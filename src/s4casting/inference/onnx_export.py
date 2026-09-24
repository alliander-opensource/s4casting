# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

"""Export a trained s4casting model to ONNX.

The exported graph is inference-only: it takes the input series and its mask and
returns the raw output-head tensor. Loss computation, the target arguments and the
sampling-interval arguments are dropped, because none of them affect the prediction.

The graph structure depends only on the TOML configuration, so a model can be
exported (and the whole pipeline validated) with randomly initialised weights before
a checkpoint exists. Supplying a checkpoint later changes parameter values only.
"""

import io
import json
import math
import pathlib
import typing
import warnings

import numpy as np
import tomlkit
import torch
from torch import nn

from s4casting import factories as fc
from s4casting.core.config import Configuration, IOConfiguration, MachineConfiguration, ModelConfiguration
from s4casting.data.files.loader import FileAccess

DEFAULT_OPSET = 18
MINUTES_PER_DAY = 24 * 60

#: Reasons an architecture cannot currently be traced into an ONNX graph. Verified by
#: attempting the export, not assumed; see check_export_supported for the details.
UNSUPPORTED_MODELS = {
    "chronos": (
        "the Chronos backbone has never been exercised by this exporter, and its wrapped "
        "HuggingFace generation loop is unlikely to trace"
    ),
}

UNSUPPORTED_SSM_KERNELS = {
    "s4": (
        "SSMKernelDPLR.forward branches on self.l_kernel.item() and grows its kernel in a "
        "while loop, so tracing fails with GuardOnDataDependentSymNode. The kernel setup has "
        "to move out of the traced forward pass first"
    ),
    "gru": (
        "GruBlock.forward does not accept the rate argument SequenceResidualBlock passes it, "
        "so this kernel raises TypeError in plain PyTorch too and needs fixing before export"
    ),
}


class GaussianMixtureQuantiles(nn.Module):
    """Turn Gaussian mixture parameters into quantiles, in a traceable way.

    ``core.distributions.gmm_to_quantiles`` cannot be exported: it calls ``.item()``,
    builds ``torch.distributions`` objects and loops in Python over ``torch.nonzero``
    results. This computes the same thing with pure tensor ops, so it becomes part of the
    ONNX graph.

    The mixture CDF is a weighted sum of normal CDFs, and a normal CDF is an ``erf``,
    which ONNX supports. Inverting it is a bisection with a fixed iteration count, so the
    control flow is static and traces cleanly. All quantiles are solved at once rather
    than in a Python loop, which keeps the graph small.
    """

    def __init__(self, quantile_values: typing.Sequence[float], iterations: int = 30, sigma_span: float = 10.0):
        """Initialize the quantile layer.

        Args:
            quantile_values (typing.Sequence[float]): Quantiles to compute, in (0, 1).
            iterations (int): Bisection steps. Each one halves the bracket. The result
                stops improving at 25 in float32; the default leaves a margin. Every step
                costs graph nodes and runtime, so raising this is rarely worth it.
            sigma_span (float): How many standard deviations to bracket around the
                component means before bisecting.
        """
        super().__init__()
        self.iterations = iterations
        self.sigma_span = sigma_span
        self.register_buffer("quantiles", torch.tensor(list(quantile_values), dtype=torch.float32))

    def _cdf(self, x: torch.Tensor, pi: torch.Tensor, sigma: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """Evaluate the mixture CDF at x.

        Args:
            x (torch.Tensor): Points of shape (..., Q).
            pi (torch.Tensor): Normalised weights of shape (..., G).
            sigma (torch.Tensor): Standard deviations of shape (..., G).
            mu (torch.Tensor): Means of shape (..., G).

        Returns:
            torch.Tensor: CDF values of shape (..., Q).
        """
        z = (x.unsqueeze(-1) - mu.unsqueeze(-2)) / (sigma.unsqueeze(-2) * math.sqrt(2.0))
        return (pi.unsqueeze(-2) * 0.5 * (1.0 + torch.erf(z))).sum(-1)

    def forward(self, logpi: torch.Tensor, sigma: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """Invert the mixture CDF at each configured quantile.

        Args:
            logpi (torch.Tensor): Log mixture weights of shape (..., G).
            sigma (torch.Tensor): Standard deviations of shape (..., G).
            mu (torch.Tensor): Means of shape (..., G).

        Returns:
            torch.Tensor: Quantiles of shape (..., Q).
        """
        pi = torch.softmax(logpi, dim=-1)

        quantiles = self.quantiles.to(mu.dtype)
        lower = (mu - self.sigma_span * sigma).amin(-1, keepdim=True).expand(*mu.shape[:-1], quantiles.shape[0])
        upper = (mu + self.sigma_span * sigma).amax(-1, keepdim=True).expand(*mu.shape[:-1], quantiles.shape[0])

        lower, upper = lower.contiguous(), upper.contiguous()
        for _ in range(self.iterations):
            middle = 0.5 * (lower + upper)
            below = self._cdf(middle, pi, sigma, mu) < quantiles
            lower = torch.where(below, middle, lower)
            upper = torch.where(below, upper, middle)

        return 0.5 * (lower + upper)


class InferenceWrapper(nn.Module):
    """Tensor-in/tensor-out view of an s4casting model, suitable for tracing.

    The wrapped models return ``(prediction, loss)`` and take the sampling intervals as
    plain Python ints. ONNX graphs have neither optional tensor inputs nor a loss branch,
    so this wrapper pins the intervals to their configured values, never passes a target,
    and returns the prediction alone.
    """

    def __init__(
        self,
        model: nn.Module,
        input_interval: int,
        output_interval: int,
        quantile_layer: nn.Module | None = None,
    ) -> None:
        """Initialize the InferenceWrapper.

        Args:
            model (nn.Module): The raw (non-DDP) model to wrap.
            input_interval (int): Input sampling interval in minutes.
            output_interval (int): Output sampling interval in minutes.
            quantile_layer (nn.Module | None): When given, the graph gains a second output
                holding quantiles derived from the mixture parameters.
        """
        super().__init__()
        self.model = model
        self.input_interval = input_interval
        self.output_interval = output_interval
        self.quantile_layer = quantile_layer

    def forward(self, x: torch.Tensor, xm: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run the prediction branch of the wrapped model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, time, features).
            xm (torch.Tensor): Mask tensor of shape (batch, time, features).

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: The output-head tensor. For a
                GMM head that is (batch, time, n_out_features, n_gaussians, 3) holding
                (logpi, sigma, mu); for a quantile head it is
                (batch, time, n_out_features, n_quantiles). When a quantile layer is
                attached, a second tensor of shape (batch, time, n_out_features,
                n_quantiles) is returned alongside it.
        """
        prediction, _ = self.model(x, xm, self.input_interval, self.output_interval)

        if self.quantile_layer is None:
            return prediction

        logpi, sigma, mu = prediction.unbind(dim=-1)
        return prediction, self.quantile_layer(logpi, sigma, mu)


def check_export_supported(config: Configuration) -> None:
    """Reject architectures known not to trace, and warn about partially verified ones.

    Failing here keeps the error close to the cause. Left to torch.export, an unsupported
    architecture surfaces as a several-hundred-line symbolic-shape trace that says nothing
    about which part of this repository is responsible.

    Args:
        config (Configuration): Loaded configuration.

    Raises:
        NotImplementedError: If the configured architecture cannot be exported.
    """
    architecture = config.model.model

    if architecture in UNSUPPORTED_MODELS:
        raise NotImplementedError(
            f"ONNX export is not implemented for model='{architecture}': "
            f"{UNSUPPORTED_MODELS[architecture]}. Only model='transformer' is fully supported; "
            f"model='ssm' works with the 's6' kernel."
        )

    if architecture != "ssm":
        return

    kernel = config.model.ssm.kernel if config.model.ssm else None

    if kernel in UNSUPPORTED_SSM_KERNELS:
        raise NotImplementedError(
            f"ONNX export is not implemented for model='ssm' with kernel='{kernel}': "
            f"{UNSUPPORTED_SSM_KERNELS[kernel]}. Use kernel='s6', or model='transformer'."
        )

    warnings.warn(
        "Exporting model='ssm' with kernel='s6'. The export always builds on CPU, which "
        "selects the naive Mamba implementation in s4casting.model.mambacpu, while training "
        "on GPU runs the CUDA kernels in s4casting.model.mamba. The two share parameter names "
        "and shapes, so a checkpoint loads either way, but they have not been compared "
        "numerically. Check the exported predictions against the PyTorch model on GPU before "
        "relying on this artefact.",
        UserWarning,
        stacklevel=2,
    )


def load_configuration(path: str | Configuration) -> Configuration:
    """Load a Configuration from a TOML file.

    Falls back to validating only the sections the exporter needs when the full
    configuration cannot be validated. The top-level validator opens every dataset
    referenced under ``[io.features]``, which is not necessarily available on the
    machine performing the export.

    Args:
        path (str | Configuration): Path to the TOML file, or an already-built Configuration.

    Returns:
        Configuration: The loaded configuration.

    Raises:
        ValueError: If the path does not point at a ``.toml`` file.
        FileNotFoundError: If the configuration file does not exist.
    """
    if isinstance(path, Configuration):
        return path

    # The path comes straight from the command line: resolve it and only accept a
    # regular TOML file, so nothing else on the filesystem can be read through here.
    config_path = pathlib.Path(path).expanduser().resolve()
    if config_path.suffix.lower() != ".toml":
        raise ValueError(f"Config file must be a .toml file: {path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = tomlkit.load(f).unwrap()

    try:
        return Configuration(**raw)
    except Exception:
        # Dataset files are absent; validate only what determines the graph.
        return Configuration.model_construct(
            machine=MachineConfiguration(**raw["machine"]),
            model=ModelConfiguration(**raw["model"]),
            io=IOConfiguration(**raw["io"]),
        )


def count_input_features(io_config: IOConfiguration) -> int:
    """Count the input features the model expects, mirroring the model factory.

    Datasets sharing a prefix (all ``measurements_*`` sources, for instance) are stacked
    into the same channel, so they are collapsed before counting.

    This duplicates the feature counting in
    factories.model_container.provide_model_container and has to stay in step with it.
    Exporting the count from the factory would be better, but that means changing a
    function the trainer depends on.

    Args:
        io_config (IOConfiguration): IO configuration.

    Returns:
        int: Number of features along the last axis of the model input.
    """
    unique_datasets = {name.split("_")[0]: dataset for name, dataset in io_config.features.items()}
    return sum(
        len(dataset.subset_features) if dataset.subset_features else dataset.n_features
        for dataset in unique_datasets.values()
    )


def sequence_length(model_config: ModelConfiguration) -> int:
    """Compute the total number of time steps the model consumes.

    The context window covers the forecast horizon, so this is the full window handed to
    the model, not the history alone.

    Args:
        model_config (ModelConfiguration): Model configuration.

    Returns:
        int: Number of time steps.
    """
    return int(model_config.context_window[0] * MINUTES_PER_DAY / model_config.base_sample_interval_minutes)


def horizon_length(model_config: ModelConfiguration) -> int:
    """Compute the number of forecast time steps.

    Args:
        model_config (ModelConfiguration): Model configuration.

    Returns:
        int: Number of forecast time steps.
    """
    return int(model_config.predict_width * MINUTES_PER_DAY / model_config.base_sample_interval_minutes)


def load_checkpoint_weights(model: nn.Module, checkpoint_path: str, device: str = "cpu") -> dict[str, str]:
    """Load model weights from a training checkpoint in place.

    Only the model weights are read; the optimizer state a checkpoint also carries is
    irrelevant for export. Checkpoints written under DDP carry a ``module.`` prefix that
    is stripped here.

    Args:
        model (nn.Module): Model to load the weights into.
        checkpoint_path (str): Path to the checkpoint written by the Checkpointer.
        device (str): Device to map the tensors onto.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.

    Returns:
        dict[str, str]: Where the weights came from, for embedding in the artefact.
    """
    is_remote = checkpoint_path.startswith(("s3://", "http://", "https://"))
    if not is_remote and not pathlib.Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = FileAccess(checkpoint_path).load_pydantic()
    # weights_only restricts unpickling to tensors and primitive containers.
    state_dict = torch.load(io.BytesIO(checkpoint["torch_model"]), map_location=device, weights_only=True)
    state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)

    return {
        "s4casting.checkpoint": checkpoint_path,
        "s4casting.checkpoint_iteration": str(checkpoint["iteration"]),
        "s4casting.checkpoint_loss": str(checkpoint["loss"]),
    }


def build_model(config: Configuration, checkpoint_path: str | None = None) -> nn.Module:
    """Build the model described by the configuration, optionally with trained weights.

    Args:
        config (Configuration): Loaded configuration.
        checkpoint_path (str | None): Checkpoint to load. Weights stay randomly
            initialised when omitted, which is enough to validate the export itself.

    Returns:
        nn.Module: The model in eval mode on CPU.
    """
    machine = fc.provide_machine(MachineConfiguration(device_kind="cpu"), rng_base_seed=config.run.seed)
    model = fc.provide_model_container(config.model, config.io, machine).raw_model

    if checkpoint_path:
        load_checkpoint_weights(model, checkpoint_path)

    return model.eval()


def example_inputs(config: Configuration, batch_size: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Build example inputs shaped like a real inference request.

    The mask zeroes the target channel over the forecast horizon, which is what marks
    those steps as the values to predict.

    Note the default batch size of 2. ``torch.export`` treats any axis whose example
    length is 0 or 1 as a constant, so tracing with a batch of 1 silently bakes the batch
    size into the graph even when it is declared dynamic.

    Args:
        config (Configuration): Loaded configuration.
        batch_size (int): Batch size of the example inputs.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The example (x, xm) pair.
    """
    steps = sequence_length(config.model)
    horizon = horizon_length(config.model)
    features = count_input_features(config.io)

    x = torch.randn(batch_size, steps, features)
    xm = torch.ones(batch_size, steps, features)
    xm[:, -horizon:, : config.model.n_out_features] = 0
    return x, xm


def _dynamic_shapes(patch_size: int, dynamic_time: bool) -> dict[str, dict[int, typing.Any]]:
    """Describe which input axes stay symbolic in the exported graph.

    The time axis is expressed as a multiple of the patch size, because the patch encoder
    reshapes the series into whole patches and cannot handle a remainder. Declaring the
    constraint here means the runtime rejects a bad length up front rather than computing
    something meaningless.

    Args:
        patch_size (int): Patch size of the patch encoder.
        dynamic_time (bool): Whether to keep the time axis dynamic.

    Returns:
        dict[str, dict[int, typing.Any]]: Mapping accepted by torch.onnx.export.
    """
    axes: dict[int, typing.Any] = {0: "batch"}

    if dynamic_time:
        patches = torch.export.Dim("patches", min=2, max=4096)
        axes[1] = patch_size * patches

    return {"x": axes, "xm": dict(axes)}


def _metadata(config: Configuration) -> dict[str, str]:
    """Collect everything a consumer needs to interpret the graph outputs.

    Stored in the ONNX metadata so the artefact stays self-describing once it leaves this
    repository. The individual keys cover what a consumer needs to read the prediction
    tensor; ``s4casting.config`` carries the whole configuration for provenance.

    The authentication section is excluded. Pydantic already masks its SecretStr fields,
    but an exported model is meant to be handed to other people and that section holds
    nothing worth shipping.

    Args:
        config (Configuration): Loaded configuration.

    Returns:
        dict[str, str]: Metadata key/value pairs.
    """
    return {
        "s4casting.config": json.dumps(
            config.model_dump(mode="json", exclude={"authentication"}),
            indent=2,
            default=str,
        ),
        "s4casting.model": config.model.model,
        "s4casting.output_head": config.model.output_head.arch,
        "s4casting.n_gaussians": str(config.model.output_head.n_gaussians),
        "s4casting.quantile_values": ",".join(str(q) for q in config.model.output_head.quantile_values),
        "s4casting.n_out_features": str(config.model.n_out_features),
        "s4casting.n_input_features": str(count_input_features(config.io)),
        "s4casting.feature_order": ",".join(config.io.feature_order),
        "s4casting.patch_size": str(config.model.patch_encoder.patch_size),
        "s4casting.sequence_length": str(sequence_length(config.model)),
        "s4casting.horizon_length": str(horizon_length(config.model)),
        "s4casting.base_sample_interval_minutes": str(config.model.base_sample_interval_minutes),
    }


def export_onnx(
    config: str | Configuration,
    output_path: str,
    checkpoint_path: str | None = None,
    *,
    dynamic_time: bool = False,
    external_data: bool = False,
    opset_version: int = DEFAULT_OPSET,
    with_quantiles: bool = False,
    verify: bool = True,
) -> tuple[pathlib.Path, dict[int, float]]:
    """Export a configured model to an ONNX file.

    Args:
        config (str | Configuration): Path to the TOML configuration, or a Configuration.
        output_path (str): Destination ``.onnx`` path.
        checkpoint_path (str | None): Checkpoint to load. Omit to export random weights,
            which validates the graph without needing a trained model.
        dynamic_time (bool): Keep the time axis dynamic, in multiples of the patch size.
            The batch axis is always dynamic.
        external_data (bool): Write weights to a sidecar ``.onnx.data`` file. A single
            self-contained file is easier to ship and stays valid below 2GB.
        opset_version (int): ONNX opset to target.
        with_quantiles (bool): For a GMM head, add a second output holding quantiles
            derived from the mixture parameters, so consumers need no post-processing.
            Ignored for a quantile head, whose output already is quantiles.
        verify (bool): Compare the exported graph against PyTorch before returning.

    Raises:
        NotImplementedError: If the configured architecture cannot be exported.
        RuntimeError: If the exporter returns no program.

    Returns:
        tuple[pathlib.Path, dict[int, float]]: The path written, and the maximum absolute
            deviation from PyTorch per batch size (empty when verification is skipped).
    """
    configuration = load_configuration(config)
    check_export_supported(configuration)

    model = build_model(configuration)
    checkpoint_metadata = load_checkpoint_weights(model, checkpoint_path) if checkpoint_path else {}

    emit_quantiles = with_quantiles and configuration.model.output_head.arch == "gmm"
    if with_quantiles and not emit_quantiles:
        warnings.warn(
            f"with_quantiles is ignored for output_head.arch="
            f"'{configuration.model.output_head.arch}': that head already outputs quantiles.",
            UserWarning,
            stacklevel=2,
        )

    quantile_layer = (
        GaussianMixtureQuantiles(configuration.model.output_head.quantile_values) if emit_quantiles else None
    )

    wrapper = InferenceWrapper(
        model,
        input_interval=configuration.model.input_sample_intervals_minutes[0],
        output_interval=configuration.model.output_sample_intervals_minutes[0],
        quantile_layer=quantile_layer,
    ).eval()

    output_names = ["prediction", "quantiles"] if emit_quantiles else ["prediction"]
    inputs = example_inputs(configuration)

    with torch.no_grad():
        program = torch.onnx.export(
            wrapper,
            inputs,
            dynamo=True,
            opset_version=opset_version,
            input_names=["x", "xm"],
            output_names=output_names,
            dynamic_shapes=_dynamic_shapes(configuration.model.patch_encoder.patch_size, dynamic_time),
        )

    # export() only returns None when it is given a destination to write to, which is not
    # the case here: the program is saved further down, after the metadata is attached.
    if program is None:
        raise RuntimeError("torch.onnx.export returned no ONNXProgram")

    # ONNXProgram.model_proto is a regenerated view, so metadata has to go on the IR model.
    program.model.metadata_props.update(
        _metadata(configuration) | checkpoint_metadata | {"s4casting.outputs": ",".join(output_names)}
    )

    destination = pathlib.Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    program.save(str(destination), external_data=external_data)

    deviations = verify_onnx(str(destination), wrapper, configuration) if verify else {}
    return destination, deviations


def verify_onnx(
    onnx_path: str,
    wrapper: nn.Module,
    config: Configuration,
    *,
    batch_sizes: tuple[int, ...] = (1, 3),
    tolerance: float = 1e-4,
) -> dict[int, float]:
    """Check an exported graph against the PyTorch module it came from.

    An export that produces a loadable file can still be wrong: tracing can bake a shape
    in as a constant or drop a branch, and neither shows up until the outputs are
    compared. This runs both implementations on identical inputs.

    The module must be the one that was exported, not an equivalent rebuild. Nothing in
    the factories seeds the RNG, so two builds of the same configuration hold different
    random weights whenever no checkpoint pins them.

    Args:
        onnx_path (str): Path to the exported ONNX file.
        wrapper (nn.Module): The module that was exported.
        config (Configuration): The configuration used for the export.
        batch_sizes (tuple[int, ...]): Batch sizes to compare, exercising the dynamic axis.
        tolerance (float): Maximum tolerated absolute deviation.

    Returns:
        dict[int, float]: Maximum absolute deviation per batch size.

    Raises:
        AssertionError: If any batch size deviates by more than the tolerance.
    """
    # onnxruntime is an optional dependency: exporting does not need a runtime.
    import onnxruntime as ort  # noqa: PLC0415

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    deviations: dict[int, float] = {}
    for batch_size in batch_sizes:
        x, xm = example_inputs(config, batch_size=batch_size)

        with torch.no_grad():
            expected = wrapper(x.clone(), xm.clone())

        # A wrapper emitting quantiles returns a tuple; compare every output, not just the first.
        expected_tensors = expected if isinstance(expected, tuple) else (expected,)
        actual_tensors = session.run(None, {"x": x.numpy(), "xm": xm.numpy()})

        deviation = max(
            float(np.abs(reference.numpy() - produced).max())
            for reference, produced in zip(expected_tensors, actual_tensors, strict=True)
        )
        deviations[batch_size] = deviation

        if deviation > tolerance:
            raise AssertionError(
                f"ONNX output deviates from PyTorch by {deviation:.3e} at batch size {batch_size} "
                f"(tolerance {tolerance:.0e})"
            )

    return deviations


def read_metadata(onnx_path: str) -> dict[str, typing.Any]:
    """Read the s4casting metadata back out of an exported model.

    Lets a consumer recover how a model was configured and trained without this
    repository, the original TOML, or the checkpoint.

    Args:
        onnx_path (str): Path to the exported ONNX file.

    Returns:
        dict[str, typing.Any]: The metadata with the ``s4casting.`` prefix stripped.
            ``config`` is parsed back into a nested dict; the rest are strings.
    """
    import onnx  # noqa: PLC0415

    model = onnx.load(onnx_path, load_external_data=False)
    metadata = {
        entry.key.removeprefix("s4casting."): entry.value
        for entry in model.metadata_props
        if entry.key.startswith("s4casting.")
    }

    if "config" in metadata:
        metadata["config"] = json.loads(metadata["config"])

    return metadata
