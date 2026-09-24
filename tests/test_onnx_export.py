# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import math
import pathlib
import warnings

import numpy as np
import pytest
import torch
from pydantic import SecretStr

from s4casting.core.config import (
    ChronosConfiguration,
    Configuration,
    SSMConfiguration,
    TransformerConfiguration,
)
from s4casting.inference.onnx_export import (
    build_model,
    check_export_supported,
    count_input_features,
    example_inputs,
    export_onnx,
    read_metadata,
    sequence_length,
)
from tests.utils import load_config

# The ONNX toolchain is an optional extra (`uv sync --extra onnx`). Skip rather than
# fail collection when it is absent, so `pytest tests` still works without it.
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")


@pytest.fixture()
def transformer_config() -> Configuration:
    """Build a small transformer configuration that exports quickly.

    Returns:
        Configuration: A transformer configuration with a GMM head.
    """
    cfg = load_config()
    cfg.model.model = "transformer"
    cfg.model.transformer = TransformerConfiguration(latent_dim=32, n_heads=2, n_layers=1, mlp_layers=2)
    cfg.model.loss.loss = "nll"
    cfg.model.output_head.arch = "gmm"
    cfg.model.output_head.n_gaussians = 2
    return cfg


def test_export_matches_pytorch(transformer_config: Configuration, tmp_path: pathlib.Path) -> None:
    """The exported graph reproduces the PyTorch outputs, including at unseen batch sizes.

    Args:
        transformer_config: Small transformer configuration.
        tmp_path: Pytest temporary directory.
    """
    path, deviations = export_onnx(transformer_config, str(tmp_path / "model.onnx"))

    assert path.is_file()
    assert deviations, "export should have verified itself"
    for deviation in deviations.values():
        assert deviation < 1e-4


def test_batch_axis_is_dynamic(transformer_config: Configuration, tmp_path: pathlib.Path) -> None:
    """The batch axis stays symbolic, so one artefact serves any batch size.

    Guards the 0/1 specialisation trap: tracing with a batch of 1 silently freezes the
    batch size even though it is declared dynamic.

    Args:
        transformer_config: Small transformer configuration.
        tmp_path: Pytest temporary directory.
    """
    path, _ = export_onnx(transformer_config, str(tmp_path / "model.onnx"), verify=False)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape[0] == "batch"

    for batch_size in (1, 4):
        x, xm = example_inputs(transformer_config, batch_size=batch_size)
        prediction = session.run(None, {"x": x.numpy(), "xm": xm.numpy()})[0]
        assert prediction.shape[0] == batch_size
        assert not np.isnan(prediction).any()


def test_export_without_checkpoint_is_shaped_correctly(transformer_config: Configuration) -> None:
    """Shapes are fully determined by the configuration, so they hold before training.

    Args:
        transformer_config: Small transformer configuration.
    """
    model = build_model(transformer_config)
    x, xm = example_inputs(transformer_config)

    with torch.no_grad():
        prediction, loss = model(x, xm, 15, 15)

    assert loss is None
    assert x.shape == (2, sequence_length(transformer_config.model), count_input_features(transformer_config.io))
    assert prediction.shape == (
        2,
        sequence_length(transformer_config.model),
        transformer_config.model.n_out_features,
        transformer_config.model.output_head.n_gaussians,
        3,
    )


def test_metadata_describes_the_outputs(transformer_config: Configuration, tmp_path: pathlib.Path) -> None:
    """The artefact carries what a consumer needs to interpret the prediction tensor.

    Args:
        transformer_config: Small transformer configuration.
        tmp_path: Pytest temporary directory.
    """
    path, _ = export_onnx(transformer_config, str(tmp_path / "model.onnx"), verify=False)

    metadata = {p.key: p.value for p in onnx.load(str(path), load_external_data=False).metadata_props}

    assert metadata["s4casting.model"] == "transformer"
    assert metadata["s4casting.output_head"] == "gmm"
    assert metadata["s4casting.n_gaussians"] == "2"
    assert metadata["s4casting.horizon_length"]


@pytest.mark.parametrize(
    ("kernel", "expected"),
    [
        ("s4", "l_kernel.item()"),
        ("gru", "rate argument"),
    ],
)
def test_unsupported_ssm_kernels_are_rejected(transformer_config: Configuration, kernel: str, expected: str) -> None:
    """Untraceable SSM kernels fail with an explanation instead of a symbolic-shape dump.

    Args:
        transformer_config: Small configuration to adapt.
        kernel: SSM kernel under test.
        expected: Fragment of the reason the message must carry.
    """
    transformer_config.model.model = "ssm"
    transformer_config.model.ssm = SSMConfiguration(kernel=kernel, n_layers=1)

    with pytest.raises(NotImplementedError, match=expected):
        check_export_supported(transformer_config)


def test_chronos_is_rejected(transformer_config: Configuration) -> None:
    """Chronos is refused rather than exported untested.

    Args:
        transformer_config: Small configuration to adapt.
    """
    transformer_config.model.model = "chronos"
    transformer_config.model.chronos = ChronosConfiguration()

    with pytest.raises(NotImplementedError, match="chronos"):
        check_export_supported(transformer_config)


def test_s6_kernel_is_allowed_but_warns(transformer_config: Configuration) -> None:
    """The s6 kernel exports, but flags that the CPU and CUDA kernels differ.

    Args:
        transformer_config: Small configuration to adapt.
    """
    transformer_config.model.model = "ssm"
    transformer_config.model.ssm = SSMConfiguration(kernel="s6", n_layers=1)

    with pytest.warns(UserWarning, match="mambacpu"):
        check_export_supported(transformer_config)


def test_transformer_is_supported_silently(transformer_config: Configuration) -> None:
    """The supported path raises nothing and warns about nothing.

    Args:
        transformer_config: Small transformer configuration.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_export_supported(transformer_config)


def test_full_config_is_embedded(transformer_config: Configuration, tmp_path: pathlib.Path) -> None:
    """The whole configuration travels with the artefact, so training settings stay recoverable.

    Args:
        transformer_config: Small transformer configuration.
        tmp_path: Pytest temporary directory.
    """
    path, _ = export_onnx(transformer_config, str(tmp_path / "model.onnx"), verify=False)

    config = read_metadata(str(path))["config"]

    assert config["model"]["transformer"]["latent_dim"] == 32
    assert config["model"]["output_head"]["arch"] == "gmm"
    assert config["optimizer"]["learning_rate"] == transformer_config.optimizer.learning_rate
    assert config["run"]["seed"] == transformer_config.run.seed
    assert "io" in config
    assert "training" in config


def test_credentials_never_reach_the_artefact(transformer_config: Configuration, tmp_path: pathlib.Path) -> None:
    """The authentication section is excluded, so an exported model is safe to hand over.

    Args:
        transformer_config: Small transformer configuration.
        tmp_path: Pytest temporary directory.
    """
    transformer_config.authentication.wandb_api_key = SecretStr("SUPER-SECRET-TOKEN-123")

    path, _ = export_onnx(transformer_config, str(tmp_path / "model.onnx"), verify=False)

    assert b"SUPER-SECRET-TOKEN-123" not in path.read_bytes()
    assert "authentication" not in read_metadata(str(path))["config"]


def test_quantiles_output_matches_the_mixture(transformer_config: Configuration, tmp_path: pathlib.Path) -> None:
    """The in-graph quantiles genuinely invert the mixture CDF.

    Checked against ground truth rather than against gmm_to_quantiles, which snaps to a
    1000-point grid and is the less accurate of the two.

    Args:
        transformer_config: Small transformer configuration.
        tmp_path: Pytest temporary directory.
    """
    quantile_values = transformer_config.model.output_head.quantile_values
    path, _ = export_onnx(transformer_config, str(tmp_path / "model.onnx"), with_quantiles=True, verify=False)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [o.name for o in session.get_outputs()] == ["prediction", "quantiles"]

    x, xm = example_inputs(transformer_config)
    prediction, quantiles = session.run(None, {"x": x.numpy(), "xm": xm.numpy()})

    assert quantiles.shape[-1] == len(quantile_values)
    assert (np.diff(quantiles, axis=-1) >= -1e-4).all(), "quantiles must not decrease with q"

    logpi, sigma, mu = torch.tensor(prediction).unbind(-1)
    weights = torch.exp(logpi)
    weights = weights / weights.sum(-1, keepdim=True)

    z = (torch.tensor(quantiles).unsqueeze(-1) - mu.unsqueeze(-2)) / (sigma.unsqueeze(-2) * math.sqrt(2.0))
    cdf = (weights.unsqueeze(-2) * 0.5 * (1.0 + torch.erf(z))).sum(-1)

    deviation = (cdf - torch.tensor(quantile_values, dtype=cdf.dtype)).abs().max()
    assert deviation < 1e-4, f"CDF at the returned quantiles is off by {deviation:.2e}"


def test_quantiles_flag_is_ignored_for_a_quantile_head(
    transformer_config: Configuration, tmp_path: pathlib.Path
) -> None:
    """A quantile head already outputs quantiles, so the flag warns instead of duplicating them.

    Args:
        transformer_config: Small transformer configuration.
        tmp_path: Pytest temporary directory.
    """
    transformer_config.model.loss.loss = "pinball"
    transformer_config.model.output_head.arch = "quantile"

    with pytest.warns(UserWarning, match="already outputs quantiles"):
        path, _ = export_onnx(transformer_config, str(tmp_path / "model.onnx"), with_quantiles=True, verify=False)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [o.name for o in session.get_outputs()] == ["prediction"]


def test_time_axis_can_be_dynamic(transformer_config: Configuration, tmp_path: pathlib.Path) -> None:
    """One artefact serves several sequence lengths, constrained to whole patches.

    Args:
        transformer_config: Small transformer configuration.
        tmp_path: Pytest temporary directory.
    """
    patch_size = transformer_config.model.patch_encoder.patch_size
    path, _ = export_onnx(transformer_config, str(tmp_path / "model.onnx"), dynamic_time=True, verify=False)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert not isinstance(session.get_inputs()[0].shape[1], int), "time axis should stay symbolic"

    features = count_input_features(transformer_config.io)
    for steps in (patch_size * 32, patch_size * 64):
        x = np.zeros((2, steps, features), dtype=np.float32)
        prediction = session.run(None, {"x": x, "xm": np.ones_like(x)})[0]
        assert prediction.shape[:2] == (2, steps)
        assert not np.isnan(prediction).any()


def test_time_axis_rejects_a_partial_patch(transformer_config: Configuration, tmp_path: pathlib.Path) -> None:
    """A length that is not a whole number of patches is refused, not silently mangled.

    Args:
        transformer_config: Small transformer configuration.
        tmp_path: Pytest temporary directory.
    """
    patch_size = transformer_config.model.patch_encoder.patch_size
    path, _ = export_onnx(transformer_config, str(tmp_path / "model.onnx"), dynamic_time=True, verify=False)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    features = count_input_features(transformer_config.io)
    x = np.zeros((2, patch_size * 32 + 1, features), dtype=np.float32)

    with pytest.raises(Exception, match=r"[Rr]eshape"):
        session.run(None, {"x": x, "xm": np.ones_like(x)})
