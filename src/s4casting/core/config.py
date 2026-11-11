# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import json
import warnings
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    SecretStr,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)

from s4casting.data.dataset.interface import ContextWindowAlignment


class StrictModel(BaseModel):
    """Base for every configuration model: unknown keys raise instead of being silently ignored.

    Pydantic's default is to drop keys that match no field, which turns config typos and
    misplaced sections into silently different experiments (a misspelled field simply leaves the
    default in charge). All configuration models inherit from this so a wrong key fails loudly at
    startup, naming the file section and key.
    """

    model_config = ConfigDict(extra="forbid")


class DType(StrEnum):
    """DType enumeration for model precision."""

    Float16 = "float16"
    Float32 = "float32"
    Float64 = "float64"
    BFloat16 = "bfloat16"


DTYPE_MAP: dict[DType, torch.dtype] = {
    DType.Float16: torch.float16,
    DType.Float32: torch.float32,
    DType.Float64: torch.float64,
    DType.BFloat16: torch.bfloat16,
}


class LogLevel(StrEnum):
    """Log level enumeration."""

    Debug = "debug"
    Info = "info"
    Warning = "warning"
    Error = "error"
    Fatal = "fatal"


class WandbMode(StrEnum):
    """Supported Weights & Biases run modes."""

    Online = "online"
    Offline = "offline"


class WandbLoggingConfiguration(StrictModel):
    """Weights & Biases logging configuration."""

    project: str = Field(min_length=1, description="Weights & Biases project name.")
    run_id: str | None = Field(None, description="Existing Weights & Biases run ID to resume.")
    notes: str | None = Field(None, description="Notes attached to the Weights & Biases run.")
    mode: WandbMode = Field(WandbMode.Online, description="Weights & Biases run mode.")


class MLflowLoggingConfiguration(StrictModel):
    """MLflow logging configuration."""

    uri: str = Field(
        "http://127.0.0.1:5000/",
        min_length=1,
        description="MLflow tracking URI: an HTTP(S) endpoint or any URI accepted by "
        "mlflow.set_tracking_uri (e.g. a local server, a cluster service, or a managed "
        "tracking backend).",
    )
    workspace: str = Field("default", min_length=1, description="MLflow workspace name.")
    experiment: str = Field("dev", min_length=1, description="MLflow experiment name.")


class LoggingConfiguration(StrictModel):
    """Experiment logging configuration."""

    wandb: WandbLoggingConfiguration | None = Field(
        None,
        description="Weights & Biases settings.",
    )
    mlflow: MLflowLoggingConfiguration | None = Field(
        None,
        description="MLflow settings.",
    )


class AuthenticationConfiguration(StrictModel):
    """Authentication configuration."""

    aws_access_key_id: SecretStr | None = Field(
        default=None,
        description="AWS access key ID used for authenticating with AWS services.",
    )
    aws_secret_access_key: SecretStr | None = Field(
        default=None,
        description="AWS secret access key paired with the access key ID.",
    )
    aws_session_token: SecretStr | None = Field(
        default=None,
        description="Optional AWS session token for temporary credentials (STS).",
    )
    wandb_api_key: SecretStr | None = Field(
        default=None,
        description="Weights & Biases API key used to authenticate runs and uploads.",
    )
    envoy_session: SecretStr | None = Field(
        default=None,
        description="Optional Envoy session cookie/token for authenticated requests.",
    )


class OptimizerConfiguration(StrictModel):
    """Optimizer configuration."""

    learning_rate: float = Field(1e-3, description="Base learning rate used by the optimizer.")
    beta1: float = Field(0.9, description="First momentum coefficient (RAdam β1).")
    beta2: float = Field(0.99, description="Second momentum coefficient (RAdam β2).")
    eps: float = Field(1e-8, description="Small constant for numerical stability.")
    weight_decay: float = Field(1e-2, description="Decoupled (AdamW-style) weight decay, scaled by the learning rate.")
    gradient_clipping: float = Field(0.0, description="Gradient norm clipping threshold; 0 disables clipping.")


class SchedulerConfiguration(StrictModel):
    """Learning rate scheduler configuration."""

    mode: Literal["min", "max"] = Field("min", description="Direction of improvement for the monitored metric.")
    factor: float = Field(0.3, description="LR multiplicative reduction factor (new_lr = lr * factor).")
    patience: int = Field(2, description="Steps without improvement before reducing the LR.")
    threshold: float = Field(1e-4, description="Minimal metric change to count as an improvement.")
    threshold_mode: Literal["rel", "abs"] = Field(
        "rel", description="Compare improvements in absolute or relative terms."
    )
    cooldown: int = Field(0, description="Steps to wait after LR reduction before resuming monitoring.")
    min_lr: float = Field(5e-5, description="Lower bound for the learning rate.")
    eps: float = Field(1e-8, description="Minimum LR change to apply, avoiding tiny no-op updates.")


class LossConfiguration(StrictModel):
    """Loss configuration."""

    loss: Literal["mse", "nll", "pinball"] = Field(
        "nll",
        description="Training loss function: 'mse' (mean squared error), "
        "'nll' (negative log-likelihood), or 'pinball' (quantile).",
    )
    alpha_clip: NonNegativeFloat = Field(
        0.0,
        description="Non-negative clamp for alpha/shape parameters to stabilize training; 0 disables clipping.",
    )
    mask_mode: Literal["masked", "unmasked"] = Field(
        "masked",
        description="Mode for loss calculation.",
    )
    sigma_regularisation_factor: NonNegativeFloat = Field(
        0.0,
        description="Prevents sigma from growing too big",
    )
    components: dict[str, float] = Field(
        default_factory=lambda: {"primary": 1.0},
        description=(
            "Named loss components and their scalar weights. "
            "Keys must match terms accumulated in the model forward pass. "
            "Example: {primary = 1.0, weather = 0.5}"
        ),
    )


class ChronosConfiguration(StrictModel):
    """Chronos-2 fine-tuning configuration."""

    model_id: str = Field(
        "autogluon/chronos-2-small",
        description="HuggingFace model ID for the pretrained Chronos-2 model.",
    )
    freeze_backbone: bool = Field(
        True,
        description="Freeze all non-LoRA backbone parameters during fine-tuning.",
    )
    use_lora: bool = Field(
        True,
        description="Apply LoRA adapters for parameter-efficient fine-tuning.",
    )
    lora_rank: int = Field(8, description="LoRA rank (r).")
    lora_alpha: float = Field(32.0, description="LoRA scaling factor (alpha).")
    lora_target_modules: list[str] = Field(
        ["q", "v"],
        description="Attention projection names to apply LoRA to.",
    )


class SSMConfiguration(StrictModel):
    """State Space Model configuration."""

    kernel: Literal["s4", "s6", "gru"] = Field(
        "s4",
        description="SSM kernel variant to use: 's4', 's6', or 'gru' baseline.",
    )
    n_layers: PositiveInt = Field(
        4,
        description="Number of stacked layers in the SSM/GRU block.",
    )
    mixer_size: PositiveInt | None = Field(
        None,
        description="Whether to do time domain mixing, limits maximum model context width.",
    )
    latent_dim: NonNegativeInt = Field(256, description="Dimensionality of the latent space.")


class TransformerConfiguration(StrictModel):
    """Transformer model configuration."""

    latent_dim: NonNegativeInt = Field(256, description="Dimensionality of the latent space.")
    n_heads: NonNegativeInt = Field(8, description="Number of attention heads per layer.")
    n_layers: NonNegativeInt = Field(6, description="Number of Transformer layers (encoder blocks).")
    dropout: float = Field(0.0, ge=0, lt=1, description="Dropout rate applied to attention/MLP.")
    attn_bias: bool = Field(True, description="Enable bias terms in attention projections.")
    mlp_layers: NonNegativeInt = Field(2, description="Number of MLP layers per block.")
    causal: bool = Field(
        False,
        description="Make the time attention causal, so tokens only attend to the past. Group "
        "attention (across features) is never causal: feature order carries no causal structure, "
        "and a causal mask there would cut the target off from its covariates.",
    )


class OutputHeadConfiguration(StrictModel):
    """Output head configuration."""

    arch: Literal["gmm", "quantile"] = Field(
        "gmm",
        description="Output head type: 'gmm' (Gaussian mixture) or 'quantile' (quantile regression).",
    )
    n_gaussians: PositiveInt | None = Field(
        4,
        description="Number of mixture components when arch='gmm'; ignored for 'quantile'.",
    )
    quantile_values: list[float] = Field(
        [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99],
        description="Quantiles to predict when arch='quantile' (values in (0,1], sorted ascending).",
    )


class PatchEncoderConfiguration(StrictModel):
    """Patch encoder configuration."""

    arch: Literal["linear", "gemma", "ss"] = Field(
        "linear",
        description="Patch encoder architecture: 'linear', 'gemma', or 'ss'.",
    )
    patch_size: PositiveInt = Field(8, description="Size of each patch.")
    n_layers: PositiveInt = Field(8, description="Number of layers in the patch encoder.")


class PatchDecoderConfiguration(StrictModel):
    """Patch decoder configuration."""

    arch: Literal["linear", "none"] = Field(
        "linear",
        description="Patch decoder architecture: 'linear' or 'none'.",
    )
    patch_size: PositiveInt = Field(8, description="Size of each patch.")
    n_layers: PositiveInt = Field(8, description="Number of layers in the patch decoder.")


class MetricsConfiguration(StrictModel):
    """Configuration controlling which evaluation metrics are computed.

    Each flag enables or disables computation of the corresponding metric.
    """

    crps: bool = Field(default=True, description="Compute Continuous Ranked Probability Score (CRPS).")
    mae: bool = Field(default=False, description="Compute Mean Absolute Error (MAE).")
    precision: bool = Field(default=False, description="Compute Precision.")
    recall: bool = Field(default=False, description="Compute Recall.")
    fbeta: bool = Field(default=False, description="Compute F-beta score.")
    odn_monthly_mape: bool = Field(
        default=True, description="Compute Monthly Mean Absolute Percentage Error (MAPE) for ODN."
    )
    ldn_monthly_mape: bool = Field(
        default=True, description="Compute Monthly Mean Absolute Percentage Error (MAPE) for LDN."
    )
    loss: bool = Field(default=True, description="Compute loss.")


class ModelConfiguration(StrictModel):
    """Model configuration."""

    base_sample_interval_minutes: PositiveInt = Field(
        15, description="Base sampling interval (minutes) used for internal alignment."
    )
    input_sample_intervals_minutes: list[PositiveInt] = Field(
        [15, 60], description="Input sampling intervals (minutes). Can be multiple for multi-resolution inputs."
    )
    output_sample_intervals_minutes: list[PositiveInt] = Field(
        [15], description="Output sampling intervals (minutes). Can be multiple for multi-resolution outputs."
    )

    context_window: list[PositiveInt] = Field(
        [32],
        description=("Context window size in days. Can be a single int (legacy) or a list for multi-window inputs."),
    )

    predict_width: PositiveInt | tuple[float, float] = Field(
        2,
        description=(
            "Forecast horizon. Either absolute days (int) or a fraction range tuple (min, max) of the sample "
            "length, each in [0.05, 0.5]. The range is drawn per sample under the 'randomprediction' and 'cpm' "
            "tasks, so the horizon is expressed independently of the sample rate."
        ),
    )

    alignment: ContextWindowAlignment = Field(
        ContextWindowAlignment.Daily, description="Temporal alignment strategy for input/output windows."
    )
    model: Literal["ssm", "transformer", "chronos"] = Field(
        "ssm", description="Model type: 'ssm', 'transformer', or 'chronos'."
    )
    ssm: SSMConfiguration | None = Field(None, description="SSM-specific settings (required when model='ssm').")
    transformer: TransformerConfiguration | None = Field(
        None, description="Transformer-specific settings (required when model='transformer')."
    )
    chronos: ChronosConfiguration | None = Field(
        None, description="Chronos-2 fine-tuning settings (required when model='chronos')."
    )
    patch_encoder: PatchEncoderConfiguration = Field(
        default_factory=PatchEncoderConfiguration, description="Patch encoder configuration."
    )
    patch_decoder: PatchDecoderConfiguration = Field(
        default_factory=PatchDecoderConfiguration, description="Patch decoder configuration."
    )
    output_head: OutputHeadConfiguration = Field(
        default_factory=OutputHeadConfiguration, description="Output head configuration."
    )
    loss: LossConfiguration = Field(default_factory=LossConfiguration, description="Loss configuration.")
    internal_dtype: DType = Field(DType.Float32, description="Internal data type used for computations.")
    norm_clamp: float = Field(10.0, description="Clamping value for normalization.")
    norm_eps: float = Field(1e-5, description="Epsilon value for normalization.")
    n_trainable_parameters: PositiveInt | None = Field(
        default=None, init=False, description="Number of trainable parameters. To be determined by the factory."
    )
    days_per_month: PositiveInt = Field(30, description="Number of days per month used for calculations.")
    n_out_features: PositiveInt = Field(1, description="Number of output features.")


class MachineConfiguration(StrictModel):
    """Machine configuration."""

    device_kind: Literal["cuda", "cpu", "mps"] = Field(
        "cpu", description="Kind of device to use: 'cuda', 'cpu', or 'mps'."
    )
    local_rank: int = Field(0, description="Local rank for distributed training.")
    ddp: bool = Field(False, description="Whether to use Distributed Data Parallel (DDP).")
    ddp_loss_sync: bool = Field(True, description="Whether to synchronize loss across DDP processes.")


class RunConfiguration(StrictModel):
    """Run configuration."""

    seed: int = Field(42069, description="Random seed for reproducibility.")
    run_start_date: str = Field(datetime.now(UTC).strftime("%Y-%m-%d"), description="Start date of the run.")
    log_level: LogLevel = Field(LogLevel.Info, description="Logging level.")


class DatasetConfiguration(StrictModel):
    """Dataset configuration."""

    location: str = Field(..., description="Location of the dataset.")
    loader: Literal["sqlite", "parquet", "time", "sideload", "croissant"] = Field(
        "sqlite", description="Type of loader to use for the dataset."
    )
    nearest_neighbor: bool = Field(False, description="Apply nearest-neighbour spatial matching (e.g. weather grids).")
    n_features: int = Field(1, description="Number of features in the dataset.")
    main_dataset: str = Field("", description="Name of the main dataset.")
    subset_features: list[str] = Field([], description="List of subset features to use from the dataset.")
    include_location: bool = Field(
        False,
        description="For loader='time' only: also emit the location's latitude and longitude after the unix "
        "time, so the models add a spatial embedding. Off by default, because a model trained without location "
        "transfers to series that carry no coordinates (external datasets, benchmarks).",
    )

    @model_validator(mode="after")
    def _time_loader_channels(self) -> "DatasetConfiguration":
        """Derive the time loader's channel count from ``include_location``.

        The time loader emits the unix time alone or followed by latitude and longitude, so its channel count
        is 1 or 3 and follows from the flag. ``n_features`` need not be written for it; an explicit value that
        disagrees with the flag is rejected. The flag is meaningless for any other loader and is rejected there.

        Raises:
            ValueError: If ``include_location`` is set on a non-time loader, or ``n_features`` contradicts it.

        Returns:
            DatasetConfiguration: The validated configuration.
        """
        if self.loader != "time":
            if self.include_location:
                raise ValueError("include_location is only valid for loader='time'.")
            return self
        expected = 3 if self.include_location else 1
        if "n_features" in self.model_fields_set and self.n_features != expected:
            raise ValueError(
                f"loader='time' with include_location={self.include_location} provides {expected} channel(s); "
                f"n_features={self.n_features} contradicts it. Leave n_features unset for the time loader."
            )
        self.n_features = expected
        return self


class IOConfiguration(StrictModel):
    """IO configuration."""

    feature_order: list[str] = Field(..., description="Order of features/datasets for input/output.")
    features: dict[str, DatasetConfiguration] = Field(..., description="Dictionary of (feature) datasets.")
    output: str = Field(..., description="Location to save outputs.")
    load_checkpoint: str | None = Field(None, description="Path to load checkpoint from.")
    gap_skip_perc: PositiveInt = Field(
        10, ge=0, le=100, description="Percentage of context window of hours to skip for gaps."
    )
    context_window_valid_ratio: float = Field(0.8, description="Valid ratio for input window.")
    hash_datasets: bool = Field(False, description="Whether to hash the datasets to be logged")
    to_memory: bool = Field(False, description="Whether to move the memmap'd data to CPU memory")
    interval_context_limits: dict[int, dict[str, int]] = Field(
        default={
            5: {"min_days": 7, "max_days": 10},
            10: {"min_days": 7, "max_days": 14},
            15: {"min_days": 11, "max_days": 32},
            30: {"min_days": 16, "max_days": 64},
            60: {"min_days": 16, "max_days": 720},
            1440: {"min_days": 32, "max_days": 720},
            10080: {"min_days": 64, "max_days": 720},
        },
        description="Pairs of valid interval and min max days in the context windows",
    )

    def input_feature_counts(self) -> tuple[int, int]:
        """Count the model input channels implied by this configuration.

        Datasets sharing a prefix (``measurements_cdb``, ``measurements_eweld``, ...) form one feature group
        and are counted once. The time loader contributes its channels (unix time, optionally latitude and
        longitude) at the end of the input.

        Returns:
            tuple[int, int]: (data channels: target + covariates, time channels).
        """
        n_features = sum(
            len(x.subset_features) if x.subset_features else x.n_features
            for x in {k.split("_")[0]: v for k, v in self.features.items()}.values()
        )
        n_time_features = sum(
            self.features[name].n_features for name in self.feature_order if self.features[name].loader == "time"
        )
        return n_features - n_time_features, n_time_features


class StefBeamBenchmark(StrictModel):
    """DOCSTRING."""

    targets_file: str = Field(..., description="Targets file for StefBeam.")
    context_window_days: PositiveInt = Field(32, description="Context window size in days.")
    predict_window_days: PositiveInt = Field(2, description="Prediction window size in days.")
    input_sample_interval_minutes: PositiveInt = Field(15, description="stef beam benchmark sample rate")


class LocalBenchmark(StrictModel):
    """DOCSTRING."""

    locations: list[str] = Field(
        [
            "Ameland||benchmark",
            "Buurmalsen||benchmark",
            "Dronten||benchmark",
            "Middenmeer||benchmark",
            "Weesp||benchmark",
            "Westwoud||benchmark",
            "Buu-RS||10-G||V01-benchmark",
            "Buu-RS||10-G||V02-benchmark",
        ],
        description="Locations for local benchmark.",
    )
    thresholds: list[float] | None = Field(None, description="Thresholds for local benchmark.")
    start_date: datetime = Field(datetime(2022, 1, 1, tzinfo=UTC), description="Start date for benchmarking.")
    phase: int = Field(0, description="Phase for local benchmark.")
    n_day_ahead: PositiveInt = Field(1, description="Number of days ahead for local benchmark.")
    context_window_days: PositiveInt = Field(..., description="Context window size in days.")
    predict_window_days: PositiveInt = Field(..., description="Prediction window size in days.")
    predict_dim: PositiveInt = Field(0, description="Prediction dimension of dataset.")
    output_sample_interval_minutes: PositiveInt = Field(
        None, description="Sample interval in minutes for benchmarking output."
    )
    input_sample_interval_minutes: PositiveInt = Field(15, description="stef beam benchmark sample rate")
    alignment: ContextWindowAlignment | None = Field(None, description="Alignment for context window.")


# TODO
class GiftEvalBenchmark(StrictModel):
    """DOCSTRING."""

    source: str = Field(..., description="GiftEval source identifier.")
    input_sample_interval_minutes: PositiveInt = Field(15, description="stef beam benchmark sample rate")


FEV_BENCH_REF = "v0.10.0"
"""fev release whose benchmark definition and published baseline results are used by default."""
FEV_BENCH_URL = f"https://raw.githubusercontent.com/autogluon/fev/{FEV_BENCH_REF}/benchmarks/fev_bench"


class FevBenchBenchmark(StrictModel):
    """fev-bench (Shchur et al., 2025) evaluation through the ``fev`` library.

    Every selected task in ``tasks_file`` is forecast zero-shot with the model under training and scored by
    ``fev`` (SQL, MASE, WQL, WAPE). Results are written as fev summary CSVs and, when ``baseline_results`` are
    given, ranked against those published baselines with ``fev.leaderboard`` (skill score and win rate).
    The task list and the baselines are read from the fev repository at a pinned release by default.
    """

    tasks_file: str = Field(
        f"{FEV_BENCH_URL}/tasks.yaml",
        description="fev benchmark YAML (path or URL): a `tasks:` list of fev.Task definitions.",
    )
    task_subset: list[str] | Literal["energy", "all"] = Field(
        "energy",
        description="Which tasks of the file to run: 'energy' (the 26 Energy-domain tasks of the fev-bench "
        "paper), 'all', or an explicit list of `dataset_config` names.",
    )
    num_windows: PositiveInt | None = Field(
        None,
        description="Override the rolling windows per task for cheaper runs. Changes the task identity, so "
        "the leaderboard against the published baselines is skipped when set.",
    )
    context_window_days: PositiveInt | None = Field(
        None,
        description="History handed to the model in days. None uses io.interval_context_limits[rate].max_days, "
        "i.e. the longest context the model was trained with at that sample rate.",
    )
    max_context_samples: PositiveInt | None = Field(
        None,
        description="Budget for the total window (history + horizon) handed to the model, in samples. For the "
        "s4casting models it replaces the longest-trained-window cap (a series shorter than the budget still "
        "keeps its own length), decoupling the benchmark context from the model.context_window / "
        "io.interval_context_limits training settings; for a model that declares its own context length "
        "(Chronos-2: 8192, Timestral: 32768) the smaller of the two applies. None keeps the trained windows "
        "or the model's own cap.",
    )
    model_name: str = Field("s4casting", description="Model name recorded in the fev summaries.")
    datasets_prefix: str | None = Field(
        None,
        description="Local or S3 mirror of autogluon/fev_datasets laid out as <prefix>/<dataset_config>/*.parquet "
        "(exported as FEV_DATASETS_PREFIX). None downloads from the Hugging Face Hub.",
    )
    num_proc: PositiveInt = Field(1, description="Processes used by `datasets` while preparing each task.")
    baseline_results: list[str] = Field(
        [f"{FEV_BENCH_URL}/results/seasonal_naive.csv", f"{FEV_BENCH_URL}/results/chronos-2.csv"],
        description="fev result CSVs (paths or URLs) to rank against; one must contain `baseline_model`.",
    )
    baseline_model: str = Field(
        "Seasonal Naive", description="`model_name` in `baseline_results` that anchors the skill score."
    )
    n_resamples: PositiveInt = Field(1000, description="Bootstrap resamples for the leaderboard confidence intervals.")
    trained_on_datasets: list[str] = Field(
        [],
        description="`dataset_config` names that overlap the training corpus; flagged as leakage in the summaries.",
    )
    output_dir: str = Field("fev-bench", description="Subdirectory of io.output for the summary CSVs.")


class BenchmarkingConfiguration(StrictModel):
    """Shared + named benchmark presets."""

    eval_quantiles: list[float] = Field(
        [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
        description="Evaluation quantiles for benchmarking.",
    )
    batch_size: PositiveInt = Field(
        64,
        description="Series per forward pass in every benchmark (local, stef-beam, fev-bench). Benchmarks run "
        "without gradients, so this can be far larger than training.batch_size; it only trades speed for memory.",
    )

    benchmarks: dict[str, StefBeamBenchmark | LocalBenchmark | GiftEvalBenchmark | FevBenchBenchmark] = Field(
        default_factory=dict,
        description="Dictionary of named benchmark presets.",
    )


class ValidationConfiguration(StrictModel):
    """Validation configuration."""

    split_type: Literal["time", "random", "location"] = Field("time", description="Type of split for validation.")
    start_year: PositiveInt = Field(2024, description="Start year for validation if split_type is 'time'.")
    percentage: PositiveInt = Field(5, description="Percentage for validation.")


class CPMConfiguration(StrictModel):
    """Contiguous Patch Masking (CPM) configuration, after TiRex (Auer et al., 2025).

    Per training sample a block length ``continuous_mask ~ U{continuous_min, ..., continuous_max}`` (in model
    patches) and a masking probability ``p_mask ~ U(0, p_max)`` are drawn. The context is cut into blocks of
    ``continuous_mask`` patches and each block of the target series is masked with probability ``p_mask``.
    Masked blocks are hidden from the model input and added to the loss target, on top of the fixed forecast
    window at the end of the sample.

    The block-length bounds and the protected leading history are configured as durations and converted to
    patches per sample rate by :meth:`patches_by_rate`, so multi-rate training masks comparable stretches of
    history at every rate.
    """

    p_max: float = Field(
        0.25,
        ge=0.0,
        le=1.0,
        description="Upper bound of the per-sample block masking probability; p_mask ~ U(0, p_max).",
    )
    continuous_max_days: float = Field(
        2.0,
        gt=0.0,
        description="Upper bound of the length of one contiguous mask block, as a duration in days. Converted "
        "to patches per sample rate (at least one patch): 2 days is 24 patches of 8 at 15 minutes, 6 patches "
        "at hourly rate and one 8-day patch at daily rate.",
    )
    continuous_min_days: float | None = Field(
        None,
        gt=0.0,
        description="Lower bound of the length of one contiguous mask block, as a duration in days; None means "
        "one patch at every rate.",
    )
    min_context_days: float | None = Field(
        None,
        ge=0.0,
        description="Leading history of the sample that CPM never masks, as a duration in days, so under causal "
        "attention every masked block keeps at least this much visible history to predict from. None means "
        "no protected history.",
    )
    apply_to_validation: bool = Field(
        False,
        description="Also apply CPM to the validation set. Off by default so the validation loss "
        "stays comparable with runs that use the plain 'prediction' task.",
    )

    @model_validator(mode="after")
    def _check_bounds(self) -> "CPMConfiguration":
        """Check that the block length range is well formed.

        The per-field bounds (p_max in [0, 1], durations positive, min_context_days at least 0) are declared
        on the fields; this is the one cross-field rule. The task dataset relies on this validation and does
        not re-check the values.

        Raises:
            ValueError: If continuous_min_days exceeds continuous_max_days.

        Returns:
            CPMConfiguration: The validated configuration.
        """
        if self.continuous_min_days is not None and self.continuous_min_days > self.continuous_max_days:
            raise ValueError(
                f"training.cpm.continuous_min_days ({self.continuous_min_days}) must not exceed "
                f"continuous_max_days ({self.continuous_max_days})."
            )
        return self

    def patches_by_rate(self, patch_size: int, sample_intervals_minutes: list[int]) -> dict[int, tuple[int, int, int]]:
        """Resolve the block-length bounds and the protected history to patches for every sample rate.

        Args:
            patch_size (int): Model patch size in samples.
            sample_intervals_minutes (list[int]): Sample rates that will be trained.

        Returns:
            dict[int, tuple[int, int, int]]: rate -> (continuous_min, continuous_max, min_context_patches).
        """

        def patches(days: float, rate: int, floor: int) -> int:
            return max(floor, round(days * 24 * 60 / (rate * patch_size)))

        return {
            rate: (
                1 if self.continuous_min_days is None else patches(self.continuous_min_days, rate, 1),
                patches(self.continuous_max_days, rate, 1),
                0 if self.min_context_days is None else patches(self.min_context_days, rate, 0),
            )
            for rate in sample_intervals_minutes
        }


class TrainingConfiguration(StrictModel):
    """Training configuration."""

    task: Literal["prediction", "masking", "randomprediction", "cpm"] = Field(
        "prediction",
        description="Training task : 'prediction' (fixed window), "
        "'masking' (random masking), 'randomprediction' (random prediction window percentage), "
        "or 'cpm' (fixed window plus contiguous patch masking of the target inside the context).",
    )
    cpm: CPMConfiguration = Field(
        default_factory=CPMConfiguration,
        description="Contiguous Patch Masking settings, used when task='cpm'.",
    )
    precision: Literal["fp32", "bf16"] = Field(
        "fp32",
        description="Mixed precision for the training step: 'fp32' (full precision) or 'bf16' "
        "(recommended on H100: autocast compute in bfloat16; bf16 keeps fp32's exponent range, so "
        "no loss scaling is needed). Master weights, normalization statistics and the loss always "
        "stay fp32, so checkpoints are precision-independent. Only supported for "
        "model='transformer'; on CPU it silently runs fp32. TF32 matmuls are always enabled on "
        "CUDA, independent of this setting.",
    )
    max_retries: NonNegativeInt = Field(
        10,
        description="Max retries for rejection sampling, invalid due to changing prediction/context windows."
        "Default 0 means no rejection sampling.",
    )

    gradient_accumulation_steps: int = Field(2, description="Number of gradient accumulation steps.")
    batch_size: int = Field(32, description="Batch size for training.")
    evaluation_interval: int = Field(50, description="Interval for evaluation during training.")
    checkpoint_interval: int = Field(1000, description="Interval for saving checkpoints during training.")
    benchmarking_interval: int = Field(100, description="Interval for benchmarking during training.")
    maximum_steps: int = Field(10000, description="Maximum number of training steps.")
    n_samples_per_epoch: PositiveInt | None = Field(
        default=None, init=False, description="Populated at runtime with the number of samples per epoch."
    )
    iteration: int = Field(1, description="Current iteration number. Set to 0 for zero-shot performance.")
    covariate_dropout: bool = False
    covariate_dropout_all_p: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="With covariate_dropout, the probability that a training sample loses every covariate. "
        "The default draw keeps a uniform random number of covariates, which with many slots almost never "
        "removes all of them (2.6% with 38 slots), while a third of the fev-bench energy tasks carry none.",
    )
    covariate_horizon_dropout_p: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="With covariate_dropout, the probability that a kept covariate is hidden over (part of) the "
        "forecast window while staying visible in the context. Half of those draws hide the whole forecast window "
        "(a past-only covariate, as fev-bench's past_dynamic_columns are presented) and half hide it from a random "
        "point onward (a forecast that runs out before the horizon does). The context never loses a covariate the "
        "forecast window keeps, which is the only shape inference produces.",
    )


def validate_covariate_dropout(training: TrainingConfiguration, model: ModelConfiguration) -> None:
    """Check that covariate dropout is only configured for models that are trained with it.

    Chronos fine-tuning does not support it: the batcher presents a dropped covariate as an all-masked
    channel, which Chronos-2 attends to as an all-missing variate of its group and which degrades its
    forecasts. Supporting it would mean dropping such channels inside the wrapper, which is deliberately
    not done.

    Args:
        training (TrainingConfiguration): Training configuration.
        model (ModelConfiguration): Model configuration.

    Raises:
        ValueError: If covariate dropout is enabled for model='chronos'.
    """
    if training.covariate_dropout and model.model == "chronos":
        raise ValueError(
            "training.covariate_dropout is not supported for model='chronos': Chronos-2 attends to a dropped "
            "(all-masked) covariate as an all-missing variate. Fine-tune Chronos with covariate_dropout = false."
        )


def validate_precision(training: TrainingConfiguration, model: ModelConfiguration) -> None:
    """Check that the configured training precision is supported by the model architecture.

    Mixed precision is only wired for the transformer: the S4 kernel's FFT path and the Mamba
    selective scan are not validated under autocast, and Chronos manages its own precision.

    Args:
        training (TrainingConfiguration): Training configuration.
        model (ModelConfiguration): Model configuration.

    Raises:
        ValueError: If a half precision is configured for a non-transformer model.
    """
    if training.precision != "fp32" and model.model != "transformer":
        raise ValueError(
            f"training.precision='{training.precision}' is only supported for model='transformer'; "
            f"model='{model.model}' must train at fp32."
        )
    if model.internal_dtype != DType.Float32:
        warnings.warn(
            "model.internal_dtype no longer controls the training precision; use training.precision "
            "('fp32', 'bf16' or 'fp16') instead.",
            UserWarning,
            stacklevel=2,
        )


def get_data_range_spans(dataset_dict: DatasetConfiguration) -> tuple[datetime, datetime]:
    """Get the min and max datetime from the spans parquet file of a dataset.

    Args:
        dataset_dict (DatasetConfiguration): The dataset configuration.

    Returns:
        tuple[datetime, datetime]: The minimum and maximum datetimes in the dataset.
    """
    dataset_json_location = Path(dataset_dict.location)
    with Path(dataset_json_location).open(encoding="utf-8") as f:
        meta = json.load(f)

    df_spans = pd.read_parquet(dataset_json_location.parent / meta["spans"])
    interval_sec = int(meta["sample_interval_minutes"]) * 60
    starts = df_spans["datetime_start"]
    ends = starts + df_spans["num_values"] * interval_sec

    start_dt = datetime.fromtimestamp(starts.min(), tz=UTC)
    end_dt = datetime.fromtimestamp(ends.max(), tz=UTC)
    return start_dt, end_dt


class Configuration(BaseSettings):
    """Main configuration for S4 casting."""

    model_config = SettingsConfigDict(extra="forbid")

    machine: MachineConfiguration = Field(..., description="Machine settings.")
    io: IOConfiguration = Field(..., description="Input/Output settings.")
    authentication: AuthenticationConfiguration = Field(
        default_factory=AuthenticationConfiguration, description="Authentication settings."
    )
    logging: LoggingConfiguration = Field(default_factory=LoggingConfiguration, description="Logging settings.")
    model: ModelConfiguration = Field(default_factory=ModelConfiguration, description="Model settings.")
    run: RunConfiguration = Field(default_factory=RunConfiguration, description="Run settings.")
    optimizer: OptimizerConfiguration = Field(default_factory=OptimizerConfiguration, description="Optimizer settings.")
    scheduler: SchedulerConfiguration = Field(default_factory=SchedulerConfiguration, description="Scheduler settings.")
    training: TrainingConfiguration = Field(default_factory=TrainingConfiguration, description="Training settings.")
    benchmarking: BenchmarkingConfiguration = Field(
        default_factory=BenchmarkingConfiguration, description="Benchmarking settings."
    )
    validation: ValidationConfiguration = Field(
        default_factory=ValidationConfiguration, description="Validation settings."
    )
    metrics: MetricsConfiguration = Field(default_factory=MetricsConfiguration, description="Metrics settings.")

    @model_validator(mode="after")
    def _post_configuration_checks(self) -> "Configuration":
        """Perform post-initialization checks on the configuration.

        Raises:
            ValueError: If weather data does not cover the required date ranges.

        Returns:
            Configuration: The validated configuration object.
        """
        parquet_ranges: dict[str, tuple[datetime, datetime]] = {}
        for name, ds in self.io.features.items():
            if ds.loader == "parquet":
                pq_min, pq_max = get_data_range_spans(ds)
                parquet_ranges[name] = (pq_min, pq_max)

        if "weather" in self.io.features and self.io.features["weather"].loader == "parquet":
            wx_min, wx_max = get_data_range_spans(self.io.features["weather"])
            for name, (pq_min, pq_max) in parquet_ranges.items():
                if wx_min > pq_min:
                    raise ValueError(f"Weather starts later ({wx_min}) than '{name}' ({pq_min})")
                if wx_max < pq_max:
                    raise ValueError(f"Weather ends earlier ({wx_max}) than '{name}' ({pq_max})")
        validate_precision(self.training, self.model)
        validate_covariate_dropout(self.training, self.model)
        # The task/width compatibility checks live in _check_task_configuration, which supersedes
        # the inline checks the mixed-precision branch carried.
        self._check_task_configuration()
        return self

    def _check_task_configuration(self) -> None:
        """Check that the training task and the prediction window width are compatible.

        Raises:
            ValueError: If predict_width has the wrong type for the configured task.
        """
        task = self.training.task
        predict_width = self.model.predict_width
        if task == "randomprediction":
            if isinstance(predict_width, float):
                raise ValueError("training.task='randomprediction' requires predict_width to be int or list[int].")
            if isinstance(predict_width, list) and any(isinstance(x, int) for x in predict_width):
                raise ValueError("training.task='randomprediction' requires predict_width to be list[float].")
        elif task == "prediction" and not isinstance(predict_width, int):
            raise ValueError("training.task='prediction' requires predict_width to be int.")
        elif task == "cpm" and not isinstance(predict_width, int | tuple):
            raise ValueError("training.task='cpm' requires predict_width to be int or a (min, max) fraction range.")

        if task == "cpm" and self.model.model != "transformer":
            warnings.warn(
                "training.task='cpm' is designed for model='transformer', whose patch encoder feeds the mask "
                f"to the model. With model='{self.model.model}' the masked context patches only reach the model "
                "through the generic input mask, which has not been validated for CPM.",
                UserWarning,
                stacklevel=2,
            )
