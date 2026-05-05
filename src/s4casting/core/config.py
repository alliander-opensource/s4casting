# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt, PositiveInt, SecretStr, model_validator
from pydantic_settings import (
    BaseSettings,
)

from s4casting.data.dataset.interface import ContextWindowAlignment


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


class AuthenticationConfiguration(BaseModel):
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


class OptimizerConfiguration(BaseModel):
    """Optimizer configuration."""

    learning_rate: float = Field(1e-3, description="Base learning rate used by the optimizer.")
    beta1: float = Field(0.9, description="First momentum coefficient (RAdam β1).")
    beta2: float = Field(0.99, description="Second momentum coefficient (RAdam β2).")
    eps: float = Field(1e-8, description="Small constant for numerical stability.")
    weight_decay: float = Field(1e-2, description="L2 weight decay regularization strength.")
    gradient_clipping: float = Field(0.0, description="Gradient norm clipping threshold; 0 disables clipping.")


class SchedulerConfiguration(BaseModel):
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


class LossConfiguration(BaseModel):
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


class ChronosConfiguration(BaseModel):
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


class SSMConfiguration(BaseModel):
    """State Space Model configuration."""

    kernel: Literal["s4", "s6", "gru"] = Field(
        "s4",
        description="SSM kernel variant to use: 's4', 's6', or 'gru' baseline.",
    )
    n_layers: PositiveInt = Field(
        4,
        description="Number of stacked layers in the SSM/GRU block.",
    )


class TransformerConfiguration(BaseModel):
    """Transformer model configuration."""

    latent_dim: NonNegativeInt = Field(1024, description="Number of latent dimension.")
    n_heads: NonNegativeInt = Field(8, description="Number of attention heads per layer.")
    n_layers: NonNegativeInt = Field(6, description="Number of Transformer layers (encoder blocks).")
    dropout: float = Field(0.0, ge=0, lt=1, description="Dropout rate applied to attention/MLP.")
    is_causal: bool = Field(False, description="Whether to use causal attention.")
    attn_bias: bool = Field(True, description="Enable bias terms in attention projections.")
    mlp_bias: bool = Field(True, description="Enable bias terms in MLP layers.")
    mlp_layers: NonNegativeInt = Field(2, description="Number of MLP layers per block.")
    mlp_activation: Literal["identity", "tanh", "relu", "gelu", "elu", "silu", "glu", "sigmoid", "softplus"] = Field(
        "gelu",
        description="Activation function used in MLP blocks.",
    )
    use_cross_attention: bool = False
    context_n_layers: NonNegativeInt = 4


class OutputHeadConfiguration(BaseModel):
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


class PatchEncoderConfiguration(BaseModel):
    """Patch encoder configuration."""

    arch: Literal["linear", "gemma", "ss"] = Field(
        "linear",
        description="Patch encoder architecture: 'linear', 'gemma', or 'ss'.",
    )
    patch_size: PositiveInt = Field(8, description="Size of each patch.")
    n_layers: PositiveInt = Field(8, description="Number of layers in the patch encoder.")


class PatchDecoderConfiguration(BaseModel):
    """Patch decoder configuration."""

    arch: Literal["linear", "none"] = Field(
        "linear",
        description="Patch decoder architecture: 'linear' or 'none'.",
    )
    patch_size: PositiveInt = Field(8, description="Size of each patch.")
    n_layers: PositiveInt = Field(8, description="Number of layers in the patch decoder.")


class MetricsConfiguration(BaseModel):
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


class ModelConfiguration(BaseModel):
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
            "Forecast horizon. Either absolute days (int) or a percent range tuple "
            "(min_percent, max_percent) where each is in [0.05, 0.5]."
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
    latent_dim: NonNegativeInt = Field(256, description="Dimensionality of the latent space.")


class MachineConfiguration(BaseModel):
    """Machine configuration."""

    device_kind: Literal["cuda", "cpu", "mps"] = Field(
        "cpu", description="Kind of device to use: 'cuda', 'cpu', or 'mps'."
    )
    local_rank: int = Field(0, description="Local rank for distributed training.")
    ddp: bool = Field(False, description="Whether to use Distributed Data Parallel (DDP).")
    ddp_loss_sync: bool = Field(True, description="Whether to synchronize loss across DDP processes.")


class RunConfiguration(BaseModel):
    """Run configuration."""

    seed: int = Field(42069, description="Random seed for reproducibility.")
    run_start_date: str = Field(datetime.now(UTC).strftime("%Y-%m-%d"), description="Start date of the run.")
    log_level: LogLevel = Field(LogLevel.Info, description="Logging level.")
    persist_to_wandb_project: str | None = Field(None, description="WandB project name for persistence.")
    wandb_runid: str | None = Field(None, description="WandB run ID.")
    wandb_notes: str | None = Field(None, description="Notes for WandB run.")
    wandb_online: bool = Field(True, description="Online logging of wandb run (if you have internet access)")


class DatasetConfiguration(BaseModel):
    """Dataset configuration."""

    location: str = Field(..., description="Location of the dataset.")
    loader: Literal["sqlite", "parquet", "time", "sideload", "croissant"] = Field(
        "sqlite", description="Type of loader to use for the dataset."
    )
    nearest_neighbor: bool = Field(False, description="Apply nearest-neighbour spatial matching (e.g. weather grids).")
    n_features: int = Field(1, description="Number of features in the dataset.")
    main_dataset: str = Field("", description="Name of the main dataset.")
    subset_features: list[str] = Field([], description="List of subset features to use from the dataset.")


class IOConfiguration(BaseModel):
    """IO configuration."""

    feature_order: list[str] = Field(..., description="Order of features/datasets for input/output.")
    features: dict[str, DatasetConfiguration] = Field(..., description="Dictionary of (feature) datasets.")
    output: str = Field(..., description="Location to save outputs.")
    load_checkpoint: str | None = Field(None, description="Path to load checkpoint from.")
    iteration: int = Field(0, description="Current iteration number.")
    gap_skip_hours: int = Field(1, description="Number of hours to skip for gaps.")
    context_window_valid_ratio: float = Field(0.8, description="Valid ratio for input window.")
    hash_datasets: bool = Field(False, description="Whether to hash the datasets to be logged")
    to_memory: bool = Field(False, description="Whether to move the memmap'd data to CPU memory")


class StefBeamBenchmark(BaseModel):
    """DOCSTRING."""

    targets_file: str = Field(..., description="Targets file for StefBeam.")
    context_window_days: PositiveInt = Field(32, description="Context window size in days.")
    predict_window_days: PositiveInt = Field(2, description="Prediction window size in days.")
    input_sample_interval_minutes: PositiveInt = Field(15, description="stef beam benchmark sample rate")


class LocalBenchmark(BaseModel):
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
class GiftEvalBenchmark(BaseModel):
    """DOCSTRING."""

    source: str = Field(..., description="GiftEval source identifier.")
    input_sample_interval_minutes: PositiveInt = Field(15, description="stef beam benchmark sample rate")


class BenchmarkingConfiguration(BaseModel):
    """Shared + named benchmark presets."""

    eval_quantiles: list[float] = Field(
        [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
        description="Evaluation quantiles for benchmarking.",
    )

    benchmarks: dict[str, StefBeamBenchmark | LocalBenchmark | GiftEvalBenchmark] = Field(
        default_factory=dict,
        description="Dictionary of named benchmark presets.",
    )


class ValidationConfiguration(BaseModel):
    """Validation configuration."""

    split_type: Literal["time", "random", "location"] = Field("time", description="Type of split for validation.")
    start_year: PositiveInt = Field(2024, description="Start year for validation if split_type is 'time'.")
    percentage: PositiveInt = Field(5, description="Percentage for validation.")


class TrainingConfiguration(BaseModel):
    """Training configuration."""

    task: Literal["prediction", "masking", "randomprediction"] = Field(
        "prediction",
        description="Training task : 'prediction' (fixed window), "
        "'masking' (random masking), or 'randomprediction' (random prediction window percentage).",
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

    machine: MachineConfiguration = Field(..., description="Machine settings.")
    io: IOConfiguration = Field(..., description="Input/Output settings.")
    authentication: AuthenticationConfiguration = Field(
        default_factory=AuthenticationConfiguration, description="Authentication settings."
    )
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
        # apply check for prediction mode and prediciton window widths:
        if self.training.task == "randomprediction":
            if isinstance(self.model.predict_width, float):
                raise ValueError("training.task='randomprediction' requires predict_width to be int or list[int].")
            if isinstance(self.model.predict_width, list) and any(isinstance(x, int) for x in self.model.predict_width):
                raise ValueError("training.task='randomprediction' requires predict_width to be list[float].")
        elif self.training.task == "prediction":
            if isinstance(self.model.predict_width, float):
                raise ValueError("training.task='prediction' requires predict_width to be int.")
            if isinstance(self.model.predict_width, list):
                raise ValueError("training.task='prediction' requires predict_width to be int.")

        return self
