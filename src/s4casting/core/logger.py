# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import csv
import hashlib
import logging
import os
import re
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar

import mlflow
import numpy as np
import pandas as pd
import wandb
from mlflow import MlflowClient  # type: ignore[possibly-missing-import]  # guarded export, ty cannot see it
from mlflow.exceptions import MlflowException
from openstef_beam.benchmarking import read_evaluation_reports
from plotly.graph_objects import Figure
from tqdm import tqdm

from s4casting.core.config import (
    AuthenticationConfiguration,
    MLflowLoggingConfiguration,
    RunConfiguration,
    TrainingConfiguration,
    WandbLoggingConfiguration,
    WandbMode,
)
from s4casting.core.context import Context
from s4casting.core.hooks import CommonHooks, TrainingHooks
from s4casting.eval.external_data_eval import (
    build_summary_from_reports,
    get_external_benchmarks,
)


class LoggerInterface:
    """Base interface for all logger implementations."""

    def __init__(self, hookable: CommonHooks | TrainingHooks) -> None:
        """Initialize the LoggerInterface.

        Args:
            hookable (CommonHooks | TrainingHooks): Hookable object to register hooks.
        """
        hookable.finished.register(self.finished)

        if isinstance(hookable, TrainingHooks):
            hookable.evaluate.register(self.evaluate)
            hookable.benchmark_metrics.register(self.benchmark_metrics)

    def finished(self, context: Context) -> None:
        """Run when training is finished.

        Args:
            context (Context): Training context.
        """
        self.report_eval(context, iteration=context.configuration.training.maximum_steps)

    def evaluate(self, context: Context, iteration: int) -> None:
        """Run when evaluation is triggered.

        Args:
            context (Context): Training context.
            iteration (int): Current training iteration.
        """
        self.report_eval(context, iteration)

    def benchmark_metrics(self, context: Context, iteration: int) -> None:
        """Run when benchmark metrics are triggered.

        Args:
            context (Context): Training context.
            iteration (int): Current training iteration.
        """
        pass

    def report_eval(self, context: Context, iteration: int | None) -> None:
        """Report evaluation metrics.

        Args:
            context (Context): Training context.
            iteration (int | None): Current training iteration.
        """
        pass


class CSVLogger(LoggerInterface):
    """Logger saving benchmark metrics to CSV."""

    def __init__(
        self, hookable: CommonHooks | TrainingHooks, run_config: RunConfiguration, output_dir: str = "results"
    ) -> None:
        """Initialize the CSVLogger.

        Args:
            hookable (CommonHooks | TrainingHooks): Hookable object to register hooks.
            run_config (RunConfiguration): Run configuration.
            output_dir (str): Directory to save CSV file.
        """
        super().__init__(hookable)
        try:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.headers = None
            # name csv based on seed initialization
            self.csv_file = self.output_dir / f"benchmark_metrics_{run_config.seed}_{run_config.run_start_date}.csv"

            if isinstance(hookable, TrainingHooks):
                hookable.benchmark_metrics.register(self.save_metrics)

        except Exception as e:
            raise Exception(f"Failed to initialize CSVLogger: {e!s}")

    def _get_flattened_metrics(self, metrics: dict, location: str) -> dict:
        """Flatten nested metrics dict and retrieve values.

        Args:
            metrics: value dict
            location: location for benchmark signal

        Returns:
            flattened dict of metrics
        """
        flattened = {"iteration": self.current_iteration, "location": location}

        for key, value in metrics.items():
            if isinstance(value, dict):
                for quantile, qvalue in value.items():
                    column_name = f"{key}_{quantile}"
                    flattened[column_name] = qvalue
            else:
                flattened[key] = value

        return flattened

    def save_metrics(self, context: Context, iteration: int) -> None:
        """Save benchmark metrics to CSV file.

        Args:
            context: training context containing metrics
            iteration: current training iteration
        """
        try:
            self.current_iteration = iteration
            metrics = context.benchmark_metrics
            location = context.benchmark_location

            row_data = self._get_flattened_metrics(metrics, location)

            # initialize headers for csv
            if self.headers is None:
                self.headers = list(row_data.keys())
                with self.csv_file.open("w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.headers)

            with self.csv_file.open("a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([row_data[header] for header in self.headers])

        except Exception as e:
            raise Exception(f"Error saving metrics to CSV: {e!s}")


class StdLogger(LoggerInterface):
    """Standard output logger using tqdm for progress bar."""

    def __init__(self, hookable: CommonHooks | TrainingHooks, config: TrainingConfiguration) -> None:
        """Initialize the StdLogger.

        Args:
            hookable (CommonHooks | TrainingHooks): Hookable object to register hooks.
            config (TrainingConfiguration): Training configuration.
        """
        self.progress_bar = tqdm(total=config.maximum_steps, desc="Training", dynamic_ncols=True)
        super().__init__(hookable)
        if isinstance(hookable, TrainingHooks):
            hookable.step.register(self.report_eval)
            hookable.eval_plot.register(self.save_eval_plot)
            hookable.benchmark_plot.register(self.save_benchmark_plot)

    def report_eval(self, context: Context, iteration: int | None) -> None:
        """Report evaluation metrics to standard output.

        Args:
            context (Context): Training context.
            iteration (int | None): Current training iteration.
        """
        # hack to get progress_bar and iterations to track
        self.progress_bar.update(iteration - self.progress_bar.n)
        metric_name = f"validation_loss_{context.input_validation_sample_rate}_{context.output_validation_sample_rate}"
        if bool(context.eval_metrics):  # check if metrics have been computed
            to_print = {
                "iteration": context.trainer.iteration,  # type: ignore[attr-defined]
                "loss": context.loss,
                metric_name: context.validation_loss,
            }
            for key, value in context.eval_metrics.items():  # -2 for 90th percentile
                to_print[f"validation_{key.capitalize()}"] = value[-2] if isinstance(value, np.ndarray) else value

        else:
            to_print = {
                "iteration": context.trainer.iteration,  # type: ignore[union-attr]
                "loss": context.loss,
                metric_name: context.validation_loss,
            }

        if bool(context.benchmark_metrics):
            for key, value in context.benchmark_metrics.items():  # -2 for 90th percentile
                if isinstance(value, dict):
                    # log specific metrics for plotting
                    for q, _metric in value.items():
                        to_print[f"benchmark_{context.benchmark_location}_{key.capitalize()}_Q{q:.2f}"] = _metric
                else:
                    to_print[f"benchmark_{context.benchmark_location}_{key.capitalize()}"] = value
        self.progress_bar.set_postfix(to_print)

    def save_eval_plot(self, context: Context, iteration: int | None, fig: Figure, outdir="plots") -> None:
        """Save evaluation plot to disk.

        Args:
            context (Context): Training context.
            iteration (int | None): Current training iteration.
            fig (Figure): Plotly figure to save.
            outdir (str): Output directory to save plots.
        """
        Path(outdir).mkdir(parents=True, exist_ok=True)
        fig.write_image(
            f"{outdir}/eval_{context.input_validation_sample_rate}_{context.output_validation_sample_rate}_{iteration}.png"
        )

    def save_benchmark_plot(
        self,
        context: Context,
        iteration: int | None,
        fig: Figure,
        plot_type,
        outdir="plots",
    ) -> None:
        """Save benchmark plot to disk.

        Args:
            context (Context): Training context.
            iteration (int | None): Current training iteration.
            fig (Figure): Plotly figure to save.
            plot_type: Type of benchmark plot.
            outdir (str): Output directory to save plots.
        """
        Path(outdir).mkdir(parents=True, exist_ok=True)
        fig.write_image(f"{outdir}/benchmark_{context.benchmark_location}_{plot_type}_{iteration}.png")


class WandbLogger(LoggerInterface):
    """Wandb logger for tracking experiments."""

    def __init__(
        self,
        hookable: CommonHooks | TrainingHooks,
        config: WandbLoggingConfiguration,
        run_config: RunConfiguration,
        output_dir: str = "results",
        auth: AuthenticationConfiguration | None = AuthenticationConfiguration(),
    ) -> None:
        """Initialize the WandbLogger.

        Args:
            hookable (CommonHooks | TrainingHooks): Hookable object to register hooks.
            config: Weights & Biases logging configuration.
            run_config (RunConfiguration): Run configuration.
            output_dir (str): Directory containing CSV benchmark results.
            auth (AuthenticationConfiguration | None): Authentication configuration.
        """
        self.config = config
        self.mode = config.mode
        if config.mode is WandbMode.Online:
            if auth is None or auth.wandb_api_key is None:
                raise ValueError("WandB online mode requires an API key, but none was provided")
            wandb.login(key=auth.wandb_api_key.get_secret_value())
        super().__init__(hookable)
        output_dir: Path = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_file = output_dir / f"benchmark_metrics_{run_config.seed}_{run_config.run_start_date}.csv"

        hookable.start.register(self.start)

        if isinstance(hookable, TrainingHooks):
            hookable.checkpoint.register(self.checkpoint)
            hookable.step.register(self.step)
            hookable.eval_plot.register(self.save_eval_plot)
            hookable.benchmark_plot.register(self.save_benchmark_plot)
            hookable.benchmark_metrics.register(self.report_benchmark)
            hookable.benchmark_complete.register(self.benchmark_complete)

    def start(self, context: Context) -> None:
        """Initialize wandb run.

        Args:
            context (Context): Training context.
        """
        wandb.init(
            project=self.config.project,
            mode=self.mode,
            config=context.tracking_parameters,
            id=self.config.run_id,
            resume="allow",
            notes=self.config.notes,
        )

    def finished(self, context: Context) -> None:
        """Log final metrics and cleanly close wandb."""
        # This will call WandbLogger.report_eval via LoggerInterface
        super().finished(context)

        try:
            wandb.finish()
        except Exception as e:
            logging.warning("wandb.finish() failed during shutdown: %s", e)

    def save_eval_plot(self, context: Context, iteration: int | None, fig: Figure) -> None:
        """Save evaluation plot to wandb.

        Args:
            context (Context): Training context.
            iteration (int | None): Current training iteration.
            fig (Figure): Plotly figure to save.
        """
        plot_name = (
            f"training_plot_sample_rate={context.input_validation_sample_rate}_{context.output_validation_sample_rate}"
        )
        wandb.log(
            {plot_name: wandb.Plotly(fig)},
            step=iteration,
        )

    def save_benchmark_plot(
        self,
        context: Context,
        iteration: int | None,
        fig: Figure,
        plot_type,
    ) -> None:
        """Save benchmark plot to wandb.

        Args:
            context (Context): Training context.
            iteration (int | None): Current training iteration.
            fig (Figure): Plotly figure to save.
            plot_type: Type of benchmark plot.
        """
        with suppress(Exception):
            wandb.log(
                {f"benchmark_{context.benchmark_location}_{plot_type}": wandb.Plotly(fig)},
                step=iteration,
            )

    def report_eval(self, context: Context, iteration: int | None) -> None:
        """Report evaluation metrics to wandb.

        Args:
            context (Context): Training context.
            iteration (int | None): Current training iteration.
        """
        for key, value in context.eval_metrics.items():  # -2 for 90th percentile
            if not isinstance(value, dict):
                metric_name = (
                    f"validation_{key}_{context.input_validation_sample_rate}_{context.output_validation_sample_rate}"
                )
                wandb.log(
                    {metric_name: value},
                    step=iteration,
                )
        metric_name = f"validation_loss_{context.input_validation_sample_rate}_{context.output_validation_sample_rate}"

        wandb.log(
            {metric_name: context.validation_loss},
            step=iteration,
        )

    def report_benchmark(self, context: Context, iteration: int | None) -> None:
        """Report benchmark metrics to wandb.

        Args:
            context (Context): Training context.
            iteration (int | None): Current training iteration.
        """
        columns, rows = [], np.array([])
        for key, value in context.benchmark_metrics.items():
            # Append data for wandb table
            if isinstance(value, dict):
                columns.append(key)
                np_vals = np.array(tuple(value.values()))[None, ...]
                rows = np_vals if rows.size == 0 else np.concatenate([rows, np_vals])

                # log specific metrics for plotting
                for q, _metric in value.items():
                    wandb.log(
                        {f"benchmark_{context.benchmark_location}_{key}_Q{q:.2f}": _metric},
                        step=iteration,
                    )
            else:
                wandb.log(
                    {f"benchmark_{context.benchmark_location}_{key}": value},
                    step=iteration,
                )
        if rows.size != 0:
            table = wandb.Table(columns=columns, data=rows.T)
            wandb.log(
                {f"benchmark_{context.benchmark_location}_quantile_scores": table},
                step=iteration,
            )

    def checkpoint(self, context: Context, _iteration: int) -> None:
        """Checkpoint the model and optimizer to wandb at the current iteration.

        Args:
            context (Context): Training context.
            _iteration (int): Current training iteration.
        """
        if context.checkpointer.last_checkpoint is not None:  # type: ignore[union-attr]
            wandb.save(context.checkpointer.last_checkpoint.as_local_path())  # type: ignore[union-attr]

    def step(self, context: Context, iteration: int | None) -> None:
        """Log training loss to wandb at each step.

        Args:
            context (Context): Training context.
            iteration (int | None): Current training iteration.
        """
        wandb.log({"training_loss": context.loss, "epoch": context.trainer.epoch}, step=iteration)  # type: ignore[union-attr]

    @staticmethod
    def log_html_panel(name: str, html_path: Path, iteration: int) -> None:
        """Log HTML panel to wandb.

        Args:
            name (str): Name of the panel.
            html_path (Path): Path to the HTML file.
            iteration (int): Current training iteration.
        """
        html_text = html_path.read_text(encoding="utf-8", errors="ignore")
        wandb.log({name: wandb.Html(html_text)}, step=iteration)

    def benchmark_complete(self, context: Context, iteration: int) -> None:
        """Log benchmark completion metrics to wandb.

        Args:
            context (Context): Training context.
            iteration (int): Current training iteration.
        """
        # Check if we are in the medium term forecasting domain
        if context.configuration.benchmarking.benchmarks.get("LocalBenchmark") is not None:
            df = get_external_benchmarks(self.csv_file, iteration)
            if "ldn_monthly_mape" in df.columns:
                wandb.log({"mean_predicted_ldn_monthly_mape": df["ldn_monthly_mape"].abs().mean()}, step=iteration)
            if "odn_monthly_mape" in df.columns:
                wandb.log({"mean_predicted_odn_monthly_mape": df["odn_monthly_mape"].abs().mean()}, step=iteration)

        # else, we'll log stef-beam metrics
        if context.configuration.benchmarking.benchmarks.get("StefBeamBenchmark") is not None:
            metric_keys: dict[str, str] = {
                "F2.0": "f2",
                "effective_precision": "effective_precision",
                "effective_recall": "effective_recall",
                "precision": "precision",
                "recall": "recall",
                "rCRPS": "rCRPS",
                "rMAE": "rMAE",
            }

            if hasattr(context, "stefbeam_targets") and hasattr(context, "stefbeam_storage"):
                reports = read_evaluation_reports(
                    targets=context.stefbeam_targets,  # type: ignore[union-attr]
                    storage=context.stefbeam_storage,  # type: ignore[union-attr]
                    run_name=context.stefbeam_run_name,  # type: ignore[union-attr]
                )

                # get available_at filters
                available_ats = str(reports[0][1].subset_reports[0].filtering)

                df_summary = build_summary_from_reports(
                    reports=reports,
                    metrics_wanted=set(metric_keys.keys()),
                    filtering=available_ats,
                )

                to_log = {}
                for _, r in df_summary.iterrows():
                    c = metric_keys[r["metric"]]
                    k = f"mean_{c}_{r['group']}_Q{r['quantile']}"
                    v = r["mean"]
                    to_log[k] = float(v)

                wandb.log(to_log, step=iteration)

                # log all html panels
                if available_ats:
                    for p in [
                        "best_f2",
                        "precision_at_best_f2",
                        "recall_at_best_f2",
                        "rCRPS_grouped",
                        "rMAE_grouped",
                    ]:
                        fname = p + ".html"
                        html_path = context.stefbeam_results_path / available_ats / fname  # type: ignore[union-attr]
                        if html_path.exists():
                            self.log_html_panel(p, html_path, iteration)


class MLflowLogger(LoggerInterface):
    """MLflow logger for tracking experiments."""

    _MAX_METRIC_NAME_LENGTH: ClassVar[int] = 250
    _PARAM_BATCH_SIZE: ClassVar[int] = 100
    _INVALID_METRIC_NAME_CHARACTER: ClassVar[re.Pattern[str]] = re.compile(
        r"[^\w.\- /]" if os.name == "nt" else r"[^\w.\- /:]"
    )
    _PREFLIGHT_ENV: ClassVar[dict[str, str]] = {
        "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "0",
        "MLFLOW_HTTP_REQUEST_TIMEOUT": "5",
    }

    def __init__(
        self,
        hookable: CommonHooks | TrainingHooks,
        config: MLflowLoggingConfiguration,
        run_config: RunConfiguration,
        output_dir: str = "results",
    ) -> None:
        """Initialize the MLflow logger.

        Args:
            hookable: Hookable object to register hooks.
            config: MLflow logging configuration.
            run_config: Shared run configuration.
            output_dir: Directory containing CSV benchmark results.
        """
        self.config = config
        self.run_id: str | None = None
        self.metric_names: dict[str, str] = {}
        self.preflight(config)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.csv_file = output_path / f"benchmark_metrics_{run_config.seed}_{run_config.run_start_date}.csv"

        super().__init__(hookable)

        hookable.start.register(self.start)
        if isinstance(hookable, TrainingHooks):
            hookable.checkpoint.register(self.checkpoint)
            hookable.step.register(self.step)
            hookable.eval_plot.register(self.save_eval_plot)
            hookable.benchmark_plot.register(self.save_benchmark_plot)
            hookable.benchmark_metrics.register(self.report_benchmark)
            hookable.benchmark_complete.register(self.benchmark_complete)

    @classmethod
    def preflight(cls, config: MLflowLoggingConfiguration) -> None:
        """Validate the MLflow service and prepare its workspace and experiment.

        Args:
            config: MLflow logging configuration.

        """
        previous_env = {key: os.environ.get(key) for key in cls._PREFLIGHT_ENV}
        os.environ.update(cls._PREFLIGHT_ENV)
        mlflow.set_tracking_uri(config.uri)
        client = MlflowClient(tracking_uri=config.uri)

        if config.workspace != "default":
            logging.warning(
                "Using a non-default MLflow workspace (%s) may cause images in the img_grid UI panel to not be shown. "
                "See https://github.com/mlflow/mlflow/issues/22794"
            )
        try:
            try:
                client.get_workspace(config.workspace)
            except MlflowException as exc:
                if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
                    raise

                try:
                    client.create_workspace(
                        config.workspace,
                    )
                except MlflowException as exc:
                    if exc.error_code != "RESOURCE_ALREADY_EXISTS":
                        raise

            mlflow.set_workspace(config.workspace)  # type: ignore[possibly-missing-attribute]
            mlflow.set_experiment(config.experiment)
        finally:
            for key, value in previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    @classmethod
    def _sanitize_metric_name(cls, name: str) -> str:
        """Return a valid, path-safe MLflow metric name.

        Args:
            name: Original metric name.

        Returns:
            Sanitized metric name.

        Raises:
            TypeError: If the metric name is not a string.
        """
        if not isinstance(name, str):
            raise TypeError(f"MLflow metric name must be a string, got {name!r}")

        segments = []
        for segment in name.split("/"):
            sanitized_segment = cls._INVALID_METRIC_NAME_CHARACTER.sub("_", segment)
            segments.append("_" if sanitized_segment in {"", ".", ".."} else sanitized_segment)

        sanitized_name = "/".join(segments)
        if len(sanitized_name) > cls._MAX_METRIC_NAME_LENGTH:
            digest = hashlib.sha256(sanitized_name.encode()).hexdigest()[:8]
            prefix_length = cls._MAX_METRIC_NAME_LENGTH - len(digest) - 1
            sanitized_name = f"{sanitized_name[:prefix_length]}_{digest}"
        return sanitized_name

    @staticmethod
    def _sanitize_artifact_component(name: str) -> str:
        """Return a safe single artifact path component.

        Args:
            name: Original artifact component.

        Returns:
            Sanitized artifact component.
        """
        sanitized = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
        return "_" if sanitized in {"", ".", ".."} else sanitized

    def _prepare_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Sanitize metric names and reject collisions.

        Metric values are intentionally left untouched so MLflow can apply its
        supported scalar conversion rules to the complete batch.

        Args:
            metrics: Metrics keyed by their s4casting names.

        Returns:
            Metrics keyed by valid MLflow names.

        Raises:
            ValueError: If distinct metric names resolve to the same MLflow name.
        """
        prepared = {}
        for original_name, value in metrics.items():
            sanitized_name = self._sanitize_metric_name(original_name)
            previous_name = self.metric_names.get(sanitized_name)
            if previous_name is not None and previous_name != original_name:
                raise ValueError(
                    f"MLflow metric names {previous_name!r} and {original_name!r} both resolve to {sanitized_name!r}"
                )
            self.metric_names[sanitized_name] = original_name
            prepared[sanitized_name] = value
        return prepared

    @staticmethod
    def _flatten_params(params: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        """Flatten nested configuration dictionaries for MLflow parameters.

        Args:
            params: Nested parameter dictionary.
            prefix: Prefix accumulated during recursion.

        Returns:
            Flat dot-separated parameter mapping.
        """
        flattened = {}
        for key, value in params.items():
            name = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flattened.update(MLflowLogger._flatten_params(value, name))
            else:
                flattened[name] = value
        return flattened

    def _log_metrics(self, metrics: dict[str, Any], iteration: int | None) -> None:
        """Log a metric batch after sanitizing its names.

        Args:
            metrics: Metrics to log.
            iteration: Training iteration.
        """
        mlflow.log_metrics(  # type: ignore[possibly-missing-attribute]
            self._prepare_metrics(metrics),
            step=0 if iteration is None else iteration,
            synchronous=True,
        )

    def _log_figure(self, name: str, fig: Figure, iteration: int | None) -> None:
        """Log an interactive artifact and a time-stepped MLflow image.

        Args:
            name: Figure name.
            fig: Plotly figure.
            iteration: Current training iteration.
        """
        step = 0 if iteration is None else iteration
        filename = self._sanitize_artifact_component(name)
        artifact_root = f"steps/step-{step:08d}/figures/{filename}"
        mlflow.log_figure(fig, f"{artifact_root}.html")  # type: ignore[possibly-missing-attribute]
        # MLflow does not natively save plotly figures, or byte-level images (fig.to_image('png')),
        # so we need to save as PNG and then log.
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / f"{filename}.png"
            fig.write_image(str(image_path))
            mlflow.log_image(  # type: ignore[possibly-missing-attribute]
                mlflow.Image(str(image_path)),  # type: ignore[possibly-missing-attribute]
                key=f"figures/{filename}",
                step=step,
                synchronous=True,
            )

    def start(self, context: Context) -> None:
        """Start an MLflow run and log non-secret configuration.

        Args:
            context: Training context.
        """
        mlflow.set_tracking_uri(self.config.uri)
        mlflow.set_workspace(self.config.workspace)  # type: ignore[possibly-missing-attribute]
        mlflow.set_experiment(self.config.experiment)
        run = mlflow.start_run(log_system_metrics=True)  # type: ignore[possibly-missing-attribute]
        self.run_id = run.info.run_id

        flattened = self._flatten_params(context.tracking_parameters)
        items = list(flattened.items())
        for start in range(0, len(items), self._PARAM_BATCH_SIZE):
            mlflow.log_params(dict(items[start : start + self._PARAM_BATCH_SIZE]), synchronous=True)  # type: ignore[possibly-missing-attribute]

    def finished(self, context: Context) -> None:
        """Log final metrics and mark the MLflow run finished.

        Args:
            context: Training context.
        """
        super().finished(context)
        if self.run_id is not None:
            mlflow.end_run(status="FINISHED")  # type: ignore[possibly-missing-attribute]
            self.run_id = None

    def save_eval_plot(self, context: Context, iteration: int | None, fig: Figure) -> None:
        """Save an evaluation plot to MLflow.

        Args:
            context: Training context.
            iteration: Current training iteration.
            fig: Plotly figure.
        """
        self._log_figure(
            (
                f"training_plot_sample_rate={context.input_validation_sample_rate}_"
                f"{context.output_validation_sample_rate}"
            ),
            fig,
            iteration,
        )

    def save_benchmark_plot(
        self,
        context: Context,
        iteration: int | None,
        fig: Figure,
        plot_type: str,
    ) -> None:
        """Save a benchmark plot to MLflow.

        Args:
            context: Training context.
            iteration: Current training iteration.
            fig: Plotly figure.
            plot_type: Benchmark plot type.
        """
        self._log_figure(f"benchmark_{context.benchmark_location}_{plot_type}", fig, iteration)

    def report_eval(self, context: Context, iteration: int | None) -> None:
        """Report evaluation metrics to MLflow.

        Args:
            context: Training context.
            iteration: Current training iteration.
        """
        metrics = {
            (f"validation_{key}_{context.input_validation_sample_rate}_{context.output_validation_sample_rate}"): value
            for key, value in context.eval_metrics.items()
            if not isinstance(value, dict)
        }
        metrics[f"validation_loss_{context.input_validation_sample_rate}_{context.output_validation_sample_rate}"] = (
            context.validation_loss
        )
        self._log_metrics(metrics, iteration)

    def report_benchmark(self, context: Context, iteration: int) -> None:
        """Report benchmark metrics and quantile rows to MLflow.

        Args:
            context: Training context.
            iteration: Current training iteration.
        """
        metrics = {}
        quantile_metrics = {}
        for key, value in context.benchmark_metrics.items():
            if isinstance(value, dict):
                quantile_metrics[key] = value
                metrics.update({
                    f"benchmark_{context.benchmark_location}_{key}_Q{quantile:.2f}": metric
                    for quantile, metric in value.items()
                })
            else:
                metrics[f"benchmark_{context.benchmark_location}_{key}"] = value
        self._log_metrics(metrics, iteration)

        if quantile_metrics:
            quantiles = sorted({quantile for values in quantile_metrics.values() for quantile in values})
            rows = [
                {
                    "step": iteration,
                    "location": context.benchmark_location,
                    "quantile": quantile,
                    **{key: values.get(quantile) for key, values in quantile_metrics.items()},
                }
                for quantile in quantiles
            ]
            mlflow.log_table(pd.DataFrame(rows), artifact_file="benchmark_quantile_scores.json")  # type: ignore[possibly-missing-attribute]

    def checkpoint(self, context: Context, iteration: int) -> None:
        """Upload the latest checkpoint to MLflow.

        Args:
            context: Training context.
            iteration: Current training iteration.
        """
        if context.checkpointer.last_checkpoint is not None:  # type: ignore[union-attr]
            mlflow.log_artifact(  # type: ignore[possibly-missing-attribute]
                context.checkpointer.last_checkpoint.as_local_path(),  # type: ignore[union-attr]
                artifact_path=f"steps/step-{iteration:08d}/checkpoints",
            )

    def step(self, context: Context, iteration: int | None) -> None:
        """Log training metrics to MLflow.

        Args:
            context: Training context.
            iteration: Current training iteration.
        """
        self._log_metrics(
            {
                "training_loss": context.loss,
                "epoch": context.trainer.epoch,  # type: ignore[union-attr]
            },
            iteration,
        )

    def log_html_panel(self, name: str, html_path: Path, iteration: int) -> None:
        """Upload an existing HTML panel to MLflow.

        Args:
            name: Panel name.
            html_path: Local HTML path.
            iteration: Current training iteration.
        """
        filename = self._sanitize_artifact_component(name)
        mlflow.log_artifact(  # type: ignore[possibly-missing-attribute]
            str(html_path),
            artifact_path=f"steps/step-{iteration:08d}/html/{filename}",
        )

    def benchmark_complete(self, context: Context, iteration: int) -> None:
        """Log aggregate benchmark completion metrics and HTML panels.

        Args:
            context: Training context.
            iteration: Current training iteration.
        """
        if context.configuration.benchmarking.benchmarks.get("LocalBenchmark") is not None:
            df = get_external_benchmarks(self.csv_file, iteration)
            metrics = {}
            if "ldn_monthly_mape" in df.columns:
                metrics["mean_predicted_ldn_monthly_mape"] = df["ldn_monthly_mape"].abs().mean()
            if "odn_monthly_mape" in df.columns:
                metrics["mean_predicted_odn_monthly_mape"] = df["odn_monthly_mape"].abs().mean()
            if metrics:
                self._log_metrics(metrics, iteration)

        if context.configuration.benchmarking.benchmarks.get("StefBeamBenchmark") is not None:
            metric_keys: dict[str, str] = {
                "F2.0": "f2",
                "effective_precision": "effective_precision",
                "effective_recall": "effective_recall",
                "precision": "precision",
                "recall": "recall",
                "rCRPS": "rCRPS",
                "rMAE": "rMAE",
            }

            if hasattr(context, "stefbeam_targets") and hasattr(context, "stefbeam_storage"):
                reports = read_evaluation_reports(
                    targets=context.stefbeam_targets,  # type: ignore[union-attr]
                    storage=context.stefbeam_storage,  # type: ignore[union-attr]
                    run_name=context.stefbeam_run_name,  # type: ignore[union-attr]
                )
                available_ats = str(reports[0][1].subset_reports[0].filtering)
                df_summary = build_summary_from_reports(
                    reports=reports,
                    metrics_wanted=set(metric_keys),
                    filtering=available_ats,
                )
                metrics = {
                    f"mean_{metric_keys[row['metric']]}_{row['group']}_Q{row['quantile']}": float(row["mean"])
                    for _, row in df_summary.iterrows()
                }
                self._log_metrics(metrics, iteration)

                if available_ats:
                    for panel in [
                        "best_f2",
                        "precision_at_best_f2",
                        "recall_at_best_f2",
                        "rCRPS_grouped",
                        "rMAE_grouped",
                    ]:
                        html_path = context.stefbeam_results_path / available_ats / f"{panel}.html"  # type: ignore[union-attr]
                        if html_path.exists():
                            self.log_html_panel(panel, html_path, iteration)
