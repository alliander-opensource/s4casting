# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import csv
import logging
from contextlib import suppress
from pathlib import Path

import numpy as np
import wandb
from openstef_beam.benchmarking import read_evaluation_reports
from plotly.graph_objects import Figure
from tqdm import tqdm

from s4casting.core.config import (
    AuthenticationConfiguration,
    RunConfiguration,
    TrainingConfiguration,
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
        run_config: RunConfiguration,
        auth: AuthenticationConfiguration | None,
        output_dir: str = "results",
    ) -> None:
        """Initialize the WandbLogger.

        Args:
            hookable (CommonHooks | TrainingHooks): Hookable object to register hooks.
            run_config (RunConfiguration): Run configuration.
            auth (AuthenticationConfiguration): Authentication configuration.
            output_dir (str): Directory to save results.
        """
        if not run_config.persist_to_wandb_project:
            logging.info("Not persisting to wandb")
            return

        if not auth.wandb_api_key and run_config.wandb_online:  # type: ignore[union-attr]
            raise RuntimeError("Wandb API key not provided, cannot persist to wandb")

        if run_config.wandb_online:  # type: ignore[union-attr]
            wandb.login(key=auth.wandb_api_key.get_secret_value())  # type: ignore[union-attr]

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
        # Note: model_dump with mode json hides secret strings -> exactly what we want
        if context.configuration.run.wandb_runid and context.configuration.run.wandb_online:  # type: ignore[union-attr]
            wandb.init(
                project=context.configuration.run.persist_to_wandb_project,
                config=context.configuration.model_dump(mode="json"),
                id=context.configuration.run.wandb_runid,
                resume="must",
                notes=context.configuration.run.wandb_notes,
            )
            wandb.config.update({
                "dataset_hash_measurements": context.measurements_hash,
                "dataset_hash_weather": context.weather_hash,
            })
        elif context.configuration.run.wandb_online:  # type: ignore[union-attr]
            wandb.init(
                project=context.configuration.run.persist_to_wandb_project,
                config=context.configuration.model_dump(mode="json"),
                notes=context.configuration.run.wandb_notes,
            )
            wandb.config.update({
                "dataset_hash_measurements": context.measurements_hash,
                "dataset_hash_weather": context.weather_hash,
            })
        else:
            wandb.init(
                project="forecasting-s4",
                entity="alliander",
                mode="offline",
                config=context.configuration.model_dump(mode="json"),
                notes=context.configuration.run.wandb_notes,
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
