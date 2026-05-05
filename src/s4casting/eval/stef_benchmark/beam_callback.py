# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

from datetime import time, timedelta
from pathlib import Path

from openstef_beam.analysis import AnalysisConfig, AnalysisScope
from openstef_beam.analysis.models import AnalysisAggregation
from openstef_beam.analysis.visualizations import GroupedTargetMetricVisualization
from openstef_beam.benchmarking.benchmarks import create_liander2024_benchmark_runner
from openstef_beam.benchmarking.benchmarks.liander2024 import Liander2024TargetProvider
from openstef_beam.benchmarking.callbacks import StrictExecutionCallback
from openstef_beam.benchmarking.storage import LocalBenchmarkStorage
from openstef_core.types import AvailableAt, Quantile

from s4casting.core.context import Context
from s4casting.eval.stef_benchmark.model_interface import S4ModelInterface


def run_stefbeam(context: Context, iteration: int):
    """Run the STEFBeam benchmark.

    This does multiple things:
    - builds benchmark pipeline
    - sets up model interface
    - runs benchmarking
    - logs results to wandb

    Args:
        context: Training context
        iteration (int): Current training iteration
    """
    assert context.configuration.model.base_sample_interval_minutes == 15, (
        "Base sample interval must be 15 minutes to run stef beam."
    )
    if not context.machine.main_process:
        return

    if not hasattr(context, "_stefbeam_pipeline"):
        output_dir = Path(context.configuration.io.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        benchmark_dir = Path("data") / "liander2024"

        storage = LocalBenchmarkStorage(
            output_dir / "stef-beam",
            skip_when_existing=False,
        )

        target_provider = Liander2024TargetProvider(
            data_dir=benchmark_dir,
            use_profiles=False,
            use_prices=False,
            targets_file_path=context.configuration.benchmarking.benchmarks["StefBeamBenchmark"].targets_file,  # type: ignore[possibly-missing-attribute]
        )
        analysis_config = AnalysisConfig(
            visualization_providers=[
                GroupedTargetMetricVisualization(name="rMAE_grouped", metric="rMAE", quantile=Quantile(0.5)),
                GroupedTargetMetricVisualization(name="rCRPS_grouped", metric="rCRPS"),
                GroupedTargetMetricVisualization(
                    name="best_f2", metric="effective_F2.0", selector_metric="effective_F2.0"
                ),
                GroupedTargetMetricVisualization(
                    name="precision_at_best_f2", metric="effective_precision", selector_metric="effective_F2.0"
                ),
                GroupedTargetMetricVisualization(
                    name="recall_at_best_f2", metric="effective_recall", selector_metric="effective_F2.0"
                ),
            ]
        )

        pipeline = create_liander2024_benchmark_runner(
            data_dir=benchmark_dir,
            storage=storage,
            callbacks=[StrictExecutionCallback()],
            target_provider=target_provider,
        )
        # limit plots and evaluation to expedite benchmarking
        pipeline.analysis_config = analysis_config
        pipeline.backtest_config.predict_interval = timedelta(hours=24)
        pipeline.backtest_config.align_time = time.fromisoformat("06:00+00")

        # ensure evaluation only uses the D-1 06:00 availability
        pipeline.evaluation_config.available_ats = [AvailableAt.from_string("D-1T06:00")]

        context._stefbeam_pipeline = pipeline  # type: ignore
        context._stefbeam_storage = storage  # type: ignore

    pipeline = context._stefbeam_pipeline  # type: ignore
    storage = context._stefbeam_storage  # type: ignore

    # model interface
    def model_factory(_benchmark_context, target):
        return S4ModelInterface(
            context=context,
            target=target,
        )

    run_name = f"iter_{iteration:07d}"
    pipeline.run(forecaster_factory=model_factory, run_name=run_name, n_processes=1)

    # get analytics dir
    analytics_scope = AnalysisScope(
        aggregation=AnalysisAggregation.GROUP,
        run_name=run_name,
    )
    target_analytics_dir = storage.get_analysis_path(analytics_scope)

    context.stefbeam_results_path = target_analytics_dir  # type: ignore
    context.stefbeam_targets = pipeline.target_provider.get_targets()  # type: ignore
    context.stefbeam_storage = storage  # type: ignore
    context.stefbeam_run_name = run_name  # type: ignore

    # hook for logging results to wandb
    context.trainer.hooks.benchmark_complete.call(context, iteration)  # type: ignore

    context.model_container.model.train(mode=True)
