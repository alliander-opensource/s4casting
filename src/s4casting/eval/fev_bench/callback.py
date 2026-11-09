# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

"""Run the fev-bench tasks with the model under training and summarise the results."""

from __future__ import annotations

import logging
import os
import time
import warnings
from pathlib import Path

import fev
import pandas as pd
import torch

from s4casting.core.config import FevBenchBenchmark
from s4casting.core.context import Context
from s4casting.eval.fev_bench.adapter import FevForecaster, UnsupportedTaskError

logger = logging.getLogger(__name__)

BENCHMARK_KEY = "FevBenchBenchmark"
METRIC_COLUMNS = ("SQL", "MASE", "WQL", "WAPE")

# The Energy domain of fev-bench (Shchur et al., 2025, arXiv:2509.26468, Appendix A.3): 26 of the 100 tasks.
ENERGY_TASKS = (
    "proenfo_gfc12",
    "proenfo_gfc14",
    "proenfo_gfc17",
    "entsoe_15T",
    "entsoe_30T",
    "entsoe_1H",
    "epf_be",
    "epf_de",
    "epf_fr",
    "epf_np",
    "epf_pjm",
    "solar_with_weather_15T",
    "solar_with_weather_1H",
    "solar_1D",
    "solar_1W",
    "kdd_cup_2022_10T",
    "kdd_cup_2022_30T",
    "kdd_cup_2022_1D",
    "ETT_15T",
    "ETT_1D",
    "ETT_1H",
    "ETT_1W",
    "ercot_1H",
    "ercot_1D",
    "ercot_1W",
    "ercot_1M",
)


def load_tasks(bench: FevBenchBenchmark) -> list[fev.Task]:
    """Load the fev tasks selected by the benchmark configuration.

    Args:
        bench (FevBenchBenchmark): Benchmark settings.

    Returns:
        list[fev.Task]: Tasks in file order, restricted to ``task_subset`` and with ``num_windows`` applied.

    Raises:
        ValueError: If ``task_subset`` names a task that is not in the file.
    """
    tasks = fev.Benchmark.from_yaml(bench.tasks_file).tasks
    if bench.task_subset != "all":
        wanted = ENERGY_TASKS if bench.task_subset == "energy" else bench.task_subset
        available = {t.dataset_config for t in tasks}
        unknown = sorted(set(wanted) - available)
        if unknown:
            raise ValueError(f"task_subset names tasks that are not in {bench.tasks_file}: {unknown}")
        tasks = [t for t in tasks if t.dataset_config in wanted]
    if bench.num_windows is not None:
        tasks = [fev.Task(**{**t.to_dict(), "num_windows": bench.num_windows}) for t in tasks]
    return tasks


def build_leaderboard(summary: pd.DataFrame, bench: FevBenchBenchmark) -> pd.DataFrame | None:
    """Rank the model against the configured baseline results with ``fev.leaderboard``.

    Args:
        summary (pd.DataFrame): fev summaries of this run, one row per task.
        bench (FevBenchBenchmark): Benchmark settings.

    Returns:
        pd.DataFrame | None: The leaderboard, or None when it cannot be computed (no baselines, a
        ``num_windows`` override that changes the task identity, or no task in common).
    """
    if not bench.baseline_results or bench.num_windows is not None or summary.empty:
        return None
    task_names = set(summary["task_name"])
    baselines = []
    for path in bench.baseline_results:
        df = pd.read_csv(path)
        baselines.append(df[df["task_name"].isin(task_names)])
    try:
        return fev.leaderboard(
            [summary, *baselines],
            baseline_model=bench.baseline_model,
            missing_strategy="drop",
            n_resamples=bench.n_resamples,
        )
    except Exception as exc:  # the leaderboard is reporting only; never abort a run for it
        logger.warning("fev leaderboard skipped: %s", exc)
        return None


def fev_bench_metrics(context: Context) -> dict[str, float]:
    """Flatten the latest fev-bench results on the context into scalar metrics for the loggers.

    Args:
        context (Context): Context holding ``fev_bench_summary`` and ``fev_bench_leaderboard``.

    Returns:
        dict[str, float]: Per-task metrics as ``fev_bench/<task>/<metric>`` plus aggregates.
    """
    summary: pd.DataFrame = context.fev_bench_summary  # type: ignore[attr-defined]
    metrics: dict[str, float] = {}
    for _, row in summary.iterrows():
        for col in METRIC_COLUMNS:
            if col in row and pd.notna(row[col]):
                metrics[f"fev_bench/{row['task_name']}/{col}"] = float(row[col])
    if not summary.empty:
        metrics["fev_bench/mean_test_error"] = float(summary["test_error"].mean())
    metrics["fev_bench/num_tasks"] = float(len(summary))
    metrics["fev_bench/num_skipped"] = float(len(context.fev_bench_skipped))  # type: ignore[attr-defined]
    leaderboard: pd.DataFrame | None = context.fev_bench_leaderboard  # type: ignore[attr-defined]
    model_name = context.configuration.benchmarking.benchmarks[BENCHMARK_KEY].model_name  # type: ignore[union-attr]
    if leaderboard is not None and model_name in leaderboard.index:
        row = leaderboard.loc[model_name]
        for col in (
            "skill_score",
            "skill_score_lower",
            "skill_score_upper",
            "win_rate",
            "win_rate_lower",
            "win_rate_upper",
        ):
            if col in row and pd.notna(row[col]):
                metrics[f"fev_bench/{col}"] = float(row[col])
    return metrics


def run_fevbench(context: Context, iteration: int) -> None:
    """Forecast every configured fev task, write the summaries and rank them.

    Results are stored on the context (``fev_bench_summary``, ``fev_bench_leaderboard``,
    ``fev_bench_results_path``, ``fev_bench_skipped``, ``fev_bench_iteration``) and the
    ``benchmark_complete`` hook is fired so the loggers pick them up.

    Args:
        context (Context): Training context.
        iteration (int): Current training iteration.
    """
    if not context.machine.main_process:
        return
    bench = context.configuration.benchmarking.benchmarks.get(BENCHMARK_KEY)
    if bench is None:
        return
    assert isinstance(bench, FevBenchBenchmark)
    if bench.datasets_prefix:
        os.environ["FEV_DATASETS_PREFIX"] = bench.datasets_prefix

    model = context.model_container.raw_model.to(context.machine.benchmarking_device)
    model.train(mode=False)
    precision = context.configuration.training.precision
    # A model with its own fev integration (Chronos-2's predict_fev) is always scored through it, so its
    # numbers follow the published recipe: its own context length, every task covariate, and no rate or
    # window handling by the adapter. The s4casting models have none and go through the adapter.
    pipeline = getattr(model, "pipeline", None)
    if not hasattr(pipeline, "predict_fev"):
        pipeline = None
    forecaster = FevForecaster(
        model,
        context.configuration,
        bench,
        context.machine.benchmarking_device,
        autocast=lambda: context.machine.autocast(precision),
    )
    batch_size = context.configuration.benchmarking.batch_size

    summaries: list[dict] = []
    skipped: list[str] = []
    for task in load_tasks(bench):
        task_name = task.task_name or str(task.dataset_config)
        started = time.monotonic()
        try:
            if pipeline is not None:
                # the model's own recipe, as in the published results: its context length, all covariates
                with torch.no_grad():
                    predictions, _ = pipeline.predict_fev(task, batch_size=batch_size)
            else:
                predictions = [
                    forecaster.predict_window(task, window) for window in task.iter_windows(num_proc=bench.num_proc)
                ]
        except UnsupportedTaskError as exc:
            logger.warning("fev-bench task %s skipped: %s", task_name, exc)
            skipped.append(task_name)
            continue
        summary = task.evaluation_summary(
            predictions,
            model_name=bench.model_name,
            inference_time_s=time.monotonic() - started,
            trained_on_this_dataset=task.dataset_config in bench.trained_on_datasets,
            extra_info={"iteration": iteration},
        )
        logger.info(
            "fev-bench %s at iteration %d: %s=%.4f", task_name, iteration, summary["eval_metric"], summary["test_error"]
        )
        summaries.append(summary)

    if skipped:
        # The skill score and win rate below only cover the tasks that ran, so they are not comparable
        # with leaderboards over the full task set (the published fev-bench numbers included).
        warnings.warn(
            f"fev-bench: {len(skipped)} of {len(skipped) + len(summaries)} tasks skipped ({', '.join(skipped)}); "
            "the leaderboard covers the remaining tasks only and is not comparable with results over the full "
            "task set. Compare per task, or train the missing sample rates.",
            UserWarning,
            stacklevel=2,
        )
    summary_df = pd.DataFrame(summaries)
    output_dir = Path(context.configuration.io.output) / bench.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"iter_{iteration:07d}.csv"
    summary_df.to_csv(results_path, index=False)
    leaderboard = build_leaderboard(summary_df, bench)
    if leaderboard is not None:
        logger.info("fev-bench leaderboard at iteration %d:\n%s", iteration, leaderboard.to_string())

    context.fev_bench_summary = summary_df  # type: ignore[attr-defined]
    context.fev_bench_leaderboard = leaderboard  # type: ignore[attr-defined]
    context.fev_bench_results_path = results_path  # type: ignore[attr-defined]
    context.fev_bench_skipped = skipped  # type: ignore[attr-defined]
    context.fev_bench_iteration = iteration  # type: ignore[attr-defined]

    if context.trainer is not None:
        context.trainer.hooks.benchmark_complete.call(context, iteration)
    model.train(mode=True)
