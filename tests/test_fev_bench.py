# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import math
from pathlib import Path

import datasets
import fev
import numpy as np
import pandas as pd
import pytest
import yaml

from s4casting import factories as fc
from s4casting.core.config import Configuration, DatasetConfiguration, FevBenchBenchmark, TransformerConfiguration
from s4casting.core.context import Context
from s4casting.eval.fev_bench.adapter import (
    FevForecaster,
    UnsupportedTaskError,
    interval_minutes,
)
from s4casting.eval.fev_bench.callback import (
    ENERGY_TASKS,
    build_leaderboard,
    fev_bench_metrics,
    load_tasks,
    run_fevbench,
)
from scripts.train import train
from tests.utils import load_config

HORIZON = 24
QUANTILES = [0.1, 0.5, 0.9]


def _write_dataset(path: Path, freq: str = "h", n_series: int = 3, n_steps: int = 24 * 40, targets=("target",)) -> Path:
    """Write a small fev-format parquet dataset with one known and one past covariate.

    Returns:
        Path: The parquet file.
    """
    rng = np.random.default_rng(0)
    timestamps = pd.date_range("2024-01-01", periods=n_steps, freq=freq)
    t = np.arange(n_steps)
    rows = {"id": [f"s{i}" for i in range(n_series)], "timestamp": [timestamps.tolist()] * n_series}
    for name in targets:
        rows[name] = [
            (10 + 3 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.3, n_steps)).tolist() for _ in range(n_series)
        ]
    rows["temp"] = [(15 + 5 * np.cos(2 * np.pi * t / 24)).tolist()] * n_series
    rows["irr"] = [np.clip(np.sin(2 * np.pi * t / 24), 0, None).tolist()] * n_series
    datasets.Dataset.from_dict(rows).to_parquet(str(path))
    return path


def _task(path: Path, **overrides) -> fev.Task:
    kwargs = {
        "dataset_path": str(path),
        "horizon": HORIZON,
        "num_windows": 2,
        "seasonality": 24,
        "eval_metric": "SQL",
        "extra_metrics": ["MASE", {"name": "WQL", "epsilon": 1.0}],
        "quantile_levels": QUANTILES,
        "known_dynamic_columns": ["temp"],
        "past_dynamic_columns": ["irr"],
    }
    kwargs.update(overrides)
    return fev.Task(**kwargs)


def _config(
    rates: list[int], n_covariate_slots: int = 2, with_time: bool = True, n_time_features: int = 1
) -> Configuration:
    """Small CPU SSM configuration with the requested training rates and covariate slots.

    Returns:
        Configuration: The configuration.
    """
    cfg = load_config()
    cfg.machine.device_kind = "cpu"
    cfg.model.input_sample_intervals_minutes = rates
    cfg.model.output_sample_intervals_minutes = rates
    cfg.io.feature_order = ["measurements"]
    if n_covariate_slots:
        cfg.io.features["weather"] = DatasetConfiguration(location="", loader="sideload", n_features=n_covariate_slots)
        cfg.io.feature_order.append("weather")
    if with_time:
        cfg.io.features["time"] = DatasetConfiguration(
            location="", loader="time", include_location=n_time_features == 3
        )
        cfg.io.feature_order.append("time")
    return cfg


def _forecaster(cfg: Configuration, **bench_overrides) -> tuple[FevForecaster, Context]:
    settings = {
        "context_window_days": 4,
        "baseline_results": [],
        "task_subset": "all",
        **bench_overrides,
    }
    bench = FevBenchBenchmark(**settings)
    cfg.benchmarking.benchmarks["FevBenchBenchmark"] = bench
    cfg.benchmarking.batch_size = 2
    machine = fc.provide_machine(cfg.machine, rng_base_seed=cfg.run.seed)
    model_container = fc.provide_model_container(cfg.model, cfg.io, machine)
    optimizer = fc.provide_optimizer(cfg.optimizer, model_container.raw_model.parameters())
    trainer = fc.provide_trainer(config=cfg.training, optimizer=cfg.optimizer, machine=machine)
    context = Context(
        configuration=cfg,
        model_container=model_container,
        optimizer=optimizer,
        scheduler=None,
        machine=machine,
        trainer=trainer,
    )
    return FevForecaster(model_container.raw_model.eval(), cfg, bench, "cpu"), context


def _assert_valid_summary(task: fev.Task, summary: dict) -> None:
    assert summary["model_name"] == "s4casting"
    assert math.isfinite(summary["test_error"])
    assert summary["test_error"] >= 0
    assert math.isfinite(summary["MASE"])
    assert summary["num_forecasts"] == task.num_windows * 3 * len(task.target_columns)


def test_interval_minutes_is_inferred_from_timestamps():
    """Sample intervals are inferred from the median timestamp spacing."""
    assert interval_minutes(pd.date_range("2024-01-01", periods=5, freq="10min").tolist()) == 10
    assert interval_minutes(pd.date_range("2024-01-01", periods=5, freq="D").tolist()) == 1440


@pytest.mark.parametrize(("arch", "n_time_features"), [("gmm", 1), ("quantile", 1), ("gmm", 3)])
def test_forecaster_predictions_are_scored_by_fev(tmp_path, arch, n_time_features):
    """Predictions at a training rate follow the fev schema for both output heads and both time encodings."""
    cfg = _config([15, 60], n_time_features=n_time_features)
    cfg.model.output_head.arch = arch
    cfg.model.loss.loss = "nll" if arch == "gmm" else "pinball"
    cfg.model.output_head.quantile_values = [0.1, 0.5, 0.9]
    forecaster, _ = _forecaster(cfg)
    task = _task(_write_dataset(tmp_path / "hourly.parquet"))
    predictions = [forecaster.predict_window(task, w) for w in task.iter_windows(num_proc=1)]
    assert len(predictions) == task.num_windows
    first = predictions[0][0]
    assert set(first) == {"predictions", "0.1", "0.5", "0.9"}
    assert all(len(v) == HORIZON for v in first.values())
    assert first["predictions"] == first["0.5"]
    assert all(np.isfinite(first[k]).all() for k in first)
    _assert_valid_summary(task, task.evaluation_summary(predictions, model_name="s4casting"))


def test_forecaster_never_resamples(tmp_path):
    """A task at a rate the model was not trained on is unsupported, never resampled; a trained rate runs."""
    forecaster, _ = _forecaster(_config([15]))
    task = _task(_write_dataset(tmp_path / "ten_min.parquet", freq="10min", n_steps=6 * 24 * 10), horizon=36)
    with pytest.raises(UnsupportedTaskError, match=r"10-minute data, but the model was trained at \[15\]"):
        forecaster.predict_window(task, task.get_window(0, num_proc=1))
    forecaster, _ = _forecaster(_config([10, 15]))
    predictions = [forecaster.predict_window(task, w) for w in task.iter_windows(num_proc=1)]
    assert len(predictions[0][0]["predictions"]) == 36
    _assert_valid_summary(task, task.evaluation_summary(predictions, model_name="s4casting"))


def test_forecaster_skips_rates_without_a_valid_context_window(tmp_path):
    """A trained rate whose context windows are all pruned is unsupported (skipped), not a crash."""
    cfg = _config([15, 60])
    cfg.model.context_window = [4]  # 4 days hourly = 96 samples, but the limits below exclude it
    cfg.io.interval_context_limits = {15: {"min_days": 4, "max_days": 8}, 60: {"min_days": 8, "max_days": 16}}
    forecaster, _ = _forecaster(cfg, context_window_days=None)
    task = _task(_write_dataset(tmp_path / "hourly.parquet"), num_windows=1)
    with pytest.raises(UnsupportedTaskError, match=r"no context window .* valid at 60 minutes"):
        forecaster.predict_window(task, task.get_window(0, num_proc=1))


def test_forecaster_handles_multivariate_tasks_and_missing_covariate_slots(tmp_path):
    """Multivariate targets are forecast per column and recombined; a model without covariate slots still runs."""
    forecaster, _ = _forecaster(_config([60], n_covariate_slots=0, with_time=False))
    path = _write_dataset(tmp_path / "multi.parquet", targets=("a", "b"))
    task = _task(path, target=["a", "b"])
    predictions = [forecaster.predict_window(task, w) for w in task.iter_windows(num_proc=1)]
    assert set(predictions[0].keys()) == {"a", "b"}
    _assert_valid_summary(task, task.evaluation_summary(predictions, model_name="s4casting"))


def test_load_tasks_subset_and_window_override(tmp_path):
    """The task file is filtered by task_subset and num_windows is applied; unknown names fail loudly."""
    tasks_file = tmp_path / "tasks.yaml"
    entries = [{**_task(_write_dataset(tmp_path / f"{n}.parquet")).to_dict(), "task_name": n} for n in ("one", "two")]
    tasks_file.write_text(yaml.safe_dump({"tasks": entries}, sort_keys=False))
    bench = FevBenchBenchmark(tasks_file=str(tasks_file), num_windows=1, baseline_results=[], task_subset="all")
    tasks = load_tasks(bench)
    assert [t.task_name for t in tasks] == ["one", "two"]
    assert all(t.num_windows == 1 for t in tasks)
    bench.task_subset = ["missing"]
    with pytest.raises(ValueError, match="not in"):
        load_tasks(bench)


def test_energy_subset_selects_the_paper_energy_tasks(tmp_path):
    """The default 'energy' subset keeps exactly the 26 Energy-domain tasks and drops the rest."""
    tasks_file = tmp_path / "tasks.yaml"
    names = [*ENERGY_TASKS, "m5_1D", "jena_weather_1H"]
    entries = [{**_task(_write_dataset(tmp_path / "shared.parquet")).to_dict(), "dataset_config": n} for n in names]
    tasks_file.write_text(yaml.safe_dump({"tasks": entries}, sort_keys=False))
    bench = FevBenchBenchmark(tasks_file=str(tasks_file), baseline_results=[])
    assert bench.task_subset == "energy"
    assert [t.dataset_config for t in load_tasks(bench)] == list(ENERGY_TASKS)
    assert len(ENERGY_TASKS) == 26


def test_run_fevbench_writes_summaries_and_metrics(tmp_path):
    """The hook writes the fev summary CSV, fills the context and exposes flat metrics for the loggers."""
    cfg = _config([60])
    cfg.io.output = str(tmp_path / "out")
    tasks_file = tmp_path / "tasks.yaml"
    task_dict = {**_task(_write_dataset(tmp_path / "toy.parquet")).to_dict(), "task_name": "toy"}
    tasks_file.write_text(yaml.safe_dump({"tasks": [task_dict]}, sort_keys=False))
    _, context = _forecaster(cfg, tasks_file=str(tasks_file))

    run_fevbench(context, iteration=7)

    assert context.fev_bench_iteration == 7
    summary = pd.read_csv(context.fev_bench_results_path)
    assert list(summary["task_name"]) == ["toy"]
    metrics = fev_bench_metrics(context)
    assert metrics["fev_bench/num_tasks"] == 1
    assert metrics["fev_bench/num_skipped"] == 0
    assert math.isfinite(metrics["fev_bench/toy/SQL"])
    assert "fev_bench/toy/MASE" in metrics
    assert context.fev_bench_leaderboard is None  # no baselines configured


def test_run_fevbench_warns_when_skipped_tasks_break_comparability(tmp_path):
    """A task at an untrained rate is skipped, counted, and flagged as breaking leaderboard comparability."""
    cfg = _config([60])
    cfg.io.output = str(tmp_path / "out")
    tasks_file = tmp_path / "tasks.yaml"
    hourly = {**_task(_write_dataset(tmp_path / "hourly.parquet")).to_dict(), "task_name": "hourly"}
    daily = {
        **_task(_write_dataset(tmp_path / "daily.parquet", freq="D", n_steps=200), horizon=7, seasonality=7).to_dict(),
        "task_name": "daily",
    }
    tasks_file.write_text(yaml.safe_dump({"tasks": [hourly, daily]}, sort_keys=False))
    _, context = _forecaster(cfg, tasks_file=str(tasks_file))

    with pytest.warns(UserWarning, match=r"1 of 2 tasks skipped \(daily\).*not comparable"):
        run_fevbench(context, iteration=1)

    assert context.fev_bench_skipped == ["daily"]
    assert list(pd.read_csv(context.fev_bench_results_path)["task_name"]) == ["hourly"]
    assert fev_bench_metrics(context)["fev_bench/num_skipped"] == 1


def test_build_leaderboard_ranks_against_baselines(tmp_path):
    """A baseline CSV in fev format yields a leaderboard with the model and baseline rows."""
    cfg = _config([60])
    tasks_file = tmp_path / "tasks.yaml"
    task = _task(_write_dataset(tmp_path / "toy.parquet"))
    tasks_file.write_text(yaml.safe_dump({"tasks": [{**task.to_dict(), "task_name": "toy"}]}, sort_keys=False))
    forecaster, _ = _forecaster(cfg)
    predictions = [forecaster.predict_window(task, w) for w in task.iter_windows(num_proc=1)]
    ours = pd.DataFrame([task.evaluation_summary(predictions, model_name="s4casting")])
    baseline = ours.copy()
    baseline["model_name"] = "Seasonal Naive"
    baseline["test_error"] *= 2
    baseline_path = tmp_path / "baseline.csv"
    baseline.to_csv(baseline_path, index=False)
    bench = FevBenchBenchmark(
        tasks_file=str(tasks_file), baseline_results=[str(baseline_path)], n_resamples=10, task_subset="all"
    )

    leaderboard = build_leaderboard(ours, bench)

    assert leaderboard is not None
    assert set(leaderboard.index) == {"s4casting", "Seasonal Naive"}
    assert leaderboard.loc["s4casting", "skill_score"] > leaderboard.loc["Seasonal Naive", "skill_score"]
    assert build_leaderboard(ours, FevBenchBenchmark(baseline_results=[])) is None


def test_train_end_to_end_cpu_with_fev_bench(tmp_path, load_config_gmm: Configuration):
    """Training runs the fev-bench hook at iteration 0 (zero-shot), at each interval and for the final model."""
    cfg = load_config_gmm
    cfg.machine.device_kind = "cpu"
    cfg.training.iteration = 0  # zero-shot benchmark before the first step
    cfg.io.output = str(tmp_path / "out")
    tasks_file = tmp_path / "tasks.yaml"
    # the test config trains at 15 minutes only and the benchmark never resamples, so the toy task is 15-minute
    task = _task(_write_dataset(tmp_path / "toy.parquet", freq="15min"), num_windows=1)
    tasks_file.write_text(yaml.safe_dump({"tasks": [{**task.to_dict(), "task_name": "toy"}]}, sort_keys=False))
    cfg.benchmarking.benchmarks["FevBenchBenchmark"] = FevBenchBenchmark(
        tasks_file=str(tasks_file), context_window_days=4, baseline_results=[], task_subset="all"
    )
    cfg.training.batch_size = 8
    cfg.training.evaluation_interval = 1
    cfg.training.checkpoint_interval = 1
    cfg.training.benchmarking_interval = 1
    cfg.training.maximum_steps = 2

    train(cfg)

    written = sorted(p.name for p in (tmp_path / "out" / "fev-bench").glob("iter_*.csv"))
    assert written == ["iter_0000000.csv", "iter_0000001.csv", "iter_0000002.csv"]


def test_forecaster_pads_inputs_like_training(tmp_path):
    """Model inputs are left-padded to the batcher's padded length, with the padding zero and masked."""
    cfg = _config([15, 60])
    cfg.model.context_window = [4, 8]  # 8 days at 15 min = 768 samples; 4 days hourly = 96 samples
    cfg.io.interval_context_limits = {15: {"min_days": 4, "max_days": 8}, 60: {"min_days": 4, "max_days": 4}}
    forecaster, _ = _forecaster(cfg, context_window_days=None)
    assert forecaster.padded_samples == 768  # the test config is an SSM, which the batcher pads
    cfg_transformer = _config([15, 60])
    cfg_transformer.model.model = "transformer"
    cfg_transformer.model.transformer = TransformerConfiguration(latent_dim=32, n_heads=4, n_layers=1)
    assert _forecaster(cfg_transformer, context_window_days=None)[0].padded_samples == 0
    seen = {}
    real_forward = forecaster.model.forward

    def spy(x, xm, *args, **kwargs):
        seen["shape"] = tuple(x.shape)
        # hourly sequence: 4 days = 96 samples = 72 context + 24 horizon, left-padded to 768
        seen["pad_zero"] = bool((x[:, : 768 - 96] == 0).all()) and bool((xm[:, : 768 - 96] == 0).all())
        seen["real_mask"] = bool(xm[:, 768 - 96 : -24, 0].all())
        return real_forward(x, xm, *args, **kwargs)

    forecaster.model.forward = spy
    task = _task(_write_dataset(tmp_path / "hourly.parquet"), num_windows=1)
    predictions = forecaster.predict_window(task, task.get_window(0, num_proc=1))
    assert seen["shape"][1] == 768
    assert seen["pad_zero"]
    assert seen["real_mask"]
    assert len(predictions[0]["predictions"]) == HORIZON


def test_forecaster_uses_the_series_length_without_padding(tmp_path):
    """The transformer gets a short series at its own length, a long one capped at the longest trained window."""
    cfg = _config([60])
    cfg.model.model = "transformer"
    cfg.model.transformer = TransformerConfiguration(latent_dim=32, n_heads=4, n_layers=1)
    cfg.model.context_window = [16, 32, 64]  # hourly: 384, 768 and 1536 samples
    forecaster, _ = _forecaster(cfg, context_window_days=None)
    assert forecaster.trained_lengths(60) == [384, 768, 1536]
    assert forecaster.sequence_samples(60) == 1536  # without a series length: the longest
    seen = {}
    real_forward = forecaster.model.forward

    def spy(x, xm, *args, **kwargs):
        seen["length"] = x.shape[1]
        seen["leading_masked"] = int((xm[0, :, 0] == 0).to(int).cumprod(0).sum())
        return real_forward(x, xm, *args, **kwargs)

    forecaster.model.forward = spy
    # 10 days: 216 h history + 24 h horizon = 240 samples, below every trained window -> used as is
    task = _task(_write_dataset(tmp_path / "short.parquet", n_steps=24 * 10), num_windows=1)
    forecaster.predict_window(task, task.get_window(0, num_proc=1))
    assert seen["length"] == 240
    assert seen["leading_masked"] == 0
    # 40 days: 936 + 24 = 960, between trained windows -> used as is, no padding
    task = _task(_write_dataset(tmp_path / "medium.parquet", n_steps=24 * 40), num_windows=1)
    forecaster.predict_window(task, task.get_window(0, num_proc=1))
    assert seen["length"] == 960
    assert seen["leading_masked"] == 0
    # 80 days: 1896 + 24 > 1536 -> capped at the longest trained window, history cut not padded
    task = _task(_write_dataset(tmp_path / "long.parquet", n_steps=24 * 80), num_windows=1)
    forecaster.predict_window(task, task.get_window(0, num_proc=1))
    assert seen["length"] == 1536
    assert seen["leading_masked"] == 0
