# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import torch

from s4casting.core.config import BenchmarkingConfiguration
from s4casting.core.context import Context
from s4casting.core.functional import run_in_batches
from s4casting.core.hooks import CommonHooks, TrainingHooks
from s4casting.core.tasks import PredictionTaskDataset
from s4casting.data.dataset.dataset import IntervalDataset, TimeseriesDataset
from s4casting.data.dataset.indexes import get_timestamps, intervals_for_location, location_id
from s4casting.data.utils import collate_single_interval
from s4casting.eval.evaluator_heads import EvaluatorHead
from s4casting.eval.stef_benchmark.beam_callback import run_stefbeam


class Benchmarker:
    """Benchmarker for evaluating model performance on specified locations."""

    def __init__(
        self,
        hookable: CommonHooks | TrainingHooks,
        bench_config: BenchmarkingConfiguration,
        head_evaluator: EvaluatorHead,
    ) -> None:
        """Initialize the Benchmarker.

        Args:
            hookable (CommonHooks | TrainingHooks): Hookable object to register hooks.
            bench_config (BenchmarkingConfiguration): Benchmarking configuration.
            head_evaluator (EvaluatorHead): Evaluator for reporting results.
        """
        self.bench_config = bench_config.benchmarks.get("LocalBenchmark")
        self.head_evaluator = head_evaluator
        self.hooks = hookable
        self.hooks.finished.register(self.finished)

        if isinstance(self.hooks, TrainingHooks):
            # check which benchmarks to run
            if bench_config.benchmarks.get("LocalBenchmark") is not None:
                self.hooks.benchmark.register(self.benchmark)

            if bench_config.benchmarks.get("StefBeamBenchmark") is not None:
                self.hooks.stef_beam.register(run_stefbeam)

    def finished(self, context: Context) -> None:
        """Run benchmarking when training is finished.

        Args:
            context (Context): Training context.
        """
        self.benchmark(context, iteration=context.configuration.training.maximum_steps)

    def sample_dataset(self, location: str, context: Context):
        """Samples location at a specific sample rate.

        Args:
            location (str): location to be benchmarked
            context (Context):  training context object

        Returns:
            Sampled dataset and some extraneous parameters.

        """
        config = context.configuration

        # get intervals for location and specified time
        intervals = intervals_for_location(
            context.batcher.benchmark.intervals_dataset.intervals,  # type: ignore[attr-defined]
            location,
        )

        # create benchmark dataset copy with those intervals
        dataset = TimeseriesDataset(
            IntervalDataset(
                intervals,
                # Hack to make sure that the long term data is properly aligned
                config.model.alignment * 60
                if self.bench_config.alignment is None
                else self.bench_config.alignment * 60,
                self.bench_config.phase,
            ),
            context.batcher.benchmark.context_window,  # type: ignore[attr-defined]
            context.batcher.benchmark.datas,  # type: ignore[attr-defined]
            self.bench_config.input_sample_interval_minutes * 60,
        )
        predict_window = (
            self.bench_config.predict_window_days * 24 * 60
        ) // self.bench_config.input_sample_interval_minutes
        location_dataset = PredictionTaskDataset(
            dataset,
            self.bench_config.predict_dim,  # type: ignore[attr-defined]
            predict_window,
        )

        times = get_timestamps(
            np.array(dataset.intervals_dataset),
            self.bench_config.context_window_days - self.bench_config.predict_window_days,
            self.bench_config.predict_window_days,
            self.bench_config.input_sample_interval_minutes,
            self.bench_config.n_day_ahead,
        )
        task, sample_config = next(
            iter(torch.utils.data.DataLoader(location_dataset, batch_size=0xFFFFFF, collate_fn=collate_single_interval))  # ty: ignore[invalid-argument-type]
        )
        # annoying way to get the sign of the dataset
        loc_id = location_id(location)
        sign = None
        for d in dataset.datas[0]:
            loc = d.locations.get(loc_id)
            if loc is None:
                continue

            sign = loc.get("sign")
            if sign is not None:
                break
        return (
            [x.float().to(context.machine.benchmarking_device) for x in task],
            times,
            sign,
            sample_config,
        )

    def benchmark(self, context: Context, iteration) -> None:
        """Run benchmarking on specified locations.

        Args:
            context (Context): Training context.
            iteration (int): Current training iteration.
        """
        with torch.no_grad():
            context.model_container.model.train(mode=False)
            benchmarking_model = context.model_container.raw_model.to(context.machine.benchmarking_device)
            for location in self.bench_config.locations:
                context.benchmark_location = location
                (X, xm, Y, ym), times, sign, sample_config = self.sample_dataset(location, context)
                output_interval = (
                    self.bench_config.output_sample_interval_minutes
                    if self.bench_config.output_sample_interval_minutes is not None
                    else sample_config.sample_interval_minutes
                )
                prediction, loss = run_in_batches(
                    benchmarking_model,
                    context.configuration.training.batch_size,
                    (X, xm, Y, ym),
                    sample_config.sample_interval_minutes,
                    output_interval,
                )
                self.head_evaluator.report(
                    context=context,
                    prediction=prediction[:, :, 0, ...],
                    X=X,
                    Xm=xm,
                    Y=Y,
                    Ym=ym,
                    loss=loss.item(),
                    iteration=iteration,
                    location=location,
                    times=times,
                    sign=sign,
                    report_type="benchmark",
                    sample_config=sample_config,
                    output_interval=output_interval,
                    n_day_ahead=self.bench_config.n_day_ahead,
                )
            context.model_container.model.train(mode=True)
            # check if the hooks belong in train or hooks, probably the latter
            self.hooks.benchmark_complete.call(context, iteration)
