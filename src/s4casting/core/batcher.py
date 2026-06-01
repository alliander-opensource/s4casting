# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

from collections import defaultdict

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import ConcatDataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from s4casting.core.config import (
    BenchmarkingConfiguration,
    IOConfiguration,
    ModelConfiguration,
    RunConfiguration,
    TrainingConfiguration,
    ValidationConfiguration,
)
from s4casting.core.context import Context
from s4casting.core.hooks import TrainingHooks
from s4casting.core.machine import Machine
from s4casting.core.tasks import PredictionTaskDataset, RandomMaskingTaskDataset, VariablePredictionTaskDataset
from s4casting.data.dataset.dataset import (
    IntervalDataset,
    TimeseriesDataset,
    initialize_per_source_datasets,
)
from s4casting.data.dataset.indexes import (
    add_duration,
    fill_gaps,
    intervals_for_date,
    intervals_for_location,
    intervals_for_year,
    substract,
)
from s4casting.data.utils import build_valid_context_sampling_pairs, collate_single_interval


class Batcher:
    """Creates DataLoaders for training, validation, and benchmarking."""

    def __init__(
        self,
        io_config: IOConfiguration,
        model_config: ModelConfiguration,
        train_config: TrainingConfiguration,
        validation_config: ValidationConfiguration,
        bench_config: BenchmarkingConfiguration,
        run_config: RunConfiguration,
        machine: Machine,
        hooks: TrainingHooks | None = None,
    ):
        """Initialize the Batcher.

        Args:
            io_config (IOConfiguration): IO configuration.
            model_config (ModelConfiguration): Model configuration.
            train_config (TrainingConfiguration): Training configuration.
            validation_config (ValidationConfiguration): Validation configuration.
            bench_config (BenchmarkingConfiguration): Benchmarking configuration.
            run_config (RunConfiguration): Run configuration.
            machine (Machine): Machine configuration.
            hooks (TrainingHooks | None): Training hooks.
        """
        # create Dataloader for each source
        self.datasets_per_source = defaultdict(dict)
        self.datasets_per_source, intervals = initialize_per_source_datasets(
            io_config, model_config, self.datasets_per_source
        )

        valid_context_windows = build_valid_context_sampling_pairs(
            context_days=model_config.context_window,
            sample_intervals_minutes=model_config.input_sample_intervals_minutes,
            min_points=32,
            max_context_len=model_config.ssm.mixer_size if model_config.ssm is not None else None,
            interval_context_limits=io_config.interval_context_limits,
        )
        context_sample_rates = [
            (r["context_days"], r["sample_interval_minutes"]) for r in valid_context_windows["valid_pairs"]
        ]

        # check if we have specified local benchmarks and remove them from trianing set if so
        local_bench = bench_config.benchmarks.get("LocalBenchmark")
        if local_bench is not None:
            bench_context_window_minutes = local_bench.context_window_days * 24 * 60  # type: ignore[possibly-missing-attribute]
            locations = local_bench.locations  # type: ignore[possibly-missing-attribute]
            start_date = int(local_bench.start_date.timestamp())  # type: ignore[possibly-missing-attribute]
        else:
            bench_context_window_minutes = 0
            locations = []
            start_date = 0

        benchmark_intervals, non_benchmark_intervals_temp = self.benchmark_split(
            intervals,
            bench_context_window_minutes,
            locations,
            start_date,
        )

        # Fill gaps for each combination of context window and sample rate
        non_benchmark_intervals = []
        for context_days, sample_rate in context_sample_rates:
            context_window_minutes = context_days * 24 * 60
            _non_benchmark_intervals = fill_gaps(
                non_benchmark_intervals_temp,
                int(24 * context_days * (io_config.gap_skip_perc / 100)),
                context_days,
                io_config.context_window_valid_ratio,
            )
            _non_benchmark_intervals = add_duration(_non_benchmark_intervals, -context_window_minutes * 60)
            non_benchmark_intervals.append(_non_benchmark_intervals)

        self.train, self.validation, self.benchmark = self.train_test_split(
            self.datasets_per_source,
            bench_context_window_minutes,
            validation_config,
            run_config,
            model_config,
            non_benchmark_intervals,
            benchmark_intervals,
            [s for _, s in context_sample_rates],
            [c * 24 * 60 for c, _ in context_sample_rates],
        )

        self.train_ds_lengths = [len(t) for t in self.train]
        self.validation_ds_lengths = [len(t) for t in self.validation]
        self.train = self._get_task_dataset(
            train_config.task,
            ConcatDataset(self.train),
            valid_context_windows["recommended_max_context_samples"],
            train_config.max_retries,
            model_config.alignment,
            model_config.base_sample_interval_minutes,
            model_config.predict_width,
        )
        # Validation uses train configuration for tasks
        self.validation = self._get_task_dataset(
            train_config.task,
            ConcatDataset(self.validation),
            valid_context_windows["recommended_max_context_samples"],
            train_config.max_retries,
            model_config.alignment,
            model_config.base_sample_interval_minutes,
            model_config.predict_width,
        )
        self.train_loader, self.validation_loader = self.create_data_loaders(train_config, machine, run_config)
        # hack for now  to prevent errors
        # Set config vals
        train_config.n_samples_per_epoch = len(self.train_loader)

        if hooks is not None:
            hooks.epoch.register(self.update_sampler_epoch)  # type: ignore[possibly-missing-attribute]

    def benchmark_split(
        self,
        intervals: NDArray,
        context_window_minutes: int,
        locations: list,
        start_date: int,
    ) -> tuple[NDArray, NDArray]:
        """Split benchmarking intervals from normal intervals.

        Args:
            intervals (np.array): Array of benchmark intervals.
            context_window_minutes (int): Context window in minutes.
            locations (tuple): benchmarking locations.
            start_date (tuple): minumum benchmarking date.

        Returns:
            tuple[NDArray, NDArray]: Benchmark intervals, non benchmark intervals.
        """
        if locations:
            benchmark_intervals = np.concatenate([
                intervals_for_date(intervals_for_location(intervals, x), start_date) for x in locations
            ])
        else:
            benchmark_intervals = np.zeros((0, 2), dtype=int)

        non_benchmark_intervals = substract(intervals, benchmark_intervals)
        benchmark_intervals = add_duration(benchmark_intervals, -context_window_minutes * 60)
        return benchmark_intervals, non_benchmark_intervals

    @staticmethod
    def train_test_split(
        datasets_per_source: dict,
        context_window_minutes: int,
        validation_config: ValidationConfiguration,
        run_config: RunConfiguration,
        model_config: ModelConfiguration,
        non_benchmark_intervals: list[NDArray],
        benchmark_intervals: NDArray,
        sample_rates: list,
        context_windows_minutes: list,
    ) -> tuple[list[TimeseriesDataset], list[TimeseriesDataset], TimeseriesDataset]:
        """Split the data into training, validation, and benchmarking datasets.

        This has been moved to a new function due to linting limits
        create time-based validation set from 2024 onward

        Args:
            datasets_per_source (dict): Datasets per data source.
            context_window_minutes (int): Context window in minutes.
            validation_config (ValidationConfiguration): Validation configuration.
            run_config (RunConfiguration): Run configuration.
            model_config (ModelConfiguration): Model configuration.
            non_benchmark_intervals (np.array): Array of non-benchmark intervals.
            benchmark_intervals (np.array): Array of benchmark intervals.
            sample_rates (Sequence[int]): Sequence of sample rates.
            context_windows_minutes (Sequence[int]):Sequence of context window sizes, in minutes

        Returns:
            tuple[list[TimeseriesDataset], list[TimeseriesDataset], TimeseriesDataset]: Training, validation,
                and benchmarking datasets.
        """
        val_intervals = []
        for _non_benchmark_intervals in non_benchmark_intervals:
            if validation_config.split_type == "time":
                chosen_year = validation_config.start_year
                val_intervals.append(intervals_for_year(_non_benchmark_intervals, chosen_year))

            elif validation_config.split_type == "random":
                val_indices = np.random.default_rng(run_config.seed).choice(
                    np.arange(len(_non_benchmark_intervals)),
                    size=int((validation_config.percentage / 100) * len(_non_benchmark_intervals)),
                    replace=False,
                )
                val_intervals.append(_non_benchmark_intervals[val_indices])
            elif validation_config.split_type == "location":
                raise NotImplementedError("Location split not implemented yet ")

        dataset = [list(v.values()) for v in datasets_per_source.values()]
        align = model_config.alignment * 60
        sample = model_config.base_sample_interval_minutes * 60

        train = [
            TimeseriesDataset(
                IntervalDataset(substract(_non_benchmark_intervals, _val_intervals), align),
                _context_window_minutes * 60,  # context_window_minutes * 60,
                dataset,
                sample_rate_minutes * 60,  # sample,
            )
            for (
                sample_rate_minutes,
                _context_window_minutes,
                _non_benchmark_intervals,
                _val_intervals,
            ) in zip(
                sample_rates,
                context_windows_minutes,
                non_benchmark_intervals,
                val_intervals,
            )
        ]
        validation = [
            TimeseriesDataset(
                IntervalDataset(_val_intervals, align),
                _context_window_minutes * 60,  # context_window_minutes * 60,
                dataset,
                sample_rate_minutes * 60,  # sample,
            )
            for sample_rate_minutes, _context_window_minutes, _val_intervals in zip(
                sample_rates, context_windows_minutes, val_intervals
            )
        ]

        benchmark = TimeseriesDataset(
            IntervalDataset(benchmark_intervals, align), context_window_minutes * 60, dataset, sample
        )

        return train, validation, benchmark

    def update_sampler_epoch(self, context: Context, _iteration: int | None) -> None:  # type: ignore
        """Update the epoch for the distributed sampler.

        Args:
            context (Context): Training context.
            _iteration (int | None): Current training iteration. (Unused)
        """
        if context.machine.ddp:
            self.train_loader.batch_sampler.set_epoch(context.trainer.epoch)  # type: ignore

    def create_data_loaders(
        self, train_config: TrainingConfiguration, machine: Machine, run_config: RunConfiguration
    ) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Create data loaders for training and validation.

        Note: we offset the seed by iteration, this should only be triggered when resuming training.

        Args:
            train_config (TrainingConfiguration): Training configuration.
            machine (Machine): Machine configuration.
            run_config (RunConfiguration): Run configuration.

        Returns:
            tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: Training and validation data loaders.
        """
        if machine.ddp:
            train_sampler = DistributedSampler(
                self.train,
                num_replicas=machine.world_size,
                rank=machine.ddp.global_rank,
                shuffle=True,
                seed=run_config.seed + train_config.iteration,
                drop_last=True,
            )
            validation_sampler = DistributedSampler(
                self.validation,
                num_replicas=machine.world_size,
                rank=machine.ddp.global_rank,
                shuffle=True,
                seed=run_config.seed,
                drop_last=False,
            )
        else:
            train_sampler = RandomSampler(
                self.train, generator=torch.Generator().manual_seed(run_config.seed + train_config.iteration)
            )
            validation_sampler = RandomSampler(
                self.validation, generator=torch.Generator().manual_seed(run_config.seed + 1028)
            )

        train_loader = torch.utils.data.DataLoader(
            self.train,
            batch_size=train_config.batch_size,
            sampler=train_sampler,
            drop_last=True,
            collate_fn=collate_single_interval,
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
        )

        validation_loader = torch.utils.data.DataLoader(
            self.validation,
            batch_size=train_config.batch_size,
            sampler=validation_sampler,
            collate_fn=collate_single_interval,
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
        )

        return train_loader, validation_loader

    @staticmethod
    def _get_task_dataset(
        task_name: str,
        dataset: torch.utils.data.Dataset,
        max_context_samples: int,
        max_retries: int,
        alignment: int,
        sample_rate: int,
        predict_width: int | tuple[float, float] = 2,
    ) -> PredictionTaskDataset | RandomMaskingTaskDataset | VariablePredictionTaskDataset:
        """Get the task dataset based on the task name.

        Args:
            task_name (str): The name of the task ("prediction", "masking", or "randomprediction").
            dataset: The dataset to wrap in the task dataset.
            max_context_samples: Maximum number for context_samples to zero pad to.
            max_retries (int) : Maximum number of retries for rejection sampling.
            alignment (int): data alignment.
            sample_rate (int): base sample rate of dataset.
            predict_width (int or tuple[float, float]): prediction window if using prediction task.

        Returns:
            torch.utils.data.Dataset: The task dataset.
        """
        if task_name == "prediction":
            if isinstance(predict_width, list):
                raise ValueError("For a prediction task prediction_width must be an int.")
            return PredictionTaskDataset(
                dataset,
                max_context_samples,
                max_retries,
                0,
                (predict_width * 24 * 60) // sample_rate,  # type: ignore
            )

        if task_name == "masking":
            return RandomMaskingTaskDataset(
                dataset,
                max_context_samples,
                max_retries,
                alignment // sample_rate,
            )  # type: ignore

        if task_name == "randomprediction":
            if isinstance(predict_width, int):
                raise ValueError("Task 'randomprediction' requires predict width of list.")
            return VariablePredictionTaskDataset(
                dataset,
                max_context_samples,
                max_retries,
                predict_dim=0,
                min_predict_width_perc=predict_width[0],
                max_predict_width_perc=predict_width[1],
            )

        raise ValueError(f"Unknown task: {task_name}")
