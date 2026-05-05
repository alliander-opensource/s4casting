# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

from collections import defaultdict
from itertools import product

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import ConcatDataset

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
from s4casting.data.utils import ConcatDatasetSampler, collate_single_interval


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

        context_sample_rates = list(
            product(
                model_config.context_window,
                model_config.input_sample_intervals_minutes,
            )
        )

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
            sample_rate_factor = sample_rate / model_config.base_sample_interval_minutes
            context_window_minutes = context_days * 24 * 60
            _non_benchmark_intervals = fill_gaps(
                non_benchmark_intervals_temp,
                io_config.gap_skip_hours,
                context_days * sample_rate_factor,
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
            model_config.alignment,
            model_config.base_sample_interval_minutes,
            model_config.predict_width,
        )
        # Validation uses train configuration for tasks
        self.validation = self._get_task_dataset(
            train_config.task,
            ConcatDataset(self.validation),
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

        Args:
            train_config (TrainingConfiguration): Training configuration.
            machine (Machine): Machine configuration.
            run_config (RunConfiguration): Run configuration.

        Returns:
            tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: Training and validation data loaders.
        """
        # Create a sampler to properly distribute the data across multiple GPUs
        ddp_kwargs = {"num_replicas": machine.world_size, "rank": machine.ddp.global_rank} if machine.ddp else {}

        train_sampler = ConcatDatasetSampler(
            self.train_ds_lengths,
            train_config.batch_size,
            drop_last=True,
            seed=run_config.seed,
            **ddp_kwargs,
        )

        train_loader = torch.utils.data.DataLoader(
            self.train,
            batch_sampler=train_sampler,
            collate_fn=collate_single_interval,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )

        validation_sampler = ConcatDatasetSampler(
            self.validation_ds_lengths,
            train_config.batch_size,
            drop_last=False,
            seed=run_config.seed,
        )

        validation_loader = torch.utils.data.DataLoader(
            self.validation,
            batch_sampler=validation_sampler,
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
        alignment: int,
        sample_rate: int,
        predict_width: int | tuple[float, float] = 2,
    ) -> PredictionTaskDataset | RandomMaskingTaskDataset | VariablePredictionTaskDataset:
        """Get the task dataset based on the task name.

        Args:
            task_name (str): The name of the task ("prediction", "masking", or "randomprediction").
            dataset: The dataset to wrap in the task dataset.
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
                0,
                (predict_width * 24 * 60) // sample_rate,  # type: ignore
            )

        if task_name == "masking":
            return RandomMaskingTaskDataset(
                dataset,
                alignment // sample_rate,
            )  # type: ignore

        if task_name == "randomprediction":
            if isinstance(predict_width, int):
                raise ValueError("Task 'randomprediction' requires predict width of list.")
            return VariablePredictionTaskDataset(
                dataset,
                predict_dim=0,
                min_predict_width_perc=predict_width[0],
                max_predict_width_perc=predict_width[1],
            )

        raise ValueError(f"Unknown task: {task_name}")
