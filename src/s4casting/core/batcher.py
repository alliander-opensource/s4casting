# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

from calendar import timegm
from collections import defaultdict

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import ConcatDataset, Sampler

from s4casting.core.config import (
    BenchmarkingConfiguration,
    CPMConfiguration,
    IOConfiguration,
    ModelConfiguration,
    RunConfiguration,
    TrainingConfiguration,
    ValidationConfiguration,
)
from s4casting.core.context import Context
from s4casting.core.hooks import TrainingHooks
from s4casting.core.machine import Machine
from s4casting.core.tasks import (
    ContiguousPatchMaskingTaskDataset,
    PredictionTaskDataset,
    RandomMaskingTaskDataset,
    VariablePredictionTaskDataset,
)
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
    substract,
)
from s4casting.data.dataset.summary import summarise_buckets
from s4casting.data.utils import build_valid_context_sampling_pairs, collate_single_interval


def _limit_worker_threads(_worker_id: int) -> None:
    """Keep every torch op in a DataLoader worker single-threaded.

    Each worker process otherwise sizes torch's intra-op thread pool to os.cpu_count(), which
    ignores cgroup CPU quotas; N workers x M threads then thrash the cores and microsecond-scale
    ops (mask building, covariate permutation) cost milliseconds. Runs in the worker before any
    dataset code, which is early enough because the pool is created lazily on first use.

    Args:
        _worker_id (int): Worker index (unused).
    """
    torch.set_num_threads(1)


def pads_to_fixed_length(model_config: ModelConfiguration) -> bool:
    """Whether training samples must be zero-padded to one fixed length.

    Only the SSM needs it: its mixer is a fixed linear layer over the sequence. The transformer works on
    patches with rotary positions, and a batch always comes from one (context window, rate) group, so its
    samples already share a length and padding would only add masked tokens.

    Args:
        model_config (ModelConfiguration): Model configuration.

    Returns:
        bool: True when samples are padded to the longest valid sequence.
    """
    return model_config.model == "ssm"


class SingleRateBatchSampler(Sampler[list[int]]):
    """Yield batches that each come from one sub-dataset of a ``ConcatDataset``.

    The training set is a concatenation of one dataset per (context window, sample rate) pair. The
    model decodes a batch at a single rate (the loss reshapes by the input/output ratio), so a batch must
    never mix rates. Indices are shuffled within every group, cut into batches, and the batch order is
    shuffled, all from a seed that advances with :meth:`set_epoch`. Under DDP every rank builds the same
    batch list and takes every ``num_replicas``-th batch, so ranks stay in step and see disjoint data.
    """

    def __init__(
        self,
        group_lengths: list[int],
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        """Initialize the sampler.

        Args:
            group_lengths (list[int]): Length of every sub-dataset, in concatenation order.
            batch_size (int): Samples per batch.
            shuffle (bool): Shuffle samples within a group and the batch order.
            drop_last (bool): Drop the incomplete last batch of every group.
            seed (int): Base seed; the epoch is added to it.
            num_replicas (int): Number of DDP ranks sharing the batches.
            rank (int): Rank of this process.
        """
        self.group_lengths = [int(n) for n in group_lengths]
        self.offsets = np.concatenate([[0], np.cumsum(self.group_lengths)[:-1]]).tolist()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Advance the shuffle seed so every epoch draws a different order.

        Args:
            epoch (int): Epoch index.
        """
        self.epoch = epoch

    def _num_batches_total(self) -> int:
        per_group = [n // self.batch_size if self.drop_last else -(-n // self.batch_size) for n in self.group_lengths]
        total = sum(per_group)
        return total - total % self.num_replicas

    def __len__(self) -> int:
        """Return the number of batches this rank yields per epoch.

        Returns:
            int: Number of batches.
        """
        return self._num_batches_total() // self.num_replicas

    def __iter__(self):
        """Yield index lists, each within a single rate group.

        Memory stays at one int64 permutation per group (as a plain random sampler) plus one group id and
        one chunk index per batch; the index lists themselves are built lazily, one batch at a time.

        Yields:
            list[int]: Indices into the concatenated dataset.
        """
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        perms = [
            (torch.randperm(length, generator=generator) if self.shuffle else torch.arange(length)) + offset
            for offset, length in zip(self.offsets, self.group_lengths, strict=True)
        ]
        batches_per_group = [
            n // self.batch_size if self.drop_last else -(-n // self.batch_size) for n in self.group_lengths
        ]
        # one entry per batch: which group it comes from, and which chunk of that group's permutation
        group_of_batch = torch.cat([
            torch.full((count,), group, dtype=torch.long) for group, count in enumerate(batches_per_group)
        ])
        if self.shuffle:
            group_of_batch = group_of_batch[torch.randperm(len(group_of_batch), generator=generator)]
        chunk_of_batch = torch.empty_like(group_of_batch)
        for group in range(len(self.group_lengths)):
            members = group_of_batch == group
            chunk_of_batch[members] = torch.arange(int(members.sum()))
        total = self._num_batches_total()
        for group, chunk in zip(
            group_of_batch[self.rank : total : self.num_replicas].tolist(),
            chunk_of_batch[self.rank : total : self.num_replicas].tolist(),
            strict=True,
        ):
            start = chunk * self.batch_size
            yield perms[group][start : start + self.batch_size].tolist()


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

        # get metadata covariate indices to prevent permutation
        self.n_time_features = sum(
            io_config.features[name].n_features
            for name in io_config.feature_order
            if io_config.features[name].loader == "time"
        )
        # The mixer is a fixed Linear(mixer_size, mixer_size) over the post-encoder
        # sequence. The linear PatchEncoder collapses patch_size raw steps into one
        # token, so the raw-context cap is scaled by patch_size; the SSEncoder keeps
        # the sequence length, so it is not.
        if model_config.ssm is not None and model_config.ssm.mixer_size is not None:
            max_context_len = model_config.ssm.mixer_size
            if model_config.patch_encoder.arch == "linear":
                max_context_len *= model_config.patch_encoder.patch_size
        else:
            max_context_len = None

        # A patch encoder cuts the sequence into whole patches, so every (context window, rate) pair
        # must give a multiple of patch_size samples; the SSEncoder keeps the raw sequence length.
        patch_size = model_config.patch_encoder.patch_size if model_config.patch_encoder.arch != "ss" else None
        valid_context_windows = build_valid_context_sampling_pairs(
            context_days=model_config.context_window,
            sample_intervals_minutes=model_config.input_sample_intervals_minutes,
            min_points=32,
            max_context_len=max_context_len,
            patch_size=patch_size,
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
        # What the bucket lengths stand for: start positions, not distinct data. The loggers report it on
        # run start, so the exposure per rate and the real amount of coarse-rate data are visible in every run.
        self.bucket_summary = summarise_buckets(self.train)
        padded_length = (
            valid_context_windows["recommended_max_context_samples"] if pads_to_fixed_length(model_config) else None
        )
        self.train = self._get_task_dataset(
            train_config.task,
            ConcatDataset(self.train),
            padded_length,
            train_config.max_retries,
            model_config.alignment,
            model_config.base_sample_interval_minutes,
            model_config.predict_width,
            n_time_features=self.n_time_features,
            covariate_dropout=train_config.covariate_dropout,
            patch_size=model_config.patch_encoder.patch_size,
            cpm=train_config.cpm,
            sample_intervals_minutes=model_config.input_sample_intervals_minutes,
        )
        self.train.covariate_dropout_all_p = train_config.covariate_dropout_all_p
        self.train.covariate_horizon_dropout_p = train_config.covariate_horizon_dropout_p
        # Validation uses train configuration for tasks, see validation_task for the CPM exception.
        validation_task = self.validation_task(train_config)
        validation_cpm = train_config.cpm
        if validation_task == "prediction" and not isinstance(model_config.predict_width, int):
            # A fraction-range horizon is drawn by the CPM dataset; keep that draw for validation but
            # switch the context masks off so the validation loss stays a plain forecasting loss.
            validation_task = "cpm"
            validation_cpm = train_config.cpm.model_copy(update={"p_max": 0.0})
        self.validation = self._get_task_dataset(
            validation_task,
            ConcatDataset(self.validation),
            padded_length,
            train_config.max_retries,
            model_config.alignment,
            model_config.base_sample_interval_minutes,
            model_config.predict_width,
            n_time_features=self.n_time_features,
            covariate_dropout=False,
            patch_size=model_config.patch_encoder.patch_size,
            cpm=validation_cpm,
            sample_intervals_minutes=model_config.input_sample_intervals_minutes,
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
        for _non_benchmark_intervals, _ctx_minutes in zip(non_benchmark_intervals, context_windows_minutes):
            if validation_config.split_type == "time":
                # Strict temporal holdout: a training window starting at `s` reads
                # [s, s + context window], so any window start within one context
                # window of the cutoff would leak post-cutoff data into training.
                # All window starts from (cutoff - context) onwards go to validation.
                cutoff = timegm((validation_config.start_year, 1, 1, 0, 0, 0))
                val_intervals.append(intervals_for_date(_non_benchmark_intervals, cutoff - _ctx_minutes * 60))

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
        num_replicas = machine.world_size if machine.ddp else 1
        rank = machine.ddp.global_rank if machine.ddp else 0
        # Every batch comes from one (context window, sample rate) group; see SingleRateBatchSampler.
        train_sampler = SingleRateBatchSampler(
            self.train_ds_lengths,
            train_config.batch_size,
            drop_last=True,
            seed=run_config.seed + train_config.iteration,
            num_replicas=num_replicas,
            rank=rank,
        )
        validation_sampler = SingleRateBatchSampler(
            self.validation_ds_lengths,
            train_config.batch_size,
            drop_last=False,
            seed=run_config.seed + 1028,
            num_replicas=num_replicas,
            rank=rank,
        )

        train_loader = torch.utils.data.DataLoader(
            self.train,
            batch_sampler=train_sampler,
            collate_fn=collate_single_interval,
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
            worker_init_fn=_limit_worker_threads,
        )

        validation_loader = torch.utils.data.DataLoader(
            self.validation,
            batch_sampler=validation_sampler,
            collate_fn=collate_single_interval,
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
            worker_init_fn=_limit_worker_threads,
        )

        return train_loader, validation_loader

    @staticmethod
    def validation_task(train_config: TrainingConfiguration) -> str:
        """Task used for the validation set.

        CPM falls back to the plain prediction task unless ``training.cpm.apply_to_validation`` is set, so the
        validation loss stays comparable with 'prediction' runs. With a fraction-range ``predict_width`` the
        batcher keeps the CPM dataset for validation with ``p_max = 0`` (no context masks) so the horizon is
        drawn the same way.

        Args:
            train_config (TrainingConfiguration): Training configuration.

        Returns:
            str: The validation task name.
        """
        if train_config.task == "cpm" and not train_config.cpm.apply_to_validation:
            return "prediction"
        return train_config.task

    @staticmethod
    def _get_task_dataset(
        task_name: str,
        dataset: torch.utils.data.Dataset,
        max_context_samples: int,
        max_retries: int,
        alignment: int,
        sample_rate: int,
        predict_width: int | tuple[float, float] = 2,
        *,
        n_time_features: int,
        covariate_dropout: bool,
        patch_size: int | None = None,
        cpm: CPMConfiguration | None = None,
        sample_intervals_minutes: list[int] | None = None,
    ) -> (
        PredictionTaskDataset
        | RandomMaskingTaskDataset
        | VariablePredictionTaskDataset
        | ContiguousPatchMaskingTaskDataset
    ):
        """Get the task dataset based on the task name.

        Args:
            task_name (str): The name of the task ("prediction", "masking", "randomprediction", or "cpm").
            dataset: The dataset to wrap in the task dataset.
            max_context_samples: Length to zero pad every sample to, or None for no padding.
            max_retries (int) : Maximum number of retries for rejection sampling.
            alignment (int): data alignment.
            sample_rate (int): base sample rate of dataset.
            predict_width (int or tuple[float, float]): prediction window in days, or a (min, max) fraction
                range of the sample length for the "randomprediction" and "cpm" tasks.
            n_time_features: The number of time features (1 - unix time; 3 - unix time, lat, lon).
            covariate_dropout: Boolean, if training with covariate dropout.
            patch_size (int | None): Model patch size in samples; required for the "cpm" task.
            cpm (CPMConfiguration | None): Contiguous patch masking settings; required for the "cpm" task.
            sample_intervals_minutes (list[int] | None): Rates that will be trained; resolves the per-rate CPM
                block sizes when the CPM settings are given in days.

        Returns:
            torch.utils.data.Dataset: The task dataset.
        """
        if task_name == "prediction":
            if not isinstance(predict_width, int):
                raise ValueError("For a prediction task prediction_width must be an int.")
            return PredictionTaskDataset(
                dataset,
                max_context_samples,
                max_retries,
                0,
                (predict_width * 24 * 60) // sample_rate,
                n_time_features=n_time_features,
                covariate_dropout=covariate_dropout,
                base_sample_interval_minutes=sample_rate,
            )

        if task_name == "masking":
            return RandomMaskingTaskDataset(
                dataset,
                max_context_samples,
                max_retries,
                alignment // sample_rate,
                n_time_features=n_time_features,
                covariate_dropout=covariate_dropout,
            )  # type: ignore

        if task_name == "cpm":
            if not isinstance(predict_width, int | tuple | list):
                raise ValueError("For a cpm task prediction_width must be an int or a (min, max) fraction range.")
            if patch_size is None or cpm is None:
                raise ValueError("Task 'cpm' requires patch_size and a CPMConfiguration.")
            rates = sorted({*(sample_intervals_minutes or []), sample_rate})
            patches_by_rate = cpm.patches_by_rate(patch_size, rates)
            continuous_min, continuous_max, min_context_patches = patches_by_rate[sample_rate]
            fixed_width = predict_width if isinstance(predict_width, int) else 1
            return ContiguousPatchMaskingTaskDataset(
                dataset,
                max_context_samples,
                max_retries,
                0,
                (fixed_width * 24 * 60) // sample_rate,
                n_time_features=n_time_features,
                covariate_dropout=covariate_dropout,
                patch_size=patch_size,
                p_max=cpm.p_max,
                continuous_min=continuous_min,
                continuous_max=continuous_max,
                min_context_patches=min_context_patches,
                base_sample_interval_minutes=sample_rate,
                patches_by_rate=patches_by_rate,
                predict_width_range=None if isinstance(predict_width, int) else tuple(predict_width),
            )

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
                n_time_features=n_time_features,
                covariate_dropout=covariate_dropout,
            )

        raise ValueError(f"Unknown task: {task_name}")
