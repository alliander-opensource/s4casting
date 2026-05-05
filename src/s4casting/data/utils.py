# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import default_collate


@dataclass
class SampleConfig:
    """Class for keeping track of parameters when sampling from dataset."""

    location: int  # TODO map this back to a string?
    sample_interval_minutes: int
    context_window_days: int
    start_timestamp: float
    n_features: int
    predict_window_days: float | None = None
    # TODO: add feature names?


@dataclass
class SampleConfigBatch:
    """Class for keeping track of parameters when collating configs from dataset."""

    location: torch.Tensor  # [B] long
    sample_interval_minutes: int
    context_window_days: int
    start_timestamp: torch.Tensor  # [B] float
    n_features: int
    predict_window_days: torch.Tensor


def collate_sample_configs(configs: list[SampleConfig]) -> SampleConfigBatch:
    """Collate SampleConfigs into SampleConfigBatch.

    Returns:
        SampleConfigBatch: returns a validated batch of configs for each sample.
    """
    # Always collate required numeric fields into tensors
    location = torch.tensor([c.location for c in configs], dtype=torch.long)
    sample_interval_minutes = torch.tensor([c.sample_interval_minutes for c in configs], dtype=torch.long)
    context_window_days = torch.tensor([c.context_window_days for c in configs], dtype=torch.long)
    start_timestamp = torch.tensor([c.start_timestamp for c in configs], dtype=torch.float64)
    n_features = torch.tensor([c.n_features for c in configs], dtype=torch.long)
    predict_window_days = torch.tensor([c.predict_window_days for c in configs], dtype=torch.float32)

    if not torch.all(sample_interval_minutes == sample_interval_minutes[0]):
        unique = torch.unique(sample_interval_minutes).tolist()
        raise ValueError(f"Mixed sample_interval in batch: {unique}")

    if not torch.all(context_window_days == context_window_days[0]):
        unique = torch.unique(context_window_days).tolist()
        raise ValueError(f"Mixed context_window_days in batch: {unique}")

    if not torch.all(n_features == n_features[0]):
        unique = torch.unique(n_features).tolist()
        raise ValueError(f"Mixed n_features in batch: {unique}")

    return SampleConfigBatch(
        location=location,
        sample_interval_minutes=sample_interval_minutes[0].item(),
        context_window_days=context_window_days[0].item(),
        start_timestamp=start_timestamp,
        n_features=n_features[0].item(),
        predict_window_days=predict_window_days,
    )


def collate_single_interval(batch):
    """Collate a batch of TaskSample objects that all share the same sample interval.

    Returns:
        task_batch: TaskSample with each field stacked along the batch dimension.
        sample_interval: The single sample interval for the batch.
    """
    task_samples, configs = zip(*batch)

    task_batch = default_collate(task_samples)
    configs = collate_sample_configs(configs)

    return task_batch, configs


class ConcatDatasetSampler(torch.utils.data.Sampler):
    """Batch sampler for a torch.utils.data.ConcatDataset.

    It that guarantees each mini-batch is drawn from exactly one underlying dataset.
    """

    def __init__(
        self,
        train_ds_lengths,
        batch_size,
        drop_last=True,
        num_replicas=1,
        rank=0,
        seed=0,
    ):
        """Init fn for sampler.

        Parameters
        ----------
        train_ds_lengths (tuple) : Tuple of lengths of each dataset.
        batch_size (int) : Number of samples per batch.
        drop_last (bool) : Drop samples that dont fit in batch.
        num_replicas (int): Worldsize i.e. number of GPUs.
        rank (int): Which gpu is being used.
        seed (int): batcher seed.
        """
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        self.batch_samplers = []
        base = 0
        for L in train_ds_lengths:
            self.batch_samplers.append(
                list(
                    torch.utils.data.BatchSampler(
                        torch.utils.data.SubsetRandomSampler(range(base, base + L)),
                        batch_size=batch_size,
                        drop_last=drop_last,
                    )
                )
            )
            base += L

        self.cumsum = np.cumsum([len(bs) for bs in self.batch_samplers])
        self.total_batches = int(self.cumsum[-1]) if len(self.cumsum) else 0

    def set_epoch(self, epoch):
        """Set the epoch to change the shuffle order for ddp."""
        self.epoch = epoch

    def __iter__(self):
        """Yield batches of indices such that each batch is drawn from a single underlying dataset."""
        if self.total_batches == 0:
            return

        g = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.total_batches, generator=g).tolist()

        if self.num_replicas > 1:
            total_size = (self.total_batches // self.num_replicas) * self.num_replicas
            order = order[:total_size]
            order = order[self.rank : total_size : self.num_replicas]

        for idx in order:
            i = int(np.searchsorted(self.cumsum, idx, "right"))
            prev = int(self.cumsum[i - 1]) if i else 0
            yield self.batch_samplers[i][idx - prev]

    def __len__(self):
        """Return the nominal number of batches produced by the sampler."""
        if self.num_replicas > 1:
            return self.total_batches // self.num_replicas
        return self.total_batches
