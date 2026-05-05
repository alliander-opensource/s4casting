# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

from dataclasses import dataclass

import torch
from torch.utils.data import default_collate


@dataclass
class SampleConfig:
    """Class for keeping track of parameters when sampling from dataset."""

    location: int  # TODO map this back to a string?
    sample_interval_minutes: int
    context_window_samples: int
    start_timestamp: float
    n_features: int
    predict_window_samples: float | None = None
    # TODO: add feature names?


@dataclass
class SampleConfigBatch:
    """Class for keeping track of parameters when collating configs from dataset."""

    location: torch.Tensor  # [B] long
    sample_interval_minutes: torch.Tensor  # [B] float
    context_window_samples: torch.Tensor  # [B] long
    start_timestamp: torch.Tensor  # [B] float
    n_features: int
    predict_window_samples: torch.Tensor


def collate_sample_configs(configs: list[SampleConfig]) -> SampleConfigBatch:
    """Collate SampleConfigs into SampleConfigBatch.

    Returns:
        SampleConfigBatch: returns a validated batch of configs for each sample.
    """
    # Always collate required numeric fields into tensors
    location = torch.tensor([c.location for c in configs], dtype=torch.long)
    sample_interval_minutes = torch.tensor([c.sample_interval_minutes for c in configs], dtype=torch.long)
    context_window_samples = torch.tensor([c.context_window_samples for c in configs], dtype=torch.long)
    start_timestamp = torch.tensor([c.start_timestamp for c in configs], dtype=torch.float64)
    n_features = torch.tensor([c.n_features for c in configs], dtype=torch.long)
    predict_window_samples = torch.tensor([c.predict_window_samples for c in configs], dtype=torch.float32)

    if not torch.all(n_features == n_features[0]):
        unique = torch.unique(n_features).tolist()
        raise ValueError(f"Mixed n_features in batch: {unique}")

    return SampleConfigBatch(
        location=location,
        sample_interval_minutes=sample_interval_minutes,
        context_window_samples=context_window_samples,
        start_timestamp=start_timestamp,
        n_features=n_features[0].item(),
        predict_window_samples=predict_window_samples,
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
