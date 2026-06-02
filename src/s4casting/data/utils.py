# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import warnings
from dataclasses import dataclass

import torch
from torch.utils.data import default_collate


def build_valid_context_sampling_pairs(
    context_days=(7, 10, 14, 16, 32, 64, 364),
    sample_intervals_minutes=(5, 10, 15, 30, 60, 1440, 10080),
    min_points=32,
    max_context_len=None,
    interval_context_limits={
        5: {"min_days": 7, "max_days": 10},
        10: {"min_days": 7, "max_days": 14},
        15: {"min_days": 11, "max_days": 32},
        30: {"min_days": 16, "max_days": 64},
        60: {"min_days": 16, "max_days": 64},
        1440: {"min_days": 32, "max_days": 364},
        10080: {"min_days": 64, "max_days": 364},
    },
):
    """Build valid (context_days, sample_interval_minutes) pairs for a fixed padded model input.

    All sample intervals are specified in minutes:
        5      = 5 minutes
        10     = 10 minutes
        60     = 1 hour
        1440   = 1 day
        10080  = 1 week

    A pair is valid if:
      - the sample count is an integer
      - sample count >= min_points
      - sample count <= max_context_samples (if max_context_samples is provided)
      - it satisfies optional manual pruning rules in interval_context_limits

    Args:
        context_days: iterable of allowed context widths in days
        sample_intervals_minutes: iterable of allowed sample intervals in minutes
        min_points: minimum raw sample count required
        max_context_len: fixed padded input length; if None, inferred from valid pairs
        interval_context_limits: optional dict of the form
            {
                5:     {"min_days": 7,   "max_days": 10},
                10:    {"min_days": 7,   "max_days": 14},
                15:    {"min_days": 14,  "max_days": 32},
                30:    {"min_days": 16,  "max_days": 64},
                60:    {"min_days": 16,  "max_days": 64},
                1440:  {"min_days": 32,  "max_days": 364},
                10080: {"min_days": 364, "max_days": 364},
            }

    Returns:
        dict with:
            - "valid_pairs": list of dicts
            - "recommended_max_context_samples": smallest exact padded length covering all valid pairs
    """
    valid = []

    for days in context_days:
        context_minutes = days * 1440

        for interval_minutes in sample_intervals_minutes:
            # Optional policy pruning
            if interval_context_limits and interval_minutes in interval_context_limits:
                limits = interval_context_limits[interval_minutes]
                if days < limits["min_days"] or days > limits["max_days"]:
                    warnings.warn(
                        f"Skipping ({days}d, {interval_minutes}min): context window outside allowed range "
                        f"[{limits['min_days']}d, {limits['max_days']}d] for {interval_minutes}-minute interval.",
                        stacklevel=2,
                    )
                    continue

            points = context_minutes / interval_minutes

            # Must align exactly
            if abs(points - round(points)) > 1e-9:
                warnings.warn(
                    f"Skipping ({days}d, {interval_minutes}min): {days * 1440} minutes is not exactly "
                    f"divisible by {interval_minutes}-minute interval (would give {points:.4f} points).",
                    stacklevel=2,
                )
                continue

            points = round(points)

            if points < min_points:
                warnings.warn(
                    f"Skipping ({days}d, {interval_minutes}min): {points} points is below the minimum of {min_points}.",
                    stacklevel=2,
                )
                continue

            if max_context_len is not None and points > max_context_len:
                warnings.warn(
                    f"Skipping ({days}d, {interval_minutes}min): {points} points exceeds max_context_len "
                    f"of {max_context_len}.",
                    stacklevel=2,
                )
                continue

            valid.append({
                "context_days": days,
                "sample_interval_minutes": interval_minutes,
                "points": points,
            })

    if not valid:
        raise ValueError("No valid (context_days, sample_interval_minutes) pairs found.")

    recommended_max_context_samples = max(row["points"] for row in valid)

    final_max_context_samples = max_context_len if max_context_len is not None else recommended_max_context_samples

    # Add padding metadata
    for row in valid:
        row["pad_left"] = final_max_context_samples - row["points"]
        row["padded_length"] = final_max_context_samples

    return {
        "valid_pairs": valid,
        "recommended_max_context_samples": recommended_max_context_samples,
    }


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
