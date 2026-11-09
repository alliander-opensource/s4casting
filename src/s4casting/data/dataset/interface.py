# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, NamedTuple

import torch

if TYPE_CHECKING:
    from s4casting.core.config import Configuration


class ContextWindowAlignment(int, Enum):
    """Enumeration for context window alignment."""

    Minute = 1
    FiveMinute = 5
    TenMinute = 10
    FifteenMinute = 15
    Hourly = 60
    TwelveHourly = 720
    Daily = 1440
    Weekly = 1440 * 7
    Monthly = 1440 * 28
    TwoMonthly = 2 * 1440 * 28


class SpanIndexInfo(NamedTuple):
    """Information about span indices in a dataset."""

    sample_interval_minutes: int
    context_window_minutes: int
    prediction_window_minutes: int
    alignment: ContextWindowAlignment


def dataset_datetime_convention(timestamp: datetime) -> int:
    """Convert a datetime object to the dataset's datetime convention (minutes since epoch).

    Args:
        timestamp (datetime): The datetime object to convert.

    Returns:
        int: The timestamp in minutes since epoch.
    """
    # Minutes since epoch
    assert timestamp.tzinfo is not None, "Timestamp must have timezone information."
    return int((timestamp - datetime(1970, 1, 1, tzinfo=UTC)).total_seconds())


def get_datetime_from_convention(timestamp: int) -> datetime:
    """Convert a timestamp in the dataset's datetime convention (minutes since epoch) to a datetime object.

    Args:
        timestamp (int): The timestamp in minutes since epoch.

    Returns:
        datetime: The corresponding datetime object.
    """
    return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(minutes=timestamp)


def resample_batch(batch, t, min_coverage=0.5):
    """Resample a batch of sequences to a new length: period means down, linear interpolation up.

    Downsampling (``t`` below the sequence length) averages every output period over the input samples it
    covers, ignoring NaNs; a period with fewer than ``min_coverage`` of its samples valid is NaN. That is what
    meters and system operators report at coarser rates (a 15-minute value is the mean of three 5-minute
    readings, a daily value the mean of the day). Point sampling the series instead, as interpolation does,
    turns a daily or weekly series into draws from the daily cycle. Non-integer ratios (10-minute data at
    15 minutes) average over the overlapping samples. Upsampling (``t`` above the sequence length) linearly
    interpolates between the known samples, which is right for smooth covariates such as hourly weather.

    Args:
        batch (np.ndarray): Input batch of shape (batch_size, seq_len, n_features).
        t (int): Target sequence length.
        min_coverage (float): Fraction of valid input samples a downsampled period needs to be non-NaN.

    Returns:
        np.ndarray: Resampled batch of shape (batch_size, t, n_features).
    """
    n = batch.shape[1]
    if t == n:
        return batch
    x = torch.as_tensor(batch).permute(0, 2, 1)  # (B, F, T)
    if t < n:
        valid = torch.isfinite(x)
        values = torch.where(valid, x, torch.zeros_like(x)).double()
        coverage = torch.nn.functional.adaptive_avg_pool1d(valid.double(), t)
        total = torch.nn.functional.adaptive_avg_pool1d(values, t)
        mean = total / coverage.clamp_min(1e-12)
        out = torch.where(coverage >= min_coverage, mean, torch.full_like(mean, float("nan")))
        return out.permute(0, 2, 1).to(x.dtype).numpy()
    return (
        torch.nn.functional.interpolate(x.to("cpu"), size=t, mode="linear", align_corners=True).permute(0, 2, 1).numpy()
    )


def get_ordered_feature_names(configuration: "Configuration") -> list[str]:
    """Get ordered list of feature names based on configuration.

    Args:
        configuration (Configuration): The configuration object.

    Returns:
        list[str]: Ordered list of feature names.
    """
    ordered_feature_names = []
    for feat_name in configuration.io.feature_order:
        if feat_name == "measurements":
            continue
        ds = configuration.io.features.get(feat_name)
        if not ds:
            continue
        if ds.subset_features:
            ordered_feature_names.extend(ds.subset_features)
        # Provide sensible defaults for known loaders
        elif ds.loader == "time":
            ordered_feature_names.extend(["unixtime", "latitude", "longitude"][: ds.n_features])

    return ordered_feature_names
