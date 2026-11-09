# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

"""How much data each (context window, sample rate) bucket really holds.

A bucket's length is its number of window start positions, one per location per alignment step (a day), so
buckets with the same window length have the same length whatever their rate, and consecutive starts are
near-duplicates. The measures here say what is behind that count: how many locations, how many days of data
they cover, how many non-overlapping windows that makes, and how many target values exist at the bucket's rate.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from s4casting.data.dataset.indexes import just_location, just_time

DAY = 86_400


def summarise_buckets(datasets: Sequence) -> pd.DataFrame:
    """Describe every (context window, rate) bucket of a split.

    Args:
        datasets (Sequence): The per-bucket ``TimeseriesDataset`` objects of a split, before concatenation.

    Returns:
        pd.DataFrame: One row per bucket with ``context_days``, ``rate_minutes``, ``start_positions``,
        ``locations``, ``covered_days`` (distinct days of data the bucket can read), ``windows`` (non-overlapping
        windows that covers) and ``values_at_rate`` (target values those days hold at the bucket's rate).
    """
    rows = []
    for d in datasets:
        iv = np.asarray(d.intervals_dataset.intervals)
        context_s = int(d.context_window)
        rate_s = int(d.out_sample_interval)
        if len(iv) == 0:
            covered_s, n_loc = 0, 0
        else:
            # each row [s, e) is a run of valid start positions; the data it reads spans e - s + one window
            covered_s = int((just_time(iv[:, 1]) - just_time(iv[:, 0])).sum()) + len(iv) * context_s
            n_loc = len(np.unique(just_location(iv[:, 0])))
        rows.append({
            "context_days": context_s // DAY,
            "rate_minutes": rate_s // 60,
            "start_positions": len(d),
            "locations": n_loc,
            "covered_days": covered_s / DAY,
            "windows": covered_s / context_s,
            "values_at_rate": covered_s // rate_s,
        })
    return pd.DataFrame(rows)


def summarise_rates(buckets: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the bucket summary per sample rate.

    Args:
        buckets (pd.DataFrame): Output of :func:`summarise_buckets`.

    Returns:
        pd.DataFrame: One row per rate: number of buckets, total start positions, most locations and most covered
        days any bucket of the rate reaches (the shortest window sees the most data), the non-overlapping windows
        summed over the rate's buckets, and the target values at that rate.
    """
    if buckets.empty:
        return buckets
    out = (
        buckets
        .groupby("rate_minutes")
        .agg(
            buckets=("context_days", "size"),
            context_days=("context_days", lambda s: sorted(set(s))),
            start_positions=("start_positions", "sum"),
            locations=("locations", "max"),
            covered_days=("covered_days", "max"),
            windows=("windows", "sum"),
            values_at_rate=("values_at_rate", "max"),
        )
        .reset_index()
    )
    return out.sort_values("rate_minutes").reset_index(drop=True)


def format_summary(buckets: pd.DataFrame) -> str:
    """Render both tables for a log line.

    Args:
        buckets (pd.DataFrame): Output of :func:`summarise_buckets`.

    Returns:
        str: The bucket table followed by the per-rate table.
    """
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:,.0f}".format):
        per_bucket = buckets.to_string(index=False)
        per_rate = summarise_rates(buckets).to_string(index=False)
    return f"per bucket:\n{per_bucket}\nper rate:\n{per_rate}"
