# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import datetime
import hashlib
from calendar import timegm
from random import Random

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def get_timestamps(
    intervals: NDArray,
    input_width_days: int,
    predict_width_days: int,
    output_sample_interval_minutes: int,
    days_ahead: int,
) -> NDArray:
    """Get times for n day ahead intervals.

    Args:
        intervals (NDarary): Interval in minutes.
        input_width_days (int): Input width days.
        predict_width_days (int): Prediction width days.
        output_sample_interval_minutes (int): Output sample interval minutes.
        days_ahead (int): number of days ahead

    Returns:
        NDArray: Array of datetime objects representing the times.
    """
    end_of_window = datetime.timedelta(days=input_width_days + predict_width_days - 1)
    start_dates = [
        datetime.datetime.fromtimestamp(start_date, tz=datetime.UTC) + end_of_window
        for start_date in intervals & ((1 << 32) - 1)
    ]

    # use start_dates to construct correct dateranges
    times = pd.date_range(
        start=start_dates[0],
        periods=days_ahead * 24 * 60 / output_sample_interval_minutes,
        freq=datetime.timedelta(minutes=output_sample_interval_minutes),
        tz="UTC",
    )
    for start_date in start_dates[1:]:
        temp_index = pd.date_range(
            start=start_date,
            periods=days_ahead * 24 * 60 / output_sample_interval_minutes,
            freq=datetime.timedelta(minutes=output_sample_interval_minutes),
            tz="UTC",
        )
        times = times.union(temp_index)
    return times


def fill_gaps(
    intervals: NDArray,
    gap_skip_hours: int,
    context_width_days: int,
    context_window_valid_ratio: float,
) -> NDArray:
    """Allow gaps in prediction and context window.

    Args:
        intervals (NDarary): Interval in minutes.
        gap_skip_hours (int): gap length to be filled
        context_width_days (int): Input width days.
        context_window_valid_ratio (float):  Percentage of valid data in input window.

    Returns:
        NDArray: updates intervals.
    """
    # simple way to allow data with gaps, jump any gap bigger than some constant, an hour in this case
    gaps = invert(intervals)
    intervals = union(intervals, gaps[distances(gaps) <= gap_skip_hours * 3600])

    # get all points that when sampled will give >=50% good data in the input window
    # and >=80% good data in the prediction window
    # in this case we do both, jump over tiny gaps for free, big gaps are allowed up to some % of total
    context_window = context_width_days * 24 * 3600
    valid_context = get_intervals_with_coverage(
        intervals, context_window, int(context_window * context_window_valid_ratio), int(context_window * 0.05)
    )
    return valid_context.astype(np.int64)


def just_time(a):
    """Get time component from combined location-time integer.

    Args:
        a (np.ndarray): Array of combined location-time integers.

    Returns:
        np.ndarray: Array of time components.
    """
    return a & ((1 << 32) - 1)


def just_location(a):
    """Get location component from combined location-time integer.

    Args:
        a (np.ndarray): Array of combined location-time integers.

    Returns:
        np.ndarray: Array of location components.
    """
    return a >> 32


def distances(A):
    """Get distances of intervals.

    Args:
        A (np.ndarray): Array of intervals of shape (n, 2).

    Returns:
        np.ndarray: Array of distances of shape (n,).
    """
    return A[:, 1] - A[:, 0]


# functions to work with intervals (left inclusive, right exclusive (0,1) contains only the sample 0)
def location_id(name: str):
    """Get location ID from location name.

    Args:
        name (str): Location name.

    Returns:
        int: Location ID.
    """
    # hash any string to 32-bit integer
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "little") & (2**31 - 1)


def clean(A, check=True):
    """Clean intervals.

    Gives the following guarantees:
    - intervals are sorted
    - no empty intervals (and negative length intervals are removed)
    - no intervals completely contained in other intervals
    - perfectly adjacent, or overlapping, intervals are merged.

    Args:
        A (np.ndarray): Array of intervals of shape (n, 2).
        check (bool, optional): Whether to check idempotency. Defaults to True.

    Returns:
        np.ndarray: Cleaned intervals.
    """
    A = A[np.argsort(A[:, 0])]
    A = A[A[:, 1] - A[:, 0] > 0]
    sub = np.where(A[:-1, 1] - A[1:, 1] >= 0)[0]
    A = np.delete(A, sub + 1, axis=0)
    merge = np.where(A[:-1, 1] - A[1:, 0] >= 0)[0]
    if len(merge) == 0:
        return A

    starts = np.flatnonzero(np.r_[True, np.diff(merge) != 1])
    ends = np.r_[starts[1:] - 1, len(merge) - 1]
    first = merge[starts]  # first i in each run
    last = merge[ends]  # last i in each run
    A[first, 1] = np.maximum(A[first, 1], A[last + 1, 1])

    A = np.delete(A, merge + 1, axis=0)

    if check:
        assert np.array_equal(A, clean(A, False)), "Not idempotent!"

    return A


def invert(A):
    """Invert intervals.

    Args:
        A (np.ndarray): Array of intervals of shape (n, 2).

    Returns:
        np.ndarray: Inverted intervals.
    """
    return clean(np.hstack([(-(2**62),), A.reshape((-1,)), (2**62,)]).reshape((-1, 2)))


def intersect_points(x, A):
    """Find intersection points between a set of points and intervals.

    Args:
        x (np.ndarray): Array of points.
        A (np.ndarray): Array of intervals of shape (n, 2).

    Returns:
        tuple: Indices of points and intervals that intersect.
    """
    # returns 2 arrays of indexpairs (i, j) such that x[i] is within A[j]
    if A.shape[0] == 0:
        return np.zeros((0,)), np.zeros((0,))
    i = np.searchsorted(A[:, 0], x, side="right") - 1
    valid = (x < A[i, 1]) & (i >= 0)
    return np.where(valid)[0], i[valid]  # type: ignore


def intersect_single(A, s, e):
    """Intersect intervals with a single interval.

    Args:
        A (np.ndarray): Array of intervals of shape (n, 2).
        s (int): Start of the interval.
        e (int): End of the interval.

    Returns:
        np.ndarray: Intersected intervals.
    """
    A = np.clip(A, s, e)
    return A[distances(A) > 0]


def intersect(A, B):
    """Intersect two sets of intervals.

    Args:
        A (np.ndarray): First array of intervals of shape (n, 2).
        B (np.ndarray): Second array of intervals of shape (m, 2).

    Returns:
        np.ndarray: Intersected intervals.
    """
    # A n B
    return substract(A, substract(A, B))


def substract(a, b):
    """Substract intervals.

    Args:
        a (np.ndarray): First array of intervals of shape (n, 2).
        b (np.ndarray): Second array of intervals of shape (m, 2).

    Returns:
        np.ndarray: Resulting intervals after substraction.
    """
    # A \ B
    a, b = clean(a), clean(b)
    if len(a) == 0 or len(b) == 0:
        return a
    p = np.concatenate([a.ravel(), b.ravel()])
    sort_idx = np.argsort(p, kind="stable")
    p = p[sort_idx]
    is_a = sort_idx < len(a) * 2
    is_s = (sort_idx & 1) == 0
    As = is_a & is_s
    Ae = is_a & ~is_s
    Bs = ~is_a & is_s
    Be = ~is_a & ~is_s
    openA = (np.cumsum(As) - np.cumsum(Ae)) > 0
    openB = (np.cumsum(Bs) - np.cumsum(Be)) > 0
    keep = (As[:-1] & Ae[1:] & ~openB[:-1]) | (Be[:-1] & Ae[1:]) | (As[:-1] & Bs[1:]) | (Be[:-1] & Bs[1:] & openA[:-1])
    keep = np.where(keep)[0]
    out = np.dstack([p[keep], p[keep + 1]])[0]
    return clean(out)


def union(A, B):
    """Union of two sets of intervals.

    Args:
        A (np.ndarray): First array of intervals of shape (n, 2).
        B (np.ndarray): Second array of intervals of shape (m, 2).

    Returns:
        np.ndarray: Union of the two sets of intervals.
    """
    # A u B
    return clean(np.vstack((A, B)))


def add_duration(A, n):
    """Add duration to intervals.

    Args:
        A (np.ndarray): Array of intervals of shape (n, 2).
        n (int): Duration to add.

    Returns:
        np.ndarray: Intervals with added duration.
    """
    # increases all invervals' length by n
    return clean(np.column_stack((A[:, 0], A[:, 0] + (A[:, 1] - A[:, 0] + n))))


def align(A, mod):
    """Align intervals with a given modulus.

    Args:
        A (np.ndarray): Array of intervals of shape (n, 2).
        mod (int): Modulus for alignment.

    Returns:
        np.ndarray: Aligned intervals.
    """
    # aligns start and end times (not locations) with `mod` units
    # starts will be moved forward, ends backwards. You will receive a subset.
    # this function is idempotent
    start = (np.ceil((A[:, 0] % 2**32) / mod) * mod).astype(int) + (A[:, 0] // 2**32) * 2**32
    end = (np.floor((A[:, 1] % 2**32) / mod) * mod).astype(int) + (A[:, 1] // 2**32) * 2**32
    return clean(np.column_stack((start, end)))


def align_start(A, mod, phase=0):
    """Align start times of intervals with a given modulus and phase.

    Args:
        A (np.ndarray): Array of intervals of shape (n, 2).
        mod (int): Modulus for alignment.
        phase (int, optional): Phase for alignment. Defaults to 0.

    Returns:
        np.ndarray: Aligned intervals.
    """
    # aligns start and end times (not locations) with `mod` units
    # starts will be moved forward, ends backwards. You will receive a subset.
    # this function is idempotent
    A = A - (phase % mod)
    start = (np.ceil((A[:, 0] % 2**32) / mod) * mod).astype(int) + (A[:, 0] // 2**32) * 2**32
    A = clean(np.column_stack((start, A[:, 1])))
    return A + (phase % mod)


def indexable_spans(A, B, c, p):
    """Get indexable spans from intervals.

    Args:
        A (np.ndarray): Array of data intervals of shape (n, 2).
        B (np.ndarray): Array of bad data intervals of shape (m, 2).
        c (int): Context length.
        p (int): Prediction length.

    Returns:
        np.ndarray: Indexable spans.
    """
    # given some spans of data A and some spans of 'bad data' B masks
    # will return all spans from which you can read `c+p` values such that
    # the range c:c+p has no intersection with B.
    return substract(add_duration(A, -(c + p)), add_duration(B - (c + p + 1), p - 1))


def to_intervals(spans, s_per_sample):
    """Convert spans to intervals.

    Args:
        spans (np.ndarray): Array of spans.
        s_per_sample (int): Samples per sample duration.

    Returns:
        np.ndarray: Converted intervals.
    """
    # get 1D intervals from collection of spans
    start = spans[:, 1] * (2**32) + spans[:, 2]
    end = start + spans[:, 3] * s_per_sample
    return np.sort(np.column_stack((start, end)), axis=0)


def sample_points(A, n, mod):
    """Sample points from intervals.

    Args:
        A (np.ndarray): Array of intervals of shape (n, 2).
        n (int): Number of points to sample.
        mod (int): Modulus for sampling.

    Returns:
        np.ndarray: Sampled points.
    """
    A = clean(A)
    An = A[:, 1] - A[:, 0]
    selected = np.random.choice(len(A), size=n, p=An / An.sum())
    selected = A[selected, :]
    start_time = (np.ceil((selected[:, 0] % 2**32) / mod) * mod).astype(int)
    end_time = (np.floor((selected[:, 1] % 2**32) / mod) * mod).astype(int)
    random_time = np.random.rand(n) * (end_time - start_time)
    return (selected[:, 0] // 2**32) * 2**32 + start_time + (random_time // mod).astype(int) * mod


def sample_all_points(A, n):
    """Sample all points from intervals with a given step.

    Args:
        A (np.ndarray): Array of intervals of shape (n, 2).
        n (int): Step size for sampling.

    Returns:
        np.ndarray: Sampled points.
    """
    A = clean(np.column_stack([np.ceil(A[:, 0] / n) * n, np.floor(A[:, 1] / n) * n]).astype(int))
    return np.concatenate([np.arange(i, j + 1, n) for (i, j) in A])


def points_to_indices(spans, x, s_per_sample):
    """Convert points to indices based on spans.

    Args:
        spans (np.ndarray): Array of spans.
        x (np.ndarray): Array of points.
        s_per_sample (int): Samples per sample duration.

    Returns:
        np.ndarray: Converted indices.
    """
    R = to_intervals(spans, s_per_sample)
    Ai, Ri = intersect_points(x, R)
    return spans[Ri, 3] + (x[Ai] - R[Ri, 0]) // s_per_sample


def point_to_index(spans, R, x, s_per_sample):
    """Convert a single point to an index based on spans.

    Args:
        spans (np.ndarray): Array of spans.
        R (np.ndarray): Array of intervals.
        x (int): Point to convert.
        s_per_sample (int): Samples per sample duration.

    Returns:
        int: Converted index.
    """
    x = np.array([x])
    Ai, Ri = intersect_points(x, R)
    loc = R[Ri, 0] >> 32
    start_time_s = R[Ri, 0] & ((1 << 32) - 1)
    end_time_s = R[Ri, 1] & ((1 << 32) - 1)
    Si = np.where(
        (spans[:, 1] == loc) & (spans[:, 2] <= start_time_s) & (spans[:, 2] + spans[:, 3] * s_per_sample >= end_time_s)
    )[0]
    return (spans[Si, 0] + (x[Ai] - R[Ri, 0]) // s_per_sample)[0]


def point_in_intervals(A, x):
    """Check if a point is in any of the intervals.

    Args:
        A (np.ndarray): Array of intervals of shape (n, 2).
        x (int): Point to check.

    Returns:
        bool: True if the point is in any interval, False otherwise.
    """
    return np.any((A[:, 0] <= x) & (A[:, 1] > x))


def to_spans(spans, A, s_per_sample):
    """Convert intervals to spans.

    Args:
        spans (np.ndarray): Array of spans.
        A (np.ndarray): Array of intervals of shape (n, 2).
        s_per_sample (int): Samples per sample duration.

    Returns:
        np.ndarray: Converted spans.
    """
    assert s_per_sample > 0, "Duration of a sample must be more than 0, otherwise all intervals will be empty"
    # get spans from mutated 1D intervals original reference is needed to reconstruct index
    return np.column_stack((
        points_to_indices(spans, A[:, 0], s_per_sample),
        A[:, 0] // (2**32),
        A[:, 0] % (2**32),
        (A[:, 1] - A[:, 0]) // s_per_sample,
    ))


def filter_locations(index, locations):
    """Filter intervals based on given locations.

    Args:
        index (np.ndarray): Array of intervals of shape (n, 2).
        locations (set): Set of location IDs to filter by.

    Returns:
        np.ndarray: Filtered intervals.
    """
    return index[np.isin(index[:, 0] // (2**32), list(locations))]


def split_indexes_on_given(indexes, locations):
    """Split intervals based on given locations.

    Args:
        indexes (np.ndarray): Array of intervals of shape (n, 2).
        locations (set): Set of location IDs to split by.

    Returns:
        tuple: Two arrays of intervals, first with given locations, second without.
    """
    return (
        indexes[np.isin(indexes[:, 0] // (2**32), list(locations)), :],
        indexes[~np.isin(indexes[:, 0] // (2**32), list(locations)), :],
    )


def split_intersected_indexes(indexes, percentage: int, seed: int = 0):
    """Split intervals based on intersection with a percentage of locations.

    Args:
        indexes (np.ndarray): Array of intervals of shape (n, 2).
        percentage (int): Percentage of locations to include in the first split.
        seed (int, optional): Seed for randomization. Defaults to 0.

    Returns:
        tuple: Two arrays of intervals, first with selected locations, second with the rest.
    """
    locs = list(set(indexes[:, 0] // (2**32)))
    Random(seed).shuffle(locs)
    return split_indexes_on_given(indexes, locs[: len(locs) * percentage // 100])


def get_intervals_with_coverage(A, N, M, resolution):
    """Get intervals with sufficient coverage.

    Will return intervals containing all points x for which `total_length((x..x+N) & A) >= M`
    doing this sample perfect is quite slow, so computes accurate to within `resolution` samples
    something like resolution = (N-M)//10 is a good rule of thumb.
    For 80% true data that would result in 20%/10 = +-2% error.

    Args:
        A (np.ndarray): Array of intervals of shape (n, 2).
        N (int): Length of the interval to check.
        M (int): Minimum required coverage within the interval.
        resolution (int): Resolution for coverage checking.

    Returns:
        np.ndarray: Intervals with sufficient coverage.
    """
    # shift all intervals by each step 0..N
    # this will cause all points not in A to be covered by as many intervals
    # as how many samples will be reached from there within N steps.
    t = np.arange(round(N / resolution), dtype=np.int64) * resolution
    S = np.stack([(A[:, 0, None] - t).ravel(), (A[:, 1, None] - t).ravel()], axis=1)

    # for any index, how many intervals in S do we hit?
    # convert S to a flat event list:
    p = S.ravel()
    is_start = np.tile([True, False], S.shape[0])
    order = np.lexsort((~is_start, p))
    p, is_start = p[order], is_start[order]

    # use cumsum to get amount of active intervals per index
    c = np.cumsum(np.where(is_start, 1, -1))

    # select all indexes that have sufficient active intervals
    idx = (c[:-1] >= round(M / resolution)) & (p[:-1] != p[1:])
    return union(add_duration(A, -M), np.stack([p[:-1][idx], p[1:][idx] + resolution + 1], axis=1))


def intervals_for_year(intervals: np.ndarray, year: int) -> np.ndarray:
    """Get intervals for a specific year.

    Args:
        intervals (np.ndarray): Array of intervals of shape (n, 2).
        year (int): Year to filter intervals for.

    Returns:
        np.ndarray: Intervals for the specified year.
    """
    return clean(
        (just_location(intervals) << 32)
        | np.clip(just_time(intervals), timegm((year, 1, 1, 0, 0, 0)), timegm((year + 1, 1, 1, 0, 0, 0)))
    )


def intervals_for_location(intervals, location: str):
    """Get the substet of `dataset.intervals` that match location.

    Args:
        intervals (np.ndarray): Array of intervals of shape (n, 2).
        location (str): Location name.

    Returns:
        np.ndarray: Intervals for the specified location.
    """
    return intervals[(intervals[:, 0] >> 32) == location_id(location), :]


def intervals_for_date(intervals: np.ndarray, date: int) -> np.ndarray:
    """Get intervals for a specific year.

    Args:
        intervals (np.ndarray): Array of intervals of shape (n, 2).
        date (int): minumum date.

    Returns:
        np.ndarray: Intervals for the specified year.
    """
    return clean((just_location(intervals) << 32) | np.clip(just_time(intervals), date, None))
