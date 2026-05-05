# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree  # type: ignore[import]

from s4casting.data.dataset.indexes import intersect_single


def build_point_index(spans: np.ndarray, s_per_sample: int) -> np.ndarray:
    """Build a (N, 3) index of (start, end, data_offset) sorted by location+time for fast lookup.

    Returns:
        np.ndarray: Array of shape (N, 3) with columns (start, end, data_offset).
    """
    o = np.lexsort((spans[:, 2], spans[:, 1]))
    s = (spans[o, 1] << 32) + spans[o, 2]
    e = s + spans[o, 3] * int(s_per_sample)
    return np.column_stack((s, e, spans[o, 0]))


def point_to_index_fast(idx: np.ndarray, x: int, sample_interval: int) -> int:
    """Return the flat data index for time x, or -1 if x is not covered by any span."""
    i = np.searchsorted(idx[:, 0], x, side="right") - 1
    if i < 0 or x >= idx[i, 1]:
        return -1
    return int(idx[i, 2] + (np.int64(x) - idx[i, 0]) // sample_interval)


@dataclass
class NumpyData:
    """Provides data access to a memmapped numpy dataset with intervals."""

    data: NDArray
    sample_interval: int
    intervals: NDArray
    spans: NDArray
    locations: dict
    cumsum: NDArray = field(init=False)
    pointcache: NDArray = field(init=False)

    def __post_init__(self) -> None:
        """Compute derived lookup structures after dataclass initialisation."""
        self.cumsum = np.cumsum((self.intervals[:, 1] - self.intervals[:, 0] - 1) // self.sample_interval + 1)
        self.pointcache = build_point_index(self.spans, self.sample_interval)

    def read(self, start: int, end: int) -> np.ndarray:
        """Read data between start and end times.

        Returns:
            np.ndarray: T x F array; NaN where no data exists within the window.
        """
        if end <= start:
            return np.empty(0, dtype=float)
        out = np.full(((end - start - 1) // self.sample_interval + 1, self.data.shape[1]), np.nan, dtype=float)
        for s_i, e_i in intersect_single(self.intervals, start, end):
            count = (e_i - s_i - 1) // self.sample_interval + 1
            dst0 = (s_i - start) // self.sample_interval
            src0 = point_to_index_fast(self.pointcache, s_i, self.sample_interval)
            x = self.data[src0 : src0 + count]
            if len(x):
                out[dst0 : dst0 + len(x)] = x
        return out


@dataclass
class NumpyDataMatchNearby(NumpyData):
    """NumpyData that maps queries to the nearest available weather grid location."""

    reference_locations: dict
    closest_location: dict = field(init=False)

    def __post_init__(self) -> None:
        """Build the nearest-neighbour location mapping after dataclass initialisation."""
        super().__post_init__()
        locs = sorted(self.reference_locations.items())
        dataset_locs = sorted(self.locations.items())
        closest_index = cKDTree([(x["lon"], x["lat"]) for _, x in dataset_locs]).query(
            [(x["lon"], x["lat"]) for _, x in locs], k=1
        )[1]
        self.closest_location = {locs[i][0]: dataset_locs[j][0] for i, j in enumerate(closest_index)}

    def read(self, start: int, end: int) -> np.ndarray:
        """Read data, remapping the location in start/end to the nearest available grid point.

        Returns:
            np.ndarray: T x F array for the nearest grid location.
        """
        if (start >> 32) not in self.closest_location:
            loc = next(iter(self.closest_location.values())) << 32
        else:
            loc = self.closest_location[start >> 32] << 32
        return super().read((start & ((1 << 32) - 1)) | loc, (end & ((1 << 32) - 1)) | loc)
