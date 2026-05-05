# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

"""Load a Croissant Parquet release into NumpyData / NumpyDataMatchNearby.

The released format is one row per (location, span) for measurements and one row
per (location, span, feature) for weather. This adapter rebuilds the flat
memmap-backed (-1, dim) layout that NumpyData expects so the rest of the
dataset pipeline (read/NaN-fill on gaps, IntervalDataset, TimeseriesDataset)
runs unchanged.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from s4casting.data.dataset.indexes import to_intervals
from s4casting.data.dataset.numpy_data import NumpyData


def _to_memmap(flat: np.ndarray, dim: int) -> np.ndarray:
    """Write a flat float32 buffer to a tempfile and return a read-only memmap reshaped to (-1, dim).

    Returns:
        np.ndarray: Read-only memmap-backed array of shape (-1, dim).
    """
    with tempfile.NamedTemporaryFile(prefix="s4c_croissant_", suffix=".np", delete=False) as tmp:
        np.ascontiguousarray(flat, dtype=np.float32).tofile(tmp)
        path = tmp.name
    return np.memmap(path, dtype="float32", mode="r").reshape(-1, dim)


def _list_array(table: pa.Table, col_name: str) -> pa.ListArray:
    """Return a single (non-chunked) ListArray for the named column.

    Returns:
        pa.ListArray: Unchunked list array for the column.
    """
    col = table.column(col_name).combine_chunks()
    if isinstance(col, pa.ChunkedArray):
        col = col.chunk(0)
    return col


def _build_weather_array(
    order: np.ndarray,
    n_spans: int,
    n_features: int,
    span_loc: np.ndarray,
    span_start: np.ndarray,
    span_num: np.ndarray,
    loc_ids: np.ndarray,
    start_times: np.ndarray,
    nums: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    feat_idx_arr: np.ndarray,
    flat: np.ndarray,
    offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Reconstruct the dense (total_t, n_features) array from per-feature sparse rows.

    Returns:
        tuple: (rebuilt array, spans array (N,4), locations dict)
    """
    span_offsets = np.empty(n_spans, dtype=np.int64)
    span_offsets[0] = 0
    np.cumsum(span_num[:-1], out=span_offsets[1:])

    rebuilt = np.empty((int(span_num.sum()), n_features), dtype=np.float32)

    for s in range(n_spans):
        block_rows = order[s * n_features : (s + 1) * n_features]
        loc = int(span_loc[s])
        st = int(span_start[s])
        n = int(span_num[s])
        cursor = int(span_offsets[s])
        for r in block_rows:
            ri = int(r)
            if int(loc_ids[ri]) != loc or int(start_times[ri]) != st or int(nums[ri]) != n:
                raise ValueError(
                    f"Inconsistent feature rows for span (loc={loc}, start={st}, num={n}); "
                    f"row {ri} has (loc={int(loc_ids[ri])}, start={int(start_times[ri])}, num={int(nums[ri])})"
                )
            rebuilt[cursor : cursor + n, int(feat_idx_arr[ri])] = flat[offsets[ri] : offsets[ri + 1]]

    spans = np.column_stack((span_offsets, span_loc, span_start, span_num)).astype(np.int64)

    locations: dict = {}
    for r in order[::n_features]:
        ri = int(r)
        lid = int(loc_ids[ri])
        if lid not in locations:
            locations[lid] = {"lat": float(lats[ri]), "lon": float(lons[ri]), "name": str(lid)}

    return rebuilt, spans, locations


def load_measurements_from_croissant(release_dir: Path, *, to_memory: bool = False) -> NumpyData:
    """Load measurements.parquet into a NumpyData (dim=1).

    Output is shape-compatible with load_memmap(): same fields, same
    NumpyData.read() NaN-fill semantics on gaps and on NaN-valued samples.

    Returns:
        NumpyData: Loaded measurements dataset.
    """
    table = pq.read_table(release_dir / "measurements.parquet")
    if table.num_rows == 0:
        raise ValueError(f"{release_dir / 'measurements.parquet'} has 0 rows")

    sample_interval_s = int(table.column("sample_interval_s")[0].as_py())

    nums = table.column("num_values").to_numpy().astype(np.int64)
    n = table.num_rows
    spans = np.empty((n, 4), dtype=np.int64)
    spans[:, 0] = np.concatenate(([0], np.cumsum(nums[:-1])))
    spans[:, 1] = table.column("location_id").to_numpy()
    spans[:, 2] = table.column("start_time").to_numpy()
    spans[:, 3] = nums

    values_arr = _list_array(table, "values")
    flat = values_arr.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)

    data = np.ascontiguousarray(flat).reshape(-1, 1) if to_memory else _to_memmap(flat, dim=1)

    loc_ids = table.column("location_id").to_numpy()
    lats = table.column("lat").to_numpy()
    lons = table.column("lon").to_numpy()
    locations: dict = {}
    for raw_lid, lat, lon in zip(loc_ids, lats, lons):
        lid = int(raw_lid)
        if lid not in locations:
            locations[lid] = {"lat": float(lat), "lon": float(lon), "name": str(lid)}

    return NumpyData(
        data=data,
        sample_interval=sample_interval_s,
        intervals=to_intervals(spans, sample_interval_s),
        spans=spans,
        locations=locations,
    )


def load_weather_from_croissant(
    release_dir: Path,
    *,
    subset_features: list[str] | None = None,
    to_memory: bool = False,
) -> NumpyData:
    """Load weather.parquet into a NumpyData.

    Nearest-neighbour location matching is left to the caller (dataset.py wraps
    all weather sources in NumpyDataMatchNearby once measurement locations are known).

    Returns:
        NumpyData: Loaded weather dataset.
    """
    table = pq.read_table(release_dir / "weather.parquet")
    if table.num_rows == 0:
        raise ValueError(f"{release_dir / 'weather.parquet'} has 0 rows")

    sample_interval_s = int(table.column("sample_interval_s")[0].as_py())

    feature_names_col = table.column("feature_name").to_pylist()
    feature_names = list(dict.fromkeys(feature_names_col))
    n_features = len(feature_names)
    feat_idx = {f: i for i, f in enumerate(feature_names)}

    loc_ids = table.column("weather_location_id").to_numpy()
    start_times = table.column("start_time").to_numpy()
    nums = table.column("num_values").to_numpy()
    lats = table.column("lat").to_numpy()
    lons = table.column("lon").to_numpy()

    values_arr = _list_array(table, "values")
    flat = values_arr.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    offsets = values_arr.offsets.to_numpy()

    feat_idx_arr = np.fromiter((feat_idx[f] for f in feature_names_col), dtype=np.int32, count=len(feature_names_col))

    n_rows = table.num_rows
    if n_rows % n_features != 0:
        raise ValueError(f"weather.parquet has {n_rows} rows, not a multiple of n_features={n_features}")

    order = np.lexsort((feat_idx_arr, start_times, loc_ids))
    n_spans = n_rows // n_features
    head_rows = order[::n_features]

    rebuilt, spans, locations = _build_weather_array(
        order,
        n_spans,
        n_features,
        loc_ids[head_rows].astype(np.int64),
        start_times[head_rows].astype(np.int64),
        nums[head_rows].astype(np.int64),
        loc_ids,
        start_times,
        nums,
        lats,
        lons,
        feat_idx_arr,
        flat,
        offsets,
    )

    if subset_features:
        idx = [feat_idx[f] for f in subset_features]
        rebuilt = rebuilt[:, idx]
        n_features = len(idx)

    data = np.ascontiguousarray(rebuilt) if to_memory else _to_memmap(rebuilt.ravel(), dim=n_features)

    return NumpyData(
        data=data,
        sample_interval=sample_interval_s,
        intervals=to_intervals(spans, sample_interval_s),
        spans=spans,
        locations=locations,
    )
