# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from s4casting.core.config import IOConfiguration, ModelConfiguration
from s4casting.data.dataset.croissant_adapter import load_measurements_from_croissant, load_weather_from_croissant
from s4casting.data.dataset.indexes import (
    align_start,
    intersect,
    just_location,
    just_time,
    location_id,
    to_intervals,
    union,
)
from s4casting.data.dataset.interface import resample_batch
from s4casting.data.dataset.numpy_data import NumpyData, NumpyDataMatchNearby
from s4casting.data.files.loader import FileAccess
from s4casting.data.utils import SampleConfig


@staticmethod
def _get_locations(meta: dict, to_location_id: bool = False) -> dict:
    locations = {int(i): x for i, x in meta["locations"].items()}
    for x, y in locations.items():
        if "name" not in y:
            y["name"] = str(x)

    if to_location_id:
        return {location_id(y["name"]): y for x, y in locations.items()}

    return locations


@staticmethod
def load_memmap(file_path: str, to_memory: bool, feature_names: list[str] = []):
    """Load a memmapped dataset from file path.

    Args:
        file_path (str): Path to the dataset JSON file.
        to_memory (bool): Move memmap to memory.
        feature_names (list[str], optional): List of feature names to select. Defaults to []

    Returns:
        list[NumpyData]: List of NumpyData datasets.
    """
    meta = FileAccess(file_path).load_json()
    locations = _get_locations(meta)

    spans = FileAccess(str(Path(file_path).parent / meta["spans"])).load_parquet().to_numpy()
    data = np.memmap(
        FileAccess(str(Path(file_path).parent / meta["dataset"])).as_local_path(), dtype="float32", mode="r"
    ).reshape((-1, int(meta["dimension"])))

    if to_memory:
        data = np.array(data)

    unique_locs = np.unique(spans[:, 1])
    lookup = {loc: location_id(locations[loc]["name"]) for loc in unique_locs if loc in locations}
    mapped = np.fromiter((lookup.get(x, -1) for x in spans[:, 1]), dtype=spans[:, 1].dtype)
    spans[mapped != -1, 1] = mapped[mapped != -1]
    spans = spans[mapped != -1]

    locations = {location_id(y["name"]): y for x, y in locations.items()}

    if feature_names:
        data = data[:, [meta["feature_names"].index(x) for x in feature_names]]

    return NumpyData(
        data,
        int(meta["sample_interval_minutes"]) * 60,
        to_intervals(spans, int(meta["sample_interval_minutes"]) * 60),
        spans,
        locations,
    )


def load_external_data(file_path: str):
    """Load external data specified by dict.

    Args:
        file_path (str): Location of external datasets.

    Returns:
        NumpyTimeseriesDataset: The loaded sideloaded dataset, or None if not found.
    """
    meta = FileAccess(file_path).load_json()
    locations = _get_locations(meta, to_location_id=True)
    freq = int(meta["sample_interval_minutes"])

    all_sideloaded_data = []
    all_sideloaded_spans = []
    for i, loc in locations.items():
        if "filename" in loc:
            # load parquet and do some basic cleaning for robustness
            df = pd.read_parquet(loc["filename"])
            # for now divide the measurements by 1 million
            df["measurements"] = df["measurements"] / 1e6
            df["time"] = pd.to_datetime(df["time"], unit="s").dt.round(f"{freq}min")  # type: ignore
            df = df.drop_duplicates("time", keep="last")
            df.set_index("time", inplace=True)
            df = df.reindex(
                pd.date_range(start=df.index.min(), end=df.index.max(), freq=f"{freq}min"), fill_value=np.nan
            ).fillna(value=0)

            start_timestamp = df.index.min().timestamp()
            data = np.float32(df["measurements"].to_numpy().reshape([-1, 1]))

            all_sideloaded_spans.append([sum(len(x) for x in all_sideloaded_data), i, start_timestamp, len(data)])  # ty: ignore[invalid-argument-type]
            all_sideloaded_data.append(data)
    if not all_sideloaded_data:
        return None
    return NumpyData(
        np.concatenate(all_sideloaded_data),
        freq * 60,
        to_intervals(np.array(all_sideloaded_spans, dtype=int), freq * 60),
        np.array(all_sideloaded_spans, dtype=int),
        {i: loc for i, loc in locations.items() if "filename" in loc},
    )


class IntervalDataset:
    """Allows sampling evenly from a set of intervals at some resolution."""

    def __init__(self, intervals: np.ndarray, alignment: int, phase: int = 0):
        """Initialize the IntervalDataset.

        Args:
            intervals (np.ndarray): Array of shape (N, 2), each row is [start, end].
            alignment (int): How many samples apart should two adjacent items be?
            phase (int, optional): Phase offset for alignment. Defaults to 0.
        """
        self.alignment = alignment
        self.phase = phase
        self.intervals = align_start(intervals, alignment, phase) if alignment > 1 else intervals
        s = self.intervals[:, 0]
        e = self.intervals[:, 1]
        steps_per_interval = (e - s - 1) // alignment + 1
        self.cumsum = np.cumsum(steps_per_interval)
        self.n = steps_per_interval.sum()

    def __len__(self):
        """Return the total number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return self.n

    def __getitem__(self, idx: int) -> int:
        """Return the absolute time 't' corresponding to a global index 'idx'.

        Note that 't' here contains both location and time into 1 dimension.

        Args:
            idx (int): Index in the range [0, n).

        Returns:
            int: The time 't' corresponding to the given index, snapped by `alignment`.
        """
        i = np.searchsorted(self.cumsum, idx, side="right")
        # offset within that interval
        offset = idx - (self.cumsum[i - 1] if i > 0 else 0)
        return self.intervals[i, 0] + offset * self.alignment


class TimeseriesDataset:
    """Dataset that provides timeseries samples based on intervals and multiple data sources."""

    def __init__(
        self,
        intervals_dataset: IntervalDataset,
        context_window: int,
        datas: list[list[NumpyData]],
        out_sample_interval: int,
    ):
        """Initialize the TimeseriesDataset.

        Args:
            intervals_dataset (IntervalDataset): The dataset of intervals.
            context_window (int): The context window size.
            datas (list[list[NumpyData]]): The data sources.
            out_sample_interval (int): The output sample interval.
        """
        self.intervals_dataset = intervals_dataset
        self.context_window = context_window
        self.datas = datas
        self.out_sample_interval = out_sample_interval

    def __len__(self):
        """Return the total number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return len(self.intervals_dataset)

    def __getitem__(self, idx: int):
        """Get the timeseries sample at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            torch.Tensor: The timeseries sample tensor.
        """
        t = self.intervals_dataset[idx]
        # sample = [[(T=2592, F=1), (T=2592, F=1)], [(T=216, F=4)]
        sample = [[d.read(t, t + self.context_window) for d in f] for f in self.datas]
        # resample to all be equivalent in shape
        # sample = [[(T=2592, F=1), (T=2592, F=1)], [(T=2592, F=4)]
        sample = [
            [resample_batch(d.copy()[np.newaxis, ...], self.context_window // self.out_sample_interval)[0] for d in f]
            for f in sample
        ]
        # merge over concat (eg: keep one sample)
        # sample = [(N=2, T=2592, F=1), (N=1, T=2592, F=4)]
        sample = [np.stack(f) for f in sample]
        sample = [np.where(np.isnan(feature).all(0), np.nan, np.nansum(feature, 0)) for feature in sample]
        # merge over features
        sample = np.concatenate(sample, axis=-1)
        return torch.tensor(sample), SampleConfig(
            int(just_location(t)),
            self.out_sample_interval / 60,
            self.context_window / (24 * 60 * 60),
            int(just_time(t)),
            sample.shape[1],
        )


@dataclass
class TimeData:
    """Provides features that encode location on interval line in terms of (time, lon, lat)."""

    sample_interval: int
    locations: dict

    def read(self, start: int, end: int) -> np.ndarray:
        """Read time and location data between start and end times.

        Args:
            start (int): Start time.
            end (int): End time.

        Returns:
            np.ndarray: T x 3 = (end-start-1)//self.sample_interval+1 x 3 with (time, lat, lon).
        """
        n = (end - start - 1) // self.sample_interval + 1
        loc_meta = self.locations[just_location(start)]
        times = np.arange(n, dtype=np.float32) * self.sample_interval + just_time(start)
        times = times[:, None]  # shape (n, 1)
        coords = np.tile([loc_meta["lat"], loc_meta["lon"]], (n, 1))
        return np.concatenate((times, coords), axis=-1)


def initialize_per_source_datasets(
    config: IOConfiguration, model_config: ModelConfiguration, dataset_per_source: defaultdict
) -> tuple[defaultdict, NDArray]:
    """Initialize per-source datasets based on configuration.

    Args:
        config (IOConfiguration): IO configuration.
        model_config (ModelConfiguration): Model configuration.
        dataset_per_source (typing.DefaultDict): Default dictionary to hold per-source datasets.

    Returns:
        typing.DefaultDict: Initialized per-source datasets.
    """
    all_locations: dict = {}

    for name in config.feature_order:
        cfg = config.features[name]

        if cfg.loader == "time":
            dataset_per_source[name][name] = TimeData(model_config.base_sample_interval_minutes * 60, all_locations)
            continue

        if cfg.loader == "sideload":
            data = load_external_data(cfg.location)
        elif cfg.loader == "parquet":
            data = load_memmap(cfg.location, config.to_memory, cfg.subset_features)
        elif cfg.loader == "croissant":
            if cfg.nearest_neighbor:
                data = load_weather_from_croissant(
                    Path(cfg.location), subset_features=cfg.subset_features or None, to_memory=config.to_memory
                )
            else:
                data = load_measurements_from_croissant(Path(cfg.location), to_memory=config.to_memory)
        else:
            raise ValueError(f"Unknown loader: {cfg.loader!r}")

        dataset_per_source[name.split("_")[0]][name] = data
        all_locations.update(data.locations)

    # Wrap all nearest_neighbor sources with spatial matching now that all
    # measurement locations are known.
    for name, cfg in config.features.items():
        if not cfg.nearest_neighbor:
            continue
        group = name.split("_")[0]
        wdata = dataset_per_source[group][name]
        dataset_per_source[group][name] = NumpyDataMatchNearby(
            wdata.data,
            wdata.sample_interval,
            wdata.intervals,
            wdata.spans,
            wdata.locations,
            reference_locations=all_locations,
        )

    nn_groups = {name.split("_")[0] for name, cfg in config.features.items() if cfg.nearest_neighbor}
    intervals = reduce(
        intersect,
        [
            reduce(union, [x.intervals for x in v.values()])
            for name, v in dataset_per_source.items()
            if name not in nn_groups and name != "time"
        ],
    )
    return dataset_per_source, intervals


def hash_memmap(mm: np.memmap, chunk: int = 1_000_000):
    """Compute a deterministic hash of the numerical contents of a single memmap.

    This function reads the memmap in fixed-size chunks to avoid loading the entire
    array into memory at once. It computes a SHA-256 hash of the concatenated byte
    representations of all numerical values in the memmap.

    Args:
        mm (np.memmap): The memmap-backed NumPy array to hash.
        chunk (int): The number of elements to read at a time from the memmap.

    Returns:
        str: A SHA-256 hex digest representing the contents of the memmap.
    """
    h = hashlib.sha256()
    for start in range(0, mm.size, chunk):
        end = min(start + chunk, mm.size)
        h.update(mm[start:end].tobytes())
    return h.hexdigest()


def hash_all_memmaps(combined_dataset: dict[str, NumpyData]) -> str:
    """Compute a single deterministic hash representing the combined content of all memmaps in the dataset.

    Each NumpyData object may contain a memmap-backed array (`data`). This function:
    - Computes an individual hash for each memmap's stored values.
    - Sorts these per-memmap hashes to ensure the final result is independent of
      dictionary key order.
    - Combines the sorted hashes into one global SHA-256 digest.

    The resulting hash changes if and only if the underlying numerical contents of
    any memmap change, regardless of how the dataset is keyed or ordered.

    Args:
        combined_dataset (dict[str, NumpyData]):
            A dictionary mapping measurement identifiers to NumpyData objects.

    Returns:
        str: A single SHA-256 hex digest representing all memmaps in the dataset.
    """
    h = hashlib.sha256()
    hashes = [hash_memmap(ds.data) for ds in combined_dataset.values()]  # ty: ignore[invalid-argument-type]
    for hmem in sorted(hashes):
        h.update(hmem.encode())
    return h.hexdigest()
