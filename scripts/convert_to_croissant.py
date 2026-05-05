#!/usr/bin/env python3

# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

"""Convert LianderPower memmap datasets to a Croissant release.

Outputs:
    <out_dir>/measurements.parquet
    <out_dir>/weather.parquet
    <out_dir>/croissant.json

Measurements link to weather by `weather_location_id`. The measurement JSON
should contain either that field for each location or anonymized `lat`/`lon`
coordinates that can be matched to the weather JSON.

Usage:
    uv run python scripts/convert_to_croissant.py \
        --measurements data/anon_measurements_k5.json \
        --weather data/weather_openmeteo_anon.json \
        --out lianderpower-anon-v1 \
        --dataset-url https://<final-dataset-landing-page> \
        --citation "<dataset or paper citation>"
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

EARTH_RADIUS_KM = 6371.0088
VALID_SENSOR_UNITS = {"P", "I", "X"}
VALID_MEASUREMENT_TYPES = {"pseudo_aggregate", "native_aggregate"}

DEFAULT_WEATHER_METADATA: dict[str, Any] = {
    "weather_provider": "Open-Meteo Historical Weather API",
    "weather_api_endpoint": "https://archive-api.open-meteo.com/v1/archive",
    "weather_license": "CC-BY-4.0",
    "weather_license_url": "https://creativecommons.org/licenses/by/4.0/",
    "weather_citation": (
        "Zippenfenig, P. (2023). Open-Meteo.com Weather API [Computer software]. "
        "Zenodo. https://doi.org/10.5281/zenodo.7970649"
    ),
    "weather_variables": [
        {
            "name": "temperature_2m",
            "unit": "degree Celsius",
            "height": "2 m",
            "temporal_semantics": "hourly instantaneous value, as returned by Open-Meteo",
        },
        {
            "name": "wind_speed_100m",
            "unit": "km h^-1",
            "height": "100 m",
            "temporal_semantics": "hourly instantaneous value, as returned by Open-Meteo",
        },
        {
            "name": "shortwave_radiation",
            "unit": "W m^-2",
            "temporal_semantics": "hourly radiation variable, as returned by Open-Meteo",
        },
        {
            "name": "direct_normal_irradiance",
            "unit": "W m^-2",
            "temporal_semantics": "hourly radiation variable, as returned by Open-Meteo",
        },
    ],
    "weather_query_coordinates": "anonymized LianderPower coordinates, not original asset coordinates",
    "weather_time_zone": "UTC",
    "weather_sample_interval_s": 3600,
    "weather_model": "best_match",
    "weather_model_selection": (
        "No Open-Meteo models parameter was specified in the query. The release therefore used "
        "Open-Meteo's default Best Match model selection."
    ),
    "weather_processing": (
        "Hourly weather spans are stored at 3600-second resolution in weather.parquet. They are "
        "not expanded to the 5-minute power cadence in the Croissant files; downstream loaders "
        "align or resample weather covariates to measurement timestamps."
    ),
    "weather_modifications": (
        "Queried at anonymized coordinates and stored as LianderPower weather covariates. Spatial "
        "jittering and anonymized-coordinate querying introduce a privacy-preserving mismatch from "
        "original asset coordinates."
    ),
    "open_meteo_endorsement": "Open-Meteo does not endorse LianderPower.",
}


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------


def _load_meta_and_arrays(json_path: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    for key in ("dataset", "spans", "dimension", "sample_interval_minutes", "locations"):
        if key not in meta:
            raise KeyError(f"{json_path} is missing required key: {key}")

    spans = pq.read_table(json_path.parent / meta["spans"]).to_pandas().to_numpy()
    dim = int(meta["dimension"])
    data = np.memmap(json_path.parent / meta["dataset"], dtype="float32", mode="r").reshape((-1, dim))
    return meta, spans, data


def _normalise_sensor_unit(value: Any) -> str:
    unit = str(value).strip().upper()
    return unit if unit in VALID_SENSOR_UNITS else "X"


def _normalise_measurement_type(info: dict[str, Any]) -> str:
    value = str(info.get("measurement_type", "")).strip().lower().replace("-", "_").replace(" ", "_")
    if value in VALID_MEASUREMENT_TYPES:
        return value

    # Full pre-pruning v6 metadata used measurement_type for P/I/X and stored
    # pseudo/native status in flags. Interpret those flags if present.
    if bool(info.get("is_total")) or bool(info.get("is_already_aggregated")):
        return "native_aggregate"
    return "pseudo_aggregate"


def _safe_location_name(location_id: int, info: dict[str, Any], measurement_type: str) -> str:
    """Return a public-safe location/sensor name.

    Full internal metadata used coordinate-bearing names such as
    grid_52.123456_5.123456_g0. Those are deliberately not propagated.
    """
    name = str(info.get("name", "")).strip()
    if name.startswith("pseudo_aggregate_") or name.startswith("native_aggregate_"):
        return name
    return f"{measurement_type}_{location_id:05d}"


def _resolve_measurement_location_ids(
    meta: dict[str, Any], spans: np.ndarray
) -> tuple[np.ndarray, dict[int, dict[str, Any]]]:
    """Filter spans to IDs present in meta and normalize public location metadata.

    Returns:
        Tuple of (filtered_spans, locations) where locations maps ID to public info.
    """
    raw_locations = {int(i): dict(x) for i, x in meta["locations"].items()}
    locations: dict[int, dict[str, Any]] = {}

    for loc_id, info in raw_locations.items():
        measurement_type = _normalise_measurement_type(info)
        sensor_unit = _normalise_sensor_unit(info.get("sensor_unit", info.get("measurement_type", "X")))
        public_info: dict[str, Any] = {
            "name": _safe_location_name(loc_id, info, measurement_type),
            "sensor_unit": sensor_unit,
            "measurement_type": measurement_type,
        }

        # Public release coordinates. Used for weather linking and optionally
        # written to measurements.parquet.
        if "lat" in info and "lon" in info:
            try:
                public_info["_lat"] = float(info["lat"])
                public_info["_lon"] = float(info["lon"])
            except (TypeError, ValueError):
                pass
        if "weather_location_id" in info:
            with contextlib.suppress(TypeError, ValueError):
                public_info["weather_location_id"] = int(info["weather_location_id"])
        locations[loc_id] = public_info

    keep = np.fromiter((int(x) in locations for x in spans[:, 1]), dtype=bool, count=len(spans))
    return spans[keep], locations


def _filter_weather_spans(meta: dict[str, Any], spans: np.ndarray) -> tuple[np.ndarray, dict[int, dict[str, Any]]]:
    """Drop weather spans whose location is missing from the JSON locations dict.

    Returns:
        Tuple of (filtered_spans, locations) mapping weather location IDs to info dicts.
    """
    locations = {int(i): dict(x) for i, x in meta["locations"].items()}
    keep = np.fromiter((int(x) in locations for x in spans[:, 1]), dtype=bool, count=len(spans))
    return spans[keep], locations


def _assert_no_overlap(spans: np.ndarray, sample_interval_s: int) -> None:
    if len(spans) == 0:
        return
    order = np.lexsort((spans[:, 2], spans[:, 1]))
    s = spans[order]
    same_loc = s[1:, 1] == s[:-1, 1]
    prev_end = s[:-1, 2] + s[:-1, 3] * sample_interval_s
    overlap = same_loc & (s[1:, 2] < prev_end)
    if overlap.any():
        i = int(np.where(overlap)[0][0])
        raise ValueError(
            f"Overlapping spans for location {int(s[i, 1])}: "
            f"prev ends at {int(prev_end[i])}, next starts at {int(s[i + 1, 2])}"
        )


# ---------------------------------------------------------------------------
# Arrays and weather-link helpers
# ---------------------------------------------------------------------------


def _build_list_array(chunks: list[np.ndarray]) -> pa.Array:
    """Build a pa.LargeListArray<float32> from chunks safely.

    Returns:
        A PyArrow LargeListArray of float32 values.
    """
    lengths = np.fromiter((len(c) for c in chunks), dtype=np.int64, count=len(chunks))
    offsets = np.empty(len(chunks) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:], dtype=np.int64)
    flat = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
    return pa.LargeListArray.from_arrays(
        pa.array(offsets, type=pa.int64()),
        pa.array(flat, type=pa.float32()),
    )


def _haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    phi1 = math.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + math.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _weather_location_arrays(weather_locations: dict[int, dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids: list[int] = []
    lats: list[float] = []
    lons: list[float] = []
    for wid, info in sorted(weather_locations.items()):
        if "lat" not in info or "lon" not in info:
            continue
        ids.append(int(wid))
        lats.append(float(info["lat"]))
        lons.append(float(info["lon"]))
    return np.array(ids, dtype=np.int64), np.array(lats, dtype=np.float64), np.array(lons, dtype=np.float64)


def _weather_location_records(weather_locations: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """Embedded unique weather-location records for Croissant joins.

    Returns:
        List of records with weather_location_id, lat, lon keys.
    """
    records: list[dict[str, Any]] = []
    for wid, info in sorted(weather_locations.items()):
        if "lat" not in info or "lon" not in info:
            continue
        records.append({
            "weather_locations/weather_location_id": int(wid),
            "weather_locations/lat": float(info["lat"]),
            "weather_locations/lon": float(info["lon"]),
        })
    return records


def _coord_key(lat: float, lon: float, ndigits: int = 6) -> tuple[float, float]:
    """Rounded coordinate key for matching anonymized measurement and weather locations.

    The anonymized coordinates in the measurement and weather JSONs are typically
    the same values rounded to six decimals. Matching exactly first avoids
    arbitrary nearest-neighbor ties when several measurement series share one
    anonymized location.

    Returns:
        Tuple of (rounded_lat, rounded_lon).
    """
    return (round(float(lat), ndigits), round(float(lon), ndigits))


def _weather_coord_lookup(weather_locations_by_id: dict[int, dict[str, Any]]) -> dict[tuple[float, float], int]:
    """Build a rounded-coordinate lookup for weather locations.

    Ambiguous rounded coordinates are skipped so nearest-neighbor matching can
    choose a unique weather location.

    Returns:
        Dict mapping (lat, lon) coordinate tuples to weather_location_id.
    """
    first_seen: dict[tuple[float, float], int] = {}
    collisions: set[tuple[float, float]] = set()
    for wid, info in weather_locations_by_id.items():
        if "lat" not in info or "lon" not in info:
            continue
        key = _coord_key(float(info["lat"]), float(info["lon"]))
        if key in first_seen and first_seen[key] != int(wid):
            collisions.add(key)
        else:
            first_seen[key] = int(wid)
    for key in collisions:
        first_seen.pop(key, None)
    return first_seen


def _assign_weather_ids(
    location_ids: np.ndarray,
    locations_by_id: dict[int, dict[str, Any]],
    weather_locations_by_id: dict[int, dict[str, Any]],
) -> tuple[np.ndarray, str]:
    """Assign one weather_location_id to each measurement span.

    Returns:
        Tuple of (weather_location_ids_array, method_description_string).
    """
    unique_location_ids = sorted({int(x) for x in location_ids.tolist()})
    if not unique_location_ids:
        return np.empty(0, dtype=np.int64), "no measurement spans"

    weather_ids, weather_lats, weather_lons = _weather_location_arrays(weather_locations_by_id)
    weather_id_set = {int(x) for x in weather_locations_by_id}
    weather_by_coord = _weather_coord_lookup(weather_locations_by_id)
    selected_by_location: dict[int, int] = {}
    method_counts: Counter[str] = Counter()
    missing_locations: list[int] = []
    invalid_weather_ids: list[tuple[int, int]] = []

    for loc_id in unique_location_ids:
        info = locations_by_id[loc_id]
        weather_id: int | None = None

        if "weather_location_id" in info:
            weather_id = int(info["weather_location_id"])
            method_counts["metadata"] += 1
        elif "_lat" in info and "_lon" in info:
            lat = float(info["_lat"])
            lon = float(info["_lon"])
            coord_key = _coord_key(lat, lon)

            if coord_key in weather_by_coord:
                weather_id = int(weather_by_coord[coord_key])
                method_counts["exact_coordinate"] += 1
            elif len(weather_ids):
                nearest = int(np.argmin(_haversine_km(lat, lon, weather_lats, weather_lons)))
                weather_id = int(weather_ids[nearest])
                method_counts["nearest_coordinate"] += 1

        if weather_id is None:
            missing_locations.append(loc_id)
            continue

        if weather_id not in weather_id_set:
            invalid_weather_ids.append((loc_id, weather_id))
            continue

        selected_by_location[loc_id] = weather_id

    if missing_locations:
        preview = missing_locations[:10]
        suffix = "..." if len(missing_locations) > len(preview) else ""
        raise RuntimeError(
            "Could not assign weather_location_id for measurement location_id(s): "
            f"{preview}{suffix}. Add weather_location_id or anonymized lat/lon to the measurement JSON."
        )

    if invalid_weather_ids:
        preview = invalid_weather_ids[:10]
        suffix = "..." if len(invalid_weather_ids) > len(preview) else ""
        raise RuntimeError(
            "Measurement metadata references weather_location_id values absent from the weather JSON: "
            f"{preview}{suffix}."
        )

    ids = np.fromiter(
        (selected_by_location[int(loc_id)] for loc_id in location_ids),
        dtype=np.int64,
        count=len(location_ids),
    )

    labels = {
        "metadata": "metadata weather_location_id",
        "exact_coordinate": "exact anonymized-coordinate match",
        "nearest_coordinate": "nearest anonymized-coordinate match",
    }
    method = "; ".join(labels[key] for key in labels if method_counts[key])
    return ids, method


def _extract_weather_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Return weather metadata, preferring values already present in weather JSON."""
    out = dict(DEFAULT_WEATHER_METADATA)
    for key in DEFAULT_WEATHER_METADATA:
        if key in meta:
            out[key] = meta[key]
    return out


def _weather_variable_lookup(weather_metadata: dict[str, Any]) -> dict[str, dict[str, str]]:
    variables = weather_metadata.get("weather_variables", [])
    lookup: dict[str, dict[str, str]] = {}
    for record in variables:
        if not isinstance(record, dict) or "name" not in record:
            continue
        lookup[str(record["name"])] = {str(k): str(v) for k, v in record.items()}
    return lookup


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def convert_measurements(
    json_path: Path,
    out_path: Path,
    weather_locations_by_id: dict[int, dict[str, Any]],
    *,
    include_measurement_coordinates: bool = True,
) -> dict[str, Any]:
    """Convert a measurements JSON+memmap dataset to measurements.parquet.

    Returns:
        Dict of summary statistics and metadata for the converted dataset.
    """
    meta, spans, data = _load_meta_and_arrays(json_path)
    sample_interval_s = int(meta["sample_interval_minutes"]) * 60
    dim = int(meta["dimension"])
    if dim != 1:
        raise ValueError(f"Expected measurement dimension=1, got dimension={dim}")

    spans, locations_by_id = _resolve_measurement_location_ids(meta, spans)
    _assert_no_overlap(spans, sample_interval_s)

    n = len(spans)
    loc_ids = spans[:, 1].astype(np.int64)
    start_times = spans[:, 2].astype(np.int64)
    nums = spans[:, 3].astype(np.int32)
    sample_intervals = np.full(n, sample_interval_s, dtype=np.int32)

    names = [locations_by_id[int(l)]["name"] for l in loc_ids]
    sensor_units = [locations_by_id[int(l)]["sensor_unit"] for l in loc_ids]
    measurement_types = [locations_by_id[int(l)]["measurement_type"] for l in loc_ids]
    weather_location_ids, weather_link_method = _assign_weather_ids(loc_ids, locations_by_id, weather_locations_by_id)

    measurement_lats: np.ndarray | None = None
    measurement_lons: np.ndarray | None = None
    if include_measurement_coordinates:
        missing_coords = sorted({
            int(l)
            for l in loc_ids.tolist()
            if "_lat" not in locations_by_id[int(l)] or "_lon" not in locations_by_id[int(l)]
        })
        if missing_coords:
            raise RuntimeError(
                "Measurement coordinates are requested for measurements.parquet, but coordinates are missing for "
                f"{len(missing_coords)} measurement location(s), including {missing_coords[:10]}."
            )
        measurement_lats = np.fromiter(
            (float(locations_by_id[int(l)]["_lat"]) for l in loc_ids), dtype=np.float64, count=n
        )
        measurement_lons = np.fromiter(
            (float(locations_by_id[int(l)]["_lon"]) for l in loc_ids), dtype=np.float64, count=n
        )

    chunks = [
        np.ascontiguousarray(data[int(sid) : int(sid) + int(num), 0], dtype=np.float32)
        for sid, num in zip(spans[:, 0], spans[:, 3])
    ]

    columns: dict[str, pa.Array] = {
        "location_id": pa.array(loc_ids),
        "location_name": pa.array(names, type=pa.string()),
        "sensor_unit": pa.array(sensor_units, type=pa.string()),
        "measurement_type": pa.array(measurement_types, type=pa.string()),
        "weather_location_id": pa.array(weather_location_ids),
        "start_time": pa.array(start_times),
        "sample_interval_s": pa.array(sample_intervals),
        "num_values": pa.array(nums),
        "values": _build_list_array(chunks),
    }
    if include_measurement_coordinates:
        columns = {
            **{
                "location_id": columns["location_id"],
                "location_name": columns["location_name"],
                "lat": pa.array(measurement_lats),
                "lon": pa.array(measurement_lons),
            },
            **{k: v for k, v in columns.items() if k not in {"location_id", "location_name"}},
        }
    table = pa.table(columns)
    pq.write_table(table, out_path, compression="zstd")

    unique_location_ids = sorted({int(x) for x in loc_ids.tolist()})
    locations_used = [locations_by_id[i] for i in unique_location_ids]
    sensor_unit_counts = dict.fromkeys(("P", "I", "X"), 0)
    sensor_unit_counts.update(Counter(loc["sensor_unit"] for loc in locations_used))
    sensor_unit_counts = {k: int(v) for k, v in sorted(sensor_unit_counts.items())}
    measurement_type_counts = {
        k: int(v) for k, v in sorted(Counter(loc["measurement_type"] for loc in locations_used).items())
    }

    meta_sensor_unit_counts = meta.get("sensor_unit_counts", sensor_unit_counts)
    if isinstance(meta_sensor_unit_counts, dict) and "all_locations" in meta_sensor_unit_counts:
        meta_sensor_unit_counts = meta_sensor_unit_counts["all_locations"]

    weather_id_by_location_id: dict[int, int] = {}
    for loc_id, weather_id in zip(loc_ids, weather_location_ids):
        weather_id_by_location_id.setdefault(int(loc_id), int(weather_id))
    weather_link_available = bool(weather_id_by_location_id)
    total_values = int(nums.astype(np.int64).sum())
    n_nan_values = int(sum(np.isnan(chunk).sum() for chunk in chunks))

    return {
        "sample_interval_s": sample_interval_s,
        "sample_interval_minutes": int(meta["sample_interval_minutes"]),
        "dimension": dim,
        "n_rows": n,
        "n_locations": len(unique_location_ids),
        "source_dataset": str(meta["dataset"]),
        "source_spans": str(meta["spans"]),
        "sensor_unit_counts": {str(k): int(v) for k, v in dict(meta_sensor_unit_counts).items()},
        "computed_sensor_unit_counts": sensor_unit_counts,
        "measurement_type_counts": measurement_type_counts,
        "min_active_count": meta.get("min_active_count"),
        "weather_link_method": weather_link_method,
        "weather_link_available": weather_link_available,
        "measurement_coordinates_included": bool(include_measurement_coordinates),
        "n_values": total_values,
        "n_nan_values": n_nan_values,
        "n_valid_values": total_values - n_nan_values,
        "location_records": _location_records(
            {loc_id: locations_by_id[loc_id] for loc_id in unique_location_ids},
            weather_id_by_location_id,
        ),
    }


def convert_weather(json_path: Path, out_path: Path) -> dict[str, Any]:
    """Convert a weather JSON+memmap dataset to weather.parquet.

    Returns:
        Dict of summary statistics, feature names, and metadata for the weather data.
    """
    meta, spans, data = _load_meta_and_arrays(json_path)
    sample_interval_s = int(meta["sample_interval_minutes"]) * 60
    if "feature_names" in meta:
        feature_names = list(meta["feature_names"])
    elif "weather_variables" in meta:
        feature_names = [str(x["name"]) for x in meta["weather_variables"] if isinstance(x, dict) and "name" in x]
    else:
        feature_names = [str(x["name"]) for x in DEFAULT_WEATHER_METADATA["weather_variables"]]
    dim = int(meta["dimension"])
    if dim != len(feature_names):
        raise ValueError(f"dimension={dim} but feature_names has {len(feature_names)} entries")

    spans, locations = _filter_weather_spans(meta, spans)
    _assert_no_overlap(spans, sample_interval_s)

    weather_metadata = _extract_weather_metadata(meta)
    weather_metadata["weather_sample_interval_s"] = int(sample_interval_s)
    # Ensure the manifest contains one variable description for every stored feature,
    # even if a future weather JSON adds a feature that is not in the defaults.
    variables = list(weather_metadata.get("weather_variables", []))
    seen_variables = {str(v.get("name")) for v in variables if isinstance(v, dict) and "name" in v}
    for feature_name in feature_names:
        if str(feature_name) not in seen_variables:
            variables.append({
                "name": str(feature_name),
                "unit": "unknown",
                "height": "",
                "temporal_semantics": "hourly weather variable stored in weather.parquet",
            })
    weather_metadata["weather_variables"] = variables
    variable_lookup = _weather_variable_lookup(weather_metadata)

    n_spans = len(spans)
    n_rows = n_spans * dim

    loc_ids = np.repeat(spans[:, 1].astype(np.int64), dim)
    start_times = np.repeat(spans[:, 2].astype(np.int64), dim)
    nums = np.repeat(spans[:, 3].astype(np.int32), dim)
    sample_intervals = np.full(n_rows, sample_interval_s, dtype=np.int32)
    feature_col = np.tile(np.array(feature_names, dtype=object), n_spans)

    lats = np.fromiter((float(locations[int(l)]["lat"]) for l in loc_ids), dtype=np.float64, count=n_rows)
    lons = np.fromiter((float(locations[int(l)]["lon"]) for l in loc_ids), dtype=np.float64, count=n_rows)
    feature_units = [variable_lookup.get(str(f), {}).get("unit", "") for f in feature_col]

    chunks: list[np.ndarray] = []
    n_nan_values = 0
    for sid, num in zip(spans[:, 0], spans[:, 3]):
        block = np.ascontiguousarray(data[int(sid) : int(sid) + int(num), :], dtype=np.float32)
        for f_idx in range(dim):
            chunk = np.ascontiguousarray(block[:, f_idx])
            n_nan_values += int(np.isnan(chunk).sum())
            chunks.append(chunk)

    table = pa.table({
        "weather_location_id": pa.array(loc_ids),
        "lat": pa.array(lats),
        "lon": pa.array(lons),
        "feature_name": pa.array(feature_col, type=pa.string()),
        "feature_unit": pa.array(feature_units, type=pa.string()),
        "start_time": pa.array(start_times),
        "sample_interval_s": pa.array(sample_intervals),
        "num_values": pa.array(nums),
        "values": _build_list_array(chunks),
    })
    pq.write_table(table, out_path, compression="zstd")

    return {
        "sample_interval_s": sample_interval_s,
        "sample_interval_minutes": int(meta["sample_interval_minutes"]),
        "dimension": dim,
        "feature_names": feature_names,
        "n_rows": n_rows,
        "n_locations": len(set(spans[:, 1].tolist())),
        "source_dataset": str(meta["dataset"]),
        "source_spans": str(meta["spans"]),
        "locations_by_id": locations,
        "weather_location_records": _weather_location_records(locations),
        "weather_metadata": weather_metadata,
        "n_values": int(nums.astype(np.int64).sum()),
        "n_nan_values": n_nan_values,
        "n_valid_values": int(nums.astype(np.int64).sum()) - n_nan_values,
    }


# ---------------------------------------------------------------------------
# Croissant manifest
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _field(
    rs: str,
    name: str,
    dtype: str,
    file_id: str,
    *,
    repeated: bool = False,
    description: str | None = None,
    references: str | None = None,
) -> dict[str, Any]:
    f: dict[str, Any] = {
        "@type": "cr:Field",
        "@id": f"{rs}/{name}",
        "name": name,
        "dataType": dtype,
        "source": {"fileObject": {"@id": file_id}, "extract": {"column": name}},
    }
    if repeated:
        # Croissant 1.1 represents vector/list-valued columns using isArray.
        f["isArray"] = True
        f["arrayShape"] = "-1"
    if description:
        f["description"] = description
    if references:
        f["references"] = {"@id": references}
    return f


def _embedded_field(rs: str, name: str, dtype: str, *, description: str | None = None) -> dict[str, Any]:
    f: dict[str, Any] = {
        "@type": "cr:Field",
        "@id": f"{rs}/{name}",
        "name": name,
        "dataType": dtype,
    }
    if description:
        f["description"] = description
    return f


def _property_value(name: str, value: Any, description: str | None = None) -> dict[str, Any]:
    if isinstance(value, (dict, list)):
        encoded_value: Any = json.dumps(value, sort_keys=True, ensure_ascii=False)
    elif value is None:
        encoded_value = ""
    else:
        encoded_value = value
    out: dict[str, Any] = {"@type": "sc:PropertyValue", "name": name, "value": encoded_value}
    if description:
        out["description"] = description
    return out


def _sensor_unit_count_records(sensor_unit_counts: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for unit in ("P", "I", "X"):
        records.append({
            "sensor_unit_counts/sensor_unit": unit,
            "sensor_unit_counts/count": int(sensor_unit_counts.get(unit, 0)),
        })
    for unit, count in sorted(sensor_unit_counts.items()):
        if unit not in {"P", "I", "X"}:
            records.append({"sensor_unit_counts/sensor_unit": str(unit), "sensor_unit_counts/count": int(count)})
    return records


def _weather_variable_records(weather_metadata: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for variable in weather_metadata.get("weather_variables", []):
        if not isinstance(variable, dict):
            continue
        records.append({
            "weather_variables/name": str(variable.get("name", "")),
            "weather_variables/unit": str(variable.get("unit", "")),
        })
    return records


def _location_records(
    locations_by_id: dict[int, dict[str, Any]],
    weather_id_by_location_id: dict[int, int],
) -> list[dict[str, Any]]:
    """Embedded public metadata for released measurement series.

    Returns:
        List of records with location_id, name, sensor_unit, measurement_type, weather_location_id.
    """
    records: list[dict[str, Any]] = []
    for loc_id in sorted(locations_by_id):
        info = locations_by_id[loc_id]
        records.append({
            "locations/location_id": int(loc_id),
            "locations/name": str(info.get("name", loc_id)),
            "locations/sensor_unit": str(info.get("sensor_unit", "X")),
            "locations/measurement_type": str(info.get("measurement_type", "unknown_aggregate")),
            "locations/weather_location_id": int(weather_id_by_location_id[int(loc_id)]),
        })
    return records


def _metadata_records(rs: str, items: dict[str, Any]) -> list[dict[str, Any]]:
    """Embedded key-value metadata rows for compact release-level metadata.

    Returns:
        List of dicts with {rs}/key and {rs}/value string fields.
    """
    rows: list[dict[str, Any]] = []
    for key, raw in items.items():
        if isinstance(raw, (dict, list)):
            encoded: Any = json.dumps(raw, sort_keys=True, ensure_ascii=False)
        elif raw is None:
            encoded = ""
        else:
            encoded = raw
        rows.append({f"{rs}/key": str(key), f"{rs}/value": str(encoded)})
    return rows


def _additional_properties(m_info: dict[str, Any], w_info: dict[str, Any]) -> list[dict[str, Any]]:
    weather_metadata = w_info["weather_metadata"]
    props = [
        _property_value(
            "measurement_source_dataset",
            m_info["source_dataset"],
            "Source memmap filename named in the measurement JSON.",
        ),
        _property_value(
            "measurement_source_spans", m_info["source_spans"], "Source spans filename named in the measurement JSON."
        ),
        _property_value("measurement_dimension", m_info["dimension"]),
        _property_value("measurement_sample_interval_minutes", m_info["sample_interval_minutes"]),
        _property_value(
            "measurement_min_active_count",
            m_info.get("min_active_count"),
            "Minimum active contributors required for a non-NaN pseudo-aggregate value.",
        ),
        _property_value("sensor_unit_counts", m_info.get("sensor_unit_counts", {})),
        _property_value("measurement_type_counts", m_info.get("measurement_type_counts", {})),
        _property_value("weather_link_available", m_info.get("weather_link_available", False)),
        _property_value("weather_link_method", m_info.get("weather_link_method", "")),
        _property_value(
            "weather_source_dataset",
            w_info["source_dataset"],
            "Source weather memmap filename named in the weather JSON.",
        ),
        _property_value(
            "weather_source_spans", w_info["source_spans"], "Source weather spans filename named in the weather JSON."
        ),
    ]
    for key in DEFAULT_WEATHER_METADATA:
        props.append(_property_value(key, weather_metadata.get(key, "")))
    return props


def write_croissant_manifest(
    out_dir: Path,
    m_info: dict[str, Any],
    w_info: dict[str, Any],
    *,
    dataset_url: str,
    citation: str,
    version: str,
    license_url: str,
    date_published: str,
) -> None:
    """Write a Croissant 1.1 JSON-LD manifest describing the Parquet files."""
    m_pq = out_dir / "measurements.parquet"
    w_pq = out_dir / "weather.parquet"
    weather_metadata = w_info["weather_metadata"]
    weather_variables_text = ", ".join(w_info["feature_names"])
    min_active = m_info.get("min_active_count")
    measurement_release_metadata = {
        "source_dataset_memmap": m_info.get("source_dataset", ""),
        "source_spans": m_info.get("source_spans", ""),
        "dimension": m_info.get("dimension", ""),
        "sample_interval_minutes": m_info.get("sample_interval_minutes", ""),
        "sample_interval_s": m_info.get("sample_interval_s", ""),
        "min_active_count": m_info.get("min_active_count", ""),
        "sensor_unit_counts": m_info.get("sensor_unit_counts", {}),
        "computed_sensor_unit_counts": m_info.get("computed_sensor_unit_counts", {}),
        "measurement_type_counts": m_info.get("measurement_type_counts", {}),
        "n_measurement_spans": m_info.get("n_rows", ""),
        "n_released_measurement_series": m_info.get("n_locations", ""),
        "n_measurement_values": m_info.get("n_values", ""),
        "n_measurement_nan_values": m_info.get("n_nan_values", ""),
        "n_measurement_valid_values": m_info.get("n_valid_values", ""),
        "weather_link_available": m_info.get("weather_link_available", False),
        "weather_link_method": m_info.get("weather_link_method", ""),
        "measurement_coordinates_included": m_info.get("measurement_coordinates_included", False),
        "measurement_coordinates": (
            "included as anonymized/jittered lat/lon in measurements.parquet"
            if m_info.get("measurement_coordinates_included")
            else "not included in measurements.parquet or public measurement JSON"
        ),
    }
    measurement_fields = [
        _field(
            "measurements",
            "location_id",
            "sc:Integer",
            "measurements-pq",
            description=(
                "Public measurement time-series identifier; corresponds to the integer key "
                "in the public measurement JSON locations dictionary."
            ),
        ),
        _field(
            "measurements",
            "location_name",
            "sc:Text",
            "measurements-pq",
            description=(
                "Public-safe anonymized name, e.g. pseudo_aggregate_00012 or native_aggregate_00354. "
                "Original sensor names are not released."
            ),
        ),
    ]
    if m_info.get("measurement_coordinates_included"):
        measurement_fields.extend([
            _field(
                "measurements",
                "lat",
                "sc:Float",
                "measurements-pq",
                description="Anonymized/jittered measurement latitude; not an original asset coordinate.",
            ),
            _field(
                "measurements",
                "lon",
                "sc:Float",
                "measurements-pq",
                description="Anonymized/jittered measurement longitude; not an original asset coordinate.",
            ),
        ])
    measurement_fields.extend([
        _field(
            "measurements",
            "sensor_unit",
            "sc:Text",
            "measurements-pq",
            description=(
                "Public physical quantity code: P=active power, I=current, X=other or ambiguous native aggregate."
            ),
        ),
        _field(
            "measurements",
            "measurement_type",
            "sc:Text",
            "measurements-pq",
            description=(
                "Aggregation provenance: pseudo_aggregate for anonymized active-power pseudo-sensors; "
                "native_aggregate for pre-existing SCADA summation sensors."
            ),
        ),
        _field(
            "measurements",
            "weather_location_id",
            "sc:Integer",
            "measurements-pq",
            description=(
                f"Join key to weather/weather_location_id. Link method: {m_info.get('weather_link_method', '')}."
            ),
        ),
        _field(
            "measurements",
            "start_time",
            "sc:Integer",
            "measurements-pq",
            description="Unix epoch seconds for the first sample in this span.",
        ),
        _field("measurements", "sample_interval_s", "sc:Integer", "measurements-pq"),
        _field("measurements", "num_values", "sc:Integer", "measurements-pq"),
        _field("measurements", "values", "sc:Float", "measurements-pq", repeated=True),
    ])

    weather_release_metadata = {
        "source_dataset_memmap": w_info.get("source_dataset", ""),
        "source_spans": w_info.get("source_spans", ""),
        "dimension": w_info.get("dimension", ""),
        "sample_interval_minutes": w_info.get("sample_interval_minutes", ""),
        "sample_interval_s": w_info.get("sample_interval_s", ""),
        "feature_names": w_info.get("feature_names", []),
        "n_weather_rows": w_info.get("n_rows", ""),
        "n_weather_locations": w_info.get("n_locations", ""),
        "n_weather_values": w_info.get("n_values", ""),
        "n_weather_nan_values": w_info.get("n_nan_values", ""),
        "n_weather_valid_values": w_info.get("n_valid_values", ""),
        **{k: v for k, v in weather_metadata.items() if k != "weather_variables"},
    }

    manifest: dict[str, Any] = {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "citeAs": "cr:citeAs",
            "column": "cr:column",
            "conformsTo": "dct:conformsTo",
            "cr": "http://mlcommons.org/croissant/",
            "rai": "http://mlcommons.org/croissant/RAI/",
            "data": {"@id": "cr:data", "@type": "@json"},
            "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
            "dct": "http://purl.org/dc/terms/",
            "examples": {"@id": "cr:examples", "@type": "@json"},
            "extract": "cr:extract",
            "field": "cr:field",
            "fileProperty": "cr:fileProperty",
            "fileObject": "cr:fileObject",
            "fileSet": "cr:fileSet",
            "format": "cr:format",
            "includes": "cr:includes",
            "isArray": "cr:isArray",
            "isLiveDataset": "cr:isLiveDataset",
            "jsonPath": "cr:jsonPath",
            "key": "cr:key",
            "md5": "cr:md5",
            "parentField": "cr:parentField",
            "path": "cr:path",
            "recordSet": "cr:recordSet",
            "references": "cr:references",
            "regex": "cr:regex",
            "repeated": "cr:repeated",
            "replace": "cr:replace",
            "sc": "https://schema.org/",
            "separator": "cr:separator",
            "source": "cr:source",
            "subField": "cr:subField",
            "transform": "cr:transform",
            "arrayShape": "cr:arrayShape",
            "equivalentProperty": "cr:equivalentProperty",
            "prov": "http://www.w3.org/ns/prov#",
            "value": "cr:value",
        },
        "@type": "sc:Dataset",
        "name": "LianderPower",
        "description": (
            "LianderPower is an anonymized power-grid time-series dataset derived from operational "
            "SCADA telemetry of Liander, a Dutch distribution system operator. The public release "
            "contains aggregate measurement time series and Open-Meteo weather covariates. It does "
            "not contain exact asset coordinates, topology, switching states, feeder identifiers, "
            "customer information, or raw single-sensor measurements. Measurement rows are stored as "
            "contiguous spans; gaps between spans are implicit."
        ),
        "version": version,
        "license": license_url,
        "citeAs": citation,
        "citation": citation,
        "url": dataset_url,
        "datePublished": date_published,
        "disclaimer": "https://www.liander.nl/over-ons/open-data/disclaimer",
        "conformsTo": "http://mlcommons.org/croissant/1.1",
        "isBasedOn": [
            {
                "@type": "sc:Dataset",
                "name": "Internal Liander operational SCADA telemetry",
                "description": (
                    "Internal operational source data used to derive the anonymized public aggregate release."
                ),
            },
            {
                "@type": "sc:SoftwareApplication",
                "name": weather_metadata.get("weather_provider", "Open-Meteo Historical Weather API"),
                "url": weather_metadata.get("weather_api_endpoint", "https://archive-api.open-meteo.com/v1/archive"),
                "license": weather_metadata.get("weather_license_url", "https://creativecommons.org/licenses/by/4.0/"),
                "citation": weather_metadata.get("weather_citation", ""),
            },
        ],
        "prov:wasDerivedFrom": [
            {
                "@type": "sc:Dataset",
                "name": "Internal Liander operational SCADA telemetry",
                "description": "Internal source data; not publicly released.",
            },
            {
                "@type": "sc:SoftwareApplication",
                "name": weather_metadata.get("weather_provider", "Open-Meteo Historical Weather API"),
                "url": weather_metadata.get("weather_api_endpoint", "https://archive-api.open-meteo.com/v1/archive"),
            },
        ],
        "prov:wasGeneratedBy": {
            "@type": "prov:Activity",
            "name": "LianderPower anonymization and Croissant conversion pipeline",
            "description": (
                "Filtering, anonymization, aggregation, temporal masking, "
                "weather covariate retrieval, and Parquet/Croissant conversion."
            ),
        },
        "creator": {"@type": "sc:Organization", "name": "Alliander / Liander"},
        "publisher": {"@type": "sc:Organization", "name": "Alliander / Liander"},
        "maintainer": {"@type": "sc:Organization", "name": "System Operations AI R&D team, Alliander"},
        "additionalProperty": _additional_properties(m_info, w_info),
        "rai:dataCollection": (
            "Operational SCADA telemetry was collected from Liander distribution-grid assets between "
            "2013 and 2024 and converted into anonymized aggregate measurement time series. Weather "
            "covariates were queried from Open-Meteo at anonymized LianderPower coordinates."
        ),
        "rai:dataCollectionType": ["Direct measurement", "Web API", "Secondary Data analysis"],
        "rai:dataCollectionRawData": (
            "Raw internal data consist of SCADA measurements from distribution-grid sensors. The raw "
            "data, original sensor names, and original asset coordinates are not released."
        ),
        "rai:dataPreprocessingProtocol": [
            "Raw SCADA spans are quality-filtered, short spans are removed, "
            "and near-duplicate sensors are filtered before anonymization.",
            "Non-aggregate active-power sensors are grouped into pseudo-aggregates. "
            "Non-NaN pseudo-aggregate values require at least min_active_count active contributors.",
            "Native summation sensors are copied only as pre-existing aggregate measurements. "
            "Original asset coordinates and original sensor names are not released; "
            "any coordinates in the public Parquet files are anonymized/jittered coordinates.",
            "Weather covariates are stored hourly and are not expanded to "
            "the measurement cadence in the Croissant files.",
        ],
        "rai:dataManipulationProtocol": (
            "The anonymization pipeline removes Wadden Island sensors, spatially jitters "
            "and merges low-density locations, aggregates active-power sensors, masks "
            "pseudo-aggregate values below the contributor-count threshold, and strips "
            "exact coordinates, topology, feeder identifiers, switching states, customer "
            "information, and raw single-sensor measurements from the public release."
        ),
        "rai:dataLimitations": [
            "The dataset is intended for aggregate load/power-grid time-series modeling, "
            "not for reconstructing exact grid topology, asset locations, switching states, "
            "or customer-level behavior.",
            "Anonymization changes spatial semantics: weather is queried at anonymized "
            "coordinates and may not match exact asset microclimates.",
            "The data are from Liander's service area in the Netherlands and may not "
            "represent transmission grids, other countries, or distribution systems "
            "with different network design and operating practices.",
            "Operational artifacts such as constant segments, abrupt regime changes, "
            "polarity changes, and missingness should be treated as part of the data "
            "distribution rather than as annotation errors.",
        ],
        "rai:dataBiases": [
            "Coverage reflects Liander's asset base, sensor deployment history, and operational "
            "data availability; it is not a statistically uniform sample of all Dutch or "
            "European distribution-grid assets.",
            "Aggregation and temporal masking smooth some localized high-frequency dynamics, "
            "which may disproportionately affect tasks involving feeder-level volatility "
            "or intermittent generation.",
            "Weather covariates are gridded model/reanalysis estimates queried at anonymized "
            "coordinates, which can introduce location-dependent mismatch.",
        ],
        "rai:personalSensitiveInformation": (
            "The release does not include customer identifiers, customer-level measurements, "
            "addresses, exact coordinates, feeder identifiers, topology, switching states, "
            "or raw single-sensor measurements. It does contain anonymized regional geography "
            "through weather coordinates and aggregate grid time series, which remain "
            "security-sensitive infrastructure context."
        ),
        "rai:dataUseCases": [
            "Training, validation, and evaluation of time-series forecasting models "
            "for aggregate distribution-grid measurements.",
            "Studying robustness of forecasters under operational artifacts, long time spans, "
            "missingness, and weather covariates.",
            "Not recommended for customer-level inference, grid-topology reconstruction, "
            "asset localization, market gaming, or operational control without additional "
            "protected operational data and governance review.",
        ],
        "rai:dataSocialImpact": (
            "The intended positive impact is to support reproducible research on power-grid "
            "forecasting and congestion-aware planning under privacy and infrastructure-security "
            "constraints. Risks include attempted inference about sensitive grid infrastructure "
            "or inappropriate transfer of conclusions outside Liander's operational context; "
            "mitigations include aggregation, temporal masking, coordinate anonymization, "
            "Wadden exclusion, and governance review."
        ),
        "rai:hasSyntheticData": False,
        "rai:dataReleaseMaintenancePlan": (
            "The dataset is intended to be maintained by the System Operations AI R&D team at "
            "Alliander and updated annually with newly releasable measurement data, subject to "
            "data-governance review."
        ),
        "distribution": [
            {
                "@type": "cr:FileObject",
                "@id": "measurements-pq",
                "name": "measurements.parquet",
                "encodingFormat": "application/vnd.apache.parquet",
                "contentUrl": "measurements.parquet",
                "sha256": _sha256(m_pq),
                "contentSize": f"{m_pq.stat().st_size} B",
            },
            {
                "@type": "cr:FileObject",
                "@id": "weather-pq",
                "name": "weather.parquet",
                "encodingFormat": "application/vnd.apache.parquet",
                "contentUrl": "weather.parquet",
                "sha256": _sha256(w_pq),
                "contentSize": f"{w_pq.stat().st_size} B",
            },
        ],
        "recordSet": [
            {
                "@type": "cr:RecordSet",
                "@id": "measurements",
                "name": "measurements",
                "description": (
                    f"One row per measurement time series and contiguous span. Sample interval: "
                    f"{m_info['sample_interval_s']} seconds. {m_info['n_rows']} spans across "
                    f"{m_info['n_locations']} aggregate measurement time series. `values` is a "
                    f"list<float32> of length `num_values`. Pseudo-aggregate values with fewer "
                    f"than {min_active} active contributors are NaN. Gaps between spans for the "
                    f"same location_id are implicit. Measurement coordinates, when present, are "
                    f"anonymized/jittered coordinates rather than original asset positions."
                ),
                "field": measurement_fields,
            },
            {
                "@type": "cr:RecordSet",
                "@id": "locations",
                "name": "locations",
                "description": (
                    "Embedded public metadata for each released aggregate measurement time series. "
                    "weather_location_id is the join key to weather.parquet. Coordinates, when "
                    "needed, are stored in measurements.parquet; original asset coordinates and "
                    "original sensor names are not included."
                ),
                "key": {"@id": "locations/location_id"},
                "field": [
                    _embedded_field("locations", "location_id", "sc:Integer"),
                    _embedded_field("locations", "name", "sc:Text"),
                    _embedded_field(
                        "locations",
                        "sensor_unit",
                        "sc:Text",
                        description="P=active power, I=current, X=other or ambiguous native aggregate.",
                    ),
                    _embedded_field(
                        "locations", "measurement_type", "sc:Text", description="pseudo_aggregate or native_aggregate."
                    ),
                    _embedded_field(
                        "locations",
                        "weather_location_id",
                        "sc:Integer",
                        description="Public join key to weather/weather_location_id.",
                    ),
                ],
                "data": m_info.get("location_records", []),
            },
            {
                "@type": "cr:RecordSet",
                "@id": "weather_locations",
                "name": "weather_locations",
                "description": (
                    "Embedded unique anonymized weather-query locations. Measurements link to "
                    "these locations through weather_location_id."
                ),
                "key": {"@id": "weather_locations/weather_location_id"},
                "field": [
                    _embedded_field("weather_locations", "weather_location_id", "sc:Integer"),
                    _embedded_field(
                        "weather_locations",
                        "lat",
                        "sc:Float",
                        description="Anonymized weather query latitude; not an original asset coordinate.",
                    ),
                    _embedded_field(
                        "weather_locations",
                        "lon",
                        "sc:Float",
                        description="Anonymized weather query longitude; not an original asset coordinate.",
                    ),
                ],
                "data": w_info.get("weather_location_records", []),
            },
            {
                "@type": "cr:RecordSet",
                "@id": "weather",
                "name": "weather",
                "description": (
                    f"One row per anonymized weather location, contiguous span, and weather feature. Sample interval: "
                    f"{w_info['sample_interval_s']} seconds. Features: {weather_variables_text}. "
                    f"{w_info['n_rows']} rows across {w_info['n_locations']} weather locations. "
                    f"Coordinates are anonymized LianderPower weather-query coordinates, "
                    "not original asset coordinates. "
                    f"Provider: {weather_metadata.get('weather_provider')}. "
                    f"Model selection: {weather_metadata.get('weather_model_selection')}"
                ),
                "field": [
                    _field(
                        "weather",
                        "weather_location_id",
                        "sc:Integer",
                        "weather-pq",
                        description=(
                            "Public weather-location identifier. Measurements join through "
                            "measurements/weather_location_id."
                        ),
                    ),
                    _field(
                        "weather",
                        "lat",
                        "sc:Float",
                        "weather-pq",
                        description="Anonymized weather query latitude; not an original asset coordinate.",
                    ),
                    _field(
                        "weather",
                        "lon",
                        "sc:Float",
                        "weather-pq",
                        description="Anonymized weather query longitude; not an original asset coordinate.",
                    ),
                    _field("weather", "feature_name", "sc:Text", "weather-pq", description="Weather variable name."),
                    _field("weather", "feature_unit", "sc:Text", "weather-pq", description="Weather variable unit."),
                    _field(
                        "weather",
                        "start_time",
                        "sc:Integer",
                        "weather-pq",
                        description="Unix epoch seconds for the first weather sample in this span.",
                    ),
                    _field("weather", "sample_interval_s", "sc:Integer", "weather-pq"),
                    _field("weather", "num_values", "sc:Integer", "weather-pq"),
                    _field("weather", "values", "sc:Float", "weather-pq", repeated=True),
                ],
            },
            {
                "@type": "cr:RecordSet",
                "@id": "sensor_unit_counts",
                "name": "sensor_unit_counts",
                "description": "Counts of released aggregate measurement time series by public sensor_unit code.",
                "key": {"@id": "sensor_unit_counts/sensor_unit"},
                "field": [
                    _embedded_field("sensor_unit_counts", "sensor_unit", "sc:Text"),
                    _embedded_field("sensor_unit_counts", "count", "sc:Integer"),
                ],
                "data": _sensor_unit_count_records(m_info.get("sensor_unit_counts", {})),
            },
            {
                "@type": "cr:RecordSet",
                "@id": "measurement_release_metadata",
                "name": "measurement_release_metadata",
                "description": (
                    "Embedded key-value summary copied from or derived from the public measurement JSON metadata."
                ),
                "key": {"@id": "measurement_release_metadata/key"},
                "field": [
                    _embedded_field("measurement_release_metadata", "key", "sc:Text"),
                    _embedded_field("measurement_release_metadata", "value", "sc:Text"),
                ],
                "data": _metadata_records("measurement_release_metadata", measurement_release_metadata),
            },
            {
                "@type": "cr:RecordSet",
                "@id": "weather_release_metadata",
                "name": "weather_release_metadata",
                "description": (
                    "Embedded key-value metadata describing the Open-Meteo source, query, license, and processing."
                ),
                "key": {"@id": "weather_release_metadata/key"},
                "field": [
                    _embedded_field("weather_release_metadata", "key", "sc:Text"),
                    _embedded_field("weather_release_metadata", "value", "sc:Text"),
                ],
                "data": _metadata_records("weather_release_metadata", weather_release_metadata),
            },
            {
                "@type": "cr:RecordSet",
                "@id": "weather_variables",
                "name": "weather_variables",
                "description": "Open-Meteo weather variables included in weather.parquet.",
                "key": {"@id": "weather_variables/name"},
                "field": [
                    _embedded_field("weather_variables", "name", "sc:Text"),
                    _embedded_field("weather_variables", "unit", "sc:Text"),
                ],
                "data": _weather_variable_records(weather_metadata),
            },
        ],
    }

    (out_dir / "croissant.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Convert measurement and weather JSON+memmap files to a Croissant release."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, required=True, help="Path to anon_measurements.json")
    parser.add_argument("--weather", type=Path, required=True, help="Path to weather_openmeteo_anon.json")
    parser.add_argument("--out", type=Path, required=True, help="Output release directory")
    parser.add_argument(
        "--omit-measurement-coordinates",
        dest="include_measurement_coordinates",
        action="store_false",
        help="Leave anonymized measurement lat/lon out of measurements.parquet.",
    )
    parser.set_defaults(include_measurement_coordinates=True)
    parser.add_argument(
        "--dataset-url",
        default="https://www.liander.nl/over-ons/open-data/",
        help="Canonical URL of the dataset landing page. Replace with the final hosting URL before submission.",
    )
    parser.add_argument(
        "--citation",
        default=(
            "LianderPower dataset, version 1.0.0. Cite the accompanying NeurIPS Evaluations "
            "& Datasets paper and dataset DOI once assigned."
        ),
        help="Dataset or paper citation string, preferably BibTeX once available.",
    )
    parser.add_argument("--version", default="1.0.0", help="Dataset release version.")
    parser.add_argument(
        "--license-url",
        default="https://creativecommons.org/licenses/by/4.0/",
        help="Dataset license URL. Default: CC BY 4.0.",
    )
    parser.add_argument(
        "--date-published",
        default=datetime.now(tz=UTC).date().isoformat(),
        help="Publication date in YYYY-MM-DD format.",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Weather is converted first because measurements link to its locations.
    log.info("[1/3] weather: %s", args.weather)
    w_info = convert_weather(args.weather, args.out / "weather.parquet")
    log.info(
        "      -> %s  (%d rows, %d locations)",
        args.out / "weather.parquet",
        w_info["n_rows"],
        w_info["n_locations"],
    )

    log.info("[2/3] measurements: %s", args.measurements)
    m_info = convert_measurements(
        args.measurements,
        args.out / "measurements.parquet",
        w_info["locations_by_id"],
        include_measurement_coordinates=args.include_measurement_coordinates,
    )
    log.info(
        "      -> %s  (%d spans, %d locations)",
        args.out / "measurements.parquet",
        m_info["n_rows"],
        m_info["n_locations"],
    )
    log.info("      weather link: %s", m_info["weather_link_method"])

    log.info("[3/3] croissant.json")
    # Do not expose weather location dictionaries in the manifest.
    w_info_for_manifest = dict(w_info)
    w_info_for_manifest.pop("locations_by_id", None)
    write_croissant_manifest(
        args.out,
        m_info,
        w_info_for_manifest,
        dataset_url=args.dataset_url,
        citation=args.citation,
        version=args.version,
        license_url=args.license_url,
        date_published=args.date_published,
    )
    log.info("Done: %s", args.out)


if __name__ == "__main__":
    main()
