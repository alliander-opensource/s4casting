# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

"""Validate that load_memmap and the croissant loaders return equivalent data.

Compares:
  - load_memmap()                    vs load_measurements_from_croissant()
  - load_memmap() (weather)          vs load_weather_from_croissant()

Locations are matched by (lat, lon) since the two formats use different ID
schemes (legacy uses a sha256-derived hash; croissant uses the raw integer key).

Usage:
    uv run python scripts/validate_croissant.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from s4casting.data.dataset.croissant_adapter import (
    load_measurements_from_croissant,
    load_weather_from_croissant,
)
from s4casting.data.dataset.dataset import load_memmap
from s4casting.data.dataset.numpy_data import NumpyData

MEASUREMENTS_JSON = Path("data/anon_measurements.json")
WEATHER_JSON = Path("data/weather_openmeteo_anon_cdb_only.json")
CROISSANT_DIR = Path("data/LianderPower")

ATOL = 1e-5  # float32 round-trip tolerance

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _loc_by_latlon(nd: NumpyData) -> dict[tuple[float, float], int]:
    """Return {(lat, lon): loc_id} rounded to 6 dp to absorb float noise."""
    return {(round(v["lat"], 6), round(v["lon"], 6)): k for k, v in nd.locations.items()}


def _spans_for_loc(nd: NumpyData, loc_id: int) -> np.ndarray:
    """Return spans rows for a single location, sorted by start_time."""
    rows = nd.spans[nd.spans[:, 1] == loc_id]
    return rows[np.argsort(rows[:, 2])]


def _data_for_span(nd: NumpyData, span_row: np.ndarray) -> np.ndarray:
    """Return the raw data slice for a single span row."""
    idx, _, _, num = span_row
    return nd.data[int(idx) : int(idx) + int(num)]


def check(label: str, condition: bool, detail: str = "") -> bool:
    """Log a PASS/FAIL line and return the condition.

    Returns:
        bool: The condition value passed in.
    """
    status = "PASS" if condition else "FAIL"
    suffix = f": {detail}" if detail else ""
    log.info("  [%s] %s%s", status, label, suffix)
    return condition


# ---------------------------------------------------------------------------
# measurement comparison
# ---------------------------------------------------------------------------


def validate_measurements(legacy: NumpyData, croissant: NumpyData) -> int:
    """Compare legacy and croissant measurement datasets for equality.

    Returns:
        int: Number of failed checks.
    """
    failures = 0
    log.info("\n=== measurements ===")

    if not check(
        "sample_interval matches",
        legacy.sample_interval == croissant.sample_interval,
        f"{legacy.sample_interval}s vs {croissant.sample_interval}s",
    ):
        failures += 1

    leg_map = _loc_by_latlon(legacy)
    cro_map = _loc_by_latlon(croissant)

    only_legacy = set(leg_map) - set(cro_map)
    only_croissant = set(cro_map) - set(leg_map)
    shared = set(leg_map) & set(cro_map)

    if not check(
        "same location count",
        len(only_legacy) == 0 and len(only_croissant) == 0,
        f"shared={len(shared)}, only-legacy={len(only_legacy)}, only-croissant={len(only_croissant)}",
    ):
        failures += 1

    span_mismatches = 0
    value_mismatches = 0
    locations_checked = 0

    for latlon in shared:
        leg_id = leg_map[latlon]
        cro_id = cro_map[latlon]
        leg_spans = _spans_for_loc(legacy, leg_id)
        cro_spans = _spans_for_loc(croissant, cro_id)

        if len(leg_spans) != len(cro_spans):
            span_mismatches += 1
            continue

        for ls, cs in zip(leg_spans, cro_spans):
            if ls[2] != cs[2] or ls[3] != cs[3]:  # start_time, num_values
                span_mismatches += 1
                break
            leg_vals = _data_for_span(legacy, ls)
            cro_vals = _data_for_span(croissant, cs)
            if not np.allclose(leg_vals, cro_vals, atol=ATOL, equal_nan=True):
                value_mismatches += 1
                break

        locations_checked += 1

    if not check(
        "all shared locations have matching spans", span_mismatches == 0, f"{span_mismatches} locations differ"
    ):
        failures += 1
    if not check(
        "all shared locations have matching values", value_mismatches == 0, f"{value_mismatches} locations differ"
    ):
        failures += 1

    log.info("  (checked %d/%d shared locations)", locations_checked, len(shared))
    return failures


# ---------------------------------------------------------------------------
# weather comparison
# ---------------------------------------------------------------------------


def validate_weather(legacy: NumpyData, croissant: NumpyData) -> int:
    """Compare legacy and croissant weather datasets for equality.

    Returns:
        int: Number of failed checks.
    """
    failures = 0
    log.info("\n=== weather ===")

    if not check(
        "sample_interval matches",
        legacy.sample_interval == croissant.sample_interval,
        f"{legacy.sample_interval}s vs {croissant.sample_interval}s",
    ):
        failures += 1

    if not check(
        "feature dimension matches",
        legacy.data.shape[1] == croissant.data.shape[1],
        f"{legacy.data.shape[1]} vs {croissant.data.shape[1]}",
    ):
        failures += 1
        return failures  # remaining checks would be meaningless

    leg_map = _loc_by_latlon(legacy)
    cro_map = _loc_by_latlon(croissant)

    only_legacy = set(leg_map) - set(cro_map)
    only_croissant = set(cro_map) - set(leg_map)
    shared = set(leg_map) & set(cro_map)

    if not check(
        "same location count",
        len(only_legacy) == 0 and len(only_croissant) == 0,
        f"shared={len(shared)}, only-legacy={len(only_legacy)}, only-croissant={len(only_croissant)}",
    ):
        failures += 1

    span_mismatches = 0
    value_mismatches = 0
    locations_checked = 0

    for latlon in shared:
        leg_id = leg_map[latlon]
        cro_id = cro_map[latlon]
        leg_spans = _spans_for_loc(legacy, leg_id)
        cro_spans = _spans_for_loc(croissant, cro_id)

        if len(leg_spans) != len(cro_spans):
            span_mismatches += 1
            continue

        for ls, cs in zip(leg_spans, cro_spans):
            if ls[2] != cs[2] or ls[3] != cs[3]:
                span_mismatches += 1
                break
            leg_vals = _data_for_span(legacy, ls)  # (T, n_features)
            cro_vals = _data_for_span(croissant, cs)
            if not np.allclose(leg_vals, cro_vals, atol=ATOL, equal_nan=True):
                value_mismatches += 1
                break

        locations_checked += 1

    if not check(
        "all shared locations have matching spans", span_mismatches == 0, f"{span_mismatches} locations differ"
    ):
        failures += 1
    if not check(
        "all shared locations have matching values", value_mismatches == 0, f"{value_mismatches} locations differ"
    ):
        failures += 1

    log.info("  (checked %d/%d shared locations)", locations_checked, len(shared))
    return failures


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and run all validation checks."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements-json", type=Path, default=MEASUREMENTS_JSON)
    parser.add_argument("--weather-json", type=Path, default=WEATHER_JSON)
    parser.add_argument("--croissant-dir", type=Path, default=CROISSANT_DIR)
    args = parser.parse_args()

    log.info("Loading legacy measurements from %s ...", args.measurements_json)
    leg_m = load_memmap(str(args.measurements_json), to_memory=True)

    log.info("Loading croissant measurements from %s ...", args.croissant_dir)
    cro_m = load_measurements_from_croissant(args.croissant_dir, to_memory=True)

    log.info("Loading legacy weather from %s ...", args.weather_json)
    leg_w = load_memmap(str(args.weather_json), to_memory=True)

    log.info("Loading croissant weather from %s ...", args.croissant_dir)
    cro_w = load_weather_from_croissant(args.croissant_dir, to_memory=True)

    total_failures = 0
    total_failures += validate_measurements(leg_m, cro_m)
    total_failures += validate_weather(leg_w, cro_w)

    log.info("\n%s", "=" * 40)
    if total_failures == 0:
        log.info("All checks passed.")
    else:
        log.info("%d check(s) FAILED.", total_failures)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
