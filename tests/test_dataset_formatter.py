# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize(
    ("output_dir", "prefix"),
    [
        (Path("/tmp/tests/output_test"), "external_data_wrapped"),
        (Path("/tmp/tests/weather_output"), "weather"),
    ],
)
def test_dataset_formatter_output(output_dir: Path, prefix: str, request: pytest.FixtureRequest):
    """Test that checks if the dataset formatter for weather and external data works.

    Args:
        output_dir (Path): Directory where output files are stored.
        prefix (str): Prefix used for output files.
        request (pytest.FixtureRequest): Pytest request object to access session config.
    """
    if prefix == "weather" and not getattr(request.session.config, "weather_api_available", True):
        pytest.skip("Weather data unavailable because weather API is not available.")
    # 1. Check files exist
    np_file = output_dir / f"{prefix}.np"
    parquet_file = output_dir / f"{prefix}_spans.parquet"
    json_file = output_dir / f"{prefix}.json"

    assert np_file.exists(), f"Missing {np_file}"
    assert parquet_file.exists(), f"Missing {parquet_file}"
    assert json_file.exists(), f"Missing {json_file}"

    # 2. Validate spans parquet
    spans_df = pd.read_parquet(parquet_file)
    expected_columns = {"id", "location", "datetime_start", "num_values"}
    assert set(spans_df.columns) == expected_columns
    assert len(spans_df) > 0

    # 3. Validate metadata JSON
    with json_file.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    expected_keys = {"dimension", "sample_interval_minutes", "dataset", "spans", "locations"}
    assert expected_keys.issubset(metadata.keys())

    # Check locations dict is not empty
    assert isinstance(metadata["locations"], dict)
    assert len(metadata["locations"]) > 0

    # Check lat/lon exist for each location
    for _, loc_data in metadata["locations"].items():
        assert loc_data["lon"] is not None
        assert loc_data["lat"] is not None

    # check if memmap has correct size
    data_memmap = np.memmap(output_dir / f"{prefix}.np", mode="r", dtype="float32")
    n_features = 38 if prefix == "weather" else 1

    assert data_memmap.shape[0] == spans_df["num_values"].sum() * n_features
