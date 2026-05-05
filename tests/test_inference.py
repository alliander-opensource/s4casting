# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import pathlib

import pytest

from s4casting.core.config import Configuration
from s4casting.inference.runner import run_inference


@pytest.mark.parametrize(
    "config_fixture",
    [
        ("load_config_gmm"),
        ("load_config_quantile"),
    ],
)
def test_inference(request, config_fixture: str) -> None:
    """Tests the inference script functionality.

    Args:
        request: Pytest request object to access fixtures.
        config_fixture: Name of the configuration fixture to use.
    """
    # Define test parameters
    cfg: Configuration = request.getfixturevalue(config_fixture)

    data_path = "data/tests/sinusoid_data_raw/location_a.parquet"
    checkpoint_path = f"out/checkpoint_tmp_{cfg.model.output_head.arch}_{cfg.model.loss.loss}.pt"
    target_col = "measurements"
    time_col = "timestamp"
    save_path_pickle = "data/inference_results.pkl"

    # Run the inference function
    run_inference(
        config=cfg,
        data_path=data_path,
        checkpoint_path=checkpoint_path,
        target_col=target_col,
        time_col=time_col,
        save_path_pickle=save_path_pickle,
    )

    assert pathlib.Path(save_path_pickle).exists(), "Pickle file was not created."
