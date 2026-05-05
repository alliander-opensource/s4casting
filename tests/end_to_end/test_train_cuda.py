# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import pandas as pd
import pytest

from s4casting.core.config import Configuration
from scripts.train import train
from tests.utils import requires_cuda

BASE_CONFIGS = [
    "tests/assets/gmm.toml",
    "tests/assets/quantiles.toml",
]


@requires_cuda
@pytest.mark.parametrize("config_fixture", ["load_config_gmm", "load_config_quantile"])
def test_train_end_to_end_cuda(request, config_fixture):
    """End-to-end training smoke test.

    Args:
        request: Pytest request object to access fixtures.
        config_fixture: Name of the configuration fixture to use.
    """
    cfg: Configuration = request.getfixturevalue(config_fixture)
    cfg.machine.device_kind = "cuda"

    # Run; test passes if no exception is raised
    train(cfg)

    df_metrics = pd.read_csv(f"out/benchmark_metrics_{cfg.run.seed}_{cfg.run.run_start_date}.csv")

    mae_05 = df_metrics["mae_0.5"].item()

    print(f"MAE(0.5): {mae_05} for config {config_fixture}")  # noqa: T201
    assert mae_05 < 0.2

    # TODO: add more assertions based on expected metrics, after refactoring GMM, like calibration and width


@pytest.mark.weather
@requires_cuda
def test_train_end_to_end_weather_cuda(load_config_with_weather: Configuration):
    """End-to-end training smoke test.

    Args:
        load_config_with_weather (Configuration): Configuration fixture with weather data.
    """
    cfg: Configuration = load_config_with_weather
    cfg.machine.device_kind = "cuda"

    # Run; test passes if no exception is raised
    train(cfg)
