# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import pytest

from s4casting.core.config import Configuration
from scripts.train import train

BASE_CONFIGS = [
    "tests/assets/gmm.toml",
    "tests/assets/quantiles.toml",
]


@pytest.mark.parametrize("config_fixture", ["load_config_gmm", "load_config_quantile"])
def test_train_end_to_end_cpu(request, config_fixture):
    """End-to-end training smoke test.

    Args:
        request: Pytest request object to access fixtures.
        config_fixture: Name of the configuration fixture to use.
    """
    # Load existing config

    cfg: Configuration = request.getfixturevalue(config_fixture)

    cfg.machine.device_kind = "cpu"

    cfg.training.batch_size = 8
    cfg.training.evaluation_interval = 1
    cfg.training.checkpoint_interval = 1
    cfg.training.benchmarking_interval = 10
    cfg.training.maximum_steps = 2

    # Run; test passes if no exception is raised
    train(cfg)


@pytest.mark.weather
def test_train_end_to_end_weather_cpu(load_config_with_weather: Configuration):
    """End-to-end training smoke test.

    Args:
        load_config_with_weather (Configuration): Configuration fixture with weather data.
    """
    cfg: Configuration = load_config_with_weather
    cfg.machine.device_kind = "cpu"
    cfg.training.batch_size = 8
    cfg.training.evaluation_interval = 1
    cfg.training.checkpoint_interval = 1
    cfg.training.benchmarking_interval = 10
    cfg.training.maximum_steps = 2

    # Run; test passes if no exception is raised
    train(cfg)
