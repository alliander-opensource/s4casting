# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import shutil

import pytest

from s4casting.core.config import Configuration, DatasetConfiguration
from tests.utils import (
    create_locations_file,
    create_sinusoid_test_data,
    create_sinusoid_test_data_raw,
    create_tmp_model_checkpoint,
    create_weather_data,
    load_config,
)


@pytest.fixture()
def load_config_gmm() -> Configuration:
    """Load configuration from a TOML file for testing purposes.

    Returns:
        Configuration: The loaded configuration object.
    """
    cfg = load_config()
    cfg.model.loss.loss = "nll"
    cfg.model.output_head.arch = "gmm"
    cfg.model.output_head.n_gaussians = 2
    create_tmp_model_checkpoint(config=cfg)
    return cfg


@pytest.fixture()
def load_config_quantile() -> Configuration:
    """Load configuration from a TOML file for testing purposes.

    Returns:
        Configuration: The loaded configuration object.
    """
    cfg = load_config()
    cfg.model.loss.loss = "pinball"
    cfg.model.output_head.arch = "quantile"
    cfg.model.output_head.quantile_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    cfg.benchmarking.eval_quantiles = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    create_tmp_model_checkpoint(config=cfg)
    return cfg


@pytest.fixture()
def load_config_with_weather() -> Configuration:
    """Load configuration with weather data for testing purposes.

    Returns:
        Configuration: The loaded configuration object with weather data.
    """
    cfg = load_config()
    cfg.io.feature_order = ["measurements", "weather"]
    cfg.io.features["weather"] = DatasetConfiguration(
        location="/tmp/tests/weather_output/weather.json",
        loader="parquet",
        nearest_neighbor="true",
        main_dataset="measurements",
        n_features=38,
        subset_features=[
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "apparent_temperature",
            "precipitation",
            "rain",
            "snowfall",
            "snow_depth",
            "weather_code",
            "pressure_msl",
            "surface_pressure",
            "cloud_cover",
            "cloud_cover_low",
            "cloud_cover_mid",
            "cloud_cover_high",
            "et0_fao_evapotranspiration",
            "vapour_pressure_deficit",
            "wind_speed_10m",
            "wind_speed_100m",
            "wind_direction_10m",
            "wind_direction_100m",
            "wind_gusts_10m",
            "soil_temperature_0_to_7cm",
            "soil_temperature_7_to_28cm",
            "soil_temperature_28_to_100cm",
            "soil_temperature_100_to_255cm",
            "soil_moisture_0_to_7cm",
            "soil_moisture_7_to_28cm",
            "soil_moisture_28_to_100cm",
            "soil_moisture_100_to_255cm",
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",
            "direct_normal_irradiance",
            "global_tilted_irradiance",
            "terrestrial_radiation",
            "is_day",
            "sunshine_duration",
        ],
    )

    cfg.training.evaluation_interval = 5
    cfg.training.benchmarking_interval = 5
    cfg.training.maximum_steps = 5
    return cfg


@pytest.fixture()
def load_config_with_time() -> Configuration:
    """Load configuration with time data for testing purposes.

    Returns:
        Configuration: The loaded configuration object with time data.
    """
    cfg = load_config()
    cfg.io.feature_order = ["measurements", "time"]
    cfg.io.features["time"] = DatasetConfiguration(
        location="",
        loader="time",
        main_dataset="measurements",
        n_features=3,
    )

    cfg.training.batch_size = 8
    cfg.training.evaluation_interval = 1
    cfg.training.checkpoint_interval = 1
    cfg.training.benchmarking_interval = 10
    cfg.training.maximum_steps = 2
    return cfg


def pytest_sessionstart(session: pytest.Session):
    """Create temp sinusoid dataset files at startup."""
    config = session.config
    config.weather_api_available = True

    create_sinusoid_test_data_raw()
    create_locations_file()
    create_sinusoid_test_data()

    try:
        create_weather_data()
    except Exception as e:
        config.weather_api_available = False
        print(f"Warning: {e}")  # noqa: T201


def pytest_sessionfinish():
    """Cleanup temp sinusoid dataset files at shutdown."""
    shutil.rmtree("/tmp/tests/", ignore_errors=True)


def pytest_runtest_setup(item: pytest.Item):
    """Skip tests requiring weather data if not available.

    Args:
        item: The pytest test item.
    """
    if not getattr(item.session.config, "weather_api_available", True) and item.get_closest_marker("weather"):
        pytest.skip("Weather data unavailable")
