# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import pytest

from s4casting.core.config import Configuration, TransformerConfiguration
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


def test_train_end_to_end_cpu_cpm(load_config_gmm: Configuration):
    """End-to-end training smoke test with the contiguous patch masking task.

    Args:
        load_config_gmm (Configuration): Configuration fixture with a GMM head.
    """
    cfg: Configuration = load_config_gmm
    cfg.machine.device_kind = "cpu"
    cfg.training.task = "cpm"
    cfg.training.cpm.p_max = 0.5
    cfg.training.cpm.continuous_max_days = 1 / 3  # 4 patches of 8 at 15 minutes

    cfg.training.batch_size = 8
    cfg.training.evaluation_interval = 1
    cfg.training.checkpoint_interval = 1
    cfg.training.benchmarking_interval = 10
    cfg.training.maximum_steps = 2

    # Run; test passes if no exception is raised
    train(cfg)


def test_train_end_to_end_cpu_cpm_transformer(load_config_gmm: Configuration):
    """End-to-end training smoke test with CPM on the transformer backbone, masking validation as well.

    Args:
        load_config_gmm (Configuration): Configuration fixture with a GMM head.
    """
    cfg: Configuration = load_config_gmm
    cfg.machine.device_kind = "cpu"
    cfg.model.model = "transformer"
    cfg.model.transformer = TransformerConfiguration(latent_dim=32, n_heads=4, n_layers=1)
    cfg.training.task = "cpm"
    cfg.training.cpm.p_max = 0.5
    cfg.training.cpm.continuous_max_days = 1 / 3  # 4 patches of 8 at 15 minutes
    cfg.training.cpm.apply_to_validation = True

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


@pytest.mark.parametrize("model", ["ssm", "transformer"])
def test_train_end_to_end_cpu_multi_rate(load_config_gmm: Configuration, model: str):
    """Multi-rate training: the forecast window follows each batch's rate (2 samples at daily rate, not 192).

    Args:
        load_config_gmm (Configuration): Configuration fixture with a GMM head.
        model (str): Backbone under test.
    """
    cfg: Configuration = load_config_gmm
    cfg.machine.device_kind = "cpu"
    cfg.model.model = model
    if model == "transformer":
        cfg.model.transformer = TransformerConfiguration(latent_dim=32, n_heads=4, n_layers=1, causal=True)
        cfg.training.task = "cpm"
        cfg.training.cpm.continuous_max_days = 1
        cfg.training.cpm.min_context_days = 1
        cfg.model.predict_width = (0.05, 0.3)
    cfg.model.context_window = [11, 16, 32]
    cfg.model.input_sample_intervals_minutes = [15, 60, 1440]
    cfg.model.output_sample_intervals_minutes = [15, 60, 1440]

    cfg.training.batch_size = 8
    cfg.training.evaluation_interval = 1
    cfg.training.checkpoint_interval = 1
    cfg.training.benchmarking_interval = 10
    cfg.training.maximum_steps = 3

    train(cfg)


@pytest.mark.parametrize("n_time_features", [1, 3])
def test_train_end_to_end_cpu_time_encodings(load_config_with_time: Configuration, n_time_features: int):
    """Training runs with the time loader emitting unix time alone (1) or unix time plus latitude/longitude (3).

    Args:
        load_config_with_time (Configuration): Configuration fixture with the time loader.
        n_time_features (int): Channels emitted by the time loader.
    """
    cfg: Configuration = load_config_with_time
    cfg.machine.device_kind = "cpu"
    cfg.io.features["time"].n_features = n_time_features

    cfg.training.batch_size = 8
    cfg.training.evaluation_interval = 1
    cfg.training.checkpoint_interval = 1
    cfg.training.benchmarking_interval = 10
    cfg.training.maximum_steps = 2

    train(cfg)
