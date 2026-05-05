# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import pytest

from s4casting.core.config import Configuration, StefBeamBenchmark
from scripts.train import train
from tests.utils import reduce_targets_yaml_to_single_location, requires_cuda


@requires_cuda
@pytest.mark.parametrize("config_fixture", ["load_config_gmm", "load_config_quantile"])
def test_stefbeam_smoke(request, config_fixture: str):
    """End-to-end training smoke test.

    Args:
        request: Pytest request object to access fixtures.
        config_fixture: Name of the configuration fixture to use.
    """
    reduce_targets_yaml_to_single_location()

    cfg: Configuration = request.getfixturevalue(config_fixture)
    cfg.benchmarking.benchmarks["StefBeamBenchmark"] = StefBeamBenchmark(
        targets_file="liander2024_targets_one_location.yaml",
        predict_window_days=2,
        input_sample_interval_minutes=15,
    )
    cfg.machine.device_kind = "cuda"
    cfg.training.batch_size = 8
    cfg.training.evaluation_interval = 1
    cfg.training.checkpoint_interval = 1
    cfg.training.benchmarking_interval = 1
    cfg.training.maximum_steps = 2
    train(cfg)
