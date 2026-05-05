# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import s4casting.core.config as cfg
from s4casting.factories.machine import provide_machine
from s4casting.factories.model_container import provide_model_container


def test_create_model(tmp_path):
    """Tests that the model container is created correctly with the specified configuration.

    Args:
        tmp_path (pathlib.Path): Temporary path for creating dummy dataset locations.
    """
    config = cfg.ModelConfiguration(ssm=cfg.SSMConfiguration())
    io_config = cfg.IOConfiguration(
        feature_order=["main"],
        features={"main": cfg.DatasetConfiguration(name="main", location=str(tmp_path / "main"))},
        output=str(tmp_path / "output"),
        load_checkpoint=None,
    )
    machine_config = cfg.MachineConfiguration(device_kind="cpu")
    machine = provide_machine(machine_config, rng_base_seed=0)
    model_container = provide_model_container(config, io_config=io_config, machine=machine)
    assert len(model_container.raw_model.ss_layers) == config.ssm.n_layers
