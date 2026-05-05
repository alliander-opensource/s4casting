# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

from s4casting.core.config import IOConfiguration, OptimizerConfiguration, TrainingConfiguration
from s4casting.core.machine import Machine
from s4casting.core.trainer import Trainer


def provide_trainer(
    config: TrainingConfiguration, io: IOConfiguration, optimizer: OptimizerConfiguration, machine: Machine
) -> Trainer:
    """Provide a Trainer instance.

    Args:
        config (TrainingConfiguration): Training configuration.
        io (IOConfiguration): IO configuration.
        optimizer (OptimizerConfiguration): Optimizer configuration.
        machine (Machine): Machine information.

    Returns:
        Trainer: An instance of Trainer.
    """
    assert config.gradient_accumulation_steps % machine.world_size == 0, (
        f"Gradient accumulation steps {config.gradient_accumulation_steps} "
        f"needs to be divisible by world_size {machine.world_size}"
    )
    return Trainer(
        config,
        io,
        optimizer,
        machine,
        gradient_accumulation_steps=config.gradient_accumulation_steps // machine.world_size,
    )
