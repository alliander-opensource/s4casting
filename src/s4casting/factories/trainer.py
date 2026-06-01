# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

from s4casting.core.config import OptimizerConfiguration, TrainingConfiguration
from s4casting.core.machine import Machine
from s4casting.core.trainer import Trainer


def provide_trainer(config: TrainingConfiguration, optimizer: OptimizerConfiguration, machine: Machine) -> Trainer:
    """Provide a Trainer instance.

    Args:
        config (TrainingConfiguration): Training configuration.
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
        optimizer,
        machine,
        gradient_accumulation_steps=config.gradient_accumulation_steps // machine.world_size,
    )
