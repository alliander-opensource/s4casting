# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.optim.optimizer import Optimizer

from s4casting.core.config import SchedulerConfiguration


def provide_scheduler(config: SchedulerConfiguration, optimizer: Optimizer) -> LRScheduler:
    """Provide a learning rate scheduler instance.

    Args:
        config (SchedulerConfiguration): Scheduler configuration.
        optimizer (Optimizer): Optimizer for which to create the scheduler.

    Returns:
        LRScheduler: An instance of learning rate scheduler.
    """
    return ReduceLROnPlateau(
        optimizer=optimizer,
        mode=config.mode,
        factor=config.factor,
        patience=config.patience,
        threshold=config.threshold,
        threshold_mode=config.threshold_mode,
        cooldown=config.cooldown,
        min_lr=config.min_lr,
        eps=config.eps,
    )
