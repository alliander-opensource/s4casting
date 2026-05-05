# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

from torch.optim import RAdam
from torch.optim.optimizer import Optimizer, ParamsT

from s4casting.core.config import OptimizerConfiguration


def provide_optimizer(config: OptimizerConfiguration, model_parameters: ParamsT) -> Optimizer:
    """Provide an Optimizer instance.

    Args:
        config (OptimizerConfiguration): Optimizer configuration.
        model_parameters (ParamsT): Model parameters to optimize.

    Returns:
        Optimizer: An instance of Optimizer.
    """
    return RAdam(
        params=model_parameters,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )
