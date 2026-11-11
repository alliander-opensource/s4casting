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
        # AdamW-style decay (scaled by the learning rate). RAdam's default couples the decay into the
        # gradient before Adam's normalisation, which drives every small-gradient weight to zero at one
        # learning rate per step: a 66M run with weight_decay = 0.01 had its LayerNorm, attention and MLP
        # weights at 1e-8 to 1e-42 by step 50k and emitted a constant forecast.
        decoupled_weight_decay=True,
    )
