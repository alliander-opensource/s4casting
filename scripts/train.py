# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import os

from s4casting.core.wandb_setup import setup_wandb  # type: ignore

wandb_root = ""
if any(var in os.environ for var in ["AWS_EXECUTION_ENV", "AWS_REGION", "AWS_DEFAULT_REGION"]):
    wandb_root = "/mnt/sagemaker-nvme/wandb_storage"
elif "SINGULARITY_CONTAINER" in os.environ:
    wandb_root = "/gpfs/projects/ehpc605/wandb_storage"
setup_wandb(wandb_root)

import random

import numpy as np
import torch

from s4casting import factories as fc
from s4casting.core.authenticate import configure_authentication
from s4casting.core.cli import get_configuration
from s4casting.core.config import Configuration
from s4casting.core.context import Context
from s4casting.core.logger import CSVLogger, StdLogger, WandbLogger


def train(config: Configuration):
    """Main training function.

    Args:
        config (Configuration): Configuration object
    """
    if config.authentication:
        configure_authentication(config.authentication)

    machine = fc.provide_machine(config.machine, rng_base_seed=config.run.seed)

    # all ranks use the same seed for model init so DDP starts with identical weights
    torch.manual_seed(config.run.seed)
    random.seed(config.run.seed)
    np.random.seed(config.run.seed)

    model_container = fc.provide_model_container(config.model, config.io, machine)

    # diverge seeds per rank for data sampling
    torch.manual_seed(machine.local_seed)
    random.seed(machine.local_seed + 0x1FF)
    np.random.seed(machine.local_seed + 0x2FF)

    optimizer = fc.provide_optimizer(config.optimizer, model_container.raw_model.parameters())
    scheduler = fc.provide_scheduler(config.scheduler, optimizer)
    trainer = fc.provide_trainer(config=config.training, io=config.io, optimizer=config.optimizer, machine=machine)
    checkpointer = fc.provide_checkpointer(config.io, trainer.hooks)
    evaluator_head = fc.provide_evaluator_head(config.model, trainer.hooks)
    evaluator = fc.provide_evaluation(trainer.hooks, evaluator_head)
    batcher = fc.provide_batcher(
        config.io,
        config.model,
        config.training,
        config.validation,
        config.benchmarking,
        config.run,
        machine,
        trainer.hooks,
    )
    benchmarker = fc.provide_benchmark(config.benchmarking, trainer.hooks, evaluator_head)

    if machine.main_process:
        StdLogger(trainer.hooks, config.training)
        WandbLogger(trainer.hooks, config.run, config.authentication, config.io.output)
        CSVLogger(trainer.hooks, config.run, config.io.output)

    ctx = Context(
        configuration=config,
        model_container=model_container,
        optimizer=optimizer,
        scheduler=scheduler,
        machine=machine,
        trainer=trainer,
        checkpointer=checkpointer,
        evaluator=evaluator,
        benchmarker=benchmarker,
        batcher=batcher,
    )

    with machine.context(config.model.internal_dtype):
        trainer.train(ctx)


if __name__ == "__main__":
    train(get_configuration())
