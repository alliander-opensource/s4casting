# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import math

import torch
import torch.distributed as dist

from s4casting.core.config import IOConfiguration, OptimizerConfiguration, TrainingConfiguration
from s4casting.core.context import Context
from s4casting.core.functional import select_rate
from s4casting.core.hooks import TrainingHooks
from s4casting.core.machine import Machine


class Trainer:
    """Trainer for managing the training process."""

    def __init__(
        self,
        config: TrainingConfiguration,
        io: IOConfiguration,
        optimizer: OptimizerConfiguration,
        machine: Machine,
        gradient_accumulation_steps: int,
    ) -> None:
        """Initialize the Trainer.

        Args:
            config (TrainingConfiguration): Training configuration.
            io (IOConfiguration): IO configuration.
            optimizer (OptimizerConfiguration): Optimizer configuration.
            machine (Machine): Machine information.
            gradient_accumulation_steps (int): Number of gradient accumulation steps.
        """
        self._config = config
        self._io = io
        self._optimizer = optimizer
        self._iteration = self._io.iteration
        self._scores = {}
        self._gradient_accumulation_steps = gradient_accumulation_steps
        self._evaluation_interval = config.evaluation_interval
        self._checkpoint_interval = config.checkpoint_interval
        self._benchmarking_interval = config.benchmarking_interval
        self._ddp = machine.ddp
        self._ddp_loss_sync = machine.ddp_loss_sync
        self._main_process = machine.main_process
        self.hooks = TrainingHooks()
        # TODO: probably need to init seed for random.

    @property
    def iteration(self) -> int:
        """Get the current training iteration.

        Returns:
            int: Current training iteration.
        """
        return self._iteration

    def get_lr(self):
        """Get the current learning rate based on cosine decay schedule.

        Returns:
            float: Current learning rate.
        """
        it = self.iteration
        base = self._optimizer.learning_rate
        if it > self._config.maximum_steps:
            return base / 10

        decay_ratio = it / self._config.maximum_steps
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return (base / 10) + coeff * (base - (base / 10))

    def train(self, context: Context) -> None:
        """Run the training process.

        Args:
            context (Context): Training context.
        """
        self.hooks.start.call(context)

        # Baseline benchmark at iteration 0 so wandb has a pre-training reference point.
        if self._iteration == 0 and self._main_process:
            self.hooks.benchmark.call(context, 0)
            self.hooks.stef_beam.call(context, 0)

        while self._iteration < self._config.maximum_steps:
            self.epoch = self._iteration // self._config.n_samples_per_epoch
            if context.machine.ddp:
                context.batcher.train_loader.sampler.set_epoch(self.epoch)  # type: ignore[possibly-missing-attribute]
            for X_all, sample_config in context.batcher.train_loader:  # type: ignore[attr-defined]
                if self._iteration > 0 and self._main_process:
                    # Only needed after first set of iterations for the main process
                    if self._iteration % self._config.evaluation_interval == 0:
                        self.hooks.evaluate.call(context, self._iteration)
                    if self._iteration % self._config.checkpoint_interval == 0:
                        self.hooks.checkpoint.call(context, self._iteration)
                    if self._iteration % self._config.benchmarking_interval == 0:
                        self.hooks.benchmark.call(context, self._iteration)
                        self.hooks.stef_beam.call(context, self._iteration)
                if self._iteration > 0 and (self._iteration % self._config.n_samples_per_epoch == 0):
                    self.hooks.epoch.call(context, self._iteration)

                self._train_step(context, X_all, sample_config)
                self.hooks.step.call(context, self._iteration)
                if self._iteration >= self._config.maximum_steps:
                    break

        self.hooks.finished.call(context)

    def _train_step(self, context: Context, X_all, sample_config) -> None:
        total_loss = 0
        context.optimizer.zero_grad(set_to_none=True)

        B = X_all[0].shape[0] // self._gradient_accumulation_steps
        grad_accu = self._gradient_accumulation_steps
        for micro_step in range(self._gradient_accumulation_steps):
            if self._ddp:
                context.model_container.ddp.require_backward_grad_sync = (  # type: ignore[attr-defined]
                    micro_step == self._gradient_accumulation_steps - 1
                )

            X, Xm, Y, Ym = (
                x[micro_step * B : (micro_step + 1) * B].float().to(context.machine.torch_device) for x in X_all
            )
            input_sample_interval_minutes = sample_config.sample_interval_minutes[
                micro_step * B : (micro_step + 1) * B
            ].to(context.machine.torch_device)

            output_interval = select_rate(
                input_sample_interval_minutes, context.configuration.model.output_sample_intervals_minutes
            )  # ty: ignore[possibly-missing-attribute]
            _, loss = context.model_container.model(
                X,
                Xm,
                input_sample_interval_minutes,
                output_interval,
                Y,
                Ym,
            )

            loss = loss / grad_accu
            loss.backward()
            total_loss = total_loss + loss.item()

        if self._optimizer.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(
                context.model_container.model.parameters(), self._optimizer.gradient_clipping
            )

        context.optimizer.step()
        lr = self.get_lr()
        for g in context.optimizer.param_groups:
            g["lr"] = lr

        # Adjust the logged loss value to be the average of all processes
        # Slows down a bit because processes need to wait for each other. Optionally disable via config.
        if self._ddp and self._ddp_loss_sync:
            total_loss = torch.tensor(total_loss, device=context.machine.torch_device)
            dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)  # type: ignore[attr-defined]
            total_loss = total_loss.item()
        context.loss = total_loss
        self._iteration += 1
