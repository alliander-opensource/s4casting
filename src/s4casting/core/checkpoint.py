# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import io
import logging

import torch
from pydantic import BaseModel

from s4casting.core.context import Context
from s4casting.core.hooks import CommonHooks, TrainingHooks
from s4casting.data.files.loader import FileAccess
from s4casting.inference.weights import MODULE_PREFIX, SAFETENSORS_SUFFIX, read_safetensors


class SavedCheckpoint(BaseModel):
    """Saved checkpoint structure."""

    torch_model: bytes
    torch_optimizer: bytes
    iteration: int
    eval_metrics: dict
    benchmark_metrics: dict
    loss: float


class Checkpointer:
    """Checkpointer for saving and loading model checkpoints."""

    def __init__(
        self,
        hookable: CommonHooks | TrainingHooks,
        *,
        load: str | None = None,
        save_directory: str | None = None,
    ) -> None:
        """Initialize the Checkpointer.

        Args:
            hookable (CommonHooks | TrainingHooks): Hookable object to register hooks.
            load (str | None): Path to load checkpoint from.
            save_directory (str | None): Directory to save checkpoints to.
        """
        self._load = load
        self._save_directory = save_directory
        self.last_checkpoint: FileAccess | None = None

        hookable.start.register(self.load)
        hookable.finished.register(self.finished)

        if isinstance(hookable, TrainingHooks):
            hookable.checkpoint.register(self.checkpoint)

    def load(self, context: Context) -> None:
        """Load checkpoint into the model and optimizer.

        Args:
            context (Context): Training context.
        """
        if not self._load:
            return

        if self._load.endswith(SAFETENSORS_SUFFIX):
            self._warm_start(context)
            return

        checkpoint = FileAccess(self._load).load_pydantic()
        # weights_only restricts unpickling to tensors and primitive containers.
        state_dict = torch.load(
            io.BytesIO(checkpoint["torch_model"]), map_location=context.machine.torch_device, weights_only=True
        )
        context.model_container.model.load_state_dict(self._align_ddp_keys(context, state_dict))
        context.optimizer.load_state_dict(
            torch.load(
                io.BytesIO(checkpoint["torch_optimizer"]),
                map_location=context.machine.torch_device,
                weights_only=True,
            )
        )

        context.model_container.model.to(context.machine.torch_device)

    def _warm_start(self, context: Context) -> None:
        """Initialise the model from a released safetensors weights file.

        Released weights carry no optimizer state, so this is a warm start rather than a
        resume: the optimizer, scheduler and iteration counter start fresh.

        Args:
            context (Context): Training context.
        """
        state_dict, metadata = read_safetensors(self._load)  # type: ignore[arg-type]
        context.model_container.model.load_state_dict(self._align_ddp_keys(context, state_dict))
        context.model_container.model.to(context.machine.torch_device)
        logging.info(
            "Warm start from %s (source iteration %s, code %s); optimizer state starts fresh.",
            self._load,
            metadata.get("s4casting.checkpoint_iteration", "unknown"),
            metadata.get("s4casting.code_tag", "unknown"),
        )

    @staticmethod
    def _align_ddp_keys(context: Context, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Add or strip the ``module.`` prefix so the keys match the (non-)DDP model in use.

        Args:
            context (Context): Training context, consulted for the DDP flag.
            state_dict (dict[str, torch.Tensor]): Weights as read from disk.

        Returns:
            dict[str, torch.Tensor]: Weights under keys the current model accepts.
        """
        first_key = next(iter(state_dict.keys()))
        if not bool(context.machine.ddp) and first_key.startswith(MODULE_PREFIX):
            return {key.removeprefix(MODULE_PREFIX): value for key, value in state_dict.items()}
        if bool(context.machine.ddp) and not first_key.startswith(MODULE_PREFIX):
            return {MODULE_PREFIX + key: value for key, value in state_dict.items()}
        return state_dict

    def save(self, context: Context, iteration: int | None) -> None:
        """Save checkpoint from the model and optimizer.

        Args:
            context (Context): Training context.
            iteration (int | None): Current training iteration.
        """
        self.last_checkpoint = FileAccess(
            self._save_directory + ("/last_checkpoint.pt" if iteration is None else f"/checkpoint_{iteration}.pt")
        )

        model_buffer = io.BytesIO()
        torch.save(context.model_container.model.state_dict(), model_buffer)
        model_bytes = model_buffer.getvalue()

        optimizer_buffer = io.BytesIO()
        torch.save(context.optimizer.state_dict(), optimizer_buffer)
        optimizer_bytes = optimizer_buffer.getvalue()

        checkpoint = SavedCheckpoint(
            torch_model=model_bytes,
            torch_optimizer=optimizer_bytes,
            iteration=context.trainer.iteration,  # type: ignore[attr-defined]
            eval_metrics=context.eval_metrics,
            benchmark_metrics=context.benchmark_metrics,
            loss=context.loss,
        )
        self.last_checkpoint.save_pydantic(checkpoint)

    def checkpoint(self, context: Context, iteration: int) -> None:
        """Checkpoint the model and optimizer at the current iteration.

        Args:
            context (Context): Training context.
            iteration (int): Current training iteration.
        """
        if not self._save_directory:
            return

        self.save(context, iteration)

    def finished(self, context: Context) -> None:
        """Save final checkpoint when training is finished.

        Args:
            context (Context): Training context.
        """
        if self._save_directory:
            self.save(context, None)
