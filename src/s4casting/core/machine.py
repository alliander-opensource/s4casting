# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

from collections.abc import Generator
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from typing import Literal

import torch
import torch.amp
from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt
from torch.distributed import destroy_process_group, init_process_group  # type: ignore[possibly-missing-import]


class MachineDDP(BaseModel):
    """Configuration for Distributed Data Parallel (DDP) setup."""

    backend: Literal["nccl", "gloo"]
    global_rank: NonNegativeInt
    local_rank: NonNegativeInt
    world_size: PositiveInt


class Machine(BaseModel):
    """Machine configuration for training and evaluation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    torch_device_kind: Literal["cuda", "cpu", "mps"]
    torch_device: torch.device
    total_memory_bytes: int
    ddp: MachineDDP | None
    ddp_loss_sync: bool = True
    rng_base_seed: int

    def model_post_init(self, _context):  # type: ignore
        """Initialize the process group for DDP if configured.

        Args:
            _context: Pydantic context (not used).
        """
        if self.ddp is not None:
            # Process group initialization is required for DDP model creation.
            init_process_group(self.ddp.backend, timeout=timedelta(minutes=30))

    def __del__(self):
        """Destroy the process group for DDP if it was initialized."""
        if self.ddp is not None:
            destroy_process_group()

    @contextmanager
    def context(self) -> Generator[None, None, None]:
        """Context manager for setting up the device and TF32 matmuls.

        Mixed precision is not applied here: a global autocast would also cover backward, optimizer
        steps and checkpointing. The forward+loss call sites open :meth:`autocast` themselves.

        Yields:
            Generator[None, None, None]: Context manager generator.
        """
        if self.torch_device_kind == "cuda":
            with torch.cuda.device(self.torch_device):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                yield
        else:
            yield

    def autocast(self, precision: Literal["fp32", "bf16"]):
        """Get the autocast context for one forward+loss call at the configured training precision.

        Args:
            precision (str): 'fp32' or 'bf16' (see training.precision).

        Returns:
            Context manager: torch.amp.autocast on CUDA for 'bf16', otherwise a no-op, so CPU runs
            (tests) silently stay fp32.
        """
        if self.torch_device_kind != "cuda" or precision == "fp32":
            return nullcontext()
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    @property
    def local_seed_offset(self) -> int:
        """Returns the local seed offset for the current process.

        Returns:
            int: Local seed offset.
        """
        return self.ddp.global_rank if self.ddp is not None else 0

    @property
    def local_seed(self) -> int:
        """Returns the local seed for the current process.

        Returns:
            int: Local seed.
        """
        return self.local_seed_offset + self.rng_base_seed

    @property
    def main_process(self) -> bool:
        """Check if the current process is the main process.

        Returns:
            bool: True if main process, False otherwise.
        """
        return self.ddp is None or self.ddp.global_rank == 0

    @property
    def benchmarking_device(self) -> str:
        """Get the device to be used for benchmarking.

        This is used because ddp seems to go into deadlock when we dont
        explicitly move model and data to a specific gpu.

        Returns:
            str: Device string for benchmarking.
        """
        # TODO: figure out the root cause of these problems
        return self.torch_device

    @property
    def world_size(self) -> int:
        """Get the world size for distributed training.

        Returns:
            int: World size.
        """
        return 1 if self.ddp is None else self.ddp.world_size
