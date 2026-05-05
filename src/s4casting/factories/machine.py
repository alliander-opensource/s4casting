# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import os

from torch import cuda, device

from s4casting.core.config import MachineConfiguration
from s4casting.core.machine import Machine, MachineDDP

TWO_GIGABYTE = 2 * 1024 * 1024 * 1024


def provide_machine(config: MachineConfiguration, rng_base_seed: int) -> Machine:
    """Provide a Machine instance.

    Args:
        config (MachineConfiguration): Machine configuration.
        rng_base_seed (int): Base seed for RNG.

    Returns:
        Machine: An instance of Machine.
    """
    torch_device_kind = config.device_kind
    ddp: MachineDDP | None = None
    local_rank = 0

    if config.ddp:
        for var in ["LOCAL_RANK", "RANK", "WORLD_SIZE"]:
            assert var in os.environ, (
                f"DDP enabled, but {var} not defined. (Did you use torchrun to start the training?)"
            )

        local_rank = int(os.environ["LOCAL_RANK"])
        ddp = MachineDDP(
            global_rank=int(os.environ["RANK"]),
            local_rank=local_rank,
            world_size=int(os.environ["WORLD_SIZE"]),
            backend="nccl" if torch_device_kind == "cuda" else "gloo",
        )
    else:
        local_rank = config.local_rank

    if torch_device_kind == "cuda":
        cuda.set_device(local_rank)

    torch_device = device(f"cuda:{local_rank}" if torch_device_kind == "cuda" else torch_device_kind)

    return Machine(
        torch_device_kind=torch_device_kind,
        torch_device=torch_device,
        total_memory_bytes=(
            cuda.get_device_properties(torch_device).total_memory if torch_device_kind == "cuda" else TWO_GIGABYTE
        ),
        ddp=ddp,
        ddp_loss_sync=config.ddp_loss_sync,
        rng_base_seed=rng_base_seed,
    )
