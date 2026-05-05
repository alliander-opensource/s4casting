# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

from s4casting.core.checkpoint import Checkpointer
from s4casting.core.config import IOConfiguration
from s4casting.core.hooks import CommonHooks, TrainingHooks


def provide_checkpointer(config: IOConfiguration, hookable: CommonHooks | TrainingHooks) -> Checkpointer:
    """Provide a Checkpointer instance.

    Args:
        config (IOConfiguration): IO configuration.
        hookable (CommonHooks | TrainingHooks): Hookable object to register hooks.

    Returns:
        Checkpointer: An instance of Checkpointer.
    """
    return Checkpointer(hookable, load=config.load_checkpoint, save_directory=config.output)
