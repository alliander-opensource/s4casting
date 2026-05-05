# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

from s4casting.core.batcher import Batcher
from s4casting.core.config import (
    BenchmarkingConfiguration,
    IOConfiguration,
    ModelConfiguration,
    RunConfiguration,
    TrainingConfiguration,
    ValidationConfiguration,
)
from s4casting.core.hooks import TrainingHooks
from s4casting.core.machine import Machine


def provide_batcher(
    config: IOConfiguration,
    model_config: ModelConfiguration,
    train_config: TrainingConfiguration,
    validation_config: ValidationConfiguration,
    bench_config: BenchmarkingConfiguration,
    run_config: RunConfiguration,
    machine: Machine,
    hooks: TrainingHooks | None = None,
):
    """Provide a Batcher instance.

    Args:
        config (IOConfiguration): IO configuration.
        model_config (ModelConfiguration): Model configuration.
        train_config (TrainingConfiguration): Training configuration.
        validation_config (ValidationConfiguration): Validation configuration.
        bench_config (BenchmarkingConfiguration): Benchmarking configuration.
        run_config (RunConfiguration): Run configuration.
        machine (Machine): Machine information.
        hooks (TrainingHooks | None): Training hooks.

    Returns:
        Batcher: An instance of Batcher.
    """
    return Batcher(
        config, model_config, train_config, validation_config, bench_config, run_config, machine, hooks=hooks
    )
