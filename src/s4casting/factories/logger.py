# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import logging

from s4casting.core.config import (
    AuthenticationConfiguration,
    LoggingConfiguration,
    RunConfiguration,
    TrainingConfiguration,
)
from s4casting.core.hooks import CommonHooks, TrainingHooks
from s4casting.core.logger import CSVLogger, LoggerInterface, MLflowLogger, StdLogger, WandbLogger


def provide_loggers(
    config: LoggingConfiguration,
    run_config: RunConfiguration,
    training_config: TrainingConfiguration,
    auth: AuthenticationConfiguration,
    output_dir: str,
    hookable: CommonHooks | TrainingHooks,
) -> list[LoggerInterface]:
    """Construct and register the configured logging providers.

    Args:
        config: Logging configuration.
        run_config: Shared run configuration.
        training_config: Training configuration.
        auth: Authentication configuration.
        output_dir: Local output directory.
        hookable: Hookable object to register logger callbacks on.

    Returns:
        Configured logger instances in report order.
    """
    loggers: list[LoggerInterface] = [
        StdLogger(hookable, training_config),
        CSVLogger(hookable, run_config, output_dir),
    ]
    logger_names = ["std", "csv"]

    if config.wandb is not None:
        loggers.append(WandbLogger(hookable, config.wandb, run_config, output_dir, auth))
        logger_names.append("wandb")

    if config.mlflow is not None:
        loggers.append(MLflowLogger(hookable, config.mlflow, run_config, output_dir))
        logger_names.append("mlflow")

    logging.info("Initialized logging providers: %s", ", ".join(logger_names))
    return loggers
