# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

from typing import TYPE_CHECKING

from s4casting.data.dataset.dataset import hash_all_memmaps

if TYPE_CHECKING:
    from torch.optim.lr_scheduler import LRScheduler
    from torch.optim.optimizer import Optimizer

    from s4casting.core.batcher import Batcher
    from s4casting.core.benchmarker import Benchmarker
    from s4casting.core.checkpoint import Checkpointer
    from s4casting.core.config import Configuration
    from s4casting.core.evaluator import MetricsEvaluator  # type: ignore[attr-defined]
    from s4casting.core.machine import Machine
    from s4casting.core.model_container import ModelContainer
    from s4casting.core.trainer import Trainer


class Context:
    """Context for managing the state of the training and evaluation process."""

    def __init__(
        self,
        *,
        configuration: "Configuration",
        model_container: "ModelContainer",
        optimizer: "Optimizer",
        scheduler: "LRScheduler | None",
        machine: "Machine",
        batcher: "Batcher | None" = None,
        trainer: "Trainer | None" = None,
        checkpointer: "Checkpointer | None" = None,
        evaluator: "MetricsEvaluator | None" = None,
        benchmarker: "Benchmarker | None" = None,
    ) -> None:
        """Initialize the Context.

        Args:
            configuration (Configuration): Configuration object.
            model_container (ModelContainer): Model container.
            optimizer (Optimizer): Optimizer.
            scheduler (LRScheduler): Learning rate scheduler.
            machine (Machine): Machine information.
            batcher (Batcher | None): Batcher for data loading.
            trainer (Trainer | None): Trainer for training process.
            checkpointer (Checkpointer | None): Checkpointer for saving/loading checkpoints.
            evaluator (MetricsEvaluator | None): Evaluator for metrics calculation.
            benchmarker (Benchmarker | None): Benchmarker for model evaluation.
        """
        self.configuration = configuration
        self.model_container = model_container
        self.trainer = trainer
        self.batcher = batcher
        self.machine = machine
        self.checkpointer = checkpointer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.evaluator = evaluator
        self.benchmarker = benchmarker
        self.eval_metrics: dict = {}
        self.benchmark_metrics: dict = {}
        self.benchmark_location: str = ""
        self.loss: float = 0.0
        self.validation_loss: float = 0.0
        self.input_validation_sample_rate: int = 15
        self.output_validation_sample_rate: int = 15
        dps = self.batcher.datasets_per_source if self.batcher else {}
        self.measurements_hash = (
            hash_all_memmaps(dps["measurements"]) if "measurements" in dps and configuration.io.hash_datasets else ""
        )
        self.weather_hash = (
            hash_all_memmaps(dps["weather"]) if "weather" in dps and configuration.io.hash_datasets else ""
        )
