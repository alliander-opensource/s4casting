# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import torch

from s4casting.core.context import Context
from s4casting.core.functional import select_rate
from s4casting.core.hooks import CommonHooks, TrainingHooks
from s4casting.eval.evaluator_heads import EvaluatorHead


class Evaluator:
    """Evaluator for calculating metrics on the validation set."""

    def __init__(self, hookable: CommonHooks | TrainingHooks, head_evaluator: EvaluatorHead) -> None:
        """Initialize the Evaluator.

        Args:
            hookable (CommonHooks | TrainingHooks): Hookable object to register hooks.
            head_evaluator (EvaluatorHead): Evaluator for reporting results.
        """
        self.hooks = hookable
        hookable.finished.register(self.finished)
        self.head_evaluator = head_evaluator
        self._data_iter = None

        if isinstance(hookable, TrainingHooks):
            hookable.evaluate.register(self.evaluate)

    def finished(self, context: Context) -> None:
        """Run evaluation when training is finished.

        Args:
            context (Context): Training context.
        """
        self.evaluate(context, iteration=context.configuration.training.maximum_steps)

    def evaluate(self, context: Context, iteration: int) -> None:
        """Run evaluation on the validation set.

        Note: we only evaluate one batch now as there are no guarantees
              that consecutive batches are of the same sample rate,

        Args:
            context (Context): Training context.
            iteration (int): Current training iteration.
        """
        try:
            task, sample_config = next(self._data_iter)
        except (TypeError, StopIteration):
            self._data_iter = iter(context.batcher.validation_loader)  # ty: ignore[possibly-missing-attribute]
            task, sample_config = next(iter(context.batcher.validation_loader))  # ty: ignore[possibly-missing-attribute]

        model_config = context.configuration.model
        with torch.no_grad():
            context.model_container.model.train(mode=False)
            evaluation_model = context.model_container.raw_model.to(context.machine.benchmarking_device)

            X, Xm, Y, Ym = (x.float().to(context.machine.benchmarking_device) for x in task)
            output_interval = select_rate(
                sample_config.sample_interval_minutes, model_config.output_sample_intervals_minutes
            )
            prediction, loss = evaluation_model(X, Xm, sample_config.sample_interval_minutes, output_interval, Y, Ym)

            context.validation_loss = loss.item()
            context.input_validation_sample_rate = sample_config.sample_interval_minutes
            context.output_validation_sample_rate = output_interval

            self.head_evaluator.report(
                context=context,
                prediction=prediction[0:1, :, 0, ...],
                X=X[0:1],
                Xm=Xm[0:1],
                Y=Y[0:1],
                Ym=Ym[0:1],
                loss=context.validation_loss,
                iteration=iteration,
                report_type="evaluation",
                sample_config=sample_config,
                output_interval=output_interval,
                n_day_ahead=sample_config.predict_window_days[0].item(),
            )

            context.model_container.model.train(mode=True)
