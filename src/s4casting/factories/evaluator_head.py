# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

from s4casting.core.config import ModelConfiguration
from s4casting.core.hooks import CommonHooks, TrainingHooks
from s4casting.eval.evaluator_heads import EvaluatorHead


def provide_evaluator_head(
    model_config: ModelConfiguration,
    hookable: CommonHooks | TrainingHooks,
) -> EvaluatorHead:
    """Provide a evaluation head instance.

    Args:
        model_config (ModelConfiguration): Model configuration.
        hookable (CommonHooks | TrainingHooks): Hookable object to register hooks.

    Returns:
        evaluator_head: An instance of EvaluatorHead.
    """
    return EvaluatorHead(
        model_config.output_head.arch,
        hookable,
    )
