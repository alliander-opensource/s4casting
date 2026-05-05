# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

from s4casting.core.evaluator import Evaluator
from s4casting.core.hooks import CommonHooks, TrainingHooks
from s4casting.eval.evaluator_heads import EvaluatorHead


def provide_evaluation(
    hookable: CommonHooks | TrainingHooks,
    evaluator_head: EvaluatorHead,
) -> Evaluator:
    """Provide an Evaluator instance.

    Args:
        hookable (CommonHooks | TrainingHooks): Hookable object to register hooks.
        evaluator_head (EvaluatorHead): evaluation_head for benchmarking

    Returns:
        Evaluator: An instance of Evaluator.
    """
    return Evaluator(hookable, evaluator_head)
