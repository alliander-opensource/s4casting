# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

from s4casting.core.benchmarker import Benchmarker
from s4casting.core.config import BenchmarkingConfiguration
from s4casting.core.hooks import CommonHooks, TrainingHooks
from s4casting.eval.evaluator_heads import EvaluatorHead


def provide_benchmark(
    bench_config: BenchmarkingConfiguration,
    hookable: CommonHooks | TrainingHooks,
    evaluator_head: EvaluatorHead,
) -> Benchmarker:
    """Provide a Benchmarker instance.

    Args:
        bench_config (BenchmarkingConfiguration): Benchmarking configuration.
        hookable (CommonHooks | TrainingHooks): Hookable object to register hooks.
        evaluator_head (EvaluatorHead): evaluation_head for benchmarking

    Returns:
        Benchmarker: An instance of Benchmarker.
    """
    return Benchmarker(hookable, bench_config, evaluator_head)
