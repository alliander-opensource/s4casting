# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

"""Run the fev-bench tasks configured under [benchmarking.benchmarks.FevBenchBenchmark] on a checkpoint.

Example:
    uv run python scripts/run_fevbench.py configs/transformer.toml --checkpoint out/checkpoint_10000.pt
"""

import argparse
import logging
import sys

from s4casting import factories as fc
from s4casting.core.checkpoint import Checkpointer
from s4casting.core.config import Configuration
from s4casting.core.context import Context
from s4casting.core.hooks import CommonHooks
from s4casting.eval.fev_bench.callback import BENCHMARK_KEY, run_fevbench
from s4casting.inference.runner import load_config


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        argv: Arguments to parse; None uses sys.argv.

    Returns:
        Parsed arguments.
    """
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint on the configured fev-bench tasks.")
    ap.add_argument("config", help="Path to the experiment TOML with a FevBenchBenchmark section.")
    ap.add_argument("--checkpoint", help="Checkpoint .pt to load; defaults to io.load_checkpoint from the config.")
    ap.add_argument("--iteration", type=int, default=0, help="Iteration recorded in the results and logs.")
    ap.add_argument("--tasks", nargs="*", help="dataset_config names to run instead of the configured subset.")
    ap.add_argument("--num-windows", type=int, help="Override the rolling windows per task (disables the leaderboard).")
    return ap.parse_args(argv)


def build_context(config: Configuration, checkpoint: str | None) -> Context:
    """Build the minimal context the benchmark needs: model, machine, hooks and loggers.

    Args:
        config: Experiment configuration.
        checkpoint: Checkpoint path, or None to run the freshly initialised model.

    Returns:
        Context with the model loaded.
    """
    machine = fc.provide_machine(config.machine, rng_base_seed=config.run.seed)
    model_container = fc.provide_model_container(config.model, config.io, machine)
    optimizer = fc.provide_optimizer(config.optimizer, model_container.raw_model.parameters())
    trainer = fc.provide_trainer(config=config.training, optimizer=config.optimizer, machine=machine)
    fc.provide_loggers(
        config.logging, config.run, config.training, config.authentication, config.io.output, trainer.hooks
    )
    context = Context(
        configuration=config,
        model_container=model_container,
        optimizer=optimizer,
        scheduler=None,
        machine=machine,
        trainer=trainer,
    )
    if checkpoint:
        Checkpointer(hookable=CommonHooks(), load=checkpoint).load(context)
    return context


def main(argv=None) -> None:
    """Run the benchmark and print the results.

    Args:
        argv: Command line arguments; None uses sys.argv.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    config = load_config(args.config)
    bench = config.benchmarking.benchmarks.get(BENCHMARK_KEY)
    if bench is None:
        sys.exit(f"{args.config} has no [benchmarking.benchmarks.{BENCHMARK_KEY}] section.")
    if args.tasks:
        bench.task_subset = args.tasks  # type: ignore[union-attr]
    if args.num_windows:
        bench.num_windows = args.num_windows  # type: ignore[union-attr]

    context = build_context(config, args.checkpoint or config.io.load_checkpoint)
    context.trainer.hooks.start.call(context)  # type: ignore[union-attr]
    run_fevbench(context, args.iteration)
    context.trainer.hooks.finished.call(context)  # type: ignore[union-attr]

    summary = context.fev_bench_summary  # type: ignore[attr-defined]
    columns = [c for c in ("task_name", "test_error", "MASE", "WQL", "WAPE", "inference_time_s") if c in summary]
    print(summary[columns].to_string(index=False))  # noqa: T201
    print(f"Summaries written to {context.fev_bench_results_path}")  # noqa: T201  # ty: ignore[unresolved-attribute]
    leaderboard = context.fev_bench_leaderboard  # type: ignore[attr-defined]
    if leaderboard is not None:
        print(leaderboard.to_string())  # noqa: T201


if __name__ == "__main__":
    main(sys.argv[1:])
