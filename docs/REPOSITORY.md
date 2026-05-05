<!--
SPDX-FileCopyrightText: Contributors to the s4casting project

SPDX-License-Identifier: MPL-2.0
-->

# Repository Overview

This document provides a high-level overview of the S4Casting code repository, outlining its structure, core components and execution flow.

## Top-Level Layout

| Path | Purpose |
| ---- | ------- |
| `configs/` | TOML configuration files for training, inference, benchmarking. |
| `scripts/` | Entry points (training, inference, benchmarking, utilities). |
| `src/s4casting/core/` | Core orchestration: context, batcher, benchmark runner, hooks, logging, checkpointing, configuration. |
| `src/s4casting/data/` | Data loading abstractions: datasets, intervals, indexing, weather & measurement interfaces. |
| `src/s4casting/model/` | Model container, architecture components (SSM, Transformer, patch encoder/decoder, output heads). |
| `src/s4casting/eval/` | Evaluators, metric heads, metrics logic, GMM utilities. |
| `src/s4casting/visualisation/` | Plotting helpers (training and benchmarking forecast visualization). |
| `notebooks/` | Interactive explanation and analysis (evaluation, inspection). |
| `docs/` | Markdown documentation (configuration, data sources, evaluation, this overview). |
| `tests/` | (If present) Unit tests. |
| `pyproject.toml` | Project metadata and dependencies. |

## Execution Flow (Training Loop)

1. Parse configuration (core/config.py).
2. Build components via factories (model, optimizer, scheduler, batcher, evaluator, logger, checkpoint).
3. Initialize Context with runtime objects.
4. Batcher creates train and validation DataLoaders (prediction or masking task).
5. Loop:
   a. Fetch batch (X, Xm, Y, Ym).
   b. Forward pass → distribution params (GMM or quantiles).
   c. Compute loss (NLL / Pinball / MSE).
   d. Backward, clip (optional), optimizer step, scheduler step (optional).
   e. Emit hooks:
      - step: every iteration (progress, live logging).
      - evaluate: run validation metrics.
      - checkpoint: persist model + optimizer state.
      - benchmark: run location benchmarking.
6. Evaluator runs validation metrics at evaluation_interval.
7. Logger records metrics, progress, plots.
8. Checkpointer saves state at checkpoint_interval.
9. Stop at maximum_steps → finished hook → artifacts persisted.

## Core Modules 

- `core/context.py`: Central state object.
- `core/batcher.py`: Interval slicing, dataset wrapping, DataLoader creation.
- `core/benchmarker.py`: benchmark runner: Location‑based benchmarking sampling & reporting.
- `core/hooks.py`: Event system (step, evaluate, benchmark_complete, etc.).
- `core/logger.py`: Stdout / CSV / WandB logging via hook callbacks.
- `core/checkpoint.py`: Save/load model + optimizer state.
- `core/config.py`: Pydantic models for all config sections.
- `core/trainer.py`: Main training loop orchestration.