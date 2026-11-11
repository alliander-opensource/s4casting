<!--
SPDX-FileCopyrightText: Contributors to the s4casting project

SPDX-License-Identifier: MPL-2.0
-->

### Model Configuration

Model behavior and training setup are fully defined through TOML configuration files (see `configs/*.toml`).
Each section controls a specific component of the pipeline, from data loading to optimization.

Configuration files are strictly validated: an unknown or misspelled key raises an error at
startup that names the offending section and key, instead of being silently ignored and leaving
the default value in charge.

Below is an overview of the most relevant sections and what they control. Some default values are provided as examples and can be used as a good starting point.

#### **[machine]**

Defines how and where the model runs.

| Key           | Description                                | Example  |
| ------------- | ------------------------------------------ | -------- |
| `device_kind` | Execution device (`cpu` or `cuda`).        | `cuda`   |
| `ddp`         | Enable/disable distributed training (DDP). | `false`  |


#### **[run]**

Reproducibility and randomness control.

| Key         | Description                      | Example  |
| ----------- | -------------------------------- | -------- |
| `seed`      | Random seed for reproducibility. | `42069`  |
| `log_level` | Python logging level.            | `"info"` |

#### **[logging]**

Standard output and CSV logging are always enabled. Configure W&B and MLflow
by adding their respective sections:

```toml
[logging.wandb]
project = "forecasting-s4"
notes = "Optional run notes"
mode = "online"

[logging.mlflow]
uri = "http://127.0.0.1:5000/"
experiment = "dev"
```

**`[logging.wandb]`**

| Key       | Description                                            | Example             |
| --------- | ------------------------------------------------------ | ------------------- |
| `project` | Weights & Biases project name.                         | `"forecasting-s4"`  |
| `run_id`  | Existing W&B run ID to resume (optional).              | `"abc123"`          |
| `notes`   | Notes attached to the run (optional).                  | `"baseline run"`    |
| `mode`    | Run mode: `online` or `offline`.                       | `"online"`          |

W&B supports `online` and `offline` modes. Online mode requires
`authentication.wandb_api_key`; use offline mode when no key is configured.
When `run_id` is configured, W&B resumes an existing run or creates it when it
does not exist (`resume = "allow"`).

**`[logging.mlflow]`**

| Key          | Description                                                        | Example                     |
| ------------ | ------------------------------------------------------------------ | --------------------------- |
| `uri`        | MLflow tracking URI (free-form string).                            | `"http://127.0.0.1:5000/"`  |
| `workspace`  | MLflow workspace name.                                             | `"default"`                 |
| `experiment` | MLflow experiment name.                                            | `"dev"`                     |

The `uri` accepts any tracking URI understood by `mlflow.set_tracking_uri`, e.g.
a local tracking server (`mlflow server`, the default `http://127.0.0.1:5000/`),
a remote HTTP(S) tracking server, or a managed tracking backend. MLflow always
records system metrics and creates the configured workspace when it does not
exist. Its default workspace is `default`, which is recommended when logging
images due to an upstream
[MLflow Image Grid limitation](https://github.com/mlflow/mlflow/issues/22794).
See `configs/mlflow.toml` for a complete example configuration.

#### **[training]**

| Key                           | Description                               | Value     	|
| ----------------------------- | ----------------------------------------- | --------- 	|
| `batch_size`                  | Batch size per step                       | `32`      	|
| `gradient_accumulation_steps` | Gradient accumulation steps               | `1`       	|
| `evaluation_interval`         | Evaluate every N steps                    | `1000`    	|
| `checkpoint_interval`         | Save checkpoint every N steps             | `1000`    	|
| `benchmarking_interval`       | Run benchmarking every N steps            | `10_000`  	|
| `maximum_steps`               | Total training steps                      | `500_000` 	|
| `task`               			| Training task: `prediction`, `masking`, `randomprediction`, or `cpm` | `prediction` 	|
| `precision`                   | Training precision: `fp32` or `bf16` (mixed precision via autocast on CUDA; transformer only). Master weights, normalization statistics and the loss stay fp32, so checkpoints are precision-independent. On CPU, `bf16` silently runs fp32. | `fp32` |
| `covariate_dropout`           | Randomly drop covariates during training  | `true`    	|
| `covariate_dropout_all_p`     | With `covariate_dropout`, probability that a sample loses every covariate | `0.0` |
| `covariate_horizon_dropout_p` | With `covariate_dropout`, probability that a kept covariate is hidden over the forecast window while staying visible in the context: half the draws hide the whole window (a past-only covariate, as fev-bench presents `past_dynamic_columns`), half hide it from a random point onward (a forecast that runs out). The forecast window never keeps a covariate the context lost. | `0.0` |

#### **[training.cpm]**

Contiguous Patch Masking (CPM) after TiRex (Auer et al., 2025), active when `task = "cpm"`. On top of the fixed
forecast window, the target series inside the context is cut into blocks of `continuous_mask` model patches and each
block is masked with probability `p_mask`. Masked blocks are hidden from the model input and included in the loss. Per
sample, `continuous_mask ~ U{continuous_min, ..., continuous_max}` and `p_mask ~ U(0, p_max)`. The block-length bounds
and the protected leading history are given as durations in days and converted to patches per sample rate (at least
one patch), so multi-rate training masks comparable stretches of history at every rate. Covariates and time features
are never masked. Designed for `model = "transformer"`, whose patch encoder passes the mask to the model.

| Key                    | Description                                                                  | Value   |
| ---------------------- | ---------------------------------------------------------------------------- | ------- |
| `p_max`                | Upper bound of the per-sample block masking probability                      | `0.25`  |
| `continuous_max_days`  | Maximum length of one contiguous mask block, in days (2 days = 24 patches of 8 at 15 min, 6 patches hourly, one 8-day patch daily) | `2` |
| `continuous_min_days`  | Minimum length of one contiguous mask block, in days (unset: one patch at every rate) | unset |
| `min_context_days`     | Leading history that is never masked, in days (unset: none)                  | unset |
| `apply_to_validation`  | Also mask the validation set; off keeps the validation loss comparable       | `false` |

#### **[model]**

Core model architecture and input/output settings.

| Key                              	| Description                             	| Value  |
| -------------------------------- 	| --------------------------------------- 	| ------ |
| `context_window`               	| Context window size in days (list for multi-window inputs) | `[32]`   |
| `predict_width`             		| Forecast horizon: days (int, rescaled to each sample's rate under multi-rate training) or a `[min, max]` fraction range of the sample length drawn per sample (`randomprediction` and `cpm` tasks), which is rate independent. | `2`    |
| `base_sample_interval_minutes`   	| Base sampling interval (minutes)        	| `15`    |
| `input_sample_intervals_minutes` 	| Input sampling intervals for multi‑rate training (minutes). Each value must be a multiple of `base_sample_interval_minutes`. | `[15, 60]` |
| `output_sample_intervals_minutes` | Output sampling intervals for predictions (minutes). Each value must be a multiple of `base_sample_interval_minutes`. | `[15]` |
| `alignment`                      	| Temporal alignment window (minutes)     | `1440` |
| `model`                          	| Backbone architecture                   | `ssm`  |

#### **[model.components]**

| Key                         | Description                                               | Choices / Examples                |
| --------------------------- | --------------------------------------------------------- | --------------------------------- |
| **`[model.loss]`**          | Defines the training loss function used for optimization. | `loss`: "mse", "nll", "pinball"   |
| **`[model.output_head]`**   | Specifies the output layer or distribution type.          | `arch`: "gmm", "quantile"         |
| **`[model.patch_encoder]`** | Encodes temporal patches of input data before modeling.   | `arch`: "linear", "gemma", "ss"   |
| **`[model.patch_decoder]`** | Decodes or reconstructs temporal output patches.          | `arch`: "linear", "none"          |

The choice of `model.loss.loss` must match `model.output_head.arch`:
- `arch = "gmm"` → use `loss = "nll"` (likelihood over Gaussian mixture).
- `arch = "quantile"` → use `loss = "pinball"` (quantile regression).

### **[optimizer]**

Standard optimizer configuration.

| Key                 | Description                 | Value  |
| ------------------- | --------------------------- | ------ |
| `learning_rate`     | Base learning rate          | `3e-5` |
| `weight_decay`      | Optimizer weight decay      | `0`    |
| `gradient_clipping` | Gradient clipping threshold | `1`    |

#### **[benchmarking]**

Defines evaluation targets and thresholds for monitoring. Split up into local, stef and gift benchmarks.

| Key              | Description                                                                                           | Default |
| ---------------- | ----------------------------------------------------------------------------------------------------- | ------- |
| `eval_quantiles` | Quantile levels reported by the benchmarks.                                                           | `[0.01, 0.05, 0.1, ..., 0.95, 0.99]` |
| `batch_size`     | Series per forward pass in every benchmark. Benchmarks run without gradients, so this can be far larger than `training.batch_size`; it only trades speed for memory. | `64` |

#### **benchmarking.benchmarks.localbenchmark**

| Key                   | Description                                        | Example            |
| --------------------- | -------------------------------------------------- | ------------------ |
| `locations`           | List of benchmark sites or time series.            | `["Ameland"]`      |
| `thresholds`          | Corresponding performance thresholds per location. | `[-4, -15.9, ...]` |
| `n_day_ahead`         | Forecast horizon for benchmarking.                 | `1`                |
| `context_window_days` | Context window size in days.                       | `32`               |
| `predict_window_days` | Prediction window size in days.                    | `2`                |

#### **benchmarking.benchmarks.stefbeambenchmark**

| Key                             | Description                             | Example                        |
| ------------------------------- | --------------------------------------- | ------------------------------ |
| `targets_file`                  | List of targets for stef beam.          | `liander2024_targets.yaml`     |
| `context_window_days`           | Context window size in days.            | `32`                           |
| `predict_window_days`           | Prediction window size in days.         | `2`                            |
| `input_sample_interval_minutes` | Benchmark input sample rate (minutes).  | `15`                           |

#### **benchmarking.benchmarks.FevBenchBenchmark**

Zero-shot evaluation on [fev-bench](https://arxiv.org/abs/2509.26468) tasks through the `fev` library. The
task definitions and the published Seasonal Naive and Chronos-2 results are read from the
[fev repository](https://github.com/autogluon/fev) at a pinned release, and by default only the 26
Energy-domain tasks run; every run is ranked against the baselines (skill score and win rate with bootstrap
confidence intervals). See [EVALUATION.md](EVALUATION.md#fev-bench) for how series are presented to the model. A model with its own fev integration (Chronos-2) is always scored through it rather than
through the s4casting adapter, so its numbers follow the published recipe.

| Key                    | Description                                                                                   | Default |
| ---------------------- | --------------------------------------------------------------------------------------------- | ------- |
| `tasks_file`           | fev benchmark YAML (`tasks:` list), path or URL.                                              | upstream `tasks.yaml` at the pinned fev release |
| `task_subset`          | `"energy"` (the 26 Energy-domain tasks), `"all"`, or a list of `dataset_config` names.        | `"energy"` |
| `num_windows`          | Override the rolling windows per task. Changes the task identity, so the leaderboard is skipped. | unset |
| `context_window_days`  | Cap on the history given to the model. Unset uses the longest window trained at the task's rate; shorter series are used at their own length, never padded. | unset |
| `model_name`           | Name recorded in the fev summaries.                                                           | `s4casting` |
| `datasets_prefix`      | Local/S3 mirror of `autogluon/fev_datasets` (`<prefix>/<dataset_config>/*.parquet`); unset streams from the Hugging Face Hub. | unset |
| `baseline_results`     | fev result CSVs (paths or URLs) to rank against.                                              | upstream Seasonal Naive and Chronos-2 results |
| `baseline_model`       | `model_name` in the baselines that anchors the skill score.                                   | `Seasonal Naive` |
| `trained_on_datasets`  | `dataset_config` names that overlap the training corpus; flagged as leakage.                  | `[]` |
| `output_dir`           | Subdirectory of `io.output` for the summary CSVs (`iter_<n>.csv`).                            | `fev-bench` |

Results are logged as `fev_bench/<task>/<metric>` (SQL, MASE, WQL, WAPE), `fev_bench/skill_score` and
`fev_bench/win_rate` to W&B and MLflow, and can be produced for a checkpoint without training:

```bash
uv run python scripts/run_fevbench.py configs/transformer.toml --checkpoint out/checkpoint_10000.pt
```

#### **[io]**

Configures data input and output.

| Key             | Description                                   | Example                                   |
| --------------- | --------------------------------------------- | ----------------------------------------- |
| `feature_order` | Features that are used for training.          | `["measurements_cdb", "weather", "time"]` |
| `output`        | Location to save outputs.                     | `"out/"`                                  |

Each feature named in `feature_order` is configured in its own
`[io.features.<name>]` section with at least a `location` (path to the dataset)
and a `loader` (e.g. `"croissant"`, `"parquet"`, `"sqlite"`, `"time"`); see
`configs/cpu.toml` for a complete example.

The `time` feature (`loader = "time"`) holds the metadata channels of every sample. By default it is the unix
time alone, from which the models derive time-of-day, day-of-week and day-of-year embeddings; with
`include_location = true` the location's latitude and longitude follow it and the models add a spatial
embedding. Its channel count follows from the flag, so `n_features` is not needed for it.
Time-only is the default: a model trained without location transfers to series that carry no coordinates,
such as the external datasets and benchmarks. A checkpoint trained with location loads into a time-only model
with a warning (its spatial embedding is dropped).

#### **[authentication]**

Credentials and external service tokens (optional). Provide only what you need.

| Key                    | Description                         | Example                |
| ---------------------- | ----------------------------------- | ---------------------- |
| `wandb_api_key`        | API key to enable Weights & Biases logging. | `"...your key..."`     |

W&B online mode requires `wandb_api_key`. Configure
`logging.wandb.mode = "offline"` when no API key is available.
