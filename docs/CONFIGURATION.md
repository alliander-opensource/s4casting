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
| `task`               			| Training task 	                        | `prediction` 	|

#### **[model]**

Core model architecture and input/output settings.

| Key                              	| Description                             	| Value  |
| -------------------------------- 	| --------------------------------------- 	| ------ |
| `context_window`               	| Context window size in days (list for multi-window inputs) | `[32]`   |
| `predict_width`             		| Forecast horizon                        	| `2`    |
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

#### **[authentication]**

Credentials and external service tokens (optional). Provide only what you need.

| Key                    | Description                         | Example                |
| ---------------------- | ----------------------------------- | ---------------------- |
| `wandb_api_key`        | API key to enable Weights & Biases logging. | `"...your key..."`     |

W&B online mode requires `wandb_api_key`. Configure
`logging.wandb.mode = "offline"` when no API key is available.
