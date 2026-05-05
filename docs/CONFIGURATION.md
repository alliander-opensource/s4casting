<!--
SPDX-FileCopyrightText: Contributors to the s4casting project

SPDX-License-Identifier: MPL-2.0
-->

### Model Configuration

Model behavior and training setup are fully defined through TOML configuration files (see `configs/*.toml`).
Each section controls a specific component of the pipeline, from data loading to optimization.

Below is an overview of the most relevant sections and what they control. Some default values are provided as examples and can be used as a good starting point.

#### **[machine]**

Defines how and where the model runs.

| Key           | Description                                | Example  |
| ------------- | ------------------------------------------ | -------- |
| `device_kind` | Execution device (`cpu` or `cuda`).        | `cuda`   |
| `ddp`         | Enable/disable distributed training (DDP). | `false`  |


#### **[run]**

Reproducibility and randomness control.

| Key    | Description                      | Example |
| ------ | -------------------------------- | ------- |
| `seed` | Random seed for reproducibility. | `42069` |
| `persist_to_wandb_project` | Name of WandB project to log to (requires `authentication.wandb_api_key`). | `"forecasting-s4"`  |

#### **[training]**

| Key                           | Description                               | Value     	|
| ----------------------------- | ----------------------------------------- | --------- 	|
| `batch_size`                  | Batch size per step                       | `32`      	|
| `evaluation_interval`         | Evaluate every N steps                    | `1000`    	|
| `checkpoint_interval`         | Save checkpoint every N steps             | `1000`    	|
| `benchmarking_interval`       | Run benchmarking every N steps            | `10_000`  	|
| `maximum_steps`               | Total training steps                      | `500_000` 	|
| `task`               			| Training task 	                        | `prediction` 	|

#### **[model]**

Core model architecture and input/output settings.

| Key                              	| Description                             	| Value  |
| -------------------------------- 	| --------------------------------------- 	| ------ |
| `context_window`               	| context window               				| `32`   |
| `predict_width`             		| Forecast horizon                        	| `2`    |
| `base_sample_interval_minutes`   	| Base sampling interval (minutes)        	| `15`    |
| `input_sample_intervals_minutes` 	| Input sampling intervals for multi‑rate training (minutes). Each value must be a multiple of `base_sample_interval_minutes`. | `[15, 60]` |
| `output_sample_interval_minutes` 	| Output sampling intervals for predictions (minutes). Each value must be a multiple of `base_sample_interval_minutes`. | `[15]` |
| `alignment`                      	| Temporal alignment window (minutes)     | `1440` |
| `model`                          	| Backbone architecture                   | `ssm`  |

#### **[model.components]**

| Key                         | Description                                               | Choices / Examples                |
| --------------------------- | --------------------------------------------------------- | --------------------------------- |
| **`[model.loss]`**          | Defines the training loss function used for optimization. | `loss`: "mse", "nll", "pinball"   |
| **`[model.output_head]`**   | Specifies the output layer or distribution type.          | `arch`: "gmm", "quantile"         |
| **`[model.patch_encoder]`** | Encodes temporal patches of input data before modeling.   | `arch`: "linear", "conv", "gemma" |
| **`[model.patch_decoder]`** | Decodes or reconstructs temporal output patches.          | `arch`: "linear", "conv", "none"  |

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

| Key           | Description                                        | Example            |
| ------------- | -------------------------------------------------- | ------------------ |
| `locations`   | List of benchmark sites or time series.            | `["Ameland"]`      |
| `thresholds`  | Corresponding performance thresholds per location. | `[-4, -15.9, ...]` |
| `n_day_ahead` | Forecast horizon for benchmarking.                 | `1`                |

#### **benchmarking.benchmarks.stefbeambenchmark**

| Key           | Description                                        | Example            |
| ------------- | -------------------------------------------------- | ------------------ |
| `targets_file`   | List of targets for stef beam.            | `liander2024_targets.yaml`      |
| `input_window_days`  | Number of input days. | `30` |
| `n_day_ahead` | Forecast horizon for benchmarking.                 | `1`                |

#### **[io]**

Configures data input and output.

| Key             | Description                                   | Example                                   |
| --------------- | --------------------------------------------- | ----------------------------------------- |
| `feature_order` | Features that are used for training .         | `["measurements_cdb", "weather", "time"]` |

#### **[authentication]**

Credentials and external service tokens (optional). Provide only what you need.

| Key                    | Description                         | Example                |
| ---------------------- | ----------------------------------- | ---------------------- |
| `wandb_api_key`        | API key to enable Weights & Biases logging. | `"...your key..."`     |

Without `wandb_api_key` logging falls back to stdout/CSV only.
