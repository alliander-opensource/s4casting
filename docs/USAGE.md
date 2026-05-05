<!--
SPDX-FileCopyrightText: Contributors to the s4casting project

SPDX-License-Identifier: MPL-2.0
-->

# USAGE

After installation, you're ready to run the code.

## Configuration Files

- Default configurations are provided in `.toml` format.
    - **`cpu.toml`** – for running on CPU only  
    - **`cuda.toml`** – for single-GPU CUDA execution  
    - **`cuda-ddp.toml`** – for multi-GPU distributed training using DDP (Distributed Data Parallel)  
    - **`cuda_medium_term.toml`** – for experiments for 1 year into the future.   
    - **`inference.toml`** – for running inference pipelines  
    - **`mps.toml`** – for Apple Silicon (M1/M2) using Metal Performance Shaders (MPS)  
    - **`transformer.toml`** – for transformer-specific model configurations
- Files placed in `configs/local/` are **ignored by Git**, making it a safe place for personal or machine-specific configs.

## Example usages

For **Distributed Data Parallel (DDP)** training, use `torchrun` to spawn multiple processes.  
For non-DDP usage, you can replace `torchrun` with `python`.

#### Inference
```bash
# Inference on cpu
uv run python3 scripts/inference.py --config-path configs/cuda-tiny.toml --data data/ameland.csv --target-col load --checkpoint data/checkpoint_tiny_99k.pt --show-plots
```

#### Training
```bash
# Train on CPU
uv run torchrun --standalone scripts/train.py configs/cpu.toml

# Train on CUDA (single GPU)
uv run torchrun --standalone scripts/train.py configs/cuda.toml

# Train on CUDA with DDP (multi-GPU)
uv run torchrun --standalone --nproc_per_node=4 scripts/train.py configs/cuda-ddp.toml

# overriding parameters on CLI
uv run torchrun --standalone scripts/train.py configs/cpu.toml --run.persist_to_wandb_project=main_project
uv run python scripts/train.py configs/cpu.toml --run.persist_to_wandb_project=main_project

# overriding parameters via env variables
export S4_run__persist_to_wandb_project="main_project"
uv run torchrun --standalone scripts/train.py configs/cpu.toml
```

#### Creating datasets in our format
The process of creating datasets in our format is handled by the `scripts/format_dataset.py` script. The process is fully described in `notebooks/00_data_preparation.ipynb`. The data used in the following examples are created by running a test once. Important to note is that all our data is measured for the UTC timezone.

```bash
uv run python3 scripts/format_dataset.py   --folder data/tests/sinusoid_data_raw  --output_dir data/example/output_test/   --target_col measurements   --time_col timestamp 

# To include location metadata (longitude and latitude), add the locations_file argument
uv run python3 scripts/format_dataset.py   --folder data/tests/sinusoid_data_raw --output_dir data/example/output_test/   --target_col measurements   --time_col timestamp --locations_file data/tests/sinusoid_locations/locations.csv
```

To incorporate weather data, it must first be downloaded manually from Open-Meteo using `scripts/download_weather_data.py`. Once downloaded, it can be referenced in your .toml configuration.

```bash
# example command to download weather data from open-meteo
uv run python3 scripts/download_weather_data.py --coords_csv data/tests/sinusoid_locations/locations.csv --start_date 2023-01-01 --end_date 2023-06-01 --output_dir data/example/weather_output
```

If you want more API calls than the default, put your Open-Meteo API key in a local .env file, see `.env.example` as reference.

## Track training with Weights & Biases (W&B)

Enable logging to W&B by adding this to your .toml:

```toml
[run]
persist_to_wandb_project = "forecasting-s4"  
wandb_notes = "<SOME NOTES ABOUT THE RUN>"    

[authentication]
wandb_api_key = "<API-KEY-FROM-WANDB>"       
```

Notes:
- W&B collects metrics, system stats, and artifacts so you can view runs on wandb.ai.
- The API key authenticates your session, store it securely (e.g., in configs/local/*.toml or env vars).
- If these fields are not set, W&B logging stays disabled and the logging is done locally.

## Running in CodeEditor (VSCode on SageMaker)

> When using the CodeEditor application in SageMaker, always start your run inside a `screen` session. The terminal may freeze after a few hours of inactivity, and `screen` helps you resume your work safely.

### Screen Commands

- **Install screen** (required after each reboot):
  ```bash
  sudo apt-get install screen
  ```
- **Start a screen session**:
  ```bash
  screen
  ```
- **Detach from a session**:  
  Press `CTRL+A`, then `D`
- **Reconnect to a session**:
  ```bash
  screen -r
  ```
- **List active sessions**:
  ```bash
  screen -ls
  ```

