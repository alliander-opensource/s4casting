# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tomlkit
import torch
import yaml

from s4casting import factories as fc
from s4casting.core.config import Configuration
from s4casting.core.context import Context
from s4casting.data.preparation.dataset_formatter import DatasetFormatter
from s4casting.data.preparation.weather import WeatherDatasetFormatter


def requires_cuda(func):
    """Decorator to skip tests if CUDA is not available.

    Args:
        func: The test function to decorate.

    Returns:
        The decorated test function.
    """
    return pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")(func)


def reduce_targets_yaml_to_single_location() -> None:
    """Reduce the stef50 targets.yaml to a single location for faster testing."""
    src = pathlib.Path("data/liander2024/liander2024_targets.yaml")
    original = yaml.safe_load(src.read_text(encoding="utf-8"))
    backup = src.with_name(src.stem + "_one_location.yaml")
    backup.write_text(yaml.safe_dump([original[0]], sort_keys=False), encoding="utf-8")


def load_config() -> Configuration:
    """Load configuration from a TOML file for testing purposes.

    Returns:
        Configuration: The loaded configuration object.
    """
    with pathlib.Path("tests/assets/test.toml").open("r", encoding="utf-8") as f:
        data = tomlkit.load(f)

    return Configuration(**data.unwrap())


def create_sinusoid_dataframe(n: int, interval_min: int, start_time: pd.Timestamp) -> pd.DataFrame:
    """Create a DataFrame containing a sinusoidal signal.

    Args:
        n: Number of samples.
        interval_min: Sampling interval in minutes.
        start_time: Start time for the data.

    Returns:
        pd.DataFrame: DataFrame with timestamp and measurements columns.
    """
    steps_per_day = 24 * 60 // interval_min
    period_steps = 4 * steps_per_day
    t = np.arange(n)
    signal = 10.0 * np.sin(2 * np.pi * t / period_steps)
    index = pd.date_range(start_time, periods=n, freq=f"{interval_min}min")
    return pd.DataFrame({"timestamp": index, "measurements": signal})


def create_sinusoid_test_data() -> None:
    """Create sinusoid test data files."""
    output_dir_name = "/tmp/tests/output_test"
    output_dir = Path(output_dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    DatasetFormatter(
        folder="data/tests/sinusoid_data_raw",
        output_prefix="external_data_wrapped",
        output_dir=output_dir_name,
        target_col="measurements",
        time_col="timestamp",
        sample_interval_minutes=5,
        locations_file="data/tests/sinusoid_locations/locations.csv",
    ).run()


def create_sinusoid_test_data_raw(
    locations_folder_raw: str = "data/tests/sinusoid_data_raw",
    n_spans: int = 100,
    span_days: int = 14,
    interval_min: int = 5,
) -> None:
    """Create raw sinusoid test data files.

    Args:
        locations_folder_raw: Folder to save the raw data files.
        n_spans: Number of spans to create.
        span_days: Number of days per span.
        interval_min: Sampling interval in minutes.
    """
    # Parameters
    samples_per_day = 24 * 60 // interval_min
    N = span_days * samples_per_day
    n = n_spans * N

    output_root = Path(locations_folder_raw)
    output_root.mkdir(parents=True, exist_ok=True)
    create_sinusoid_dataframe(n=n, interval_min=interval_min, start_time=pd.Timestamp("2023-01-01")).to_parquet(
        f"{locations_folder_raw}/location_a.parquet"
    )
    create_sinusoid_dataframe(n=n, interval_min=interval_min, start_time=pd.Timestamp("2023-01-01")).to_parquet(
        f"{locations_folder_raw}/location_b.parquet"
    )
    create_sinusoid_dataframe(n=N, interval_min=interval_min, start_time=pd.Timestamp("2023-01-01")).to_parquet(
        f"{locations_folder_raw}/location_c.parquet"
    )


def create_locations_file(locations_folder: str = "data/tests/sinusoid_locations") -> None:
    """Create a locations CSV file for testing purposes.

    Args:
        locations_folder: Folder containing the locations CSV file.
    """
    df_locations = pd.DataFrame({
        "name": ["location_a", "location_b", "location_c"],
        "lon": [5.842, 5.852, 5.862],
        "lat": [51.645, 51.655, 51.675],
    })
    pathlib.Path(locations_folder).mkdir(parents=True, exist_ok=True)
    df_locations.to_csv(f"{locations_folder}/locations.csv", index=False)


def create_weather_data(locations_folder: str = "data/tests/sinusoid_locations") -> None:
    """Create a locations CSV file for testing purposes.

    Args:
        locations_folder: Folder containing the locations CSV file.
    """
    WeatherDatasetFormatter(
        df_locations=f"{locations_folder}/locations.csv",
        start_date="2023-01-01",
        end_date="2023-06-01",
        output_prefix="weather",
        output_dir="/tmp/tests/weather_output",
    ).run()


def create_tmp_model_checkpoint(
    config: Configuration,
) -> None:
    """Create a temporary model checkpoint with optional head/loss overrides.

    Args:
        config: Configuration object to use for creating the checkpoint.
    """
    machine = fc.provide_machine(config.machine, config.run.seed)
    model_container = fc.provide_model_container(config.model, config.io, machine)
    optimizer = fc.provide_optimizer(config.optimizer, model_container.raw_model.parameters())
    scheduler = fc.provide_scheduler(config.scheduler, optimizer)
    trainer = fc.provide_trainer(config=config.training, optimizer=config.optimizer, machine=machine)
    checkpointer = fc.provide_checkpointer(config.io, trainer.hooks)
    evaluator_head = fc.provide_evaluator_head(config.model, trainer.hooks)
    evaluator = fc.provide_evaluation(trainer.hooks, evaluator_head)
    benchmarker = fc.provide_benchmark(config.benchmarking, trainer.hooks, evaluator_head)

    context = Context(
        configuration=config,
        model_container=model_container,
        optimizer=optimizer,
        scheduler=scheduler,
        machine=machine,
        trainer=trainer,
        checkpointer=checkpointer,
        evaluator=evaluator,
        benchmarker=benchmarker,
        batcher=None,
    )
    suffix = f"tmp_{config.model.output_head.arch}_{config.model.loss.loss}"
    checkpointer.save(context, iteration=suffix)
