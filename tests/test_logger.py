# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import logging
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import URLError
from urllib.request import urlopen

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.exceptions import RESOURCE_DOES_NOT_EXIST, MlflowException
from plotly.graph_objects import Figure

from s4casting.core.config import (
    MLflowLoggingConfiguration,
    WandbLoggingConfiguration,
)
from s4casting.core.context import Context
from s4casting.core.hooks import TrainingHooks
from s4casting.core.logger import MLflowLogger
from s4casting.factories.logger import provide_loggers
from tests.utils import load_config as load_test_config


@pytest.fixture(scope="module")
def workspace_mlflow_server(tmp_path_factory):
    """Run a local SQL-backed MLflow server with workspace support.

    Yields:
        Tracking URI of the running server.
    """
    server_root = tmp_path_factory.mktemp("mlflow-server")
    artifact_root = server_root / "artifacts"
    artifact_root.mkdir()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    uri = f"http://127.0.0.1:{port}"
    server_log = (server_root / "server.log").open("w+")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow",
            "server",
            "--backend-store-uri",
            f"sqlite:///{server_root / 'mlflow.db'}",
            "--default-artifact-root",
            artifact_root.as_uri(),
            "--enable-workspaces",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--workers",
            "1",
        ],
        stdout=server_log,
        stderr=subprocess.STDOUT,
    )

    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and process.poll() is None:
            try:
                with urlopen(f"{uri}/health", timeout=1) as response:
                    if response.status == 200:
                        break
            except (TimeoutError, URLError):
                time.sleep(0.2)
        else:
            server_log.flush()
            server_log.seek(0)
            pytest.fail(f"MLflow server did not start:\n{server_log.read()}")

        yield uri
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        server_log.close()


def test_logger_factory_constructs_local_loggers(caplog):
    """The factory always constructs standard output and CSV loggers."""
    config = load_test_config()

    with (
        caplog.at_level(logging.INFO),
        patch("s4casting.factories.logger.StdLogger", return_value="std") as std_logger,
        patch("s4casting.factories.logger.CSVLogger", return_value="csv") as csv_logger,
        patch("s4casting.factories.logger.WandbLogger") as wandb_logger,
        patch("s4casting.factories.logger.MLflowLogger") as mlflow_logger,
    ):
        loggers = provide_loggers(
            config.logging,
            config.run,
            config.training,
            config.authentication,
            config.io.output,
            TrainingHooks(),
        )

    assert loggers == ["std", "csv"]
    std_logger.assert_called_once()
    csv_logger.assert_called_once()
    wandb_logger.assert_not_called()
    mlflow_logger.assert_not_called()
    assert "Initialized logging providers: std, csv" in caplog.text


def test_logger_factory_constructs_configured_remote_loggers(caplog):
    """The factory constructs each configured remote logger once."""
    config = load_test_config()
    config.logging.wandb = WandbLoggingConfiguration(project="test")
    config.logging.mlflow = MLflowLoggingConfiguration()

    with (
        caplog.at_level(logging.INFO),
        patch("s4casting.factories.logger.StdLogger", return_value="std") as std_logger,
        patch("s4casting.factories.logger.CSVLogger", return_value="csv") as csv_logger,
        patch("s4casting.factories.logger.WandbLogger", return_value="wandb") as wandb_logger,
        patch("s4casting.factories.logger.MLflowLogger", return_value="mlflow") as mlflow_logger,
    ):
        loggers = provide_loggers(
            config.logging,
            config.run,
            config.training,
            config.authentication,
            config.io.output,
            TrainingHooks(),
        )

    assert loggers == ["std", "csv", "wandb", "mlflow"]
    std_logger.assert_called_once()
    csv_logger.assert_called_once()
    wandb_logger.assert_called_once()
    mlflow_logger.assert_called_once()
    assert "Initialized logging providers: std, csv, wandb, mlflow" in caplog.text


def test_context_prepares_complete_tracking_parameters():
    """Runtime dataset metadata is included before loggers receive the context."""
    config = load_test_config()
    config.io.hash_datasets = True
    batcher = SimpleNamespace(datasets_per_source={"measurements": object(), "weather": object()})

    with patch("s4casting.core.context.hash_all_memmaps", side_effect=["measurements", "weather"]):
        context = Context(
            configuration=config,
            model_container=Mock(),
            optimizer=Mock(),
            scheduler=None,
            machine=Mock(),
            batcher=batcher,
        )

    assert "authentication" not in context.tracking_parameters
    assert context.tracking_parameters["dataset_hash_measurements"] == "measurements"
    assert context.tracking_parameters["dataset_hash_weather"] == "weather"


def test_mlflow_preflight_creates_missing_workspace():
    """MLflow preflight creates the configured workspace when absent."""
    workspace = "some_nonexistent_workspace"
    config = MLflowLoggingConfiguration(workspace=workspace)
    client = Mock()
    client.get_workspace.side_effect = MlflowException(
        "missing",
        error_code=RESOURCE_DOES_NOT_EXIST,
    )

    with (
        patch("s4casting.core.logger.MlflowClient", return_value=client),
        patch("s4casting.core.logger.mlflow.set_tracking_uri"),
        patch("s4casting.core.logger.mlflow.set_workspace"),
        patch("s4casting.core.logger.mlflow.set_experiment"),
    ):
        MLflowLogger.preflight(config)

    client.create_workspace.assert_called_once_with(
        workspace,
    )


def test_mlflow_preflight_propagates_workspace_errors(monkeypatch: pytest.MonkeyPatch):
    """Unexpected MLflow workspace errors retain their original details."""
    client = Mock()
    error = MlflowException("workspace endpoint unavailable")
    client.get_workspace.side_effect = error
    monkeypatch.setattr("s4casting.core.logger.MlflowClient", Mock(return_value=client))
    monkeypatch.setattr("s4casting.core.logger.mlflow.set_tracking_uri", Mock())

    with pytest.raises(MlflowException, match="workspace endpoint unavailable") as raised:
        MLflowLogger.preflight(MLflowLoggingConfiguration())

    assert raised.value is error


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        ("benchmark/type?", "benchmark/type_"),
        ("benchmark//n_samples", "benchmark/_/n_samples"),
        ("/benchmark/", "_/benchmark/_"),
        ("benchmark/./value", "benchmark/_/value"),
        ("benchmark/../value", "benchmark/_/value"),
    ],
)
def test_mlflow_metric_name_sanitization(original, expected):
    """MLflow metric names are valid as characters and normalized paths."""
    assert MLflowLogger._sanitize_metric_name(original) == expected


def test_mlflow_metric_name_length_and_collisions():
    """Metric names remain portable and never silently overwrite a peer."""
    original = "metric/" + ("x" * 300)
    sanitized = MLflowLogger._sanitize_metric_name(original)
    assert len(sanitized) == 250

    logger = object.__new__(MLflowLogger)
    logger.metric_names = {}
    logger._prepare_metrics({"metric?": 1})
    with pytest.raises(ValueError, match="both resolve"):
        logger._prepare_metrics({"metric!": 2})


def test_mlflow_adapter_passes_metric_values_to_sdk_unchanged():
    """The adapter lets MLflow apply its supported scalar conversions."""
    logger = object.__new__(MLflowLogger)
    logger.run_id = "run-id"
    logger.metric_names = {}
    values = {
        "boolean": True,
        "numeric_string": "2.5",
        "numpy_scalar": np.float32(3.5),
        "numpy_array": np.array([4.5]),
        "not_finite": np.inf,
    }

    with patch("s4casting.core.logger.mlflow.log_metrics") as log_metrics:
        logger._log_metrics(values, 7)

    passed_values = log_metrics.call_args.args[0]
    assert all(passed_values[key] is value for key, value in values.items())
    assert log_metrics.call_args.kwargs == {"step": 7, "synchronous": True}


def test_mlflow_quantile_table_uses_one_stable_path():
    """Repeated benchmark steps append rows to one MLflow table artifact."""
    logger = object.__new__(MLflowLogger)
    logger._log_metrics = Mock()
    context = SimpleNamespace(
        benchmark_location="location?",
        benchmark_metrics={
            "mae": {0.1: np.float32(1.0), 0.9: np.float32(2.0)},
            "loss": 3.0,
        },
    )

    with patch("s4casting.core.logger.mlflow.log_table") as log_table:
        logger.report_benchmark(context, 10)
        logger.report_benchmark(context, 20)

    assert log_table.call_args_list[0].kwargs["artifact_file"] == "benchmark_quantile_scores.json"
    assert log_table.call_args_list[1].kwargs["artifact_file"] == "benchmark_quantile_scores.json"
    first_table = log_table.call_args_list[0].args[0]
    second_table = log_table.call_args_list[1].args[0]
    pd.testing.assert_series_equal(first_table["step"], pd.Series([10, 10], name="step"))
    pd.testing.assert_series_equal(second_table["step"], pd.Series([20, 20], name="step"))
    assert first_table["location"].tolist() == ["location?", "location?"]


def test_mlflow_figure_logs_html_and_png():
    """Plotly figures are logged as an HTML artifact and a time-stepped image."""
    logger = object.__new__(MLflowLogger)
    figure = Mock()

    with (
        patch("s4casting.core.logger.mlflow.log_figure") as log_figure,
        patch("s4casting.core.logger.mlflow.Image", return_value="image") as image,
        patch("s4casting.core.logger.mlflow.log_image") as log_image,
    ):
        logger._log_figure("benchmark/location?", figure, 12)

    log_figure.assert_called_once_with(figure, "steps/step-00000012/figures/benchmark_location_.html")
    image.assert_called_once()
    image_path = Path(image.call_args.args[0])
    assert image_path.name == "benchmark_location_.png"
    figure.write_image.assert_called_once_with(str(image_path))
    log_image.assert_called_once_with(
        "image",
        key="figures/benchmark_location_",
        step=12,
        synchronous=True,
    )


def test_mlflow_sql_server_workspace_and_run_lifecycle(workspace_mlflow_server, tmp_path):
    """Exercise MLflow workspaces, runs, metrics, tables, and artifacts over HTTP."""
    config = load_test_config()
    mlflow_config = MLflowLoggingConfiguration.model_construct(
        uri=workspace_mlflow_server,
        workspace="s4casting",
        experiment="dev",
    )
    config.logging.mlflow = mlflow_config

    logger = MLflowLogger(TrainingHooks(), mlflow_config, config.run, str(tmp_path))
    MLflowLogger.preflight(mlflow_config)
    context = SimpleNamespace(
        configuration=config,
        tracking_parameters={
            **config.model_dump(mode="json", exclude={"authentication"}),
            "dataset_hash_measurements": "measurements-hash",
            "dataset_hash_weather": "weather-hash",
        },
        benchmark_location="integration",
        benchmark_metrics={"mae": {0.1: 1.0, 0.9: 2.0}},
        eval_metrics={},
        input_validation_sample_rate=15,
        output_validation_sample_rate=60,
        validation_loss=0.5,
    )

    logger.start(context)
    finished_run_id = logger.run_id
    logger._log_metrics({"training_loss": 1.25}, 2)
    logger.report_benchmark(context, 2)
    logger.report_benchmark(context, 3)
    logger._log_figure("forecast", Figure().add_scatter(x=[0, 1], y=[0, 1]), 2)
    html_path = tmp_path / "panel.html"
    html_path.write_text("<html><body>panel</body></html>")
    logger.log_html_panel("summary", html_path, 2)
    logger.finished(context)

    assert finished_run_id is not None
    client = mlflow.MlflowClient(tracking_uri=workspace_mlflow_server)
    assert client.get_workspace("s4casting").name == "s4casting"
    assert client.get_experiment_by_name("dev") is not None

    finished_run = client.get_run(finished_run_id)
    assert finished_run.info.status == "FINISHED"
    assert finished_run.info.run_name
    assert finished_run.data.params["dataset_hash_measurements"] == "measurements-hash"
    assert finished_run.data.metrics["training_loss"] == 1.25

    table = mlflow.load_table("benchmark_quantile_scores.json", run_ids=[finished_run_id])
    assert table["step"].tolist() == [2, 2, 3, 3]
    figure_artifacts = client.list_artifacts(finished_run_id, "steps/step-00000002/figures")
    assert [artifact.path for artifact in figure_artifacts] == ["steps/step-00000002/figures/forecast.html"]
    html_artifact = Path(client.download_artifacts(finished_run_id, "steps/step-00000002/figures/forecast.html"))
    assert "cdn.plot.ly" in html_artifact.read_text(encoding="utf-8")
    image_artifacts = client.list_artifacts(finished_run_id, "images")
    image_paths = {artifact.path for artifact in image_artifacts}
    assert any(
        path.startswith("images/figures~forecast+step+2+timestamp+") and path.endswith(".png") for path in image_paths
    )
    assert any(
        path.startswith("images/figures~forecast+step+2+timestamp+") and path.endswith(".webp") for path in image_paths
    )
    panel_artifacts = client.list_artifacts(finished_run_id, "steps/step-00000002/html/summary")
    assert [artifact.path for artifact in panel_artifacts] == ["steps/step-00000002/html/summary/panel.html"]
