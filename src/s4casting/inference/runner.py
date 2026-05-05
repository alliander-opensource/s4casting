# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import pathlib
import pickle
import typing

import pandas as pd
import plotly.graph_objects as go
import tomlkit
import torch

from s4casting import factories as fc
from s4casting.core.checkpoint import Checkpointer
from s4casting.core.config import Configuration
from s4casting.core.context import Context
from s4casting.core.distributions import gmm_to_quantiles
from s4casting.core.hooks import CommonHooks
from s4casting.visualisation.quantile_plotter import plot_inference_quantiles


class DataFrameInferenceRunner:
    """Run most basic model inference on a Pandas DataFrame."""

    def __init__(
        self,
        config: Configuration,
        target_col: str,
        checkpoint_path: str | None = None,
        save_path: str | None = None,
    ) -> None:
        """Initialize DataFrameInferenceRunner."""
        self.config = config
        self._init_components()
        self.device = config.machine.device_kind
        self.model = self.context.model_container.raw_model.to(self.device)
        self.model.eval()
        self.save_path = save_path or "out/df_inference_results.pkl"
        self.input_interval_min = self.config.model.base_sample_interval_minutes
        # Here the 0th context window is used as this inference script is only called with the
        # test.toml config, with is a list with only 1 entry
        self.context_steps_cfg = int(
            ((self.config.model.context_window[0] - self.config.model.predict_width) * 24 * 60)
            / self.input_interval_min
        )
        self.predict_steps_cfg = int(self.config.model.predict_width * 24 * 60 / self.input_interval_min)
        self.param_dtype = next(self.model.parameters()).dtype
        self.x = None
        self.xm = None
        self.predictions = None
        self.expected_feature_order = [target_col]  # could be extended in future with weather data
        self.results: dict[str, typing.Any] = {}
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        print(  # noqa: T201
            f"[Runner] context={self.context_steps_cfg} \n"
            f"predict={self.predict_steps_cfg} \n"
            f"interval={self.input_interval_min}min"
        )

    def _init_components(self) -> None:
        """Initialize machine, model container, optimizer, and wrap into a Context."""
        self.machine = fc.provide_machine(self.config.machine, rng_base_seed=self.config.run.seed)
        self.model_container = fc.provide_model_container(self.config.model, self.config.io, self.machine)
        optimizer = fc.provide_optimizer(self.config.optimizer, self.model_container.raw_model.parameters())
        # Minimal context; scheduler & batcher not needed for inference
        self.context = Context(
            configuration=self.config,
            model_container=self.model_container,
            optimizer=optimizer,
            scheduler=None,
            machine=self.machine,
            batcher=None,
        )

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load checkpoint via Checkpointer without relying on hook triggers.

        Handles missing path gracefully.

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        p = pathlib.Path(checkpoint_path)
        if not p.is_file():
            raise AssertionError(f"Checkpoint not found at {checkpoint_path}")
        hooks = CommonHooks()  # dummy hook container
        cp = Checkpointer(hookable=hooks, load=str(p))
        cp.load(self.context)
        print(f"[Runner] Loaded weights from {p}")  # noqa: T201

    def _tensorize(self, df: pd.DataFrame) -> torch.Tensor:
        """Convert DataFrame to 3D tensor (B, T, F); batch size fixed to 1.

        Args:
            df: Input DataFrame.

        Returns:
            Tensor of shape (1, T, F).
        """
        values = torch.tensor(df.to_numpy(), dtype=torch.float32)
        if values.ndim == 2:
            values = values.unsqueeze(0)
        return values.to(self.device).to(self.param_dtype)

    def _build_mask(self, x: torch.Tensor, context_steps: int, target_idx: int = 0) -> torch.Tensor:
        """Build mask tensor; zeros out target feature during forecast horizon.

        Args:
            x: Input tensor (B, T, F).
            context_steps: Number of context time steps.
            target_idx: Index of target feature.

        Returns:
            Mask tensor (B, T, F) of dtype int.
        """
        base_mask = ~torch.isnan(x)
        base_mask[:, context_steps:, target_idx] = False
        return base_mask.int()

    def forward(self, x: torch.Tensor, xm: torch.Tensor) -> torch.Tensor:
        """Run model forward pass returning raw output tensor.

        Args:
            x: Input tensor (B, T, F).
            xm: Mask tensor (B, T, F).

        Returns:
            Output tensor (B, T, F, C, P).
        """
        with torch.no_grad():
            out, _ = self.model(
                x,
                xm,
                self.config.model.input_sample_intervals_minutes[0],
                self.config.model.output_sample_intervals_minutes[0],
            )
        return out

    def prep_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for inference.

        This is done by ensuring correct columns and extending dataframe with forecast horizon.

        Args:
            df: Input DataFrame to prepare.

        Returns:
            Prepared DataFrame with context and forecast horizon.
        """
        df = df.tail(self.context_steps_cfg)
        if isinstance(df.index, pd.DatetimeIndex):
            freq = f"{self.input_interval_min}T"
            start = df.index[-1] + pd.Timedelta(minutes=self.input_interval_min)
            new_index = pd.date_range(start=start, periods=self.predict_steps_cfg, freq=freq)
        else:
            last = df.index[-1]
            new_index = range(last + 1, last + 1 + self.predict_steps_cfg)

        extension = pd.DataFrame(index=new_index, columns=df.columns, dtype="float64")

        if set(df.columns) != set(self.expected_feature_order):
            raise ValueError(
                f"Input dataframe columns do not match expected Model IO.\n"
                f"Expected: {sorted(self.expected_feature_order)}\n"
                f"Received: {sorted(df.columns)}"
            )
        # All NaNs by default; keep them
        return pd.concat([df, extension])

    def inference(self, df: pd.DataFrame):
        """Execute full inference pipeline on DataFrame.

        Args:
            df: Input DataFrame with context + forecast horizon.
        """
        df_predictions = self.prep_df(df.copy())
        x = self._tensorize(df_predictions)
        xm = self._build_mask(x, self.context_steps_cfg)
        if self.config.model.output_head.arch == "gmm":
            self.predictions = self.forward(x, xm)[:, -self.predict_steps_cfg :, 0, :, :]

            (logpi, sigma, mu) = self.predictions.unbind(dim=-1)

            quantile_values = gmm_to_quantiles(
                torch.exp(logpi), sigma, mu, self.context.configuration.model.output_head.quantile_values
            )
        elif self.config.model.output_head.arch == "quantile":
            self.predictions = self.forward(x, xm)[:, -self.predict_steps_cfg :, 0, :]
            quantile_values = self.predictions
        else:
            raise NotImplementedError(
                f"Inference not implemented for architecture: {self.config.model.output_head.arch}"
            )

        self.x = x[:, : -self.predict_steps_cfg, :]
        self.xm = xm[:, : -self.predict_steps_cfg, :]

        self.results = {
            "X": self.x.cpu(),
            "Xm": self.xm.cpu(),
            "predictions": quantile_values.cpu(),
            "forecast_start": df_predictions.iloc[-self.predict_steps_cfg :].index[0],
        }

    def save_predictions_pickle(self) -> None:
        """Save predictions and context to dictionary."""
        with pathlib.Path(self.save_path).open("wb") as f:
            self.results["config"] = self.config
            pickle.dump(self.results, f)

    def save_predictions_parquet(self, path: str, location_name: str) -> None:
        """Save predictions to Parquet file.

        Args:
            path: Path to save the Parquet file.
            location_name: Name of the location for the predictions.
        """
        quantiles = self.config.model.output_head.quantile_values
        path_obj = pathlib.Path(path) / f"predictions_{location_name}.parquet"
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        index = pd.date_range(
            start=self.results["forecast_start"],
            periods=self.predict_steps_cfg,
            freq=f"{self.input_interval_min}min",
        )
        data = {}
        for i, q in enumerate(quantiles):
            data[q] = self.results["predictions"][0, :, i].numpy()
        df_out = pd.DataFrame(data, index=index)

        df_out.to_parquet(str(path_obj))

    def plot_results(self) -> go.Figure:
        """Plot inference results using the Plotter class.

        Returns:
            go.Figure: Plotly figure object containing the inference quantiles plot.
        """
        return plot_inference_quantiles(
            self.x.cpu(), self.results["predictions"], self.config.model.output_head.quantile_values
        )


def load_config(path: str | Configuration) -> Configuration:
    """Load Configuration from TOML file.

    Args:
        path: Path to the TOML configuration file or Configuration object.

    Returns:
        Loaded Configuration object.
    """
    if isinstance(path, Configuration):
        return path
    if not pathlib.Path(path).is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with pathlib.Path(path).open("r", encoding="utf-8") as f:
        data = tomlkit.load(f)
    return Configuration(**data.unwrap())


def load_dataframe(path: str, time_col: str = "time") -> pd.DataFrame:
    """Load DataFrame from CSV or Parquet file, ensuring datetime index if possible.

    Args:
        path: Path to the data file (CSV or Parquet).
        time_col: Name of the time column to set as index if not already.

    Returns:
        Loaded Pandas DataFrame.
    """
    p = pathlib.Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p, index_col=0, parse_dates=True)
    if time_col in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        try:
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.set_index(time_col)
        except Exception:
            pass
    return df


def run_inference(
    config: str | Configuration,
    data_path: str,
    checkpoint_path: str,
    target_col: str = "measurements",
    time_col: str = "time",
    save_path_predictions: str | None = None,
    save_path_pickle: str | None = None,
    plot_path: str | None = None,
    show_plots: bool = False,
) -> None:
    """CLI helper to execute inference and optional saving/plotting.

    Args:
        config: Model configuration object or path to TOML file.
        data_path: Path to input data (parquet or csv).
        checkpoint_path: Path to checkpoint .pt file.
        target_col: Target column name in data.
        time_col: Time column name in data.
        save_path_predictions: Where to store output predictions Parquet (omit to disable saving).
        save_path_pickle: Where to store output pickle (omit to disable saving).
        plot_path: Path to save plot HTML.
        show_plots: Whether to display plots.
    """
    df = load_dataframe(data_path, time_col=time_col)
    config = load_config(config)
    runner = DataFrameInferenceRunner(
        config=config,
        checkpoint_path=checkpoint_path,
        target_col=target_col,
        save_path=save_path_pickle,
    )
    runner.inference(df)

    if save_path_predictions:
        location_name = pathlib.Path(data_path).stem  # e.g., "location_a" from "location_a.parquet"
        runner.save_predictions_parquet(save_path_predictions, location_name=location_name)
        print(f"[OK] Saved predictions -> {save_path_predictions}")  # noqa: T201

    if save_path_pickle:
        runner.save_predictions_pickle()
        print(f"[OK] Saved pickle -> {save_path_pickle}")  # noqa: T201

    fig = runner.plot_results()
    if show_plots:
        fig.show()
        print(f"[OK] Plot saved -> {plot_path}")  # noqa: T201
    if plot_path:
        pathlib.Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(plot_path)
