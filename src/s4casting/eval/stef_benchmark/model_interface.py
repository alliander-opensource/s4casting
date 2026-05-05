# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import torch
from openstef_beam.backtesting.backtest_forecaster import (
    BacktestBatchForecasterMixin,
    BacktestForecasterConfig,
    BacktestForecasterMixin,
)
from openstef_beam.backtesting.restricted_horizon_timeseries import RestrictedHorizonVersionedTimeSeries
from openstef_beam.benchmarking.models import BenchmarkTarget
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.types import Quantile
from torch.utils.data import DataLoader

from s4casting.core.context import Context
from s4casting.core.distributions import gmm_to_quantiles
from s4casting.eval.stef_benchmark.datasets.horizon_window import HorizonWindowDataset


class S4ModelInterface(BacktestForecasterMixin, BacktestBatchForecasterMixin):
    """Model interface for S4 models in StefBeam benchmarking."""

    def __init__(self, context: Context, target: BenchmarkTarget | None = None):
        """Initialize the S4 Model Interface."""
        super().__init__()
        self.context = context
        self.target = target

        if "time" in context.batcher.datasets_per_source:  # ty: ignore[possibly-missing-attribute]
            raise Exception("Cannot run stef beam with time as an input feature")

        # Provide s4 configurations to s4interface
        cfg = context.configuration.model
        predict_window_days = context.configuration.benchmarking.benchmarks["StefBeamBenchmark"].predict_window_days  # type: ignore[possibly-missing-attribute]
        input_window_days = (
            context.configuration.benchmarking.benchmarks["StefBeamBenchmark"].context_window_days - predict_window_days  # type: ignore[possibly-missing-attribute]
        )

        # define quantiles
        self._quantiles = [Quantile(q) for q in context.configuration.benchmarking.eval_quantiles]
        self._config = BacktestForecasterConfig(
            requires_training=False,
            batch_size=context.configuration.training.batch_size,  # type: ignore[call-arg]
            predict_sample_interval=timedelta(
                minutes=context.configuration.benchmarking.benchmarks["StefBeamBenchmark"].input_sample_interval_minutes  # type: ignore[possibly-missing-attribute]
            ),
            predict_length=timedelta(days=predict_window_days),
            predict_min_length=timedelta(days=predict_window_days),
            predict_context_length=timedelta(days=input_window_days + 1),
            predict_context_min_coverage=1.0,
            training_context_length=timedelta(days=0),
            training_context_min_coverage=0.0,
        )
        if cfg.output_head.arch == "quantile" and not set(self._quantiles).issubset(
            set(cfg.output_head.quantile_values)
        ):
            missing = set(self._quantiles) - set(cfg.output_head.quantile_values)
            raise ValueError(
                f"Quantile output head cannot produce eval quantiles {missing}. "
                f"Either add them to output_head.quantile_values or remove from benchmarking.eval_quantiles"
            )

    @property
    def config(self) -> BacktestForecasterConfig:
        """Get the model interface configuration.

        Returns:
            ModelInterfaceConfig: The model interface configuration.
        """
        return self._config

    @property
    def quantiles(self) -> list[Quantile]:
        """Return the list of quantiles that this forecaster predicts."""
        return self._quantiles

    @property
    def batch_size(self) -> int | None:
        """Batch size for prediction. None means process all at once."""
        return self.context.configuration.training.batch_size

    @torch.no_grad()
    def predict_batch(self, batch: list[RestrictedHorizonVersionedTimeSeries]):
        """Predicts a batch of HorizonTransforms.

        Args:
            batch (list[HorizonTransform]): List of HorizonTransform objects.

        Returns:
            list[pd.DataFrame]: List of DataFrames containing predictions for each horizon.
        """
        ds = HorizonWindowDataset(
            horizons=batch,
            cfg=self.context.configuration,
            device=self.context.machine.benchmarking_device,
        )
        loader = DataLoader(
            ds,
            batch_size=self.context.configuration.training.batch_size,
            shuffle=False,
            pin_memory=False,
            collate_fn=lambda b: b,
        )

        all_predictions = []
        ts_list_all = []

        n_predict = (
            self.context.configuration.benchmarking.benchmarks["StefBeamBenchmark"].predict_window_days * 24 * 60
        ) // self.context.configuration.model.base_sample_interval_minutes

        for mini_batch in loader:
            # stack tensors
            X = torch.stack([d["X"] for d in mini_batch], dim=0)
            xm = torch.stack([d["xm"] for d in mini_batch], dim=0)
            ts_list = [d["ts"] for d in mini_batch]  # length == batch
            ts_list_all.extend(ts_list)

            X = X.to(self.context.machine.benchmarking_device, non_blocking=True)
            xm = xm.to(self.context.machine.benchmarking_device, non_blocking=True)

            # forward pass model
            # Note that this assumes that the base rate is 15 minutes.
            pred = self.context.model_container.raw_model(
                X,
                xm,
                input_interval=torch.tensor(
                    [self.context.configuration.model.base_sample_interval_minutes] * X.shape[0], device=X.device
                ),
                output_interval=torch.tensor(
                    [self.context.configuration.model.base_sample_interval_minutes] * X.shape[0], device=X.device
                ),
            )[0]
            all_predictions.append(pred[:, -n_predict:, 0, ...])

        full_pred = torch.cat(all_predictions, dim=0)

        if self.context.configuration.model.output_head.arch == "gmm":
            logpi, sigma, mu = (x for x in full_pred.unbind(dim=-1))  # (B, T, G, 3)
            qs = (
                gmm_to_quantiles(
                    torch.exp(logpi),
                    sigma,
                    mu,
                    [float(q) for q in self._quantiles],
                )
                .detach()
                .cpu()
                .numpy()
            )
        elif self.context.configuration.model.output_head.arch == "quantile":
            # Find which quantile from our model output lines up with the default stef quantiles
            model_quantiles = self.context.configuration.model.output_head.quantile_values
            indexes = [model_quantiles.index(float(q)) for q in self._quantiles]
            qs = full_pred.detach().cpu().numpy()[:, :, indexes]

        results = []
        sample_interval = timedelta(minutes=self.context.configuration.model.base_sample_interval_minutes)
        for b in range(qs.shape[0]):
            data_dict = {q.format(): qs[b, :, i] for i, q in enumerate(self._quantiles)}

            df = pd.DataFrame(
                data_dict,
                index=pd.to_datetime(ts_list_all[b], utc=True, unit="ns"),
            )

            results.append(
                TimeSeriesDataset(
                    data=df,
                    sample_interval=sample_interval,
                )
            )

        return results
