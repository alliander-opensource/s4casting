# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import scoringrules as sr
import torch
from sklearn import metrics

from s4casting.core.config import MetricsConfiguration, ModelConfiguration
from s4casting.core.functional import quantile_pool1d


class Metrics:
    """Base class for Metrics."""

    def __init__(
        self,
        Y: torch.Tensor,
        predict_window_days: int,
        model_config: ModelConfiguration,
        metrics_config: MetricsConfiguration,
        input_sample_interval_minutes: int,
        output_sample_interval_minutes: int,
        quantiles: torch.Tensor,
        quantile_values: list,
        sign: str | None,
        loss: float,
        beta: int = 10,
        climits: torch.Tensor | None = None,
        mape_odn_quantile_value: float = 0.01,
        mape_ldn_quantile_value: float = 0.99,
    ):
        """Initialize Metrics class.

        Args:
            quantiles (torch.Tensor): Predicted quantiles tensor.
            quantile_values (torch.Tensor): quantile values.
            Y (torch.Tensor): Ground truth tensor.
            Y (int): prediction window size.
            predict_window_days (int): prediction window size.
            model_config (ModelConfiguration): Model configuration for setting up config
            metrics_config (MetricsConfiguration): Metrics configuration for setting up config
            sign (str): Specifies whether to calculate metrics for "LDN", "ODN", or "BOTH".
            input_sample_interval_minutes (int): Benchmarking sampling rate in the case of multirate models
            output_sample_interval_minutes (int): Benchmarking sampling rate in the case of multirate models
            loss (float): precomputed loss.
            climits (torch.Tensor, optional): Congestion limit per batch.
                Defaults to 80% of the max value if not provided.
            beta (int, optional): Weighting factor between precision and recall for the F-beta score. Defaults to 10.
            mape_ldn_quantile_value (float, optional): which quantile to use for the mape calculation. Defaults 0.99.
            mape_odn_quantile_value (float, optional): which quantile to use for the mape calculation. Defaults 0.01.
        """
        self.Y = Y
        self.model_config = model_config
        self.metrics_config = metrics_config
        self.input_sample_interval_minutes = input_sample_interval_minutes
        self.output_sample_interval_minutes = output_sample_interval_minutes
        self.quantiles = quantiles
        self.quantile_values = quantile_values
        self.sign = sign
        self._loss = loss
        self.beta = beta
        self._B = self.Y.shape[0]
        self._L = self.Y.shape[1]
        self.climits = [climits]

        self.month_kernel_size = (self.model_config.days_per_month * 24 * 60) // self.input_sample_interval_minutes

        # Get the 1st and 99th quantile indexes
        self.odn_index = np.argwhere(
            np.array(self.model_config.output_head.quantile_values) == mape_odn_quantile_value
        )[0][0]
        self.ldn_index = np.argwhere(
            np.array(self.model_config.output_head.quantile_values) == mape_ldn_quantile_value
        )[0][0]
        self.sufficient_days = predict_window_days >= model_config.days_per_month

        # check whether input and outputs rates are the same
        # as CRPS, and peak metrics require that they are
        self.single_rate = self.input_sample_interval_minutes == self.output_sample_interval_minutes

        self.registry: dict[str, Callable[..., Any] | None] = {
            "crps": self.crps if self.single_rate else None,
            "mae": self.mae if self.single_rate else None,
            "precision": self.precision if self.single_rate and climits is not None else None,
            "recall": self.recall if self.single_rate and climits is not None else None,
            "fbeta": self.fbeta if self.single_rate and climits is not None else None,
            "ldn_monthly_mape": self.ldn_monthly_mape if self.sufficient_days else None,
            "odn_monthly_mape": self.odn_monthly_mape if self.sufficient_days else None,
            "loss": self.loss,
        }

    def get_metrics(self) -> dict:
        """Get all valid metrics.

        Returns:
            dict containing metrics.

        """
        cfg = self.metrics_config.model_dump()

        results: dict[str, Any] = {}
        for name, metric in self.registry.items():
            if cfg.get(name, False) and metric is not None:
                results[name] = metric()

        return results

    def loss(self) -> torch.Tensor:
        """Get loss.

        Returns:
            torch.Tensor: Loss across the entire tensor.
        """
        return self._loss

    def ldn_monthly_mape(self) -> torch.Tensor | None:
        """Calculate Monthly MAPE for ODN = .

        Returns:
            torch.Tensor: CRPS across the entire tensor.
        """
        if self.sign in ["LDN", "BOTH"]:
            predicted_ldn = quantile_pool1d(
                self.quantiles[..., self.ldn_index].unsqueeze(dim=0),
                kernel_size=self.model_config.days_per_month,
                stride=self.model_config.days_per_month,
                quantile=1.0,
            )[0, 0, :]
            ground_truth_ldn = quantile_pool1d(
                self.Y.swapaxes(1, 2), kernel_size=self.month_kernel_size, stride=self.month_kernel_size, quantile=1.0
            )[0, 0, :]
            return (torch.sqrt((ground_truth_ldn - predicted_ldn) ** 2) / ground_truth_ldn).mean().item()
        return None

    def odn_monthly_mape(self) -> torch.Tensor | None:
        """Calculate Monthly MAPE for ODN.

        TODO: Merge odn and LDN.
              Make monthly pooling happen elsewhere.

        Returns:
            torch.Tensor: CRPS across the entire tensor.
        """
        if self.sign in ["ODN", "BOTH"]:
            predicted_odn = quantile_pool1d(
                self.quantiles[..., self.odn_index].unsqueeze(dim=0),
                kernel_size=self.model_config.days_per_month,
                stride=self.model_config.days_per_month,
                quantile=0.0,
            )[0, 0, :]

            ground_truth_odn = quantile_pool1d(
                self.Y.swapaxes(1, 2), kernel_size=self.month_kernel_size, stride=self.month_kernel_size, quantile=0.0
            )[0, 0, :]

            return (torch.sqrt((ground_truth_odn - predicted_odn) ** 2) / ground_truth_odn).mean().item()
        return None

    def crps(self) -> torch.Tensor:
        """Calculate CRPS = sum_{n=0}^N (cdf[n] - steps[n])**2.

        Returns:
            torch.Tensor: CRPS across the entire tensor.
        """
        return (
            sr
            .crps_quantile(self.Y[..., 0], self.quantiles, self.quantile_values, m_axis=-1, backend="torch")
            .mean()
            .item()
        )

    def fbeta(
        self,
    ) -> dict:
        """Calculate fbeta score for each quantile.

        Returns:
            dict: F-beta scores for each quantile.
        """
        f_beta = {}
        for q in self.quantile_values:
            f_beta[q] = np.nan_to_num(
                (1 + self.beta**2)
                * self.recall()[q]
                * self.precision()[q]
                / ((self.beta**2) * self.precision()[q] + self.recall()[q]),
                nan=0.0,
                posinf=0,
                neginf=0,
            )
        return f_beta

    def precision(
        self,
    ) -> dict:
        """Calculate precision for each quantile.

        Returns:
            dict: Precision scores for each quantile.
        """
        warnings.warn("Precision per sample not day")

        precision = np.zeros([self._B, len(self.quantile_values)])
        precision_dict = {}
        for b in range(self._B):
            for q in range(len(self.quantile_values)):
                if self.climits[b] >= 0:  # threshold is positive
                    precision[b, q] = metrics.precision_score(
                        self.Y[b, :, 0] > self.climits[b],
                        self.quantiles[b, :, q] > self.climits[b],
                        zero_division=0,
                    )
                else:
                    precision[b, q] = metrics.precision_score(
                        self.Y[b, :, 0] <= self.climits[b],
                        self.quantiles[b, :, q] <= self.climits[b],
                        zero_division=0,
                    )

        precision = np.nan_to_num(precision.mean(axis=0), nan=0.0, posinf=0, neginf=0)
        for i, q in enumerate(self.quantile_values):
            precision_dict[q] = precision[i]

        return precision_dict

    def recall(
        self,
    ) -> dict:
        """Calculate recall for each quantile.

        Returns:
            dict: Recall scores for each quantile.

        """
        warnings.warn("Recall per sample not day")

        recall = np.zeros([self._B, len(self.quantile_values)])
        recall_dict = {}
        for b in range(self._B):
            for q in range(len(self.quantile_values)):
                if self.climits[b] >= 0:  # threshold is positive
                    recall[b, q] = metrics.recall_score(
                        self.Y[b, :, 0] > self.climits[b],
                        self.quantiles[b, :, q] > self.climits[b],
                        zero_division=0,
                    )
                else:  # threshold is negative
                    recall[b, q] = metrics.recall_score(
                        self.Y[b, :, 0] <= self.climits[b],
                        self.quantiles[b, :, q] <= self.climits[b],
                        zero_division=0,
                    )

        recall = np.nan_to_num(recall.mean(axis=0), nan=0.0, posinf=0, neginf=0)
        for i, q in enumerate(self.quantile_values):
            recall_dict[q] = recall[i]

        return recall_dict

    def mae(
        self,
    ) -> dict:
        """Calculate mae.

        Returns:
            dict: MAE for each quantile.
        """
        mae = np.zeros([self._B, len(self.quantile_values)])
        mae_dict = {}
        for b in range(self._B):
            qs = torch.quantile(self.Y[b, ..., 0], torch.tensor([0.01, 0.99]))
            measurement_range = torch.abs(qs[0] - qs[1])
            for q in range(len(self.quantile_values)):
                mae[b, q] = (
                    metrics.mean_absolute_error(
                        torch.nan_to_num(self.Y)[b, :, 0], torch.nan_to_num(self.quantiles)[b, :, q]
                    )
                    / measurement_range
                )
        mae = np.nan_to_num(mae.mean(axis=0), nan=0.0, posinf=0, neginf=0)

        for i, q in enumerate(self.quantile_values):
            mae_dict[q] = mae[i]
        return mae_dict
