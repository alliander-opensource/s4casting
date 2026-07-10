# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import torch
from numpy.typing import NDArray

from s4casting.core.context import Context
from s4casting.core.distributions import gmm_to_quantiles
from s4casting.core.hooks import CommonHooks, TrainingHooks
from s4casting.data.dataset.interface import get_ordered_feature_names
from s4casting.eval.metrics import Metrics
from s4casting.visualisation import plot_quantiles


class EvaluatorHead:
    """Base class for head evaluators."""

    # TODO: this could probably be refactored to be stateless (i.e. not a class)

    def __init__(
        self,
        head_type: str,
        hookable: CommonHooks | TrainingHooks,
    ) -> None:
        """Initialize the EvaluatorHead.

        Args:
            head_type (str): gmm or quantile
            hookable (CommonHooks | TrainingHooks): Hooks for training or common.
        """
        self.hooks = hookable
        self.head_type = head_type

    def reshape_batches(
        self,
        X: torch.Tensor,
        Xm: torch.Tensor,
        Y: torch.Tensor,
        Ym: torch.Tensor,
        quantiles: torch.Tensor,
        input_window_days: int,
        n_day_ahead: int,
        input_interval: int,
        output_interval: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reshape batches of predictions to a single batch of all predictions.

        Flatten a batch of windowed inputs, targets, masks, and predictions into
        a single continuous sequence.

        The function:
          * Slices out the relevant context / prediction windows.
          * Handles GMM or quantile prediction formats.
          * Collapses `(B, T, F)` tensors into `(1, B*T, F)`.

        Args:
            X (torch.Tensor): Input data.
            Xm (torch.Tensor): Input data mask.
            Y (torch.Tensor): Ground truth data.
            Ym (torch.Tensor): Ground truth data mask.
            quantiles (torch.Tensor): Model outputs.
            input_window_days (int): Number of input days.
            n_day_ahead (int): Number of predict days.
            input_interval (int): Input sample rate of eval step.
            output_interval (int): Output sample rate of eval step.

        Returns:
            (torch.Tensor): Input data.
            (torch.Tensor): Input data mask.
            (torch.Tensor): Ground truth data.
            (torch.Tensor): Ground truth data mask.
            (torch.Tensor): Model outputs.
            (torch.Tensor): Weather.
        """
        n_input_predict = int(n_day_ahead * 24 * 60) // input_interval
        n_output_predict = int(n_day_ahead * 24 * 60) // output_interval
        n_context = int(input_window_days * 24 * 60) // input_interval
        quantiles = quantiles[:, -n_output_predict:, ...].detach().cpu()
        # Hack so we can plot weather in both interleaved and single batch settings
        feature_mask = Ym[0].sum(dim=-2) == 0
        Y[:, :, feature_mask] = X[:, :, feature_mask]

        X = X[:, :n_context, :].detach().cpu()
        Xm = Xm[:, :n_context, :].detach().cpu()
        Y = Y[:, -n_input_predict:, :].detach().cpu()
        Ym = Ym[:, -n_input_predict:, :].detach().cpu()
        quantiles = quantiles.reshape((1, quantiles.shape[0] * quantiles.shape[1], quantiles.shape[-1]))

        Y = Y.reshape((1, Y.shape[0] * Y.shape[1], Y.shape[2]))
        X = X.reshape((1, X.shape[0] * X.shape[1], X.shape[2]))
        Ym = Ym.reshape((1, Ym.shape[0] * Ym.shape[1], Ym.shape[2]))
        Xm = Xm.reshape((1, Xm.shape[0] * Xm.shape[1], Xm.shape[2]))
        return X, Xm, Y, Ym, quantiles  # type: ignore[invalid-return-type]

    def report(
        self,
        context: Context,
        prediction: torch.Tensor,
        X: torch.Tensor,
        Xm: torch.Tensor,
        Y: torch.Tensor,
        Ym: torch.Tensor,
        loss: float,
        iteration: int,
        report_type: str,
        context_window_days: int,
        predict_window_days: int,
        input_interval: int,
        output_interval: int,
        n_day_ahead: int,
        location: str | None = None,
        times: NDArray | None = None,
        sign: str | None = None,
    ) -> None:
        """Report the evaluation results.

        Args:
            context (Context): Training or evaluation context.
            prediction (torch.Tensor): Model predictions either quantile or gmm.
            X (torch.Tensor): Input data.
            Xm (torch.Tensor): Input mask.
            Y (torch.Tensor): Target data.
            Ym (torch.Tensor): Target mask.
            loss (float): The mean loss of the signal
            iteration (int): Current iteration number.
            location (str): Location identifier.
            times (NDArray): Dates for the predictions.
            sign (str): Sign for metrics calculation.
            report_type (str): Type of report (e.g., "benchmark", "evaluation", "inference").
            output_interval (int): Output sample rate of eval step.
            n_day_ahead (int): Days ahead for the forecast (different from prediction width).
            context_window_days (int): Context window in days.
            predict_window_days (int): Prediction  window in days.
            input_interval (int): Input sample interval.
        """
        if self.head_type == "gmm":
            logpi, sigma, mu = (x for x in prediction.unbind(dim=-1))  # (B, T, G, 3)
            prediction = gmm_to_quantiles(
                torch.exp(logpi),
                sigma,
                mu,
                context.configuration.model.output_head.quantile_values,
            )

        X, Xm, Y, Ym, quantiles = self.reshape_batches(
            X=X,
            Xm=Xm,
            Y=Y,
            Ym=Ym,
            quantiles=prediction,
            input_window_days=context_window_days - predict_window_days,
            n_day_ahead=n_day_ahead,
            input_interval=input_interval,
            output_interval=output_interval,
        )

        # Ugly: should probably be part of the dataset itself.
        local_bench = context.configuration.benchmarking.benchmarks.get("LocalBenchmark")  # type: ignore[possibly-missing-attribute]
        climits = (
            local_bench.thresholds[local_bench.locations.index(location)]  # type: ignore[index]
            if ((local_bench is not None) and (location is not None) and (local_bench.thresholds is not None))  # type: ignore[possibly-missing-attribute]
            else None
        )
        metrics = Metrics(
            output_sample_interval_minutes=output_interval,
            predict_window_days=predict_window_days,
            input_sample_interval_minutes=input_interval,
            climits=climits,  # type: ignore[arg-type]
            quantiles=quantiles,
            Y=Y,
            model_config=context.configuration.model,
            metrics_config=context.configuration.metrics,
            sign=sign,
            quantile_values=context.configuration.model.output_head.quantile_values,
            loss=loss,
        )
        fig = plot_quantiles(
            quantiles,
            context.configuration.model.output_head.quantile_values,
            X,
            Xm,
            Y,
            Ym,
            times,  # ty: ignore[invalid-argument-type]
            input_interval,
            output_interval,
            report_type,
            "short" if input_interval == output_interval else "medium",
            feature_names=get_ordered_feature_names(context.configuration),
        )

        if report_type == "benchmark":
            self.hooks.benchmark_plot.call(context, iteration, fig, f"{location}_forecast")
            context.benchmark_metrics.update(metrics.get_metrics())
            self.hooks.benchmark_metrics.call(context, iteration)

        elif report_type == "evaluation":
            context.eval_metrics.update(metrics.get_metrics())
            self.hooks.eval_plot.call(context, iteration, fig)
