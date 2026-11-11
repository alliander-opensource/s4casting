# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

"""Forecast fev evaluation windows with an s4casting model.

The adapter turns the ``past_data`` / ``future_data`` of a :class:`fev.EvaluationWindow` into the
``(X, xm)`` tensors the models consume, runs the model, and returns predictions in the format
:meth:`fev.Task.evaluation_summary` expects.

Input layout (last axis of ``X``):

* channel 0: the target, masked over the forecast horizon;
* channels 1..n: the task's covariates in task order, each in its own slot. Known-future covariates
  keep their horizon values, past-only covariates are masked there. Slots the task does not fill are
  zero and masked, exactly as covariate dropout presents them in training. Covariate order carries
  no meaning for the models (training permutes it), so no mapping onto weather names is needed;
* the trailing channels, when the model was trained with the time loader: unix time in seconds and, for a
  model trained with location, latitude and longitude set to the spatial embedding's reference point
  (fev series carry no coordinates, so the location contributes nothing beyond the embedding's bias).

For models the training batcher zero-pads to one fixed length (the SSM), windows are left-padded the same
way so benchmark inputs are shaped exactly like training inputs; the transformer gets unpadded windows, as
in training.

Series are handed to the model at their own sample rate, which must be one of the training rates: the
benchmark never resamples. A task at a rate the model was not trained on is reported as unsupported and
skipped, so its score is absent rather than a resampling artefact.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext

import datasets
import fev
import numpy as np
import pandas as pd
import torch

from s4casting.core.batcher import pads_to_fixed_length
from s4casting.core.config import Configuration, FevBenchBenchmark
from s4casting.core.distributions import gmm_to_quantiles
from s4casting.data.utils import build_valid_context_sampling_pairs
from s4casting.model._encoders import NL_CENTER, TIME_AND_LOCATION

logger = logging.getLogger(__name__)

PREDICTIONS = "predictions"
POINT_QUANTILE = 0.5


class UnsupportedTaskError(ValueError):
    """Raised when a fev task cannot be presented to the model (e.g. a sample rate without a training rate)."""


def interval_minutes(timestamps: list) -> int:
    """Infer the sample interval of a series from its timestamps.

    Args:
        timestamps (list): Timestamps of one series, as returned by ``datasets``.

    Returns:
        int: Median spacing in whole minutes.

    Raises:
        UnsupportedTaskError: If the series has fewer than two timestamps.
    """
    ts = pd.DatetimeIndex(timestamps)
    if len(ts) < 2:
        raise UnsupportedTaskError("A series needs at least two timestamps to infer its sample interval.")
    return round(float(np.median((ts[1:] - ts[:-1]).total_seconds())) / 60)


def _as_float(values: list) -> np.ndarray:
    """Convert a ``datasets`` column (floats with ``None`` for missing values) to a float array with NaNs.

    Returns:
        np.ndarray: 1-D float64 array.
    """
    return pd.Series(values, dtype="float64").to_numpy()


def _unix_seconds(timestamp) -> float:
    return pd.Timestamp(timestamp).timestamp()


class FevForecaster:
    """Forecast every series of a fev evaluation window with an s4casting model."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: Configuration,
        bench: FevBenchBenchmark,
        device: str | torch.device,
        autocast: Callable[[], AbstractContextManager] | None = None,
    ) -> None:
        """Initialize the forecaster.

        Args:
            model (torch.nn.Module): Model with the s4casting forward signature ``(x, xm, in_rate, out_rate)``.
            config (Configuration): Full configuration the model was built from.
            bench (FevBenchBenchmark): Benchmark settings.
            device (str | torch.device): Device to run on.
            autocast (Callable[[], AbstractContextManager] | None): Factory for the precision context.

        Raises:
            ValueError: If a quantile output head cannot produce the point forecast quantile.
        """
        self.model = model
        self.config = config
        self.bench = bench
        self.device = device
        self.autocast = autocast or nullcontext
        self.model_rates = sorted(config.model.input_sample_intervals_minutes)
        # Sequence and horizon lengths are floored/padded to whole patches of the s4casting patch encoder;
        # a model that patches internally (Chronos) declares patch_alignment = 1 and gets every sample.
        self.patch_size = int(getattr(model, "patch_alignment", config.model.patch_encoder.patch_size))
        n_data_features, self.n_time_features = config.io.input_feature_counts()
        # Models that take any number of covariates (Chronos) get every task covariate, as in their own fev
        # integration; the s4casting models have the fixed slot count they trained with.
        self.any_covariates = bool(getattr(model, "accepts_any_covariates", False))
        self.n_covariate_slots = n_data_features - 1
        # A model-declared context length (Chronos-2: 8192 samples) or a benchmark budget replaces the
        # trained windows as the cap; when both are set the smaller applies.
        caps = [c for c in (getattr(model, "max_context_samples", None), bench.max_context_samples) if c]
        self.max_context_samples: int | None = min(caps) if caps else None
        self.any_rate = bool(getattr(model, "accepts_any_rate", False))
        self.head = config.model.output_head.arch
        self.head_quantiles = [float(q) for q in config.model.output_head.quantile_values]
        if self.head == "quantile" and not self._has_quantile(POINT_QUANTILE):
            raise ValueError("The quantile output head must include 0.5 to provide fev's point forecast.")
        if self.n_time_features == TIME_AND_LOCATION:
            logger.warning(
                "The model was trained with location; fev-bench series carry no coordinates, so every series is "
                "placed at the spatial embedding's reference point (%.2f, %.2f).",
                *NL_CENTER,
            )
        self._warned_truncation: set[str] = set()
        self._trained_lengths: dict[int, list[int]] = {}
        self.padded_samples = self._padded_samples()
        self._rate_seconds = 0
        self.levels: list[float] = []

    def _padded_samples(self) -> int:
        """Sequence length the training batcher zero-pads every sample to, so benchmark inputs match it.

        Returns:
            int: The padded length in samples (the longest valid (context window, rate) pair) for models the
            batcher pads (the SSM), otherwise 0 and no padding is added.
        """
        if self.bench.context_window_days is not None or not pads_to_fixed_length(self.config.model):
            return 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            pairs = build_valid_context_sampling_pairs(
                context_days=list(self.config.model.context_window),
                sample_intervals_minutes=list(self.model_rates),
                max_context_len=None,
                interval_context_limits=self.config.io.interval_context_limits,
            )
        if not pairs["valid_pairs"]:
            return 0
        return int(pairs["recommended_max_context_samples"])

    def _has_quantile(self, level: float) -> bool:
        return any(math.isclose(level, q, abs_tol=1e-9) for q in self.head_quantiles)

    def _quantile_index(self, level: float) -> int:
        for i, q in enumerate(self.head_quantiles):
            if math.isclose(level, q, abs_tol=1e-9):
                return i
        raise ValueError(
            f"Quantile output head cannot produce level {level}; add it to model.output_head.quantile_values."
        )

    def trained_lengths(self, rate: int) -> list[int]:
        """Sequence lengths the model trained with at ``rate``, ascending, as multiples of the patch size.

        ``context_window_days`` overrides them with that single length.

        Args:
            rate (int): Model sample rate in minutes.

        Returns:
            list[int]: Sequence lengths in samples.

        Raises:
            UnsupportedTaskError: If the configuration has no context window for this rate.
        """
        if rate in self._trained_lengths:
            return self._trained_lengths[rate]
        if self.bench.context_window_days is not None:
            days_list = [self.bench.context_window_days]
        else:
            # The helper warns for every context window it prunes at this rate; that pruning is the
            # point of the call here, so the warnings carry no information.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                try:
                    pairs = build_valid_context_sampling_pairs(
                        context_days=list(self.config.model.context_window),
                        sample_intervals_minutes=[rate],
                        max_context_len=None,
                        patch_size=self.patch_size,
                        interval_context_limits=self.config.io.interval_context_limits,
                    )["valid_pairs"]
                except ValueError:  # the helper raises when every window is pruned at this rate
                    pairs = []
            if not pairs:
                raise UnsupportedTaskError(
                    f"no context window of model.context_window {list(self.config.model.context_window)} is valid "
                    f"at {rate} minutes under io.interval_context_limits[{rate}] = "
                    f"{self.config.io.interval_context_limits.get(rate)} (whole days, multiple of patch_size "
                    f"{self.patch_size}); add one, or set FevBenchBenchmark.context_window_days."
                )
            days_list = [pair["context_days"] for pair in pairs]
        lengths = set()
        for days in days_list:
            samples = (days * 24 * 60) // rate
            lengths.add(max(self.patch_size, samples - samples % self.patch_size))
        self._trained_lengths[rate] = sorted(lengths)
        return self._trained_lengths[rate]

    def sequence_samples(self, rate: int, available: int | None = None) -> int:
        """Total sequence length (history plus forecast window) handed to the model at ``rate``.

        The transformer gets the series at its own length, capped at the longest window it trained with
        at that rate (or at ``max_context_samples`` when a model or the benchmark declares a budget) and
        rounded up to whole patches: a series shorter than the cap keeps every sample it
        has and is left-padded by at most patch_size - 1 masked rows (masked gaps are routine in training;
        dropping up to a patch of real history is not free, 7 of 39 weeks on solar weekly). It trains on
        unpadded windows of many lengths with relative positions and timestamp features, so a length
        between two trained ones is not a train-test mismatch. Models the batcher pads to one fixed length
        (the SSM) get the longest trained window and are padded exactly as in training. Without
        ``available`` the longest trained window is returned.

        Args:
            rate (int): Model sample rate in minutes.
            available (int | None): Samples the series can supply, history plus the padded horizon.

        Returns:
            int: Sequence length in samples, a multiple of the patch size.
        """
        longest = self.trained_lengths(rate)[-1] if self.max_context_samples is None else self.max_context_samples
        longest -= longest % self.patch_size
        if available is None or self.padded_samples:
            return longest
        return max(self.patch_size, min(longest, math.ceil(available / self.patch_size) * self.patch_size))

    def _covariate_columns(
        self, task: fev.Task, task_name: str, extra_past: list[str] | None = None
    ) -> tuple[list[str], list[str]]:
        """Known-future and past-only covariate columns of a task, truncated to the model's slots.

        Args:
            task (fev.Task): The task.
            task_name (str): Name for the truncation warning.
            extra_past (list[str] | None): Additional past-only columns (the other targets of a multivariate
                task); they come last, so they are the first to go when the slots run out.

        Returns:
            tuple[list[str], list[str]]: Known-future columns, past-only columns.
        """
        known = list(task.known_dynamic_columns)
        past = list(task.past_dynamic_columns) + list(extra_past or [])
        n_needed = len(known) + len(past)
        if self.any_covariates:
            return known, past
        if n_needed > self.n_covariate_slots:
            if task_name not in self._warned_truncation:
                logger.warning(
                    "%s has %d covariates but the model has %d covariate slots; keeping the first %d "
                    "(known covariates first).",
                    task_name,
                    n_needed,
                    self.n_covariate_slots,
                    self.n_covariate_slots,
                )
                self._warned_truncation.add(task_name)
            known = known[: self.n_covariate_slots]
            past = past[: max(0, self.n_covariate_slots - len(known))]
        return known, past

    def _series_window(
        self,
        past: dict,
        future: dict,
        known: list[str],
        past_only: list[str],
        n_context: int,
        n_horizon: int,
        n_horizon_padded: int,
        n_pad: int,
    ) -> np.ndarray:
        """Build one (n_pad + n_context + n_horizon_padded, F) window at the task rate; NaN marks masked entries.

        ``n_pad`` leading rows are all NaN (zero and masked after conversion, time channel included), exactly
        like the zero padding the training batcher prepends to every sample.

        Returns:
            np.ndarray: The window.
        """
        n_slots = len(known) + len(past_only) if self.any_covariates else self.n_covariate_slots
        n_channels = 1 + n_slots + self.n_time_features
        target = _as_float(past["target"])
        n_past = len(target)
        series = np.full((n_past + n_horizon, n_channels), np.nan)
        series[:n_past, 0] = target
        for slot, col in enumerate(known, start=1):
            series[:n_past, slot] = _as_float(past[col])
            series[n_past:, slot] = _as_float(future[col])
        for slot, col in enumerate(past_only, start=1 + len(known)):
            series[:n_past, slot] = _as_float(past[col])

        # keep the tail of the history, left-padded when the series is shorter than the context
        history = series[:n_past][-n_context:]
        if history.shape[0] < n_context:
            history = np.concatenate([np.full((n_context - history.shape[0], n_channels), np.nan), history])
        window = np.concatenate([history, series[n_past:]])  # (n_context + n_horizon, F)
        if n_horizon_padded > n_horizon:
            window = np.concatenate([window, np.full((n_horizon_padded - n_horizon, n_channels), np.nan)])
        window[n_context:, 0] = np.nan  # forecast horizon: the target is never visible
        if self.n_time_features:
            # dense unix time on the model-rate grid, anchored on the first forecast timestamp so the
            # temporal embedding sees real timestamps over padding as well
            forecast_start = _unix_seconds(future["timestamp"][0])
            time_column = n_channels - self.n_time_features
            window[:, time_column] = forecast_start + np.arange(-n_context, n_horizon_padded) * self._rate_seconds
            if self.n_time_features == TIME_AND_LOCATION:
                window[:, time_column + 1 :] = NL_CENTER
        if n_pad > 0:
            window = np.concatenate([np.full((n_pad, n_channels), np.nan), window])
        return window

    def _forward(self, windows: np.ndarray, rate: int, first: int, n_horizon: int) -> torch.Tensor:
        """Run the model on stacked windows and return quantile forecasts of shape (B, n_horizon, Q).

        Args:
            windows (np.ndarray): Stacked windows, NaN where masked.
            rate (int): Model sample rate in minutes.
            first (int): Index of the first forecast sample in the window.
            n_horizon (int): Number of forecast samples to return.

        Returns:
            torch.Tensor: Quantile forecasts at ``self.levels``.
        """
        x = torch.from_numpy(np.nan_to_num(windows).astype(np.float32)).to(self.device)
        xm = torch.from_numpy(np.isfinite(windows).astype(np.float32)).to(self.device)
        with torch.no_grad(), self.autocast():
            out, _ = self.model(x, xm, rate, rate)
        if out.shape[1] < x.shape[1]:
            # horizon-only output (Chronos): the model returns the forecast rows alone, in window order
            pred = out[:, :n_horizon, 0, ...].float()
        else:
            pred = out[:, first : first + n_horizon, 0, ...].float()
        if self.head == "gmm":
            logpi, sigma, mu = pred.unbind(dim=-1)
            return gmm_to_quantiles(torch.exp(logpi), sigma, mu, self.levels)
        return pred[..., [self._quantile_index(q) for q in self.levels]]

    def predict_window(self, task: fev.Task, window: fev.EvaluationWindow) -> datasets.DatasetDict | list[dict]:
        """Forecast every series of one evaluation window.

        Multivariate tasks are forecast one target column at a time (each column becomes its own series
        with the task's covariates) and recombined into fev's multivariate format. The other target
        columns are handed to each series as past-only covariates, so the group attention can use the
        cross-variate information a joint forecast would see.

        Args:
            task (fev.Task): The task the window belongs to.
            window (fev.EvaluationWindow): The evaluation window.

        Returns:
            datasets.DatasetDict | list[dict]: Predictions accepted by :meth:`fev.Task.evaluation_summary`.

        Raises:
            UnsupportedTaskError: If the task's sample rate is not one of the model's training rates.
        """
        task_name = task.task_name or str(task.dataset_config)
        self.levels = sorted({float(q) for q in task.quantile_levels} | {POINT_QUANTILE})
        past_data, future_data = fev.convert_input_data(
            window, adapter="datasets", as_univariate=True, univariate_target_column="target"
        )
        # fev orders the univariate series item-major, cycling through the target columns; the other
        # targets of the same item join each series as past-only covariates named other_1, other_2, ...
        extra_past: list[dict] = [{} for _ in range(len(past_data))]
        extra_names: list[str] = []
        if task.is_multivariate:
            columns = list(task.target_columns)
            extra_names = [f"other_{j}" for j in range(1, len(columns))]
            multivariate_past, _ = window.get_input_data()
            items = [{c: row[c] for c in columns} for row in multivariate_past]
            for k in range(len(past_data)):
                own = columns[k % len(columns)]
                others = [c for c in columns if c != own]
                extra_past[k] = dict(zip(extra_names, (items[k // len(columns)][c] for c in others), strict=True))
        rate = interval_minutes(past_data[0]["timestamp"])
        if not self.any_rate and rate not in self.model_rates:
            raise UnsupportedTaskError(
                f"{rate}-minute data, but the model was trained at {self.model_rates} minutes; the benchmark does "
                "not resample, so add that rate to model.input_sample_intervals_minutes to score this task."
            )
        self._rate_seconds = rate * 60
        n_horizon = task.horizon
        n_horizon_padded = math.ceil(n_horizon / self.patch_size) * self.patch_size
        history = int(np.median([len(item["target"]) for item in past_data]))
        sequence = self.sequence_samples(rate, history + n_horizon_padded)
        n_context = max(self.patch_size, sequence - n_horizon_padded)
        logger.info(
            "fev-bench %s: %d-minute data, median history %d, window %d of %s (%d history + %d horizon).",
            task_name,
            rate,
            history,
            sequence,
            f"cap {self.max_context_samples}" if self.max_context_samples else f"trained {self.trained_lengths(rate)}",
            n_context,
            n_horizon_padded,
        )
        n_pad = max(0, self.padded_samples - n_context - n_horizon_padded)
        known, past_only = self._covariate_columns(task, task_name, extra_past=extra_names)

        predictions: list[dict] = []
        n_series = len(past_data)
        batch_size = self.config.benchmarking.batch_size
        for start in range(0, n_series, batch_size):
            rows = range(start, min(start + batch_size, n_series))
            windows = np.stack([
                self._series_window(
                    {**past_data[i], **extra_past[i]},
                    future_data[i],
                    known,
                    past_only,
                    n_context,
                    n_horizon,
                    n_horizon_padded,
                    n_pad,
                )
                for i in rows
            ])
            quantiles = self._forward(windows, rate, n_pad + n_context, n_horizon).cpu().numpy()  # (B, n_horizon, Q)
            for b in range(quantiles.shape[0]):
                forecast = {str(q): quantiles[b, :, i].astype(np.float64).tolist() for i, q in enumerate(self.levels)}
                forecast[PREDICTIONS] = forecast[str(POINT_QUANTILE)]
                predictions.append({
                    k: v for k, v in forecast.items() if k == PREDICTIONS or float(k) in task.quantile_levels
                })
        if task.is_multivariate:
            return fev.combine_univariate_predictions_to_multivariate(predictions, target_columns=task.target_columns)
        return predictions
