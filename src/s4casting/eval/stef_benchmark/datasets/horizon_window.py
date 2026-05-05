# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta

import numpy as np
import torch
from openstef_beam.backtesting.restricted_horizon_timeseries import RestrictedHorizonVersionedTimeSeries
from openstef_core.utils.datetime import align_datetime
from torch.utils.data import Dataset


class HorizonWindowDataset(Dataset):
    """Turns a list[RestrictedHorizonVersionedTimeSeries] into tensors."""

    def __init__(
        self,
        horizons: list[RestrictedHorizonVersionedTimeSeries],
        cfg,
        device,
    ):
        """Initialize the dataset.

        Args:
            horizons : List of RestrictedHorizonVersionedTimeSeries objects.
            cfg: Configuration object with dataset parameters.
            device: Device to load the tensors onto.
        """
        self.horizons = horizons
        self.cfg = cfg.benchmarking.benchmarks["StefBeamBenchmark"]
        self.feature_order = cfg.io.feature_order
        self.device = device
        self.n_predict = (
            cfg.benchmarking.benchmarks["StefBeamBenchmark"].predict_window_days * 24 * 60
        ) // cfg.benchmarking.benchmarks["StefBeamBenchmark"].input_sample_interval_minutes

    def __len__(self):
        """Return the number of horizons in the dataset.

        Returns:
            int: Number of horizons.
        """
        return len(self.horizons)

    def __getitem__(self, idx):
        """Get the item at the specified index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: A dictionary containing the input and output tensors.
        """
        h = self.horizons[idx]
        # align horizon to day boundary
        aligned = align_datetime(h.horizon, interval=timedelta(hours=24), mode="floor")
        window_dataset = h.get_window(
            start=aligned - timedelta(days=self.cfg.context_window_days - self.cfg.predict_window_days),
            end=aligned + timedelta(days=self.cfg.predict_window_days),
        )
        w = window_dataset.data

        """
        this is made really badly (by jessica 2025),
        can you add a TODO comment to match the loaded features to what's specified in self.feature_order
        """
        # build weather features if specified in configuration
        if any("weather" in item for item in self.feature_order):
            inp = np.stack(
                [
                    w["load"].values,
                    w["temperature_2m"].values,
                    w["wind_speed_80m"].values,
                    w["shortwave_radiation"].values,
                    w["direct_normal_irradiance"].values,
                ],
                axis=-1,
            ).astype("float32", copy=False)
        else:
            inp = np.stack(
                [
                    w["load"].values,
                ],
                axis=-1,
            ).astype("float32", copy=False)

        X = torch.from_numpy(inp)
        xm = torch.ones_like(X)
        xm[-self.n_predict :, 0] = 0  # mask load, mimic training where prediction values are zeroed

        # mask out nans
        nan_mask = torch.isnan(X)
        xm[nan_mask] = 0
        X = torch.nan_to_num(X) * xm

        if not np.isfinite(X).all():
            raise ValueError("NaNs found in inputs")

        return {
            "X": X,
            "xm": xm,
            "ts": w.index[-self.n_predict :].astype("int64").tolist(),  # timestamps
        }
