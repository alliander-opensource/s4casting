# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

from collections import namedtuple

import torch

from s4casting.core.functional import nanmax, nanmin

TaskSample = namedtuple("TaskSample", ["X", "Xm", "Y", "Ym"])


class TaskDataset:
    """Dataset wrapper that provides input and output masks for each sample."""

    def __init__(self, dataset):
        """Initialize the TaskDataset.

        Args:
            dataset: The underlying dataset to wrap.
        """
        self.dataset = dataset
        self.predict_window_samples = None
        self.predict_dim = None

    def __len__(self):
        """Get the length of the dataset.\

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.dataset)

    def get_masks(self, sample, _sample_interval):
        """Get the input and output masks for a given sample.

        Args:
            sample: The sample for which to get the masks.
            _sample_interval: The sample sample_interval.

        Returns:
            tuple: A tuple containing the input mask and output mask.
        """
        return (torch.ones(sample.shape), torch.ones(sample.shape))

    def valid_predict_window(
        self,
        sample: torch.Tensor,
        eps: float = 1e-6,
        peak_threshold: float = 10,
        offset_threshold: float = 0.8,
    ) -> bool:
        """Detect large differences between context and prediction window.

        Args:
            sample (torch.Tensor): Sample to be evaluated.
            eps (float): Small value to avoid division by zero.
            peak_threshold (float): Threshold ratio to determine flatness.
            offset_threshold (float): Threshold for relative offset magnitude.

        Returns:
            bool: True if prediction window is valid, False otherwise.
        """
        x = sample[: -self.predict_window_samples, self.predict_dim]
        y = sample[-self.predict_window_samples :, self.predict_dim]

        # get range by subtracting min from max values (use quantiles to remove outliers)
        range_context = torch.nanquantile(x, 0.99) - torch.nanquantile(x, 0.01)
        range_predict = nanmax(y) - nanmin(y)

        ratio = range_predict / (range_context + eps)

        # Reject if prediction window is much more volatile
        if ratio > peak_threshold:
            return False

        median_context = torch.nanmedian(x)
        median_predict = torch.nanmedian(y)

        offset = torch.abs(median_predict - median_context)
        scaled_offset = offset / (range_context + eps)

        return not scaled_offset > offset_threshold

    def __getitem__(self, idx):
        """Get the task sample at the specified index.

        Args:
            idx: The index of the sample to retrieve.

        Returns:
            TaskSample: A named tuple containing the input data, input mask, output data, and output mask.
        """
        X, sample_config = self.dataset[idx]
        xm, ym = self.get_masks(X, sample_config.sample_interval_minutes)
        # A hack so that prediction window is accessible downstream
        sample_config.predict_window_days = (self.predict_window_samples * sample_config.sample_interval_minutes) // (
            24 * 60
        )

        # Note we only do this according to prediction window split
        # But could theoretically be done with random masking as well.
        if isinstance(self, (PredictionTaskDataset, VariablePredictionTaskDataset)) and not self.valid_predict_window(
            X
        ):
            # i.e. if no valid then mask the whole sample
            ym = torch.zeros_like(ym)
            xm = torch.zeros_like(ym)

        # set prediction window

        return TaskSample(torch.nan_to_num(X) * xm, xm, torch.nan_to_num(X.detach().clone()) * ym, ym), sample_config


class PredictionTaskDataset(TaskDataset):
    """Dataset wrapper for prediction tasks."""

    def __init__(self, dataset, predict_dim, predict_window_samples):
        """Initialize the PredictionTaskDataset.

        Args:
            dataset: The underlying dataset to wrap.
            predict_dim: The dimension to predict.
            predict_window_samples: The window size for prediction.
        """
        super().__init__(dataset)
        self.predict_window_samples = predict_window_samples
        self.predict_dim = predict_dim

    def get_masks(self, sample, _sample_interval):
        """Get the input and output masks for prediction tasks.

        Args:
            sample: The sample for which to get the masks.
            _sample_interval: The sample sample_interval.

        Returns:
            tuple: A tuple containing the input mask and output mask.
        """
        x = torch.ones(sample.shape)
        y = torch.zeros(sample.shape)
        x[-self.predict_window_samples :, self.predict_dim] = 0
        y[-self.predict_window_samples :, self.predict_dim] = 1

        # Mask out any nans
        x[torch.isnan(sample)] = 0
        y[torch.isnan(sample)] = 0

        return (x, y)


class VariablePredictionTaskDataset(TaskDataset):
    """Dataset wrapper that randomly selects prediction window size within a percentage range.

    Unlike PredictionTaskDataset which uses a fixed prediction window, this class
    randomly selects the prediction window size between min and max percentages
    of the sample length each time a sample is retrieved.
    """

    def __init__(self, dataset, predict_dim, min_predict_width_perc, max_predict_width_perc):
        """Initialize the VariablePredictionTaskDataset.

        Args:
            dataset: The underlying dataset to wrap.
            predict_dim: The dimension to predict.
            min_predict_width_perc: Minimum prediction window as a percentage
                of sample length (0.0 to 1.0).
            max_predict_width_perc: Maximum prediction window as a percentage
                of sample length (0.0 to 1.0).
        """
        super().__init__(dataset)
        self.predict_dim = predict_dim
        self.min_predict_width_perc = min_predict_width_perc
        self.max_predict_width_perc = max_predict_width_perc

    def get_masks(self, sample, sample_interval):
        """Get the input and output masks with randomly sized prediction window.

        Randomly selects a prediction window size between min and max percentages
        of the sample length, then masks all samples within that window.

        Args:
            sample: The sample for which to get the masks.
            sample_interval: The sample sample_interval.

        Returns:
            tuple: A tuple containing the input mask and output mask.
        """
        xm = torch.ones(sample.shape)
        ym = torch.zeros(sample.shape)

        # Randomly select prediction window percentage and compute window size
        random_perc = (
            torch.rand(1).item() * (self.max_predict_width_perc - self.min_predict_width_perc)
            + self.min_predict_width_perc
        )
        self.predict_window_samples = max(int((1 * 24 * 60) // sample_interval), int(sample.shape[0] * random_perc))

        # Mask the prediction window
        xm[-self.predict_window_samples :, self.predict_dim] = 0
        ym[-self.predict_window_samples :, self.predict_dim] = 1

        # Mask out any nans
        xm[torch.isnan(sample)] = 0
        ym[torch.isnan(sample)] = 0

        return (xm, ym)


class RandomMaskingTaskDataset(TaskDataset):
    """Dataset wrapper that applies random masking to samples."""

    def __init__(self, dataset, min_mask_size, mask_fraction=0.3):
        """Initialize the RandomMaskingTaskDataset.

        Args:
            dataset: The dataset whose samples are to be masked.
            min_mask_size: The min_mask_size in samples.
                       Should be a multiple of the model's `patch_size`.
            mask_fraction: The fraction of mask samples.
        """
        super().__init__(dataset)
        self.min_mask_size = min_mask_size
        self.mask_fraction = mask_fraction

    def get_masks(self, sample, _sample_interval):
        """Get the input and output masks with random masking.

        Args:
            sample: The sample for which to get the masks.
            _sample_interval: The sample sample_interval.

        Returns:
            tuple: A tuple containing the input mask and output mask.
        """
        ym = torch.zeros_like(sample)
        while ym.sum() == 0:
            masked_patches = torch.rand(sample.shape[0] // self.min_mask_size) <= self.mask_fraction
            masked_patches = masked_patches.repeat_interleave(self.min_mask_size)
            ym[masked_patches] = 1

        ym[torch.isnan(sample)] = 0
        xm = 1 - ym

        return (xm, ym)
