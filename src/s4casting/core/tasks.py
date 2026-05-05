# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import warnings
from collections import namedtuple
import torch

from s4casting.core.functional import nanmax, nanmin
from s4casting.data.utils import SampleConfig

TaskSample = namedtuple("TaskSample", ["X", "Xm", "Y", "Ym"])


class TaskDataset:
    """Dataset wrapper that provides input and output masks for each sample."""

    def __init__(self, dataset, max_context_samples):
        """Initialize the TaskDataset.

        Args:
            dataset: The underlying dataset to wrap.
            max_context_samples: The maximum number of samples.
        """
        self.dataset = dataset
        self.max_context_samples = max_context_samples
        self.predict_window_samples = None
        self.predict_dim = None

    def __len__(self):
        """Get the length of the dataset.\

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.dataset)

    def get_masks(self, sample):
        """Get the input and output masks for a given sample.

        Args:
            sample: The sample for which to get the masks.

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
        x_min_samples: int = 10,
        y_min_samples: int = 10,
    ) -> bool:
        """Detect large differences between context and prediction window.

        Args:
            sample (torch.Tensor): Sample to be evaluated.
            eps (float): Small value to avoid division by zero.
            peak_threshold (float): Threshold ratio to determine flatness.
            offset_threshold (float): Threshold for relative offset magnitude.
            x_min_samples (int): Minimum number of samples in context window.
            y_min_samples (int): Minimum number of samples in prediction window.

        Returns:
            bool: True if prediction window is valid, False otherwise.
        """
        x = sample[: -self.predict_window_samples, self.predict_dim]
        y = sample[-self.predict_window_samples :, self.predict_dim]

        if len(x) < x_min_samples or len(y) < y_min_samples:
            return False

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

    def zero_pad(
        self,
        X: torch.Tensor,
        xm: torch.Tensor,
        ym: torch.Tensor,
        sample_config: SampleConfig,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, SampleConfig]:
        """Zero pad context to maximum width.

        Args:
            X (torch.Tensor): Sample to be padded.
            xm (torch.Tensor): Sample context mask to be padded.
            ym (torch.Tensor): Sample prediction mask  to be padded.
            sample_config (SampleConfig): Sample config to be updated

        Returns:
            X (torch.Tensor): Padded sample.
            xm (torch.Tensor): Padded context mask.
            ym (torch.Tensor): Padded prediction mask.
            sample_config  (SampleConfig): Sample configuration.

        """
        _diff = self.max_context_samples - X.shape[0]
        zeros = torch.zeros([_diff, X.shape[1]], device=X.device)
        X = torch.cat([zeros, X], dim=0)
        xm = torch.cat([zeros, xm], dim=0)
        ym = torch.cat([zeros, ym], dim=0)
        sample_config.context_window_samples = self.max_context_samples
        return (X, xm, ym, sample_config)

    def __getitem__(self, idx, _max_retries=10):
        """Get the task sample at the specified index.

        Note that this function does rejection sampling.

        Args:
            idx: The index of the sample to retrieve.
            _max_retries: Number of retries at a diffent index if data is not valid.

        Returns:
            TaskSample: A named tuple containing the input data, input mask, output data, and output mask.
            SampleConfig: A named tuple of the configuration Parameters for a givem sample.
        """
        for attempt in range(_max_retries):
            X, sample_config = self.dataset[idx]
            xm, ym = self.get_masks(X)
            sample_config.predict_window_samples = self.predict_window_samples

            if xm.sum() < 10 or ym.sum() < 10:
                idx = torch.randint(len(self.dataset), (1,)).item()
                continue

            if isinstance(
                self, (PredictionTaskDataset, VariablePredictionTaskDataset)
            ) and not self.valid_predict_window(X):
                if attempt == _max_retries - 1:
                    warnings.warn(
                        f"Could not find a valid sample after {_max_retries} attempts, "
                        f"returning zero-masked sample at idx={idx}."
                        f"\n{sample_config}"
                    )
                    break
                idx = torch.randint(len(self.dataset), (1,)).item()
                continue

            break  # Valid sample found, exit loop
        X, xm, ym, sample_config = self.zero_pad(X, xm, ym, sample_config)
        return TaskSample(
            torch.nan_to_num(X) * xm,
            xm,
            torch.nan_to_num(X.detach().clone()) * ym,
            ym,
        ), sample_config


class PredictionTaskDataset(TaskDataset):
    """Dataset wrapper for prediction tasks."""

    def __init__(self, dataset, max_context_samples, predict_dim, predict_window_samples):
        """Initialize the PredictionTaskDataset.

        Args:
            dataset: The underlying dataset to wrap.
            max_context_samples: The maximum number of samples.
            predict_dim: The dimension to predict.
            predict_window_samples: The window size for prediction.
        """
        super().__init__(dataset, max_context_samples)
        self.predict_window_samples = predict_window_samples
        self.predict_dim = predict_dim

    def get_masks(self, sample):
        """Get the input and output masks for prediction tasks.

        Args:
            sample: The sample for which to get the masks.

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

    def __init__(self, dataset, max_context_samples, predict_dim, min_predict_width_perc, max_predict_width_perc):
        """Initialize the VariablePredictionTaskDataset.

        Args:
            dataset: The underlying dataset to wrap.
            max_context_samples: The maximum number of samples.
            predict_dim: The dimension to predict.
            min_predict_width_perc: Minimum prediction window as a percentage
                of sample length (0.0 to 1.0).
            max_predict_width_perc: Maximum prediction window as a percentage
                of sample length (0.0 to 1.0).
        """
        super().__init__(dataset, max_context_samples)
        self.predict_dim = predict_dim
        self.min_predict_width_perc = min_predict_width_perc
        self.max_predict_width_perc = max_predict_width_perc

    def get_masks(self, sample):
        """Get the input and output masks with randomly sized prediction window.

        Randomly selects a prediction window size between min and max percentages
        of the sample length, then masks all samples within that window.

        Args:
            sample: The sample for which to get the masks.

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
        self.predict_window_samples = max(10, int(sample.shape[0] * random_perc))

        # Mask the prediction window
        xm[-self.predict_window_samples :, self.predict_dim] = 0
        ym[-self.predict_window_samples :, self.predict_dim] = 1

        # Mask out any nans
        xm[torch.isnan(sample)] = 0
        ym[torch.isnan(sample)] = 0

        return (xm, ym)


class RandomMaskingTaskDataset(TaskDataset):
    """Dataset wrapper that applies random masking to samples."""

    def __init__(self, dataset, max_context_samples, min_mask_size, mask_fraction=0.3):
        """Initialize the RandomMaskingTaskDataset.

        Args:
            dataset: The dataset whose samples are to be masked.
            max_context_samples: The maximum number of samples.
            min_mask_size: The min_mask_size in samples.
                       Should be a multiple of the model's `patch_size`.
            mask_fraction: The fraction of mask samples.
        """
        super().__init__(dataset, max_context_samples)
        self.min_mask_size = min_mask_size
        self.mask_fraction = mask_fraction

    def get_masks(self, sample):
        """Get the input and output masks with random masking.

        Args:
            sample: The sample for which to get the masks.

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
