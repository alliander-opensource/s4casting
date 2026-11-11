# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import bisect
import warnings
from collections import namedtuple

import torch

from s4casting.core.functional import nanmax, nanmin

TaskSample = namedtuple("TaskSample", ["X", "Xm", "Y", "Ym"])


class TaskDataset:
    """Dataset wrapper that provides input and output masks for each sample."""

    def __init__(self, dataset, max_context_samples, max_retries):
        """Initialize the TaskDataset.

        Args:
            dataset: The underlying dataset to wrap.
            max_context_samples: Length every sample is left-padded to with zeros, or None to leave samples
                at their own length (a batch always comes from one (context window, rate) group, so its samples
                already share a length; only models with a fixed-length layer need the padding).
            max_retries: Number of retries at a diffent index if data is not valid.
        """
        self.dataset = dataset
        self.max_context_samples = max_context_samples
        self.max_retries = max(1, max_retries)
        self.predict_window_samples = None
        self.covariate_dropout_all_p = 0.0  # set by the batcher from training.covariate_dropout_all_p
        self.covariate_horizon_dropout_p = 0.0  # set by the batcher from training.covariate_horizon_dropout_p
        self.predict_dim = None

    def redraw_index(self, idx: int) -> int:
        """Draw a replacement index for rejection sampling.

        The training set concatenates one dataset per (context window, sample rate) pair and a batch must
        stay within one of them, so the redraw is confined to the sub-dataset that ``idx`` belongs to.

        Args:
            idx: Index of the rejected sample.

        Returns:
            int: A random index from the same sub-dataset (or the whole dataset when it is not concatenated).
        """
        bounds = getattr(self.dataset, "cumulative_sizes", None)
        if not bounds:
            return int(torch.randint(len(self.dataset), (1,)).item())
        group = bisect.bisect_right(bounds, idx)
        low = 0 if group == 0 else bounds[group - 1]
        return int(torch.randint(low, bounds[group], (1,)).item())

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
            _sample_interval: The sample interval in minutes (unused in the base implementation).

        Returns:
            tuple: A tuple containing the input mask and output mask.
        """
        return (torch.ones(sample.shape), torch.ones(sample.shape))

    def window_samples(self, _sample_interval) -> int:
        """Forecast window of the current sample in samples.

        The base implementation has no forecast window; prediction tasks override it so the window
        follows the sample's own rate under multi-rate training.

        Args:
            _sample_interval: Sample interval of the current sample in minutes (unused here).

        Returns:
            int: Number of forecast samples at the end of the sample.
        """
        return 0

    def valid_predict_window(
        self,
        sample: torch.Tensor,
        predict_window_samples: int,
        eps: float = 1e-6,
        peak_threshold: float = 10,
        offset_threshold: float = 0.8,
    ) -> bool:
        """Detect large differences between context and prediction window.

        Args:
            sample (torch.Tensor): Sample to be evaluated.
            predict_window_samples (int): Forecast window of this sample in samples.
            eps (float): Small value to avoid division by zero.
            peak_threshold (float): Threshold ratio to determine flatness.
            offset_threshold (float): Threshold for relative offset magnitude.

        Returns:
            bool: True if prediction window is valid, False otherwise.
        """
        if not 0 < predict_window_samples < sample.shape[0]:
            return False
        x = sample[:-predict_window_samples, self.predict_dim]
        y = sample[-predict_window_samples:, self.predict_dim]

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
        self, X: torch.Tensor, xm: torch.Tensor, ym: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Zero pad context to maximum width; a no-op when no padded length is configured.

        Args:
            X (torch.Tensor): Sample to be padded.
            xm (torch.Tensor): Sample context mask to be padded.
            ym (torch.Tensor): Sample prediction mask  to be padded.

        Returns:
            X (torch.Tensor): Padded sample.
            xm (torch.Tensor): Padded context mask.
            ym (torch.Tensor): Padded prediction mask.

        """
        if self.max_context_samples is None:
            return (X, xm, ym)
        _diff = self.max_context_samples - X.shape[0]

        if _diff == 0:
            return (X, xm, ym)
        if _diff < 0:
            raise ValueError("Context width greater than pad width")

        zeros = torch.zeros([_diff, X.shape[1]], device=X.device)
        X = torch.cat([zeros, X], dim=0)
        xm = torch.cat([zeros, xm], dim=0)
        ym = torch.cat([zeros, ym], dim=0)
        return (X, xm, ym)

    def drop_covariates(self, X, xm, ym, n_predict: int = 0):
        """Randomly drop covariates for training group attention.

        Covariates are permuted and a random number of them is dropped over the whole sample. Of the kept
        ones, each is then hidden over the tail of the forecast window with probability
        ``covariate_horizon_dropout_p``: half of those draws hide the whole forecast window (a past-only
        covariate) and half hide it from a random point onward (a forecast that runs out). The context
        always keeps every covariate the forecast window keeps, matching what inference can present.

        Args:
            X: X tensor.
            xm: Mask tensor.
            ym: Mask tensor.
            n_predict: Forecast window at the end of the sample in samples; 0 disables the horizon cut.

        Returns:
            xm: Mask tensor.
        """
        if not self.covariate_dropout:
            return X, xm, ym

        indices = torch.arange(
            self.predict_dim + 1,
            xm.shape[1] - self.n_time_features,
            device=xm.device,
        )
        n_covariates = len(indices)

        if n_covariates == 0:
            return X, xm, ym

        perm = torch.randperm(n_covariates, device=X.device)
        X[:, indices] = X[:, indices][:, perm]
        xm[:, indices] = xm[:, indices][:, perm]
        ym[:, indices] = ym[:, indices][:, perm]

        if self.covariate_dropout_all_p > 0 and float(torch.rand(())) < self.covariate_dropout_all_p:
            n_keep = 0  # a covariate-free sample, as the no-covariate benchmark tasks present them
        else:
            n_keep = torch.randint(0, n_covariates + 1, (), device=X.device).item()
        xm[:, indices[n_keep:]] = 0
        ym[:, indices[n_keep:]] = 0

        if self.covariate_horizon_dropout_p > 0 and 0 < n_predict < X.shape[0]:
            horizon_start = X.shape[0] - n_predict
            for channel in indices[:n_keep]:
                if float(torch.rand(())) >= self.covariate_horizon_dropout_p:
                    continue
                # past-only (cut at the start of the forecast window) or a forecast that runs out partway
                cut = 0 if n_predict == 1 or float(torch.rand(())) < 0.5 else int(torch.randint(1, n_predict, ()))
                xm[horizon_start + cut :, channel] = 0

        return X, xm, ym

    def __getitem__(self, idx):
        """Get the task sample at the specified index.

        Note that this function does rejection sampling if max_retries > 0.

        Args:
            idx: The index of the sample to retrieve.

        Returns:
            TaskSample: A named tuple containing the input data, input mask, output data, and output mask.
            SampleConfig: A named tuple of the configuration Parameters for a givem sample.
        """
        for attempt in range(self.max_retries):
            X, sample_config = self.dataset[idx]
            xm, ym = self.get_masks(X, sample_config.sample_interval_minutes)
            n_predict = self.window_samples(sample_config.sample_interval_minutes)
            X, xm, ym = self.drop_covariates(X, xm, ym, n_predict)
            sample_config.predict_window_samples = n_predict

            if xm.sum() < 10 or ym.sum() < 10:
                idx = self.redraw_index(idx)
                continue

            if isinstance(
                self, (PredictionTaskDataset, VariablePredictionTaskDataset)
            ) and not self.valid_predict_window(X, n_predict):
                if attempt == self.max_retries - 1:
                    warnings.warn(
                        f"Could not find a valid sample after {self.max_retries} attempts, "
                        f"returning zero-masked sample at idx={idx}."
                        f"\n{sample_config}"
                    )
                    break
                idx = self.redraw_index(idx)
                continue

            break  # Valid sample found, exit loop
        X, xm, ym = self.zero_pad(X, xm, ym)
        sample_config.context_window_samples = X.shape[0]
        return TaskSample(
            torch.nan_to_num(X) * xm,
            xm,
            torch.nan_to_num(X.detach().clone()) * ym,
            ym,
        ), sample_config


class PredictionTaskDataset(TaskDataset):
    """Dataset wrapper for prediction tasks."""

    def __init__(
        self,
        dataset,
        max_context_samples,
        max_retries,
        predict_dim,
        predict_window_samples,
        n_time_features,
        covariate_dropout,
        base_sample_interval_minutes=None,
    ):
        """Initialize the PredictionTaskDataset.

        Args:
            dataset: The underlying dataset to wrap.
            max_context_samples:  The max number of samples in a sample.
            max_retries: Number of retries at a different index if data is not valid.
            predict_dim: The dimension to predict.
            predict_window_samples: The window size for prediction in samples at
                ``base_sample_interval_minutes``.
            n_time_features: The number of time features (1 - unix time; 3 - unix time, lat, lon).
            covariate_dropout: Boolean, if training with covariate dropout.
            base_sample_interval_minutes: Rate at which ``predict_window_samples`` is expressed. When set,
                the window is rescaled to each sample's own rate so it spans the same duration under
                multi-rate training (a 2-day window is 192 samples at 15 minutes but 2 at daily rate).
                None keeps a fixed window in samples.
        """
        super().__init__(dataset, max_context_samples, max_retries)
        self.predict_window_samples = predict_window_samples
        self.base_sample_interval_minutes = base_sample_interval_minutes
        self.predict_dim = predict_dim
        self.n_time_features = n_time_features
        self.covariate_dropout = covariate_dropout

    def window_samples(self, sample_interval) -> int:
        """Forecast window at the sample's rate.

        Args:
            sample_interval: Sample interval of the current sample in minutes.

        Returns:
            int: Forecast samples, at least 1.
        """
        if self.base_sample_interval_minutes is None:
            return self.predict_window_samples
        minutes = self.predict_window_samples * self.base_sample_interval_minutes
        return max(1, round(minutes / float(sample_interval)))

    def get_masks(self, sample, sample_interval):
        """Get the input and output masks for prediction tasks.

        Args:
            sample: The sample for which to get the masks.
            sample_interval: The sample interval in minutes; sets the window length under multi-rate training.

        Returns:
            tuple: A tuple containing the input mask and output mask.
        """
        n_predict = self.window_samples(sample_interval)
        xm = torch.ones(sample.shape)
        ym = torch.zeros(sample.shape)
        xm[-n_predict:, self.predict_dim] = 0
        ym[-n_predict:, self.predict_dim] = 1

        # Mask out any nans
        xm[torch.isnan(sample)] = 0
        ym[torch.isnan(sample)] = 0

        return (xm, ym)


class VariablePredictionTaskDataset(TaskDataset):
    """Dataset wrapper that randomly selects prediction window size within a percentage range.

    Unlike PredictionTaskDataset which uses a fixed prediction window, this class
    randomly selects the prediction window size between min and max percentages
    of the sample length each time a sample is retrieved.
    """

    def __init__(
        self,
        dataset,
        max_context_samples,
        max_retries,
        predict_dim,
        min_predict_width_perc,
        max_predict_width_perc,
        n_time_features,
        covariate_dropout,
    ):
        """Initialize the VariablePredictionTaskDataset.

        Args:
            dataset: The underlying dataset to wrap.
            max_context_samples: maximum context witdth.
            max_retries: Number of retries at a different index if data is not valid.
            predict_dim: The dimension to predict.
            min_predict_width_perc: Minimum prediction window as a percentage
                of sample length (0.0 to 1.0).
            max_predict_width_perc: Maximum prediction window as a percentage
                of sample length (0.0 to 1.0).
            n_time_features: The number of time features (1 - unix time; 3 - unix time, lat, lon).
            covariate_dropout: Boolean, if training with covariate dropout.
        """
        super().__init__(dataset, max_context_samples, max_retries)
        self.predict_dim = predict_dim
        self.min_predict_width_perc = min_predict_width_perc
        self.max_predict_width_perc = max_predict_width_perc
        self.n_time_features = n_time_features
        self.covariate_dropout = covariate_dropout
        self.predict_window_samples = 0

    def window_samples(self, _sample_interval) -> int:
        """Forecast window drawn by the last :meth:`get_masks` call.

        Args:
            _sample_interval: Sample interval of the current sample in minutes (unused; the draw fixed it).

        Returns:
            int: Forecast samples.
        """
        return self.predict_window_samples

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

    def __init__(
        self,
        dataset,
        max_context_samples,
        max_retries,
        min_mask_size,
        n_time_features,
        covariate_dropout,
        mask_fraction=0.3,
        predict_dim=0,
    ):
        """Initialize the RandomMaskingTaskDataset.

        Args:
            dataset: The dataset whose samples are to be masked.
            max_context_samples: maximum context witdth.
            max_retries: Number of retries at a different index if data is not valid.
            min_mask_size: The min_mask_size in samples.
                       Should be a multiple of the model's `patch_size`.
            mask_fraction: The fraction of mask samples.
            predict_dim: The target dimension.
            n_time_features: The number of time features (1 - unix time; 3 - unix time, lat, lon).
            covariate_dropout: Boolean, if training with covariate dropout.
        """
        super().__init__(dataset, max_context_samples, max_retries)
        self.min_mask_size = min_mask_size
        self.mask_fraction = mask_fraction
        self.predict_dim = predict_dim
        self.n_time_features = n_time_features
        self.covariate_dropout = covariate_dropout

    def get_masks(self, sample, _sample_interval):
        """Get the input and output masks with random masking.

        Args:
            sample: The sample for which to get the masks.
            _sample_interval: The sample interval in minutes (unused for random masking).

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


class ContiguousPatchMaskingTaskDataset(PredictionTaskDataset):
    """Prediction task with Contiguous Patch Masking (CPM) of the target inside the context.

    Follows TiRex (Auer et al., 2025). Per sample a block length ``continuous_mask ~ U{continuous_min, ...,
    continuous_max}`` (in model patches) and a masking probability ``p_mask ~ U(0, p_max)`` are drawn. The
    visible context of the target series is cut into blocks of ``continuous_mask * patch_size`` samples,
    aligned to the patch grid counted from the end
    of the sample, and each block is masked with probability ``p_mask``. Masked blocks are hidden from the input
    mask and added to the target mask, so the loss covers them. Neighbouring masked blocks merge into longer runs.

    The fixed forecast window at the end of the sample is always masked, exactly as in
    :class:`PredictionTaskDataset`. Covariates and time features are never masked by CPM.
    """

    def __init__(
        self,
        dataset,
        max_context_samples,
        max_retries,
        predict_dim,
        predict_window_samples,
        n_time_features,
        covariate_dropout,
        *,
        patch_size,
        p_max=0.25,
        continuous_min=1,
        continuous_max=24,
        max_redraws=5,
        min_context_patches=0,
        base_sample_interval_minutes=None,
        patches_by_rate=None,
        predict_width_range=None,
    ):
        """Initialize the ContiguousPatchMaskingTaskDataset.

        Args:
            dataset: The underlying dataset to wrap.
            max_context_samples: The max number of samples in a sample.
            max_retries: Number of retries at a different index if data is not valid.
            predict_dim: The dimension to predict.
            predict_window_samples: The window size for prediction.
            n_time_features: The number of time features (1 - unix time; 3 - unix time, lat, lon).
            covariate_dropout: Boolean, if training with covariate dropout.
            patch_size: Model patch size in samples; mask blocks are multiples of it.
            p_max: Upper bound of the per-sample block masking probability, in [0, 1].
            continuous_min: Lower bound of the number of consecutive patches per block, at least 1.
            continuous_max: Upper bound of the number of consecutive patches per block, at least continuous_min.
            max_redraws: Redraws of the CPM mask when it would hide the whole visible context of the target.
            min_context_patches: Leading patches of the sample that are never masked, so under
                causal attention every masked block keeps at least this much visible history to
                predict from. 0 disables the exclusion.
            base_sample_interval_minutes: See :class:`PredictionTaskDataset`.
            patches_by_rate: Optional mapping sample rate (minutes) -> (continuous_min, continuous_max,
                min_context_patches)
                that overrides the fixed patch counts per rate, so block durations stay comparable under
                multi-rate training.
            predict_width_range: Optional (min, max) fraction of the sample length. When set, the forecast
                window is drawn per sample from this range (at least one patch, leaving at least one patch of
                context) instead of using ``predict_window_samples``, which makes the horizon independent of
                the sample rate and trains the model on a spread of horizons.
        """
        super().__init__(
            dataset,
            max_context_samples,
            max_retries,
            predict_dim,
            predict_window_samples,
            n_time_features,
            covariate_dropout,
            base_sample_interval_minutes=base_sample_interval_minutes,
        )
        self.patch_size = int(patch_size)
        self.p_max = float(p_max)
        self.continuous_min = int(continuous_min)
        self.continuous_max = int(continuous_max)
        self.max_redraws = max(1, int(max_redraws))
        self.min_context_samples = int(min_context_patches) * self.patch_size
        self.patches_by_rate = patches_by_rate
        if predict_width_range is not None:
            low, high = (float(x) for x in predict_width_range)
            if not 0.0 < low <= high < 1.0:
                raise ValueError("CPM predict_width_range must satisfy 0 < min <= max < 1.")
            predict_width_range = (low, high)
        self.predict_width_range = predict_width_range
        self._drawn_window: int | None = None

    def draw_window(self, n_samples: int) -> int:
        """Draw the forecast window for one sample from ``predict_width_range``.

        Args:
            n_samples: Length of the sample.

        Returns:
            int: Forecast samples, between one patch and the sample length minus one patch.
        """
        low, high = self.predict_width_range  # type: ignore[misc]
        fraction = low + float(torch.rand(()).item()) * (high - low)
        n_predict = int(n_samples * fraction)
        return int(min(max(n_predict, self.patch_size), max(self.patch_size, n_samples - self.patch_size)))

    def window_samples(self, sample_interval) -> int:
        """Forecast window of the current sample: the drawn one when a fraction range is configured.

        Args:
            sample_interval: Sample interval of the current sample in minutes.

        Returns:
            int: Forecast samples.
        """
        if self.predict_width_range is not None and self._drawn_window is not None:
            return self._drawn_window
        return super().window_samples(sample_interval)

    def block_bounds(self, sample_interval) -> tuple[int, int, int]:
        """Block-length bounds (patches) and protected history (samples) for a sample rate.

        Args:
            sample_interval: Sample interval of the current sample in minutes; None uses the fixed counts.

        Returns:
            tuple[int, int, int]: (continuous_min, continuous_max, min_context_samples).
        """
        if self.patches_by_rate is None or sample_interval is None:
            return self.continuous_min, self.continuous_max, self.min_context_samples
        continuous_min, continuous_max, min_context_patches = self.patches_by_rate[int(sample_interval)]
        return continuous_min, continuous_max, min_context_patches * self.patch_size

    def sample_context_mask(self, n_samples: int, n_context: int, sample_interval=None) -> torch.Tensor:
        """Draw one CPM mask over the context of a sample.

        Blocks are aligned to the patch grid anchored at the end of the sample, so the grid is unchanged by the
        zero padding that is later prepended in :meth:`TaskDataset.zero_pad`. A partial patch shared between the
        context and the forecast window is never masked.

        Args:
            n_samples: Total length of the sample (context plus forecast window).
            n_context: Number of context samples, i.e. samples before the forecast window.
            sample_interval: Sample interval in minutes, selecting per-rate block bounds when configured.

        Returns:
            torch.Tensor: Boolean tensor of shape [n_context], True where the target is masked.
        """
        mask = torch.zeros(n_context, dtype=torch.bool)
        continuous_min, continuous_max, min_context_samples = self.block_bounds(sample_interval)

        continuous_mask = int(torch.randint(continuous_min, continuous_max + 1, ()).item())
        p_mask = float(torch.rand(()).item()) * self.p_max
        block = continuous_mask * self.patch_size

        # Drop the context samples that fall in a patch shared with the forecast window, so every block
        # boundary lies on the patch grid anchored at the end of the sample.
        n_forecast = n_samples - n_context
        shared = (-n_forecast) % self.patch_size
        n_usable = n_context - shared
        # Blocks stay end-anchored on the patch grid; the leading `min_context_samples` are simply
        # not maskable, so under causal attention no block has to be predicted from (almost) no
        # history - those near-impossible targets were a loss spike source.
        maskable = n_usable - min_context_samples
        n_blocks = maskable // block
        if n_blocks <= 0 or p_mask <= 0.0:
            return mask

        masked_blocks = torch.rand(n_blocks) < p_mask
        start = n_usable - n_blocks * block
        mask[start:n_usable] = masked_blocks.repeat_interleave(block)
        return mask

    def get_masks(self, sample, sample_interval):
        """Get the input and output masks: fixed forecast window plus contiguous patch masks in the context.

        Args:
            sample: The sample for which to get the masks, shape [T, F].
            sample_interval: The sample interval in minutes; sets the forecast window length.

        Returns:
            tuple: A tuple containing the input mask and output mask.
        """
        if self.predict_width_range is not None:
            self._drawn_window = self.draw_window(sample.shape[0])
        xm, ym = super().get_masks(sample, sample_interval)

        n_samples = sample.shape[0]
        n_context = n_samples - self.window_samples(sample_interval)
        if n_context <= 0:
            return (xm, ym)

        visible = xm[:n_context, self.predict_dim].bool()
        cpm = torch.zeros(n_context, dtype=torch.bool)
        for _ in range(self.max_redraws):
            candidate = self.sample_context_mask(n_samples, n_context, sample_interval)
            if (visible & ~candidate).any():
                cpm = candidate
                break

        idx = cpm.nonzero().squeeze(-1)
        if idx.numel() == 0:
            return (xm, ym)

        xm[idx, self.predict_dim] = 0
        # Only real (non-NaN) values can be loss targets.
        ym[idx, self.predict_dim] = (~torch.isnan(sample[idx, self.predict_dim])).to(ym.dtype)
        return (xm, ym)
