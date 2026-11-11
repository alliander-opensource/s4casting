# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import pytest
import torch
from torch.utils.data import ConcatDataset

import s4casting.core.config as cfg
from s4casting.core.batcher import Batcher, SingleRateBatchSampler, pads_to_fixed_length
from s4casting.core.tasks import ContiguousPatchMaskingTaskDataset, PredictionTaskDataset
from s4casting.data.utils import SampleConfig, build_valid_context_sampling_pairs
from s4casting.factories.machine import provide_machine
from s4casting.factories.model_container import provide_model_container

PATCH = 8
PREDICT = 3 * PATCH
N_FEATURES = 4  # target, one covariate, two more columns


def _cpm_task(**overrides) -> ContiguousPatchMaskingTaskDataset:
    kwargs = {
        "dataset": [],
        "max_context_samples": 0,
        "max_retries": 1,
        "predict_dim": 0,
        "predict_window_samples": PREDICT,
        "n_time_features": 0,
        "covariate_dropout": False,
        "patch_size": PATCH,
        "p_max": 1.0,
        "continuous_min": 1,
        "continuous_max": 2,
    }
    kwargs.update(overrides)
    return ContiguousPatchMaskingTaskDataset(**kwargs)


def _runs(mask: torch.Tensor) -> list[tuple[int, int]]:
    """Return [start, end) index pairs of the True runs in a 1-D boolean mask."""
    padded = torch.cat([torch.tensor([False]), mask.bool(), torch.tensor([False])]).int()
    edges = torch.nonzero(padded[1:] - padded[:-1]).flatten().tolist()
    return list(zip(edges[::2], edges[1::2]))


class _ListDataset:
    """Minimal dataset returning (sample, SampleConfig) like TimeseriesDataset."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx].clone()
        return sample, SampleConfig(0, 15, sample.shape[0], 0.0, sample.shape[1])


def test_cpm_masks_respect_the_window_the_grid_and_nans():
    """Tail masked, only the target touched, whole patches on the end-anchored grid, NaNs excluded, context kept.

    The sample length is not a multiple of the patch size, so the grid must count from the end of the sample.
    """
    torch.manual_seed(0)
    task = _cpm_task()
    n_samples = 30 * PATCH + 5  # not a multiple of the patch size: the grid must count from the end
    n_context = n_samples - PREDICT
    sample = torch.randn(n_samples, N_FEATURES)
    nan_rows = torch.arange(0, n_context, 7)  # NaNs in the context; the tail must stay fully masked
    sample[nan_rows, 0] = float("nan")
    masked_context = False
    for _ in range(100):
        xm, ym = task.get_masks(sample, 15)
        assert torch.all(xm[-PREDICT:, 0] == 0)
        assert torch.all(ym[-PREDICT:, 0] == 1)
        assert torch.all(xm[:, 1:] == 1)
        assert torch.all(ym[:, 1:] == 0)
        assert torch.all(xm[nan_rows, 0] == 0)
        assert torch.all(ym[nan_rows, 0] == 0)
        assert xm[:n_context, 0].sum() > 0
        # the drawn blocks (before NaN exclusion splits them) are whole patches on the grid counted from the end
        for start, end in _runs(task.sample_context_mask(n_samples, n_context)):
            assert (n_samples - start) % PATCH == 0
            assert (n_samples - end) % PATCH == 0
            masked_context = True
    assert masked_context


def test_p_max_zero_matches_the_prediction_task():
    """With p_max = 0 the masks are identical to those of PredictionTaskDataset."""
    plain = PredictionTaskDataset([], 0, 1, 0, PREDICT, 0, False)
    sample = torch.randn(30 * PATCH, N_FEATURES)
    sample[10:14, 0] = float("nan")
    xm_ref, ym_ref = plain.get_masks(sample, 15)
    xm, ym = _cpm_task(p_max=0.0).get_masks(sample, 15)
    assert torch.equal(xm, xm_ref)
    assert torch.equal(ym, ym_ref)


def test_masked_fraction_tracks_p_max():
    """With single-patch blocks the expected masked fraction of the context is p_max / 2 (p_mask ~ U(0, p_max))."""
    torch.manual_seed(1)
    task = _cpm_task(p_max=0.6, continuous_min=1, continuous_max=1)
    n_samples, n_context = 40 * PATCH, 40 * PATCH - PREDICT
    fractions = torch.stack([task.sample_context_mask(n_samples, n_context).float().mean() for _ in range(1000)])
    assert abs(fractions.mean().item() - 0.3) < 0.03


def test_getitem_keeps_masks_on_the_grid_after_padding():
    """Through __getitem__ (rejection sampling, zero padding) masked runs still sit on the padded patch grid."""
    torch.manual_seed(2)
    n_samples, padded = 30 * PATCH + 4, 40 * PATCH
    task = _cpm_task(dataset=_ListDataset([torch.randn(n_samples, N_FEATURES)]), max_context_samples=padded)
    (x, xm, y, ym), config = task[0]
    assert x.shape == (padded, N_FEATURES)
    assert config.predict_window_samples == PREDICT
    assert torch.all(xm[: padded - n_samples] == 0)
    assert torch.all(x[xm == 0] == 0)
    assert torch.all(y[ym == 0] == 0)
    for start, end in _runs(ym[: padded - PREDICT, 0]):
        assert start % PATCH == 0
        assert end % PATCH == 0


def test_transformer_trains_on_cpm_masks(tmp_path):
    """The transformer gives a finite loss and gradients when context patches are masked."""
    torch.manual_seed(3)
    model_config = cfg.ModelConfiguration(
        model="transformer",
        input_sample_intervals_minutes=[15],
        output_sample_intervals_minutes=[
            15
        ],  # equal to the input intervals, otherwise the decoder passes patches through
        transformer=cfg.TransformerConfiguration(latent_dim=32, n_heads=4, n_layers=1),
        output_head=cfg.OutputHeadConfiguration(arch="gmm", n_gaussians=2),
        patch_encoder=cfg.PatchEncoderConfiguration(patch_size=PATCH),
        patch_decoder=cfg.PatchDecoderConfiguration(patch_size=PATCH),
    )
    io_config = cfg.IOConfiguration(
        feature_order=["main", "time"],
        features={
            "main": cfg.DatasetConfiguration(location=str(tmp_path / "main"), loader="parquet"),
            "time": cfg.DatasetConfiguration(location="", loader="time"),
        },
        output=str(tmp_path / "output"),
    )
    machine = provide_machine(cfg.MachineConfiguration(device_kind="cpu"), rng_base_seed=0)
    model = provide_model_container(model_config, io_config=io_config, machine=machine).raw_model

    n_samples = 24 * PATCH
    time = torch.stack(
        [1.7e9 + 900.0 * torch.arange(n_samples), torch.full((n_samples,), 52.0), torch.full((n_samples,), 5.0)], -1
    )
    sample = torch.cat([torch.randn(n_samples, 1), time], dim=-1)
    xm, ym = _cpm_task(n_time_features=1).get_masks(sample, 15)
    x, y = (sample * xm)[None], (sample * ym)[None]
    x[..., -3:] = time  # the task never masks time features
    interval = torch.tensor([15])

    _, loss = model(x, xm[None], interval, interval, y, ym[None])
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_batcher_builds_cpm_task_from_config():
    """task='cpm' is wired to ContiguousPatchMaskingTaskDataset with the configured parameters."""
    task = Batcher._get_task_dataset(
        "cpm", [], 3072, 1, 1440, 15, 2,
        n_time_features=1, covariate_dropout=True, patch_size=PATCH,
      cpm=cfg.CPMConfiguration(p_max=0.4, continuous_min_days=1 / 6, continuous_max_days=0.5),
    )  # fmt: skip
    assert isinstance(task, ContiguousPatchMaskingTaskDataset)
    # 1/6 and 1/2 day are 2 and 6 patches of 8 at 15 minutes
    assert (task.patch_size, task.p_max, task.continuous_min, task.continuous_max) == (PATCH, 0.4, 2, 6)
    assert task.predict_window_samples == 2 * 24 * 60 // 15
    assert task.covariate_dropout is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"predict_width": 0.3, "patch_size": PATCH, "cpm": cfg.CPMConfiguration()}, "must be an int or"),
        ({"predict_width": 2, "patch_size": None, "cpm": cfg.CPMConfiguration()}, "requires patch_size"),
        ({"predict_width": 2, "patch_size": PATCH, "cpm": None}, "requires patch_size"),
    ],
)
def test_batcher_rejects_incomplete_cpm_setup(kwargs, message):
    """The batcher refuses a cpm task with a bare float predict width, no patch size, or no CPM settings."""
    with pytest.raises(ValueError, match=message):
        Batcher._get_task_dataset(
            "cpm",
            [],
            3072,
            1,
            1440,
            15,
            kwargs["predict_width"],
            n_time_features=1,
            covariate_dropout=False,
            patch_size=kwargs["patch_size"],
            cpm=kwargs["cpm"],
        )


def test_validation_task_falls_back_to_prediction_unless_requested():
    """Validation uses 'prediction' for a cpm run unless apply_to_validation is set; other tasks pass through."""
    cpm = cfg.TrainingConfiguration(task="cpm", cpm=cfg.CPMConfiguration())
    assert Batcher.validation_task(cpm) == "prediction"
    cpm.cpm.apply_to_validation = True
    assert Batcher.validation_task(cpm) == "cpm"
    assert Batcher.validation_task(cfg.TrainingConfiguration(task="randomprediction")) == "randomprediction"


def test_cpm_configuration_validation():
    """CPMConfiguration enforces its bounds."""
    with pytest.raises(ValueError, match="continuous_min_days"):
        cfg.CPMConfiguration(continuous_min_days=5, continuous_max_days=4)
    with pytest.raises(ValueError, match="p_max"):
        cfg.CPMConfiguration(p_max=1.5)
    with pytest.raises(ValueError, match="continuous_max_days"):
        cfg.CPMConfiguration(continuous_max_days=0)


@pytest.mark.parametrize(
    ("sample_interval", "expected"),
    [(15, 192), (30, 96), (60, 48), (1440, 2), (10080, 1)],
)
def test_prediction_window_follows_the_sample_rate(sample_interval, expected):
    """With a base rate the 2-day window is rescaled to each sample's rate; without one it is fixed."""
    task = PredictionTaskDataset(
        [], 0, 1, 0, 192, n_time_features=0, covariate_dropout=False, base_sample_interval_minutes=15
    )
    assert task.window_samples(sample_interval) == expected
    sample = torch.randn(max(expected + 8, 16), 2)
    xm, ym = task.get_masks(sample, sample_interval)
    assert int(ym[:, 0].sum()) == expected
    assert int((xm[:, 0] == 0).sum()) == expected
    fixed = PredictionTaskDataset([], 0, 1, 0, 192, n_time_features=0, covariate_dropout=False)
    assert fixed.window_samples(sample_interval) == 192


def test_cpm_uses_the_rate_scaled_window():
    """CPM keeps the whole rescaled forecast window masked and only adds blocks inside the remaining context."""
    task = _cpm_task(
        p_max=1.0,
        continuous_min=1,
        continuous_max=1,
        patch_size=2,
        base_sample_interval_minutes=15,
        predict_window_samples=192,
    )
    sample = torch.randn(32, 1)
    xm, ym = task.get_masks(sample, 1440)  # 2-day window is 2 samples at daily rate
    assert ym[-2:, 0].tolist() == [1.0, 1.0]
    assert (xm[-2:, 0] == 0).all()
    assert ym[:-2, 0].sum() < 30  # some context stays visible


def test_valid_predict_window_rejects_windows_that_leave_no_context():
    """A forecast window that is empty or covers the whole sample is invalid instead of crashing nanquantile."""
    task = PredictionTaskDataset([], 0, 1, 0, 4, n_time_features=0, covariate_dropout=False)
    sample = torch.randn(4, 1)
    assert not task.valid_predict_window(sample, 4)
    assert not task.valid_predict_window(sample, 0)
    assert task.valid_predict_window(torch.ones(16, 1), 4)


def test_single_rate_batch_sampler_never_mixes_groups():
    """Every batch stays inside one group, groups are exhausted, and DDP ranks split the batches disjointly."""
    lengths = [10, 7, 3]
    sampler = SingleRateBatchSampler(lengths, 4, drop_last=False, seed=1)
    batches = list(sampler)
    assert len(batches) == len(sampler) == 3 + 2 + 1
    bounds = [(0, 10), (10, 17), (17, 20)]
    for batch in batches:
        assert any(all(lo <= i < hi for i in batch) for lo, hi in bounds)
    assert sorted(i for b in batches for i in b) == list(range(20))

    strict = SingleRateBatchSampler(lengths, 4, drop_last=True, seed=1)
    assert len(list(strict)) == len(strict) == 2 + 1 + 0

    ranks = [list(SingleRateBatchSampler(lengths, 4, drop_last=False, seed=1, num_replicas=2, rank=r)) for r in (0, 1)]
    assert len(ranks[0]) == len(ranks[1]) == 3
    assert not {i for b in ranks[0] for i in b} & {i for b in ranks[1] for i in b}

    sampler.set_epoch(1)
    assert list(sampler) != batches


def _dropout_task(**overrides) -> PredictionTaskDataset:
    task = PredictionTaskDataset([], None, 1, 0, PREDICT, n_time_features=0, covariate_dropout=True)
    task.covariate_dropout_all_p = 0.0
    for key, value in overrides.items():
        setattr(task, key, value)
    return task


def test_horizon_dropout_hides_kept_covariates_only_over_the_forecast_tail():
    """A kept covariate stays visible in the context and loses a tail of the forecast window; target untouched."""
    torch.manual_seed(3)
    task = _dropout_task(covariate_horizon_dropout_p=1.0)
    n_samples, n_covariates = 10 * PATCH, 4
    n_context = n_samples - PREDICT
    cuts: set[int] = set()
    for _ in range(100):
        sample = torch.randn(n_samples, 1 + n_covariates)
        xm, ym = task.get_masks(sample, 15)
        ym_before = ym.clone()
        _, xm, ym = task.drop_covariates(sample, xm.clone(), ym, PREDICT)
        assert torch.equal(ym, ym_before)  # hiding a covariate adds no loss target
        assert torch.all(xm[:n_context, 0] == 1)
        assert torch.all(xm[n_context:, 0] == 0)
        for channel in range(1, 1 + n_covariates):
            column = xm[:, channel]
            if column.sum() == 0:
                continue  # dropped over the whole sample by the ordinary draw
            assert torch.all(column[:n_context] == 1)  # the context keeps every kept covariate
            horizon = column[n_context:]
            visible = int(horizon.sum())
            assert torch.all(horizon[:visible] == 1)  # visible up to one cut ...
            assert torch.all(horizon[visible:] == 0)  # ... then hidden to the end
            cuts.add(visible)
    assert 0 in cuts  # past-only covariates occur
    assert any(0 < cut < PREDICT for cut in cuts)  # and forecasts that run out partway
    assert PREDICT not in cuts  # with p = 1 every kept covariate loses at least its last sample


def test_horizon_dropout_is_off_by_default_and_without_a_window():
    """With the probability at 0, or no forecast window, kept covariates keep their horizon values."""
    torch.manual_seed(0)
    n_samples = 10 * PATCH
    for task, n_predict in [(_dropout_task(), PREDICT), (_dropout_task(covariate_horizon_dropout_p=1.0), 0)]:
        sample = torch.randn(n_samples, 4)
        xm, ym = task.get_masks(sample, 15)
        _, xm, _ = task.drop_covariates(sample, xm, ym, n_predict)
        for channel in range(1, 4):
            assert xm[:, channel].sum() in (0, n_samples)  # whole-sample keep or drop only


def test_getitem_passes_the_forecast_window_to_the_horizon_dropout():
    """Through __getitem__ the cut lands on the sample's own forecast window."""
    torch.manual_seed(5)
    n_samples = 10 * PATCH
    dataset = _ListDataset([torch.randn(n_samples, 4) for _ in range(4)])
    task = PredictionTaskDataset(dataset, None, 1, 0, PREDICT, n_time_features=0, covariate_dropout=True)
    task.covariate_dropout_all_p = 0.0
    task.covariate_horizon_dropout_p = 1.0
    seen_cut = False
    for _ in range(40):
        (_, xm, _, _), config = task[0]
        assert config.predict_window_samples == PREDICT
        for channel in range(1, 4):
            if xm[:, channel].sum() > 0:
                assert torch.all(xm[: n_samples - PREDICT, channel] == 1)
                seen_cut |= bool(xm[-1, channel] == 0)
    assert seen_cut


def test_cpm_block_sizes_in_days_resolve_per_rate():
    """Durations become patch counts per rate with a floor of one patch, and 0 history stays 0."""
    cpm = cfg.CPMConfiguration(continuous_max_days=2, min_context_days=2)
    assert cpm.patches_by_rate(8, [15, 60, 1440]) == {15: (1, 24, 24), 60: (1, 6, 6), 1440: (1, 1, 0)}
    assert cfg.CPMConfiguration(continuous_max_days=2, continuous_min_days=1).patches_by_rate(8, [15]) == {
        15: (12, 24, 0)
    }
    assert cfg.CPMConfiguration().patches_by_rate(8, [15]) == {15: (1, 24, 0)}  # defaults: 2 days, no history
    with pytest.raises(ValueError, match="must not exceed continuous_max_days"):
        cfg.CPMConfiguration(continuous_max_days=1, continuous_min_days=2)

    task = _cpm_task(
        p_max=1.0,
        patches_by_rate={15: (1, 24, 24), 1440: (1, 1, 0)},
        predict_window_samples=192,
        base_sample_interval_minutes=15,
    )
    torch.manual_seed(0)
    _, ym = task.get_masks(torch.randn(64, 1), 1440)  # daily: 1-patch blocks, no protected history
    assert int(ym[-2:, 0].sum()) == 2
    assert int(ym[:-2, 0].sum()) > 0  # CPM masks now exist at daily rate


def test_cpm_draws_the_forecast_window_from_a_fraction_range():
    """With a fraction range the window is drawn per sample within [min, max] of its length, one patch minimum."""
    task = _cpm_task(p_max=0.0, patch_size=4, predict_width_range=(0.1, 0.3))
    torch.manual_seed(3)
    windows = set()
    for n_samples in (64, 128):
        for _ in range(20):
            _, ym = task.get_masks(torch.randn(n_samples, 1), 15)
            n_predict = int(ym[:, 0].sum())
            assert 0.1 * n_samples - 1 <= n_predict <= 0.3 * n_samples
            assert task.window_samples(15) == n_predict
            windows.add(n_predict)
    assert len(windows) > 3
    # tiny samples still keep one patch of context and one patch of horizon
    _, ym = task.get_masks(torch.randn(8, 1), 15)
    assert int(ym[:, 0].sum()) == 4
    with pytest.raises(ValueError, match="0 < min <= max < 1"):
        _cpm_task(predict_width_range=(0.5, 0.2))
    from_batcher = Batcher._get_task_dataset(
        "cpm",
        [],
        3072,
        1,
        1440,
        15,
        [0.05, 0.3],
        n_time_features=1,
        covariate_dropout=False,
        patch_size=PATCH,
        cpm=cfg.CPMConfiguration(),
    )
    assert from_batcher.predict_width_range == (0.05, 0.3)
    with pytest.raises(ValueError, match="must be an int"):
        Batcher._get_task_dataset(
            "prediction", [], 3072, 1, 1440, 15, [0.05, 0.3], n_time_features=1, covariate_dropout=False
        )


def test_rejection_sampling_redraws_within_the_same_rate_group():
    """A rejected sample is replaced from its own sub-dataset, never from another (context, rate) group."""

    class _Fixed(torch.utils.data.Dataset):
        def __init__(self, n):
            self.n = n

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            return idx

    task = PredictionTaskDataset(ConcatDataset([_Fixed(10), _Fixed(7), _Fixed(3)]), 0, 1, 0, 4, 0, False)
    torch.manual_seed(0)
    for idx, (low, high) in [(3, (0, 10)), (10, (10, 17)), (16, (10, 17)), (19, (17, 20))]:
        draws = {task.redraw_index(idx) for _ in range(200)}
        assert draws <= set(range(low, high))
        assert len(draws) > 1
    plain = PredictionTaskDataset(_Fixed(5), 0, 1, 0, 4, 0, False)
    assert {plain.redraw_index(0) for _ in range(100)} <= set(range(5))


def test_padding_is_optional_and_only_for_the_ssm():
    """Without a padded length samples keep their own length; the batcher pads only for the SSM."""
    task = PredictionTaskDataset([], None, 1, 0, 4, 0, False)
    X = torch.randn(16, 2)
    padded = task.zero_pad(X, torch.ones(16, 2), torch.zeros(16, 2))
    assert padded[0].shape == (16, 2)
    assert padded[0] is X
    fixed = PredictionTaskDataset([], 24, 1, 0, 4, 0, False)
    assert fixed.zero_pad(X, torch.ones(16, 2), torch.zeros(16, 2))[0].shape == (24, 2)
    assert pads_to_fixed_length(cfg.ModelConfiguration(model="ssm"))
    assert not pads_to_fixed_length(cfg.ModelConfiguration(model="transformer"))


def test_context_pairs_not_divisible_by_patch_size_are_dropped():
    """A window whose sample count is not a multiple of the patch size is skipped, so it never reaches the encoder."""
    limits = {1440: {"min_days": 128, "max_days": 720}, 10080: {"min_days": 224, "max_days": 720}}
    with pytest.warns(UserWarning, match="not a multiple of the patch size"):
        pairs = build_valid_context_sampling_pairs(
            context_days=[364, 448],
            sample_intervals_minutes=[1440, 10080],
            patch_size=PATCH,
            interval_context_limits=limits,
        )["valid_pairs"]
    assert {(p["context_days"], p["sample_interval_minutes"]) for p in pairs} == {(448, 1440), (448, 10080)}
    assert all(p["points"] % PATCH == 0 for p in pairs)
    # without a patch size every exactly divisible pair is kept, as before
    pairs = build_valid_context_sampling_pairs(
        context_days=[364, 448], sample_intervals_minutes=[1440, 10080], interval_context_limits=limits
    )["valid_pairs"]
    assert len(pairs) == 4
