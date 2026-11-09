# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from s4casting.data.dataset.interface import resample_batch


def test_downsampling_takes_the_period_mean():
    """Coarser rates average every period, so identical hours give identical hourly values."""
    hour = [2, 2, 2, 8, 8, 2, 2, 2, 2, 8, 8, 2]  # one hour of 5-minute readings
    x = np.array(hour * 4, dtype=np.float32).reshape(1, -1, 1)
    np.testing.assert_allclose(resample_batch(x, 4)[0, :, 0], [4.0] * 4)
    np.testing.assert_allclose(
        resample_batch(x, 16)[0, :, 0], [2.0, 6.0, 2.0, 6.0] * 4
    )  # 15-minute means of 3 readings


def test_downsampling_ignores_gaps_up_to_the_coverage_threshold():
    """A period keeps its mean while at least half its samples are valid, and is NaN below that."""
    x = np.arange(12, dtype=np.float32).reshape(1, -1, 1)
    x[0, 0, 0] = np.nan  # one of three samples missing: mean of the other two
    x[0, 3:5, 0] = np.nan  # two of three missing: NaN
    out = resample_batch(x, 4)[0, :, 0]
    np.testing.assert_allclose(out[0], 1.5)
    assert np.isnan(out[1])
    np.testing.assert_allclose(out[2:], [7.0, 10.0])


def test_downsampling_handles_non_integer_ratios_and_several_features():
    """10-minute data at 15 minutes averages the overlapping samples; every feature is pooled alike."""
    x = np.stack([np.arange(6, dtype=np.float32), np.ones(6, dtype=np.float32)], axis=-1)[None]  # (1, 6, 2)
    out = resample_batch(x, 4)[0]
    assert out.shape == (4, 2)
    np.testing.assert_allclose(out[:, 1], 1.0)
    assert np.all(np.diff(out[:, 0]) > 0)
    np.testing.assert_allclose(out[:, 0].mean(), x[0, :, 0].mean(), atol=0.25)


def test_upsampling_still_interpolates():
    """Finer rates fill in between the known samples, as before."""
    x = np.array([0.0, 3.0, 6.0], dtype=np.float32).reshape(1, -1, 1)
    np.testing.assert_allclose(resample_batch(x, 7)[0, :, 0], np.linspace(0, 6, 7))
    assert resample_batch(x, 3) is x
