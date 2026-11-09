# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest
import torch

import s4casting.core.config as cfg
from s4casting.data.dataset.dataset import TimeData
from s4casting.factories.machine import provide_machine
from s4casting.factories.model_container import provide_model_container
from s4casting.model._encoders import PatchEncoder, SeperateTime  # noqa: PLC2701


def test_create_model(tmp_path):
    """Tests that the model container is created correctly with the specified configuration.

    Args:
        tmp_path (pathlib.Path): Temporary path for creating dummy dataset locations.
    """
    config = cfg.ModelConfiguration(ssm=cfg.SSMConfiguration())
    io_config = cfg.IOConfiguration(
        feature_order=["main"],
        features={"main": cfg.DatasetConfiguration(location=str(tmp_path / "main"))},
        output=str(tmp_path / "output"),
        load_checkpoint=None,
    )
    machine_config = cfg.MachineConfiguration(device_kind="cpu")
    machine = provide_machine(machine_config, rng_base_seed=0)
    model_container = provide_model_container(config, io_config=io_config, machine=machine)
    assert len(model_container.raw_model.ss_layers) == config.ssm.n_layers


def _io_config_with_time(tmp_path, n_time_features: int) -> cfg.IOConfiguration:
    return cfg.IOConfiguration(
        feature_order=["main", "time"],
        features={
            "main": cfg.DatasetConfiguration(location=str(tmp_path / "main")),
            "time": cfg.DatasetConfiguration(location="", loader="time", include_location=n_time_features == 3),
        },
        output=str(tmp_path / "output"),
        load_checkpoint=None,
    )


def _time_channels(batch: int, n_samples: int, n_time_features: int) -> torch.Tensor:
    t = torch.full((batch, n_samples, n_time_features), 1_700_000_000.0)
    t[..., 0] += torch.arange(n_samples) * 900.0
    if n_time_features == 3:
        t[..., 1] = 52.0
        t[..., 2] = 5.0
    return t


@pytest.mark.parametrize("n_time_features", [1, 3])
def test_transformer_accepts_time_only_and_time_with_location(tmp_path, n_time_features: int):
    """The transformer runs with a unix-time channel alone (1) and with latitude/longitude appended (3).

    Args:
        tmp_path (pathlib.Path): Temporary path for the dummy dataset locations.
        n_time_features (int): Channels emitted by the time loader.
    """
    patch = 8
    config = cfg.ModelConfiguration(
        model="transformer",
        input_sample_intervals_minutes=[15],
        output_sample_intervals_minutes=[15],
        transformer=cfg.TransformerConfiguration(latent_dim=32, n_heads=4, n_layers=1),
        output_head=cfg.OutputHeadConfiguration(arch="gmm", n_gaussians=2),
        patch_encoder=cfg.PatchEncoderConfiguration(patch_size=patch),
        patch_decoder=cfg.PatchDecoderConfiguration(patch_size=patch),
    )
    machine = provide_machine(cfg.MachineConfiguration(device_kind="cpu"), rng_base_seed=0)
    model = provide_model_container(config, io_config=_io_config_with_time(tmp_path, n_time_features), machine=machine)
    model = model.raw_model
    assert (model.spatial_embedding is not None) == (n_time_features == 3)

    batch, n_samples = 2, 4 * patch
    x = torch.cat([torch.randn(batch, n_samples, 1), _time_channels(batch, n_samples, n_time_features)], dim=-1)
    xm = torch.ones_like(x)
    xm[:, -patch:, 0] = 0
    y = x[..., :1].clone()
    ym = torch.zeros_like(y)
    ym[:, -patch:, 0] = 1
    interval = torch.tensor([15] * batch)
    pred, loss = model(x * xm, xm, interval, interval, y, ym)
    assert torch.isfinite(loss)
    assert pred.shape[:2] == (batch, n_samples)


@pytest.mark.parametrize("n_time_features", [1, 3])
def test_separate_time_encoder_strips_the_time_channels(n_time_features: int):
    """The SSM-side wrapper strips the trailing time channels before the patch encoder and adds their embeddings.

    Args:
        n_time_features (int): Channels emitted by the time loader.
    """
    patch, latent, n_data = 4, 16, 2
    encoder = SeperateTime(PatchEncoder(latent, n_data, patch), n_time_features)
    assert (encoder.spat_enc is not None) == (n_time_features == 3)
    batch, n_samples = 2, 3 * patch
    x = torch.cat([torch.randn(batch, n_samples, n_data), _time_channels(batch, n_samples, n_time_features)], dim=-1)
    out = encoder(x, 1.0, patch)
    assert out.shape == (batch, n_samples // patch, latent)
    with pytest.raises(ValueError, match="n_time_features"):
        SeperateTime(PatchEncoder(latent, n_data, patch), 2)


def test_time_loader_channels_follow_include_location():
    """The time loader is unix time alone, or unix time with latitude and longitude, chosen by include_location."""
    assert cfg.DatasetConfiguration(location="", loader="time").n_features == 1
    assert cfg.DatasetConfiguration(location="", loader="time", include_location=True).n_features == 3
    with pytest.raises(ValueError, match="contradicts"):
        cfg.DatasetConfiguration(location="", loader="time", n_features=3)
    with pytest.raises(ValueError, match="only valid for loader='time'"):
        cfg.DatasetConfiguration(location="x", loader="parquet", include_location=True)


def test_time_data_emits_location_only_when_asked():
    """TimeData returns the unix time alone by default and appends latitude/longitude for three features."""
    locations = {7: {"name": "loc", "lat": 52.1, "lon": 5.3}}
    start = 7 * 2**32 + 1_700_000_000
    end = start + 4 * 900
    time_only = TimeData(900, locations).read(start, end)
    with_location = TimeData(900, locations, n_features=3).read(start, end)
    assert time_only.shape == (4, 1)
    assert with_location.shape == (4, 3)
    np.testing.assert_allclose(with_location[:, 0], time_only[:, 0])
    np.testing.assert_allclose(with_location[:, 1:], [[52.1, 5.3]] * 4, rtol=1e-6)
