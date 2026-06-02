# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

from math import pi

import torch
from einops import rearrange
from pydantic import NonNegativeInt
from torch import nn
from torch.nn import functional as F

from s4casting.model._blocks import GruBlock, S4Block, S6Block, SequenceResidualBlock

DAY = 86_400
WEEK = 604_800
YEAR = int(DAY * 365.25)
EPOCH_2012 = 1_325_376_000
NL_CENTER = (52.15, 5.25)


class SSEncoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        n_features,
        n_layers,
        patch_size,
        kernel="s6",
        backend="keops",
    ):
        super().__init__()
        self.expand = nn.Linear(n_features, latent_dim)
        _kernel = {"s4": S4Block, "s6": S6Block, "gru": GruBlock}[kernel]
        self.ss_layers = nn.ModuleList([
            SequenceResidualBlock(latent_dim, _kernel, backend=backend) for _ in range(n_layers)
        ])

    def forward(self, x, input_interval, output_interval):
        x = self.expand(x)
        for layer in self.ss_layers:
            x = layer(x, input_interval / output_interval)
        return x


class PatchEncoder(nn.Module):
    """Simple linear patch encoder."""

    def __init__(
        self,
        latent_dim: NonNegativeInt,
        n_features: NonNegativeInt,
        patch_size: NonNegativeInt,
    ):
        """Initialize the PatchEncoder.

        Args:
            latent_dim (NonNegativeInt): Latent dimension size.
            n_features (NonNegativeInt): Number of input features.
            patch_size (NonNegativeInt): Size of each patch.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.embed = nn.Linear(n_features * patch_size, latent_dim)

    def forward(self, x, *wargs):
        """Forward pass of the PatchEncoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_features)
            *wargs: Additional keyword arguments to catch the input/ouput rats.


        Returns:
            torch.Tensor: Encoded tensor of shape (batch_size, n_patches, latent_dim)
        """
        # do full independent connection of values in patch to embedding
        x = rearrange(x, "b (t p) f -> b t (p f)", p=self.patch_size)
        return self.embed(x)  # b t e


class PatchDecoder(nn.Module):
    """Simple linear patch decoder."""

    # TODO: does it really make sense that the decoder is in the "encoders" directory?
    def __init__(
        self,
        latent_dim: NonNegativeInt,
        out_emb: NonNegativeInt,
        patch_size: NonNegativeInt,
        input_sample_intervals_minutes: list[NonNegativeInt],
        output_sample_intervals_minutes: list[NonNegativeInt],
        arch: str = "linear",
    ):
        """Initialize the PatchDecoder.

        Args:
            latent_dim (NonNegativeInt): Latent dimension size.
            out_emb (NonNegativeInt): Output embedding size.
            patch_size (NonNegativeInt): Size of each patch.
            input_sample_intervals_minutes (NonNegativeInt): Input sample interval in minutes.
            output_sample_intervals_minutes (NonNegativeInt): Output sample interval in minutes.
            arch (str): Encoder architecture type.
        """
        super().__init__()
        # TODO: this should be if model_sample_interval is not equal
        self.passthrough = (
            sorted(input_sample_intervals_minutes) != sorted(output_sample_intervals_minutes) or arch != "linear"
        )
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.project = nn.Sequential(
            nn.LayerNorm(2 * latent_dim), nn.Linear(2 * latent_dim, out_emb * patch_size), nn.GELU()
        )

    def forward(self, x, **kwargs):
        """Forward pass of the PatchDecoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_patches, latent_dim)
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Decoded tensor of shape (batch_size, seq_len, out_emb)
        """
        if self.passthrough:
            return x
        # concat the embeddings of the current and next patch
        x = F.pad(x, pad=(0, 0, 0, 1, 0, 0), mode="constant", value=0)
        x = torch.cat([x[:, :-1, :], x[:, 1:, :]], dim=-1)

        x = self.project(x)
        return rearrange(x, "b t (p e) -> b (t p) e", p=self.patch_size)


class PatchMaskEncoder(PatchEncoder):
    """Patches a timeseries, and replaces masked values with a learnable mask embedding (or zero-embedding)."""

    def __init__(
        self,
        latent_dim: NonNegativeInt,
        n_features: NonNegativeInt,
        patch_size: NonNegativeInt,
        train_mask: bool = True,
    ):
        """Initialize the PatchMaskEncoder.

        Args:
            latent_dim (NonNegativeInt): Latent dimension size.
            n_features (NonNegativeInt): Number of input features.
            patch_size (NonNegativeInt): Size of each patch.
            train_mask (bool): Whether to train the mask embedding.
        """
        super().__init__(latent_dim, n_features, patch_size)
        self.mask_embedding = nn.Parameter(torch.zeros(latent_dim), requires_grad=train_mask)

    def forward(self, x: torch.Tensor, xm: torch.Tensor, *wargs):
        """Forward pass of the PatchMaskEncoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_features).
            xm (torch.Tensor): Mask tensor of shape (batch_size, seq_len, n_features).
            *wargs: Additional keyword arguments to catch the input/ouput rats.

        Returns:
            - x_patch (torch.Tensor): Encoded tensor with masked patches.
            Shape: (batch_size, n_patches, latent_dim).
            - xm_patch (torch.Tensor): Patch mask tensor.
            Shape: (batch_size, n_patches, latent_dim).
        """
        x_embed = super().forward(x)
        # If any of the elements of the patch, or feature were unmasked, keep the entire embedding
        # else, you will mask out weather in the prediction window.
        xm_patch = rearrange(xm, "b (t p) f -> b t p f", p=self.patch_size)
        xm_patch = xm_patch.max(dim=3, keepdim=False).values.max(dim=2, keepdim=True).values
        x_masked = torch.where(xm_patch.bool(), x_embed, self.mask_embedding)
        return x_masked, xm_patch


class TemporalEmbedding(nn.Module):
    """Temporal embedding module.

    Compose cyclic encodings (sin/cos) for
    time of day (2),
    day of week (2),
    day of year (2),
    linear trend of years since 20120101 (1).
    """

    def __init__(self, patch_size: int, latent_dim: int):
        """Initialize the TemporalEmbedding.

        Args:
            patch_size (int): Size of each patch.
            latent_dim (int): Latent dimension size.
        """
        super().__init__()
        self.latent_dim = latent_dim
        # The 7 features from above
        self.proj = nn.Linear(7, latent_dim)
        self.norm = nn.LayerNorm(7)
        torch.nn.init.zeros_(self.proj.weight)
        torch.nn.init.ones_(self.norm.weight)
        torch.nn.init.zeros_(self.norm.bias)
        self.time_pool = nn.AvgPool1d(patch_size, patch_size, ceil_mode=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Forward pass of the TemporalEmbedding.

        Args:
            t (torch.Tensor): Timestamps tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Temporal embeddings of shape (batch_size, latent_dim).
        """
        if t is None:
            return torch.zeros(self.latent_dim, device=t.device)

        angles = [2 * pi * (t % p) / p for p in [DAY, WEEK * 3, YEAR]]
        feats = [f(x) for x in angles for f in (torch.sin, torch.cos)]
        feats.append(torch.tanh(((t - EPOCH_2012) / YEAR - 6.5) / 6.5))
        x = self.norm(torch.stack(feats, dim=-1))
        return self.time_pool(self.proj(x).transpose(1, 2)).transpose(1, 2)


class SpatialEmbedding(nn.Module):
    """Spatial embedding module."""

    def __init__(self, latent_dim: int):
        """Initialize the SpatialEmbedding.

        Args:
            latent_dim (int): Latent dimension size.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.proj = nn.Linear(2, latent_dim)
        torch.nn.init.zeros_(self.proj.weight)

    def forward(self, latlon):
        """Forward pass of the SpatialEmbedding.

        Args:
            latlon (torch.Tensor): Latitude and longitude tensor of shape (batch_size, 2).

        Returns:
            torch.Tensor: Spatial embeddings of shape (batch_size x 1 x n_embd).
        """
        if latlon is None:
            return torch.zeros(self.latent_dim, device=self.geo_means.device)  # n_embd

        return self.proj(latlon - torch.tensor(NL_CENTER, device=latlon.device))  # B x 1 x n_embd


class SeperateLocTime(nn.Module):
    """Adds temporal and spatial embeddings to the encoded patches."""

    def __init__(self, encoder):
        """Initialize the SeperateLocTime.

        Args:
            encoder: Patch encoder instance.
        """
        super().__init__()
        self.latent_dim = encoder.latent_dim
        self.patch_size = encoder.patch_size
        self.temp_enc = TemporalEmbedding(encoder.patch_size, encoder.latent_dim)
        self.spat_enc = SpatialEmbedding(encoder.latent_dim)
        self.encoder = encoder

    def forward(self, x, sample_rate_conversion_factor, patch_size):
        """Forward pass of the SeperateLocTime.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_features + 3).
            sample_rate_conversion_factor (float): convert between base sample rate.
            patch_size (int): how many samples per patch.

        Returns:
            torch.Tensor: Encoded tensor of shape (batch_size, n_patches, latent_dim).
        """
        return (
            self.encoder(x[:, :, :-3], sample_rate_conversion_factor, patch_size)
            + self.temp_enc(x[:, :, -3])
            + self.spat_enc(x[:, -1:, -2:])
        )
