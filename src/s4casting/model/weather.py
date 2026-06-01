# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import torch
from torch import nn


class WeatherAuxTask(nn.Module):
    """Self-supervised auxiliary task that learns to forecast masked weather features.

    During training, a random trailing fraction of each sequence's weather features
    is zeroed out and the model learns to reconstruct those values from the latent
    representation. During eval (self.training=False) the mask fraction collapses to
    zero so no features are hidden and compute_loss returns a zero scalar.

    Only valid when patch_size=1 (no temporal compression), so the encoder output
    is per-timestep and the forecaster can map directly to (B, T, C).

    When n_weather_features=0 both methods are no-ops.
    """

    def __init__(self, latent_dim: int, n_weather_features: int):
        """Initialize WeatherAuxTask.

        Args:
            latent_dim: Size of the latent representation produced by the patch encoder.
            n_weather_features: Number of auxiliary weather channels (features after the
                first target feature). Set to 0 to disable.
        """
        super().__init__()
        self.n_weather_features = n_weather_features
        # max(1, ...) avoids nn.Linear(latent_dim, 0) when disabled; the layer is
        # never called when n_weather_features=0.
        self.weather_forecaster = nn.Linear(latent_dim, max(1, n_weather_features))

    def prepare(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly mask weather features and return targets for the auxiliary loss.

        Args:
            x: Normalised input tensor of shape (B, T, F).

        Returns:
            x_masked: x with a random trailing fraction of weather channels zeroed.
            weather_gt: Ground-truth values in the masked region, zero elsewhere. Shape (B, T, C).
            weather_mask: Binary float mask (1=unmasked, 0=masked). Shape (B, T, C).
        """
        B, T = x.shape[:2]
        C = self.n_weather_features

        if C == 0 or not self.training:
            empty = x.new_empty(B, T, 0)
            return x, empty, empty

        fractions = torch.rand(B, device=x.device)
        n_masked = (fractions * T).long()
        time_idx = torch.arange(T, device=x.device).view(1, T)
        weather_mask = (time_idx < (T - n_masked).unsqueeze(1)).float().unsqueeze(-1).expand(B, T, C)

        weather_gt = x[..., 1 : 1 + C].clone() * (1 - weather_mask)
        x = x.clone()
        x[..., 1 : 1 + C] = x[..., 1 : 1 + C] * weather_mask
        return x, weather_gt, weather_mask

    def compute_loss(
        self,
        x_enc: torch.Tensor,
        weather_gt: torch.Tensor,
        weather_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE over the masked weather region.

        Args:
            x_enc: Per-timestep encoder representation of shape (B, T, E).
            weather_gt: Ground-truth weather values in the masked region. Shape (B, T, C).
            weather_mask: Binary mask (1=unmasked, 0=masked). Shape (B, T, C).

        Returns:
            Scalar MSE loss over the masked region; zero when n_weather_features=0.
        """
        if self.n_weather_features == 0:
            return x_enc.new_zeros(())

        weather_forecast = self.weather_forecaster(x_enc)  # (B, T, C)
        n_masked = (1 - weather_mask).sum().clamp(min=1)
        return ((weather_forecast - weather_gt) ** 2 * (1 - weather_mask)).sum() / n_masked
