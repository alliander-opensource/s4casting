# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import torch
from einops import rearrange
from pydantic import NonNegativeInt
from torch import nn


class GMMHead(nn.Module):
    """Gaussian Mixture Model output head."""

    def __init__(self, latent_dim: NonNegativeInt, n_gaussian: NonNegativeInt, n_out_features: NonNegativeInt):
        """Initialize the GMMHead.

        Args:
            latent_dim (NonNegativeInt): Dimension of the latent representation.
            n_gaussian (NonNegativeInt): Number of Gaussian components.
            n_out_features (NonNegativeInt): Number of output features.
        """
        super().__init__()
        self.n_gaussian = n_gaussian
        self.logpi = nn.Sequential(nn.Linear(latent_dim, n_gaussian * n_out_features), nn.LogSoftmax(dim=-1))
        self.sigma = nn.Linear(latent_dim, n_gaussian * n_out_features)
        self.mu = nn.Linear(latent_dim, n_gaussian * n_out_features)

    def forward(self, x: torch.Tensor):
        """Forward pass of the GMMHead.

        Sigma is forced positive and trained in logspace by taking the exponent of the linear layer output

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, latent_dim)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: logpi, sigma, mu, of
                                                                shape (batch_size, seq_len, n_out_features, n_gaussian).
        """
        logpi = rearrange(self.logpi(x), "b t (f g) -> b t f g", g=self.n_gaussian)
        sigma = rearrange(torch.exp(self.sigma(x)), "b t (f g) -> b t f g", g=self.n_gaussian)
        mu = rearrange(self.mu(x), "b t (f g) -> b t f g", g=self.n_gaussian)
        return torch.stack([logpi, sigma, mu], dim=-1)


class QuantileHead(nn.Module):
    """Quantile output head."""

    def __init__(self, latent_dim: NonNegativeInt, n_out_features: NonNegativeInt, quantile_values: tuple[float]):
        """Initialize the QuantileHead.

        Args:
            latent_dim (NonNegativeInt): Dimension of the latent representation.
            n_out_features (NonNegativeInt): Number of output features.
            quantile_values (tuple[float]): Tuple of quantile values to predict.
        """
        super().__init__()
        self.quantile_values = quantile_values
        self.n_out_features = n_out_features
        self.quantile_head = nn.Linear(latent_dim, n_out_features * len(quantile_values))

    def forward(self, x: torch.Tensor):
        """Forward pass of the QuantileHead.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor of shape.
        """
        return rearrange(self.quantile_head(x), "b t (f q) -> b t f q", f=self.n_out_features)
