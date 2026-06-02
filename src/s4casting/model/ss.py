# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import torch
from torch import nn

from s4casting.core.loss import CompositeLoss, SubsetNLLLoss
from s4casting.model._blocks import GruBlock, S4Block, S6Block, SequenceResidualBlock
from s4casting.model._encoders import PatchDecoder, PatchEncoder, SeperateLocTime
from s4casting.model._heads import GMMHead
from s4casting.model._norm import denorm, norm, norm_target
from s4casting.model.weather import WeatherAuxTask


class SSModel(nn.Module):
    """State Space-based time series forecasting model."""

    # NOTE: n_gaussian = 0 reverts back to predicting a single value per timestep
    def __init__(
        self,
        latent_dim=256,
        n_layer=4,
        kernel="s4",
        backend="keops",
        mixer_size=None,
        patch_size=1,
        n_out_features=1,
        norm_clamp=10.0,
        norm_eps=1e-5,
        loss_fn: nn.Module = SubsetNLLLoss(1, "masked"),
        composite_loss: nn.Module = CompositeLoss({"primary": 1.0}),
        output_head: nn.Module = GMMHead(256, 2, 1),
        patch_encoder: nn.Module = PatchEncoder(256, 5, 8),
        patch_decoder: nn.Module = PatchDecoder(256, 256, 8, [15], [15]),
        base_sample_interval_minutes: int = 15,
        n_weather_features: int = 0,
    ):
        """Initialize the SSModel.

        Args:
            latent_dim (int): Latent dimension size.
            n_layer (int): Number of S4/GRU layers.
            kernel (str): Kernel type ("s4", "s6", or "gru").
            backend (str): Backend to use for the kernel.
            mixer_size (int|None): Time domain mixer size.
            patch_size (int): Size of the patches for patch encoding/decoding.
            n_out_features (int): Number of output features.
            norm_clamp (float): Clamping value for normalization.
            norm_eps (float): Epsilon for norm.
            loss_fn (nn.Module): Loss function module.
            composite_loss (nn.Module): Combines named loss terms with per-component weights.
            output_head (nn.Module): Output head module.
            patch_encoder (nn.Module): Patch encoder module.
            patch_decoder (nn.Module): Patch decoder module.
            base_sample_interval_minutes (int): Base sample interval.
            n_weather_features (int): Number of auxiliary weather channels for the masking
                loss. Set to 0 to disable.
        """
        super().__init__()
        self.patch_size = patch_size
        self.kernel = kernel
        self.loss_fn = loss_fn
        self.composite_loss = composite_loss
        self.norm_clamp = norm_clamp
        self.norm_eps = norm_eps
        self.n_out_features = n_out_features
        self.patch_decoder = patch_decoder
        self.patch_encoder = patch_encoder
        self.output_head = output_head
        self.latent_dim = latent_dim
        self.base_sample_interval_minutes = base_sample_interval_minutes
        self.weather_aux = WeatherAuxTask(latent_dim, n_weather_features)

        Kernel = {"s4": S4Block, "s6": S6Block, "gru": GruBlock}[kernel]
        self.ss_layers = nn.ModuleList([
            SequenceResidualBlock(self.latent_dim, Kernel, backend=backend, mixer_size=mixer_size)
            for _ in range(n_layer)
        ])

    def forward(self, x, xm, input_interval, output_interval, y=None, ym=None):
        """Forward pass of the SSModel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_features).
            xm (torch.Tensor): Mask tensor of shape (batch_size, seq_len, n_features).
            y (torch.Tensor | None): Target tensor of shape (batch_size, seq_len, n_features).
            ym (torch.Tensor | None): Target mask tensor of shape (batch_size, seq_len, n_features).
            input_interval (torch.Tensor | float): Used for multi-rate training of state space models.
            output_interval (torch.Tensor | float): Used for multi-rate training of state space models.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, n_out_features).
            torch.Tensor | None: Loss tensor if target is provided, else None.
        """
        assert x.shape[1] % self.patch_size == 0, (
            f"Sequence length {x.shape[1]=} must be multiple of {self.patch_size=}"
        )

        x, mean_in, std_in = norm(
            x=x,
            xm=xm,
            eps=self.norm_eps,
            clamp=self.norm_clamp,
            dims=torch.arange(x.shape[-1] - 3 * isinstance(self.patch_encoder, SeperateLocTime)),
        )
        x, weather_gt, weather_mask = self.weather_aux.prepare(x)
        x = self.patch_encoder(x, input_interval, output_interval)  # B T F -> B T/P E
        x_enc = x
        for layer in self.ss_layers:
            x = layer(x, output_interval / self.base_sample_interval_minutes)

        x = self.patch_decoder(x)
        x = self.output_head(x)

        loss = None
        if y is not None:
            y = norm_target(
                mean_in=mean_in[..., : self.n_out_features], std_in=std_in[..., : self.n_out_features], y=y, ym=ym
            )
            _losses: dict[str, torch.Tensor] = {
                "primary": self.loss_fn(
                    out=x, target=y, input_interval=input_interval, output_interval=output_interval, mask=ym
                ),
                "weather": self.weather_aux.compute_loss(x_enc, weather_gt, weather_mask),
            }
            loss = self.composite_loss(_losses)

        x = denorm(
            mean_in=mean_in[..., : self.n_out_features].unsqueeze(-1),
            std_in=std_in[..., : self.n_out_features].unsqueeze(-1),
            x=x,
            output_type=type(self.output_head),  # type: ignore
            xm=xm,
        )

        return x, loss
