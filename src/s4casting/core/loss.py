# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

from typing import Literal

import torch
from einops import rearrange
from pydantic import NonNegativeFloat
from torch import nn


class SoftClipLoss(nn.Module):
    """Soft clipping loss function."""

    def __init__(self, alpha: NonNegativeFloat = 0.1):
        """Soft clipping loss function. Asymptotically reduces the loss for large values.

        The asymptote of the loss lies at 1/alpha. So large loss values will be soft clipped to a/alpha
        If alpha = 0, reduces to unaltered loss.
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, loss_in: torch.Tensor):
        """Applies soft clipping to the input loss.

        Args:
            loss_in (torch.Tensor): Input loss tensor with arbitrary shape.

        Returns:
            torch.Tensor: Tensor with same shape as loss_in.
        """
        return loss_in / (1 + self.alpha * loss_in)


class SubsetNLLLoss(nn.Module):
    """Negative log likelihood loss on a subset of the data.

    If input sample rate is not the same as as output, then we use the long term loss strategy.
    """

    def __init__(
        self,
        sigma_regularisation_factor: float = 0.0,
        mode: Literal["masked", "unmasked"] = "masked",
    ):
        """Calculates the NLL on a subset of the given data.

        mode: "masked" calculates the loss on the masked values, "unmasked" calculates the loss on the unmasked values

        Args:
            sigma_regularisation_factor (float): prevents sigma from growing too big.
            mode (Literal["masked", "unmasked"]): Mode for loss calculation.
        """
        super().__init__()
        self.mode = mode
        self.sigma_regularisation_factor = sigma_regularisation_factor

    def forward(
        self,
        out: torch.Tensor,
        target: torch.Tensor,
        input_interval: int,
        output_interval: int,
        mask: torch.Tensor | None = None,
    ):
        """Calculates the nll loss assuming a out is a tensor containing GMM parameters.

        Target and mask represent a [batch_size, seq_len] tensor representing the target for this GMM.

        Args:
            out (torch.Tensor): Tuple of (pi, sigma, mu) with shape
                (batch_size, seq_len, n_out_features, <n_gaussians> or <quantile_values>, <mu, pi, sigma> or 1).
            target (torch.Tensor): Tensor with one dimension less than the GMM parameters.
                Shape: (batch_size, seq_len, n_out_features).
            input_interval (int): Input sample rate of eval step.
            output_interval (int): Output sample rate of eval step.
            mask (torch.Tensor, optional): Tensor determining on which values to calculate the loss.
                1 = calculate, 0 = ignore. Shape: same as `target`. If no mask is supplied,
                the loss will be calculated on the entire signal.

        Returns:
            torch.Tensor: Average loss over entire signal as torch.tensor with single value
        """
        if mask is None:
            mask = torch.ones_like(target)
        mask = mask.bool()
        if self.mode == "unmasked":
            mask = ~mask

        # Suppose target is now (B, T, S, F, 1): e.g. [32, 3072, 3, 8, 1]
        # mu, sigma: (B, T, 1, 1, D) - expand from (B, T, 1, D) if needed

        # Note: clone is necessary for gradient flow
        logpi, sigma, mu = (t.unsqueeze(2).clone() for t in out.unbind(dim=-1))  # -> (B, T, 1, 1, D)

        # reshape such that a number of samples fit inside each distribution
        target = rearrange(
            target * mask,
            "b (t s) f  -> b t s f",
            s=output_interval // input_interval,
            f=target.shape[-1],
        ).unsqueeze(-1)
        mask = rearrange(
            mask,
            "b (t s) f  -> b t s f",
            s=output_interval // input_interval,
            f=mask.shape[-1],
        )

        nll = -nn.GaussianNLLLoss(reduction="none", eps=1e-6)(
            input=mu,
            target=target,
            var=sigma**2,
        )
        nll = -torch.logsumexp(logpi + nll, dim=-1)[mask]
        return nll.mean() + self.sigma_regularisation_factor * (out[..., 1].clone().pow(2).mean())


class SubsetPinballLoss(nn.Module):
    """Calculates the Pinball on a subset of the given data.

    If input sample rate is not the same as as output, then we use the long term loss strategy

    """

    def __init__(
        self,
        quantile_values: tuple[float],
        mode: Literal["masked", "unmasked"] = "masked",
    ):
        """Initializes the SubsetPinballLoss module.

        Args:
            quantile_values (tuple[float]): Quantile values for pinball loss calculation.
            mode (Literal["masked", "unmasked"], optional): Mode for loss calculation. Defaults to "masked".
        """
        super().__init__()
        self.mode = mode
        self.quantile_values = quantile_values

    def forward(
        self,
        out: torch.Tensor,
        target: torch.Tensor,
        input_interval: int,  # noqa
        output_interval: int,  # noqa
        mask: torch.Tensor | None = None,
    ):
        """Calculates the pinball loss.

        Args:
            out (torch.Tensor): Model output of shape (batch_size, seq_len, n_out_features, <quantile_values>,).
            target (torch.Tensor): Target tensor of shape (batch_size, seq_len, n_out_features).
            mask (torch.Tensor, optional): Mask tensor determining on which values to calculate the loss.
            input_interval (int): Input sample rate of eval step. (Not used)
            output_interval (int): Output sample rate of eval step. (Not used)

        Returns:
            torch.Tensor: Average loss over entire signal as torch.tensor with single value
        """
        if mask is None:
            mask = torch.ones_like(target)
        mask = mask.bool()
        if self.mode == "unmasked":
            mask = ~mask
        # TODO: This is broken, target has to be reshaped based on input and output rates.
        total_loss = 0.0
        target = target.view([target.shape[0], out.shape[1], -1])
        mask = mask.view([mask.shape[0], out.shape[1], -1])
        target = target * mask
        # Note this only applies the loss over the first output feature dimension
        for i, q in enumerate(self.quantile_values):
            errors = target - out[..., 0, i : i + 1]
            loss = torch.where(errors >= 0, q * errors, (q - 1) * errors)
            total_loss += loss[mask].mean()
        return total_loss / len(self.quantile_values)
