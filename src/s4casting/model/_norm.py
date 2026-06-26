# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import torch
from pydantic import NonNegativeFloat, PositiveInt, confloat

from s4casting.model._heads import GMMHead, QuantileHead


def get_statistics(x: torch.Tensor, eps: confloat(ge=0) = 1.0):  # type: ignore
    """Obtain std, mu from given x tensor.

    Ignores nan values in determining the statistics.

    Args:
        x: tensor of shape (batch_size, seq_len, n_features)
        eps: stability parameter to add to std

    Returns:
        mean: tensor of shape (batch_size, 1, n_features)
        std: tensor of shape (batch_size, 1, n_features).
    """
    mean = torch.nanmean(x, dim=1, keepdim=True)
    std = nanstd(x, dim=1, keepdim=True) + eps
    return mean, std


def norm(
    x: torch.Tensor,
    xm: torch.Tensor | None = None,
    eps: confloat(ge=0) = 1.0,  # type: ignore
    clamp: NonNegativeFloat | None = None,
    dims=None,
):
    """Normalize x over the 1st dimension.

    The normalization is done only based on the values with xm=1
    And only values where xm=1 are normalized, the others are left as is.

    Args:
        x: tensor of shape (batch_size, seq_len, n_emb)
        xm: mask tensor of shape (batch_size, seq_len, n_emb), defines which values to use for normalization
        eps: stability parameter to use. Also commonly abused to prevent exploding x with low std by setting eps to 1.0
        clamp: clamp value to use for the normalized x
        dims: dimensions of the data which statistics will be used for normalization.

    Returns:
        normalized tensor of shape (batch_size, seq_len, n_emb)
    """
    x_in = x
    if xm is None:
        xm = torch.ones_like(x)
    if dims is None:
        dims = torch.arange(x.shape[-1])
    x = torch.where(xm.bool(), x, torch.nan)
    mean, std = get_statistics(x, eps)
    x[:, :, dims] = (x[:, :, dims] - mean[:, :, dims]) / std[:, :, dims]
    x = torch.where(xm.bool(), x, x_in)
    if clamp is not None:
        x[:, :, dims] = x[:, :, dims].clamp(min=-clamp, max=clamp)
    x = torch.nan_to_num(x)
    return x, mean, std


def norm_target(
    mean_in: torch.Tensor,
    std_in: torch.Tensor,
    y: torch.Tensor,
    ym: torch.Tensor | None = None,
    clamp: NonNegativeFloat | None = None,
):
    """Normalize the target tensor y using the provided mean and std.

    Ignore the values where ym=0 when applying norm; these remain untouched.

    Args:
        mean_in: tensor of shape (batch_size, 1, n_features)
        std_in: tensor of shape (batch_size, 1, n_features)
        y: tensor of shape (batch_size, seq_len, n_features)
        ym: mask tensor of shape (batch_size, seq_len, n_features), defining which y values not to touch during norm
        clamp: clamp value to use for the normalized y

    Returns:
        normalized tensor of shape (batch_size, seq_len, n_features)
    """
    if ym is None:
        ym = torch.ones_like(y)
    y_norm = (y - mean_in) / std_in
    y = torch.where(ym.bool(), y_norm, y)
    if clamp is not None:
        y = y.clamp(min=-clamp, max=clamp)
    return torch.nan_to_num(y)


def denorm(
    mean_in: torch.Tensor,
    std_in: torch.Tensor,
    x: torch.Tensor,
    xm: torch.Tensor | None = None,
    output_type: GMMHead | QuantileHead = QuantileHead,  # type: ignore
    stat_dims: list[PositiveInt] = [0],
):
    """Denormalize the tensor x using the provided mean and std.

    Args:
        mean_in: tensor of shape (batch_size, 1, n_features)
        std_in: tensor of shape (batch_size, 1, n_features)
        x: tensor of shape (batch_size, seq_len, n_out_features, ...)
        xm: mask tensor of shape (batch_size, seq_len, n_out_features), defining which x values not to
            touch during denorm
        output_type: type of output head used, determines denorm method
        stat_dims: dimensions of the data which statistics will be used for denormalization.

    Returns:
        denormalized tensor of shape (batch_size, seq_len, n_out_features, ...)
    """
    # Denormalize a timeseries based on predetermined mean_in and std_in
    # Ignore the mask=0 values for denormalization
    if xm is None:
        xm = torch.ones_like(x)

    # TODO: not the nicest, but doing it like this because transformer still needs denorm_GMM
    if output_type == GMMHead:
        x[..., 1], x[..., 2] = denorm_GMM(mean_in, std_in, x[..., 1], x[..., 2])
    elif output_type == QuantileHead:
        x = x * std_in + mean_in
    return x


def denorm_GMM(
    mean_in: torch.Tensor,
    std_in: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    stat_dims: list[PositiveInt] = [0],
):
    """Denormalize std and mu parameters.

    Note: pi is not changed by denormalization.
    Make sure you select the right dimensions of the mean_in and std_in tensors before passing to this functions. E.g.,
    if you are forecasting measurements, and measurements are the zeroth dimensions of the input data -> select the
    zeroth dimension of the mean_in and std_in tensors (but keep singleton dimensions).

    Args:
        mean_in: tensor of shape (batch_size, 1, n_out_features, 1)
        std_in: tensor of shape (batch_size, 1, n_out_features, 1)
        sigma: tensor of shape (batch_size, seq_len, n_out_features, n_gaussian)
        mu: tensor of shape (batch_size, seq_len, n_out_features, n_gaussian)
        stat_dims: dimensions of the data which statistics will be used for denormalization.

    Returns:
        sigma: tensor of shape (batch_size, seq_len, n_out_features, n_gaussian)
        mu: tensor of shape (batch_size, seq_len, n_out_features, n_gaussian)
    """
    mu = mu * std_in + mean_in
    sigma *= std_in
    return sigma, mu


def nanstd(x: torch.Tensor, dim: PositiveInt | None = None, keepdim=False, correction=1):
    """Calculate the standard deviation of x along dimension dim, ignoring nan values.

    See https://pytorch.org/docs/stable/generated/torch.std.html.

    Args:
        x: tensor
        dim: dimension along which to calculate the standard deviation
        keepdim: whether to keep the dimension which we calculate the standard deviation over
        correction: correction factor for unbiased standard deviation, see pytorch docs

    Returns:
        tensor of shape ()
    """
    x_var = (x - torch.nanmean(x, dim=dim, keepdim=True)) ** 2
    N = torch.sum(~torch.isnan(x_var), dim=dim, keepdim=keepdim)
    var = 1 / (N - correction).clip(0) * torch.nansum(x_var, dim=dim, keepdim=keepdim)
    return torch.sqrt(var)
