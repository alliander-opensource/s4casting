# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import random

import torch
from torch.nn.functional import max_pool1d, pad


def build_valid_context_sampling_pairs(
    context_days=(7, 10, 14, 16, 32, 64, 364),
    sample_intervals_minutes=(5, 10, 15, 30, 60, 1440, 10080),
    min_points=32,
    max_context_len=3072,
    interval_context_limits={
        5: {"min_days": 7, "max_days": 10},
        10: {"min_days": 7, "max_days": 16},
        15: {"min_days": 7, "max_days": 32},
        30: {"min_days": 7, "max_days": 64},
        60: {"min_days": 7, "max_days": 128},
        1440: {"min_days": 32, "max_days": 728},
        10080: {"min_days": 64, "max_days": 728},
    },
):
    """Build valid (context_days, sample_interval_minutes) pairs for a fixed padded model input.

    All sample intervals are specified in minutes:
        5      = 5 minutes
        10     = 10 minutes
        60     = 1 hour
        1440   = 1 day
        10080  = 1 week

    A pair is valid if:
      - the sample count is an integer
      - sample count >= min_points
      - sample count <= max_context_len (if max_context_len is provided)
      - it satisfies optional manual pruning rules in interval_context_limits

    Args:
        context_days: iterable of allowed context widths in days
        sample_intervals_minutes: iterable of allowed sample intervals in minutes
        min_points: minimum raw sample count required
        max_context_len: fixed padded input length; if None, inferred from valid pairs
        interval_context_limits: optional dict of the form
            {
                5:     {"min_days": 7,   "max_days": 10},
                10:    {"min_days": 7,   "max_days": 14},
                15:    {"min_days": 14,  "max_days": 32},
                30:    {"min_days": 16,  "max_days": 64},
                60:    {"min_days": 16,  "max_days": 64},
                1440:  {"min_days": 32,  "max_days": 364},
                10080: {"min_days": 364, "max_days": 364},
            }

    Returns:
        dict with:
            - "valid_pairs": list of dicts
            - "recommended_max_context_len": smallest exact padded length covering all valid pairs
    """
    valid = []

    for days in context_days:
        context_minutes = days * 1440

        for interval_minutes in sample_intervals_minutes:
            # Optional policy pruning
            if interval_context_limits and interval_minutes in interval_context_limits:
                limits = interval_context_limits[interval_minutes]
                if days < limits["min_days"] or days > limits["max_days"]:
                    continue

            points = context_minutes / interval_minutes

            # Must align exactly
            if abs(points - round(points)) > 1e-9:
                continue

            points = round(points)

            if points < min_points:
                continue

            if max_context_len is not None and points > max_context_len:
                continue

            valid.append({
                "context_days": days,
                "sample_interval_minutes": interval_minutes,
                "points": points,
            })

    if not valid:
        raise ValueError("No valid (context_days, sample_interval_minutes) pairs found.")

    recommended_max_context_len = max(row["points"] for row in valid)

    final_max_context_len = max_context_len if max_context_len is not None else recommended_max_context_len

    # Add padding metadata
    for row in valid:
        row["pad_left"] = final_max_context_len - row["points"]
        row["padded_length"] = final_max_context_len

    return {
        "valid_pairs": valid,
        "recommended_max_context_len": recommended_max_context_len,
    }


def nanmin(x: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Compute the minimum of a tensor while ignoring NaN values.

    Args:
        x (torch.tensor): Input tensor.
        dim (int | None): Dimension along which to compute the minimum.
            If None, computes the global minimum of all non-NaN values.

    Returns:
        torch.tensor: Minimum value(s) with NaNs ignored.
    """
    # Replace NaNs with +inf so they don't affect min
    replaced = torch.where(torch.isnan(x), torch.tensor(float("inf"), device=x.device), x)

    if dim is None:
        return replaced.min()
    return replaced.min(dim=dim).values


def nanmax(x: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Compute the maximum of a tensor while ignoring NaN values.

    Args:
        x (torch.tensor): Input tensor.
        dim (int | None): Dimension along which to compute the maximum.
            If None, computes the global maximum of all non-NaN values.

    Returns:
        torch.tensor: Maximum value(s) with NaNs ignored.
    """
    # Replace NaNs with -inf so they don't affect max
    replaced = torch.where(torch.isnan(x), torch.tensor(float("-inf"), device=x.device), x)

    if dim is None:
        return replaced.max()
    return replaced.max(dim=dim).values


def to_cpu(x: torch.Tensor):
    """Place tensor on CPU if its on GPU.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.tensor: Tensor on cpu .
    """
    if x.get_device() != -1:
        x = x.detach().cpu()
    return x


def run_in_batches(fn, B, inputs, input_rate, output_rate) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a function in batches to avoid memory issues and return loss.

    Args:
        fn (callable): Function to run in batches.
        B (int): Batch size.
        inputs (list[torch.tensor]): List of input tensors.
        input_rate (int): input sample rate of batch.
        output_rate (int): output sample rate of batch.

    Returns:
        torch.tensor: Concatenated output tensor.
        float: loss.
    """
    outputs, losses = [], torch.tensor(0.0, device=inputs[0].device)
    for i in range(0, inputs[0].shape[0], B):
        x, xm, y, ym = [x[i : i + B] for x in inputs]
        out, loss = fn(x, xm, input_rate, output_rate, y, ym)
        outputs.append(out)
        losses += loss
    return torch.cat(outputs, dim=0), losses / len(outputs)


def quantile_pool1d(x, kernel_size, stride=None, padding=0, dilation=1, quantile=0.5) -> torch.Tensor:
    """Perform 1D quantile pooling over the last dimension of input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (N, C, L)
        kernel_size (int): Size of pooling window
        stride (int): Stride of the pooling window. Defaults to kernel_size
        padding (int): Zero-padding to apply on both sides of input
        dilation (int): Dilation factor for pooling
        quantile (float): Quantile to compute (between 0 and 1)

    Returns:
        torch.Tensor: Output tensor after quantile pooling
    """
    if stride is None:
        stride = kernel_size

    # Pad input if needed
    if padding > 0:
        x = pad(x, (padding, padding), mode="constant", value=0)

    # Apply unfolding to get sliding windows: shape becomes (N, C, L_out, kernel_size)
    x_unfolded = x.unfold(dimension=2, size=kernel_size, step=stride)  # (N, C, L_out, kernel_size)

    # Apply dilation if needed
    if dilation > 1:
        x_unfolded = x_unfolded[..., ::dilation]

    # Compute quantile across kernel_size dimension
    return torch.quantile(x_unfolded, q=quantile, dim=-1)


def resample(data: torch.Tensor, patch_size, maxpool=True) -> torch.Tensor:
    """Max pool input data tensor.

    Args:
        data (torch.Tensor): Input tensor of shape (..., L, C)
        patch_size (int): Size of the pooling window
        maxpool (bool): If True, perform max pooling; if False, perform min pooling

    Returns:
        torch.Tensor: Pooled tensor

    """
    sign = 1 if maxpool else -1
    return sign * max_pool1d(sign * data[..., 0], kernel_size=patch_size).unsqueeze(-1)


def select_rate(
    input_rate: int,
    output_sample_intervals_minutes: list[int],
    transcoding: bool = False,
) -> int | torch.Tensor:
    """Randomly choose an output sample interval that is greater than or equal to the given input sample interval.

    Args:
        input_rate (int): input_sample rate for batch.
        output_sample_intervals_minutes(list[int]): Possible output sample rate.
        transcoding (bool): If input and output rates can be different.

    Returns:
        int : selected sample rate.

    Raises:
        ValueError: If no valid output sample interval exists.
    """
    if not transcoding:
        return input_rate
    valid_rates = [rate for rate in output_sample_intervals_minutes if rate >= input_rate]

    if not valid_rates:
        raise ValueError(f"No output sample interval >= input_rate ({input_rate})")

    return random.choice(valid_rates)
