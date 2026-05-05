# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from s4casting.model._s4_kernel import SSMKernelDPLR


class FFTConv(nn.Module):
    """FFT-based convolutional layer using S4 kernels."""

    def __init__(
        self,
        d_model,
        transposed=False,
        backend="keops",
        **kernel_args,
    ):
        """Initialize the FFTConv layer.

        Args:
            d_model (int): Dimension of the model.
            transposed (bool): Whether the input is transposed.
            backend (str): Backend to use for S4 kernel.
            **kernel_args: Additional arguments for the S4 kernel.
        """
        super().__init__()
        self.d_model = d_model
        self.transposed = transposed
        self.D = nn.Parameter(torch.randn(1, self.d_model))
        self.kernel = SSMKernelDPLR(
            d_model=self.d_model,
            l_max=None,
            channels=1,
            backend=backend,
            **kernel_args,
        )

    def forward(self, x, rate=1.0, state=None, **kwargs):
        """Forward pass of the FFTConv layer.

        Args:
            x: (B D L) if self.transposed else (B L D).
            rate: Scaling factor for the kernel.
            state: Optional state tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            y: Output tensor of shape (B D L) if self.transposed else (B L D).
            next_state: Optional next state tensor.
        """
        # Always work with (B D L) dimension in this module
        if not self.transposed:
            x = x.transpose(-1, -2)
        L = x.size(-1)

        # Compute SS Kernel
        k, k_state = self.kernel(L=L, rate=rate, state=state)  # (C H L) (B C H L)

        k_f = torch.fft.rfft(k, n=L + L)  # (C H L)
        x_f = torch.fft.rfft(x, n=L + L)  # (B H L)
        y_f = torch.einsum("bhl,chl->bchl", x_f, k_f)
        y = torch.fft.irfft(y_f, n=L + L)[..., :L]  # (B C H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + torch.einsum("bhl,ch->bchl", x, self.D)

        if state is not None:
            y = y + k_state
            # TODO
            next_state = self.kernel.forward_state(x, state)
        else:
            next_state = None

        y = rearrange(y, "b c h l -> b (c h) l")

        if not self.transposed:
            y = y.transpose(-1, -2)
        return y, next_state


class S4Block(nn.Module):
    """S4 Block using FFTConv layer."""

    def __init__(self, d_model, backend="keops", **layer_args):
        """Initialize the S4Block.

        Args:
            d_model (int): Dimension of the model.
            backend (str): Backend to use for S4 kernel.
            **layer_args: Additional arguments for the S4 kernel.
        """
        super().__init__()
        self.d_model = d_model
        self.layer = FFTConv(
            d_model,
            transposed=False,
            backend=backend,
            **layer_args,
        )
        self.output_linear = nn.Sequential(nn.Linear(self.d_model, self.d_model * 2, bias=True), nn.GLU(dim=-1))

    def forward(self, x, lengths=None, **kwargs):
        """Forward pass of the S4Block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model
            lengths (torch.Tensor | None): Optional lengths tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            y (torch.Tensor): Output tensor of shape (batch_size, seq_len, d_model).
            state: Optional state tensor.
        """
        y, state = self.layer(x, **kwargs)
        y = F.gelu(y)
        y = self.output_linear(y)
        return y, state


class S6Block(nn.Module):
    """S6 Block using Mamba layer."""

    def __init__(self, d_model, backend="keops", **layer_args):
        """Initialize the S6Block.

        Args:
            d_model (int): Dimension of the model.
            backend (str): Backend to use for S4 kernel.
            **layer_args: Additional arguments for the S4 kernel.
        """
        if backend == "naive":
            from s4casting.model.mambacpu import Mamba  # noqa
        else:
            from s4casting.model.mamba import Mamba  # noqa

        super().__init__()
        self.d_model = d_model
        self.layer = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2, **layer_args)
        self.output_linear = nn.Sequential(nn.Linear(self.d_model, self.d_model * 2, bias=True), nn.GLU(dim=-1))
        self.mixer = nn.Linear(3072, 3072, bias=True)

    def forward(self, x, rate=1, lengths=None, **kwargs):
        """Forward pass of the S6Block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            lengths (torch.Tensor | None): Optional lengths tensor.
            rate: Scaling factor for the kernel.
            **kwargs: Additional keyword arguments.

        Returns:
            y (torch.Tensor): Output tensor of shape (batch_size, seq_len, d_model).
            state: Optional state tensor.
        """
        y = self.layer(x, rate=rate)
        y = F.gelu(y)
        y = self.output_linear(y)
        y = self.mixer(y.swapaxes(1, 2)).swapaxes(1, 2)
        return y, None


def heinsen_associative_scan_log(log_coeffs, log_values):
    """Heinsen's associative scan in log space.

    Args:
        log_coeffs (torch.Tensor): Log coefficients tensor of shape (batch_size, seq_len).
        log_values (torch.Tensor): Log values tensor of shape (batch_size, seq_len).

    Returns:
        torch.Tensor: Resulting tensor after scan operation of shape (batch_size, seq_len).
    """
    a_star = log_coeffs.cumsum(dim=1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim=1)
    log_h = a_star + log_h0_plus_b_star
    return log_h.exp()


def log_g(x):
    """Logarithmic GELU activation function.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Activated tensor.
    """
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))


class GruBlock(nn.Module):
    """GRU Block using Heinsen's associative scan."""

    def __init__(self, d_model, backend="keops", **layer_args):
        """Initialize the GruBlock.

        Args:
            d_model (int): Dimension of the model.
            backend (str): Backend to use.
            **layer_args: Additional arguments.
        """
        super().__init__()
        self.to_hidden_and_gate = nn.Linear(d_model, d_model * 4, bias=False)
        self.to_out = nn.Linear(d_model * 2, d_model, bias=False)

    def forward(self, x, prev_hidden=None):
        """Forward pass of the GruBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            prev_hidden (torch.Tensor | None): Previous hidden state tensor.

        Returns:
            out (torch.Tensor): Output tensor of shape (batch_size, seq_len, d_model
            state (torch.Tensor): Final hidden state tensor of shape (batch_size, d_model).
        """
        hidden, gate = self.to_hidden_and_gate(x).chunk(2, dim=-1)
        log_coeffs = -F.softplus(gate)
        log_z = -F.softplus(-gate)
        log_tilde_h = log_g(hidden)
        log_values = log_z + log_tilde_h
        out = heinsen_associative_scan_log(log_coeffs, log_values)
        out = out[:, -x.shape[1] :]
        state = out[:, -1:]
        out = self.to_out(out)
        return out, state


class SequenceResidualBlock(nn.Module):
    """Sequence Residual Block with LayerNorm and S4/GRU layer."""

    def __init__(self, d_input, kernel, backend="keops"):
        """Initialize the SequenceResidualBlock.

        Args:
            d_input (int): Dimension of the input.
            kernel (nn.Module): Kernel layer (S4Block or GruBlock).
            backend (str): Backend to use for the kernel.
        """
        super().__init__()
        self.layer = kernel(d_input, backend)
        self.norm = torch.nn.LayerNorm((d_input,))

    def forward(self, x, rate=1, **kwargs):
        """Forward pass of the SequenceResidualBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_input).
            rate: Scaling factor for the kernel.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_input).
        """
        y = x
        y = self.norm(rearrange(y, "b ... d -> b (...) d")).view(y.shape)
        y, _new_state = self.layer(y, rate=rate, **kwargs)
        return x + y
