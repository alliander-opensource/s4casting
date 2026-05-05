# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import plotly.graph_objects as go
import torch
from numpy.typing import NDArray
from plotly.subplots import make_subplots

from s4casting.core.functional import quantile_pool1d


def plot_quantiles(
    quantiles: torch.Tensor,
    quantile_values: list,
    X: torch.Tensor,
    Xm: torch.Tensor,
    Y: torch.Tensor,
    Ym: torch.Tensor,
    times: NDArray,
    input_sample_interval_minutes: int,
    output_sample_interval_minutes: int,
    report_type: str,
    time_horizon: str,
    feature_names: list[str] | None = None,
):
    """Dispatch plotting based on report type and time horizon.

    Note: the last dimensions of X and Y are assumed to be weather features

    Args:
        quantiles (torch.tensor): Quantile predictions.
        quantile_values (list): Which quantile do the quantiles belong to.
        times (NDArray): Time stamps associated with the predictions.
        X (np.ndarray): Input data tensor of shape (B, T, F).
        Xm (np.ndarray): Input mask tensor of shape (B, T, F).
        Y (np.ndarray): Output data tensor of shape (B, T, F).
        Ym (np.ndarray): Output mask tensor of shape (B, T, F).
        quantiles (np.ndarray): Quantile predictions of shape (B, T, Q).
        input_sample_interval_minutes (int): Input sample rate in mins.
        output_sample_interval_minutes (int): Output sample rate in mins.
        report_type (str): one of {"training", "evaluation", "benchmark"}
        time_horizon (str): one of {"short", "medium"}
        feature_names (Optional[List[str]]): Names of the features for plotting.

    Returns:
        go.Figure
    """
    patch_size = output_sample_interval_minutes // input_sample_interval_minutes

    # Medium-term always uses quantile-pooled plots
    if time_horizon == "medium":
        return plot_medium_term_training_quantiles(
            X,
            Xm,
            Ym,
            Y,
            quantiles,
            patch_size,
        )

    if time_horizon == "short":
        # Training / evaluation (quantile head only)
        if report_type in ("training", "evaluation"):
            return plot_short_term_training_quantiles(
                X, Xm, Y, Ym, quantiles, quantile_values, feature_names=feature_names
            )

        # Benchmarking
        if report_type == "benchmark":
            return plot_short_term_benchmark_quantiles(
                Y, Ym, times, quantiles, quantile_values, feature_names=feature_names
            )

        raise ValueError(f"Unknown report_type: {report_type!r}")

    # Unknown horizon
    raise ValueError(f"Unknown time_horizon: {time_horizon!r}")


def plot_short_term_training_quantiles(
    X, Xm, Y, Ym, quantiles, quantile_values, feature_names: list[str] | None = None
) -> go.Figure:
    """Plot short-term training data using a quantile head.

    Note: the last dimensions of X and Y are assumed to be weather features
    Shows:
      - High-resolution context X and target Y
      - Predicted quantiles for the forecast horizon (short-term)

    Returns:
        go.Figure: Plotly figure object containing the benchmarking forecast plot.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Extract weather:
    weather_x = X[:, :, Ym[0].sum(dim=-2) == 0]
    weather_y = Y[:, :, Ym[0].sum(dim=-2) == 0]
    weather = torch.cat([weather_x, weather_y], dim=1)

    # Sequence lengths
    seq_len = X.shape[1]
    pred_len = Y.shape[1]

    # Time indices (short-term, same resolution for X and Y)
    time_x = np.arange(seq_len)
    time_y = np.arange(seq_len, seq_len + pred_len)

    # High-resolution context and target
    fig.add_trace(
        go.Scatter(
            x=time_x,
            y=X[0, :, 0],
            mode="lines",
            line={"color": "blue", "width": 1},
            name="X (context)",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=time_x,
            y=Xm[0, :, 0],
            mode="lines",
            line={"color": "red", "width": 1},
            name="Mask X",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=time_y,
            y=Y[0, :, 0],
            mode="lines",
            line={"color": "black", "width": 1},
            name="Y (GT)",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=time_y,
            y=Ym[0, :, 0],
            mode="lines",
            line={"color": "red", "width": 1},
            name="Mask Y",
        ),
        secondary_y=False,
    )

    # Prediction quantiles (short-term, no pooling)
    # Fade from transparent to opaque green
    num_q = quantiles.shape[-1]
    alphas = np.linspace(0.25, 1.0, num_q)
    colours = [f"rgba(0, 255, 0, {a})" for a in alphas]

    for i, q_val in enumerate(quantile_values):
        fig.add_trace(
            go.Scatter(
                x=time_y,
                y=quantiles[0, :, i],
                mode="lines",
                line={"color": colours[i]},
                name=f"Prediction quantile {q_val}",
            ),
            secondary_y=False,
        )

    # Show weather data
    for i in range(weather.shape[-1]):
        name = feature_names[i] if feature_names and i < len(feature_names) else f"Weather [{i}]"
        if name in ("unixtime", "latitude", "longitude"):
            continue
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([time_x, time_y]),
                y=weather[0, ..., i],
                name=name,
            ),
            secondary_y=True,
        )

    # Axes and layout
    fig.update_yaxes(title_text="Power values", secondary_y=False)
    fig.update_yaxes(title_text="Weather values", secondary_y=True)

    fig.update_layout(
        title="Short-term Training Forecast",
        xaxis_title="Time Step",
        width=1000,
        height=600,
        legend={
            "x": 0,
            "y": 1,
            "xanchor": "left",
            "yanchor": "top",
        },
    )

    return fig


def plot_short_term_benchmark_quantiles(
    Y, Ym, times, quantiles, quantile_values, downsample_rate: int = 10, feature_names: list[str] | None = None
) -> go.Figure:
    """Plot short-term benchmarking forecast using a quantile head.

    Shows:
      - Ground truth Y (downsampled)
      - Predicted quantiles (downsampled)
      - Weather covariates on secondary y-axis

    Note: the last dimensions of X and Y are assumed to be weather features

    Returns:
        go.Figure: Plotly figure object containing the benchmarking forecast plot.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    weather = Y[:, :, Ym[0].sum(dim=-2) == 0]

    # Downsample ground truth and covariates
    Y = Y[:, ::downsample_rate, :]
    weather = weather[:, ::downsample_rate, :]
    quantiles = quantiles[:, ::downsample_rate, :]

    # Time index from provided timestamps, downsampled
    idx_y = times[::downsample_rate]

    # Ground truth
    fig.add_trace(
        go.Scatter(
            x=idx_y,
            y=Y[0, :, 0],
            name="Y (Ground truth)",
            mode="lines",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=idx_y,
            y=Ym[0, :, 0],
            mode="lines",
            line={"color": "red", "width": 1},
            name="Y (Mask)",
        ),
        secondary_y=False,
    )

    # Predicted quantiles
    num_q = quantiles.shape[-1]
    alphas = np.linspace(0.25, 1.0, num_q)
    colours = [f"rgba(0, 255, 0, {a})" for a in alphas]

    for i, q_val in enumerate(quantile_values):
        fig.add_trace(
            go.Scatter(
                x=idx_y,
                y=quantiles[0, :, i],
                mode="lines",
                line={"color": colours[i]},
                name=f"Prediction quantile {q_val}",
            ),
            secondary_y=False,
        )

    # Weather covariates (secondary y-axis)
    for i in range(weather.shape[-1]):
        name = feature_names[i] if feature_names and i < len(feature_names) else f"Weather [{i}]"
        if name in ("unixtime", "latitude", "longitude"):
            continue
        fig.add_trace(
            go.Scatter(
                x=idx_y,
                y=weather[0, :, i],
                name=name,
                mode="lines",
            ),
            secondary_y=True,
        )

    # Axes and layout
    fig.update_xaxes(title_text="Time step", showgrid=True)
    fig.update_yaxes(title_text="Power value", showgrid=True, secondary_y=False)
    fig.update_yaxes(title_text="Weather value", showgrid=True, secondary_y=True)

    fig.update_layout(
        width=1000,
        height=1000,
        title="Short-term Forecast (Quantile Head)",
        legend={
            "x": 0,
            "y": 1,
            "xanchor": "left",
            "yanchor": "top",
        },
    )

    return fig


def plot_medium_term_training_quantiles(
    X,
    Xm,
    Ym,
    Y,
    quantiles,
    patch_size,
) -> go.Figure:
    """Plot medium term training quantiles.

    Plot prediction on training data with both high-resolution and quantile-pooled data.
    This is for the medium term setting.

    Returns:
        go.Figure: Plotly figure object containing the medium term training quantiles plot.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    seq_len = X.shape[1]
    pred_len = Y.shape[1]

    # Time indices
    time_x_hr = np.arange(seq_len)  # high-resolution time index for X
    time_y_hr = np.arange(seq_len, seq_len + pred_len)  # high-resolution time index for Y

    time_x_quant = np.arange(0, seq_len, patch_size) + patch_size // 2  # centered pooled indices
    time_y_quant = np.arange(seq_len, seq_len + pred_len, patch_size) + patch_size // 2

    # Plot high-resolution input X
    fig.add_trace(
        go.Scatter(
            x=time_x_hr,
            y=X[0, :, 0],
            mode="lines",
            line={"color": "blue", "width": 1},
            name="High-res X",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=time_x_hr,
            y=Xm[0, :, 0],
            mode="lines",
            line={"color": "red", "width": 1},
            name="Mask X",
        ),
        secondary_y=False,
    )

    # Plot high-resolution ground truth Y
    fig.add_trace(
        go.Scatter(
            x=time_y_hr,
            y=Y[0, :, 0],
            mode="lines",
            line={"color": "black", "width": 1},
            name="High-res Y (GT)",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=time_y_hr,
            y=Ym[0, :, 0],
            mode="lines",
            line={"color": "red", "width": 1},
            name="Mask Y",
        ),
        secondary_y=False,
    )

    # Plot pooled quantiles for input X
    for q in [0.00, 1.00]:
        vals = quantile_pool1d(X.swapaxes(1, 2), kernel_size=patch_size, stride=patch_size, quantile=q)[0, 0, :]
        fig.add_trace(
            go.Scatter(
                x=time_x_quant,
                y=vals,
                name=f"Context quantile {q:.2f}",
                line={"dash": "dot"},
            ),
            secondary_y=False,
        )

    # Plot prediction quantiles
    alphas = np.linspace(0.25, 1.0, quantiles.shape[-1])
    colours = [f"rgba(0, 255, 0, {a})" for a in alphas]
    for q in range(quantiles.shape[-1]):
        fig.add_trace(
            go.Scatter(
                x=time_y_quant,
                y=quantiles[0, :, q],
                mode="lines",
                line={"color": colours[q]},
                name=f"Prediction quantile {q}",
            ),
            secondary_y=False,
        )

    # Plot pooled quantiles for ground truth Y
    for q in [0.00, 1.00]:
        vals = quantile_pool1d(Y.swapaxes(1, 2), kernel_size=patch_size, stride=patch_size, quantile=q)[0, 0, :]
        fig.add_trace(
            go.Scatter(
                x=time_y_quant,
                y=vals,
                name=f"Ground truth quantile {q:.2f}",
                line={"dash": "dash"},
            ),
            secondary_y=False,
        )

    # Axes titles
    fig.update_yaxes(title_text="Power values", secondary_y=False)
    fig.update_yaxes(title_text="Weather values", secondary_y=True)

    # Layout
    fig.update_layout(
        title="Time Series Prediction with Quantiles and High-Resolution Data",
        xaxis_title="Time Step",
        width=1000,
        height=600,
        legend={
            "x": 0,
            "y": 1,
            "xanchor": "left",
            "yanchor": "top",
        },
    )

    return fig


def plot_inference_quantiles(X, quantiles, quantile_values: list[float]) -> go.Figure:
    """Plot inference results: input series and quantile predictions only.

    Args:
        X (torch.Tensor): Input data tensor of shape (B, T, F).
        quantiles (torch.Tensor): Quantile predictions of shape (B, T, Q).
        quantile_values (list[float]): List of quantile values corresponding to the last dimension

    Returns:
        go.Figure: Plotly figure object containing the inference quantiles plot.
    """
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    seq_len = X.shape[1]
    pred_len = quantiles.shape[1]
    idx_x = np.arange(seq_len)
    idx_y = np.arange(seq_len, seq_len + pred_len)

    fig.add_trace(
        go.Scatter(
            x=idx_x,
            y=X[0, :, 0],
            mode="lines",
            name="X (Input Series)",
            line={"color": "blue"},
        )
    )

    num_q = quantiles.shape[-1]
    alphas = np.linspace(0.25, 1.0, num_q)
    colours = [f"rgba(0, 255, 0, {a})" for a in alphas]

    for i, q_val in enumerate(quantile_values):
        fig.add_trace(
            go.Scatter(
                x=idx_y,
                y=quantiles[0, :, i],
                mode="lines",
                line={"color": colours[i]},
                name=f"Prediction quantile {q_val}",
            )
        )

    fig.update_yaxes(title_text="Power values")
    fig.update_xaxes(title_text="Time Step")
    fig.update_layout(
        title="Time Series Inference Forecast",
        width=1000,
        height=600,
        legend={
            "x": 0,
            "y": 1,
            "xanchor": "left",
            "yanchor": "top",
        },
    )
    return fig
