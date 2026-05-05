# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

"""Experimental GMM heatmap plotting utilities.

NOTE: This is experimental code meant to keep the GMM heatmap plotting functionality.
It has not been thoroughly tested, but should work as is.
"""

import numpy as np
import plotly.graph_objects as go
import torch
from numpy.typing import NDArray
from plotly.subplots import make_subplots

from s4casting.core.distributions import gmm_bounds, gmm_to_pdf, gmm_to_quantiles


def gmm_training_plot(
    X: torch.Tensor, Y: torch.Tensor, logpi: torch.Tensor, sigma: torch.Tensor, mu: torch.Tensor, n_points=50
) -> go.Figure:
    """Plot prediction on training data.

    The shape of X and Y are [1,T,F].
    The shape of mu, pi, sigma are [1,T,#Gaussian]

    Args:
        X (np.ndarray): Input data tensor of shape (B, T, D).
        Y (np.ndarray): Output data tensor of shape (B, T, D).
        logpi (torch.tensor): mixing factor of GMM.
        sigma (torch.tensor): mixing factor of GMM.
        mu (torch.tensor): mean of GMM.
        n_points (int): n samples per discrete pdf.

    Returns:
        go.Figure: Plotly figure object containing the training data plot.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Calculate x bounds
    idx_x = np.arange(X.shape[1])
    idx_y = np.arange(X.shape[1], X.shape[1] + Y.shape[1])

    # Plot X (Input Series)
    fig.add_trace(
        go.Scatter(
            x=idx_x,
            y=X[0, ..., 0],
            mode="lines",
            name="X (Input Series)",
            line={"color": "blue"},
        ),
        secondary_y=False,
    )
    min_val, max_val = gmm_bounds(sigma, mu)
    V = np.linspace(min_val, max_val, n_points)
    # Note: mu's is [B,T,G]
    pdf = gmm_to_pdf(torch.exp(logpi), sigma, mu, torch.Tensor(V))
    z_data = np.transpose(pdf[0, ...].numpy())

    # Create a heatmap for the PDF values
    fig.add_trace(
        go.Heatmap(
            z=z_data,
            x=idx_y,
            y=V,
            colorscale="Hot",
            colorbar={"title": "Probability Density"},
        ),
        secondary_y=False,
    )

    # Plot Y (True Continuation)
    fig.add_trace(
        go.Scatter(
            x=idx_y,
            y=Y[0, ..., 0],
            mode="lines",
            name="Y (True Continuation)",
            line={"color": "green", "width": 1.0},
            opacity=0.7,
        ),
        secondary_y=False,
    )

    fig.update_yaxes(title_text="Power values", secondary_y=False)
    fig.update_yaxes(range=[min_val, max_val], secondary_y=False)

    # Update layout
    fig.update_layout(
        title="Time Series Prediction",
        xaxis_title="Time Step",
        width=1000,
        height=1000,
        legend={
            "x": 0,  # Position the legend in the top-left corner
            "y": 1,
            "xanchor": "left",
            "yanchor": "top",
        },
    )

    return fig


def gmm_benchmarking_forecast_plot(
    Y: torch.Tensor,
    logpi: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    weather: torch.Tensor,
    quantile_values: list,
    times: NDArray,
    downsample_rate: int = 10,
    n_points: int = 50,
) -> go.Figure:
    """Plot predictions on benchmarking data.

    The shape of Y are [1,T,F].
    The shape of mu, pi, sigma are [1,T,#Gaussian]

    Args:
        Y (np.ndarray): Output data tensor of shape (B, T, D).
        logpi (torch.tensor): mixing factor of GMM.
        sigma (torch.tensor): mixing factor of GMM.
        mu (torch.tensor): mean of GMM.
        quantile_values (list): Which quantile do the quantiles belong to.
        weather (torch.tensor): Weather data associated with the predictions.
        times (NDArray): Time stamps associated with the predictions.
        downsample_rate (int, optional): Rate at which to downsample the data for plotting
        n_points (int): n samples per discrete pdf.

    Returns:
        go.Figure: Plotly figure object containing the benchmarking forecast plot.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    ######################################################
    # Get prediction and scale ##############
    ######################################################

    Y = Y[:, ::downsample_rate, :]
    weather = weather[:, ::downsample_rate, :]

    # Calculate x bounds
    idx_y = times[::downsample_rate]  # np.arange(Y.shape[1])

    min_val, max_val = gmm_bounds(sigma, mu)
    V = np.linspace(min_val, max_val, n_points)
    pdf = gmm_to_pdf(torch.exp(logpi), sigma, mu, torch.Tensor(V))
    quantiles = gmm_to_quantiles(torch.exp(logpi), sigma, mu, quantile_values)
    z_data = np.transpose(pdf[0, ::downsample_rate, :].numpy())

    ######################################################
    # plot full forecast #################
    ######################################################

    # Create a heatmap for the PDF values
    heatmap = go.Heatmap(
        z=z_data,
        x=idx_y,
        y=V,
        name="Forecast",
        colorscale="Tempo",
        colorbar={"title": "Forecast"},
    )

    fig.add_trace(heatmap, secondary_y=False)

    fig.add_trace(
        go.Scatter(
            x=idx_y,
            y=Y[0, ..., 0],
            name="Y (Ground truth)",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=idx_y,
            y=quantiles[0, ::downsample_rate, -2].numpy(),  # -2 for the 90th quantile
            name="90th Quantile",
        ),
        secondary_y=False,
    )

    # Show weather data
    for i in range(weather.shape[-1]):
        fig.add_trace(
            go.Scatter(
                x=idx_y,
                y=weather[0, ..., i],
                name=f"Weather [{i}]",
            ),
            secondary_y=True,
        )

    ######################################################
    # Update layout ###################
    ######################################################
    fig.update_xaxes(title_text="Time step", showgrid=True)
    fig.update_yaxes(title_text="Power value", showgrid=True, secondary_y=False)
    fig.update_yaxes(title_text="Weather value", showgrid=True, secondary_y=True)

    fig.update_layout(
        width=1000,
        height=1000,
        title="Forecast",
        legend={
            "x": 0,  # Position the legend in the top-left corner
            "y": 1,
            "xanchor": "left",
            "yanchor": "top",
        },
    )

    return fig
