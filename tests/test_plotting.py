# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path

import numpy as np
import torch
from plotly.graph_objects import Figure

from s4casting.visualisation import plot_quantiles


def test_plotting():
    """Tests the plotting functionality of the updated Plotter class."""
    # ------------------------------
    # Base dimensions (simple small test)
    # ------------------------------
    B = 1
    context_window = 48  # X length
    pred_window = 24  # Y length
    n_features = 3
    n_quantiles = 5

    input_interval = 5  # minutes
    output_interval = 15  # minutes

    # ------------------------------
    # Build input tensors
    # ------------------------------
    X = torch.randn(B, context_window, n_features)
    Xm = torch.ones_like(X)

    Y = torch.randn(B, pred_window, n_features)
    Ym = torch.ones_like(Y)

    quantiles = torch.randn(B, pred_window, n_quantiles)
    quantile_values = torch.linspace(0, 1, n_quantiles)

    # Times array for benchmarking
    times = np.arange(context_window + pred_window)

    # ------------------------------
    # Instantiate the plotter
    # ------------------------------
    fig = plot_quantiles(
        quantiles=quantiles,
        quantile_values=quantile_values,
        X=X,
        Xm=Xm,
        Y=Y,
        Ym=Ym,
        times=times,
        input_sample_interval_minutes=input_interval,
        output_sample_interval_minutes=output_interval,
        report_type="training",
        time_horizon="short",  # use short-term test path
    )

    # ------------------------------
    # Save and assert
    # ------------------------------
    outdir = Path("plots")
    outdir.mkdir(parents=True, exist_ok=True)

    fig.write_image(outdir / "plotter_test.png")

    assert isinstance(fig, Figure), "plotting must return a plotly Figure"
