# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import torch


def gmm_bounds(sigma: torch.Tensor, mu: torch.Tensor, scale: float = 4.0, pad: float = 0.2) -> tuple[float, float]:
    """Calculate bounds for GMM.

    Args:
        sigma (torch.tensor): covariance of GMM.
        mu (torch.tensor): mean of GMM.
        scale (float): # of sigmas away from mean.
        pad (float): percentage of padding from min/max

    Returns:
        (float,float): min and max values.
    """
    lower_bound = mu - scale * sigma
    upper_bound = mu + scale * sigma

    min_val = lower_bound.min().item()
    max_val = upper_bound.max().item()
    padding = (max_val - min_val) * pad

    return (min_val - padding, max_val + padding)


def gmm_to_cdf(
    pi: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    xs: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Convert Gaussian Mixture Model using torch.distributions to cdf.

    Args:
        pi (torch.tensor): mixing factor of GMM.
        sigma (torch.tensor): covariance of GMM.
        mu (torch.tensor): mean of GMM.
        xs (torch.tensor): linspace of values to compute the cdf on.
        eps (float): epsilon parameter to ensure numeral stability

    Returns:
        cdf (torch.Tensor): CDF predictions from GMM

    """
    # make cdf using torch.distributions
    cdf = torch.zeros((sigma.shape[0], sigma.shape[1], len(xs)))

    mix = torch.distributions.Categorical(torch.nan_to_num(pi) + eps)
    comp = torch.distributions.Normal(loc=torch.nan_to_num(mu) + eps, scale=torch.nan_to_num(sigma) + eps)
    gmm = torch.distributions.MixtureSameFamily(mixture_distribution=mix, component_distribution=comp)

    # NOTE: I tried to implement this using torch.distributions.Independant
    #       in order to make event_shape known, but it seems that
    #       the Independant.cdf() function is not defined.
    for i, x in enumerate(xs):
        cdf[:, :, i] = gmm.cdf(torch.nan_to_num(x))

    return cdf


def gmm_to_pdf(
    pi: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    xs: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Convert Gaussian Mixture Model using torch.distributions to pdf.

    Args:
        pi (torch.tensor): mixing factor of GMM.
        sigma (torch.tensor): covariance of GMM.
        mu (torch.tensor): mean of GMM.
        xs (torch.tensor): linspace of values to compute the pdf on.
        eps (float): epsilon parameter to ensure numeral stability

    Returns:
        pdf (torch.Tensor): pdf predictions from GMM

    """
    # make pdf using torch.distributions
    pdf = torch.zeros((sigma.shape[0], sigma.shape[1], len(xs)))

    mix = torch.distributions.Categorical(torch.nan_to_num(pi) + eps)
    comp = torch.distributions.Normal(loc=torch.nan_to_num(mu) + eps, scale=torch.nan_to_num(sigma) + eps)
    gmm = torch.distributions.MixtureSameFamily(mixture_distribution=mix, component_distribution=comp)
    for i, x in enumerate(xs):
        pdf[:, :, i] = torch.exp(gmm.log_prob(torch.nan_to_num(x)))

    return pdf


def gmm_to_quantiles(
    pi: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    quantile_values: list,
    n: int = 40,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Quantiles of a Gaussian mixture, solved per forecast by bisection of the mixture CDF.

    Every (batch, step) element gets its own search interval, ``mu +- 6 sigma`` over its components, so the
    precision is relative to that element's own scale. The previous implementation evaluated the CDF on one
    grid of 1000 points spanning the whole batch and snapped each quantile up to the first grid point at or
    above the level; with series of different magnitude in one batch (a national load next to a small
    zone, as the benchmarks batch them) the grid step was a sizeable fraction of the small series, which
    biased its quantiles upward by tens of percent while the large series were unaffected.

    Args:
        pi (torch.Tensor): Mixture weights, shape (B, L, G).
        sigma (torch.Tensor): Component standard deviations, shape (B, L, G).
        mu (torch.Tensor): Component means, shape (B, L, G).
        quantile_values (list): Quantile levels in (0, 1).
        n (int): Bisection iterations; 40 gives a relative precision of 1e-12 of the search interval.
        eps (float): Floor added to the weights and scales for numerical stability.

    Returns:
        torch.Tensor: Quantiles, shape (B, L, len(quantile_values)).
    """
    pi = torch.nan_to_num(pi.float()) + eps
    sigma = torch.nan_to_num(sigma.float()) + eps
    mu = torch.nan_to_num(mu.float())
    mixture = torch.distributions.MixtureSameFamily(
        mixture_distribution=torch.distributions.Categorical(probs=pi),
        component_distribution=torch.distributions.Normal(loc=mu, scale=sigma),
    )
    lower = (mu - 6 * sigma).amin(dim=-1)  # (B, L)
    upper = (mu + 6 * sigma).amax(dim=-1)
    quantiles = torch.empty((*mu.shape[:-1], len(quantile_values)), dtype=torch.float32, device=mu.device)
    for q_idx, level in enumerate(quantile_values):
        low, high = lower.clone(), upper.clone()
        for _ in range(n):
            mid = (low + high) / 2
            below = mixture.cdf(mid) < level
            low = torch.where(below, mid, low)
            high = torch.where(below, high, mid)
        quantiles[..., q_idx] = (low + high) / 2
    return quantiles
