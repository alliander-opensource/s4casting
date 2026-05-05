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
    n: int = 1000,
) -> torch.Tensor:
    """Convert Gaussian Mixture Model using torch.distributions to quantiles.

    TODO: Maybe this should be moved to another location? s4casting/eval/funcitonal.py?
    GMM has 3 parameters pi, mu, sigma which determine the means, standard deviations and amounts.
    Defaults to 4*sigma ranges.

    Args:
        pi (torch.tensor): log factor of GMM.
        sigma (torch.tensor): covariance of GMM.
        mu (torch.tensor): mean of GMM.
        quantile_values (list): which quantiles.
        n (int): n samples per discrete pdf.

    Returns:
        quantiles (torch.Tensor): Quantile predictions from GMM

    """
    B = sigma.shape[0]
    L = sigma.shape[1]

    # get ranges
    min_val, max_val = gmm_bounds(sigma, mu)
    xs = torch.linspace(min_val, max_val, n, device=mu.device)

    # make cdf using torch.distributions
    cdf = gmm_to_cdf(pi, sigma, mu, xs)

    # make quantiles
    quantiles = torch.zeros((B, L, len(quantile_values)))

    for q_idx, q in enumerate(quantile_values):
        c_idxs = torch.argmax((cdf >= q).int(), dim=2)  # get first True index for every forecast

        # Handle edge case where the CDF might not exceed the quantile value within the bounds
        zero_indexes = torch.nonzero(c_idxs == 0, as_tuple=False)
        for z in zero_indexes:
            b, l = z[0], z[1]  # Extract batch and sequence indices
            if torch.all(cdf[b, l, :] < q):  # Check along the n_samples dimension
                c_idxs[b, l] = len(xs) - 1  # Set to the last index in xs

        quantiles[:, :, q_idx] = xs[c_idxs]

    return quantiles
