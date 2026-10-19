# -*- coding: utf-8 -*-
"""The Gaussian search family: density, scores, the closed-form raw defect
(Eq. 47 of the main article), the positive-quadratic blow-up time (Eq. 50), and
the analytic Gaussian defect for a sinusoidal fitness (Eq. S3 of the supplement).
"""
import numpy as np
from .quadrature import expectation

__all__ = [
    "gaussian_pdf", "gaussian_scores", "raw_defect_gaussian",
    "blowup_time", "analytic_sinusoid_defect",
]


def gaussian_pdf(x, mu, var):
    """N(mu, var) density evaluated at x."""
    return np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2.0 * np.pi * var)


def gaussian_scores(x, mu, var):
    """Centered Gaussian score directions spanning the relative tangent space.

    Returns an array of shape (2, len(x)) with rows Z and Z^2 - 1, where
    Z = (x - mu)/sqrt(var). Their span equals span{partial_mu log rho, partial_var log rho}.
    """
    z = (x - mu) / np.sqrt(var)
    return np.vstack([z, z ** 2 - 1.0])


def raw_defect_gaussian(f_vals, mu, var, p, grid):
    """Closed-form raw invariance defect under a Gaussian state, Eq. (47):

        delta^2 = Var[f] - Cov(f, X)^2 / var - Cov(f, (X-mu)^2)^2 / (2 var^2).

    f_vals is f sampled on grid.x; p is the current Gaussian density on the grid.
    """
    x = grid.x
    Ef = expectation(f_vals, p, grid)
    varf = expectation((f_vals - Ef) ** 2, p, grid)
    cov_fx = expectation((f_vals - Ef) * (x - mu), p, grid)
    cov_fq = expectation((f_vals - Ef) * ((x - mu) ** 2 - var), p, grid)
    d2 = varf - cov_fx ** 2 / var - cov_fq ** 2 / (2.0 * var ** 2)
    return max(d2, 0.0)


def blowup_time(c, var0):
    """Finite-time validity horizon for f(x) = c x^2 with c > 0, Eq. (50):
    t* = 1 / (2 c var0). Returns inf for c <= 0."""
    return np.inf if c <= 0 else 1.0 / (2.0 * c * var0)


def analytic_sinusoid_defect(eps, k, var):
    """Analytic Gaussian squared defect for f(x) = eps sin(k x) at N(0, var), Eq. (S3):

        delta^2 = eps^2 [ (1 - e^{-2 k^2 var}) / 2 - k^2 var e^{-k^2 var} ].
    """
    ks = k * k * var
    return eps ** 2 * (0.5 * (1.0 - np.exp(-2.0 * ks)) - ks * np.exp(-ks))
