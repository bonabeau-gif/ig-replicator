# -*- coding: utf-8 -*-
"""Score dictionaries for near-invariant families under shaping (Section 5.5).

Each builder returns an array of shape (K, n) sampled on the grid, uncentered;
the projection routine centers them under the current density. Dictionaries:

  gaussian_scores     : {Z, Z^2 - 1} with Z = (x - mu)/sqrt(var)
  legendre_rank       : shifted Legendre polynomials P_1..P_K of the quantile q
  rank_bins           : K indicator bins of the quantile q on [0,1]
  rbf                 : K radial basis functions exp(-(x-c)^2 / 2h^2)

Combined dictionaries are formed by vertical stacking, e.g.
    np.vstack([gaussian_scores(...), legendre_rank(q, K)]).
"""
import numpy as np
from numpy.polynomial import legendre as L

__all__ = ["gaussian_scores", "legendre_rank", "rank_bins", "rbf", "stack"]


def gaussian_scores(x, mu, var):
    """Ordinary Gaussian tangent directions {Z, Z^2 - 1}."""
    z = (np.asarray(x, float) - mu) / np.sqrt(var)
    return np.vstack([z, z ** 2 - 1.0])


def legendre_rank(q, K):
    """Shifted Legendre features P_1..P_K evaluated at quantile q in (0,1).

    The quantile is mapped to [-1, 1] by t = 2q - 1; P_0 (constant) is omitted
    because the projection centers all features.
    """
    q = np.asarray(q, float)
    t = 2.0 * q - 1.0
    rows = []
    for k in range(1, K + 1):
        c = np.zeros(k + 1)
        c[k] = 1.0
        rows.append(L.legval(t, c))
    return np.vstack(rows)


def rank_bins(q, K):
    """K indicator bins of the quantile q on a uniform partition of [0,1]."""
    q = np.asarray(q, float)
    edges = np.linspace(0.0, 1.0, K + 1)
    rows = []
    for j in range(K):
        lo, hi = edges[j], edges[j + 1]
        if j == K - 1:
            rows.append(((q >= lo) & (q <= hi)).astype(float))
        else:
            rows.append(((q >= lo) & (q < hi)).astype(float))
    return np.vstack(rows)


def rbf(x, centers, h):
    """Radial basis functions exp(-(x - c)^2 / (2 h^2)) at the given centers.

    h may be a scalar or an array matching centers.
    """
    x = np.asarray(x, float)
    centers = np.atleast_1d(np.asarray(centers, float))
    h = np.broadcast_to(np.asarray(h, float), centers.shape)
    return np.vstack([np.exp(-0.5 * (x - c) ** 2 / hc ** 2) for c, hc in zip(centers, h)])


def stack(*blocks):
    """Vertically stack dictionary blocks into one (K, n) array."""
    return np.vstack([np.atleast_2d(b) for b in blocks])
