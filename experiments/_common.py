# -*- coding: utf-8 -*-
"""Shared helpers for the shaped-invariance experiments (S2-S5)."""
import numpy as np
from igr import (Grid, gaussian_pdf, normalize, gaussian_scores, legendre_rank,
                 rank_bins, rbf, stack, weighted_quantile, W_linear, W_elite,
                 W_smooth_elite, shaped_projection)

SHAPINGS = {
    "linear rank": W_linear,
    "elite top 25%": lambda u: W_elite(u, 0.25),
    "smooth top 25%": lambda u: W_smooth_elite(u, 0.25, 0.05),
}


def build_dictionaries(x, p, grid, mu, var, which, rbf_centers=None):
    """Return {name: scores array} for the requested dictionary specs on this state.

    `which` is a list of ("name", kind, K) tuples; kind in
    {gaussian, legendre, bins, rbf}. Rank features use the quantile of a reference
    fitness, so the quantile q is supplied per call by the caller via closure.
    """
    raise NotImplementedError  # dictionaries are built inline per experiment below


def r2_table(f_vals, p, grid, mu, var, dict_specs, shapings=None, ridge=0.0):
    """Compute R^2_shape for each (dictionary, shaping). Returns a dict of dicts.

    dict_specs: list of (name, builder) where builder(q) -> scores array (K, n),
    q being the quantile of f under p. Gaussian scores do not depend on q.
    """
    shapings = shapings or SHAPINGS
    q = weighted_quantile(f_vals, p, grid)
    out = {}
    for name, builder in dict_specs:
        scores = builder(q)
        row = {}
        for sname, W in shapings.items():
            res = shaped_projection(f_vals, p, grid, W, scores, ridge=ridge)
            row[sname] = res
        out[name] = row
    return out


def gauss_only(mu, var, x):
    return lambda q: gaussian_scores(x, mu, var)


def gauss_plus_legendre(mu, var, x, K):
    return lambda q: stack(gaussian_scores(x, mu, var), legendre_rank(q, K))


def gauss_plus_bins(mu, var, x, K):
    return lambda q: stack(gaussian_scores(x, mu, var), rank_bins(q, K))


def gauss_plus_rbf(mu, var, x, centers, h):
    return lambda q: stack(gaussian_scores(x, mu, var), rbf(x, centers, h))
