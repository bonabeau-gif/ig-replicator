# -*- coding: utf-8 -*-
"""Quantile shaping (Section 5.1) and the shaped defect diagnostic (Eqs. 56-59).

The population quantile q_theta(x) = CDF of f under rho_theta, evaluated at f(x),
is computed as the weighted empirical CDF of the raw fitness values on the grid.
Shaping functions W map (0,1) to R: linear W(u)=u, elite W(u)=1{u>=1-tau}, and
smooth elite W(u)=logistic((u-(1-tau))/h). The centered shaped signal is
w_theta = W(q_theta) - integral_0^1 W. Its L2(p) variance is Var_U(W), a constant.
"""
import numpy as np
from .quadrature import expectation
from .defect import project

__all__ = [
    "weighted_quantile", "W_linear", "W_elite", "W_smooth_elite",
    "var_U", "shaped_signal", "shaped_projection",
]


def weighted_quantile(f_vals, p, grid):
    """Weighted empirical CDF of f under density p, evaluated at each grid point.

    q_i = P(f(X) <= f(x_i)) with a midpoint convention, computed by stable sort.
    For continuous f this returns values approximately uniform on (0,1).
    """
    f_vals = np.asarray(f_vals, float)
    mass = np.maximum(p, 0.0) * grid.dx
    mass = mass / mass.sum()
    order = np.argsort(f_vals, kind="mergesort")
    cum = np.cumsum(mass[order])
    q_sorted = cum - 0.5 * mass[order]        # midpoint CDF
    q = np.empty_like(q_sorted)
    q[order] = q_sorted
    return np.clip(q, 1e-12, 1.0 - 1e-12)


def W_linear(u):
    """Linear rank shaping W(u) = u."""
    return np.asarray(u, float)


def W_elite(u, tau=0.25):
    """Elite top-tau selection W(u) = 1{u >= 1 - tau}."""
    return (np.asarray(u, float) >= (1.0 - tau)).astype(float)


def W_smooth_elite(u, tau=0.25, h=0.05):
    """Smooth elite selection W(u) = logistic((u - (1 - tau)) / h)."""
    z = (np.asarray(u, float) - (1.0 - tau)) / h
    return 1.0 / (1.0 + np.exp(-z))


def var_U(W, n=200001):
    """Var_{U(0,1)}(W(U)) by fine quadrature on (0,1)."""
    u = np.linspace(0, 1, n)
    w = W(u)
    mean = np.trapezoid(w, u)
    return float(np.trapezoid((w - mean) ** 2, u))


def shaped_signal(f_vals, p, grid, W):
    """Centered shaped relative fitness w_theta = W(q) - E[W(q)], Eq. (56)."""
    q = weighted_quantile(f_vals, p, grid)
    Wq = W(q)
    return Wq - expectation(Wq, p, grid), q


def shaped_projection(f_vals, p, grid, W, scores, ridge=0.0):
    """Shaped defect and R^2_shape for a score dictionary. Returns ProjectionResult.

    R^2_shape = (b^T F^+ b) / Var_U(W); the result's .r2 uses E_p[w^2] which equals
    Var_U(W) up to grid discretization, so the two normalizations agree numerically.
    """
    w, q = shaped_signal(f_vals, p, grid, W)
    return project(w, scores, p, grid, ridge=ridge)
