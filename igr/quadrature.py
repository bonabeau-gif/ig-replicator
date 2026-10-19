# -*- coding: utf-8 -*-
"""One-dimensional quadrature and density utilities.

All population (quadrature) calculations in Supplementary Sections S1-S5 use a
normalized grid on a bounded interval. Densities are stored as arrays sampled on
the grid and normalized so that the trapezoidal integral is one.
"""
import numpy as np

__all__ = [
    "Grid", "normalize", "integrate", "expectation", "moments",
    "l1_distance", "tv_distance", "kl_divergence", "relative_l2",
]


class Grid:
    """A uniform 1-D quadrature grid on [a, b] with n points."""

    def __init__(self, a, b, n):
        self.a, self.b, self.n = float(a), float(b), int(n)
        self.x = np.linspace(a, b, n)
        self.dx = self.x[1] - self.x[0]

    def __len__(self):
        return self.n

    def __repr__(self):
        return f"Grid([{self.a}, {self.b}], n={self.n})"


def integrate(p, grid):
    """Trapezoidal integral of array p over the grid."""
    return float(np.trapezoid(p, grid.x))


def normalize(p, grid):
    """Return p rescaled to integrate to one on the grid (clipped nonnegative)."""
    p = np.maximum(np.asarray(p, float), 0.0)
    z = integrate(p, grid)
    if z <= 0:
        raise ValueError("density has nonpositive mass")
    return p / z


def expectation(h, p, grid):
    """E_p[h] = integral h p dx on the grid."""
    return float(np.trapezoid(h * p, grid.x))


def moments(p, grid):
    """Return (mean, variance) of the density p on the grid."""
    m = expectation(grid.x, p, grid)
    v = expectation((grid.x - m) ** 2, p, grid)
    return m, v


def l1_distance(p, q, grid):
    """L1 density distance integral |p - q| dx."""
    return float(np.trapezoid(np.abs(p - q), grid.x))


def tv_distance(p, q, grid):
    """Total-variation distance = 0.5 * L1."""
    return 0.5 * l1_distance(p, q, grid)


def kl_divergence(p, q, grid, eps=1e-300):
    """KL(p || q) = integral p log(p/q) dx on the grid."""
    p = np.maximum(p, 0.0)
    ratio = np.log(np.maximum(p, eps)) - np.log(np.maximum(q, eps))
    return float(np.trapezoid(np.where(p > 0, p * ratio, 0.0), grid.x))


def relative_l2(p, q, grid):
    """Relative L2 density error ||p - q||_2 / ||q||_2 on the grid."""
    num = np.sqrt(float(np.trapezoid((p - q) ** 2, grid.x)))
    den = np.sqrt(float(np.trapezoid(q ** 2, grid.x)))
    return num / den if den > 0 else np.nan
