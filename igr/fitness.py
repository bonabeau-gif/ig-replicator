# -*- coding: utf-8 -*-
"""Fitness functions used across the supplementary experiments.

Where the clean supplement states a fitness explicitly (the S3 perturbed family,
the two-dimensional landscape, the stalling fitness) the exact form is used.
Where only a qualitative description is given (the S1 "pure sinusoid", the S4/S11
"two modes near +/-2"), a representative form is defined here and documented so
the figures reproduce the described behavior.
"""
import numpy as np

__all__ = [
    "quadratic", "perturbed_quadratic", "sinusoid",
    "multimodal", "stalling", "landscape_2d",
]


def quadratic(x):
    """Affine-quadratic base fitness f(x) = 0.30 x - 0.20 x^2 (in the Gaussian class)."""
    x = np.asarray(x, float)
    return 0.30 * x - 0.20 * x ** 2


def perturbed_quadratic(eps, a=0.30, b=0.20, k=1.5):
    """f_eps(x) = a x - b x^2 + eps sin(k x), the S3 multi-step family."""
    def f(x):
        x = np.asarray(x, float)
        return a * x - b * x ** 2 + eps * np.sin(k * x)
    return f


def sinusoid(eps=1.2, k=1.5):
    """Pure sinusoidal fitness f(x) = eps sin(k x); eps=1.2 turns the t=2 density bimodal."""
    def f(x):
        return eps * np.sin(k * np.asarray(x, float))
    return f


def multimodal(width=0.5, centers=(-2.0, 2.0)):
    """Two-mode fitness: sum of Gaussian bumps near the given centers.

    f(x) = sum_c exp(-(x - c)^2 / (2 width^2)). Used for multimodal elite shaping.
    """
    centers = np.asarray(centers, float)

    def f(x):
        x = np.asarray(x, float)
        return sum(np.exp(-0.5 * (x - c) ** 2 / width ** 2) for c in centers)
    return f


def stalling(a=2.0):
    """Symmetric double-well fitness f(x) = -(x^2 - a^2)^2 (Corollary 3.3 / Experiment S10)."""
    def f(x):
        x = np.asarray(x, float)
        return -((x ** 2 - a ** 2) ** 2)
    return f


def landscape_2d(eps):
    """Two-dimensional S5 landscape, f(x1, x2) =
       0.35 x1 - 0.18 x1^2 - 0.12 x2^2 + 0.08 x1 x2 + eps sin(1.3 x1) cos(1.1 x2)."""
    def f(X):
        X = np.atleast_2d(np.asarray(X, float))
        x1, x2 = X[..., 0], X[..., 1]
        return (0.35 * x1 - 0.18 * x1 ** 2 - 0.12 * x2 ** 2 + 0.08 * x1 * x2
                + eps * np.sin(1.3 * x1) * np.cos(1.1 * x2))
    return f
