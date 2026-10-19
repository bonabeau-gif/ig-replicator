# -*- coding: utf-8 -*-
"""Student-t search family (Section 4.3).

The Student-t family is invariant not under polynomial fitness but under rational
fitness of the form [alpha z + beta z^2] / (nu + z^2) + gamma, with z = (x-mu)/sigma.
Scores are given by Eq. (54).
"""
import numpy as np
from scipy.special import gammaln

__all__ = ["student_t_pdf", "student_t_scores", "rational_conjugate_fitness"]


def student_t_pdf(x, mu, sigma, nu):
    """Student-t_nu density with location mu and scale sigma."""
    z = (np.asarray(x, float) - mu) / sigma
    logc = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi) - np.log(sigma)
    return np.exp(logc - (nu + 1) / 2 * np.log1p(z ** 2 / nu))


def student_t_scores(x, mu, sigma, nu):
    """Centered Student-t score directions spanning the relative tangent space, Eq. (54):

        partial_mu log rho = (nu + 1) z / ((nu + z^2) sigma)
        partial_s log rho  = -1 + (nu + 1) z^2 / (nu + z^2),   s = log sigma.
    """
    z = (np.asarray(x, float) - mu) / sigma
    s_mu = (nu + 1) * z / ((nu + z ** 2) * sigma)
    s_s = -1.0 + (nu + 1) * z ** 2 / (nu + z ** 2)
    return np.vstack([s_mu, s_s])


def rational_conjugate_fitness(alpha, beta, gamma, mu, sigma, nu):
    """Return the Student-t conjugate fitness f(x) = [alpha z + beta z^2]/(nu + z^2) + gamma."""
    def f(x):
        z = (np.asarray(x, float) - mu) / sigma
        return (alpha * z + beta * z ** 2) / (nu + z ** 2) + gamma
    return f
