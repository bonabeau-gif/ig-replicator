# -*- coding: utf-8 -*-
"""Replicator truth and the two Gaussian surrogates.

The exact fixed-fitness replicator trajectory is the exponential tilt of
Proposition 5, rho_t propto rho_0 exp(t f), evaluated on the grid. Against it we
compare two Gaussian surrogates:

  param : the Fisher projection on relative velocities, i.e. the moment-closure
          ODE of Eq. (S1) / Eq. (42). This is the object of the theorems.
  proj  : the ordinary L2(dx) projection on density velocities (Appendix C of the
          main manuscript), retained only for comparison.
"""
import numpy as np
from scipy.integrate import solve_ivp
from .quadrature import normalize, expectation, moments
from .gaussian import gaussian_pdf

__all__ = ["exact_tilt", "param_rhs", "integrate_param", "proj_rhs", "integrate_proj"]


def exact_tilt(rho0, f_vals, t, grid):
    """Exact replicator density at time t: normalize(rho0 * exp(t f))."""
    logw = t * f_vals
    logw -= logw.max()
    return normalize(rho0 * np.exp(logw), grid)


# ----------------------------------------------------------------- param (Fisher / moment)
def param_rhs(mu, var, f_callable, grid):
    """Right-hand side of the Fisher-projected moment ODE, Eq. (S1):

        mu_dot  = E_g[(X - mu) f(X)]
        var_dot = E_g[(X - mu)^2 f(X)] - var * E_g[f(X)],

    with g = N(mu, var), expectations taken by quadrature on the grid.
    """
    x = grid.x
    g = gaussian_pdf(x, mu, max(var, 1e-12))
    g = normalize(g, grid)
    fx = f_callable(x)
    Ef = expectation(fx, g, grid)
    mu_dot = expectation((x - mu) * fx, g, grid)
    var_dot = expectation((x - mu) ** 2 * fx, g, grid) - var * Ef
    return mu_dot, var_dot


def integrate_param(mu0, var0, f_callable, grid, t_span, t_eval=None,
                    rtol=1e-9, atol=1e-12, method="DOP853"):
    """Integrate the param moment flow over t_span. Returns (t, mu, var)."""
    def rhs(t, y):
        md, vd = param_rhs(y[0], max(y[1], 1e-12), f_callable, grid)
        return [md, vd]
    sol = solve_ivp(rhs, t_span, [mu0, var0], t_eval=t_eval,
                    rtol=rtol, atol=atol, method=method, dense_output=False)
    return sol.t, sol.y[0], sol.y[1]


# ----------------------------------------------------------------- proj (L2(dx) density velocity)
def proj_rhs(mu, var, f_callable, grid):
    """Right-hand side of the ordinary L2(dx) density-velocity projection.

    The true density velocity is rho (f - E[f]); the Gaussian density-velocity
    basis is partial_mu rho and partial_var rho. Minimizing the L2(dx) distance
    between density velocities gives the normal equations M [mu_dot, var_dot] = rhs,
    with M_jk = integral (partial_j rho)(partial_k rho) dx.
    """
    x = grid.x
    rho = normalize(gaussian_pdf(x, mu, max(var, 1e-12)), grid)
    fx = f_callable(x)
    Ef = expectation(fx, rho, grid)
    dmu = rho * (x - mu) / var                         # partial_mu rho
    dvar = rho * ((x - mu) ** 2 - var) / (2.0 * var ** 2)   # partial_var rho
    true_dv = rho * (fx - Ef)                           # true density velocity
    basis = [dmu, dvar]
    M = np.array([[np.trapezoid(bi * bj, x) for bj in basis] for bi in basis])
    rhs = np.array([np.trapezoid(true_dv * bi, x) for bi in basis])
    sol = np.linalg.solve(M + 1e-18 * np.eye(2), rhs)
    return float(sol[0]), float(sol[1])


def integrate_proj(mu0, var0, f_callable, grid, t_span, t_eval=None,
                   rtol=1e-9, atol=1e-12, method="DOP853"):
    """Integrate the proj density-velocity flow over t_span. Returns (t, mu, var)."""
    def rhs(t, y):
        md, vd = proj_rhs(y[0], max(y[1], 1e-12), f_callable, grid)
        return [md, vd]
    sol = solve_ivp(rhs, t_span, [mu0, var0], t_eval=t_eval,
                    rtol=rtol, atol=atol, method=method)
    return sol.t, sol.y[0], sol.y[1]
