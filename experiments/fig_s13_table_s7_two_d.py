# -*- coding: utf-8 -*-
"""Experiment S5 (two dimensions): bivariate Gaussian projection. Figure S13, Table S7.

The full bivariate Gaussian family N(mu, Sigma) is evolved under a nonquadratic,
nonseparable landscape. At eps = 0 the fitness is affine-quadratic and the defect
is numerical zero; as eps grows, defect, L1 error, and KL rise together.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from igr import fitness
from igr.plotting import setup, save, save_table, COLORS

EPS = [0.0, 0.1, 0.2, 0.4, 0.6]
T = 0.8


def grid_2d(n=181, lim=5.0):
    g = np.linspace(-lim, lim, n)
    X1, X2 = np.meshgrid(g, g, indexing="ij")
    dx = g[1] - g[0]
    return X1, X2, dx


def gauss2d(X1, X2, mu, S):
    det = S[0, 0] * S[1, 1] - S[0, 1] ** 2
    inv = np.array([[S[1, 1], -S[0, 1]], [-S[0, 1], S[0, 0]]]) / det
    d1 = X1 - mu[0]; d2 = X2 - mu[1]
    q = inv[0, 0] * d1 ** 2 + 2 * inv[0, 1] * d1 * d2 + inv[1, 1] * d2 ** 2
    return np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(det))


def normalize2d(p, dx):
    return p / (p.sum() * dx * dx)


def E(h, p, dx):
    return float((h * p).sum() * dx * dx)


def moment_rhs(state, fvals, X1, X2, dx):
    mu = state[:2]; S = np.array([[state[2], state[3]], [state[3], state[4]]])
    p = normalize2d(gauss2d(X1, X2, mu, S), dx)
    Ef = E(fvals, p, dx)
    d1 = X1 - mu[0]; d2 = X2 - mu[1]
    mu1 = E(d1 * fvals, p, dx); mu2 = E(d2 * fvals, p, dx)
    S11 = E(d1 * d1 * fvals, p, dx) - S[0, 0] * Ef
    S12 = E(d1 * d2 * fvals, p, dx) - S[0, 1] * Ef
    S22 = E(d2 * d2 * fvals, p, dx) - S[1, 1] * Ef
    return [mu1, mu2, S11, S12, S22]


def defect_2d(fvals, mu, S, p, X1, X2, dx):
    d1 = X1 - mu[0]; d2 = X2 - mu[1]
    scores = [d1, d2, d1 * d1 - S[0, 0], d2 * d2 - S[1, 1], d1 * d2 - S[0, 1]]
    scores = [s - E(s, p, dx) for s in scores]
    Ef = E(fvals, p, dx); w = fvals - Ef
    F = np.array([[E(si * sj, p, dx) for sj in scores] for si in scores])
    b = np.array([E(w * si, p, dx) for si in scores])
    var_w = E(w * w, p, dx)
    return max(var_w - b @ np.linalg.pinv(F) @ b, 0.0)


def run():
    setup()
    X1, X2, dx = grid_2d()
    rho0 = normalize2d(gauss2d(X1, X2, np.zeros(2), np.eye(2)), dx)
    rows = []
    defects = []; l1s = []; kls = []
    for eps in EPS:
        f = fitness.landscape_2d(eps)
        fvals = f(np.stack([X1, X2], axis=-1))
        sol = solve_ivp(lambda t, y: moment_rhs(y, fvals, X1, X2, dx),
                        (0, T), [0, 0, 1, 0, 1], rtol=1e-8, atol=1e-10, method="DOP853")
        mu = sol.y[:2, -1]; S = np.array([[sol.y[2, -1], sol.y[3, -1]], [sol.y[3, -1], sol.y[4, -1]]])
        p = normalize2d(gauss2d(X1, X2, mu, S), dx)
        tilt = normalize2d(rho0 * np.exp(T * (fvals - fvals.max())), dx)
        d2 = np.sqrt(defect_2d(fvals, mu, S, p, X1, X2, dx))   # report delta, as in Table S7
        l1 = float(np.abs(tilt - p).sum() * dx * dx)
        kl = float(np.where(tilt > 0, tilt * (np.log(np.maximum(tilt, 1e-300)) -
                                              np.log(np.maximum(p, 1e-300))), 0.0).sum() * dx * dx)
        rows.append({"eps": eps, "defect": d2, "L1 error": l1, "KL": kl,
                     "proj mean": f"({mu[0]:.3f}, {mu[1]:.3f})"})
        defects.append(d2); l1s.append(l1); kls.append(kl)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    save_table(df, "tableS7_two_d")

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.plot(EPS, defects, "o-", color=COLORS["param"], label="defect")
    ax.plot(EPS, l1s, "s-", color=COLORS["accent"], label=r"$L^1$ error")
    ax.plot(EPS, kls, "^-", color=COLORS["bins"], label="KL")
    ax.set_xlabel(r"$\epsilon$"); ax.set_ylabel("value at $t=0.8$")
    ax.set_title("Figure S13. Two-dimensional Gaussian projection"); ax.legend()
    fig.tight_layout(); print("saved", save(fig, "figS13_two_d"))


if __name__ == "__main__":
    run()
