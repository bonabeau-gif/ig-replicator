# -*- coding: utf-8 -*-
"""Experiment S17: dictionary cost in dimension. Table S8.

The multimodal elite-shaping problem is lifted to d dimensions: the shaped target
is elite top-25% on a single projected coordinate u = a^T x under a broad
N(0, 2.5^2 I) baseline. Three augmentations of the full Gaussian base are grown
until the shaped explained variance reaches 0.9: structured rank (rank features of
the projected coordinate), separable rank (per-axis rank features), and tiled RBF.

The structured construction reaches the target at every dimension with a constant
number of rank features beyond the Gaussian base d(d+3)/2; the rank of a point is a
scalar, so the augmentation is dimension-free. R^2 is estimated by Monte Carlo.
Set IGR_NMC to override the sample size (default 40000).
"""
import os
import numpy as np
import pandas as pd
from numpy.polynomial import legendre as L
from igr import fitness
from igr.plotting import save_table

NMC = int(os.environ.get("IGR_NMC", "40000"))
SIGMA = 2.5
TAU = 0.25
DIMS = [1, 2, 3, 5, 10]


def gaussian_base(X, sigma=SIGMA):
    """Full Gaussian base scores in d dimensions, standardized: linear z_j and
    centered quadratic z_i z_j (diagonal as z_j^2 - 1), z = x/sigma."""
    N, d = X.shape
    Z = X / sigma
    rows = [Z[:, j] for j in range(d)]
    for i in range(d):
        for j in range(i, d):
            rows.append(Z[:, i] * Z[:, j] - (1.0 if i == j else 0.0))
    return np.vstack(rows)


def legendre_of(u):
    r = np.empty_like(u)
    order = np.argsort(u, kind="mergesort")
    r[order] = (np.arange(len(u)) + 0.5) / len(u)
    return r


def leg_features(rank, K):
    t = 2 * rank - 1
    return np.vstack([L.legval(t, np.eye(K + 1)[k]) for k in range(1, K + 1)])


def r2_shape(scores, w):
    Sc = scores - scores.mean(1, keepdims=True)
    wc = w - w.mean()
    N = Sc.shape[1]
    F = Sc @ Sc.T / N
    b = Sc @ wc / N
    expl = float(b @ np.linalg.pinv(F) @ b)
    return expl / np.var(wc), float(np.linalg.cond(F))


def run():
    rng = np.random.default_rng(0)
    f = fitness.multimodal(width=0.6)
    rows = []
    for d in DIMS:
        X = rng.normal(0.0, SIGMA, size=(NMC, d))
        a = np.zeros(d); a[0] = 1.0            # projected coordinate
        u = X @ a
        fu = f(u)                              # multimodal fitness of the projected coord
        qf = legendre_of(fu)                   # fitness quantile
        w = (qf >= 1 - TAU).astype(float)      # elite top-25% on the fitness
        base = gaussian_base(X)
        base_count = d * (d + 3) // 2

        # structured rank: base + Legendre features of the (scalar) fitness quantile
        qu = qf
        struct = ("not reached", np.nan)
        for K in range(1, 13):
            scores = np.vstack([base, leg_features(qu, K)])
            r2, cond = r2_shape(scores, w)
            if r2 >= 0.9:
                struct = (base_count + K, cond); break
        else:
            struct = (f"not reached (R2={r2:.2f})", cond)

        # separable rank: base + per-axis Legendre features
        sep = None
        for m in range(1, 7):
            feats = [leg_features(legendre_of(X[:, j]), m) for j in range(d)]
            scores = np.vstack([base] + feats)
            r2, cond = r2_shape(scores, w)
            if r2 >= 0.9:
                sep = (base_count + d * m, cond); break
        if sep is None:
            sep = (f"not reached (R2={r2:.2f})", cond)

        # tiled RBF: base + RBF at tiled centers (capped)
        tiled = None
        centers_pool = rng.normal(0.0, SIGMA, size=(200, d))
        for Kc in (9, 18, 36, 60):
            C = centers_pool[:Kc]; h = SIGMA
            rbf = np.exp(-0.5 * ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1) / h ** 2).T
            scores = np.vstack([base, rbf])
            r2, cond = r2_shape(scores, w)
            if r2 >= 0.9:
                tiled = (base_count + Kc, cond); break
        if tiled is None:
            tiled = (f"not reached (R2={r2:.2f})", cond)

        rows.append({"d": d, "Gaussian base": base_count,
                     "structured rank": struct[0], "structured cond": round(struct[1], 1) if np.isfinite(struct[1]) else np.nan,
                     "separable rank": sep[0], "tiled RBF": tiled[0]})
        print(f"d={d:2d}: base={base_count:3d}  structured={struct[0]} (cond {struct[1]:.0f})  "
              f"separable={sep[0]}  tiled={tiled[0]}")
    df = pd.DataFrame(rows)
    save_table(df, "tableS8_dictionary_cost")
    print("\n", df.to_string(index=False))


if __name__ == "__main__":
    run()
