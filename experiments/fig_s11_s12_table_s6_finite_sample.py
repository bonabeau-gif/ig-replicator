# -*- coding: utf-8 -*-
"""Experiment S7: finite-sample estimation of shaped projected directions.
Figures S11 and S12, Table S6.

The multimodal elite-shaping example is evaluated under a broad N(0, 2.5^2)
baseline. Population values use dense quadrature; sample estimates use
N in {32,64,128,256,512} and ridge in {0,1e-6,1e-4,1e-2}, each averaged over
independent samples, evaluated out of sample under the population density.

Set IGR_NSAMPLES to override the number of samples per setting (paper used 150;
default here is 60 to keep runtime modest).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from igr import (Grid, gaussian_pdf, normalize, gaussian_scores, legendre_rank,
                 rank_bins, rbf, stack, weighted_quantile, W_elite, expectation, project)
from igr import fitness
from igr.plotting import setup, save, save_table, COLORS

BROAD_VAR = 2.5 ** 2
TAU = 0.25
VARW = TAU * (1 - TAU)
NS = [32, 64, 128, 256, 512]
LAMS = [0.0, 1e-6, 1e-4, 1e-2]
NSAMPLES = int(os.environ.get("IGR_NSAMPLES", "60"))
CENT9 = np.linspace(-4, 4, 9); H9 = 0.8 * (CENT9[1] - CENT9[0])


def pop_scores(name, g):
    """Population centered scores (K, n), under broad Gaussian, and the quantile."""
    p = normalize(gaussian_pdf(g.x, 0.0, BROAD_VAR), g)
    fx = fitness.multimodal(width=0.6)(g.x)
    q = weighted_quantile(fx, p, g)
    if name == "Gaussian":
        S = gaussian_scores(g.x, 0, BROAD_VAR)
    elif name == "Gauss+Legendre K=8":
        S = stack(gaussian_scores(g.x, 0, BROAD_VAR), legendre_rank(q, 8))
    elif name == "Gauss+RBF K=9":
        S = stack(gaussian_scores(g.x, 0, BROAD_VAR), rbf(g.x, CENT9, H9))
    elif name == "Gauss+rank bins K=8":
        S = stack(gaussian_scores(g.x, 0, BROAD_VAR), rank_bins(q, 8))
    means = np.array([expectation(s, p, g) for s in S])
    Sc = S - means[:, None]
    w_pop = (W_elite(q, TAU) - expectation(W_elite(q, TAU), p, g))
    return p, Sc, w_pop


def sample_scores(name, xs, fxs):
    order = np.argsort(fxs, kind="mergesort")
    rank = np.empty_like(fxs); rank[order] = (np.arange(len(fxs)) + 0.5) / len(fxs)
    w = W_elite(rank, TAU) - TAU
    if name == "Gaussian":
        z = xs / 2.5; S = np.vstack([z, z ** 2 - 1])
    elif name == "Gauss+Legendre K=8":
        z = xs / 2.5; S = np.vstack([z, z ** 2 - 1, legendre_rank(rank, 8)])
    elif name == "Gauss+RBF K=9":
        z = xs / 2.5; S = np.vstack([z, z ** 2 - 1, rbf(xs, CENT9, H9)])
    elif name == "Gauss+rank bins K=8":
        z = xs / 2.5; S = np.vstack([z, z ** 2 - 1, rank_bins(rank, 8)])
    Sc = S - S.mean(1, keepdims=True)
    return Sc, w - w.mean()


def fit_and_test(name, N, lam, rng, p, Sc_pop, w_pop, grid, f):
    xs = rng.normal(0.0, 2.5, N); fxs = f(xs)
    Ss, ws = sample_scores(name, xs, fxs)
    G = Ss @ Ss.T / N
    rhs = Ss @ ws / N
    md = np.trace(G) / G.shape[0]
    v = np.linalg.solve(G + lam * md * np.eye(G.shape[0]) + 1e-12 * np.eye(G.shape[0]), rhs)
    pred = v @ Sc_pop
    resid = w_pop - pred
    r2 = 1.0 - expectation(resid ** 2, p, grid) / VARW
    return r2


def run():
    setup()
    g = Grid(-10, 10, 4001)
    f = fitness.multimodal(width=0.6)
    dicts = ["Gaussian", "Gauss+Legendre K=8", "Gauss+RBF K=9", "Gauss+rank bins K=8"]
    rows = []
    best_curve = {}          # for Fig S11
    rbf_by_lambda = {}       # for Fig S12
    for name in dicts:
        p, Sc_pop, w_pop = pop_scores(name, g)
        pop_r2 = project(w_pop, Sc_pop, p, g, center=False).r2
        cond = float(np.linalg.cond(np.array([[expectation(si * sj, p, g) for sj in Sc_pop] for si in Sc_pop])))
        curve_med = []; curve_lo = []; curve_hi = []
        for N in NS:
            per_lambda = {}
            for lam in LAMS:
                rng = np.random.default_rng(1000 + N)
                r2s = np.array([fit_and_test(name, N, lam, rng, p, Sc_pop, w_pop, g, f)
                                for _ in range(NSAMPLES)])
                per_lambda[lam] = r2s
            meds = {lam: np.median(v) for lam, v in per_lambda.items()}
            best_lam = max(meds, key=meds.get)
            r2s = per_lambda[best_lam]
            med = np.median(r2s); lo, hi = np.percentile(r2s, [10, 90])
            curve_med.append(med); curve_lo.append(lo); curve_hi.append(hi)
            rows.append({"dictionary": name, "N": N, "pop R2": pop_r2, "best lambda": best_lam,
                         "median test R2": med, "10-90%": f"[{lo:.3f}, {hi:.3f}]",
                         "median cond": cond})
            if name == "Gauss+RBF K=9":
                rbf_by_lambda[N] = {lam: np.median(v) for lam, v in per_lambda.items()}
        best_curve[name] = (curve_med, curve_lo, curve_hi)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    save_table(df, "tableS6_finite_sample")

    figA, axA = plt.subplots(figsize=(5.0, 3.6))
    for name, col in zip(dicts, [COLORS["muted"], COLORS["param"], COLORS["bins"], COLORS["accent"]]):
        med, lo, hi = best_curve[name]
        axA.errorbar(NS, med, yerr=[np.array(med) - np.array(lo), np.array(hi) - np.array(med)],
                     marker="o", ms=3, capsize=2, color=col, label=name)
    axA.set_xscale("log", base=2); axA.set_xlabel("sample size N")
    axA.set_ylabel("out-of-sample $R^2_{shape}$"); axA.set_ylim(-0.5, 1.05)
    axA.set_title("Figure S11. Finite-sample estimation"); axA.legend(fontsize=7)
    figA.tight_layout(); print("saved", save(figA, "figS11_finite_sample"))

    figB, axB = plt.subplots(figsize=(5.0, 3.6))
    for lam, col in zip(LAMS, [COLORS["muted"], COLORS["accent"], COLORS["param"], COLORS["bins"]]):
        axB.plot(NS, [rbf_by_lambda[N][lam] for N in NS], "o-", ms=3, color=col, label=f"$\\lambda$={lam}")
    axB.set_xscale("log", base=2); axB.set_xlabel("sample size N")
    axB.set_ylabel("median out-of-sample $R^2$"); axB.set_ylim(-0.5, 1.05)
    axB.set_title("Figure S12. Ridge regularization, 9-RBF dictionary"); axB.legend(fontsize=7)
    figB.tight_layout(); print("saved", save(figB, "figS12_ridge"))


if __name__ == "__main__":
    run()
