# -*- coding: utf-8 -*-
"""Experiment S4: multimodal fitness with elite shaping. Figure S7, Table S3.

The raw fitness has two modes near +/-2, evaluated under a broad Gaussian baseline
with elite top-25% shaping. The single Gaussian is a poor near-invariant family;
rank bins are exact by construction; Legendre-rank and RBF dictionaries are strong
but the RBF Gram matrix is far more ill-conditioned. A two-component mixture
captures basin splitting but not the within-basin elite threshold.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from igr import (Grid, gaussian_pdf, normalize, gaussian_scores, legendre_rank,
                 rank_bins, rbf, stack, weighted_quantile, W_elite, shaped_signal, project)
from igr import fitness
from igr.plotting import setup, save, save_table

BROAD_VAR = 2.5 ** 2
TAU = 0.25


def mixture_pdf_scores(x, mus, sigmas, pis):
    comps = np.vstack([pis[k] * gaussian_pdf(x, mus[k], sigmas[k] ** 2) for k in range(len(mus))])
    rho = comps.sum(0)
    resp = comps / np.maximum(rho, 1e-300)
    rows = []
    for k in range(len(mus)):
        rows.append(resp[k] * (x - mus[k]) / sigmas[k] ** 2)                       # d/dmu_k
        rows.append(resp[k] * ((x - mus[k]) ** 2 / sigmas[k] ** 2 - 1.0))          # d/dlog sigma_k
    rows.append(resp[0] / pis[0] - resp[1] / pis[1])                               # d/dpi
    return rho, np.vstack(rows)


def evaluate(name, p, scores, f_vals, grid):
    res = project(shaped_signal(f_vals, p, grid, lambda u: W_elite(u, TAU))[0],
                  scores, p, grid)
    return {"family": name, "R2_shape": res.r2, "delta_shape": res.defect,
            "directions": scores.shape[0], "condition no.": res.cond}


def run():
    setup()
    g = Grid(-10, 10, 24001)
    f = fitness.multimodal(width=0.6, centers=(-2.0, 2.0))
    fx = f(g.x)
    pG = normalize(gaussian_pdf(g.x, 0.0, BROAD_VAR), g)
    qG = weighted_quantile(fx, pG, g)
    centers9 = np.linspace(-4, 4, 9); h9 = 0.8 * (centers9[1] - centers9[0])
    centers17 = np.linspace(-5, 5, 17); h17 = 0.8 * (centers17[1] - centers17[0])

    rows = []
    rows.append(evaluate("Gauss + rank bins K=8", pG,
                         stack(gaussian_scores(g.x, 0, BROAD_VAR), rank_bins(qG, 8)), fx, g))
    rows.append(evaluate("Gauss + Legendre K=8", pG,
                         stack(gaussian_scores(g.x, 0, BROAD_VAR), legendre_rank(qG, 8)), fx, g))
    rows.append(evaluate("Flow-like RBF K=17", pG,
                         stack(gaussian_scores(g.x, 0, BROAD_VAR), rbf(g.x, centers17, h17)), fx, g))
    rows.append(evaluate("Gauss + RBF K=9", pG,
                         stack(gaussian_scores(g.x, 0, BROAD_VAR), rbf(g.x, centers9, h9)), fx, g))
    # 2-component mixture family (its own state and scores)
    pM, mix_scores = mixture_pdf_scores(g.x, [-2.0, 2.0], [1.0, 1.0], [0.5, 0.5])
    pM = normalize(pM, g)
    rows.append(evaluate("2-component Gaussian mixture", pM, mix_scores, fx, g))
    rows.append(evaluate("Single Gaussian", pG, gaussian_scores(g.x, 0, BROAD_VAR), fx, g))

    df = pd.DataFrame(rows).sort_values("R2_shape", ascending=False).reset_index(drop=True)
    print(df.to_string(index=False))
    save_table(df, "tableS3_multimodal")

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.barh(df.family, df.R2_shape, color="#1f6f8b")
    ax.set_xlabel(r"$R^2_{shape}$"); ax.set_xlim(0, 1.05)
    ax.set_title("Figure S7. Multimodal objective under top-25% shaping")
    ax.invert_yaxis()
    fig.tight_layout(); print("saved", save(fig, "figS7_multimodal"))


if __name__ == "__main__":
    run()
