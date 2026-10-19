# -*- coding: utf-8 -*-
"""Experiment S5: does the shaped defect predict one-step tracking error?
Figure S8, Table S4.

For an ensemble of nonquadratic landscapes, the true one-step shaped update
p_true propto p exp(eta w) is compared with the projected update
p_proj propto p exp(eta Pi w), and the initial shaped defect is correlated with
the one-step error under relative L2, TV, and KL for several step sizes eta.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from igr import (Grid, gaussian_pdf, normalize, gaussian_scores, weighted_quantile,
                 W_elite, shaped_signal, project, relative_l2, tv_distance, kl_divergence)
from igr import fitness
from igr.plotting import setup, save, save_table, COLORS

MU, VAR = 0.0, 1.0
ETAS = [0.25, 0.5, 1.0]


def one_step(p, w, proj_w, eta, grid):
    a = normalize(p * np.exp(eta * (w - w.max())), grid)
    b = normalize(p * np.exp(eta * (proj_w - proj_w.max())), grid)
    return a, b


def run():
    setup()
    g = Grid(-10, 10, 24001)
    p = normalize(gaussian_pdf(g.x, MU, VAR), g)
    W = lambda u: W_elite(u, 0.25)
    scores = gaussian_scores(g.x, MU, VAR)

    # ensemble of landscapes spanning a range of shaped defects
    problems = []
    for eps in np.linspace(0.0, 0.8, 11):
        for k in (1.0, 1.5, 2.0):
            problems.append(fitness.perturbed_quadratic(eps, k=k))
    records = {eta: {"defect": [], "relL2": [], "TV": [], "KL": []} for eta in ETAS}
    for f in problems:
        fx = f(g.x)
        w, q = shaped_signal(fx, p, g, W)
        res = project(w, scores, p, g)
        proj_w = res.coef @ (gaussian_scores(g.x, MU, VAR) -
                             np.array([[np.trapezoid(s * p, g.x)] for s in gaussian_scores(g.x, MU, VAR)]))
        for eta in ETAS:
            a, b = one_step(p, w, proj_w, eta, g)
            records[eta]["defect"].append(res.defect)
            records[eta]["relL2"].append(relative_l2(a, b, g))
            records[eta]["TV"].append(tv_distance(a, b, g))
            records[eta]["KL"].append(kl_divergence(a, b, g))

    rows = []
    for eta in ETAS:
        d = np.array(records[eta]["defect"])
        for metric in ("relL2", "TV", "KL"):
            e = np.array(records[eta][metric])
            m = np.isfinite(e)
            rows.append({"eta": eta, "metric": metric,
                         "Pearson": pearsonr(d[m], e[m]).statistic,
                         "Spearman": spearmanr(d[m], e[m]).statistic})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    save_table(df, "tableS4_defect_predicts")

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    cols = {"relL2": COLORS["param"], "TV": COLORS["accent"], "KL": COLORS["bins"]}
    eta = 1.0
    d = np.array(records[eta]["defect"])
    for metric, col in cols.items():
        e = np.array(records[eta][metric]); e = e / np.nanmax(e)
        ax.scatter(d, e, s=16, color=col, label=metric, alpha=0.8)
    ax.set_xlabel(r"initial shaped defect $\delta_{shape}$")
    ax.set_ylabel("one-step error (normalized)")
    ax.set_title(r"Figure S8. Shaped defect predicts" + "\n" + r"one-step error ($\eta=1$)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    print("saved", save(fig, "figS8_defect_predicts"))


if __name__ == "__main__":
    run()
