# -*- coding: utf-8 -*-
"""Experiment S3: quadratic fitness f(x) = -x^2 before and after shaping.
Figure S6, Table S2. The raw quadratic is Gaussian-invariant (R^2 = 1); rank
shaping moves the target out of the Gaussian tangent space.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from igr import Grid, gaussian_pdf, normalize, expectation, project
from igr.plotting import setup, save, save_table
from _common import gauss_only, gauss_plus_legendre, gauss_plus_bins, r2_table

MU, VAR = 0.0, 1.0


def raw_r2(f_vals, p, grid, builder):
    """R^2 for the raw (unshaped) centered fitness projected onto the dictionary."""
    from igr import weighted_quantile
    q = weighted_quantile(f_vals, p, grid)
    scores = builder(q)
    w = f_vals - expectation(f_vals, p, grid)
    return project(w, scores, p, grid).r2


def run():
    setup()
    g = Grid(-12, 12, 24001)
    p = normalize(gaussian_pdf(g.x, MU, VAR), g)
    f = -g.x ** 2
    specs = [
        ("Gauss + Legendre rank, K=4", gauss_plus_legendre(MU, VAR, g.x, 4)),
        ("Gauss + Legendre rank, K=8", gauss_plus_legendre(MU, VAR, g.x, 8)),
        ("Gauss + rank bins, K=8", gauss_plus_bins(MU, VAR, g.x, 8)),
        ("Gaussian scores", gauss_only(MU, VAR, g.x)),
    ]
    res = r2_table(f, p, g, MU, VAR, specs)
    rows = []
    for name, builder in specs:
        rows.append({"family": name, "raw f=-x^2": raw_r2(f, p, g, builder),
                     "linear rank": res[name]["linear rank"].r2,
                     "elite top 25%": res[name]["elite top 25%"].r2,
                     "smooth top 25%": res[name]["smooth top 25%"].r2})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    save_table(df, "tableS2_quadratic_shaping")

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    cols = list(df.columns[1:]); x = np.arange(len(df)); w = 0.2
    for j, c in enumerate(cols):
        ax.bar(x + (j - 1.5) * w, df[c].values, w, label=c)
    ax.set_xticks(x); ax.set_xticklabels([n.replace("Gauss + ", "").replace(", ", "\n") for n in df.family],
                                         fontsize=6.5)
    ax.set_ylabel(r"$R^2$"); ax.set_ylim(0, 1.05); ax.legend(fontsize=7)
    ax.set_title("Figure S6. Raw quadratic is invariant; shaped quadratic is not")
    fig.tight_layout()
    print("saved", save(fig, "figS6_quadratic_shaping"))


if __name__ == "__main__":
    run()
