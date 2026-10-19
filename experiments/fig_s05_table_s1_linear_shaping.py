# -*- coding: utf-8 -*-
"""Experiment S2: linear fitness f(x) = x under Gaussian search. Figure S5, Table S1."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from igr import Grid, gaussian_pdf, normalize
from igr.plotting import setup, save, save_table, COLORS
from _common import r2_table, gauss_only, gauss_plus_legendre, gauss_plus_bins

MU, VAR = 0.0, 1.0


def run():
    setup()
    g = Grid(-12, 12, 24001)
    p = normalize(gaussian_pdf(g.x, MU, VAR), g)
    f = g.x.copy()
    specs = [
        ("Gauss + Legendre rank, K=16", gauss_plus_legendre(MU, VAR, g.x, 16)),
        ("Gauss + Legendre rank, K=2", gauss_plus_legendre(MU, VAR, g.x, 2)),
        ("Gauss + Legendre rank, K=4", gauss_plus_legendre(MU, VAR, g.x, 4)),
        ("Gauss + Legendre rank, K=8", gauss_plus_legendre(MU, VAR, g.x, 8)),
        ("Gauss + rank bins, K=8", gauss_plus_bins(MU, VAR, g.x, 8)),
        ("Gaussian scores", gauss_only(MU, VAR, g.x)),
    ]
    res = r2_table(f, p, g, MU, VAR, specs)
    rows = []
    for name, _ in specs:
        rows.append({"family": name,
                     "linear rank W(u)=u": res[name]["linear rank"].r2,
                     "elite top 25%": res[name]["elite top 25%"].r2,
                     "smooth top 25%": res[name]["smooth top 25%"].r2})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    save_table(df, "tableS1_linear_shaping")

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    cols = list(df.columns[1:]); x = np.arange(len(df)); w = 0.25
    for j, c in enumerate(cols):
        ax.bar(x + (j - 1) * w, df[c].values, w, label=c)
    ax.set_xticks(x); ax.set_xticklabels([n.replace("Gauss + ", "").replace(", ", "\n") for n in df.family],
                                         rotation=0, fontsize=6.5)
    ax.set_ylabel(r"$R^2_{shape}$"); ax.set_ylim(0, 1.05); ax.legend(fontsize=7)
    ax.set_title("Figure S5. Linear fitness under Gaussian search")
    fig.tight_layout()
    print("saved", save(fig, "figS5_linear_shaping"))


if __name__ == "__main__":
    run()
