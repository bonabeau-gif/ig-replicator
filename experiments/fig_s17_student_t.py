# -*- coding: utf-8 -*-
"""Experiment S15: Student-t conjugacy along the quadratic-to-rational homotopy.
Figure S17.

The Gaussian family is invariant under quadratic fitness; the Student-t family is
invariant under rational fitness [alpha z + beta z^2]/(nu + z^2). Along the
homotopy f_s = (1-s) q + s r, the Gaussian and Student-t explained fractions cross,
and the Gaussian explanation of the pure rational fitness approaches 1 as nu grows
(R^2 = 0.99312 at nu = 30, above the 0.99 ceiling that the blanket threshold used).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from igr import (Grid, gaussian_pdf, normalize, gaussian_scores, expectation, project,
                 student_t_pdf, student_t_scores)
from igr.plotting import setup, save, save_table, COLORS


def raw_r2(f_vals, p, scores, grid):
    w = f_vals - expectation(f_vals, p, grid)
    return project(w, scores, p, grid).r2


def run():
    setup()
    g = Grid(-80, 80, 60001)
    z = g.x
    f_quad = z ** 2
    svals = np.linspace(0, 1, 21)

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5))
    for nu, col in [(3.0, COLORS["param"]), (8.0, COLORS["accent"])]:
        pG = normalize(gaussian_pdf(z, 0.0, 1.0), g)
        pT = normalize(student_t_pdf(z, 0.0, 1.0, nu), g)
        sG = gaussian_scores(z, 0.0, 1.0)
        sT = student_t_scores(z, 0.0, 1.0, nu)
        f_rat = z ** 2 / (nu + z ** 2)
        rg = []; rt = []
        for s in svals:
            f_s = (1 - s) * f_quad + s * f_rat
            rg.append(raw_r2(f_s, pG, sG, g))
            rt.append(raw_r2(f_s, pT, sT, g))
        ax[0].plot(svals, rg, "-", color=col, label=f"Gaussian, $\\nu$={nu:.0f}")
        ax[0].plot(svals, rt, "--", color=col, label=f"Student-$t$, $\\nu$={nu:.0f}")
    ax[0].set_xlabel("homotopy parameter s (quadratic -> rational)")
    ax[0].set_ylabel(r"explained fraction $R^2$")
    ax[0].set_title("(a) explained fractions cross"); ax[0].legend(fontsize=7)

    # Gaussian explanation of the pure rational fitness vs nu
    nus = [2, 3, 5, 8, 12, 20, 30, 50]
    pG = normalize(gaussian_pdf(z, 0.0, 1.0), g); sG = gaussian_scores(z, 0.0, 1.0)
    rows = []
    for nu in nus:
        f_rat = z ** 2 / (nu + z ** 2)
        r2 = raw_r2(f_rat, pG, sG, g)
        rows.append({"nu": nu, "Gaussian R2 of rational fitness": r2})
    df = pd.DataFrame(rows)
    ax[1].plot(nus, df["Gaussian R2 of rational fitness"], "o-", color=COLORS["bins"])
    ax[1].axhline(0.99, ls=":", color="0.5", label="0.99 ceiling")
    ax[1].set_xlabel(r"$\nu$"); ax[1].set_ylabel(r"Gaussian $R^2$ of rational fitness")
    ax[1].set_title("(b) separation shrinks as $\\nu$ grows"); ax[1].legend(fontsize=7)
    fig.suptitle("Figure S17. Gaussian and Student-t explained fractions", y=1.02)
    fig.tight_layout(); print("saved", save(fig, "figS17_student_t"))
    print(df.to_string(index=False))
    save_table(df, "tableS_student_t_nu")


if __name__ == "__main__":
    run()
