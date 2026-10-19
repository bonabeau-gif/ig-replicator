# -*- coding: utf-8 -*-
"""Multi-step tracking and calibration of the trajectory estimate (Section S3).
Figures S9 and S10, Table S5.

The true trajectory is the exact tilt; the surrogate follows the Fisher-projected
moment flow. We report the accumulated L1 error, the instantaneous defect, the
unamplified accumulated defect, and the Gronwall envelope of Eq. (49).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from igr import (Grid, gaussian_pdf, normalize, exact_tilt, integrate_param,
                 raw_defect_gaussian, l1_distance)
from igr import fitness
from igr.plotting import setup, save, save_table, COLORS

EPS = [0.0, 0.05, 0.10, 0.20, 0.40, 0.60]
T = 1.0


def trajectory(eps, g, teval):
    f = fitness.perturbed_quadratic(eps)
    fx = f(g.x)
    rho0 = normalize(gaussian_pdf(g.x, 0.0, 1.0), g)
    t, mu, var = integrate_param(0.0, 1.0, f, g, (0, T), teval, rtol=1e-10, atol=1e-13)
    l1 = np.empty(len(t)); dfc = np.empty(len(t))
    for i in range(len(t)):
        true_d = exact_tilt(rho0, fx, t[i], g)
        surr = normalize(gaussian_pdf(g.x, mu[i], var[i]), g)
        l1[i] = l1_distance(true_d, surr, g)
        dfc[i] = np.sqrt(raw_defect_gaussian(fx, mu[i], var[i], surr, g))
    M = 0.5 * (fx.max() - fx.min())           # bounded-support oscillation constant
    L = 3.0 * M
    return t, l1, dfc, L


def run():
    setup()
    g = Grid(-6, 6, 2401)
    teval = np.linspace(0, T, 501)

    figE, axE = plt.subplots(figsize=(4.2, 3.4))
    rows = []
    store = {}
    for eps in EPS:
        t, l1, dfc, L = trajectory(eps, g, teval)
        store[eps] = (t, l1, dfc, L)
        axE.plot(t, l1, label=f"$\\epsilon$={eps}")
        acc = np.trapezoid(dfc, t)
        env = (dfc.max() / L) * (np.exp(L * T) - 1.0) if L > 0 else 0.0
        rows.append({"eps": eps, "delta(theta_1)": dfc[-1], "L1 error": l1[-1],
                     "int_0^1 delta ds": acc, "Eq(49) envelope": env})
    axE.set_xlabel("t"); axE.set_ylabel(r"accumulated $L^1$ error")
    axE.set_title("Figure S9. Multi-step tracking error"); axE.legend(fontsize=7)
    figE.tight_layout(); print("saved", save(figE, "figS9_multistep"))

    # calibration at eps = 0.4
    t, l1, dfc, L = store[0.40]
    acc = np.array([np.trapezoid(dfc[:i + 1], t[:i + 1]) for i in range(len(t))])
    env = (np.maximum.accumulate(dfc) / L) * (np.exp(L * t) - 1.0)
    figC, axC = plt.subplots(figsize=(4.6, 3.4))
    axC.plot(t, l1, color=COLORS["true"], label=r"actual $L^1$ error")
    axC.plot(t, acc, "--", color=COLORS["param"], label=r"$\int_0^t\delta\,ds$")
    axC.plot(t, env, ":", color=COLORS["bins"], label="Gronwall envelope (49)")
    axC.set_yscale("log"); axC.set_xlabel("t"); axC.set_ylabel("error / bound")
    axC.set_title(r"Figure S10. Calibration at $\epsilon=0.4$"); axC.legend(fontsize=7)
    figC.tight_layout(); print("saved", save(figC, "figS10_calibration"))

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    save_table(df, "tableS5_multistep")


if __name__ == "__main__":
    run()
