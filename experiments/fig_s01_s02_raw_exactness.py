# -*- coding: utf-8 -*-
"""Experiment S1: raw-fitness exactness and near-exactness. Figures S1 and S2.

The exact replicator tilt is compared at t = 2 with the param (Fisher moment)
and proj (L2(dx) density-velocity) Gaussian surrogates, for three fitness
functions spanning quadratic to strongly non-Gaussian.
"""
import numpy as np
import matplotlib.pyplot as plt
from igr import (Grid, gaussian_pdf, normalize, moments, exact_tilt,
                 integrate_param, integrate_proj, l1_distance, kl_divergence)
from igr import fitness
from igr.plotting import setup, save, COLORS

CASES = [
    ("quadratic", fitness.quadratic),
    ("quadratic + small sinusoid", fitness.perturbed_quadratic(0.30)),
    ("pure sinusoid", fitness.sinusoid(1.2, 1.5)),
]
MU0, VAR0, T = 0.0, 1.0, 2.0


def run():
    setup()
    g = Grid(-10, 10, 24001)
    rho0 = normalize(gaussian_pdf(g.x, MU0, VAR0), g)
    teval = np.linspace(0, T, 101)

    figD, axD = plt.subplots(2, 3, figsize=(10.5, 5.2))
    figM, axM = plt.subplots(2, 3, figsize=(11, 5.2))
    print(f"{'case':28} {'L1 param':>10} {'L1 proj':>10} {'KL param':>10} {'KL proj':>10}")
    for j, (name, f) in enumerate(CASES):
        fx = f(g.x)
        rho_true = exact_tilt(rho0, fx, T, g)
        tp, mp, vp = integrate_param(MU0, VAR0, f, g, (0, T), teval)
        to, mo, vo = integrate_proj(MU0, VAR0, f, g, (0, T), teval)
        dens_param = normalize(gaussian_pdf(g.x, mp[-1], vp[-1]), g)
        dens_proj = normalize(gaussian_pdf(g.x, mo[-1], vo[-1]), g)
        l1p = l1_distance(rho_true, dens_param, g); l1o = l1_distance(rho_true, dens_proj, g)
        klp = kl_divergence(rho_true, dens_param, g); klo = kl_divergence(rho_true, dens_proj, g)
        print(f"{name:28} {l1p:>10.4g} {l1o:>10.4g} {klp:>10.4g} {klo:>10.4g}")

        # densities (top row) and fitness + initial density (bottom row)
        ax = axD[0, j]
        ax.fill_between(g.x, rho_true, color=COLORS["true"], alpha=0.15)
        ax.plot(g.x, rho_true, color=COLORS["true"], label="true")
        ax.plot(g.x, dens_param, "--", color=COLORS["param"], label="param")
        ax.plot(g.x, dens_proj, ":", color=COLORS["proj"], label="proj")
        ax.set_xlim(-5, 5); ax.set_title(f"({'abc'[j]}) {name}")
        if j == 0: ax.legend()
        ax2 = axD[1, j]
        ax2.plot(g.x, fx, color=COLORS["accent"], label="f(x)")
        ax2.plot(g.x, rho0 / rho0.max() * np.nanmax(fx), color=COLORS["muted"], alpha=0.6, label=r"$\rho_0$ (scaled)")
        ax2.set_xlim(-5, 5); ax2.set_xlabel("x")
        if j == 0: ax2.legend()

        # moment trajectories
        mu_true = np.array([moments(exact_tilt(rho0, fx, t, g), g)[0] for t in teval])
        var_true = np.array([moments(exact_tilt(rho0, fx, t, g), g)[1] for t in teval])
        axM[0, j].plot(teval, mu_true, color=COLORS["true"], label="true")
        axM[0, j].plot(tp, mp, "--", color=COLORS["param"], label="param")
        axM[0, j].plot(to, mo, ":", color=COLORS["proj"], label="proj")
        axM[0, j].set_title(f"({'abc'[j]}) {name}"); axM[0, j].set_ylabel(r"$\mu(t)$")
        if j == 0: axM[0, j].legend()
        axM[1, j].plot(teval, var_true, color=COLORS["true"])
        axM[1, j].plot(tp, vp, "--", color=COLORS["param"])
        axM[1, j].plot(to, vo, ":", color=COLORS["proj"])
        axM[1, j].set_ylabel(r"$\sigma^2(t)$"); axM[1, j].set_xlabel("t")

    axD[0, 0].set_ylabel("density"); axM.ravel()
    figD.suptitle("Figure S1. Replicator truth vs Gaussian surrogates at t = 2", y=1.01)
    figM.suptitle("Figure S2. Moment trajectories", y=1.01)
    figD.tight_layout(); figM.tight_layout()
    p1 = save(figD, "figS1_densities"); p2 = save(figM, "figS2_moments")
    print("saved", p1, p2)


if __name__ == "__main__":
    run()
