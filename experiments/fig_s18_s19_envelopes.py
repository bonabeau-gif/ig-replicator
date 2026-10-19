# -*- coding: utf-8 -*-
"""Figure S18 (Experiment S18): one-sided Lipschitz envelope.

The worst-case Gronwall constant L = 3M is replaced by the sampled logarithmic
norm (one-sided Lipschitz constant) of the projected moment-flow Jacobian along
the trajectory. The resulting envelope is far tighter than the worst-case bound
while still dominating the observed tracking error.

Figure S19 (Experiment S19): the family floor against dynamic drift.

The tracking error is decomposed into the irreducible Gaussian family floor
(true tilt vs its moment-matched Gaussian) and the projected path's excess over
that moment-matched m-projection, whose small-time slope is ~2.
"""
import numpy as np
import matplotlib.pyplot as plt
from igr import (Grid, gaussian_pdf, normalize, moments, exact_tilt, integrate_param,
                 raw_defect_gaussian, l1_distance, param_rhs)
from igr import fitness
from igr.plotting import setup, save, COLORS


def moment_jacobian(mu, var, f, g, h=1e-5):
    def rhs(m, v):
        return np.array(param_rhs(m, v, f, g))
    d_mu = (rhs(mu + h, var) - rhs(mu - h, var)) / (2 * h)
    d_var = (rhs(mu, var + h) - rhs(mu, var - h)) / (2 * h)
    return np.column_stack([d_mu, d_var])


def fig_s18(g):
    teval = np.linspace(0, 1.0, 201)
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    maxlog = -np.inf
    for eps in (0.2, 0.4, 0.6):
        f = fitness.sinusoid(eps, 1.5); fx = f(g.x)   # smooth normalized perturbation
        t, mu, var = integrate_param(0.0, 1.0, f, g, (0, 1.0), teval, rtol=1e-10, atol=1e-13)
        rho0 = normalize(gaussian_pdf(g.x, 0.0, 1.0), g)
        l1 = np.empty(len(t)); dfc = np.empty(len(t)); logn = np.empty(len(t))
        for i in range(len(t)):
            surr = normalize(gaussian_pdf(g.x, mu[i], var[i]), g)
            l1[i] = l1_distance(exact_tilt(rho0, fx, t[i], g), surr, g)
            dfc[i] = np.sqrt(raw_defect_gaussian(fx, mu[i], var[i], surr, g))
            J = moment_jacobian(mu[i], var[i], f, g)
            logn[i] = np.max(np.linalg.eigvalsh(0.5 * (J + J.T)))
        maxlog = max(maxlog, logn.max())
        mu_ls = max(logn.max(), 1e-6)
        acc = np.array([np.trapezoid(dfc[:i + 1], t[:i + 1]) for i in range(len(t))])
        env_one = np.array([np.trapezoid(np.exp(mu_ls * (t[i] - t[:i + 1])) * dfc[:i + 1], t[:i + 1])
                            for i in range(len(t))])
        M = 0.5 * (fx.max() - fx.min()); Lworst = 3 * M
        env_worst = (dfc.max() / Lworst) * (np.exp(Lworst * t) - 1.0)
        if eps == 0.4:
            ax.plot(t, l1, color=COLORS["true"], label="observed error")
            ax.plot(t, acc, "--", color=COLORS["param"], label=r"accumulated defect")
            ax.plot(t, env_one, "-.", color=COLORS["accent"], label="one-sided envelope")
            ax.plot(t, env_worst, ":", color=COLORS["bins"], label="worst-case envelope")
    ax.set_yscale("log"); ax.set_xlabel("t"); ax.set_ylabel("error / bound")
    ax.set_title(f"Figure S18. One-sided vs worst-case envelope\n(max log norm {maxlog:.3f})")
    ax.legend(fontsize=7)
    fig.tight_layout(); print(f"max sampled logarithmic norm = {maxlog:.3f}")
    print("saved", save(fig, "figS18_one_sided"))


def fig_s19(g):
    f = fitness.perturbed_quadratic(0.40); fx = f(g.x)
    rho0 = normalize(gaussian_pdf(g.x, 0.0, 1.0), g)
    teval = np.linspace(0, 1.0, 201)
    t, mu, var = integrate_param(0.0, 1.0, f, g, (0, 1.0), teval, rtol=1e-10, atol=1e-13)
    floor = np.empty(len(t)); excess = np.empty(len(t)); total = np.empty(len(t))
    for i in range(len(t)):
        tilt = exact_tilt(rho0, fx, t[i], g)
        mt, vt = moments(tilt, g)
        mproj = normalize(gaussian_pdf(g.x, mt, vt), g)
        path = normalize(gaussian_pdf(g.x, mu[i], var[i]), g)
        floor[i] = l1_distance(tilt, mproj, g)
        excess[i] = l1_distance(mproj, path, g)
        total[i] = l1_distance(tilt, path, g)
    m = (t > 0.02) & (t < 0.2)
    slope = np.polyfit(np.log(t[m]), np.log(np.maximum(excess[m], 1e-15)), 1)[0]
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.plot(t, total, color=COLORS["true"], label="total error")
    ax.plot(t, floor, "--", color=COLORS["param"], label="Gaussian family floor")
    ax.plot(t, excess, "-.", color=COLORS["bins"], label=f"excess over m-projection (slope {slope:.2f})")
    ax.set_xlabel("t"); ax.set_ylabel(r"$L^1$ distance")
    ax.set_title("Figure S19. Family floor vs dynamic drift"); ax.legend(fontsize=7)
    fig.tight_layout(); print(f"excess small-time slope = {slope:.3f}")
    print("saved", save(fig, "figS19_family_floor"))


def run():
    setup()
    g = Grid(-10, 10, 24001)
    fig_s18(g)
    fig_s19(g)


if __name__ == "__main__":
    run()
