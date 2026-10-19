# -*- coding: utf-8 -*-
"""Three core-identity experiments.

Figure S14 (Experiment S9): the Fisher rate decomposition. The full mean-fitness
rate equals the fitness variance, the projected rate is the explained component,
and the gap is the squared defect.

Figure S15 (Experiment S10): exact symmetric stalling for f(x) = -(x^2 - 4)^2 and
the symmetry-breaking map of the projected mean.

Figure S16 (Experiment S11): convergence orders, moment error slope ~2 and density
error slope ~1 on log-log axes.
"""
import numpy as np
import matplotlib.pyplot as plt
from igr import (Grid, gaussian_pdf, normalize, expectation, moments, exact_tilt,
                 integrate_param, raw_defect_gaussian, gaussian_scores, project,
                 l1_distance)
from igr import fitness
from igr.plotting import setup, save, COLORS


def fig_s14(g):
    f = fitness.perturbed_quadratic(0.40)
    fx = f(g.x)
    teval = np.linspace(0, 2.0, 81)
    t, mu, var = integrate_param(0.0, 1.0, f, g, (0, 2.0), teval)
    full = np.empty(len(t)); proj = np.empty(len(t)); gap = np.empty(len(t))
    for i in range(len(t)):
        p = normalize(gaussian_pdf(g.x, mu[i], var[i]), g)
        Ef = expectation(fx, p, g)
        full[i] = expectation((fx - Ef) ** 2, p, g)             # Var_P(f)
        res = project(fx - Ef, gaussian_scores(g.x, mu[i], var[i]), p, g)
        proj[i] = res.r2 * res.var_w                            # explained = c^T F^+ c
        gap[i] = res.defect2
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    ax.plot(t, full, color=COLORS["true"], label=r"full rate = $Var_P(f)$")
    ax.plot(t, proj, "--", color=COLORS["param"], label="projected rate")
    ax.fill_between(t, proj, full, color=COLORS["bins"], alpha=0.25, label=r"gap = $\delta^2$")
    ax.set_xlabel("t"); ax.set_ylabel("mean-fitness increase rate")
    ax.set_title("Figure S14. Numerical rate decomposition"); ax.legend(fontsize=8)
    fig.tight_layout(); print("saved", save(fig, "figS14_rate_decomposition"))


def fig_s15(g):
    a = 2.0; var_star = a ** 2 / 3.0
    f = fitness.stalling(a); fx = f(g.x)
    p_star = normalize(gaussian_pdf(g.x, 0.0, var_star), g)
    d = np.sqrt(raw_defect_gaussian(fx, 0.0, var_star, p_star, g))
    tilt = exact_tilt(p_star, fx, 5.0, g)
    # symmetry-breaking map: sign of projected mean after a finite horizon
    mu0s = np.linspace(-0.6, 0.6, 61)
    signs = []
    for m0 in mu0s:
        t, mu, var = integrate_param(m0, var_star, f, g, (0, 12.0),
                                     np.array([0, 12.0]), rtol=1e-9, atol=1e-12)
        signs.append(np.sign(mu[-1]) if abs(mu[-1]) > 1e-3 else 0.0)
    signs = np.array(signs)
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.3))
    ax[0].fill_between(g.x, tilt, color=COLORS["true"], alpha=0.18, label="tilt at t=5 (double well)")
    ax[0].plot(g.x, tilt, color=COLORS["true"])
    ax[0].plot(g.x, p_star, "--", color=COLORS["param"], label=r"stalled $\mathcal{N}(0,4/3)$")
    ax[0].set_xlim(-5, 5); ax[0].set_xlabel("x"); ax[0].set_ylabel("density")
    ax[0].set_title(f"(a) stalled Gaussian, defect $\\delta$={d:.3f}"); ax[0].legend(fontsize=7)
    cols = np.where(signs > 0, 0, np.where(signs < 0, 1, 2))
    ax[1].scatter(mu0s, signs, c=[["#b0451f", "#1f6f8b", "#888888"][c] for c in cols], s=14)
    ax[1].set_xlabel(r"$\mu_0$"); ax[1].set_ylabel(r"sign of $\mu(T)$")
    ax[1].set_title("(b) symmetry-breaking outcome")
    fig.suptitle("Figure S15. Exact Gaussian stalling and symmetry breaking", y=1.02)
    fig.tight_layout(); print("saved", save(fig, "figS15_stalling"))


def fig_s16(g):
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.3))
    ts = np.logspace(-2.3, -0.5, 10)
    print("convergence slopes:")
    for eps, col in [(0.1, COLORS["param"]), (0.2, COLORS["accent"]),
                     (0.4, COLORS["bins"]), (0.6, COLORS["muted"])]:
        f = fitness.perturbed_quadratic(eps); fx = f(g.x)
        rho0 = normalize(gaussian_pdf(g.x, 0.0, 1.0), g)
        merr = []; derr = []
        for t in ts:
            tt, mu, var = integrate_param(0.0, 1.0, f, g, (0, t), np.array([0, t]),
                                          rtol=1e-11, atol=1e-13)
            true = exact_tilt(rho0, fx, t, g)
            mt, vt = moments(true, g)
            merr.append(abs(mu[-1] - mt) + abs(var[-1] - vt))
            derr.append(l1_distance(true, normalize(gaussian_pdf(g.x, mu[-1], var[-1]), g), g))
        sm = np.polyfit(np.log(ts), np.log(merr), 1)[0]
        sd = np.polyfit(np.log(ts), np.log(derr), 1)[0]
        print(f"  eps={eps}: moment slope {sm:.4f}, density slope {sd:.4f}")
        ax[0].loglog(ts, merr, "o-", color=col, ms=3, label=f"$\\epsilon$={eps}")
        ax[1].loglog(ts, derr, "o-", color=col, ms=3)
    ax[0].set_title("(a) moment error (slope ~ 2)"); ax[0].set_xlabel("t"); ax[0].set_ylabel("moment error")
    ax[1].set_title("(b) density error (slope ~ 1)"); ax[1].set_xlabel("t"); ax[1].set_ylabel(r"$L^1$ error")
    ax[0].legend(fontsize=7)
    fig.suptitle("Figure S16. Convergence-order test", y=1.02)
    fig.tight_layout(); print("saved", save(fig, "figS16_convergence"))


def run():
    setup()
    g = Grid(-10, 10, 24001)
    fig_s14(g)
    fig_s15(g)
    fig_s16(g)


if __name__ == "__main__":
    run()
