# -*- coding: utf-8 -*-
"""Figure S3: the invariance defect delta(theta).

(a) delta along the projected trajectory for a quadratic and two nonquadratic
    fitnesses; the quadratic stays at the floating-point floor.
(b) delta at t = 2 for f(x) = 0.30 x - 0.20 x^2 + eps sin(1.5 x) as eps grows.
"""
import numpy as np
import matplotlib.pyplot as plt
from igr import Grid, raw_defect_gaussian, integrate_param
from igr import fitness
from igr.plotting import setup, save, COLORS

MU0, VAR0, T = 0.0, 1.0, 2.0


def defect_along(f, g, teval):
    t, mu, var = integrate_param(MU0, VAR0, f, g, (0, T), teval)
    fx = f(g.x)
    d = np.array([raw_defect_gaussian(fx, mu[i], var[i], _pdf(g, mu[i], var[i]), g)
                  for i in range(len(t))])
    return t, np.sqrt(np.maximum(d, 0.0))


def _pdf(g, mu, var):
    from igr import gaussian_pdf, normalize
    return normalize(gaussian_pdf(g.x, mu, var), g)


def run():
    setup()
    g = Grid(-10, 10, 24001)
    teval = np.linspace(0, T, 81)

    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    cases = [("quadratic", fitness.quadratic, COLORS["param"]),
             (r"$\epsilon=0.3$ sinusoid", fitness.perturbed_quadratic(0.30), COLORS["accent"]),
             (r"$\epsilon=0.6$ sinusoid", fitness.perturbed_quadratic(0.60), COLORS["bins"])]
    for name, f, col in cases:
        t, d = defect_along(f, g, teval)
        ax[0].plot(t, np.maximum(d, 1e-16), color=col, label=name)
    ax[0].set_yscale("log"); ax[0].set_xlabel("t"); ax[0].set_ylabel(r"defect $\delta(\theta_t)$")
    ax[0].set_title("(a) defect along the trajectory"); ax[0].legend()

    eps_grid = np.linspace(0, 0.6, 25)
    d_eps = []
    for eps in eps_grid:
        f = fitness.perturbed_quadratic(eps)
        _, d = defect_along(f, g, np.array([0.0, T]))
        d_eps.append(d[-1])
    ax[1].plot(eps_grid, d_eps, "o-", color=COLORS["param"], ms=3)
    ax[1].set_xlabel(r"perturbation amplitude $\epsilon$"); ax[1].set_ylabel(r"defect at $t=2$")
    ax[1].set_title("(b) defect grows with nonquadratic amplitude")
    fig.tight_layout()
    print("saved", save(fig, "figS3_defect_departure"))


if __name__ == "__main__":
    run()
