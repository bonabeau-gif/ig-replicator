# -*- coding: utf-8 -*-
"""Figure S4: finite-time validity under f(x) = c x^2 with c > 0 (Section 3.11).

The variance equation var_dot = 2 c var^2 gives var(t) = var0 / (1 - 2 c var0 t),
which blows up at t* = 1/(2 c var0). Numerical param trajectories are compared
with the closed form up to 0.985 t*.
"""
import numpy as np
import matplotlib.pyplot as plt
from igr import Grid, integrate_param, blowup_time
from igr.plotting import setup, save, COLORS

VAR0 = 1.0


def run():
    setup()
    g = Grid(-14, 14, 40001)
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for c, col in [(0.089, COLORS["param"]), (0.20, COLORS["bins"])]:
        tstar = blowup_time(c, VAR0)
        teval = np.linspace(0, 0.985 * tstar, 120)
        f = lambda x, c=c: c * np.asarray(x, float) ** 2
        t, mu, var = integrate_param(0.0, VAR0, f, g, (0, teval[-1]), teval,
                                     rtol=1e-10, atol=1e-13)
        closed = VAR0 / (1.0 - 2.0 * c * VAR0 * t)
        ax.plot(t, var, color=col, label=f"c={c}, numerical ($t^*$={tstar:.3g})")
        ax.plot(t, closed, "--", color=col, alpha=0.7, label=f"c={c}, closed form")
    ax.set_xlabel("t"); ax.set_ylabel(r"$\sigma^2(t)$"); ax.set_yscale("log")
    ax.set_title("Figure S4. Finite-time validity under positive quadratic curvature")
    ax.legend(fontsize=7)
    fig.tight_layout()
    print("saved", save(fig, "figS4_finite_time"))


if __name__ == "__main__":
    run()
