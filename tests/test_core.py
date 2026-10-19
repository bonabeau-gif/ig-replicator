# -*- coding: utf-8 -*-
"""Validate the core machinery against analytic values stated in the supplement."""
import numpy as np
from scipy.stats import norm
import igr
from igr import Grid, gaussian_pdf, raw_defect_gaussian, blowup_time, analytic_sinusoid_defect
from igr import var_U, W_linear, W_elite, gaussian_scores, shaped_projection, normalize


def test_quadratic_zero_defect():
    g = Grid(-12, 12, 24001)
    p = normalize(gaussian_pdf(g.x, 0.0, 1.0), g)
    f = 0.30 * g.x - 0.20 * g.x ** 2
    d2 = raw_defect_gaussian(f, 0.0, 1.0, p, g)
    assert d2 < 1e-8, d2


def test_sinusoid_defect_matches_analytic():
    g = Grid(-14, 14, 60001)
    eps, k, var = 0.5, 1.5, 1.0
    p = normalize(gaussian_pdf(g.x, 0.0, var), g)
    f = eps * np.sin(k * g.x)
    num = raw_defect_gaussian(f, 0.0, var, p, g)
    ana = analytic_sinusoid_defect(eps, k, var)
    assert abs(num - ana) < 1e-4, (num, ana)


def test_blowup_times():
    assert abs(blowup_time(0.089, 1.0) - 5.61798) < 1e-3
    assert abs(blowup_time(0.20, 1.0) - 2.5) < 1e-9


def test_stalling_defect():
    a = 2.0
    var_star = a ** 2 / 3.0            # 4/3
    g = Grid(-10, 10, 40001)
    p = normalize(gaussian_pdf(g.x, 0.0, var_star), g)
    f = -((g.x ** 2 - a ** 2) ** 2)
    d2 = raw_defect_gaussian(f, 0.0, var_star, p, g)
    expected = np.sqrt(8 * a ** 8 / 27.0) ** 2   # = 8 a^8 / 27
    assert abs(np.sqrt(d2) - np.sqrt(expected)) < 1e-2, (np.sqrt(d2), np.sqrt(expected))


def test_var_U():
    assert abs(var_U(W_linear) - 1.0 / 12.0) < 1e-5
    assert abs(var_U(lambda u: W_elite(u, 0.25)) - 0.25 * 0.75) < 1e-4


def test_elite_gaussian_r2():
    # Eq (S4): linear fitness, elite tau=0.25, ordinary Gaussian scores -> R^2 ~ 0.661
    g = Grid(-12, 12, 24001)
    p = normalize(gaussian_pdf(g.x, 0.0, 1.0), g)
    f = g.x.copy()
    scores = gaussian_scores(g.x, 0.0, 1.0)
    res = shaped_projection(f, p, g, lambda u: W_elite(u, 0.25), scores)
    a_tau = norm.ppf(0.75)
    phi = norm.pdf(a_tau)
    d2_ana = 0.25 * 0.75 - phi ** 2 - 0.5 * a_tau ** 2 * phi ** 2
    r2_ana = 1 - d2_ana / (0.25 * 0.75)
    assert abs(res.r2 - r2_ana) < 0.02, (res.r2, r2_ana)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("all core checks passed")
