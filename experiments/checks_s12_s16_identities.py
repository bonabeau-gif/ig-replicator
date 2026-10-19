# -*- coding: utf-8 -*-
"""Numerical identity checks (Experiments S12-S16, text-only results).

S12 coordinate invariance (Prop 2(i)): the represented relative velocity is
    unchanged under an invertible reparametrization of the family.
S13 monotone invariance (Prop 8): shaped quantities are identical under a strictly
    increasing transform of the objective, while the raw defect varies.
S16 frequency-dependent fitness (Cor 6.1): a pairwise kernel of degree <= 2 in the
    focal trait keeps the Gaussian defect at zero; a Gaussian smoothing kernel does not.
"""
import numpy as np
from igr import (Grid, gaussian_pdf, normalize, expectation, gaussian_scores, project,
                 weighted_quantile, W_elite, shaped_signal, raw_defect_gaussian)
from igr import fitness


def check_coordinate_invariance(g):
    p = normalize(gaussian_pdf(g.x, 0.0, 1.0), g)
    f = fitness.perturbed_quadratic(0.4)(g.x)
    w = f - expectation(f, p, g)
    # parametrization A: scores {Z, Z^2 - 1}
    sA = gaussian_scores(g.x, 0.0, 1.0)
    rA = project(w, sA, p, g)
    velA = rA.coef @ (sA - np.array([[expectation(s, p, g)] for s in sA]))
    # parametrization B: an invertible linear remix of the same span
    T = np.array([[1.3, 0.4], [-0.7, 1.1]])
    sB = T @ sA
    rB = project(w, sB, p, g)
    velB = rB.coef @ (sB - np.array([[expectation(s, p, g)] for s in sB]))
    diff = float(np.max(np.abs(velA - velB)))
    print(f"S12 coordinate invariance: max |velocity_A - velocity_B| = {diff:.2e}")


def check_monotone_invariance(g):
    p = normalize(gaussian_pdf(g.x, 0.0, 1.0), g)
    base = fitness.perturbed_quadratic(0.4)(g.x)
    maps = {"identity": lambda y: y, "shift+scale": lambda y: 3 * y + 5,
            "cube": lambda y: y ** 3, "arctan": lambda y: np.arctan(y),
            "exp": lambda y: np.exp(0.5 * y)}
    scores = gaussian_scores(g.x, 0.0, 1.0)
    r2s = []; raw = []
    for name, m in maps.items():
        fm = m(base)
        res = project(shaped_signal(fm, p, g, lambda u: W_elite(u, 0.25))[0], scores, p, g)
        r2s.append(res.r2)
        raw.append(raw_defect_gaussian(fm, 0.0, 1.0, p, g))
    print(f"S13 monotone invariance: shaped R^2 spread = {max(r2s) - min(r2s):.2e} "
          f"(identical); raw defect varies by factor {max(raw) / max(min(raw), 1e-300):.1f}")


def check_frequency_dependent(g):
    p = normalize(gaussian_pdf(g.x, 0.0, 1.0), g)
    Pz = p.copy()  # interacting-trait distribution = current population
    # degree-2 pairwise kernel: Pi(x,z) = a0(z) + a1(z) x + a2(z) x^2
    a0 = np.cos(g.x); a1 = 0.5 * np.ones_like(g.x); a2 = -0.3 * np.ones_like(g.x)
    # f(x|P) = A0 + A1 x + A2 x^2 with A_k = E_z[a_k(z)] -> quadratic in x
    A0 = expectation(np.cos(g.x), Pz, g); A1 = 0.5; A2 = -0.3
    f_poly = A0 + A1 * g.x + A2 * g.x ** 2
    d_poly = raw_defect_gaussian(f_poly, 0.0, 1.0, p, g)
    # Gaussian smoothing kernel Pi(x,z) = exp(-(x-z)^2 / 2b^2); f(x|P) = convolution
    b = 0.8
    f_kernel = np.array([expectation(np.exp(-0.5 * (xi - g.x) ** 2 / b ** 2), Pz, g) for xi in g.x[::50]])
    xk = g.x[::50]
    # defect of the kernel fitness on the subsampled grid via a local Gaussian projection
    pk = normalize(gaussian_pdf(xk, 0.0, 1.0), Grid(xk[0], xk[-1], len(xk)))
    gk = Grid(xk[0], xk[-1], len(xk))
    d_kernel = raw_defect_gaussian(f_kernel, 0.0, 1.0, normalize(gaussian_pdf(xk, 0, 1), gk), gk)
    print(f"S16 frequency-dependent: degree-2 kernel defect = {d_poly:.2e} (zero); "
          f"Gaussian kernel defect = {d_kernel:.3e} (positive)")


def run():
    g = Grid(-10, 10, 24001)
    check_coordinate_invariance(g)
    check_monotone_invariance(g)
    check_frequency_dependent(g)


if __name__ == "__main__":
    run()
