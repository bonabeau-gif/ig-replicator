# -*- coding: utf-8 -*-
"""The projection and defect machinery, Eqs. (25) and (57)-(59).

Given a centered target signal w(x) and a score dictionary s_1,...,s_K, all
sampled on the grid and weighted by the current density p, the defect is the
L2(p) residual of w after least-squares projection onto span{s_k}:

    F_{jk} = E_p[s_j s_k],  b_k = E_p[w s_k],  v* = F^+ b,
    delta^2 = E_p[w^2] - b^T F^+ b,   R^2 = (b^T F^+ b) / E_p[w^2].

Scores are centered under p before the Gram matrix is formed, and the
Moore-Penrose pseudoinverse is computed by SVD.
"""
import numpy as np
from .quadrature import expectation

__all__ = ["center_scores", "project", "ProjectionResult"]


class ProjectionResult:
    def __init__(self, defect2, r2, coef, cond, var_w):
        self.defect2 = defect2          # squared defect
        self.defect = np.sqrt(max(defect2, 0.0))
        self.r2 = r2                    # explained fraction
        self.coef = coef                # v* coefficients
        self.cond = cond                # Gram condition number
        self.var_w = var_w              # E_p[w^2]

    def __repr__(self):
        return f"ProjectionResult(R2={self.r2:.6f}, defect={self.defect:.3e}, cond={self.cond:.3e})"


def center_scores(scores, p, grid):
    """Subtract the p-weighted mean from each row of scores (shape (K, n))."""
    scores = np.atleast_2d(np.asarray(scores, float))
    means = np.array([expectation(s, p, grid) for s in scores])
    return scores - means[:, None]


def project(w, scores, p, grid, ridge=0.0, center=True):
    """Project centered signal w onto span(scores) in L2(p). Returns ProjectionResult.

    ridge, if positive, is added as ridge * mean(diag(F)) * I before the solve
    (Tikhonov regularization, as in the finite-sample experiments).
    """
    w = np.asarray(w, float)
    if center:
        scores = center_scores(scores, p, grid)
    else:
        scores = np.atleast_2d(np.asarray(scores, float))
    K = scores.shape[0]
    F = np.array([[expectation(si * sj, p, grid) for sj in scores] for si in scores])
    b = np.array([expectation(w * si, p, grid) for si in scores])
    var_w = expectation(w * w, p, grid)
    if ridge > 0:
        F = F + ridge * (np.trace(F) / max(K, 1)) * np.eye(K)
    Finv = np.linalg.pinv(F, rcond=1e-12)
    explained = float(b @ Finv @ b)
    coef = Finv @ b
    defect2 = max(var_w - explained, 0.0)
    r2 = explained / var_w if var_w > 0 else np.nan
    cond = float(np.linalg.cond(F)) if K > 0 else np.nan
    return ProjectionResult(defect2, r2, coef, cond, var_w)
