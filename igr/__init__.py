# -*- coding: utf-8 -*-
"""igr: information-geometric replicator dynamics, invariance defect, and fitness shaping.

Reference implementation for the supplementary experiments of
"Information-geometric properties of replicator dynamics with continuous traits"
(E. Bonabeau). The package provides the exact replicator tilt, the Fisher-projected
Gaussian moment flow, the raw and shaped invariance defects, the score dictionaries,
and the Student-t conjugate family used in Supplementary Experiments S1-S19.
"""
from . import quadrature, gaussian, replicator, defect, shaping, dictionaries, student_t, fitness
from .quadrature import (Grid, normalize, integrate, expectation, moments,
                         l1_distance, tv_distance, kl_divergence, relative_l2)
from .gaussian import gaussian_pdf, gaussian_scores, raw_defect_gaussian, blowup_time, analytic_sinusoid_defect
from .replicator import exact_tilt, integrate_param, integrate_proj, param_rhs, proj_rhs
from .defect import project, ProjectionResult
from .shaping import weighted_quantile, W_linear, W_elite, W_smooth_elite, var_U, shaped_signal, shaped_projection
from .dictionaries import legendre_rank, rank_bins, rbf, stack
from .student_t import student_t_pdf, student_t_scores, rational_conjugate_fitness

__version__ = "1.0.0"
