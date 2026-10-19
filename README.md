# The invariance defect of continuous-trait parametric replicator dynamics: reproducibility package

Reference implementation for the supplementary experiments of *The invariance defect of 
continuous-trait parametric replicator dynamics* (E. Bonabeau). The package
reproduces the simulations, figures, and tables of Supplementary Experiments S1-S19:
exact invariance, shaped selection, multi-step tracking, finite-sample estimation,
a two-dimensional landscape, and the direct identity tests.

## The object under study

For a search family with relative scores $s_\theta = \nabla_\theta \log \rho_\theta$,
a target signal $w$ (the centered fitness, raw or rank-shaped), and the current
density $\rho_\theta$, the **invariance defect** is the $L^2(\rho_\theta)$ residual
of $w$ after least-squares projection onto the score span:

$$
\delta^2 = \mathbb{E}[w^2] - b^{\top} F^{+} b, \qquad F = \mathbb{E}[s s^{\top}], \qquad b = \mathbb{E}[w s], \qquad R^2 = \frac{b^{\top} F^{+} b}{\mathbb{E}[w^2]}.
$$

It is zero exactly when the family is forward-invariant under the (shaped) fitness.
The package computes this for the Gaussian and Student-t families, the raw and
shaped targets, and four score dictionaries, and compares the exact replicator tilt
with its Fisher-projected Gaussian surrogate.

## Install

```bash
python -m pip install -e .
# or just install the dependencies and run from the repo root:
python -m pip install -r requirements.txt
```

Python 3.9+ with NumPy, SciPy, scikit-learn, pandas, matplotlib, and tabulate.

## Reproduce everything

```bash
python run_all.py            # regenerates every figure (figures/) and table (tables/)
python run_all.py --quick    # smaller sample sizes for the sampling experiments
python run_all.py S07 S11    # run selected experiment groups only
```

Each experiment is also runnable on its own, for example
`python experiments/fig_s05_table_s1_linear_shaping.py`.

## Package layout

```
igr/                      core library
  quadrature.py           grids, normalization, moments, L1/TV/KL
  gaussian.py             Gaussian density, scores, closed-form defect (Eq 47), blow-up (Eq 50)
  replicator.py           exact tilt (Prop 5); param (Fisher moment) and proj (L2(dx)) surrogates
  defect.py               projection machinery, Eq 25 / 57-59 (Gram, pseudoinverse, R^2)
  shaping.py              quantile shaping, shaping functions, shaped defect
  dictionaries.py         Gaussian, Legendre-rank, rank-bin, and RBF score dictionaries
  student_t.py            Student-t scores (Eq 54) and rational conjugate fitness
  fitness.py              fitness functions used across the experiments
  plotting.py             shared style and output paths
experiments/              one script per figure/table group
tests/                    analytic checks against values stated in the supplement
run_all.py                driver
```

## Figure and table map

| Produces | Script | Experiment |
|---|---|---|
| Fig S1, S2 | `fig_s01_s02_raw_exactness.py` | raw-fitness exactness |
| Fig S3 | `fig_s03_defect_departure.py` | defect vs departure from Gaussian |
| Fig S4 | `fig_s04_finite_time.py` | finite-time validity / blow-up |
| Fig S5, Table S1 | `fig_s05_table_s1_linear_shaping.py` | linear fitness, shaping |
| Fig S6, Table S2 | `fig_s06_table_s2_quadratic_shaping.py` | quadratic fitness, shaping |
| Fig S7, Table S3 | `fig_s07_table_s3_multimodal.py` | multimodal elite shaping |
| Fig S8, Table S4 | `fig_s08_table_s4_defect_predicts.py` | defect predicts tracking error |
| Fig S9, S10, Table S5 | `fig_s09_s10_table_s5_multistep.py` | multi-step tracking, calibration |
| Fig S11, S12, Table S6 | `fig_s11_s12_table_s6_finite_sample.py` | finite-sample estimation |
| Fig S13, Table S7 | `fig_s13_table_s7_two_d.py` | two-dimensional landscape |
| Fig S14, S15, S16 | `fig_s14_s15_s16_core.py` | rate decomposition, stalling, convergence |
| Fig S17 | `fig_s17_student_t.py` | Student-t conjugacy homotopy |
| Table S8 | `table_s8_dictionary_cost.py` | dictionary cost in dimension |
| Fig S18, S19 | `fig_s18_s19_envelopes.py` | one-sided Lipschitz, family floor |
| Checks S12-S16 | `checks_s12_s16_identities.py` | coordinate/monotone/frequency identities |

## Validation

`python tests/test_core.py` checks the core against analytic values stated in the
supplement: zero quadratic defect, the Eq (S3) sinusoid defect, the blow-up times
5.618 and 2.50, the symmetric-stalling defect $\sqrt{8a^8/27}=8.709$, $\mathrm{Var}_U(W)$,
and the Eq (S4) elite Gaussian $R^2 = 0.661$. Running the experiments reproduces
Tables S1, S2, S4, S5, S7 to three significant figures and the qualitative content
of the rest.

## Notes

A few settings:

- the perturbation amplitudes for the S1 "small sinusoid" and "pure sinusoid" cases;
- the RBF centers and bandwidth in the multimodal and finite-sample experiments;
- the Gaussian-bump width of the multimodal fitness;
- the smooth-elite logistic bandwidth `h = 0.05`.

The finite-sample experiment defaults to 60 samples per setting (the paper used 150);
override with `IGR_NSAMPLES`. The dictionary-cost experiment estimates $R^2$ by Monte
Carlo and finds the structured-rank augmentation to be dimension-free (a constant
number of rank features beyond the Gaussian base at every dimension), the paper's
central result for that table; override the sample size with `IGR_NMC`.
