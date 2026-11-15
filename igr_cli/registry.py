# -*- coding: utf-8 -*-
"""The parameter registry: the one source of truth for everything configurable.

`describe`, `template`, `validate`, the JSON Schema (schema/config.schema.json),
and the run-directory `schema.json` are all generated from the structures below.
This module is pure data plus small helpers; it imports nothing heavy.

Conventions
-----------
* Parameter names are dotted paths inside the configuration's `parameters:` block.
* `default` is the value used when the key is omitted; REQUIRED marks a parameter
  that must be given (the library function has no default).
* `range` is {"min", "max", "min_exclusive", "max_exclusive"} or None when the
  source does not state one. Ranges are only given where the source states or
  the formula requires them (`source` says which).
* `when` restricts a parameter to configurations where another parameter has one
  of the listed values (for example, the fitness block that matches fitness.kind).
* The model is dimensionless: the source states no physical units. Units below
  are the ones the formulas imply (trait units for x, replicator time 1/fitness
  for t); fitness coefficients have unit None.

New parameters go here first; everything else picks them up automatically.
"""

REQUIRED = "<required>"
SCHEMA_VERSION = 1

TRAIT = "trait units (x)"
TRAIT2 = "trait units squared"
TIME = "replicator time (1/fitness units; t multiplies f in rho0*exp(t f))"
DIMLESS = "dimensionless"


def _rng(lo=None, hi=None, lo_ex=False, hi_ex=False):
    return {"min": lo, "max": hi, "min_exclusive": lo_ex, "max_exclusive": hi_ex}


POSITIVE = _rng(0, None, lo_ex=True)
NONNEG = _rng(0, None)
UNIT_OPEN = _rng(0, 1, lo_ex=True, hi_ex=True)
UNIT_CLOSED = _rng(0, 1)

# --------------------------------------------------------------------------- models
MODELS = {
    "trajectory": {
        "description": (
            "One-dimensional fixed-fitness replicator dynamics from a Gaussian start "
            "N(mu0, var0): the exact trajectory (exponential tilt rho0*exp(t f), "
            "Proposition 5) compared with Gaussian surrogates over [0, t_end]. "
            "'param' is the Fisher-projected moment flow (Eq. S1/42, the object of the "
            "theorems); 'proj' is the ordinary L2(dx) density-velocity projection "
            "(Appendix C, comparison only). Records moments, L1/KL tracking error, and "
            "the raw invariance defect along each surrogate path (Figs S1-S3, S9, S10, S14)."),
        "library": "igr.exact_tilt, igr.integrate_param, igr.integrate_proj, igr.raw_defect_gaussian",
        "randomness": "none (deterministic quadrature and ODE integration)",
    },
    "projection": {
        "description": (
            "Static invariance-defect diagnostic at one search state: the raw or "
            "rank-shaped centered fitness w is projected in L2(rho) onto a score "
            "dictionary (family scores, optionally augmented with Legendre-rank, "
            "rank-bin, or RBF features). Reports R^2, the defect delta, and the Gram "
            "condition number for every (dictionary, target) pair "
            "(Tables S1-S3, Eq. S4, Fig S17, Experiment S10 stalling defect)."),
        "library": "igr.project, igr.shaped_projection, igr.gaussian_scores, igr.student_t_scores, igr.legendre_rank, igr.rank_bins, igr.rbf",
        "randomness": "none (deterministic quadrature)",
    },
    "experiment": {
        "description": (
            "Run one of the repository's supplementary experiment scripts unchanged "
            "(experiments/*.py, the same code run_all.py calls) and collect its tables "
            "(CSV, Markdown) and figures (PNG) inside the run directory. Use it to "
            "reproduce a published table or figure. Requires the repository checkout "
            "(editable install or running from the repo root)."),
        "library": "experiments/<script>.run()",
        "randomness": ("fixed seeds inside the scripts: S11 uses numpy default_rng(1000 + N) "
                       "per sample size, S18 (Table S8) uses default_rng(0). The config "
                       "seed is recorded but does not change these."),
    },
}

# Experiment groups, mirroring run_all.EXPERIMENTS (id, module, produces).
EXPERIMENTS = [
    ("S01", "fig_s01_s02_raw_exactness", "Fig S1, S2", "raw-fitness exactness at t=2: exact tilt vs param/proj surrogates"),
    ("S03", "fig_s03_defect_departure", "Fig S3", "defect along the trajectory and vs perturbation amplitude"),
    ("S04", "fig_s04_finite_time", "Fig S4", "finite-time blow-up of the variance under f = c x^2, c > 0"),
    ("S05", "fig_s05_table_s1_linear_shaping", "Fig S5, Table S1", "linear fitness under rank shaping, R^2 per dictionary"),
    ("S06", "fig_s06_table_s2_quadratic_shaping", "Fig S6, Table S2", "quadratic fitness before and after shaping"),
    ("S07", "fig_s07_table_s3_multimodal", "Fig S7, Table S3", "multimodal fitness, elite shaping, dictionaries and a 2-component mixture"),
    ("S08", "fig_s08_table_s4_defect_predicts", "Fig S8, Table S4", "shaped defect vs one-step tracking error (Pearson/Spearman)"),
    ("S09", "fig_s09_s10_table_s5_multistep", "Fig S9, S10, Table S5", "multi-step tracking error, accumulated defect, Eq. 49 envelope"),
    ("S11", "fig_s11_s12_table_s6_finite_sample", "Fig S11, S12, Table S6", "finite-sample estimation of shaped directions with ridge"),
    ("S13", "fig_s13_table_s7_two_d", "Fig S13, Table S7", "two-dimensional bivariate Gaussian projection"),
    ("S14", "fig_s14_s15_s16_core", "Fig S14, S15, S16", "rate decomposition, symmetric stalling, convergence orders"),
    ("S17", "fig_s17_student_t", "Fig S17", "Student-t conjugacy homotopy, Gaussian R^2 of rational fitness vs nu"),
    ("S18", "table_s8_dictionary_cost", "Table S8", "dictionary cost in dimension (Monte Carlo)"),
    ("S19", "fig_s18_s19_envelopes", "Fig S18, S19", "one-sided Lipschitz envelope; family floor vs dynamic drift"),
    ("CHK", "checks_s12_s16_identities", "Checks S12-S16", "coordinate, monotone, and frequency-dependent identity checks (printed only)"),
]
EXPERIMENT_IDS = [e[0] for e in EXPERIMENTS]

# --------------------------------------------------------------------------- fitness kinds
# Each kind: description, formula, source, and its own parameters (registered as
# fitness.<kind>.<name>). All fitness functions are of one trait variable x.
FITNESS_KINDS = {
    "quadratic": {
        "formula": "f(x) = 0.30 x - 0.20 x^2",
        "description": "Affine-quadratic base fitness of Experiment S1; inside the Gaussian invariance class (defect zero).",
        "source": "igr.fitness.quadratic",
        "params": [],
    },
    "perturbed_quadratic": {
        "formula": "f(x) = a x - b x^2 + eps sin(k x)",
        "description": "The S3 multi-step family: quadratic plus a sinusoidal perturbation that moves it out of the Gaussian class as eps grows.",
        "source": "igr.fitness.perturbed_quadratic",
        "params": [
            ("a", "number", 0.30, None, "Linear coefficient. Moves the mean.", "igr.fitness.perturbed_quadratic default a=0.30"),
            ("b", "number", 0.20, None, "Quadratic coefficient (f contains -b x^2). b > 0 is concave; b < 0 is convex and, with eps = 0, the Gaussian variance blows up at t* = 1/(2|b| var0) (Eq. 50).", "igr.fitness.perturbed_quadratic default b=0.20; Eq. 50"),
            ("eps", "number", REQUIRED, None, "Perturbation amplitude. eps = 0 is exactly Gaussian-invariant; the defect grows with |eps| (Fig S3b). Experiments use 0 to 0.8.", "igr.fitness.perturbed_quadratic (no default); values used in S1, S3, S9"),
            ("k", "number", 1.5, None, "Perturbation wavenumber.", "igr.fitness.perturbed_quadratic default k=1.5"),
        ],
    },
    "polynomial": {
        "formula": "f(x) = sum_i coefficients[i] x^i",
        "description": "General polynomial fitness (coefficients in increasing degree). Degree <= 2 is in the Gaussian invariance class; e.g. [0, 1] is the linear fitness of S2/Table S1, [0, 0, -1] the quadratic of Table S2, [0, 0, c] the blow-up case of Fig S4.",
        "source": "added by the CLI layer to express the polynomial fitnesses that experiments S2, S3, S4 define inline",
        "params": [
            ("coefficients", "list[number]", REQUIRED, None, "Polynomial coefficients c0, c1, c2, ... (increasing degree). At least one entry.", "experiments S2 (f=x), S3 (f=-x^2), S4 (f=c x^2)"),
        ],
    },
    "sinusoid": {
        "formula": "f(x) = eps sin(k x)",
        "description": "Pure sinusoidal fitness; strongly non-Gaussian. eps = 1.2 turns the t = 2 density bimodal (S1). Eq. S3 gives its Gaussian defect in closed form.",
        "source": "igr.fitness.sinusoid; README notes the S1 amplitude is a representative choice",
        "params": [
            ("eps", "number", 1.2, None, "Amplitude.", "igr.fitness.sinusoid default eps=1.2"),
            ("k", "number", 1.5, None, "Wavenumber.", "igr.fitness.sinusoid default k=1.5"),
        ],
    },
    "multimodal": {
        "formula": "f(x) = sum_c exp(-(x - c)^2 / (2 width^2))",
        "description": "Two-mode fitness (Gaussian bumps), used for multimodal elite shaping (S4, S7, S11, S17). Note: the library default width is 0.5 but every shipped experiment uses width = 0.6.",
        "source": "igr.fitness.multimodal; README notes the bump width is a representative choice",
        "params": [
            ("width", "number", 0.5, None, "Bump width (standard deviation of each bump). Experiments use 0.6.", "igr.fitness.multimodal default width=0.5"),
            ("centers", "list[number]", [-2.0, 2.0], None, "Bump centers in trait units.", "igr.fitness.multimodal default centers=(-2.0, 2.0)"),
        ],
    },
    "stalling": {
        "formula": "f(x) = -(x^2 - a^2)^2",
        "description": "Symmetric double well (Corollary 3.3 / Experiment S10). The Gaussian surrogate stalls at N(0, a^2/3) with defect sqrt(8 a^8 / 27) (8.709 for a = 2).",
        "source": "igr.fitness.stalling",
        "params": [
            ("a", "number", 2.0, None, "Well location +/- a. The Gaussian surrogate stalls at N(0, a^2/3) (set state.var or initial.var yourself).", "igr.fitness.stalling default a=2.0"),
        ],
    },
    "rational_conjugate": {
        "formula": "f(x) = (alpha z + beta z^2)/(nu + z^2) + gamma, z = (x - mu)/sigma",
        "description": "Student-t conjugate (rational) fitness: the Student-t family with the same nu is invariant under it (Section 4.3).",
        "source": "igr.student_t.rational_conjugate_fitness",
        "params": [
            ("alpha", "number", REQUIRED, None, "Coefficient of z.", "igr.student_t.rational_conjugate_fitness (no default)"),
            ("beta", "number", REQUIRED, None, "Coefficient of z^2.", "igr.student_t.rational_conjugate_fitness (no default)"),
            ("gamma", "number", REQUIRED, None, "Additive constant (does not change any defect or trajectory).", "igr.student_t.rational_conjugate_fitness (no default)"),
            ("mu", "number", REQUIRED, None, "Center of z, trait units.", "igr.student_t.rational_conjugate_fitness (no default)"),
            ("sigma", "number", REQUIRED, POSITIVE, "Scale of z, trait units.", "scale parameter; the Student-t family requires sigma > 0 (log sigma in igr.student_t)"),
            ("nu", "number", REQUIRED, POSITIVE, "Degrees of freedom of the conjugate Student-t family.", "Student-t degrees of freedom, nu > 0"),
        ],
    },
    "quadratic_rational_homotopy": {
        "formula": "f_s(x) = (1 - s) x^2 + s x^2/(nu + x^2)",
        "description": "The S17 homotopy from the Gaussian-conjugate quadratic (s = 0) to the Student-t-conjugate rational fitness (s = 1), at mu = 0, sigma = 1.",
        "source": "experiments/fig_s17_student_t.py",
        "params": [
            ("s", "number", REQUIRED, UNIT_CLOSED, "Homotopy parameter, quadratic (0) to rational (1).", "S17 sweeps s over linspace(0, 1, 21)"),
            ("nu", "number", REQUIRED, POSITIVE, "nu of the rational endpoint.", "S17 uses nu = 3, 8 and 2..50"),
        ],
    },
}
FITNESS_MODELS = ["trajectory", "projection"]

TARGET_SIGNALS = {
    "raw": "Centered raw fitness w = f - E[f] (the unshaped defect).",
    "linear": "Linear rank shaping W(u) = u of the fitness quantile q.",
    "elite": "Elite top-tau selection W(u) = 1{u >= 1 - tau}.",
    "smooth_elite": "Smooth elite W(u) = logistic((u - (1 - tau))/h).",
}
DICT_AUGMENTS = {
    "none": "Family scores only ({Z, Z^2-1} for Gaussian; Eq. 54 for Student-t).",
    "legendre_rank": "Family scores + shifted Legendre polynomials P_1..P_K of the fitness quantile.",
    "rank_bins": "Family scores + K indicator bins of the fitness quantile.",
    "rbf": "Family scores + radial basis functions exp(-(x-c)^2/(2h^2)) at the given centers.",
}

# --------------------------------------------------------------------------- parameters


def _p(name, type_, default, description, applies_to, unit=None, range_=None,
       options=None, when=None, constraints=None, source=None, items=None):
    entry = {"name": name, "type": type_, "default": default, "range": range_,
             "unit": unit, "description": description, "applies_to": list(applies_to),
             "constraints": constraints or [], "source": source}
    if options is not None:
        entry["options"] = list(options)
    if when is not None:
        entry["when"] = when
    if items is not None:
        entry["items"] = items
    return entry


def _build_parameters():
    P = []
    TP = ["trajectory", "projection"]
    P.append(_p("fitness.kind", "enum", REQUIRED,
                "Which fitness function f(x) to use. Its parameters go in the block "
                "fitness.<kind> (only that block is allowed). Kinds: "
                + "; ".join(f"{k}: {v['formula']}" for k, v in FITNESS_KINDS.items()) + ".",
                TP, options=list(FITNESS_KINDS), source="igr/fitness.py, igr/student_t.py, experiments"))
    for kind, spec in FITNESS_KINDS.items():
        for (pname, ptype, pdef, prange, pdesc, psrc) in spec["params"]:
            P.append(_p(f"fitness.{kind}.{pname}", ptype, pdef, f"[{kind}: {spec['formula']}] {pdesc}", TP,
                        unit=TRAIT if pname in ("centers", "mu", "sigma") and kind != "quadratic_rational_homotopy" else None,
                        range_=prange, when={"fitness.kind": [kind]}, source=psrc,
                        constraints=(["len(fitness.polynomial.coefficients) >= 1"] if pname == "coefficients" else
                                     ["fitness.multimodal.width != 0"] if pname == "width" else
                                     ["len(fitness.multimodal.centers) >= 1"] if pname == "centers" else [])))
    # quadrature grid
    P.append(_p("grid.a", "number", -10.0, "Left end of the uniform quadrature grid. All densities live on [a, b]; mass outside is truncated.",
                TP, unit=TRAIT, constraints=["grid.b > grid.a"], source="igr.Grid; Grid(-10, 10, 24001) in S1, S3, S7, S8, S14, S19"))
    P.append(_p("grid.b", "number", 10.0, "Right end of the quadrature grid.",
                TP, unit=TRAIT, constraints=["grid.b > grid.a"], source="igr.Grid; Grid(-10, 10, 24001) in most experiments"))
    P.append(_p("grid.n", "integer", 24001, "Number of grid points (trapezoidal quadrature). Cost scales linearly with n. Experiments use 2401 to 60001.",
                TP, unit=None, range_=_rng(2, None), source="igr.Grid needs n >= 2 (dx = x[1] - x[0])"))
    # trajectory
    TR = ["trajectory"]
    P.append(_p("initial.mu", "number", 0.0, "Mean of the initial Gaussian density rho0 = N(mu0, var0).", TR, unit=TRAIT,
                source="MU0 = 0.0 in experiments S1, S3, S9, S14, S19"))
    P.append(_p("initial.var", "number", 1.0, "Variance of the initial Gaussian density.", TR, unit=TRAIT2, range_=POSITIVE,
                constraints=["time.t_end < 1/(2 c var0) for convex quadratic fitness (Eq. 50)"],
                source="VAR0 = 1.0 in experiments; gaussian_pdf takes sqrt(var)"))
    P.append(_p("time.t_end", "number", 2.0, "Final replicator time. The exact tilt is evaluated and the surrogate ODEs integrated on [0, t_end].",
                TR, unit=TIME, range_=POSITIVE,
                constraints=["time.t_end < blow-up time t* = 1/(2 c var0) when the fitness is quadratic with positive curvature c (polynomial degree 2 with c2 > 0, or perturbed_quadratic with b < 0 and eps = 0); Eq. 50"],
                source="T = 2.0 in S1, S3, S14; T = 1.0 in S9, S19"))
    P.append(_p("time.n_points", "integer", 101, "Number of equally spaced output times in [0, t_end], including both ends.",
                TR, range_=_rng(2, None), source="teval = linspace(0, T, 101) in S1; 501 in S9"))
    P.append(_p("surrogates", "list[enum]", ["param", "proj"],
                "Gaussian surrogates to integrate and compare with the exact tilt. 'param' = Fisher moment flow (Eq. S1); 'proj' = L2(dx) density-velocity projection (comparison only).",
                TR, options=["param", "proj"], constraints=["len(surrogates) >= 1", "no duplicates"],
                source="igr.integrate_param, igr.integrate_proj; S1 uses both"))
    P.append(_p("solver.rtol", "number", 1e-9, "Relative tolerance of scipy solve_ivp for the surrogate ODEs.", TR, range_=POSITIVE,
                source="igr.integrate_param default rtol=1e-9; S9 uses 1e-10"))
    P.append(_p("solver.atol", "number", 1e-12, "Absolute tolerance of scipy solve_ivp.", TR, range_=POSITIVE,
                source="igr.integrate_param default atol=1e-12; S9 uses 1e-13"))
    P.append(_p("solver.method", "enum", "DOP853", "scipy solve_ivp integration method.", TR,
                options=["DOP853", "RK45", "RK23", "Radau", "BDF", "LSODA"],
                source="igr.integrate_param default method='DOP853'; options are scipy.integrate.solve_ivp methods"))
    # projection
    PR = ["projection"]
    P.append(_p("state.family", "enum", "gaussian", "Search family of the current state rho_theta and of the base score dictionary.",
                PR, options=["gaussian", "student_t"], source="igr.gaussian, igr.student_t"))
    P.append(_p("state.mu", "number", 0.0, "Location of the state density.", PR, unit=TRAIT,
                source="MU = 0.0 in S2, S3, S5, S17"))
    P.append(_p("state.var", "number", 1.0, "Variance of the Gaussian state N(mu, var). Experiments use 1.0 and 2.5^2 = 6.25 (broad baseline of S4/S7).",
                PR, unit=TRAIT2, range_=POSITIVE, when={"state.family": ["gaussian"]},
                source="VAR = 1.0 in S2, S3; BROAD_VAR = 6.25 in S7"))
    P.append(_p("state.sigma", "number", 1.0, "Scale of the Student-t state.", PR, unit=TRAIT, range_=POSITIVE,
                when={"state.family": ["student_t"]}, source="sigma = 1.0 in S17"))
    P.append(_p("state.nu", "number", REQUIRED, "Degrees of freedom of the Student-t state.", PR, range_=POSITIVE,
                when={"state.family": ["student_t"]}, source="nu = 3, 8 in S17"))
    P.append(_p("targets", "list[mapping]",
                [{"signal": "raw"}, {"signal": "linear"}, {"signal": "elite", "tau": 0.25},
                 {"signal": "smooth_elite", "tau": 0.25, "h": 0.05}],
                "Target signals to project. Shaped targets use the fitness quantile q under the state density (Section 5.1); 'raw' is the unshaped centered fitness.",
                PR, constraints=["len(targets) >= 1"],
                source="Table S2 columns (raw, linear, elite 25%, smooth 25%); igr.shaping",
                items={
                    "signal": {"type": "enum", "default": REQUIRED, "options": list(TARGET_SIGNALS), "range": None, "unit": None,
                               "description": "; ".join(f"{k}: {v}" for k, v in TARGET_SIGNALS.items())},
                    "tau": {"type": "number", "default": 0.25, "range": UNIT_OPEN, "unit": DIMLESS, "when_signal": ["elite", "smooth_elite"],
                            "description": "Elite fraction (top tau of the quantile range). Larger tau selects more of the population."},
                    "h": {"type": "number", "default": 0.05, "range": POSITIVE, "unit": DIMLESS, "when_signal": ["smooth_elite"],
                          "description": "Logistic bandwidth of the smooth elite (README: h = 0.05 is a representative choice)."},
                }))
    P.append(_p("dictionaries", "list[mapping]",
                [{"augment": "none"}, {"augment": "legendre_rank", "K": 4},
                 {"augment": "legendre_rank", "K": 8}, {"augment": "rank_bins", "K": 8}],
                "Score dictionaries. Each is the family's own scores plus an optional augmentation; R^2 rises with richer dictionaries.",
                PR, constraints=["len(dictionaries) >= 1"],
                source="Table S2 rows; igr.dictionaries",
                items={
                    "augment": {"type": "enum", "default": REQUIRED, "options": list(DICT_AUGMENTS), "range": None, "unit": None,
                                "description": "; ".join(f"{k}: {v}" for k, v in DICT_AUGMENTS.items())},
                    "K": {"type": "integer", "default": REQUIRED, "range": _rng(1, None), "unit": None, "when_augment": ["legendre_rank", "rank_bins"],
                          "description": "Number of rank features (Legendre degree 1..K, or K bins)."},
                    "centers": {"type": "list[number]", "default": REQUIRED, "range": None, "unit": TRAIT, "when_augment": ["rbf"],
                                "description": "RBF centers (S7 uses linspace(-4, 4, 9) and linspace(-5, 5, 17))."},
                    "h": {"type": "number", "default": REQUIRED, "range": POSITIVE, "unit": TRAIT, "when_augment": ["rbf"],
                          "description": "RBF bandwidth (S7 uses 0.8 x center spacing)."},
                }))
    P.append(_p("ridge", "number", 0.0, "Tikhonov ridge added as ridge*mean(diag F)*I to the Gram matrix before the pseudoinverse.",
                PR, unit=DIMLESS, range_=NONNEG, source="igr.project ridge=0.0"))
    # experiment
    EX = ["experiment"]
    P.append(_p("experiment.id", "enum", REQUIRED, "Which supplementary experiment group to run (same ids as run_all.py).",
                EX, options=EXPERIMENT_IDS, source="run_all.EXPERIMENTS"))
    P.append(_p("experiment.nsamples", "integer", 60, "Independent samples per (dictionary, N, lambda) setting in S11. The paper used 150; run_all --quick uses 20. Replaces the IGR_NSAMPLES environment variable.",
                EX, range_=_rng(1, None), when={"experiment.id": ["S11"]},
                source="experiments/fig_s11_s12_table_s6_finite_sample.py NSAMPLES default 60; README"))
    P.append(_p("experiment.nmc", "integer", 40000, "Monte Carlo sample size for Table S8 (id S18). run_all --quick uses 15000. Replaces the IGR_NMC environment variable.",
                EX, range_=_rng(1, None), when={"experiment.id": ["S18"]},
                source="experiments/table_s8_dictionary_cost.py NMC default 40000; README"))
    return P


PARAMETERS = _build_parameters()
PARAM_INDEX = {p["name"]: p for p in PARAMETERS}

TOP_LEVEL = {
    "schema": {"type": "integer", "default": SCHEMA_VERSION, "description": "Configuration format version. Must be 1."},
    "model": {"type": "enum", "default": REQUIRED, "options": list(MODELS), "description": "Which model to run."},
    "seed": {"type": "integer", "default": "chosen at run time and recorded", "range": _rng(0, None),
             "description": ("Random seed, recorded in the run directory. No model in this package draws random "
                             "numbers from it: trajectory and projection are deterministic and the experiment "
                             "scripts use their own fixed seeds. Always set it anyway, so runs stay reproducible "
                             "if a stochastic model is added.")},
    "parameters": {"type": "mapping", "default": {}, "description": "Model parameters (see describe --model <m>)."},
    "sweep": {"type": "mapping", "default": None,
              "description": "Optional, used only by `sweep`: {dotted.parameter: [values, ...]}; the cartesian product is run, one run directory per point."},
}

# --------------------------------------------------------------------------- outputs
def _o(name, unit, description, shape, file):
    return {"name": name, "unit": unit, "description": description, "shape": shape, "file": file}


SURR_COLS = [
    ("mu_{s}", TRAIT, "Mean of surrogate {s}."),
    ("var_{s}", TRAIT2, "Variance of surrogate {s}."),
    ("l1_{s}", DIMLESS, "L1 density distance between the exact tilt and surrogate {s} (TV = l1/2)."),
    ("kl_{s}", DIMLESS, "KL(exact || surrogate {s}) on the grid (nats)."),
    ("defect_{s}", "fitness units", "Raw invariance defect delta (Eq. 47) at the surrogate {s} state."),
    ("fitness_var_{s}", "fitness units squared", "Full mean-fitness rate Var_g[f] under surrogate {s} (Fig S14)."),
    ("explained_rate_{s}", "fitness units squared", "Projected (explained) rate Var_g[f] - delta^2 under surrogate {s} (Fig S14); the gap is delta^2."),
]

OUTPUTS = {
    "trajectory": [
        _o("t", TIME, "Output time.", "time series", "outputs/trajectory.csv"),
        _o("mu_true", TRAIT, "Mean of the exact tilt rho0*exp(t f) (normalized).", "time series", "outputs/trajectory.csv"),
        _o("var_true", TRAIT2, "Variance of the exact tilt.", "time series", "outputs/trajectory.csv"),
    ] + [_o(c.format(s=s), u, d.format(s=s), "time series", "outputs/trajectory.csv")
         for s in ("param", "proj") for (c, u, d) in SURR_COLS] + [
        _o("x", TRAIT, "Grid point.", "per-grid-point table", "outputs/densities_final.csv"),
        _o("rho_true", "1/trait units", "Exact tilt density at t_end.", "per-grid-point table", "outputs/densities_final.csv"),
        _o("rho_param", "1/trait units", "Param surrogate density at t_end (if requested).", "per-grid-point table", "outputs/densities_final.csv"),
        _o("rho_proj", "1/trait units", "Proj surrogate density at t_end (if requested).", "per-grid-point table", "outputs/densities_final.csv"),
    ],
    "projection": [
        _o("dictionary", None, "Dictionary label, e.g. 'gaussian + legendre_rank K=8'.", "per-pair table", "outputs/projection.csv"),
        _o("target", None, "Target label, e.g. 'elite tau=0.25'.", "per-pair table", "outputs/projection.csv"),
        _o("directions", None, "Number of score directions in the dictionary.", "per-pair table", "outputs/projection.csv"),
        _o("r2", DIMLESS, "Explained fraction R^2 = b^T F^+ b / E[w^2] (R^2_shape for shaped targets).", "per-pair table", "outputs/projection.csv"),
        _o("defect", "fitness units (raw) or dimensionless (shaped)", "Defect delta = sqrt(E[w^2] - b^T F^+ b).", "per-pair table", "outputs/projection.csv"),
        _o("defect2", "square of defect unit", "Squared defect delta^2.", "per-pair table", "outputs/projection.csv"),
        _o("var_w", "square of defect unit", "E[w^2]; equals Var_U(W) for shaped targets up to discretization.", "per-pair table", "outputs/projection.csv"),
        _o("cond", DIMLESS, "Condition number of the Gram matrix F.", "per-pair table", "outputs/projection.csv"),
    ],
    "experiment": [
        _o("tables", None, "Each table the script saves, as outputs/tables/<name>.csv and .md (columns described in schema.json).", "per-row table", "outputs/tables/"),
        _o("figures", None, "Each figure the script saves, as outputs/figures/<name>.png.", "image", "outputs/figures/"),
        _o("reported_lines", None, "Numeric lines the script prints (copied into summary.json; full text in log.txt).", "text", "summary.json"),
    ],
}

INTERVENTIONS = []
INTERVENTIONS_NOTE = ("No interventions are supported: every model uses a fixed fitness for the whole run "
                      "(the exact tilt of Proposition 5 is the fixed-fitness solution). Change the fitness "
                      "between runs instead and use `compare` or `sweep`.")

# Constants in the source that users are not expected to edit; listed by describe.
FIXED = [
    {"name": "variance floor", "value": 1e-12, "where": "igr.replicator integrate_param/integrate_proj", "description": "Variance is clipped at 1e-12 inside the surrogate RHS."},
    {"name": "pinv rcond", "value": 1e-12, "where": "igr.defect.project", "description": "Relative cutoff of the Moore-Penrose pseudoinverse."},
    {"name": "quantile clip", "value": 1e-12, "where": "igr.shaping.weighted_quantile", "description": "Quantiles are clipped to [1e-12, 1 - 1e-12]; midpoint CDF convention."},
    {"name": "proj regularizer", "value": 1e-18, "where": "igr.replicator.proj_rhs", "description": "1e-18*I added to the 2x2 normal matrix."},
    {"name": "KL floor", "value": 1e-300, "where": "igr.quadrature.kl_divergence", "description": "Densities floored at 1e-300 inside the log."},
    {"name": "Gronwall constant", "value": "L = 3M, M = (max f - min f)/2 on the grid", "where": "experiments/fig_s09 (used by trajectory summary)", "description": "Worst-case constant of the Eq. 49 envelope."},
    {"name": "experiment grids and settings", "value": "hard-coded per script", "where": "experiments/*.py", "description": "Each experiment script fixes its own grid, fitness, and sweep values; the experiment model runs them as published."},
]


def param_applies(entry, model):
    return model in entry["applies_to"]


def params_for(model):
    return [p for p in PARAMETERS if param_applies(p, model)]


def output_units(model):
    return {o["name"]: o["unit"] for o in OUTPUTS.get(model, [])}
