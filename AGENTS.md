# AGENTS.md

## What this is

Reference code for *The invariance defect of continuous-trait parametric replicator dynamics*
(E. Bonabeau). It compares the exact fixed-fitness replicator trajectory (the exponential tilt
rho0 exp(t f)) with Gaussian (and Student-t) surrogates, and measures the **invariance defect**:
the part of the raw or rank-shaped fitness that the family's score directions cannot represent
(delta = 0 exactly when the family is forward-invariant). It reproduces Supplementary
Experiments S1-S19. The model is dimensionless; time t multiplies f in exp(t f).

## Install (verified from a fresh clone)

```bash
git clone https://github.com/bonabeau-gif/ig-replicator.git && cd ig-replicator
python3 -m venv ../venv-igr && . ../venv-igr/bin/activate     # Python >= 3.9
python -m pip install -e .        # editable: the `experiment` model needs experiments/ from the checkout
igr describe                      # or, without the console script: python -m igr_cli describe
```

## The interface

One command, `igr`. **Start with `igr describe`.** Add `--json` to any subcommand for exactly one
JSON document on stdout (logs go to stderr). Exit codes: 0 ok, 1 runtime failure, 2 invalid
usage or configuration. Errors with `--json`: `{"ok": false, "errors": [{field, value, reason, hint}]}`.

| Command | Purpose |
|---|---|
| `igr describe [--model M] [--param NAME] [--outputs] [--schema]` | models, parameters (type, default, unit, range, source), outputs, fixed constants |
| `igr template MODEL [--fitness KIND] [--family F] [--experiment ID]` | complete commented YAML config with defaults |
| `igr validate CONFIG` | every problem at once, plus runtime/memory/output estimate (`-` reads stdin) |
| `igr run CONFIG --out DIR` | one run into a new run directory |
| `igr query RUN summary\|outputs\|series\|stats\|at\|final\|table\|crossing` | compact answers from a finished run |
| `igr compare RUN RUN [...]` | differing parameters and every summary difference |
| `igr sweep CONFIG --out DIR` | cartesian sweep from the config's `sweep:` block, plus `index.json` |

Models: `trajectory` (exact tilt vs `param` Fisher moment flow and `proj` L2 surrogate over time),
`projection` (static R^2 / defect of a raw or shaped target on a score dictionary, Gaussian or
Student-t state), `experiment` (runs one published experiment script, ids S01...S19, CHK).
Interventions: none (fitness is fixed during a run).

## Worked example (about 3 s)

```bash
igr template trajectory > my.yaml                  # fitness 0.3x - 0.2x^2 + 0.3 sin(1.5x), N(0,1), t_end = 2
igr validate my.yaml --json                        # ok + estimate
igr run my.yaml --out runs/demo --json             # writes runs/demo/
igr query runs/demo summary --json                 # final L1/KL error, defect, Eq. 49 envelope
igr query runs/demo crossing l1_param 0.05 --json  # when the param surrogate's L1 error first exceeds 0.05
```

`surrogates.param.l1_final` = 0.12906 and `surrogates.proj.l1_final` = 0.08293 (Experiment S1).
Changing one parameter: edit the YAML (or use a `sweep:` block, see `examples/sweep_eps.yaml`),
run into a new directory, then `igr compare runs/demo runs/other --json`.

More examples, each with a sidecar `.md` holding commands and measured values:
`examples/minimal_multistep.yaml` (Table S5 row), `typical_s1_small_sinusoid.yaml` (S1),
`projection_linear_shaping.yaml` (Table S1, Eq. S4 R^2 = 0.661), `projection_multimodal_rbf.yaml`
(Table S3), `experiment_table_s5.yaml` (experiment model), `sweep_eps.yaml` (all of Table S5).

## Configuration

```yaml
schema: 1
model: projection            # trajectory | projection | experiment
seed: 42
parameters:                  # dotted names from `igr describe`, nested: fitness.kind -> fitness: {kind: ...}
  fitness: {kind: sinusoid, sinusoid: {eps: 0.5, k: 1.5}}   # params of the chosen kind go in fitness.<kind>
  state: {family: gaussian, mu: 0.0, var: 1.0}
  targets: [{signal: raw}, {signal: elite, tau: 0.25}]       # raw | linear | elite (tau) | smooth_elite (tau, h)
  dictionaries: [{augment: none}, {augment: legendre_rank, K: 8}]   # none | legendre_rank (K) | rank_bins (K) | rbf (centers, h)
```

Sweeps: add a top-level block `sweep: {fitness.perturbed_quadratic.eps: [0.0, 0.2, 0.4], grid.n: [2401, 4801]}`
(dotted names, list of values each); `igr sweep` runs the cartesian product (last key varies fastest)
into `point_000/`, `point_001/`, ... and writes `index.json` with each point's parameters and summary.

The JSON Schema is `schema/config.schema.json`. `run` writes `config.resolved.yaml` (every
default explicit), `manifest.json` (status, command, times, git commit, versions, seed, peak
memory, error), `summary.json`, `schema.json` (every file and column with units), `log.txt`, and
`outputs/` (CSV; experiment runs also get `outputs/tables/` and `outputs/figures/`).

## Rules for using it well

- Validate before running: `igr validate cfg.yaml --json`. It reports every error and an estimate.
- Always set `seed`. (No model currently draws from it: trajectory and projection are
  deterministic and the experiment scripts use fixed internal seeds; it is recorded anyway.)
- Never parse raw CSVs when a query exists: use `query summary`, `stats`, `series --points N`,
  `at T`, `final`, `table --where col=value --columns a,b`, `crossing NAME THRESHOLD`.
- Check the estimate before large runs. Cost scales with `grid.n` x `time.n_points`.
  `experiment` S14 takes about 15 s; S18 uses about 1 GB of memory at the default `nmc`.
  The first run after installation takes about 10 s longer (one-time import and font caches).
- Use a new `--out` directory per run (existing non-empty directories are refused without `--force`).
- YAML 1.1 reads `1e-9` as a string; write `1.0e-9` (validate warns and converts).
- `fitness.multimodal.width` defaults to the library value 0.5; every published experiment uses 0.6.

## Verify the installation

```bash
python tests/test_core.py     # analytic checks of the library, about 1 s
python tests/test_cli.py      # CLI layer + smoke test of examples/minimal_multistep.yaml, about 10 s
```

## Known limits (from the project's documentation)

- Some settings are representative choices because the supplement does not state them: the S1
  sinusoid amplitudes, the RBF centers and bandwidth, the multimodal bump width, and the smooth-elite
  bandwidth h = 0.05 (README, Notes). Tables S1, S2, S4, S5, S7 reproduce to three significant
  figures; the other experiments reproduce qualitatively.
- The finite-sample experiment (S11) defaults to 60 samples per setting; the paper used 150
  (`experiment.nsamples`). Table S8 is a Monte Carlo estimate (`experiment.nmc`).
- Under convex quadratic fitness (f = c x^2, c > 0) the Gaussian variance blows up at
  t* = 1/(2 c var0) (Eq. 50); `validate` rejects `t_end >= t*` where t* is exact.
- Everything is computed by quadrature on a bounded grid `[grid.a, grid.b]`: mass outside is
  truncated, and the Eq. 49 envelope uses M = (max f - min f)/2 on that grid, so it depends on the
  grid extent. `proj` is a comparison surrogate only; the theorems concern `param`.
- `trajectory` and `projection` are one-dimensional; the two-dimensional landscape exists only as
  experiment S13. Fitness is fixed (frequency-dependent fitness appears only in the CHK checks).

## For contributors

- Run both test files above; the CLI tests also check the old entry points against baseline values.
- New configurable quantities go in `igr_cli/registry.py` first; `describe`, `template`,
  `validate`, and the schema are generated from it. Then regenerate the schema:
  `igr describe --schema --json > schema/config.schema.json` (a test checks they match).
- Model logic stays in `igr/` and `experiments/`; `igr_cli/` only parses, validates, calls, and writes files.
- `python run_all.py` and `python experiments/<script>.py` still work and write to `figures/` and `tables/`.
