# -*- coding: utf-8 -*-
"""Model runners: thin calls into the igr library and the experiment scripts.

Each runner takes the flattened resolved parameters and an output directory,
writes its files under <out>/outputs/, and returns the summary mapping. Heavy
libraries are imported inside the runners only.
"""
import csv
import io
import os
import sys

from . import registry as R


def q(value, unit):
    """A summary quantity with its unit."""
    if value is not None:
        value = float(value)
    return {"value": value, "unit": unit}


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def experiments_dir():
    d = os.path.join(repo_root(), "experiments")
    return d if os.path.isfile(os.path.join(d, "_common.py")) else None


def _fmt(v):
    if isinstance(v, float):
        return repr(v)
    return v


def write_csv(path, header, columns):
    """Write equal-length columns as CSV with a header row (floats as shortest repr)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = zip(*columns)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        for row in rows:
            w.writerow([_fmt(v) for v in row])


# --------------------------------------------------------------------------- fitness
def fitness_label(flat):
    kind = flat["fitness.kind"]
    spec = R.FITNESS_KINDS[kind]
    args = {p[0]: flat.get(f"fitness.{kind}.{p[0]}") for p in spec["params"]}
    return {"kind": kind, "formula": spec["formula"], "parameters": args}


def build_fitness(flat):
    """Return f(x) for the configured fitness, using the library constructors."""
    import numpy as np
    from igr import fitness as F
    from igr.student_t import rational_conjugate_fitness
    kind = flat["fitness.kind"]
    g = lambda n: flat[f"fitness.{kind}.{n}"]  # noqa: E731
    if kind == "quadratic":
        return F.quadratic
    if kind == "perturbed_quadratic":
        return F.perturbed_quadratic(g("eps"), a=g("a"), b=g("b"), k=g("k"))
    if kind == "polynomial":
        coef = [float(c) for c in g("coefficients")]

        def f(x):
            x = np.asarray(x, float)
            out = np.zeros_like(x)
            for i, c in enumerate(coef):
                if c != 0.0:
                    out = out + c * x ** i
            return out
        return f
    if kind == "sinusoid":
        return F.sinusoid(g("eps"), g("k"))
    if kind == "multimodal":
        return F.multimodal(width=g("width"), centers=tuple(g("centers")))
    if kind == "stalling":
        return F.stalling(g("a"))
    if kind == "rational_conjugate":
        return rational_conjugate_fitness(g("alpha"), g("beta"), g("gamma"), g("mu"), g("sigma"), g("nu"))
    if kind == "quadratic_rational_homotopy":
        s, nu = g("s"), g("nu")

        def f(x):
            z = np.asarray(x, float)
            return (1 - s) * z ** 2 + s * (z ** 2 / (nu + z ** 2))
        return f
    raise ValueError(f"unknown fitness kind {kind!r}")


# --------------------------------------------------------------------------- trajectory
def run_trajectory(flat, out_dir, log):
    import numpy as np
    import igr
    g = igr.Grid(flat["grid.a"], flat["grid.b"], flat["grid.n"])
    f = build_fitness(flat)
    fx = f(g.x)
    mu0, var0 = flat["initial.mu"], flat["initial.var"]
    T, npts = flat["time.t_end"], flat["time.n_points"]
    rtol, atol, method = flat["solver.rtol"], flat["solver.atol"], flat["solver.method"]
    rho0 = igr.normalize(igr.gaussian_pdf(g.x, mu0, var0), g)
    teval = np.linspace(0, T, npts)

    paths = {}
    for s in flat["surrogates"]:
        integ = igr.integrate_param if s == "param" else igr.integrate_proj
        log(f"integrating {s} surrogate on [0, {T}] ({method}, rtol={rtol}, atol={atol})")
        ts, mu, var = integ(mu0, var0, f, g, (0, T), teval, rtol=rtol, atol=atol, method=method)
        if len(ts) != npts or not np.all(np.isfinite(var)) or np.any(var <= 0):
            raise RuntimeError(f"{s} surrogate integration stopped at t = {ts[-1] if len(ts) else 0:.6g} "
                               f"before t_end = {T} (solver failure or variance blow-up, Eq. 50)")
        paths[s] = (mu, var)

    cols = {"t": teval, "mu_true": np.empty(npts), "var_true": np.empty(npts)}
    for s in paths:
        for c in ("mu", "var", "l1", "kl", "defect", "fitness_var", "explained_rate"):
            cols[f"{c}_{s}"] = np.empty(npts)
    final_true = None
    final_surr = {}
    for i, t in enumerate(teval):
        true_d = igr.exact_tilt(rho0, fx, t, g)
        cols["mu_true"][i], cols["var_true"][i] = igr.moments(true_d, g)
        for s, (mu, var) in paths.items():
            surr = igr.normalize(igr.gaussian_pdf(g.x, mu[i], var[i]), g)
            d2 = igr.raw_defect_gaussian(fx, mu[i], var[i], surr, g)
            Ef = igr.expectation(fx, surr, g)
            fv = igr.expectation((fx - Ef) ** 2, surr, g)
            cols[f"mu_{s}"][i], cols[f"var_{s}"][i] = mu[i], var[i]
            cols[f"l1_{s}"][i] = igr.l1_distance(true_d, surr, g)
            cols[f"kl_{s}"][i] = igr.kl_divergence(true_d, surr, g)
            cols[f"defect_{s}"][i] = np.sqrt(d2)
            cols[f"fitness_var_{s}"][i] = fv
            cols[f"explained_rate_{s}"][i] = fv - d2
            if i == npts - 1:
                final_surr[s] = surr
        if i == npts - 1:
            final_true = true_d

    header = list(cols)
    write_csv(os.path.join(out_dir, "outputs", "trajectory.csv"), header, [cols[h].tolist() for h in header])
    dh = ["x", "rho_true"] + [f"rho_{s}" for s in final_surr]
    write_csv(os.path.join(out_dir, "outputs", "densities_final.csv"), dh,
              [g.x.tolist(), final_true.tolist()] + [final_surr[s].tolist() for s in final_surr])

    units = R.output_units("trajectory")
    summary = {"model": "trajectory", "fitness": fitness_label(flat),
               "initial": {"mu": q(mu0, R.TRAIT), "var": q(var0, R.TRAIT2)},
               "t_end": q(T, R.TIME),
               "true": {"mu_final": q(cols["mu_true"][-1], R.TRAIT), "var_final": q(cols["var_true"][-1], R.TRAIT2)},
               "surrogates": {}}
    for s in paths:
        d = cols[f"defect_{s}"]
        imax = int(np.argmax(d))
        summary["surrogates"][s] = {
            "mu_final": q(cols[f"mu_{s}"][-1], R.TRAIT),
            "var_final": q(cols[f"var_{s}"][-1], R.TRAIT2),
            "l1_final": q(cols[f"l1_{s}"][-1], units["l1_param"]),
            "tv_final": q(0.5 * cols[f"l1_{s}"][-1], R.DIMLESS),
            "kl_final": q(cols[f"kl_{s}"][-1], units["kl_param"]),
            "l1_max": q(cols[f"l1_{s}"].max(), R.DIMLESS),
            "defect_final": q(d[-1], units["defect_param"]),
            "defect_max": q(d[imax], units["defect_param"]),
            "defect_max_time": q(teval[imax], R.TIME),
            "defect_integral": q(np.trapezoid(d, teval), "fitness units x time"),
        }
    if "param" in paths:
        M = 0.5 * (fx.max() - fx.min())
        L = 3.0 * M
        d = cols["defect_param"]
        env = (d.max() / L) * (np.exp(L * T) - 1.0) if L > 0 else 0.0
        summary["gronwall_envelope_eq49"] = {
            "value": float(env), "unit": R.DIMLESS,
            "note": "worst-case Eq. 49 envelope (max defect / L)(exp(L t_end) - 1) with L = 3M, "
                    "M = (max f - min f)/2 on the grid (as in Table S5); depends on the grid extent"}
    from .config import blowup_time
    tb = blowup_time(flat)
    summary["blowup_time_eq50"] = q(tb, R.TIME) if tb is not None else None
    return summary


# --------------------------------------------------------------------------- projection
def _target_label(t):
    s = t["signal"]
    if s == "elite":
        return f"elite tau={t['tau']}"
    if s == "smooth_elite":
        return f"smooth_elite tau={t['tau']} h={t['h']}"
    return s


def _dict_label(family, d):
    a = d["augment"]
    if a == "none":
        return family
    if a == "rbf":
        return f"{family} + rbf K={len(d['centers'])} h={d['h']}"
    return f"{family} + {a} K={d['K']}"


def run_projection(flat, out_dir, log):
    import numpy as np
    import igr
    g = igr.Grid(flat["grid.a"], flat["grid.b"], flat["grid.n"])
    fx = build_fitness(flat)(g.x)
    fam = flat["state.family"]
    mu = flat["state.mu"]
    if fam == "gaussian":
        var = flat["state.var"]
        p = igr.normalize(igr.gaussian_pdf(g.x, mu, var), g)
        base = igr.gaussian_scores(g.x, mu, var)
        state = {"family": fam, "mu": q(mu, R.TRAIT), "var": q(var, R.TRAIT2)}
    else:
        sigma, nu = flat["state.sigma"], flat["state.nu"]
        p = igr.normalize(igr.student_t_pdf(g.x, mu, sigma, nu), g)
        base = igr.student_t_scores(g.x, mu, sigma, nu)
        state = {"family": fam, "mu": q(mu, R.TRAIT), "sigma": q(sigma, R.TRAIT), "nu": q(nu, None)}
    qnt = igr.weighted_quantile(fx, p, g)
    W = {"linear": igr.W_linear,
         "elite": lambda t: (lambda u: igr.W_elite(u, t["tau"])),
         "smooth_elite": lambda t: (lambda u: igr.W_smooth_elite(u, t["tau"], t["h"]))}
    rows = []
    for d in flat["dictionaries"]:
        a = d["augment"]
        if a == "none":
            scores = base
        elif a == "legendre_rank":
            scores = igr.stack(base, igr.legendre_rank(qnt, d["K"]))
        elif a == "rank_bins":
            scores = igr.stack(base, igr.rank_bins(qnt, d["K"]))
        else:
            scores = igr.stack(base, igr.rbf(g.x, np.asarray(d["centers"], float), d["h"]))
        for t in flat["targets"]:
            if t["signal"] == "raw":
                res = igr.project(fx - igr.expectation(fx, p, g), scores, p, g, ridge=flat["ridge"])
            else:
                Wf = W["linear"] if t["signal"] == "linear" else W[t["signal"]](t)
                res = igr.shaped_projection(fx, p, g, Wf, scores, ridge=flat["ridge"])
            rows.append({"dictionary": _dict_label(fam, d), "target": _target_label(t),
                         "directions": int(scores.shape[0]), "r2": float(res.r2), "defect": float(res.defect),
                         "defect2": float(res.defect2), "var_w": float(res.var_w), "cond": float(res.cond)})
    header = ["dictionary", "target", "directions", "r2", "defect", "defect2", "var_w", "cond"]
    write_csv(os.path.join(out_dir, "outputs", "projection.csv"), header, [[r[h] for r in rows] for h in header])
    return {"model": "projection", "fitness": fitness_label(flat), "state": state,
            "ridge": flat["ridge"],
            "units": {"r2": R.DIMLESS, "defect": "fitness units for raw targets, dimensionless for shaped targets",
                      "cond": R.DIMLESS},
            "results": [{k: r[k] for k in ("dictionary", "target", "directions", "r2", "defect", "cond")} for r in rows]}


# --------------------------------------------------------------------------- experiment
def _read_table(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]

    def conv(v):
        try:
            x = float(v)
            return int(x) if v.strip().lstrip("-").isdigit() else x
        except ValueError:
            return v
    return header, [[conv(v) for v in r] for r in body]


def run_experiment(flat, out_dir, log):
    import importlib
    from contextlib import redirect_stdout
    eid = flat["experiment.id"]
    _, module, produces, desc = next(e for e in R.EXPERIMENTS if e[0] == eid)
    exp = experiments_dir()
    if exp is None:
        raise RuntimeError("experiments/ directory not found next to the igr package; "
                           "install with `pip install -e .` from a repository checkout")
    root = repo_root()
    for p in (exp, root):
        if p not in sys.path:
            sys.path.insert(0, p)
    import igr.plotting as P
    fig_dir = os.path.join(out_dir, "outputs", "figures")
    tab_dir = os.path.join(out_dir, "outputs", "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)
    P.FIG_DIR, P.TAB_DIR = fig_dir, tab_dir          # redirect the scripts' output paths
    mod = importlib.import_module(module)
    settings = {}
    if eid == "S11":
        mod.NSAMPLES = flat["experiment.nsamples"]       # instead of IGR_NSAMPLES
        settings["nsamples"] = mod.NSAMPLES
    if eid == "S18":
        mod.NMC = flat["experiment.nmc"]                 # instead of IGR_NMC
        settings["nmc"] = mod.NMC
    buf = io.StringIO()
    log(f"running experiments/{module}.py ({produces})")
    with redirect_stdout(buf):
        mod.run()
    text = buf.getvalue()
    for line in text.splitlines():
        log("  | " + line)
    tables = {}
    for name in sorted(os.listdir(tab_dir)):
        if name.endswith(".csv"):
            header, body = _read_table(os.path.join(tab_dir, name))
            tables[name[:-4]] = {"file": f"outputs/tables/{name}", "columns": header, "rows": body}
    figures = sorted(f"outputs/figures/{n}" for n in os.listdir(fig_dir) if n.endswith(".png"))
    lines = [ln.rstrip() for ln in text.splitlines()
             if ln.strip() and not ln.startswith("saved ")]
    summary = {"model": "experiment",
               "experiment": {"id": eid, "script": f"experiments/{module}.py", "produces": produces, "description": desc},
               "settings": settings, "tables": tables, "figures": figures,
               "units": "table values are dimensionless (R^2, correlations, errors, defects) unless the column name says otherwise",
               "reported_lines": lines[:200] if not tables else [],
               "reported_lines_note": ("printed numeric results (the script saves no table)" if not tables
                                       else "this experiment saves tables; the printed text is in log.txt")}
    return summary


RUNNERS = {"trajectory": run_trajectory, "projection": run_projection, "experiment": run_experiment}


def schema_for_run(model, flat, out_dir):
    """Describe every file and column in outputs/ (generated from the registry)."""
    outs = {o["name"]: o for o in R.OUTPUTS[model]}
    files = []
    if model == "trajectory":
        surr = flat["surrogates"]
        cols = ["t", "mu_true", "var_true"] + [c.format(s=s).split("{")[0] for s in surr for c, _, _ in R.SURR_COLS]
        files.append({"file": "outputs/trajectory.csv", "format": "csv", "shape": "time series, one row per output time",
                      "rows": flat["time.n_points"],
                      "columns": [{"name": c, "unit": outs[c]["unit"], "description": outs[c]["description"]} for c in cols]})
        dcols = ["x", "rho_true"] + [f"rho_{s}" for s in surr]
        files.append({"file": "outputs/densities_final.csv", "format": "csv", "shape": "one row per grid point, at t_end",
                      "rows": flat["grid.n"],
                      "columns": [{"name": c, "unit": outs[c]["unit"], "description": outs[c]["description"]} for c in dcols]})
    elif model == "projection":
        cols = ["dictionary", "target", "directions", "r2", "defect", "defect2", "var_w", "cond"]
        files.append({"file": "outputs/projection.csv", "format": "csv", "shape": "one row per (dictionary, target) pair",
                      "rows": len(flat["dictionaries"]) * len(flat["targets"]),
                      "columns": [{"name": c, "unit": outs[c]["unit"], "description": outs[c]["description"]} for c in cols]})
    else:
        tab_dir = os.path.join(out_dir, "outputs", "tables")
        for name in sorted(os.listdir(tab_dir)) if os.path.isdir(tab_dir) else []:
            path = os.path.join(tab_dir, name)
            if name.endswith(".csv"):
                header, body = _read_table(path)
                files.append({"file": f"outputs/tables/{name}", "format": "csv", "shape": "table", "rows": len(body),
                              "columns": [{"name": c, "unit": None,
                                           "description": f"column '{c}' as written by the experiment script (see its table in the supplement)"}
                                          for c in header]})
            elif name.endswith(".md"):
                files.append({"file": f"outputs/tables/{name}", "format": "markdown", "shape": "table (same content as the CSV)"})
        fig_dir = os.path.join(out_dir, "outputs", "figures")
        for name in sorted(os.listdir(fig_dir)) if os.path.isdir(fig_dir) else []:
            files.append({"file": f"outputs/figures/{name}", "format": "png", "shape": "image"})
    return {"model": model, "files": files}
