# -*- coding: utf-8 -*-
"""Declarative configuration: load, validate, resolve, template, JSON Schema, cost.

Everything here is generated from igr_cli.registry. No numerical libraries are
imported, so `validate`, `template`, and `describe` stay fast.
"""
import copy
import difflib
import itertools
import json
import math
import os
import re

from . import registry as R

NUM_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


class ConfigError(Exception):
    """Raised with a list of structured errors {field, value, reason, hint}."""

    def __init__(self, errors):
        super().__init__("; ".join(f"{e['field']}: {e['reason']}" for e in errors))
        self.errors = errors


def err(field, value, reason, hint=None):
    return {"field": field, "value": value, "reason": reason, "hint": hint}


# --------------------------------------------------------------------------- load
def load(path):
    """Load a YAML or JSON configuration file ('-' reads standard input)."""
    import sys
    if path == "-":
        text = sys.stdin.read()
        base = os.getcwd()
    else:
        if not os.path.exists(path):
            raise ConfigError([err("<config>", path, "configuration file not found",
                                   "pass the path of a .yaml or .json file, or '-' for stdin")])
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        base = os.path.dirname(os.path.abspath(path))
    try:
        if path.endswith(".json"):
            data = json.loads(text)
        else:
            import yaml
            data = yaml.safe_load(text)
    except Exception as exc:  # parse error
        raise ConfigError([err("<config>", path, f"could not parse configuration: {exc}",
                               "check YAML/JSON syntax; `igr template <model>` prints a valid example")])
    if not isinstance(data, dict):
        raise ConfigError([err("<config>", type(data).__name__, "configuration must be a mapping",
                               "start from `igr template <model>`")])
    return data, base


# --------------------------------------------------------------------------- helpers
def get_path(d, dotted):
    cur = d
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None, False
        cur = cur[k]
    return cur, True


def set_path(d, dotted, value):
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def _range_text(r):
    if not r:
        return None
    lo = "(" if r.get("min_exclusive") else "["
    hi = ")" if r.get("max_exclusive") or r.get("max") is None else "]"
    a = "-inf" if r.get("min") is None else r["min"]
    b = "inf" if r.get("max") is None else r["max"]
    return f"{lo}{a}, {b}{hi}"


def _in_range(v, r):
    if not r:
        return True
    if r.get("min") is not None:
        if r.get("min_exclusive") and not v > r["min"]:
            return False
        if not r.get("min_exclusive") and not v >= r["min"]:
            return False
    if r.get("max") is not None:
        if r.get("max_exclusive") and not v < r["max"]:
            return False
        if not r.get("max_exclusive") and not v <= r["max"]:
            return False
    return True


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _coerce(field, value, typ, options, rng, errors, warnings):
    """Type-check one value. Returns the (possibly coerced) value."""
    if typ == "number":
        if isinstance(value, str) and NUM_RE.match(value.strip()):
            warnings.append(f"{field}: string {value!r} read as a number (YAML 1.1 needs a dot, e.g. 1.0e-9)")
            value = float(value)
        if not _is_num(value):
            errors.append(err(field, value, "expected a finite number", f"range {_range_text(rng) or 'unrestricted'}"))
            return value
        value = float(value) if isinstance(value, int) else value
    elif typ == "integer":
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(err(field, value, "expected an integer", f"range {_range_text(rng) or 'unrestricted'}"))
            return value
    elif typ == "enum":
        if value not in options:
            close = difflib.get_close_matches(str(value), [str(o) for o in options], n=1)
            errors.append(err(field, value, "not one of the allowed options",
                              (f"did you mean {close[0]!r}? " if close else "") + f"options: {options}"))
        return value
    elif typ.startswith("list["):
        inner = typ[5:-1]
        if not isinstance(value, list):
            errors.append(err(field, value, f"expected a list of {inner}", None))
            return value
        if inner in ("number", "enum", "integer"):
            return [_coerce(f"{field}[{i}]", v, inner, options, rng, errors, warnings) for i, v in enumerate(value)]
        return value
    if typ in ("number", "integer") and not _in_range(value, rng):
        errors.append(err(field, value, f"out of range {_range_text(rng)}", f"use a value in {_range_text(rng)}"))
    return value


def _when_ok(entry, flat):
    w = entry.get("when")
    if not w:
        return True
    for k, allowed in w.items():
        if flat.get(k) not in allowed:
            return False
    return True


# --------------------------------------------------------------------------- list items
def _validate_items(entry, value, errors, warnings):
    """Validate a list[mapping] parameter (targets, dictionaries). Returns resolved list."""
    name = entry["name"]
    items = entry["items"]
    disc = next(iter(items))                 # discriminator key: signal / augment
    cond_key = f"when_{disc}"
    out = []
    if not isinstance(value, list):
        errors.append(err(name, value, "expected a list of mappings", f"example: {json.dumps(entry['default'])}"))
        return value
    if len(value) == 0:
        errors.append(err(name, value, "must contain at least one entry", f"example: {json.dumps(entry['default'])}"))
    for i, item in enumerate(value):
        f = f"{name}[{i}]"
        if isinstance(item, str):              # shorthand: "raw" / "none"
            item = {disc: item}
        if not isinstance(item, dict):
            errors.append(err(f, item, "expected a mapping", f"keys: {list(items)}"))
            continue
        d = item.get(disc)
        res = {}
        spec = items[disc]
        if disc not in item:
            errors.append(err(f"{f}.{disc}", None, "missing required key", f"options: {spec['options']}"))
            continue
        res[disc] = _coerce(f"{f}.{disc}", d, "enum", spec["options"], None, errors, warnings)
        for k, v in item.items():
            if k == disc:
                continue
            if k not in items:
                close = difflib.get_close_matches(k, list(items), n=1)
                errors.append(err(f"{f}.{k}", v, "unknown key",
                                  (f"did you mean {close[0]!r}? " if close else "") + f"valid keys: {list(items)}"))
                continue
            if d not in items[k].get(cond_key, []):
                errors.append(err(f"{f}.{k}", v, f"not used when {disc} is {d!r}",
                                  f"{k} applies only to {disc} in {items[k].get(cond_key)}"))
                continue
            res[k] = _coerce(f"{f}.{k}", v, items[k]["type"], items[k].get("options"), items[k].get("range"), errors, warnings)
        for k, s in items.items():
            if k == disc or d not in s.get(cond_key, []) or k in res:
                continue
            if s["default"] == R.REQUIRED:
                errors.append(err(f"{f}.{k}", None, f"required when {disc} is {d!r}", s["description"]))
            else:
                res[k] = copy.deepcopy(s["default"])
        if "centers" in res and isinstance(res["centers"], list) and len(res["centers"]) == 0:
            errors.append(err(f"{f}.centers", [], "must contain at least one center", "e.g. [-4, -3, ..., 4]"))
        out.append(res)
    return out


# --------------------------------------------------------------------------- validate
def _collect_leaves(params, errors, prefix=""):
    """Walk the parameters mapping and return {dotted: value} for registered leaves."""
    leaves = {}
    if not isinstance(params, dict):
        errors.append(err("parameters" + ("." + prefix.rstrip(".") if prefix else ""), params,
                          "expected a mapping", None))
        return leaves
    for k, v in params.items():
        path = prefix + str(k)
        if path in R.PARAM_INDEX:
            leaves[path] = v
        elif any(n.startswith(path + ".") for n in R.PARAM_INDEX):
            if isinstance(v, dict):
                leaves.update(_collect_leaves(v, errors, path + "."))
            else:
                errors.append(err(path, v, "expected a mapping (this is a group of parameters)",
                                  "children: " + ", ".join(n for n in R.PARAM_INDEX if n.startswith(path + "."))))
        else:
            close = difflib.get_close_matches(path, list(R.PARAM_INDEX), n=1, cutoff=0.6)
            errors.append(err(path, v, "unknown parameter",
                              f"did you mean {close[0]!r}?" if close else "run `igr describe` for the list of parameters"))
    return leaves


def validate(data, base_dir=None, check_model_params=True):
    """Validate a configuration mapping. Returns (resolved, warnings) or raises ConfigError
    with every problem found."""
    errors, warnings = [], []
    for k in data:
        if k not in R.TOP_LEVEL:
            close = difflib.get_close_matches(str(k), list(R.TOP_LEVEL), n=1)
            errors.append(err(str(k), data[k], "unknown top-level key",
                              (f"did you mean {close[0]!r}? " if close else "") + f"valid keys: {list(R.TOP_LEVEL)}"))
    schema = data.get("schema", R.SCHEMA_VERSION)
    if schema != R.SCHEMA_VERSION:
        errors.append(err("schema", schema, "unsupported configuration format version", "use schema: 1"))
    model = data.get("model")
    if model is None:
        errors.append(err("model", None, "missing required key", f"options: {list(R.MODELS)}"))
    elif model not in R.MODELS:
        close = difflib.get_close_matches(str(model), list(R.MODELS), n=1)
        errors.append(err("model", model, "unknown model",
                          (f"did you mean {close[0]!r}? " if close else "") + f"options: {list(R.MODELS)}"))
        model = None
    seed = data.get("seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
        errors.append(err("seed", seed, "seed must be a non-negative integer", "e.g. seed: 42"))
    params = data.get("parameters", {}) or {}
    leaves = _collect_leaves(params, errors)
    sweep = data.get("sweep")
    if sweep is not None and not isinstance(sweep, dict):
        errors.append(err("sweep", sweep, "expected a mapping {parameter: [values]}", None))
    if model is None:
        raise ConfigError(errors)

    flat = {}
    # discriminators first (fitness.kind, state.family, experiment.id) so `when` can be checked
    order = sorted(R.params_for(model), key=lambda e: 0 if "when" not in e else 1)
    for entry in order:
        name = entry["name"]
        if not _when_ok(entry, flat):
            continue
        if name in leaves:
            v = leaves[name]
            if entry["type"] == "list[mapping]":
                flat[name] = _validate_items(entry, v, errors, warnings)
            else:
                flat[name] = _coerce(name, v, entry["type"], entry.get("options"), entry.get("range"), errors, warnings)
        elif entry["default"] == R.REQUIRED:
            errors.append(err(name, None, "missing required parameter",
                              entry["description"] + (f" Options: {entry['options']}" if entry.get("options") else "")))
        else:
            flat[name] = copy.deepcopy(entry["default"])
    # given parameters that the model or the chosen variant does not use
    for name, v in leaves.items():
        entry = R.PARAM_INDEX[name]
        if model not in entry["applies_to"]:
            errors.append(err(name, v, f"not used by model {model!r}",
                              f"applies to {entry['applies_to']}; remove it"))
        elif not _when_ok(entry, flat):
            cond = "; ".join(f"{k} in {vals}" for k, vals in entry["when"].items())
            errors.append(err(name, v, f"not used with the chosen variant (requires {cond})", "remove it or change the variant"))

    _check_constraints(model, flat, errors, warnings)
    if isinstance(sweep, dict):
        for k, vals in sweep.items():
            if k not in R.PARAM_INDEX or model not in R.PARAM_INDEX[k]["applies_to"]:
                close = difflib.get_close_matches(str(k), [p["name"] for p in R.params_for(model)], n=1)
                errors.append(err(f"sweep.{k}", vals, "not a parameter of this model",
                                  f"did you mean {close[0]!r}?" if close else "run `igr describe --model " + model + "`"))
            elif not isinstance(vals, list) or not vals:
                errors.append(err(f"sweep.{k}", vals, "expected a non-empty list of values", "e.g. [0.1, 0.2, 0.4]"))
    if errors:
        raise ConfigError(errors)

    nested = {}
    for entry in R.params_for(model):
        if entry["name"] in flat:
            set_path(nested, entry["name"], flat[entry["name"]])
    resolved = {"schema": R.SCHEMA_VERSION, "model": model, "seed": seed, "parameters": nested}
    if sweep:
        resolved["sweep"] = sweep
    return resolved, warnings


def flatten_params(resolved):
    flat = {}

    def walk(d, prefix):
        for k, v in d.items():
            path = prefix + k
            if path in R.PARAM_INDEX:
                flat[path] = v
            elif isinstance(v, dict):
                walk(v, path + ".")
    walk(resolved.get("parameters", {}), "")
    return flat


def blowup_time(flat):
    """Blow-up time of the Gaussian variance (Eq. 50) if the fitness is a convex quadratic."""
    kind = flat.get("fitness.kind")
    var0 = flat.get("initial.var")
    if var0 is None:
        return None
    c = None
    if kind == "polynomial":
        co = flat.get("fitness.polynomial.coefficients") or []
        if isinstance(co, list) and len(co) == 3 and all(_is_num(x) for x in co) and co[2] > 0:
            c = co[2]
        elif isinstance(co, list) and len(co) > 3 and all(_is_num(x) for x in co) and all(x == 0 for x in co[3:]) and co[2] > 0:
            c = co[2]
    elif kind == "perturbed_quadratic":
        b, eps = flat.get("fitness.perturbed_quadratic.b"), flat.get("fitness.perturbed_quadratic.eps")
        if _is_num(b) and _is_num(eps) and eps == 0 and b < 0:
            c = -b
    if c is None or not _is_num(var0) or var0 <= 0:
        return None
    return 1.0 / (2.0 * c * var0)


def _check_constraints(model, flat, errors, warnings):
    if model == "experiment":
        from .models import experiments_dir
        if experiments_dir() is None:
            errors.append(err("experiment.id", flat.get("experiment.id"),
                              "the experiment scripts (experiments/*.py) were not found next to the installed package",
                              "install from a repository checkout with `pip install -e .`"))
    a, b = flat.get("grid.a"), flat.get("grid.b")
    if _is_num(a) and _is_num(b) and not b > a:
        errors.append(err("grid.b", b, f"constraint grid.b > grid.a violated (grid.a = {a})", "the grid is [a, b] with a < b"))
    if model == "trajectory":
        tstar = blowup_time(flat)
        t_end = flat.get("time.t_end")
        if tstar is not None and _is_num(t_end) and t_end >= tstar:
            errors.append(err("time.t_end", t_end,
                              f"constraint time.t_end < t* violated: the Gaussian variance blows up at t* = 1/(2 c var0) = {tstar:.6g} (Eq. 50)",
                              f"use t_end < {tstar:.6g} (Fig S4 stops at 0.985 t*)"))
        s = flat.get("surrogates")
        if isinstance(s, list):
            if len(s) == 0:
                errors.append(err("surrogates", s, "must contain at least one surrogate", "options: ['param', 'proj']"))
            if len(set(map(str, s))) != len(s):
                errors.append(err("surrogates", s, "duplicate entries", "list each surrogate once"))
        if flat.get("fitness.kind") == "perturbed_quadratic" and _is_num(flat.get("fitness.perturbed_quadratic.b")) \
                and flat["fitness.perturbed_quadratic.b"] < 0 and flat.get("fitness.perturbed_quadratic.eps") not in (0, 0.0):
            warnings.append("fitness.perturbed_quadratic.b < 0 with eps != 0: convex fitness, the surrogate variance may blow up (Eq. 50 gives t* only for eps = 0)")
    co = flat.get("fitness.polynomial.coefficients")
    if flat.get("fitness.kind") == "polynomial" and isinstance(co, list) and len(co) == 0:
        errors.append(err("fitness.polynomial.coefficients", co, "must contain at least one coefficient", "e.g. [0, 1] for f(x) = x"))
    if flat.get("fitness.kind") == "multimodal":
        w = flat.get("fitness.multimodal.width")
        if _is_num(w) and w == 0:
            errors.append(err("fitness.multimodal.width", w, "constraint width != 0 violated (division by width^2)", "experiments use 0.6"))
        c = flat.get("fitness.multimodal.centers")
        if isinstance(c, list) and len(c) == 0:
            errors.append(err("fitness.multimodal.centers", c, "must contain at least one center", "default [-2.0, 2.0]"))


# --------------------------------------------------------------------------- sweep
def sweep_points(resolved):
    sweep = resolved.get("sweep") or {}
    keys = list(sweep)
    points = []
    for combo in itertools.product(*[sweep[k] for k in keys]):
        cfg = copy.deepcopy(resolved)
        cfg.pop("sweep", None)
        for k, v in zip(keys, combo):
            set_path(cfg["parameters"], k, v)
        points.append((dict(zip(keys, combo)), cfg))
    return points


# --------------------------------------------------------------------------- cost estimate
# Calibrated by timing `igr run` on the audit machine (Apple silicon laptop, Python 3.12,
# numpy 2.5). Machine-dependent; treat as order-of-magnitude.
STARTUP_S = 0.45                       # CLI start + numpy/scipy import
TRAJ_PER_GRIDPOINT_TIME = 1.0e-8       # exact tilt + moments, per grid point per output time
TRAJ_PER_GRIDPOINT_TIME_SURR = 2.6e-8  # distances + defect per surrogate, per grid point per output time
TRAJ_PER_GRIDPOINT_SOLVE = 4.5e-6      # one surrogate ODE solve (~hundreds of RHS evaluations), per grid point
PROJ_PER_GRIDPOINT_DIR2 = 5.0e-9       # projection, per grid point per direction^2
# Measured wall-clock seconds and peak memory (MB) of `igr run` for each experiment at default settings.
EXPERIMENT_COST = {
    "S01": (2.0, 152), "S03": (2.3, 117), "S04": (1.5, 114), "S05": (1.3, 158), "S06": (1.2, 147),
    "S07": (1.2, 161), "S08": (1.6, 167), "S09": (1.7, 150), "S11": (1.9, 150), "S13": (1.3, 148),
    "S14": (15.5, 134), "S17": (1.4, 155), "S18": (2.2, 941), "S19": (2.4, 119), "CHK": (0.8, 106),
}


def estimate(resolved):
    flat = flatten_params(resolved)
    model = resolved["model"]
    if model == "trajectory":
        n = flat["grid.n"]
        npts = flat["time.n_points"]
        S = len(flat["surrogates"])
        T = flat["time.t_end"]
        secs = STARTUP_S + n * (npts * (TRAJ_PER_GRIDPOINT_TIME + S * TRAJ_PER_GRIDPOINT_TIME_SURR)
                                + S * TRAJ_PER_GRIDPOINT_SOLVE * (max(T, 0.1) / 2.0) ** 0.2)
        mem = 90 + n * 8 * 40 / 1e6
        out = (npts * (3 + 7 * S) * 22 + n * (2 + S) * 22) / 1e6
        method = ("linear in grid.n x time.n_points plus one adaptive ODE solve per surrogate, "
                  "calibrated on the audit machine (stiff fitness or tight tolerances take longer)")
    elif model == "projection":
        n = flat["grid.n"]
        dirs = [2 + (d.get("K") or len(d.get("centers") or [])) for d in flat["dictionaries"]]
        secs = STARTUP_S + sum(n * (k * k + 4 * k) * PROJ_PER_GRIDPOINT_DIR2 for k in dirs) * len(flat["targets"])
        mem = 80 + n * 8 * (max(dirs) + 10) / 1e6
        out = len(dirs) * len(flat["targets"]) * 160 / 1e6
        method = "quadrature cost ~ grid.n x directions^2 per (dictionary, target) pair, calibrated on the audit machine"
    else:
        eid = flat["experiment.id"]
        secs, mem = EXPERIMENT_COST[eid]
        if eid == "S11":
            secs = 1.6 + (secs - 1.6) * flat["experiment.nsamples"] / 60.0
        if eid == "S18":
            secs = 1.4 + (secs - 1.4) * flat["experiment.nmc"] / 40000.0
            mem = 110 + (mem - 110) * flat["experiment.nmc"] / 40000.0
        out = 0.3 if eid != "CHK" else 0.001
        method = "measured runtime and peak memory of this experiment script during the audit, scaled by nsamples/nmc"
    return {"runtime_s": round(secs, 2), "memory_mb": round(mem, 0), "output_mb": round(out, 3),
            "method": method, "machine_dependent": True,
            "note": ("steady-state estimate; the first run after installation can take about 10 s longer "
                     "(one-time import caching; matplotlib builds its font cache on first use)")}


# --------------------------------------------------------------------------- template
TEMPLATE_EXAMPLES = {
    "fitness.kind": ("perturbed_quadratic", "Experiments S1, S3, S9"),
    "fitness.perturbed_quadratic.eps": (0.3, "the S1 'quadratic + small sinusoid' case"),
    "fitness.polynomial.coefficients": ([0.0, 1.0], "f(x) = x, Experiment S2 / Table S1"),
    "fitness.rational_conjugate.alpha": (0.0, "S17 uses alpha = 0"),
    "fitness.rational_conjugate.beta": (1.0, "S17 uses beta = 1"),
    "fitness.rational_conjugate.gamma": (0.0, "S17 uses gamma = 0"),
    "fitness.rational_conjugate.mu": (0.0, "S17"),
    "fitness.rational_conjugate.sigma": (1.0, "S17"),
    "fitness.rational_conjugate.nu": (3.0, "S17"),
    "fitness.quadratic_rational_homotopy.s": (0.5, "S17 sweeps s in [0, 1]"),
    "fitness.quadratic_rational_homotopy.nu": (3.0, "S17"),
    "state.nu": (3.0, "S17"),
    "experiment.id": ("S05", "Table S1"),
}


def template_config(model, fitness=None, family=None, experiment=None):
    """Return (config mapping, {dotted: comment}) for a complete, valid config."""
    data = {"schema": R.SCHEMA_VERSION, "model": model, "seed": 0, "parameters": {}}
    overrides = {}
    if fitness:
        overrides["fitness.kind"] = fitness
    if family:
        overrides["state.family"] = family
    if experiment:
        overrides["experiment.id"] = experiment
    if model == "projection" and not fitness:
        overrides["fitness.kind"] = "polynomial"
    comments = {}
    flat = {}
    for entry in sorted(R.params_for(model), key=lambda e: 0 if "when" not in e else 1):
        name = entry["name"]
        if not _when_ok(entry, flat):
            continue
        note = ""
        if name in overrides:
            val = overrides[name]
        elif entry["default"] == R.REQUIRED:
            val, src = TEMPLATE_EXAMPLES[name]
            note = f"REQUIRED (example value: {src}). "
        else:
            val = copy.deepcopy(entry["default"])
        flat[name] = val
        bits = [note + entry["description"]]
        if entry.get("unit"):
            bits.append(f"unit: {entry['unit']}")
        if entry.get("options"):
            bits.append(f"options: {entry['options']}")
        if entry["type"] in ("number", "integer", "list[number]"):
            bits.append(f"range: {_range_text(entry.get('range')) or 'not stated in source'}")
        if entry.get("constraints"):
            bits.append("constraints: " + "; ".join(entry["constraints"]))
        comments[name] = " | ".join(bits)
    nested = {}
    for entry in R.params_for(model):
        if entry["name"] in flat:
            set_path(nested, entry["name"], flat[entry["name"]])
    data["parameters"] = nested
    return data, comments


def _yaml_scalar(v):
    import yaml
    s = yaml.safe_dump(v, default_flow_style=True, width=10 ** 6).strip()
    if s.endswith("\n...") or s.endswith("..."):
        s = s.replace("\n...", "").rstrip(".").strip() if s != "..." else s
    return s


def to_commented_yaml(data, comments, header=None):
    lines = []
    if header:
        lines += [f"# {h}" for h in header]
    top = R.TOP_LEVEL
    for k in ("schema", "model", "seed"):
        lines.append(f"# {top[k]['description']}")
        lines.append(f"{k}: {_yaml_scalar(data[k])}")
    lines.append("parameters:")

    def walk(d, prefix, indent):
        for k, v in d.items():
            path = prefix + k
            pad = "  " * indent
            if path in R.PARAM_INDEX:
                lines.append(f"{pad}# {comments.get(path, '')}")
                lines.append(f"{pad}{k}: {_yaml_scalar(v)}")
            else:
                lines.append(f"{pad}{k}:")
                walk(v, path + ".", indent + 1)
    walk(data["parameters"], "", 1)
    return "\n".join(lines) + "\n"


def dump_yaml(data):
    import yaml
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=None, width=100)


# --------------------------------------------------------------------------- JSON Schema
def _type_schema(typ, options=None, rng=None):
    if typ == "number":
        s = {"type": "number"}
    elif typ == "integer":
        s = {"type": "integer"}
    elif typ == "enum":
        s = {"enum": options}
    elif typ.startswith("list["):
        s = {"type": "array", "items": _type_schema(typ[5:-1], options, rng)}
        return s
    else:
        s = {}
    if rng and typ in ("number", "integer"):
        if rng.get("min") is not None:
            s["exclusiveMinimum" if rng.get("min_exclusive") else "minimum"] = rng["min"]
        if rng.get("max") is not None:
            s["exclusiveMaximum" if rng.get("max_exclusive") else "maximum"] = rng["max"]
    return s


def _items_schema(entry):
    items = entry["items"]
    disc = next(iter(items))
    cond_key = f"when_{disc}"
    variants = []
    for opt in items[disc]["options"]:
        props = {disc: {"const": opt}}
        req = [disc]
        for k, s in items.items():
            if k == disc or opt not in s.get(cond_key, []):
                continue
            props[k] = dict(_type_schema(s["type"], s.get("options"), s.get("range")), description=s["description"])
            if s["default"] == R.REQUIRED:
                req.append(k)
            else:
                props[k]["default"] = s["default"]
        variants.append({"type": "object", "properties": props, "required": req, "additionalProperties": False})
    return {"type": "array", "minItems": 1, "items": {"oneOf": variants + [{"enum": items[disc]["options"]}]}}


def _nest(keys, leaf, require):
    """Schema fragment placing `leaf` at a dotted path; `require` makes every level required."""
    node = leaf
    for k in reversed(keys):
        frag = {"properties": {k: node}}
        if require:
            frag["required"] = [k]
        node = frag
    return node


def _merge(a, b):
    for k, v in b.items():
        if k == "required":
            a.setdefault("required", [])
            a["required"] += [x for x in v if x not in a["required"]]
        elif k == "properties":
            props = a.setdefault("properties", {})
            for pk, pv in v.items():
                if pk in props and isinstance(props[pk], dict) and isinstance(pv, dict):
                    _merge(props[pk], pv)
                else:
                    props[pk] = pv
        else:
            a[k] = v
    return a


def _params_schema(model):
    root = {"type": "object", "properties": {}, "additionalProperties": False}
    conds = []
    for entry in R.params_for(model):
        keys = entry["name"].split(".")
        node = root
        for k in keys[:-1]:
            node = node["properties"].setdefault(k, {"type": "object", "properties": {}, "additionalProperties": False})
        if entry["type"] == "list[mapping]":
            s = _items_schema(entry)
        else:
            s = _type_schema(entry["type"], entry.get("options"), entry.get("range"))
        s["description"] = entry["description"] + (f" Unit: {entry['unit']}." if entry.get("unit") else "")
        if entry["default"] != R.REQUIRED:
            s["default"] = entry["default"]
        node["properties"][keys[-1]] = s
        if entry["default"] == R.REQUIRED and "when" not in entry:
            parent = root
            for k in keys[:-1]:
                parent.setdefault("required", [])
                if k not in parent["required"]:
                    parent["required"].append(k)
                parent = parent["properties"][k]
            parent.setdefault("required", []).append(keys[-1])
        if "when" in entry:
            conds.append(entry)
    # conditional blocks: fitness.<kind>.* only with the matching kind, state.* per
    # family, experiment.* per id. if <discriminator in values> then <required leaves>
    # else <those leaves forbidden>.
    allof = []
    groups = {}
    for e in conds:
        (ck, vals), = e["when"].items()
        groups.setdefault((ck, tuple(vals)), []).append(e)
    for (ck, vals), entries in groups.items():
        cond = _nest(ck.split("."), {"enum": list(vals)}, require=True)
        if R.PARAM_INDEX[ck]["default"] in vals:   # discriminator omitted -> its default applies
            cond = {"anyOf": [cond, {"not": _nest(ck.split("."), {}, require=True)}]}
        then_s, else_s = {}, {}
        for e in entries:
            keys = e["name"].split(".")
            if e["default"] == R.REQUIRED:
                _merge(then_s, _nest(keys, {}, require=True))
            _merge(else_s, _nest(keys, False, require=False))
        allof.append({"if": cond, "then": then_s, "else": else_s})
    if allof:
        root["allOf"] = allof
    return root


def json_schema():
    defs = {m: _params_schema(m) for m in R.MODELS}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "igr-cli/config.schema.json",
        "title": "igr run configuration (schema 1)",
        "description": "Generated from igr_cli/registry.py by `igr describe --schema`. Do not edit by hand.",
        "type": "object",
        "required": ["model"],
        "additionalProperties": False,
        "properties": {
            "schema": {"const": 1, "description": R.TOP_LEVEL["schema"]["description"]},
            "model": {"enum": list(R.MODELS), "description": R.TOP_LEVEL["model"]["description"]},
            "seed": {"type": "integer", "minimum": 0, "description": R.TOP_LEVEL["seed"]["description"]},
            "parameters": {"type": "object"},
            "sweep": {"type": "object", "additionalProperties": {"type": "array", "minItems": 1},
                      "description": R.TOP_LEVEL["sweep"]["description"]},
        },
        "allOf": [{"if": {"properties": {"model": {"const": m}}, "required": ["model"]},
                   "then": {"properties": {"parameters": {"$ref": f"#/$defs/{m}"}}}} for m in R.MODELS],
        "$defs": defs,
    }
