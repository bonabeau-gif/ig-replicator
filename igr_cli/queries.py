# -*- coding: utf-8 -*-
"""Query and compare commands. They read run-directory files only (standard library
only, no numerical imports) and never rerun a model."""
import csv
import json
import os


class QueryError(Exception):
    def __init__(self, field, value, reason, hint=None):
        super().__init__(reason)
        self.error = {"field": field, "value": value, "reason": reason, "hint": hint}


MAIN_TABLE = {"trajectory": "outputs/trajectory.csv", "projection": "outputs/projection.csv"}


# --------------------------------------------------------------------------- loading
def _load_json(run, name):
    p = os.path.join(run, name)
    if not os.path.exists(p):
        raise QueryError("run", run, f"{name} not found; not a completed run directory",
                         "pass a directory written by `igr run`")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def load_run(run):
    if not os.path.isdir(run):
        raise QueryError("run", run, "run directory not found", "pass a directory written by `igr run`")
    manifest = _load_json(run, "manifest.json")
    if manifest.get("status") != "completed":
        raise QueryError("run", run, f"run status is {manifest.get('status')!r}, not 'completed'",
                         f"error: {manifest.get('error')}")
    return manifest, _load_json(run, "summary.json"), _load_json(run, "schema.json")


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def read_csv(run, rel):
    p = os.path.join(run, rel)
    if not os.path.exists(p):
        raise QueryError("file", rel, "output file not found", "see `igr query <run> outputs`")
    with open(p, newline="", encoding="utf-8") as fh:
        r = csv.reader(fh)
        header = next(r)
        rows = [[_num(v) for v in row] for row in r]
    return header, rows


def _units(schema, rel):
    for f in schema["files"]:
        if f["file"] == rel:
            return {c["name"]: c.get("unit") for c in f.get("columns", [])}
    return {}


def _table_for(run, schema, model, file=None):
    if file:
        rel = file if file.startswith("outputs/") else None
        if rel is None:
            cands = [f["file"] for f in schema["files"] if f.get("format") == "csv" and
                     (os.path.basename(f["file"]) in (file, file + ".csv"))]
            if not cands:
                raise QueryError("--file", file, "no such CSV output",
                                 "options: " + ", ".join(f["file"] for f in schema["files"] if f.get("format") == "csv"))
            rel = cands[0]
        return rel
    if model in MAIN_TABLE:
        return MAIN_TABLE[model]
    csvs = [f["file"] for f in schema["files"] if f.get("format") == "csv"]
    if len(csvs) == 1:
        return csvs[0]
    raise QueryError("--file", None, "this run has several tables; choose one",
                     "options: " + ", ".join(csvs) if csvs else "this experiment saves no tables; see `query summary`")


def _column(header, rows, name, rel):
    if name not in header:
        import difflib
        close = difflib.get_close_matches(name, header, n=1)
        raise QueryError("name", name, f"no column {name!r} in {rel}",
                         (f"did you mean {close[0]!r}? " if close else "") + f"columns: {header}")
    i = header.index(name)
    return [r[i] for r in rows]


def _require_time(model, what):
    if model != "trajectory":
        raise QueryError("query", what, f"'{what}' needs a time axis; only the trajectory model records one",
                         "use `query <run> table` or `query <run> stats <column>`")


def _envelope(query, run, result, approximation=None, units=None):
    return {"ok": True, "query": query, "run": run, "result": result,
            "units": units, "approximation": approximation}


# --------------------------------------------------------------------------- queries
def q_summary(run, args):
    _, summary, _ = load_run(run)
    return _envelope("summary", run, summary, None, "each quantity carries its own unit")


def q_outputs(run, args):
    manifest, _, schema = load_run(run)
    files = []
    for f in schema["files"]:
        p = os.path.join(run, f["file"])
        entry = {"file": f["file"], "format": f.get("format"), "shape": f.get("shape"),
                 "bytes": os.path.getsize(p) if os.path.exists(p) else None}
        if "rows" in f:
            entry["rows"] = f["rows"]
        if "columns" in f:
            entry["columns"] = [{"name": c["name"], "unit": c.get("unit"), "description": c.get("description")} for c in f["columns"]]
        files.append(entry)
    return _envelope("outputs", run, {"model": manifest["model"], "files": files})


def _select(n, points):
    if n <= points:
        return list(range(n))
    idx = sorted({round(i * (n - 1) / (points - 1)) for i in range(points)})
    return idx


def q_series(run, args):
    manifest, _, schema = load_run(run)
    model = manifest["model"]
    _require_time(model, "series")
    rel = MAIN_TABLE[model]
    header, rows = read_csv(run, rel)
    t = _column(header, rows, "t", rel)
    y = _column(header, rows, args.name, rel)
    sel = [i for i in range(len(t)) if (args.t_from is None or t[i] >= args.t_from) and (args.t_to is None or t[i] <= args.t_to)]
    if args.points < 2:
        raise QueryError("--points", args.points, "must be at least 2", "default 100")
    pick = [sel[i] for i in _select(len(sel), args.points)]
    units = _units(schema, rel)
    approx = None
    if len(pick) < len(sel):
        approx = (f"downsampled {len(sel)} recorded points to {len(pick)} by evenly spaced index selection "
                  "(first and last kept, no averaging or interpolation)")
    return _envelope("series", run, {"name": args.name, "n_recorded": len(sel), "n_returned": len(pick),
                                     "t": [t[i] for i in pick], "values": [y[i] for i in pick]},
                     approx, {"t": units.get("t"), args.name: units.get(args.name)})


def q_stats(run, args):
    manifest, _, schema = load_run(run)
    model = manifest["model"]
    rel = _table_for(run, schema, model, args.file)
    header, rows = read_csv(run, rel)
    y = _column(header, rows, args.name, rel)
    vals = [v for v in y if isinstance(v, float) and v == v]
    if not vals:
        raise QueryError("name", args.name, "column is not numeric", f"numeric columns: {[h for h, v in zip(header, rows[0]) if isinstance(v, float)]}")
    imax = max(range(len(y)), key=lambda i: y[i] if isinstance(y[i], float) and y[i] == y[i] else float("-inf"))
    imin = min(range(len(y)), key=lambda i: y[i] if isinstance(y[i], float) and y[i] == y[i] else float("inf"))
    res = {"name": args.name, "file": rel, "n": len(vals), "min": min(vals), "max": max(vals),
           "mean": sum(vals) / len(vals), "final": y[-1]}
    approx = "mean is the arithmetic mean of recorded rows (not a time average)"
    if "t" in header:
        t = _column(header, rows, "t", rel)
        res["t_at_max"] = t[imax]
        res["t_at_min"] = t[imin]
        tw = sum(0.5 * (y[i] + y[i + 1]) * (t[i + 1] - t[i]) for i in range(len(t) - 1))
        res["time_average"] = tw / (t[-1] - t[0]) if t[-1] > t[0] else None
        approx += "; time_average uses the trapezoidal rule over recorded times"
    else:
        labels = [h for h, v in zip(header, rows[0]) if isinstance(v, str)]
        res["row_at_max"] = {h: rows[imax][header.index(h)] for h in labels}
        res["row_at_min"] = {h: rows[imin][header.index(h)] for h in labels}
    units = _units(schema, rel)
    return _envelope("stats", run, res, approx, {args.name: units.get(args.name), "t": units.get("t")})


def q_at(run, args):
    manifest, _, schema = load_run(run)
    model = manifest["model"]
    _require_time(model, "at")
    rel = MAIN_TABLE[model]
    header, rows = read_csv(run, rel)
    t = _column(header, rows, "t", rel)
    i = min(range(len(t)), key=lambda k: abs(t[k] - args.time))
    row = dict(zip(header, rows[i]))
    approx = None if t[i] == args.time else \
        f"nearest recorded time t = {t[i]!r} to the requested {args.time!r} (|dt| = {abs(t[i] - args.time):.3g}); no interpolation"
    return _envelope("at", run, {"requested_time": args.time, "t": t[i], "values": row}, approx, _units(schema, rel))


def _density_summary(run):
    header, rows = read_csv(run, "outputs/densities_final.csv")
    x = [r[0] for r in rows]
    out = {}
    for j, name in enumerate(header[1:], start=1):
        p = [r[j] for r in rows]
        trap = lambda h: sum(0.5 * (h[i] * p[i] + h[i + 1] * p[i + 1]) * (x[i + 1] - x[i]) for i in range(len(x) - 1))  # noqa: E731
        mass = trap([1.0] * len(x))
        mean = trap(x) / mass
        var = trap([(xi - mean) ** 2 for xi in x]) / mass
        k = max(range(len(p)), key=lambda i: p[i])
        # count interior local maxima above 1% of the peak (modality)
        thr = 0.01 * p[k]
        modes = [x[i] for i in range(1, len(p) - 1) if p[i] > p[i - 1] and p[i] >= p[i + 1] and p[i] > thr]
        out[name] = {"mass": mass, "mean": mean, "variance": var, "peak_density": p[k], "argmax_x": x[k],
                     "n_modes": len(modes), "mode_locations": modes[:10]}
    return out


def q_final(run, args):
    manifest, summary, schema = load_run(run)
    model = manifest["model"]
    if model == "trajectory":
        header, rows = read_csv(run, MAIN_TABLE[model])
        res = {"t": rows[-1][0], "values": dict(zip(header, rows[-1])), "densities_at_t_end": _density_summary(run)}
        approx = ("density moments recomputed from outputs/densities_final.csv with the trapezoidal rule; "
                  "modes are interior local maxima above 1% of the peak")
        return _envelope("final", run, res, approx, _units(schema, MAIN_TABLE[model]))
    if model == "projection":
        return _envelope("final", run, {"results": summary["results"]}, "static model: the final state is the full result table",
                         summary.get("units"))
    return _envelope("final", run, {"tables": {k: {"file": v["file"], "rows": len(v["rows"]), "columns": v["columns"]}
                                               for k, v in summary["tables"].items()},
                                    "figures": summary["figures"], "reported_lines": summary["reported_lines"]},
                     "experiment: lists the saved tables; read one with `query <run> table --file <name>`")


def q_table(run, args):
    manifest, _, schema = load_run(run)
    rel = _table_for(run, schema, manifest["model"], args.file)
    header, rows = read_csv(run, rel)
    for cond in args.where or []:
        if "=" not in cond:
            raise QueryError("--where", cond, "expected column=value", "e.g. --where target=elite tau=0.25")
        k, v = cond.split("=", 1)
        col = _column(header, rows, k, rel)
        vv = _num(v)
        rows = [r for r, c in zip(rows, col) if c == vv or str(c) == v]
    cols = header
    if args.columns:
        cols = [c.strip() for c in args.columns.split(",")]
        for c in cols:
            _column(header, [], c, rel)
        rows = [[r[header.index(c)] for c in cols] for r in rows]
    n = len(rows)
    approx = None
    if n > args.limit:
        rows = rows[:args.limit]
        approx = f"first {args.limit} of {n} matching rows (raise --limit for more)"
    units = _units(schema, rel)
    return _envelope("table", run, {"file": rel, "columns": cols, "n_matching": n, "rows": rows},
                     approx, {c: units.get(c) for c in cols})


def q_crossing(run, args):
    manifest, _, schema = load_run(run)
    model = manifest["model"]
    _require_time(model, "crossing")
    rel = MAIN_TABLE[model]
    header, rows = read_csv(run, rel)
    t = _column(header, rows, "t", rel)
    y = _column(header, rows, args.name, rel)
    thr = args.threshold
    up = args.direction == "up"
    hit = None
    for i in range(len(t)):
        if (up and y[i] >= thr) or (not up and y[i] <= thr):
            if i == 0:
                hit = {"t": t[0], "index": 0, "interpolated": False}
            else:
                y0, y1 = y[i - 1], y[i]
                frac = (thr - y0) / (y1 - y0) if y1 != y0 else 0.0
                hit = {"t": t[i - 1] + frac * (t[i] - t[i - 1]), "bracket": [t[i - 1], t[i]], "index": i, "interpolated": True}
            break
    units = _units(schema, rel)
    res = {"name": args.name, "threshold": thr, "direction": args.direction, "crossed": hit is not None,
           "first_crossing": hit, "value_at_end": y[-1], "t_end": t[-1]}
    return _envelope("crossing", run, res,
                     "linear interpolation between the two recorded times that bracket the first crossing" if hit and hit.get("interpolated") else None,
                     {"t": units.get("t"), args.name: units.get(args.name)})


QUERIES = {"summary": q_summary, "outputs": q_outputs, "series": q_series, "stats": q_stats,
           "at": q_at, "final": q_final, "table": q_table, "crossing": q_crossing}


# --------------------------------------------------------------------------- compare
def _flatten_summary(obj, prefix="", out=None):
    out = {} if out is None else out
    if isinstance(obj, dict):
        if "value" in obj and (isinstance(obj["value"], (int, float)) or obj["value"] is None) and set(obj) <= {"value", "unit", "note"}:
            out[prefix.rstrip(".")] = (obj["value"], obj.get("unit"))
            return out
        for k, v in obj.items():
            _flatten_summary(v, f"{prefix}{k}.", out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, dict) and "dictionary" in v and "target" in v:
                key = f"{v['dictionary']} | {v['target']}"
                for k, vv in v.items():
                    if isinstance(vv, (int, float)) and not isinstance(vv, bool):
                        out[f"{prefix}[{key}].{k}"] = (vv, None)
            elif isinstance(v, list) and prefix.endswith("rows."):
                for j, vv in enumerate(v):
                    if isinstance(vv, (int, float)) and not isinstance(vv, bool):
                        out[f"{prefix}[{i}][{j}]"] = (vv, None)
            else:
                _flatten_summary(v, f"{prefix}[{i}].", out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix.rstrip(".")] = (obj, None)
    return out


def _flatten_params(d, prefix="", out=None):
    out = {} if out is None else out
    for k, v in d.items():
        if isinstance(v, dict):
            _flatten_params(v, f"{prefix}{k}.", out)
        else:
            out[prefix + k] = v
    return out


def _resolved(run):
    import yaml
    p = os.path.join(run, "config.resolved.yaml")
    if not os.path.exists(p):
        raise QueryError("run", run, "config.resolved.yaml not found", "pass a directory written by `igr run`")
    with open(p, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def compare(runs, limit=200):
    base = runs[0]
    _, sa, _ = load_run(base)
    ca = _resolved(base)
    fa = _flatten_params({"model": ca.get("model"), "seed": ca.get("seed"), **ca.get("parameters", {})})
    va = _flatten_summary(sa)
    comparisons = []
    for other in runs[1:]:
        _, sb, _ = load_run(other)
        cb = _resolved(other)
        fb = _flatten_params({"model": cb.get("model"), "seed": cb.get("seed"), **cb.get("parameters", {})})
        pdiff = [{"parameter": k, "a": fa.get(k), "b": fb.get(k)} for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)]
        vb = _flatten_summary(sb)
        sdiff = []
        for k in [k for k in va if k in vb] + [k for k in vb if k not in va]:
            a = va.get(k, (None, None))[0]
            b = vb.get(k, (None, None))[0]
            if a == b:
                continue
            d = {"quantity": k, "unit": (va.get(k) or vb.get(k))[1], "a": a, "b": b}
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                d["difference"] = b - a
                d["relative"] = (b - a) / abs(a) if a else None
            sdiff.append(d)
        truncated = len(sdiff) > limit
        comparisons.append({"a": base, "b": other, "parameters_differing": pdiff,
                            "summary_differences": sdiff[:limit], "n_summary_differences": len(sdiff),
                            "n_summary_identical": sum(1 for k in va if k in vb and va[k][0] == vb[k][0]),
                            "truncated": truncated})
    return {"ok": True, "query": "compare", "runs": runs, "result": comparisons,
            "units": "each summary difference carries its unit; difference = b - a, relative = (b - a)/|a|",
            "approximation": None}
