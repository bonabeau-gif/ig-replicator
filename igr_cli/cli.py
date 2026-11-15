# -*- coding: utf-8 -*-
"""`igr`: command-line interface for agents and people.

Subcommands: describe, template, validate, run, query, compare, sweep.
With --json, standard output carries exactly one JSON document; logs go to stderr.
Exit codes: 0 success, 1 runtime failure, 2 invalid usage or invalid configuration.
"""
import argparse
import json
import os
import sys

EXAMPLES = {
    "describe": "igr describe --model trajectory --json",
    "template": "igr template trajectory --fitness perturbed_quadratic > my.yaml",
    "validate": "igr validate examples/minimal_multistep.yaml --json",
    "run": "igr run examples/minimal_multistep.yaml --out runs/minimal --json",
    "query": "igr query runs/minimal summary --json",
    "compare": "igr compare runs/a runs/b --json",
    "sweep": "igr sweep examples/sweep_eps.yaml --out runs/sweep_eps --json",
}


def _want_json(argv):
    return "--json" in argv


def emit(doc, as_json, text=None):
    from .rundir import clean
    if as_json:
        sys.stdout.write(json.dumps(clean(doc), allow_nan=False) + "\n")
    elif text is not None:
        sys.stdout.write(text if text.endswith("\n") else text + "\n")
    else:
        sys.stdout.write(json.dumps(clean(doc), indent=2, allow_nan=False) + "\n")


def fail(errors, as_json, code):
    if as_json:
        emit({"ok": False, "errors": errors}, True)
    else:
        for e in errors:
            hint = f"  (hint: {e['hint']})" if e.get("hint") else ""
            print(f"error: {e['field']} = {e['value']!r}: {e['reason']}{hint}", file=sys.stderr)
    return code


class Parser(argparse.ArgumentParser):
    """argparse that reports usage errors as structured JSON when --json is present."""

    def error(self, message):
        if _want_json(sys.argv):
            emit({"ok": False, "errors": [{"field": "<arguments>", "value": " ".join(sys.argv[1:]),
                                           "reason": message, "hint": f"see `{self.prog} --help`"}]}, True)
            sys.exit(2)
        super().error(message)


# --------------------------------------------------------------------------- describe
def cmd_describe(args):
    from . import registry as R
    from .config import json_schema, _range_text
    if args.schema:
        emit(json_schema(), True)
        return 0
    if args.param:
        hits = [p for p in R.PARAMETERS if p["name"] == args.param or p["name"].startswith(args.param + ".")]
        if not hits:
            import difflib
            close = difflib.get_close_matches(args.param, list(R.PARAM_INDEX), n=3)
            return fail([{"field": "--param", "value": args.param, "reason": "unknown parameter",
                          "hint": f"did you mean {close}?" if close else "run `igr describe`"}], args.json, 2)
        emit({"ok": True, "parameters": hits}, args.json)
        return 0
    if args.model:
        if args.model not in R.MODELS:
            return fail([{"field": "--model", "value": args.model, "reason": "unknown model",
                          "hint": f"options: {list(R.MODELS)}"}], args.json, 2)
        doc = {"ok": True, "model": args.model, **R.MODELS[args.model],
               "parameters": R.params_for(args.model), "outputs": R.OUTPUTS[args.model],
               "interventions": R.INTERVENTIONS, "interventions_note": R.INTERVENTIONS_NOTE}
        if args.model in R.FITNESS_MODELS:
            doc["fitness_kinds"] = {k: {kk: v[kk] for kk in ("formula", "description", "source")} for k, v in R.FITNESS_KINDS.items()}
        if args.model == "experiment":
            doc["experiments"] = [{"id": e[0], "script": f"experiments/{e[1]}.py", "produces": e[2], "description": e[3]} for e in R.EXPERIMENTS]
        if args.json:
            emit(doc, True)
            return 0
        lines = [f"model {args.model}: {R.MODELS[args.model]['description']}", "", "parameters:"]
        for p in R.params_for(args.model):
            d = "REQUIRED" if p["default"] == R.REQUIRED else json.dumps(p["default"])
            w = f" [when {p['when']}]" if p.get("when") else ""
            lines.append(f"  {p['name']} ({p['type']}, default {d}, unit {p['unit']}, range {_range_text(p['range'])}){w}")
            lines.append(f"      {p['description']}")
        lines.append("outputs:")
        for o in R.OUTPUTS[args.model]:
            lines.append(f"  {o['name']} [{o['unit']}] {o['shape']} in {o['file']}: {o['description']}")
        lines.append(f"interventions: none. {R.INTERVENTIONS_NOTE}")
        emit(None, False, "\n".join(lines))
        return 0
    if args.outputs:
        emit({"ok": True, "outputs": R.OUTPUTS}, args.json)
        return 0
    doc = {"ok": True, "package": "igr-invariance", "schema_version": R.SCHEMA_VERSION,
           "what": ("Invariance defect of continuous-trait parametric replicator dynamics: exact replicator "
                    "tilt vs Gaussian/Student-t surrogates, raw and rank-shaped defects, score dictionaries."),
           "models": {k: v["description"] for k, v in R.MODELS.items()},
           "top_level_keys": R.TOP_LEVEL,
           "parameters": R.PARAMETERS,
           "fitness_kinds": {k: {kk: v[kk] for kk in ("formula", "description", "source")} for k, v in R.FITNESS_KINDS.items()},
           "experiments": [{"id": e[0], "script": f"experiments/{e[1]}.py", "produces": e[2], "description": e[3]} for e in R.EXPERIMENTS],
           "outputs": R.OUTPUTS, "interventions": R.INTERVENTIONS, "interventions_note": R.INTERVENTIONS_NOTE,
           "fixed_constants": R.FIXED,
           "filters": "describe --model <m> | --param <name> | --outputs | --schema"}
    if args.json:
        emit(doc, True)
        return 0
    lines = ["igr: " + doc["what"], "", "models:"]
    for k, v in R.MODELS.items():
        lines.append(f"  {k:11s} {v['description'][:150]}...")
        lines.append(f"              {len(R.params_for(k))} parameters; outputs: {', '.join(sorted({o['file'] for o in R.OUTPUTS[k]}))}")
    lines += ["", "fitness kinds: " + ", ".join(R.FITNESS_KINDS),
              "experiments: " + ", ".join(R.EXPERIMENT_IDS),
              "interventions: none (fixed fitness per run)",
              "", "detail: igr describe --model <m> | --param <name> | --outputs | --schema | --json"]
    emit(None, False, "\n".join(lines))
    return 0


# --------------------------------------------------------------------------- template
def cmd_template(args):
    from . import registry as R
    from .config import template_config, to_commented_yaml
    errs = []
    if args.model not in R.MODELS:
        errs.append({"field": "model", "value": args.model, "reason": "unknown model", "hint": f"options: {list(R.MODELS)}"})
    if args.fitness and args.fitness not in R.FITNESS_KINDS:
        errs.append({"field": "--fitness", "value": args.fitness, "reason": "unknown fitness kind", "hint": f"options: {list(R.FITNESS_KINDS)}"})
    if args.family and args.family not in ("gaussian", "student_t"):
        errs.append({"field": "--family", "value": args.family, "reason": "unknown family", "hint": "options: ['gaussian', 'student_t']"})
    if args.experiment and args.experiment not in R.EXPERIMENT_IDS:
        errs.append({"field": "--experiment", "value": args.experiment, "reason": "unknown experiment id", "hint": f"options: {R.EXPERIMENT_IDS}"})
    if errs:
        return fail(errs, args.json, 2)
    data, comments = template_config(args.model, args.fitness, args.family, args.experiment)
    if args.json:
        emit(data, True)
    else:
        emit(None, False, to_commented_yaml(data, comments, header=[
            f"igr configuration template for model '{args.model}' (generated from the registry).",
            "Check with: igr validate <this file>   Run with: igr run <this file> --out <dir>"]))
    return 0


# --------------------------------------------------------------------------- validate
def _load_validate(path):
    from .config import load, validate
    data, base = load(path)
    return validate(data, base)


def cmd_validate(args):
    from .config import ConfigError, estimate, sweep_points, validate
    try:
        resolved, warnings = _load_validate(args.config)
        doc = {"ok": True, "config": args.config, "model": resolved["model"], "warnings": warnings}
        if resolved.get("sweep"):
            pts = sweep_points(resolved)
            ests = []
            for _, cfg in pts:
                r, _ = validate(cfg)
                ests.append(estimate(r))
            doc["sweep_points"] = len(pts)
            doc["estimate"] = {"runtime_s": round(sum(e["runtime_s"] for e in ests), 2),
                               "memory_mb": max(e["memory_mb"] for e in ests),
                               "output_mb": round(sum(e["output_mb"] for e in ests), 3),
                               "method": "sum over sweep points of: " + ests[0]["method"], "machine_dependent": True}
        else:
            doc["estimate"] = estimate(resolved)
        if resolved.get("seed") is None:
            warnings.append("seed omitted: `run` will choose one and record it")
        doc["resolved"] = resolved
    except ConfigError as exc:
        return fail(exc.errors, args.json, 2)
    if args.json:
        emit(doc, True)
    else:
        e = doc["estimate"]
        msg = [f"ok: {args.config} is a valid '{doc['model']}' configuration",
               f"estimate: ~{e['runtime_s']} s, ~{e['memory_mb']:.0f} MB memory, ~{e['output_mb']} MB output ({e['method']})"]
        msg += [f"warning: {w}" for w in warnings]
        emit(None, False, "\n".join(msg))
    return 0


# --------------------------------------------------------------------------- run / sweep
def _seed(resolved):
    if resolved.get("seed") is None:
        import secrets
        resolved["seed"] = secrets.randbelow(2 ** 31)
        note = f"seed omitted; chose seed {resolved['seed']} and recorded it in config.resolved.yaml"
        print(note, file=sys.stderr)
        return note
    return None


def cmd_run(args):
    from .config import ConfigError, estimate
    from .rundir import RunDirError, execute
    try:
        resolved, warnings = _load_validate(args.config)
    except ConfigError as exc:
        return fail(exc.errors, args.json, 2)
    if resolved.get("sweep"):
        return fail([{"field": "sweep", "value": resolved["sweep"], "reason": "configuration has a sweep block",
                      "hint": "use `igr sweep <config> --out <dir>`, or remove the sweep block"}], args.json, 2)
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    note = _seed(resolved)
    try:
        summary, manifest = execute(resolved, args.out, sys.argv, estimate=estimate(resolved),
                                    force=args.force, quiet=args.quiet, seed_note=note)
    except RunDirError as exc:
        return fail([{"field": "--out", "value": args.out, "reason": str(exc), "hint": "choose a new directory or pass --force"}], args.json, 2)
    except Exception as exc:
        return fail([{"field": "<run>", "value": args.out, "reason": f"{type(exc).__name__}: {exc}",
                      "hint": "see manifest.json and log.txt in the run directory"}], args.json, 1)
    doc = {"ok": True, "run": args.out, "status": manifest["status"], "duration_s": manifest["duration_s"],
           "seed": resolved["seed"], "summary": summary}
    emit(doc, args.json, None if args.json else
         f"completed in {manifest['duration_s']} s -> {args.out}\nnext: igr query {args.out} summary")
    return 0


def cmd_sweep(args):
    from .config import ConfigError, estimate, sweep_points, validate
    from .rundir import RunDirError, execute, prepare, write_json
    from .queries import _flatten_summary
    try:
        resolved, warnings = _load_validate(args.config)
    except ConfigError as exc:
        return fail(exc.errors, args.json, 2)
    if not resolved.get("sweep"):
        return fail([{"field": "sweep", "value": None, "reason": "configuration has no sweep block",
                      "hint": "add sweep: {dotted.parameter: [v1, v2, ...]}"}], args.json, 2)
    note = _seed(resolved)
    points = sweep_points(resolved)
    errors, checked = [], []
    for i, (vals, cfg) in enumerate(points):
        try:
            r, _ = validate(cfg)
            checked.append((vals, r))
        except ConfigError as exc:
            errors += [dict(e, field=f"sweep point {i} {vals}: {e['field']}") for e in exc.errors]
    if errors:
        return fail(errors, args.json, 2)
    try:
        prepare(args.out, force=args.force)
    except RunDirError as exc:
        return fail([{"field": "--out", "value": args.out, "reason": str(exc), "hint": "choose a new directory or pass --force"}], args.json, 2)
    index = {"sweep": resolved["sweep"], "model": resolved["model"], "seed": resolved["seed"], "points": []}
    status = 0
    for i, (vals, r) in enumerate(checked):
        d = os.path.join(args.out, f"point_{i:03d}")
        entry = {"index": i, "parameters": vals, "dir": os.path.basename(d)}
        try:
            summary, manifest = execute(r, d, sys.argv, estimate=estimate(r), quiet=args.quiet, seed_note=note)
            entry["status"] = "completed"
            entry["summary"] = {k: v[0] for k, v in _flatten_summary(summary).items()}
            if len(json.dumps(entry["summary"])) > 4000:
                entry["summary"] = dict(list(entry["summary"].items())[:60])
                entry["summary_truncated"] = True
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            status = 1
        index["points"].append(entry)
        write_json(os.path.join(args.out, "index.json"), index)
    doc = {"ok": status == 0, "sweep": args.out, "points": len(checked), "index": os.path.join(args.out, "index.json"),
           "failed": [p["index"] for p in index["points"] if p["status"] != "completed"]}
    if status:
        doc["errors"] = [{"field": f"sweep point {p['index']}", "value": p["parameters"], "reason": p.get("error"), "hint": "see that point's manifest.json"}
                         for p in index["points"] if p["status"] != "completed"]
    emit(doc, args.json, None if args.json else f"sweep of {len(checked)} points -> {args.out}/index.json")
    return status


# --------------------------------------------------------------------------- query / compare
def cmd_query(args):
    from .queries import QUERIES, QueryError
    try:
        doc = QUERIES[args.what](args.run, args)
    except QueryError as exc:
        return fail([exc.error], args.json, 2)
    emit(doc, args.json)
    return 0


def cmd_compare(args):
    from .queries import QueryError, compare
    if len(args.runs) < 2:
        return fail([{"field": "runs", "value": args.runs, "reason": "need at least two run directories", "hint": EXAMPLES["compare"]}], args.json, 2)
    try:
        doc = compare(args.runs, limit=args.limit)
    except QueryError as exc:
        return fail([exc.error], args.json, 2)
    emit(doc, args.json)
    return 0


# --------------------------------------------------------------------------- parser
def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="print exactly one JSON document on stdout")
    p = Parser(prog="igr", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
               epilog="Start with: igr describe. Agent guide: AGENTS.md.")
    p.add_argument("--version", action="store_true", help="print the package version")
    sub = p.add_subparsers(dest="cmd", parser_class=Parser)

    def add(name, help_, **kw):
        return sub.add_parser(name, help=help_, description=help_, parents=[common],
                              epilog=f"example: {EXAMPLES[name]}", **kw)

    d = add("describe", "Report models, parameters, outputs, and interventions from the registry.")
    d.add_argument("--model", help="detail for one model")
    d.add_argument("--param", help="detail for one parameter (or a dotted prefix, e.g. fitness.sinusoid)")
    d.add_argument("--outputs", action="store_true", help="list every recorded output")
    d.add_argument("--schema", action="store_true", help="print the configuration JSON Schema")
    d.set_defaults(fn=cmd_describe)

    t = add("template", "Print a complete, valid, commented configuration for a model.")
    t.add_argument("model", help="trajectory | projection | experiment")
    t.add_argument("--fitness", help="fitness kind to fill in (trajectory, projection)")
    t.add_argument("--family", help="state family for projection: gaussian | student_t")
    t.add_argument("--experiment", help="experiment id for the experiment model (e.g. S05)")
    t.set_defaults(fn=cmd_template)

    v = add("validate", "Check a configuration without running it; report every problem and a cost estimate.")
    v.add_argument("config", help="YAML or JSON file, or - for stdin")
    v.set_defaults(fn=cmd_validate)

    r = add("run", "Execute one run and write a self-describing run directory.")
    r.add_argument("config")
    r.add_argument("--out", required=True, help="run directory to create (must be new or empty)")
    r.add_argument("--force", action="store_true", help="replace an existing run directory")
    r.add_argument("--quiet", action="store_true", help="do not mirror log lines to stderr")
    r.set_defaults(fn=cmd_run)

    s = add("sweep", "Run the cartesian product in the config's sweep block, one run directory per point, plus index.json.")
    s.add_argument("config")
    s.add_argument("--out", required=True)
    s.add_argument("--force", action="store_true")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(fn=cmd_sweep)

    qp = add("query", "Answer targeted questions about a finished run (reads files only).")
    qp.add_argument("run", help="run directory")
    qs = qp.add_subparsers(dest="what", parser_class=Parser, required=True)

    def qadd(name, help_, example):
        return qs.add_parser(name, help=help_, description=help_, parents=[common], epilog=f"example: {example}")
    qadd("summary", "Contents of summary.json.", "igr query runs/minimal summary --json")
    qadd("outputs", "What was recorded: files, columns, units, sizes.", "igr query runs/minimal outputs --json")
    x = qadd("series", "One time series, downsampled to at most --points points.", "igr query runs/minimal series l1_param --points 20 --json")
    x.add_argument("name")
    x.add_argument("--from", dest="t_from", type=float)
    x.add_argument("--to", dest="t_to", type=float)
    x.add_argument("--points", type=int, default=100)
    x = qadd("stats", "Min, max, mean, final value, and time (or row) of the max of one column.", "igr query runs/minimal stats defect_param --json")
    x.add_argument("name")
    x.add_argument("--file", help="table to use when the run has several (experiment model)")
    x = qadd("at", "Recorded state at the time nearest to TIME.", "igr query runs/minimal at 0.5 --json")
    x.add_argument("time", type=float)
    qadd("final", "Final state (trajectory: last row and density moments/modes at t_end).", "igr query runs/minimal final --json")
    x = qadd("table", "Rows of an output table, filtered with --where col=value.", "igr query runs/proj table --where 'target=elite tau=0.25' --json")
    x.add_argument("--file")
    x.add_argument("--where", action="append", help="column=value (repeatable)")
    x.add_argument("--columns", help="comma-separated columns to keep")
    x.add_argument("--limit", type=int, default=50)
    x = qadd("crossing", "First time a series crosses a threshold (e.g. when L1 error exceeds 0.1).", "igr query runs/minimal crossing l1_param 0.05 --json")
    x.add_argument("name")
    x.add_argument("threshold", type=float)
    x.add_argument("--direction", choices=["up", "down"], default="up")
    qp.set_defaults(fn=cmd_query)

    c = add("compare", "Parameters that differ and differences in every summary quantity between runs.")
    c.add_argument("runs", nargs="+")
    c.add_argument("--limit", type=int, default=200, help="max summary differences listed per pair")
    c.set_defaults(fn=cmd_compare)
    return p


def main(argv=None):
    sys.dont_write_bytecode = True
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args = parser.parse_args(argv)
    args.json = "--json" in argv        # robust to flag position (argparse subparser defaults)
    if getattr(args, "version", False) and not args.cmd:
        from . import __version__
        print(__version__)
        return 0
    if not args.cmd:
        parser.print_help(sys.stderr)
        return 2
    try:
        return args.fn(args)
    except KeyboardInterrupt:
        return 1
    except Exception as exc:  # unexpected: still a structured failure
        return fail([{"field": "<internal>", "value": args.cmd, "reason": f"{type(exc).__name__}: {exc}",
                      "hint": "please report this"}], getattr(args, "json", False), 1)


if __name__ == "__main__":
    sys.exit(main())
