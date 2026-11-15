# -*- coding: utf-8 -*-
"""Tests for the agent/CLI layer (igr_cli) and a smoke test of the minimal example.

Run with `python tests/test_cli.py` (same convention as tests/test_core.py) or pytest.
Takes about 15 s. Every expected value below was measured by running the code.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXAMPLES = os.path.join(ROOT, "examples")
sys.path.insert(0, ROOT)

# Table S5, eps = 0.4 row (baseline run of experiments/fig_s09_s10_table_s5_multistep.py)
MINIMAL_EXPECTED = {
    "surrogates.param.defect_final": 0.13515837433055547,
    "surrogates.param.l1_final": 0.11699120514655909,
    "surrogates.param.defect_integral": 0.1705046997112487,
    "gronwall_envelope_eq49": 27684.960941838923,
}
# Table S5 and Table S1 as produced by the original scripts before the layer was added.
TABLE_S5 = [
    [0.0, 1.1229990020379613e-05, 3.001367648659598e-08, 8.727053290484953e-05, 21.37428534245636],
    [0.05, 0.019019142396116892, 0.015914551279757973, 0.022140116111642738, 1758.244855566629],
    [0.1, 0.037550756465718454, 0.03154102505012239, 0.04409403910029617, 3865.8596614856096],
    [0.2, 0.07288924868155669, 0.061753099608481674, 0.08733871651880928, 9372.590304191068],
    [0.4, 0.13515837433055547, 0.11699120514655909, 0.1705046997112487, 27684.960941838923],
    [0.6, 0.18428352942415174, 0.16393303718548347, 0.24818208563402253, 61511.62367241338],
]
TABLE_S1_GAUSSIAN = [0.9549296667206199, 0.6610784375404001, 0.8096069732329497]  # linear, elite, smooth
RTOL = 1e-9


def igr(*args, stdin=None, cwd=None):
    """Run the CLI as a subprocess; return (exit code, stdout, stderr)."""
    p = subprocess.run([sys.executable, "-m", "igr_cli", *args], input=stdin, capture_output=True,
                       text=True, cwd=cwd or ROOT)
    return p.returncode, p.stdout, p.stderr


def igr_json(*args, **kw):
    code, out, err = igr(*args, "--json", **kw)
    return code, json.loads(out)     # must be exactly one JSON document


def close(a, b, rtol=RTOL):
    return abs(a - b) <= rtol * max(abs(a), abs(b), 1e-300)


def _tree_bytes(d):
    out = {}
    for root, _, files in os.walk(d):
        for f in files:
            p = os.path.join(root, f)
            out[os.path.relpath(p, d)] = open(p, "rb").read()
    return out


# --------------------------------------------------------------------------- discovery
def test_describe_json_covers_registry():
    from igr_cli import registry as R
    code, d = igr_json("describe")
    assert code == 0 and d["ok"]
    names = {p["name"] for p in d["parameters"]}
    assert names == set(R.PARAM_INDEX), names ^ set(R.PARAM_INDEX)
    for p in d["parameters"]:
        for k in ("type", "default", "unit", "description", "range", "applies_to"):
            assert k in p, (p["name"], k)
        assert p["description"], p["name"]
    assert set(d["models"]) == set(R.MODELS)
    assert set(d["outputs"]) == set(R.MODELS)
    assert "interventions" in d
    for m in R.MODELS:
        code, dm = igr_json("describe", "--model", m)
        assert code == 0 and dm["parameters"] and dm["outputs"]


def test_schema_file_matches_registry():
    code, s = igr_json("describe", "--schema")
    assert code == 0
    with open(os.path.join(ROOT, "schema", "config.schema.json")) as fh:
        committed = json.load(fh)
    assert s == committed, "regenerate: igr describe --schema --json > schema/config.schema.json"


# --------------------------------------------------------------------------- templates and validation
def test_templates_validate_for_every_model():
    from igr_cli import registry as R
    cases = [[m] for m in R.MODELS]
    cases += [["trajectory", "--fitness", k] for k in R.FITNESS_KINDS]
    cases += [["projection", "--fitness", k] for k in R.FITNESS_KINDS]
    cases += [["projection", "--family", "student_t"]]
    cases += [["experiment", "--experiment", e] for e in R.EXPERIMENT_IDS]
    for c in cases:
        code, yaml_text, _ = igr("template", *c)          # commented YAML
        assert code == 0, c
        code, d = igr_json("validate", "-", stdin=yaml_text)
        assert code == 0 and d["ok"], (c, d)
        assert d["estimate"]["runtime_s"] > 0


def test_invalid_config_reports_every_error():
    bad = (
        "schema: 1\nmodel: trajectory\nseed: 1\nparameters:\n"
        "  fitness: {kind: polynomial, polynomial: {coefficients: [0.0, 0.0, 0.2]}}\n"
        "  grid: {n: 1}\n"                         # out of range (n >= 2)
        "  time: {t_end: 3.0, n_pionts: 50}\n"     # misspelled key; t_end >= t* = 2.5 (Eq. 50)
    )
    code, d = igr_json("validate", "-", stdin=bad)
    assert code == 2 and d["ok"] is False
    fields = {e["field"]: e for e in d["errors"]}
    assert "grid.n" in fields and "range" in fields["grid.n"]["reason"]
    assert "time.n_pionts" in fields and "time.n_points" in fields["time.n_pionts"]["hint"]
    assert "time.t_end" in fields and "t*" in fields["time.t_end"]["reason"]
    for e in d["errors"]:
        assert set(e) == {"field", "value", "reason", "hint"}


def test_clean_channels_on_failure():
    for args in (["run", "/nonexistent.yaml", "--out", "/tmp/never"], ["query", "/nonexistent", "summary"],
                 ["template", "nosuchmodel"], ["describe", "--param", "nope"], ["run"]):
        code, out, _ = igr(*args, "--json")
        doc = json.loads(out)
        assert code == 2 and doc["ok"] is False and doc["errors"], args


# --------------------------------------------------------------------------- smoke test + reproducibility
def _flat(summary):
    from igr_cli.queries import _flatten_summary
    return {k: v[0] for k, v in _flatten_summary(summary).items()}


def test_smoke_minimal_example():
    tmp = tempfile.mkdtemp()
    try:
        cfg = os.path.join(EXAMPLES, "minimal_multistep.yaml")
        code, v = igr_json("validate", cfg)
        assert code == 0 and v["ok"]
        run = os.path.join(tmp, "run")
        code, r = igr_json("run", cfg, "--out", run, "--quiet")
        assert code == 0 and r["status"] == "completed"
        code, q = igr_json("query", run, "summary")
        assert code == 0
        flat = _flat(q["result"])
        for k, want in MINIMAL_EXPECTED.items():
            assert close(flat[k], want), (k, flat[k], want)
        for f in ("config.resolved.yaml", "manifest.json", "summary.json", "schema.json", "log.txt",
                  "outputs/trajectory.csv", "outputs/densities_final.csv"):
            assert os.path.exists(os.path.join(run, f)), f
        # the resolved config alone reproduces the run, from another location
        moved = os.path.join(tmp, "moved")
        shutil.copytree(run, moved)
        code, _ = igr_json("run", os.path.join(moved, "config.resolved.yaml"), "--out", os.path.join(tmp, "rerun"), "--quiet")
        assert code == 0
        assert _tree_bytes(os.path.join(run, "outputs")) == _tree_bytes(os.path.join(tmp, "rerun", "outputs"))
        # every query answers in compact JSON
        for qargs in (["summary"], ["outputs"], ["series", "l1_param", "--points", "20"], ["stats", "defect_param"],
                      ["at", "0.5"], ["final"], ["table", "--limit", "5"], ["crossing", "l1_param", "0.05"]):
            code, out, _ = igr("query", run, *qargs, "--json")
            assert code == 0 and json.loads(out)["ok"] and len(out) < 20000, qargs
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_same_seed_gives_identical_outputs():
    tmp = tempfile.mkdtemp()
    try:
        cfg = os.path.join(EXAMPLES, "projection_multimodal_rbf.yaml")
        for name in ("a", "b"):
            code, _ = igr_json("run", cfg, "--out", os.path.join(tmp, name), "--quiet")
            assert code == 0
        a, b = (os.path.join(tmp, n) for n in "ab")
        assert _tree_bytes(os.path.join(a, "outputs")) == _tree_bytes(os.path.join(b, "outputs"))
        assert open(os.path.join(a, "summary.json"), "rb").read() == open(os.path.join(b, "summary.json"), "rb").read()
        code, c = igr_json("compare", a, b)
        assert code == 0 and c["result"][0]["n_summary_differences"] == 0
        # refusing to overwrite
        code, d = igr_json("run", cfg, "--out", a)
        assert code == 2 and d["errors"][0]["field"] == "--out"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- old entry points
def _run_script(module):
    """Run an experiment script's run() exactly as run_all.py does, with its outputs
    redirected to a temporary directory instead of the repository's figures/ and tables/."""
    import importlib
    import io
    from contextlib import redirect_stdout
    sys.path.insert(0, os.path.join(ROOT, "experiments"))
    import igr.plotting as P
    tmp = tempfile.mkdtemp()
    old = (P.FIG_DIR, P.TAB_DIR)
    P.FIG_DIR = P.TAB_DIR = tmp
    try:
        with redirect_stdout(io.StringIO()):
            importlib.import_module(module).run()
        return tmp
    finally:
        P.FIG_DIR, P.TAB_DIR = old


def _csv(path):
    import csv
    with open(path) as fh:
        rows = list(csv.reader(fh))
    return rows[0], rows[1:]


def test_old_entry_points_reproduce_baseline():
    tmp = _run_script("fig_s09_s10_table_s5_multistep")
    try:
        _, rows = _csv(os.path.join(tmp, "tableS5_multistep.csv"))
        for got, want in zip(rows, TABLE_S5):
            assert all(close(float(g), w) for g, w in zip(got, want)), (got, want)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    tmp = _run_script("fig_s05_table_s1_linear_shaping")
    try:
        _, rows = _csv(os.path.join(tmp, "tableS1_linear_shaping.csv"))
        gauss = [r for r in rows if r[0] == "Gaussian scores"][0]
        assert all(close(float(g), w) for g, w in zip(gauss[1:], TABLE_S1_GAUSSIAN)), gauss
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_projection_model_matches_table_s1():
    tmp = tempfile.mkdtemp()
    try:
        run = os.path.join(tmp, "p")
        code, _ = igr_json("run", os.path.join(EXAMPLES, "projection_linear_shaping.yaml"), "--out", run, "--quiet")
        assert code == 0
        code, t = igr_json("query", run, "table", "--where", "dictionary=gaussian", "--columns", "target,r2")
        got = {r[0]: r[1] for r in t["result"]["rows"]}
        want = dict(zip(["linear", "elite tau=0.25", "smooth_elite tau=0.25 h=0.05"], TABLE_S1_GAUSSIAN))
        for k, w in want.items():
            assert close(got[k], w), (k, got[k], w)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    import time
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            t0 = time.time()
            fn()
            print(f"  ok  {name}  ({time.time() - t0:.1f}s)")
    print("all CLI checks passed")
