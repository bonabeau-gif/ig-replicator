# -*- coding: utf-8 -*-
"""Self-describing run directories: manifest, resolved config, summary, schema, log."""
import datetime as _dt
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time

RUN_FILES = ("config.resolved.yaml", "manifest.json", "summary.json", "schema.json", "log.txt")
DEPENDENCIES = ("igr-invariance", "numpy", "scipy", "pandas", "matplotlib", "scikit-learn", "tabulate", "PyYAML")


class RunDirError(Exception):
    pass


def clean(obj):
    """Make an object strict-JSON safe: non-finite floats become None, numpy scalars become Python."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            obj = obj.item()
        except Exception:
            pass
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_json(path, obj, indent=2):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(clean(obj), fh, indent=indent, allow_nan=False)
        fh.write("\n")


def prepare(out_dir, force=False, kind="run"):
    """Create an empty output directory. Refuse to reuse a non-empty one unless --force
    and it is recognizably a previous run or sweep directory."""
    if os.path.exists(out_dir) and not os.path.isdir(out_dir):
        raise RunDirError(f"{out_dir} exists and is not a directory")
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        if not force:
            raise RunDirError(f"output directory {out_dir} exists and is not empty; choose a new --out or pass --force")
        if not (os.path.exists(os.path.join(out_dir, "manifest.json")) or os.path.exists(os.path.join(out_dir, "index.json"))):
            raise RunDirError(f"--force refused: {out_dir} does not look like a run or sweep directory "
                              "(no manifest.json or index.json); remove it yourself")
        for name in os.listdir(out_dir):
            p = os.path.join(out_dir, name)
            shutil.rmtree(p) if os.path.isdir(p) and not os.path.islink(p) else os.remove(p)
    os.makedirs(out_dir, exist_ok=True)


def code_version():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    info = {"package_root": root, "git_commit": None, "git_dirty": None}
    try:
        c = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        if c.returncode == 0:
            info["git_commit"] = c.stdout.strip()
            d = subprocess.run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10)
            info["git_dirty"] = bool(d.stdout.strip()) if d.returncode == 0 else None
    except Exception:
        pass
    return info


def dependency_versions():
    from importlib import metadata
    out = {}
    for name in DEPENDENCIES:
        try:
            out[name] = metadata.version(name)
        except Exception:
            out[name] = None
    return out


def peak_rss_mb():
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(r / (1024 * 1024) if sys.platform == "darwin" else r / 1024, 1)
    except Exception:
        return None


def now():
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


class Logger:
    """Writes timestamped lines to <run>/log.txt and mirrors them to stderr."""

    def __init__(self, path, quiet=False):
        self.fh = open(path, "a", encoding="utf-8")
        self.quiet = quiet

    def __call__(self, msg):
        line = f"[{now()}] {msg}"
        self.fh.write(line + "\n")
        self.fh.flush()
        if not self.quiet:
            print(msg, file=sys.stderr)

    def close(self):
        self.fh.close()


def execute(resolved, out_dir, argv, estimate=None, force=False, quiet=False, seed_note=None):
    """Run one resolved configuration into out_dir. Returns (summary, manifest)."""
    from . import models
    from .config import dump_yaml, flatten_params
    prepare(out_dir, force=force)
    with open(os.path.join(out_dir, "config.resolved.yaml"), "w", encoding="utf-8") as fh:
        fh.write("# Fully resolved configuration: every default explicit. Rerun with\n"
                 "#   igr run config.resolved.yaml --out <new-dir>\n")
        fh.write(dump_yaml(resolved))
    manifest = {
        "status": "running", "model": resolved["model"], "command": list(argv),
        "cwd": os.getcwd(), "started": now(), "ended": None, "duration_s": None,
        "seed": resolved.get("seed"), "seed_note": seed_note,
        "code_version": code_version(), "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "dependencies": dependency_versions(),
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "estimate": estimate, "peak_rss_mb": None, "error": None,
    }
    mpath = os.path.join(out_dir, "manifest.json")
    write_json(mpath, manifest)
    log = Logger(os.path.join(out_dir, "log.txt"), quiet=quiet)
    log(f"run {resolved['model']} -> {out_dir}")
    if seed_note:
        log(seed_note)
    t0 = time.time()
    flat = flatten_params(resolved)
    try:
        summary = models.RUNNERS[resolved["model"]](flat, out_dir, log)
        write_json(os.path.join(out_dir, "summary.json"), summary)
        write_json(os.path.join(out_dir, "schema.json"), models.schema_for_run(resolved["model"], flat, out_dir))
        manifest["status"] = "completed"
    except BaseException as exc:  # record and re-raise
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        summary = None
        log(f"FAILED: {type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["ended"] = now()
        manifest["duration_s"] = round(time.time() - t0, 3)
        manifest["peak_rss_mb"] = peak_rss_mb()
        write_json(mpath, manifest)
        log(f"status {manifest['status']} in {manifest['duration_s']} s")
        log.close()
    return summary, manifest
