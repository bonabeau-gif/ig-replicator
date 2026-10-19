#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regenerate every supplementary figure and table.

Usage:
    python run_all.py              # full run
    python run_all.py --quick      # smaller sample sizes for the sampling experiments
    python run_all.py S05 S07      # run only the listed experiment ids

Figures land in figures/, tables in tables/ (CSV and Markdown).
"""
import sys
import os
import time
import importlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

# (id, module, produces)
EXPERIMENTS = [
    ("S01", "fig_s01_s02_raw_exactness", "Fig S1, S2"),
    ("S03", "fig_s03_defect_departure", "Fig S3"),
    ("S04", "fig_s04_finite_time", "Fig S4"),
    ("S05", "fig_s05_table_s1_linear_shaping", "Fig S5, Table S1"),
    ("S06", "fig_s06_table_s2_quadratic_shaping", "Fig S6, Table S2"),
    ("S07", "fig_s07_table_s3_multimodal", "Fig S7, Table S3"),
    ("S08", "fig_s08_table_s4_defect_predicts", "Fig S8, Table S4"),
    ("S09", "fig_s09_s10_table_s5_multistep", "Fig S9, S10, Table S5"),
    ("S11", "fig_s11_s12_table_s6_finite_sample", "Fig S11, S12, Table S6"),
    ("S13", "fig_s13_table_s7_two_d", "Fig S13, Table S7"),
    ("S14", "fig_s14_s15_s16_core", "Fig S14, S15, S16"),
    ("S17", "fig_s17_student_t", "Fig S17"),
    ("S18", "table_s8_dictionary_cost", "Table S8"),
    ("S19", "fig_s18_s19_envelopes", "Fig S18, S19"),
    ("CHK", "checks_s12_s16_identities", "Checks S12-S16"),
]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--quick" in sys.argv:
        os.environ.setdefault("IGR_NSAMPLES", "20")
        os.environ.setdefault("IGR_NMC", "15000")
    selected = [e for e in EXPERIMENTS if (not args or e[0] in args)]
    print(f"Running {len(selected)} experiment group(s)\n" + "=" * 60)
    for eid, mod, produces in selected:
        t0 = time.time()
        print(f"[{eid}] {mod}  ->  {produces}")
        try:
            m = importlib.import_module(mod)
            m.run()
        except Exception as exc:  # keep going, report at the end
            print(f"  FAILED: {exc!r}")
        print(f"  done in {time.time() - t0:.1f}s\n")
    print("=" * 60 + "\nAll requested experiments finished.")


if __name__ == "__main__":
    main()
