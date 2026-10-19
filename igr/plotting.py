# -*- coding: utf-8 -*-
"""Shared matplotlib style and figure/table output paths."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(HERE, "figures")
TAB_DIR = os.path.join(HERE, "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

COLORS = {
    "true": "#111111", "param": "#1f6f8b", "proj": "#d08020",
    "accent": "#3a7d44", "muted": "#888888", "bins": "#b0451f",
}


def setup():
    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 140,
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.6,
    })


def save(fig, name):
    """Save a figure into figures/ by base name (without extension)."""
    path = os.path.join(FIG_DIR, name + ".png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_table(df, name, float_format="%.4g"):
    """Save a dataframe as CSV and a Markdown table into tables/."""
    import pandas as pd  # noqa
    csv = os.path.join(TAB_DIR, name + ".csv")
    md = os.path.join(TAB_DIR, name + ".md")
    df.to_csv(csv, index=False)
    with open(md, "w") as fh:
        fh.write(df.to_markdown(index=False, floatfmt=".4g"))
    return csv, md
