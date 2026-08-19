"""
Before-vs-after PAAR comparison, all 8 cases: shows why the PAAR method
(identifiable-subspace reparameterization + band-validity screening +
D-optimal adaptive resampling, report/main.tex) mattered.

Data: results_dir("report")/before_after_paar/before_after_paar.csv, itself
a verified transcription (see before_after_paar/verify_before_after_paar.py)
of two already-published tables (report/main.tex's baseline table and
report/tables/final_results_2pct.tex) -- no new simulations, no new fitting.

Every visual knob lives in the PlotConfig dataclass below.

Run: python3.9 plot_before_after_paar.py [--csv PATH] [--output-dir DIR]
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # validation/
import config

import csv
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class PlotConfig:
    """Every adjustable knob for this figure, in one place."""
    figsize: tuple = (10.0, 6.0)
    dpi: int = 200
    bar_width: float = 0.38
    before_color: str = "#d7301f"
    after_color: str = "#2c7fb8"
    criterion_color: str = "black"
    criterion_linestyle: str = ":"
    criterion_linewidth: float = 1.3
    edge_linewidth: float = 0.6
    value_label_fontsize: float = 8.5
    title_fontsize: float = 13.0
    subtitle_fontsize: float = 10.0
    axis_label_fontsize: float = 10.5
    tick_label_fontsize: float = 10.0
    legend_fontsize: float = 9.5
    case_label_fontsize: float = 10.0
    yscale: str = "log"          # "log" or "linear" -- log recommended: Case 6 spans 0.9%-59%
    ymin: float = 0.3
    ymax: float = 100.0
    grid_alpha: float = 0.25
    title: str = "Why PAAR mattered: worst-case prediction error, before vs. after"
    subtitle: str = ("Same 8 combustion-mechanism cases, before any of PAAR's fixes vs. after "
                      "the full method (identifiable reparameterization + band-validity "
                      "screening + D-optimal resampling where needed)")
    left_bar_label: str = "BEFORE PAAR (original pipeline)"
    right_bar_label: str = "AFTER PAAR (this project's final result)"
    metric_label: str = "before_test_pct"   # which "before" column to plot: test error is the
                                             # fair comparison against after's hold-out error
    after_metric_label: str = "after_holdout_pct"
    output_formats: tuple = ("pdf", "png")


def load_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k: (int(v) if k == "case" else float(v)) for k, v in r.items()})
    return sorted(rows, key=lambda r: r["case"])


def make_figure(rows, cfg: PlotConfig, out_dir, tag=""):
    os.makedirs(out_dir, exist_ok=True)
    cases = [r["case"] for r in rows]
    before = [r[cfg.metric_label] for r in rows]
    after = [r[cfg.after_metric_label] for r in rows]
    criterion = rows[0]["criterion_pct"]

    x = np.arange(len(cases))
    fig, ax = plt.subplots(figsize=cfg.figsize)

    b1 = ax.bar(x - cfg.bar_width / 2, before, width=cfg.bar_width,
                color=cfg.before_color, edgecolor="black", linewidth=cfg.edge_linewidth,
                label=cfg.left_bar_label, zorder=3)
    b2 = ax.bar(x + cfg.bar_width / 2, after, width=cfg.bar_width,
                color=cfg.after_color, edgecolor="black", linewidth=cfg.edge_linewidth,
                label=cfg.right_bar_label, zorder=3)

    ax.axhline(criterion, color=cfg.criterion_color, linestyle=cfg.criterion_linestyle,
               linewidth=cfg.criterion_linewidth, zorder=2,
               label=f"{criterion:g}% acceptable-error threshold")

    for rect, val in zip(b1, before):
        ax.annotate(f"{val:.2f}", (rect.get_x() + rect.get_width() / 2, val),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=cfg.value_label_fontsize, color=cfg.before_color, fontweight="bold")
    for rect, val in zip(b2, after):
        ax.annotate(f"{val:.2f}", (rect.get_x() + rect.get_width() / 2, val),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=cfg.value_label_fontsize, color=cfg.after_color, fontweight="bold")

    if cfg.yscale == "log":
        ax.set_yscale("log")
        ax.set_ylim(cfg.ymin, cfg.ymax)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Case {c}" for c in cases], fontsize=cfg.case_label_fontsize)
    ax.set_ylabel("Worst-case prediction error (%) — log scale, lower is better",
                   fontsize=cfg.axis_label_fontsize)
    ax.tick_params(axis="y", labelsize=cfg.tick_label_fontsize)
    ax.grid(axis="y", which="both", alpha=cfg.grid_alpha, zorder=0)
    ax.set_axisbelow(True)

    fig.suptitle(cfg.title, fontsize=cfg.title_fontsize, y=0.99)
    ax.set_title(cfg.subtitle, fontsize=cfg.subtitle_fontsize, pad=10)
    ax.legend(fontsize=cfg.legend_fontsize, loc="upper left", framealpha=0.9)

    # headline callout for the most dramatic case -- placed to the right of
    # its bar and below the legend so neither the callout nor the bar's own
    # value label (drawn above the bar) is ever obscured
    worst_i = int(np.argmax(before))
    ax.annotate(
        f"Case {cases[worst_i]}'s error dropped\nfrom {before[worst_i]:.1f}% to {after[worst_i]:.2f}%",
        xy=(x[worst_i] + cfg.bar_width / 2, after[worst_i]),
        xytext=(x[worst_i] + 1.3, after[worst_i] * 0.4),
        fontsize=cfg.value_label_fontsize + 1, ha="left",
        arrowprops=dict(arrowstyle="->", color=cfg.after_color, lw=1.2),
        bbox=dict(boxstyle="round", fc="white", ec=cfg.after_color, alpha=0.9, lw=0.6))

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    paths = []
    for ext in cfg.output_formats:
        p = os.path.join(out_dir, f"before_after_paar_comparison{tag}.{ext}")
        fig.savefig(p, dpi=cfg.dpi, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=os.path.join(
        config.results_dir("report"), "before_after_paar", "before_after_paar.csv"))
    ap.add_argument("--output-dir", default=config.figures_dir("report"))
    args = ap.parse_args()

    rows = load_csv(args.csv)
    cfg = PlotConfig()
    paths = make_figure(rows, cfg, args.output_dir)
    for p in paths:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
