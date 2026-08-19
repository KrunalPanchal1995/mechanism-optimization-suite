"""
Recreated, plainly-labeled 10-fold CV figures -- reads ONLY the CSVs in
results_dir("tenfold")/csv/ (produced by export_10fold_csv.py), never the
raw JSON, so the data and the plotting are fully separated per the
2026-07-31 request: to change what's plotted, edit the CSVs (or re-run the
export); to change how it looks, edit PlotConfig below and re-run this
script. No knob is hardcoded inside the plotting calls themselves.

Original version of these figures used dense technical labels (`tol=inf,
k=20, ridge=0`, bare "MRE (%)" axis, unlabeled 2% dashed line) with no
explanation of what a "fold" is. This version uses plain-language titles,
per-panel status sentences, demotes hyperparameters to a small footnote,
and adds a glossary caption -- reproducing the exact same published numbers
(verified against tables/baseline_10fold.tex and tables/final_10fold.tex,
worst-fold values match to 3 decimal places).

Produces:
  fig_10fold_before.pdf/png   -- per-case, per-fold grid, BEFORE the fix
  fig_10fold_after.pdf/png    -- same, AFTER the fix
  fig_10fold_summary.pdf/png  -- single-panel worst-fold-per-case comparison

Run: python3.9 export_10fold_csv.py     # (in results_dir("tenfold")/csv/, once)
     python3.9 plot_10fold_evolution.py [--output-dir DIR] [--csv-dir DIR]
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # validation/
import config

import csv
from collections import defaultdict
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CRITERION = config.CRITERION


@dataclass
class PlotConfig:
    """Every adjustable knob for the 10-fold figures, in one place. Nothing
    below is hardcoded elsewhere in this file's plotting functions."""
    # -- figure / subplot layout --
    grid_figsize: tuple = (16.0, 7.0)
    grid_nrows: int = 2
    grid_ncols: int = 4
    grid_hspace: float = 0.65
    grid_wspace: float = 0.30
    grid_top: float = 0.80     # leaves room for suptitle + subtitle
    grid_bottom: float = 0.22  # leaves room for the shared x-label, legend, and glossary
    summary_figsize: tuple = (9.5, 5.5)
    dpi: int = 200
    output_formats: tuple = ("pdf", "png")

    # -- colors --
    pass_color: str = "#2c7fb8"
    fail_color: str = "#d7301f"
    train_color: str = "#999999"
    criterion_color: str = "black"
    before_bar_color: str = "#d7301f"
    after_bar_color: str = "#2c7fb8"

    # -- line/marker/bar geometry --
    bar_width: float = 0.6
    train_marker_size: float = 3.0
    train_line_width: float = 1.0
    criterion_line_width: float = 1.3
    criterion_line_style: str = ":"
    summary_bar_width: float = 0.35
    edge_line_width: float = 0.6

    # -- font sizes --
    suptitle_fontsize: float = 13.0
    subtitle_fontsize: float = 10.5
    panel_title_fontsize: float = 9.0
    axis_label_fontsize: float = 8.5
    tick_label_fontsize: float = 7.5
    legend_fontsize: float = 9.0
    footnote_fontsize: float = 7.0
    value_label_fontsize: float = 8.5

    # -- plain-language text (edit here, not in the plotting code) --
    grid_title_before: str = "Does accuracy hold up on data the model has never seen?"
    grid_subtitle_before: str = ("10 different test splits per case, BEFORE the fix "
                                  "(last-production hyperparameters, every point scored)")
    grid_title_after: str = "Does accuracy hold up on data the model has never seen?"
    grid_subtitle_after: str = ("Same test, AFTER the fix (worst-of-10-fold hyperparameter "
                                 "search + scoring only physically-plausible points)")
    xlabel: str = "Test split # (1 of 10 independent random 10% slices of the data)"
    ylabel: str = "Prediction error on held-out data (%) — lower is better"
    criterion_label_fmt: str = "{crit:g}% acceptable-error threshold"
    glossary: str = ("\"Fold\" = one of 10 independent ways to split the data into 90% "
                      "train / 10% test, so every point gets tested exactly once. "
                      "\"MRE\" = maximum relative error, the single worst prediction "
                      "in that split, as a percent.")


def load_per_fold_csv(path):
    by_case = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            by_case[int(r["case"])].append(r)
    for case in by_case:
        by_case[case].sort(key=lambda r: int(r["fold"]))
    return by_case


def load_summary_csv(path):
    with open(path) as f:
        return {int(r["case"]): r for r in csv.DictReader(f)}


def plot_grid(by_case, cfg: PlotConfig, title, subtitle, out_name, out_dir):
    cases = sorted(by_case.keys())
    fig, axes = plt.subplots(cfg.grid_nrows, cfg.grid_ncols, figsize=cfg.grid_figsize)
    axes = np.atleast_1d(axes).ravel()

    for i, case in enumerate(cases):
        ax = axes[i]
        rows = by_case[case]
        x = np.array([int(r["fold"]) for r in rows])
        worst = np.array([float(r["worst_pct"]) for r in rows])
        train = np.array([float(r["train_pct"]) for r in rows])
        test = np.array([float(r["test_pct"]) for r in rows])
        colors = [cfg.fail_color if w > CRITERION else cfg.pass_color for w in worst]

        ax.bar(x, test, color=colors, width=cfg.bar_width, zorder=3,
               edgecolor="black", linewidth=cfg.edge_line_width)
        ax.plot(x, train, "o--", color=cfg.train_color, ms=cfg.train_marker_size,
                lw=cfg.train_line_width, zorder=4)
        ax.axhline(CRITERION, color=cfg.criterion_color, ls=cfg.criterion_line_style,
                   lw=cfg.criterion_line_width, zorder=2)

        n_fail = int((worst > CRITERION).sum())
        status = "FAILED" if n_fail > 0 else "PASSED"
        color = cfg.fail_color if n_fail > 0 else "#2b8a3e"
        if n_fail > 0:
            worst_fold = int(x[np.argmax(worst)])
            sentence = f"{status} — split #{worst_fold + 1} alone reached {worst.max():.1f}% error"
        else:
            sentence = f"{status} — every split stayed under {CRITERION:g}%"
        ax.set_title(f"Case {case}\n{sentence}", fontsize=cfg.panel_title_fontsize, color=color)

        hp = rows[0]
        footnote = f"tol={hp['tolerance']}, k={hp['k']}, ridge={float(hp['ridge_alpha']):g}"
        ax.text(0.02, 0.97, footnote, transform=ax.transAxes, fontsize=cfg.footnote_fontsize,
                 va="top", ha="left", color="#555555")

        # Shared-axis convention: the (verbose) x-label and y-label are drawn
        # ONCE at the figure level (see below), not per panel -- with 4
        # panels per row, per-panel copies of this much text collide with
        # the neighboring panel no matter how much hspace is added.
        ax.tick_params(labelsize=cfg.tick_label_fontsize)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v + 1) for v in x])
        ymax = max(test.max(), train.max(), CRITERION) * 1.15
        ax.set_ylim(0, ymax)

    for i in range(len(cases), len(axes)):
        axes[i].axis("off")

    fig.text(0.5, cfg.grid_bottom - 0.045, cfg.xlabel, ha="center",
             fontsize=cfg.axis_label_fontsize + 0.5)
    fig.text(0.01, (cfg.grid_top + cfg.grid_bottom) / 2, cfg.ylabel, va="center",
             rotation="vertical", fontsize=cfg.axis_label_fontsize + 0.5)

    crit_label = cfg.criterion_label_fmt.format(crit=CRITERION)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=cfg.pass_color, label="Test split ≤ threshold"),
        plt.Rectangle((0, 0), 1, 1, color=cfg.fail_color, label="Test split > threshold"),
        plt.Line2D([0], [0], color=cfg.train_color, marker="o", ms=4, ls="--", label="Training error"),
        plt.Line2D([0], [0], color=cfg.criterion_color, ls=cfg.criterion_line_style, label=crit_label),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=cfg.legend_fontsize,
               bbox_to_anchor=(0.5, 0.06))
    fig.text(0.5, 0.01, cfg.glossary, ha="center", fontsize=cfg.footnote_fontsize, color="#666666")
    fig.suptitle(title, fontsize=cfg.suptitle_fontsize, y=0.985)
    fig.text(0.5, 0.94, subtitle, ha="center", fontsize=cfg.subtitle_fontsize)
    fig.subplots_adjust(hspace=cfg.grid_hspace, wspace=cfg.grid_wspace,
                         top=cfg.grid_top, bottom=cfg.grid_bottom)

    paths = []
    for ext in cfg.output_formats:
        p = os.path.join(out_dir, f"{out_name}.{ext}")
        fig.savefig(p, dpi=cfg.dpi, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def plot_summary(summary, cfg: PlotConfig, out_dir, out_name="fig_10fold_summary"):
    cases = sorted(summary.keys())
    before = [float(summary[c]["worst_fold_before_pct"]) for c in cases]
    after = [float(summary[c]["worst_fold_after_pct"]) for c in cases]

    x = np.arange(len(cases))
    fig, ax = plt.subplots(figsize=cfg.summary_figsize)
    b1 = ax.bar(x - cfg.summary_bar_width / 2, before, width=cfg.summary_bar_width,
                color=cfg.before_bar_color, label="Before fix", zorder=3,
                edgecolor="black", linewidth=cfg.edge_line_width)
    b2 = ax.bar(x + cfg.summary_bar_width / 2, after, width=cfg.summary_bar_width,
                color=cfg.after_bar_color, label="After fix", zorder=3,
                edgecolor="black", linewidth=cfg.edge_line_width)
    ax.axhline(CRITERION, color=cfg.criterion_color, ls=cfg.criterion_line_style,
               lw=cfg.criterion_line_width, label=cfg.criterion_label_fmt.format(crit=CRITERION))

    for rect, v in zip(b1, before):
        ax.annotate(f"{v:.1f}", (rect.get_x() + rect.get_width() / 2, v),
                     textcoords="offset points", xytext=(0, 3), ha="center",
                     fontsize=cfg.value_label_fontsize, color=cfg.before_bar_color)
    for rect, v in zip(b2, after):
        ax.annotate(f"{v:.2f}", (rect.get_x() + rect.get_width() / 2, v),
                     textcoords="offset points", xytext=(0, 3), ha="center",
                     fontsize=cfg.value_label_fontsize, color=cfg.after_bar_color)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Case {c}" for c in cases])
    ax.set_ylabel("Worst result across all 10 test splits (%) — lower is better",
                   fontsize=cfg.axis_label_fontsize + 1)
    ax.set_title("The single number each case's reliability hinges on: its worst split, "
                  "before vs. after the fix", fontsize=cfg.subtitle_fontsize + 1.5)
    ax.legend(fontsize=cfg.legend_fontsize)
    ax.tick_params(labelsize=cfg.tick_label_fontsize + 1)
    fig.tight_layout()

    paths = []
    for ext in cfg.output_formats:
        p = os.path.join(out_dir, f"{out_name}.{ext}")
        fig.savefig(p, dpi=cfg.dpi, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv-dir", default=os.path.join(config.results_dir("tenfold"), "csv"))
    ap.add_argument("--output-dir", default=config.figures_dir("tenfold"))
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = PlotConfig()
    before = load_per_fold_csv(os.path.join(args.csv_dir, "10fold_per_fold_before.csv"))
    after = load_per_fold_csv(os.path.join(args.csv_dir, "10fold_per_fold_after.csv"))
    summary = load_summary_csv(os.path.join(args.csv_dir, "10fold_case_summary.csv"))

    for p in plot_grid(before, cfg, cfg.grid_title_before, cfg.grid_subtitle_before,
                        "fig_10fold_before", args.output_dir):
        print(f"wrote {p}")
    for p in plot_grid(after, cfg, cfg.grid_title_after, cfg.grid_subtitle_after,
                        "fig_10fold_after", args.output_dir):
        print(f"wrote {p}")
    for p in plot_summary(summary, cfg, args.output_dir):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
