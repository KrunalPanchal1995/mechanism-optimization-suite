"""
Per-case legacy-retraining-pass trajectory, for the supplementary materials
(report/supplementary/main.tex, "Per-case retraining-pass trajectory").

Source: TEST_CORRELATIONS_CORRECTED/CASE_STUDIES/hybrid_tol1.0_alpha0.6_
20260730-001553/Prediction_errors.csv (an append-only historical log across
many runs) -- the LAST self-consistent Case-0..Case-7 block in that file was
isolated (rows 105-131) and confirmed to exactly match the "After legacy
re-training" column already published in report/main.tex's baseline table
(Case 4: 2.65%/1.74%, Case 5: 4.08%/2.43%, Case 6: 1.76%/0.97%, Case 7:
1.43%/0.60%, Case 3: 11.75% train after 5 passes -- all match to the digit).
This is the SAME run report/main.tex already summarizes, not a different or
ambiguous one; this figure just shows the pass-by-pass trajectory instead of
only the final value. This is the PRE-PAAR pipeline's own legacy retraining
loop (the one PAAR's Stage 3 replaced) -- not PAAR's D-optimal resampling.

Run: python3.9 plot_legacy_retraining_trajectory.py [--csv PATH] [--output-dir DIR]
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


@dataclass
class PlotConfig:
    figsize: tuple = (9.0, 5.5)
    dpi: int = 200
    marker_size: float = 6.0
    line_width: float = 1.8
    criterion_color: str = "black"
    criterion_linestyle: str = ":"
    title_fontsize: float = 12.5
    axis_label_fontsize: float = 10.5
    tick_label_fontsize: float = 9.5
    legend_fontsize: float = 9.0
    colors: tuple = ("#d7301f", "#fc8d59", "#2c7fb8", "#2ca25f", "#756bb1")


def load(path):
    data = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            case = int(r["case"])
            data.setdefault(case, []).append(
                (int(r["pass"]), float(r["train_pct"]), float(r["test_pct"])))
    for case in data:
        data[case].sort()
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=os.path.join(
        config.results_dir("report"), "before_after_paar", "legacy_retraining_trajectory.csv"))
    ap.add_argument("--output-dir", default=os.path.join(
        config.REPO_ROOT, "report", "supplementary", "figures"))
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = PlotConfig()
    data = load(args.csv)
    cases = sorted(data.keys())

    fig, ax = plt.subplots(figsize=cfg.figsize)
    for i, case in enumerate(cases):
        pts = data[case]
        passes = [p[0] for p in pts]
        test = [p[2] for p in pts]
        color = cfg.colors[i % len(cfg.colors)]
        ax.plot(passes, test, "o-", color=color, ms=cfg.marker_size, lw=cfg.line_width,
                label=f"Case {case}")

    ax.axhline(2.0, color=cfg.criterion_color, linestyle=cfg.criterion_linestyle,
               linewidth=1.2, label="2% acceptable-error threshold")
    ax.set_xlabel("Legacy retraining pass # (0 = before any retraining)",
                   fontsize=cfg.axis_label_fontsize)
    ax.set_ylabel("Hold-out prediction error (%)", fontsize=cfg.axis_label_fontsize)
    ax.set_title("Pre-PAAR legacy retraining loop: pass-by-pass trajectory\n"
                  "(the loop PAAR's Stage 3 replaced -- not PAAR's own D-optimal resampling)",
                  fontsize=cfg.title_fontsize)
    ax.tick_params(labelsize=cfg.tick_label_fontsize)
    ax.legend(fontsize=cfg.legend_fontsize, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(args.output_dir, f"legacy_retraining_trajectory.{ext}")
        fig.savefig(p, dpi=cfg.dpi, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    for p in paths:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
