#!/usr/bin/env python3
"""
convergence_plotter.py
======================
Two-panel convergence diagnostic for PyGAD + L-BFGS-B optimisation runs.

  Panel A – All function evaluations (scatter) overlaid with the
            best-so-far envelope (running minimum).
  Panel B – Best objective per generation (GA phase) and per-call
            running minimum (L-BFGS-B phase), both on a single axis
            using the global function-call count as X.

Data sources
------------
  samplefile.txt  (default)
      Format written by OptimizationTool.py (after bug fix):
          generation,count,obj,fitness
      GA rows    : integer generation number
      L-BFGS-B   : literal string "LB" in the generation column

  Objective.txt   (used when --every-n N is given)
      One float per line (every single function call).
      Simpler; no generation / phase info.

Update modes (--live)
---------------------
  Default        – redraw once per new completed generation
                   (polls samplefile.txt; generation column increments).
  --every-n N    – redraw every N new lines in Objective.txt.

Auto log-scale
--------------
  Y-axis switches to log if  max(obj) / min(obj) > 10 ** log_threshold
  (default threshold = 2, i.e. two orders of magnitude).

Usage
-----
  # Offline (save PNG):
  python convergence_plotter.py

  # Live, per-generation update from samplefile.txt:
  python convergence_plotter.py --live

  # Live, every 100 calls from Objective.txt:
  python convergence_plotter.py --live --every-n 100

  # Custom paths / options:
  python convergence_plotter.py --samplefile run1/samplefile.txt \\
      --outfile run1_conv.png --log-threshold 1.5 --live --interval 3
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np

# ── Backend selection must happen before pyplot is imported ──────────────────
def _pick_backend(live: bool) -> str:
    import matplotlib
    if not live:
        matplotlib.use("Agg")
        return "Agg"
    for _b in ("TkAgg", "Qt5Agg", "Qt4Agg", "WXAgg", "GTK3Agg", "GTK4Agg"):
        try:
            matplotlib.use(_b)
            import matplotlib.pyplot  # noqa: trigger backend load
            return _b
        except Exception:
            continue
    # Fall back to Agg (offline) and warn
    matplotlib.use("Agg")
    print("[warn] No interactive backend found; falling back to Agg (offline).",
          file=sys.stderr)
    return "Agg"


# ── Parse arguments first so we know whether --live is set ──────────────────
def _get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--samplefile", default="samplefile.txt",
                   help="Path to samplefile.txt  [default: %(default)s]")
    p.add_argument("--objfile", default="Objective.txt",
                   help="Path to Objective.txt (used with --every-n)  "
                        "[default: %(default)s]")
    p.add_argument("--live", action="store_true",
                   help="Keep the window open and update in real-time")
    p.add_argument("--every-n", type=int, default=None, metavar="N",
                   help="Update every N new lines in Objective.txt "
                        "(overrides per-generation polling of samplefile.txt)")
    p.add_argument("--interval", type=float, default=2.0,
                   help="Live polling interval in seconds  [default: %(default)s]")
    p.add_argument("--outfile", default="convergence_plot.png",
                   help="Output filename for offline / snapshot saves  "
                        "[default: %(default)s]")
    p.add_argument("--log-threshold", type=float, default=2.0,
                   dest="log_threshold",
                   help="Switch Y to log-scale when max/min > 10**this  "
                        "[default: %(default)s]")
    p.add_argument("--title", default="",
                   help="Optional super-title appended to the figure")
    return p.parse_args()


ARGS = _get_args()
_BACKEND = _pick_backend(ARGS.live)

import matplotlib                      # noqa: E402  (import after backend set)
import matplotlib.pyplot as plt        # noqa: E402
import matplotlib.ticker as ticker     # noqa: E402

# ── Colour palette ───────────────────────────────────────────────────────────
C = {
    "ga_scatter":  "#5B8DB8",   # soft blue   – individual GA evaluations
    "ga_run_min":  "#C0392B",   # red         – best-so-far (Panel A, GA)
    "lb_scatter":  "#E67E22",   # orange      – individual L-BFGS-B evaluations
    "lb_run_min":  "#8E44AD",   # purple      – best-so-far (Panel A, L-BFGS-B)
    "gen_best":    "#27AE60",   # green       – per-gen best (Panel B, GA)
    "gen_run_min": "#C0392B",   # red         – running min per gen (Panel B, GA)
    "lb_b_trace":  "#8E44AD",   # purple      – L-BFGS-B trace (Panel B)
    "sep_vline":   "#7F8C8D",   # grey        – GA/L-BFGS-B phase separator
    "obj_scatter": "#5B8DB8",   # blue        – every-n mode scatter
    "obj_run_min": "#C0392B",   # red         – every-n mode envelope
}


# ════════════════════════════════════════════════════════════════════════════
# I.  Parsers
# ════════════════════════════════════════════════════════════════════════════

def parse_samplefile(filepath: str):
    """
    Parse samplefile.txt.

    Returns
    -------
    ga_data : list of (gen:int, count:int, obj:float, fitness:float)
    lb_data : list of (count:int, obj:float, fitness:float)
    """
    ga_data, lb_data = [], []
    try:
        with open(filepath, "r") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 4:
                    continue
                try:
                    gen_s   = parts[0].strip()
                    count   = int(parts[1].strip())
                    obj     = float(parts[2].strip())
                    fitness = float(parts[3].strip())
                except (ValueError, IndexError):
                    continue
                # skip rows where obj is 0.0 AND fitness is suspicious
                # (artefact of the old bug where self.objective was always 0)
                if obj == 0.0 and fitness > 1e8:
                    continue
                if gen_s.upper() == "LB":
                    lb_data.append((count, obj, fitness))
                else:
                    try:
                        ga_data.append((int(gen_s), count, obj, fitness))
                    except ValueError:
                        continue
    except FileNotFoundError:
        pass
    return ga_data, lb_data


def parse_objfile(filepath: str):
    """
    Parse Objective.txt: one float per line.

    Returns list of (call_index:int, obj:float).
    """
    data = []
    try:
        with open(filepath, "r") as fh:
            for idx, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    data.append((idx, float(raw)))
                except ValueError:
                    continue
    except FileNotFoundError:
        pass
    return data


# ════════════════════════════════════════════════════════════════════════════
# II.  Derived quantities
# ════════════════════════════════════════════════════════════════════════════

def running_minimum(values):
    """Cumulative minimum (best-so-far) for a sequence of values."""
    out, cur = [], float("inf")
    for v in values:
        cur = min(cur, v)
        out.append(cur)
    return out


def best_per_gen(ga_data):
    """
    For each generation, find the evaluation with the lowest obj and
    record its global call count as the X coordinate.

    Returns
    -------
    counts    : list[int]   global call count at which gen's best was found
    best_objs : list[float] best obj in that generation
    run_min   : list[float] running minimum across generations
    """
    gen_entries = defaultdict(list)
    for gen, count, obj, fitness in ga_data:
        gen_entries[gen].append((count, obj))

    gens = sorted(gen_entries.keys())
    counts, best_objs = [], []
    for g in gens:
        pts = gen_entries[g]
        best_count, best_obj = min(pts, key=lambda t: t[1])
        counts.append(best_count)
        best_objs.append(best_obj)

    # sort by count (should already be sorted but be safe)
    paired = sorted(zip(counts, best_objs), key=lambda t: t[0])
    counts    = [p[0] for p in paired]
    best_objs = [p[1] for p in paired]
    run_min   = running_minimum(best_objs)
    return counts, best_objs, run_min


# ════════════════════════════════════════════════════════════════════════════
# III.  Auto log-scale helper
# ════════════════════════════════════════════════════════════════════════════

def _apply_log_scale(ax, values, axis: str = "y", threshold: float = 2.0):
    """
    Switch the given axis to log-scale if the finite positive values span
    more than `threshold` orders of magnitude.

    Returns True when log-scale was applied, False otherwise.
    """
    pos = [v for v in values if np.isfinite(v) and v > 0]
    if len(pos) < 2:
        return False
    ratio = max(pos) / min(pos)
    use_log = ratio > 10 ** threshold
    setter = ax.set_yscale if axis == "y" else ax.set_xscale
    setter("log" if use_log else "linear")
    if use_log:
        # Nice log-format for axis labels
        if axis == "y":
            ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
        else:
            ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    return use_log


# ════════════════════════════════════════════════════════════════════════════
# IV.  Figure factory
# ════════════════════════════════════════════════════════════════════════════

def make_figure():
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(15, 5.5),
        gridspec_kw={"wspace": 0.30},
    )
    return fig, ax_a, ax_b


# ════════════════════════════════════════════════════════════════════════════
# V.  Core draw routine
# ════════════════════════════════════════════════════════════════════════════

def _waiting_message(ax_a, ax_b, msg="Waiting for data…"):
    for ax in (ax_a, ax_b):
        ax.cla()
        ax.text(0.5, 0.5, msg, ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray",
                style="italic")
        ax.set_xticks([])
        ax.set_yticks([])


def draw_every_n(fig, ax_a, ax_b, obj_data, log_threshold: float):
    """
    Draw both panels using Objective.txt data (every-N mode).
    X-axis = global function-call index for both panels.
    """
    ax_a.cla(); ax_b.cla()

    if not obj_data:
        _waiting_message(ax_a, ax_b, "Objective.txt is empty or not found yet.")
        return

    indices  = [d[0] for d in obj_data]
    obj_vals = [d[1] for d in obj_data]
    run_min  = running_minimum(obj_vals)

    # ── Panel A ──────────────────────────────────────────────────────────────
    ax_a.scatter(indices, obj_vals, s=5, alpha=0.25,
                 color=C["obj_scatter"], zorder=2, label="All evaluations")
    ax_a.plot(indices, run_min, color=C["obj_run_min"], lw=2.0,
              zorder=3, label="Best so far")
    ax_a.set_xlabel("Function call #", fontsize=10)
    ax_a.set_ylabel("Objective value", fontsize=10)
    ax_a.set_title("Panel A – All evaluations + best-so-far", fontsize=11,
                   fontweight="bold")
    ax_a.legend(fontsize=8, framealpha=0.7)
    _apply_log_scale(ax_a, obj_vals, threshold=log_threshold)

    # ── Panel B ──────────────────────────────────────────────────────────────
    ax_b.plot(indices, run_min, color=C["gen_run_min"], lw=2.2,
              label="Running minimum")
    # Mark the update sub-sample with larger dots
    n_step = max(1, len(obj_data) // 40)
    idx_sub = indices[::n_step]
    rm_sub  = run_min[::n_step]
    ax_b.scatter(idx_sub, rm_sub, s=30, color=C["gen_run_min"],
                 zorder=4, alpha=0.8)
    ax_b.set_xlabel("Function call #", fontsize=10)
    ax_b.set_ylabel("Best objective (running min)", fontsize=10)
    ax_b.set_title("Panel B – Running minimum trace", fontsize=11,
                   fontweight="bold")
    ax_b.legend(fontsize=8, framealpha=0.7)
    _apply_log_scale(ax_b, run_min, threshold=log_threshold)

    # status annotation
    best_val = run_min[-1]
    total    = len(obj_data)
    ax_b.annotate(
        f"Best = {best_val:.4e}\n({total} evals total)",
        xy=(0.97, 0.97), xycoords="axes fraction",
        ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="gray", alpha=0.8),
    )
    fig.canvas.draw_idle()


def draw_samplefile(fig, ax_a, ax_b,
                    ga_data, lb_data,
                    log_threshold: float):
    """
    Draw both panels from samplefile.txt data.
    X-axis = global function-call count (for both panels and both phases).
    """
    ax_a.cla(); ax_b.cla()

    if not ga_data and not lb_data:
        _waiting_message(ax_a, ax_b,
                         "samplefile.txt is empty or not found yet.\n"
                         "Start the optimisation or check the file path.")
        return

    # ── collect all obj values for scale decisions ────────────────────────
    all_objs = [d[2] for d in ga_data] + [d[1] for d in lb_data]

    # ─────────────────────────────────────────────────────────────────────────
    # Panel A – all evaluations
    # ─────────────────────────────────────────────────────────────────────────
    all_counts_sorted = []
    all_objs_sorted   = []

    if ga_data:
        ga_counts = [d[1] for d in ga_data]
        ga_objs   = [d[2] for d in ga_data]
        ax_a.scatter(
            ga_counts, ga_objs, s=4, alpha=0.20,
            color=C["ga_scatter"], zorder=2,
            label=f"GA evals ({len(ga_counts):,})",
        )
        all_counts_sorted.extend(zip(ga_counts, ga_objs))

    if lb_data:
        lb_counts = [d[0] for d in lb_data]
        lb_objs   = [d[1] for d in lb_data]
        ax_a.scatter(
            lb_counts, lb_objs, s=20, alpha=0.65, marker="D",
            color=C["lb_scatter"], zorder=3,
            label=f"L-BFGS-B evals ({len(lb_counts):,})",
        )
        all_counts_sorted.extend(zip(lb_counts, lb_objs))
        # Vertical phase separator
        if ga_data:
            sep_x = min(lb_counts) - 0.5
            ax_a.axvline(sep_x, color=C["sep_vline"], lw=1.2,
                         ls="--", alpha=0.65, label="Phase boundary")

    # Running minimum over ALL evaluations (global best-so-far)
    if all_counts_sorted:
        paired = sorted(all_counts_sorted, key=lambda t: t[0])
        sc = [t[0] for t in paired]
        so = [t[1] for t in paired]
        rm = running_minimum(so)

        # Split the running-min line by phase for colour coding
        if lb_data and ga_data:
            sep_count = min(d[0] for d in lb_data)
            ga_mask = [c < sep_count for c in sc]
            lb_mask = [c >= sep_count for c in sc]

            sc_ga = [c for c, m in zip(sc, ga_mask) if m]
            rm_ga = [r for r, m in zip(rm, ga_mask) if m]
            sc_lb = [c for c, m in zip(sc, lb_mask) if m]
            rm_lb = [r for r, m in zip(rm, lb_mask) if m]

            if sc_ga:
                ax_a.plot(sc_ga, rm_ga, color=C["ga_run_min"], lw=2.2,
                          zorder=4, label="Best so far (GA)")
            if sc_lb:
                ax_a.plot(sc_lb, rm_lb, color=C["lb_run_min"], lw=2.2,
                          zorder=4, ls="--", label="Best so far (L-BFGS-B)")
        else:
            ax_a.plot(sc, rm, color=C["ga_run_min"], lw=2.2,
                      zorder=4, label="Best so far")

    ax_a.set_xlabel("Function call #", fontsize=10)
    ax_a.set_ylabel("Objective value", fontsize=10)
    ax_a.set_title("Panel A – All evaluations + best-so-far envelope",
                   fontsize=11, fontweight="bold")
    ax_a.legend(fontsize=7.5, loc="upper right", framealpha=0.7)
    _apply_log_scale(ax_a, all_objs, threshold=log_threshold)

    # ─────────────────────────────────────────────────────────────────────────
    # Panel B – best per generation (GA) + L-BFGS-B trace
    # ─────────────────────────────────────────────────────────────────────────
    panel_b_vals = []

    if ga_data:
        gen_counts, gen_best, gen_run_min = best_per_gen(ga_data)
        # Per-generation best dots
        ax_b.scatter(
            gen_counts, gen_best, s=22, color=C["gen_best"],
            zorder=3, alpha=0.75, label="Per-gen best (GA)",
        )
        # Running minimum across generations
        ax_b.plot(
            gen_counts, gen_run_min,
            color=C["gen_run_min"], lw=2.3, zorder=4,
            label="Running min (GA)",
        )
        panel_b_vals.extend(gen_best)

    if lb_data:
        lb_counts = [d[0] for d in lb_data]
        lb_objs   = [d[1] for d in lb_data]
        lb_rm     = running_minimum(lb_objs)
        ax_b.plot(
            lb_counts, lb_rm,
            color=C["lb_b_trace"], lw=2.2, ls="--",
            zorder=4, label="Running min (L-BFGS-B)",
        )
        # Mark individual L-BFGS-B calls with small crosses
        ax_b.scatter(lb_counts, lb_objs, s=10, marker="x",
                     color=C["lb_b_trace"], alpha=0.40, zorder=2)
        if ga_data:
            ax_b.axvline(min(lb_counts) - 0.5,
                         color=C["sep_vline"], lw=1.2, ls="--", alpha=0.65)
        panel_b_vals.extend(lb_objs)

    ax_b.set_xlabel("Function call # (global, monotonic)", fontsize=10)
    ax_b.set_ylabel("Best objective value", fontsize=10)
    ax_b.set_title("Panel B – Best per generation + L-BFGS-B trace",
                   fontsize=11, fontweight="bold")
    ax_b.legend(fontsize=7.5, loc="upper right", framealpha=0.7)
    _apply_log_scale(ax_b, panel_b_vals, threshold=log_threshold)

    # ── status annotations ────────────────────────────────────────────────
    if all_objs:
        best_overall = min(all_objs)
        n_ga  = len(ga_data)
        n_lb  = len(lb_data)
        n_gen = len(set(d[0] for d in ga_data)) if ga_data else 0
        phase = "L-BFGS-B" if lb_data else "GA"
        status = (
            f"Phase: {phase}\n"
            f"Best obj: {best_overall:.4e}\n"
            f"GA evals: {n_ga:,}  ({n_gen} gen)\n"
            f"LB evals: {n_lb:,}"
        )
        ax_b.annotate(
            status,
            xy=(0.97, 0.97), xycoords="axes fraction",
            ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="gray", alpha=0.85),
        )

    fig.canvas.draw_idle()


# ════════════════════════════════════════════════════════════════════════════
# VI.  Update-trigger helpers (live mode)
# ════════════════════════════════════════════════════════════════════════════

class _SamplefileWatcher:
    """Fires when the maximum completed generation increases."""
    def __init__(self):
        self._last_max_gen = -1
        self._last_lb_len  = 0

    def has_new_data(self, ga_data, lb_data) -> bool:
        cur_max_gen = max((d[0] for d in ga_data), default=-1)
        cur_lb_len  = len(lb_data)
        changed = (
            cur_max_gen > self._last_max_gen
            or cur_lb_len > self._last_lb_len
        )
        if changed:
            self._last_max_gen = cur_max_gen
            self._last_lb_len  = cur_lb_len
        return changed


class _EveryNWatcher:
    """Fires when Objective.txt has grown by at least N lines."""
    def __init__(self, n: int):
        self._n         = n
        self._last_batch = -1

    def has_new_data(self, obj_data) -> bool:
        cur_batch = len(obj_data) // self._n
        changed   = cur_batch > self._last_batch
        if changed:
            self._last_batch = cur_batch
        return changed


# ════════════════════════════════════════════════════════════════════════════
# VII.  Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    args      = ARGS
    every_n   = args.every_n
    threshold = args.log_threshold

    fig, ax_a, ax_b = make_figure()

    # Build the super-title
    base_title = "Optimisation Convergence Diagnostics"
    if args.title:
        base_title = f"{base_title} — {args.title}"
    fig.suptitle(base_title, fontsize=13, fontweight="bold")

    # ── OFFLINE mode ──────────────────────────────────────────────────────
    if not args.live:
        if every_n is not None:
            obj_data = parse_objfile(args.objfile)
            draw_every_n(fig, ax_a, ax_b, obj_data, threshold)
        else:
            ga_data, lb_data = parse_samplefile(args.samplefile)
            draw_samplefile(fig, ax_a, ax_b, ga_data, lb_data, threshold)

        fig.subplots_adjust(top=0.92, wspace=0.30)
        fig.savefig(args.outfile, dpi=150, bbox_inches="tight")
        print(f"[convergence_plotter] Saved → {args.outfile}")
        return

    # ── LIVE mode ─────────────────────────────────────────────────────────
    print(f"[convergence_plotter] Live mode | backend={_BACKEND} | "
          f"interval={args.interval}s")

    if every_n is not None:
        print(f"[convergence_plotter] Source: {args.objfile}  "
              f"(update every {every_n} calls)")
        watcher = _EveryNWatcher(every_n)
        _waiting_message(ax_a, ax_b,
                         f"Watching {args.objfile}\n"
                         f"(update every {every_n} function calls)")
        fig.subplots_adjust(top=0.92, wspace=0.30)
        plt.ion()
        plt.show(block=False)

        while plt.fignum_exists(fig.number):
            obj_data = parse_objfile(args.objfile)
            if watcher.has_new_data(obj_data):
                draw_every_n(fig, ax_a, ax_b, obj_data, threshold)
                fig.subplots_adjust(top=0.92, wspace=0.30)
                fig.canvas.flush_events()
                # auto-save snapshot
                fig.savefig(args.outfile, dpi=120, bbox_inches="tight")
            plt.pause(args.interval)

    else:
        print(f"[convergence_plotter] Source: {args.samplefile}  "
              f"(update per generation)")
        watcher = _SamplefileWatcher()
        _waiting_message(ax_a, ax_b,
                         f"Watching {args.samplefile}\n"
                         "(update on each completed generation)")
        fig.subplots_adjust(top=0.92, wspace=0.30)
        plt.ion()
        plt.show(block=False)

        while plt.fignum_exists(fig.number):
            ga_data, lb_data = parse_samplefile(args.samplefile)
            if watcher.has_new_data(ga_data, lb_data):
                draw_samplefile(fig, ax_a, ax_b, ga_data, lb_data, threshold)
                fig.subplots_adjust(top=0.92, wspace=0.30)
                fig.canvas.flush_events()
                # auto-save snapshot on every update
                fig.savefig(args.outfile, dpi=120, bbox_inches="tight")
            plt.pause(args.interval)

    print("[convergence_plotter] Window closed.")


if __name__ == "__main__":
    main()
