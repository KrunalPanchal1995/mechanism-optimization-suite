"""
Visualization utilities for forward UQ propagation results.

Five plot types:
  1. plot_ign_delay_bands        — T vs τ with prediction uncertainty bands
  2. plot_parity_with_uncertainty — parity plot with MC error bars
  3. plot_sobol_heatmap           — first-order Sobol index heatmap
  4. plot_coverage_summary        — per-dataset PI coverage bar chart
  5. plot_reaction_uncertainty_ranking — ranked bar chart of mean Sobol S1
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict
from typing import Dict, List, Optional

warnings.filterwarnings("ignore", category=UserWarning)

# ── Shared style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

_CMAP = plt.cm.tab10


# ---------------------------------------------------------------------------
# 1. Ignition-delay (T vs τ) uncertainty bands
# ---------------------------------------------------------------------------

def plot_ign_delay_bands(
    predictions: Dict[int, np.ndarray],
    target_list: List,
    output_dir: str,
    percentiles: tuple = (5, 25, 75, 95),
) -> None:
    """
    Plot T [K] vs τ or S_u with prior uncertainty bands per dataset.

    For Tig / RCM cases: log-scale τ [μs] vs 1000/T [K⁻¹] (Arrhenius plot)
    For Fls cases: T [K] vs S_u [cm/s]
    """
    os.makedirs(output_dir, exist_ok=True)

    # Group cases by (dataset_id, target_type)
    groups: Dict[tuple, List] = defaultdict(list)
    for i, case in enumerate(target_list):
        if i not in predictions:
            continue
        key = (getattr(case, "dataSet_id", str(i)), getattr(case, "target", "Tig"))
        groups[key].append((i, case, predictions[i]))

    for (ds_id, t_type), items in groups.items():
        fig, ax = plt.subplots(figsize=(6, 4))

        # Sort by temperature for clean line plots
        items.sort(key=lambda x: x[1].temperature)

        Tvals    = np.array([c.temperature for _, c, _ in items])
        obs_vals = np.array([c.observed     for _, c, _ in items])

        is_tig = any(k in t_type for k in ("Tig", "RCM", "JSR"))
        is_fls = ("Fls" in t_type or "Fsl" in t_type)

        # Compute prediction percentiles for each case
        eta_pcts = {}
        for pct in [5, 25, 50, 75, 95]:
            eta_pcts[pct] = []
        for _, _case, eta_arr in items:
            for pct in [5, 25, 50, 75, 95]:
                eta_pcts[pct].append(np.percentile(eta_arr, pct))

        # Convert η to physical
        if is_tig:
            x_axis = 1000.0 / Tvals          # 1000/T [K⁻¹]
            x_label = "1000 / T  [K⁻¹]"
            y_obs   = obs_vals                 # [μs]
            y_pcts  = {p: np.exp(np.clip(np.array(v), -30, 30)) / 10.0
                       for p, v in eta_pcts.items()}
            y_label = r"$\tau_{ign}$  [$\mu$s]"
            log_y   = True
        elif is_fls:
            x_axis  = Tvals
            x_label = "T  [K]"
            y_obs   = obs_vals                # [cm/s]
            y_pcts  = {p: np.exp(np.clip(np.array(v), -30, 30))
                       for p, v in eta_pcts.items()}
            y_label = r"$S_u$  [cm/s]"
            log_y   = False
        else:
            x_axis  = Tvals
            x_label = "T  [K]"
            y_obs   = obs_vals
            y_pcts  = {p: np.array(v) for p, v in eta_pcts.items()}
            y_label = "Observable"
            log_y   = False

        # Plot bands (90% and 50%)
        ax.fill_between(x_axis, y_pcts[5],  y_pcts[95],
                        alpha=0.20, color="steelblue", label="90% PI")
        ax.fill_between(x_axis, y_pcts[25], y_pcts[75],
                        alpha=0.40, color="steelblue", label="50% PI")
        ax.plot(x_axis, y_pcts[50], "-", color="steelblue", lw=1.5, label="Median")
        ax.scatter(x_axis, y_obs, c="black", s=25, zorder=5, label="Experimental")

        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        safe_id = str(ds_id).replace("/", "_").replace(" ", "_")
        ax.set_title(f"{safe_id}  ({t_type})")
        ax.legend(fontsize=8, framealpha=0.7)
        fig.tight_layout()

        fname = os.path.join(output_dir, f"ign_delay_bands_{safe_id}.pdf")
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)

    print(f"  [plot] Ignition-delay band plots → {output_dir}/ign_delay_bands_*.pdf")


# ---------------------------------------------------------------------------
# 2. Parity plot with uncertainty
# ---------------------------------------------------------------------------

def plot_parity_with_uncertainty(
    statistics_list: List[Dict],
    output_dir: str,
) -> None:
    """
    η_obs vs η_mean (predicted) parity plot.

    Vertical bars  = ±1 σ_pred (from MC distribution)
    Horizontal bars = ±σ_obs / |η_obs|  (experimental uncertainty)
    Points outside ±2 σ_pred are highlighted in red.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not statistics_list:
        return

    eta_obs  = np.array([s["eta_obs"]  for s in statistics_list])
    eta_mean = np.array([s["eta_mean"] for s in statistics_list])
    eta_std  = np.array([s["eta_std"]  for s in statistics_list])
    # Experimental relative σ
    sigma_obs = np.array([abs(s.get("std_dvtn", 0) / max(abs(s["observed"]), 1e-12))
                           for s in statistics_list])

    residuals = eta_mean - eta_obs
    outside   = np.abs(residuals) > 2 * eta_std

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    # Inside 2σ
    ax.errorbar(eta_obs[~outside], eta_mean[~outside],
                yerr=eta_std[~outside], xerr=sigma_obs[~outside],
                fmt="o", ms=4, color="steelblue", ecolor="steelblue",
                elinewidth=0.6, capsize=2, alpha=0.7, label="Within 2σ")
    # Outside 2σ
    if outside.any():
        ax.errorbar(eta_obs[outside], eta_mean[outside],
                    yerr=eta_std[outside], xerr=sigma_obs[outside],
                    fmt="^", ms=5, color="crimson", ecolor="crimson",
                    elinewidth=0.7, capsize=2, alpha=0.9, label="Outside 2σ")

    lims = [min(eta_obs.min(), eta_mean.min()) - 0.3,
            max(eta_obs.max(), eta_mean.max()) + 0.3]
    ax.plot(lims, lims,        "k-",  lw=0.8, label="Parity")
    ax.plot(lims, [l + 2 * eta_std.mean() for l in lims], "k--", lw=0.5, alpha=0.4)
    ax.plot(lims, [l - 2 * eta_std.mean() for l in lims], "k--", lw=0.5, alpha=0.4)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(r"Observed  $\eta_{obs}$")
    ax.set_ylabel(r"Predicted  $\hat{\eta}$ (MC mean)")
    ax.set_title("Forward UQ Parity Plot")
    ax.legend(fontsize=8)
    fig.tight_layout()

    fname = os.path.join(output_dir, "parity_forward_uq.pdf")
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Parity plot → {fname}")


# ---------------------------------------------------------------------------
# 3. Sobol index heatmap (reactions × cases)
# ---------------------------------------------------------------------------

def plot_sobol_heatmap(
    sobol_df: pd.DataFrame,
    output_dir: str,
    top_n: int = 20,
    index_col: str = "S1",
) -> None:
    """
    Heatmap of first-order Sobol indices: rows = reactions, columns = cases.
    Only the top_n reactions (by mean S1) are shown.
    """
    os.makedirs(output_dir, exist_ok=True)

    if sobol_df.empty:
        return

    pivot = sobol_df.pivot_table(
        index="reaction", columns="case_index",
        values=index_col, aggfunc="mean", fill_value=0.0
    )

    # Keep top-N by mean S1
    pivot["_mean"] = pivot.mean(axis=1)
    pivot = pivot.nlargest(top_n, "_mean").drop(columns="_mean")

    if pivot.empty:
        return

    n_rxn  = len(pivot)
    n_case = len(pivot.columns)
    fig_h  = max(4, n_rxn * 0.35)
    fig_w  = max(5, n_case * 0.30)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    mat = pivot.values
    im  = ax.imshow(mat, aspect="auto", cmap="YlOrRd",
                    vmin=0, vmax=max(mat.max(), 1e-6))
    fig.colorbar(im, ax=ax, label=f"Sobol {index_col}", fraction=0.03, pad=0.02)

    ax.set_yticks(range(n_rxn))
    ax.set_yticklabels([_shorten(r) for r in pivot.index], fontsize=7)
    ax.set_xticks(range(n_case))
    ax.set_xticklabels([f"c{c}" for c in pivot.columns], fontsize=6,
                        rotation=90)
    ax.set_xlabel("Target case index")
    ax.set_ylabel("Reaction")
    ax.set_title(f"First-order Sobol index  S₁  (top {top_n} reactions)")
    fig.tight_layout()

    fname = os.path.join(output_dir, "sobol_heatmap.pdf")
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Sobol heatmap → {fname}")


# ---------------------------------------------------------------------------
# 4. Coverage summary bar chart
# ---------------------------------------------------------------------------

def plot_coverage_summary(
    statistics_list: List[Dict],
    output_dir: str,
) -> None:
    """
    Per-dataset bar chart: % of experimental points within the 90% PI band.
    Reference line at 90% (the nominal coverage target).
    """
    os.makedirs(output_dir, exist_ok=True)

    if not statistics_list:
        return

    # Group by dataset_id
    ds_counts: Dict[str, List[int]] = defaultdict(list)
    for s in statistics_list:
        ds_counts[s["dataset_id"]].append(int(s.get("covered_90", 0)))

    ds_ids   = sorted(ds_counts.keys())
    coverage = [np.mean(ds_counts[ds]) * 100 for ds in ds_ids]

    fig, ax = plt.subplots(figsize=(max(5, len(ds_ids) * 0.7), 4))
    colors  = ["steelblue" if c >= 80 else "tomato" for c in coverage]
    bars    = ax.bar(range(len(ds_ids)), coverage, color=colors, edgecolor="k",
                      linewidth=0.5)
    ax.axhline(90, color="k", linestyle="--", lw=0.8, label="90% target")
    ax.set_xticks(range(len(ds_ids)))
    ax.set_xticklabels([_shorten(d, 14) for d in ds_ids], rotation=45, ha="right",
                        fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Coverage of 90% PI  [%]")
    ax.set_title("Experimental data coverage by prior uncertainty band")
    ax.legend(fontsize=8)

    for bar, cov in zip(bars, coverage):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{cov:.0f}%", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fname = os.path.join(output_dir, "coverage_summary.pdf")
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Coverage summary → {fname}")


# ---------------------------------------------------------------------------
# 5. Reaction uncertainty ranking
# ---------------------------------------------------------------------------

def plot_reaction_uncertainty_ranking(
    sobol_df: pd.DataFrame,
    output_dir: str,
    top_n: int = 15,
) -> None:
    """
    Horizontal bar chart of mean first-order Sobol S1 per reaction,
    averaged over all target cases. Shows top_n most influential reactions.
    Also includes total-order ST for comparison.
    """
    os.makedirs(output_dir, exist_ok=True)

    if sobol_df.empty:
        return

    mean_df = (sobol_df.groupby("reaction")[["S1", "ST"]]
               .mean()
               .sort_values("ST", ascending=True)
               .tail(top_n))

    if mean_df.empty:
        return

    n    = len(mean_df)
    fig, ax = plt.subplots(figsize=(6, max(3, n * 0.35)))

    y     = np.arange(n)
    width = 0.38
    ax.barh(y + width / 2, mean_df["ST"].values, width,
            label="Total-order ST", color="steelblue", alpha=0.85)
    ax.barh(y - width / 2, mean_df["S1"].values, width,
            label="First-order S₁", color="darkorange", alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels([_shorten(r) for r in mean_df.index], fontsize=8)
    ax.set_xlabel("Mean Sobol index (averaged over all target cases)")
    ax.set_title(f"Top {top_n} reactions by uncertainty contribution")
    ax.legend(fontsize=8)
    ax.set_xlim(left=0)
    fig.tight_layout()

    fname = os.path.join(output_dir, "reaction_uncertainty_ranking.pdf")
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Reaction ranking → {fname}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shorten(label: str, max_len: int = 22) -> str:
    """Shorten a long reaction string for axis labels."""
    s = str(label)
    if len(s) <= max_len:
        return s
    # Keep reactants side up to max_len chars
    return s[:max_len - 2] + "…"


# =============================================================================
# postprocess() -- "process the results" flag entry point
# =============================================================================
# Same shared CLI contract as src/MUQ-SAC/VisualAid.py's postprocess() (see
# that file's docstring for the full rationale) via
# mso_io.postprocess_base. forward_uq.py's own driving script persists
# ``forward_uq_statistics.csv`` and ``sobol_indices.csv`` to disk (the two
# tabular outputs plot_sobol_heatmap/plot_reaction_uncertainty_ranking/
# plot_coverage_summary above are built from) -- postprocess() reloads those
# and re-drives the existing plot_* functions with no reimplementation.
# plot_ign_delay_bands/plot_parity_with_uncertainty additionally need a live
# target_list + predictions dict that forward_uq.py does not persist to
# disk -- skipped with a clear note, same degrade-honestly convention as
# MUQ-SAC/VisualAid.py's NominalParityAnalyzer skip.

import sys as _sys  # noqa: E402

_this_dir_pp = os.path.dirname(os.path.abspath(__file__))
_src_dir_pp = os.path.normpath(os.path.join(_this_dir_pp, ".."))
if _src_dir_pp not in _sys.path:
    _sys.path.append(_src_dir_pp)
try:
    from mso_io.postprocess_base import (  # noqa: E402
        PostProcessResult, add_postprocess_args, default_out_dir, discover,
        formats_from_args, write_manifest, write_table,
    )
except ImportError:  # pragma: no cover
    PostProcessResult = None

_UQ_PATTERNS = {
    "statistics": "**/forward_uq_statistics.csv",
    "sobol": "**/sobol_indices.csv",
}


def postprocess(run_dir, out_dir=None, journal=False, csv_only=False,
                 formats=("png",), intermediate=False):
    """Regenerate forward-UQ plots/tables from an existing (or
    still-running) ``forward_uq.py`` run directory. Never raises on missing
    files. Returns a :class:`mso_io.postprocess_base.PostProcessResult`."""
    if PostProcessResult is None:
        raise ImportError("mso_io.postprocess_base not importable -- is src/ on sys.path?")

    run_dir = os.path.abspath(run_dir)
    out_dir = str(default_out_dir(run_dir, out_dir))
    found = discover(run_dir, _UQ_PATTERNS)
    result = PostProcessResult(run_dir=run_dir, out_dir=out_dir, partial=bool(intermediate))
    figures_dir = os.path.join(out_dir, "figures")
    tables_dir = os.path.join(out_dir, "tables")

    stats_df = None
    if found["statistics"]:
        stats_df = pd.read_csv(str(found["statistics"][0]))
        result.tables.append(write_table(stats_df, "forward_uq_statistics", tables_dir))
        if not csv_only:
            try:
                plot_coverage_summary(stats_df.to_dict("records"), figures_dir)
                fname = os.path.join(figures_dir, "coverage_summary.pdf")
                if os.path.exists(fname):
                    result.figures.append(fname)
            except Exception as exc:
                result.skipped.append(f"coverage summary: {type(exc).__name__}: {exc}")
    else:
        result.skipped.append(
            f"coverage summary: no forward_uq_statistics.csv found under {run_dir}"
            + (" (UQ propagation not yet finished)" if intermediate else ""))

    sobol_df = None
    if found["sobol"]:
        sobol_df = pd.read_csv(str(found["sobol"][0]))
        result.tables.append(write_table(sobol_df, "sobol_indices", tables_dir))
        if not csv_only:
            try:
                plot_sobol_heatmap(sobol_df, figures_dir)
                plot_reaction_uncertainty_ranking(sobol_df, figures_dir)
                for fname in ("sobol_heatmap.pdf", "reaction_uncertainty_ranking.pdf"):
                    fpath = os.path.join(figures_dir, fname)
                    if os.path.exists(fpath):
                        result.figures.append(fpath)
            except Exception as exc:
                result.skipped.append(f"sobol plots: {type(exc).__name__}: {exc}")
    else:
        result.skipped.append(
            f"sobol plots: no sobol_indices.csv found under {run_dir}"
            + (" (Sobol analysis not yet run)" if intermediate else " (Sobol analysis may not have been requested)"))

    result.skipped.append(
        "ignition-delay bands / parity-with-uncertainty: need a live target_list + "
        "predictions dict that forward_uq.py does not persist to disk -- not attempted")

    if journal:
        journal_dir = os.path.join(out_dir, "journal_data")
        if stats_df is not None:
            result.journal_tables.append(write_table(stats_df, "journal_forward_uq_statistics", journal_dir))
        if sobol_df is not None:
            top = (sobol_df.groupby("reaction")[["S1", "ST"]].mean()
                   .sort_values("ST", ascending=False).reset_index())
            result.journal_tables.append(write_table(top, "journal_sobol_ranking", journal_dir))

    write_manifest(result, out_dir)
    return result


def _main():
    import argparse
    parser = argparse.ArgumentParser(
        description="forward_uq uq_plots: standalone plotting library, plus "
                     "--postprocess to regenerate plots/tables from an "
                     "existing forward_uq.py run directory.")
    add_postprocess_args(parser)
    args = parser.parse_args()
    if not args.postprocess:
        parser.error("nothing to do -- pass --postprocess RUN_DIR")
    result = postprocess(
        args.postprocess, out_dir=args.out, journal=args.journal,
        csv_only=args.csv_only, formats=formats_from_args(args),
        intermediate=args.intermediate,
    )
    print(f"[uq_plots.postprocess] {len(result.figures)} figure(s), "
          f"{len(result.tables)} table(s), {len(result.journal_tables)} journal table(s) "
          f"-> {result.out_dir}")
    for note in result.skipped:
        print(f"  skipped: {note}")


if __name__ == "__main__":
    _main()
