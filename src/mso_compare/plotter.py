"""
plotter.py
----------
Unified flexible IDT comparison plotter for the mso_compare framework.

Replaces the four separate plotting scripts (``rms_HTC.py``,
``rms_HTC_ALL.py``, ``rms_HTC_comparison_two_sets.py``,
``rms_4_LTT.py``) with a single, mode-driven module.

Supported modes
---------------
``auto``          — Infer from CSV columns which stages are present.
``nominal_only``  — Experimental data + Nominal simulation only.
``nominal_vs_s1`` — Experimental data + Nominal + Stage-1.
``nominal_vs_s2`` — Experimental data + Nominal + Stage-2.
``all_stages``    — Experimental data + Nominal + Stage-1 + Stage-2.

A separate function handles the *two-threshold* scenario where the same
nominal + optimised results are compared for two different parameter sets
(e.g. δ = 0.05 vs δ = 0.01).

Bug fixes over original scripts
---------------------------------
* The output filename in ``rms_HTC_ALL.py`` used ``phi_numeric`` from the
  **last** loop iteration — all PDFs except the last were overwritten.
  Fixed by deriving the output name from the φ sub-folder name.
* ``rms_4_LTT.py`` raised ``KeyError`` for any φ value not in the hardcoded
  marker dict.  Fixed with a per-folder default pool.
* Both scripts plotted all pressure conditions with a single φ-based marker,
  making conditions indistinguishable.  The new code assigns a distinct
  marker to each (φ, P) CSV file.

Public API
----------
plot_comparison(data_dir, plot_dir, mode, ...)
plot_two_threshold_comparison(data_dir_set1, data_dir_set2, plot_dir, ...)
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for batch/server use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import parse_phi_pressure

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

# Line styles per simulation stage
_LINE_STYLE: Dict[str, str] = {
    "Nominal":  "-.",
    "Stage-1":  "dotted",
    "Stage-2":  "-",
}

# Display labels (used in legend entries alongside the ε value)
_STAGE_DISPLAY: Dict[str, str] = {
    "Nominal":  "Nominal",
    "Stage-1":  "Stage-1 (Opt.)",
    "Stage-2":  "Stage-2 (Opt.)",
}

# Stages checked in this fixed order so legend entries are consistent
_STAGE_ORDER = ("Nominal", "Stage-1", "Stage-2")

# Marker pools: one pool per φ key; falls back to _DEFAULT_MARKERS
_MARKER_POOLS: Dict[str, List[str]] = {
    "phi_0_25": ["1", "2", "3", "4"],
    "phi_0_5":  ["o", "s", "D", "^"],
    "phi_1_0":  ["P", "X", "v", "<"],
    "phi_2_0":  ["*", "h", ">", "8"],
}
_DEFAULT_MARKERS = ["o", "s", "D", "^", "P", "X", "v", "<", "*", "h", "1", "2"]

# Colour map (20 distinct colours recycled if needed)
_CMAP = plt.cm.tab20(np.linspace(0, 1, 20))


# ---------------------------------------------------------------------------
# Mode helpers
# ---------------------------------------------------------------------------

def _detect_mode(df: pd.DataFrame) -> str:
    """Infer the plotting mode from columns present in *df*."""
    has_s1 = "Stage-1" in df.columns
    has_s2 = "Stage-2" in df.columns
    if has_s1 and has_s2:
        return "all_stages"
    if has_s1:
        return "nominal_vs_s1"
    if has_s2:
        return "nominal_vs_s2"
    return "nominal_only"


def _stage_series(
    mode: str,
    df:   pd.DataFrame,
) -> List[Tuple[str, str, str]]:
    """
    Return ``[(col_name, linestyle, display_label), …]`` for *mode*,
    filtered to columns that actually exist in *df*.
    """
    wanted: Dict[str, List[str]] = {
        "nominal_only":  ["Nominal"],
        "nominal_vs_s1": ["Nominal", "Stage-1"],
        "nominal_vs_s2": ["Nominal", "Stage-2"],
        "all_stages":    ["Nominal", "Stage-1", "Stage-2"],
    }
    cols = wanted.get(mode, ["Nominal"])
    return [
        (c, _LINE_STYLE[c], _STAGE_DISPLAY[c])
        for c in cols
        if c in df.columns
    ]


# ---------------------------------------------------------------------------
# Error function
# ---------------------------------------------------------------------------

def _chi_sq(
    pred: np.ndarray,
    obs:  np.ndarray,
    unc:  np.ndarray,
) -> float:
    """Weighted sum-of-squares: ε = Σ((pred−obs)/σ)²"""
    mask = ~np.isnan(pred)
    if not mask.any():
        return float("nan")
    e = (pred[mask] - obs[mask]) / unc[mask]
    return float(e @ e)


# ---------------------------------------------------------------------------
# Per-φ plot
# ---------------------------------------------------------------------------

def _plot_phi_folder(
    phi_folder:    Path,
    plot_dir:      Path,
    mode:          str   = "auto",
    unc_frac:      float = 0.10,
    stage_labels:  Optional[Dict[str, str]] = None,
    title_suffix:  str   = "",
    figsize:       Tuple[float, float] = (12, 8),
) -> Optional[Path]:
    """
    Plot all CSV files in *phi_folder* on one figure and save as PDF.

    Each CSV in the folder corresponds to one (φ, P) condition.  All
    conditions share the same figure (one axes), distinguished by colour and
    marker.

    Returns
    -------
    Path | None
        Path to the saved PDF, or ``None`` if the folder was empty.
    """
    labels = stage_labels or _STAGE_DISPLAY
    csv_files = sorted(phi_folder.glob("*.csv"))
    if not csv_files:
        logger.warning("No CSV files in %s — skipping.", phi_folder)
        return None

    # Detect mode from the first file if "auto"
    actual_mode = mode
    if mode == "auto":
        actual_mode = _detect_mode(pd.read_csv(csv_files[0]))

    fig, ax = plt.subplots(figsize=figsize)

    phi_key     = phi_folder.name                   # e.g. "phi_0_5"
    marker_pool = _MARKER_POOLS.get(phi_key, _DEFAULT_MARKERS)
    c_idx = m_idx = 0

    for csv_path in csv_files:
        # ── Parse condition from filename ──────────────────────────────────
        try:
            phi_str, pres_str = parse_phi_pressure(csv_path.stem)
        except ValueError:
            logger.warning("Cannot parse φ/P from '%s' — skipped.", csv_path.name)
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            logger.error("Cannot read %s: %s", csv_path, exc)
            continue

        # ── Extract T, Obs, σ (µs → ms) ───────────────────────────────────
        T_col   = df.columns[1]
        obs_col = df.columns[2]

        dT      = 1000.0 / df[T_col].to_numpy(dtype=float)
        obs_ms  = df[obs_col].to_numpy(dtype=float) / 1000.0
        unc_ms  = unc_frac * obs_ms

        color   = _CMAP[c_idx % len(_CMAP)]
        c_idx  += 1
        marker  = marker_pool[m_idx % len(marker_pool)]
        m_idx  += 1

        phi_disp = phi_str.replace("_", ".")
        exp_lbl  = fr"Exp:  $\Phi$ = {phi_disp},  P = {pres_str} atm"

        # ── Experimental data ──────────────────────────────────────────────
        ax.errorbar(
            dT, obs_ms,
            yerr=unc_ms,
            fmt=marker,
            markersize=10,
            fillstyle="full",
            color=color,
            ecolor="black",
            label=exp_lbl,
            zorder=3,
        )

        # ── Simulation lines ───────────────────────────────────────────────
        for col, ls, base_lbl in _stage_series(actual_mode, df):
            sim_ms = df[col].to_numpy(dtype=float) / 1000.0
            eps    = _chi_sq(sim_ms, obs_ms, unc_ms)
            disp   = labels.get(col, base_lbl)
            ax.plot(
                dT, sim_ms,
                linestyle=ls,
                color=color,
                alpha=0.85,
                label=fr"{disp}  ($\epsilon$ = {eps:.2f})",
            )

    if not ax.get_lines():
        plt.close(fig)
        return None

    _format_axes(ax)

    # ── Split legend ───────────────────────────────────────────────────────
    handles, lbls = ax.get_legend_handles_labels()
    exp_h  = [h for h, l in zip(handles, lbls) if "Exp" in l]
    exp_l  = [l for l in lbls if "Exp" in l]
    sim_h  = [h for h, l in zip(handles, lbls) if "Exp" not in l]
    sim_l  = [l for l in lbls if "Exp" not in l]

    leg1 = ax.legend(
        exp_h, exp_l,
        loc="upper left",
        title="Experimental Data",
        fontsize=11, borderpad=1.2,
    )
    ax.add_artist(leg1)

    sim_title = "Mechanism Comparison"
    if title_suffix:
        sim_title += f"  —  {title_suffix}"
    ax.legend(
        sim_h, sim_l,
        loc="lower right",
        title=sim_title,
        fontsize=11, borderpad=1.2,
    )

    # ── Save ───────────────────────────────────────────────────────────────
    out_dir  = plot_dir / phi_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"IDT_{phi_key}_{actual_mode}.pdf"

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Per-φ plot saved → %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Combined multi-panel plot
# ---------------------------------------------------------------------------

def _plot_combined(
    data_dir:   Path,
    plot_dir:   Path,
    mode:       str   = "auto",
    unc_frac:   float = 0.10,
    stage_labels: Optional[Dict[str, str]] = None,
    title_suffix: str = "",
    panel_w:    float = 10.0,
    panel_h:    float = 6.0,
) -> Optional[Path]:
    """
    Generate a combined multi-panel figure: one subplot per φ sub-folder.

    Layout: up to 3 columns, wrapping into additional rows as needed.

    Returns
    -------
    Path | None
        Path to the saved PDF, or ``None`` if no data was found.
    """
    labels      = stage_labels or _STAGE_DISPLAY
    phi_folders = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    n_phi       = len(phi_folders)

    if n_phi == 0:
        logger.warning("No φ sub-folders in %s — combined plot skipped.", data_dir)
        return None

    ncols = min(n_phi, 3)
    nrows = math.ceil(n_phi / ncols)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(panel_w * ncols, panel_h * nrows),
        squeeze=False,
    )

    actual_mode_global = mode   # may be overridden per panel

    for ax_idx, phi_folder in enumerate(phi_folders):
        row_i = ax_idx // ncols
        col_i = ax_idx % ncols
        ax    = axes[row_i][col_i]

        phi_key  = phi_folder.name
        phi_disp = phi_key.replace("phi_", "Φ = ").replace("_", ".")

        csv_files = sorted(phi_folder.glob("*.csv"))
        if not csv_files:
            ax.set_visible(False)
            continue

        actual_mode = actual_mode_global
        if actual_mode == "auto":
            actual_mode = _detect_mode(pd.read_csv(csv_files[0]))

        marker_pool = _MARKER_POOLS.get(phi_key, _DEFAULT_MARKERS)
        c_idx = m_idx = 0

        for csv_path in csv_files:
            try:
                phi_str, pres_str = parse_phi_pressure(csv_path.stem)
                df = pd.read_csv(csv_path)
            except Exception as exc:
                logger.warning("Skipping %s: %s", csv_path.name, exc)
                continue

            T_col   = df.columns[1]
            obs_col = df.columns[2]

            dT     = 1000.0 / df[T_col].to_numpy(dtype=float)
            obs_ms = df[obs_col].to_numpy(dtype=float) / 1000.0
            unc_ms = unc_frac * obs_ms

            color  = _CMAP[c_idx % len(_CMAP)]
            c_idx += 1
            marker = marker_pool[m_idx % len(marker_pool)]
            m_idx += 1

            ax.errorbar(
                dT, obs_ms,
                yerr=unc_ms,
                fmt=marker,
                markersize=8,
                fillstyle="full",
                color=color,
                ecolor="black",
                label=f"P = {pres_str} atm",
                zorder=3,
            )

            for col, ls, base_lbl in _stage_series(actual_mode, df):
                sim_ms = df[col].to_numpy(dtype=float) / 1000.0
                eps    = _chi_sq(sim_ms, obs_ms, unc_ms)
                disp   = labels.get(col, base_lbl)
                ax.plot(
                    dT, sim_ms,
                    linestyle=ls,
                    color=color,
                    alpha=0.85,
                    label=f"{disp}  (ε={eps:.1f})",
                )

        ax.set_yscale("log")
        ax.set_title(phi_disp, fontsize=12, fontweight="bold")
        ax.set_xlabel(r"1000/T  (K$^{-1}$)", fontsize=11)
        ax.set_ylabel(r"IDT  (ms)",           fontsize=11)
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.tick_params(axis="both", which="minor", labelsize=8)
        ax.legend(fontsize=7, loc="best", borderpad=0.6, ncol=1)

    # Hide unused subplot panels
    for idx in range(n_phi, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    suptitle = "Ignition Delay Time — Mechanism Comparison"
    if title_suffix:
        suptitle += f"\n{title_suffix}"
    fig.suptitle(suptitle, fontsize=14, y=1.01)

    plt.tight_layout()

    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / f"combined_{actual_mode}.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Combined plot saved → %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Two-threshold comparison
# ---------------------------------------------------------------------------

def plot_two_threshold_comparison(
    data_dir_set1: Path | str,
    data_dir_set2: Path | str,
    plot_dir:      Path | str,
    set1_label:    str   = "δ = 0.05",
    set2_label:    str   = "δ = 0.01",
    unc_frac:      float = 0.10,
    figsize:       Tuple[float, float] = (12, 8),
    plot_combined: bool  = True,
) -> None:
    """
    Compare two sets of optimisation results that differ only in a single
    parameter (e.g. sparsity threshold δ) against the same nominal mechanism.

    Parameters
    ----------
    data_dir_set1, data_dir_set2 : Path | str
        Merged data directories for the two parameter sets.  Each must
        contain φ sub-folders → standardised CSV files.
    plot_dir : Path | str
        Output folder for the comparison PDFs.
    set1_label, set2_label : str
        Legend labels for the two sets (e.g. ``"δ = 0.05"``).
    unc_frac : float
        σ = fraction × obs  (default 0.10).
    figsize : tuple
        Figure size per (φ, P) figure.
    plot_combined : bool
        If ``True``, also generate a combined multi-panel figure.
    """
    data_dir_set1 = Path(data_dir_set1)
    data_dir_set2 = Path(data_dir_set2)
    plot_dir      = Path(plot_dir)

    phi_folders = sorted([d for d in data_dir_set1.iterdir() if d.is_dir()])

    for phi_folder in phi_folders:
        phi_key   = phi_folder.name
        phi2_fold = data_dir_set2 / phi_key

        if not phi2_fold.exists():
            logger.warning("φ folder missing in set-2: %s — skipped.", phi_key)
            continue

        csv_files = sorted(phi_folder.glob("*.csv"))
        fig, ax   = plt.subplots(figsize=figsize)

        marker_pool = _MARKER_POOLS.get(phi_key, _DEFAULT_MARKERS)
        c_idx = m_idx = 0

        for csv_path in csv_files:
            csv2_path = phi2_fold / csv_path.name
            if not csv2_path.exists():
                logger.warning(
                    "File missing from set-2: %s — skipped.", csv_path.name
                )
                continue

            try:
                phi_str, pres_str = parse_phi_pressure(csv_path.stem)
                df1 = pd.read_csv(csv_path)
                df2 = pd.read_csv(csv2_path)
            except Exception as exc:
                logger.warning("Skipping %s: %s", csv_path.name, exc)
                continue

            T_col   = df1.columns[1]
            obs_col = df1.columns[2]

            dT     = 1000.0 / df1[T_col].to_numpy(dtype=float)
            obs_ms = df1[obs_col].to_numpy(dtype=float) / 1000.0
            unc_ms = unc_frac * obs_ms

            color   = _CMAP[c_idx % len(_CMAP)]
            c_idx  += 1
            marker  = marker_pool[m_idx % len(marker_pool)]
            m_idx  += 1

            phi_disp = phi_str.replace("_", ".")
            exp_lbl  = fr"Exp:  $\Phi$ = {phi_disp},  P = {pres_str} atm"

            ax.errorbar(
                dT, obs_ms,
                yerr=unc_ms,
                fmt=marker,
                markersize=10,
                fillstyle="full",
                color=color,
                ecolor="black",
                label=exp_lbl,
                zorder=3,
            )

            # Nominal (same in both sets — use set-1)
            if "Nominal" in df1.columns:
                nom_ms  = df1["Nominal"].to_numpy(dtype=float) / 1000.0
                eps_nom = _chi_sq(nom_ms, obs_ms, unc_ms)
                ax.plot(
                    dT, nom_ms,
                    linestyle="-.",
                    color=color,
                    alpha=0.85,
                    label=fr"Nominal  ($\epsilon$ = {eps_nom:.2f})",
                )

            # Optimised column: prefer Stage-2, fall back to Stage-1
            opt_col = "Stage-2" if "Stage-2" in df1.columns else "Stage-1"

            for df_s, ls, lbl in (
                (df1, "dotted", set1_label),
                (df2, "-",      set2_label),
            ):
                if opt_col in df_s.columns:
                    opt_ms = df_s[opt_col].to_numpy(dtype=float) / 1000.0
                    eps    = _chi_sq(opt_ms, obs_ms, unc_ms)
                    ax.plot(
                        dT, opt_ms,
                        linestyle=ls,
                        color=color,
                        alpha=0.85,
                        label=fr"Opt. ({lbl})  ($\epsilon$ = {eps:.2f})",
                    )

        _format_axes(ax)

        handles, lbls = ax.get_legend_handles_labels()
        exp_h = [h for h, l in zip(handles, lbls) if "Exp" in l]
        exp_l = [l for l in lbls if "Exp" in l]
        sim_h = [h for h, l in zip(handles, lbls) if "Exp" not in l]
        sim_l = [l for l in lbls if "Exp" not in l]

        leg1 = ax.legend(
            exp_h, exp_l,
            loc="upper left",
            title="Experimental Data",
            fontsize=11, borderpad=1.2,
        )
        ax.add_artist(leg1)
        ax.legend(
            sim_h, sim_l,
            loc="lower right",
            title=f"Threshold Comparison\n({set1_label} vs {set2_label})",
            fontsize=11, borderpad=1.2,
        )

        plt.tight_layout()
        out_dir  = plot_dir / phi_key
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"two_threshold_{phi_key}.pdf"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Two-threshold plot saved → %s", out_path)

    # Optional combined multi-panel
    if plot_combined:
        _plot_combined(
            data_dir=data_dir_set1,
            plot_dir=plot_dir,
            mode="auto",
            unc_frac=unc_frac,
            title_suffix=f"Threshold Comparison: {set1_label} vs {set2_label}",
        )


# ---------------------------------------------------------------------------
# Axis formatting helper
# ---------------------------------------------------------------------------

def _format_axes(ax: plt.Axes) -> None:
    """Apply standard axis formatting for IDT plots."""
    ax.set_yscale("log")
    ax.set_xlabel(
        r"Temperature  ($\frac{1000}{T}$,  K$^{-1}$)",
        fontsize=14,
    )
    ax.set_ylabel(
        r"Ignition Delay Time  ($\tau$,  ms)",
        fontsize=14,
    )
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which="minor", labelsize=10)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def plot_comparison(
    data_dir:     str | Path,
    plot_dir:     str | Path,
    mode:         str   = "auto",
    unc_frac:     float = 0.10,
    plot_per_phi: bool  = True,
    plot_combined: bool = True,
    title_suffix: str   = "",
    stage_labels: Optional[Dict[str, str]] = None,
) -> None:
    """
    Generate IDT comparison plots from a folder of merged CSVs.

    *data_dir* must contain φ sub-folders (``phi_0_5/``, ``phi_1_0/``, …),
    each holding standardised CSV files (``Phi_0_5_P_10.csv``, …).

    Parameters
    ----------
    data_dir : str | Path
        Folder of merged CSVs organised by φ sub-folder.
    plot_dir : str | Path
        Output folder for PDFs.
    mode : str
        ``'auto'`` | ``'nominal_only'`` | ``'nominal_vs_s1'`` |
        ``'nominal_vs_s2'`` | ``'all_stages'``
    unc_frac : float
        Uncertainty fraction (σ = unc_frac × obs).  Default 0.10.
    plot_per_phi : bool
        Generate one PDF per φ sub-folder.
    plot_combined : bool
        Generate a single combined multi-panel PDF.
    title_suffix : str
        Optional subtitle appended to figure titles.
    stage_labels : dict | None
        Override the default display labels for simulation stages.
    """
    data_dir = Path(data_dir)
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    phi_folders = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    if not phi_folders:
        # Flat layout: CSVs directly in data_dir (legacy fallback)
        if any(data_dir.glob("*.csv")):
            phi_folders = [data_dir]
            logger.info(
                "Flat CSV layout detected in %s — treating as single φ group.", data_dir
            )
        else:
            logger.warning("No φ sub-folders or CSV files found in %s.", data_dir)
            return

    if plot_per_phi:
        logger.info("Generating per-φ plots from: %s", data_dir)
        for pf in phi_folders:
            _plot_phi_folder(
                phi_folder=pf,
                plot_dir=plot_dir,
                mode=mode,
                unc_frac=unc_frac,
                stage_labels=stage_labels,
                title_suffix=title_suffix,
            )

    if plot_combined:
        logger.info("Generating combined multi-panel plot from: %s", data_dir)
        _plot_combined(
            data_dir=data_dir,
            plot_dir=plot_dir,
            mode=mode,
            unc_frac=unc_frac,
            stage_labels=stage_labels,
            title_suffix=title_suffix,
        )
