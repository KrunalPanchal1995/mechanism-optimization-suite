#!/usr/bin/env python3
"""
modules/plotting.py

Sensitivity plot generation.

CHANGES vs original sensitivity_plotter.py:
- PRESERVES MASTER-LIST ORDER. The original re-sorted every plot by
  np.argsort(np.abs(vals)), which directly contradicted the design
  requirement ("No resorting by sensitivity ... makes plots directly
  comparable"). Order now follows the classification/master file.
- Uses the canonical nested sens_db[T][rxn_id] from sensitivity_parser
  and the shared normalize_group from interfaces (no duplicate readers,
  no second parse).

Output tree:
    <output_dir>/HTC/HTC_SA.pdf
    <output_dir>/HTC/All_Temperatures/HTC_700K.pdf ...
    <output_dir>/HTC/HTC_Temperature_Comparison.pdf
    (same for LTC)
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .interfaces import normalize_group


# --- Times New Roman where available, graceful fallback otherwise --------
def _configure_font():
    """Prefer Times New Roman; fall back to a serif Times-like face."""
    candidates = ["Times New Roman", "Times", "Nimbus Roman",
                  "Liberation Serif", "DejaVu Serif"]
    available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), None)
    if chosen:
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = [chosen] + plt.rcParams.get(
            "font.serif", [])
    else:
        plt.rcParams["font.family"] = "serif"
    # mathtext to match
    plt.rcParams["mathtext.fontset"] = "dejavuserif"


_configure_font()


def _save(fig, outbase):
    fig.savefig(outbase + ".pdf", bbox_inches="tight")
    fig.savefig(outbase + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _ordered_reactions(group_df, sens_db_keys):
    """Reaction IDs in master-file order (no sensitivity re-sort).

    NOTE on display order: matplotlib's barh draws index 0 at the BOTTOM.
    To show the most-sensitive (first in the classification file) at the
    TOP, callers invert the y-axis after plotting via _top_down(ax).
    """
    return [int(r) for r in group_df["Reaction_ID"].tolist()]


def _top_down(ax):
    """Put the first reaction at the top of the plot (descending order)."""
    ax.invert_yaxis()


def create_two_temperature_plot(group_name, group_df, sens_db,
                                temperatures, outdir,
                                threshold=0.05, xlim=(-0.30, 0.30)):
    """Grouped bar plot at the lowest and highest available temperature."""
    if len(temperatures) < 2:
        print(f"{group_name}: need >=2 temperatures for paired plot.")
        return
    T1, T2 = temperatures[0], temperatures[-1]
    reactions = _ordered_reactions(group_df, sens_db.keys())
    if not reactions:
        return

    vals1 = [sens_db.get(T1, {}).get(r, 0.0) for r in reactions]
    vals2 = [sens_db.get(T2, {}).get(r, 0.0) for r in reactions]
    ypos = np.arange(len(reactions))
    width = 0.40

    fig, ax = plt.subplots(figsize=(8, max(6, len(reactions) * 0.30)))
    ax.barh(ypos - width / 2, vals1, height=width, label=f"T = {T1:.0f} K")
    ax.barh(ypos + width / 2, vals2, height=width, label=f"T = {T2:.0f} K")
    ax.axvline(0, color="black")
    ax.axvline(threshold, linestyle="--", color="black",
               label=f"Threshold={threshold}")
    ax.axvline(-threshold, linestyle="--", color="black")
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"R{r}" for r in reactions])
    ax.set_xlabel("Sensitivity Coefficient")
    ax.set_ylabel("Reaction ID")
    ax.set_xlim(*xlim)
    ax.legend()
    ax.set_title(f"{group_name} Sensitivity Analysis")
    _top_down(ax)
    _save(fig, os.path.join(outdir, f"{group_name}_SA"))


def create_all_temperature_plots(group_name, group_df, sens_db,
                                 temperatures, outdir, threshold=0.05):
    """One single-temperature plot per temperature, master order preserved."""
    os.makedirs(outdir, exist_ok=True)
    reactions = _ordered_reactions(group_df, sens_db.keys())
    if not reactions:
        return
    ypos = np.arange(len(reactions))

    for T in temperatures:
        vals = [sens_db.get(T, {}).get(r, 0.0) for r in reactions]
        fig, ax = plt.subplots(figsize=(8, max(6, len(reactions) * 0.30)))
        ax.barh(ypos, vals)
        ax.axvline(0, color="black")
        ax.axvline(threshold, linestyle="--", color="black",
                   label=f"Threshold={threshold}")
        ax.axvline(-threshold, linestyle="--", color="black")
        ax.set_yticks(ypos)
        ax.set_yticklabels([f"R{r}" for r in reactions])
        ax.set_xlabel("Sensitivity Coefficient")
        ax.set_ylabel("Reaction ID")
        ax.legend()
        ax.set_title(f"{group_name} @ {T:.0f} K")
        _top_down(ax)
        _save(fig, os.path.join(outdir, f"{group_name}_{int(T)}K"))


def create_per_case_plots(group_name, group_df, case_db,
                          outdir, threshold=0.05):
    """One plot per (case, temperature) combination.

    case_db is keyed on (case_id, temperature) -> {rxn_id: sensitivity}
    (from sensitivity_parser.build_case_database). Multiple cases at the
    same temperature each get their own plot; the case id and temperature
    both appear in the filename, e.g.:

        HTC_T680K_case0.pdf
        HTC_T900K_case4.pdf
        HTC_T1200K_case2.pdf
        HTC_T1200K_case7.pdf   (two different cases, same temperature)

    Reaction order follows the classification file (no sensitivity re-sort).
    """
    os.makedirs(outdir, exist_ok=True)
    reactions = _ordered_reactions(group_df, case_db.keys())
    if not reactions:
        return
    ypos = np.arange(len(reactions))

    # sort keys by temperature then case for tidy, deterministic output
    keys = sorted(
        case_db.keys(),
        key=lambda k: (k[1], (k[0] is None, k[0])),
    )

    for case_id, T in keys:
        vals = [case_db[(case_id, T)].get(r, 0.0) for r in reactions]
        fig, ax = plt.subplots(figsize=(8, max(6, len(reactions) * 0.30)))
        ax.barh(ypos, vals)
        ax.axvline(0, color="black")
        ax.axvline(threshold, linestyle="--", color="black",
                   label=f"Threshold={threshold}")
        ax.axvline(-threshold, linestyle="--", color="black")
        ax.set_yticks(ypos)
        ax.set_yticklabels([f"R{r}" for r in reactions])
        ax.set_xlabel("Sensitivity Coefficient")
        ax.set_ylabel("Reaction ID")
        ax.legend()

        case_label = "NA" if case_id is None else str(case_id)
        ax.set_title(f"{group_name} @ {T:.0f} K (case {case_label})")
        _top_down(ax)
        fname = f"{group_name}_T{int(T)}K_case{case_label}"
        _save(fig, os.path.join(outdir, fname))


def create_temperature_comparison_plot(group_name, group_df, sens_db,
                                       temperatures, output_file,
                                       threshold=0.05, colors=None):
    """Side-by-side grouped bars across an arbitrary set of temperatures.

    colors : optional list of matplotlib color specs, one per temperature.
    If None, matplotlib's default cycle is used. If fewer colors than
    temperatures are given, the list is cycled.
    """
    reactions = _ordered_reactions(group_df, sens_db.keys())
    if not reactions:
        return
    nT = len(temperatures)
    if nT == 0:
        return
    ypos = np.arange(len(reactions))
    width = 0.8 / nT
    offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, nT)

    fig, ax = plt.subplots(figsize=(10, max(6, len(reactions) * 0.30)))
    for k, T in enumerate(temperatures):
        vals = [sens_db.get(T, {}).get(r, 0.0) for r in reactions]
        color = None
        if colors:
            color = colors[k % len(colors)]
        ax.barh(ypos + offsets[k], vals, height=width,
                label=f"{T:.0f} K", color=color)
    ax.axvline(0, color="black")
    ax.axvline(threshold, linestyle="--", color="black",
               label=f"Threshold={threshold}")
    ax.axvline(-threshold, linestyle="--", color="black")
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"R{r}" for r in reactions])
    ax.set_xlabel("Sensitivity Coefficient")
    ax.set_ylabel("Reaction ID")
    ax.legend()
    ax.set_title(f"{group_name} Temperature Comparison")
    _top_down(ax)
    _save(fig, output_file)


def generate_group_plots(classification_df, sens_db, output_dir,
                         threshold=0.05, compare_temperatures=None,
                         case_db=None, compare_colors=None):
    """Top-level: per-group paired, all-temperature, and comparison plots.

    compare_colors : optional list of colors for the comparison plot,
    one per temperature (cycled if shorter).

    If case_db is supplied (keyed on (case_id, temperature) from
    sensitivity_parser.build_case_database), an additional per-case plot
    is written for every (case, T) combination under <group>/Per_Case/,
    with the case id in the filename.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = classification_df.copy()
    if "Group_norm" not in df.columns:
        df["Group_norm"] = df["Group"].apply(normalize_group)

    temperatures = sorted(sens_db.keys())

    for group in ("HTC", "LTC"):
        gdf = df[df["Group_norm"] == group]
        if gdf.empty:
            continue
        gdir = os.path.join(output_dir, group)
        os.makedirs(gdir, exist_ok=True)

        create_two_temperature_plot(
            group, gdf, sens_db, temperatures, gdir, threshold
        )
        create_all_temperature_plots(
            group, gdf, sens_db, temperatures,
            os.path.join(gdir, "All_Temperatures"), threshold
        )
        if case_db is not None:
            create_per_case_plots(
                group, gdf, case_db,
                os.path.join(gdir, "Per_Case"), threshold
            )
        if compare_temperatures:
            create_temperature_comparison_plot(
                group, gdf, sens_db,
                [float(t) for t in compare_temperatures],
                os.path.join(gdir, f"{group}_Temperature_Comparison"),
                threshold=threshold, colors=compare_colors,
            )
