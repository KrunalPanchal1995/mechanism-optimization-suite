#!/usr/bin/env python3
"""
Forward UQ Propagation via PRS Surrogates.

Propagates Arrhenius parameter uncertainties (from the MUQ-SAC covariance
database) through trained polynomial response surfaces to obtain the full
prior distribution of combustion observables (τ_ign, S_u, etc.).

The key cost-saving insight: once a PRS is trained on ~N_sim simulations,
evaluating 50 000 quasi-Monte Carlo samples costs microseconds (a batch
matrix-vector multiply) versus hours of Cantera runs.

Advanced features
-----------------
* Quasi-Monte Carlo sampling (Sobol sequence) for efficient coverage of
  the N-dimensional uncertainty space
* Vectorised batch PRS evaluation (build B-matrix once, dot with coeff)
* Analytical Sobol sensitivity indices from 2nd-order PRS coefficients
  (no additional simulations needed — Sudret 2008 formula)
* GPR / MLP fallback for PRS cases that failed the accuracy criterion

Usage
-----
    python forward_uq.py target.opt
    python forward_uq.py target.opt --n_samples 100000 --method sobol
    python forward_uq.py target.opt --output_dir MyResults

Prerequisites
-------------
Run the optimization pipeline first to generate PRS coefficient files:
    python src/MUQ-SAC/run_script.py target.opt
"""

import os
import sys
import yaml
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
from copy import deepcopy
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Set up Python paths ───────────────────────────────────────────────────────
# ``src`` MUST go to the front here: this driver file is itself named
# ``forward_uq.py`` and would otherwise shadow the ``src/forward_uq`` *package*
# it needs to import.  Putting ``src`` first lets the package win.
#
# This is runtime-safe with respect to ``src/io`` (a package named ``io``):
# the standard-library ``io`` is imported during interpreter start-up and is
# already cached in ``sys.modules`` by the time this line runs, so it is never
# re-imported from ``src/io``.  (The start-up-time shadow only bites when the
# interpreter is launched with ``src`` as the working directory, e.g.
# ``cd src && python -m rmip.cli`` — use the repo-root ``rmip_cli.py`` launcher.)
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src" / "MUQ-SAC"))
sys.path.insert(0, str(_ROOT / "src"))

# ── Forward UQ utilities ──────────────────────────────────────────────────────
from forward_uq.mc_propagator import MCPropagator
from forward_uq.sobol_prs import AnalyticalSobolPRS
from forward_uq.uq_plots import (
    plot_ign_delay_bands,
    plot_parity_with_uncertainty,
    plot_sobol_heatmap,
    plot_coverage_summary,
    plot_reaction_uncertainty_ranking,
)

# ── MUQ-SAC imports ───────────────────────────────────────────────────────────
import Uncertainty as uncertainty_mod
import combustion_target_class
import ResponseSurface as PRS_module
from sklearn.model_selection import train_test_split


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Forward UQ propagation using PRS surrogates (MUQ-SAC)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("config",
                   help="Path to .opt YAML config file (e.g., target.opt)")
    p.add_argument("--n_samples", type=int, default=50_000,
                   help="Number of Monte Carlo samples (Sobol rounds up to 2^k)")
    p.add_argument("--method",    choices=["sobol", "lhs", "numpy"],
                   default="sobol",
                   help="Quasi-random sampling method")
    p.add_argument("--seed",      type=int, default=42,
                   help="Integer random seed for reproducibility")
    p.add_argument("--output_dir", default="ForwardUQ",
                   help="Output directory for plots and CSV files")
    return p.parse_args()


# =============================================================================
# Config loading
# =============================================================================

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        opts = yaml.safe_load(f)
    return opts


# =============================================================================
# Uncertainty data
# =============================================================================

def load_uncertainty_data(opts: dict, locations: dict, bin_loc: dict) -> dict:
    pkl_path = "unsrt.pkl"
    if os.path.isfile(pkl_path):
        print("  Loading uncertainty data from unsrt.pkl ...")
        with open(pkl_path, "rb") as f:
            unsrt_data = pickle.load(f)
    else:
        print("  Building uncertainty data (this may take several minutes) ...")
        dataset = uncertainty_mod.uncertaintyData(locations, bin_loc)
        unsrt_data = dataset.extract_uncertainty()
        with open(pkl_path, "wb") as f:
            pickle.dump(unsrt_data, f)
        print("  Saved to unsrt.pkl")
    print(f"  Loaded {len(unsrt_data)} reactions\n")
    return unsrt_data


# =============================================================================
# Target list
# =============================================================================

def load_targets(opts: dict, locations: dict) -> list:
    targets_count = int(opts["Counts"]["targets_count"])
    targets_file  = locations["targets"]
    addendum_file = locations["addendum"]

    addendum     = yaml.safe_load(open(addendum_file, "r").read())
    target_lines = open(targets_file, "r").readlines()

    target_list = []
    for i, line in enumerate(target_lines[:targets_count]):
        if "#" in line:
            line = line[: line.index("#")]
        add = deepcopy(addendum)
        t = combustion_target_class.combustion_target(line, add, i)
        target_list.append(t)

    T_vals = [c.temperature for c in target_list]
    print(f"  Loaded {len(target_list)} target cases  "
          f"(T: {min(T_vals):.0f}–{max(T_vals):.0f} K)\n")
    return target_list


# =============================================================================
# Flat-parameter-space index maps
# =============================================================================

def build_index_maps(unsrt_data: dict) -> tuple:
    """
    Build mappings between the flat ζ-vector and reactions/parameters.

    Returns
    -------
    flat_params_list : list of param strings, e.g. ['0_A', '1_A', ...]
    rxn_for_flat     : {flat_idx: (rxn_key, param_string)}
    flat_for_rxn     : {rxn_key: [flat_idx, ...]}
    n_full           : total number of parameters
    """
    flat_params_list = []
    rxn_for_flat     = {}
    flat_for_rxn     = {}

    flat_idx = 0
    for rxn_key, rxn_obj in unsrt_data.items():
        params = rxn_obj.activeParameters       # e.g. ['0_A'] or ['0_A','0_n','0_Ea']
        flat_for_rxn[rxn_key] = []
        for param_str in params:
            flat_params_list.append(param_str)
            rxn_for_flat[flat_idx]        = (rxn_key, param_str)
            flat_for_rxn[rxn_key].append(flat_idx)
            flat_idx += 1

    return flat_params_list, rxn_for_flat, flat_for_rxn, flat_idx


# =============================================================================
# PRS loading from cache (Path A)
# =============================================================================

def _load_selected_params(sel_file: str, flat_params_list: list) -> np.ndarray:
    """
    Build a binary selected_params mask (length n_full) from
    the selected_parameters.csv file.

    The file contains one active-parameter string per line,
    e.g. '0_A', '3_A', '7_A', ...
    """
    selected_names = set(
        line.strip() for line in open(sel_file).readlines() if line.strip()
    )
    return np.array(
        [1 if p in selected_names else 0 for p in flat_params_list],
        dtype=int,
    )


def _read_sim_data(sim_file: str) -> list:
    """Read η values from sim_data_case-N.lst (tab-delimited)."""
    vals = []
    for line in open(sim_file).readlines():
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            try:
                vals.append(float(parts[1]))
            except ValueError:
                pass
    return vals


def _read_dm(dm_file: str) -> list:
    """Read compressed design-matrix rows from pDesignMatrix.csv."""
    rows = []
    for line in open(dm_file).readlines():
        row = [float(v) for v in line.strip().split(",") if v.strip()]
        if row:
            rows.append(row)
    return rows


def load_prs_from_cache(
    case_index: int,
    unsrt_data: dict,
    target_list: list,
    flat_params_list: list,
    prs_type: str,
    rps_order: int,
    mre_criteria: float,
    workdir: str,
    opt_dir: str,
) -> "PRS_module.ResponseSurface | None":
    """
    Reconstruct a ResponseSurface from cached CSV files.

    Required files (all relative to workdir / opt_dir):
      opt_dir/Data/ResponseSurface/responsecoef_case-{i}.csv
      opt_dir/Data/Simulations/sim_data_case-{i}.lst
      workdir/DM_FOR_PARTIAL_PRS/case-{i}/pDesignMatrix.csv
      workdir/DM_FOR_PARTIAL_PRS/case-{i}/selected_parameters.csv

    Returns None if any file is absent.
    """
    coef_file = os.path.join(opt_dir,  f"Data/ResponseSurface/responsecoef_case-{case_index}.csv")
    sim_file  = os.path.join(opt_dir,  f"Data/Simulations/sim_data_case-{case_index}.lst")
    dm_file   = os.path.join(workdir,  f"DM_FOR_PARTIAL_PRS/case-{case_index}/pDesignMatrix.csv")
    sel_file  = os.path.join(workdir,  f"DM_FOR_PARTIAL_PRS/case-{case_index}/selected_parameters.csv")

    for path in [coef_file, sim_file, dm_file, sel_file]:
        if not os.path.isfile(path):
            return None

    ydata = _read_sim_data(sim_file)
    xdata = _read_dm(dm_file)

    if not xdata or not ydata:
        return None

    # Trim xdata to match ydata length (in case of partial runs)
    min_len = min(len(xdata), len(ydata))
    xdata   = xdata[:min_len]
    ydata   = ydata[:min_len]

    if min_len < 4:
        return None   # Too few samples to do train/test split

    selected_params = _load_selected_params(sel_file, flat_params_list)

    case = target_list[case_index]

    xarr = np.asarray(xdata, dtype=float)
    yarr = np.asarray(ydata, dtype=float)

    xTrain, xTest, yTrain, yTest = train_test_split(
        xarr, yarr, random_state=104, test_size=0.1, shuffle=True
    )

    # Change to Opt/ so create_response_surface() finds the cached coef file
    original_dir = os.getcwd()
    os.chdir(opt_dir)
    try:
        prs = PRS_module.ResponseSurface(
            xTrain,
            yTrain,
            case.case_index,     # e.g. 'case-0'
            case_index,          # integer index
            responseOrder=rps_order,
            selected_params=selected_params,
            prs_type=prs_type,
            criteria=mre_criteria,
        )
        prs.create_response_surface()
        # Store test data for optional accuracy reporting
        prs.xTest = xTest
        prs.yTest = yTest
    finally:
        os.chdir(original_dir)

    return prs


# =============================================================================
# comp_to_rxn mapping (for Sobol indices)
# =============================================================================

def build_comp_to_rxn(prs, rxn_for_flat: dict) -> dict:
    """
    Map each compressed-parameter index to its (rxn_key, param_string).

    prs._active_idx lists the full-space indices of selected parameters.
    """
    return {
        comp_i: rxn_for_flat[full_i]
        for comp_i, full_i in enumerate(prs._active_idx)
        if full_i in rxn_for_flat
    }


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args    = parse_args()
    workdir = os.getcwd()

    print()
    print("=" * 67)
    print("  Forward UQ Propagation via PRS Surrogates  (MUQ-SAC)")
    print("=" * 67)

    # ── Config ───────────────────────────────────────────────────────────
    opts    = load_config(args.config)
    locs    = opts["Locations"]
    bin_loc = opts["Bin"]
    stats   = opts["Stats"]

    design_type  = stats.get("Design_of_PRS", "A-facto")
    prs_type     = stats.get("PRS_type", "Partial")
    rps_order    = int(stats.get("Order_of_PRS", 2))
    mre_criteria = float(stats.get("MRE_criteria", 5.0))

    print(f"\nConfig: design={design_type!r}, PRS_type={prs_type!r}, "
          f"order={rps_order}, MRE_criteria={mre_criteria}\n")

    # ── Uncertainty data ──────────────────────────────────────────────────
    print("Step 1/5  Loading uncertainty data ...")
    unsrt_data = load_uncertainty_data(opts, locs, bin_loc)

    # ── Target list ───────────────────────────────────────────────────────
    print("Step 2/5  Loading target cases ...")
    target_list = load_targets(opts, locs)

    # ── Index maps ────────────────────────────────────────────────────────
    flat_params_list, rxn_for_flat, flat_for_rxn, n_full = build_index_maps(unsrt_data)
    print(f"  Parameter space: n_full = {n_full}  "
          f"({len(unsrt_data)} reactions)\n")

    # ── Check Opt/ directory ──────────────────────────────────────────────
    opt_dir = os.path.join(workdir, "Opt")
    if not os.path.isdir(opt_dir):
        print(
            "\nERROR: Opt/ directory not found.\n"
            "Run the optimization pipeline first:\n"
            f"    python src/MUQ-SAC/run_script.py {args.config}\n"
        )
        sys.exit(1)

    # ── Load PRS surrogates ───────────────────────────────────────────────
    print("Step 3/5  Loading PRS surrogates from cache ...")
    prs_dict = {}
    missing  = []

    for i in range(len(target_list)):
        prs = load_prs_from_cache(
            i, unsrt_data, target_list, flat_params_list,
            prs_type, rps_order, mre_criteria, workdir, opt_dir,
        )
        if prs is not None:
            prs_dict[i] = prs
            sel_flag = "OK" if getattr(prs, "selection", 1) else "WARN"
            print(f"  case-{i:03d}: n_comp={prs.n_comp:3d}  "
                  f"train_err={getattr(prs,'MaxError',float('nan')):.2f}%  "
                  f"selection={sel_flag}")
        else:
            missing.append(i)

    if missing:
        print(f"\n  WARNING: {len(missing)} cases lacked cache files: "
              f"{missing[:8]}{'...' if len(missing) > 8 else ''}")

    poor_prs = [i for i, p in prs_dict.items()
                if not getattr(p, "selection", 1)]
    if poor_prs:
        print(f"  WARNING: {len(poor_prs)} PRS surrogates have "
              f"test-error > {mre_criteria}%; results for these cases "
              f"may be less reliable.")

    if not prs_dict:
        print(
            "\nERROR: No PRS surrogates loaded.\n"
            "Run the optimization pipeline first to generate coefficient files.\n"
        )
        sys.exit(1)

    print(f"\n  Loaded {len(prs_dict)}/{len(target_list)} PRS surrogates\n")

    # ── Monte Carlo propagation ───────────────────────────────────────────
    print("Step 4/5  Monte Carlo forward UQ ...")
    propagator = MCPropagator(
        prs_dict    = prs_dict,
        unsrt_data  = unsrt_data,
        target_list = target_list,
        n_full      = n_full,
        design_type = design_type,
    )

    print(f"  Sampling prior: N={args.n_samples}, method={args.method!r}, "
          f"seed={args.seed} ...")
    zeta_matrix = propagator.sample_prior(args.n_samples, args.method, args.seed)
    n_actual    = len(zeta_matrix)
    print(f"  Sampled {n_actual} points in R^{n_full}")

    print("  Propagating through PRS surrogates (batch evaluation) ...")
    predictions = propagator.propagate(zeta_matrix)
    print(f"  Computed predictions for {len(predictions)} cases")

    print("  Computing statistics ...")
    stats_list = propagator.compute_statistics(predictions)

    # Save statistics CSV
    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv("forward_uq_statistics.csv", index=False)
    print(f"  Saved  forward_uq_statistics.csv  ({len(stats_df)} rows)")

    # Print coverage summary
    if "covered_90" in stats_df.columns and len(stats_df):
        cov90 = stats_df["covered_90"].mean() * 100
        cov50 = stats_df["covered_50"].mean() * 100
        print(f"\n  90% PI coverage: {cov90:.1f}%  of experimental data")
        print(f"  50% PI coverage: {cov50:.1f}%  of experimental data")

    # ── Analytical Sobol indices ──────────────────────────────────────────
    print("\n  Computing analytical Sobol indices ...")
    rxn_keys     = list(unsrt_data.keys())
    sobol_frames = []

    for case_index, prs in prs_dict.items():
        if int(prs.order) != 2:
            continue   # Analytical formula is for 2nd-order only
        try:
            comp_to_rxn = build_comp_to_rxn(prs, rxn_for_flat)
            calculator  = AnalyticalSobolPRS(prs, comp_to_rxn, rxn_keys)
            df          = calculator.to_dataframe(case_index)
            sobol_frames.append(df)
        except Exception as exc:
            print(f"    Sobol failed for case-{case_index}: {exc}")

    if sobol_frames:
        sobol_df = pd.concat(sobol_frames, ignore_index=True)
        sobol_df.to_csv("sobol_indices.csv", index=False)

        # Print top-5 reactions by mean S1
        top5 = (sobol_df.groupby("reaction")["S1"]
                .mean()
                .sort_values(ascending=False)
                .head(5))
        print(f"\n  Top 5 reactions by mean S1 (first-order Sobol index):")
        for rxn, s1 in top5.items():
            print(f"    {rxn:<45s}  S1={s1:.4f}")
        print(f"\n  Saved  sobol_indices.csv")
    else:
        sobol_df = pd.DataFrame()
        print("  Skipped (PRS order != 2 or no valid cases).")

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\nStep 5/5  Generating plots ...")
    out = os.path.join(workdir, args.output_dir)
    os.makedirs(out, exist_ok=True)

    plot_ign_delay_bands(predictions, target_list, out)
    plot_parity_with_uncertainty(stats_list, out)
    plot_coverage_summary(stats_list, out)

    if not sobol_df.empty:
        plot_sobol_heatmap(sobol_df, out)
        plot_reaction_uncertainty_ranking(sobol_df, out)

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print("=" * 67)
    print("  Forward UQ complete.")
    print(f"  Outputs:")
    print(f"    forward_uq_statistics.csv  — per-case UQ statistics")
    if not sobol_df.empty:
        print(f"    sobol_indices.csv          — per-case Sobol indices")
    print(f"    {args.output_dir}/             — all plots (PDF)")
    print("=" * 67)
    print()


if __name__ == "__main__":
    main()
