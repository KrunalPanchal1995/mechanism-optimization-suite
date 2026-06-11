"""
error_metrics.py
----------------
Computes the weighted chi-squared error function for each simulation stage
and writes per-condition error CSV files that mirror the merge output structure.

Error function
--------------
The error metric is the weighted sum of squares:

    ε = Σᵢ [ (sim_i − obs_i) / σᵢ ]²

where  σᵢ = uncertainty_fraction × obs_i  (default 10 %).

Output format
-------------
Each error CSV contains two columns::

    Stage,Error_Function
    Nominal,97.56
    Stage-1,82.96
    Stage-2,77.64

The set of ``Stage`` rows present reflects which simulation columns exist in
the corresponding merged CSV (e.g. a ``nominal_s1`` merged CSV produces only
``Nominal`` and ``Stage-1`` rows).

Output layout
-------------
The error output tree mirrors the merge output tree exactly::

    error_root/
    ├── all/
    │   ├── phi_0_5/
    │   │   ├── Phi_0_5_P_10.csv
    │   │   └── Phi_0_5_P_20.csv
    │   ├── phi_1_0/
    │   └── phi_2_0/
    ├── nominal_s1/
    └── nominal_s2/

Public API
----------
chi_squared(pred, obs, unc)                           — scalar error
compute_errors_from_df(df, obs_col, unc_frac)         — dict of {stage: ε}
generate_error_csvs(merge_root, error_root, unc_frac) — full folder walk
summarise_errors(error_root)                          — summary DataFrame
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Stage column names checked in order (order determines CSV row order)
_STAGE_COLS = ("Nominal", "Stage-1", "Stage-2")


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def chi_squared(
    pred: np.ndarray,
    obs:  np.ndarray,
    unc:  np.ndarray,
) -> float:
    """
    Compute the weighted sum-of-squares error:

        ε = Σ [ (pred_i − obs_i) / unc_i ]²

    ``NaN`` entries in *pred* are silently excluded (they represent failed
    simulations).

    Parameters
    ----------
    pred, obs, unc : np.ndarray
        1-D arrays of equal length.

    Returns
    -------
    float
        The error value, or ``nan`` if all *pred* entries are ``NaN``.
    """
    mask = ~np.isnan(pred)
    if not mask.any():
        return float("nan")
    err = (pred[mask] - obs[mask]) / unc[mask]
    return float(err @ err)


def compute_errors_from_df(
    df:                  pd.DataFrame,
    obs_col:             str   = "Obs(us)",
    uncertainty_fraction: float = 0.10,
) -> Dict[str, float]:
    """
    Compute the chi-squared error for every simulation-stage column present
    in *df*.

    The IDT values are expected to be in the **same unit** in both the
    observed and simulated columns (the unit cancels in the ratio).

    Parameters
    ----------
    df : pd.DataFrame
        A merged IDT CSV loaded into a DataFrame.
    obs_col : str
        Name of the observed IDT column (default ``"Obs(us)"``).
    uncertainty_fraction : float
        σ = fraction × obs  (default 0.10 → 10 %).

    Returns
    -------
    dict
        ``{stage_label: chi_sq_value}`` for every stage column found in *df*.
        Empty dict if *obs_col* is absent or no stage columns are present.
    """
    if obs_col not in df.columns:
        logger.warning("Observed column '%s' not found in DataFrame.", obs_col)
        return {}

    obs = df[obs_col].to_numpy(dtype=float)
    unc = uncertainty_fraction * obs

    return {
        col: chi_squared(df[col].to_numpy(dtype=float), obs, unc)
        for col in _STAGE_COLS
        if col in df.columns
    }


# ---------------------------------------------------------------------------
# Folder-level processing
# ---------------------------------------------------------------------------

def generate_error_csvs(
    merge_root:           Path,
    error_root:           Path,
    uncertainty_fraction:  float = 0.10,
    obs_col:              str   = "Obs(us)",
) -> None:
    """
    Walk *merge_root*, compute chi-squared errors for every merged CSV, and
    write mirror error CSVs under *error_root*.

    Parameters
    ----------
    merge_root : Path
        Root of the merge output (produced by :func:`~mso_compare.merge.merge_simulation_results`).
    error_root : Path
        Root where error CSVs will be written (mirrors *merge_root* layout).
    uncertainty_fraction : float
        σ = fraction × obs  (default 0.10).
    obs_col : str
        Name of the observed IDT column in the merged CSVs.
    """
    if not merge_root.exists():
        logger.warning(
            "Merge root does not exist: %s — skipping error-metric generation.",
            merge_root,
        )
        return

    csv_files = sorted(merge_root.rglob("*.csv"))
    if not csv_files:
        logger.warning("No CSV files found under: %s", merge_root)
        return

    logger.info(
        "Computing error metrics for %d file(s) under %s …",
        len(csv_files), merge_root,
    )

    n_written = 0
    for csv_path in csv_files:
        rel_path   = csv_path.relative_to(merge_root)
        error_path = error_root / rel_path

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            logger.error("Failed to read %s: %s", csv_path, exc)
            continue

        errors = compute_errors_from_df(
            df,
            obs_col=obs_col,
            uncertainty_fraction=uncertainty_fraction,
        )

        if not errors:
            logger.warning(
                "No computable stage columns in %s — error CSV not written.", csv_path
            )
            continue

        # Write error CSV
        error_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [(stage, f"{val:.6f}") for stage, val in errors.items()]
        error_df = pd.DataFrame(rows, columns=["Stage", "Error_Function"])
        error_df.to_csv(error_path, index=False)
        n_written += 1
        logger.debug("  Error CSV → %s", error_path)

    logger.info("✓ Error metrics written: %d file(s) → %s", n_written, error_root)


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

def summarise_errors(error_root: Path) -> pd.DataFrame:
    """
    Collect all error CSVs under *error_root* into a single summary DataFrame.

    Useful for printing or exporting an overview of all ε values.

    Returns
    -------
    pd.DataFrame
        Columns: ``comparison_set``, ``phi_folder``, ``filename``,
        ``stage``, ``error_function``.
    """
    records = []
    for csv_path in sorted(error_root.rglob("*.csv")):
        rel   = csv_path.relative_to(error_root)
        parts = rel.parts          # e.g. ("all", "phi_0_5", "Phi_0_5_P_10.csv")

        comparison_set = parts[0] if len(parts) >= 1 else ""
        phi_folder     = parts[1] if len(parts) >= 2 else ""
        fname          = parts[-1]

        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                records.append(
                    {
                        "comparison_set": comparison_set,
                        "phi_folder":     phi_folder,
                        "filename":       fname,
                        "stage":          row["Stage"],
                        "error_function": float(row["Error_Function"]),
                    }
                )
        except Exception as exc:
            logger.warning("Skipping %s: %s", csv_path, exc)

    return pd.DataFrame(records)
