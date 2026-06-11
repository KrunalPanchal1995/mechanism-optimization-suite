"""
merge.py
--------
Merges IDT simulation CSV results from up to three mechanism runs and
writes an organised output structure.

Output structure
----------------
After merging, files are organised by:

1. **Comparison set** — which simulation stages are present:
   ``all``          (Nominal + Stage-1 + Stage-2)
   ``nominal_s1``   (Nominal + Stage-1)
   ``nominal_s2``   (Nominal + Stage-2)
   ``nominal_only`` (Nominal only — no optimised stages)

2. **φ sub-folder** — e.g. ``phi_0_5``, ``phi_1_0``, ``phi_2_0``

3. **Standard filename** — e.g. ``Phi_0_5_P_10.csv``

Column layout in output CSVs
-----------------------------
``DS_ID, T, Obs(us), Nominal [, Stage-1] [, Stage-2]``

Public API
----------
merge_simulation_results(nominal_folder, stage1_folder, stage2_folder,
                         output_root)

Bug fixes over original merge_idt_results.py
---------------------------------------------
* Filename parser uses a robust regex (not fragile ``split("_phi")``).
* Missing counterpart files in stage folders only emit a warning; the
  remaining common files are still processed.
* Output directories are created automatically.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .utils import parse_phi_pressure, phi_folder_name, standardise_filename

logger = logging.getLogger(__name__)

# The column name used for the simulated IDT in every input CSV
_SIM_COL = "Nominal"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_csvs(folder: Path) -> Dict[str, pd.DataFrame]:
    """
    Load all ``.csv`` files in *folder*.

    Returns
    -------
    dict
        ``{filename: DataFrame}``

    Raises
    ------
    FileNotFoundError
        If no CSV files are found.
    """
    csvs = sorted(folder.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in: {folder}")
    return {f.name: pd.read_csv(f) for f in csvs}


def _left_join_stage(
    base_df:   pd.DataFrame,
    stage_df:  pd.DataFrame,
    col_name:  str,
    join_key:  str = "DS_ID",
) -> pd.DataFrame:
    """
    Left-join the simulated IDT column from *stage_df* onto *base_df*,
    renaming it to *col_name*.

    Both DataFrames must contain *join_key* and ``Nominal`` columns.
    """
    slim = (
        stage_df[[join_key, _SIM_COL]]
        .rename(columns={_SIM_COL: col_name})
    )
    return base_df.merge(slim, on=join_key, how="left")


def _save(df: pd.DataFrame, path: Path) -> None:
    """Write *df* as CSV to *path*, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("    Saved → %s", path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def merge_simulation_results(
    nominal_folder: Path,
    stage1_folder:  Optional[Path],
    stage2_folder:  Optional[Path],
    output_root:    Path,
    join_key:       str = "DS_ID",
) -> None:
    """
    Merge simulation CSV results and write an organised output structure.

    Parameters
    ----------
    nominal_folder : Path
        Folder containing per-dataset CSVs from the nominal mechanism run.
        Each CSV must have the columns: ``DS_ID``, ``T``, ``Obs(us)``, ``Nominal``.
    stage1_folder : Path | None
        Folder for Stage-1 optimised mechanism results.  Pass ``None`` if
        Stage-1 results are not available.
    stage2_folder : Path | None
        Folder for Stage-2 optimised mechanism results.  Pass ``None`` if
        Stage-2 results are not available.
    output_root : Path
        Root directory where merged outputs will be written.  Sub-folders
        ``all/``, ``nominal_s1/``, ``nominal_s2/`` (or ``nominal_only/``)
        are created automatically under this root.
    join_key : str
        Column used to align rows across the three DataFrames (default ``DS_ID``).

    Output structure::

        output_root/
        ├── all/
        │   ├── phi_0_5/
        │   │   ├── Phi_0_5_P_10.csv
        │   │   └── Phi_0_5_P_20.csv
        │   ├── phi_1_0/
        │   └── phi_2_0/
        ├── nominal_s1/
        │   └── phi_*/...
        ├── nominal_s2/
        │   └── phi_*/...
        └── (or nominal_only/ if no stage folders given)
    """
    # ── Load CSV files ─────────────────────────────────────────────────────
    logger.info("Loading nominal CSVs from: %s", nominal_folder)
    nominal_files: Dict[str, pd.DataFrame] = _load_csvs(nominal_folder)

    stage1_files: Dict[str, pd.DataFrame] = {}
    stage2_files: Dict[str, pd.DataFrame] = {}

    if stage1_folder and stage1_folder.exists():
        logger.info("Loading Stage-1 CSVs from: %s", stage1_folder)
        stage1_files = _load_csvs(stage1_folder)
    elif stage1_folder:
        logger.warning("Stage-1 folder not found: %s", stage1_folder)

    if stage2_folder and stage2_folder.exists():
        logger.info("Loading Stage-2 CSVs from: %s", stage2_folder)
        stage2_files = _load_csvs(stage2_folder)
    elif stage2_folder:
        logger.warning("Stage-2 folder not found: %s", stage2_folder)

    # ── Find common filenames ──────────────────────────────────────────────
    common = set(nominal_files)

    if stage1_files:
        only_in_nominal = common - set(stage1_files)
        if only_in_nominal:
            logger.warning(
                "Files present in nominal but missing in Stage-1 (will be skipped): %s",
                sorted(only_in_nominal),
            )
        common &= set(stage1_files)

    if stage2_files:
        only_before_s2 = common - set(stage2_files)
        if only_before_s2:
            logger.warning(
                "Files present in nominal/Stage-1 but missing in Stage-2 (will be skipped): %s",
                sorted(only_before_s2),
            )
        common &= set(stage2_files)

    if not common:
        raise ValueError(
            "No matching CSV filenames found across the provided input folders."
        )

    logger.info("Merging %d file(s) …", len(common))

    # ── Process each file ──────────────────────────────────────────────────
    for fname in sorted(common):
        # Parse φ and P from the original dataset-ID filename
        try:
            phi_str, pres_str = parse_phi_pressure(fname)
        except ValueError:
            logger.warning("Cannot parse φ/P from '%s' — skipped.", fname)
            continue

        std_fname  = standardise_filename(fname)      # e.g. "Phi_0_5_P_10.csv"
        phi_folder = phi_folder_name(phi_str)         # e.g. "phi_0_5"

        logger.info("  %-50s → %s / %s", fname, phi_folder, std_fname)

        nominal_df = nominal_files[fname].rename(columns={_SIM_COL: "Nominal"})

        # ── nominal_only ──────────────────────────────────────────────────
        if not stage1_files and not stage2_files:
            _save(
                nominal_df,
                output_root / "nominal_only" / phi_folder / std_fname,
            )
            continue

        # ── Build all three merged frames in one pass ──────────────────────
        if stage1_files and stage2_files:
            merged = (
                nominal_df
                .pipe(_left_join_stage, stage1_files[fname], "Stage-1", join_key)
                .pipe(_left_join_stage, stage2_files[fname], "Stage-2", join_key)
            )
            _save(merged,                                   output_root / "all"        / phi_folder / std_fname)
            _save(merged.drop(columns=["Stage-2"]),         output_root / "nominal_s1" / phi_folder / std_fname)
            _save(merged.drop(columns=["Stage-1"]),         output_root / "nominal_s2" / phi_folder / std_fname)

        elif stage1_files:
            merged = nominal_df.pipe(_left_join_stage, stage1_files[fname], "Stage-1", join_key)
            _save(merged, output_root / "nominal_s1" / phi_folder / std_fname)

        elif stage2_files:
            merged = nominal_df.pipe(_left_join_stage, stage2_files[fname], "Stage-2", join_key)
            _save(merged, output_root / "nominal_s2" / phi_folder / std_fname)

    logger.info("✓ Merge complete. Output root: %s", output_root)
