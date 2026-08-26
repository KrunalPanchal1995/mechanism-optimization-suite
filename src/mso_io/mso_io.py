"""
mso_io.py — Input/Output Parsers
==================================
Handles all reading of:
  • YAML metadata
  • Error CSV files  (one file per φ/P condition, from a folder)
  • Cost CSV files   (Case_ID, Total_Rxns, Active_Rxns, p-PRS_Cost, Full_PRS_Cost)
  • PRS stats CSV    (Case_ID, Training_MRE, Testing_MRE, Threshold)
  • Sensitivity CSV  (optional)
  • Convergence CSV  (optional)
"""

import re
import warnings
from pathlib import Path

import pandas as pd
import yaml


# ══════════════════════════════════════════════════════════════════════════════
# YAML metadata
# ══════════════════════════════════════════════════════════════════════════════

def parse_yaml_metadata(path: str) -> dict:
    """Load fuel/study metadata from a YAML file."""
    p = Path(path)
    if not p.exists():
        warnings.warn(f"Metadata YAML not found: {path}. Using empty dict.")
        return {}
    with open(p, 'r', encoding='utf-8') as f:
        meta = yaml.safe_load(f)
    return meta or {}


# ══════════════════════════════════════════════════════════════════════════════
# Filename → (φ, P) extraction
# ══════════════════════════════════════════════════════════════════════════════

# Patterns supported (case-insensitive):
#   phi_(0_5)_P_(10).csv
#   PHI_0_5_p_10.csv
#   phi_(2_0)_P_(40).csv
#   phi_(0.5)_P_(10).csv    ← dot already in place
_PHI_P_RE = re.compile(
    r'(?i)'
    r'phi[\s_\(]*([0-9][0-9_\.]*)[\s_\)]*'   # group 1 → phi value
    r'p[\s_\(]*([0-9]+)'                        # group 2 → P value
)

def extract_phi_p(filename: str):
    """
    Extract (phi, P) from a filename.

    Examples
    --------
    'phi_(0_5)_P_(10).csv'   → (0.5, 10)
    'PHI_2_0_p_40.csv'       → (2.0, 40)
    'phi_(0.5)_P_(10).csv'   → (0.5, 10)
    """
    stem = Path(filename).stem
    m = _PHI_P_RE.search(stem)
    if not m:
        raise ValueError(
            f"Cannot extract φ / P from filename: '{filename}'.\n"
            f"  Expected format: phi_(X_X)_P_(XX) (case-insensitive).")
    phi_raw = m.group(1).replace('_', '.')
    phi_raw = phi_raw.strip('.')          # remove trailing dot (e.g. '2.')
    p_raw   = m.group(2)
    return float(phi_raw), int(p_raw)


# ══════════════════════════════════════════════════════════════════════════════
# Error CSV folder parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_error_csvs_from_folder(folder: str) -> dict:
    """
    Read all *.csv files from *folder*.  Each file must represent one
    (φ, P) condition and contain two columns:

        col-1  stage label   (e.g. "Nominal", "Stage-1", "Stage-2")
        col-2  ε value       (numeric)

    Returns
    -------
    dict  {(phi, P): {stage_label: eps_value}}

    The stage labels are taken directly from column-1 of each CSV so the
    user controls them completely.
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Error CSV folder not found: {folder}")

    csv_files = sorted(
        list(folder_path.glob('*.csv')) +
        list(folder_path.glob('*.CSV'))
    )
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {folder}")

    data = {}
    for f in csv_files:
        phi, p = extract_phi_p(f.name)
        df = pd.read_csv(f, header=0)
        if df.shape[1] < 2:
            raise ValueError(
                f"Error CSV must have ≥2 columns (label, ε). File: {f}")
        col_label = df.columns[0]
        col_eps   = df.columns[1]
        cond = {}
        for _, row in df.iterrows():
            label = str(row[col_label]).strip()
            try:
                cond[label] = float(row[col_eps])
            except ValueError:
                pass  # skip header-like rows if any
        data[(phi, p)] = cond

    return data


# ══════════════════════════════════════════════════════════════════════════════
# Cost CSV parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_cost_csv(path: str) -> pd.DataFrame:
    """
    Load cost CSV with expected columns:
        Case_ID, Total_Rxns, Active_Rxns, p-PRS_Cost, Full_PRS_Cost

    Returns cleaned DataFrame (duplicates dropped).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Cost CSV not found: {path}")
    df = pd.read_csv(p)
    df = df.drop_duplicates().reset_index(drop=True)

    required = {'Total_Rxns', 'Active_Rxns', 'p-PRS_Cost', 'Full_PRS_Cost'}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Cost CSV '{path}' is missing columns: {missing}.\n"
            f"  Found: {list(df.columns)}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PRS statistics CSV parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_prs_stats_csv(path: str) -> pd.DataFrame:
    """
    Load PRS statistics CSV with expected columns:
        Case_ID, Training_MRE, Testing_MRE, Threshold

    MRE values should be in percent (%).

    Returns cleaned DataFrame.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PRS stats CSV not found: {path}")
    df = pd.read_csv(p)
    df = df.drop_duplicates().reset_index(drop=True)

    # Flexible column matching (case-insensitive)
    col_map = {}
    for col in df.columns:
        cl = col.lower().replace(' ', '_').replace('-', '_')
        if 'train' in cl and 'mre' in cl:
            col_map['Training_MRE'] = col
        elif 'test' in cl and 'mre' in cl:
            col_map['Testing_MRE'] = col
        elif 'threshold' in cl:
            col_map['Threshold'] = col
        elif 'case' in cl or 'id' in cl:
            col_map['Case_ID'] = col

    # Rename to standard names
    df = df.rename(columns={v: k for k, v in col_map.items()})

    if 'Training_MRE' not in df.columns or 'Testing_MRE' not in df.columns:
        raise ValueError(
            f"PRS stats CSV must have Training_MRE and Testing_MRE columns.\n"
            f"  Found: {list(df.columns)}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Sensitivity CSV parser  (optional, for S1 heatmap)
# ══════════════════════════════════════════════════════════════════════════════

def parse_sensitivity_csv(path: str) -> pd.DataFrame:
    """
    Load sensitivity coefficients CSV.
    Expected columns: Reaction_ID (or similar), then one column per condition
    (e.g. phi05_P10, phi10_P20, ...) containing |S_i| values.

    Alternatively:  Reaction_ID, Phi, Pressure, Sensitivity

    The function auto-detects wide vs. long format.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Sensitivity CSV not found: {path}")
    df = pd.read_csv(p)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Convergence CSV parser  (optional, for S4 plot)
# ══════════════════════════════════════════════════════════════════════════════

def parse_convergence_csv(path: str) -> pd.DataFrame:
    """
    Load optimizer convergence history.
    Expected columns: Iteration, Best_Objective  (and optionally Stage)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Convergence CSV not found: {path}")
    df = pd.read_csv(p)
    return df
