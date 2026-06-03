#!/usr/bin/env python3
"""
modules/interfaces.py

CANONICAL DATA CONTRACTS for the Active Parameter Framework.

Every module in this package agrees on the schemas defined here.
This file is the single source of truth; if two modules disagree
about a column name or dict structure, this file wins.

--------------------------------------------------------------------
1. SENSITIVITY DATABASE  (nested-dict form, used for plotting/lookup)
--------------------------------------------------------------------
    sens_db[temperature][reaction_id] = sensitivity   (float)

    - temperature : float  (Kelvin)
    - reaction_id : int    (1-based, == YAML index + 1)
    - sensitivity : float

--------------------------------------------------------------------
2. DETAILED RECORDS  (flat list of dicts, used for screening)
--------------------------------------------------------------------
    Each record is a dict with EXACTLY these keys:
        File           : str
        Case_ID        : int | None
        Temperature    : float | None
        Reaction_ID    : int
        Reaction       : str     (equation string as it appears in SA file)
        Sensitivity    : float
        Abs_Sensitivity: float

--------------------------------------------------------------------
3. REACTION LOOKUP  (from the YAML mechanism)
--------------------------------------------------------------------
    lookup[reaction_id] = {
        "Reaction_ID"  : int,
        "YAML_Index"   : int,    (0-based)
        "Reaction"     : str,    (equation from YAML)
        "Reaction_Type": str,    (Elementary/Duplicate/PLOG/PLOG-Duplicate/
                                  ThirdBody/Falloff/Chebyshev)
    }

    NOTE: keys are Capitalized. Do NOT use "equation"/"type"/"yaml_index".

--------------------------------------------------------------------
4. MASTER (SCREENING OUTPUT)  -- this is a RESULT, not an input
--------------------------------------------------------------------
    Columns:
        Reaction_ID, Reaction, Reaction_Type,
        Maximum_Abs_Sensitivity, Occurrences
    Reaction_ID is stored as a plain int.

--------------------------------------------------------------------
5. CLASSIFICATION FILE  (USER-SUPPLIED INPUT for XML / LaTeX / plots)
--------------------------------------------------------------------
    Required columns : Reaction_ID, Reaction, Group, f
    Optional columns : data_type, Reaction_Type, RxnCount

    - Reaction_ID : int (no "R" prefix internally)
    - Group       : raw label; normalize via normalize_group()
    - f           : uncertainty factor (the "Unsrt" column is an accepted
                    alias and is renamed to f on load)
    - data_type   : defaults to "constant;end_points" when absent/blank

    This file is DISTINCT from the screening output (#4). Screening tells
    you which reactions matter; the classification file is how YOU assign
    groups and uncertainty factors before generating XML/tables/plots.
"""

from __future__ import annotations

import re
import pandas as pd

DEFAULT_DATA_TYPE = "constant;end_points"

# Column name constants (use these instead of string literals everywhere)
COL_RID = "Reaction_ID"
COL_RXN = "Reaction"
COL_TYPE = "Reaction_Type"
COL_GROUP = "Group"
COL_F = "f"
COL_DATA_TYPE = "data_type"
COL_RXNCOUNT = "RxnCount"


# ============================================================
# Group normalization  (single canonical implementation)
# ============================================================

_HTC_ALIASES = {
    "htc", "high", "high temperature",
    "high_temperature", "high-temperature",
}
_LTC_ALIASES = {
    "ltc", "low", "low temperature",
    "low_temperature", "low-temperature",
}


def normalize_group(value):
    """Map any accepted HTC/LTC label to 'HTC' / 'LTC'.

    Returns 'UNKNOWN' for None/NaN, otherwise the upper-cased original
    string when it matches neither alias set.
    """
    if value is None:
        return "UNKNOWN"
    try:
        if pd.isna(value):
            return "UNKNOWN"
    except (TypeError, ValueError):
        pass

    v = str(value).strip().lower()
    if v in _HTC_ALIASES:
        return "HTC"
    if v in _LTC_ALIASES:
        return "LTC"
    return str(value).strip().upper()


# ============================================================
# Species extraction  (single canonical implementation)
# ============================================================

_STOICH_RE = re.compile(r"^\d+(\.\d+)?\s*")


def extract_species_from_equation(equation):
    """Return the set of species names in a reaction equation.

    Handles '<=>', '=>', '<=' and strips leading stoichiometric
    coefficients (e.g. '2 H2O' -> 'H2O'). Drops the '(+M)' / '+M'
    third-body markers and bare 'M'.
    """
    eq = (
        str(equation)
        .replace("<=>", "=")
        .replace("=>", "=")
        .replace("<=", "=")
    )
    # remove falloff third-body markers like (+M), (+ H2O)
    eq = re.sub(r"\(\+\s*[^)]*\)", "", eq)

    species = set()
    for side in eq.split("="):
        for term in side.split("+"):
            term = term.strip()
            term = _STOICH_RE.sub("", term).strip()
            if term and term != "M":
                species.add(term)
    return species


# ============================================================
# Classification-file loader  (the user-supplied input, #5)
# ============================================================

def load_classification_file(path):
    """Load and validate the user-supplied classification CSV.

    - Accepts 'Unsrt' as an alias for 'f'.
    - Adds data_type='constant;end_points' when missing/blank.
    - Coerces Reaction_ID to int (stripping any leading 'R').
    - Adds normalized 'Group_norm' column.

    Returns a validated DataFrame.
    """
    df = pd.read_csv(path)

    # Accept 'Temperature_Regime' as a Group alias
    if COL_GROUP not in df.columns and "Temperature_Regime" in df.columns:
        df[COL_GROUP] = df["Temperature_Regime"]

    # f / Unsrt handling
    if COL_F not in df.columns:
        if "Unsrt" in df.columns:
            df[COL_F] = df["Unsrt"]
        else:
            raise ValueError(
                f"Classification file '{path}' must contain an "
                f"'f' or 'Unsrt' column."
            )

    for required in (COL_RID, COL_RXN, COL_GROUP):
        if required not in df.columns:
            raise ValueError(
                f"Classification file '{path}' missing required "
                f"column '{required}'. Found: {list(df.columns)}"
            )

    # data_type defaulting
    if COL_DATA_TYPE not in df.columns:
        df[COL_DATA_TYPE] = DEFAULT_DATA_TYPE
    else:
        df[COL_DATA_TYPE] = (
            df[COL_DATA_TYPE]
            .fillna(DEFAULT_DATA_TYPE)
            .replace("", DEFAULT_DATA_TYPE)
        )

    # Reaction_ID -> int (strip leading 'R' if present)
    df[COL_RID] = (
        df[COL_RID]
        .astype(str)
        .str.strip()
        .str.replace(r"^[Rr]", "", regex=True)
        .astype(int)
    )

    df["Group_norm"] = df[COL_GROUP].apply(normalize_group)
    return df
