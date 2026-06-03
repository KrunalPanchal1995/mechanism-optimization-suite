#!/usr/bin/env python3
"""
modules/sensitivity_parser.py

Parses SA files into the TWO canonical forms defined in interfaces.py:

  1. build_sensitivity_database()  -> sens_db[T][rxn_id] = sensitivity
                                       (for plotting / fast lookup)
  2. build_detailed_records()      -> list[dict] per contract #2
                                       (for screening)

Both are produced from the same single pass over the files via
_read_all(), so there is no second parsing step and the two views
can never drift apart.

CHANGES vs original:
- Single regex (case-insensitive) for temperature, shared with the
  plotter. The old plotter had its own non-case-insensitive copy.
- read_sensitivity_file now also returns Abs_Sensitivity.
"""

import os
import re
from collections import defaultdict

import pandas as pd

try:
    from tqdm import tqdm
except Exception:                      # pragma: no cover
    def tqdm(x, **kwargs):
        return x


_CASE_RE = re.compile(r"case[_\-]?(\d+)", re.I)
_TEMP_RE = re.compile(r"T[_\-]?(\d+(?:\.\d+)?)", re.I)


def parse_filename(filename):
    """(case_id|None, temperature|None) from an SA filename."""
    cm = _CASE_RE.search(filename)
    tm = _TEMP_RE.search(filename)
    case_id = int(cm.group(1)) if cm else None
    temperature = float(tm.group(1)) if tm else None
    return case_id, temperature


def read_sensitivity_file(filepath):
    """Read one SA file: columns Sensitivity, RxnID, Reaction.

    Robust to a leading tab and to runs of consecutive tabs (some Cantera
    SA dumps prefix every data row with a tab, which would otherwise make
    the first split field empty and silently drop the whole file). Empty
    fields are discarded before parsing, so the first three *non-empty*
    fields are taken as sensitivity, reaction id, and reaction.

    The first line is treated as a header and skipped.

    Returns list[dict] with Reaction_ID, Reaction, Sensitivity, Abs_Sensitivity.
    """
    rows = []
    with open(filepath, "r") as f:
        lines = f.readlines()

    for line in lines[1:]:
        # drop empty fields produced by leading/duplicate tabs
        parts = [p for p in line.rstrip("\n").split("\t") if p.strip() != ""]
        if len(parts) < 3:
            continue
        try:
            sensitivity = float(parts[0])
            reaction_id = int(parts[1])
            # reaction may itself contain tabs/spaces; rejoin the remainder
            reaction = " ".join(parts[2:]).strip()
        except (ValueError, IndexError):
            continue
        rows.append({
            "Reaction_ID": reaction_id,
            "Reaction": reaction,
            "Sensitivity": sensitivity,
            "Abs_Sensitivity": abs(sensitivity),
        })
    return rows


def _list_txt(directory):
    return sorted(f for f in os.listdir(directory) if f.endswith(".txt"))


def _read_all(sensitivity_directory, desc="Reading SA files"):
    """Single pass: yields (filename, case_id, temperature, rows)."""
    for filename in tqdm(_list_txt(sensitivity_directory), desc=desc):
        case_id, temperature = parse_filename(filename)
        rows = read_sensitivity_file(
            os.path.join(sensitivity_directory, filename)
        )
        yield filename, case_id, temperature, rows


# ============================================================
# Canonical form #1 : nested dict
# ============================================================

def build_sensitivity_database(sensitivity_directory):
    """sens_db[temperature][reaction_id] = sensitivity.

    WARNING: this view collapses cases. If two files share the same
    temperature, the later one overwrites the earlier. Use
    build_case_database() when you need per-case data.
    """
    database = defaultdict(dict)
    for _fn, _cid, temperature, rows in _read_all(sensitivity_directory):
        if temperature is None:
            continue
        for row in rows:
            database[temperature][row["Reaction_ID"]] = row["Sensitivity"]
    return dict(database)


def build_case_database(sensitivity_directory):
    """case_db[(case_id, temperature)][reaction_id] = sensitivity.

    Preserves every (case, T) combination separately, so multiple cases
    at the same temperature do not overwrite each other. case_id is the
    integer parsed from the filename, or None if absent.
    """
    database = defaultdict(dict)
    for _fn, case_id, temperature, rows in _read_all(
            sensitivity_directory, desc="Building case database"):
        if temperature is None:
            continue
        key = (case_id, temperature)
        for row in rows:
            database[key][row["Reaction_ID"]] = row["Sensitivity"]
    return dict(database)


def get_case_temperature_pairs(case_db):
    """Sorted list of (case_id, temperature) keys.

    Sorts by temperature first, then case_id, with None case_ids last.
    """
    return sorted(
        case_db.keys(),
        key=lambda k: (k[1], (k[0] is None, k[0])),
    )


# ============================================================
# Canonical form #2 : flat detailed records
# ============================================================

def build_detailed_records(sensitivity_directory):
    """list[dict] following interfaces.py contract #2."""
    records = []
    for filename, case_id, temperature, rows in _read_all(
            sensitivity_directory, desc="Building detailed records"):
        for row in rows:
            records.append({
                "File": filename,
                "Case_ID": case_id,
                "Temperature": temperature,
                "Reaction_ID": row["Reaction_ID"],
                "Reaction": row["Reaction"],
                "Sensitivity": row["Sensitivity"],
                "Abs_Sensitivity": row["Abs_Sensitivity"],
            })
    return records


def build_detailed_dataframe(sensitivity_directory):
    """Same as build_detailed_records but as a DataFrame."""
    return pd.DataFrame(build_detailed_records(sensitivity_directory))


# ============================================================
# Small helpers
# ============================================================

def get_available_temperatures(sens_db):
    return sorted(sens_db.keys())


def export_sensitivity_database(sens_db, outfile):
    rows = []
    for T in sorted(sens_db.keys()):
        for rxn_id, sens in sens_db[T].items():
            rows.append({
                "Temperature": T, "Reaction_ID": rxn_id, "Sensitivity": sens,
            })
    pd.DataFrame(rows).to_csv(outfile, index=False)


def summarize_database(sens_db):
    temps = sorted(sens_db.keys())
    reactions = set()
    for T in temps:
        reactions.update(sens_db[T].keys())
    print("\nSensitivity Database Summary")
    print("-" * 40)
    print(f"Temperatures : {len(temps)}")
    print(f"Reactions    : {len(reactions)}")
    if temps:
        print(f"T Range      : {min(temps)} - {max(temps)} K")
