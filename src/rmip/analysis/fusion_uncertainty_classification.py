"""fusion_uncertainty_classification.py -- bridge from fusion's per-reaction
uncertainty bands (FusionUncertaintyEngine) to the classification CSV
format src/ActiveParameterScreen.py / src/modules/interfaces.py expect
(Reaction_ID, Reaction, Reaction_Type, Group, f).

KRUNAL_JULY_23.md point 6: "It must also generate the uncertainty data for
all the reactions. You must provide an output file so it can be used as
input to uncertainty file generator. The format must match what the
updated Uncertainty Generator requires." Point 5's rework made that
"updated Uncertainty Generator" mean src/ActiveParameterScreen.py's
pipeline, whose input contract is this classification CSV (verified
against the real DATASET/METHYL_BUTANOATE_2026/.../results/
master_active_reactions_CLASSIFIED.csv fixture and
src/modules/interfaces.py::load_classification_file's required columns).

FusionUncertaintyRecord bands one factor per grid temperature, not one
factor per HTC/LTC-style group the classification format expects -- this
module splits the record's T_grid at a configurable boundary and reports
the worst-case (max) factor within each half as that group's `f`, so a
downstream user sees the SAME "high/low temperature chemistry" grouping
convention as a hand-classified SA screening would produce.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional

from rmip.analysis.fusion_uncertainty import FusionUncertaintyRecord

_COLUMNS = ["Reaction_ID", "Reaction", "Reaction_Type", "Group", "f"]


def _reaction_type(rec: FusionUncertaintyRecord) -> str:
    return "Falloff/PLOG" if rec.high_p_limit_only else "Elementary"


def _group_factor(rec: FusionUncertaintyRecord, T_lo: float, T_hi: float) -> Optional[float]:
    vals = [f for T, f in zip(rec.T_grid_K, rec.f_T) if T_lo <= T < T_hi]
    return max(vals) if vals else None


def build_classification_rows(
    records: List[FusionUncertaintyRecord],
    split_K: float = 1000.0,
    decimals: int = 4,
) -> List[dict]:
    """One row per (reaction, HTC/LTC group) with a nonzero grid overlap.
    Reaction_ID is a stable 1-based sequence over the input record order
    (this bridge has no independent mechanism-DB reaction id to reuse --
    ActiveParameterScreen's own contract only requires Reaction_ID be a
    unique integer key, not that it trace back to any particular source)."""
    rows: List[dict] = []
    next_id = 1
    for rec in records:
        rtype = _reaction_type(rec)
        for group, (T_lo, T_hi) in (("LTC", (0.0, split_K)),
                                     ("HTC", (split_K, float("inf")))):
            f = _group_factor(rec, T_lo, T_hi)
            if f is None:
                continue
            rows.append({
                "Reaction_ID": next_id,
                "Reaction": rec.formula_string,
                "Reaction_Type": rtype,
                "Group": group,
                "f": round(f, decimals),
            })
            next_id += 1
    return rows


def write_classification_csv(
    records: List[FusionUncertaintyRecord],
    path: str,
    split_K: float = 1000.0,
) -> str:
    rows = build_classification_rows(records, split_K=split_K)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return str(out)
