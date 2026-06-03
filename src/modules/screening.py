#!/usr/bin/env python3
"""
modules/screening.py

Core reaction screening.

CHANGES vs original:
- Consumes the canonical DETAILED RECORDS list (interfaces contract #2),
  NOT a {filename: [rows]} dict. The original iterated the parser's
  nested {T: {rxn_id: sens}} dict as if it were {filename: [rows]},
  which raised TypeError.
- reaction_lookup is read with Capitalized keys (Reaction / Reaction_Type
  / YAML_Index) matching mechanism_utils.build_reaction_lookup.
- species extraction delegated to interfaces (no local duplicate).

Returns a ScreeningResult with five components:
    detailed_rows, master_rows, species_rows, summary_rows, mismatch_log
"""

from collections import Counter, namedtuple

from .interfaces import extract_species_from_equation
from .mechanism_utils import reaction_passes_carbon_filter

ScreeningResult = namedtuple(
    "ScreeningResult",
    ["detailed_rows", "master_rows", "species_rows",
     "summary_rows", "mismatch_log"],
)


def run_screening(detailed_records, reaction_lookup, carbon_map,
                  threshold=0.05, carbon_threshold=0):
    """Screen detailed SA records.

    Parameters
    ----------
    detailed_records : list[dict]
        From sensitivity_parser.build_detailed_records().
    reaction_lookup : dict
        From mechanism_utils.build_reaction_lookup().
    carbon_map : dict
        From mechanism_utils.build_species_carbon_map().
    threshold : float
        Abs-sensitivity cutoff.
    carbon_threshold : int
        Minimum carbon number; <=0 disables carbon filtering.
    """
    detailed_rows = []
    master_stats = {}
    species_counter = Counter()
    mismatch_log = []

    for row in detailed_records:
        rxn_id = int(row["Reaction_ID"])
        equation = row["Reaction"]
        sensitivity = float(row["Sensitivity"])
        temperature = row.get("Temperature")
        case_id = row.get("Case_ID")

        if abs(sensitivity) < threshold:
            continue

        keep, max_carbon = reaction_passes_carbon_filter(
            equation, carbon_map, carbon_threshold
        )
        if not keep:
            continue

        yaml_equation = None
        yaml_index = None
        rxn_type = "Unknown"

        entry = reaction_lookup.get(rxn_id)
        master_key = rxn_id          # default: itself
        if entry is not None:
            yaml_equation = entry["Reaction"]
            yaml_index = entry["YAML_Index"]
            rxn_type = entry["Reaction_Type"]

            # collapse duplicate pairs to the lowest ID in the group, so a
            # same-equation pair (e.g. R100 & R534) becomes a single master
            # row keyed on the lowest ID.
            dup_group = entry.get("Duplicate_Group", (rxn_id,))
            if dup_group:
                master_key = min(dup_group)

            if (yaml_equation is not None
                    and yaml_equation.strip() != equation.strip()):
                mismatch_log.append(
                    f"Reaction ID: {rxn_id}\n"
                    f"SA   : {equation}\n"
                    f"YAML : {yaml_equation}\n\n"
                )

        detailed_rows.append({
            "Case_ID": case_id,
            "Temperature": temperature,
            "Reaction_ID": rxn_id,
            "YAML_Index": yaml_index,
            "Reaction": equation,
            "Reaction_Type": rxn_type,
            "Sensitivity": sensitivity,
            "Abs_Sensitivity": abs(sensitivity),
            "Max_Carbon": max_carbon,
        })

        if master_key not in master_stats:
            master_stats[master_key] = {
                "Reaction": equation,
                "Reaction_Type": rxn_type,
                "Maximum_Abs_Sensitivity": abs(sensitivity),
                "Occurrences": 1,
            }
        else:
            m = master_stats[master_key]
            m["Occurrences"] += 1
            m["Maximum_Abs_Sensitivity"] = max(
                m["Maximum_Abs_Sensitivity"], abs(sensitivity)
            )

        for sp in extract_species_from_equation(equation):
            species_counter[sp] += 1

    # master rows (sorted by descending max sensitivity for readability)
    master_rows = [
        {
            "Reaction_ID": rid,
            "Reaction": info["Reaction"],
            "Reaction_Type": info["Reaction_Type"],
            "Maximum_Abs_Sensitivity": info["Maximum_Abs_Sensitivity"],
            "Occurrences": info["Occurrences"],
        }
        for rid, info in master_stats.items()
    ]
    master_rows.sort(
        key=lambda r: r["Maximum_Abs_Sensitivity"], reverse=True
    )

    species_rows = [
        {
            "Species": sp,
            "Carbon_Number": carbon_map.get(sp, 0),
            "Occurrences": count,
        }
        for sp, count in species_counter.items()
    ]

    summary_counter = Counter(r["Reaction_Type"] for r in detailed_rows)
    summary_rows = [
        {"Reaction_Type": rtype, "Count": count}
        for rtype, count in summary_counter.items()
    ]

    return ScreeningResult(
        detailed_rows, master_rows, species_rows,
        summary_rows, mismatch_log,
    )
