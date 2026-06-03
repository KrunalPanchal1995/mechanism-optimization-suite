#!/usr/bin/env python3
"""
modules/mechanism_utils.py

Utilities for reading and processing Cantera YAML mechanisms.

CHANGES vs original:
- reaction_lookup keys are now Capitalized and match interfaces.py
  (Reaction_ID, YAML_Index, Reaction, Reaction_Type). The original
  mixed these with lowercase 'equation'/'type'/'yaml_index', which
  broke screening.py.
- species extraction delegated to interfaces.extract_species_from_equation
  (was duplicated three times across the codebase).
"""

import yaml

from .interfaces import extract_species_from_equation  # canonical


# ============================================================
# YAML Loading
# ============================================================

def load_mechanism(yaml_file):
    """Load a Cantera YAML mechanism into a dict."""
    with open(yaml_file, "r") as f:
        return yaml.safe_load(f)


# ============================================================
# Species Utilities
# ============================================================

def build_species_carbon_map(mech):
    """species name -> carbon number (0 if no carbon)."""
    carbon_map = {}
    for sp in mech.get("species", []):
        composition = sp.get("composition", {}) or {}
        carbon_map[sp["name"]] = composition.get("C", 0)
    return carbon_map


def get_species_names(mech):
    return [sp["name"] for sp in mech.get("species", [])]


# ============================================================
# Reaction Type
# ============================================================

def get_rxn_type(rxn_data):
    """Classify a YAML reaction dict into a framework reaction type."""
    rtype = rxn_data.get("type")
    if rtype is not None:
        if rtype == "three-body":
            return "ThirdBody"
        if rtype == "falloff":
            return "Falloff"
        if rtype == "Chebyshev":
            return "Chebyshev"
        if rtype == "pressure-dependent-Arrhenius":
            if rxn_data.get("duplicate", False):
                return "PLOG-Duplicate"
            return "PLOG"

    if rxn_data.get("duplicate", False):
        return "Duplicate"
    return "Elementary"


# ============================================================
# Reaction Lookup   (Reaction_ID == YAML index + 1)
# ============================================================

def build_reaction_lookup(mech):
    """lookup[reaction_id] -> dict per interfaces.py contract #3.

    Adds two duplicate-awareness fields:
      - Duplicate_Group : sorted tuple of ALL reaction IDs sharing this
        exact equation (length 1 if unique). Lets screening collapse a
        duplicate pair to a single master entry and lets XML emit one
        a/b block instead of two separate reactions.
      - RxnCount : for PLOG / PLOG-Duplicate, len(rate-constants); None
        otherwise. The XML writer subtracts 2 (High+Low) from this.
    """
    reactions = mech.get("reactions", [])

    # map equation -> [all ids] so duplicates can be grouped
    eq_to_ids = {}
    for i, rxn in enumerate(reactions):
        eq = rxn.get("equation", "")
        eq_to_ids.setdefault(eq, []).append(i + 1)

    lookup = {}
    for i, rxn in enumerate(reactions):
        reaction_id = i + 1
        eq = rxn.get("equation", "")
        rtype = get_rxn_type(rxn)

        # count pressure points for PLOG variants
        rxn_count = None
        if rtype in ("PLOG", "PLOG-Duplicate"):
            rates = rxn.get("rate-constants", [])
            rxn_count = len(rates) if rates else None

        lookup[reaction_id] = {
            "Reaction_ID": reaction_id,
            "YAML_Index": i,
            "Reaction": eq,
            "Reaction_Type": rtype,
            "Duplicate_Group": tuple(sorted(eq_to_ids[eq])),
            "RxnCount": rxn_count,
            "Raw_Data": rxn,
        }
    return lookup


# ============================================================
# Carbon filter
# ============================================================

def reaction_passes_carbon_filter(equation, carbon_map, carbon_threshold):
    """(passes, max_carbon). threshold<=0 => always passes."""
    if carbon_threshold <= 0:
        return True, 0
    max_carbon = 0
    for sp in extract_species_from_equation(equation):
        c = carbon_map.get(sp, 0)
        if c > max_carbon:
            max_carbon = c
    return max_carbon >= carbon_threshold, max_carbon


# ============================================================
# Validation
# ============================================================

def validate_sa_reaction(reaction_id, sa_equation, lookup):
    """Validate an SA reaction string against the YAML at the same ID."""
    result = {
        "found": False, "match": False,
        "yaml_equation": None, "reaction_type": None, "yaml_index": None,
    }
    if reaction_id not in lookup:
        return result
    entry = lookup[reaction_id]
    result["found"] = True
    result["yaml_equation"] = entry["Reaction"]
    result["reaction_type"] = entry["Reaction_Type"]
    result["yaml_index"] = entry["YAML_Index"]
    result["match"] = (entry["Reaction"].strip() == str(sa_equation).strip())
    return result
