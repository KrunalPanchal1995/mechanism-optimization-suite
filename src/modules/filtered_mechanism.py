#!/usr/bin/env python3
"""
modules/filtered_mechanism.py

Mechanism reduction utilities.

CHANGES vs original:
- species extraction delegated to interfaces (was a 3rd duplicate copy
  that used a different split rule than screening/mechanism_utils).
- generate_filtered_mechanism signature aligned with how the driver
  calls it (takes a mechanism dict + reaction_ids or species set).
"""

from copy import deepcopy

import yaml

from .interfaces import extract_species_from_equation, normalize_group


def collect_species_from_reactions(reaction_ids, mechanism):
    species_keep = set()
    reactions = mechanism.get("reactions", [])
    n = len(reactions)
    for rid in reaction_ids:
        rid = int(rid)
        if rid < 1 or rid > n:
            continue
        species_keep.update(
            extract_species_from_equation(reactions[rid - 1].get("equation", ""))
        )
    return species_keep


def filter_by_reaction_ids(mechanism, reaction_ids):
    reaction_ids = {int(i) for i in reaction_ids}
    reduced = deepcopy(mechanism)
    selected, species_keep = [], set()
    for i, rxn in enumerate(mechanism.get("reactions", []), start=1):
        if i not in reaction_ids:
            continue
        selected.append(rxn)
        species_keep.update(
            extract_species_from_equation(rxn.get("equation", ""))
        )
    reduced["reactions"] = selected
    reduced["species"] = [
        sp for sp in mechanism.get("species", [])
        if sp["name"] in species_keep
    ]
    return reduced


def filter_by_species(mechanism, species_keep):
    species_keep = set(species_keep)
    reduced = deepcopy(mechanism)
    reduced["species"] = [
        sp for sp in mechanism.get("species", [])
        if sp["name"] in species_keep
    ]
    filtered = []
    for rxn in mechanism.get("reactions", []):
        species = extract_species_from_equation(rxn.get("equation", ""))
        if species.issubset(species_keep):
            filtered.append(rxn)
    reduced["reactions"] = filtered
    return reduced


def save_mechanism(mechanism, outfile):
    with open(outfile, "w") as f:
        yaml.safe_dump(mechanism, f, sort_keys=False)


def mechanism_statistics(mechanism):
    return {
        "species_count": len(mechanism.get("species", [])),
        "reaction_count": len(mechanism.get("reactions", [])),
    }


def generate_filtered_mechanism(mechanism, reaction_ids=None,
                                species_keep=None,
                                outfile="filtered_mechanism.yaml"):
    if reaction_ids is not None:
        reduced = filter_by_reaction_ids(mechanism, reaction_ids)
    elif species_keep is not None:
        reduced = filter_by_species(mechanism, species_keep)
    else:
        raise ValueError("Supply either reaction_ids or species_keep.")
    save_mechanism(reduced, outfile)
    print(f"[Mech] Written -> {outfile}  {mechanism_statistics(reduced)}")
    return reduced
