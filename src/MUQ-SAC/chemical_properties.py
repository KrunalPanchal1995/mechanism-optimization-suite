"""
chemical_properties.py

Centralized chemical metadata used by the dataset manipulation utilities.

This module intentionally contains *only* data and small helper functions so
it can be imported by lightweight parsers without pulling heavy dependencies.

Definitions
-----------
STOICH_O2_PER_MOL_FUEL:
    Stoichiometric O2 requirement (mol O2 per mol fuel) for complete oxidation.
    These values are used to compute equivalence ratio:

        phi = O2_required / O2_actual

    where:
        O2_required = sum_i nu_O2_i * X_fuel_i
        O2_actual   = X_O2  (mole fraction of O2 in the mixture)

MOLECULAR_WEIGHTS:
    Molecular weights in g/mol for selected species (optional metadata).

Notes
-----
- Keys are stored in a normalized (upper-case) form.
- If you add new fuels, update STOICH_O2_PER_MOL_FUEL with the correct O2 demand.
"""

from __future__ import annotations

from typing import Dict

def normalize_species_name(name: str) -> str:
    """Normalize species keys to a canonical form (upper-case, stripped)."""
    return name.strip().upper()

# Molecular weights [g/mol] (only used when needed)
MOLECULAR_WEIGHTS: Dict[str, float] = {
    "H2": 2.016,
    "CO": 28.010,
    "O2": 31.998,
    "N2": 28.014,
    "AR": 39.948,
    "HE": 4.003,
    "CO2": 44.009,
    "H2O": 18.015,
    "CH4": 16.043,
    "C3H8": 44.097,
    "NC7H16": 100.205,
    "C4H6": 54.092,
    "OME3": 136.0,          # keep as in your original file (placeholder if needed)
    "MB-C5H10O2": 86.0,     # placeholder
    "CH2O": 30.026,
    "NC12H26": 170.334,
    "C7H8": 92.141,
    "IC16H34": 226.441,
    "C12H23": 167.0,        # placeholder
}

# Stoichiometric O2 requirement (mol O2 / mol fuel)
STOICH_O2_PER_MOL_FUEL: Dict[str, float] = {
    "H2": 0.5,     # H2 + 0.5 O2 -> H2O
    "CO": 0.5,     # CO + 0.5 O2 -> CO2
    "CH4": 2.0,
    "C3H8": 5.0,
    "NC7H16": 11.0,
    "C4H6": 5.5,
    "OME3": 6.0,
    "MB-C5H10O2": 6.5,
    "CH2O": 1.0,
    "NC12H26": 18.5,  # NOTE: verify; keep your original if different
    "C7H8": 9.0,      # NOTE: verify; keep your original if different
    "IC16H34": 24.5,  # NOTE: verify; keep your original if different
    "C12H23": 17.75,
    # Inerts / oxidizer
    "O2": 0.0,
    "N2": 0.0,
    "AR": 0.0,
    "HE": 0.0,
    "CO2": 0.0,
    "H2O": 0.0,
}

def get_o2_stoich(species: str) -> float:
    """Return stoichiometric O2 requirement for a fuel species (default 0.0)."""
    return STOICH_O2_PER_MOL_FUEL.get(normalize_species_name(species), 0.0)

def get_molecular_weight(species: str) -> float | None:
    """Return molecular weight in g/mol if known."""
    return MOLECULAR_WEIGHTS.get(normalize_species_name(species))
