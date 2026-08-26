"""
gui_common/fuel_data.py
=========================
Fuel name normalization/validation, extracted from generate_target_opt.py's
FuelNormalizer class -- with one deliberate change from the original:

MW/nu_O2 lookups now delegate to chem_data.py's compute_mw()/compute_nu_o2()
instead of maintaining a second, verbatim-duplicated table. The original
FuelNormalizer.CANONICAL_FUELS embedded the exact same numbers
(H2=2.016, CH4=16.043, NC7H16=100.205, ...) as chem_data.py's
DEFAULT_FUEL_MW/DEFAULT_STOICH_O2 -- confirmed identical, not just similar,
during the GUI-separation planning research (see
GUI_SEPARATION_AND_IMPROVEMENT_PLAN_2026-07-12_142230.md §1.4). This class's
actual unique value-add is alias resolution (case-insensitive matching,
"n-heptane"/"nheptane" -> "NC7H16") and SMILES lookup, neither of which
chem_data.py provides -- that's what's kept here.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import chem_data  # noqa: E402  (repo-root module, path fixed up above)


class FuelNormalizer:
    """
    Normalize and validate fuel species across GUI tabs and mechanism files.
    Handles:
    - Case-insensitive matching (H2 ≈ h2, NC7H16 ≈ nc7h16)
    - Alias resolution (n-Heptane → NC7H16, heptane → NC7H16)
    - Cross-file consistency (XML vs CSV vs mechanism species)

    MW/nu_O2: delegates to chem_data.compute_mw()/compute_nu_o2() (single
    source of truth). ALIASES/SMILES: this class's own data, chem_data.py
    has neither.
    """
    # canonical → set of recognized aliases (lowercase)
    ALIASES = {
        "H2":         {"h2", "hydrogen"},
        "CO":         {"co", "carbon monoxide"},
        "CH4":        {"ch4", "methane"},
        "C2H4":       {"c2h4", "ethene", "ethylene"},
        "C2H6":       {"c2h6", "ethane"},
        "C3H8":       {"c3h8", "propane"},
        "C4H10":      {"c4h10", "butane"},
        "C4H6":       {"c4h6", "butene", "butadiene"},
        "CH2O":       {"ch2o", "formaldehyde"},
        "CH3OH":      {"ch3oh", "methanol"},
        "C2H5OH":     {"c2h5oh", "ethanol"},
        "NH3":        {"nh3", "ammonia"},
        "N2H4":       {"n2h4", "hydrazine"},
        "NC7H16":     {"nc7h16", "n-heptane", "nheptane", "heptane", "n-c7h16"},
        "NC12H26":    {"nc12h26", "n-dodecane", "n-c12h26", "dodecane"},
        "C7H8":       {"c7h8", "toluene"},
        "IC16H34":    {"ic16h34", "isohexadecane", "i-c16h34"},
        "C12H23":     {"c12h23"},
        "OME3":       {"ome3", "polymethoxymethane"},
        "MB-C5H10O2": {"mb-c5h10o2", "methyl butyrate"},
    }

    # canonical → SMILES (None where the exact isomer/structure is uncertain,
    # so species-translation treats these as low-confidence rather than
    # silently matching on a guess).
    SMILES = {
        "H2": "[H][H]", "CO": "[C-]#[O+]", "CH4": "C", "C2H4": "C=C",
        "C2H6": "CC", "C3H8": "CCC", "C4H10": "CCCC", "C4H6": "C=CC=C",
        "CH2O": "C=O", "CH3OH": "CO", "C2H5OH": "CCO", "NH3": "N", "N2H4": "NN",
        "NC7H16": "CCCCCCC", "NC12H26": "CCCCCCCCCCCC", "C7H8": "Cc1ccccc1",
        "IC16H34": None, "C12H23": None,
        "OME3": "COCOCOCOC", "MB-C5H10O2": "CCCC(=O)OC",
    }

    @staticmethod
    def normalize(fuel_input: str) -> str:
        """
        Normalize fuel name to canonical form.
        E.g. 'h2' → 'H2', 'n-heptane' → 'NC7H16', 'nheptane' → 'NC7H16'
        Returns canonical form or original input if no match found.
        """
        normalized = fuel_input.strip().lower()
        for canonical, aliases in FuelNormalizer.ALIASES.items():
            if normalized == canonical.lower() or normalized in aliases:
                return canonical
        return fuel_input  # fallback to original if not in table

    @staticmethod
    def get_mw(fuel: str):
        """Get molecular weight (g/mol) for a fuel via chem_data.compute_mw().
        Returns None if not found."""
        canonical = FuelNormalizer.normalize(fuel)
        mw = chem_data.compute_mw(canonical)
        return mw if mw else None

    @staticmethod
    def get_nu_o2(fuel: str):
        """Get stoichiometric O2 (mol O2 / mol fuel) via chem_data.compute_nu_o2().
        Returns None if not found."""
        canonical = FuelNormalizer.normalize(fuel)
        nu = chem_data.compute_nu_o2(canonical)
        return nu if nu else None

    @staticmethod
    def get_smiles(fuel: str):
        """Get SMILES for a well-known fuel. Returns None if not found/uncertain."""
        canonical = FuelNormalizer.normalize(fuel)
        for fuel_key, smiles in FuelNormalizer.SMILES.items():
            if canonical == fuel_key or canonical.lower() == fuel_key.lower():
                return smiles
        return None

    @staticmethod
    def is_known_fuel(fuel: str) -> bool:
        """Check if fuel is in the canonical database."""
        normalized = fuel.strip().lower()
        for canonical, aliases in FuelNormalizer.ALIASES.items():
            if normalized == canonical.lower() or normalized in aliases:
                return True
        return False

    @staticmethod
    def validate_fuel_against_mechanism(fuel: str, mechanism_species: list) -> tuple:
        """
        Validate a fuel against mechanism species list.
        Returns (is_valid, message, suggestions)
        """
        canonical = FuelNormalizer.normalize(fuel)
        mechanism_species_normalized = [s.lower() for s in mechanism_species]

        if canonical.lower() in mechanism_species_normalized:
            return (True, f"✓ Fuel '{canonical}' found in mechanism", [])

        # Try fuzzy match
        suggestions = [s for s in mechanism_species if canonical.lower() in s.lower() or s.lower() in canonical.lower()]
        if suggestions:
            return (False, f"Fuel '{canonical}' NOT in mechanism. Did you mean:", suggestions)

        return (False, f"Fuel '{canonical}' NOT found in mechanism. Available fuels: {', '.join(mechanism_species[:5])}...", [])
