"""
rmip/io/cantera_reader.py
=========================
Load a Cantera YAML (or CTI) mechanism file and return the same plain-dict
schema as ``chemkin_reader.py`` (design decision D4).

Usage::

    from rmip.io.cantera_reader import load_cantera_yaml

    species_list, rxn_list = load_cantera_yaml("mechanism.yaml")
"""
from __future__ import annotations

import json
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("rmip.io.cantera")

SpeciesDict  = Dict[str, Any]
ReactionDict = Dict[str, Any]

# Cantera gas constant [J/mol/K]
_R_SI = 8314.46261815324  # J/kmol/K → but Cantera uses SI J/mol when .standard = True


def load_cantera_yaml(
    path: str,
    phase_name: Optional[str] = None,
) -> Tuple[List[SpeciesDict], List[ReactionDict]]:
    """
    Parse a Cantera YAML (or CTI) mechanism file.

    Parameters
    ----------
    path : str
        Path to ``.yaml`` or ``.cti`` file.
    phase_name : str, optional
        Name of the gas phase to load.  Defaults to the first gas phase found.

    Returns
    -------
    species_list : list[dict]
    rxn_list     : list[dict]
    """
    try:
        import cantera as ct
    except ImportError:
        raise ImportError(
            "Cantera is required to read YAML mechanisms. "
            "Install via: pip install cantera"
        )

    log.info("Loading Cantera YAML: %s", path)
    try:
        gas = ct.Solution(path, phase_name) if phase_name else ct.Solution(path)
    except Exception as exc:
        raise RuntimeError(f"Cantera failed to load '{path}': {exc}") from exc

    log.info("  Loaded: %d species, %d reactions", gas.n_species, gas.n_reactions)

    species_list = [_parse_species(sp) for sp in gas.species()]
    rxn_list     = [_parse_reaction(rx, gas) for rx in gas.reactions()]
    return species_list, rxn_list


# ---------------------------------------------------------------------------
# Private — species
# ---------------------------------------------------------------------------

def _parse_species(sp) -> SpeciesDict:
    d: SpeciesDict = {
        "name":           sp.name,
        "smiles":         None,
        "formula":        _formula_from_composition(sp.composition),
        "mw_kg_mol":      sp.molecular_weight / 1000.0,   # g/mol → kg/mol
        "thermo_type":    None,
        "tmin_K":         None,
        "tmid_K":         None,
        "tmax_K":         None,
        "coeffs_low":     None,
        "coeffs_high":    None,
        "h298_kJ_mol":    None,
        "s298_J_molK":    None,
        "thermo_comment": None,
    }
    _extract_thermo(sp.thermo, d)
    return d


def _extract_thermo(thermo, d: SpeciesDict) -> None:
    tn = type(thermo).__name__
    if "NasaPoly2" in tn or tn == "NasaPoly2":
        d["thermo_type"] = "nasa7"
        d["tmin_K"]      = float(thermo.min_temp)
        d["tmid_K"]      = float(thermo.coeffs[0])
        d["tmax_K"]      = float(thermo.max_temp)
        # Cantera coeffs layout: [Tmid, a1_hi..a7_hi, a1_lo..a7_lo]
        d["coeffs_high"] = [float(c) for c in thermo.coeffs[1:8]]
        d["coeffs_low"]  = [float(c) for c in thermo.coeffs[8:15]]


def _formula_from_composition(comp: Dict[str, float]) -> str:
    """Build an empirical formula string from a Cantera composition dict."""
    order = ["C", "H", "O", "N", "S"]
    parts = []
    for el in order:
        if el in comp:
            n = int(comp[el])
            parts.append(el if n == 1 else f"{el}{n}")
    for el, n in sorted(comp.items()):
        if el not in order:
            nn = int(n)
            parts.append(el if nn == 1 else f"{el}{nn}")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Private — reactions
# ---------------------------------------------------------------------------

def _parse_reaction(rx, gas) -> ReactionDict:
    reactants = [(k, float(v)) for k, v in rx.reactants.items()]
    products  = [(k, float(v)) for k, v in rx.products.items()]

    kinetics_type, A_si, A_orig, A_units, n, Ea_si, Ea_orig, Ea_units, kdata = \
        _parse_kinetics(rx)

    return {
        "formula_string":          rx.equation,
        "reversible":              bool(rx.reversible),
        "family":                  None,
        "duplicate":               bool(rx.duplicate),
        "kinetics_type":           kinetics_type,
        "pressure_dependent":      _is_pdep(rx),
        "has_third_body":          _has_third_body(rx),
        "A_si":                    A_si,
        "A_orig":                  A_orig,
        "A_units":                 A_units,
        "n":                       n,
        "Ea_si":                   Ea_si,
        "Ea_orig":                 Ea_orig,
        "Ea_units":                Ea_units,
        "kinetics_data":           kdata,
        "reactants":               reactants,
        "products":                products,
        "third_body_efficiencies": _extract_efficiencies(rx),
        "rmg_comment":             None,
    }


def _parse_kinetics(rx) -> Tuple:
    tn = type(rx).__name__

    if tn == "ElementaryReaction":
        rate = rx.rate
        A, n, Ea = _arrhenius_si(rate)
        return "arrhenius", A, A, None, n, Ea, Ea, "J/mol", {}

    if tn == "ThreeBodyReaction":
        rate = rx.rate
        A, n, Ea = _arrhenius_si(rate)
        return "third_body", A, A, None, n, Ea, Ea, "J/mol", {}

    if tn == "FalloffReaction":
        kdata: Dict[str, Any] = {}
        lo  = rx.low_rate
        hi  = rx.high_rate
        A_lo, n_lo, Ea_lo = _arrhenius_si(lo)
        A_hi, n_hi, Ea_hi = _arrhenius_si(hi)
        kdata["low"]  = {"A_si": A_lo, "n": n_lo, "Ea_si": Ea_lo}
        kdata["high"] = {"A_si": A_hi, "n": n_hi, "Ea_si": Ea_hi}
        falloff_type = type(rx.falloff).__name__
        if "Troe" in falloff_type:
            kdata["falloff"] = "troe"
            kdata["troe_params"] = list(rx.falloff.parameters)
            return "troe", A_hi, A_hi, None, n_hi, Ea_hi, Ea_hi, "J/mol", kdata
        return "lindemann", A_hi, A_hi, None, n_hi, Ea_hi, Ea_hi, "J/mol", kdata

    if tn == "PlogReaction":
        kdata = {"pressures": [], "arrhenius": []}
        for P, rate in rx.rates:
            A, n, Ea = _arrhenius_si(rate)
            kdata["pressures"].append(float(P))
            kdata["arrhenius"].append({"A_si": A, "n": n, "Ea_si": Ea})
        # representative at 1 atm
        A = n = Ea = None
        for i, P in enumerate(kdata["pressures"]):
            if abs(P - 101325.0) < 50000:
                entry = kdata["arrhenius"][i]
                A, n, Ea = entry["A_si"], entry["n"], entry["Ea_si"]
                break
        return "plog", A, A, None, n, Ea, Ea, "J/mol", kdata

    if tn == "ChebyshevReaction":
        cheb = rx.rate
        kdata = {
            "Tmin": float(cheb.Tmin), "Tmax": float(cheb.Tmax),
            "Pmin": float(cheb.Pmin), "Pmax": float(cheb.Pmax),
            "coeffs": cheb.coeffs.tolist(),
        }
        return "chebyshev", None, None, None, None, None, None, None, kdata

    return tn.lower(), None, None, None, None, None, None, None, {}


def _arrhenius_si(rate) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Extract (A, n, Ea_J_per_mol) in SI from a Cantera Arrhenius-like rate."""
    try:
        A  = float(rate.pre_exponential_factor)
        n  = float(rate.temperature_exponent)
        Ea = float(rate.activation_energy)   # Cantera returns J/mol
        return A, n, Ea
    except Exception:
        return None, None, None


def _is_pdep(rx) -> bool:
    return type(rx).__name__ in ("FalloffReaction", "PlogReaction", "ChebyshevReaction")


def _has_third_body(rx) -> bool:
    return type(rx).__name__ in ("ThreeBodyReaction", "FalloffReaction")


def _extract_efficiencies(rx) -> List[Tuple[str, float]]:
    try:
        effs = rx.efficiencies or {}
    except (AttributeError, ValueError):
        effs = {}
    return [(name, float(eff)) for name, eff in effs.items()]
