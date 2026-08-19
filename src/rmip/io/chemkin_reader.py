"""
rmip/io/chemkin_reader.py
=========================
Load a CHEMKIN mechanism file via RMG-Py and return plain Python dicts.
No RMG-Py objects leak out of this module (design decision D4).

Usage::

    from rmip.io.chemkin_reader import load_chemkin

    species_list, rxn_list = load_chemkin(
        mech_path  = "n_heptane.inp",
        thermo_path = "thermo.dat",   # optional
    )

Each dict in ``species_list``::

    {
        "name":            str,   # as used in mechanism
        "smiles":          None,  # populated by parser after glossary lookup
        "formula":         str,   # e.g. "C7H16"
        "mw_kg_mol":       float,
        "thermo_type":     "nasa7",
        "tmin_K":          float,
        "tmid_K":          float,
        "tmax_K":          float,
        "coeffs_low":      [float×7],
        "coeffs_high":     [float×7],
        "thermo_comment":  str,
    }

Each dict in ``rxn_list``::

    {
        "formula_string":   str,
        "reversible":       bool,
        "family":           str | None,
        "duplicate":        bool,
        "kinetics_type":    str,
        "pressure_dependent": bool,
        "has_third_body":   bool,
        "A_si":             float | None,
        "A_orig":           float | None,
        "A_units":          str   | None,
        "n":                float | None,
        "Ea_si":            float | None,
        "Ea_orig":          float | None,
        "Ea_units":         str   | None,
        "kinetics_data":    dict,   # PLOG/Troe/Chebyshev details
        "reactants":        [(name, stoich), ...],
        "products":         [(name, stoich), ...],
        "third_body_efficiencies": [(species_name, efficiency), ...],
        "rmg_comment":      str | None,
    }
"""
from __future__ import annotations

import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("rmip.io.chemkin")

SpeciesDict = Dict[str, Any]
ReactionDict = Dict[str, Any]


def load_chemkin(
    mech_path:   str,
    thermo_path: Optional[str] = None,
    transport_path: Optional[str] = None,  # ignored — D6
    dict_path: Optional[str] = None,
    check_duplicates: bool = False,
) -> Tuple[List[SpeciesDict], List[ReactionDict]]:
    """
    Parse a CHEMKIN mechanism using RMG-Py.

    Parameters
    ----------
    mech_path : str
        Path to the CHEMKIN mechanism .inp file.
    thermo_path : str, optional
        Separate thermodynamic data file (.dat).  If None, thermo is assumed
        to be embedded in *mech_path*.
    transport_path : str, optional
        Ignored — transport data is out of scope (D6).
    dict_path : str, optional
        RMG species dictionary .txt path (for resonance structure generation).
    check_duplicates : bool
        Flag duplicate reactions during RMG-Py loading.

    Returns
    -------
    species_list : list[dict]
    rxn_list     : list[dict]
    """
    # Try RMG-Py first (full-featured parser with resonance structures).
    # Fall back to built-in pure-Python parser if RMG-Py is not installed.
    try:
        from rmgpy.chemkin import load_chemkin_file
        _HAS_RMG_PY = True
    except ImportError:
        _HAS_RMG_PY = False
        from rmip.io._chemkin_parser import ChemkinParser

    log.info("Loading CHEMKIN file: %s", mech_path)

    if _HAS_RMG_PY:
        kwargs: Dict[str, Any] = {
            "use_chemkin_names": True,
            "check_duplicates": check_duplicates,
        }
        if thermo_path:
            kwargs["thermo_path"] = thermo_path
        if dict_path:
            kwargs["dictionary_path"] = dict_path

        # RMG-Py API varies across versions — handle gracefully
        try:
            species, rxns = load_chemkin_file(mech_path, **kwargs)
        except TypeError:
            # Older versions may not accept keyword arguments
            species, rxns = load_chemkin_file(mech_path)

        log.info("  Raw: %d species, %d reactions (via RMG-Py)", len(species), len(rxns))

        species_list = [_parse_species(sp) for sp in species]
        rxn_list     = [_parse_reaction(rx) for rx in rxns]
    else:
        log.warning(
            "RMG-Py not available; using built-in CHEMKIN parser. "
            "For full resonance structure support, install RMG-Py."
        )
        parser = ChemkinParser()
        species_list, rxn_list = parser.parse(mech_path, thermo_path=thermo_path)

    log.info("  Raw: %d species, %d reactions", len(species_list), len(rxn_list))
    return species_list, rxn_list


# ---------------------------------------------------------------------------
# Private — species parsing
# ---------------------------------------------------------------------------

def _parse_species(sp) -> SpeciesDict:
    d: SpeciesDict = {
        "name":           sp.label,
        "smiles":         None,
        "formula":        sp.molecule[0].get_formula() if sp.molecule else None,
        "mw_kg_mol":      None,
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

    # Molecular weight
    try:
        d["mw_kg_mol"] = sp.molecule[0].get_molecular_weight() / 1000.0  # g/mol → kg/mol
    except Exception:
        pass

    # Thermo
    if sp.thermo is not None:
        _extract_thermo(sp.thermo, d)

    return d


def _extract_thermo(thermo, d: SpeciesDict) -> None:
    """Extract NASA-7 polynomial data from an rmgpy thermo object."""
    class_name = type(thermo).__name__
    if class_name == "NASA":
        d["thermo_type"] = "nasa7"
        polys = thermo.polynomials
        if len(polys) >= 2:
            lo = polys[0]
            hi = polys[1]
            d["tmin_K"]     = float(lo.Tmin.value_si)
            d["tmid_K"]     = float(lo.Tmax.value_si)
            d["tmax_K"]     = float(hi.Tmax.value_si)
            d["coeffs_low"] = [float(c) for c in lo.coeffs]
            d["coeffs_high"]= [float(c) for c in hi.coeffs]
        d["thermo_comment"] = getattr(thermo, "comment", None)

    # Enthalpy and entropy at 298 K
    try:
        d["h298_kJ_mol"] = float(thermo.get_enthalpy(298.15)) / 1e3
        d["s298_J_molK"] = float(thermo.get_entropy(298.15))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Private — reaction parsing
# ---------------------------------------------------------------------------

def _parse_reaction(rx) -> ReactionDict:
    reactants = _get_stoich(rx.reactants)
    products  = _get_stoich(rx.products)

    kinetics_type, A_si, A_orig, A_units, n, Ea_si, Ea_orig, Ea_units, kdata = \
        _parse_kinetics(rx.kinetics)

    d: ReactionDict = {
        "formula_string":        _make_formula(reactants, products, rx.reversible),
        "reversible":            bool(rx.reversible),
        "family":                _get_family(rx),
        "duplicate":             bool(getattr(rx, "duplicate", False)),
        "kinetics_type":         kinetics_type,
        "pressure_dependent":    _is_pdep(rx),
        "has_third_body":        _has_third_body(rx),
        "A_si":                  A_si,
        "A_orig":                A_orig,
        "A_units":               A_units,
        "n":                     n,
        "Ea_si":                 Ea_si,
        "Ea_orig":               Ea_orig,
        "Ea_units":              Ea_units,
        "kinetics_data":         kdata,
        "reactants":             reactants,
        "products":              products,
        "third_body_efficiencies": _extract_efficiencies(rx),
        "rmg_comment":           getattr(rx, "comment", None),
    }
    return d


def _get_stoich(species_list) -> List[Tuple[str, float]]:
    """
    RMG-Py sometimes duplicates species to encode stoich > 1.
    Use a Counter to sum them.
    """
    counter: Counter = Counter()
    for sp in species_list:
        counter[sp.label] += 1
    return [(name, float(count)) for name, count in counter.items()]


def _make_formula(
    reactants: List[Tuple[str, float]],
    products:  List[Tuple[str, float]],
    reversible: bool,
) -> str:
    def side(pairs):
        parts = []
        for name, coeff in pairs:
            parts.append(name if coeff == 1.0 else f"{int(coeff)} {name}")
        return " + ".join(parts)
    arrow = " <=> " if reversible else " => "
    return side(reactants) + arrow + side(products)


def _parse_kinetics(kin):
    """Return (kinetics_type, A_si, A_orig, A_units, n, Ea_si, Ea_orig, Ea_units, kdata)."""
    if kin is None:
        return "unknown", None, None, None, None, None, None, None, {}

    cn = type(kin).__name__

    if cn == "Arrhenius":
        return _from_arrhenius(kin)
    elif cn in ("PDepArrhenius", "MultiPDepArrhenius"):
        return _from_plog(kin)
    elif cn == "Troe":
        return _from_troe(kin)
    elif cn == "Lindemann":
        return _from_lindemann(kin)
    elif cn == "ThirdBody":
        A, n, Ea, A_u, Ea_u = _arrhenius_params(kin.arrheniusLow)
        return "third_body", A, A, A_u, n, Ea, Ea, Ea_u, {}
    elif cn == "Chebyshev":
        return _from_chebyshev(kin)
    else:
        return cn.lower(), None, None, None, None, None, None, None, {}


def _from_arrhenius(kin):
    A_si, n, Ea_si, A_u, Ea_u = _arrhenius_params(kin)
    return "arrhenius", A_si, A_si, A_u, n, Ea_si, Ea_si, Ea_u, {}


def _arrhenius_params(kin):
    """Extract (A_si, n, Ea_si, A_units, Ea_units) from an Arrhenius object."""
    try:
        A_si  = float(kin.A.value_si)
        n     = float(kin.n.value_si)
        Ea_si = float(kin.Ea.value_si)  # J/mol in SI
        A_u   = str(kin.A.units) if hasattr(kin.A, "units") else None
        Ea_u  = str(kin.Ea.units) if hasattr(kin.Ea, "units") else "J/mol"
    except Exception:
        return None, None, None, None, None
    return A_si, n, Ea_si, A_u, Ea_u


def _from_plog(kin):
    kdata: Dict[str, Any] = {"pressures": [], "arrhenius": []}
    try:
        pressures = kin.pressures
        if hasattr(pressures, "value_si"):
            pressures = pressures.value_si
        for P, arr in zip(pressures, kin.arrhenius):
            A_si, n, Ea_si, A_u, Ea_u = _arrhenius_params(arr)
            kdata["pressures"].append(float(P))
            kdata["arrhenius"].append({"A_si": A_si, "n": n, "Ea_si": Ea_si})
    except Exception as e:
        log.debug("PLOG parse error: %s", e)
    # Representative at 1 atm
    A_si = n = Ea_si = None
    for i, P in enumerate(kdata["pressures"]):
        if abs(P - 101325.0) < 50000:
            entry = kdata["arrhenius"][i]
            A_si, n, Ea_si = entry["A_si"], entry["n"], entry["Ea_si"]
            break
    return "plog", A_si, A_si, None, n, Ea_si, Ea_si, "J/mol", kdata


def _from_troe(kin):
    kdata: Dict[str, Any] = {"alpha": None, "T1": None, "T2": None, "T3": None,
                              "low": {}, "high": {}}
    try:
        kdata["alpha"] = float(kin.alpha)
        kdata["T1"]    = float(kin.T1.value_si) if kin.T1 else None
        kdata["T2"]    = float(kin.T2.value_si) if kin.T2 else None
        kdata["T3"]    = float(kin.T3.value_si) if kin.T3 else None
        A_lo, n_lo, Ea_lo, _, _ = _arrhenius_params(kin.arrheniusLow)
        A_hi, n_hi, Ea_hi, _, _ = _arrhenius_params(kin.arrheniusHigh)
        kdata["low"]  = {"A_si": A_lo, "n": n_lo, "Ea_si": Ea_lo}
        kdata["high"] = {"A_si": A_hi, "n": n_hi, "Ea_si": Ea_hi}
    except Exception as e:
        log.debug("Troe parse error: %s", e)
    A_si, n, Ea_si, A_u, Ea_u = _arrhenius_params(getattr(kin, "arrheniusHigh", kin))
    return "troe", A_si, A_si, A_u, n, Ea_si, Ea_si, Ea_u, kdata


def _from_lindemann(kin):
    A_si, n, Ea_si, A_u, Ea_u = _arrhenius_params(kin.arrheniusHigh)
    kdata = {
        "low":  dict(zip(["A_si","n","Ea_si"], _arrhenius_params(kin.arrheniusLow)[:3])),
        "high": {"A_si": A_si, "n": n, "Ea_si": Ea_si},
    }
    return "lindemann", A_si, A_si, A_u, n, Ea_si, Ea_si, Ea_u, kdata


def _from_chebyshev(kin):
    kdata: Dict[str, Any] = {"Tmin": None, "Tmax": None, "Pmin": None, "Pmax": None,
                              "coeffs": []}
    try:
        kdata["Tmin"]   = float(kin.Tmin.value_si)
        kdata["Tmax"]   = float(kin.Tmax.value_si)
        kdata["Pmin"]   = float(kin.Pmin.value_si)
        kdata["Pmax"]   = float(kin.Pmax.value_si)
        kdata["coeffs"] = kin.coeffs.tolist()
    except Exception:
        pass
    return "chebyshev", None, None, None, None, None, None, None, kdata


def _extract_efficiencies(rx) -> List[Tuple[str, float]]:
    effs = []
    kin  = rx.kinetics
    if kin is None:
        return effs
    efficiencies = getattr(kin, "efficiencies", None)
    if efficiencies is None:
        efficiencies = getattr(kin, "thirdBodyEfficiencies", None)
    if not efficiencies:
        return effs
    for sp, eff in efficiencies.items():
        name = getattr(sp, "label", None) or str(sp)
        effs.append((name, float(eff)))
    return effs


def _is_pdep(rx) -> bool:
    kin = rx.kinetics
    return kin is not None and type(kin).__name__ in (
        "PDepArrhenius", "MultiPDepArrhenius", "Chebyshev", "Troe", "Lindemann"
    )


def _has_third_body(rx) -> bool:
    kin = rx.kinetics
    return kin is not None and type(kin).__name__ in (
        "ThirdBody", "Troe", "Lindemann"
    )


def _get_family(rx) -> Optional[str]:
    try:
        template = rx.template
        return template[0].label if template else None
    except Exception:
        return getattr(rx, "family", None)
