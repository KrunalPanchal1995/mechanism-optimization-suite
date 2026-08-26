"""
rmip/io/cantera_reader.py
=========================
Load a Cantera YAML (or CTI) mechanism file and return the same plain-dict
schema as ``chemkin_reader.py`` (design decision D4).

Usage::

    from rmip.io.cantera_reader import load_cantera_yaml

    species_list, rxn_list = load_cantera_yaml("mechanism.yaml")

Cantera 3 notes (B1/B2 in review.md, 2026-07-14)
-------------------------------------------------
Cantera >=3.0 replaced the old per-type reaction classes
(``ElementaryReaction``, ``ThreeBodyReaction``, ``FalloffReaction``, ...)
with a single ``Reaction`` class whose kinetics live on ``rx.rate`` (an
``ArrheniusRate``/``FalloffRate``/``PlogRate``/``ChebyshevRate``...), with
``rx.reaction_type`` giving a clean string discriminator ("Arrhenius",
"three-body-Arrhenius", "falloff-Troe", "falloff-Lindemann", "falloff-SRI",
"pressure-dependent-Arrhenius", "Chebyshev") and third-body data moved to
``rx.third_body``. Dispatching on the old class names (as this module did
before) matches nothing, so every reaction fell through to the "unknown"
branch with NULL kinetics.

Separately, ``rate.pre_exponential_factor``/``.activation_energy`` are
**kmol**-based SI (m, kmol, s, J/kmol), not the mol-based SI (m, mol, s,
J/mol) convention the rest of this suite (chemkin parser, fusion,
estimator) uses. ``_arrhenius_mol_si`` converts at read time: Ea/1000 is
unconditional, A is divided by ``1000**(order-1)`` where ``order`` is the
number of concentration terms in the rate law (reactant stoichiometric sum,
+1 per third body) -- e.g. a plain bimolecular reaction has order=2 so A is
divided by 1000; a third-body reaction's single rate has order=3; a
falloff reaction's low-pressure rate has order=(bimolecular order)+1 while
its high-pressure rate keeps the bimolecular order.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("rmip.io.cantera")

SpeciesDict  = Dict[str, Any]
ReactionDict = Dict[str, Any]


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

    n_hollow = sum(1 for r in rxn_list if r["A_si"] is None and r["kinetics_type"] not in ("chebyshev",))
    if n_hollow:
        log.warning(
            "  %d/%d reactions have no usable A_si (unrecognized reaction_type "
            "or Chebyshev, which this suite doesn't fit an Arrhenius summary for)",
            n_hollow, len(rxn_list),
        )
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
    _extract_transport(sp, d)
    if d["thermo_type"] is None:
        log.warning(
            "  Species '%s': unrecognized thermo type %s -- no thermo stored "
            "(will be absent from any CHEMKIN export, which will hard-fail "
            "solvers that need it)",
            sp.name, type(sp.thermo).__name__,
        )
    return d


def _extract_transport(sp, d: SpeciesDict) -> None:
    """
    Pull Lennard-Jones transport out of a Cantera Species into ``d['transport']``
    (Å / K / Debye / Å³ — the native YAML units), so the parser can persist it to
    the ``species`` transport columns at parse time (Phase-3 follow-on).
    """
    try:
        raw = sp.input_data.get("transport") if hasattr(sp, "input_data") else None
    except Exception:                       # noqa: BLE001
        raw = None
    if not raw:
        return
    d["transport"] = {
        "transport_geometry": raw.get("geometry"),
        "well_depth_K":       raw.get("well-depth"),
        "diameter_A":         raw.get("diameter"),
        "dipole_D":           raw.get("dipole", 0.0),
        "polarizability_A3":  raw.get("polarizability", 0.0),
        "rot_relax":          raw.get("rotational-relaxation", 0.0),
        "transport_source":   "cantera",
    }


def _extract_thermo(thermo, d: SpeciesDict) -> None:
    tn = type(thermo).__name__
    if tn == "NasaPoly2":
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

    kinetics_type, A_si, n, Ea_si, kdata = _parse_kinetics(rx)

    return {
        "formula_string":          rx.equation,
        "reversible":              bool(rx.reversible),
        "family":                  None,
        "duplicate":               bool(rx.duplicate),
        "kinetics_type":           kinetics_type,
        "pressure_dependent":      _is_pdep(rx),
        "has_third_body":          _has_third_body(rx),
        "A_si":                    A_si,
        # Cantera doesn't expose the mechanism-file-original A/Ea (only
        # re-derived kmol-SI values, already captured above) -- leaving
        # these NULL rather than mislabeling SI values as "original units"
        # (the pre-B2 bug: chemkin_writer's SI-path CHEMKIN export was off
        # by ~1e9 because A_orig/Ea_orig held SI numbers under a units
        # header that claimed otherwise).
        "A_orig":                  None,
        "A_units":                 None,
        "n":                       n,
        "Ea_si":                   Ea_si,
        "Ea_orig":                 None,
        "Ea_units":                None,
        "kinetics_data":           kdata,
        "reactants":               reactants,
        "products":                products,
        "third_body_efficiencies": _extract_efficiencies(rx),
        "rmg_comment":             None,
    }


def _parse_kinetics(rx) -> Tuple:
    """Returns (kinetics_type, A_si, n, Ea_si, kinetics_data) with A_si/Ea_si
    in this suite's mol-based SI convention (see module docstring)."""
    reaction_type = getattr(rx, "reaction_type", "") or ""
    reactant_order = sum(rx.reactants.values())

    if reaction_type == "Arrhenius":
        A, n, Ea = _arrhenius_mol_si(rx.rate, reactant_order)
        return "arrhenius", A, n, Ea, {}

    if reaction_type == "three-body-Arrhenius":
        A, n, Ea = _arrhenius_mol_si(rx.rate, reactant_order + 1)
        return "third_body", A, n, Ea, {}

    if reaction_type.startswith("falloff-"):
        rate = rx.rate
        kdata: Dict[str, Any] = {}
        A_lo, n_lo, Ea_lo = _arrhenius_mol_si(rate.low_rate, reactant_order + 1)
        A_hi, n_hi, Ea_hi = _arrhenius_mol_si(rate.high_rate, reactant_order)
        kdata["low"]  = {"A_si": A_lo, "n": n_lo, "Ea_si": Ea_lo}
        kdata["high"] = {"A_si": A_hi, "n": n_hi, "Ea_si": Ea_hi}
        sub_type = reaction_type.split("-", 1)[1].lower()  # troe / lindemann / sri
        kdata["falloff"] = sub_type
        if sub_type == "troe":
            kdata["troe_params"] = [float(c) for c in rate.falloff_coeffs]
            kinetics_type = "troe"
        elif sub_type == "sri":
            kdata["sri_params"] = [float(c) for c in rate.falloff_coeffs]
            kinetics_type = "sri"
        else:
            kinetics_type = "lindemann"
        # Representative A/n/Ea for the summary columns = high-pressure limit
        # (matches this suite's pre-existing convention for falloff rows).
        return kinetics_type, A_hi, n_hi, Ea_hi, kdata

    if reaction_type in ("pressure-dependent-Arrhenius",
                         "three-body-pressure-dependent-Arrhenius"):
        # The "three-body-" variant is a real, if rare, Cantera 3.x reaction
        # type: a PLOG reaction (rx.rate is still a plain PlogRate with the
        # same .rates API) that also carries an (often-empty-efficiency)
        # rx.third_body -- e.g. Glarborg 2018's 'CH2O + H <=> CO + H2 + H'.
        # Before this fix it fell into the "unrecognized" branch below and
        # got stored with kinetics_type='three-body-pressure-dependent-
        # arrhenius' and EVERY kinetics field NULL/empty -- silently
        # correct-looking (no crash) until something downstream (the
        # fusion engine's verbatim-copy -> ChemkinWriter export) tried to
        # actually write out a rate constant for it and hit "no usable A
        # value (A_si and A_orig both NULL)". Found 2026-07-22 running the
        # full 5-mechanism ammonia fusion (Phase 5) -- 2 such reactions in
        # Glarborg 2018 alone. has_third_body (a separate DB column, set by
        # the caller via rx.third_body) already records the third-body
        # fact; kinetics_type stays "plog" like any other PLOG reaction.
        kdata = {"pressures": [], "arrhenius": []}
        for P, rate in rx.rate.rates:
            A, n, Ea = _arrhenius_mol_si(rate, reactant_order)
            kdata["pressures"].append(float(P))
            kdata["arrhenius"].append({"A_si": A, "n": n, "Ea_si": Ea})
        # representative at 1 atm
        A = n = Ea = None
        for i, P in enumerate(kdata["pressures"]):
            if abs(P - 101325.0) < 50000:
                entry = kdata["arrhenius"][i]
                A, n, Ea = entry["A_si"], entry["n"], entry["Ea_si"]
                break
        return "plog", A, n, Ea, kdata

    if reaction_type == "Chebyshev":
        cheb = rx.rate
        tmin, tmax = cheb.temperature_range
        pmin, pmax = cheb.pressure_range
        kdata = {
            "Tmin": float(tmin), "Tmax": float(tmax),
            "Pmin": float(pmin), "Pmax": float(pmax),
            "coeffs": cheb.data.tolist(),
        }
        return "chebyshev", None, None, None, kdata

    log.warning(
        "  Reaction '%s': unrecognized Cantera reaction_type=%r -- "
        "kinetics not stored", rx.equation, reaction_type,
    )
    return (reaction_type or "unknown").lower(), None, None, None, {}


def _arrhenius_mol_si(rate, order) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Extract (A, n, Ea) from a Cantera Arrhenius-like rate object,
    converting Cantera 3's kmol-based SI (m, kmol, s, J/kmol) to this
    suite's mol-based SI (m, mol, s, J/mol) convention. ``order`` is the
    number of concentration terms in the rate expression and sets the
    volume-per-mol dimensionality of A: A_mol = A_kmol / 1000**(order-1)."""
    try:
        A_kmol  = float(rate.pre_exponential_factor)
        n       = float(rate.temperature_exponent)
        Ea_kmol = float(rate.activation_energy)  # Cantera 3: J/kmol
    except Exception:
        return None, None, None
    A_mol  = A_kmol / (1000.0 ** (order - 1))
    Ea_mol = Ea_kmol / 1000.0
    return A_mol, n, Ea_mol


def _is_pdep(rx) -> bool:
    rt = getattr(rx, "reaction_type", "") or ""
    return rt.startswith("falloff-") or rt in ("pressure-dependent-Arrhenius", "Chebyshev")


def _has_third_body(rx) -> bool:
    return rx.third_body is not None


def _extract_efficiencies(rx) -> List[Tuple[str, float]]:
    tb = rx.third_body
    if tb is None:
        return []
    try:
        effs = tb.efficiencies or {}
    except (AttributeError, ValueError):
        effs = {}
    return [(name, float(eff)) for name, eff in effs.items()]
