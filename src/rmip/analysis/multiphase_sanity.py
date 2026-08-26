"""
rmip/analysis/multiphase_sanity.py
==================================
W5c — liquid / multiphase composition sanity (P25).

The P25 report: a liquid-phase solvent set *by mole fraction* came out ~10x
below its known pure concentration, and the root cause was never identified.
This module turns that failure into a diagnosable finding, via two
independent probes on every condensed (non-gas, non-surface) phase:

1. **Internal identity** — for each species, set the phase to the pure
   composition (X_i = 1) and verify Cantera's reported concentration equals
   the independently computed ``rho / W_i``.  A violation means the phase's
   activity/EoS model and its composition handling disagree — the
   "mole-fraction vs concentration confusion" class of P25.

2. **Reference magnitude** — compare the pure-species molar concentration
   against a small built-in table of well-known pure-liquid concentrations
   (water 55.3 M, ethanol 17.1 M, ...), matched by **elemental composition**
   (never by name, so H2O(l)/WATER/H2O all match, and isomers of different
   phases don't false-match by label).  An order-of-magnitude mismatch is
   flagged with the likely cause: a wrong species density / molar volume in
   the ``equation-of-state`` entry — which reproduces the P25 gap exactly
   (a 10x-low density gives a 10x-low concentration).

Also checks that condensed-phase densities are physically plausible.
All findings are **warnings** — like thermo_provenance, disagreement never
fails a verdict on its own; whether a phase model is wrong is a human call.

Surfaces two ways: the ``multiphase_sanity`` check id in ``rmip check``
(auto-skips mechanisms with no condensed phases — zero noise for pure-gas
mechanisms), or standalone via ``MultiphaseSanityChecker(path).findings()``.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Pure-liquid molar concentrations at ~298 K, mol/L, keyed by elemental
# composition (physical constants; CRC-handbook values).
_PURE_LIQUID_REFS: Dict[Tuple[Tuple[str, int], ...], Tuple[str, float, float]] = {}


def _key(comp: Dict[str, int]) -> Tuple[Tuple[str, int], ...]:
    return tuple(sorted((str(e).capitalize(), int(n)) for e, n in comp.items()
                        if n))


def _add_ref(name: str, comp: Dict[str, int], conc_mol_L: float,
             rho_kg_m3: float) -> None:
    _PURE_LIQUID_REFS[_key(comp)] = (name, conc_mol_L, rho_kg_m3)


_add_ref("water",      {"H": 2, "O": 1},          55.35, 997.0)
_add_ref("methanol",   {"C": 1, "H": 4, "O": 1},  24.55, 792.0)
_add_ref("ethanol",    {"C": 2, "H": 6, "O": 1},  17.13, 789.0)
_add_ref("acetone",    {"C": 3, "H": 6, "O": 1},  13.50, 784.0)
_add_ref("benzene",    {"C": 6, "H": 6},          11.17, 876.0)
_add_ref("toluene",    {"C": 7, "H": 8},           9.38, 867.0)
_add_ref("n-heptane",  {"C": 7, "H": 16},          6.79, 680.0)
_add_ref("iso-octane", {"C": 8, "H": 18},          5.75, 692.0)
_add_ref("n-decane",   {"C": 10, "H": 22},         5.13, 730.0)
_add_ref("n-dodecane", {"C": 12, "H": 26},         4.40, 750.0)


class MultiphaseSanityChecker:
    """Run the P25 composition-sanity probes on a mechanism YAML's condensed
    phases.

    Parameters
    ----------
    path : str — Cantera YAML file.
    T_K, P_Pa : evaluation state (default 298.15 K, 1 atm — where the
        reference concentrations hold).
    tol_log10 : reference-magnitude flag threshold in decades (default 0.5,
        i.e. a factor ~3; the P25 case was a full decade).
    identity_rtol : relative tolerance for the internal identity probe.
    """

    def __init__(self, path: str, T_K: float = 298.15, P_Pa: float = 101325.0,
                 tol_log10: float = 0.5, identity_rtol: float = 0.01) -> None:
        self.path = str(path)
        self.T_K = float(T_K)
        self.P_Pa = float(P_Pa)
        self.tol_log10 = float(tol_log10)
        self.identity_rtol = float(identity_rtol)

    # ------------------------------------------------------------------
    def condensed_phase_names(self) -> List[str]:
        # rmip.analysis.yaml_safe, not plain yaml.safe_load: stock PyYAML
        # resolves a bare species name "NO" as a YAML-1.1 bool (see
        # yaml_safe.py docstring), silently corrupting species-name lists.
        from rmip.analysis.yaml_safe import safe_load_path
        doc = safe_load_path(self.path) or {}
        out = []
        gas_like = {"ideal-gas", "Redlich-Kwong", "Peng-Robinson", "plasma"}
        for ph in doc.get("phases", []) or []:
            model = str(ph.get("thermo", ""))
            if model in gas_like or "surface" in model or "edge" in model:
                continue
            out.append(ph["name"])
        return out

    # ------------------------------------------------------------------
    def findings(self) -> List[Any]:
        from rmip.analysis.mechanism_check import Finding
        import cantera as ct

        findings: List[Finding] = []
        for name in self.condensed_phase_names():
            try:
                phase = ct.Solution(self.path, name)
            except Exception as exc:                       # noqa: BLE001
                findings.append(Finding(
                    check="multiphase_sanity", severity="warning",
                    category="phase-load-failure",
                    message=f"Condensed phase '{name}' failed to load: "
                            f"{str(exc).strip().splitlines()[-1]}",
                    location=name, details={"error": str(exc)}))
                continue
            try:
                # Real-gas phases (Redlich-Kwong etc.) are not ideal-gas by
                # thermo model but are still gases — not this check's business.
                if phase.phase_of_matter == "gas":
                    continue
            except Exception:                              # noqa: BLE001
                pass
            findings.extend(self._check_phase(phase, name))
        return findings

    # ------------------------------------------------------------------
    def _check_phase(self, phase, phase_name: str) -> List[Any]:
        from rmip.analysis.mechanism_check import Finding
        findings: List[Finding] = []

        for j, sp in enumerate(phase.species_names):
            try:
                if phase.n_species > 1:
                    X = {sp: 1.0}
                    phase.TPX = self.T_K, self.P_Pa, X
                else:
                    phase.TP = self.T_K, self.P_Pa
            except Exception as exc:                       # noqa: BLE001
                findings.append(Finding(
                    check="multiphase_sanity", severity="warning",
                    category="composition-set-failure",
                    message=f"Setting pure-{sp} composition on phase "
                            f"'{phase_name}' failed: {exc}",
                    location=f"{phase_name}/{sp}"))
                continue

            rho = float(phase.density)                     # kg/m^3
            W = float(phase.molecular_weights[j])          # kg/kmol
            c_expected = rho / W                           # kmol/m^3 == mol/L

            # -- probe 1: internal identity (P25's confusion class) -------
            c_reported = float(phase.concentrations[j])
            if c_expected > 0 and not math.isclose(
                    c_reported, c_expected,
                    rel_tol=self.identity_rtol, abs_tol=1e-30):
                findings.append(Finding(
                    check="multiphase_sanity", severity="warning",
                    category="concentration-identity",
                    message=(
                        f"Phase '{phase_name}': pure {sp} set by mole "
                        f"fraction reports {c_reported:.3g} mol/L but "
                        f"density/W gives {c_expected:.3g} mol/L "
                        f"({c_reported / c_expected:.2g}x) — the phase's "
                        "activity/EoS model and its composition handling "
                        "disagree (mole-fraction vs concentration "
                        "confusion, P25)."),
                    location=f"{phase_name}/{sp}",
                    details={"c_reported_mol_L": c_reported,
                             "c_expected_mol_L": c_expected}))

            # -- probe 2: reference magnitude ------------------------------
            comp = {e: phase.n_atoms(sp, e) for e in phase.element_names
                    if phase.n_atoms(sp, e)}
            ref = _PURE_LIQUID_REFS.get(_key(comp))
            if ref is not None and c_expected > 0:
                ref_name, c_ref, rho_ref = ref
                dec = math.log10(c_expected / c_ref)
                if abs(dec) > self.tol_log10:
                    findings.append(Finding(
                        check="multiphase_sanity", severity="warning",
                        category="pure-concentration-mismatch",
                        message=(
                            f"Phase '{phase_name}': pure {sp} comes out at "
                            f"{c_expected:.3g} mol/L, but {ref_name} is "
                            f"{c_ref:.4g} mol/L ({10**abs(dec):.1f}x "
                            f"{'below' if dec < 0 else 'above'}) — likely a "
                            f"wrong species density / molar volume in the "
                            f"equation-of-state (phase density here "
                            f"{phase.density:.4g} kg/m^3; {ref_name} is "
                            f"~{rho_ref:.0f})."),
                        location=f"{phase_name}/{sp}",
                        details={"c_pure_mol_L": c_expected,
                                 "c_ref_mol_L": c_ref,
                                 "reference": ref_name,
                                 "log10_ratio": dec,
                                 "phase_density_kg_m3": float(phase.density),
                                 "ref_density_kg_m3": rho_ref}))

            # -- density plausibility --------------------------------------
            if not (50.0 <= rho <= 25000.0):
                findings.append(Finding(
                    check="multiphase_sanity", severity="warning",
                    category="implausible-density",
                    message=(f"Phase '{phase_name}' with pure {sp} has "
                             f"density {rho:.4g} kg/m^3 — outside the "
                             "plausible condensed-phase range (50–25000)."),
                    location=f"{phase_name}/{sp}",
                    details={"density_kg_m3": rho}))

        return findings
