"""
rmip/simulation/surface_crosscheck.py
=====================================
W5b — Surface-mechanism cross-solver validator (P21 detector).

RMG's own surface solver and Cantera have been observed to give different
results for the same surface mechanism (P21, an open interoperability
discrepancy whose fix lives inside RMG).  This module operationalises that
as a *detector*: run the same surface case through Cantera's Interface
kinetics and through an RMG-side evaluation, align species/reactions, and
diff

- per-reaction standard-state equilibrium constants, compared as
  ``log10 K = -ΔG°_rxn / (RT ln 10)`` — a dimensionless, unit-convention-free
  quantity both sides can compute from their own thermo (the P21 Keq
  inconsistency class), and
- steady-state surface coverages (when the RMG side provides them).

RMG side — two modes
--------------------
1. **Live** (needs the RMG env): ``_rmg_surface_backend.py`` is launched via
   subprocess in the RMG interpreter (same pattern as
   ``generation/reaction_family_kb.py``); it loads the mechanism's CHEMKIN
   gas+surface files with ``rmgpy.chemkin.load_chemkin_file`` and reports
   per-reaction log10 K from RMG's thermo.  RMG cannot read Cantera YAML, so
   live mode requires the CHEMKIN-format surface input
   (``rmg_chemkin_gas=``/``rmg_chemkin_surf=``).
2. **Offline** (``rmg_json=``): a pre-computed RMG-side result JSON —
   ``{"reactions": [{"equation": ..., "log10_keq": ...}, ...],
   "coverages": {name: theta, ...}}`` — produced by the user's own RMG run.
   This is also how the diff engine is tested on machines without RMG.

Reactions are aligned by a canonicalised equation (sides sorted, arrows and
whitespace normalised); species by name, with an optional ``species_map``
translating RMG names to Cantera names.  Unmatched entries are reported,
never silently dropped.
"""

from __future__ import annotations

import json
import logging
import math
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

_LN10 = math.log(10.0)
_R_KMOL = 8314.462618          # J / kmol / K  (Cantera's molar basis)

_DEFAULT_RMG_PYTHON = "/home/krithika/anaconda3/envs/rmg_env/bin/python"
_BACKEND = Path(__file__).parent / "_rmg_surface_backend.py"


# ---------------------------------------------------------------------------
# Report data model
# ---------------------------------------------------------------------------
@dataclass
class ReactionComparison:
    equation: str                       # Cantera-side equation string
    log10_keq_ct: float
    log10_keq_rmg: float
    flagged: bool

    @property
    def delta(self) -> float:
        return self.log10_keq_rmg - self.log10_keq_ct


@dataclass
class CoverageComparison:
    species: str
    theta_ct: float
    theta_rmg: float
    flagged: bool

    @property
    def delta(self) -> float:
        return self.theta_rmg - self.theta_ct


@dataclass
class CrossCheckReport:
    mechanism: str
    T_K: float
    P_atm: float
    reactions: List[ReactionComparison] = field(default_factory=list)
    coverages: List[CoverageComparison] = field(default_factory=list)
    unmatched_cantera: List[str] = field(default_factory=list)
    unmatched_rmg: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def divergent(self) -> bool:
        return (any(r.flagged for r in self.reactions)
                or any(c.flagged for c in self.coverages))

    def ranked_reactions(self) -> List[ReactionComparison]:
        return sorted(self.reactions, key=lambda r: -abs(r.delta))

    def to_findings(self):
        """Emit mechanism_check-style ``Finding``s, ranked by magnitude."""
        from rmip.analysis.mechanism_check import Finding
        out = []
        for r in self.ranked_reactions():
            if not r.flagged:
                continue
            out.append(Finding(
                check="surface_crosscheck", severity="warning",
                category="keq-divergence",
                message=(f"Cantera and RMG disagree on Keq: "
                         f"Δlog10 K = {r.delta:+.2f} "
                         f"(Cantera {r.log10_keq_ct:.2f}, "
                         f"RMG {r.log10_keq_rmg:.2f}) at {self.T_K:.0f} K"),
                location=r.equation,
                details={"log10_keq_cantera": r.log10_keq_ct,
                         "log10_keq_rmg": r.log10_keq_rmg,
                         "T_K": self.T_K},
            ))
        for c in sorted(self.coverages, key=lambda c: -abs(c.delta)):
            if not c.flagged:
                continue
            out.append(Finding(
                check="surface_crosscheck", severity="warning",
                category="coverage-divergence",
                message=(f"Steady-state coverage diverges: "
                         f"Δθ = {c.delta:+.3f} (Cantera {c.theta_ct:.3f}, "
                         f"RMG {c.theta_rmg:.3f})"),
                location=c.species,
                details={"theta_cantera": c.theta_ct, "theta_rmg": c.theta_rmg},
            ))
        return out

    def to_dict(self) -> Dict:
        return {
            "mechanism": self.mechanism,
            "T_K": self.T_K, "P_atm": self.P_atm,
            "divergent": self.divergent,
            "reactions": [
                {"equation": r.equation, "log10_keq_cantera": r.log10_keq_ct,
                 "log10_keq_rmg": r.log10_keq_rmg, "delta": r.delta,
                 "flagged": r.flagged}
                for r in self.ranked_reactions()],
            "coverages": [
                {"species": c.species, "theta_cantera": c.theta_ct,
                 "theta_rmg": c.theta_rmg, "delta": c.delta,
                 "flagged": c.flagged}
                for c in self.coverages],
            "unmatched_cantera": self.unmatched_cantera,
            "unmatched_rmg": self.unmatched_rmg,
            "notes": self.notes,
        }

    def render_text(self) -> str:
        lines = [
            f"Surface cross-check — {self.mechanism}",
            f"  T = {self.T_K:.1f} K, P = {self.P_atm:g} atm",
            f"  verdict: {'DIVERGENT' if self.divergent else 'consistent'}",
            "",
            f"  {'Δlog10K':>9}  {'Cantera':>9}  {'RMG':>9}  reaction",
        ]
        for r in self.ranked_reactions():
            mark = " ⚠" if r.flagged else "  "
            lines.append(f"  {r.delta:>+9.3f}  {r.log10_keq_ct:>9.3f}  "
                         f"{r.log10_keq_rmg:>9.3f}  {r.equation}{mark}")
        if self.coverages:
            lines += ["", f"  {'Δθ':>9}  {'Cantera':>9}  {'RMG':>9}  species"]
            for c in sorted(self.coverages, key=lambda c: -abs(c.delta)):
                mark = " ⚠" if c.flagged else "  "
                lines.append(f"  {c.delta:>+9.4f}  {c.theta_ct:>9.4f}  "
                             f"{c.theta_rmg:>9.4f}  {c.species}{mark}")
        for grp, label in ((self.unmatched_cantera, "Cantera-only"),
                           (self.unmatched_rmg, "RMG-only")):
            if grp:
                lines += ["", f"  {label} (no counterpart on the other side):"]
                lines += [f"    - {e}" for e in grp]
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Equation canonicalisation (alignment key)
# ---------------------------------------------------------------------------
def canonical_equation(eq: str) -> str:
    """Normalise a reaction equation so trivially-reordered/re-arrowed forms
    of the same reaction compare equal: sides sorted, arrows unified,
    whitespace stripped."""
    s = eq.strip()
    arrow = "<=>"
    for a in ("<=>", "=>", "="):
        if a in s:
            arrow = a
            break
    lhs, rhs = re.split(r"<=>|=>|(?<![<>=])=(?![<>=])", s, maxsplit=1)

    def side(x: str) -> str:
        terms = [t.strip() for t in x.split("+")]
        return " + ".join(sorted(t for t in terms if t))

    return f"{side(lhs)} {arrow} {side(rhs)}"


# ---------------------------------------------------------------------------
# The checker
# ---------------------------------------------------------------------------
class SurfaceCrossChecker:
    """Compare a surface mechanism between Cantera and an RMG-side result.

    Parameters
    ----------
    mech : str — Cantera YAML with a gas and an ``ideal-surface`` phase
        (adjacent bulk phases are loaded automatically).
    T_K, P_atm, X : the shared case condition.
    rmg_python : RMG-env interpreter for live mode.
    rmg_chemkin_gas, rmg_chemkin_surf : CHEMKIN files for live mode (RMG
        cannot read Cantera YAML).
    species_map : optional {rmg_name: cantera_name} translation.
    tol_log10_keq : |Δ log10 K| flag threshold (0.5 ≈ a factor of 3).
    tol_coverage : absolute Δθ flag threshold.
    """

    def __init__(
        self,
        mech: str,
        gas_phase: Optional[str] = None,
        surface_phase: Optional[str] = None,
        T_K: float = 1000.0,
        P_atm: float = 1.0,
        X: Union[str, Dict[str, float]] = "O2:0.21, N2:0.79",
        rmg_python: Optional[str] = None,
        rmg_chemkin_gas: Optional[str] = None,
        rmg_chemkin_surf: Optional[str] = None,
        species_map: Optional[Dict[str, str]] = None,
        tol_log10_keq: float = 0.5,
        tol_coverage: float = 0.05,
    ) -> None:
        self.mech = str(mech)
        self.gas_phase = gas_phase
        self.surface_phase = surface_phase
        self.T_K = float(T_K)
        self.P_atm = float(P_atm)
        self.X = X
        self.rmg_python = rmg_python or _DEFAULT_RMG_PYTHON
        self.rmg_chemkin_gas = rmg_chemkin_gas
        self.rmg_chemkin_surf = rmg_chemkin_surf
        self.species_map = dict(species_map or {})
        self.tol_log10_keq = float(tol_log10_keq)
        self.tol_coverage = float(tol_coverage)

    # ------------------------------------------------------------------
    # Cantera side
    # ------------------------------------------------------------------
    def cantera_side(self) -> Dict:
        """Evaluate the case with Cantera Interface kinetics.

        Returns the same JSON-able shape the RMG side uses, so the diff
        engine is symmetric:
        ``{"reactions": [{"equation", "log10_keq"}], "coverages": {...}}``
        """
        import cantera as ct
        # rmip.analysis.yaml_safe, not plain yaml.safe_load: stock PyYAML
        # resolves a bare species name "NO" as a YAML-1.1 bool, silently
        # corrupting species-name lists (see yaml_safe.py docstring).
        from rmip.analysis.yaml_safe import safe_load_path

        doc = safe_load_path(self.mech)
        phases = {p["name"]: p for p in doc.get("phases", [])}

        surf_name = self.surface_phase
        if surf_name is None:
            cands = [n for n, p in phases.items()
                     if "surface" in str(p.get("thermo", ""))]
            if len(cands) != 1:
                raise ValueError(
                    f"Cannot auto-detect the surface phase (candidates "
                    f"{cands or 'none'}); pass surface_phase= explicitly.")
            surf_name = cands[0]

        gas_name = self.gas_phase
        if gas_name is None:
            cands = [n for n, p in phases.items()
                     if str(p.get("thermo", "")) == "ideal-gas"]
            if len(cands) != 1:
                raise ValueError(
                    f"Cannot auto-detect the gas phase (candidates "
                    f"{cands or 'none'}); pass gas_phase= explicitly.")
            gas_name = cands[0]

        adjacent = phases[surf_name].get("adjacent-phases") or [gas_name]
        sols = [ct.Solution(self.mech, n) for n in adjacent]
        gas = next((s for s in sols if s.name == gas_name), sols[0])
        surf = ct.Interface(self.mech, surf_name, sols)

        gas.TPX = self.T_K, self.P_atm * 101325.0, self.X
        for s in sols:
            s.TP = self.T_K, self.P_atm * 101325.0
        surf.TP = self.T_K, self.P_atm * 101325.0

        # Steady-state coverages at fixed gas composition.
        surf.advance_coverages(1.0e4)

        # Dimensionless standard-state K from ΔG°: unit-convention-free.
        dg = surf.delta_standard_gibbs                    # J/kmol
        log10_keq = [-float(g) / (_R_KMOL * self.T_K * _LN10) for g in dg]

        return {
            "reactions": [
                {"equation": surf.reaction(i).equation,
                 "log10_keq": log10_keq[i]}
                for i in range(surf.n_reactions)],
            "coverages": {n: float(t) for n, t in
                          zip(surf.species_names, surf.coverages)},
        }

    # ------------------------------------------------------------------
    # RMG side
    # ------------------------------------------------------------------
    def rmg_side(self, rmg_json: Optional[str] = None) -> Dict:
        if rmg_json:
            with open(rmg_json) as fh:
                return json.load(fh)

        if not (self.rmg_chemkin_gas and self.rmg_chemkin_surf):
            raise RuntimeError(
                "Live RMG mode needs the CHEMKIN-format surface mechanism "
                "(rmg_chemkin_gas= and rmg_chemkin_surf= / --rmg-chemkin-gas "
                "and --rmg-chemkin-surf): RMG cannot read Cantera YAML. "
                "Alternatively pass a pre-computed RMG result via "
                "rmg_json= / --rmg-json.")
        if not Path(self.rmg_python).exists():
            raise RuntimeError(
                f"RMG interpreter not found at {self.rmg_python}. "
                "Pass rmg_python= / --rmg-python for your RMG env, or use "
                "offline mode (--rmg-json).")

        from rmip.simulation.subprocess_util import (
            resolve_timeout, sanitized_subprocess_env)
        cfg = {
            "chemkin_gas": str(self.rmg_chemkin_gas),
            "chemkin_surf": str(self.rmg_chemkin_surf),
            "T_K": self.T_K,
        }
        proc = subprocess.run(
            [self.rmg_python, str(_BACKEND), json.dumps(cfg)],
            capture_output=True, text=True,
            env=sanitized_subprocess_env(),
            timeout=resolve_timeout(None))
        if proc.returncode != 0:
            raise RuntimeError(
                f"RMG surface backend failed (rc={proc.returncode}):\n"
                f"{proc.stderr[-4000:]}")
        return json.loads(proc.stdout)

    # ------------------------------------------------------------------
    # Diff engine
    # ------------------------------------------------------------------
    def run(self, rmg_json: Optional[str] = None) -> CrossCheckReport:
        ct_res = self.cantera_side()
        rmg_res = self.rmg_side(rmg_json=rmg_json)

        report = CrossCheckReport(mechanism=self.mech, T_K=self.T_K,
                                  P_atm=self.P_atm)

        # ---- reactions, matched on canonicalised equations --------------
        def rmg_translate(eq: str) -> str:
            for rmg_name, ct_name in self.species_map.items():
                eq = re.sub(rf"(?<![\w(]){re.escape(rmg_name)}(?![\w)])",
                            ct_name, eq)
            return eq

        ct_rxns = {canonical_equation(r["equation"]): r
                   for r in ct_res.get("reactions", [])}
        rmg_rxns = {canonical_equation(rmg_translate(r["equation"])): r
                    for r in rmg_res.get("reactions", [])}

        for key, cr in ct_rxns.items():
            rr = rmg_rxns.get(key)
            if rr is None:
                report.unmatched_cantera.append(cr["equation"])
                continue
            delta = float(rr["log10_keq"]) - float(cr["log10_keq"])
            report.reactions.append(ReactionComparison(
                equation=cr["equation"],
                log10_keq_ct=float(cr["log10_keq"]),
                log10_keq_rmg=float(rr["log10_keq"]),
                flagged=abs(delta) > self.tol_log10_keq))
        for key, rr in rmg_rxns.items():
            if key not in ct_rxns:
                report.unmatched_rmg.append(rr["equation"])

        # ---- coverages (only if the RMG side has them) -------------------
        rmg_cov_raw = rmg_res.get("coverages") or {}
        rmg_cov = {self.species_map.get(k, k): v
                   for k, v in rmg_cov_raw.items()}
        ct_cov = ct_res.get("coverages") or {}
        if rmg_cov:
            for name, th_ct in ct_cov.items():
                if name not in rmg_cov:
                    continue
                th_rmg = float(rmg_cov[name])
                report.coverages.append(CoverageComparison(
                    species=name, theta_ct=float(th_ct), theta_rmg=th_rmg,
                    flagged=abs(th_rmg - float(th_ct)) > self.tol_coverage))
        else:
            report.notes.append(
                "RMG side provided no coverages — Keq-only comparison "
                "(supply coverages via the RMG-result JSON to diff them too).")

        return report
