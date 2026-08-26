"""
mech_diagnostics.py
-------------------
Cantera-based mechanism diagnostics for MUQ-SAC workflows.

Three checks are implemented:

1. **NASA-7 thermo continuity at Tmid** — the high- and low-T polynomials must
   agree at the midpoint temperature.  A discontinuity > ``thermo_tol`` K (in Cp/R)
   indicates a bad fit or wrong polynomial order.

2. **Collision-limit check** — elementary bimolecular reactions whose pre-exponential
   factor ``A`` exceeds the gas-kinetic collision-limit (~10^14 cm³/(mol·s) in CGS)
   are flagged.  These are almost always analogy-rate estimates that were not
   re-fitted and will produce unphysical rates at high T.

3. **Reverse-rate sanity** — for exothermic reactions where the equilibrium constant
   K_eq >> 1, the reverse rate must be much smaller than the forward.  Reactions where
   `k_rev / k_fwd > rev_ratio_tol` at ``T_ref`` are flagged as potentially overestimated
   reverse rates.

Usage (standalone)::

    python mech_diagnostics.py mech.yaml [--thermo-tol 0.01] [--coll-limit 1e14]
                                          [--rev-tol 0.1] [--T-ref 1500]

Usage (Python API)::

    from mech_diagnostics import MechDiagnostics
    diag = MechDiagnostics("mech.yaml")
    report = diag.run()
    diag.print_report(report)

The diagnostics require Cantera (≥ 3.0).  If Cantera is absent the module still
imports; ``run()`` raises ``ImportError`` with a clear message.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ThermoIssue:
    species_name: str
    tmid: float
    delta_cp_over_R: float   # |Cp_high(Tmid) - Cp_low(Tmid)| / R


@dataclass
class CollisionLimitIssue:
    reaction_index: int
    equation: str
    A_cgs: float             # A in cm³/(mol·s) or s⁻¹


@dataclass
class ReverseRateIssue:
    reaction_index: int
    equation: str
    k_fwd: float
    k_rev: float
    ratio: float             # k_rev / k_fwd at T_ref
    T_ref: float


@dataclass
class DiagnosticsReport:
    mechanism_file: str
    n_species: int
    n_reactions: int
    thermo_issues: List[ThermoIssue] = field(default_factory=list)
    collision_limit_issues: List[CollisionLimitIssue] = field(default_factory=list)
    reverse_rate_issues: List[ReverseRateIssue] = field(default_factory=list)

    @property
    def n_issues(self) -> int:
        return (len(self.thermo_issues)
                + len(self.collision_limit_issues)
                + len(self.reverse_rate_issues))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MechDiagnostics:
    """
    Parameters
    ----------
    mech_file : str
        Path to a Cantera YAML (or CTI) mechanism file.
    thermo_tol : float
        Max allowed |ΔCp/R| discontinuity at Tmid (default 0.01).
    coll_limit_cgs : float
        Pre-exponential collision-limit in CGS cm³/(mol·s).  Reactions with A above
        this value are flagged (default 1e14).
    rev_ratio_tol : float
        Flag if k_rev/k_fwd > this at T_ref (default 0.1).
    T_ref : float
        Temperature (K) used for reverse-rate check (default 1500.0).
    """

    def __init__(
        self,
        mech_file: str,
        thermo_tol: float = 0.01,
        coll_limit_cgs: float = 1e14,
        rev_ratio_tol: float = 0.1,
        T_ref: float = 1500.0,
    ):
        self.mech_file = mech_file
        self.thermo_tol = thermo_tol
        self.coll_limit_cgs = coll_limit_cgs
        self.rev_ratio_tol = rev_ratio_tol
        self.T_ref = T_ref

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> DiagnosticsReport:
        try:
            import cantera as ct
        except ImportError:
            raise ImportError(
                "Cantera is required for mech_diagnostics. "
                "Install with: pip install cantera"
            )

        gas = ct.Solution(self.mech_file)
        report = DiagnosticsReport(
            mechanism_file=self.mech_file,
            n_species=gas.n_species,
            n_reactions=gas.n_reactions,
        )

        self._check_thermo(gas, report)
        self._check_collision_limit(gas, report)
        self._check_reverse_rates(gas, report)
        return report

    @staticmethod
    def print_report(report: DiagnosticsReport) -> None:
        sep = "─" * 70
        print(f"\n{sep}")
        print(f"  Mechanism Diagnostics: {report.mechanism_file}")
        print(f"  {report.n_species} species  |  {report.n_reactions} reactions  |  {report.n_issues} issues found")
        print(sep)

        if report.thermo_issues:
            print(f"\n  [1] NASA-7 Tmid discontinuity  ({len(report.thermo_issues)} species)")
            print(f"      {'Species':<25} {'Tmid':>7}  {'|ΔCp/R|':>9}")
            for iss in sorted(report.thermo_issues, key=lambda x: -x.delta_cp_over_R):
                print(f"      {iss.species_name:<25} {iss.tmid:>7.0f}  {iss.delta_cp_over_R:>9.4f}")
        else:
            print("\n  [1] NASA-7 Tmid discontinuity  ✓  none found")

        if report.collision_limit_issues:
            print(f"\n  [2] Collision-limit exceeded  ({len(report.collision_limit_issues)} reactions)")
            print(f"      {'Rxn':>4}  {'A (CGS)':>14}  Equation")
            for iss in sorted(report.collision_limit_issues, key=lambda x: -x.A_cgs):
                print(f"      {iss.reaction_index:>4}  {iss.A_cgs:>14.3e}  {iss.equation}")
        else:
            print("\n  [2] Collision-limit check       ✓  none found")

        if report.reverse_rate_issues:
            print(f"\n  [3] Reverse-rate overestimate  ({len(report.reverse_rate_issues)} reactions)"
                  f"  [T_ref = {report.reverse_rate_issues[0].T_ref} K]")
            print(f"      {'Rxn':>4}  {'k_fwd':>12}  {'k_rev':>12}  {'ratio':>8}  Equation")
            for iss in sorted(report.reverse_rate_issues, key=lambda x: -x.ratio):
                print(f"      {iss.reaction_index:>4}  {iss.k_fwd:>12.3e}  {iss.k_rev:>12.3e}"
                      f"  {iss.ratio:>8.4f}  {iss.equation}")
        else:
            print("\n  [3] Reverse-rate sanity check   ✓  none found")

        print(f"\n{sep}\n")

    # ------------------------------------------------------------------
    # Private checkers
    # ------------------------------------------------------------------

    def _check_thermo(self, gas, report: DiagnosticsReport) -> None:
        """NASA-7 Cp/R continuity at Tmid for every species."""
        for sp in gas.species():
            thermo = sp.thermo
            # NasaPoly2 has coeffs for [low, high] T ranges
            if not hasattr(thermo, 'coeffs'):
                continue
            coeffs = thermo.coeffs    # 15-element array: [Tmid, 7 low, 7 high]
            if len(coeffs) < 15:
                continue
            tmid = coeffs[0]
            lo = coeffs[1:8]          # low-T polynomial coefficients
            hi = coeffs[8:15]         # high-T polynomial coefficients
            # Cp/R = a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4
            cp_lo = lo[0] + lo[1]*tmid + lo[2]*tmid**2 + lo[3]*tmid**3 + lo[4]*tmid**4
            cp_hi = hi[0] + hi[1]*tmid + hi[2]*tmid**2 + hi[3]*tmid**3 + hi[4]*tmid**4
            delta = abs(cp_hi - cp_lo)
            if delta > self.thermo_tol:
                report.thermo_issues.append(
                    ThermoIssue(species_name=sp.name, tmid=tmid, delta_cp_over_R=delta)
                )

    def _check_collision_limit(self, gas, report: DiagnosticsReport) -> None:
        """Flag bimolecular reactions whose A exceeds the collision limit (CGS)."""
        import cantera as ct
        for i, rxn in enumerate(gas.reactions()):
            rate = rxn.rate
            # Only check simple Arrhenius (ElementaryReaction)
            if not hasattr(rate, 'pre_exponential_factor'):
                continue
            A_si = rate.pre_exponential_factor   # SI: m³/(kmol·s) or s⁻¹
            n_reactants = sum(
                v for v in rxn.reactants.values()
            )
            if n_reactants < 2:
                continue
            # Convert bimolecular A: SI (m³/kmol/s) → CGS (cm³/mol/s)
            # 1 m³/kmol/s = 1e3 cm³/mol/s  (×1e6 for m³→cm³, ×1e-3 for kmol→mol)
            A_cgs = A_si * 1e3
            if A_cgs > self.coll_limit_cgs:
                report.collision_limit_issues.append(
                    CollisionLimitIssue(
                        reaction_index=i,
                        equation=rxn.equation,
                        A_cgs=A_cgs,
                    )
                )

    def _check_reverse_rates(self, gas, report: DiagnosticsReport) -> None:
        """Flag reactions where k_rev / k_fwd exceeds the tolerance at T_ref."""
        import cantera as ct
        gas.TPX = self.T_ref, ct.one_atm, {gas.species_names[0]: 1.0}
        kf = gas.forward_rate_constants   # shape (n_reactions,)
        kr = gas.reverse_rate_constants   # shape (n_reactions,)
        for i, rxn in enumerate(gas.reactions()):
            if kf[i] <= 0.0:
                continue
            ratio = kr[i] / kf[i]
            if ratio > self.rev_ratio_tol:
                report.reverse_rate_issues.append(
                    ReverseRateIssue(
                        reaction_index=i,
                        equation=rxn.equation,
                        k_fwd=kf[i],
                        k_rev=kr[i],
                        ratio=ratio,
                        T_ref=self.T_ref,
                    )
                )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cantera-based mechanism diagnostics (thermo + kinetics checks)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("mech", help="Cantera YAML or CTI mechanism file")
    p.add_argument("--thermo-tol", type=float, default=0.01,
                   help="Max |ΔCp/R| at Tmid")
    p.add_argument("--coll-limit", type=float, default=1e14,
                   help="Collision-limit A (CGS, cm³/mol/s)")
    p.add_argument("--rev-tol", type=float, default=0.1,
                   help="k_rev/k_fwd threshold")
    p.add_argument("--T-ref", type=float, default=1500.0,
                   help="Temperature (K) for reverse-rate check")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    diag = MechDiagnostics(
        mech_file=args.mech,
        thermo_tol=args.thermo_tol,
        coll_limit_cgs=args.coll_limit,
        rev_ratio_tol=args.rev_tol,
        T_ref=args.T_ref,
    )
    report = diag.run()
    MechDiagnostics.print_report(report)
    sys.exit(0 if report.n_issues == 0 else 1)
