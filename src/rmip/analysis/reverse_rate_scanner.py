"""
rmip/analysis/reverse_rate_scanner.py
======================================
Future.md Part 2, Feature 1 — Reverse Rate Unphysicality Scanner.

Many CHEMKIN mechanisms specify only a forward Arrhenius expression for a
reversible reaction (no ``REV /`` auxiliary block).  The solver then derives
the reverse rate from the forward rate and the equilibrium constant:

    k_rev(T) = k_fwd(T) / Kc(T)

If the mechanism's thermo is inconsistent (bad NASA-7 fit, wrong ΔHf, Tmid
discontinuity) or the forward A-factor itself is unphysical, the *implied*
reverse rate can blow past the gas-kinetic collision limit — a common root
cause of the stiff-integrator convergence failures documented in
``solved_error.md``. This scanner flags exactly that.

Algorithm
---------
For each reversible reaction with full NASA-7 thermo on every participating
species:

  1. ΔH°(T), ΔS°(T) — via NASA-7 polynomials, stoichiometry-weighted sum
     over products minus reactants (third-body "M" is not a stoichiometric
     species in RMIP's schema, so it drops out naturally).
  2. ΔG°(T) = ΔH°(T) − T·ΔS°(T)
  3. Kp(T)  = exp(−ΔG°(T) / R·T)                  (dimensionless, P° = 1 atm)
  4. Kc(T)  = Kp(T) · (P°/RT)^Δn                    Δn = Σν_products − Σν_reactants
  5. k_rev(T) = k_fwd(T) / Kc(T)

Only reactions whose *reverse* reactant order (= the forward reaction's
product-side stoichiometric sum) is exactly 2 are collision-limit-checked —
collision theory gives a clean upper bound only for bimolecular steps.
Reactions with any other reverse order are evaluated for k_rev but excluded
from flagging (recorded as skipped).

Collision limit (heuristic, T-scaled): ``1e8 · sqrt(T / 298.15) m^3/mol/s``
(≈ 1e14 cm^3/mol/s at 298 K, the standard combustion-literature ceiling).
A reaction is flagged when ``k_rev / collision_limit > exceedance_threshold``
(default 1e3 — "3 orders of magnitude", per Future.md).

Known limitation
-----------------
RMIP's CHEMKIN parser currently discards ``REV /`` blocks entirely (they are
recognised and skipped — see ``io/_chemkin_parser.py``), so an explicitly
fit reverse Arrhenius cannot be distinguished from a solver-computed one in
the DB. All ``reversible=1`` reactions are scanned uniformly; if a mechanism
does specify ``REV /`` for some reactions, those flags are a false positive
on the *derived* reverse (still informative, since the solver would compute
that same value in the absence of REV — but not reflective of what the
solver will actually integrate for that specific reaction).
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from typing import Any, Dict, List, Optional

from rmip.database.schema import init_db
from rmip.types import ReverseRateFlag

log = logging.getLogger("rmip.analysis.reverse_rate_scanner")

R_GAS: float = 8.314_462_618       # J mol⁻¹ K⁻¹
P_STD: float = 101_325.0           # Pa (1 atm standard state)

_DEFAULT_T_GRID = [600.0, 900.0, 1200.0, 1500.0, 1800.0, 2200.0]


class ReverseRateScanner:
    """
    Scan a mechanism for reactions whose Keq-implied reverse rate exceeds
    the gas-kinetic collision limit.

    Parameters
    ----------
    db_path : str
        Path to the RMIP SQLite database.
    T_grid_K : list[float] | None
        Temperatures at which to evaluate k_rev(T). Defaults to a 6-point
        grid spanning typical combustion conditions (600–2200 K).
    exceedance_threshold : float
        k_rev / collision_limit ratio above which a (reaction, T) pair is
        flagged. Default 1000 (3 orders of magnitude, per Future.md).
    """

    def __init__(
        self,
        db_path: str,
        T_grid_K: Optional[List[float]] = None,
        exceedance_threshold: float = 1000.0,
    ) -> None:
        self._db = db_path
        self._conn = init_db(db_path)
        self._T_grid = T_grid_K or list(_DEFAULT_T_GRID)
        self._threshold = exceedance_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(self, mechanism_id: int) -> List[ReverseRateFlag]:
        """
        Scan every reversible reaction in ``mechanism_id`` and return
        ``ReverseRateFlag`` entries for every (reaction, T) pair evaluated
        (not just flagged ones — callers filter on ``.flagged``).
        """
        species_thermo = self._load_species_thermo(mechanism_id)
        reactions = self._conn.execute(
            """SELECT id, canonical_key, formula_string, reversible,
                      kinetics_type, A_si, n, Ea_si, kinetics_data_json,
                      has_third_body
               FROM reactions
               WHERE mechanism_id=? AND reversible=1""",
            (mechanism_id,),
        ).fetchall()

        flags: List[ReverseRateFlag] = []
        n_skipped_order = 0
        n_skipped_thermo = 0
        n_skipped_kinetics = 0

        for rx in reactions:
            stoich = self._load_stoichiometry(rx["id"])
            if not stoich:
                continue

            missing_thermo = [
                sid for sid in stoich
                if sid not in species_thermo
            ]
            if missing_thermo:
                n_skipped_thermo += 1
                continue

            reactant_order = sum(c for c, role in stoich.values() if role == "reactant")
            product_order = sum(c for c, role in stoich.values() if role == "product")
            # The REVERSE reaction's reactants are the FORWARD reaction's products.
            if abs(product_order - 2.0) > 1e-6:
                n_skipped_order += 1
                continue

            k_fwd_fn = self._forward_rate_fn(rx)
            if k_fwd_fn is None:
                n_skipped_kinetics += 1
                continue

            delta_n = product_order - reactant_order

            for T in self._T_grid:
                k_fwd = k_fwd_fn(T)
                if k_fwd is None or k_fwd <= 0:
                    continue
                Kc = self._equilibrium_constant(stoich, species_thermo, T, delta_n)
                if Kc is None or Kc <= 0:
                    continue
                k_rev = k_fwd / Kc
                limit = _collision_limit(T)
                exceedance = k_rev / limit if limit > 0 else float("inf")
                flags.append(ReverseRateFlag(
                    canonical_key=rx["canonical_key"] or "",
                    formula_string=rx["formula_string"],
                    mechanism_id=mechanism_id,
                    T_K=T,
                    k_fwd=k_fwd,
                    k_rev=k_rev,
                    collision_limit=limit,
                    exceedance_factor=exceedance,
                    flagged=exceedance > self._threshold,
                ))

        log.info(
            "ReverseRateScanner: mechanism_id=%d — %d (reaction,T) pairs evaluated, "
            "%d flagged; skipped %d (non-bimolecular reverse), %d (missing thermo), "
            "%d (unsupported kinetics type)",
            mechanism_id, len(flags), sum(1 for f in flags if f.flagged),
            n_skipped_order, n_skipped_thermo, n_skipped_kinetics,
        )
        return flags

    def store(self, mechanism_id: int, flags: List[ReverseRateFlag]) -> None:
        """Persist scan results to the ``reverse_rate_flags`` table."""
        from rmip.database.schema_phase6 import insert_reverse_rate_flags
        insert_reverse_rate_flags(
            self._conn, mechanism_id, [f.to_dict() for f in flags]
        )

    # ------------------------------------------------------------------
    # Thermo helpers
    # ------------------------------------------------------------------

    def _load_species_thermo(self, mechanism_id: int) -> Dict[int, Dict[str, Any]]:
        rows = self._conn.execute(
            """SELECT id, coeffs_low_json, coeffs_high_json, tmid_K
               FROM species WHERE mechanism_id=?""",
            (mechanism_id,),
        ).fetchall()
        out: Dict[int, Dict[str, Any]] = {}
        for r in rows:
            lo = _parse_json_list(r["coeffs_low_json"])
            hi = _parse_json_list(r["coeffs_high_json"])
            if lo is None or hi is None:
                continue
            out[r["id"]] = {
                "coeffs_low": lo,
                "coeffs_high": hi,
                "tmid_K": r["tmid_K"] or 1000.0,
            }
        return out

    def _load_stoichiometry(self, reaction_id: int) -> Dict[int, tuple]:
        """Return {species_id: (stoich_coeff, role)} for one reaction."""
        rows = self._conn.execute(
            """SELECT species_id, stoich_coeff, role
               FROM stoichiometry WHERE reaction_id=?""",
            (reaction_id,),
        ).fetchall()
        return {r["species_id"]: (r["stoich_coeff"], r["role"]) for r in rows}

    @staticmethod
    def _nasa7_H_S(coeffs: List[float], T: float) -> tuple:
        """Return (H(T) [J/mol], S(T) [J/mol/K]) from a NASA-7 coefficient set."""
        a = coeffs
        H_RT = a[0] + a[1]*T/2.0 + a[2]*T**2/3.0 + a[3]*T**3/4.0 + a[4]*T**4/5.0 + a[5]/T
        S_R  = a[0]*math.log(T) + a[1]*T + a[2]*T**2/2.0 + a[3]*T**3/3.0 + a[4]*T**4/4.0 + a[6]
        return R_GAS * T * H_RT, R_GAS * S_R

    def _equilibrium_constant(
        self,
        stoich: Dict[int, tuple],
        species_thermo: Dict[int, Dict[str, Any]],
        T: float,
        delta_n: float,
    ) -> Optional[float]:
        """Return Kc(T) for the forward reaction (products/reactants)."""
        try:
            dH = 0.0
            dS = 0.0
            for sid, (coeff, role) in stoich.items():
                th = species_thermo[sid]
                coeffs = th["coeffs_low"] if T <= th["tmid_K"] else th["coeffs_high"]
                H, S = self._nasa7_H_S(coeffs, T)
                sign = 1.0 if role == "product" else -1.0
                dH += sign * coeff * H
                dS += sign * coeff * S
            dG = dH - T * dS
            Kp = math.exp(-dG / (R_GAS * T))
            Kc = Kp * (P_STD / (R_GAS * T)) ** delta_n
            return Kc
        except (OverflowError, ValueError, ZeroDivisionError):
            return None

    # ------------------------------------------------------------------
    # Kinetics helpers
    # ------------------------------------------------------------------

    def _forward_rate_fn(self, rx: sqlite3.Row):
        """Return a callable T -> k_fwd(T), or None if unsupported."""
        ktype = (rx["kinetics_type"] or "arrhenius").lower()

        if ktype in ("arrhenius", "multi_arrhenius", "unknown", "third_body"):
            A, n, Ea = rx["A_si"], rx["n"], rx["Ea_si"]
            if A is None or Ea is None or A <= 0:
                return None
            n_val = n or 0.0

            def _k(T: float, A=A, n_val=n_val, Ea=Ea) -> Optional[float]:
                try:
                    return A * (T ** n_val) * math.exp(-Ea / (R_GAS * T))
                except (OverflowError, ValueError, ZeroDivisionError):
                    return None
            return _k

        if ktype == "plog":
            from rmip.analysis.uncertainty_bands import UncertaintyBandEstimator
            estimator = UncertaintyBandEstimator()
            kdata_json = rx["kinetics_data_json"]

            def _k_plog(T: float, kdata_json=kdata_json, estimator=estimator) -> Optional[float]:
                vals = estimator._eval_plog(kdata_json, [T])
                if not vals or vals[0] <= 0:
                    return None
                return vals[0]
            return _k_plog

        # Falloff/Chebyshev: pressure-dependent — out of scope for this
        # 1D-in-T scanner (matches uncertainty_bands.py precedent).
        return None


def _parse_json_list(raw: Optional[str]) -> Optional[List[float]]:
    if not raw:
        return None
    try:
        val = json.loads(raw) if isinstance(raw, str) else raw
        return [float(x) for x in val] if val else None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _collision_limit(T_K: float) -> float:
    """
    Gas-kinetic bimolecular collision-limit heuristic [m^3/mol/s].
    ~1e14 cm^3/mol/s at 298 K, scaled by sqrt(T/298.15) (hard-sphere theory).
    """
    return 1.0e8 * math.sqrt(T_K / 298.15)
