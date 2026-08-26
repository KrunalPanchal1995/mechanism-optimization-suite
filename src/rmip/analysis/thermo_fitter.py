"""
rmip/analysis/thermo_fitter.py
==============================
Feature #3 — NASA-7 thermo fitting + consistency repair.

Sibling to ``analysis/parameter_estimator.py``'s ``ThermoEstimate`` (which only
*averages* coefficients, and warns that the mean polynomial may violate Cp/H/S
continuity at Tmid). This module actually *produces a continuous, consistent*
NASA-7 polynomial by constrained least-squares, and repairs species whose
existing polynomial has a Tmid discontinuity.

Why this matters
----------------
A NASA-7 record is two 7-coefficient polynomials joined at ``Tmid``:

    Cp/R  = a1 + a2 T + a3 T^2 + a4 T^3 + a5 T^4
    H/RT  = a1 + a2 T/2 + a3 T^2/3 + a4 T^3/4 + a5 T^4/5 + a6/T
    S/R   = a1 lnT + a2 T + a3 T^2/2 + a4 T^3/3 + a5 T^4/4 + a7

If the low and high pieces disagree at ``Tmid`` (a common defect in hand-edited
or machine-merged mechanisms — including the coefficient-averaged output of
``ParameterEstimator.estimate_thermo``) then Cp, H or S is discontinuous. The
solver derives every missing reverse rate from ``Kc(T)`` = f(H, S); a thermo
discontinuity therefore injects an unphysical kink into ``k_rev(T)`` — one of
the documented root causes of the stiff-integrator failures that
``analysis/reverse_rate_scanner.py`` flags. Repairing continuity (while
preserving the enthalpy of formation) is a direct fix.

Method
------
1. **Cp fit** — fit the ten Cp coefficients (a1..a5 low, a1..a5 high) by
   least-squares to Cp/R sample points, subject to *exact* C0 and C1 continuity
   of Cp at Tmid (linear equality constraints, solved via the KKT system). The
   fit is done in a scaled temperature ``tau = T/1000`` for conditioning, then
   converted back to physical coefficients.
2. **a6 (enthalpy constant)** — fixed on the low piece so H(Tref) matches a
   reference enthalpy, then on the high piece by H-continuity at Tmid. Result:
   H is continuous *and* H(298.15) is preserved exactly.
3. **a7 (entropy constant)** — same construction against a reference S(Tref)
   and S-continuity at Tmid.

The output therefore satisfies Cp (C0+C1), H and S continuity at Tmid *by
construction* — the residual jumps reported in :class:`NASA7Fit` are numerical
(~1e-10), not physical.

For **repair**, the reference H(Tref)/S(Tref) and the Cp sample points are taken
from the species' *own* existing polynomial (low piece around Tref, each piece
sampled over its own range). This keeps the enthalpy of formation and the shape
of each Cp branch, and only smooths the junction — a minimal, defensible edit.

No third-party fit dependency (numpy only); no writer changes (the coeffs land
in the existing ``coeffs_low_json`` / ``coeffs_high_json`` species columns).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("rmip.analysis.thermo_fitter")

R_GAS: float = 8.314_462_618          # J mol^-1 K^-1
T_REF: float = 298.15                 # K, standard reference for Hf / S
_T_SCALE: float = 1000.0              # temperature scaling for fit conditioning

# Relative Cp/H/S jump at Tmid above which a species is deemed inconsistent.
DEFAULT_DISCONTINUITY_TOL: float = 0.01     # 1 %


# ─────────────────────────────────────────────────────────────────────────────
# NASA-7 dimensionless property evaluation
# ─────────────────────────────────────────────────────────────────────────────

def nasa7_cp_R(coeffs: Sequence[float], T: float) -> float:
    """Cp/R from a 7-coefficient NASA set."""
    a = coeffs
    return a[0] + a[1]*T + a[2]*T**2 + a[3]*T**3 + a[4]*T**4


def nasa7_H_RT(coeffs: Sequence[float], T: float) -> float:
    """H/(R·T) from a 7-coefficient NASA set (uses a6)."""
    a = coeffs
    return (a[0] + a[1]*T/2.0 + a[2]*T**2/3.0 + a[3]*T**3/4.0
            + a[4]*T**4/5.0 + a[5]/T)


def nasa7_S_R(coeffs: Sequence[float], T: float) -> float:
    """S/R from a 7-coefficient NASA set (uses a7)."""
    a = coeffs
    return (a[0]*math.log(T) + a[1]*T + a[2]*T**2/2.0
            + a[3]*T**3/3.0 + a[4]*T**4/4.0 + a[6])


def nasa7_eval(coeffs_low: Sequence[float], coeffs_high: Sequence[float],
               tmid: float, T: float) -> Tuple[float, float, float]:
    """Return (Cp/R, H/RT, S/R) picking the range appropriate to ``T``."""
    c = coeffs_low if T <= tmid else coeffs_high
    return nasa7_cp_R(c, T), nasa7_H_RT(c, T), nasa7_S_R(c, T)


# ─────────────────────────────────────────────────────────────────────────────
# Result record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NASA7Fit:
    """A fitted, continuity-consistent NASA-7 polynomial plus fit diagnostics."""

    coeffs_low: List[float]        # 7 coefficients, [Tmin, Tmid]
    coeffs_high: List[float]       # 7 coefficients, [Tmid, Tmax]
    tmin_K: float
    tmid_K: float
    tmax_K: float

    # Fit quality against the Cp sample points (dimensionless Cp/R units)
    cp_rms_error: float = 0.0
    cp_max_error: float = 0.0

    # Residual continuity jumps at Tmid AFTER the fit (should be ~1e-10, relative)
    cp_jump_rel: float = 0.0
    h_jump_rel: float = 0.0
    s_jump_rel: float = 0.0

    # Reference thermodynamics reproduced by the fit
    h298_kJ_mol: Optional[float] = None
    s298_J_molK: Optional[float] = None

    source: str = "fit"            # 'fit' | 'repair'
    notes: List[str] = field(default_factory=list)

    def as_coeffs_pair(self) -> Tuple[List[float], List[float]]:
        return self.coeffs_low, self.coeffs_high

    def to_dict(self) -> dict:
        return {
            "coeffs_low": self.coeffs_low,
            "coeffs_high": self.coeffs_high,
            "tmin_K": self.tmin_K,
            "tmid_K": self.tmid_K,
            "tmax_K": self.tmax_K,
            "cp_rms_error": self.cp_rms_error,
            "cp_max_error": self.cp_max_error,
            "cp_jump_rel": self.cp_jump_rel,
            "h_jump_rel": self.h_jump_rel,
            "s_jump_rel": self.s_jump_rel,
            "h298_kJ_mol": self.h298_kJ_mol,
            "s298_J_molK": self.s298_J_molK,
            "source": self.source,
            "notes": self.notes,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Core fitter
# ─────────────────────────────────────────────────────────────────────────────

def _constrained_lsq(A: np.ndarray, b: np.ndarray,
                     C: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Minimise ``||A x − b||^2`` subject to ``C x = d`` via the KKT system::

        [ 2 AᵀA   Cᵀ ] [x]   [ 2 Aᵀb ]
        [ C       0  ] [λ] = [ d     ]

    Returns the ``x`` block only.
    """
    n = A.shape[1]
    m = C.shape[0]
    AtA = A.T @ A
    Atb = A.T @ b
    kkt = np.zeros((n + m, n + m))
    kkt[:n, :n] = 2.0 * AtA
    kkt[:n, n:] = C.T
    kkt[n:, :n] = C
    rhs = np.concatenate([2.0 * Atb, d])
    sol = np.linalg.solve(kkt, rhs)
    return sol[:n]


def fit_nasa7(
    T_data: Sequence[float],
    cp_R_data: Sequence[float],
    h_ref_RT: float,
    s_ref_R: float,
    tmin_K: float,
    tmid_K: float,
    tmax_K: float,
    tref_K: float = T_REF,
    source: str = "fit",
) -> NASA7Fit:
    """
    Fit a continuity-consistent NASA-7 polynomial to Cp/R sample points.

    Parameters
    ----------
    T_data, cp_R_data
        Sample temperatures [K] and the corresponding dimensionless heat
        capacity ``Cp/R``. Points are split at ``tmid_K`` into the low/high fit.
    h_ref_RT
        Reference enthalpy as ``H(tref_K)/(R·tref_K)`` (dimensionless). The fit
        reproduces this exactly on the low piece.
    s_ref_R
        Reference entropy as ``S(tref_K)/R`` (dimensionless).
    tmin_K, tmid_K, tmax_K
        NASA-7 temperature ranges.
    tref_K
        Reference temperature for the enthalpy/entropy constants (default
        298.15 K, which must lie within the low range).
    source
        Provenance tag stored on the result ('fit' | 'repair').

    Notes
    -----
    Cp gets exact C0 and C1 continuity at Tmid; H and S are made continuous
    through the integration-constant construction. To constrain both Cp
    branches independently each side needs at least 5 sample points; if a side
    is underdetermined its unknowns are still solvable thanks to the continuity
    constraints, but the fit is only meaningful when both sides are sampled.
    """
    T = np.asarray(T_data, dtype=float)
    y = np.asarray(cp_R_data, dtype=float)
    if T.shape != y.shape or T.size == 0:
        raise ValueError("T_data and cp_R_data must be non-empty, equal length")

    tau = T / _T_SCALE
    tau_mid = tmid_K / _T_SCALE
    is_low = T <= tmid_K

    n_low = int(np.count_nonzero(is_low))
    n_high = int(np.count_nonzero(~is_low))
    if n_low == 0 or n_high == 0:
        raise ValueError(
            f"Need Cp points on both sides of Tmid={tmid_K} K "
            f"(got {n_low} low, {n_high} high)"
        )

    # ── Design matrix for the ten scaled Cp coefficients ──────────────────
    # Unknown vector x = [cl1..cl5, ch1..ch5] in scaled tau powers.
    def cp_basis(t: np.ndarray) -> np.ndarray:
        return np.vstack([np.ones_like(t), t, t**2, t**3, t**4]).T   # (N,5)

    N = T.size
    A = np.zeros((N, 10))
    A[is_low, 0:5] = cp_basis(tau[is_low])
    A[~is_low, 5:10] = cp_basis(tau[~is_low])

    # ── Continuity constraints at tau_mid (C0 value + C1 slope of Cp) ──────
    val = np.array([1.0, tau_mid, tau_mid**2, tau_mid**3, tau_mid**4])
    slope = np.array([0.0, 1.0, 2.0*tau_mid, 3.0*tau_mid**2, 4.0*tau_mid**3])
    C = np.zeros((2, 10))
    C[0, 0:5] = val
    C[0, 5:10] = -val
    C[1, 0:5] = slope
    C[1, 5:10] = -slope
    d = np.zeros(2)

    x = _constrained_lsq(A, y, C, d)
    c_low_scaled = x[0:5]
    c_high_scaled = x[5:10]

    # Un-scale: a_k = c_k / T_SCALE^(k-1)
    scale_pows = _T_SCALE ** np.arange(5)
    a_low = (c_low_scaled / scale_pows).tolist()
    a_high = (c_high_scaled / scale_pows).tolist()

    # ── Enthalpy constant a6 ──────────────────────────────────────────────
    # H/RT = gH(T) + a6/T,  gH = a1 + a2 T/2 + a3 T^2/3 + a4 T^3/4 + a5 T^4/5
    def gH(a: List[float], t: float) -> float:
        return (a[0] + a[1]*t/2.0 + a[2]*t**2/3.0
                + a[3]*t**3/4.0 + a[4]*t**4/5.0)

    a6_low = tref_K * (h_ref_RT - gH(a_low, tref_K))
    # H continuity at Tmid: gH_low(Tmid) + a6_low/Tmid = gH_high(Tmid) + a6_high/Tmid
    a6_high = a6_low + tmid_K * (gH(a_low, tmid_K) - gH(a_high, tmid_K))

    # ── Entropy constant a7 ───────────────────────────────────────────────
    # S/R = gS(T) + a7,  gS = a1 lnT + a2 T + a3 T^2/2 + a4 T^3/3 + a5 T^4/4
    def gS(a: List[float], t: float) -> float:
        return (a[0]*math.log(t) + a[1]*t + a[2]*t**2/2.0
                + a[3]*t**3/3.0 + a[4]*t**4/4.0)

    a7_low = s_ref_R - gS(a_low, tref_K)
    a7_high = a7_low + gS(a_low, tmid_K) - gS(a_high, tmid_K)

    coeffs_low = a_low + [a6_low, a7_low]
    coeffs_high = a_high + [a6_high, a7_high]

    # ── Diagnostics ───────────────────────────────────────────────────────
    cp_pred = np.array([
        nasa7_cp_R(coeffs_low if lo else coeffs_high, float(t))
        for t, lo in zip(T, is_low)
    ])
    resid = cp_pred - y
    cp_rms = float(np.sqrt(np.mean(resid**2)))
    cp_max = float(np.max(np.abs(resid)))

    cp_jump = _rel_jump(nasa7_cp_R(coeffs_low, tmid_K),
                        nasa7_cp_R(coeffs_high, tmid_K))
    h_jump = _rel_jump(nasa7_H_RT(coeffs_low, tmid_K) * tmid_K,
                       nasa7_H_RT(coeffs_high, tmid_K) * tmid_K)
    s_jump = _rel_jump(nasa7_S_R(coeffs_low, tmid_K),
                       nasa7_S_R(coeffs_high, tmid_K))

    h298 = nasa7_H_RT(coeffs_low, tref_K) * R_GAS * tref_K / 1000.0   # kJ/mol
    s298 = nasa7_S_R(coeffs_low, tref_K) * R_GAS                       # J/mol/K

    return NASA7Fit(
        coeffs_low=coeffs_low,
        coeffs_high=coeffs_high,
        tmin_K=float(tmin_K),
        tmid_K=float(tmid_K),
        tmax_K=float(tmax_K),
        cp_rms_error=cp_rms,
        cp_max_error=cp_max,
        cp_jump_rel=cp_jump,
        h_jump_rel=h_jump,
        s_jump_rel=s_jump,
        h298_kJ_mol=h298,
        s298_J_molK=s298,
        source=source,
    )


def refit_from_coeffs(
    coeffs_low: Sequence[float],
    coeffs_high: Sequence[float],
    tmin_K: float,
    tmid_K: float,
    tmax_K: float,
    n_points: int = 50,
    tref_K: float = T_REF,
    source: str = "repair",
) -> NASA7Fit:
    """
    Repair an existing (possibly discontinuous) NASA-7 by resampling its own
    Cp curves and refitting a continuous polynomial.

    Each branch is sampled over its own range ([Tmin, Tmid] from the low piece,
    [Tmid, Tmax] from the high piece), so the shape of both Cp curves is kept
    and only the junction is reconciled. The reference enthalpy/entropy come
    from the **low** piece at ``tref_K`` — preserving H(298) and S(298), hence
    the enthalpy of formation.
    """
    if len(coeffs_low) < 7 or len(coeffs_high) < 7:
        raise ValueError("coeffs_low and coeffs_high must each have 7 entries")

    n_side = max(6, n_points // 2)
    T_low = np.linspace(tmin_K, tmid_K, n_side)
    T_high = np.linspace(tmid_K, tmax_K, n_side)
    T_all = np.concatenate([T_low, T_high[1:]])   # avoid duplicate Tmid point

    cp_R = []
    for t in T_all:
        c = coeffs_low if t <= tmid_K else coeffs_high
        cp_R.append(nasa7_cp_R(c, float(t)))

    # Reference H/S from the low piece at Tref (preserves Hf298, S298).
    h_ref_RT = nasa7_H_RT(coeffs_low, tref_K)
    s_ref_R = nasa7_S_R(coeffs_low, tref_K)

    fit = fit_nasa7(
        T_all.tolist(), cp_R, h_ref_RT, s_ref_R,
        tmin_K, tmid_K, tmax_K, tref_K=tref_K, source=source,
    )

    # Report how bad the ORIGINAL junction was (pre-repair), for context.
    orig_cp = _rel_jump(nasa7_cp_R(coeffs_low, tmid_K),
                        nasa7_cp_R(coeffs_high, tmid_K))
    orig_h = _rel_jump(nasa7_H_RT(coeffs_low, tmid_K) * tmid_K,
                       nasa7_H_RT(coeffs_high, tmid_K) * tmid_K)
    orig_s = _rel_jump(nasa7_S_R(coeffs_low, tmid_K),
                       nasa7_S_R(coeffs_high, tmid_K))
    fit.notes.append(
        f"original Tmid jumps: Cp={orig_cp:.2e} H={orig_h:.2e} S={orig_s:.2e}"
    )
    return fit


# ─────────────────────────────────────────────────────────────────────────────
# Consistency scan (dictless helper, DB-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

def continuity_jumps(
    coeffs_low: Sequence[float],
    coeffs_high: Sequence[float],
    tmid_K: float,
) -> Dict[str, float]:
    """Relative Cp/H/S jumps at Tmid for an existing polynomial pair."""
    return {
        "cp": _rel_jump(nasa7_cp_R(coeffs_low, tmid_K),
                        nasa7_cp_R(coeffs_high, tmid_K)),
        "h": _rel_jump(nasa7_H_RT(coeffs_low, tmid_K) * tmid_K,
                       nasa7_H_RT(coeffs_high, tmid_K) * tmid_K),
        "s": _rel_jump(nasa7_S_R(coeffs_low, tmid_K),
                       nasa7_S_R(coeffs_high, tmid_K)),
    }


def is_consistent(
    coeffs_low: Sequence[float],
    coeffs_high: Sequence[float],
    tmid_K: float,
    tol: float = DEFAULT_DISCONTINUITY_TOL,
) -> bool:
    """True if all of Cp/H/S are continuous at Tmid within ``tol`` (relative)."""
    j = continuity_jumps(coeffs_low, coeffs_high, tmid_K)
    return max(j["cp"], j["h"], j["s"]) <= tol


def _rel_jump(low_val: float, high_val: float) -> float:
    """Relative discontinuity |low − high| / mean(|low|, |high|)."""
    denom = (abs(low_val) + abs(high_val)) / 2.0
    if denom <= 0:
        return 0.0
    return abs(low_val - high_val) / denom


# ─────────────────────────────────────────────────────────────────────────────
# DB-facing repairer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SpeciesThermoRepair:
    """One species' before/after thermo-consistency record."""
    species_id: int
    name: str
    tmid_K: float
    jumps_before: Dict[str, float]
    jumps_after: Optional[Dict[str, float]] = None   # None if not repaired
    repaired: bool = False
    persisted: bool = False
    cp_rms_error: Optional[float] = None
    h298_kJ_mol_before: Optional[float] = None
    h298_kJ_mol_after: Optional[float] = None
    note: str = ""

    @property
    def worst_jump_before(self) -> float:
        return max(self.jumps_before.values()) if self.jumps_before else 0.0

    def to_dict(self) -> dict:
        return {
            "species_id": self.species_id,
            "name": self.name,
            "tmid_K": self.tmid_K,
            "jumps_before": self.jumps_before,
            "jumps_after": self.jumps_after,
            "repaired": self.repaired,
            "persisted": self.persisted,
            "cp_rms_error": self.cp_rms_error,
            "h298_kJ_mol_before": self.h298_kJ_mol_before,
            "h298_kJ_mol_after": self.h298_kJ_mol_after,
            "note": self.note,
        }


class ThermoRepairer:
    """
    Scan a mechanism's species thermo for Tmid discontinuities and, optionally,
    repair them in place via :func:`refit_from_coeffs`.

    Repairs preserve H(298)/S(298) (hence the enthalpy of formation) and both Cp
    branch shapes; they only reconcile the junction. The reverse-rate scanner
    derives every solver reverse rate from the same thermo, so repairing Tmid
    continuity is expected to reduce (never manufacture) reverse-rate flags —
    :meth:`count_reverse_flags` lets a caller measure that before/after.
    """

    def __init__(self, db_path: str, tol: float = DEFAULT_DISCONTINUITY_TOL) -> None:
        from rmip.database.schema import init_db
        self._db = db_path
        self._conn = init_db(db_path)
        self._tol = tol

    # ------------------------------------------------------------------
    def scan(self, mechanism_id: int) -> List[SpeciesThermoRepair]:
        """Report the Tmid jump of every species that has full NASA-7 thermo."""
        rows = self._conn.execute(
            """SELECT id, name, tmid_K, coeffs_low_json, coeffs_high_json
               FROM species WHERE mechanism_id=?""",
            (mechanism_id,),
        ).fetchall()
        out: List[SpeciesThermoRepair] = []
        for r in rows:
            lo = _parse_coeffs(r["coeffs_low_json"])
            hi = _parse_coeffs(r["coeffs_high_json"])
            tmid = r["tmid_K"] or 1000.0
            if lo is None or hi is None:
                continue
            jumps = continuity_jumps(lo, hi, tmid)
            out.append(SpeciesThermoRepair(
                species_id=r["id"], name=r["name"], tmid_K=tmid,
                jumps_before=jumps,
                h298_kJ_mol_before=nasa7_H_RT(lo, T_REF) * R_GAS * T_REF / 1000.0,
            ))
        return out

    def inconsistent(self, mechanism_id: int) -> List[SpeciesThermoRepair]:
        """Subset of :meth:`scan` whose worst Cp/H/S jump exceeds the tolerance."""
        return [s for s in self.scan(mechanism_id)
                if s.worst_jump_before > self._tol]

    # ------------------------------------------------------------------
    def repair(
        self,
        mechanism_id: int,
        dry_run: bool = False,
        only_inconsistent: bool = True,
    ) -> List[SpeciesThermoRepair]:
        """
        Refit discontinuous species and (unless ``dry_run``) persist the new
        coefficients to the ``species`` table. Returns one record per repaired
        species with before/after jumps.
        """
        rows = self._conn.execute(
            """SELECT id, name, tmin_K, tmid_K, tmax_K,
                      coeffs_low_json, coeffs_high_json
               FROM species WHERE mechanism_id=?""",
            (mechanism_id,),
        ).fetchall()

        results: List[SpeciesThermoRepair] = []
        for r in rows:
            lo = _parse_coeffs(r["coeffs_low_json"])
            hi = _parse_coeffs(r["coeffs_high_json"])
            tmid = r["tmid_K"] or 1000.0
            if lo is None or hi is None:
                continue
            jumps_before = continuity_jumps(lo, hi, tmid)
            worst = max(jumps_before.values())
            if only_inconsistent and worst <= self._tol:
                continue

            tmin = r["tmin_K"] or 200.0
            tmax = r["tmax_K"] or 5000.0
            rec = SpeciesThermoRepair(
                species_id=r["id"], name=r["name"], tmid_K=tmid,
                jumps_before=jumps_before,
                h298_kJ_mol_before=nasa7_H_RT(lo, T_REF) * R_GAS * T_REF / 1000.0,
            )
            try:
                fit = refit_from_coeffs(lo, hi, tmin, tmid, tmax)
            except Exception as exc:  # degenerate polynomial, bad ranges, …
                rec.note = f"refit failed: {exc}"
                results.append(rec)
                continue

            rec.repaired = True
            rec.jumps_after = {
                "cp": fit.cp_jump_rel, "h": fit.h_jump_rel, "s": fit.s_jump_rel,
            }
            rec.cp_rms_error = fit.cp_rms_error
            rec.h298_kJ_mol_after = fit.h298_kJ_mol

            if not dry_run:
                self._conn.execute(
                    """UPDATE species
                       SET coeffs_low_json=?, coeffs_high_json=?,
                           h298_kJ_mol=?, s298_J_molK=?,
                           thermo_comment=COALESCE(thermo_comment,'') || ?
                       WHERE id=?""",
                    (json.dumps(fit.coeffs_low), json.dumps(fit.coeffs_high),
                     fit.h298_kJ_mol, fit.s298_J_molK,
                     " [rmip:thermo-repair Tmid-continuity]", r["id"]),
                )
                rec.persisted = True
            results.append(rec)

        if not dry_run and any(rec.persisted for rec in results):
            self._conn.commit()

        log.info(
            "ThermoRepairer: mechanism_id=%d — %d species repaired%s (tol=%.3g)",
            mechanism_id, sum(1 for r in results if r.repaired),
            " (dry-run, not persisted)" if dry_run else "", self._tol,
        )
        return results

    # ------------------------------------------------------------------
    def count_reverse_flags(self, mechanism_id: int) -> Tuple[int, int]:
        """
        Run the reverse-rate scanner and return ``(n_flagged, n_evaluated)``.

        Used to demonstrate that a thermo repair lowers the number of
        collision-limit-exceeding derived reverse rates.
        """
        try:
            from rmip.analysis.reverse_rate_scanner import ReverseRateScanner
        except Exception as exc:
            log.warning("Reverse-rate re-verification unavailable: %s", exc)
            return (0, 0)
        scanner = ReverseRateScanner(self._db)
        flags = scanner.scan(mechanism_id)
        return (sum(1 for f in flags if f.flagged), len(flags))

    def close(self) -> None:
        self._conn.close()


def _parse_coeffs(raw: Optional[str]) -> Optional[List[float]]:
    if not raw:
        return None
    try:
        val = json.loads(raw) if isinstance(raw, str) else raw
        coeffs = [float(x) for x in val]
        return coeffs if len(coeffs) == 7 else None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
