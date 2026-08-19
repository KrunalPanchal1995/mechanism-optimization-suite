"""
rmip/analysis/parameter_estimator.py
======================================
Estimates consensus Arrhenius / NASA-7 parameters when the same reaction or species
appears in multiple mechanisms with different values.

Conventions
-----------
A  : geometric mean (log-space) — correct for quantities spanning orders of magnitude
n  : arithmetic mean
Ea : arithmetic mean  (J/mol, SI)
Uncertainty bounds : min/max across all source mechanisms (used by MSOInterface as
                     lower/upper bounds for optimisation — NOT a formal DoE).
NASA-7 coefficients: arithmetic mean per coefficient; note that the mean polynomial
                     may not satisfy the exact Cp/H/S continuity at Tmid.  Flag this.

Design decisions inherited
--------------------------
D3: SI units throughout; originals also stored.
D6: Transport data — NOT handled here (out of scope).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

log = logging.getLogger("rmip.analysis.parameter_estimator")

R_GAS: float = 8.314_462_618   # J mol⁻¹ K⁻¹


# ─────────────────────────────────────────────────────────────────────────────
# Result data-classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ArrheniusEstimate:
    """Consensus Arrhenius parameters from N mechanism sources."""

    n_sources: int

    # ── Central estimates ──────────────────────────────────────────────────
    A_mean_si: float        # geometric mean of A values (SI)
    n_mean: float           # arithmetic mean of temperature exponent
    Ea_mean_si: float       # arithmetic mean of activation energy (J/mol)

    # ── Uncertainty bounds (feeds MSO lower/upper bounds) ─────────────────
    A_min_si: float
    A_max_si: float
    n_min: float
    n_max: float
    Ea_min_si: float
    Ea_max_si: float

    # ── Derived diagnostics ────────────────────────────────────────────────
    uncertainty_factor: float   # A_max / A_min  (> 100 → warn)
    Ea_range_kJ_mol: float      # (Ea_max − Ea_min) / 1000
    log_A_std: float            # std-dev of log(A) across sources
    n_std: float
    Ea_std_si: float

    # Rate-ratio at landmark temperatures
    k_ratio_at_1000K: Optional[float] = None
    k_ratio_at_1500K: Optional[float] = None

    def summary(self) -> str:
        return (
            f"ArrheniusEstimate(n={self.n_sources})  "
            f"A_mean={self.A_mean_si:.3E}  n={self.n_mean:.3f}  "
            f"Ea={self.Ea_mean_si/1000:.2f} kJ/mol  "
            f"UF={self.uncertainty_factor:.1f}×  "
            f"k-ratio@1000K={self.k_ratio_at_1000K}"
        )


@dataclass
class ThermoEstimate:
    """Consensus NASA-7 thermo data from N mechanism sources."""

    n_sources: int
    coeffs_low_mean: list[float]    # 7 coefficients, low-T range
    coeffs_high_mean: list[float]   # 7 coefficients, high-T range
    tmin_K: float
    tmid_K: float
    tmax_K: float

    # Per-coefficient standard deviations
    coeffs_low_std: list[float]  = field(default_factory=list)
    coeffs_high_std: list[float] = field(default_factory=list)

    # Continuity check: |Cp_low(Tmid) - Cp_high(Tmid)| / Cp_high(Tmid)
    continuity_error_pct: Optional[float] = None
    continuity_warning: bool = False

    def h298_kJ_mol(self) -> Optional[float]:
        """Estimate H°(298 K) from mean high-T polynomial (kJ/mol)."""
        try:
            c = self.coeffs_low_mean
            h = R_GAS * 298.0 * (
                c[0] + c[1]*298/2 + c[2]*298**2/3
                + c[3]*298**3/4 + c[4]*298**4/5 + c[5]/298
            )
            return h / 1000.0
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Main estimator class
# ─────────────────────────────────────────────────────────────────────────────

class ParameterEstimator:
    """
    Estimates consensus kinetic and thermodynamic parameters from multiple sources.

    All arithmetic on A is performed in log-space (geometric mean).
    All arithmetic on Ea and n is arithmetic (linear).
    NASA-7 coefficients are averaged coefficient-by-coefficient (arithmetic).

    Usage
    -----
    est = ParameterEstimator()

    # From a list of dicts (as stored in the DB)
    result = est.estimate_arrhenius_from_dicts([
        {"A_si": 1.0e13, "n": 0.0,  "Ea_si": 100_000},
        {"A_si": 3.2e13, "n": -0.3, "Ea_si":  95_000},
        {"A_si": 2.1e13, "n":  0.1, "Ea_si": 102_000},
    ])
    print(result.summary())

    bounds = est.arrhenius_to_mso_bounds(result)
    """

    # ── Arrhenius estimation ──────────────────────────────────────────────

    def estimate_arrhenius_from_dicts(
        self,
        kinetics_list: list[dict],
    ) -> ArrheniusEstimate:
        """
        Estimate Arrhenius parameters from a list of kinetics dicts.

        Each dict must contain at least: ``A_si``, ``n``, ``Ea_si``.
        Entries missing any required field are silently skipped.
        """
        A_vals, n_vals, Ea_vals = [], [], []
        for k in kinetics_list:
            if k.get("A_si") is not None and k.get("Ea_si") is not None:
                A_vals.append(float(k["A_si"]))
                n_vals.append(float(k.get("n", 0.0)))
                Ea_vals.append(float(k["Ea_si"]))

        if not A_vals:
            raise ValueError("No complete (A, n, Ea) entries in kinetics_list")

        return self.estimate_arrhenius(A_vals, n_vals, Ea_vals)

    def estimate_arrhenius(
        self,
        A_vals:  Sequence[float],
        n_vals:  Sequence[float],
        Ea_vals: Sequence[float],
    ) -> ArrheniusEstimate:
        """
        Core Arrhenius estimation.

        Parameters
        ----------
        A_vals  : Pre-exponential factors in SI units.
        n_vals  : Temperature exponents.
        Ea_vals : Activation energies in J/mol.
        """
        A_arr  = np.asarray(A_vals,  dtype=float)
        n_arr  = np.asarray(n_vals,  dtype=float)
        Ea_arr = np.asarray(Ea_vals, dtype=float)
        N = len(A_arr)

        if N == 0:
            raise ValueError("Empty parameter lists")

        # A: geometric mean (log-space)
        log_A    = np.log(A_arr)
        A_mean   = float(np.exp(np.mean(log_A)))
        A_min    = float(np.min(A_arr))
        A_max    = float(np.max(A_arr))
        log_A_std = float(np.std(log_A, ddof=0))

        uf = A_max / A_min if A_min > 0 else math.inf
        if uf > 100:
            log.warning(
                "Large uncertainty factor %.0f× for A — sources may represent "
                "different pressure conditions or families.", uf
            )

        # n: arithmetic
        n_mean = float(np.mean(n_arr))
        n_min  = float(np.min(n_arr))
        n_max  = float(np.max(n_arr))
        n_std  = float(np.std(n_arr, ddof=0))

        # Ea: arithmetic
        Ea_mean = float(np.mean(Ea_arr))
        Ea_min  = float(np.min(Ea_arr))
        Ea_max  = float(np.max(Ea_arr))
        Ea_std  = float(np.std(Ea_arr, ddof=0))
        Ea_range_kJ = (Ea_max - Ea_min) / 1000.0

        k_ratio_1000 = self._k_ratio(A_vals, n_vals, Ea_vals, T=1000.0)
        k_ratio_1500 = self._k_ratio(A_vals, n_vals, Ea_vals, T=1500.0)

        return ArrheniusEstimate(
            n_sources=N,
            A_mean_si=A_mean,
            n_mean=n_mean,
            Ea_mean_si=Ea_mean,
            A_min_si=A_min,
            A_max_si=A_max,
            n_min=n_min,
            n_max=n_max,
            Ea_min_si=Ea_min,
            Ea_max_si=Ea_max,
            uncertainty_factor=uf,
            Ea_range_kJ_mol=Ea_range_kJ,
            log_A_std=log_A_std,
            n_std=n_std,
            Ea_std_si=Ea_std,
            k_ratio_at_1000K=k_ratio_1000,
            k_ratio_at_1500K=k_ratio_1500,
        )

    def arrhenius_to_mso_bounds(self, est: ArrheniusEstimate) -> dict:
        """
        Convert an ArrheniusEstimate to the bounds dict format expected by MSOInterface.

        Returns both list-valued keys (``params_mean``, ``params_lb``, ``params_ub``)
        matching the OptimizationTool_v4.py JSON contract, and individual scalar keys
        for downstream diagnostics.
        """
        return {
            # ── List-valued keys (MSO JSON contract) ──────────────────────
            "params_mean":      [est.A_mean_si,  est.n_mean,  est.Ea_mean_si],
            "params_lb":        [est.A_min_si,   est.n_min,   est.Ea_min_si],
            "params_ub":        [est.A_max_si,   est.n_max,   est.Ea_max_si],
            # ── Individual scalar keys (diagnostics / display) ─────────────
            "A_mean_si":        est.A_mean_si,
            "A_lower_si":       est.A_min_si,
            "A_upper_si":       est.A_max_si,
            "n_mean":           est.n_mean,
            "n_lower":          est.n_min,
            "n_upper":          est.n_max,
            "Ea_mean_si":       est.Ea_mean_si,
            "Ea_lower_si":      est.Ea_min_si,
            "Ea_upper_si":      est.Ea_max_si,
            "uncertainty_factor": est.uncertainty_factor,
            "k_ratio_1000K":    est.k_ratio_at_1000K,
            "k_ratio_1500K":    est.k_ratio_at_1500K,
        }

    # ── NASA-7 thermo estimation ──────────────────────────────────────────

    def estimate_thermo(
        self,
        thermo_list: list[dict],
    ) -> ThermoEstimate:
        """
        Estimate mean NASA-7 thermo from multiple source dicts.

        Each dict must contain:
          ``coeffs_low``  : list[float] of 7 low-T coefficients
          ``coeffs_high`` : list[float] of 7 high-T coefficients
          ``tmin_K``, ``tmid_K``, ``tmax_K`` : float

        Transport data is NOT included (Design Decision D6).
        """
        low_arrays, high_arrays = [], []
        tmin_vals, tmid_vals, tmax_vals = [], [], []

        for td in thermo_list:
            cl = td.get("coeffs_low")
            ch = td.get("coeffs_high")
            if cl and ch and len(cl) == 7 and len(ch) == 7:
                low_arrays.append(cl)
                high_arrays.append(ch)
                tmin_vals.append(td.get("tmin_K", 300.0) or 300.0)
                tmid_vals.append(td.get("tmid_K", 1000.0) or 1000.0)
                tmax_vals.append(td.get("tmax_K", 5000.0) or 5000.0)

        if not low_arrays:
            raise ValueError("No valid NASA-7 thermo data provided")

        N = len(low_arrays)
        low_arr  = np.array(low_arrays, dtype=float)
        high_arr = np.array(high_arrays, dtype=float)

        mean_low  = np.mean(low_arr,  axis=0).tolist()
        mean_high = np.mean(high_arr, axis=0).tolist()
        std_low   = np.std(low_arr,  axis=0, ddof=0).tolist()
        std_high  = np.std(high_arr, axis=0, ddof=0).tolist()

        tmin = float(np.min(tmin_vals))
        tmid = float(np.mean(tmid_vals))   # if mechs differ in Tmid, warn
        tmax = float(np.max(tmax_vals))

        if len(set(tmid_vals)) > 1:
            log.warning(
                "Thermo sources have different Tmid values %s — "
                "using arithmetic mean Tmid=%.1f K.  "
                "The averaged polynomial may not satisfy Cp continuity at Tmid.",
                tmid_vals, tmid
            )

        # Check continuity of mean polynomial at Tmid
        cont_err = self._thermo_continuity_error(mean_low, mean_high, tmid)
        warn_cont = cont_err is not None and cont_err > 0.5   # > 0.5 % relative

        est = ThermoEstimate(
            n_sources=N,
            coeffs_low_mean=mean_low,
            coeffs_high_mean=mean_high,
            tmin_K=tmin,
            tmid_K=tmid,
            tmax_K=tmax,
            coeffs_low_std=std_low,
            coeffs_high_std=std_high,
            continuity_error_pct=cont_err,
            continuity_warning=warn_cont,
        )

        if warn_cont:
            log.warning(
                "Mean NASA-7 polynomial has %.2f%% Cp discontinuity at Tmid=%.1f K. "
                "Consider refitting or using most_complete strategy instead.",
                cont_err, tmid
            )

        return est

    # ── Rate constant utilities ───────────────────────────────────────────

    @staticmethod
    def rate_constant(A: float, n: float, Ea_si: float, T: float) -> float:
        """k(T) = A · T^n · exp(−Ea / RT)   [SI units]"""
        return A * (T ** n) * math.exp(-Ea_si / (R_GAS * T))

    def mre_vs_mean(
        self,
        A_vals:  Sequence[float],
        n_vals:  Sequence[float],
        Ea_vals: Sequence[float],
        T_grid: Optional[Sequence[float]] = None,
    ) -> float:
        """
        Mean Relative Error of all source rates vs the geometric-mean rate.

        MRE = (1/N) Σ_i |k_i(T) − k̄(T)| / k̄(T),  averaged over T_grid.
        """
        if T_grid is None:
            T_grid = [500, 700, 1000, 1200, 1500, 2000]
        est = self.estimate_arrhenius(A_vals, n_vals, Ea_vals)
        mres = []
        for T in T_grid:
            k_mean = self.rate_constant(est.A_mean_si, est.n_mean, est.Ea_mean_si, T)
            if k_mean <= 0:
                continue
            for A, n, Ea in zip(A_vals, n_vals, Ea_vals):
                k_i = self.rate_constant(A, n, Ea, T)
                mres.append(abs(k_i - k_mean) / k_mean)
        return float(np.mean(mres)) if mres else 0.0

    # ── Private helpers ───────────────────────────────────────────────────

    def _k_ratio(
        self,
        A_vals:  Sequence[float],
        n_vals:  Sequence[float],
        Ea_vals: Sequence[float],
        T: float,
    ) -> Optional[float]:
        """Max/min rate ratio at temperature T across all sources."""
        if len(A_vals) < 2:
            return None
        k_vals = [self.rate_constant(A, n, Ea, T)
                  for A, n, Ea in zip(A_vals, n_vals, Ea_vals)]
        k_min, k_max = min(k_vals), max(k_vals)
        return k_max / k_min if k_min > 0 else None

    @staticmethod
    def _thermo_continuity_error(
        cl: list[float], ch: list[float], Tmid: float
    ) -> Optional[float]:
        """
        Relative Cp discontinuity at Tmid between low-T and high-T polynomials.
        Cp/R = a1 + a2·T + a3·T² + a4·T³ + a5·T⁴
        Returns percentage error, or None if computation fails.
        """
        try:
            T = Tmid
            Cp_low  = cl[0] + cl[1]*T + cl[2]*T**2 + cl[3]*T**3 + cl[4]*T**4
            Cp_high = ch[0] + ch[1]*T + ch[2]*T**2 + ch[3]*T**3 + ch[4]*T**4
            return abs(Cp_low - Cp_high) / abs(Cp_high) * 100.0 if Cp_high != 0 else None
        except Exception:
            return None
