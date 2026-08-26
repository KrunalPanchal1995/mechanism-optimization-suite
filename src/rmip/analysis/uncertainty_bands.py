"""
rmip/analysis/uncertainty_bands.py
=====================================
Phase-4 physics core: temperature-dependent uncertainty bands for OPTIMIZE
reactions in the unified mechanism.

Algorithm
---------
For each contributing mechanism i, evaluate k_i(T) across a temperature grid:

  Arrhenius:  k_i = A_si_i · T^n_i · exp(-Ea_si_i / (R · T))
              (A, Ea in SI units; units cancel in all ratios since every
               contribution is the same reaction → same molecularity)

  PLOG:       interpolate ln k linearly in ln P space to P_ref at each T;
              uses the "pressures"/"arrhenius" blob from the DB.

  Troe / Lindemann / Chebyshev:
              if use_cantera_for_falloff=True and Cantera is importable,
              evaluate via a one-reaction Cantera Solution; otherwise SKIP
              that contribution with a logged warning (v1 limitation).

Then per temperature T_j:
  k_nom  = exp(mean_i(ln k_i))        geometric mean (log-normal A)
  k_min  = min_i k_i
  k_max  = max_i k_i
  f      = sqrt(k_max / k_min)        uncertainty factor
  sigma_lnk = ln(f) / 3              3-sigma convention (confirmed by KP)
  f_to_nom  = k_max / k_nom           diagnostic

Output: UncertaintyBand dataclass (defined in types.py).
CSV output (written by UnifiedTreeBuilder): one row per reaction × T point.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from rmip.types import ReactionKinetics, TemperatureGrid, UncertaintyBand

log = logging.getLogger("rmip.analysis.uncertainty_bands")

R_GAS: float = 8.314_462_618   # J mol⁻¹ K⁻¹

# Falloff types we skip unless Cantera is available
_FALLOFF_TYPES = frozenset({"troe", "lindemann", "falloff", "chebyshev", "sri"})


class UncertaintyBandEstimator:
    """
    Compute temperature-dependent uncertainty bands for a reaction that appears
    in multiple mechanisms (OPTIMIZE decisions in UnifiedTreeBuilder).

    Parameters
    ----------
    P_ref_atm : float
        Reference pressure for PLOG evaluation (default 1 atm).
    use_cantera_for_falloff : bool
        If True and cantera is importable, evaluate Troe/Chebyshev via a
        one-reaction Cantera Solution.  If False (default), those contributions
        are skipped with a warning.
    """

    def __init__(
        self,
        P_ref_atm: float = 1.0,
        use_cantera_for_falloff: bool = False,
    ) -> None:
        self._P_ref_atm = P_ref_atm
        self._P_ref_Pa  = P_ref_atm * 101_325.0
        self._use_cantera = use_cantera_for_falloff

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def band_for_reaction(
        self,
        contributions: List[ReactionKinetics],
        grid: TemperatureGrid,
    ) -> UncertaintyBand:
        """
        Compute the uncertainty band for one reaction across all contributions.

        Parameters
        ----------
        contributions : List[ReactionKinetics]
            One entry per contributing mechanism.
        grid : TemperatureGrid
            Temperature points at which to evaluate k(T).

        Returns
        -------
        UncertaintyBand
        """
        if not contributions:
            raise ValueError("contributions must be non-empty")

        T_pts = grid.points()
        n_T = len(T_pts)

        # Evaluate k_i(T) for each contribution
        k_matrix: List[List[float]] = []   # k_matrix[i][j] = k_i at T_j
        mech_ids_used: List[int] = []
        n_skipped = 0

        for contrib in contributions:
            k_vec = self._eval_contribution(contrib, T_pts)
            if k_vec is None:
                n_skipped += 1
                continue
            k_matrix.append(k_vec)
            mech_ids_used.append(contrib.mech_id)

        if not k_matrix:
            log.warning(
                "All contributions skipped for a reaction — returning trivial band."
            )
            trivial = [1.0] * n_T
            return UncertaintyBand(
                canonical_key="",
                formula_string="",
                T_K=T_pts,
                k_nom=trivial,
                k_min=trivial,
                k_max=trivial,
                f_uncertainty=[1.0] * n_T,
                sigma_lnk=[0.0] * n_T,
                f_to_nom=[1.0] * n_T,
                contributor_mech_ids=[c.mech_id for c in contributions],
                method="minmax",
                n_skipped=n_skipped,
            )

        # Build per-T statistics
        k_nom_list: List[float] = []
        k_min_list: List[float] = []
        k_max_list: List[float] = []
        f_list:     List[float] = []
        sigma_list: List[float] = []
        f_to_nom_list: List[float] = []

        for j in range(n_T):
            k_at_T = [k_matrix[i][j] for i in range(len(k_matrix))
                      if k_matrix[i][j] > 0]
            if not k_at_T:
                # All contributions zero/invalid at this T — propagate NaN-like 1.0
                k_nom_list.append(float("nan"))
                k_min_list.append(float("nan"))
                k_max_list.append(float("nan"))
                f_list.append(1.0)
                sigma_list.append(0.0)
                f_to_nom_list.append(1.0)
                continue

            log_k = [math.log(k) for k in k_at_T]
            k_nom = math.exp(sum(log_k) / len(log_k))   # geometric mean
            k_min = min(k_at_T)
            k_max = max(k_at_T)

            f = math.sqrt(k_max / k_min) if k_min > 0 else 1.0
            # 3-sigma convention: [k_min, k_max] spans the 3σ interval
            sigma_lnk = math.log(f) / 3.0 if f > 1.0 else 0.0
            f_to_nom  = k_max / k_nom if k_nom > 0 else 1.0

            k_nom_list.append(k_nom)
            k_min_list.append(k_min)
            k_max_list.append(k_max)
            f_list.append(f)
            sigma_list.append(sigma_lnk)
            f_to_nom_list.append(f_to_nom)

        return UncertaintyBand(
            canonical_key="",         # filled in by caller
            formula_string="",        # filled in by caller
            T_K=T_pts,
            k_nom=k_nom_list,
            k_min=k_min_list,
            k_max=k_max_list,
            f_uncertainty=f_list,
            sigma_lnk=sigma_list,
            f_to_nom=f_to_nom_list,
            contributor_mech_ids=mech_ids_used,
            method="minmax",
            n_skipped=n_skipped,
        )

    # ------------------------------------------------------------------
    # Rate-constant evaluation per kinetics type
    # ------------------------------------------------------------------

    def _eval_contribution(
        self,
        contrib: ReactionKinetics,
        T_pts: List[float],
    ) -> Optional[List[float]]:
        """
        Evaluate k(T) at all T points for one contribution.
        Returns None if the contribution should be skipped.
        """
        ktype = (contrib.kinetics_type or "arrhenius").lower()

        if ktype in ("arrhenius", "multi_arrhenius", "unknown", "third_body"):
            return self._eval_arrhenius(contrib.A_si, contrib.n, contrib.Ea_si, T_pts)

        if ktype == "plog":
            return self._eval_plog(contrib.kinetics_data_json, T_pts)

        if ktype in _FALLOFF_TYPES:
            if self._use_cantera:
                result = self._eval_via_cantera(contrib, T_pts)
                if result is not None:
                    return result
            log.warning(
                "Skipping falloff contribution (type=%s, mech_id=%d): "
                "set use_cantera_for_falloff=True to include. "
                "Pressure conditioning at P_ref=%.2f atm not applied.",
                ktype, contrib.mech_id, self._P_ref_atm,
            )
            return None

        # Unknown type — try plain Arrhenius fallback
        if contrib.A_si is not None and contrib.Ea_si is not None:
            log.debug(
                "Unknown kinetics type '%s' for mech_id=%d — evaluating as Arrhenius",
                ktype, contrib.mech_id,
            )
            return self._eval_arrhenius(contrib.A_si, contrib.n, contrib.Ea_si, T_pts)

        log.warning("Cannot evaluate k(T) for type=%s, mech_id=%d — skipping",
                    ktype, contrib.mech_id)
        return None

    @staticmethod
    def _eval_arrhenius(
        A_si: Optional[float],
        n: float,
        Ea_si: Optional[float],
        T_pts: List[float],
    ) -> Optional[List[float]]:
        """k_i(T) = A · T^n · exp(-Ea / RT) for each T in T_pts."""
        if A_si is None or Ea_si is None or A_si <= 0:
            return None
        result: List[float] = []
        for T in T_pts:
            try:
                k = A_si * (T ** n) * math.exp(-Ea_si / (R_GAS * T))
                result.append(max(k, 0.0))
            except (OverflowError, ZeroDivisionError, ValueError):
                result.append(0.0)
        return result

    def _eval_plog(
        self,
        kinetics_data_json: Optional[str],
        T_pts: List[float],
    ) -> Optional[List[float]]:
        """
        Evaluate PLOG at P_ref by interpolating ln k linearly in ln P space.

        Supports two stored formats:
          - {"pressures": [Pa, ...], "arrhenius": [{"A_si":..,"n":..,"Ea_si":..},...]}
            (produced by both _chemkin_parser and chemkin_reader)
          - {"plog_rates": [{"P_atm":..,"A":..,"n":..,"Ea":..},...]}
            (legacy format used by chemkin_writer)
        """
        if not kinetics_data_json:
            return None
        try:
            kdata = json.loads(kinetics_data_json)
        except (json.JSONDecodeError, TypeError):
            return None

        # Normalise to list of (P_Pa, A_si, n, Ea_si)
        entries: List[Tuple[float, float, float, float]] = []

        if "pressures" in kdata and "arrhenius" in kdata:
            pressures = kdata["pressures"]   # Pa
            arrhs     = kdata["arrhenius"]
            for P_Pa, arr in zip(pressures, arrhs):
                A  = arr.get("A_si") or arr.get("A", 0.0)
                n  = arr.get("n", 0.0)
                Ea = arr.get("Ea_si") or arr.get("Ea", 0.0)
                if A > 0:
                    entries.append((float(P_Pa), float(A), float(n), float(Ea)))
        elif "plog_rates" in kdata:
            # Legacy writer format: A in CGS, Ea in kcal/mol → convert to SI
            for rate in kdata["plog_rates"]:
                P_atm = rate.get("P_atm", 1.0)
                A_cgs = rate.get("A", 0.0)
                n_val = rate.get("n", 0.0)
                Ea_kcal = rate.get("Ea", 0.0)
                A_si  = A_cgs / 1e6          # cm³/mol/s → m³/mol/s (bimolecular)
                Ea_si = Ea_kcal * 4184.0     # kcal/mol → J/mol
                if A_si > 0:
                    entries.append((P_atm * 101_325.0, A_si, n_val, Ea_si))

        if not entries:
            return None

        # Sort by pressure ascending
        entries.sort(key=lambda e: e[0])
        P_vals = [e[0] for e in entries]
        P_ref  = self._P_ref_Pa

        result: List[float] = []
        for T in T_pts:
            k = self._plog_at_T_P(entries, P_vals, P_ref, T)
            result.append(k if (k is not None and k > 0) else 0.0)
        return result

    @staticmethod
    def _plog_at_T_P(
        entries: List[Tuple[float, float, float, float]],
        P_vals:  List[float],
        P_ref:   float,
        T:       float,
    ) -> Optional[float]:
        """Evaluate PLOG rate at (T, P_ref) by log-linear pressure interpolation."""
        def k_arr(A: float, n: float, Ea: float) -> float:
            try:
                return A * (T ** n) * math.exp(-Ea / (R_GAS * T))
            except (OverflowError, ValueError):
                return 0.0

        if P_ref <= P_vals[0]:
            e = entries[0]
            return k_arr(e[1], e[2], e[3])

        if P_ref >= P_vals[-1]:
            e = entries[-1]
            return k_arr(e[1], e[2], e[3])

        # Find bracketing interval
        for i in range(len(P_vals) - 1):
            if P_vals[i] <= P_ref <= P_vals[i + 1]:
                e_lo, e_hi = entries[i], entries[i + 1]
                k_lo = k_arr(e_lo[1], e_lo[2], e_lo[3])
                k_hi = k_arr(e_hi[1], e_hi[2], e_hi[3])
                if k_lo <= 0 or k_hi <= 0:
                    return max(k_lo, k_hi) if (k_lo > 0 or k_hi > 0) else None
                ln_klo = math.log(k_lo)
                ln_khi = math.log(k_hi)
                ln_Plo = math.log(P_vals[i])
                ln_Phi = math.log(P_vals[i + 1])
                frac = (math.log(P_ref) - ln_Plo) / (ln_Phi - ln_Plo)
                return math.exp(ln_klo + frac * (ln_khi - ln_klo))
        return None

    def _eval_via_cantera(
        self,
        contrib: ReactionKinetics,
        T_pts: List[float],
    ) -> Optional[List[float]]:
        """
        Evaluate k(T, P_ref) for a falloff/Chebyshev reaction using Cantera.
        Returns None if Cantera is not available or if the evaluation fails.
        """
        try:
            import cantera as ct
        except ImportError:
            return None

        kdata_str = contrib.kinetics_data_json
        if not kdata_str:
            return None
        try:
            kdata = json.loads(kdata_str)
        except (json.JSONDecodeError, TypeError):
            return None

        ktype = (contrib.kinetics_type or "").lower()

        try:
            if ktype in ("troe", "falloff", "lindemann"):
                return self._cantera_falloff(kdata, ktype, T_pts)
            if ktype == "chebyshev":
                return self._cantera_chebyshev(kdata, T_pts)
        except Exception as exc:
            log.warning("Cantera evaluation failed for mech_id=%d: %s",
                        contrib.mech_id, exc)
        return None

    def _cantera_falloff(
        self,
        kdata: dict,
        ktype: str,
        T_pts: List[float],
    ) -> Optional[List[float]]:
        """Use Cantera to evaluate a Troe/Lindemann rate at P_ref.

        M14 (review.md): this used to reference `ct.TroeFalloff` /
        `ct.FalloffReaction(high_rate=..., low_rate=..., falloff=...)` /
        `ct.NasaPoly2(...)` with a 14-element coefficient array -- all
        Cantera-2 API shapes that either don't exist in Cantera >=3
        (`ct.TroeFalloff`, the `high_rate`/`low_rate`/`falloff` kwargs) or
        silently fail (`NasaPoly2` requires a 15-element array: [Tmid] + 7
        low-T coeffs + 7 high-T coeffs, not 14). None of this was wrapped
        in a try/except beyond the `import cantera` line itself, so any
        Troe/Lindemann band request with use_cantera_for_falloff=True would
        raise uncaught. Also fixed a genuine unit bug: this codebase's
        Ea_si is J/mol (see chemkin_reader.py's `_arrhenius_params`), but
        Cantera's Python `ct.Arrhenius(A, b, Ea)` constructor expects Ea in
        J/**kmol** (confirmed against Cantera's own FalloffReaction
        docstring example: -1700 cal/mol == -7112800 J/kmol passed to the
        constructor) -- every falloff rate evaluated through this path was
        off by a factor of exp(Ea/RT * 999/1000), i.e. wildly wrong except
        coincidentally near Ea=0.
        """
        try:
            import cantera as ct
        except ImportError:
            return None

        high = kdata.get("high", {})
        low  = kdata.get("low",  {})
        if not high or not low:
            return None

        A_hi  = high.get("A_si", 0.0)
        n_hi  = high.get("n", 0.0)
        Ea_hi = high.get("Ea_si", 0.0) * 1000.0   # J/mol -> J/kmol
        A_lo  = low.get("A_si", 0.0)
        n_lo  = low.get("n", 0.0)
        Ea_lo = low.get("Ea_si", 0.0) * 1000.0    # J/mol -> J/kmol

        high_rate = ct.Arrhenius(A_hi, n_hi, Ea_hi)
        low_rate  = ct.Arrhenius(A_lo, n_lo, Ea_lo)

        if ktype in ("troe", "falloff") and "alpha" in kdata:
            alpha = kdata["alpha"]
            T1    = kdata.get("T1", 1e30)
            T2    = kdata.get("T2")
            T3    = kdata.get("T3", 1e-30)
            troe_params = [alpha, T3, T1] + ([T2] if T2 is not None else [])
            rate = ct.TroeRate(low=low_rate, high=high_rate, falloff_coeffs=troe_params)
        else:
            rate = ct.LindemannRate(low=low_rate, high=high_rate)

        try:
            rxn = ct.Reaction(reactants={"A": 1}, products={"B": 1}, rate=rate)
        except Exception:
            log.debug("Could not build falloff Reaction for band evaluation", exc_info=True)
            return None

        # Build minimal solution to evaluate k(T, P). NasaPoly2's data array
        # is [Tmid, low_coeffs(7), high_coeffs(7)] = 15 elements.
        sp_A = ct.Species("A", "C:1")
        sp_B = ct.Species("B", "C:1")
        sp_M = ct.Species("M", "N:2")
        for sp in [sp_A, sp_B, sp_M]:
            sp.thermo = ct.NasaPoly2(300, 5000, 1000,
                                      [1000.0,
                                       3.5, 0, 0, 0, 0, -745, 3.5,
                                       3.5, 0, 0, 0, 0, -745, 3.5])
        gas = ct.Solution(thermo="ideal-gas", kinetics="gas",
                          species=[sp_A, sp_B, sp_M],
                          reactions=[rxn])
        P_Pa = self._P_ref_Pa
        result: List[float] = []
        for T in T_pts:
            gas.TPX = T, P_Pa, "A:0.5, M:0.5"
            k = gas.forward_rate_constants[0]
            result.append(float(k))
        return result

    def _cantera_chebyshev(
        self,
        kdata: dict,
        T_pts: List[float],
    ) -> Optional[List[float]]:
        """Evaluate Chebyshev rate at P_ref using Cantera."""
        try:
            import cantera as ct
        except ImportError:
            return None

        Tmin  = kdata.get("Tmin", 300.0)
        Tmax  = kdata.get("Tmax", 3000.0)
        Pmin  = kdata.get("Pmin", 101.325)
        Pmax  = kdata.get("Pmax", 1.013e7)
        coeffs = kdata.get("coeffs", [])
        if not coeffs:
            return None

        try:
            rxn = ct.Reaction(
                reactants={"A": 1}, products={"B": 1},
                rate=ct.ChebyshevRate(
                    temperature_range=(Tmin, Tmax),
                    pressure_range=(Pmin, Pmax),
                    data=coeffs,
                ),
            )
        except Exception:
            log.debug("Could not build Chebyshev Reaction for band evaluation", exc_info=True)
            return None

        sp_A = ct.Species("A", "C:1")
        sp_B = ct.Species("B", "C:1")
        for sp in [sp_A, sp_B]:
            sp.thermo = ct.NasaPoly2(300, 5000, 1000,
                                      [1000.0,
                                       3.5, 0, 0, 0, 0, -745, 3.5,
                                       3.5, 0, 0, 0, 0, -745, 3.5])
        gas = ct.Solution(thermo="ideal-gas", kinetics="gas",
                          species=[sp_A, sp_B],
                          reactions=[rxn])
        P_Pa = self._P_ref_Pa
        result: List[float] = []
        for T in T_pts:
            gas.TPX = T, P_Pa, "A:1.0"
            k = gas.forward_rate_constants[0]
            result.append(float(k))
        return result
