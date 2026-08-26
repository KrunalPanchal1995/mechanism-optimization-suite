"""
rmip/analysis/fusion_uncertainty.py
====================================
Temperature-dependent uncertainty bands for a mean-fusion mechanism, built
from 2+ real source mechanisms (2026-07-14 Keromnes/Li/Davis fusion
session). Confirmed methodology (user sign-off, this session):

  A-factor UF (single scalar per reaction)
  -----------------------------------------
  UF = max over the T grid of (k_max(T) / k_mean(T)), arithmetic mean,
  worst-case over the whole range. Bounds: [A_mean/UF, A_mean*UF] with n,
  Ea held at their fused (geometric-mean) values.

  3-parameter (A, n, Ea) temperature-dependent band
  ----------------------------------------------------
  At each grid point T: mean_ln = mean_i(ln k_i(T)), f(T) =
  exp(max_i |ln k_i(T) - mean_ln|), k_gmean(T) = exp(mean_ln). Band =
  [k_gmean(T)/f(T), k_gmean(T)*f(T)] -- symmetric in ln(k), contains every
  source by construction. The upper/lower envelope points are then each
  fit to an effective 2-parameter Arrhenius form (A_env, Ea_env) at fixed
  n = n_mean via ordinary least squares on ln(k) vs 1/T, then inflated so
  the smooth fitted curve still contains the full discrete band (and hence
  every source) at every grid point.

Reuses `ReactionClassification.kinetics_set.mech_data` (already computed
by GraphComparator -- {mech_id: {A_si, n, Ea_si, kinetics_type, ...}}) so
this module never re-queries the DB directly.

Falloff (Troe/Lindemann/SRI/PLOG/Chebyshev) reactions are represented in
this suite by a single high-pressure-limit Arrhenius triple at the
DB/kinetics_set level (see cantera_reader.py/_chemkin_parser.py) -- the
band below is computed from that same high-P representative value, which
is a real but incomplete picture (it doesn't capture the falloff curve's
own pressure dependence). Each such record is tagged
``high_p_limit_only=True`` rather than silently treated as identical in
rigor to a plain Arrhenius reaction.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from rmip.types import GraphComparisonResult, ReactionClassification, UncertaintyBand

R_GAS_SI = 8.314462618  # J/mol/K -- matches fusion.py's R_GAS


# ---------------------------------------------------------------------------
# Temperature grid
# ---------------------------------------------------------------------------

def make_inverse_T_grid(T_min: float = 300.0, T_max: float = 2500.0, n: int = 80) -> List[float]:
    """Grid uniform in 1000/T (theta), not in T. Arrhenius curves are
    straight lines in ln(k) vs 1/T, so uniform-1/T spacing weights the
    envelope least-squares fit evenly across the whole extrapolation
    range instead of over-weighting the high-T end (which a uniform-T
    grid would do, since d(1/T) shrinks as T grows)."""
    theta_min = 1000.0 / T_max
    theta_max = 1000.0 / T_min
    if n < 2:
        raise ValueError("n must be >= 2")
    step = (theta_max - theta_min) / (n - 1)
    return [1000.0 / (theta_min + i * step) for i in range(n)]


def arrhenius_k(A: Optional[float], n: Optional[float], Ea: Optional[float],
                T: float) -> Optional[float]:
    """k = A * T^n * exp(-Ea/RT), SI (m,mol,s,J/mol). None on missing
    inputs or numeric overflow (extreme extrapolation)."""
    if A is None or n is None or Ea is None or A <= 0:
        return None
    try:
        return A * (T ** n) * math.exp(-Ea / (R_GAS_SI * T))
    except (OverflowError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class EnvelopeFit:
    """Effective 2-parameter Arrhenius fit (A, Ea at fixed n) to one
    envelope (upper or lower) of the discrete T-grid band, inflated so the
    smooth curve still contains every discrete grid point."""
    n_fixed:     float
    A_si_raw:    float    # unfit-inflated least-squares result
    Ea_si_raw:   float
    A_si:        float    # inflated -- use this one for MSO bounds
    Ea_si:       float
    fit_rms_ln_k: float   # RMS residual of the raw fit, ln(k) units
    max_containment_gap_ln_k: float  # how far the raw fit missed containment (0 if none)


@dataclass
class FusionUncertaintyRecord:
    canonical_key:    str
    occurrence_index: int
    formula_string:   str
    source_mech_ids:  List[int]   # mechs actually contributing to this band
    high_p_limit_only: bool       # True for troe/lindemann/sri/plog/chebyshev

    T_grid_K:      List[float]
    k_by_source:   Dict[int, List[Optional[float]]]
    k_geo_mean:    List[float]     # exp(mean_i ln k_i(T)) at each grid point
    f_T:           List[float]     # exp(max_i |ln k_i - mean_ln|) at each grid point
    k_lb:          List[float]     # discrete lower envelope = k_geo_mean/f_T
    k_ub:          List[float]     # discrete upper envelope = k_geo_mean*f_T

    params_mean: Tuple[float, float, float]  # (A_si, n, Ea_si) of the fused nominal

    uf_A: float                                  # worst-case-over-grid A-factor UF
    A_bounds_uf: Tuple[float, float]              # (A_mean/UF, A_mean*UF)

    envelope_lower: Optional[EnvelopeFit]
    envelope_upper: Optional[EnvelopeFit]

    def to_uncertainty_band(self) -> UncertaintyBand:
        """Adapt to this repo's existing UncertaintyBand type (types.py) so
        it can be fed to CanteraYAMLWriter.write(..., uncertainty_bands=...)
        to embed rmip-uncertainty keys into the fused YAML."""
        f_uncertainty = [max(f, 1.0 + 1e-12) for f in self.f_T]
        sigma_lnk = [math.log(f) / 3.0 for f in f_uncertainty]
        return UncertaintyBand(
            canonical_key=self.canonical_key,
            formula_string=self.formula_string,
            T_K=list(self.T_grid_K),
            k_nom=list(self.k_geo_mean),
            k_min=list(self.k_lb),
            k_max=list(self.k_ub),
            f_uncertainty=f_uncertainty,
            sigma_lnk=sigma_lnk,
            f_to_nom=list(f_uncertainty),  # k_max/k_nom == f (symmetric-in-ln construction)
            contributor_mech_ids=list(self.source_mech_ids),
            method="fusion_minmax_lnk",
        )

    def to_dict(self) -> dict:
        return {
            "canonical_key": self.canonical_key,
            "occurrence_index": self.occurrence_index,
            "formula_string": self.formula_string,
            "source_mech_ids": self.source_mech_ids,
            "high_p_limit_only": self.high_p_limit_only,
            "params_mean": {"A_si": self.params_mean[0], "n": self.params_mean[1],
                             "Ea_si": self.params_mean[2]},
            "uf_A": self.uf_A,
            "A_bounds_uf": {"A_lb": self.A_bounds_uf[0], "A_ub": self.A_bounds_uf[1]},
            "envelope_lower": None if self.envelope_lower is None else {
                "n": self.envelope_lower.n_fixed,
                "A_si": self.envelope_lower.A_si, "Ea_si": self.envelope_lower.Ea_si,
                "A_si_raw": self.envelope_lower.A_si_raw, "Ea_si_raw": self.envelope_lower.Ea_si_raw,
                "fit_rms_ln_k": self.envelope_lower.fit_rms_ln_k,
            },
            "envelope_upper": None if self.envelope_upper is None else {
                "n": self.envelope_upper.n_fixed,
                "A_si": self.envelope_upper.A_si, "Ea_si": self.envelope_upper.Ea_si,
                "A_si_raw": self.envelope_upper.A_si_raw, "Ea_si_raw": self.envelope_upper.Ea_si_raw,
                "fit_rms_ln_k": self.envelope_upper.fit_rms_ln_k,
            },
            "n_T_points": len(self.T_grid_K),
            "T_range_K": [min(self.T_grid_K), max(self.T_grid_K)],
        }

    def to_mso_bounds_block(self) -> dict:
        """Same shape as FusionEngine.export_mso_bounds()'s fuse_blocks
        entries, but params_lb/ub are the inflated 3-param envelope fits
        instead of the source min/max -- for downstream MSO tooling that
        already consumes that JSON schema."""
        lb = self.envelope_lower
        ub = self.envelope_upper
        return {
            "canonical_key": self.canonical_key,
            "formula_string": self.formula_string,
            "source_mech_ids": self.source_mech_ids,
            "params_mean": list(self.params_mean),
            "params_lb": None if lb is None else [lb.A_si, lb.n_fixed, lb.Ea_si],
            "params_ub": None if ub is None else [ub.A_si, ub.n_fixed, ub.Ea_si],
            "param_names": ["A_si", "n", "Ea_si"],
            "units": ["m^3/mol/s (SI, order-dependent)", "1", "J/mol"],
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

_HIGH_P_LIMIT_TYPES = {"troe", "lindemann", "sri", "plog", "falloff"}
_NO_BAND_TYPES = {"chebyshev"}  # no representative Arrhenius triple at all


class FusionUncertaintyEngine:
    """Computes FusionUncertaintyRecords for every reaction shared by >=2
    of the source mechanisms in a GraphComparisonResult."""

    def __init__(self, T_min: float = 300.0, T_max: float = 2500.0, n_grid: int = 80):
        self.T_grid = make_inverse_T_grid(T_min, T_max, n_grid)

    def compute(
        self, cmp_result: GraphComparisonResult,
    ) -> List[FusionUncertaintyRecord]:
        records: List[FusionUncertaintyRecord] = []
        for rc in cmp_result.reaction_classifications:
            if len(rc.present_in_mech_ids) < 2 or rc.kinetics_set is None:
                continue
            rec = self._compute_one(rc)
            if rec is not None:
                records.append(rec)
        return records

    def _compute_one(self, rc: ReactionClassification) -> Optional[FusionUncertaintyRecord]:
        mech_data = rc.kinetics_set.mech_data  # {mid: {A_si,n,Ea_si,kinetics_type,...}}
        ktypes = {(d.get("kinetics_type") or "arrhenius").lower() for d in mech_data.values()}

        if ktypes & _NO_BAND_TYPES:
            return None  # no representative Arrhenius triple -- nothing to band
        high_p_limit_only = bool(ktypes & _HIGH_P_LIMIT_TYPES)

        # Per-source k(T) across the grid.
        k_by_source: Dict[int, List[Optional[float]]] = {}
        for mid, d in mech_data.items():
            A, n, Ea = d.get("A_si"), d.get("n"), d.get("Ea_si")
            k_by_source[mid] = [arrhenius_k(A, n, Ea, T) for T in self.T_grid]

        # Need >=2 sources with fully-valid k(T) across the whole grid to
        # band meaningfully.
        valid_mids = [mid for mid, ks in k_by_source.items() if all(k is not None and k > 0 for k in ks)]
        if len(valid_mids) < 2:
            return None

        n_grid = len(self.T_grid)
        k_geo_mean: List[float] = []
        f_T: List[float] = []
        k_lb: List[float] = []
        k_ub: List[float] = []
        for j in range(n_grid):
            ln_ks = [math.log(k_by_source[mid][j]) for mid in valid_mids]
            mean_ln = sum(ln_ks) / len(ln_ks)
            max_dev = max(abs(lk - mean_ln) for lk in ln_ks)
            f = math.exp(max_dev)
            kg = math.exp(mean_ln)
            k_geo_mean.append(kg)
            f_T.append(f)
            k_lb.append(kg / f)
            k_ub.append(kg * f)

        # Arithmetic-mean k(T) and worst-case A-factor UF (confirmed spec:
        # arithmetic mean, NOT the geometric mean used for the band itself).
        uf = 1.0
        for j in range(n_grid):
            ks = [k_by_source[mid][j] for mid in valid_mids]
            k_arith_mean = sum(ks) / len(ks)
            k_max = max(ks)
            if k_arith_mean > 0:
                uf = max(uf, k_max / k_arith_mean)

        # Fused nominal params (mean_A_si/mean_n/mean_Ea_si from
        # GraphComparator's own aggregate -- geometric-mean A, arithmetic
        # mean n/Ea, matching FusionEngine's FUSE-bucket nominal exactly).
        A_mean = rc.mean_A_si
        n_mean = rc.mean_n
        Ea_mean = rc.mean_Ea_si
        if A_mean is None or n_mean is None or Ea_mean is None:
            # SHARED_IDENTICAL reactions don't populate mean_* (no
            # divergence to aggregate) -- fall back to the plain mean of
            # the present sources' own A/n/Ea.
            A_vals = [mech_data[m]["A_si"] for m in valid_mids]
            n_vals = [mech_data[m]["n"] for m in valid_mids]
            Ea_vals = [mech_data[m]["Ea_si"] for m in valid_mids]
            A_mean = math.exp(sum(math.log(a) for a in A_vals) / len(A_vals))
            n_mean = sum(n_vals) / len(n_vals)
            Ea_mean = sum(Ea_vals) / len(Ea_vals)

        A_bounds_uf = (A_mean / uf, A_mean * uf)

        envelope_lower = self._fit_envelope(k_lb, n_mean, direction="lower")
        envelope_upper = self._fit_envelope(k_ub, n_mean, direction="upper")

        return FusionUncertaintyRecord(
            canonical_key=rc.canonical_key,
            occurrence_index=rc.occurrence_index,
            formula_string=rc.formula_string,
            source_mech_ids=valid_mids,
            high_p_limit_only=high_p_limit_only,
            T_grid_K=list(self.T_grid),
            k_by_source=k_by_source,
            k_geo_mean=k_geo_mean,
            f_T=f_T,
            k_lb=k_lb,
            k_ub=k_ub,
            params_mean=(A_mean, n_mean, Ea_mean),
            uf_A=uf,
            A_bounds_uf=A_bounds_uf,
            envelope_lower=envelope_lower,
            envelope_upper=envelope_upper,
        )

    def _fit_envelope(
        self, k_env: List[float], n_fixed: float, direction: str,
    ) -> Optional[EnvelopeFit]:
        """Ordinary least squares: y_j = ln(k_env(T_j)) - n_fixed*ln(T_j),
        x_j = 1/T_j, fit y = ln(A_env) - (Ea_env/R)*x. Then inflate A_env
        multiplicatively so the fitted curve still contains every discrete
        grid point of k_env (the discrete band is the ground truth; the
        fit is a smooth convenience wrapper around it, never the other way
        around). `direction` is explicit ("upper": fit must sit >= k_env
        everywhere; "lower": fit must sit <= k_env everywhere) -- trying to
        infer this from residual signs is unreliable when the true curve
        isn't a pure Arrhenius shape (fixed-n fit to a band built from
        varying-n sources)."""
        assert direction in ("upper", "lower")
        n_pts = len(self.T_grid)
        if n_pts < 2:
            return None
        xs = [1.0 / T for T in self.T_grid]
        ys = [math.log(k_env[j]) - n_fixed * math.log(self.T_grid[j]) for j in range(n_pts)]

        x_bar = sum(xs) / n_pts
        y_bar = sum(ys) / n_pts
        s_xx = sum((x - x_bar) ** 2 for x in xs)
        s_xy = sum((xs[j] - x_bar) * (ys[j] - y_bar) for j in range(n_pts))
        if s_xx == 0:
            return None
        slope = s_xy / s_xx          # slope = -Ea/R
        intercept = y_bar - slope * x_bar  # intercept = ln(A)

        Ea_raw = -slope * R_GAS_SI
        A_raw = math.exp(intercept)

        # Residuals of the raw (uninflated) fit, in ln(k) space: positive
        # means the fit sits ABOVE the true discrete envelope at that point.
        residuals = []
        for j in range(n_pts):
            ln_k_fit = math.log(A_raw) + n_fixed * math.log(self.T_grid[j]) - Ea_raw / (R_GAS_SI * self.T_grid[j])
            ln_k_true = math.log(k_env[j])
            residuals.append(ln_k_fit - ln_k_true)
        rms = math.sqrt(sum(r ** 2 for r in residuals) / n_pts)
        max_gap = max(abs(r) for r in residuals)

        if direction == "upper":
            # Fit must be >= true everywhere: push A up by the worst case
            # where the raw fit fell BELOW the true envelope (negative
            # residual). No-op if the raw fit already dominates everywhere.
            worst_below = max((-r for r in residuals if r < 0), default=0.0)
            inflate_ln = worst_below
        else:
            # Fit must be <= true everywhere: push A down by the worst case
            # where the raw fit rose ABOVE the true envelope.
            worst_above = max((r for r in residuals if r > 0), default=0.0)
            inflate_ln = -worst_above

        A_inflated = A_raw * math.exp(inflate_ln)

        return EnvelopeFit(
            n_fixed=n_fixed,
            A_si_raw=A_raw, Ea_si_raw=Ea_raw,
            A_si=A_inflated, Ea_si=Ea_raw,
            fit_rms_ln_k=rms,
            max_containment_gap_ln_k=max_gap,
        )
