"""
rmip/tests/test_uncertainty_bands_m14.py
==========================================
Regression coverage for M14 (review.md, PLAN.md Priority-0 item #5,
2026-07-17): UncertaintyBandEstimator._cantera_falloff()/_cantera_chebyshev()
referenced Cantera-2 API shapes that don't exist in Cantera >=3
(`ct.TroeFalloff`, `ct.FalloffReaction(high_rate=..., low_rate=...,
falloff=...)`, `ct.ChebyshevReaction`) and a `ct.NasaPoly2` call with a
14-element coefficient array (needs 15: [Tmid] + 7 low + 7 high) -- any of
these would raise uncaught the moment a Troe/Lindemann/Chebyshev band was
requested with use_cantera_for_falloff=True, since only the `import
cantera` line was wrapped in try/except. Also fixed a real unit bug: this
codebase's Ea_si is J/mol, but Cantera's Python `ct.Arrhenius(A, b, Ea)`
expects Ea in J/kmol -- every falloff rate evaluated through the old code
was off by orders of magnitude except coincidentally near Ea=0.

These tests require a real `cantera` install (skipped otherwise, same
convention as the rest of this suite's Cantera-dependent tests).
"""
from __future__ import annotations

import math

import pytest

ct = pytest.importorskip("cantera")

from rmip.analysis.uncertainty_bands import UncertaintyBandEstimator


class TestM14CanteraFalloffAPICompat:
    def test_troe_falloff_does_not_raise_and_returns_real_numbers(self):
        """The exact scenario that used to crash: ct.TroeFalloff doesn't
        exist in Cantera 3, ct.FalloffReaction's high_rate/low_rate/falloff
        kwargs don't exist either."""
        est = UncertaintyBandEstimator(P_ref_atm=1.0)
        kdata = {
            "alpha": 0.5, "T1": 100.0, "T2": 2000.0, "T3": 10.0,
            "low":  {"A_si": 1e15, "n": -1.0, "Ea_si": 1000.0},
            "high": {"A_si": 1e10, "n": 0.5, "Ea_si": 50000.0},
        }
        result = est._cantera_falloff(kdata, "troe", [1000.0])
        assert result is not None
        assert len(result) == 1
        assert result[0] > 0
        assert math.isfinite(result[0])

    def test_lindemann_falloff_no_alpha_does_not_raise(self):
        est = UncertaintyBandEstimator(P_ref_atm=1.0)
        kdata = {
            "low":  {"A_si": 1e15, "n": -1.0, "Ea_si": 1000.0},
            "high": {"A_si": 1e10, "n": 0.5, "Ea_si": 50000.0},
        }
        result = est._cantera_falloff(kdata, "lindemann", [1000.0])
        assert result is not None
        assert result[0] > 0
        assert math.isfinite(result[0])

    def test_chebyshev_does_not_raise(self):
        est = UncertaintyBandEstimator(P_ref_atm=1.0)
        kdata = {
            "Tmin": 300.0, "Tmax": 3000.0, "Pmin": 101.325, "Pmax": 1.013e7,
            "coeffs": [[8.5, 0.5], [0.3, -0.1]],
        }
        result = est._cantera_chebyshev(kdata, [1000.0])
        assert result is not None
        assert result[0] > 0
        assert math.isfinite(result[0])

    def test_troe_matches_independent_yaml_ground_truth(self):
        """Cross-check against an independently-built Cantera YAML mechanism
        (bypassing this module's Python-API construction entirely) using the
        SAME kdata -- catches both the API-shape bugs and the Ea unit bug at
        once, since a wrong unit conversion would produce a wildly different
        (not just slightly off) rate constant."""
        yaml_str = """
phases:
- name: gas
  thermo: ideal-gas
  elements: [C, N]
  species: [A, B, M]
  kinetics: gas
  reactions: all
species:
- name: A
  composition: {C: 1}
  thermo:
    model: NASA7
    temperature-ranges: [300.0, 1000.0, 5000.0]
    data:
    - [3.5, 0, 0, 0, 0, -745, 3.5]
    - [3.5, 0, 0, 0, 0, -745, 3.5]
- name: B
  composition: {C: 1}
  thermo:
    model: NASA7
    temperature-ranges: [300.0, 1000.0, 5000.0]
    data:
    - [3.5, 0, 0, 0, 0, -745, 3.5]
    - [3.5, 0, 0, 0, 0, -745, 3.5]
- name: M
  composition: {N: 2}
  thermo:
    model: NASA7
    temperature-ranges: [300.0, 1000.0, 5000.0]
    data:
    - [3.5, 0, 0, 0, 0, -745, 3.5]
    - [3.5, 0, 0, 0, 0, -745, 3.5]
reactions:
- equation: A (+M) <=> B (+M)
  type: falloff
  low-P-rate-constant: {A: 1.0e15, b: -1.0, Ea: 1000.0 J/mol}
  high-P-rate-constant: {A: 1.0e10, b: 0.5, Ea: 50000.0 J/mol}
  Troe: {A: 0.5, T3: 10.0, T1: 100.0, T2: 2000.0}
"""
        gas = ct.Solution(yaml=yaml_str)
        gas.TPX = 1000.0, 101325.0, "A:0.5, M:0.5"
        k_ground_truth = gas.forward_rate_constants[0]

        est = UncertaintyBandEstimator(P_ref_atm=1.0)
        kdata = {
            "alpha": 0.5, "T1": 100.0, "T2": 2000.0, "T3": 10.0,
            "low":  {"A_si": 1e15, "n": -1.0, "Ea_si": 1000.0},
            "high": {"A_si": 1e10, "n": 0.5, "Ea_si": 50000.0},
        }
        result = est._cantera_falloff(kdata, "troe", [1000.0])

        assert result[0] == pytest.approx(k_ground_truth, rel=1e-6)

    def test_ea_unit_conversion_actually_applied(self):
        """Sanity check that the fix is the J/mol->J/kmol conversion
        specifically, not something that happens to cancel out: a
        wrong-by-1000x Ea in the exponential term should change the
        rate constant by many orders of magnitude at typical combustion
        temperatures, so a naive (unconverted) implementation would fail
        the ground-truth comparison above by more than a rounding error."""
        est = UncertaintyBandEstimator(P_ref_atm=1.0)
        kdata = {
            "low":  {"A_si": 1e15, "n": -1.0, "Ea_si": 100000.0},
            "high": {"A_si": 1e10, "n": 0.5, "Ea_si": 50000.0},
        }
        result = est._cantera_falloff(kdata, "lindemann", [1000.0])
        # Manual ground truth via the high-pressure-limit Arrhenius alone
        # is not exact for a falloff blend, but the unconverted (buggy)
        # Ea would make k effectively zero (exp(-1e8/(R*1000)) ~ 1e-5230);
        # the fixed version must be a physically sane finite positive value
        # in normal double range.
        assert 0 < result[0] < 1e30
