"""
rmip/tests/test_fusion_uncertainty.py
=======================================
Regression coverage for the fusion uncertainty-band module built for the
Keromnes/Li/Davis fusion session (2026-07-14). Confirmed methodology:
worst-case-over-grid A-factor UF (arithmetic mean), symmetric-in-ln(k)
3-parameter band (geometric mean +/- max deviation), fitted+inflated
envelope curves guaranteed to contain the discrete band.
"""
from __future__ import annotations

import math

import pytest

from rmip.analysis.fusion_uncertainty import (
    FusionUncertaintyEngine, arrhenius_k, make_inverse_T_grid, R_GAS_SI,
)


class TestGrid:
    def test_grid_endpoints(self):
        grid = make_inverse_T_grid(300.0, 2500.0, 80)
        assert len(grid) == 80
        assert min(grid) == pytest.approx(300.0, abs=1e-6)
        assert max(grid) == pytest.approx(2500.0, abs=1e-6)

    def test_grid_uniform_in_inverse_T(self):
        grid = make_inverse_T_grid(300.0, 2500.0, 10)
        inv = [1000.0 / T for T in grid]
        diffs = [inv[i + 1] - inv[i] for i in range(len(inv) - 1)]
        assert all(d == pytest.approx(diffs[0], rel=1e-6) for d in diffs)


class TestArrheniusK:
    def test_basic(self):
        k = arrhenius_k(1e10, 0.0, 50000.0, 1000.0)
        expected = 1e10 * math.exp(-50000.0 / (R_GAS_SI * 1000.0))
        assert k == pytest.approx(expected, rel=1e-9)

    def test_none_on_missing_params(self):
        assert arrhenius_k(None, 0.0, 1000.0, 500.0) is None
        assert arrhenius_k(1e10, None, 1000.0, 500.0) is None


class TestFusionUncertaintyEngine:
    def _make_classification(self, mech_data, n_grid=10):
        from rmip.types import ReactionClassification, ReactionClass, ReactionKineticsSet
        ks = ReactionKineticsSet(
            canonical_key="R1", formula_string="A<=>B", mech_data=mech_data,
        )
        return ReactionClassification(
            canonical_key="R1", formula_string="A<=>B",
            classification=ReactionClass.SHARED_DIVERGENT,
            present_in_mech_ids=sorted(mech_data.keys()),
            occurrence_index=0,
            kinetics_set=ks,
            mean_A_si=None, mean_n=None, mean_Ea_si=None,
        )

    def test_band_contains_all_sources_at_every_grid_point(self):
        """Sanity check required by the confirmed spec: the discrete band
        [k_lb(T), k_ub(T)] must contain every source's k(T) at every T."""
        mech_data = {
            1: {"A_si": 1.0e10, "n": 0.0, "Ea_si": 40000.0, "kinetics_type": "arrhenius"},
            2: {"A_si": 1.3e10, "n": 0.1, "Ea_si": 42000.0, "kinetics_type": "arrhenius"},
            3: {"A_si": 0.8e10, "n": -0.1, "Ea_si": 38000.0, "kinetics_type": "arrhenius"},
        }
        rc = self._make_classification(mech_data)
        engine = FusionUncertaintyEngine(T_min=300.0, T_max=2500.0, n_grid=40)
        records = engine.compute(
            type("R", (), {"reaction_classifications": [rc]})()
        )
        assert len(records) == 1
        rec = records[0]
        for j, T in enumerate(rec.T_grid_K):
            for mid in (1, 2, 3):
                k_src = arrhenius_k(mech_data[mid]["A_si"], mech_data[mid]["n"],
                                     mech_data[mid]["Ea_si"], T)
                # exp(log(x)) round-trips introduce ~1e-9 relative FP noise
                # at the source that defines an envelope edge exactly.
                assert rec.k_lb[j] <= k_src * (1 + 1e-8), (T, mid, k_src, rec.k_lb[j])
                assert k_src <= rec.k_ub[j] * (1 + 1e-8), (T, mid, k_src, rec.k_ub[j])

    def test_uf_A_is_worst_case_arithmetic_ratio(self):
        mech_data = {
            1: {"A_si": 1.0e10, "n": 0.0, "Ea_si": 40000.0, "kinetics_type": "arrhenius"},
            2: {"A_si": 5.0e10, "n": 0.0, "Ea_si": 40000.0, "kinetics_type": "arrhenius"},
        }
        rc = self._make_classification(mech_data)
        engine = FusionUncertaintyEngine(T_min=300.0, T_max=2500.0, n_grid=5)
        records = engine.compute(type("R", (), {"reaction_classifications": [rc]})())
        rec = records[0]
        # Same n/Ea for both sources -> k ratio is constant = A2/A1 = 5 at
        # every T. arithmetic mean = 3e10, k_max/k_mean = 5e10/3e10 = 5/3.
        assert rec.uf_A == pytest.approx(5.0 / 3.0, rel=1e-6)

    def test_envelope_fit_contains_discrete_band(self):
        mech_data = {
            1: {"A_si": 1.0e10, "n": 0.0, "Ea_si": 40000.0, "kinetics_type": "arrhenius"},
            2: {"A_si": 1.3e10, "n": 0.15, "Ea_si": 42000.0, "kinetics_type": "arrhenius"},
            3: {"A_si": 0.8e10, "n": -0.15, "Ea_si": 38000.0, "kinetics_type": "arrhenius"},
        }
        rc = self._make_classification(mech_data)
        engine = FusionUncertaintyEngine(T_min=300.0, T_max=2500.0, n_grid=60)
        rec = engine.compute(type("R", (), {"reaction_classifications": [rc]})())[0]
        assert rec.envelope_lower is not None
        assert rec.envelope_upper is not None
        for j, T in enumerate(rec.T_grid_K):
            k_fit_ub = arrhenius_k(rec.envelope_upper.A_si, rec.envelope_upper.n_fixed,
                                    rec.envelope_upper.Ea_si, T)
            k_fit_lb = arrhenius_k(rec.envelope_lower.A_si, rec.envelope_lower.n_fixed,
                                    rec.envelope_lower.Ea_si, T)
            assert k_fit_ub >= rec.k_ub[j] * (1 - 1e-9), (T, k_fit_ub, rec.k_ub[j])
            assert k_fit_lb <= rec.k_lb[j] * (1 + 1e-9), (T, k_fit_lb, rec.k_lb[j])

    def test_falloff_reactions_flagged_high_p_limit_only(self):
        mech_data = {
            1: {"A_si": 1.0e10, "n": 0.0, "Ea_si": 0.0, "kinetics_type": "troe"},
            2: {"A_si": 1.2e10, "n": 0.0, "Ea_si": 0.0, "kinetics_type": "troe"},
        }
        rc = self._make_classification(mech_data)
        engine = FusionUncertaintyEngine(n_grid=5)
        rec = engine.compute(type("R", (), {"reaction_classifications": [rc]})())[0]
        assert rec.high_p_limit_only is True

    def test_chebyshev_skipped_no_representative_rate(self):
        mech_data = {
            1: {"A_si": None, "n": None, "Ea_si": None, "kinetics_type": "chebyshev"},
            2: {"A_si": None, "n": None, "Ea_si": None, "kinetics_type": "chebyshev"},
        }
        rc = self._make_classification(mech_data)
        engine = FusionUncertaintyEngine(n_grid=5)
        records = engine.compute(type("R", (), {"reaction_classifications": [rc]})())
        assert records == []

    def test_uncertainty_band_adapter_round_trips(self):
        mech_data = {
            1: {"A_si": 1.0e10, "n": 0.0, "Ea_si": 40000.0, "kinetics_type": "arrhenius"},
            2: {"A_si": 1.3e10, "n": 0.0, "Ea_si": 40000.0, "kinetics_type": "arrhenius"},
        }
        rc = self._make_classification(mech_data)
        engine = FusionUncertaintyEngine(n_grid=5)
        rec = engine.compute(type("R", (), {"reaction_classifications": [rc]})())[0]
        band = rec.to_uncertainty_band()
        assert band.canonical_key == "R1"
        assert len(band.k_nom) == 5
        assert all(f >= 1.0 for f in band.f_uncertainty)
