"""
rmip/tests/test_thermo_fitter.py
================================
Tests for NASA-7 thermo fitting + consistency repair
(``rmip/analysis/thermo_fitter.py``), Future.md pt. 3.

Run with:
    python -m pytest src/rmip/tests/test_thermo_fitter.py -v
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from rmip.analysis.thermo_fitter import (
    DEFAULT_DISCONTINUITY_TOL,
    NASA7Fit,
    R_GAS,
    T_REF,
    ThermoRepairer,
    continuity_jumps,
    fit_nasa7,
    is_consistent,
    nasa7_cp_R,
    nasa7_H_RT,
    nasa7_S_R,
    refit_from_coeffs,
)

# GRI-Mech 3.0 reference coefficients (continuous, well-behaved) --------------
# N2, Tmid = 1000 K
_N2_LOW = [3.298677e00, 1.408240e-03, -3.963222e-06, 5.641515e-09,
           -2.444855e-12, -1.020900e03, 3.950372e00]
_N2_HIGH = [2.926640e00, 1.487977e-03, -5.684760e-07, 1.009704e-10,
            -6.753351e-15, -9.227977e02, 5.980528e00]

# O2, Tmid = 1000 K
_O2_LOW = [3.782456e00, -2.996734e-03, 9.847302e-06, -9.681295e-09,
           3.243728e-12, -1.063944e03, 3.657676e00]
_O2_HIGH = [3.282538e00, 1.483088e-03, -7.579667e-07, 2.094706e-10,
            -2.167178e-14, -1.088458e03, 5.453231e00]


# ---------------------------------------------------------------------------
# NASA-7 property evaluation
# ---------------------------------------------------------------------------

class TestNasa7Eval:
    def test_cp_matches_hand_calc(self):
        # Cp/R = a1 + a2 T + ... at T=300 for N2 low range
        T = 300.0
        expect = (_N2_LOW[0] + _N2_LOW[1]*T + _N2_LOW[2]*T**2
                  + _N2_LOW[3]*T**3 + _N2_LOW[4]*T**4)
        assert nasa7_cp_R(_N2_LOW, T) == pytest.approx(expect)

    def test_n2_cp_near_diatomic_limit(self):
        # N2 at 300 K: Cp/R ~ 3.5 (7/2 for a diatomic)
        assert nasa7_cp_R(_N2_LOW, 300.0) == pytest.approx(3.5, abs=0.05)

    def test_h_and_s_finite(self):
        assert nasa7_H_RT(_N2_LOW, 298.15) == pytest.approx(
            nasa7_H_RT(_N2_LOW, 298.15))
        assert nasa7_S_R(_N2_LOW, 298.15) > 0


# ---------------------------------------------------------------------------
# continuity_jumps / is_consistent
# ---------------------------------------------------------------------------

class TestContinuity:
    def test_good_polynomial_is_consistent(self):
        j = continuity_jumps(_N2_LOW, _N2_HIGH, 1000.0)
        assert j["cp"] < 1e-6
        assert j["h"] < 1e-6
        assert j["s"] < 1e-6
        assert is_consistent(_N2_LOW, _N2_HIGH, 1000.0)

    def test_discontinuous_polynomial_detected(self):
        bad_high = list(_N2_HIGH)
        bad_high[0] *= 1.05           # 5 % bump in a1 → Cp/H/S jump
        j = continuity_jumps(_N2_LOW, bad_high, 1000.0)
        assert j["cp"] > 0.01
        assert not is_consistent(_N2_LOW, bad_high, 1000.0)

    def test_rel_jump_symmetric_and_zero(self):
        assert continuity_jumps(_N2_LOW, _N2_LOW, 1000.0)["cp"] == 0.0


# ---------------------------------------------------------------------------
# fit_nasa7 — core constrained least-squares
# ---------------------------------------------------------------------------

class TestFitNasa7:
    def _sample(self, low, high, tmid, tmin=300.0, tmax=5000.0, n=40):
        import numpy as np
        Ts = np.linspace(tmin, tmax, n)
        cp = [nasa7_cp_R(low if t <= tmid else high, float(t)) for t in Ts]
        return Ts.tolist(), cp

    def test_recovers_continuous_polynomial(self):
        T, cp = self._sample(_N2_LOW, _N2_HIGH, 1000.0)
        h_ref = nasa7_H_RT(_N2_LOW, T_REF)
        s_ref = nasa7_S_R(_N2_LOW, T_REF)
        fit = fit_nasa7(T, cp, h_ref, s_ref, 300.0, 1000.0, 5000.0)
        assert fit.cp_rms_error < 1e-4
        # continuity by construction
        assert fit.cp_jump_rel < 1e-8
        assert fit.h_jump_rel < 1e-8
        assert fit.s_jump_rel < 1e-8

    def test_reproduces_reference_h298_s298(self):
        T, cp = self._sample(_O2_LOW, _O2_HIGH, 1000.0)
        h_ref = nasa7_H_RT(_O2_LOW, T_REF)
        s_ref = nasa7_S_R(_O2_LOW, T_REF)
        fit = fit_nasa7(T, cp, h_ref, s_ref, 300.0, 1000.0, 5000.0)
        # H298/S298 from the fitted low polynomial must match the references
        assert nasa7_H_RT(fit.coeffs_low, T_REF) == pytest.approx(h_ref, rel=1e-9)
        assert nasa7_S_R(fit.coeffs_low, T_REF) == pytest.approx(s_ref, rel=1e-9)

    def test_h298_kj_mol_reasonable_for_o2(self):
        # O2 is an element in its standard state → Hf(298) ~ 0 kJ/mol
        T, cp = self._sample(_O2_LOW, _O2_HIGH, 1000.0)
        fit = fit_nasa7(T, cp, nasa7_H_RT(_O2_LOW, T_REF),
                        nasa7_S_R(_O2_LOW, T_REF), 300.0, 1000.0, 5000.0)
        assert abs(fit.h298_kJ_mol) < 1.0

    def test_requires_points_on_both_sides(self):
        # all points below Tmid
        import numpy as np
        Ts = np.linspace(300.0, 900.0, 10)
        cp = [nasa7_cp_R(_N2_LOW, float(t)) for t in Ts]
        with pytest.raises(ValueError):
            fit_nasa7(Ts.tolist(), cp, 0.0, 20.0, 300.0, 1000.0, 5000.0)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            fit_nasa7([300.0, 1200.0], [3.5], 0.0, 20.0, 300.0, 1000.0, 5000.0)


# ---------------------------------------------------------------------------
# refit_from_coeffs — repair
# ---------------------------------------------------------------------------

class TestRefitFromCoeffs:
    def test_good_polynomial_unchanged(self):
        fit = refit_from_coeffs(_N2_LOW, _N2_HIGH, 300.0, 1000.0, 5000.0)
        # Cp curve reproduced across the whole range
        for T in (300.0, 700.0, 1000.0, 2000.0, 5000.0):
            src = _N2_LOW if T <= 1000.0 else _N2_HIGH
            out = fit.coeffs_low if T <= 1000.0 else fit.coeffs_high
            assert nasa7_cp_R(out, T) == pytest.approx(nasa7_cp_R(src, T), abs=1e-4)

    def test_repairs_discontinuity(self):
        bad_high = list(_N2_HIGH)
        bad_high[0] *= 1.05
        assert not is_consistent(_N2_LOW, bad_high, 1000.0)
        fit = refit_from_coeffs(_N2_LOW, bad_high, 300.0, 1000.0, 5000.0)
        assert fit.cp_jump_rel < 1e-8
        assert fit.h_jump_rel < 1e-8
        assert fit.s_jump_rel < 1e-8
        assert is_consistent(fit.coeffs_low, fit.coeffs_high, 1000.0)

    def test_repair_preserves_h298(self):
        bad_high = list(_N2_HIGH)
        bad_high[0] *= 1.10
        h298_before = nasa7_H_RT(_N2_LOW, T_REF) * R_GAS * T_REF / 1000.0
        fit = refit_from_coeffs(_N2_LOW, bad_high, 300.0, 1000.0, 5000.0)
        assert fit.h298_kJ_mol == pytest.approx(h298_before, abs=1e-6)

    def test_source_and_notes(self):
        bad_high = list(_N2_HIGH)
        bad_high[0] *= 1.05
        fit = refit_from_coeffs(_N2_LOW, bad_high, 300.0, 1000.0, 5000.0)
        assert fit.source == "repair"
        assert any("original Tmid jumps" in n for n in fit.notes)

    def test_short_coeffs_raise(self):
        with pytest.raises(ValueError):
            refit_from_coeffs([1, 2, 3], _N2_HIGH, 300.0, 1000.0, 5000.0)

    def test_fit_to_dict_roundtrip(self):
        fit = refit_from_coeffs(_N2_LOW, _N2_HIGH, 300.0, 1000.0, 5000.0)
        d = fit.to_dict()
        assert len(d["coeffs_low"]) == 7
        assert len(d["coeffs_high"]) == 7
        assert d["source"] == "repair"
        assert json.dumps(d)   # serialisable


# ---------------------------------------------------------------------------
# ThermoRepairer — DB integration
# ---------------------------------------------------------------------------

def _make_db(path: str) -> sqlite3.Connection:
    from rmip.database.schema import init_db
    return init_db(path)


def _insert_mech(conn) -> int:
    from rmip.database.schema import insert_mechanism
    mid = insert_mechanism(conn, name="M", format="chemkin")
    conn.commit()
    return mid


def _insert_species(conn, mech_id, name, low, high,
                    tmin=300.0, tmid=1000.0, tmax=5000.0) -> int:
    from rmip.database.schema import insert_species
    sid = insert_species(
        conn, mechanism_id=mech_id, name=name, thermo_type="nasa7",
        tmin_K=tmin, tmid_K=tmid, tmax_K=tmax,
        coeffs_low_json=json.dumps(low), coeffs_high_json=json.dumps(high),
    )
    conn.commit()
    return sid


class TestThermoRepairer:
    def _db(self, tmp):
        return str(Path(tmp) / "t.db")

    def test_scan_reports_all_nasa7_species(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._db(tmp)
            conn = _make_db(db)
            mid = _insert_mech(conn)
            _insert_species(conn, mid, "N2", _N2_LOW, _N2_HIGH)
            _insert_species(conn, mid, "O2", _O2_LOW, _O2_HIGH)
            conn.close()

            rep = ThermoRepairer(db)
            scan = rep.scan(mid)
            rep.close()
            assert {s.name for s in scan} == {"N2", "O2"}
            assert all(s.worst_jump_before < 1e-6 for s in scan)

    def test_inconsistent_filters(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._db(tmp)
            conn = _make_db(db)
            mid = _insert_mech(conn)
            bad_high = list(_N2_HIGH); bad_high[0] *= 1.05
            _insert_species(conn, mid, "N2", _N2_LOW, _N2_HIGH)     # good
            _insert_species(conn, mid, "BAD", _N2_LOW, bad_high)    # bad
            conn.close()

            rep = ThermoRepairer(db, tol=0.01)
            inc = rep.inconsistent(mid)
            rep.close()
            assert [s.name for s in inc] == ["BAD"]

    def test_repair_persists_and_fixes(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._db(tmp)
            conn = _make_db(db)
            mid = _insert_mech(conn)
            bad_high = list(_N2_HIGH); bad_high[0] *= 1.08
            sid = _insert_species(conn, mid, "BAD", _N2_LOW, bad_high)
            conn.close()

            rep = ThermoRepairer(db, tol=0.01)
            results = rep.repair(mid, dry_run=False)
            rep.close()
            assert len(results) == 1
            r = results[0]
            assert r.repaired and r.persisted
            assert max(r.jumps_after.values()) < 1e-8

            # DB actually updated to a consistent polynomial
            conn2 = _make_db(db)
            row = conn2.execute(
                "SELECT coeffs_low_json, coeffs_high_json, tmid_K, "
                "thermo_comment FROM species WHERE id=?", (sid,)).fetchone()
            conn2.close()
            lo = json.loads(row["coeffs_low_json"])
            hi = json.loads(row["coeffs_high_json"])
            assert is_consistent(lo, hi, row["tmid_K"])
            assert "thermo-repair" in (row["thermo_comment"] or "")

    def test_dry_run_does_not_persist(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._db(tmp)
            conn = _make_db(db)
            mid = _insert_mech(conn)
            bad_high = list(_N2_HIGH); bad_high[0] *= 1.08
            sid = _insert_species(conn, mid, "BAD", _N2_LOW, bad_high)
            conn.close()

            rep = ThermoRepairer(db, tol=0.01)
            results = rep.repair(mid, dry_run=True)
            rep.close()
            assert results[0].repaired and not results[0].persisted

            conn2 = _make_db(db)
            row = conn2.execute(
                "SELECT coeffs_high_json FROM species WHERE id=?", (sid,)).fetchone()
            conn2.close()
            # unchanged — still the bad polynomial
            assert json.loads(row["coeffs_high_json"])[0] == pytest.approx(bad_high[0])

    def test_good_only_mech_repairs_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._db(tmp)
            conn = _make_db(db)
            mid = _insert_mech(conn)
            _insert_species(conn, mid, "N2", _N2_LOW, _N2_HIGH)
            conn.close()

            rep = ThermoRepairer(db, tol=0.01)
            results = rep.repair(mid, dry_run=False)
            rep.close()
            assert results == []
