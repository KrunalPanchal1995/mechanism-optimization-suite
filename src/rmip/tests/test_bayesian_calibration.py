"""
rmip/tests/test_bayesian_calibration.py
=======================================
Tests for MUM-PCE / Bayesian calibration (Future.md pt. 6):
``rmip/analysis/bayesian_calibration.py`` + the emcee ``mcmc`` backend command.

The MUM-PCE core (PCEPoly + MAP + Laplace covariance) is pure numpy/scipy and
always runs. The emcee full-posterior test needs the Phase-5 venv and is skipped
when it is absent.

Run with:
    python -m pytest src/rmip/tests/test_bayesian_calibration.py -v
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from rmip.analysis.bayesian_calibration import (
    CalibrationTarget,
    MUMPCECalibrator,
    PCEPoly,
    insert_calibration_run,
    load_surrogate_target,
    surrogate_parameter_names,
)
from rmip.types import CalibrationResult

_VENV = "/home/krithika/.venvs/chaospy_gpc/bin/python3"
venv_required = pytest.mark.skipif(
    not os.path.exists(_VENV), reason="chaospy/emcee venv not present")


# ---------------------------------------------------------------------------
# PCEPoly
# ---------------------------------------------------------------------------

class TestPCEPoly:
    def test_eval_matches_monomial(self):
        # f = 2*x0 + 0.5*x1^2 + 3*x0*x1
        p = PCEPoly([[1, 0], [0, 2], [1, 1]], [2.0, 0.5, 3.0])
        x = [0.7, -0.4]
        expect = 2 * 0.7 + 0.5 * 0.16 + 3 * 0.7 * -0.4
        assert p.eval(x) == pytest.approx(expect)

    def test_grad_matches_finite_difference(self):
        p = PCEPoly([[1, 0], [0, 2], [1, 1]], [2.0, 0.5, 3.0])
        x = np.array([0.7, -0.4])
        g = p.grad(x)
        h = 1e-6
        fd = np.array([
            (p.eval(x + np.array([h, 0])) - p.eval(x - np.array([h, 0]))) / (2 * h),
            (p.eval(x + np.array([0, h])) - p.eval(x - np.array([0, h]))) / (2 * h),
        ])
        assert np.allclose(g, fd, atol=1e-5)

    def test_grad_safe_at_zero(self):
        p = PCEPoly([[0, 0], [1, 0], [0, 2]], [1.0, 2.0, 0.5])
        g = p.grad(np.array([0.0, 0.0]))
        assert np.allclose(g, [2.0, 0.0])       # d/dx1(0.5 x1^2)=x1=0

    def test_dim(self):
        assert PCEPoly([[1, 0, 0]], [1.0]).dim == 3

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            PCEPoly([[1, 0], [0, 1]], [1.0])      # 2 exps, 1 coeff

    def test_from_surrogate_result(self):
        from rmip.types import SurrogateResult
        sr = SurrogateResult(
            mechanism_id=1, qoi="idt", order=1, parameter_names=["a"],
            n_samples=10, pce_exponents=[[0], [1]], pce_coefficients=[0.0, 2.0])
        p = PCEPoly.from_surrogate_result(sr)
        assert p.eval([3.0]) == pytest.approx(6.0)

    def test_from_surrogate_result_empty_raises(self):
        from rmip.types import SurrogateResult
        sr = SurrogateResult(mechanism_id=1, qoi="idt", order=1,
                             parameter_names=["a"], n_samples=10)
        with pytest.raises(ValueError):
            PCEPoly.from_surrogate_result(sr)


# ---------------------------------------------------------------------------
# MUM-PCE calibrator — closed-form linear-Gaussian checks
# ---------------------------------------------------------------------------

class TestMUMPCE:
    def test_1param_1target_closed_form(self):
        a, y, sig, sp = 3.0, 1.2, 0.5, 1.0
        t = CalibrationTarget("t", PCEPoly([[1]], [a]), y, sig)
        res = MUMPCECalibrator([t], ["r0"], prior_sigma=sp,
                               bounds_sigma=6.0).calibrate()
        Lam = a**2 / sig**2 + 1.0 / sp**2
        mu = (a * y / sig**2) / Lam
        sd = 1.0 / math.sqrt(Lam)
        assert res.posterior_mean[0] == pytest.approx(mu, abs=1e-4)
        assert res.posterior_sigma[0] == pytest.approx(sd, abs=1e-4)
        assert res.converged

    def test_2param_2target_closed_form(self):
        A = np.array([[2.0, 0.5], [0.3, 1.5]])
        yv = np.array([0.8, -0.6])
        sg = np.array([0.4, 0.6])
        spv = np.array([1.0, 0.8])
        targs = [CalibrationTarget(f"t{i}", PCEPoly([[1, 0], [0, 1]], A[i].tolist()),
                                   yv[i], sg[i]) for i in range(2)]
        res = MUMPCECalibrator(targs, ["r0", "r1"], prior_sigma=spv,
                               bounds_sigma=6.0).calibrate()
        prec = np.diag(1.0 / spv**2) + sum(
            np.outer(A[i], A[i]) / sg[i]**2 for i in range(2))
        cov = np.linalg.inv(prec)
        mu = cov @ sum(A[i] * yv[i] / sg[i]**2 for i in range(2))
        assert np.allclose(res.posterior_mean, mu, atol=1e-4)
        assert np.allclose(np.sqrt(np.diag(cov)), res.posterior_sigma, atol=1e-4)

    def test_uncertainty_reduction_in_range(self):
        t = CalibrationTarget("t", PCEPoly([[1]], [5.0]), 0.5, 0.2)
        res = MUMPCECalibrator([t], ["r0"], prior_sigma=1.0).calibrate()
        red = res.uncertainty_reduction[0]
        assert 0.0 <= red < 1.0
        # posterior must be tighter than prior for a well-observed param
        assert res.posterior_sigma[0] < res.prior_sigma[0]

    def test_prior_sigma_from_factor(self):
        ps = MUMPCECalibrator.prior_sigma_from_factor(math.e**2, 3)  # ln=2 → /2 =1
        assert np.allclose(ps, 1.0)

    def test_data_pulls_map_toward_observation(self):
        # single positive-slope target, observed > prior prediction (0) → MAP > 0
        t = CalibrationTarget("t", PCEPoly([[1]], [2.0]), observed=1.0, sigma=0.3)
        res = MUMPCECalibrator([t], ["r0"], prior_sigma=1.0,
                               bounds_sigma=6.0).calibrate()
        assert res.posterior_mean[0] > 0.0
        # MAP prediction is closer to the observation than the prior prediction
        assert abs(res.predicted_map[0] - 1.0) < abs(res.predicted_prior[0] - 1.0)

    def test_dim_mismatch_raises(self):
        t = CalibrationTarget("t", PCEPoly([[1, 0]], [1.0]), 0.0, 1.0)  # dim 2
        with pytest.raises(ValueError):
            MUMPCECalibrator([t], ["only_one"], prior_sigma=1.0)

    def test_bad_sigma_raises(self):
        t = CalibrationTarget("t", PCEPoly([[1]], [1.0]), 0.0, 0.0)
        with pytest.raises(ValueError):
            MUMPCECalibrator([t], ["r0"], prior_sigma=1.0)

    def test_empty_targets_raises(self):
        with pytest.raises(ValueError):
            MUMPCECalibrator([], ["r0"], prior_sigma=1.0)


# ---------------------------------------------------------------------------
# CalibrationResult dataclass
# ---------------------------------------------------------------------------

class TestCalibrationResult:
    def _make(self):
        t = CalibrationTarget("t", PCEPoly([[1]], [2.0]), 1.0, 0.3)
        return MUMPCECalibrator([t], ["r0"], prior_sigma=1.0).calibrate()

    def test_to_dict_serialisable(self):
        d = self._make().to_dict()
        assert d["n_targets"] == 1
        assert json.dumps(d)

    def test_ranked_uncertainty_reduction(self):
        A = np.array([[5.0, 0.1]])   # r0 well-constrained, r1 barely
        t = CalibrationTarget("t", PCEPoly([[1, 0], [0, 1]], A[0].tolist()), 0.5, 0.2)
        res = MUMPCECalibrator([t], ["r0", "r1"], prior_sigma=1.0).calibrate()
        ranked = res.ranked_uncertainty_reduction()
        assert ranked[0][0] == "r0"

    def test_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "c.json"
            self._make().save(str(p))
            assert json.loads(p.read_text())["n_targets"] == 1


# ---------------------------------------------------------------------------
# DB integration — insert a surrogate row, load it as a target, persist result
# ---------------------------------------------------------------------------

def _seed_surrogate_row(db, mech_id, exps, coefs, names, factor=2.0):
    from rmip.database.migrations import run_surrogate_migrations
    conn = sqlite3.connect(db)
    run_surrogate_migrations(conn)
    cur = conn.execute(
        """INSERT INTO surrogate_runs
           (mechanism_id, qoi, method, poly_order, n_parameters, n_samples,
            n_failed_samples, n_terms, r2_insample, log_transformed, factor_bound,
            parameter_names_json, pce_exponents_json, pce_coefficients_json,
            created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (mech_id, "idt", "pce", 1, len(names), 50, 0, len(coefs), 0.99, 1, factor,
         json.dumps(names), json.dumps(exps), json.dumps(coefs),
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return rid


class TestDBIntegration:
    def test_load_and_calibrate_and_persist(self):
        from rmip.database.schema import init_db, insert_mechanism
        from rmip.database.migrations import _table_exists
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "c.db")
            conn = init_db(db)
            mid = insert_mechanism(conn, name="m", format="cantera_yaml")
            conn.commit(); conn.close()

            # two surrogate rows (targets), 2 shared params
            r1 = _seed_surrogate_row(db, mid, [[1, 0], [0, 1]], [2.0, 0.5],
                                     ["R1", "R2"])
            r2 = _seed_surrogate_row(db, mid, [[1, 0], [0, 1]], [0.3, 1.5],
                                     ["R1", "R2"])

            names = surrogate_parameter_names(db, r1)
            assert names == ["R1", "R2"]

            targets = [
                load_surrogate_target(db, r1, observed=0.8, sigma=0.4),
                load_surrogate_target(db, r2, observed=-0.6, sigma=0.6),
            ]
            res = MUMPCECalibrator(targets, names, prior_sigma=1.0).calibrate()

            run_id = insert_calibration_run(db, res, mid, [r1, r2])
            assert run_id is not None

            conn = sqlite3.connect(db)
            assert _table_exists(conn, "calibration_runs")
            row = conn.execute(
                "SELECT n_targets, n_parameters, posterior_mean_json "
                "FROM calibration_runs WHERE id=?", (run_id,)).fetchone()
            conn.close()
            assert row[0] == 2 and row[1] == 2
            assert len(json.loads(row[2])) == 2

    def test_load_surrogate_without_pce_raises(self):
        from rmip.database.schema import init_db, insert_mechanism
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "c.db")
            conn = init_db(db)
            mid = insert_mechanism(conn, name="m", format="cantera_yaml")
            conn.commit(); conn.close()
            rid = _seed_surrogate_row(db, mid, [], [], ["R1"])   # empty PCE
            with pytest.raises(ValueError):
                load_surrogate_target(db, rid, 0.0, 1.0)


# ---------------------------------------------------------------------------
# emcee full posterior (venv) — must match the MAP for a linear-Gaussian model
# ---------------------------------------------------------------------------

@venv_required
class TestMCMC:
    def test_mcmc_matches_map(self):
        A = np.array([[2.0, 0.5], [0.3, 1.5]])
        yv = np.array([0.8, -0.6])
        sg = np.array([0.4, 0.6])
        targs = [CalibrationTarget(f"t{i}", PCEPoly([[1, 0], [0, 1]], A[i].tolist()),
                                   yv[i], sg[i]) for i in range(2)]
        cal = MUMPCECalibrator(targs, ["r0", "r1"], prior_sigma=1.0, bounds_sigma=6.0)
        res = cal.calibrate()
        with tempfile.TemporaryDirectory() as tmp:
            mc = cal.run_mcmc(tmp, n_steps=3000, n_burn=800, seed=7)
        assert mc["n_samples"] > 1000
        assert np.allclose(mc["mean"], res.posterior_mean, atol=0.05)
        assert np.allclose(mc["sigma"], res.posterior_sigma, rtol=0.25)
