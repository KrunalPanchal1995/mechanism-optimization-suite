"""
rmip/tests/test_b2bdc.py
========================
Tests for Bound-to-Bound Data Collaboration (Future.md pt. 5):
``rmip/analysis/b2bdc.py``. Pure numpy/scipy — always runs (no venv/cantera).

Run with:
    python -m pytest src/rmip/tests/test_b2bdc.py -v
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from rmip.analysis.b2bdc import (
    B2BDCConstraint,
    B2BDCDataset,
    insert_b2bdc_run,
    load_constraint,
)
from rmip.analysis.bayesian_calibration import PCEPoly
from rmip.types import B2BDCResult


def _lin(coeffs, d=None):
    """Linear PCEPoly f = Σ coeffs[i]·x_i (one exponent row per dim)."""
    d = d or len(coeffs)
    exps = [[1 if j == i else 0 for j in range(d)] for i in range(d)]
    return PCEPoly(exps, list(coeffs))


# ---------------------------------------------------------------------------
# Constraint
# ---------------------------------------------------------------------------

class TestConstraint:
    def test_residual_and_margin(self):
        c = B2BDCConstraint("m", _lin([1.0]), lower=-1.0, upper=1.0)  # mid 0, half 1
        assert c.residual(np.array([0.5])) == pytest.approx(0.5)
        assert c.margin(np.array([0.5])) == pytest.approx(0.5)
        assert c.margin(np.array([1.5])) == pytest.approx(-0.5)   # outside

    def test_bad_bounds_raise(self):
        with pytest.raises(ValueError):
            B2BDCConstraint("m", _lin([1.0]), lower=1.0, upper=1.0)


# ---------------------------------------------------------------------------
# Consistency measure
# ---------------------------------------------------------------------------

class TestConsistency:
    def test_single_constraint_measure_one(self):
        # M=x, bound [-1,1], box [-2,2] → C = max(1-|x|) = 1 at x=0
        ds = B2BDCDataset([B2BDCConstraint("m", _lin([1.0]), -1.0, 1.0)],
                          ["r0"], [(-2.0, 2.0)])
        C, x, margins = ds.consistency(n_starts=10, seed=1)
        assert C == pytest.approx(1.0, abs=1e-3)
        assert abs(x[0]) < 1e-2

    def test_overlapping_bands(self):
        # M=x, bands [0,1] and [0.5,1.5] overlap [0.5,1]; C=0.5 at x=0.75
        ds = B2BDCDataset(
            [B2BDCConstraint("a", _lin([1.0]), 0.0, 1.0),
             B2BDCConstraint("b", _lin([1.0]), 0.5, 1.5)],
            ["r0"], [(-3.0, 3.0)])
        C, x, _ = ds.consistency(n_starts=12, seed=2)
        assert C == pytest.approx(0.5, abs=0.02)
        assert x[0] == pytest.approx(0.75, abs=0.02)

    def test_inconsistent_negative(self):
        # M=2x; [1.5,1.9] needs x≈0.85, [-1.9,-1.5] needs x≈-0.85 → conflict
        ds = B2BDCDataset(
            [B2BDCConstraint("a", _lin([2.0]), 1.5, 1.9),
             B2BDCConstraint("b", _lin([2.0]), -1.9, -1.5)],
            ["r0"], [(-1.0, 1.0)])
        C, _, _ = ds.consistency(n_starts=12, seed=3)
        assert C < 0.0


# ---------------------------------------------------------------------------
# Parameter bounds & prediction over the feasible set
# ---------------------------------------------------------------------------

class TestFeasibleSetBounds:
    def test_parameter_bounds_recover_interval(self):
        # 2 params; c0: x0 ∈ [0.2,0.6] (M=x0); box [-1,1]^2.
        # feasible set: x0 ∈ [0.2,0.6], x1 free in [-1,1]
        ds = B2BDCDataset(
            [B2BDCConstraint("c0", _lin([1.0, 0.0]), 0.2, 0.6)],
            ["r0", "r1"], [(-1.0, 1.0), (-1.0, 1.0)])
        pb = ds.parameter_bounds(n_starts=8, seed=1)
        assert pb["r0"][0] == pytest.approx(0.2, abs=0.02)
        assert pb["r0"][1] == pytest.approx(0.6, abs=0.02)
        assert pb["r1"][0] == pytest.approx(-1.0, abs=0.02)
        assert pb["r1"][1] == pytest.approx(1.0, abs=0.02)

    def test_prediction_interval(self):
        # feasible x0∈[0.2,0.6]; predict M=x0 → range ≈ [0.2,0.6]
        ds = B2BDCDataset(
            [B2BDCConstraint("c0", _lin([1.0]), 0.2, 0.6)],
            ["r0"], [(-1.0, 1.0)])
        pr = ds.predict(_lin([1.0]), name="p", n_starts=8, seed=1)
        assert pr["lower"] == pytest.approx(0.2, abs=0.02)
        assert pr["upper"] == pytest.approx(0.6, abs=0.02)


# ---------------------------------------------------------------------------
# analyze() one-shot
# ---------------------------------------------------------------------------

class TestAnalyze:
    def test_consistent_full_report(self):
        ds = B2BDCDataset(
            [B2BDCConstraint("M1", _lin([2.0, 0.5]), 0.3, 0.9),
             B2BDCConstraint("M2", _lin([1.0, -1.0]), -0.4, 0.2)],
            ["r0", "r1"], [(-1.0, 1.0), (-1.0, 1.0)])
        res = ds.analyze(predictions=[("x0+x1", _lin([1.0, 1.0]))],
                         n_starts=16, seed=1)
        assert isinstance(res, B2BDCResult)
        assert res.consistent and res.consistency_measure > 0
        assert set(res.parameter_bounds) == {"r0", "r1"}
        # posterior width < prior width (2.0) for both
        for name in ("r0", "r1"):
            lo, hi = res.parameter_bounds[name]
            assert (hi - lo) < 2.0
            assert 0.0 <= res.parameter_reduction[name] <= 1.0
        assert res.predictions and res.predictions[0]["name"] == "x0+x1"
        assert res.predictions[0]["lower"] <= res.predictions[0]["upper"]

    def test_inconsistent_skips_bounds(self):
        ds = B2BDCDataset(
            [B2BDCConstraint("a", _lin([2.0]), 1.5, 1.9),
             B2BDCConstraint("b", _lin([2.0]), -1.9, -1.5)],
            ["r0"], [(-1.0, 1.0)])
        res = ds.analyze(n_starts=12, seed=2)
        assert not res.consistent
        assert res.parameter_bounds == {}
        assert res.notes and "inconsistent" in res.notes[0].lower()

    def test_box_from_factor(self):
        box = B2BDCDataset.box_from_factor(np.e, 3)
        assert len(box) == 3
        assert box[0][0] == pytest.approx(-1.0)
        assert box[0][1] == pytest.approx(1.0)

    def test_dim_mismatch_raises(self):
        with pytest.raises(ValueError):
            B2BDCDataset([B2BDCConstraint("m", _lin([1.0, 0.0]), 0, 1)],
                         ["only_one"], [(-1, 1)])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            B2BDCDataset([], ["r0"], [(-1, 1)])


# ---------------------------------------------------------------------------
# DB integration
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
    def test_load_constraint_and_persist(self):
        from rmip.database.schema import init_db, insert_mechanism
        from rmip.database.migrations import _table_exists
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "b.db")
            conn = init_db(db)
            mid = insert_mechanism(conn, name="m", format="cantera_yaml")
            conn.commit(); conn.close()

            r1 = _seed_surrogate_row(db, mid, [[1, 0], [0, 1]], [2.0, 0.5],
                                     ["R1", "R2"])
            c = load_constraint(db, r1, lower=0.3, upper=0.9, name="exp1")
            assert isinstance(c, B2BDCConstraint)
            assert c.surrogate.dim == 2

            ds = B2BDCDataset([c], ["R1", "R2"],
                              B2BDCDataset.box_from_factor(2.0, 2))
            res = ds.analyze(n_starts=10, seed=1, mechanism_id=mid)
            run_id = insert_b2bdc_run(db, res, [r1])
            assert run_id is not None

            conn = sqlite3.connect(db)
            assert _table_exists(conn, "b2bdc_runs")
            row = conn.execute(
                "SELECT n_constraints, consistent, consistency_measure "
                "FROM b2bdc_runs WHERE id=?", (run_id,)).fetchone()
            conn.close()
            assert row[0] == 1
