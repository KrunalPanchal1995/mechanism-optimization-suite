"""
rmip/tests/test_experiment_design.py
====================================
Tests for optimal experiment design / DoE
(``rmip/analysis/experiment_design.py``, FEATURE_ROADMAP #8).

Uses synthetic ``surrogate_runs`` rows with hand-set monomial PCEs (no chaospy
venv needed) so the information-gain selection can be checked against a known
answer: two orthogonal linear surrogates carry the most joint information, a
near-flat one the least.

Run with:
    python -m pytest src/rmip/tests/test_experiment_design.py -v
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import pytest

from rmip.analysis.experiment_design import ExperimentDesigner, _criterion_gain
from rmip.types import DoEResult
import numpy as np


def _insert_surrogate(conn, exponents, coefficients, param_names,
                      factor_bound=2.718281828, mech_id=1, qoi="idt"):
    """Insert a minimal surrogate_runs row carrying a monomial PCE."""
    cur = conn.execute(
        """INSERT INTO surrogate_runs
           (mechanism_id, qoi, method, poly_order, n_parameters, n_samples,
            n_failed_samples, n_terms, r2_insample, qoi_mean, qoi_variance,
            log_transformed, factor_bound, parameter_names_json,
            pce_exponents_json, pce_coefficients_json, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (mech_id, qoi, "pce", 1, len(param_names), 10, 0, len(coefficients),
         0.99, 0.0, 1.0, 1, factor_bound, json.dumps(param_names),
         json.dumps(exponents), json.dumps(coefficients),
         datetime.now(timezone.utc).isoformat()),
    )
    return cur.lastrowid


@pytest.fixture()
def doe_db(tmp_path):
    from rmip.database.schema import init_db
    from rmip.database.migrations import run_surrogate_migrations
    db_path = str(tmp_path / "doe.db")
    conn = init_db(db_path)
    conn.execute("INSERT INTO mechanisms (name, format, loaded_at) VALUES (?,?,?)",
                 ("m", "cantera_yaml", datetime.now(timezone.utc).isoformat()))
    run_surrogate_migrations(conn)
    names = ["rxnA", "rxnB"]
    # A: sensitive only to param 0; B: only to param 1; C: weakly to param 0
    a = _insert_surrogate(conn, [[1, 0]], [1.0], names, qoi="expA")
    b = _insert_surrogate(conn, [[0, 1]], [1.0], names, qoi="expB")
    c = _insert_surrogate(conn, [[1, 0]], [0.1], names, qoi="expC")
    conn.commit()
    conn.close()
    return db_path, {"A": a, "B": b, "C": c}


# ---------------------------------------------------------------------------
# criterion gain (pure)
# ---------------------------------------------------------------------------

def test_criterion_gain_monotone_in_information():
    Lam = np.eye(2)
    g = np.array([1.0, 0.0])
    F_small = np.outer(g, g) * 0.1
    F_big = np.outer(g, g) * 2.0
    assert _criterion_gain(Lam, F_big, "D") > _criterion_gain(Lam, F_small, "D")
    assert _criterion_gain(Lam, F_big, "A") > _criterion_gain(Lam, F_small, "A")


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------

def test_selects_orthogonal_informative_pair(doe_db):
    db_path, ids = doe_db
    res = ExperimentDesigner(db_path).design(
        [ids["A"], ids["B"], ids["C"]], n_select=2, criterion="D", save_db=False)
    assert isinstance(res, DoEResult)
    chosen = {s["run_id"] for s in res.selected}
    # A and B (orthogonal, strong) beat the weak C
    assert chosen == {ids["A"], ids["B"]}
    # selection shrinks the uncertainty
    assert res.trace_cov_post < res.trace_cov_prior
    assert res.logdet_post > res.logdet_prior


def test_ranking_puts_weak_candidate_last(doe_db):
    db_path, ids = doe_db
    res = ExperimentDesigner(db_path).design(
        [ids["A"], ids["B"], ids["C"]], criterion="D", save_db=False)
    ranking = [r["run_id"] for r in res.ranked_candidates]
    assert ranking[-1] == ids["C"]                 # weakest is ranked last
    assert res.ranked_candidates[0]["first_step_gain"] >= \
        res.ranked_candidates[-1]["first_step_gain"]


def test_a_and_d_criteria_both_run(doe_db):
    db_path, ids = doe_db
    for crit in ("A", "D"):
        res = ExperimentDesigner(db_path).design(
            [ids["A"], ids["B"], ids["C"]], n_select=2, criterion=crit, save_db=False)
        assert res.criterion == crit
        assert res.n_selected == 2
        assert all(u >= 0 for u in res.uncertainty_reduction)


def test_persists_and_saves(doe_db, tmp_path):
    db_path, ids = doe_db
    res = ExperimentDesigner(db_path).design(
        [ids["A"], ids["B"]], criterion="D", save_db=True)
    assert res.run_id is not None
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT n_candidates, criterion FROM doe_runs WHERE id=?",
                           (res.run_id,)).fetchone()
    assert row[0] == 2 and row[1] == "D"
    out = tmp_path / "doe.json"
    res.save(str(out))
    assert json.loads(out.read_text())["n_selected"] == 2
