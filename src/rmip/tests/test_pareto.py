"""
rmip/tests/test_pareto.py
=========================
Tests for multi-objective Pareto optimization
(``rmip/analysis/pareto.py``, FEATURE_ROADMAP #9).

The NSGA-II core is checked on the classic ZDT1 benchmark (known convex front);
the mechanism optimizer is checked on synthetic surrogates that create a genuine
1-parameter trade-off (two targets for the same parameter).

Run with:
    python -m pytest src/rmip/tests/test_pareto.py -v
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pytest

from rmip.analysis.pareto import (
    ParetoOptimizer, dominates, fast_non_dominated_sort, crowding_distance, nsga2,
)
from rmip.types import ParetoResult


# ---------------------------------------------------------------------------
# NSGA-II core
# ---------------------------------------------------------------------------

def test_dominates():
    assert dominates(np.array([1.0, 2.0]), np.array([1.0, 3.0]))
    assert dominates(np.array([1.0, 2.0]), np.array([2.0, 3.0]))
    assert not dominates(np.array([1.0, 2.0]), np.array([1.0, 2.0]))   # equal
    assert not dominates(np.array([1.0, 3.0]), np.array([2.0, 2.0]))   # trade-off


def test_non_dominated_sort_layers():
    # front0: (1,4),(2,2),(4,1); front1: (3,3) dominated by (2,2)
    F = np.array([[1.0, 4.0], [2.0, 2.0], [4.0, 1.0], [3.0, 3.0]])
    fronts, rank = fast_non_dominated_sort(F)
    assert set(fronts[0]) == {0, 1, 2}
    assert fronts[1] == [3]
    assert rank[3] == 1


def test_crowding_distance_endpoints_infinite():
    F = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    d = crowding_distance(F, [0, 1, 2])
    assert np.isinf(d[0]) and np.isinf(d[2])
    assert np.isfinite(d[1])


def test_nsga2_recovers_zdt1_front():
    d = 5
    def zdt1(X):
        f1 = X[:, 0]
        g = 1.0 + 9.0 * X[:, 1:].sum(axis=1) / (d - 1)
        f2 = g * (1.0 - np.sqrt(f1 / g))
        return np.column_stack([f1, f2])

    Xf, Ff = nsga2(zdt1, np.zeros(d), np.ones(d), pop_size=40, n_gen=80, seed=7)

    # members are mutually non-dominated
    for i in range(len(Ff)):
        for j in range(len(Ff)):
            if i != j:
                assert not dominates(Ff[i], Ff[j])
    # spread along objective 1
    assert Ff[:, 0].max() - Ff[:, 0].min() > 0.5
    # converged near the true front f2 = 1 - sqrt(f1) (g→1)
    true_f2 = 1.0 - np.sqrt(np.clip(Ff[:, 0], 0, 1))
    assert np.median(Ff[:, 1] - true_f2) < 0.2


# ---------------------------------------------------------------------------
# Mechanism Pareto over synthetic surrogates
# ---------------------------------------------------------------------------

def _insert_surrogate(conn, exponents, coefficients, param_names, qoi,
                      factor_bound=2.718281828, mech_id=1):
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
def pareto_db(tmp_path):
    from rmip.database.schema import init_db
    from rmip.database.migrations import run_surrogate_migrations
    db_path = str(tmp_path / "pareto.db")
    conn = init_db(db_path)
    conn.execute("INSERT INTO mechanisms (name, format, loaded_at) VALUES (?,?,?)",
                 ("m", "cantera_yaml", datetime.now(timezone.utc).isoformat()))
    run_surrogate_migrations(conn)
    names = ["r0", "r1"]
    # both objectives depend on param 0 → conflicting targets create a trade-off
    s1 = _insert_surrogate(conn, [[1, 0]], [1.0], names, "expHi")   # M = theta0
    s2 = _insert_surrogate(conn, [[1, 0]], [1.0], names, "expLo")   # M = theta0
    conn.commit()
    conn.close()
    return db_path, s1, s2


def test_pareto_finds_tradeoff_front(pareto_db):
    db_path, s1, s2 = pareto_db
    # minimise |theta0 - 0.5| and |theta0 + 0.5| → front is theta0 ∈ [-0.5, 0.5]
    res = ParetoOptimizer(db_path).optimize(
        [(s1, 0.5), (s2, -0.5)], factor=2.718281828, n_gen=40, pop_size=40, seed=3,
        objective_names=["hi", "lo"], save_db=False)
    assert isinstance(res, ParetoResult)
    assert res.n_front >= 3
    F = np.array([s["objectives"] for s in res.front])
    # mutually non-dominated
    for i in range(len(F)):
        for j in range(len(F)):
            if i != j:
                assert not dominates(F[i], F[j])
    # on this trade-off obj1 + obj2 == |t0-0.5| + |t0+0.5| == 1 for t0 in [-0.5,0.5]
    sums = F.sum(axis=1)
    assert np.all(sums <= 1.0 + 1e-2)
    assert np.median(sums) == pytest.approx(1.0, abs=0.1)


def test_pareto_persists(pareto_db):
    db_path, s1, s2 = pareto_db
    res = ParetoOptimizer(db_path).optimize(
        [(s1, 0.5), (s2, -0.5)], factor=2.718281828, n_gen=20, pop_size=20,
        seed=1, save_db=True)
    assert res.run_id is not None
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT n_objectives, n_front FROM pareto_runs WHERE id=?",
                           (res.run_id,)).fetchone()
    assert row[0] == 2 and row[1] == res.n_front
