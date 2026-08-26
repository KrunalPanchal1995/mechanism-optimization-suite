"""
rmip/tests/test_multi_condition_reduction.py
=============================================
Tests for ``rmip.reduction.multi_condition.reduce_multi_condition``
(FEATURE_GAPS_FROM_MONOLITH.md ranked candidate #5 — the union-reduce
orchestration part; the monolith's separate 3D diverse-subset
farthest-point-sampling confirm dialog remains a distinct, still-open gap
and is NOT covered here).

Uses Cantera's built-in h2o2.yaml (10 species / 29 reactions) parsed into a
temp RMIP DB — real Cantera rate evaluation (not the unweighted fallback)
is required so two conditions at genuinely different T/P/phi actually
produce different DRG-retained species subsets at some ε (verified
empirically while building this test: T=900K/P=1atm/phi=0.5 vs.
T=2000K/P=10atm/phi=2.0 diverge on H2O2 retention for ε in [0.5, 0.7]
with eps step 0.05).

Run with:
    python -m pytest src/rmip/tests/test_multi_condition_reduction.py -v
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

ct = pytest.importorskip("cantera")

_H2O2_YAML = str(Path(ct.__path__[0]) / "data" / "h2o2.yaml")

# Wide enough / fine enough to land squarely on the empirically-verified
# divergence band (eps in [0.5, 0.7]) between the two conditions below.
_EPS_RANGE = [round(i * 0.05, 6) for i in range(41)]
_TARGET = ["H2", "O2", "H2O"]


def _parse_h2o2(db_path: str) -> int:
    from rmip.config import RMIPConfig
    from rmip.core.parser import MechanismParser
    parser = MechanismParser(RMIPConfig(db_path=db_path))
    mech_id = parser.parse(mech_path=_H2O2_YAML)
    parser.conn.close()
    return mech_id


def _build_graph(db_path: str, mech_id: int):
    from rmip.core.graph_builder import GraphBuilder
    return GraphBuilder(db_path).build_graph(
        mech_id, include_reverse_edges=False, include_spring=False)


def _two_conditions():
    from rmip.types import SimCondition
    c1 = SimCondition(T_K=900.0, P_atm=1.0, phi=0.5,
                       fuel="H2:1.0", oxidizer="O2:1.0, AR:3.76")
    c2 = SimCondition(T_K=2000.0, P_atm=10.0, phi=2.0,
                       fuel="H2:1.0", oxidizer="O2:1.0, AR:3.76")
    return c1, c2


# ---------------------------------------------------------------------------
# Exact-union correctness
# ---------------------------------------------------------------------------

def test_union_equals_exact_union_of_individual_results(tmp_path):
    from rmip.reduction.drg import DRGReducer
    from rmip.reduction.multi_condition import reduce_multi_condition

    db_path = str(tmp_path / "h2o2.db")
    mech_id = _parse_h2o2(db_path)
    G = _build_graph(db_path, mech_id)
    c1, c2 = _two_conditions()

    result = reduce_multi_condition(
        mech_id=mech_id, db_path=db_path, G=G, target_species=_TARGET,
        conditions=[c1, c2], method="DRG", epsilon_range=_EPS_RANGE,
        mech_path=_H2O2_YAML, save_db=False,
    )

    r1 = DRGReducer(db_path).reduce(
        mech_id=mech_id, G=G, target_species=_TARGET, epsilon_range=_EPS_RANGE,
        condition=c1, mech_path=_H2O2_YAML, save_db=False)
    r2 = DRGReducer(db_path).reduce(
        mech_id=mech_id, G=G, target_species=_TARGET, epsilon_range=_EPS_RANGE,
        condition=c2, mech_path=_H2O2_YAML, save_db=False)

    assert len(result.steps) == len(_EPS_RANGE)

    n_divergent = 0
    for i, eps in enumerate(_EPS_RANGE):
        step, s1, s2 = result.steps[i], r1.steps[i], r2.steps[i]
        expected_sp = set(s1.retained_species) | set(s2.retained_species)
        expected_rx = set(s1.retained_reactions) | set(s2.retained_reactions)
        # Exact set equality, not just superset.
        assert set(step.retained_species) == expected_sp, f"eps={eps}"
        assert set(step.retained_reactions) == expected_rx, f"eps={eps}"
        assert step.n_species == len(expected_sp)
        assert step.n_reactions == len(expected_rx)
        # removed = complement of the union = intersection of the two removed sets
        assert set(step.removed_species) == (
            set(s1.removed_species) & set(s2.removed_species))
        assert set(step.removed_reactions) == (
            set(s1.removed_reactions) & set(s2.removed_reactions))
        if set(s1.retained_species) != set(s2.retained_species):
            n_divergent += 1

    # Sanity: the two conditions must actually have disagreed somewhere,
    # otherwise this test would pass trivially even with a buggy union.
    assert n_divergent > 0, "test fixture did not exercise genuine divergence"


# ---------------------------------------------------------------------------
# Degenerate single-condition case: no drift vs. calling .reduce() directly
# ---------------------------------------------------------------------------

def test_single_condition_matches_direct_reduce_call(tmp_path):
    from rmip.reduction.drg import DRGReducer
    from rmip.reduction.multi_condition import reduce_multi_condition

    db_path = str(tmp_path / "h2o2.db")
    mech_id = _parse_h2o2(db_path)
    G = _build_graph(db_path, mech_id)
    c1, _ = _two_conditions()

    multi_result = reduce_multi_condition(
        mech_id=mech_id, db_path=db_path, G=G, target_species=_TARGET,
        conditions=[c1], method="DRG", mech_path=_H2O2_YAML, save_db=False,
    )
    direct_result = DRGReducer(db_path).reduce(
        mech_id=mech_id, G=G, target_species=_TARGET,
        condition=c1, mech_path=_H2O2_YAML, save_db=False,
    )

    assert len(multi_result.steps) == len(direct_result.steps)
    for ms, ds in zip(multi_result.steps, direct_result.steps):
        assert ms.threshold == ds.threshold
        assert ms.retained_species == ds.retained_species
        assert ms.retained_reactions == ds.retained_reactions
        assert ms.removed_species == ds.removed_species
        assert ms.removed_reactions == ds.removed_reactions
        assert ms.n_species == ds.n_species
        assert ms.n_reactions == ds.n_reactions


# ---------------------------------------------------------------------------
# Persistence: one reduction_runs row + one linked
# reduction_multi_condition_runs row
# ---------------------------------------------------------------------------

def test_save_db_inserts_run_and_linked_multi_condition_row(tmp_path):
    from rmip.database.migrations import run_phase5b_migrations
    from rmip.reduction.multi_condition import reduce_multi_condition

    db_path = str(tmp_path / "h2o2.db")
    mech_id = _parse_h2o2(db_path)
    G = _build_graph(db_path, mech_id)
    c1, c2 = _two_conditions()

    with sqlite3.connect(db_path) as conn:
        run_phase5b_migrations(conn)

    result = reduce_multi_condition(
        mech_id=mech_id, db_path=db_path, G=G, target_species=_TARGET,
        conditions=[c1, c2], method="DRG", epsilon_range=_EPS_RANGE,
        mech_path=_H2O2_YAML, save_db=True,
    )
    assert result.run_id is not None

    with sqlite3.connect(db_path) as conn:
        run_rows = conn.execute(
            "SELECT id, mechanism_id, method FROM reduction_runs"
        ).fetchall()
        mc_rows = conn.execute(
            "SELECT reduction_run_id, conditions_json FROM "
            "reduction_multi_condition_runs"
        ).fetchall()

    assert len(run_rows) == 1
    assert run_rows[0][0] == result.run_id
    assert run_rows[0][1] == mech_id
    assert run_rows[0][2] == "DRG"

    assert len(mc_rows) == 1
    assert mc_rows[0][0] == result.run_id
    import json as _json
    conditions_stored = _json.loads(mc_rows[0][1])
    assert len(conditions_stored) == 2
    assert conditions_stored[0]["T_K"] == 900.0
    assert conditions_stored[1]["T_K"] == 2000.0


def test_no_conditions_raises(tmp_path):
    from rmip.reduction.multi_condition import reduce_multi_condition
    db_path = str(tmp_path / "h2o2.db")
    mech_id = _parse_h2o2(db_path)
    G = _build_graph(db_path, mech_id)
    with pytest.raises(ValueError):
        reduce_multi_condition(
            mech_id=mech_id, db_path=db_path, G=G, target_species=_TARGET,
            conditions=[], save_db=False,
        )
