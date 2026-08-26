"""
rmip/reduction/multi_condition.py
==================================
Multi-condition union-reduce orchestration
(FEATURE_GAPS_FROM_MONOLITH.md ranked candidate #5 — restores the
monolith's ``_run_mechanism_reduction_multi_condition`` workflow): run the
SAME DRG/DRGEP/PFA/DRGPSA/DRGSA reducer once per experimental condition,
then union the retained species/reactions at each matching ε threshold, so
a reduced mechanism stays valid across every condition simulated rather
than just the single condition a plain ``.reduce()`` call was evaluated at.

This module does NOT reimplement any reduction algorithm — it dispatches to
the existing reducer classes (mirrors ``rmip.cli._cmd_reduce``'s own
method-to-class mapping) and only does the per-epsilon set union
afterwards. Persistence reuses a reducer instance's own ``_insert_run``
helper so exactly ONE ``reduction_runs`` row is written for the whole
union (not one per condition), plus a new
``reduction_multi_condition_runs`` row (see
``rmip.database.migrations.run_multi_condition_migrations``) linking that
run back to the list of conditions that were unioned.

Note on the removed_* complement: every per-condition ``ReductionStep``
already computes ``removed_* = all_{species,reactions} - retained_*``
against the SAME mechanism-wide universe (same ``mech_id``). So the
unioned removed set — species/reactions retained by NONE of the
conditions — is simply the INTERSECTION of the per-condition removed
sets; no separate DB re-query for "all species"/"all reactions" is
needed.

The monolith's separate 3D diverse-subset farthest-point-sampling confirm
dialog (picking WHICH conditions to run against) is a distinct, still-open
gap — this module only does the union-reduce math once conditions are
already given.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from rmip.types import ReductionResult, ReductionStep, SimCondition

log = logging.getLogger(__name__)


def _make_reducer(method: str, db_path: str, sa_json_path: Optional[str] = None):
    """Method -> reducer instance, mirroring rmip.cli._cmd_reduce's own dispatch."""
    if method == "DRG":
        from rmip.reduction.drg import DRGReducer
        return DRGReducer(db_path)
    if method == "DRGEP":
        from rmip.reduction.drgep import DRGEPReducer
        return DRGEPReducer(db_path)
    if method == "PFA":
        from rmip.reduction.pfa import PFAReducer
        return PFAReducer(db_path)
    if method == "DRGPSA":
        from rmip.reduction.drgpsa import DRGPSAReducer
        return DRGPSAReducer(db_path, sa_json_path=sa_json_path)
    if method == "DRGSA":
        from rmip.reduction.drgpsa import DRGSAReducer
        return DRGSAReducer(db_path, sa_json_path=sa_json_path)
    raise ValueError(f"Unknown reduction method: {method!r}")


def reduce_multi_condition(
    mech_id: int,
    db_path: str,
    G: Any,
    target_species: List[str],
    conditions: List[SimCondition],
    method: str = "DRG",
    epsilon_range: Optional[List[float]] = None,
    mech_path: Optional[str] = None,
    sa_json_path: Optional[str] = None,
    save_db: bool = True,
) -> ReductionResult:
    """
    Run ``method``'s reducer once per condition in ``conditions`` and union
    the retained species/reactions at each matching ε threshold.

    ``epsilon_range`` defaults exactly the way each reducer's own
    ``.reduce()`` does (51 steps in ``[0, 0.5]``) and is then passed
    explicitly into every per-condition call, so every condition's steps
    line up index-for-index at the SAME thresholds before unioning.

    A single-condition call (``len(conditions) == 1``) is degenerate with
    calling that reducer's ``.reduce(condition=conditions[0], ...)``
    directly — the retained/removed sets at every ε come out byte-for-byte
    identical, since the union of one set is itself.

    Returns
    -------
    ReductionResult
        ``steps`` holds the unioned per-ε retained/removed species and
        reactions. ``run_id`` is the single ``reduction_runs`` row id (if
        ``save_db``), not one per condition.
    """
    if not conditions:
        raise ValueError("reduce_multi_condition requires at least one condition")

    if epsilon_range is None:
        epsilon_range = [round(i * 0.01, 6) for i in range(51)]

    per_condition_results: List[ReductionResult] = []
    reducer = None
    for condition in conditions:
        reducer = _make_reducer(method, db_path, sa_json_path)
        result = reducer.reduce(
            mech_id=mech_id, G=G, target_species=target_species,
            epsilon_range=epsilon_range, condition=condition,
            mech_path=mech_path, save_db=False,
        )
        per_condition_results.append(result)

    n_steps = len(epsilon_range)
    for result in per_condition_results:
        if len(result.steps) != n_steps:
            raise RuntimeError(
                "Per-condition reduction produced a mismatched step count — "
                "cannot union at matching ε thresholds"
            )

    unioned_steps: List[ReductionStep] = []
    for i, eps in enumerate(epsilon_range):
        retained_sp: Set[str] = set()
        retained_rx: Set[str] = set()
        removed_sp_inter: Optional[Set[str]] = None
        removed_rx_inter: Optional[Set[str]] = None
        bfs_merged: Dict[str, int] = {}

        for result in per_condition_results:
            step = result.steps[i]
            retained_sp |= set(step.retained_species)
            retained_rx |= set(step.retained_reactions)

            rs = set(step.removed_species)
            rr = set(step.removed_reactions)
            removed_sp_inter = rs if removed_sp_inter is None else (removed_sp_inter & rs)
            removed_rx_inter = rr if removed_rx_inter is None else (removed_rx_inter & rr)

            if step.bfs_levels:
                for sp, lvl in step.bfs_levels.items():
                    if sp not in bfs_merged or lvl < bfs_merged[sp]:
                        bfs_merged[sp] = lvl

        unioned_steps.append(ReductionStep(
            threshold=eps,
            retained_species=sorted(retained_sp),
            retained_reactions=sorted(retained_rx),
            removed_species=sorted(removed_sp_inter or set()),
            removed_reactions=sorted(removed_rx_inter or set()),
            n_species=len(retained_sp),
            n_reactions=len(retained_rx),
            bfs_levels=bfs_merged or None,
        ))

    run_id = None
    if save_db:
        run_id = reducer._insert_run(mech_id, target_species, epsilon_range, unioned_steps)
        if run_id is not None:
            _insert_multi_condition_run(db_path, run_id, conditions)

    log.info(
        "%s multi-condition (%d conditions) reduction on mech_id=%d: %d ε steps, "
        "full=%d sp/%d rxn -> minimal=%d sp/%d rxn",
        method, len(conditions), mech_id, n_steps,
        unioned_steps[0].n_species, unioned_steps[0].n_reactions,
        unioned_steps[-1].n_species, unioned_steps[-1].n_reactions,
    )

    return ReductionResult(
        mechanism_id=mech_id,
        method=method,
        target_species=list(target_species),
        steps=unioned_steps,
        run_id=run_id,
    )


def _insert_multi_condition_run(
    db_path: str, reduction_run_id: int, conditions: List[SimCondition],
) -> Optional[int]:
    """Insert one ``reduction_multi_condition_runs`` row linking a persisted
    unioned reduction run back to the conditions it unioned."""
    from rmip.database.migrations import _table_exists, run_multi_condition_migrations

    with sqlite3.connect(db_path) as conn:
        if not _table_exists(conn, "reduction_multi_condition_runs"):
            run_multi_condition_migrations(conn)
        ts = datetime.now(timezone.utc).isoformat()
        cur = conn.execute(
            "INSERT INTO reduction_multi_condition_runs "
            "(reduction_run_id, conditions_json, created_at) VALUES (?, ?, ?)",
            (reduction_run_id, json.dumps([c.to_dict() for c in conditions]), ts),
        )
        conn.commit()
        return cur.lastrowid
