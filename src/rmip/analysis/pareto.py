"""
rmip/analysis/pareto.py
=======================
FEATURE_ROADMAP #9 — multi-objective (Pareto) optimization.

The existing optimizer collapses everything into a single weighted scalar (WMRE /
logRMSE). Real mechanism tuning trades objectives off against each other (IDT vs
LFS accuracy, high- vs low-temperature regime error, …). This computes the
**Pareto front** of non-dominated parameter sets with a self-contained NSGA-II
(non-dominated sort + crowding distance, numpy only — no new dependency).

Objectives are evaluated cheaply through stored gPC surrogates: objective *k* is
the residual ``|M_k(θ) − target_k|`` of a surrogate ``M_k`` (e.g. an IDT surrogate
at one condition vs its measured value). Parameters θ are reaction log-A
multipliers bounded by the prior box ``θ ∈ [−ln f, ln f]``.

Honest scope
------------
NSGA-II over surrogate-based objectives — fast and reproducible. Direct
simulator-in-the-loop objectives (IDT/LFS runners) are a drop-in extension of the
``objective_fn`` hook but cost real integrations per individual.

Usage
-----
>>> from rmip.analysis.pareto import ParetoOptimizer
>>> res = ParetoOptimizer("project.db").optimize(
...     objectives=[(idt_hi_run_id, 3.1), (idt_lo_run_id, 4.2)],
...     factor=2.0, n_gen=40, pop_size=40)
>>> len(res.front)
"""
from __future__ import annotations

import logging
import math
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from rmip.types import ParetoResult

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NSGA-II core (pure numpy, minimises all objectives)
# ---------------------------------------------------------------------------

def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """True if objective vector ``a`` Pareto-dominates ``b`` (all ≤, some <)."""
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(F: np.ndarray) -> Tuple[List[List[int]], np.ndarray]:
    """Return (fronts as index lists, rank per individual)."""
    n = F.shape[0]
    S: List[List[int]] = [[] for _ in range(n)]
    dom_count = np.zeros(n, dtype=int)
    rank = np.zeros(n, dtype=int)
    fronts: List[List[int]] = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(F[p], F[q]):
                S[p].append(q)
            elif dominates(F[q], F[p]):
                dom_count[p] += 1
        if dom_count[p] == 0:
            rank[p] = 0
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt: List[int] = []
        for p in fronts[i]:
            for q in S[p]:
                dom_count[q] -= 1
                if dom_count[q] == 0:
                    rank[q] = i + 1
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    fronts.pop()
    return fronts, rank


def crowding_distance(F: np.ndarray, front: Sequence[int]) -> np.ndarray:
    """Crowding distance for the individuals in ``front`` (aligned to front order)."""
    l = len(front)
    d = np.zeros(l)
    if l <= 2:
        d[:] = np.inf
        return d
    Ff = F[np.asarray(front)]
    for m in range(F.shape[1]):
        order = np.argsort(Ff[:, m])
        d[order[0]] = d[order[-1]] = np.inf
        rng = Ff[order[-1], m] - Ff[order[0], m]
        if rng == 0:
            continue
        for i in range(1, l - 1):
            d[order[i]] += (Ff[order[i + 1], m] - Ff[order[i - 1], m]) / rng
    return d


def _tournament(rank, cd_lookup, rng) -> int:
    a, b = rng.integers(0, len(rank), size=2)
    if rank[a] != rank[b]:
        return int(a if rank[a] < rank[b] else b)
    return int(a if cd_lookup[a] >= cd_lookup[b] else b)


def _sbx_crossover(p1, p2, lb, ub, rng, eta=15.0, prob=0.9):
    c1, c2 = p1.copy(), p2.copy()
    if rng.random() > prob:
        return c1, c2
    for i in range(len(p1)):
        if rng.random() > 0.5 or abs(p1[i] - p2[i]) < 1e-14:
            continue
        x1, x2 = min(p1[i], p2[i]), max(p1[i], p2[i])
        u = rng.random()
        beta = (2 * u) ** (1 / (eta + 1)) if u <= 0.5 else (1 / (2 * (1 - u))) ** (1 / (eta + 1))
        c1[i] = 0.5 * ((x1 + x2) - beta * (x2 - x1))
        c2[i] = 0.5 * ((x1 + x2) + beta * (x2 - x1))
    return np.clip(c1, lb, ub), np.clip(c2, lb, ub)


def _poly_mutation(x, lb, ub, rng, eta=20.0, prob=None):
    prob = prob if prob is not None else 1.0 / len(x)
    y = x.copy()
    for i in range(len(x)):
        if rng.random() > prob or ub[i] == lb[i]:
            continue
        u = rng.random()
        delta = (2 * u) ** (1 / (eta + 1)) - 1 if u < 0.5 else 1 - (2 * (1 - u)) ** (1 / (eta + 1))
        y[i] += delta * (ub[i] - lb[i])
    return np.clip(y, lb, ub)


def nsga2(
    objective_fn: Callable[[np.ndarray], np.ndarray],
    lb:           np.ndarray,
    ub:           np.ndarray,
    pop_size:     int = 40,
    n_gen:        int = 50,
    seed:         Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Minimise ``objective_fn`` (maps an (N, d) population to (N, m) objectives)
    with NSGA-II.  Returns ``(X_front, F_front)`` — the final non-dominated set.
    """
    rng = np.random.default_rng(seed)
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    d = lb.size

    X = rng.uniform(lb, ub, size=(pop_size, d))
    F = objective_fn(X)

    for _ in range(n_gen):
        fronts, rank = fast_non_dominated_sort(F)
        cd = np.zeros(len(X))
        for fr in fronts:
            cd[np.asarray(fr)] = crowding_distance(F, fr)

        # offspring
        Q = np.empty_like(X)
        for j in range(0, pop_size, 2):
            p1 = X[_tournament(rank, cd, rng)]
            p2 = X[_tournament(rank, cd, rng)]
            c1, c2 = _sbx_crossover(p1, p2, lb, ub, rng)
            Q[j] = _poly_mutation(c1, lb, ub, rng)
            if j + 1 < pop_size:
                Q[j + 1] = _poly_mutation(c2, lb, ub, rng)
        FQ = objective_fn(Q)

        # environmental selection from combined R = X ∪ Q
        R = np.vstack([X, Q])
        FR = np.vstack([F, FQ])
        fronts, _ = fast_non_dominated_sort(FR)
        new_idx: List[int] = []
        for fr in fronts:
            if len(new_idx) + len(fr) <= pop_size:
                new_idx.extend(fr)
            else:
                cdf = crowding_distance(FR, fr)
                order = sorted(range(len(fr)), key=lambda i: -cdf[i])
                need = pop_size - len(new_idx)
                new_idx.extend(fr[i] for i in order[:need])
                break
        idx = np.asarray(new_idx)
        X, F = R[idx], FR[idx]

    fronts, _ = fast_non_dominated_sort(F)
    front = np.asarray(fronts[0])
    return X[front], F[front]


def hypervolume_2d(F: np.ndarray, ref: Sequence[float]) -> float:
    """2-objective hypervolume dominated by the front ``F`` w.r.t. reference ``ref``."""
    pts = F[np.argsort(F[:, 0])]
    hv = 0.0
    prev_x = ref[0]
    # sweep from best x0 upward, but hypervolume is easiest sorted descending x1
    pts = pts[np.argsort(-pts[:, 1])]
    prev_y = ref[1]
    for x, y in pts:
        if x >= ref[0] or y >= ref[1]:
            continue
        hv += (ref[0] - x) * (prev_y - y)
        prev_y = y
    return float(max(hv, 0.0))


# ---------------------------------------------------------------------------
# Mechanism Pareto optimizer (surrogate objectives)
# ---------------------------------------------------------------------------

class ParetoOptimizer:
    """Multi-objective NSGA-II over reaction A-factors, objectives from surrogates."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    def optimize(
        self,
        objectives:  Sequence[Tuple[int, float]],
        factor:      Optional[float] = None,
        objective_names: Optional[Sequence[str]] = None,
        pop_size:    int = 40,
        n_gen:       int = 50,
        seed:        Optional[int] = None,
        save_db:     bool = True,
    ) -> ParetoResult:
        """
        Parameters
        ----------
        objectives : sequence[(surrogate_run_id, target_value)]
            Each objective minimises ``|M_k(θ) − target_value|`` for the stored
            surrogate ``M_k``.
        factor : float, optional
            Prior A-factor box f → θ ∈ [−ln f, ln f]. Default: first surrogate's
            stored factor_bound.
        """
        import sqlite3
        from rmip.analysis.bayesian_calibration import (
            load_surrogate_target, surrogate_parameter_names,
        )

        run_ids = [o[0] for o in objectives]
        targets = [float(o[1]) for o in objectives]
        param_names = surrogate_parameter_names(self._db_path, run_ids[0])
        d = len(param_names)

        polys = [load_surrogate_target(self._db_path, rid, 0.0, 1.0).surrogate
                 for rid in run_ids]

        conn = sqlite3.connect(self._db_path)
        row = conn.execute("SELECT factor_bound, mechanism_id FROM surrogate_runs "
                           "WHERE id=?", (run_ids[0],)).fetchone()
        conn.close()
        if factor is None:
            factor = (row[0] if row and row[0] else 2.0)
        mech_id = row[1] if row else None
        bound = math.log(factor)
        lb = np.full(d, -bound)
        ub = np.full(d, bound)

        def objective_fn(X: np.ndarray) -> np.ndarray:
            out = np.empty((X.shape[0], len(polys)))
            for i, x in enumerate(X):
                for k, poly in enumerate(polys):
                    out[i, k] = abs(poly.eval(x) - targets[k])
            return out

        Xf, Ff = nsga2(objective_fn, lb, ub, pop_size=pop_size, n_gen=n_gen, seed=seed)

        # sort the front by the first objective for a stable, readable ordering
        order = np.argsort(Ff[:, 0])
        Xf, Ff = Xf[order], Ff[order]
        obj_names = (list(objective_names) if objective_names
                     else [f"surrogate_{rid}" for rid in run_ids])
        front = [{"params": Xf[i].tolist(), "objectives": Ff[i].tolist()}
                 for i in range(len(Xf))]

        hv = None
        if Ff.shape[1] == 2 and len(Ff):
            ref = (float(Ff[:, 0].max() * 1.1 + 1e-9), float(Ff[:, 1].max() * 1.1 + 1e-9))
            hv = hypervolume_2d(Ff, ref)

        result = ParetoResult(
            mechanism_id    = mech_id,
            parameter_names = param_names,
            objective_names = obj_names,
            objective_run_ids = run_ids,
            objective_targets = targets,
            n_generations   = n_gen,
            pop_size        = pop_size,
            n_front         = len(front),
            front           = front,
            hypervolume     = hv,
        )
        result.notes.append(
            "NSGA-II over surrogate-residual objectives; simulator-in-the-loop "
            "objectives are a drop-in extension of the objective hook.")

        if save_db:
            result.run_id = _insert_pareto_run(self._db_path, result)

        log.info("Pareto (NSGA-II): %d objectives, %d generations → "
                 "%d non-dominated solutions", len(polys), n_gen, len(front))
        return result


def _insert_pareto_run(db_path: str, result: ParetoResult) -> Optional[int]:
    import json
    import sqlite3
    from datetime import datetime, timezone
    from rmip.database.migrations import run_pareto_migrations, _table_exists

    with sqlite3.connect(db_path) as conn:
        if not _table_exists(conn, "pareto_runs"):
            run_pareto_migrations(conn)
        cur = conn.execute(
            """
            INSERT INTO pareto_runs
                (mechanism_id, n_objectives, n_generations, pop_size, n_front,
                 hypervolume, parameter_names_json, objective_names_json,
                 objective_run_ids_json, objective_targets_json, front_json,
                 notes_json, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                result.mechanism_id, len(result.objective_names),
                result.n_generations, result.pop_size, result.n_front,
                result.hypervolume, json.dumps(result.parameter_names),
                json.dumps(result.objective_names),
                json.dumps(result.objective_run_ids),
                json.dumps(result.objective_targets), json.dumps(result.front),
                json.dumps(result.notes),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        return cur.lastrowid
