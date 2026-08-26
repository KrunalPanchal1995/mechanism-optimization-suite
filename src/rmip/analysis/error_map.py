"""
rmip/analysis/error_map.py
==========================
FEATURE_ROADMAP #11a — skeletal-vs-detailed error mapping.

Reduction reports a single threshold; it never tells you *where* in the operating
envelope the reduced mechanism actually holds.  This maps a reduced/skeletal
mechanism's error against the detailed one across a (T, P, φ) grid, giving a
worst-case error surface and the cells where the skeletal model breaks — the
error atlas that a deployment (or the TDAC runtime, #11b) needs.

It reuses the existing simulation stack: ``MechanismBenchmarker.run_custom_conditions``
(batched IDT/LFS over arbitrary ``SimCondition``s) and the same relative/log error
definitions as ``MechanismBenchmarker._error_metrics``, with the **detailed**
mechanism playing the role of "experiment".

Usage
-----
>>> from rmip.analysis.error_map import ErrorMapper, parse_grid
>>> conds = parse_grid("T=800:1600:9,P=1:40:3,phi=0.5:1.5:3", fuel="H2:1.0")
>>> res = ErrorMapper("project.db").map_error(
...     detailed_id=1, skeletal_id=2, sim_type="IDT", conditions=conds)
>>> res.max_rel_err, res.frac_within_tol
"""
from __future__ import annotations

import itertools
import logging
import math
from typing import Dict, List, Optional, Sequence

from rmip.types import ErrorMapResult, SimCondition

log = logging.getLogger(__name__)

_VALUE_COL = {"IDT": "idt_s", "LFS": "lfs_m_s"}


# ---------------------------------------------------------------------------
# Grid parsing
# ---------------------------------------------------------------------------

def _parse_axis(spec: str) -> List[float]:
    """
    Parse one axis: ``a:b:n`` → n linspace points; ``a,b,c`` → explicit list;
    ``a`` → single point.
    """
    spec = spec.strip()
    if ":" in spec:
        lo, hi, n = spec.split(":")
        lo, hi, n = float(lo), float(hi), int(n)
        if n <= 1:
            return [lo]
        step = (hi - lo) / (n - 1)
        return [lo + i * step for i in range(n)]
    if "," in spec:
        return [float(x) for x in spec.split(",") if x.strip()]
    return [float(spec)]


def parse_grid(
    grid:     str,
    fuel:     str,
    oxidizer: str = "O2:1.0, N2:3.76",
    default_T: float = 1000.0,
    default_P: float = 1.0,
    default_phi: float = 1.0,
) -> List[SimCondition]:
    """
    Build the cartesian product of ``T``/``P``/``phi`` axes into a list of
    ``SimCondition``.  Axis syntax (comma-separated between axes):

        "T=800:1600:9,P=1:40:3,phi=0.5:1.5:3"     # linspace per axis
        "T=800,1000,1200,phi=1.0"                 # explicit lists / scalars

    Missing axes fall back to the ``default_*`` scalars.
    """
    axes: Dict[str, List[float]] = {}
    for tok in _split_axes(grid):
        if "=" not in tok:
            continue
        key, val = tok.split("=", 1)
        axes[key.strip().lower()] = _parse_axis(val)

    Ts = axes.get("t", [default_T])
    Ps = axes.get("p", [default_P])
    phis = axes.get("phi", [default_phi])

    return [
        SimCondition(T_K=T, P_atm=P, phi=phi, fuel=fuel, oxidizer=oxidizer)
        for T, P, phi in itertools.product(Ts, Ps, phis)
    ]


def _split_axes(grid: str) -> List[str]:
    """
    Split a grid spec into axis tokens on the commas that separate axes (i.e.
    commas that precede an ``axis=`` label), leaving in-axis value commas intact.
    """
    tokens: List[str] = []
    for chunk in grid.split(","):
        if "=" in chunk or not tokens:
            tokens.append(chunk)
        else:                                   # a value comma → glue to previous
            tokens[-1] = tokens[-1] + "," + chunk
    return tokens


# ---------------------------------------------------------------------------
# Error mapper
# ---------------------------------------------------------------------------

class ErrorMapper:
    """Map a skeletal mechanism's error vs a detailed one over a condition grid."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    def map_error(
        self,
        detailed_id: int,
        skeletal_id: int,
        sim_type:    str,
        conditions:  Sequence[SimCondition],
        tolerance:   float = 0.30,
        n_workers:   int = 1,
        save_db:     bool = True,
    ) -> ErrorMapResult:
        """
        Run both mechanisms over ``conditions`` and build the error atlas.

        Parameters
        ----------
        sim_type : "IDT" | "LFS"
        tolerance : float
            Relative-error threshold defining "trustworthy" cells (default 0.30).
        """
        from rmip.analysis.benchmarking import MechanismBenchmarker

        sim_type = sim_type.upper()
        if sim_type not in _VALUE_COL:
            raise ValueError(f"sim_type must be IDT or LFS, got {sim_type!r}")
        col = _VALUE_COL[sim_type]
        conditions = list(conditions)

        bench = MechanismBenchmarker(self._db_path)
        df = bench.run_custom_conditions(
            [detailed_id, skeletal_id], sim_type, conditions, n_workers=n_workers)

        det = df[df["mechanism_id"] == detailed_id].reset_index(drop=True)
        ske = df[df["mechanism_id"] == skeletal_id].reset_index(drop=True)

        cells: List[Dict] = []
        rel_errs: List[float] = []
        log_errs: List[float] = []
        n_within = 0
        for i in range(len(conditions)):
            d_val = det[col].iloc[i] if i < len(det) else None
            s_val = ske[col].iloc[i] if i < len(ske) else None
            c = conditions[i]
            both = (d_val is not None and s_val is not None
                    and _pos(d_val) and _pos(s_val))
            cell = {
                "T_K": c.T_K, "P_atm": c.P_atm, "phi": c.phi,
                "detailed": (float(d_val) if _pos(d_val) else None),
                "skeletal": (float(s_val) if _pos(s_val) else None),
                "rel_err": None, "log_err": None, "both_converged": bool(both),
            }
            if both:
                rel = abs(float(s_val) - float(d_val)) / float(d_val)
                lge = abs(math.log10(float(s_val) / float(d_val)))
                cell["rel_err"] = rel
                cell["log_err"] = lge
                rel_errs.append(rel)
                log_errs.append(lge)
                if rel <= tolerance:
                    n_within += 1
            cells.append(cell)

        n_eval = len(rel_errs)
        worst = None
        if rel_errs:
            wi = max(range(len(cells)),
                     key=lambda k: cells[k]["rel_err"] if cells[k]["rel_err"] is not None else -1)
            worst = cells[wi]

        result = ErrorMapResult(
            detailed_mech_id = detailed_id,
            skeletal_mech_id = skeletal_id,
            sim_type         = sim_type,
            fuel             = conditions[0].fuel if conditions else "",
            oxidizer         = conditions[0].oxidizer if conditions else "",
            tolerance        = tolerance,
            n_conditions     = len(conditions),
            n_evaluated      = n_eval,
            cells            = cells,
            max_rel_err      = (max(rel_errs) if rel_errs else None),
            mean_rel_err     = (sum(rel_errs) / n_eval if n_eval else None),
            max_log_err      = (max(log_errs) if log_errs else None),
            frac_within_tol  = (n_within / n_eval if n_eval else None),
            worst_cell       = worst,
        )
        if n_eval < len(conditions):
            result.notes.append(
                f"{len(conditions) - n_eval} of {len(conditions)} cells were not "
                f"evaluated (a mechanism did not converge / ignite there).")

        if save_db:
            result.run_id = _insert_error_map_run(self._db_path, result)

        log.info(
            "error-map %s: detailed=%d vs skeletal=%d over %d cells → "
            "max rel err %.3g, %.0f%% within tol %.2f",
            sim_type, detailed_id, skeletal_id, len(conditions),
            result.max_rel_err or float("nan"),
            100 * (result.frac_within_tol or 0.0), tolerance,
        )
        return result


def _pos(v) -> bool:
    try:
        return v is not None and float(v) > 0
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# DB helper
# ---------------------------------------------------------------------------

def _insert_error_map_run(db_path: str, result: ErrorMapResult) -> Optional[int]:
    import json
    import sqlite3
    from datetime import datetime, timezone
    from rmip.database.migrations import run_error_map_migrations, _table_exists

    with sqlite3.connect(db_path) as conn:
        if not _table_exists(conn, "error_map_runs"):
            run_error_map_migrations(conn)
        cur = conn.execute(
            """
            INSERT INTO error_map_runs
                (detailed_mech_id, skeletal_mech_id, sim_type, fuel, oxidizer,
                 tolerance, n_conditions, n_evaluated, max_rel_err, mean_rel_err,
                 max_log_err, frac_within_tol, cells_json, worst_cell_json,
                 notes_json, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                result.detailed_mech_id, result.skeletal_mech_id, result.sim_type,
                result.fuel, result.oxidizer, result.tolerance,
                result.n_conditions, result.n_evaluated, result.max_rel_err,
                result.mean_rel_err, result.max_log_err, result.frac_within_tol,
                json.dumps(result.cells), json.dumps(result.worst_cell),
                json.dumps(result.notes),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        return cur.lastrowid
