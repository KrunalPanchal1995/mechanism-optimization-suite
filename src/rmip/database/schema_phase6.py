"""
rmip/database/schema_phase6.py
===============================
Phase-6 DDL additions for Future.md features 1–7:

Tables
------
  ntc_runs                 — one row per NTCDetector.detect() call (Future.md pt. 7)
  reverse_rate_flags        — one row per flagged (reaction, T) pair (Future.md pt. 1)
  mechanism_quality_scores — one row per QualityScorer.score() call (Future.md pt. 2+4)
  flux_runs                 — one row per flux analysis run, any solver (Future.md pt. 3)

All DDL uses CREATE TABLE IF NOT EXISTS and is safe to call on existing DBs.
Column-add / CHECK-widen migrations for existing tables (``simulation_runs``,
``exp_data_points``) live in ``migrations.run_phase6_migrations`` alongside the
savepoint-probe pattern used for earlier phases.

Usage
-----
    from rmip.database.schema_phase6 import upgrade_db_phase6
    upgrade_db_phase6(conn)
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_CREATE_NTC_RUNS = """
CREATE TABLE IF NOT EXISTS ntc_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    mechanism_id        INTEGER NOT NULL,
    fuel                TEXT    NOT NULL,
    phi                 REAL    NOT NULL,
    P_atm               REAL    NOT NULL,
    T_grid_json         TEXT    NOT NULL,   -- JSON list of T_K sweep points
    idt_s_json          TEXT    NOT NULL,   -- JSON list, same length (null = no ignition)
    has_ntc             INTEGER NOT NULL DEFAULT 0,
    T_NTC_onset         REAL,               -- K
    T_NTC_end           REAL,               -- K
    idt_ratio_ntc       REAL,
    n_htt_points        INTEGER NOT NULL DEFAULT 0,
    n_ltt_points        INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT    NOT NULL,
    FOREIGN KEY (mechanism_id) REFERENCES mechanisms(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_ntcruns_mech ON ntc_runs(mechanism_id);
"""

_CREATE_REVERSE_RATE_FLAGS = """
CREATE TABLE IF NOT EXISTS reverse_rate_flags (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    mechanism_id        INTEGER NOT NULL,
    canonical_key       TEXT    NOT NULL,
    formula_string      TEXT,
    T_K                 REAL    NOT NULL,
    k_fwd               REAL    NOT NULL,
    k_rev               REAL    NOT NULL,
    collision_limit     REAL    NOT NULL,
    exceedance_factor   REAL    NOT NULL,
    flagged             INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT    NOT NULL,
    FOREIGN KEY (mechanism_id) REFERENCES mechanisms(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_revflags_mech ON reverse_rate_flags(mechanism_id, flagged);
"""

_CREATE_MECHANISM_QUALITY_SCORES = """
CREATE TABLE IF NOT EXISTS mechanism_quality_scores (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    mechanism_id                INTEGER NOT NULL,
    smiles_resolution_pct       REAL    NOT NULL,
    thermo_consistency_score    REAL    NOT NULL,
    hf_agreement_score          REAL,
    ntc_coverage_score          REAL,
    rate_bounds_score           REAL    NOT NULL,
    composite_score             REAL    NOT NULL,
    n_reactions                 INTEGER NOT NULL,
    n_species                   INTEGER NOT NULL,
    n_reverse_rate_flags        INTEGER NOT NULL DEFAULT 0,
    details_json                TEXT,
    created_at                  TEXT    NOT NULL,
    FOREIGN KEY (mechanism_id) REFERENCES mechanisms(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_qscores_mech ON mechanism_quality_scores(mechanism_id);
"""

_CREATE_FLUX_RUNS = """
CREATE TABLE IF NOT EXISTS flux_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    mechanism_id        INTEGER NOT NULL,
    solver              TEXT    NOT NULL CHECK(solver IN ('cantera','CHEMKIN_PRO','FlameMaster')),
    T_K                 REAL    NOT NULL,
    P_atm               REAL    NOT NULL,
    phi                 REAL    NOT NULL,
    fuel_composition    TEXT    NOT NULL,
    element             TEXT    NOT NULL,    -- 'C' | 'H' | 'O'
    n_reactions         INTEGER NOT NULL DEFAULT 0,
    n_flux_edges        INTEGER NOT NULL DEFAULT 0,
    rop_json            TEXT,                -- JSON {canonical_key: rop}
    element_flux_json   TEXT,                -- JSON list of Sankey edges
    converged           INTEGER NOT NULL DEFAULT 1,
    cpu_time_s          REAL,
    error_msg           TEXT,
    created_at          TEXT    NOT NULL,
    FOREIGN KEY (mechanism_id) REFERENCES mechanisms(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_fluxruns_mech ON flux_runs(mechanism_id, solver);
"""


def upgrade_db_phase6(conn: sqlite3.Connection) -> None:
    """
    Add Phase-6 tables to an existing RMIP database.
    Idempotent — safe to call multiple times.
    """
    for ddl in (_CREATE_NTC_RUNS,
                _CREATE_REVERSE_RATE_FLAGS,
                _CREATE_MECHANISM_QUALITY_SCORES,
                _CREATE_FLUX_RUNS):
        conn.executescript(ddl)
    conn.commit()


# ---------------------------------------------------------------------------
# CRUD — ntc_runs
# ---------------------------------------------------------------------------

def insert_ntc_run(
    conn:            sqlite3.Connection,
    mechanism_id:    int,
    fuel:            str,
    phi:             float,
    P_atm:           float,
    T_grid_K:        List[float],
    idt_s:           List[Optional[float]],
    has_ntc:         bool,
    T_NTC_onset:     Optional[float],
    T_NTC_end:       Optional[float],
    idt_ratio_ntc:   Optional[float],
    n_htt_points:    int,
    n_ltt_points:    int,
) -> int:
    cur = conn.execute(
        """INSERT INTO ntc_runs
           (mechanism_id, fuel, phi, P_atm, T_grid_json, idt_s_json,
            has_ntc, T_NTC_onset, T_NTC_end, idt_ratio_ntc,
            n_htt_points, n_ltt_points, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (mechanism_id, fuel, phi, P_atm,
         json.dumps(T_grid_K), json.dumps(idt_s),
         int(has_ntc), T_NTC_onset, T_NTC_end, idt_ratio_ntc,
         n_htt_points, n_ltt_points, _now()),
    )
    conn.commit()
    return cur.lastrowid


def get_latest_ntc_run(
    conn:         sqlite3.Connection,
    mechanism_id: int,
) -> Optional[Dict[str, Any]]:
    """Return the most recent ntc_runs row for a mechanism, or None."""
    row = conn.execute(
        """SELECT * FROM ntc_runs WHERE mechanism_id=?
           ORDER BY created_at DESC LIMIT 1""",
        (mechanism_id,),
    ).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# CRUD — reverse_rate_flags
# ---------------------------------------------------------------------------

def insert_reverse_rate_flags(
    conn:         sqlite3.Connection,
    mechanism_id: int,
    flags:        List[Dict[str, Any]],
) -> None:
    """Bulk-insert reverse-rate flag rows (dicts matching ReverseRateFlag.to_dict())."""
    ts = _now()
    rows = [
        (mechanism_id, f["canonical_key"], f.get("formula_string"), f["T_K"],
         f["k_fwd"], f["k_rev"], f["collision_limit"], f["exceedance_factor"],
         int(f["flagged"]), ts)
        for f in flags
    ]
    conn.executemany(
        """INSERT INTO reverse_rate_flags
           (mechanism_id, canonical_key, formula_string, T_K, k_fwd, k_rev,
            collision_limit, exceedance_factor, flagged, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    conn.commit()


def get_reverse_rate_flags(
    conn:         sqlite3.Connection,
    mechanism_id: int,
    flagged_only: bool = True,
) -> List[Dict[str, Any]]:
    q = "SELECT * FROM reverse_rate_flags WHERE mechanism_id=?"
    params: list = [mechanism_id]
    if flagged_only:
        q += " AND flagged=1"
    q += " ORDER BY exceedance_factor DESC"
    return [dict(r) for r in conn.execute(q, params).fetchall()]


# ---------------------------------------------------------------------------
# CRUD — mechanism_quality_scores
# ---------------------------------------------------------------------------

def insert_quality_score(
    conn:                     sqlite3.Connection,
    mechanism_id:             int,
    smiles_resolution_pct:    float,
    thermo_consistency_score: float,
    hf_agreement_score:       Optional[float],
    ntc_coverage_score:       Optional[float],
    rate_bounds_score:        float,
    composite_score:          float,
    n_reactions:              int,
    n_species:                int,
    n_reverse_rate_flags:     int,
    details:                  Optional[Dict[str, Any]] = None,
) -> int:
    cur = conn.execute(
        """INSERT INTO mechanism_quality_scores
           (mechanism_id, smiles_resolution_pct, thermo_consistency_score,
            hf_agreement_score, ntc_coverage_score, rate_bounds_score,
            composite_score, n_reactions, n_species, n_reverse_rate_flags,
            details_json, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (mechanism_id, smiles_resolution_pct, thermo_consistency_score,
         hf_agreement_score, ntc_coverage_score, rate_bounds_score,
         composite_score, n_reactions, n_species, n_reverse_rate_flags,
         json.dumps(details) if details else None, _now()),
    )
    conn.commit()
    return cur.lastrowid


def get_latest_quality_score(
    conn:         sqlite3.Connection,
    mechanism_id: int,
) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        """SELECT * FROM mechanism_quality_scores WHERE mechanism_id=?
           ORDER BY created_at DESC LIMIT 1""",
        (mechanism_id,),
    ).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# CRUD — flux_runs
# ---------------------------------------------------------------------------

def insert_flux_run(
    conn:             sqlite3.Connection,
    mechanism_id:     int,
    solver:           str,
    T_K:              float,
    P_atm:            float,
    phi:              float,
    fuel_composition: str,
    element:          str,
    rop_by_reaction:  Dict[str, float],
    element_flux:     List[Dict[str, Any]],
    converged:        bool,
    cpu_time_s:       Optional[float] = None,
    error_msg:        Optional[str]   = None,
) -> int:
    cur = conn.execute(
        """INSERT INTO flux_runs
           (mechanism_id, solver, T_K, P_atm, phi, fuel_composition, element,
            n_reactions, n_flux_edges, rop_json, element_flux_json,
            converged, cpu_time_s, error_msg, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (mechanism_id, solver, T_K, P_atm, phi, fuel_composition, element,
         len(rop_by_reaction), len(element_flux),
         json.dumps(rop_by_reaction), json.dumps(element_flux),
         int(converged), cpu_time_s, error_msg, _now()),
    )
    conn.commit()
    return cur.lastrowid


def get_flux_runs(
    conn:         sqlite3.Connection,
    mechanism_id: int,
    solver:       Optional[str] = None,
) -> List[Dict[str, Any]]:
    q = "SELECT * FROM flux_runs WHERE mechanism_id=?"
    params: list = [mechanism_id]
    if solver:
        q += " AND solver=?"
        params.append(solver)
    q += " ORDER BY created_at DESC"
    return [dict(r) for r in conn.execute(q, params).fetchall()]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
