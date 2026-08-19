"""
rmip/database/schema_phase3.py
==============================
Phase-3 DDL additions and CRUD helpers for:
  - simulation_runs      (IDT / LFS results per mechanism)
  - exp_datasets         (experimental dataset registry)
  - exp_data_points      (individual T/P/phi → value rows)
  - mechanism_scores     (MRE / RMSE per mechanism per dataset per regime)
  - graph_comparison_sessions   (one row per compare() call)
  - reaction_classifications    (per-reaction classification in a session)
  - species_classifications     (per-species classification in a session)

Usage (call ONCE after schema.init_db):
    from rmip.database.schema_phase3 import upgrade_db
    upgrade_db(conn)

CRUD helpers follow the same pattern as schema.py:
  - plain sqlite3 only (design decision D3)
  - SI units for all numerical values
  - ISO-8601 strings for timestamps
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_PHASE3_DDL = """
-- -------------------------------------------------------------------------
-- simulation_runs
-- One row per (mechanism_id, sim_type, T_K, P_atm, phi, fuel) simulation.
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS simulation_runs (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    mechanism_id            INTEGER NOT NULL,
    sim_type                TEXT    NOT NULL CHECK(sim_type IN ('IDT','LFS')),
    T_K                     REAL    NOT NULL,
    P_atm                   REAL    NOT NULL,
    phi                     REAL    NOT NULL,
    fuel_name               TEXT    NOT NULL,
    fuel_composition        TEXT    NOT NULL,   -- Cantera string, e.g. "NC7H16:1.0"
    oxidizer_composition    TEXT    NOT NULL,   -- e.g. "O2:1.0, N2:3.76"
    result                  REAL,              -- IDT [s] or LFS [m/s]; NULL = no ignition/convergence
    converged               INTEGER NOT NULL DEFAULT 1,
    cpu_time_s              REAL,
    error_msg               TEXT,
    created_at              TEXT    NOT NULL,
    FOREIGN KEY (mechanism_id) REFERENCES mechanisms(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_simruns_mech_type
    ON simulation_runs(mechanism_id, sim_type);

CREATE INDEX IF NOT EXISTS idx_simruns_conditions
    ON simulation_runs(sim_type, T_K, P_atm, phi);

-- -------------------------------------------------------------------------
-- exp_datasets
-- Registry of experimental datasets (one entry per file load).
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS exp_datasets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    sim_type    TEXT    NOT NULL CHECK(sim_type IN ('IDT','LFS')),
    fuel_name   TEXT,
    source_ref  TEXT,
    n_points    INTEGER NOT NULL DEFAULT 0,
    filepath    TEXT,
    created_at  TEXT    NOT NULL
);

-- -------------------------------------------------------------------------
-- exp_data_points
-- Individual experimental observations linked to a dataset.
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS exp_data_points (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id  INTEGER NOT NULL,
    T_K         REAL,
    P_atm       REAL,
    phi         REAL,
    value       REAL    NOT NULL,    -- IDT [s] or LFS [m/s]
    uncertainty REAL,                -- 1-sigma absolute uncertainty; NULL if unknown
    units       TEXT    NOT NULL,    -- 's' for IDT, 'm/s' for LFS
    notes       TEXT,
    FOREIGN KEY (dataset_id) REFERENCES exp_datasets(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_exppts_dataset
    ON exp_data_points(dataset_id);

-- -------------------------------------------------------------------------
-- mechanism_scores
-- Error metrics for a mechanism against one experimental dataset.
-- One row per (mechanism_id, dataset_id, regime).
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS mechanism_scores (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    mechanism_id    INTEGER NOT NULL,
    dataset_id      INTEGER NOT NULL,
    regime          TEXT    NOT NULL DEFAULT 'all',  -- 'LTC'|'NTC'|'HTC'|'all'
    n_points        INTEGER NOT NULL,
    mre             REAL,    -- Mean Relative Error   (dimensionless)
    rmse            REAL,    -- Root Mean Squared Error (in native units)
    log_rmse        REAL,    -- RMSE of log10(sim/exp)
    rank_position   INTEGER, -- rank within the comparison set for this dataset+regime
    computed_at     TEXT    NOT NULL,
    FOREIGN KEY (mechanism_id) REFERENCES mechanisms(id) ON DELETE CASCADE,
    FOREIGN KEY (dataset_id)   REFERENCES exp_datasets(id) ON DELETE CASCADE,
    UNIQUE(mechanism_id, dataset_id, regime)
);

-- -------------------------------------------------------------------------
-- graph_comparison_sessions
-- One row per GraphComparator.compare() call.
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS graph_comparison_sessions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    mechanism_ids_json  TEXT    NOT NULL,   -- JSON list, e.g. [1,2,3]
    n_mechanisms        INTEGER NOT NULL,
    kinetic_tol_A       REAL    NOT NULL DEFAULT 0.01,   -- relative tolerance for A
    kinetic_tol_n       REAL    NOT NULL DEFAULT 0.001,  -- absolute tolerance for n
    kinetic_tol_Ea      REAL    NOT NULL DEFAULT 100.0,  -- J/mol tolerance for Ea
    notes               TEXT,
    created_at          TEXT    NOT NULL
);

-- -------------------------------------------------------------------------
-- reaction_classifications
-- Per-reaction outcome in a comparison session.
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS reaction_classifications (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id                  INTEGER NOT NULL,
    canonical_key               TEXT    NOT NULL,
    formula_string              TEXT,
    classification              TEXT    NOT NULL
                                    CHECK(classification IN
                                          ('UNIQUE','SHARED_IDENTICAL','SHARED_DIVERGENT')),
    present_in_mechs_json       TEXT    NOT NULL,   -- JSON list of mechanism IDs
    -- Aggregated kinetics (populated for SHARED_DIVERGENT and SHARED_IDENTICAL)
    mean_A_si                   REAL,   -- geometric mean of A
    mean_n                      REAL,
    mean_Ea_si                  REAL,   -- J/mol
    min_A_si                    REAL,
    max_A_si                    REAL,
    min_n                       REAL,
    max_n                       REAL,
    min_Ea_si                   REAL,
    max_Ea_si                   REAL,
    per_mech_kinetics_json      TEXT,   -- JSON {mech_id: {A_si,n,Ea_si,...}}
    FOREIGN KEY (session_id) REFERENCES graph_comparison_sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_rxncls_session
    ON reaction_classifications(session_id, classification);

CREATE INDEX IF NOT EXISTS idx_rxncls_key
    ON reaction_classifications(canonical_key);

-- -------------------------------------------------------------------------
-- species_classifications
-- Per-species outcome in a comparison session.
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS species_classifications (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id              INTEGER NOT NULL,
    inchi_key               TEXT    NOT NULL,
    canonical_smiles        TEXT,
    formula                 TEXT,
    classification          TEXT    NOT NULL
                                CHECK(classification IN ('UNIVERSAL','PARTIAL','UNIQUE')),
    present_in_mechs_json   TEXT    NOT NULL,   -- JSON list of mechanism IDs
    per_mech_thermo_json    TEXT,               -- JSON {mech_id: {coeffs_low, coeffs_high,...}}
    FOREIGN KEY (session_id) REFERENCES graph_comparison_sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_speccls_session
    ON species_classifications(session_id, classification);
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def upgrade_db(conn: sqlite3.Connection) -> None:
    """
    Idempotently add all Phase-3 tables to an existing RMIP database.
    Safe to call on a DB that already has Phase-3 tables (IF NOT EXISTS guards).
    """
    conn.executescript(_PHASE3_DDL)
    conn.commit()


# ---------------------------------------------------------------------------
# CRUD — simulation_runs
# ---------------------------------------------------------------------------

def insert_simulation_run(
    conn: sqlite3.Connection,
    mechanism_id:         int,
    sim_type:             str,
    T_K:                  float,
    P_atm:                float,
    phi:                  float,
    fuel_name:            str,
    fuel_composition:     str,
    oxidizer_composition: str,
    result:               Optional[float],
    converged:            bool,
    cpu_time_s:           Optional[float] = None,
    error_msg:            Optional[str]   = None,
) -> int:
    cur = conn.execute(
        """INSERT INTO simulation_runs
           (mechanism_id, sim_type, T_K, P_atm, phi,
            fuel_name, fuel_composition, oxidizer_composition,
            result, converged, cpu_time_s, error_msg, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (mechanism_id, sim_type, T_K, P_atm, phi,
         fuel_name, fuel_composition, oxidizer_composition,
         result, int(converged), cpu_time_s, error_msg,
         _now()),
    )
    conn.commit()
    return cur.lastrowid


def get_simulation_runs(
    conn:         sqlite3.Connection,
    mechanism_id: int,
    sim_type:     Optional[str] = None,
) -> List[Dict[str, Any]]:
    q = "SELECT * FROM simulation_runs WHERE mechanism_id=?"
    params: list = [mechanism_id]
    if sim_type:
        q += " AND sim_type=?"
        params.append(sim_type)
    q += " ORDER BY T_K, P_atm, phi"
    return [dict(r) for r in conn.execute(q, params).fetchall()]


# ---------------------------------------------------------------------------
# CRUD — exp_datasets / exp_data_points
# ---------------------------------------------------------------------------

def insert_exp_dataset(
    conn:       sqlite3.Connection,
    name:       str,
    sim_type:   str,
    fuel_name:  Optional[str] = None,
    source_ref: Optional[str] = None,
    filepath:   Optional[str] = None,
) -> int:
    cur = conn.execute(
        """INSERT OR IGNORE INTO exp_datasets
           (name, sim_type, fuel_name, source_ref, filepath, created_at)
           VALUES (?,?,?,?,?,?)""",
        (name, sim_type, fuel_name, source_ref, filepath, _now()),
    )
    conn.commit()
    if cur.lastrowid:
        return cur.lastrowid
    # dataset already existed → return existing id
    row = conn.execute(
        "SELECT id FROM exp_datasets WHERE name=?", (name,)
    ).fetchone()
    return row["id"] if row else -1


def insert_exp_data_points(
    conn:       sqlite3.Connection,
    dataset_id: int,
    points:     List[Dict[str, Any]],
) -> None:
    """
    Bulk-insert experimental data points.
    Each dict in `points` must have keys: T_K, P_atm, phi, value, units.
    Optional keys: uncertainty, notes.
    """
    rows = [
        (
            dataset_id,
            p.get("T_K"),    p.get("P_atm"), p.get("phi"),
            p["value"],      p.get("uncertainty"),
            p["units"],      p.get("notes"),
        )
        for p in points
    ]
    conn.executemany(
        """INSERT INTO exp_data_points
           (dataset_id, T_K, P_atm, phi, value, uncertainty, units, notes)
           VALUES (?,?,?,?,?,?,?,?)""",
        rows,
    )
    conn.execute(
        "UPDATE exp_datasets SET n_points=(SELECT COUNT(*) FROM exp_data_points "
        "WHERE dataset_id=?) WHERE id=?",
        (dataset_id, dataset_id),
    )
    conn.commit()


def get_exp_data_points(
    conn:       sqlite3.Connection,
    dataset_id: int,
) -> List[Dict[str, Any]]:
    return [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM exp_data_points WHERE dataset_id=? ORDER BY T_K, P_atm, phi",
            (dataset_id,),
        ).fetchall()
    ]


def list_exp_datasets(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    return [dict(r) for r in conn.execute(
        "SELECT * FROM exp_datasets ORDER BY created_at"
    ).fetchall()]


# ---------------------------------------------------------------------------
# CRUD — mechanism_scores
# ---------------------------------------------------------------------------

def upsert_mechanism_score(
    conn:         sqlite3.Connection,
    mechanism_id: int,
    dataset_id:   int,
    regime:       str,
    n_points:     int,
    mre:          Optional[float],
    rmse:         Optional[float],
    log_rmse:     Optional[float],
    rank_position: Optional[int] = None,
) -> int:
    cur = conn.execute(
        """INSERT INTO mechanism_scores
           (mechanism_id, dataset_id, regime, n_points, mre, rmse, log_rmse,
            rank_position, computed_at)
           VALUES (?,?,?,?,?,?,?,?,?)
           ON CONFLICT(mechanism_id, dataset_id, regime)
           DO UPDATE SET
               n_points=excluded.n_points, mre=excluded.mre,
               rmse=excluded.rmse, log_rmse=excluded.log_rmse,
               rank_position=excluded.rank_position,
               computed_at=excluded.computed_at""",
        (mechanism_id, dataset_id, regime, n_points, mre, rmse, log_rmse,
         rank_position, _now()),
    )
    conn.commit()
    return cur.lastrowid


def get_mechanism_scores(
    conn:      sqlite3.Connection,
    dataset_id: int,
    regime:    str = "all",
) -> List[Dict[str, Any]]:
    return [dict(r) for r in conn.execute(
        """SELECT ms.*, m.name AS mechanism_name
           FROM mechanism_scores ms
           JOIN mechanisms m ON m.id = ms.mechanism_id
           WHERE ms.dataset_id=? AND ms.regime=?
           ORDER BY ms.rank_position NULLS LAST, ms.mre""",
        (dataset_id, regime),
    ).fetchall()]


# ---------------------------------------------------------------------------
# CRUD — graph_comparison_sessions
# ---------------------------------------------------------------------------

def insert_comparison_session(
    conn:             sqlite3.Connection,
    mechanism_ids:    List[int],
    kinetic_tol_A:    float = 0.01,
    kinetic_tol_n:    float = 0.001,
    kinetic_tol_Ea:   float = 100.0,
    notes:            Optional[str] = None,
) -> int:
    cur = conn.execute(
        """INSERT INTO graph_comparison_sessions
           (mechanism_ids_json, n_mechanisms,
            kinetic_tol_A, kinetic_tol_n, kinetic_tol_Ea,
            notes, created_at)
           VALUES (?,?,?,?,?,?,?)""",
        (json.dumps(mechanism_ids), len(mechanism_ids),
         kinetic_tol_A, kinetic_tol_n, kinetic_tol_Ea,
         notes, _now()),
    )
    conn.commit()
    return cur.lastrowid


def insert_reaction_classification(
    conn:             sqlite3.Connection,
    session_id:       int,
    canonical_key:    str,
    formula_string:   Optional[str],
    classification:   str,
    present_in_mechs: List[int],
    kinetics_data:    Optional[Dict[str, Any]] = None,
    mean_A:   Optional[float] = None,
    mean_n:   Optional[float] = None,
    mean_Ea:  Optional[float] = None,
    min_A:    Optional[float] = None,
    max_A:    Optional[float] = None,
    min_n:    Optional[float] = None,
    max_n:    Optional[float] = None,
    min_Ea:   Optional[float] = None,
    max_Ea:   Optional[float] = None,
) -> int:
    cur = conn.execute(
        """INSERT INTO reaction_classifications
           (session_id, canonical_key, formula_string, classification,
            present_in_mechs_json,
            mean_A_si, mean_n, mean_Ea_si,
            min_A_si, max_A_si, min_n, max_n, min_Ea_si, max_Ea_si,
            per_mech_kinetics_json)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (session_id, canonical_key, formula_string, classification,
         json.dumps(present_in_mechs),
         mean_A, mean_n, mean_Ea,
         min_A, max_A, min_n, max_n, min_Ea, max_Ea,
         json.dumps(kinetics_data) if kinetics_data else None),
    )
    return cur.lastrowid


def insert_species_classification(
    conn:           sqlite3.Connection,
    session_id:     int,
    inchi_key:      str,
    canonical_smiles: Optional[str],
    formula:        Optional[str],
    classification: str,
    present_in_mechs: List[int],
    thermo_data:    Optional[Dict[str, Any]] = None,
) -> int:
    cur = conn.execute(
        """INSERT INTO species_classifications
           (session_id, inchi_key, canonical_smiles, formula,
            classification, present_in_mechs_json, per_mech_thermo_json)
           VALUES (?,?,?,?,?,?,?)""",
        (session_id, inchi_key, canonical_smiles, formula,
         classification, json.dumps(present_in_mechs),
         json.dumps(thermo_data) if thermo_data else None),
    )
    return cur.lastrowid


def get_comparison_session(
    conn:       sqlite3.Connection,
    session_id: int,
) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        "SELECT * FROM graph_comparison_sessions WHERE id=?", (session_id,)
    ).fetchone()
    return dict(row) if row else None


def get_reaction_classifications_for_session(
    conn:           sqlite3.Connection,
    session_id:     int,
    classification: Optional[str] = None,
) -> List[Dict[str, Any]]:
    q = "SELECT * FROM reaction_classifications WHERE session_id=?"
    params: list = [session_id]
    if classification:
        q += " AND classification=?"
        params.append(classification)
    return [dict(r) for r in conn.execute(q, params).fetchall()]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
