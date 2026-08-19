"""
rmip/database/schema_phase4.py
===============================
Phase-4 DDL additions: unified tree, reaction decisions, and uncertainty bands.

Tables
------
  unified_tree_runs    — one row per UnifiedTreeBuilder.build() call
  reaction_decisions   — one row per reaction per unified_tree_run
  uncertainty_bands    — long-format: one row per reaction × temperature point

All DDL uses CREATE TABLE IF NOT EXISTS and is safe to call on existing DBs.

Usage
-----
    from rmip.database.schema_phase4 import upgrade_db_phase4
    upgrade_db_phase4(conn)
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_CREATE_UNIFIED_TREE_RUNS = """
CREATE TABLE IF NOT EXISTS unified_tree_runs (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    unified_mechanism_id    INTEGER NOT NULL,
    mode                    TEXT    NOT NULL,           -- 'anchored' | 'optimize_all'
    anchor_mech_id          INTEGER,
    contributing_mech_ids_json TEXT NOT NULL,           -- JSON list
    manifest_path           TEXT,
    n_keep_anchor           INTEGER NOT NULL DEFAULT 0,
    n_import_single         INTEGER NOT NULL DEFAULT 0,
    n_optimize              INTEGER NOT NULL DEFAULT 0,
    n_pick_source           INTEGER NOT NULL DEFAULT 0,
    n_dropped               INTEGER NOT NULL DEFAULT 0,
    n_species               INTEGER NOT NULL DEFAULT 0,
    metadata_json           TEXT,
    created_at              TEXT    NOT NULL,
    FOREIGN KEY (unified_mechanism_id) REFERENCES mechanisms(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_utree_mech ON unified_tree_runs(unified_mechanism_id);
"""

_CREATE_REACTION_DECISIONS = """
CREATE TABLE IF NOT EXISTS reaction_decisions (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                  INTEGER NOT NULL,
    canonical_key           TEXT    NOT NULL,
    formula_string          TEXT,
    status                  TEXT    NOT NULL,           -- IN_ANCHOR|PATCH_SINGLE|PATCH_MULTI|SHARED|UNIQUE
    decision                TEXT    NOT NULL,           -- KEEP_ANCHOR|IMPORT_SINGLE|OPTIMIZE|PICK_SOURCE|DROP
    chosen_source_mech_id   INTEGER,
    source_mech_ids_json    TEXT,
    notes                   TEXT,
    FOREIGN KEY (run_id) REFERENCES unified_tree_runs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_rdec_run   ON reaction_decisions(run_id);
CREATE INDEX IF NOT EXISTS idx_rdec_ckey  ON reaction_decisions(canonical_key);
"""

_CREATE_UNCERTAINTY_BANDS = """
CREATE TABLE IF NOT EXISTS uncertainty_bands (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL,
    canonical_key   TEXT    NOT NULL,
    T_K             REAL    NOT NULL,
    k_nom           REAL,
    k_min           REAL,
    k_max           REAL,
    f_uncertainty   REAL,
    sigma_lnk       REAL,
    f_to_nom        REAL,
    method          TEXT    NOT NULL DEFAULT 'minmax',
    FOREIGN KEY (run_id) REFERENCES unified_tree_runs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_ubands_run  ON uncertainty_bands(run_id);
CREATE INDEX IF NOT EXISTS idx_ubands_key  ON uncertainty_bands(run_id, canonical_key);
"""


def upgrade_db_phase4(conn: sqlite3.Connection) -> None:
    """
    Add Phase-4 tables to an existing RMIP database.
    Idempotent — safe to call multiple times.
    """
    for ddl in (_CREATE_UNIFIED_TREE_RUNS,
                _CREATE_REACTION_DECISIONS,
                _CREATE_UNCERTAINTY_BANDS):
        conn.executescript(ddl)
    conn.commit()


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

def insert_unified_tree_run(
    conn:                   sqlite3.Connection,
    unified_mechanism_id:   int,
    mode:                   str,
    contributing_mech_ids:  List[int],
    anchor_mech_id:         Optional[int] = None,
    manifest_path:          Optional[str] = None,
    n_keep_anchor:          int = 0,
    n_import_single:        int = 0,
    n_optimize:             int = 0,
    n_pick_source:          int = 0,
    n_dropped:              int = 0,
    n_species:              int = 0,
    metadata:               Optional[Dict[str, Any]] = None,
) -> int:
    """Insert a unified_tree_runs row; return new id."""
    ts = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        """
        INSERT INTO unified_tree_runs
            (unified_mechanism_id, mode, anchor_mech_id,
             contributing_mech_ids_json, manifest_path,
             n_keep_anchor, n_import_single, n_optimize,
             n_pick_source, n_dropped, n_species,
             metadata_json, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            unified_mechanism_id,
            mode,
            anchor_mech_id,
            json.dumps(contributing_mech_ids),
            manifest_path,
            n_keep_anchor,
            n_import_single,
            n_optimize,
            n_pick_source,
            n_dropped,
            n_species,
            json.dumps(metadata or {}),
            ts,
        ),
    )
    return cur.lastrowid


def insert_reaction_decision(
    conn:                   sqlite3.Connection,
    run_id:                 int,
    canonical_key:          str,
    formula_string:         str,
    status:                 str,
    decision:               str,
    source_mech_ids:        List[int],
    chosen_source_mech_id:  Optional[int] = None,
    notes:                  str = "",
) -> int:
    """Insert one reaction_decisions row; return new id."""
    cur = conn.execute(
        """
        INSERT INTO reaction_decisions
            (run_id, canonical_key, formula_string, status, decision,
             chosen_source_mech_id, source_mech_ids_json, notes)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            run_id,
            canonical_key,
            formula_string,
            status,
            decision,
            chosen_source_mech_id,
            json.dumps(source_mech_ids),
            notes,
        ),
    )
    return cur.lastrowid


def insert_uncertainty_band_rows(
    conn:         sqlite3.Connection,
    run_id:       int,
    canonical_key: str,
    T_K_list:     List[float],
    k_nom_list:   List[float],
    k_min_list:   List[float],
    k_max_list:   List[float],
    f_list:       List[float],
    sigma_list:   List[float],
    f_to_nom_list: List[float],
    method:       str = "minmax",
) -> None:
    """Bulk-insert uncertainty_bands rows for one reaction."""
    rows = [
        (run_id, canonical_key, T, kn, kmin, kmax, f, s, ftn, method)
        for T, kn, kmin, kmax, f, s, ftn
        in zip(T_K_list, k_nom_list, k_min_list, k_max_list,
               f_list, sigma_list, f_to_nom_list)
    ]
    conn.executemany(
        """
        INSERT INTO uncertainty_bands
            (run_id, canonical_key, T_K, k_nom, k_min, k_max,
             f_uncertainty, sigma_lnk, f_to_nom, method)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
        rows,
    )
