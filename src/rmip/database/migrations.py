"""
rmip/database/migrations.py
============================
Phase-3 database schema migrations.

Adds to the existing Phase-1/Phase-2 schema (schema.py + schema_phase3.py):

  1. reactions table
     - `fusion_class` TEXT          ("VERBATIM" | "COPY" | "FUSE" | NULL)
     - `source_mechanism_ids_json` TEXT (JSON list of source mechanism DB IDs)

  2. New table: `optimization_runs`
     Records every MSOInterface session (inputs, outputs, status).

  3. New table: `fusion_metadata`
     One row per FusionEngine call; stores provenance JSON for each
     fused mechanism.

  4. New table: `completion_runs`
     Records each RMGInterface.append_to_mechanism() call.

All migrations are idempotent — safe to call multiple times.

Usage
-----
>>> from rmip.database.migrations import run_phase3_migrations
>>> import sqlite3
>>> conn = sqlite3.connect("project.db")
>>> run_phase3_migrations(conn)
>>> conn.close()

Or via CLI:
    python -m rmip.database.migrations project.db
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger("rmip.database.migrations")


# ─────────────────────────────────────────────────────────────────────────────
# DDL statements
# ─────────────────────────────────────────────────────────────────────────────

_ADD_REACTION_FUSION_CLASS = """
ALTER TABLE reactions ADD COLUMN fusion_class TEXT;
"""

_ADD_REACTION_SOURCE_IDS = """
ALTER TABLE reactions ADD COLUMN source_mechanism_ids_json TEXT;
"""

# SQLite cannot ALTER a CHECK constraint in-place — we must recreate the table.
# This DDL widens format to include 'fused' and 'optimized'.
_WIDEN_MECHANISMS_FORMAT = """
PRAGMA foreign_keys = OFF;

CREATE TABLE IF NOT EXISTS mechanisms_new (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,
    format        TEXT    NOT NULL CHECK(format IN ('chemkin','cantera_yaml','cti','fused','optimized')),
    filepath      TEXT,
    description   TEXT,
    n_species     INTEGER,
    n_reactions   INTEGER,
    loaded_at     TEXT    NOT NULL,
    metadata_json TEXT
);

INSERT OR IGNORE INTO mechanisms_new
    SELECT id, name, format, filepath, description,
           n_species, n_reactions, loaded_at, metadata_json
    FROM mechanisms;

DROP TABLE mechanisms;
ALTER TABLE mechanisms_new RENAME TO mechanisms;

PRAGMA foreign_keys = ON;
"""

# Phase-4 widen: adds 'unified' to the CHECK constraint.
_WIDEN_MECHANISMS_FORMAT_P4 = """
PRAGMA foreign_keys = OFF;

CREATE TABLE IF NOT EXISTS mechanisms_new (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,
    format        TEXT    NOT NULL CHECK(format IN ('chemkin','cantera_yaml','cti','fused','optimized','unified')),
    filepath      TEXT,
    description   TEXT,
    n_species     INTEGER,
    n_reactions   INTEGER,
    loaded_at     TEXT    NOT NULL,
    metadata_json TEXT
);

INSERT OR IGNORE INTO mechanisms_new
    SELECT id, name, format, filepath, description,
           n_species, n_reactions, loaded_at, metadata_json
    FROM mechanisms;

DROP TABLE mechanisms;
ALTER TABLE mechanisms_new RENAME TO mechanisms;

PRAGMA foreign_keys = ON;
"""

_CREATE_OPTIMIZATION_RUNS = """
CREATE TABLE IF NOT EXISTS optimization_runs (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id            INTEGER NOT NULL UNIQUE,
    fused_mechanism_id    INTEGER NOT NULL,
    optimized_mechanism_id INTEGER,
    status                TEXT NOT NULL DEFAULT 'pending',
    objective_initial     REAL,
    objective_final       REAL,
    n_fuse_reactions      INTEGER,
    n_target_conditions   INTEGER,
    settings_json         TEXT,
    input_path            TEXT,
    output_path           TEXT,
    runtime_s             REAL,
    error_msg             TEXT,
    created_at            TEXT NOT NULL,
    FOREIGN KEY (fused_mechanism_id) REFERENCES mechanisms(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_optruns_fused_mech
    ON optimization_runs(fused_mechanism_id);
"""

_CREATE_FUSION_METADATA = """
CREATE TABLE IF NOT EXISTS fusion_metadata (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    fused_mechanism_id    INTEGER NOT NULL,
    source_mechanism_ids  TEXT    NOT NULL,  -- JSON list
    comparison_session_id INTEGER,
    n_verbatim            INTEGER NOT NULL DEFAULT 0,
    n_copy                INTEGER NOT NULL DEFAULT 0,
    n_fused               INTEGER NOT NULL DEFAULT 0,
    n_species             INTEGER NOT NULL DEFAULT 0,
    uf_warn_threshold     REAL    NOT NULL DEFAULT 10.0,
    provenance_json       TEXT,
    created_at            TEXT    NOT NULL,
    FOREIGN KEY (fused_mechanism_id) REFERENCES mechanisms(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_fusion_meta_mech
    ON fusion_metadata(fused_mechanism_id);
"""

_CREATE_COMPLETION_RUNS = """
CREATE TABLE IF NOT EXISTS completion_runs (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    base_mechanism_id     INTEGER NOT NULL,
    rmg_mechanism_id      INTEGER NOT NULL,
    target_mechanism_id   INTEGER NOT NULL,
    n_missing_species     INTEGER NOT NULL DEFAULT 0,
    n_missing_reactions   INTEGER NOT NULL DEFAULT 0,
    missing_inchi_keys    TEXT,   -- JSON list
    missing_canonical_keys TEXT,  -- JSON list
    created_at            TEXT NOT NULL,
    FOREIGN KEY (base_mechanism_id)   REFERENCES mechanisms(id) ON DELETE CASCADE,
    FOREIGN KEY (rmg_mechanism_id)    REFERENCES mechanisms(id) ON DELETE CASCADE,
    FOREIGN KEY (target_mechanism_id) REFERENCES mechanisms(id) ON DELETE CASCADE
);
"""


# ─────────────────────────────────────────────────────────────────────────────
# Migration runner
# ─────────────────────────────────────────────────────────────────────────────

def run_phase3_migrations(conn: sqlite3.Connection) -> None:
    """
    Apply all Phase-3 schema migrations idempotently.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open connection to the RMIP SQLite database (Phase-1/2 schema must
        already exist, i.e. ``init_db()`` and ``upgrade_db()`` have been called).
    """
    log.info("Running Phase-3 migrations …")

    # ── 0. mechanisms: widen format CHECK to include 'fused' / 'optimized' ─
    # Detect if the current constraint already allows 'fused'; if not, recreate.
    _ensure_wide_format_constraint(conn)

    # ── 1. reactions: fusion_class column ─────────────────────────────────
    if not _column_exists(conn, "reactions", "fusion_class"):
        conn.execute(_ADD_REACTION_FUSION_CLASS)
        log.info("  Added reactions.fusion_class")
    else:
        log.debug("  reactions.fusion_class already exists — skipping")

    # ── 2. reactions: source_mechanism_ids_json column ────────────────────
    if not _column_exists(conn, "reactions", "source_mechanism_ids_json"):
        conn.execute(_ADD_REACTION_SOURCE_IDS)
        log.info("  Added reactions.source_mechanism_ids_json")
    else:
        log.debug("  reactions.source_mechanism_ids_json already exists — skipping")

    # ── 3. optimization_runs table ────────────────────────────────────────
    if not _table_exists(conn, "optimization_runs"):
        conn.executescript(_CREATE_OPTIMIZATION_RUNS)
        log.info("  Created optimization_runs table")
    else:
        log.debug("  optimization_runs already exists — skipping")

    # ── 4. fusion_metadata table ──────────────────────────────────────────
    if not _table_exists(conn, "fusion_metadata"):
        conn.executescript(_CREATE_FUSION_METADATA)
        log.info("  Created fusion_metadata table")
    else:
        log.debug("  fusion_metadata already exists — skipping")

    # ── 5. completion_runs table ──────────────────────────────────────────
    if not _table_exists(conn, "completion_runs"):
        conn.executescript(_CREATE_COMPLETION_RUNS)
        log.info("  Created completion_runs table")
    else:
        log.debug("  completion_runs already exists — skipping")

    conn.commit()
    log.info("Phase-3 migrations complete.")


# ─────────────────────────────────────────────────────────────────────────────
# CRUD helpers for new tables
# ─────────────────────────────────────────────────────────────────────────────

def insert_optimization_run(
    conn: sqlite3.Connection,
    session_id:             int,
    fused_mechanism_id:     int,
    n_fuse_reactions:       int,
    n_target_conditions:    int,
    settings:               Dict[str, Any],
    input_path:             Optional[str] = None,
    status:                 str = "pending",
) -> int:
    """Insert an optimization_runs row. Returns the new row id."""
    ts = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        """
        INSERT INTO optimization_runs
            (session_id, fused_mechanism_id, status,
             n_fuse_reactions, n_target_conditions,
             settings_json, input_path, created_at)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            session_id,
            fused_mechanism_id,
            status,
            n_fuse_reactions,
            n_target_conditions,
            json.dumps(settings),
            input_path,
            ts,
        ),
    )
    return cur.lastrowid


def update_optimization_run(
    conn: sqlite3.Connection,
    session_id:              int,
    status:                  str,
    optimized_mechanism_id:  Optional[int] = None,
    objective_initial:       Optional[float] = None,
    objective_final:         Optional[float] = None,
    runtime_s:               Optional[float] = None,
    output_path:             Optional[str] = None,
    error_msg:               Optional[str] = None,
) -> None:
    """Update an existing optimization_runs row after the run completes."""
    conn.execute(
        """
        UPDATE optimization_runs SET
            status                  = ?,
            optimized_mechanism_id  = COALESCE(?, optimized_mechanism_id),
            objective_initial       = COALESCE(?, objective_initial),
            objective_final         = COALESCE(?, objective_final),
            runtime_s               = COALESCE(?, runtime_s),
            output_path             = COALESCE(?, output_path),
            error_msg               = COALESCE(?, error_msg)
         WHERE session_id = ?
        """,
        (
            status,
            optimized_mechanism_id,
            objective_initial,
            objective_final,
            runtime_s,
            output_path,
            error_msg,
            session_id,
        ),
    )


def insert_fusion_metadata(
    conn: sqlite3.Connection,
    fused_mechanism_id:    int,
    source_mechanism_ids:  list,
    comparison_session_id: Optional[int],
    n_verbatim:            int,
    n_copy:                int,
    n_fused:               int,
    n_species:             int,
    uf_warn_threshold:     float,
    provenance:            Dict[str, Any],
) -> int:
    """Insert a fusion_metadata row. Returns the new row id."""
    ts = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        """
        INSERT INTO fusion_metadata
            (fused_mechanism_id, source_mechanism_ids,
             comparison_session_id,
             n_verbatim, n_copy, n_fused, n_species,
             uf_warn_threshold, provenance_json, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
        (
            fused_mechanism_id,
            json.dumps(source_mechanism_ids),
            comparison_session_id,
            n_verbatim, n_copy, n_fused, n_species,
            uf_warn_threshold,
            json.dumps(provenance),
            ts,
        ),
    )
    return cur.lastrowid


def insert_completion_run(
    conn: sqlite3.Connection,
    base_mechanism_id:    int,
    rmg_mechanism_id:     int,
    target_mechanism_id:  int,
    missing_inchi_keys:   list,
    missing_canonical_keys: list,
) -> int:
    """Insert a completion_runs row. Returns the new row id."""
    ts = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        """
        INSERT INTO completion_runs
            (base_mechanism_id, rmg_mechanism_id, target_mechanism_id,
             n_missing_species, n_missing_reactions,
             missing_inchi_keys, missing_canonical_keys, created_at)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            base_mechanism_id,
            rmg_mechanism_id,
            target_mechanism_id,
            len(missing_inchi_keys),
            len(missing_canonical_keys),
            json.dumps(missing_inchi_keys),
            json.dumps(missing_canonical_keys),
            ts,
        ),
    )
    return cur.lastrowid


def list_optimization_runs(conn: sqlite3.Connection) -> list:
    """Return all rows in optimization_runs as list of dicts."""
    conn.row_factory = sqlite3.Row
    return [dict(r) for r in conn.execute(
        "SELECT * FROM optimization_runs ORDER BY created_at DESC"
    )]


def list_fusion_metadata(conn: sqlite3.Connection) -> list:
    """Return all rows in fusion_metadata as list of dicts."""
    conn.row_factory = sqlite3.Row
    return [dict(r) for r in conn.execute(
        "SELECT * FROM fusion_metadata ORDER BY created_at DESC"
    )]


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_wide_format_constraint(conn: sqlite3.Connection) -> None:
    """
    Recreate the mechanisms table if its format CHECK does not yet include
    'fused' and 'optimized'.  Idempotent — if the constraint is already wide,
    this is a no-op (detected via a test INSERT that is rolled back).
    """
    # Quick probe: try to INSERT a 'fused' row inside a savepoint, then roll back.
    try:
        conn.execute("SAVEPOINT _fmt_probe")
        conn.execute(
            "INSERT INTO mechanisms (name, format, loaded_at) VALUES ('_probe','fused','x')"
        )
        conn.execute("ROLLBACK TO SAVEPOINT _fmt_probe")
        conn.execute("RELEASE SAVEPOINT _fmt_probe")
        log.debug("  mechanisms.format already accepts 'fused' — skipping constraint widen")
    except sqlite3.IntegrityError:
        conn.execute("ROLLBACK TO SAVEPOINT _fmt_probe")
        conn.execute("RELEASE SAVEPOINT _fmt_probe")
        log.info("  Widening mechanisms.format CHECK to include 'fused'/'optimized' …")
        conn.executescript(_WIDEN_MECHANISMS_FORMAT)
        log.info("  mechanisms.format constraint updated.")


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def _column_exists(conn: sqlite3.Connection, table_name: str, col_name: str) -> bool:
    cols = {
        row[1]
        for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    return col_name in cols


def run_phase4_migrations(conn: sqlite3.Connection) -> None:
    """
    Apply all Phase-4 schema migrations idempotently.

    Requires Phase-3 migrations to have already been applied.
    Creates: unified_tree_runs, reaction_decisions, uncertainty_bands tables.
    Widens mechanisms.format CHECK to include 'unified'.
    Adds reaction_label column to reactions table.
    """
    log.info("Running Phase-4 migrations …")

    # ── 0. Widen mechanisms.format to include 'unified' ──────────────────
    _ensure_unified_format_constraint(conn)

    # ── 1. Phase-4 tables via schema_phase4 ──────────────────────────────
    from rmip.database.schema_phase4 import upgrade_db_phase4
    upgrade_db_phase4(conn)

    # ── 2. Add reaction_label column to reactions (idempotent) ───────────
    _add_reaction_label_column(conn)

    log.info("Phase-4 migrations complete.")


def _add_reaction_label_column(conn: sqlite3.Connection) -> None:
    """Add reaction_label TEXT column to reactions table if not already present."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(reactions)").fetchall()}
    if "reaction_label" not in cols:
        conn.execute("ALTER TABLE reactions ADD COLUMN reaction_label TEXT")
        conn.commit()
        log.info("  Added reaction_label column to reactions table.")
    else:
        log.debug("  reaction_label column already present — skipping.")


def _ensure_unified_format_constraint(conn: sqlite3.Connection) -> None:
    """
    Recreate the mechanisms table if its format CHECK does not yet include
    'unified'.  Idempotent — probe via savepoint INSERT then rollback.
    """
    try:
        conn.execute("SAVEPOINT _fmt_probe_p4")
        conn.execute(
            "INSERT INTO mechanisms (name, format, loaded_at) VALUES ('_probe','unified','x')"
        )
        conn.execute("ROLLBACK TO SAVEPOINT _fmt_probe_p4")
        conn.execute("RELEASE SAVEPOINT _fmt_probe_p4")
        log.debug("  mechanisms.format already accepts 'unified' — skipping")
    except sqlite3.IntegrityError:
        conn.execute("ROLLBACK TO SAVEPOINT _fmt_probe_p4")
        conn.execute("RELEASE SAVEPOINT _fmt_probe_p4")
        log.info("  Widening mechanisms.format CHECK to include 'unified' …")
        conn.executescript(_WIDEN_MECHANISMS_FORMAT_P4)
        log.info("  mechanisms.format constraint updated.")


def schema_version(conn: sqlite3.Connection) -> Dict[str, bool]:
    """Report which Phase-3 schema elements exist."""
    return {
        "reactions.fusion_class":             _column_exists(conn, "reactions", "fusion_class"),
        "reactions.source_mechanism_ids_json": _column_exists(conn, "reactions", "source_mechanism_ids_json"),
        "optimization_runs":                  _table_exists(conn, "optimization_runs"),
        "fusion_metadata":                    _table_exists(conn, "fusion_metadata"),
        "completion_runs":                    _table_exists(conn, "completion_runs"),
    }


def run_phase5d_migrations(conn: sqlite3.Connection) -> None:
    """
    Phase-5D idempotent migration: add species variant columns.

    Adds ``variant_tag`` and ``canonical_name`` to the ``species`` table.
    These columns support SMILES disambiguation (OHEX/OH, HNO2/HONO, etc.).
    """
    for col, col_type in [("variant_tag", "TEXT"), ("canonical_name", "TEXT")]:
        if not _column_exists(conn, "species", col):
            conn.execute(f"ALTER TABLE species ADD COLUMN {col} {col_type}")
            log.info("Phase-5D: added species.%s", col)
        else:
            log.debug("Phase-5D: species.%s already exists — skipped", col)
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if len(sys.argv) < 2:
        print("Usage: python -m rmip.database.migrations <path/to/project.db>")
        sys.exit(1)
    db_path = sys.argv[1]
    if not Path(db_path).exists():
        print(f"Error: database not found: {db_path}")
        sys.exit(1)
    with sqlite3.connect(db_path) as conn:
        run_phase3_migrations(conn)
    print(f"Migrations applied to {db_path}")
    with sqlite3.connect(db_path) as conn:
        ver = schema_version(conn)
    print("Schema status:")
    for k, v in ver.items():
        status = "✓" if v else "✗"
        print(f"  {status}  {k}")
