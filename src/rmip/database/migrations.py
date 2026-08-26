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
#
# M17: the CREATE/INSERT/DROP/RENAME sequence is wrapped in an explicit
# BEGIN…COMMIT so it is atomic. Without it, a crash (or kill) between
# `DROP TABLE mechanisms` and the RENAME would leave the database with NO
# mechanisms table at all — permanent data loss on the central table. The
# `PRAGMA foreign_keys` toggles stay OUTSIDE the transaction (the pragma is a
# no-op while a transaction is open); executescript() commits any pending
# transaction before it runs, so foreign_keys=OFF takes effect before BEGIN.
_WIDEN_MECHANISMS_FORMAT = """
PRAGMA foreign_keys = OFF;
BEGIN;

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

COMMIT;
PRAGMA foreign_keys = ON;
"""

# Phase-4 widen: adds 'unified' to the CHECK constraint. Same atomicity contract
# as _WIDEN_MECHANISMS_FORMAT above (M17).
_WIDEN_MECHANISMS_FORMAT_P4 = """
PRAGMA foreign_keys = OFF;
BEGIN;

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

COMMIT;
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


_CREATE_REDUCTION_RUNS = """
CREATE TABLE IF NOT EXISTS reduction_runs (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    mechanism_id            INTEGER NOT NULL REFERENCES mechanisms(id) ON DELETE CASCADE,
    method                  TEXT    NOT NULL CHECK(method IN ('DRG','DRGEP','PFA','DRGPSA','DRGSA')),
    target_species_json     TEXT    NOT NULL,
    epsilon_min             REAL    NOT NULL DEFAULT 0.0,
    epsilon_max             REAL    NOT NULL DEFAULT 0.5,
    n_steps                 INTEGER NOT NULL DEFAULT 51,
    n_species_full          INTEGER,
    n_reactions_full        INTEGER,
    n_species_minimal       INTEGER,
    n_reactions_minimal     INTEGER,
    created_at              TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_reduction_runs_mech
    ON reduction_runs(mechanism_id);
"""


def run_phase5b_migrations(conn: sqlite3.Connection) -> None:
    """
    Phase-5B idempotent migration: add reduction_runs table.

    Creates the ``reduction_runs`` table that records each DRG/DRGEP/PFA/DRGPSA
    reduction sweep (metadata only; full step data is stored as JSON files).
    Also widens the CHECK constraint to allow DRGPSA and DRGSA methods if the
    table already exists with the old constraint.
    """
    log.info("Running Phase-5B migrations …")
    if not _table_exists(conn, "reduction_runs"):
        conn.executescript(_CREATE_REDUCTION_RUNS)
        log.info("  Created reduction_runs table")
    else:
        _ensure_wide_reduction_method_constraint(conn)
    conn.commit()
    log.info("Phase-5B migrations complete.")


def _ensure_wide_reduction_method_constraint(conn: sqlite3.Connection) -> None:
    """Widen reduction_runs.method CHECK to include DRGPSA/DRGSA (idempotent)."""
    # M17: idempotency must be decided from the table's own DDL, NOT a probe
    # INSERT. The old probe INSERTed mechanism_id=0, which — with foreign_keys
    # ON — fails the FK to mechanisms(id) *regardless* of whether the method
    # CHECK is already wide, so the `except` branch fired on every startup and
    # rebuilt the table needlessly (and destructively — see below). Inspecting
    # the stored CREATE TABLE text is side-effect-free and unambiguous.
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='reduction_runs'"
    ).fetchone()
    if row is None:
        # Table doesn't exist yet — created wide by schema_phase5b elsewhere.
        return
    table_sql = row[0] or ""
    if "'DRGPSA'" in table_sql and "'DRGSA'" in table_sql:
        log.debug("  reduction_runs.method CHECK already wide — skipping rebuild")
        return

    log.info("  Widening reduction_runs.method CHECK to include DRGPSA/DRGSA …")
    # M17: wrap the rebuild in an explicit transaction so a crash between the
    # RENAME and the INSERT/DROP can't lose reduction_runs. foreign_keys is
    # toggled OFF around the swap (outside the transaction, per SQLite rules).
    conn.execute("PRAGMA foreign_keys=OFF")
    try:
        conn.executescript("""
            BEGIN;
            ALTER TABLE reduction_runs RENAME TO _reduction_runs_old;

            CREATE TABLE reduction_runs (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                mechanism_id            INTEGER NOT NULL REFERENCES mechanisms(id) ON DELETE CASCADE,
                method                  TEXT    NOT NULL
                    CHECK(method IN ('DRG','DRGEP','PFA','DRGPSA','DRGSA')),
                target_species_json     TEXT    NOT NULL,
                epsilon_min             REAL    NOT NULL DEFAULT 0.0,
                epsilon_max             REAL    NOT NULL DEFAULT 0.5,
                n_steps                 INTEGER NOT NULL DEFAULT 51,
                n_species_full          INTEGER,
                n_reactions_full        INTEGER,
                n_species_minimal       INTEGER,
                n_reactions_minimal     INTEGER,
                created_at              TEXT    NOT NULL
            );

            INSERT INTO reduction_runs SELECT * FROM _reduction_runs_old;

            DROP TABLE _reduction_runs_old;

            CREATE INDEX IF NOT EXISTS idx_reduction_runs_mech
                ON reduction_runs(mechanism_id);
            COMMIT;
        """)
    finally:
        conn.execute("PRAGMA foreign_keys=ON")


def run_phase6_migrations(conn: sqlite3.Connection) -> None:
    """
    Apply all Phase-6 schema migrations idempotently (Future.md features 1–7).

    - Widens ``simulation_runs.sim_type`` CHECK to include 'JSR', 'PFR', 'FLUX'.
    - Adds ``simulation_runs.extra_json`` — generic per-sim-type payload (JSR/PFR
      full species composition + axial profile; JSON, NULL for IDT/LFS rows).
    - Adds ``exp_data_points.temp_target_class`` (HTT/LTT, Future.md pt. 7).
    - Creates ``ntc_runs``, ``reverse_rate_flags``, ``mechanism_quality_scores``,
      ``flux_runs`` tables via ``schema_phase6.upgrade_db_phase6``.
    """
    log.info("Running Phase-6 migrations …")

    _ensure_wide_sim_type_constraint(conn)
    _ensure_wide_exp_dataset_sim_type(conn)
    _add_temp_target_class_column(conn)
    _add_simulation_runs_extra_json_column(conn)
    _add_species_transport_columns(conn)

    from rmip.database.schema_phase6 import upgrade_db_phase6
    upgrade_db_phase6(conn)

    conn.commit()
    log.info("Phase-6 migrations complete.")


def _add_simulation_runs_extra_json_column(conn: sqlite3.Connection) -> None:
    """Add simulation_runs.extra_json TEXT column if not already present."""
    if not _table_exists(conn, "simulation_runs"):
        return
    if not _column_exists(conn, "simulation_runs", "extra_json"):
        conn.execute("ALTER TABLE simulation_runs ADD COLUMN extra_json TEXT")
        conn.commit()
        log.info("  Added simulation_runs.extra_json")
    else:
        log.debug("  simulation_runs.extra_json already exists — skipping")


def _ensure_wide_exp_dataset_sim_type(conn: sqlite3.Connection) -> None:
    """
    Widen ``exp_datasets.sim_type`` CHECK from ('IDT','LFS') to
    ('IDT','LFS','JSR','PFR') so ReSpecTh XML ingestion (Future.md pt. 2,
    ReSpecTh/PrIMe) can register speciation datasets, not just IDT/LFS.
    Idempotent — probed via a rolled-back savepoint INSERT (SQLite cannot ALTER
    a CHECK constraint in place).  Same recover-from-interruption logic as
    ``_ensure_wide_sim_type_constraint``.
    """
    if not _table_exists(conn, "exp_datasets"):
        return  # Phase-3 schema not applied yet; created with current CHECK later.

    try:
        conn.execute("SAVEPOINT _expsimtype_probe")
        conn.execute(
            "INSERT INTO exp_datasets (name, sim_type, created_at) "
            "VALUES ('__probe__jsr__', 'JSR', 'probe')"
        )
        conn.execute("ROLLBACK TO SAVEPOINT _expsimtype_probe")
        conn.execute("RELEASE SAVEPOINT _expsimtype_probe")
        log.debug("  exp_datasets.sim_type already accepts 'JSR'/'PFR' — skipping")
        return
    except sqlite3.IntegrityError:
        conn.execute("ROLLBACK TO SAVEPOINT _expsimtype_probe")
        conn.execute("RELEASE SAVEPOINT _expsimtype_probe")

    # Recover from an interrupted prior attempt: if exp_datasets was dropped but
    # the replacement _exp_datasets_new is still around, finish the rename.
    if not _table_exists(conn, "exp_datasets") and _table_exists(conn, "_exp_datasets_new"):
        log.warning("  Found leftover _exp_datasets_new from an interrupted "
                    "migration — recovering.")
        conn.execute("ALTER TABLE _exp_datasets_new RENAME TO exp_datasets")
        conn.commit()
        return

    log.info("  Widening exp_datasets.sim_type CHECK to include 'JSR'/'PFR' …")
    old_cols = [row[1] for row in conn.execute("PRAGMA table_info(exp_datasets)").fetchall()]
    copy_cols = ", ".join(old_cols)
    # FK-safe recreate. exp_data_points has FOREIGN KEY(dataset_id) REFERENCES
    # exp_datasets(id). Renaming the *original* table away would (with SQLite's
    # default legacy_alter_table=OFF) rewrite that child FK to the temp name, and
    # dropping the temp would then leave a dangling reference. Instead: build the
    # replacement under a temp name, drop the original, rename the replacement
    # back to exp_datasets. FKs are disabled for the swap — the PRAGMA is a no-op
    # inside a transaction, so it is set with no transaction open (executescript
    # commits any pending one first).
    conn.commit()
    conn.execute("PRAGMA foreign_keys=OFF")
    try:
        conn.executescript(f"""
            BEGIN TRANSACTION;

            CREATE TABLE _exp_datasets_new (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL UNIQUE,
                sim_type    TEXT    NOT NULL CHECK(sim_type IN ('IDT','LFS','JSR','PFR')),
                fuel_name   TEXT,
                source_ref  TEXT,
                n_points    INTEGER NOT NULL DEFAULT 0,
                filepath    TEXT,
                created_at  TEXT    NOT NULL
            );

            INSERT INTO _exp_datasets_new ({copy_cols})
                SELECT {copy_cols} FROM exp_datasets;

            DROP TABLE exp_datasets;

            ALTER TABLE _exp_datasets_new RENAME TO exp_datasets;

            COMMIT;
        """)
    finally:
        conn.execute("PRAGMA foreign_keys=ON")
    log.info("  exp_datasets.sim_type constraint updated.")


def _add_species_transport_columns(conn: sqlite3.Connection) -> None:
    """
    Add Lennard-Jones transport columns to the ``species`` table (Future.md
    pt. 4).  Additive ALTERs — no table recreation, so no FK concerns.
    ``transport_source`` records provenance: 'chemkin' / 'cantera' /
    'estimated:<method>'.
    """
    if not _table_exists(conn, "species"):
        return
    cols = {
        "transport_geometry":     "TEXT",   # 'atom' | 'linear' | 'nonlinear'
        "well_depth_K":           "REAL",   # eps/kB
        "diameter_A":             "REAL",   # sigma
        "dipole_D":               "REAL",
        "polarizability_A3":      "REAL",
        "rot_relax":              "REAL",
        "transport_source":       "TEXT",
    }
    added = []
    for col, decl in cols.items():
        if not _column_exists(conn, "species", col):
            conn.execute(f"ALTER TABLE species ADD COLUMN {col} {decl}")
            added.append(col)
    if added:
        conn.commit()
        log.info("  Added species transport columns: %s", ", ".join(added))
    else:
        log.debug("  species transport columns already present — skipping")


def _add_temp_target_class_column(conn: sqlite3.Connection) -> None:
    """Add exp_data_points.temp_target_class TEXT column if not already present."""
    if not _column_exists(conn, "exp_data_points", "temp_target_class"):
        conn.execute("ALTER TABLE exp_data_points ADD COLUMN temp_target_class TEXT")
        conn.commit()
        log.info("  Added exp_data_points.temp_target_class")
    else:
        log.debug("  exp_data_points.temp_target_class already exists — skipping")


def _ensure_wide_sim_type_constraint(conn: sqlite3.Connection) -> None:
    """
    Recreate simulation_runs if its sim_type CHECK does not yet include
    'JSR', 'PFR', 'FLUX'.  Idempotent — probed via a savepoint INSERT that is
    rolled back (SQLite cannot ALTER a CHECK constraint in place).
    """
    if not _table_exists(conn, "simulation_runs"):
        # Phase-3 schema not yet applied — nothing to widen; upgrade_db() /
        # schema_phase3.upgrade_db() will create it with the current CHECK later.
        return

    try:
        conn.execute("SAVEPOINT _simtype_probe")
        conn.execute(
            "INSERT INTO simulation_runs "
            "(mechanism_id, sim_type, T_K, P_atm, phi, fuel_name, "
            " fuel_composition, oxidizer_composition, converged, created_at) "
            "VALUES (0,'JSR',0,0,0,'x','x','x',1,'probe')"
        )
        conn.execute("ROLLBACK TO SAVEPOINT _simtype_probe")
        conn.execute("RELEASE SAVEPOINT _simtype_probe")
        log.debug("  simulation_runs.sim_type already accepts 'JSR'/'PFR'/'FLUX' — skipping")
        return
    except sqlite3.IntegrityError:
        conn.execute("ROLLBACK TO SAVEPOINT _simtype_probe")
        conn.execute("RELEASE SAVEPOINT _simtype_probe")

    # Recover from a previous interrupted widen attempt: if _simulation_runs_old
    # is still around, the prior run failed between RENAME and DROP. The
    # (possibly empty/partial) `simulation_runs` is disposable — the real data
    # is still safe in _simulation_runs_old — so drop it and restart cleanly.
    if _table_exists(conn, "_simulation_runs_old"):
        log.warning("  Found leftover _simulation_runs_old from an interrupted "
                    "migration — recovering.")
        conn.execute("DROP TABLE IF EXISTS simulation_runs")
        conn.execute("ALTER TABLE _simulation_runs_old RENAME TO simulation_runs")
        conn.commit()

    log.info("  Widening simulation_runs.sim_type CHECK to include 'JSR'/'PFR'/'FLUX' …")
    # Copy by explicit column name (not `SELECT *`) so this is robust to
    # whichever nullable columns (e.g. extra_json) have or haven't been
    # added yet on this DB — avoids a column-count mismatch either way.
    old_cols = [row[1] for row in conn.execute("PRAGMA table_info(simulation_runs)").fetchall()]
    copy_cols = ", ".join(old_cols)

    # Wrapped in an explicit transaction: executescript() does not treat a
    # multi-statement script as atomic, so without BEGIN/COMMIT a failure
    # partway (e.g. a future column-list mismatch) would leave the RENAME
    # applied but the CREATE/INSERT/DROP undone — the recovery block above
    # exists specifically because an earlier version of this function did
    # exactly that on a live test DB.
    conn.executescript(f"""
        BEGIN TRANSACTION;

        ALTER TABLE simulation_runs RENAME TO _simulation_runs_old;

        CREATE TABLE simulation_runs (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            mechanism_id            INTEGER NOT NULL,
            sim_type                TEXT    NOT NULL
                CHECK(sim_type IN ('IDT','LFS','JSR','PFR','FLUX')),
            T_K                     REAL    NOT NULL,
            P_atm                   REAL    NOT NULL,
            phi                     REAL    NOT NULL,
            fuel_name               TEXT    NOT NULL,
            fuel_composition        TEXT    NOT NULL,
            oxidizer_composition    TEXT    NOT NULL,
            result                  REAL,
            converged               INTEGER NOT NULL DEFAULT 1,
            cpu_time_s              REAL,
            error_msg               TEXT,
            created_at              TEXT    NOT NULL,
            extra_json              TEXT,
            FOREIGN KEY (mechanism_id) REFERENCES mechanisms(id) ON DELETE CASCADE
        );

        INSERT INTO simulation_runs ({copy_cols})
            SELECT {copy_cols} FROM _simulation_runs_old;

        DROP TABLE _simulation_runs_old;

        CREATE INDEX IF NOT EXISTS idx_simruns_mech_type
            ON simulation_runs(mechanism_id, sim_type);

        CREATE INDEX IF NOT EXISTS idx_simruns_conditions
            ON simulation_runs(sim_type, T_K, P_atm, phi);

        COMMIT;
    """)
    log.info("  simulation_runs.sim_type constraint updated.")


_CREATE_SENSITIVITY_RUNS = """
CREATE TABLE IF NOT EXISTS sensitivity_runs (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    mechanism_id            INTEGER NOT NULL REFERENCES mechanisms(id) ON DELETE CASCADE,
    method                  TEXT    NOT NULL CHECK(method IN ('sobol','morris')),
    n_samples               INTEGER NOT NULL,
    n_failed_samples        INTEGER NOT NULL DEFAULT 0,
    reaction_ids_json        TEXT    NOT NULL,
    factor_bound            REAL    NOT NULL DEFAULT 2.0,
    T_K                     REAL    NOT NULL,
    P_atm                   REAL    NOT NULL,
    phi                     REAL    NOT NULL,
    fuel                    TEXT    NOT NULL,
    oxidizer                TEXT    NOT NULL,
    idt_temp_delta          REAL    NOT NULL DEFAULT 400.0,
    seed                    INTEGER,
    created_at              TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sensitivity_runs_mech
    ON sensitivity_runs(mechanism_id);

CREATE TABLE IF NOT EXISTS sobol_indices (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                  INTEGER NOT NULL REFERENCES sensitivity_runs(id) ON DELETE CASCADE,
    reaction_id             INTEGER REFERENCES reactions(id) ON DELETE SET NULL,
    parameter_name          TEXT    NOT NULL,
    S1                      REAL,
    S1_conf                 REAL,
    ST                      REAL,
    ST_conf                 REAL,
    mu_star                 REAL,
    mu_star_conf            REAL,
    sigma                   REAL
);

CREATE INDEX IF NOT EXISTS idx_sobol_indices_run
    ON sobol_indices(run_id);

CREATE TABLE IF NOT EXISTS design_matrices (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                  INTEGER NOT NULL REFERENCES sensitivity_runs(id) ON DELETE CASCADE,
    method                  TEXT    NOT NULL,
    n_samples               INTEGER NOT NULL,
    n_parameters            INTEGER NOT NULL,
    seed                    INTEGER,
    matrix_path             TEXT    NOT NULL,
    created_at              TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_design_matrices_run
    ON design_matrices(run_id);
"""


def run_gsa_migrations(conn: sqlite3.Connection) -> None:
    """
    Zero-RK GSA idempotent migration: add sensitivity_runs/sobol_indices/
    design_matrices tables (ZERORK_GSA_PLAN.md Phase 3).

    Mirrors the reduction_runs pattern (Phase 5B) — metadata + indices in
    the DB, the full SALib sample matrix as a file on disk (referenced by
    design_matrices.matrix_path) since it can be large (n_samples ×
    n_parameters floats) and has no per-row query need.
    """
    log.info("Running Zero-RK GSA migrations …")
    if not _table_exists(conn, "sensitivity_runs"):
        conn.executescript(_CREATE_SENSITIVITY_RUNS)
        log.info("  Created sensitivity_runs/sobol_indices/design_matrices tables")
    conn.commit()
    log.info("Zero-RK GSA migrations complete.")


_CREATE_SURROGATE_RUNS = """
CREATE TABLE IF NOT EXISTS surrogate_runs (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    mechanism_id            INTEGER NOT NULL REFERENCES mechanisms(id) ON DELETE CASCADE,
    qoi                     TEXT    NOT NULL,
    method                  TEXT    NOT NULL DEFAULT 'pce',
    poly_order              INTEGER NOT NULL,
    n_parameters            INTEGER NOT NULL,
    n_samples               INTEGER NOT NULL,
    n_failed_samples        INTEGER NOT NULL DEFAULT 0,
    n_terms                 INTEGER NOT NULL DEFAULT 0,
    r2_insample             REAL,
    qoi_mean                REAL,
    qoi_variance            REAL,
    log_transformed         INTEGER NOT NULL DEFAULT 1,
    factor_bound            REAL,
    T_K                     REAL,
    P_atm                   REAL,
    phi                     REAL,
    fuel                    TEXT,
    oxidizer                TEXT,
    parameter_names_json    TEXT    NOT NULL,
    sobol_main_json         TEXT,
    sobol_total_json        TEXT,
    active_subspace_json    TEXT,
    pce_exponents_json      TEXT,
    pce_coefficients_json   TEXT,
    seed                    INTEGER,
    created_at              TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_surrogate_runs_mech
    ON surrogate_runs(mechanism_id);
"""


def run_surrogate_migrations(conn: sqlite3.Connection) -> None:
    """
    gPC/active-subspace surrogate idempotent migration (Future.md pt. 7): add
    the ``surrogate_runs`` table.

    Single-table design (metadata + JSON blobs for parameter names, Sobol
    indices, the active-subspace eigendecomposition, and the portable monomial
    PCE) — mirrors the flux_runs style rather than child index tables, since a
    surrogate run's outputs are small and always read as a whole. The two
    ``pce_*_json`` columns store the surrogate in a form that Bayesian/MUM-PCE
    calibration (pt. 6) can reload and evaluate as a cheap forward model.
    """
    log.info("Running surrogate (gPC/active-subspace) migrations …")
    if not _table_exists(conn, "surrogate_runs"):
        conn.executescript(_CREATE_SURROGATE_RUNS)
        log.info("  Created surrogate_runs table")
    else:
        # Additive: earlier Phase-5 DBs predate the portable-PCE columns.
        for col, decl in (("pce_exponents_json", "TEXT"),
                          ("pce_coefficients_json", "TEXT"),
                          ("factor_bound", "REAL")):
            if not _column_exists(conn, "surrogate_runs", col):
                conn.execute(f"ALTER TABLE surrogate_runs ADD COLUMN {col} {decl}")
                log.info("  Added surrogate_runs.%s", col)
    conn.commit()
    log.info("Surrogate migrations complete.")


_CREATE_CALIBRATION_RUNS = """
CREATE TABLE IF NOT EXISTS calibration_runs (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    mechanism_id                INTEGER REFERENCES mechanisms(id) ON DELETE CASCADE,
    method                      TEXT    NOT NULL DEFAULT 'mumpce',
    n_targets                   INTEGER NOT NULL,
    n_parameters                INTEGER NOT NULL,
    chi2                        REAL,
    reduced_chi2                REAL,
    converged                   INTEGER NOT NULL DEFAULT 1,
    surrogate_run_ids_json      TEXT,
    parameter_names_json        TEXT    NOT NULL,
    prior_sigma_json            TEXT,
    posterior_mean_json         TEXT,
    posterior_sigma_json        TEXT,
    posterior_cov_json          TEXT,
    uncertainty_reduction_json  TEXT,
    mcmc_mean_json              TEXT,
    mcmc_sigma_json             TEXT,
    created_at                  TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_calibration_runs_mech
    ON calibration_runs(mechanism_id);
"""


def run_calibration_migrations(conn: sqlite3.Connection) -> None:
    """
    Bayesian / MUM-PCE calibration idempotent migration (Future.md pt. 6): add
    the ``calibration_runs`` table (posterior mean/covariance + fit diagnostics,
    JSON blobs — read as a whole like surrogate_runs).
    """
    log.info("Running calibration (MUM-PCE) migrations …")
    if not _table_exists(conn, "calibration_runs"):
        conn.executescript(_CREATE_CALIBRATION_RUNS)
        log.info("  Created calibration_runs table")
    conn.commit()
    log.info("Calibration migrations complete.")


_CREATE_B2BDC_RUNS = """
CREATE TABLE IF NOT EXISTS b2bdc_runs (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    mechanism_id                INTEGER REFERENCES mechanisms(id) ON DELETE CASCADE,
    n_constraints               INTEGER NOT NULL,
    n_parameters                INTEGER NOT NULL,
    consistency_measure         REAL,
    consistent                  INTEGER NOT NULL DEFAULT 0,
    bound_type                  TEXT    NOT NULL DEFAULT 'inner',
    surrogate_run_ids_json      TEXT,
    parameter_names_json        TEXT    NOT NULL,
    prior_bounds_json           TEXT,
    parameter_bounds_json       TEXT,
    parameter_reduction_json    TEXT,
    predictions_json            TEXT,
    notes_json                  TEXT,
    created_at                  TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_b2bdc_runs_mech
    ON b2bdc_runs(mechanism_id);
"""


def run_b2bdc_migrations(conn: sqlite3.Connection) -> None:
    """
    Bound-to-Bound Data Collaboration idempotent migration (Future.md pt. 5):
    add the ``b2bdc_runs`` table (consistency measure + feasible-set parameter/
    prediction bounds, JSON blobs — read as a whole like the other Phase-6/7 runs).
    """
    log.info("Running B2BDC migrations …")
    if not _table_exists(conn, "b2bdc_runs"):
        conn.executescript(_CREATE_B2BDC_RUNS)
        log.info("  Created b2bdc_runs table")
    conn.commit()
    log.info("B2BDC migrations complete.")


_CREATE_CSP_RUNS = """
CREATE TABLE IF NOT EXISTS csp_runs (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    mechanism_id            INTEGER REFERENCES mechanisms(id) ON DELETE CASCADE,
    eval_state              TEXT    NOT NULL,
    state_T_K               REAL,
    T_K                     REAL,
    P_atm                   REAL,
    phi                     REAL,
    fuel                    TEXT,
    oxidizer                TEXT,
    n_species               INTEGER NOT NULL,
    n_reactions             INTEGER NOT NULL,
    jacobian_analytic       INTEGER NOT NULL DEFAULT 1,
    tau_threshold           REAL,
    fastest_tau             REAL,
    slowest_tau             REAL,
    n_exhausted_modes       INTEGER,
    n_qss                   INTEGER,
    mode_timescales_json    TEXT,
    qss_candidates_json     TEXT,
    excluded_species_json   TEXT,
    notes_json              TEXT,
    created_at              TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_csp_runs_mech ON csp_runs(mechanism_id);
"""


_CREATE_LUMPING_RUNS = """
CREATE TABLE IF NOT EXISTS lumping_runs (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    mechanism_id            INTEGER REFERENCES mechanisms(id) ON DELETE CASCADE,
    method                  TEXT    NOT NULL DEFAULT 'thermo_lumping',
    dHf_tol_kJ              REAL,
    dS_tol                  REAL,
    min_atoms               INTEGER,
    n_species_full          INTEGER NOT NULL,
    n_lumps                 INTEGER NOT NULL,
    n_species_removed       INTEGER,
    n_species_lumped        INTEGER,
    groups_json             TEXT,
    excluded_species_json   TEXT,
    notes_json              TEXT,
    created_at              TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_lumping_runs_mech ON lumping_runs(mechanism_id);
"""


def run_reduction_analysis_migrations(conn: sqlite3.Connection) -> None:
    """
    Phase-8 (FEATURE_ROADMAP #10) idempotent migration: add the ``csp_runs`` and
    ``lumping_runs`` tables (CSP timescale / QSS-species diagnostics and a-priori
    thermodynamic lumping proposals).  JSON blobs — read as a whole like the
    other reduction/analysis run tables.
    """
    log.info("Running reduction-analysis (CSP/lumping) migrations …")
    if not _table_exists(conn, "csp_runs"):
        conn.executescript(_CREATE_CSP_RUNS)
        log.info("  Created csp_runs table")
    if not _table_exists(conn, "lumping_runs"):
        conn.executescript(_CREATE_LUMPING_RUNS)
        log.info("  Created lumping_runs table")
    conn.commit()
    log.info("Reduction-analysis migrations complete.")


# Widen mechanisms.format to include 'reduced' (materialized skeletal / lumped /
# QSSA subset mechanisms produced by reduction/materialize.py).
_WIDEN_MECHANISMS_FORMAT_REDUCED = """
PRAGMA foreign_keys = OFF;

CREATE TABLE IF NOT EXISTS mechanisms_new (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,
    format        TEXT    NOT NULL CHECK(format IN ('chemkin','cantera_yaml','cti','fused','optimized','unified','reduced')),
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


_CREATE_ERROR_MAP_RUNS = """
CREATE TABLE IF NOT EXISTS error_map_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    detailed_mech_id    INTEGER REFERENCES mechanisms(id) ON DELETE CASCADE,
    skeletal_mech_id    INTEGER REFERENCES mechanisms(id) ON DELETE CASCADE,
    sim_type            TEXT    NOT NULL,
    fuel                TEXT,
    oxidizer            TEXT,
    tolerance           REAL,
    n_conditions        INTEGER NOT NULL,
    n_evaluated         INTEGER,
    max_rel_err         REAL,
    mean_rel_err        REAL,
    max_log_err         REAL,
    frac_within_tol     REAL,
    cells_json          TEXT,
    worst_cell_json     TEXT,
    notes_json          TEXT,
    created_at          TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_error_map_runs_det ON error_map_runs(detailed_mech_id);
"""


def run_error_map_migrations(conn: sqlite3.Connection) -> None:
    """
    FEATURE_ROADMAP #11a idempotent migration: add the ``error_map_runs`` table
    (skeletal-vs-detailed error atlas over a (T,P,φ) grid; per-cell errors as a
    JSON blob — read as a whole like the other analysis run tables).
    """
    log.info("Running error-map migrations …")
    if not _table_exists(conn, "error_map_runs"):
        conn.executescript(_CREATE_ERROR_MAP_RUNS)
        log.info("  Created error_map_runs table")
    conn.commit()
    log.info("Error-map migrations complete.")


_CREATE_TDAC_RUNS = """
CREATE TABLE IF NOT EXISTS tdac_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    mechanism_id        INTEGER REFERENCES mechanisms(id) ON DELETE CASCADE,
    T_K                 REAL,
    P_atm               REAL,
    phi                 REAL,
    fuel                TEXT,
    oxidizer            TEXT,
    eps                 REAL,
    update_every        INTEGER,
    bin_T               REAL,
    n_species_full      INTEGER,
    n_reactions_full    INTEGER,
    idt_s               REAL,
    converged           INTEGER NOT NULL DEFAULT 0,
    n_updates           INTEGER,
    n_cache_hits        INTEGER,
    n_cache_builds      INTEGER,
    mean_active_species REAL,
    min_active_species  INTEGER,
    max_active_species  INTEGER,
    target_species_json TEXT,
    notes_json          TEXT,
    created_at          TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tdac_runs_mech ON tdac_runs(mechanism_id);
"""


_CREATE_DOE_RUNS = """
CREATE TABLE IF NOT EXISTS doe_runs (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    mechanism_id                INTEGER REFERENCES mechanisms(id) ON DELETE CASCADE,
    criterion                   TEXT    NOT NULL,
    n_candidates                INTEGER NOT NULL,
    n_selected                  INTEGER NOT NULL,
    logdet_prior                REAL,
    logdet_post                 REAL,
    trace_cov_prior             REAL,
    trace_cov_post              REAL,
    parameter_names_json        TEXT,
    prior_sigma_json            TEXT,
    candidate_run_ids_json      TEXT,
    selected_json               TEXT,
    ranked_candidates_json      TEXT,
    posterior_sigma_json        TEXT,
    uncertainty_reduction_json  TEXT,
    notes_json                  TEXT,
    created_at                  TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_doe_runs_mech ON doe_runs(mechanism_id);
"""


_CREATE_PARETO_RUNS = """
CREATE TABLE IF NOT EXISTS pareto_runs (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    mechanism_id            INTEGER REFERENCES mechanisms(id) ON DELETE CASCADE,
    n_objectives            INTEGER NOT NULL,
    n_generations           INTEGER NOT NULL,
    pop_size                INTEGER NOT NULL,
    n_front                 INTEGER NOT NULL,
    hypervolume             REAL,
    parameter_names_json    TEXT,
    objective_names_json    TEXT,
    objective_run_ids_json  TEXT,
    objective_targets_json  TEXT,
    front_json              TEXT,
    notes_json              TEXT,
    created_at              TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pareto_runs_mech ON pareto_runs(mechanism_id);
"""


_CREATE_FAMILY_SCAN_RUNS = """
CREATE TABLE IF NOT EXISTS family_scan_runs (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    mechanism_id            INTEGER REFERENCES mechanisms(id) ON DELETE CASCADE,
    n_species               INTEGER NOT NULL,
    n_enumerated            INTEGER NOT NULL,
    n_missing_reactions     INTEGER NOT NULL,
    n_missing_species       INTEGER NOT NULL,
    missing_reactions_json  TEXT,
    missing_species_json    TEXT,
    families_seen_json      TEXT,
    notes_json              TEXT,
    created_at              TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_family_scan_runs_mech ON family_scan_runs(mechanism_id);
"""


def run_family_scan_migrations(conn: sqlite3.Connection) -> None:
    """
    FEATURE_ROADMAP #2 idempotent migration: add the ``family_scan_runs`` table
    (reaction-family completeness diffs — missing reactions/species per mechanism).
    """
    log.info("Running family-scan migrations …")
    if not _table_exists(conn, "family_scan_runs"):
        conn.executescript(_CREATE_FAMILY_SCAN_RUNS)
        log.info("  Created family_scan_runs table")
    conn.commit()
    log.info("Family-scan migrations complete.")


def run_pareto_migrations(conn: sqlite3.Connection) -> None:
    """
    FEATURE_ROADMAP #9 idempotent migration: add the ``pareto_runs`` table
    (multi-objective NSGA-II non-dominated fronts + hypervolume).
    """
    log.info("Running Pareto migrations …")
    if not _table_exists(conn, "pareto_runs"):
        conn.executescript(_CREATE_PARETO_RUNS)
        log.info("  Created pareto_runs table")
    conn.commit()
    log.info("Pareto migrations complete.")


def run_doe_migrations(conn: sqlite3.Connection) -> None:
    """
    FEATURE_ROADMAP #8 idempotent migration: add the ``doe_runs`` table
    (optimal-experiment-design selections + prior/posterior precision metrics).
    """
    log.info("Running DoE migrations …")
    if not _table_exists(conn, "doe_runs"):
        conn.executescript(_CREATE_DOE_RUNS)
        log.info("  Created doe_runs table")
    conn.commit()
    log.info("DoE migrations complete.")


def run_tdac_migrations(conn: sqlite3.Connection) -> None:
    """
    FEATURE_ROADMAP #11b idempotent migration: add the ``tdac_runs`` table
    (tabulated dynamic adaptive chemistry batch runs — accuracy + efficiency
    metrics per reactor).
    """
    log.info("Running TDAC migrations …")
    if not _table_exists(conn, "tdac_runs"):
        conn.executescript(_CREATE_TDAC_RUNS)
        log.info("  Created tdac_runs table")
    conn.commit()
    log.info("TDAC migrations complete.")


def run_reduction_materialize_migrations(conn: sqlite3.Connection) -> None:
    """
    Widen ``mechanisms.format`` CHECK to include ``'reduced'`` so a materialized
    skeletal/lumped/QSSA subset mechanism can be stored as a first-class
    mechanism row.  Idempotent — probes with a rolled-back INSERT (mirrors
    ``_ensure_wide_format_constraint``) and rebuilds the table only if needed.
    """
    try:
        conn.execute("SAVEPOINT _fmt_red_probe")
        conn.execute(
            "INSERT INTO mechanisms (name, format, loaded_at) VALUES ('_probe','reduced','x')"
        )
        conn.execute("ROLLBACK TO SAVEPOINT _fmt_red_probe")
        conn.execute("RELEASE SAVEPOINT _fmt_red_probe")
        log.debug("  mechanisms.format already accepts 'reduced' — skipping widen")
    except sqlite3.IntegrityError:
        conn.execute("ROLLBACK TO SAVEPOINT _fmt_red_probe")
        conn.execute("RELEASE SAVEPOINT _fmt_red_probe")
        log.info("  Widening mechanisms.format CHECK to include 'reduced' …")
        conn.executescript(_WIDEN_MECHANISMS_FORMAT_REDUCED)
    conn.commit()


_CREATE_REDUCTION_MULTI_CONDITION_RUNS = """
CREATE TABLE IF NOT EXISTS reduction_multi_condition_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    reduction_run_id    INTEGER REFERENCES reduction_runs(id) ON DELETE CASCADE,
    conditions_json     TEXT    NOT NULL,
    created_at          TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_reduction_multi_condition_runs_run
    ON reduction_multi_condition_runs(reduction_run_id);
"""


def run_multi_condition_migrations(conn: sqlite3.Connection) -> None:
    """
    FEATURE_GAPS_FROM_MONOLITH.md ranked candidate #5 idempotent migration:
    add the ``reduction_multi_condition_runs`` table, which links one
    unioned multi-condition reduction (see
    ``rmip.reduction.multi_condition.reduce_multi_condition``) back to the
    single ``reduction_runs`` row its union was persisted under, plus the
    JSON-serialised list of ``SimCondition``s that were unioned together.
    """
    log.info("Running multi-condition-reduction migrations …")
    if not _table_exists(conn, "reduction_multi_condition_runs"):
        conn.executescript(_CREATE_REDUCTION_MULTI_CONDITION_RUNS)
        log.info("  Created reduction_multi_condition_runs table")
    conn.commit()
    log.info("Multi-condition-reduction migrations complete.")


def run_canonical_key_rebuild_migration(conn: sqlite3.Connection) -> None:
    """
    Data-repair migration (not a schema change) for two parse-time fixes
    that were never backfilled into previously-ingested rows:

    1. **canonical_key** — ``core/parser.py``'s DB-build path opts into
       ``canonicalize.make_reaction_canonical_key(..., cancel_self_third_body=
       True)`` (the self-third-body-cancellation fix, bug.md 2026-07-17), but
       ``canonical_key`` is computed once at parse time and persisted — any
       DB parsed before that fix landed still carries stale keys. Stale keys
       mean a self-third-body reaction (``2H+H2<=>2H2``) and its generic-M
       twin (``H2+M<=>2H+M``) canonicalize to *different* keys instead of
       matching, so fusion's duplicate-flag logic
       (``analysis/fusion.py`` — driven purely by ``canonical_key``
       collisions) never flags them as duplicates, and the resulting
       Cantera-YAML export fails to load with "Undeclared duplicate
       reactions" (bug.md, the canonical fused-mechanism-won't-load bug).
       Rebuilt here from each reaction's stored ``stoichiometry`` rows
       (species SMILES + coefficients + role) — the same inputs
       ``parser.py::_build_canonical_key`` uses, just read back from the DB
       instead of the in-memory parse result. Rows with any unresolved
       (NULL ``canonical_smiles``) participant are left untouched — nothing
       to canonicalize yet, matching ``_build_canonical_key``'s own
       None-on-any-missing-SMILES behavior.

    2. **formula_string** — M6 (bug.md): a generic-M parenthesized falloff
       (``H2O2(+M)<=>2OH(+M)``) used to lose its ``(+M)`` tag entirely on
       parse (``_build_formula_string`` only tagged ``specific_collider``,
       e.g. ``(+AR)``/``(+HE)``, never the generic case). Full re-parse from
       the original CHEMKIN source is the only way to recover an *arbitrary*
       lost tag in general, but this specific bug's signature is narrow and
       safely detectable from stored data alone: ``kinetics_type`` in
       ``('troe', 'lindemann')`` always implies a parenthesized falloff main
       line (Troe/Lindemann parameters require a ``LOW``/``TROE`` aux line,
       which is physically incompatible with a bare, unparenthesized
       third-body reaction) — so if such a row's ``formula_string`` contains
       no ``"(+"`` at all, it can only be the generic-M-tag-lost bug (a
       specific-collider reaction always kept its tag, even before the fix).
       Repaired by appending ``" (+M)"`` to both sides.

    Idempotent: canonical_key recompute is deterministic (safe to re-run);
    formula_string repair only ever matches rows still exhibiting the exact
    bug signature (already-repaired rows contain ``"(+"`` and are excluded
    by the same WHERE clause on the next run).
    """
    from rmip.core.canonicalize import make_reaction_canonical_key

    log.info("Running canonical_key/formula_string rebuild migration …")
    conn.row_factory = sqlite3.Row

    # -- formula_string repair (M6) --------------------------------------
    fs_rows = conn.execute(
        "SELECT id, formula_string, reversible FROM reactions "
        "WHERE kinetics_type IN ('troe','lindemann') AND has_third_body=1 "
        "AND formula_string NOT LIKE '%(+%'"
    ).fetchall()
    n_formula_fixed = 0
    for r in fs_rows:
        fs = r["formula_string"] or ""
        arrow = " <=> " if r["reversible"] else " => "
        if arrow not in fs:
            continue
        lhs, rhs = fs.split(arrow, 1)
        conn.execute(
            "UPDATE reactions SET formula_string=? WHERE id=?",
            (f"{lhs} (+M){arrow}{rhs} (+M)", r["id"]),
        )
        n_formula_fixed += 1

    # -- canonical_key rebuild --------------------------------------------
    rxn_rows = conn.execute("SELECT id, reversible FROM reactions").fetchall()
    n_key_updated = 0
    for rx in rxn_rows:
        stoich = conn.execute(
            "SELECT s.canonical_smiles AS smi, st.stoich_coeff AS coeff, "
            "st.role AS role FROM stoichiometry st "
            "JOIN species s ON s.id = st.species_id WHERE st.reaction_id=?",
            (rx["id"],),
        ).fetchall()
        r_smiles = [row["smi"] for row in stoich if row["role"] == "reactant"]
        r_coeffs = [row["coeff"] for row in stoich if row["role"] == "reactant"]
        p_smiles = [row["smi"] for row in stoich if row["role"] == "product"]
        p_coeffs = [row["coeff"] for row in stoich if row["role"] == "product"]
        if not r_smiles or not p_smiles or None in r_smiles or None in p_smiles:
            continue
        new_key = make_reaction_canonical_key(
            r_smiles, p_smiles, reversible=bool(rx["reversible"]),
            reactant_coeffs=r_coeffs, product_coeffs=p_coeffs,
            cancel_self_third_body=True,
        )
        if new_key is not None:
            conn.execute(
                "UPDATE reactions SET canonical_key=? WHERE id=?",
                (new_key, rx["id"]),
            )
            n_key_updated += 1

    conn.commit()
    log.info(
        "Canonical-key rebuild migration complete: %d formula_string rows "
        "repaired, %d canonical_key rows rebuilt.",
        n_formula_fixed, n_key_updated,
    )


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
