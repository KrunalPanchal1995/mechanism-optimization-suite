"""
mso_project.schema
===================
SQLite DDL and connection helpers for a project's ``artifacts.db``.

Follows the same conventions as ``src/rmip/database/schema.py`` (stdlib
``sqlite3`` only, no ORM, ``row_factory = sqlite3.Row``, WAL mode, DDL as one
``executescript`` block) but is a distinct, unrelated database: RMIP's is
mechanism-centric (species/reactions/stoichiometry); this one is
artifact-centric (every file a run produced, wherever it lives on disk).

Tables
------
  project_info  -- single row mirroring project.yaml (name/description/tags)
  runs          -- one row per captured run (pipeline, label, working_dir, status)
  artifacts     -- one row per tracked file (path, role, kind, run_id FK)

Usage
-----
    from mso_project.schema import init_db, get_connection

    conn = init_db("/data/PROJECTS/AMMONIA_2026/artifacts.db")
    ...
    conn = get_connection("/data/PROJECTS/AMMONIA_2026/artifacts.db")  # reopen later
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1

_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS project_info (
    id            INTEGER PRIMARY KEY CHECK (id = 1),  -- single-row table
    name          TEXT    NOT NULL,
    description   TEXT,
    tags_json     TEXT,             -- JSON list of strings
    created_at    TEXT    NOT NULL, -- ISO-8601
    schema_version INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT    PRIMARY KEY,  -- str(RunRecord.run_dir()) -- stable, no new ID scheme
    pipeline      TEXT    NOT NULL,     -- e.g. "run_script", "forward_uq", "mso_compare"
    label         TEXT    NOT NULL,
    working_dir   TEXT    NOT NULL,
    started_at    TEXT    NOT NULL,     -- ISO-8601
    finished_at   TEXT,
    status        TEXT    NOT NULL DEFAULT 'running'
                          CHECK (status IN ('running', 'done', 'failed', 'unknown')),
    record_path   TEXT                 -- path to the RunRecord's own record.json, if saved
);

CREATE INDEX IF NOT EXISTS idx_runs_pipeline ON runs(pipeline);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           TEXT             REFERENCES runs(run_id) ON DELETE SET NULL,
    path             TEXT    NOT NULL,  -- absolute path
    role             TEXT    NOT NULL DEFAULT 'unclassified',
    kind             TEXT    NOT NULL DEFAULT 'other',
                              -- csv|png|pdf|pkl|yaml|json|txt|log|doc|notebook|other
    size_bytes       INTEGER,
    checksum_sha256  TEXT,             -- opt-in only, NULL for most bulk-registered files
    mtime            REAL,             -- POSIX timestamp of the file at registration time
    registered_at    TEXT    NOT NULL, -- ISO-8601
    tags_json        TEXT,             -- JSON list of strings
    report_include   TEXT    NOT NULL DEFAULT 'auto'
                              CHECK (report_include IN ('auto', 'main', 'supplementary', 'exclude')),
    UNIQUE (run_id, path)
);

CREATE INDEX IF NOT EXISTS idx_artifacts_run  ON artifacts(run_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_role ON artifacts(role);
CREATE INDEX IF NOT EXISTS idx_artifacts_kind ON artifacts(kind);
"""


def init_db(path: str) -> sqlite3.Connection:
    """Open (or create) ``artifacts.db``, run DDL, and return a connection."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_DDL)
    conn.commit()
    return conn


def get_connection(path: str) -> sqlite3.Connection:
    """Open an existing ``artifacts.db`` without re-running DDL."""
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn
