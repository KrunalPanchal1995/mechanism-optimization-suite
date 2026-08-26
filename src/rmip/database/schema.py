"""
rmip/database/schema.py
=======================
SQLite DDL, connection helpers, and CRUD primitives for the core RMIP tables.

Design decisions (from CLAUDE.md):
  D3 — Single SQLite file per project, no SQLAlchemy, stdlib sqlite3 only.
       SI units for all numeric columns; original units stored side-by-side.
  D1 — Species identity: inchi_key (25-char RDKit hash).
  D2 — Reaction identity: canonical_key (sorted SMILES frozenset string).

Tables
------
  mechanisms
  species
  reactions
  stoichiometry
  third_body_efficiencies

Usage
-----
    import sqlite3
    from rmip.database.schema import init_db, insert_mechanism, get_mechanisms

    conn = init_db("project.db")
    mech_id = insert_mechanism(conn, name="H2 test", format="chemkin",
                               filepath="h2.inp")
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Connection / initialisation
# ---------------------------------------------------------------------------

_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS mechanisms (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,
    format        TEXT    NOT NULL CHECK(format IN ('chemkin','cantera_yaml','cti','fused','optimized','unified')),
    filepath      TEXT,
    description   TEXT,
    n_species     INTEGER,
    n_reactions   INTEGER,
    loaded_at     TEXT    NOT NULL,   -- ISO-8601
    metadata_json TEXT                -- arbitrary extra info (thermo_filepath etc.)
);

CREATE TABLE IF NOT EXISTS species (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    mechanism_id     INTEGER NOT NULL REFERENCES mechanisms(id) ON DELETE CASCADE,
    name             TEXT    NOT NULL,
    canonical_smiles TEXT,
    source_smiles    TEXT,
    inchi            TEXT,
    inchi_key        TEXT,
    formula          TEXT,
    mw_kg_mol        REAL,
    charge           INTEGER DEFAULT 0,
    glossary_matched INTEGER DEFAULT 0,
    thermo_type      TEXT,
    tmin_K           REAL,
    tmid_K           REAL,
    tmax_K           REAL,
    coeffs_low_json  TEXT,   -- JSON [7 floats]
    coeffs_high_json TEXT,   -- JSON [7 floats]
    h298_kJ_mol      REAL,
    s298_J_molK      REAL,
    thermo_comment   TEXT,
    UNIQUE (mechanism_id, name)
);

CREATE INDEX IF NOT EXISTS idx_species_mech    ON species(mechanism_id);
CREATE INDEX IF NOT EXISTS idx_species_inchi   ON species(inchi_key);

CREATE TABLE IF NOT EXISTS reactions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    mechanism_id        INTEGER NOT NULL REFERENCES mechanisms(id) ON DELETE CASCADE,
    formula_string      TEXT    NOT NULL,
    canonical_key       TEXT,
    reversible          INTEGER DEFAULT 1,
    family              TEXT,
    duplicate           INTEGER DEFAULT 0,
    kinetics_type       TEXT,   -- 'arrhenius'|'plog'|'troe'|'lindemann'|
                                --  'third_body'|'chebyshev'|'multi_arrhenius'
    pressure_dependent  INTEGER DEFAULT 0,
    has_third_body      INTEGER DEFAULT 0,
    A_si                REAL,
    A_orig              REAL,
    A_units             TEXT,
    n                   REAL,
    Ea_si               REAL,
    Ea_orig             REAL,
    Ea_units            TEXT,
    kinetics_data_json  TEXT,   -- full PLOG/Troe/Chebyshev data
    reaction_label      TEXT,   -- sequential ID: R1, R2, … (mechanism-local, file order)
    rmg_comment         TEXT
);

CREATE INDEX IF NOT EXISTS idx_rxn_mech    ON reactions(mechanism_id);
CREATE INDEX IF NOT EXISTS idx_rxn_cankey  ON reactions(canonical_key);
CREATE INDEX IF NOT EXISTS idx_rxn_formula ON reactions(mechanism_id, formula_string);

CREATE TABLE IF NOT EXISTS stoichiometry (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    reaction_id INTEGER NOT NULL REFERENCES reactions(id) ON DELETE CASCADE,
    species_id  INTEGER NOT NULL REFERENCES species(id)  ON DELETE CASCADE,
    stoich_coeff REAL   NOT NULL,
    role        TEXT    NOT NULL CHECK(role IN ('reactant','product'))
);

CREATE INDEX IF NOT EXISTS idx_stoich_rxn ON stoichiometry(reaction_id);
CREATE INDEX IF NOT EXISTS idx_stoich_sp  ON stoichiometry(species_id);

CREATE TABLE IF NOT EXISTS third_body_efficiencies (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    reaction_id INTEGER NOT NULL REFERENCES reactions(id)  ON DELETE CASCADE,
    species_id  INTEGER          REFERENCES species(id)    ON DELETE SET NULL,
    species_name TEXT,           -- fallback when species_id is NULL
    efficiency  REAL    NOT NULL
);
"""


def init_db(path: str) -> sqlite3.Connection:
    """
    Open (or create) the SQLite database, run DDL, and return a connection.

    Parameters
    ----------
    path : str
        File path for the SQLite database.

    Returns
    -------
    sqlite3.Connection
        Open connection with row_factory = sqlite3.Row.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_DDL)
    conn.commit()
    return conn


def get_connection(path: str) -> sqlite3.Connection:
    """
    Open an existing SQLite database without re-running DDL.

    Parameters
    ----------
    path : str
        File path for the SQLite database.

    Returns
    -------
    sqlite3.Connection
        Open connection with row_factory = sqlite3.Row.
    """
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ---------------------------------------------------------------------------
# Mechanisms CRUD
# ---------------------------------------------------------------------------

def insert_mechanism(
    conn:        sqlite3.Connection,
    name:        str,
    format:      str,
    filepath:    Optional[str]  = None,
    description: Optional[str]  = None,
    n_species:   Optional[int]  = None,
    n_reactions: Optional[int]  = None,
    metadata:    Optional[dict] = None,
) -> int:
    """Insert a mechanism row; return the new row id."""
    cur = conn.execute(
        """INSERT INTO mechanisms
           (name, format, filepath, description, n_species, n_reactions,
            loaded_at, metadata_json)
           VALUES (?,?,?,?,?,?,?,?)""",
        (name, format, filepath, description, n_species, n_reactions,
         _now(), json.dumps(metadata or {})),
    )
    conn.commit()
    return cur.lastrowid


def get_mechanism(conn: sqlite3.Connection, mech_id: int) -> Optional[sqlite3.Row]:
    """Return the mechanism row or None."""
    return conn.execute(
        "SELECT * FROM mechanisms WHERE id = ?", (mech_id,)
    ).fetchone()


def get_mechanisms(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    """Return all mechanism rows ordered by id."""
    return conn.execute("SELECT * FROM mechanisms ORDER BY id").fetchall()


def update_mechanism_counts(
    conn: sqlite3.Connection, mech_id: int,
    n_species: int, n_reactions: int,
) -> None:
    conn.execute(
        "UPDATE mechanisms SET n_species=?, n_reactions=? WHERE id=?",
        (n_species, n_reactions, mech_id),
    )
    conn.commit()


def delete_mechanism(conn: sqlite3.Connection, mech_id: int) -> None:
    conn.execute("DELETE FROM mechanisms WHERE id=?", (mech_id,))
    conn.commit()


# ---------------------------------------------------------------------------
# Species CRUD
# ---------------------------------------------------------------------------

def insert_species(conn: sqlite3.Connection, **kwargs) -> int:
    """
    Insert one species row.  `mechanism_id` and `name` are required.
    All other keyword arguments map to column names.

    Uses INSERT OR IGNORE — if a row with the same (mechanism_id, name)
    already exists, returns the existing id.
    """
    cols = list(kwargs.keys())
    vals = list(kwargs.values())
    sql = (
        f"INSERT OR IGNORE INTO species ({', '.join(cols)}) "
        f"VALUES ({', '.join('?' * len(cols))})"
    )
    cur = conn.execute(sql, vals)
    if cur.rowcount == 0:
        row = conn.execute(
            "SELECT id FROM species WHERE mechanism_id=? AND name=?",
            (kwargs.get("mechanism_id"), kwargs.get("name"))
        ).fetchone()
        return row["id"] if row else cur.lastrowid
    return cur.lastrowid


def get_species_for_mechanism(
    conn: sqlite3.Connection, mech_id: int
) -> List[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM species WHERE mechanism_id=? ORDER BY id", (mech_id,)
    ).fetchall()


def get_species_by_inchi_key(
    conn: sqlite3.Connection, inchi_key: str, mech_id: Optional[int] = None
) -> List[sqlite3.Row]:
    if mech_id is not None:
        return conn.execute(
            "SELECT * FROM species WHERE inchi_key=? AND mechanism_id=?",
            (inchi_key, mech_id),
        ).fetchall()
    return conn.execute(
        "SELECT * FROM species WHERE inchi_key=?", (inchi_key,)
    ).fetchall()


# ---------------------------------------------------------------------------
# Reactions CRUD
# ---------------------------------------------------------------------------

def insert_reaction(conn: sqlite3.Connection, **kwargs) -> int:
    """
    Insert one reaction row.
    Required kwargs: ``mechanism_id``, ``formula_string``.
    """
    cols = list(kwargs.keys())
    vals = list(kwargs.values())
    sql = (
        f"INSERT INTO reactions ({', '.join(cols)}) "
        f"VALUES ({', '.join('?' * len(cols))})"
    )
    cur = conn.execute(sql, vals)
    return cur.lastrowid


def get_reactions_for_mechanism(
    conn: sqlite3.Connection, mech_id: int
) -> List[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM reactions WHERE mechanism_id=? ORDER BY id", (mech_id,)
    ).fetchall()


def get_reaction_by_canonical_key(
    conn: sqlite3.Connection, canonical_key: str, mech_id: Optional[int] = None
) -> List[sqlite3.Row]:
    if mech_id is not None:
        return conn.execute(
            "SELECT * FROM reactions WHERE canonical_key=? AND mechanism_id=?",
            (canonical_key, mech_id),
        ).fetchall()
    return conn.execute(
        "SELECT * FROM reactions WHERE canonical_key=?", (canonical_key,)
    ).fetchall()


# ---------------------------------------------------------------------------
# Stoichiometry CRUD
# ---------------------------------------------------------------------------

def insert_stoichiometry(
    conn:         sqlite3.Connection,
    reaction_id:  int,
    species_id:   int,
    stoich_coeff: float,
    role:         str,  # 'reactant' | 'product'
) -> int:
    cur = conn.execute(
        "INSERT INTO stoichiometry (reaction_id, species_id, stoich_coeff, role) "
        "VALUES (?,?,?,?)",
        (reaction_id, species_id, stoich_coeff, role),
    )
    return cur.lastrowid


def get_stoichiometry_for_reaction(
    conn: sqlite3.Connection, reaction_id: int
) -> List[sqlite3.Row]:
    return conn.execute(
        """SELECT s.*, sp.name AS species_name, sp.inchi_key
           FROM stoichiometry s
           JOIN species sp ON sp.id = s.species_id
           WHERE s.reaction_id = ?
           ORDER BY s.role, s.id""",
        (reaction_id,),
    ).fetchall()


# ---------------------------------------------------------------------------
# Third-body efficiencies CRUD
# ---------------------------------------------------------------------------

def insert_third_body_efficiency(
    conn:         sqlite3.Connection,
    reaction_id:  int,
    efficiency:   float,
    species_id:   Optional[int]  = None,
    species_name: Optional[str]  = None,
) -> int:
    cur = conn.execute(
        "INSERT INTO third_body_efficiencies "
        "(reaction_id, species_id, species_name, efficiency) VALUES (?,?,?,?)",
        (reaction_id, species_id, species_name, efficiency),
    )
    return cur.lastrowid


def insert_stoichiometry_batch(
    conn: sqlite3.Connection,
    rows: list[tuple],
) -> None:
    """Insert multiple stoichiometry rows at once.

    Each tuple: (reaction_id, species_id, stoich_coeff, role)
    """
    conn.executemany(
        "INSERT INTO stoichiometry (reaction_id, species_id, stoich_coeff, role)"
        " VALUES (?, ?, ?, ?)",
        rows,
    )


def insert_third_body_batch(
    conn: sqlite3.Connection,
    rows: list[tuple],
) -> None:
    """Insert multiple third-body efficiency rows at once.

    Each tuple: (reaction_id, species_id, efficiency)
    """
    conn.executemany(
        "INSERT INTO third_body_efficiencies (reaction_id, species_id, efficiency)"
        " VALUES (?, ?, ?)",
        rows,
    )


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def get_mechanism_summary(
    conn: sqlite3.Connection, mech_id: int
) -> Dict[str, Any]:
    """Return a rich summary dict for a mechanism (used by CLI + parser)."""
    mech = get_mechanism(conn, mech_id)
    if mech is None:
        raise ValueError(f"Mechanism id={mech_id} not found")
    n_sp = conn.execute(
        "SELECT COUNT(*) FROM species WHERE mechanism_id=?", (mech_id,)
    ).fetchone()[0]
    n_rx = conn.execute(
        "SELECT COUNT(*) FROM reactions WHERE mechanism_id=?", (mech_id,)
    ).fetchone()[0]
    n_glossary = conn.execute(
        "SELECT COUNT(*) FROM species WHERE mechanism_id=? AND glossary_matched=1",
        (mech_id,),
    ).fetchone()[0]
    kinetics_types = conn.execute(
        "SELECT kinetics_type, COUNT(*) AS cnt "
        "FROM reactions WHERE mechanism_id=? GROUP BY kinetics_type",
        (mech_id,),
    ).fetchall()
    return {
        "id":              mech["id"],
        "name":            mech["name"],
        "format":          mech["format"],
        "filepath":        mech["filepath"],
        "description":     mech["description"],
        "n_species":       n_sp,
        "n_reactions":     n_rx,
        "loaded_at":       mech["loaded_at"],
        "glossary_matched_fraction": n_glossary / max(n_sp, 1),
        "kinetics_types":  {r["kinetics_type"]: r["cnt"] for r in kinetics_types},
        "metadata":        json.loads(mech["metadata_json"] or "{}"),
    }


# ---------------------------------------------------------------------------
# Batch inserter
# ---------------------------------------------------------------------------

class BatchInserter:
    """
    Context-manager that accumulates rows and flushes them in batches via
    ``executemany``.  Reduces round-trip overhead for large mechanisms.

    Usage::

        with BatchInserter(conn, "species", batch_size=500) as bi:
            for row in rows:
                bi.add(row)
    """

    def __init__(
        self,
        conn:       sqlite3.Connection,
        table:      str,
        batch_size: int = 500,
    ) -> None:
        self._conn       = conn
        self._table      = table
        self._batch_size = batch_size
        self._buffer: List[dict] = []
        self._cols:   Optional[List[str]] = None

    def add(self, row: dict) -> None:
        if self._cols is None:
            self._cols = list(row.keys())
        self._buffer.append(row)
        if len(self._buffer) >= self._batch_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        cols = self._cols
        sql = (
            f"INSERT INTO {self._table} ({', '.join(cols)}) "
            f"VALUES ({', '.join('?' * len(cols))})"
        )
        self._conn.executemany(sql, [
            [r.get(c) for c in cols] for r in self._buffer
        ])
        self._conn.commit()
        self._buffer.clear()

    def __enter__(self) -> "BatchInserter":
        return self

    def __exit__(self, *_) -> None:
        self._flush()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
