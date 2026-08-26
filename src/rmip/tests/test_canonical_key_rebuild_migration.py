"""
rmip/tests/test_canonical_key_rebuild_migration.py
====================================================
Regression coverage for ``run_canonical_key_rebuild_migration`` (Phase-2 of
the 2026-08 backlog program, bug.md): a data-repair migration that backfills
two parse-time fixes into previously-ingested DB rows that predate them --

  1. canonical_key rebuilt with cancel_self_third_body=True (the
     self-third-body-cancellation fix, bug.md 2026-07-17) so a self-collider
     reaction (``2H+H2<=>2H2``) matches its generic-M twin
     (``H2+M<=>2H+M``)'s canonical_key instead of getting a different, stale
     one -- this is what unblocks fusion's duplicate-flag logic from
     correctly flagging them, which is what makes the resulting Cantera-YAML
     load without an "Undeclared duplicate reactions" error.
  2. formula_string repaired for the M6 generic-M falloff collider-tag bug
     (a generic-M parenthesized falloff, e.g. H2O2(+M)<=>2OH(+M), that lost
     its "(+M)" tag entirely at parse time).
"""
from __future__ import annotations

import sqlite3

from rmip.database.migrations import run_canonical_key_rebuild_migration
from rmip.database.schema import init_db

# Real canonical SMILES -- doesn't matter which exact string, only that both
# reactions below use the SAME species identifiers (matches the pattern in
# test_canonicalize_self_third_body.py).
H_SMILES = "[H]"
H2_SMILES = "[H][H]"


def _make_db(path: str) -> sqlite3.Connection:
    conn = init_db(path)
    return conn


def _insert_mechanism(conn: sqlite3.Connection, name: str) -> int:
    conn.execute(
        "INSERT INTO mechanisms (name, format, loaded_at) VALUES (?,?,?)",
        (name, "chemkin", "2026-08-04T00:00:00Z"),
    )
    return conn.execute("SELECT id FROM mechanisms WHERE name=?", (name,)).fetchone()[0]


def _insert_species(conn: sqlite3.Connection, mech_id: int, name: str, smiles: str) -> int:
    conn.execute(
        "INSERT INTO species (mechanism_id, name, canonical_smiles) VALUES (?,?,?)",
        (mech_id, name, smiles),
    )
    return conn.execute(
        "SELECT id FROM species WHERE mechanism_id=? AND name=?", (mech_id, name)
    ).fetchone()[0]


def _insert_reaction(
    conn: sqlite3.Connection, mech_id: int, formula_string: str, canonical_key,
    reversible=1, kinetics_type="third_body", has_third_body=1,
) -> int:
    cur = conn.execute(
        "INSERT INTO reactions (mechanism_id, formula_string, canonical_key, "
        "reversible, kinetics_type, has_third_body) VALUES (?,?,?,?,?,?)",
        (mech_id, formula_string, canonical_key, reversible, kinetics_type, has_third_body),
    )
    return cur.lastrowid


def _insert_stoich(conn: sqlite3.Connection, rxn_id: int, sp_id: int, coeff: float, role: str):
    conn.execute(
        "INSERT INTO stoichiometry (reaction_id, species_id, stoich_coeff, role) "
        "VALUES (?,?,?,?)",
        (rxn_id, sp_id, coeff, role),
    )


class TestCanonicalKeyRebuild:
    def test_self_third_body_and_generic_m_converge_to_same_key(self, tmp_path):
        db_path = str(tmp_path / "ck.db")
        conn = _make_db(db_path)
        mid = _insert_mechanism(conn, "TestMech")
        h = _insert_species(conn, mid, "H", H_SMILES)
        h2 = _insert_species(conn, mid, "H2", H2_SMILES)

        # Generic-M form: H2+M<=>2H+M -- M stripped at parse time, never
        # appears in stoichiometry (matches _chemkin_parser.py behavior).
        r_generic = _insert_reaction(conn, mid, "H2 <=> 2 H", "STALE-KEY-1")
        _insert_stoich(conn, r_generic, h2, 1.0, "reactant")
        _insert_stoich(conn, r_generic, h, 2.0, "product")

        # Explicit self-third-body form: 2H+H2<=>2H2.
        r_explicit = _insert_reaction(conn, mid, "2 H + H2 <=> 2 H2", "STALE-KEY-2")
        _insert_stoich(conn, r_explicit, h, 2.0, "reactant")
        _insert_stoich(conn, r_explicit, h2, 1.0, "reactant")
        _insert_stoich(conn, r_explicit, h2, 2.0, "product")

        conn.commit()

        # Pre-migration: stale keys differ (the exact bug that broke fusion's
        # duplicate detection).
        assert conn.execute(
            "SELECT canonical_key FROM reactions WHERE id=?", (r_generic,)
        ).fetchone()[0] != conn.execute(
            "SELECT canonical_key FROM reactions WHERE id=?", (r_explicit,)
        ).fetchone()[0]

        run_canonical_key_rebuild_migration(conn)

        key_generic = conn.execute(
            "SELECT canonical_key FROM reactions WHERE id=?", (r_generic,)
        ).fetchone()[0]
        key_explicit = conn.execute(
            "SELECT canonical_key FROM reactions WHERE id=?", (r_explicit,)
        ).fetchone()[0]
        assert key_generic is not None
        assert key_generic == key_explicit
        conn.close()

    def test_unresolved_species_left_untouched(self, tmp_path):
        """A reaction with a NULL canonical_smiles participant (unresolved
        glossary) must keep its existing canonical_key, not get overwritten
        with something bogus."""
        db_path = str(tmp_path / "unresolved.db")
        conn = _make_db(db_path)
        mid = _insert_mechanism(conn, "UnresolvedMech")
        sp_unresolved = _insert_species(conn, mid, "MYSTERY", None)
        sp_known = _insert_species(conn, mid, "H", H_SMILES)

        rxn = _insert_reaction(conn, mid, "MYSTERY <=> H", "ORIGINAL-KEY")
        _insert_stoich(conn, rxn, sp_unresolved, 1.0, "reactant")
        _insert_stoich(conn, rxn, sp_known, 1.0, "product")
        conn.commit()

        run_canonical_key_rebuild_migration(conn)

        key = conn.execute(
            "SELECT canonical_key FROM reactions WHERE id=?", (rxn,)
        ).fetchone()[0]
        assert key == "ORIGINAL-KEY"
        conn.close()

    def test_idempotent(self, tmp_path):
        db_path = str(tmp_path / "idem.db")
        conn = _make_db(db_path)
        mid = _insert_mechanism(conn, "IdemMech")
        h = _insert_species(conn, mid, "H", H_SMILES)
        h2 = _insert_species(conn, mid, "H2", H2_SMILES)
        rxn = _insert_reaction(conn, mid, "H2 <=> 2 H", "STALE")
        _insert_stoich(conn, rxn, h2, 1.0, "reactant")
        _insert_stoich(conn, rxn, h, 2.0, "product")
        conn.commit()

        run_canonical_key_rebuild_migration(conn)
        key_1 = conn.execute(
            "SELECT canonical_key FROM reactions WHERE id=?", (rxn,)
        ).fetchone()[0]

        run_canonical_key_rebuild_migration(conn)
        key_2 = conn.execute(
            "SELECT canonical_key FROM reactions WHERE id=?", (rxn,)
        ).fetchone()[0]

        assert key_1 == key_2
        conn.close()


class TestFormulaStringRepair:
    def test_generic_m_falloff_missing_tag_gets_repaired(self, tmp_path):
        db_path = str(tmp_path / "fs.db")
        conn = _make_db(db_path)
        mid = _insert_mechanism(conn, "FalloffMech")
        rxn = _insert_reaction(
            conn, mid, "H2O2 <=> 2 OH", None,
            kinetics_type="troe", has_third_body=1,
        )
        conn.commit()

        run_canonical_key_rebuild_migration(conn)

        fs = conn.execute(
            "SELECT formula_string FROM reactions WHERE id=?", (rxn,)
        ).fetchone()[0]
        assert fs == "H2O2 (+M) <=> 2 OH (+M)"
        conn.close()

    def test_specific_collider_formula_string_untouched(self, tmp_path):
        """A specific-collider falloff (e.g. "(+AR)") already carried its
        tag correctly even before the M6 fix -- must be left alone (the
        WHERE clause excludes any formula_string already containing "(+")."""
        db_path = str(tmp_path / "fs_ar.db")
        conn = _make_db(db_path)
        mid = _insert_mechanism(conn, "ArMech")
        rxn = _insert_reaction(
            conn, mid, "H2O2 (+AR) <=> 2 OH (+AR)", None,
            kinetics_type="troe", has_third_body=1,
        )
        conn.commit()

        run_canonical_key_rebuild_migration(conn)

        fs = conn.execute(
            "SELECT formula_string FROM reactions WHERE id=?", (rxn,)
        ).fetchone()[0]
        assert fs == "H2O2 (+AR) <=> 2 OH (+AR)"
        conn.close()

    def test_non_falloff_third_body_untouched(self, tmp_path):
        """A bare, non-falloff third-body reaction (kinetics_type=
        'third_body', not 'troe'/'lindemann') is outside this bug's scope --
        must not get a spurious "(+M)" appended."""
        db_path = str(tmp_path / "fs_bare.db")
        conn = _make_db(db_path)
        mid = _insert_mechanism(conn, "BareMech")
        rxn = _insert_reaction(
            conn, mid, "H2 <=> 2 H", None,
            kinetics_type="third_body", has_third_body=1,
        )
        conn.commit()

        run_canonical_key_rebuild_migration(conn)

        fs = conn.execute(
            "SELECT formula_string FROM reactions WHERE id=?", (rxn,)
        ).fetchone()[0]
        assert fs == "H2 <=> 2 H"
        conn.close()

    def test_idempotent(self, tmp_path):
        db_path = str(tmp_path / "fs_idem.db")
        conn = _make_db(db_path)
        mid = _insert_mechanism(conn, "IdemFsMech")
        rxn = _insert_reaction(
            conn, mid, "CO + O <=> CO2", None,
            kinetics_type="lindemann", has_third_body=1,
        )
        conn.commit()

        run_canonical_key_rebuild_migration(conn)
        run_canonical_key_rebuild_migration(conn)

        fs = conn.execute(
            "SELECT formula_string FROM reactions WHERE id=?", (rxn,)
        ).fetchone()[0]
        assert fs == "CO + O (+M) <=> CO2 (+M)"
        conn.close()
