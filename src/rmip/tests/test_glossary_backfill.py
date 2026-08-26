"""
rmip/tests/test_glossary_backfill.py
=====================================
Regression coverage for KRUNAL_JULY_23.md point 6: fusing Davis-2005 +
Keromenes-2013 + Li-2015 produced "Fused_Mech (fused, 0 sp / 0 rxn)" with no
error. Root cause, confirmed against the real DB (see
rmip.core.glossary_backfill's module docstring for the full analysis): those
mechanisms were ingested via ``rmip parse`` without ``--glossary``, so every
species has a null inchi_key and every reaction a null canonical_key. This is
NOT the same bug as B4 (test_fusion_b4.py) -- B4's fallback (unresolved rows
classify UNIQUE per-mechanism rather than vanishing or colliding) is a
deliberate, correct safety net that this module does not touch. This module
fixes the actual cause: species that *could* have resolved (the glossary
already lists them) but never did because no glossary was supplied at parse
time.

Uses the same minimal hand-built-DB pattern as test_fusion_b4.py.
"""
from __future__ import annotations

import csv
import json
import sqlite3

import pytest


def _make_db(path: str) -> sqlite3.Connection:
    from rmip.database.schema import init_db
    from rmip.database.schema_phase3 import upgrade_db
    from rmip.database.migrations import run_phase3_migrations
    conn = init_db(path)
    upgrade_db(conn)
    run_phase3_migrations(conn)
    return conn


def _insert_mechanism(conn: sqlite3.Connection, name: str) -> int:
    cur = conn.execute(
        """INSERT INTO mechanisms (name, format, filepath, description,
               n_species, n_reactions, loaded_at, metadata_json)
           VALUES (?,?,?,?,?,?,?,?)""",
        (name, "chemkin", "/fake/path.inp", f"Test mechanism {name}", 0, 0,
         "2024-01-01T00:00:00Z", "{}"),
    )
    return cur.lastrowid


def _insert_species(conn, mech_id, name, inchi_key=None, formula=None,
                     canonical_smiles=None) -> int:
    cur = conn.execute(
        """INSERT INTO species (mechanism_id, name, inchi_key, formula,
               canonical_smiles, thermo_type, tmin_K, tmid_K, tmax_K,
               coeffs_low_json, coeffs_high_json)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (mech_id, name, inchi_key, formula, canonical_smiles, "nasa7",
         200.0, 1000.0, 5000.0,
         json.dumps([1.0, 0, 0, 0, 0, -100.0, 1.0]),
         json.dumps([1.0, 0, 0, 0, 0, -100.0, 1.0])),
    )
    return cur.lastrowid


def _insert_reaction(conn, mech_id, formula_string, canonical_key=None,
                     A_si=1e10, n=0.5, Ea_si=50000.0, reversible=1) -> int:
    cur = conn.execute(
        """INSERT INTO reactions (mechanism_id, formula_string, canonical_key,
               reversible, kinetics_type, A_si, n, Ea_si)
           VALUES (?,?,?,?,?,?,?,?)""",
        (mech_id, formula_string, canonical_key, reversible, "arrhenius",
         A_si, n, Ea_si),
    )
    return cur.lastrowid


def _insert_stoich(conn, rxn_id, sp_id, coef, role) -> None:
    conn.execute(
        "INSERT INTO stoichiometry (reaction_id, species_id, stoich_coeff, role) "
        "VALUES (?,?,?,?)", (rxn_id, sp_id, coef, role))


@pytest.fixture
def glossary_csv(tmp_path):
    path = tmp_path / "glossary.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "smiles"])
        w.writerow(["H2", "[H][H]"])
        w.writerow(["O2", "[O][O]"])
        w.writerow(["H2O", "O"])
    return str(path)


def _build_unglossaried_mech(conn, mech_id: int, suffix: str = ""):
    """Two H2/O2 species + one reaction, exactly as they'd be stored had
    `rmip parse` run with no --glossary (blank inchi_key/canonical_smiles/
    canonical_key everywhere) -- the actual state mechanisms 2/3/4 were
    found in."""
    h2 = _insert_species(conn, mech_id, "H2")
    o2 = _insert_species(conn, mech_id, "O2")
    r = _insert_reaction(conn, mech_id, f"H2+O2<=>H2O2{suffix}", canonical_key=None)
    _insert_stoich(conn, r, h2, 1.0, "reactant")
    _insert_stoich(conn, r, o2, 1.0, "reactant")
    return h2, o2, r


class TestBackfillSpecies:
    def test_resolves_previously_unglossaried_species(self, tmp_path, glossary_csv):
        db = str(tmp_path / "t.db")
        conn = _make_db(db)
        m1 = _insert_mechanism(conn, "M1")
        _build_unglossaried_mech(conn, m1)
        conn.commit()
        conn.close()

        from rmip.core.glossary_backfill import backfill_glossary
        summary = backfill_glossary(db, [m1], glossary_csv)

        assert summary[m1]["species_resolved"] == 2
        assert summary[m1]["species_still_unresolved"] == 0

        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        rows = {r["name"]: r for r in
                conn.execute("SELECT * FROM species WHERE mechanism_id=?", (m1,))}
        assert rows["H2"]["inchi_key"]
        assert rows["O2"]["inchi_key"]
        assert rows["H2"]["canonical_smiles"]
        assert rows["H2"]["glossary_matched"] == 1

    def test_never_overwrites_an_already_resolved_species(self, tmp_path, glossary_csv):
        db = str(tmp_path / "t.db")
        conn = _make_db(db)
        m1 = _insert_mechanism(conn, "M1")
        # H2 already resolved (e.g. a prior correct parse); O2 is not.
        _insert_species(conn, m1, "H2", inchi_key="PRE_EXISTING_KEY",
                        canonical_smiles="PRE_EXISTING_SMILES")
        _insert_species(conn, m1, "O2")
        conn.commit()
        conn.close()

        from rmip.core.glossary_backfill import backfill_glossary
        summary = backfill_glossary(db, [m1], glossary_csv)

        assert summary[m1]["species_resolved"] == 1        # only O2
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        rows = {r["name"]: r for r in
                conn.execute("SELECT * FROM species WHERE mechanism_id=?", (m1,))}
        assert rows["H2"]["inchi_key"] == "PRE_EXISTING_KEY"          # untouched
        assert rows["H2"]["canonical_smiles"] == "PRE_EXISTING_SMILES"
        assert rows["O2"]["inchi_key"] and rows["O2"]["inchi_key"] != "PRE_EXISTING_KEY"

    def test_species_not_in_glossary_stays_unresolved_without_error(
            self, tmp_path, glossary_csv):
        db = str(tmp_path / "t.db")
        conn = _make_db(db)
        m1 = _insert_mechanism(conn, "M1")
        _insert_species(conn, m1, "SomeExoticRadicalXYZ")
        conn.commit()
        conn.close()

        from rmip.core.glossary_backfill import backfill_glossary
        summary = backfill_glossary(db, [m1], glossary_csv)     # must not raise

        assert summary[m1]["species_resolved"] == 0
        assert summary[m1]["species_still_unresolved"] == 1

    def test_idempotent_second_run_changes_nothing(self, tmp_path, glossary_csv):
        db = str(tmp_path / "t.db")
        conn = _make_db(db)
        m1 = _insert_mechanism(conn, "M1")
        _build_unglossaried_mech(conn, m1)
        conn.commit()
        conn.close()

        from rmip.core.glossary_backfill import backfill_glossary
        first = backfill_glossary(db, [m1], glossary_csv)
        second = backfill_glossary(db, [m1], glossary_csv)
        assert first[m1]["species_resolved"] == 2
        assert second[m1]["species_resolved"] == 0             # nothing left to do
        assert second[m1]["reactions_resolved"] == 0


class TestBackfillReactions:
    def test_reaction_canonical_key_recomputed_once_species_resolved(
            self, tmp_path, glossary_csv):
        db = str(tmp_path / "t.db")
        conn = _make_db(db)
        m1 = _insert_mechanism(conn, "M1")
        _build_unglossaried_mech(conn, m1)
        conn.commit()
        conn.close()

        from rmip.core.glossary_backfill import backfill_glossary
        summary = backfill_glossary(db, [m1], glossary_csv)

        assert summary[m1]["reactions_resolved"] == 1
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        rxn = conn.execute(
            "SELECT canonical_key FROM reactions WHERE mechanism_id=?", (m1,)
        ).fetchone()
        assert rxn["canonical_key"]

    def test_reaction_stays_unresolved_if_a_participant_cannot_resolve(
            self, tmp_path, glossary_csv):
        db = str(tmp_path / "t.db")
        conn = _make_db(db)
        m1 = _insert_mechanism(conn, "M1")
        h2 = _insert_species(conn, m1, "H2")
        mystery = _insert_species(conn, m1, "NotInGlossary")
        r = _insert_reaction(conn, m1, "H2+NotInGlossary<=>Stuff", canonical_key=None)
        _insert_stoich(conn, r, h2, 1.0, "reactant")
        _insert_stoich(conn, r, mystery, 1.0, "reactant")
        conn.commit()
        conn.close()

        from rmip.core.glossary_backfill import backfill_glossary
        summary = backfill_glossary(db, [m1], glossary_csv)

        assert summary[m1]["reactions_resolved"] == 0
        assert summary[m1]["reactions_still_unresolved"] == 1


class TestBackfillFixesRealFusionBug:
    """The actual repro: two mechanisms sharing H2/O2, both parsed without a
    glossary (100% blank inchi_key/canonical_key, exactly as mechanisms
    2/3/4 were found). Before backfill, GraphComparator's B4 fallback
    correctly refuses to guess a cross-mechanism match (species classify
    UNIQUE per-mechanism). After backfill, the shared species/reaction
    should classify UNIVERSAL/shared instead -- proving real fusion, not
    concatenation, is now possible."""

    def test_compare_shows_zero_universal_before_backfill(self, tmp_path, glossary_csv):
        db = str(tmp_path / "t.db")
        conn = _make_db(db)
        m1 = _insert_mechanism(conn, "M1")
        m2 = _insert_mechanism(conn, "M2")
        _build_unglossaried_mech(conn, m1)
        _build_unglossaried_mech(conn, m2)
        conn.commit()
        conn.close()

        from rmip.analysis.graph_comparator import GraphComparator
        result = GraphComparator(db).compare([m1, m2])
        d = result.summary_dict()
        assert d["species"]["universal"] == 0
        assert d["species"]["unique"] == 4          # H2, O2 x2 mechanisms, all UNIQUE

    def test_compare_shows_universal_species_after_backfill(self, tmp_path, glossary_csv):
        db = str(tmp_path / "t.db")
        conn = _make_db(db)
        m1 = _insert_mechanism(conn, "M1")
        m2 = _insert_mechanism(conn, "M2")
        _build_unglossaried_mech(conn, m1)
        _build_unglossaried_mech(conn, m2)
        conn.commit()
        conn.close()

        from rmip.core.glossary_backfill import backfill_glossary
        from rmip.analysis.graph_comparator import GraphComparator
        backfill_glossary(db, [m1, m2], glossary_csv)

        result = GraphComparator(db).compare([m1, m2])
        d = result.summary_dict()
        assert d["species"]["universal"] == 2       # H2 and O2 now cross-match
        assert d["species"]["unique"] == 0

    def test_fusion_no_longer_produces_empty_mechanism_after_backfill(
            self, tmp_path, glossary_csv):
        """The literal KRUNAL_JULY_23.md repro shape: fuse two mechanisms
        that were parsed without a glossary. Must NOT come out 0 sp/0 rxn."""
        db = str(tmp_path / "t.db")
        conn = _make_db(db)
        m1 = _insert_mechanism(conn, "M1")
        m2 = _insert_mechanism(conn, "M2")
        _build_unglossaried_mech(conn, m1)
        _build_unglossaried_mech(conn, m2)
        conn.commit()
        conn.close()

        from rmip.core.glossary_backfill import backfill_glossary
        from rmip.analysis.graph_comparator import GraphComparator
        from rmip.analysis.fusion import FusionEngine

        backfill_glossary(db, [m1, m2], glossary_csv)
        cmp_result = GraphComparator(db).compare([m1, m2])
        result = FusionEngine(db).fuse(cmp_result, name="Fused_after_backfill")

        assert len(result.species_records) > 0
        assert len(result.reaction_records) > 0
        # the shared H2+O2<=>H2O2 reaction should be genuinely fused/copied,
        # not present twice as two independent UNIQUE records
        assert len(result.reaction_records) == 1


class TestResolveGlossaryCli:
    def test_cli_subcommand_backfills_and_exits_zero(self, tmp_path, glossary_csv):
        db = str(tmp_path / "t.db")
        conn = _make_db(db)
        m1 = _insert_mechanism(conn, "M1")
        _build_unglossaried_mech(conn, m1)
        conn.commit()
        conn.close()

        from rmip.cli import main
        rc = main(["resolve-glossary", "--mechs", str(m1), "--db", db,
                  "--glossary", glossary_csv])
        assert rc == 0

        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        rows = list(conn.execute("SELECT inchi_key FROM species WHERE mechanism_id=?", (m1,)))
        assert all(r["inchi_key"] for r in rows)
