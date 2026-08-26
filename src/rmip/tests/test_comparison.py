"""
rmip/tests/test_comparison.py
==============================
Regression coverage for B3 (review.md, 2026-07-14): ComparisonEngine had no
test coverage at all. `_compare_species`/`_compare_reactions` filtered out
any row whose inchi_key/canonical_key was NULL (i.e. anything the glossary
didn't resolve) at the SQL level -- such rows silently vanished from every
count with no warning. At FFCM1's real glossary hit rate (28.9%), 27/38
species and ~93% of reactions disappeared from the report; with no glossary
at all, `compare` returned universal=0, unique=0/0 (arithmetically
impossible for two non-empty mechanisms) and still exited 0. Even
self-compare (same mechanism against a copy of itself) returned zeros.

Fix: unresolved rows are kept and fall back to a per-mechanism name/formula
key (so they show up as "unique" instead of disappearing), coverage % is
reported, and 0%-resolved is a hard error instead of a silent all-zero
report.
"""
from __future__ import annotations

import sqlite3
from typing import List, Optional

import pytest


def _make_db(path: str) -> sqlite3.Connection:
    from rmip.database.schema import init_db
    return init_db(path)


def _insert_mechanism(conn: sqlite3.Connection, name: str) -> int:
    cur = conn.execute(
        """
        INSERT INTO mechanisms (name, format, filepath, description,
                                n_species, n_reactions, loaded_at, metadata_json)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (name, "chemkin", "/fake/path.inp", f"Test mechanism {name}", 0, 0,
         "2024-01-01T00:00:00Z", "{}"),
    )
    return cur.lastrowid


def _insert_species(
    conn: sqlite3.Connection, mech_id: int, name: str,
    inchi_key: Optional[str], formula: str = "H2",
) -> int:
    cur = conn.execute(
        """
        INSERT INTO species (mechanism_id, name, inchi_key, formula,
                             coeffs_low_json, coeffs_high_json)
        VALUES (?,?,?,?,?,?)
        """,
        (mech_id, name, inchi_key, formula, "[1,0,0,0,0,0,0]", "[1,0,0,0,0,0,0]"),
    )
    return cur.lastrowid


def _insert_reaction(
    conn: sqlite3.Connection, mech_id: int, formula_string: str,
    canonical_key: Optional[str], A_si: float = 1e10, n: float = 0.5,
    Ea_si: float = 50000.0,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO reactions
            (mechanism_id, formula_string, canonical_key,
             reversible, kinetics_type, A_si, n, Ea_si)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (mech_id, formula_string, canonical_key, 1, "arrhenius", A_si, n, Ea_si),
    )
    return cur.lastrowid


class TestSpeciesResolutionNotSilentlyDropped:
    def test_unresolved_species_appear_as_unique_not_dropped(self, tmp_path):
        db = str(tmp_path / "c.db")
        conn = _make_db(db)
        m1 = _insert_mechanism(conn, "M1")
        m2 = _insert_mechanism(conn, "M2")
        _insert_species(conn, m1, "H2", "IK_H2")
        _insert_species(conn, m1, "MysteryA", None)   # unresolved -- glossary miss
        _insert_species(conn, m2, "H2", "IK_H2")
        conn.commit()

        from rmip.analysis.comparison import ComparisonEngine
        eng = ComparisonEngine(db)
        report = eng.compare([m1, m2])
        sp = report["species_comparison"]

        assert sp["n_universal"] == 1  # H2
        assert "MysteryA" in sp["unique"][str(m1)]  # kept, not dropped
        assert sp["n_unresolved_species"] == 1
        assert sp["species_resolution_coverage_pct"] == pytest.approx(200 / 3, abs=0.1)

    def test_zero_percent_resolution_and_zero_matches_raises(self, tmp_path):
        """Pre-fix: universal=0, unique=0/0 (impossible for 2 non-empty
        mechanisms), exit 0. Now a hard error instead -- but only when
        there's truly no signal (name fallback also finds nothing shared),
        distinguishing this from a legitimate self-compare-by-name."""
        db = str(tmp_path / "c.db")
        conn = _make_db(db)
        m1 = _insert_mechanism(conn, "M1")
        m2 = _insert_mechanism(conn, "M2")
        _insert_species(conn, m1, "SpeciesA", None)
        _insert_species(conn, m2, "SpeciesB", None)
        conn.commit()

        from rmip.analysis.comparison import ComparisonEngine
        eng = ComparisonEngine(db)
        with pytest.raises(RuntimeError, match="0%"):
            eng.compare([m1, m2])

    def test_self_compare_with_partial_glossary_resolution(self, tmp_path):
        """B3's exact scenario: "even self-compare returns zeros" when the
        glossary only resolves a fraction of species (FFCM1: 28.9%). The
        resolved subset (real inchi_key) correctly reports as universal even
        pre-fix; the point of B3 is that the *unresolved* subset used to
        vanish from the report entirely instead of showing up as unique to
        each side -- unresolved rows are namespaced per-mechanism (never
        cross-match by name alone, since that identifier isn't trustworthy),
        so self-compare's honest report is "partially universal, partially
        unverifiable" rather than a silent full match or a silent zero."""
        db = str(tmp_path / "c.db")
        conn = _make_db(db)
        m1 = _insert_mechanism(conn, "Orig")
        m2 = _insert_mechanism(conn, "Copy")
        for mid in (m1, m2):
            _insert_species(conn, mid, "H2", "IK_H2")      # glossary-resolved
            _insert_species(conn, mid, "MysteryX", None)   # glossary miss
        conn.commit()

        from rmip.analysis.comparison import ComparisonEngine
        eng = ComparisonEngine(db)
        report = eng.compare([m1, m2])
        sp = report["species_comparison"]
        assert sp["n_universal"] == 1  # H2, via real inchi_key
        # Unresolved species no longer vanish -- both sides report it.
        assert "MysteryX" in sp["unique"][str(m1)]
        assert "MysteryX" in sp["unique"][str(m2)]
        assert sp["n_unresolved_species"] == 2


class TestReactionResolutionNotSilentlyDropped:
    def test_unresolved_reactions_appear_as_unique_not_dropped(self, tmp_path):
        db = str(tmp_path / "c.db")
        conn = _make_db(db)
        m1 = _insert_mechanism(conn, "M1")
        m2 = _insert_mechanism(conn, "M2")
        _insert_reaction(conn, m1, "H2+O2<=>H2O", "R1")
        _insert_reaction(conn, m1, "Weird+Rxn<=>Stuff", None)  # unresolved
        _insert_reaction(conn, m2, "H2+O2<=>H2O", "R1")
        conn.commit()

        from rmip.analysis.comparison import ComparisonEngine
        eng = ComparisonEngine(db)
        report = eng.compare([m1, m2])
        rxn = report["reaction_comparison"]

        assert rxn["n_shared"] == 1
        assert "Weird+Rxn<=>Stuff" in rxn["unique"][str(m1)]
        assert rxn["n_unresolved_reactions"] == 1

    def test_zero_percent_resolution_and_zero_matches_raises(self, tmp_path):
        db = str(tmp_path / "c.db")
        conn = _make_db(db)
        m1 = _insert_mechanism(conn, "M1")
        m2 = _insert_mechanism(conn, "M2")
        _insert_reaction(conn, m1, "H2+O2<=>H2O", None)
        _insert_reaction(conn, m2, "CH4+O2<=>CO2+H2O", None)
        conn.commit()

        from rmip.analysis.comparison import ComparisonEngine
        eng = ComparisonEngine(db)
        with pytest.raises(RuntimeError, match="0%"):
            eng.compare([m1, m2])


class TestFullyResolvedCoverageIs100Percent(object):
    def test_all_resolved_gives_100_pct_coverage_and_no_error(self, tmp_path):
        db = str(tmp_path / "c.db")
        conn = _make_db(db)
        m1 = _insert_mechanism(conn, "M1")
        m2 = _insert_mechanism(conn, "M2")
        _insert_species(conn, m1, "H2", "IK_H2")
        _insert_species(conn, m2, "H2", "IK_H2")
        _insert_reaction(conn, m1, "H2+O2<=>H2O", "R1")
        _insert_reaction(conn, m2, "H2+O2<=>H2O", "R1")
        conn.commit()

        from rmip.analysis.comparison import ComparisonEngine
        eng = ComparisonEngine(db)
        report = eng.compare([m1, m2])
        assert report["species_comparison"]["species_resolution_coverage_pct"] == 100.0
        assert report["reaction_comparison"]["reaction_resolution_coverage_pct"] == 100.0


class TestDuplicateReactionsNotCollapsed:
    """M9 (review.md): two CHEMKIN DUPLICATE rows sharing one canonical_key
    used to collide in `_compare_reactions`'s `{canonical_key: row}` shape,
    silently dropping all but the last-read row. Fixed by keying on
    (canonical_key, occurrence_index) instead, mirroring graph_comparator.py.
    """

    def test_two_duplicate_rows_in_one_mechanism_both_survive(self, tmp_path):
        db = str(tmp_path / "c.db")
        conn = _make_db(db)
        m1 = _insert_mechanism(conn, "M1")
        m2 = _insert_mechanism(conn, "M2")
        # M1 has a DUPLICATE-marked reaction: two rows, same canonical_key,
        # different kinetics (e.g. two summed Arrhenius expressions).
        _insert_reaction(conn, m1, "A+B<=>C", "R_DUP", A_si=1e10)
        _insert_reaction(conn, m1, "A+B<=>C", "R_DUP", A_si=2e10)
        _insert_reaction(conn, m2, "A+B<=>C", "R_DUP", A_si=1e10)
        conn.commit()

        from rmip.analysis.comparison import ComparisonEngine
        eng = ComparisonEngine(db)
        report = eng.compare([m1, m2])
        rxn = report["reaction_comparison"]

        # Pre-fix: the second insert into a plain {canonical_key: row} dict
        # would silently overwrite the first, so n_total_unique_reactions
        # would be 1 (both occurrences collapsed into a single key) and one
        # of M1's two real rows would vanish entirely.
        assert rxn["n_total_unique_reactions"] == 2
        matched_A_si = sorted(
            entry[m1]["A_si"] for entry in rxn["matched"].values()
            if m1 in entry
        )
        assert matched_A_si == [1e10, 2e10]
        # Occurrence 0 in M1 pairs with M2's only row -> shared;
        # occurrence 1 in M1 has no counterpart in M2 -> unique.
        assert rxn["n_shared"] == 1
        assert rxn["n_unique"][str(m1)] == 1

    def test_kinetic_diffs_carry_occurrence_index(self, tmp_path):
        db = str(tmp_path / "c.db")
        conn = _make_db(db)
        m1 = _insert_mechanism(conn, "M1")
        m2 = _insert_mechanism(conn, "M2")
        _insert_reaction(conn, m1, "A+B<=>C", "R_DUP", A_si=1e10)
        _insert_reaction(conn, m2, "A+B<=>C", "R_DUP", A_si=1.5e10)
        conn.commit()

        from rmip.analysis.comparison import ComparisonEngine
        eng = ComparisonEngine(db)
        report = eng.compare([m1, m2])
        diffs = report["kinetic_diffs"]
        assert len(diffs) == 1
        assert diffs[0]["canonical_key"] == "R_DUP"
        assert diffs[0]["occurrence_index"] == 0
