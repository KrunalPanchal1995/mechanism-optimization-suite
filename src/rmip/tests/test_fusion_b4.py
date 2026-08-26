"""
rmip/tests/test_fusion_b4.py
=============================
Regression coverage for B4 (review.md, 2026-07-14): fusing two identical
mechanism copies used to violate identity and destroy data. At FFCM1's real
glossary hit rate (28.9%), fusing two identical FFCM1 copies produced
11/38 species and 20/291 reactions -- unresolved species/reactions (no
inchi_key / canonical_key) were silently dropped at the SQL level in both
GraphComparator (feeding FusionEngine) and FusionEngine's own lookups. NULL
kinetics also became a fabricated A_si=0.0 in the fused rows.

Root cause lived in two files:
  - graph_comparator.py's _load_all_species/_load_all_reactions filtered
    `WHERE inchi_key/canonical_key IS NOT NULL` at the SQL level.
  - fusion.py's own _fuse_species/_fuse_reactions lookups skipped rows with
    no inchi_key/canonical_key even after that.

Fix: unresolved rows fall back to a per-mechanism (mid-namespaced) key
convention shared by both files, so GraphComparator classifies them UNIQUE
(never cross-matching, since the identifier isn't trustworthy) and
FusionEngine's existing UNIQUE->VERBATIM path carries them through intact
instead of dropping them. NULL-A_si rows with no kinetics_data_json either
are excluded (nothing to fuse) rather than written as a fabricated 0.0.
"""
from __future__ import annotations

import json
import sqlite3
from typing import List, Optional

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
    coeffs_low: Optional[List[float]] = None,
    coeffs_high: Optional[List[float]] = None,
) -> int:
    coeffs_low  = coeffs_low  or [1.0, 0.0, 0.0, 0.0, 0.0, -100.0, 1.0]
    coeffs_high = coeffs_high or [1.0, 0.0, 0.0, 0.0, 0.0, -100.0, 1.0]
    cur = conn.execute(
        """
        INSERT INTO species (mechanism_id, name, inchi_key, formula,
                             canonical_smiles, thermo_type,
                             tmin_K, tmid_K, tmax_K,
                             coeffs_low_json, coeffs_high_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        (mech_id, name, inchi_key, formula, inchi_key or "", "nasa7",
         200.0, 1000.0, 5000.0, json.dumps(coeffs_low), json.dumps(coeffs_high)),
    )
    return cur.lastrowid


def _insert_reaction(
    conn: sqlite3.Connection, mech_id: int, formula_string: str,
    canonical_key: Optional[str], A_si: Optional[float] = 1e10,
    n: Optional[float] = 0.5, Ea_si: Optional[float] = 50000.0,
    kinetics_type: str = "arrhenius", kinetics_data_json: Optional[str] = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO reactions
            (mechanism_id, formula_string, canonical_key,
             reversible, kinetics_type, A_si, n, Ea_si, kinetics_data_json)
        VALUES (?,?,?,?,?,?,?,?,?)
        """,
        (mech_id, formula_string, canonical_key, 1, kinetics_type,
         A_si, n, Ea_si, kinetics_data_json),
    )
    return cur.lastrowid


def _insert_stoich(conn, rxn_id, sp_id, coef, role) -> None:
    conn.execute(
        "INSERT INTO stoichiometry (reaction_id, species_id, stoich_coeff, role) VALUES (?,?,?,?)",
        (rxn_id, sp_id, coef, role),
    )


def _build_mixed_resolution_mech(conn, mech_id: int) -> None:
    """3 species (2 resolved, 1 not), 2 reactions (1 resolved, 1 not)."""
    h2 = _insert_species(conn, mech_id, "H2", "IK_H2", formula="H2")
    o2 = _insert_species(conn, mech_id, "O2", "IK_O2", formula="O2")
    mystery = _insert_species(conn, mech_id, "MysteryX", None, formula="CxHy")

    r1 = _insert_reaction(conn, mech_id, "H2+O2<=>H2O2", "R1", A_si=1e10, n=0.5, Ea_si=50000.0)
    _insert_stoich(conn, r1, h2, 1.0, "reactant")
    _insert_stoich(conn, r1, o2, 1.0, "reactant")

    r2 = _insert_reaction(conn, mech_id, "MysteryX+O2<=>Stuff", None, A_si=2e11, n=0.1, Ea_si=30000.0)
    _insert_stoich(conn, r2, mystery, 1.0, "reactant")
    _insert_stoich(conn, r2, o2, 1.0, "reactant")


class TestSelfFuseIdentity:
    """fuse(M, M) should preserve species/reaction counts (within the
    conservative UNIQUE-not-UNIVERSAL classification unresolved rows get)."""

    def test_self_fuse_preserves_species_and_reaction_counts(self, tmp_path):
        db = str(tmp_path / "b4.db")
        conn = _make_db(db)
        m1 = _insert_mechanism(conn, "Orig")
        m2 = _insert_mechanism(conn, "Copy")
        _build_mixed_resolution_mech(conn, m1)
        _build_mixed_resolution_mech(conn, m2)
        conn.commit()
        conn.close()

        from rmip.analysis.graph_comparator import GraphComparator
        from rmip.analysis.fusion import FusionEngine

        cmp_result = GraphComparator(db).compare([m1, m2])
        result = FusionEngine(db).fuse(cmp_result, name="Fused_self")

        # Pre-fix: 11/38-style collapse (unresolved species/reactions
        # vanished). Post-fix: every species/reaction from the source
        # mechanism accounts for exactly one record in the fused output --
        # 3 species (H2, O2 real-matched; MysteryX unique-per-source, so it
        # appears once per source mechanism = 2 records) and similarly for
        # the unresolved reaction.
        assert len(result.species_records) == 4    # H2, O2 (universal) + MysteryX x2 (unique per copy)
        assert len(result.reaction_records) == 3    # R1 (universal) + unresolved rxn x2 (unique per copy)

        names = [s.name for s in result.species_records]
        assert names.count("MysteryX") == 2
        assert "H2" in names and "O2" in names

    def test_no_species_or_reaction_silently_dropped_with_zero_glossary(self, tmp_path):
        """Even at 0% glossary resolution, nothing should vanish -- everything
        becomes UNIQUE-per-source instead."""
        db = str(tmp_path / "b4.db")
        conn = _make_db(db)
        m1 = _insert_mechanism(conn, "M1")
        m2 = _insert_mechanism(conn, "M2")
        for mid, suffix in ((m1, "A"), (m2, "B")):
            sp = _insert_species(conn, mid, f"Species{suffix}", None)
            r = _insert_reaction(conn, mid, f"Rxn{suffix}", None, A_si=1e9, n=0.0, Ea_si=1000.0)
            _insert_stoich(conn, r, sp, 1.0, "reactant")
        conn.commit()
        conn.close()

        from rmip.analysis.graph_comparator import GraphComparator
        from rmip.analysis.fusion import FusionEngine

        cmp_result = GraphComparator(db).compare([m1, m2])
        result = FusionEngine(db).fuse(cmp_result, name="Fused_zero_glossary")

        assert len(result.species_records) == 2
        assert len(result.reaction_records) == 2
        for rec in result.reaction_records:
            assert rec.A_si == pytest.approx(1e9)  # not fabricated to 0.0


class TestNullKineticsRefused:
    def test_hollow_reaction_excluded_not_fabricated_to_zero(self, tmp_path):
        """A reaction with no A_si and no kinetics_data_json (e.g. a
        Chebyshev row this suite can't summarize as a single Arrhenius
        triple, or a genuinely unparseable row) must be excluded from the
        fused mechanism, never written as A_si=0.0."""
        db = str(tmp_path / "b4.db")
        conn = _make_db(db)
        m1 = _insert_mechanism(conn, "M1")
        m2 = _insert_mechanism(conn, "M2")
        for mid in (m1, m2):
            sp1 = _insert_species(conn, mid, "H2", "IK_H2")
            sp2 = _insert_species(conn, mid, "O2", "IK_O2")
            r = _insert_reaction(conn, mid, "H2+O2<=>H2O2", "R_HOLLOW",
                                  A_si=None, n=None, Ea_si=None,
                                  kinetics_type="chebyshev", kinetics_data_json=None)
            _insert_stoich(conn, r, sp1, 1.0, "reactant")
            _insert_stoich(conn, r, sp2, 1.0, "reactant")
        conn.commit()
        conn.close()

        from rmip.analysis.graph_comparator import GraphComparator
        from rmip.analysis.fusion import FusionEngine

        cmp_result = GraphComparator(db).compare([m1, m2])
        result = FusionEngine(db).fuse(cmp_result, name="Fused_hollow")

        assert len(result.reaction_records) == 0  # excluded, not fabricated

    def test_chebyshev_with_kinetics_data_preserved_not_dropped(self, tmp_path):
        """A_si=None but kinetics_data_json populated (real Chebyshev
        polynomial) must survive -- A_si=None is honest, not "refuse"."""
        db = str(tmp_path / "b4.db")
        conn = _make_db(db)
        m1 = _insert_mechanism(conn, "M1")
        m2 = _insert_mechanism(conn, "M2")
        cheb_json = json.dumps({"Tmin": 300, "Tmax": 2000, "coeffs": [[1, 2], [3, 4]]})
        for mid in (m1, m2):
            sp1 = _insert_species(conn, mid, "H2", "IK_H2")
            sp2 = _insert_species(conn, mid, "O2", "IK_O2")
            r = _insert_reaction(conn, mid, "H2+O2<=>H2O2", "R_CHEB",
                                  A_si=None, n=None, Ea_si=None,
                                  kinetics_type="chebyshev", kinetics_data_json=cheb_json)
            _insert_stoich(conn, r, sp1, 1.0, "reactant")
            _insert_stoich(conn, r, sp2, 1.0, "reactant")
        conn.commit()
        conn.close()

        from rmip.analysis.graph_comparator import GraphComparator
        from rmip.analysis.fusion import FusionEngine

        cmp_result = GraphComparator(db).compare([m1, m2])
        result = FusionEngine(db).fuse(cmp_result, name="Fused_cheb")

        assert len(result.reaction_records) == 1
        assert result.reaction_records[0].A_si is None
        assert result.reaction_records[0].kinetics_data_json == cheb_json
