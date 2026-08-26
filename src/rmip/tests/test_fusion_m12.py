"""
rmip/tests/test_fusion_m12.py
==============================
Regression coverage for M12 (bug.md, flagged "ambiguous/unconfirmed" as of
2026-07-17): species without an InChI key were claimed to be dropped from
fused stoichiometry, mass-imbalancing the output mechanism.

Re-verification (2026-07-17): reading fusion.py's `_insert_species`/
`_copy_stoichiometry` shows this was already fixed as part of B4's Phase 1
fix (2026-07-14) -- both functions re-derive a per-mechanism unresolved-
species sentinel key (`_unresolved_key(mid, name)`) instead of collapsing
every InChI-less species onto a shared `None` bucket, so stoichiometry rows
for unresolved species resolve correctly instead of colliding/dropping.
`test_fusion_b4.py` already covers the species/reaction *record* counts;
this file adds the one thing it didn't check -- that STOICHIOMETRY rows
(the actual mass-balance data) for InChI-less species survive the full
fuse() DB write, not just the in-memory FusionResult records.
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
) -> int:
    coeffs = [1.0, 0.0, 0.0, 0.0, 0.0, -100.0, 1.0]
    cur = conn.execute(
        """
        INSERT INTO species (mechanism_id, name, inchi_key, formula,
                             canonical_smiles, thermo_type,
                             tmin_K, tmid_K, tmax_K,
                             coeffs_low_json, coeffs_high_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        (mech_id, name, inchi_key, formula, inchi_key or "", "nasa7",
         200.0, 1000.0, 5000.0, json.dumps(coeffs), json.dumps(coeffs)),
    )
    return cur.lastrowid


def _insert_reaction(
    conn: sqlite3.Connection, mech_id: int, formula_string: str,
    canonical_key: Optional[str], A_si: float = 1e10,
    n: float = 0.5, Ea_si: float = 50000.0,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO reactions
            (mechanism_id, formula_string, canonical_key,
             reversible, kinetics_type, A_si, n, Ea_si, kinetics_data_json)
        VALUES (?,?,?,?,?,?,?,?,?)
        """,
        (mech_id, formula_string, canonical_key, 1, "arrhenius",
         A_si, n, Ea_si, None),
    )
    return cur.lastrowid


def _insert_stoich(conn, rxn_id, sp_id, coef, role) -> None:
    conn.execute(
        "INSERT INTO stoichiometry (reaction_id, species_id, stoich_coeff, role) VALUES (?,?,?,?)",
        (rxn_id, sp_id, coef, role),
    )


class TestM12StoichiometrySurvivesFusion:
    """A reaction built entirely from InChI-less species (the worst case --
    every stoich row's species key is a sentinel, not a real InChI) must
    keep every one of its stoichiometry rows in the fused DB output."""

    def test_stoich_rows_for_inchi_less_species_not_dropped(self, tmp_path):
        db = str(tmp_path / "m12.db")
        conn = _make_db(db)
        m1 = _insert_mechanism(conn, "M1")
        m2 = _insert_mechanism(conn, "M2")

        # Two mechanisms, each with 3 InChI-less species in one reaction
        # (A + B <=> C), spelled identically so GraphComparator has a
        # chance to (wrongly, pre-fix) collapse them onto one None bucket.
        for mid in (m1, m2):
            a = _insert_species(conn, mid, "SpeciesA", None, formula="CxHy")
            b = _insert_species(conn, mid, "SpeciesB", None, formula="CxHy")
            c = _insert_species(conn, mid, "SpeciesC", None, formula="CxHy")
            r = _insert_reaction(conn, mid, "SpeciesA+SpeciesB<=>SpeciesC", None)
            _insert_stoich(conn, r, a, 1.0, "reactant")
            _insert_stoich(conn, r, b, 1.0, "reactant")
            _insert_stoich(conn, r, c, 1.0, "product")
        conn.commit()
        conn.close()

        from rmip.analysis.graph_comparator import GraphComparator
        from rmip.analysis.fusion import FusionEngine

        cmp_result = GraphComparator(db).compare([m1, m2])
        result = FusionEngine(db).fuse(cmp_result, name="Fused_m12")

        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        fused_mech_id = result.fused_mechanism_id

        rxn_rows = conn.execute(
            "SELECT id, formula_string FROM reactions WHERE mechanism_id = ?",
            (fused_mech_id,),
        ).fetchall()
        # Unresolved reaction is per-source UNIQUE (not merged) -> one row per source mech
        assert len(rxn_rows) == 2

        for rxn in rxn_rows:
            stoich_rows = conn.execute(
                "SELECT stoich_coeff, role FROM stoichiometry WHERE reaction_id = ?",
                (rxn["id"],),
            ).fetchall()
            # Pre-fix (M12): these would silently drop to 0-2 rows as
            # InChI-less species collided on a shared None lookup key.
            assert len(stoich_rows) == 3, (
                f"reaction {rxn['id']} ({rxn['formula_string']!r}) has "
                f"{len(stoich_rows)} stoich rows, expected 3 -- mass-imbalanced "
                f"fused output, M12 has regressed"
            )
            reactant_coeff = sum(
                r["stoich_coeff"] for r in stoich_rows if r["role"] == "reactant"
            )
            product_coeff = sum(
                r["stoich_coeff"] for r in stoich_rows if r["role"] == "product"
            )
            assert reactant_coeff == pytest.approx(2.0)
            assert product_coeff == pytest.approx(1.0)
        conn.close()

    def test_mixed_resolved_and_unresolved_species_same_reaction(self, tmp_path):
        """A reaction with ONE real-InChI species and one InChI-less species
        -- the InChI-less one must still get its stoich row, not just the
        resolved one."""
        db = str(tmp_path / "m12b.db")
        conn = _make_db(db)
        m1 = _insert_mechanism(conn, "M1")
        m2 = _insert_mechanism(conn, "M2")
        for mid in (m1, m2):
            h2 = _insert_species(conn, mid, "H2", "IK_H2", formula="H2")
            mystery = _insert_species(conn, mid, "MysteryX", None, formula="CxHy")
            r = _insert_reaction(conn, mid, "H2+MysteryX<=>Stuff", None)
            _insert_stoich(conn, r, h2, 1.0, "reactant")
            _insert_stoich(conn, r, mystery, 1.0, "reactant")
        conn.commit()
        conn.close()

        from rmip.analysis.graph_comparator import GraphComparator
        from rmip.analysis.fusion import FusionEngine

        cmp_result = GraphComparator(db).compare([m1, m2])
        result = FusionEngine(db).fuse(cmp_result, name="Fused_m12_mixed")

        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        fused_mech_id = result.fused_mechanism_id
        rxn_rows = conn.execute(
            "SELECT id FROM reactions WHERE mechanism_id = ?", (fused_mech_id,),
        ).fetchall()
        for rxn in rxn_rows:
            n = conn.execute(
                "SELECT COUNT(*) FROM stoichiometry WHERE reaction_id = ?",
                (rxn["id"],),
            ).fetchone()[0]
            assert n == 2, "InChI-less reactant dropped from mixed-resolution reaction"
        conn.close()
