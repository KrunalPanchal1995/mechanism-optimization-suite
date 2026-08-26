"""
rmip/tests/test_unified_tree_m9.py
====================================
Regression coverage for the M9 (review.md) gap in unified_tree.py that
survived the fusion.py/graph_comparator.py M9 fixes: `_load_source_rows`
built `reactions_by_key` as a plain `{canonical_key: row}` dict, so two
CHEMKIN DUPLICATE rows sharing one canonical_key within a mechanism
collided -- every `ReactionDecision` referencing that canonical_key
(regardless of which real occurrence it should resolve to) silently picked
up the same last-read row's kinetics, and `ReactionDecision` itself had no
`occurrence_index` field to disambiguate them in the first place.

Fix: `ReactionDecision.occurrence_index` (copied from the originating
`ReactionClassification` at plan() time) + `reactions_by_key` keyed on
(canonical_key, occurrence_index), matching graph_comparator.py's own
convention exactly so plan()'s classification and build()'s row lookups
agree on which physical row is occurrence 0 vs 1.
"""
from __future__ import annotations

import json
import sqlite3


def _make_db(path: str) -> sqlite3.Connection:
    from rmip.database.schema import init_db
    from rmip.database.schema_phase3 import upgrade_db
    conn = init_db(path)
    upgrade_db(conn)
    return conn


def _insert_mech(conn, name: str) -> int:
    cur = conn.execute(
        """
        INSERT INTO mechanisms (name, format, filepath, description,
                                n_species, n_reactions, loaded_at, metadata_json)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (name, "chemkin", "/fake/path.inp", f"Test {name}", 0, 0,
         "2024-01-01T00:00:00Z", "{}"),
    )
    return cur.lastrowid


def _insert_species(conn, mid, name, inchi_key, formula="H2"):
    coeffs = [1.0, 0.0, 0.0, 0.0, 0.0, -100.0, 1.0]
    cur = conn.execute(
        """
        INSERT INTO species (mechanism_id, name, inchi_key, formula,
                             canonical_smiles, thermo_type,
                             tmin_K, tmid_K, tmax_K,
                             coeffs_low_json, coeffs_high_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        (mid, name, inchi_key, formula, inchi_key or "", "nasa7",
         200.0, 1000.0, 5000.0, json.dumps(coeffs), json.dumps(coeffs)),
    )
    return cur.lastrowid


def _insert_reaction(conn, mid, formula_string, canonical_key, A_si, n, Ea_si):
    cur = conn.execute(
        """
        INSERT INTO reactions
            (mechanism_id, formula_string, canonical_key,
             reversible, kinetics_type, A_si, n, Ea_si, kinetics_data_json)
        VALUES (?,?,?,?,?,?,?,?,?)
        """,
        (mid, formula_string, canonical_key, 1, "arrhenius", A_si, n, Ea_si, None),
    )
    return cur.lastrowid


def _add_stoich(conn, rxn_id, sp_id, coeff, role):
    conn.execute(
        "INSERT INTO stoichiometry (reaction_id, species_id, stoich_coeff, role) VALUES (?,?,?,?)",
        (rxn_id, sp_id, coeff, role),
    )


def _build_db_with_duplicate_reaction(tmp_path):
    """m1 has a DUPLICATE-marked reaction (two rows, same canonical_key,
    different A_si -- e.g. two summed Arrhenius terms for one physical
    step). m2 has only one row at that canonical_key, so it pairs with
    m1's occurrence 0 (SHARED/OPTIMIZE); m1's occurrence 1 has no
    counterpart anywhere (UNIQUE/IMPORT_SINGLE)."""
    db_path = str(tmp_path / "m9.db")
    conn = _make_db(db_path)
    m1 = _insert_mech(conn, "MechA")
    m2 = _insert_mech(conn, "MechB")

    for mid in (m1, m2):
        _insert_species(conn, mid, "H2", "IK_H2", formula="H2")
        _insert_species(conn, mid, "O2", "IK_O2", formula="O2")

    r0 = _insert_reaction(conn, m1, "H2+O2<=>P", "R_DUP", 1e13, 0.0, 100_000.0)
    r1 = _insert_reaction(conn, m1, "H2+O2<=>P", "R_DUP", 5e12, 0.0, 95_000.0)
    r2 = _insert_reaction(conn, m2, "H2+O2<=>P", "R_DUP", 2e13, 0.0, 90_000.0)

    for mid, rid in ((m1, r0), (m1, r1), (m2, r2)):
        h2_id = conn.execute(
            "SELECT id FROM species WHERE mechanism_id=? AND name='H2'", (mid,)
        ).fetchone()[0]
        o2_id = conn.execute(
            "SELECT id FROM species WHERE mechanism_id=? AND name='O2'", (mid,)
        ).fetchone()[0]
        _add_stoich(conn, rid, h2_id, 1.0, "reactant")
        _add_stoich(conn, rid, o2_id, 1.0, "reactant")

    conn.commit()
    conn.close()
    return db_path, m1, m2, (r0, r1, r2)


class TestM9DuplicateReactionOccurrenceIndex:
    def test_plan_assigns_distinct_occurrence_index_to_each_duplicate_row(self, tmp_path):
        db_path, m1, m2, _ = _build_db_with_duplicate_reaction(tmp_path)

        from rmip.analysis.unified_tree import UnifiedTreeBuilder

        builder = UnifiedTreeBuilder(db_path)
        plan = builder.plan([m1, m2], mode="optimize_all",
                             manifest_path=str(tmp_path / "decisions.csv"))

        dup_decisions = [d for d in plan.decisions if d.canonical_key == "R_DUP"]
        assert len(dup_decisions) == 2
        occs = sorted(d.occurrence_index for d in dup_decisions)
        assert occs == [0, 1]

        occ0 = next(d for d in dup_decisions if d.occurrence_index == 0)
        occ1 = next(d for d in dup_decisions if d.occurrence_index == 1)
        # Occurrence 0 pairs with m2's single row -> shared -> OPTIMIZE.
        assert occ0.decision == "OPTIMIZE"
        assert set(occ0.source_mech_ids) == {m1, m2}
        # Occurrence 1 has no counterpart -> unique -> IMPORT_SINGLE.
        assert occ1.decision == "IMPORT_SINGLE"
        assert occ1.source_mech_ids == [m1]

    def test_build_keeps_both_duplicate_rows_kinetics_distinct(self, tmp_path):
        db_path, m1, m2, (r0, r1, r2) = _build_db_with_duplicate_reaction(tmp_path)

        from rmip.analysis.unified_tree import UnifiedTreeBuilder
        from rmip.types import TemperatureGrid

        builder = UnifiedTreeBuilder(db_path)
        plan = builder.plan([m1, m2], mode="optimize_all",
                             manifest_path=str(tmp_path / "decisions.csv"))

        # Force IMPORT_SINGLE occurrence to copy verbatim from m1 (already
        # the only source), and check its A_si downstream is r1's own value
        # (5e12), not silently swapped for r0's (1e13) by a collision.
        grid = TemperatureGrid(T_min=1000, T_max=1000, dT=20)
        result = builder.build(
            plan=plan, name="m9_test", tgrid=grid, out_dir=str(tmp_path / "out"),
        )

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT A_si FROM reactions WHERE mechanism_id=? AND canonical_key='R_DUP' "
            "ORDER BY A_si",
            (result.unified_mechanism_id,),
        ).fetchall()

        # Pre-fix: `reactions_by_key` collapsed to a plain {canonical_key:
        # row} dict, so m1's lookup for EITHER decision returned whichever
        # of r0/r1 was read last (r1, A_si=5e12) -- the IMPORT_SINGLE copy
        # would still show 5e12 by coincidence (it's genuinely r1's own
        # value), but the OPTIMIZE mean would silently use r1's 5e12 as
        # m1's contribution instead of r0's true 1e13, giving
        # geomean(5e12, 2e13) = 1e13 exactly instead of the correct
        # geomean(1e13, 2e13) ≈ 1.4142e13. Asserting the precise value
        # (not just "the two differ") is what actually catches the bug --
        # both scenarios produce two distinct A_si values, only one is right.
        a_values = sorted(r["A_si"] for r in rows)
        assert len(a_values) == 2
        assert a_values[0] == 5e12  # IMPORT_SINGLE verbatim copy of r1
        assert abs(a_values[1] - 1e13 * (2 ** 0.5)) < 1.0  # correct OPTIMIZE geomean
        assert a_values[1] != 1e13  # the collision value the bug would have produced
