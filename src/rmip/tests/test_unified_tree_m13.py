"""
rmip/tests/test_unified_tree_m13.py
=====================================
Regression coverage for M13 (review.md, PLAN.md Priority-0 item #4,
2026-07-17): UnifiedTreeBuilder.build() used to re-pair uncertainty bands
with their originating decisions via
``zip([d for d in decisions if d.decision == "OPTIMIZE"], bands)`` -- but an
OPTIMIZE decision whose record build fails (e.g. no source mechanism
actually has that reaction, so _build_optimize_record's "no contributions
found" path returns None) is silently skipped when building `bands`, while
still appearing in that re-derived filter list. Every OPTIMIZE decision
AFTER the failed one then gets shifted by one position against `bands`,
so `insert_reaction_decision()` (keyed on the wrong `d`) and
`insert_uncertainty_band_rows()` (keyed on the right `band.canonical_key`,
stamped earlier) end up writing MISMATCHED rows into the audit DB for that
run -- a decision claiming OPTIMIZE for a reaction that was actually
dropped, an orphaned band with no matching decision row, and the true last
OPTIMIZE decision missing its audit trail entirely.

Fix: track (decision, band) pairs in lockstep as they're built, instead of
re-deriving the OPTIMIZE-decision list separately after the fact.
"""
from __future__ import annotations

import json
import sqlite3

import pytest


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


def _build_two_mech_three_shared_reactions(tmp_path):
    """m1/m2 both have R_A, R_B, R_C with divergent kinetics (forces
    OPTIMIZE for all three under optimize_all mode); R_A and R_C get
    deliberately different nominal A_si so their bands are numerically
    distinguishable in assertions."""
    db_path = str(tmp_path / "m13.db")
    conn = _make_db(db_path)
    m1 = _insert_mech(conn, "MechA")
    m2 = _insert_mech(conn, "MechB")

    for mid in (m1, m2):
        h2 = _insert_species(conn, mid, "H2", "IK_H2", formula="H2")
        o2 = _insert_species(conn, mid, "O2", "IK_O2", formula="O2")

    specs = [
        ("R_A", "H2+O2<=>P_A", 1e13, 0.0, 100_000.0, 2e13, 0.0, 90_000.0),
        ("R_B", "H2+O2<=>P_B", 1e14, 0.0, 200_000.0, 2e14, 0.0, 190_000.0),
        ("R_C", "H2+O2<=>P_C", 1e16, 0.5, 300_000.0, 2e16, 0.5, 290_000.0),
    ]
    for ck, formula, A1, n1, Ea1, A2, n2, Ea2 in specs:
        for mid, A, n, Ea in ((m1, A1, n1, Ea1), (m2, A2, n2, Ea2)):
            r = _insert_reaction(conn, mid, formula, ck, A, n, Ea)
            h2_id = conn.execute(
                "SELECT id FROM species WHERE mechanism_id=? AND name='H2'", (mid,)
            ).fetchone()[0]
            o2_id = conn.execute(
                "SELECT id FROM species WHERE mechanism_id=? AND name='O2'", (mid,)
            ).fetchone()[0]
            _add_stoich(conn, r, h2_id, 1.0, "reactant")
            _add_stoich(conn, r, o2_id, 1.0, "reactant")

    conn.commit()
    conn.close()
    return db_path, m1, m2


class TestM13BandDecisionPairingSurvivesAFailedOptimizeBuild:
    def test_middle_optimize_failure_does_not_shift_later_pairings(self, tmp_path):
        db_path, m1, m2 = _build_two_mech_three_shared_reactions(tmp_path)

        from rmip.analysis.unified_tree import UnifiedTreeBuilder
        from rmip.types import TemperatureGrid

        builder = UnifiedTreeBuilder(db_path)
        manifest = str(tmp_path / "decisions.csv")
        plan = builder.plan([m1, m2], mode="optimize_all", manifest_path=manifest)

        by_key = {d.canonical_key: d for d in plan.decisions}
        assert {"R_A", "R_B", "R_C"} <= set(by_key)
        for ck in ("R_A", "R_B", "R_C"):
            assert by_key[ck].decision == "OPTIMIZE"

        # Sabotage R_B: point it at a mechanism id with no such reaction, so
        # _build_optimize_record's "no contributions found" path fires and
        # this decision is silently dropped mid-list -- the exact scenario
        # that used to shift every later OPTIMIZE pairing by one.
        by_key["R_B"].source_mech_ids = [999999]

        # Keep decisions in R_A, R_B, R_C order (their natural order from
        # `specs` above) so R_B sits strictly between the two survivors.
        plan.decisions.sort(key=lambda d: d.canonical_key)

        grid = TemperatureGrid(T_min=1000, T_max=1000, dT=20)
        result = builder.build(
            plan=plan, name="m13_test", tgrid=grid, out_dir=str(tmp_path / "out"),
        )

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        run_row = conn.execute(
            "SELECT id FROM unified_tree_runs WHERE unified_mechanism_id=?",
            (result.unified_mechanism_id,),
        ).fetchone()
        run_id = run_row["id"]

        decision_rows = conn.execute(
            "SELECT canonical_key, decision FROM reaction_decisions WHERE run_id=?",
            (run_id,),
        ).fetchall()
        decision_keys = {r["canonical_key"] for r in decision_rows}

        band_rows = conn.execute(
            "SELECT DISTINCT canonical_key FROM uncertainty_bands WHERE run_id=?",
            (run_id,),
        ).fetchall()
        band_keys = {r["canonical_key"] for r in band_rows}
        conn.close()

        # R_B failed to build -> must NOT appear in either audit table, and
        # must not falsely claim OPTIMIZE succeeded for it.
        assert "R_B" not in decision_keys, (
            "R_B's failed OPTIMIZE build was recorded as a successful "
            "decision anyway -- band/decision pairing has shifted (M13 "
            "regression)"
        )
        assert "R_B" not in band_keys

        # R_A and R_C must BOTH have survived with correctly paired
        # decision + band rows -- pre-fix, R_C would be silently dropped
        # entirely once R_B's failure shifted the zip().
        assert decision_keys == {"R_A", "R_C"}
        assert band_keys == {"R_A", "R_C"}

    def test_band_values_not_cross_contaminated(self, tmp_path):
        """The surviving bands' k_nom values must correspond to their OWN
        reaction's Arrhenius parameters, not a neighboring one's."""
        db_path, m1, m2 = _build_two_mech_three_shared_reactions(tmp_path)

        from rmip.analysis.unified_tree import UnifiedTreeBuilder
        from rmip.types import TemperatureGrid

        builder = UnifiedTreeBuilder(db_path)
        manifest = str(tmp_path / "decisions.csv")
        plan = builder.plan([m1, m2], mode="optimize_all", manifest_path=manifest)
        by_key = {d.canonical_key: d for d in plan.decisions}
        by_key["R_B"].source_mech_ids = [999999]
        plan.decisions.sort(key=lambda d: d.canonical_key)

        grid = TemperatureGrid(T_min=1000, T_max=1000, dT=20)
        result = builder.build(
            plan=plan, name="m13_test2", tgrid=grid, out_dir=str(tmp_path / "out2"),
        )

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        run_id = conn.execute(
            "SELECT id FROM unified_tree_runs WHERE unified_mechanism_id=?",
            (result.unified_mechanism_id,),
        ).fetchone()["id"]

        k_nom_by_key = {}
        for ck in ("R_A", "R_C"):
            row = conn.execute(
                "SELECT k_nom FROM uncertainty_bands WHERE run_id=? AND canonical_key=?",
                (run_id, ck),
            ).fetchone()
            assert row is not None
            k_nom_by_key[ck] = row["k_nom"]
        conn.close()

        # R_A (A~1.5e13, Ea~95000) and R_C (A~1.5e16, Ea~295000) have very
        # different nominal rates at T=1000K -- if bands were cross-paired
        # these would collide/swap.
        assert k_nom_by_key["R_A"] != pytest.approx(k_nom_by_key["R_C"], rel=0.01)
