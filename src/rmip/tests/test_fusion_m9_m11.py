"""
rmip/tests/test_fusion_m9_m11.py
==================================
Regression coverage for M9 and M11 (review.md, 2026-07-14).

M9: CHEMKIN DUPLICATE reactions (two DB rows sharing one canonical_key --
e.g. two Arrhenius expressions summed for one reaction) used to collide in
every `{canonical_key: row}` map the comparison/fusion pipeline builds,
silently dropping all but the last-inserted duplicate. Fixed by keying on
(canonical_key, occurrence_index) throughout GraphComparator and
FusionEngine, threaded via a new `occurrence_index` field on
ReactionClassification/FusionRecord/UnionGraphSpec's bucket dicts.

M11: a FUSE-bucket reaction where any source has non-Arrhenius kinetics
(Troe/Lindemann/PLOG/SRI/Chebyshev) used to get its A/n/Ea averaged with
kinetics_data_json discarded -- a fused reaction claiming falloff/PLOG
kinetics_type with no falloff/PLOG table to back it up. Fixed by falling
back to VERBATIM-from-preferred-source (kinetics_data_json intact) whenever
any source row is non-Arrhenius.
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
    canonical_key: Optional[str], A_si: Optional[float] = 1e10,
    n: Optional[float] = 0.5, Ea_si: Optional[float] = 50000.0,
    kinetics_type: str = "arrhenius", kinetics_data_json: Optional[str] = None,
    duplicate: int = 0,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO reactions
            (mechanism_id, formula_string, canonical_key,
             reversible, kinetics_type, A_si, n, Ea_si, kinetics_data_json,
             duplicate)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
        (mech_id, formula_string, canonical_key, 1, kinetics_type,
         A_si, n, Ea_si, kinetics_data_json, duplicate),
    )
    return cur.lastrowid


def _insert_stoich(conn, rxn_id, sp_id, coef, role) -> None:
    conn.execute(
        "INSERT INTO stoichiometry (reaction_id, species_id, stoich_coeff, role) VALUES (?,?,?,?)",
        (rxn_id, sp_id, coef, role),
    )


class TestM9DuplicateReactionsSurviveFusion:
    def test_duplicate_pair_both_channels_fused_not_collided(self, tmp_path):
        """Two mechanisms each declare the SAME duplicate-reaction pair
        (two Arrhenius expressions for one equation, real CHEMKIN
        DUPLICATE convention). Pre-fix, {canonical_key: row} dicts in
        GraphComparator/FusionEngine collapsed each mechanism's 2 rows
        into 1, and further collapsed across mechanisms -- losing 3 of the
        4 total duplicate rows. Post-fix, all 4 source rows are visible to
        classification and the fused mechanism carries both duplicate
        channels through with distinct rate constants."""
        db = str(tmp_path / "dup.db")
        conn = _make_db(db)
        m1 = _insert_mechanism(conn, "M1")
        m2 = _insert_mechanism(conn, "M2")
        for mid, A_lo, A_hi in ((m1, 1.0e14, 5.0e13), (m2, 1.1e14, 5.2e13)):
            sp_h = _insert_species(conn, mid, "H", "IK_H", formula="H")
            sp_o2 = _insert_species(conn, mid, "O2", "IK_O2", formula="O2")
            sp_o = _insert_species(conn, mid, "O", "IK_O", formula="O")
            sp_oh = _insert_species(conn, mid, "OH", "IK_OH", formula="OH")
            r1 = _insert_reaction(conn, mid, "H+O2<=>O+OH", "R1",
                                   A_si=A_lo, n=0.0, Ea_si=16000.0, duplicate=1)
            r2 = _insert_reaction(conn, mid, "H+O2<=>O+OH", "R1",
                                   A_si=A_hi, n=0.0, Ea_si=15000.0, duplicate=1)
            for r in (r1, r2):
                _insert_stoich(conn, r, sp_h, 1.0, "reactant")
                _insert_stoich(conn, r, sp_o2, 1.0, "reactant")
                _insert_stoich(conn, r, sp_o, 1.0, "product")
                _insert_stoich(conn, r, sp_oh, 1.0, "product")
        conn.commit()
        conn.close()

        from rmip.analysis.graph_comparator import GraphComparator
        from rmip.analysis.fusion import FusionEngine

        cmp_result = GraphComparator(db).compare([m1, m2])
        by_occ = {rc.occurrence_index for rc in cmp_result.reaction_classifications
                  if rc.canonical_key == "R1"}
        assert by_occ == {0, 1}, "both duplicate occurrences must be classified"

        result = FusionEngine(db).fuse(cmp_result, name="Fused_dup")
        r1_records = [r for r in result.reaction_records if r.canonical_key == "R1"]
        assert len(r1_records) == 2, "both duplicate channels must survive fusion"
        a_values = sorted(r.A_si for r in r1_records)
        # geometric mean of (1.0e14,1.1e14) and of (5.0e13,5.2e13) respectively
        assert a_values[0] == pytest.approx((5.0e13 * 5.2e13) ** 0.5, rel=1e-6)
        assert a_values[1] == pytest.approx((1.0e14 * 1.1e14) ** 0.5, rel=1e-6)


class TestM11NonArrheniusFuseFallback:
    def test_troe_reaction_falls_back_to_verbatim_not_averaged(self, tmp_path):
        """A Troe (falloff) reaction shared by 2 mechanisms with different
        LOW/TROE tables used to get its high-P A/n/Ea geometric-averaged
        while kinetics_data_json was discarded -- kinetics_type="troe" with
        no falloff table. Must fall back to VERBATIM from the preferred
        source instead, keeping its real kinetics_data_json."""
        db = str(tmp_path / "troe.db")
        conn = _make_db(db)
        m1 = _insert_mechanism(conn, "M1")
        m2 = _insert_mechanism(conn, "M2")
        kdata1 = json.dumps({"low": {"A_si": 1e18, "n": -1.0, "Ea_si": 1000.0},
                              "high": {"A_si": 1e10, "n": 0.0, "Ea_si": 0.0},
                              "falloff": "troe", "troe_params": [0.5, 100, 1000, 5000]})
        kdata2 = json.dumps({"low": {"A_si": 2e18, "n": -1.1, "Ea_si": 1100.0},
                              "high": {"A_si": 1.2e10, "n": 0.0, "Ea_si": 0.0},
                              "falloff": "troe", "troe_params": [0.6, 90, 1100, 5100]})
        for mid, A_hi, kdata in ((m1, 1e10, kdata1), (m2, 1.2e10, kdata2)):
            sp_h = _insert_species(conn, mid, "H", "IK_H", formula="H")
            sp_o2 = _insert_species(conn, mid, "O2", "IK_O2", formula="O2")
            sp_ho2 = _insert_species(conn, mid, "HO2", "IK_HO2", formula="HO2")
            r = _insert_reaction(conn, mid, "H+O2(+M)<=>HO2(+M)", "R_TROE",
                                  A_si=A_hi, n=0.0, Ea_si=0.0,
                                  kinetics_type="troe", kinetics_data_json=kdata)
            _insert_stoich(conn, r, sp_h, 1.0, "reactant")
            _insert_stoich(conn, r, sp_o2, 1.0, "reactant")
            _insert_stoich(conn, r, sp_ho2, 1.0, "product")
        conn.commit()
        conn.close()

        from rmip.analysis.graph_comparator import GraphComparator
        from rmip.analysis.fusion import FusionEngine

        cmp_result = GraphComparator(db).compare([m1, m2])
        result = FusionEngine(db).fuse(cmp_result, name="Fused_troe")
        rec = next(r for r in result.reaction_records if r.canonical_key == "R_TROE")

        assert rec.kinetics_type == "troe"
        assert rec.kinetics_data_json is not None
        parsed = json.loads(rec.kinetics_data_json)
        assert "low" in parsed and "troe_params" in parsed
        # Verbatim from the preferred (lowest mech id) source, not averaged.
        assert rec.A_si == pytest.approx(1e10)
