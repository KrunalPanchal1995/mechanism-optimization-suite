"""
rmip/tests/test_graph_comparator_m10.py
=========================================
Regression coverage for M10 (review.md, PLAN.md Priority-0 item #3,
2026-07-17): GraphComparator._kinetics_are_identical() used to compare ONLY
the single representative A/n/Ea triple (the high-pressure limit for
Troe/Lindemann, the ~1atm entry for PLOG) -- two mechanisms could share that
one number while having completely different LOW-pressure parameters, Troe
alpha/T1/T2/T3, or other PLOG pressure points, and still get classified
SHARED_IDENTICAL. This silently understated real kinetic divergence during
fusion (a fused "identical" reaction could actually differ wildly at real
operating pressures).

Fix: _kinetics_are_identical now also requires (a) all mechanisms agree on
kinetics_type for the reaction, and (b) for pressure-dependent types
(troe/lindemann/sri/plog/chebyshev), the full kinetics_data_json payload to
match within tolerance -- not just the representative triple.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Optional

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


def _build_troe_mech(conn, mech_id: int, low_A: float) -> None:
    h2 = _insert_species(conn, mech_id, "H2", "IK_H2", formula="H2")
    o2 = _insert_species(conn, mech_id, "O2", "IK_O2", formula="O2")
    kdata = {
        "alpha": 0.5, "T1": 100.0, "T2": 2000.0, "T3": 10.0,
        "low":  {"A_si": low_A, "n": -1.0, "Ea_si": 1000.0},
        "high": {"A_si": 1e10, "n": 0.5, "Ea_si": 50000.0},
    }
    r = _insert_reaction(
        conn, mech_id, "H2+O2(+M)<=>H2O2(+M)", "R_TROE",
        A_si=1e10, n=0.5, Ea_si=50000.0,  # representative = high-pressure limit
        kinetics_type="troe", kinetics_data_json=json.dumps(kdata),
    )
    _insert_stoich(conn, r, h2, 1.0, "reactant")
    _insert_stoich(conn, r, o2, 1.0, "reactant")


class TestM10FalloffDivergenceDetected:
    def test_same_high_p_limit_different_low_p_not_identical(self, tmp_path):
        """Two Troe reactions sharing the same high-pressure-limit A/n/Ea
        but wildly different LOW-pressure A must NOT be SHARED_IDENTICAL."""
        db = str(tmp_path / "m10.db")
        conn = _make_db(db)
        m1 = _insert_mechanism(conn, "M1")
        m2 = _insert_mechanism(conn, "M2")
        _build_troe_mech(conn, m1, low_A=1e15)
        _build_troe_mech(conn, m2, low_A=1e20)  # 5 orders of magnitude off
        conn.commit()
        conn.close()

        from rmip.analysis.graph_comparator import GraphComparator, ReactionClass

        result = GraphComparator(db).compare([m1, m2])
        rxn = next(r for r in result.reaction_classifications if r.canonical_key == "R_TROE")
        assert rxn.classification == ReactionClass.SHARED_DIVERGENT

    def test_identical_full_troe_table_is_identical(self, tmp_path):
        """Sanity check: truly identical Troe tables (same high AND low AND
        alpha/T1/T2/T3) must still classify SHARED_IDENTICAL -- the fix must
        not become overly conservative."""
        db = str(tmp_path / "m10b.db")
        conn = _make_db(db)
        m1 = _insert_mechanism(conn, "M1")
        m2 = _insert_mechanism(conn, "M2")
        _build_troe_mech(conn, m1, low_A=1e15)
        _build_troe_mech(conn, m2, low_A=1e15)
        conn.commit()
        conn.close()

        from rmip.analysis.graph_comparator import GraphComparator, ReactionClass

        result = GraphComparator(db).compare([m1, m2])
        rxn = next(r for r in result.reaction_classifications if r.canonical_key == "R_TROE")
        assert rxn.classification == ReactionClass.SHARED_IDENTICAL

    def test_different_kinetics_type_not_identical(self, tmp_path):
        """Same representative A/n/Ea but one mechanism reports plain
        arrhenius and the other reports troe (e.g. a falloff table lost or
        never captured on one side) -- must not be SHARED_IDENTICAL even
        though the single-triple comparison alone would pass."""
        db = str(tmp_path / "m10c.db")
        conn = _make_db(db)
        m1 = _insert_mechanism(conn, "M1")
        m2 = _insert_mechanism(conn, "M2")
        for mid, ktype, kdata in (
            (m1, "arrhenius", None),
            (m2, "troe", json.dumps({
                "alpha": 0.5, "T1": 100.0, "T2": 2000.0, "T3": 10.0,
                "low": {"A_si": 1e15, "n": -1.0, "Ea_si": 1000.0},
                "high": {"A_si": 1e10, "n": 0.5, "Ea_si": 50000.0},
            })),
        ):
            h2 = _insert_species(conn, mid, "H2", "IK_H2", formula="H2")
            o2 = _insert_species(conn, mid, "O2", "IK_O2", formula="O2")
            r = _insert_reaction(
                conn, mid, "H2+O2<=>H2O2", "R_MIXED",
                A_si=1e10, n=0.5, Ea_si=50000.0,
                kinetics_type=ktype, kinetics_data_json=kdata,
            )
            _insert_stoich(conn, r, h2, 1.0, "reactant")
            _insert_stoich(conn, r, o2, 1.0, "reactant")
        conn.commit()
        conn.close()

        from rmip.analysis.graph_comparator import GraphComparator, ReactionClass

        result = GraphComparator(db).compare([m1, m2])
        rxn = next(r for r in result.reaction_classifications if r.canonical_key == "R_MIXED")
        assert rxn.classification == ReactionClass.SHARED_DIVERGENT

    def test_different_plog_pressure_point_count_not_identical(self, tmp_path):
        """Same 1-atm representative triple, but one mechanism has an extra
        PLOG pressure point the other lacks -- must not be identical."""
        db = str(tmp_path / "m10d.db")
        conn = _make_db(db)
        m1 = _insert_mechanism(conn, "M1")
        m2 = _insert_mechanism(conn, "M2")
        base_plog = {
            "pressures": [1013.25, 101325.0],
            "arrhenius": [
                {"A_si": 5e9, "n": 0.6, "Ea_si": 48000.0},
                {"A_si": 1e10, "n": 0.5, "Ea_si": 50000.0},
            ],
        }
        extra_plog = {
            "pressures": [1013.25, 101325.0, 10132500.0],
            "arrhenius": [
                {"A_si": 5e9, "n": 0.6, "Ea_si": 48000.0},
                {"A_si": 1e10, "n": 0.5, "Ea_si": 50000.0},
                {"A_si": 2e10, "n": 0.4, "Ea_si": 52000.0},
            ],
        }
        for mid, kdata in ((m1, base_plog), (m2, extra_plog)):
            h2 = _insert_species(conn, mid, "H2", "IK_H2", formula="H2")
            o2 = _insert_species(conn, mid, "O2", "IK_O2", formula="O2")
            r = _insert_reaction(
                conn, mid, "H2+O2<=>H2O2", "R_PLOG",
                A_si=1e10, n=0.5, Ea_si=50000.0,
                kinetics_type="plog", kinetics_data_json=json.dumps(kdata),
            )
            _insert_stoich(conn, r, h2, 1.0, "reactant")
            _insert_stoich(conn, r, o2, 1.0, "reactant")
        conn.commit()
        conn.close()

        from rmip.analysis.graph_comparator import GraphComparator, ReactionClass

        result = GraphComparator(db).compare([m1, m2])
        rxn = next(r for r in result.reaction_classifications if r.canonical_key == "R_PLOG")
        assert rxn.classification == ReactionClass.SHARED_DIVERGENT

    def test_plain_arrhenius_still_works_unaffected(self, tmp_path):
        """Sanity check: two plain-Arrhenius reactions within tolerance are
        unaffected by the M10 fix (kinetics_type check is a no-op for a
        single shared type, no kinetics_data_json comparison triggered)."""
        db = str(tmp_path / "m10e.db")
        conn = _make_db(db)
        m1 = _insert_mechanism(conn, "M1")
        m2 = _insert_mechanism(conn, "M2")
        for mid in (m1, m2):
            h2 = _insert_species(conn, mid, "H2", "IK_H2", formula="H2")
            o2 = _insert_species(conn, mid, "O2", "IK_O2", formula="O2")
            r = _insert_reaction(
                conn, mid, "H2+O2<=>H2O2", "R_ARR",
                A_si=1e10, n=0.5, Ea_si=50000.0, kinetics_type="arrhenius",
            )
            _insert_stoich(conn, r, h2, 1.0, "reactant")
            _insert_stoich(conn, r, o2, 1.0, "reactant")
        conn.commit()
        conn.close()

        from rmip.analysis.graph_comparator import GraphComparator, ReactionClass

        result = GraphComparator(db).compare([m1, m2])
        rxn = next(r for r in result.reaction_classifications if r.canonical_key == "R_ARR")
        assert rxn.classification == ReactionClass.SHARED_IDENTICAL
