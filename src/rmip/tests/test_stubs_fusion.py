"""
rmip/tests/test_stubs_fusion.py
================================
Test suite for `rmip.stubs.FusionEngine` (M7), the UnionGraphSpec-facing
adapter over the already-implemented, already-tested
`rmip.analysis.fusion.FusionEngine` (a *different* class, same name,
different module/interface — see `stubs.py`'s docstring for why two exist).

Coverage
--------
  fuse() end-to-end: UnionGraphSpec -> GraphComparisonResult reconstruction
    -> core FusionEngine delegation -> DB row -> CHEMKIN .inp/.dat files.
  fuse_overrides: post-insert UPDATE of FUSE-bucket reactions.
  write_cantera: optional ck2yaml conversion (skipped if ck2yaml absent).
  validate_element_balance(): both the balanced and the violation path.

Real-DB caveat (important, discovered while writing these tests): every
mechanism currently in `outputs/project.db` (both `chemkin`-parsed and
`cantera_yaml`-parsed) has `species.inchi_key`/`reactions.canonical_key`
entirely NULL — `glossary_matched=0` for all species, meaning the
name->SMILES->InChI resolution step in `core/parser.py` was never exercised
for any of them (no `--glossary` file covering these species names was
used at parse time). `GraphComparator.compare()` matches species by
inchi_key and reactions by canonical_key (design decisions D1/D2), so on
real project.db mechanisms today it produces an empty comparison —
0 species, 0 reactions classified, and FusionEngine.fuse() (correctly)
builds an empty fused mechanism from that. This is a pre-existing,
separate data-pipeline gap (glossary matching), not a defect in
FusionEngine — hence these tests build glossary-resolved fixture data by
hand, mirroring `test_phase3.py::TestFusionEngine`'s established pattern,
exactly the same way that class's own tests do (it has the identical
real-DB limitation).

Run with:
    python -m pytest src/rmip/tests/test_stubs_fusion.py -v
"""
from __future__ import annotations

import json
import math
import sqlite3
import tempfile
from pathlib import Path

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# DB fixture helpers (formula strings are real/balanced, unlike
# test_phase3.py's fixture which never varies `formula` across species —
# element-balance testing needs accurate per-species formulas)
# ─────────────────────────────────────────────────────────────────────────────

def _make_db(path: str) -> sqlite3.Connection:
    from rmip.database.schema import init_db
    from rmip.database.schema_phase3 import upgrade_db
    from rmip.database.migrations import run_phase3_migrations

    conn = init_db(path)
    upgrade_db(conn)
    run_phase3_migrations(conn)
    return conn


def _insert_mechanism(conn, name: str, n_sp: int, n_rxn: int) -> int:
    cur = conn.execute(
        "INSERT INTO mechanisms (name, format, filepath, description, "
        "n_species, n_reactions, loaded_at, metadata_json) VALUES (?,?,?,?,?,?,?,?)",
        (name, "chemkin", "/fake.inp", f"Test {name}", n_sp, n_rxn,
         "2024-01-01T00:00:00Z", "{}"),
    )
    return cur.lastrowid


def _insert_species(conn, mech_id: int, name: str, inchi_key: str, formula: str) -> int:
    cur = conn.execute(
        "INSERT INTO species (mechanism_id, name, inchi_key, formula, "
        "canonical_smiles, thermo_type, tmin_K, tmid_K, tmax_K, "
        "coeffs_low_json, coeffs_high_json, glossary_matched) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (mech_id, name, inchi_key, formula, f"[{name}]", "nasa7",
         200.0, 1000.0, 5000.0,
         json.dumps([1.0]*7), json.dumps([2.0]*7), 1),
    )
    return cur.lastrowid


def _insert_reaction(conn, mech_id: int, formula_string: str, canonical_key: str,
                      A_si: float = 1e10, n: float = 0.5, Ea_si: float = 50000.0) -> int:
    cur = conn.execute(
        "INSERT INTO reactions (mechanism_id, formula_string, canonical_key, "
        "reversible, kinetics_type, A_si, A_orig, A_units, n, Ea_si, Ea_orig, Ea_units) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (mech_id, formula_string, canonical_key, 1, "arrhenius",
         A_si, A_si, "SI", n, Ea_si, Ea_si, "J/mol"),
    )
    return cur.lastrowid


def _insert_stoich(conn, rxn_id: int, sp_id: int, coeff: float, role: str) -> None:
    conn.execute(
        "INSERT INTO stoichiometry (reaction_id, species_id, stoich_coeff, role) VALUES (?,?,?,?)",
        (rxn_id, sp_id, coeff, role),
    )


# ─────────────────────────────────────────────────────────────────────────────
# H2 combustion fixture: 2 mechanisms, one VERBATIM + one COPY + one FUSE
# reaction, all element-balanced.
#
#   Species (shared inchi_keys across both mechs): H2, O2, H2O, H, O, OH
#
#   R_A "2 H2 + O2 <=> 2 H2O"   -- identical A in both mechs -> COPY
#   R_B "H2 + O <=> H + OH"     -- divergent A across mechs  -> FUSE
#   R_C "2 O + M <=> O2 + M"    -- present in mech1 only     -> VERBATIM
# ─────────────────────────────────────────────────────────────────────────────

def _build_h2_fixture(tmp_path) -> tuple:
    db = str(tmp_path / "fusion_stub.db")
    conn = _make_db(db)

    mid1 = _insert_mechanism(conn, "Mech1", n_sp=6, n_rxn=3)
    mid2 = _insert_mechanism(conn, "Mech2", n_sp=6, n_rxn=2)

    sp = {}
    for mid_key, mid in (("m1", mid1), ("m2", mid2)):
        sp[(mid_key, "H2")]  = _insert_species(conn, mid, "H2",  "IK_H2",  "H2")
        sp[(mid_key, "O2")]  = _insert_species(conn, mid, "O2",  "IK_O2",  "O2")
        sp[(mid_key, "H2O")] = _insert_species(conn, mid, "H2O", "IK_H2O", "H2O")
        sp[(mid_key, "H")]   = _insert_species(conn, mid, "H",   "IK_H",   "H")
        sp[(mid_key, "O")]   = _insert_species(conn, mid, "O",   "IK_O",   "O")
        sp[(mid_key, "OH")]  = _insert_species(conn, mid, "OH",  "IK_OH",  "HO")

    # R_A: COPY (identical A in both mechs)
    ra1 = _insert_reaction(conn, mid1, "2 H2 + O2 <=> 2 H2O", "CK_A", A_si=1e10)
    ra2 = _insert_reaction(conn, mid2, "2 H2 + O2 <=> 2 H2O", "CK_A", A_si=1e10)
    _insert_stoich(conn, ra1, sp[("m1", "H2")],  2.0, "reactant")
    _insert_stoich(conn, ra1, sp[("m1", "O2")],  1.0, "reactant")
    _insert_stoich(conn, ra1, sp[("m1", "H2O")], 2.0, "product")
    _insert_stoich(conn, ra2, sp[("m2", "H2")],  2.0, "reactant")
    _insert_stoich(conn, ra2, sp[("m2", "O2")],  1.0, "reactant")
    _insert_stoich(conn, ra2, sp[("m2", "H2O")], 2.0, "product")

    # R_B: FUSE (divergent A)
    rb1 = _insert_reaction(conn, mid1, "H2 + O <=> H + OH", "CK_B", A_si=1e8)
    rb2 = _insert_reaction(conn, mid2, "H2 + O <=> H + OH", "CK_B", A_si=1e11)
    _insert_stoich(conn, rb1, sp[("m1", "H2")], 1.0, "reactant")
    _insert_stoich(conn, rb1, sp[("m1", "O")],  1.0, "reactant")
    _insert_stoich(conn, rb1, sp[("m1", "H")],  1.0, "product")
    _insert_stoich(conn, rb1, sp[("m1", "OH")], 1.0, "product")
    _insert_stoich(conn, rb2, sp[("m2", "H2")], 1.0, "reactant")
    _insert_stoich(conn, rb2, sp[("m2", "O")],  1.0, "reactant")
    _insert_stoich(conn, rb2, sp[("m2", "H")],  1.0, "product")
    _insert_stoich(conn, rb2, sp[("m2", "OH")], 1.0, "product")

    # R_C: VERBATIM (mech1 only) -- third body M never a stoichiometry row
    rc1 = _insert_reaction(conn, mid1, "2 O + M <=> O2 + M", "CK_C", A_si=5e9)
    _insert_stoich(conn, rc1, sp[("m1", "O")],  2.0, "reactant")
    _insert_stoich(conn, rc1, sp[("m1", "O2")], 1.0, "product")

    conn.commit()
    return db, mid1, mid2


def _build_spec(db: str, mid1: int, mid2: int):
    """Build a real UnionGraphSpec via GraphComparator.build_union_spec(),
    fed a hand-built GraphComparisonResult (mirrors test_phase3.py's
    _make_comparison_result pattern -- see this file's module docstring for
    why real project.db mechanisms can't produce a non-empty one today)."""
    from rmip.types import (
        GraphComparisonResult, ReactionClassification, SpeciesClassification,
        ReactionClass, SpeciesClass,
    )
    from rmip.analysis.graph_comparator import GraphComparator

    sp_clsf = [
        SpeciesClassification(inchi_key="IK_H2", canonical_smiles="[H][H]",
                               formula="H2", classification=SpeciesClass.UNIVERSAL,
                               present_in_mech_ids=[mid1, mid2]),
        SpeciesClassification(inchi_key="IK_O2", canonical_smiles="[O][O]",
                               formula="O2", classification=SpeciesClass.UNIVERSAL,
                               present_in_mech_ids=[mid1, mid2]),
        SpeciesClassification(inchi_key="IK_H2O", canonical_smiles="O",
                               formula="H2O", classification=SpeciesClass.UNIVERSAL,
                               present_in_mech_ids=[mid1, mid2]),
        SpeciesClassification(inchi_key="IK_H", canonical_smiles="[H]",
                               formula="H", classification=SpeciesClass.UNIVERSAL,
                               present_in_mech_ids=[mid1, mid2]),
        SpeciesClassification(inchi_key="IK_O", canonical_smiles="[O]",
                               formula="O", classification=SpeciesClass.UNIVERSAL,
                               present_in_mech_ids=[mid1, mid2]),
        SpeciesClassification(inchi_key="IK_OH", canonical_smiles="[OH]",
                               formula="HO", classification=SpeciesClass.UNIVERSAL,
                               present_in_mech_ids=[mid1, mid2]),
    ]
    rxn_clsf = [
        ReactionClassification(
            canonical_key="CK_A", formula_string="2 H2 + O2 <=> 2 H2O",
            classification=ReactionClass.SHARED_IDENTICAL,
            present_in_mech_ids=[mid1, mid2],
        ),
        ReactionClassification(
            canonical_key="CK_B", formula_string="H2 + O <=> H + OH",
            classification=ReactionClass.SHARED_DIVERGENT,
            present_in_mech_ids=[mid1, mid2],
            mean_A_si=math.sqrt(1e8 * 1e11), mean_n=0.5, mean_Ea_si=50000.0,
            min_A_si=1e8, max_A_si=1e11, min_n=0.5, max_n=0.5,
            min_Ea_si=50000.0, max_Ea_si=50000.0,
        ),
        ReactionClassification(
            canonical_key="CK_C", formula_string="2 O + M <=> O2 + M",
            classification=ReactionClass.UNIQUE, present_in_mech_ids=[mid1],
        ),
    ]
    result = GraphComparisonResult(
        session_id=1, mechanism_ids=[mid1, mid2],
        species_classifications=sp_clsf, reaction_classifications=rxn_clsf,
    )
    cmp = GraphComparator(db)
    return cmp.build_union_spec(result)


# ─────────────────────────────────────────────────────────────────────────────
# fuse()
# ─────────────────────────────────────────────────────────────────────────────

class TestFusionEngineFuse:
    def test_spec_bucket_reconstruction_is_lossless(self, tmp_path):
        """The core assumption behind stubs.FusionEngine.fuse(): concatenating
        spec's 3 reaction buckets + species_map reproduces the original flat
        classification lists exactly."""
        db, mid1, mid2 = _build_h2_fixture(tmp_path)
        spec = _build_spec(db, mid1, mid2)

        all_reactions = (
            list(spec.verbatim_reactions.values())
            + list(spec.copy_reactions.values())
            + list(spec.fuse_reactions.values())
        )
        assert {r.canonical_key for r in all_reactions} == {"CK_A", "CK_B", "CK_C"}
        assert set(spec.species_map.keys()) == {"IK_H2", "IK_O2", "IK_H2O", "IK_H", "IK_O", "IK_OH"}

    def test_fuse_creates_new_mechanism_with_correct_counts(self, tmp_path):
        from rmip.stubs import FusionEngine

        db, mid1, mid2 = _build_h2_fixture(tmp_path)
        spec = _build_spec(db, mid1, mid2)

        engine = FusionEngine(spec, db)
        out_dir = str(tmp_path / "out")
        fused_id = engine.fuse("H2_Fused", output_dir=out_dir)

        assert fused_id > mid2  # new row, higher id than sources

        with sqlite3.connect(db) as conn:
            mech_row = conn.execute(
                "SELECT format, n_species, n_reactions FROM mechanisms WHERE id=?", (fused_id,)
            ).fetchone()
            assert mech_row[0] == "fused"
            assert mech_row[2] == 3  # CK_A, CK_B, CK_C

            treatments = dict(conn.execute(
                "SELECT canonical_key, fusion_class FROM reactions WHERE mechanism_id=?", (fused_id,)
            ).fetchall())
        assert treatments["CK_A"] == "COPY"
        assert treatments["CK_B"] == "FUSE"
        assert treatments["CK_C"] == "VERBATIM"

    def test_fuse_writes_chemkin_files(self, tmp_path):
        from rmip.stubs import FusionEngine

        db, mid1, mid2 = _build_h2_fixture(tmp_path)
        spec = _build_spec(db, mid1, mid2)
        engine = FusionEngine(spec, db)
        out_dir = tmp_path / "out"
        engine.fuse("H2_Fused", output_dir=str(out_dir))

        inp_path = out_dir / "H2_Fused.inp"
        dat_path = out_dir / "H2_Fused.dat"
        assert inp_path.exists() and inp_path.stat().st_size > 0
        assert dat_path.exists() and dat_path.stat().st_size > 0
        inp_text = inp_path.read_text()
        assert "REACTIONS" in inp_text
        assert "END" in inp_text

    def test_fuse_overrides_apply_to_fuse_reaction(self, tmp_path):
        from rmip.stubs import FusionEngine

        db, mid1, mid2 = _build_h2_fixture(tmp_path)
        spec = _build_spec(db, mid1, mid2)
        engine = FusionEngine(spec, db)

        fused_id = engine.fuse(
            "H2_Fused", output_dir=str(tmp_path / "out"),
            fuse_overrides={"CK_B": {"A_si": 9.9e9, "n": 0.75, "Ea_si": 12345.0}},
        )

        with sqlite3.connect(db) as conn:
            row = conn.execute(
                "SELECT A_si, n, Ea_si FROM reactions WHERE mechanism_id=? AND canonical_key=?",
                (fused_id, "CK_B"),
            ).fetchone()
        assert row == (9.9e9, 0.75, 12345.0)

    def test_fuse_without_overrides_keeps_geometric_mean(self, tmp_path):
        from rmip.stubs import FusionEngine

        db, mid1, mid2 = _build_h2_fixture(tmp_path)
        spec = _build_spec(db, mid1, mid2)
        engine = FusionEngine(spec, db)
        fused_id = engine.fuse("H2_Fused", output_dir=str(tmp_path / "out"))

        with sqlite3.connect(db) as conn:
            a_si = conn.execute(
                "SELECT A_si FROM reactions WHERE mechanism_id=? AND canonical_key='CK_B'",
                (fused_id,),
            ).fetchone()[0]
        assert math.isclose(a_si, math.sqrt(1e8 * 1e11), rel_tol=1e-4)

    @pytest.mark.skipif(
        __import__("shutil").which("ck2yaml") is None, reason="ck2yaml not on PATH",
    )
    def test_fuse_write_cantera_produces_yaml(self, tmp_path):
        from rmip.stubs import FusionEngine

        db, mid1, mid2 = _build_h2_fixture(tmp_path)
        spec = _build_spec(db, mid1, mid2)
        engine = FusionEngine(spec, db)
        out_dir = tmp_path / "out"
        engine.fuse("H2_Fused", output_dir=str(out_dir), write_cantera=True)

        yaml_path = out_dir / "H2_Fused.yaml"
        assert yaml_path.exists() and yaml_path.stat().st_size > 0


# ─────────────────────────────────────────────────────────────────────────────
# validate_element_balance()
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateElementBalance:
    def test_balanced_fused_mechanism_reports_no_violations(self, tmp_path):
        from rmip.stubs import FusionEngine

        db, mid1, mid2 = _build_h2_fixture(tmp_path)
        spec = _build_spec(db, mid1, mid2)
        engine = FusionEngine(spec, db)
        fused_id = engine.fuse("H2_Fused", output_dir=str(tmp_path / "out"))

        report = engine.validate_element_balance(fused_id)
        assert report == {"balanced": True, "violations": []}

    def test_unbalanced_reaction_is_flagged(self, tmp_path):
        """Directly construct a tiny DB with one deliberately-unbalanced
        reaction (extra O on the product side with no source) to verify the
        violation-detection path in isolation from the fuse() pipeline."""
        from rmip.stubs import FusionEngine
        from rmip.types import UnionGraphSpec

        db = str(tmp_path / "unbalanced.db")
        conn = _make_db(db)
        mid = _insert_mechanism(conn, "Broken", n_sp=2, n_rxn=1)
        h2 = _insert_species(conn, mid, "H2", "IK_H2", "H2")
        o3 = _insert_species(conn, mid, "O3", "IK_O3", "O3")  # extra O, unbalanced on purpose
        rxn = _insert_reaction(conn, mid, "H2 <=> O3", "CK_BROKEN")
        _insert_stoich(conn, rxn, h2, 1.0, "reactant")
        _insert_stoich(conn, rxn, o3, 1.0, "product")
        conn.commit()

        # validate_element_balance() only needs db_path -- spec is unused for
        # this method, so an empty placeholder spec is fine.
        spec = UnionGraphSpec(mechanism_ids=[mid], comparison_session_id=0)
        engine = FusionEngine(spec, db)
        report = engine.validate_element_balance(mid)

        assert report["balanced"] is False
        assert "H2 <=> O3" in report["violations"]

    def test_empty_mechanism_reports_balanced(self, tmp_path):
        from rmip.stubs import FusionEngine
        from rmip.types import UnionGraphSpec

        db = str(tmp_path / "empty.db")
        conn = _make_db(db)
        mid = _insert_mechanism(conn, "Empty", n_sp=0, n_rxn=0)
        conn.commit()

        spec = UnionGraphSpec(mechanism_ids=[mid], comparison_session_id=0)
        engine = FusionEngine(spec, db)
        report = engine.validate_element_balance(mid)
        assert report == {"balanced": True, "violations": []}
