"""
rmip/tests/test_fusion_a_orig_units.py
========================================
Regression coverage for the 2026-07-17 fusion ignition bug (bug.md,
"real-mechanism second-tier GUI stress test" section): FusionEngine's
VERBATIM/COPY/PRIORITY record builders wrote the fused reaction's A_orig
column equal to A_si (SI-basis, m^3/mol/s for a bimolecular reaction)
instead of the true original-units (CGS, cm^3/mol/s) value carried on the
source row -- a factor of 1e6 too small.

ChemkinWriter's default use_original_units=True export path takes a
non-NULL A_orig completely literally (it does not actually check A_units),
so every such export silently wrote the wrong A-factor under a CHEMKIN
{length: cm, quantity: mol} header. Confirmed end-to-end: a real H2/O2/Ar
mixture that ignites in 3us standalone showed zero temperature rise after
50ms in the fused mechanism.

Fix: VERBATIM/COPY/PRIORITY records now copy A_orig/A_units/Ea_orig/
Ea_units verbatim from the chosen source row (same convention
cantera_reader.py already uses for its own A_orig=None case, see
test_cantera_reader.py::test_A_orig_left_null_not_mislabeled). The
arithmetic-mean FUSE bucket has no single source's original-units value to
point to, so it leaves A_orig/Ea_orig as NULL -- ChemkinWriter's
A_orig-is-None fallback then correctly derives CGS units from A_si via a
real SI->CGS conversion instead of writing the SI number out unconverted.
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


def _insert_reaction_with_orig(
    conn: sqlite3.Connection, mech_id: int, formula_string: str,
    canonical_key: Optional[str],
    A_si: float, A_orig: float, A_units: str,
    n: float, Ea_si: float, Ea_orig: float, Ea_units: str,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO reactions
            (mechanism_id, formula_string, canonical_key, reversible,
             kinetics_type, A_si, A_orig, A_units, n,
             Ea_si, Ea_orig, Ea_units)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (mech_id, formula_string, canonical_key, 1, "arrhenius",
         A_si, A_orig, A_units, n, Ea_si, Ea_orig, Ea_units),
    )
    return cur.lastrowid


def _insert_stoich(conn, rxn_id, sp_id, coef, role) -> None:
    conn.execute(
        "INSERT INTO stoichiometry (reaction_id, species_id, stoich_coeff, role) VALUES (?,?,?,?)",
        (rxn_id, sp_id, coef, role),
    )


# Real Keromnes-2013 O+H2<=>H+OH values that exposed this bug: SI A is
# 1e6x smaller than the true CGS original (bimolecular SI->CGS factor).
A_SI   = 0.0508
A_ORIG = 50800.0
EA_SI  = 26317.0
EA_ORIG_CAL = 6290.0


class TestVerbatimCopyPreservesTrueOriginalUnits:
    def _build_shared_reaction_mech(self, conn, mech_id: int) -> None:
        h2 = _insert_species(conn, mech_id, "H2", "IK_H2", formula="H2")
        oh = _insert_species(conn, mech_id, "OH", "IK_OH", formula="OH")
        o  = _insert_species(conn, mech_id, "O",  "IK_O",  formula="O")
        h  = _insert_species(conn, mech_id, "H",  "IK_H",  formula="H")
        r = _insert_reaction_with_orig(
            conn, mech_id, "O+H2<=>H+OH", "RK_O_H2",
            A_si=A_SI, A_orig=A_ORIG, A_units="cm3/mol/s",
            n=2.67, Ea_si=EA_SI, Ea_orig=EA_ORIG_CAL, Ea_units="cal/mol",
        )
        _insert_stoich(conn, r, o,  1.0, "reactant")
        _insert_stoich(conn, r, h2, 1.0, "reactant")
        _insert_stoich(conn, r, h,  1.0, "product")
        _insert_stoich(conn, r, oh, 1.0, "product")

    def test_copy_treatment_preserves_true_A_orig(self, tmp_path):
        """Two identical mechanisms -> UNIVERSAL/COPY. Pre-fix, the fused
        row's A_orig was silently overwritten with A_si (0.0508) instead of
        the true CGS value (50800.0) -- a 1e6x understatement that
        ChemkinWriter would then export verbatim as if it needed no
        conversion."""
        db = str(tmp_path / "copy.db")
        conn = _make_db(db)
        m1 = _insert_mechanism(conn, "M1")
        m2 = _insert_mechanism(conn, "M2")
        self._build_shared_reaction_mech(conn, m1)
        self._build_shared_reaction_mech(conn, m2)
        conn.commit()
        conn.close()

        from rmip.analysis.graph_comparator import GraphComparator
        from rmip.analysis.fusion import FusionEngine

        cmp_result = GraphComparator(db).compare([m1, m2])
        result = FusionEngine(db).fuse(cmp_result, name="Fused_copy")

        assert len(result.reaction_records) == 1
        rec = result.reaction_records[0]
        assert rec.A_si == pytest.approx(A_SI)
        assert rec.A_orig == pytest.approx(A_ORIG)
        assert rec.A_orig != pytest.approx(rec.A_si)
        assert rec.Ea_orig == pytest.approx(EA_ORIG_CAL)

        # And confirm what actually landed in the DB, since that's what
        # ChemkinWriter reads.
        conn = sqlite3.connect(db)
        row = conn.execute(
            "SELECT A_si, A_orig, A_units, Ea_orig, Ea_units FROM reactions "
            "WHERE mechanism_id=?", (result.fused_mechanism_id,),
        ).fetchone()
        conn.close()
        db_A_si, db_A_orig, db_A_units, db_Ea_orig, db_Ea_units = row
        assert db_A_orig == pytest.approx(A_ORIG)
        assert db_A_units == "cm3/mol/s"
        assert db_Ea_orig == pytest.approx(EA_ORIG_CAL)
        assert db_Ea_units == "cal/mol"

    def test_priority_treatment_preserves_true_A_orig(self, tmp_path):
        """Priority-pick mode: same source-of-truth requirement as COPY."""
        db = str(tmp_path / "priority.db")
        conn = _make_db(db)
        m1 = _insert_mechanism(conn, "Keromnes")
        m2 = _insert_mechanism(conn, "Li")
        self._build_shared_reaction_mech(conn, m1)
        # m2 has a different (lower-priority) A value for the same reaction
        h2 = _insert_species(conn, m2, "H2", "IK_H2", formula="H2")
        oh = _insert_species(conn, m2, "OH", "IK_OH", formula="OH")
        o  = _insert_species(conn, m2, "O",  "IK_O",  formula="O")
        h  = _insert_species(conn, m2, "H",  "IK_H",  formula="H")
        r = _insert_reaction_with_orig(
            conn, m2, "O+H2<=>H+OH", "RK_O_H2",
            A_si=0.099, A_orig=99000.0, A_units="cm3/mol/s",
            n=2.6, Ea_si=26500.0, Ea_orig=6335.0, Ea_units="cal/mol",
        )
        _insert_stoich(conn, r, o,  1.0, "reactant")
        _insert_stoich(conn, r, h2, 1.0, "reactant")
        _insert_stoich(conn, r, h,  1.0, "product")
        _insert_stoich(conn, r, oh, 1.0, "product")
        conn.commit()
        conn.close()

        from rmip.analysis.graph_comparator import GraphComparator
        from rmip.analysis.fusion import FusionEngine

        cmp_result = GraphComparator(db).compare([m1, m2])
        result = FusionEngine(db).fuse(
            cmp_result, name="Fused_priority", priority_order=[m1, m2],
        )

        assert len(result.reaction_records) == 1
        rec = result.reaction_records[0]
        # Priority picks m1 (Keromnes) verbatim, including its true A_orig.
        assert rec.A_si == pytest.approx(A_SI)
        assert rec.A_orig == pytest.approx(A_ORIG)


class TestFuseBucketLeavesOrigNull:
    def test_averaged_fuse_record_has_no_fabricated_A_orig(self, tmp_path):
        """When mechanisms disagree (real FUSE/mean bucket), the resulting
        A_si is a mean with no single source's original-units
        value behind it -- A_orig/Ea_orig must be None (NULL), not a copy of
        the mean SI value under a claimed 'SI' A_units, so ChemkinWriter's
        A_orig-is-None fallback does the real SI->CGS conversion instead of
        writing the SI-magnitude number out unconverted."""
        db = str(tmp_path / "fuse.db")
        conn = _make_db(db)
        m1 = _insert_mechanism(conn, "M1")
        m2 = _insert_mechanism(conn, "M2")
        for mid, A_si, A_orig in ((m1, 1.0e10, 1.0e16), (m2, 2.0e10, 2.0e16)):
            h2 = _insert_species(conn, mid, "H2", "IK_H2", formula="H2")
            o2 = _insert_species(conn, mid, "O2", "IK_O2", formula="O2")
            r = _insert_reaction_with_orig(
                conn, mid, "H2+O2<=>H2O2", "R_DISAGREE",
                A_si=A_si, A_orig=A_orig, A_units="cm3/mol/s",
                n=0.5, Ea_si=50000.0, Ea_orig=11950.0, Ea_units="cal/mol",
            )
            _insert_stoich(conn, r, h2, 1.0, "reactant")
            _insert_stoich(conn, r, o2, 1.0, "reactant")
        conn.commit()
        conn.close()

        from rmip.analysis.graph_comparator import GraphComparator
        from rmip.analysis.fusion import FusionEngine

        cmp_result = GraphComparator(db).compare([m1, m2])
        result = FusionEngine(db).fuse(cmp_result, name="Fused_disagree")

        assert len(result.reaction_records) == 1
        rec = result.reaction_records[0]
        assert rec.treatment.value.lower() == "fuse"
        assert rec.A_si == pytest.approx((1.0e10 * 2.0e10) ** 0.5)  # geometric mean
        assert rec.A_orig is None
        assert rec.Ea_orig is None

        conn = sqlite3.connect(db)
        row = conn.execute(
            "SELECT A_orig, Ea_orig FROM reactions WHERE mechanism_id=?",
            (result.fused_mechanism_id,),
        ).fetchone()
        conn.close()
        assert row == (None, None)


class TestChemkinExportUsesTrueUnits:
    def test_verbatim_export_writes_true_cgs_A_not_si_magnitude(self, tmp_path):
        """End-to-end: write the fused mechanism to CHEMKIN with
        use_original_units=True (ChemkinWriter's default, and what
        mean_value_fusion_gui.py / rmip.stubs.FusionEngine use) and confirm
        the emitted A-factor is the true CGS value, not the SI-magnitude
        number that used to leak through as A_orig."""
        db = str(tmp_path / "export.db")
        conn = _make_db(db)
        m1 = _insert_mechanism(conn, "M1")
        m2 = _insert_mechanism(conn, "M2")
        h2 = _insert_species(conn, m1, "H2", "IK_H2", formula="H2")
        oh = _insert_species(conn, m1, "OH", "IK_OH", formula="OH")
        o  = _insert_species(conn, m1, "O",  "IK_O",  formula="O")
        h  = _insert_species(conn, m1, "H",  "IK_H",  formula="H")
        r = _insert_reaction_with_orig(
            conn, m1, "O+H2<=>H+OH", "RK_O_H2",
            A_si=A_SI, A_orig=A_ORIG, A_units="cm3/mol/s",
            n=2.67, Ea_si=EA_SI, Ea_orig=EA_ORIG_CAL, Ea_units="cal/mol",
        )
        _insert_stoich(conn, r, o,  1.0, "reactant")
        _insert_stoich(conn, r, h2, 1.0, "reactant")
        _insert_stoich(conn, r, h,  1.0, "product")
        _insert_stoich(conn, r, oh, 1.0, "product")
        h2b = _insert_species(conn, m2, "H2", "IK_H2", formula="H2")
        ohb = _insert_species(conn, m2, "OH", "IK_OH", formula="OH")
        ob  = _insert_species(conn, m2, "O",  "IK_O",  formula="O")
        hb  = _insert_species(conn, m2, "H",  "IK_H",  formula="H")
        r2 = _insert_reaction_with_orig(
            conn, m2, "O+H2<=>H+OH", "RK_O_H2",
            A_si=A_SI, A_orig=A_ORIG, A_units="cm3/mol/s",
            n=2.67, Ea_si=EA_SI, Ea_orig=EA_ORIG_CAL, Ea_units="cal/mol",
        )
        _insert_stoich(conn, r2, ob,  1.0, "reactant")
        _insert_stoich(conn, r2, h2b, 1.0, "reactant")
        _insert_stoich(conn, r2, hb,  1.0, "product")
        _insert_stoich(conn, r2, ohb, 1.0, "product")
        conn.commit()
        conn.close()

        from rmip.analysis.graph_comparator import GraphComparator
        from rmip.analysis.fusion import FusionEngine
        from rmip.io.chemkin_writer import ChemkinWriter

        cmp_result = GraphComparator(db).compare([m1, m2])
        result = FusionEngine(db).fuse(cmp_result, name="Fused_export")

        out_path = str(tmp_path / "fused.inp")
        writer = ChemkinWriter(db, use_original_units=True)
        writer.write(result.fused_mechanism_id, out_path)

        text = open(out_path).read()
        # The true CGS A-factor (50800, formatted as 5.08...E+04) must
        # appear; the wrong SI-magnitude number (0.0508, i.e. 5.08E-02)
        # must not.
        assert "5.080000E+04" in text
        assert "5.080000E-02" not in text
