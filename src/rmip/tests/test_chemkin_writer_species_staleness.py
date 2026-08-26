"""
rmip/tests/test_chemkin_writer_species_staleness.py
=====================================================
Regression coverage for the "stale formula_string species name" bug found
2026-07-17 while stress-testing mechanism fusion through the GUI with 3
dummy mechanisms that deliberately spell every shared species differently.

`ChemkinWriter._format_rxn_string()` used to trust each reaction's cached
`formula_string` column verbatim. That column is written once at PARSE
time using the reaction's ORIGINAL SOURCE mechanism's own species spelling.
After fusion picks ONE canonical representative name per species identity
for the merged SPECIES block, a reaction whose formula_string came from a
different source mechanism still names the OLD spelling -- a species never
declared in the fused mechanism's own SPECIES block. Real mechanisms with
even slightly different spellings for the same species (case differences,
different literature naming) hit this, not just an artificial torture test.

Fixed: `_format_rxn_string` now always rebuilds the reaction string from
`stoich` (species names via a proper FK join against THIS mechanism's own
species table), falling back to formula_string only when stoich is empty.
The one thing formula_string uniquely carried -- specific-collider
annotations like "(+M)"/"(+AR)" for falloff reactions, stripped from
stoichiometry at parse time with no other persisted home -- is preserved by
extracting it via regex and re-applying it to the freshly-built string.
"""
from __future__ import annotations

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
        """
        INSERT INTO mechanisms (name, format, filepath, description,
                                n_species, n_reactions, loaded_at, metadata_json)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (name, "chemkin", "/fake/path.inp", f"Test mechanism {name}", 0, 0,
         "2024-01-01T00:00:00Z", "{}"),
    )
    return cur.lastrowid


def _insert_species(conn, mech_id, name, formula="H2") -> int:
    coeffs = [1.0, 0.0, 0.0, 0.0, 0.0, -100.0, 1.0]
    cur = conn.execute(
        """
        INSERT INTO species (mechanism_id, name, inchi_key, formula,
                             canonical_smiles, thermo_type,
                             tmin_K, tmid_K, tmax_K,
                             coeffs_low_json, coeffs_high_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        (mech_id, name, f"IK_{name}", formula, "", "nasa7",
         200.0, 1000.0, 5000.0, json.dumps(coeffs), json.dumps(coeffs)),
    )
    return cur.lastrowid


def _insert_reaction(conn, mech_id, formula_string, kinetics_type="arrhenius",
                      kinetics_data_json=None) -> int:
    cur = conn.execute(
        """
        INSERT INTO reactions
            (mechanism_id, formula_string, canonical_key,
             reversible, kinetics_type, A_si, n, Ea_si, kinetics_data_json,
             has_third_body, pressure_dependent)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        (mech_id, formula_string, "R1", 1, kinetics_type,
         1.0e10, 0.5, 50000.0, kinetics_data_json, 1, 1),
    )
    return cur.lastrowid


def _insert_stoich(conn, rxn_id, sp_id, coef, role) -> None:
    conn.execute(
        "INSERT INTO stoichiometry (reaction_id, species_id, stoich_coeff, role) VALUES (?,?,?,?)",
        (rxn_id, sp_id, coef, role),
    )


class TestStaleFormulaStringNotTrusted:
    def test_reaction_string_uses_declared_species_names_not_stale_formula_string(self, tmp_path):
        """formula_string says "protium" (a name from a DIFFERENT, now-
        discarded source mechanism spelling) but this mechanism's own
        SPECIES table (and this reaction's stoichiometry FK) says
        "canonical_h". The written REACTIONS block must use "canonical_h",
        never the stale "protium" -- otherwise the written .inp declares
        SPECIES canonical_h/... but its own REACTIONS block references an
        undeclared species, which ck2yaml/Zero-RK both reject."""
        db = str(tmp_path / "stale.db")
        conn = _make_db(db)
        mid = _insert_mechanism(conn, "M1")
        sp_h  = _insert_species(conn, mid, "canonical_h", formula="H")
        sp_o2 = _insert_species(conn, mid, "canonical_o2", formula="O2")
        sp_o  = _insert_species(conn, mid, "canonical_o", formula="O")
        sp_oh = _insert_species(conn, mid, "canonical_oh", formula="OH")
        # Stale formula_string: names from a source mechanism that no
        # longer matches this mechanism's own SPECIES declarations.
        rxn = _insert_reaction(conn, mid, "protium+dioxygen<=>oxo+hydroxyl_x")
        _insert_stoich(conn, rxn, sp_h, 1.0, "reactant")
        _insert_stoich(conn, rxn, sp_o2, 1.0, "reactant")
        _insert_stoich(conn, rxn, sp_o, 1.0, "product")
        _insert_stoich(conn, rxn, sp_oh, 1.0, "product")
        conn.commit()
        conn.close()

        from rmip.io.chemkin_writer import ChemkinWriter
        out_path = str(tmp_path / "out.inp")
        ChemkinWriter(db).write(mid, out_path, include_thermo_in_mech=False)
        text = open(out_path).read()

        assert "protium" not in text, "stale source-mechanism species name must not leak into output"
        assert "dioxygen" not in text
        assert "oxo" not in text
        assert "hydroxyl_x" not in text
        assert "canonical_h+canonical_o2<=>canonical_o+canonical_oh" in text.replace(" ", "")

    def test_specific_collider_annotation_still_preserved(self, tmp_path):
        """The one thing formula_string uniquely carries -- a specific
        third-body collider tag like "(+AR)" on a falloff reaction, which
        stoichiometry alone can't reconstruct -- must survive even though
        the species-name text itself is no longer trusted from
        formula_string."""
        db = str(tmp_path / "collider.db")
        conn = _make_db(db)
        mid = _insert_mechanism(conn, "M1")
        sp_h  = _insert_species(conn, mid, "canonical_h", formula="H")
        sp_o2 = _insert_species(conn, mid, "canonical_o2", formula="O2")
        sp_ho2 = _insert_species(conn, mid, "canonical_ho2", formula="HO2")
        rxn = _insert_reaction(
            conn, mid, "protium+dioxygen(+AR)<=>hydroperoxyl_x(+AR)",
            kinetics_type="troe",
            kinetics_data_json=json.dumps({
                "low": {"A_si": 1.0e9, "n": 0.0, "Ea_si": 0.0},
                "alpha": 0.5, "T3": 1e-30, "T1": 1e30,
            }),
        )
        _insert_stoich(conn, rxn, sp_h, 1.0, "reactant")
        _insert_stoich(conn, rxn, sp_o2, 1.0, "reactant")
        _insert_stoich(conn, rxn, sp_ho2, 1.0, "product")
        conn.commit()
        conn.close()

        from rmip.io.chemkin_writer import ChemkinWriter
        out_path = str(tmp_path / "out.inp")
        ChemkinWriter(db).write(mid, out_path, include_thermo_in_mech=False)
        text = open(out_path).read()

        assert "protium" not in text
        assert "(+AR)" in text
        compact = text.replace(" ", "")
        assert "canonical_h+canonical_o2(+AR)<=>canonical_ho2(+AR)" in compact
