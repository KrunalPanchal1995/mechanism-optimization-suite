"""
rmip/tests/test_materialize.py
==============================
Tests for the shared "materialize reduced/subset mechanism" primitive
(``rmip/reduction/materialize.py``, Work Package 0).

A full mechanism (species + reactions + stoichiometry) is parsed into a temp DB
from Cantera's shipped ``h2o2.yaml``; subsets are then materialised and checked
structurally and (for the balanced subset) by reloading in Cantera.

Run with:
    python -m pytest src/rmip/tests/test_materialize.py -v
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

ct = pytest.importorskip("cantera")

from rmip.reduction.materialize import materialize_subset  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture — parse h2o2.yaml into a temp RMIP DB
# ---------------------------------------------------------------------------

@pytest.fixture()
def h2o2_full_db(tmp_path):
    from rmip.config import RMIPConfig
    from rmip.core.parser import MechanismParser

    yaml_path = Path(ct.__path__[0]) / "data" / "h2o2.yaml"
    db_path = str(tmp_path / "mat_test.db")
    parser = MechanismParser(RMIPConfig(db_path=db_path))
    mech_id = parser.parse(mech_path=str(yaml_path))
    parser.conn.close()
    return db_path, mech_id


def _names(db_path, mech_id):
    with sqlite3.connect(db_path) as conn:
        return {r[0] for r in conn.execute(
            "SELECT name FROM species WHERE mechanism_id=?", (mech_id,))}


def _formulas(db_path, mech_id):
    with sqlite3.connect(db_path) as conn:
        return [r[0] for r in conn.execute(
            "SELECT formula_string FROM reactions WHERE mechanism_id=?", (mech_id,))]


def _rxn_ids_with_species(db_path, mech_id, sp_name):
    with sqlite3.connect(db_path) as conn:
        return {r[0] for r in conn.execute(
            """SELECT DISTINCT s.reaction_id FROM stoichiometry s
               JOIN species sp ON sp.id = s.species_id
               JOIN reactions r ON r.id = s.reaction_id
               WHERE r.mechanism_id=? AND sp.name=?""", (mech_id, sp_name))}


# ---------------------------------------------------------------------------
# Structural subset
# ---------------------------------------------------------------------------

def test_subset_drops_species_and_skips_its_reactions(h2o2_full_db):
    db_path, mech_id = h2o2_full_db
    full_names = _names(db_path, mech_id)
    assert "H2O2" in full_names
    all_formulas = _formulas(db_path, mech_id)
    n_rxn_with_h2o2 = len(_rxn_ids_with_species(db_path, mech_id, "H2O2"))
    assert n_rxn_with_h2o2 > 0

    # retain every species except H2O2, offer ALL reactions → reactions that
    # reference H2O2 must be auto-skipped (participant not retained).
    keep = full_names - {"H2O2"}
    new_id = materialize_subset(db_path, mech_id, species=keep,
                                reactions=all_formulas, new_name="h2o2_no_H2O2")

    new_names = _names(db_path, new_id)
    assert "H2O2" not in new_names
    assert new_names == keep
    with sqlite3.connect(db_path) as conn:
        n_new_rxn = conn.execute(
            "SELECT COUNT(*) FROM reactions WHERE mechanism_id=?", (new_id,)
        ).fetchone()[0]
        fmt = conn.execute("SELECT format FROM mechanisms WHERE id=?",
                           (new_id,)).fetchone()[0]
    assert fmt == "reduced"
    assert n_new_rxn == len(all_formulas) - n_rxn_with_h2o2


def test_subset_metadata_records_provenance(h2o2_full_db):
    import json
    db_path, mech_id = h2o2_full_db
    keep = _names(db_path, mech_id) - {"H2O2"}
    new_id = materialize_subset(db_path, mech_id, species=keep,
                                reactions=_formulas(db_path, mech_id),
                                new_name="prov")
    with sqlite3.connect(db_path) as conn:
        meta = json.loads(conn.execute(
            "SELECT metadata_json FROM mechanisms WHERE id=?", (new_id,)).fetchone()[0])
    assert meta["materialized_from"] == mech_id
    assert meta["n_species_full"] == len(_names(db_path, mech_id))


# ---------------------------------------------------------------------------
# Cantera round-trip on the balanced subset
# ---------------------------------------------------------------------------

def test_materialized_subset_loads_in_cantera(h2o2_full_db, tmp_path):
    from rmip.io.cantera_writer import CanteraYAMLWriter
    db_path, mech_id = h2o2_full_db
    keep = _names(db_path, mech_id) - {"H2O2"}
    new_id = materialize_subset(db_path, mech_id, species=keep,
                                reactions=_formulas(db_path, mech_id),
                                new_name="h2o2_skeletal")

    out = tmp_path / "skeletal.yaml"
    CanteraYAMLWriter(db_path).write(new_id, str(out))
    gas = ct.Solution(str(out))                       # must not raise
    assert gas.n_species == len(keep)
    assert "H2O2" not in gas.species_names
    # it still runs: set a state and read rates
    gas.TPX = 1200.0, ct.one_atm, "H2:2,O2:1,N2:3.76"
    assert gas.n_reactions > 0


# ---------------------------------------------------------------------------
# Lump collapse (structural remap of stoichiometry)
# ---------------------------------------------------------------------------

def test_lump_collapses_member_onto_representative(h2o2_full_db):
    db_path, mech_id = h2o2_full_db
    full_names = _names(db_path, mech_id)
    # synthetic lump (mechanics only — not real isomers): collapse O onto OH.
    lump = {"OH": ["OH", "O"]}
    keep = full_names                                  # offer all; O collapses away
    new_id = materialize_subset(
        db_path, mech_id, species=keep, reactions=_formulas(db_path, mech_id),
        new_name="lumped", lump_map=lump)

    new_names = _names(db_path, new_id)
    assert "O" not in new_names            # member collapsed away
    assert "OH" in new_names               # representative kept
    assert len(new_names) == len(full_names) - 1

    # a reaction that referenced O now references OH instead (no dangling O ids)
    with sqlite3.connect(db_path) as conn:
        o_refs = conn.execute(
            """SELECT COUNT(*) FROM stoichiometry s
               JOIN species sp ON sp.id=s.species_id
               JOIN reactions r ON r.id=s.reaction_id
               WHERE r.mechanism_id=? AND sp.name='O'""", (new_id,)).fetchone()[0]
    assert o_refs == 0
