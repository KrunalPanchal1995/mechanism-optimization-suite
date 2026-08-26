"""
rmip/tests/test_lumping.py
==========================
Tests for a-priori thermodynamic chemical/isomer lumping
(``rmip/reduction/lumping.py``, Phase 8 / FEATURE_ROADMAP #10).

The grouping logic (union-find single-linkage) is exercised directly on
synthetic thermochemistry; the Cantera-facing extraction and the full
``ChemicalLumper`` run against Cantera's shipped ``h2o2`` mechanism, which has
no isomers (so a correct run proposes zero lumps).

Run with:
    python -m pytest src/rmip/tests/test_lumping.py -v
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

ct = pytest.importorskip("cantera")

from rmip.reduction.lumping import (  # noqa: E402
    ChemicalLumper,
    _formula_str,
    composition_key,
    single_linkage_lumps,
    species_thermo_298,
    total_atoms,
)
from rmip.types import LumpingResult  # noqa: E402


# ---------------------------------------------------------------------------
# Pure grouping helpers
# ---------------------------------------------------------------------------

def test_composition_key_isomers_match():
    a = composition_key({"C": 7, "H": 15})
    b = composition_key({"H": 15.0, "C": 7.0})     # order / float-count invariant
    c = composition_key({"C": 7, "H": 16})
    assert a == b
    assert a != c
    assert total_atoms({"C": 7, "H": 15}) == 22


def test_formula_str():
    assert _formula_str((("C", 7), ("H", 15))) == "C7H15"
    assert _formula_str((("H", 2), ("O", 1))) == "H2O"


def test_single_linkage_merges_close_isomers():
    # three C4H9 isomers: A,B thermochemically close; C far away → two lumps
    members = ["A", "B", "C"]
    hf = [10.0, 12.0, 60.0]        # kJ/mol
    s = [300.0, 305.0, 340.0]      # J/mol/K
    lumps = single_linkage_lumps(members, hf, s, dHf_tol=5.0, dS_tol=20.0)
    lumps_as_sets = [set(g) for g in lumps]
    assert {"A", "B"} in lumps_as_sets
    assert {"C"} in lumps_as_sets


def test_single_linkage_chains_transitively():
    # single-linkage: A-B linked, B-D linked (A-D not directly) → one lump A,B,D
    members = ["A", "B", "D"]
    hf = [10.0, 14.0, 18.0]
    s = [300.0, 304.0, 308.0]
    lumps = single_linkage_lumps(members, hf, s, dHf_tol=5.0, dS_tol=6.0)
    assert len(lumps) == 1
    assert set(lumps[0]) == {"A", "B", "D"}


def test_single_linkage_orders_by_stability():
    members = ["hi", "lo", "mid"]
    hf = [30.0, 10.0, 20.0]
    s = [300.0, 300.0, 300.0]
    lumps = single_linkage_lumps(members, hf, s, dHf_tol=100.0, dS_tol=100.0)
    assert len(lumps) == 1
    assert lumps[0][0] == "lo"     # most stable (lowest Hf) first


def test_tolerance_zero_splits_everything():
    members = ["A", "B", "C"]
    hf = [10.0, 11.0, 12.0]
    s = [300.0, 301.0, 302.0]
    lumps = single_linkage_lumps(members, hf, s, dHf_tol=0.0, dS_tol=0.0)
    assert len(lumps) == 3


# ---------------------------------------------------------------------------
# Cantera-facing extraction
# ---------------------------------------------------------------------------

def test_species_thermo_298_wellformed():
    gas = ct.Solution("h2o2.yaml")
    hf, s = species_thermo_298(gas)
    assert hf.shape == (gas.n_species,)
    assert s.shape == (gas.n_species,)
    assert np.all(np.isfinite(hf)) and np.all(np.isfinite(s))
    # extraction must not disturb the caller's gas state
    assert gas.n_species == 10


# ---------------------------------------------------------------------------
# ChemicalLumper end-to-end (h2o2 has no isomers)
# ---------------------------------------------------------------------------

@pytest.fixture()
def h2o2_db(tmp_path):
    from rmip.database.schema import init_db

    yaml_path = Path(ct.__path__[0]) / "data" / "h2o2.yaml"
    db_path = str(tmp_path / "lump_test.db")
    conn = init_db(db_path)
    conn.execute(
        "INSERT INTO mechanisms (name, format, filepath, loaded_at) VALUES (?,?,?,?)",
        ("h2o2", "cantera_yaml", str(yaml_path),
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    mech_id = conn.execute("SELECT id FROM mechanisms").fetchone()[0]
    conn.close()
    return db_path, mech_id


def test_lumper_no_isomers_zero_lumps(h2o2_db):
    db_path, mech_id = h2o2_db
    res = ChemicalLumper(db_path).propose(mech_id)
    assert isinstance(res, LumpingResult)
    assert res.n_species_full == 10
    assert res.n_lumps == 0
    assert res.n_species_removed == 0
    assert res.n_species_lumped == res.n_species_full
    assert any("no isomer" in n.lower() for n in res.notes)


def test_lumper_persists(h2o2_db):
    db_path, mech_id = h2o2_db
    res = ChemicalLumper(db_path).propose(mech_id)
    assert res.run_id is not None
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT n_lumps, method FROM lumping_runs WHERE id=?",
                           (res.run_id,)).fetchone()
    assert row[0] == 0 and row[1] == "thermo_lumping"


def test_lumper_save_roundtrip(h2o2_db, tmp_path):
    db_path, mech_id = h2o2_db
    res = ChemicalLumper(db_path).propose(mech_id, save_db=False)
    out = tmp_path / "lump.json"
    res.save(str(out))
    import json
    d = json.loads(out.read_text())
    assert d["n_species_full"] == 10
    assert d["groups"] == []
