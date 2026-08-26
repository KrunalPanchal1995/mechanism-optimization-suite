"""
rmip/tests/test_reduction_export.py
===================================
Tests for the Phase-8 follow-on exporters that turn a CSP/lumping diagnostic
into a runnable reduced mechanism (``reduction/csp.py::export_qssa`` and
``reduction/lumping.py::export_lumped``), both built on ``materialize_subset``.

The mechanism is parsed into a temp DB (export needs the DB rows, not just a
Cantera-loadable filepath). h2o2 exercises the pure skeletal QSSA subset; gri30
exercises isomer lumping (CH2OH/CH3O), including the null/duplicate collapse the
lump induces.

Run with:
    python -m pytest src/rmip/tests/test_reduction_export.py -v
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

ct = pytest.importorskip("cantera")


def _parse(db_path, yaml_name):
    from rmip.config import RMIPConfig
    from rmip.core.parser import MechanismParser
    yaml_path = Path(ct.__path__[0]) / "data" / yaml_name
    parser = MechanismParser(RMIPConfig(db_path=db_path))
    mech_id = parser.parse(mech_path=str(yaml_path))
    parser.conn.close()
    return mech_id


# ---------------------------------------------------------------------------
# QSSA / skeletal export (h2o2)
# ---------------------------------------------------------------------------

def test_export_qssa_removes_qss_and_loads(tmp_path):
    from rmip.reduction.csp import CSPAnalyzer, export_qssa
    db_path = str(tmp_path / "h.db")
    mech_id = _parse(db_path, "h2o2.yaml")

    res = CSPAnalyzer(mech_id, db_path).analyze(
        T_K=1200, P_atm=1.0, phi=1.0, fuel="H2:1.0", at="ignition",
        max_qss=2, save_db=False)
    assert res.n_qss == 2
    removed = set(res.qss_species())

    out = tmp_path / "qssa.yaml"
    new_id = export_qssa(db_path, res, str(out), new_name="h2o2_qssa")

    # new mechanism drops exactly the QSS species
    with sqlite3.connect(db_path) as conn:
        new_names = {r[0] for r in conn.execute(
            "SELECT name FROM species WHERE mechanism_id=?", (new_id,))}
        fmt = conn.execute("SELECT format FROM mechanisms WHERE id=?",
                           (new_id,)).fetchone()[0]
    assert fmt == "reduced"
    assert removed.isdisjoint(new_names)

    gas = ct.Solution(str(out))                       # Cantera-valid
    assert gas.n_species == 10 - len(removed)
    # kinetics are intact enough to evaluate rates
    gas.TPX = 1200.0, ct.one_atm, "H2:2,O2:1,N2:3.76"
    assert len(gas.net_production_rates) == gas.n_species


def test_export_qssa_metadata_records_removed(tmp_path):
    import json
    from rmip.reduction.csp import CSPAnalyzer, export_qssa
    db_path = str(tmp_path / "h.db")
    mech_id = _parse(db_path, "h2o2.yaml")
    res = CSPAnalyzer(mech_id, db_path).analyze(
        T_K=1200, P_atm=1.0, phi=1.0, fuel="H2:1.0", at="ignition",
        max_qss=3, save_db=False)
    new_id = export_qssa(db_path, res, None)
    with sqlite3.connect(db_path) as conn:
        meta = json.loads(conn.execute(
            "SELECT metadata_json FROM mechanisms WHERE id=?", (new_id,)).fetchone()[0])
    assert sorted(meta["qssa_removed_species"]) == sorted(res.qss_species())


def test_export_qssa_requires_parsed_mechanism(tmp_path):
    # a filepath-only mechanism row (no species rows) must fail cleanly
    from datetime import datetime, timezone
    from rmip.database.schema import init_db
    from rmip.reduction.csp import export_qssa
    from rmip.types import CSPResult, SimCondition
    db_path = str(tmp_path / "empty.db")
    conn = init_db(db_path)
    conn.execute("INSERT INTO mechanisms (name,format,filepath,loaded_at) VALUES (?,?,?,?)",
                 ("x", "cantera_yaml", "/x.yaml",
                  datetime.now(timezone.utc).isoformat()))
    conn.commit(); conn.close()
    res = CSPResult(mechanism_id=1, condition=SimCondition(1200, 1, 1, "H2:1"),
                    eval_state="ignition", state_T_K=1200, n_species=0,
                    n_reactions=0, jacobian_analytic=True, tau_threshold=1e-6,
                    fastest_tau=None, slowest_tau=None, n_exhausted_modes=0)
    with pytest.raises(ValueError, match="no species rows"):
        export_qssa(db_path, res, None)


# ---------------------------------------------------------------------------
# Lumping export (gri30 — CH2OH/CH3O isomers)
# ---------------------------------------------------------------------------

def test_export_lumped_collapses_isomers_and_loads(tmp_path):
    from rmip.reduction.lumping import ChemicalLumper, export_lumped
    db_path = str(tmp_path / "g.db")
    mech_id = _parse(db_path, "gri30.yaml")

    # loose tolerances so the CH2OH/CH3O isomer pair lumps
    res = ChemicalLumper(db_path).propose(
        mech_id, dHf_tol_kJ=60.0, dS_tol=40.0, save_db=False)
    assert res.n_lumps >= 1
    members = {m for g in res.groups for m in g["members"]}
    assert {"CH2OH", "CH3O"} <= members

    out = tmp_path / "lumped.yaml"
    new_id = export_lumped(db_path, res, str(out), new_name="gri_lumped")

    with sqlite3.connect(db_path) as conn:
        new_names = {r[0] for r in conn.execute(
            "SELECT name FROM species WHERE mechanism_id=?", (new_id,))}
    # the non-representative members are gone; representatives remain
    reps = {g["representative"] for g in res.groups}
    collapsed = members - reps
    assert reps <= new_names
    assert collapsed.isdisjoint(new_names)
    assert len(new_names) == 53 - res.n_species_removed

    gas = ct.Solution(str(out))                        # Cantera-valid despite lump
    assert gas.n_species == len(new_names)
    assert "CH3O" not in gas.species_names             # collapsed member gone
    assert "CH3OH" in gas.species_names                # methanol untouched (not CH3O)
    # no reaction lists a collapsed member as a participant (token-exact, so
    # CH3OH is not a false match for CH3O)
    for i in range(gas.n_reactions):
        parts = set(gas.reaction(i).reactants) | set(gas.reaction(i).products)
        assert "CH3O" not in parts
