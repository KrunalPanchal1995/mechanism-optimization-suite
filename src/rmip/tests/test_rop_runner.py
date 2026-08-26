"""
rmip/tests/test_rop_runner.py
=============================
Tests for the native Cantera rate-of-production / reaction-path-flux diagnostic
(``rmip/simulation/rop_runner.py``).

The pure numeric helpers are exercised directly on Cantera's shipped ``h2o2``
mechanism (no DB needed).  The full ``ROPRunner`` is exercised end-to-end
against a temporary DB whose single mechanism row points at that same YAML.

Run with:
    python -m pytest src/rmip/tests/test_rop_runner.py -v
"""
from __future__ import annotations

import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

ct = pytest.importorskip("cantera")

from rmip.simulation.rop_runner import (  # noqa: E402
    ROPRunner,
    build_element_flux,
    build_rop_by_reaction,
    _stoich_matrices,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _staged_h2o2_gas():
    """An H2/O2 gas advanced to its ignition point (chemistry fully active)."""
    gas = ct.Solution("h2o2.yaml")
    gas.set_equivalence_ratio(1.0, fuel="H2:1.0", oxidizer="O2:1.0, N2:3.76")
    gas.TP = 1200.0, 5.0 * ct.one_atm
    r = ct.IdealGasReactor(gas)
    net = ct.ReactorNet([r])
    # advance until a clear temperature rise (ignition) then stop
    t = 0.0
    while t < 1.0 and (r.T - 1200.0) < 400.0:
        t += 1e-5
        net.advance(t)
    return gas


@pytest.fixture(scope="module")
def h2o2_gas():
    return _staged_h2o2_gas()


@pytest.fixture()
def h2o2_db(tmp_path):
    """A minimal RMIP DB with one mechanism row pointing at Cantera's h2o2.yaml."""
    from rmip.database.schema import init_db

    yaml_path = Path(ct.__path__[0]) / "data" / "h2o2.yaml"
    assert yaml_path.exists(), "Cantera h2o2.yaml not found in data dir"

    db_path = str(tmp_path / "rop_test.db")
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


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_stoich_matrices_shape(h2o2_gas):
    rsc, psc = _stoich_matrices(h2o2_gas)
    assert rsc.shape == (h2o2_gas.n_species, h2o2_gas.n_reactions)
    assert psc.shape == (h2o2_gas.n_species, h2o2_gas.n_reactions)


def test_build_rop_by_reaction(h2o2_gas):
    net_rates = h2o2_gas.net_rates_of_progress
    rop = build_rop_by_reaction(h2o2_gas, net_rates)
    # keyed by reaction identity (equation here); duplicates collapse, so there
    # are at most n_reactions keys, and rates are summed per identity.
    n_unique = len({h2o2_gas.reaction(i).equation for i in range(h2o2_gas.n_reactions)})
    assert len(rop) == n_unique
    assert n_unique < h2o2_gas.n_reactions, "h2o2 is expected to contain duplicates"
    assert any(abs(v) > 0 for v in rop.values()), "expected non-zero ROP at ignition"
    assert h2o2_gas.reaction(0).equation in rop
    # total signed ROP is conserved when collapsing duplicates
    assert sum(rop.values()) == pytest.approx(float(net_rates.sum()), rel=1e-9)


def test_build_rop_uses_formula_map(h2o2_gas):
    net_rates = h2o2_gas.net_rates_of_progress
    eq0 = h2o2_gas.reaction(0).equation
    rop = build_rop_by_reaction(h2o2_gas, net_rates, {eq0: "CANON-0"})
    assert "CANON-0" in rop
    assert eq0 not in rop  # remapped to canonical key


def test_element_flux_h_edges_wellformed(h2o2_gas):
    net_rates = h2o2_gas.net_rates_of_progress
    edges = build_element_flux(h2o2_gas, net_rates, "H")
    assert edges, "H flux should be non-empty for an active H2/O2 system"
    for e in edges:
        assert set(e) == {"source", "target", "reaction", "value"}
        assert e["source"] in h2o2_gas.species_names
        assert e["target"] in h2o2_gas.species_names
        assert e["source"] != e["target"]
        assert isinstance(e["value"], float)
        # every edge species must actually carry the traced element
        assert h2o2_gas.n_atoms(e["source"], "H") > 0
        assert h2o2_gas.n_atoms(e["target"], "H") > 0


def test_element_flux_absent_element_is_empty(h2o2_gas):
    # h2o2 mechanism has no carbon -> no C edges, and no crash
    net_rates = h2o2_gas.net_rates_of_progress
    assert build_element_flux(h2o2_gas, net_rates, "C") == []


def test_element_flux_conserves_element(h2o2_gas):
    """Total source-side flux out of a reaction equals its |q| * (element atoms)."""
    net_rates = h2o2_gas.net_rates_of_progress
    edges = build_element_flux(h2o2_gas, net_rates, "O")
    # pick the single most active reaction and check its edges sum to |q|*totalO
    i = int(max(range(h2o2_gas.n_reactions), key=lambda k: abs(net_rates[k])))
    eq = h2o2_gas.reaction(i).equation
    q = net_rates[i]
    rsc, _ = _stoich_matrices(h2o2_gas)
    import numpy as np
    o_atoms = np.array([h2o2_gas.n_atoms(s, "O") for s in h2o2_gas.species_names])
    total_o = float((rsc[:, i] * o_atoms).sum())
    got = sum(e["value"] for e in edges if e["reaction"] == eq)
    if total_o > 0 and q != 0:
        assert got == pytest.approx(abs(q) * total_o, rel=1e-6)


# ---------------------------------------------------------------------------
# End-to-end runner (temp DB -> FluxResult)
# ---------------------------------------------------------------------------

def test_runner_end_to_end(h2o2_db):
    db_path, mech_id = h2o2_db
    runner = ROPRunner(mech_id, db_path)
    res = runner.run_single(
        T_K=1200, P_atm=5.0, phi=1.0,
        fuel="H2:1.0", oxidizer="O2:1.0, N2:3.76",
        element="H", at="ignition",
    )
    assert res.converged, res.error_msg
    assert res.solver == "cantera"
    assert res.element == "H"
    assert len(res.rop_by_reaction) > 0
    assert len(res.element_flux) > 0
    assert res.cpu_time_s is not None and res.cpu_time_s >= 0


def test_runner_at_initial_and_time(h2o2_db):
    db_path, mech_id = h2o2_db
    runner = ROPRunner(mech_id, db_path)
    r_init = runner.run_single(T_K=1200, P_atm=5.0, phi=1.0, fuel="H2:1.0",
                               element="O", at="initial")
    r_time = runner.run_single(T_K=1200, P_atm=5.0, phi=1.0, fuel="H2:1.0",
                               element="O", at="time:1e-4")
    assert r_init.converged and r_time.converged
    # both produce a full reaction set
    assert len(r_init.rop_by_reaction) == len(r_time.rop_by_reaction)


def test_runner_bad_at_returns_error(h2o2_db):
    db_path, mech_id = h2o2_db
    runner = ROPRunner(mech_id, db_path)
    res = runner.run_single(T_K=1200, P_atm=5.0, phi=1.0, fuel="H2:1.0",
                            at="nonsense")
    assert not res.converged
    assert res.error_msg and "nonsense" in res.error_msg


def test_formula_to_key_logs_on_collision(tmp_path, caplog):
    """M6 follow-on: formula_string collisions used to be silent (last write
    wins) -- exactly the failure mode that hid the generic-M falloff
    collider-tag bug (_chemkin_parser.py::_build_formula_string) for
    months. Two reaction rows sharing one formula_string but different
    canonical_key must now log a loud warning naming both keys."""
    import logging

    from rmip.database.schema import init_db

    db_path = str(tmp_path / "collision.db")
    conn = init_db(db_path)
    conn.execute(
        "INSERT INTO mechanisms (name, format, filepath, loaded_at) VALUES (?,?,?,?)",
        ("collision_mech", "chemkin", "unused.inp", "2026-08-04T00:00:00Z"),
    )
    mech_id = conn.execute("SELECT id FROM mechanisms").fetchone()[0]
    for key in ("CANON-A", "CANON-B"):
        conn.execute(
            "INSERT INTO reactions (mechanism_id, formula_string, canonical_key) "
            "VALUES (?,?,?)",
            (mech_id, "H2O2 <=> 2 OH", key),
        )
    conn.commit()
    conn.close()

    runner = ROPRunner(mech_id, db_path)
    with caplog.at_level(logging.WARNING, logger="rmip.simulation.rop_runner"):
        mapping = runner._formula_to_key()

    assert mapping["H2O2 <=> 2 OH"] in ("CANON-A", "CANON-B")
    assert any("formula_string collision" in r.message for r in caplog.records)
    assert any("CANON-A" in r.message and "CANON-B" in r.message
               for r in caplog.records)


def test_formula_to_key_no_warning_when_keys_agree(tmp_path, caplog):
    """Sanity check: two rows sharing formula_string but the SAME
    canonical_key (a genuine duplicate reaction, correctly flagged) must
    NOT trigger the collision warning."""
    import logging

    from rmip.database.schema import init_db

    db_path = str(tmp_path / "dup.db")
    conn = init_db(db_path)
    conn.execute(
        "INSERT INTO mechanisms (name, format, filepath, loaded_at) VALUES (?,?,?,?)",
        ("dup_mech", "chemkin", "unused.inp", "2026-08-04T00:00:00Z"),
    )
    mech_id = conn.execute("SELECT id FROM mechanisms").fetchone()[0]
    for _ in range(2):
        conn.execute(
            "INSERT INTO reactions (mechanism_id, formula_string, canonical_key) "
            "VALUES (?,?,?)",
            (mech_id, "H + OH <=> H2O", "SAME-KEY"),
        )
    conn.commit()
    conn.close()

    runner = ROPRunner(mech_id, db_path)
    with caplog.at_level(logging.WARNING, logger="rmip.simulation.rop_runner"):
        runner._formula_to_key()

    assert not any("formula_string collision" in r.message for r in caplog.records)


def test_runner_persists_via_insert_flux_run(h2o2_db):
    """The CLI persistence path: a FluxResult round-trips into flux_runs."""
    from rmip.database.schema import init_db
    from rmip.database.schema_phase6 import insert_flux_run, upgrade_db_phase6

    db_path, mech_id = h2o2_db
    conn = init_db(db_path)
    upgrade_db_phase6(conn)

    runner = ROPRunner(mech_id, db_path)
    res = runner.run_single(T_K=1200, P_atm=5.0, phi=1.0, fuel="H2:1.0",
                            element="H", at="ignition")
    run_id = insert_flux_run(
        conn, mechanism_id=res.mechanism_id, solver=res.solver,
        T_K=res.condition.T_K, P_atm=res.condition.P_atm, phi=res.condition.phi,
        fuel_composition=res.condition.fuel, element=res.element,
        rop_by_reaction=res.rop_by_reaction, element_flux=res.element_flux,
        converged=res.converged, cpu_time_s=res.cpu_time_s,
    )
    row = conn.execute(
        "SELECT solver, n_reactions, n_flux_edges, converged "
        "FROM flux_runs WHERE id=?", (run_id,)
    ).fetchone()
    conn.close()
    assert row[0] == "cantera"
    assert row[1] == len(res.rop_by_reaction)
    assert row[2] == len(res.element_flux)
    assert row[3] == 1
