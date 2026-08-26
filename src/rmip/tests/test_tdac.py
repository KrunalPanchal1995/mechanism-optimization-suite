"""
rmip/tests/test_tdac.py
=======================
Tests for the tabulated dynamic adaptive chemistry runner
(``rmip/simulation/tdac_runner.py``, FEATURE_ROADMAP #11b).

The pure DRGEP helpers run on a staged h2o2 gas; the full ``TDACRunner`` runs
end-to-end against a temp DB, checking that the on-the-fly reduction reproduces
the full-mechanism IDT while using fewer active species and reusing tabulated
active-sets.

Run with:
    python -m pytest src/rmip/tests/test_tdac.py -v
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

ct = pytest.importorskip("cantera")

from rmip.simulation.tdac_runner import (  # noqa: E402
    TDACRunner,
    drgep_active_species,
    drgep_coefficients,
    reaction_species_lists,
)
from rmip.types import TDACResult  # noqa: E402


def _staged_h2o2_gas():
    gas = ct.Solution("h2o2.yaml")
    gas.set_equivalence_ratio(1.0, fuel="H2:1.0", oxidizer="O2:1.0, N2:3.76")
    gas.TP = 1200.0, ct.one_atm
    r = ct.IdealGasReactor(gas)
    net = ct.ReactorNet([r])
    net.advance(3e-5)                      # into the reacting induction period
    return gas


@pytest.fixture(scope="module")
def h2o2_gas():
    return _staged_h2o2_gas()


@pytest.fixture()
def h2o2_db(tmp_path):
    from rmip.config import RMIPConfig
    from rmip.core.parser import MechanismParser
    db_path = str(tmp_path / "tdac.db")
    parser = MechanismParser(RMIPConfig(db_path=db_path))
    mech_id = parser.parse(mech_path=str(Path(ct.__path__[0]) / "data" / "h2o2.yaml"))
    parser.conn.close()
    return db_path, mech_id


# ---------------------------------------------------------------------------
# Pure DRGEP helpers
# ---------------------------------------------------------------------------

def test_drgep_coefficients_wellformed(h2o2_gas):
    r = drgep_coefficients(h2o2_gas)
    n = h2o2_gas.n_species
    assert r.shape == (n, n)
    assert np.all(r >= 0) and np.all(r <= 1 + 1e-12)
    assert np.allclose(np.diag(r), 0.0)
    assert np.all(np.isfinite(r))


def test_drgep_active_includes_targets(h2o2_gas):
    ti = [h2o2_gas.species_index(s) for s in ("H2", "O2")]
    active = drgep_active_species(h2o2_gas, ti, eps=0.01)
    assert set(ti) <= active
    assert len(active) <= h2o2_gas.n_species


def test_drgep_smaller_eps_is_superset(h2o2_gas):
    ti = [h2o2_gas.species_index(s) for s in ("H2", "O2")]
    loose = drgep_active_species(h2o2_gas, ti, eps=0.2)     # aggressive
    tight = drgep_active_species(h2o2_gas, ti, eps=0.001)   # conservative
    assert loose <= tight                                   # smaller eps ⇒ ⊇
    assert len(tight) >= len(loose)


def test_reaction_species_lists(h2o2_gas):
    lists = reaction_species_lists(h2o2_gas)
    assert len(lists) == h2o2_gas.n_reactions
    assert all(isinstance(s, set) and s for s in lists)


# ---------------------------------------------------------------------------
# End-to-end TDAC
# ---------------------------------------------------------------------------

def test_tdac_matches_full_idt_with_reduction(h2o2_db):
    from rmip.simulation.idt_runner import IDTRunner
    db_path, mech_id = h2o2_db
    full = IDTRunner(mech_id, db_path, t_end_s=1.0).run_single(1200, 1, 1, "H2:1.0").idt_s

    res = TDACRunner(mech_id, db_path, t_end_s=1.0).run_single(
        T_K=1200, P_atm=1.0, phi=1.0, fuel="H2:1.0", eps=0.05,
        update_every=10, save_db=False)

    assert isinstance(res, TDACResult)
    assert res.converged and res.idt_s is not None
    # on-the-fly reduction reproduces the detailed IDT
    assert abs(res.idt_s - full) / full < 0.10
    # ... while genuinely running reduced chemistry
    assert res.mean_active_species < res.n_species_full
    assert res.n_updates > 0
    assert res.n_cache_builds >= 1


def test_tdac_tabulation_reuses_states(h2o2_db):
    db_path, mech_id = h2o2_db
    res = TDACRunner(mech_id, db_path, t_end_s=1.0).run_single(
        T_K=1200, P_atm=1.0, phi=1.0, fuel="H2:1.0", eps=0.05,
        update_every=10, save_db=False)
    # the burnt-gas plateau recurs → tabulation cache scores hits
    assert res.n_cache_hits > 0
    assert 0.0 < res.hit_rate <= 1.0


def test_tdac_eps_controls_active_set(h2o2_db):
    db_path, mech_id = h2o2_db
    runner = TDACRunner(mech_id, db_path, t_end_s=1.0)
    aggressive = runner.run_single(T_K=1200, P_atm=1, phi=1, fuel="H2:1.0",
                                   eps=0.2, save_db=False)
    conservative = runner.run_single(T_K=1200, P_atm=1, phi=1, fuel="H2:1.0",
                                     eps=1e-4, save_db=False)
    # a smaller eps keeps at least as many species active on average
    assert conservative.mean_active_species >= aggressive.mean_active_species - 1e-9


def test_tdac_persists_and_saves(h2o2_db, tmp_path):
    db_path, mech_id = h2o2_db
    res = TDACRunner(mech_id, db_path, t_end_s=1.0).run_single(
        T_K=1200, P_atm=1, phi=1, fuel="H2:1.0", eps=0.05, save_db=True)
    assert res.run_id is not None
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT n_species_full, converged FROM tdac_runs WHERE id=?",
            (res.run_id,)).fetchone()
    assert row[0] == 10 and row[1] == 1

    out = tmp_path / "tdac.json"
    res.save(str(out))
    import json
    d = json.loads(out.read_text())
    assert d["hit_rate"] is not None
    assert d["mean_active_fraction"] is not None
