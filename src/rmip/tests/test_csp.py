"""
rmip/tests/test_csp.py
======================
Tests for the CSP timescale / QSS-species diagnostic
(``rmip/reduction/csp.py``, Phase 8 / FEATURE_ROADMAP #10).

Pure numeric helpers are exercised directly on Cantera's shipped ``h2o2``
mechanism (no DB); the full ``CSPAnalyzer`` runs end-to-end against a temporary
DB whose single mechanism row points at that same YAML.

Run with:
    python -m pytest src/rmip/tests/test_csp.py -v
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

ct = pytest.importorskip("cantera")

from rmip.reduction.csp import (  # noqa: E402
    CSPAnalyzer,
    auto_threshold,
    chemical_jacobian,
    count_exhausted_modes,
    csp_pointers,
    mode_decomposition,
    species_qss_metrics,
    timescales_from_eigs,
)
from rmip.types import CSPResult  # noqa: E402

_RADICALS = {"H", "O", "OH", "HO2", "H2O2"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _staged_h2o2_gas():
    """H2/O2 gas advanced to (near) its ignition point — chemistry fully active."""
    gas = ct.Solution("h2o2.yaml")
    gas.set_equivalence_ratio(1.0, fuel="H2:1.0", oxidizer="O2:1.0, N2:3.76")
    gas.TP = 1200.0, ct.one_atm
    r = ct.IdealGasReactor(gas)
    net = ct.ReactorNet([r])
    t = 0.0
    while t < 1.0 and (r.T - 1200.0) < 400.0:
        t += 1e-6
        net.advance(t)
    return gas


@pytest.fixture(scope="module")
def h2o2_gas():
    return _staged_h2o2_gas()


@pytest.fixture()
def h2o2_db(tmp_path):
    from rmip.database.schema import init_db

    yaml_path = Path(ct.__path__[0]) / "data" / "h2o2.yaml"
    assert yaml_path.exists()
    db_path = str(tmp_path / "csp_test.db")
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

def test_chemical_jacobian_shape_and_analytic(h2o2_gas):
    J, analytic = chemical_jacobian(h2o2_gas)
    n = h2o2_gas.n_species
    assert J.shape == (n, n)
    assert np.all(np.isfinite(J))
    # Cantera 3.x ships the analytic derivative
    assert analytic is True


def test_fd_jacobian_matches_analytic(h2o2_gas):
    from rmip.reduction.csp import _fd_chemical_jacobian
    J_an, _ = chemical_jacobian(h2o2_gas)
    J_fd = _fd_chemical_jacobian(h2o2_gas)
    # The physically meaningful content is the fast end of the spectrum; the FD
    # fallback reproduces the dominant decay rates to a few percent.
    ev_an = np.sort(np.abs(np.linalg.eigvals(J_an).real))[::-1][:4]
    ev_fd = np.sort(np.abs(np.linalg.eigvals(J_fd).real))[::-1][:4]
    assert np.allclose(ev_an, ev_fd, rtol=0.05)


def test_mode_decomposition_ordering(h2o2_gas):
    J, _ = chemical_jacobian(h2o2_gas)
    eigvals, A, B = mode_decomposition(J)
    # fastest (largest |Re|) first
    re = np.abs(eigvals.real)
    assert np.all(np.diff(re) <= 1e-6)
    # B is the (pseudo)inverse of A: B @ A ≈ I
    assert np.allclose((B @ A).real, np.eye(A.shape[0]), atol=1e-6)


def test_timescales_conserved_modes_are_huge():
    eig = np.array([-1e8 + 0j, -1e3, 0.0, -1e-3])
    tau = timescales_from_eigs(eig)
    assert tau[0] == pytest.approx(1e-8, rel=1e-6)
    assert tau[2] >= 1e29          # the zero (conserved) eigenvalue
    assert np.all(np.isfinite(tau))


def test_csp_pointers_rows_normalised(h2o2_gas):
    J, _ = chemical_jacobian(h2o2_gas)
    _, A, B = mode_decomposition(J)
    P = csp_pointers(A, B)
    assert P.shape == (h2o2_gas.n_species, h2o2_gas.n_species)
    assert np.all(P >= 0)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-9)


def test_auto_threshold_ignores_conserved_modes():
    # four fast chemical modes, a clear gap, then two ~infinite conserved modes
    taus = [1e-8, 4e-8, 9e-8, 2e-7, 3e-6, 5e-6, 5e8, 5e8]
    thr = auto_threshold(taus)
    # split must land in the real chemical gap (2e-7 … 3e-6), not up near 5e8
    assert 2e-7 < thr < 3e-6
    assert count_exhausted_modes(taus, thr) == 4


def test_species_qss_metrics_radicals_shortlived(h2o2_gas):
    m = species_qss_metrics(h2o2_gas)
    # radicals are far shorter-lived than the stable bulk species
    tau_radical = np.mean([m[s]["lifetime"] for s in ("H", "O", "OH")])
    assert tau_radical < m["H2O"]["lifetime"]
    assert tau_radical < m["N2"]["lifetime"]
    assert all(m[s]["defect"] >= 0 for s in m)


# ---------------------------------------------------------------------------
# CSPAnalyzer end-to-end
# ---------------------------------------------------------------------------

def test_analyze_identifies_radical_qss(h2o2_db):
    db_path, mech_id = h2o2_db
    an = CSPAnalyzer(mech_id, db_path)
    res = an.analyze(T_K=1200, P_atm=1.0, phi=1.0, fuel="H2:1.0", at="ignition")

    assert isinstance(res, CSPResult)
    assert res.n_species == 10
    assert res.jacobian_analytic is True
    assert res.n_exhausted_modes >= 1
    assert res.fastest_tau < res.slowest_tau
    # some classic radical is flagged QSS, and not everything is
    qss = set(res.qss_species())
    assert qss & _RADICALS, f"expected a radical in QSS set, got {qss}"
    assert 0 < res.n_qss < res.n_species
    # fuel / oxidiser components are never reported as QSS
    assert "H2" not in qss and "O2" not in qss and "N2" not in qss


def test_analyze_excludes_and_caps(h2o2_db):
    db_path, mech_id = h2o2_db
    an = CSPAnalyzer(mech_id, db_path)
    res = an.analyze(T_K=1200, P_atm=1.0, phi=1.0, fuel="H2:1.0",
                     at="ignition", max_qss=2, exclude=["HO2"])
    assert res.n_qss <= 2
    assert "HO2" not in res.qss_species()
    assert "HO2" in res.excluded_species


def test_analyze_explicit_threshold_monotone(h2o2_db):
    db_path, mech_id = h2o2_db
    an = CSPAnalyzer(mech_id, db_path)
    tight = an.analyze(T_K=1200, P_atm=1.0, phi=1.0, fuel="H2:1.0",
                       at="ignition", tau_threshold=1e-8, save_db=False)
    loose = an.analyze(T_K=1200, P_atm=1.0, phi=1.0, fuel="H2:1.0",
                       at="ignition", tau_threshold=1e-4, save_db=False)
    # a looser (larger) threshold can only admit more QSS species and more modes
    assert loose.n_qss >= tight.n_qss
    assert loose.n_exhausted_modes >= tight.n_exhausted_modes


def test_analyze_persists_and_roundtrips(h2o2_db, tmp_path):
    db_path, mech_id = h2o2_db
    an = CSPAnalyzer(mech_id, db_path)
    res = an.analyze(T_K=1200, P_atm=1.0, phi=1.0, fuel="H2:1.0", at="ignition")
    assert res.run_id is not None
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT n_qss, eval_state FROM csp_runs WHERE id=?",
                           (res.run_id,)).fetchone()
    assert row[0] == res.n_qss and row[1] == "ignition"

    out = tmp_path / "csp.json"
    res.save(str(out))
    import json
    d = json.loads(out.read_text())
    assert d["n_qss"] == res.n_qss
    assert d["qss_candidates"][0]["species"] in res.qss_species()


def test_analyze_initial_state_runs(h2o2_db):
    # unburned mixture: the diagnostic must still produce a well-formed result
    db_path, mech_id = h2o2_db
    an = CSPAnalyzer(mech_id, db_path)
    res = an.analyze(T_K=1200, P_atm=1.0, phi=1.0, fuel="H2:1.0",
                     at="initial", save_db=False)
    assert res.eval_state == "initial"
    assert len(res.mode_timescales) == res.n_species
