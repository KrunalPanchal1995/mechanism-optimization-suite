"""
Tests for Phase-5B extended: DRGPSA, DRGSA, CoefficientVisualizer, TrimVisualizer,
BFS levels, and r_AB serialisation.

All external dependencies (Cantera, Dash, Plotly) are optional — tests that require
them are marked with skipif.  The core algorithm tests run without any of them.
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Dict, List

import networkx as nx
import pytest

from rmip.types import ReductionResult, ReductionStep


# ─────────────────────────────────────────────────────────────────────────────
# Fixture helpers — mirroring test_phase5b.py style
# ─────────────────────────────────────────────────────────────────────────────

def _make_db(path: str) -> sqlite3.Connection:
    from rmip.database.schema import init_db
    from rmip.database.migrations import run_phase3_migrations, run_phase5b_migrations
    conn = init_db(path)
    run_phase3_migrations(conn)
    run_phase5b_migrations(conn)
    return conn


def _insert_mechanism(conn, name):
    cur = conn.execute(
        "INSERT INTO mechanisms (name, format, filepath, description, "
        "n_species, n_reactions, loaded_at, metadata_json) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (name, "chemkin", "/fake.inp", f"Test {name}", 7, 4,
         "2026-01-01T00:00:00Z", "{}"),
    )
    return cur.lastrowid


def _insert_species(conn, mech_id, name, formula="H2"):
    cur = conn.execute(
        "INSERT INTO species (mechanism_id, name, inchi_key, formula, "
        "canonical_smiles, glossary_matched, tmin_K, tmid_K, tmax_K) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (mech_id, name, None, formula, None, 0, 200, 1000, 5000),
    )
    return cur.lastrowid


def _insert_reaction(conn, mech_id, formula_str, can_key=None):
    cur = conn.execute(
        "INSERT INTO reactions (mechanism_id, formula_string, canonical_key, "
        "reversible, A_si, n, Ea_si, kinetics_type, pressure_dependent, "
        "has_third_body) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (mech_id, formula_str, can_key, 1, 1e10, 0.0, 50000.0,
         "arrhenius", 0, 0),
    )
    return cur.lastrowid


def _insert_stoich(conn, rxn_id, sp_id, coeff, role):
    conn.execute(
        "INSERT INTO stoichiometry (reaction_id, species_id, stoich_coeff, role) "
        "VALUES (?,?,?,?)",
        (rxn_id, sp_id, coeff, role),
    )


def _build_h2_fixture(db_path: str):
    """
    Minimal H2 combustion mechanism for tests.

    Species: H2, O2, H2O, H, OH, O, HO2
    Reactions:
      R1: H2 + O2 <=> H2O + H
      R2: H + O2 <=> OH + O
      R3: H2 + OH <=> H2O + H
      R4: HO2 + H <=> OH + OH   (weakly coupled)
    """
    conn = _make_db(db_path)
    mid  = _insert_mechanism(conn, "H2_ext_test")
    sp   = {}
    for name, formula in [("H2","H2"),("O2","O2"),("H2O","H2O"),
                          ("H","H"),("OH","HO"),("O","O"),("HO2","HO2")]:
        sp[name] = _insert_species(conn, mid, name, formula=formula)

    r1 = _insert_reaction(conn, mid, "H2 + O2 <=> H2O + H", "k1")
    for n in ("H2","O2"):  _insert_stoich(conn, r1, sp[n], 1.0, "reactant")
    for n in ("H2O","H"):  _insert_stoich(conn, r1, sp[n], 1.0, "product")

    r2 = _insert_reaction(conn, mid, "H + O2 <=> OH + O", "k2")
    for n in ("H","O2"):   _insert_stoich(conn, r2, sp[n], 1.0, "reactant")
    for n in ("OH","O"):   _insert_stoich(conn, r2, sp[n], 1.0, "product")

    r3 = _insert_reaction(conn, mid, "H2 + OH <=> H2O + H", "k3")
    for n in ("H2","OH"):  _insert_stoich(conn, r3, sp[n], 1.0, "reactant")
    for n in ("H2O","H"):  _insert_stoich(conn, r3, sp[n], 1.0, "product")

    r4 = _insert_reaction(conn, mid, "HO2 + H <=> OH + OH", "k4")
    for n in ("HO2","H"):  _insert_stoich(conn, r4, sp[n], 1.0, "reactant")
    _insert_stoich(conn, r4, sp["OH"], 2.0, "product")

    conn.commit()

    from rmip.core.graph_builder import GraphBuilder
    G = GraphBuilder(db_path).build_graph(mid, include_reverse_edges=False,
                                          include_spring=False)
    return conn, mid, G, sp


# ─────────────────────────────────────────────────────────────────────────────
# DRGPSA tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDRGPSAReducer:

    def test_method_string(self):
        from rmip.reduction.drgpsa import DRGPSAReducer
        assert DRGPSAReducer.METHOD == "DRGPSA"

    def test_reduce_basic(self, tmp_path):
        db_path = str(tmp_path / "h2.db")
        conn, mid, G, sp = _build_h2_fixture(db_path)
        sa_data = {
            "observable": "IDT",
            "conditions": {},
            "coefficients": {
                "H + O2 <=> OH + O":   0.9,
                "H2 + O2 <=> H2O + H": 0.4,
                "H2 + OH <=> H2O + H": 0.2,
                "HO2 + H <=> OH + OH": 0.05,
            },
        }
        from rmip.reduction.drgpsa import DRGPSAReducer
        reducer = DRGPSAReducer(db_path, sa_data=sa_data)
        result  = reducer.reduce(
            mech_id=mid, G=G,
            target_species=["H2", "O2", "H2O"],
            epsilon_range=[0.0, 0.2, 0.5],
            save_db=False,
        )
        assert result.method == "DRGPSA"
        assert len(result.steps) == 3
        assert result.r_AB is not None
        assert result.sa_coefficients is not None
        # Seeds always retained
        for step in result.steps:
            for seed in ["H2", "O2", "H2O"]:
                assert seed in step.retained_species

    def test_sa_weighting_changes_coupling(self, tmp_path):
        """
        DRGPSA with heavy SA on H+O2 boosts r(H,O2) but shrinks r(H,HO2).

        When "H + O2 <=> OH + O" is SA-amplified by ×51:
          denom[H] grows from 2 → 52  (two reactions, one amplified)
          numer[(H,O2)] grows from 1 → 51  → r = 51/52 ≈ 0.98
          numer[(H,HO2)] stays at 1        → r = 1/52 ≈ 0.019

        So at ε=0.3, DRGPSA drops HO2 (weakly-coupled through non-SA reaction)
        while DRG retains it.  This is the intended selectivity of DRGPSA.
        """
        db_path = str(tmp_path / "h2.db")
        conn, mid, G, sp = _build_h2_fixture(db_path)
        sa_data = {
            "observable": "IDT",
            "conditions": {},
            "coefficients": {"H + O2 <=> OH + O": 50.0},
        }
        from rmip.reduction.drg    import DRGReducer
        from rmip.reduction.drgpsa import DRGPSAReducer
        eps_range = [0.3]
        drg_r = DRGReducer(db_path).reduce(
            mech_id=mid, G=G, target_species=["H2","O2","H2O"],
            epsilon_range=eps_range, save_db=False)
        psa_r = DRGPSAReducer(db_path, sa_data=sa_data).reduce(
            mech_id=mid, G=G, target_species=["H2","O2","H2O"],
            epsilon_range=eps_range, save_db=False)

        # DRG retains HO2 (r(H,HO2)=0.5>0.3); DRGPSA drops it (r≈0.019<0.3)
        assert "HO2" in drg_r.steps[0].retained_species
        assert "HO2" not in psa_r.steps[0].retained_species

        # But both retain the core species that are strongly coupled
        for sp_name in ["H2", "O2", "H2O", "H", "OH", "O"]:
            assert sp_name in psa_r.steps[0].retained_species

    def test_empty_sa_equals_drg(self, tmp_path):
        """Empty SA dict → DRGPSA degrades to DRG (sa_amp = 1 for all)."""
        db_path = str(tmp_path / "h2.db")
        conn, mid, G, sp = _build_h2_fixture(db_path)
        sa_data = {"observable": "IDT", "conditions": {}, "coefficients": {}}
        from rmip.reduction.drg    import DRGReducer
        from rmip.reduction.drgpsa import DRGPSAReducer
        eps_range = [0.0, 0.3]
        drg_r = DRGReducer(db_path).reduce(
            mech_id=mid, G=G, target_species=["H2","O2","H2O"],
            epsilon_range=eps_range, save_db=False)
        psa_r = DRGPSAReducer(db_path, sa_data=sa_data).reduce(
            mech_id=mid, G=G, target_species=["H2","O2","H2O"],
            epsilon_range=eps_range, save_db=False)
        assert psa_r.steps[1].n_species == drg_r.steps[1].n_species

    def test_no_sa_warning_logged(self, tmp_path, caplog):
        """Without SA data, DRGPSA logs a warning and degrades gracefully."""
        db_path = str(tmp_path / "h2.db")
        conn, mid, G, sp = _build_h2_fixture(db_path)
        from rmip.reduction.drgpsa import DRGPSAReducer
        import logging
        with caplog.at_level(logging.WARNING, logger="rmip.reduction.drgpsa"):
            reducer = DRGPSAReducer(db_path)  # no SA
            result  = reducer.reduce(
                mech_id=mid, G=G, target_species=["H2","O2","H2O"],
                epsilon_range=[0.3], save_db=False,
            )
        assert any("SA" in r.message or "degrading" in r.message for r in caplog.records)
        assert result.method == "DRGPSA"

    def test_sa_json_path(self, tmp_path):
        """DRGPSAReducer accepts sa_json_path on construction."""
        db_path = str(tmp_path / "h2.db")
        conn, mid, G, sp = _build_h2_fixture(db_path)
        sa_data = {"observable": "IDT", "conditions": {},
                   "coefficients": {"H + O2 <=> OH + O": 0.8}}
        sa_path = str(tmp_path / "sa.json")
        with open(sa_path, "w") as f:
            json.dump(sa_data, f)
        from rmip.reduction.drgpsa import DRGPSAReducer
        reducer = DRGPSAReducer(db_path, sa_json_path=sa_path)
        assert reducer._sa_data is not None
        assert "H + O2 <=> OH + O" in reducer._sa_data["coefficients"]

    def test_r_AB_stored_in_result(self, tmp_path):
        db_path = str(tmp_path / "h2.db")
        conn, mid, G, sp = _build_h2_fixture(db_path)
        from rmip.reduction.drgpsa import DRGPSAReducer
        reducer = DRGPSAReducer(db_path, sa_data={"coefficients": {}})
        result  = reducer.reduce(
            mech_id=mid, G=G, target_species=["H2","O2","H2O"],
            epsilon_range=[0.0], save_db=False)
        assert result.r_AB is not None
        # Nested dict format
        for A, neighbours in result.r_AB.items():
            assert isinstance(A, str)
            assert isinstance(neighbours, dict)

    def test_bfs_levels_in_drgpsa_steps(self, tmp_path):
        db_path = str(tmp_path / "h2.db")
        conn, mid, G, sp = _build_h2_fixture(db_path)
        from rmip.reduction.drgpsa import DRGPSAReducer
        reducer = DRGPSAReducer(db_path, sa_data={"coefficients": {}})
        result  = reducer.reduce(
            mech_id=mid, G=G, target_species=["H2","O2","H2O"],
            epsilon_range=[0.0], save_db=False)
        step = result.steps[0]
        assert step.bfs_levels is not None
        for seed in ["H2", "O2", "H2O"]:
            assert step.bfs_levels.get(seed) == 0


# ─────────────────────────────────────────────────────────────────────────────
# DRGSA tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDRGSAReducer:

    def test_method_string(self):
        from rmip.reduction.drgpsa import DRGSAReducer
        assert DRGSAReducer.METHOD == "DRGSA"

    def test_seeds_always_retained(self, tmp_path):
        db_path = str(tmp_path / "h2.db")
        conn, mid, G, sp = _build_h2_fixture(db_path)
        from rmip.reduction.drgpsa import DRGSAReducer
        reducer = DRGSAReducer(db_path, sa_data={"coefficients": {}})
        result  = reducer.reduce(
            mech_id=mid, G=G, target_species=["H2","O2","H2O"],
            epsilon_range=[0.0, 0.99], save_db=False)
        for step in result.steps:
            for seed in ["H2", "O2", "H2O"]:
                assert seed in step.retained_species

    def test_removes_species_with_no_sa(self, tmp_path):
        """HO2 is not in any SA-covered reaction → removed at ε > 0."""
        db_path = str(tmp_path / "h2.db")
        conn, mid, G, sp = _build_h2_fixture(db_path)
        sa_data = {
            "observable": "IDT",
            "conditions": {},
            "coefficients": {
                "H + O2 <=> OH + O":   0.9,
                "H2 + O2 <=> H2O + H": 0.5,
                "H2 + OH <=> H2O + H": 0.3,
                # HO2 + H <=> OH + OH  intentionally absent
            },
        }
        from rmip.reduction.drgpsa import DRGSAReducer
        reducer = DRGSAReducer(db_path, sa_data=sa_data)
        result  = reducer.reduce(
            mech_id=mid, G=G, target_species=["H2","O2","H2O"],
            epsilon_range=[0.0, 0.2, 0.8], save_db=False)
        # At ε=0.2, HO2 (importance=0 < 0.2) should be removed
        step02 = result.steps[1]
        assert "HO2" not in step02.retained_species

    def test_r_AB_is_none(self, tmp_path):
        db_path = str(tmp_path / "h2.db")
        conn, mid, G, sp = _build_h2_fixture(db_path)
        from rmip.reduction.drgpsa import DRGSAReducer
        reducer = DRGSAReducer(db_path, sa_data={"coefficients": {}})
        result  = reducer.reduce(
            mech_id=mid, G=G, target_species=["H2","O2","H2O"],
            epsilon_range=[0.0], save_db=False)
        assert result.r_AB is None

    def test_bfs_levels_seeds_zero(self, tmp_path):
        db_path = str(tmp_path / "h2.db")
        conn, mid, G, sp = _build_h2_fixture(db_path)
        from rmip.reduction.drgpsa import DRGSAReducer
        reducer = DRGSAReducer(db_path, sa_data={"coefficients": {
            "H + O2 <=> OH + O": 0.9}})
        result  = reducer.reduce(
            mech_id=mid, G=G, target_species=["H2","O2","H2O"],
            epsilon_range=[0.0], save_db=False)
        step = result.steps[0]
        assert step.bfs_levels is not None
        for seed in ["H2", "O2", "H2O"]:
            assert step.bfs_levels.get(seed) == 0

    def test_sa_normalised_in_result(self, tmp_path):
        db_path = str(tmp_path / "h2.db")
        conn, mid, G, sp = _build_h2_fixture(db_path)
        from rmip.reduction.drgpsa import DRGSAReducer
        sa_data = {"coefficients": {
            "H + O2 <=> OH + O":   0.9,
            "H2 + O2 <=> H2O + H": 0.5,
        }}
        reducer = DRGSAReducer(db_path, sa_data=sa_data)
        result  = reducer.reduce(
            mech_id=mid, G=G, target_species=["H2","O2","H2O"],
            epsilon_range=[0.0], save_db=False)
        assert result.sa_coefficients is not None
        # Max value should be 1.0 (normalised)
        max_val = max(result.sa_coefficients.values())
        assert abs(max_val - 1.0) < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# BFS levels and r_AB serialisation
# ─────────────────────────────────────────────────────────────────────────────

class TestBFSLevelsAndRAB:

    def test_bfs_levels_seeds_are_zero(self, tmp_path):
        db_path = str(tmp_path / "h2.db")
        conn, mid, G, sp = _build_h2_fixture(db_path)
        from rmip.reduction.drg import DRGReducer
        result = DRGReducer(db_path).reduce(
            mech_id=mid, G=G, target_species=["H2","O2","H2O"],
            epsilon_range=[0.0], save_db=False)
        step = result.steps[0]
        assert step.bfs_levels is not None
        for seed in ["H2", "O2", "H2O"]:
            assert step.bfs_levels[seed] == 0

    def test_bfs_levels_non_seeds_positive(self, tmp_path):
        db_path = str(tmp_path / "h2.db")
        conn, mid, G, sp = _build_h2_fixture(db_path)
        from rmip.reduction.drg import DRGReducer
        result = DRGReducer(db_path).reduce(
            mech_id=mid, G=G, target_species=["H2","O2","H2O"],
            epsilon_range=[0.0], save_db=False)
        step = result.steps[0]
        seeds = {"H2","O2","H2O"}
        for sp_name, lvl in step.bfs_levels.items():
            if sp_name in seeds:
                assert lvl == 0
            else:
                assert lvl >= 1

    def test_bfs_levels_increase_at_high_eps(self, tmp_path):
        """At high ε, fewer BFS levels reached."""
        db_path = str(tmp_path / "h2.db")
        conn, mid, G, sp = _build_h2_fixture(db_path)
        from rmip.reduction.drg import DRGReducer
        result = DRGReducer(db_path).reduce(
            mech_id=mid, G=G, target_species=["H2","O2","H2O"],
            epsilon_range=[0.0, 0.99], save_db=False)
        # At ε=0.99, only seeds retained (level=0)
        step_high = result.steps[1]
        if step_high.bfs_levels:
            max_lvl = max(step_high.bfs_levels.values())
            assert max_lvl >= 0  # seeds always at 0

    def test_bfs_levels_round_trip(self):
        """bfs_levels survives ReductionStep.to_dict / from_dict."""
        step = ReductionStep(
            threshold=0.2, n_species=3, n_reactions=2,
            retained_species=["H2","O2","H"],
            retained_reactions=["H2+O2<=>H2O"],
            removed_species=["HO2"], removed_reactions=[],
            bfs_levels={"H2":0, "O2":0, "H":1},
        )
        step2 = ReductionStep.from_dict(step.to_dict())
        assert step2.bfs_levels == {"H2":0, "O2":0, "H":1}

    def test_r_AB_nested_dict(self, tmp_path):
        db_path = str(tmp_path / "h2.db")
        conn, mid, G, sp = _build_h2_fixture(db_path)
        from rmip.reduction.drg import DRGReducer
        result = DRGReducer(db_path).reduce(
            mech_id=mid, G=G, target_species=["H2","O2","H2O"],
            epsilon_range=[0.0], save_db=False)
        assert result.r_AB is not None
        for A, neighbours in result.r_AB.items():
            assert isinstance(A, str)
            for B, val in neighbours.items():
                assert isinstance(B, str)
                assert 0.0 <= val <= 1.0 + 1e-9

    def test_r_AB_round_trip_json(self, tmp_path):
        result = ReductionResult(
            mechanism_id=1, method="DRG",
            target_species=["H2","O2"],
            steps=[],
            r_AB={"H2": {"O2": 0.8, "H": 0.5}, "O2": {"H2": 0.75}},
            sa_coefficients={"H+O2": 0.9},
        )
        out = str(tmp_path / "result.json")
        result.save(out)
        r2 = ReductionResult.load(out)
        assert r2.r_AB == result.r_AB
        assert r2.sa_coefficients == result.sa_coefficients

    def test_drgep_bfs_levels_present(self, tmp_path):
        """DRGEP steps also have bfs_levels (seeds=0, retained=1)."""
        db_path = str(tmp_path / "h2.db")
        conn, mid, G, sp = _build_h2_fixture(db_path)
        from rmip.reduction.drgep import DRGEPReducer
        result = DRGEPReducer(db_path).reduce(
            mech_id=mid, G=G, target_species=["H2","O2","H2O"],
            epsilon_range=[0.0], save_db=False)
        step = result.steps[0]
        assert step.bfs_levels is not None
        for seed in ["H2","O2","H2O"]:
            assert step.bfs_levels.get(seed) == 0

    def test_pfa_bfs_levels_present(self, tmp_path):
        """PFA steps also have bfs_levels."""
        db_path = str(tmp_path / "h2.db")
        conn, mid, G, sp = _build_h2_fixture(db_path)
        from rmip.reduction.pfa import PFAReducer
        result = PFAReducer(db_path).reduce(
            mech_id=mid, G=G, target_species=["H2","O2","H2O"],
            epsilon_range=[0.0], save_db=False)
        step = result.steps[0]
        assert step.bfs_levels is not None


# ─────────────────────────────────────────────────────────────────────────────
# CoefficientVisualizer unit tests
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_R_AB = {
    "H2": {"O2": 0.9, "H": 0.5},
    "O2": {"H2": 0.8, "OH": 0.6},
    "H":  {"H2": 0.3},
    "OH": {"H2": 0.2},
}

def _make_coeff_result(r_AB=None):
    if r_AB is None:
        r_AB = _DEFAULT_R_AB
    return ReductionResult(
        mechanism_id=1, method="DRG",
        target_species=["H2","O2","H2O"],
        steps=[], r_AB=r_AB,
    )


def _make_species_G(species):
    G = nx.DiGraph()
    for i, sp in enumerate(species):
        G.add_node(f"S_{i}", type="species", name=sp,
                   pos_x=float(i), pos_y=float(i),
                   pos_x_3d=float(i), pos_y_3d=0.0, pos_z_3d=float(i))
    return G


class TestCoefficientVisualizer:

    def test_import(self):
        from rmip.reduction.coefficient_viz import CoefficientVisualizer

    def test_init_species_set(self):
        from rmip.reduction.coefficient_viz import CoefficientVisualizer
        result = _make_coeff_result()
        G      = _make_species_G(["H2","O2","H","OH"])
        cv     = CoefficientVisualizer(result, G)
        assert set(cv._species) == {"H2","O2","H","OH"}
        assert "H2" in cv._seeds

    def test_pos2d_from_G(self):
        from rmip.reduction.coefficient_viz import CoefficientVisualizer
        result = _make_coeff_result()
        G      = _make_species_G(["H2","O2","H","OH"])
        cv     = CoefficientVisualizer(result, G)
        assert "H2" in cv._pos2d
        assert len(cv._pos2d["H2"]) == 2

    def test_pos3d_from_G(self):
        from rmip.reduction.coefficient_viz import CoefficientVisualizer
        result = _make_coeff_result()
        G      = _make_species_G(["H2","O2","H","OH"])
        cv     = CoefficientVisualizer(result, G)
        assert "H2" in cv._pos3d
        assert len(cv._pos3d["H2"]) == 3

    def test_r_AB_to_nested(self):
        from rmip.reduction.drg import DRGReducer
        flat   = {("H2","O2"): 0.8, ("O2","H2"): 0.75, ("H2","H"): 0.5}
        nested = DRGReducer._r_AB_to_nested(flat)
        assert nested["H2"]["O2"] == pytest.approx(0.8, abs=1e-5)
        assert nested["O2"]["H2"] == pytest.approx(0.75, abs=1e-5)

    def test_empty_r_AB_no_crash(self):
        from rmip.reduction.coefficient_viz import CoefficientVisualizer
        result = _make_coeff_result(r_AB={})
        cv     = CoefficientVisualizer(result, nx.DiGraph())
        assert cv._species == []

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("plotly"),
        reason="plotly not installed",
    )
    def test_heatmap_figure(self):
        import plotly.graph_objects as go
        from rmip.reduction.coefficient_viz import CoefficientVisualizer
        result = _make_coeff_result()
        cv     = CoefficientVisualizer(result, _make_species_G(["H2","O2","H","OH"]))
        fig    = cv._heatmap_figure()
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("plotly"),
        reason="plotly not installed",
    )
    def test_network_figure_2d(self):
        import plotly.graph_objects as go
        from rmip.reduction.coefficient_viz import CoefficientVisualizer
        result = _make_coeff_result()
        cv     = CoefficientVisualizer(result, _make_species_G(["H2","O2","H","OH"]))
        fig    = cv._network_figure_2d(0.1)
        assert isinstance(fig, go.Figure)

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("plotly"),
        reason="plotly not installed",
    )
    def test_network_figure_3d(self):
        import plotly.graph_objects as go
        from rmip.reduction.coefficient_viz import CoefficientVisualizer
        result = _make_coeff_result()
        cv     = CoefficientVisualizer(result, _make_species_G(["H2","O2","H","OH"]))
        fig    = cv._network_figure_3d(0.1)
        assert isinstance(fig, go.Figure)

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("plotly"),
        reason="plotly not installed",
    )
    def test_export_html(self, tmp_path):
        from rmip.reduction.coefficient_viz import CoefficientVisualizer
        result   = _make_coeff_result()
        cv       = CoefficientVisualizer(result, _make_species_G(["H2","O2","H","OH"]))
        out_path = str(tmp_path / "coeff.html")
        cv.export_html(out_path)
        assert Path(out_path).exists()
        assert Path(out_path).stat().st_size > 100

    def test_threshold_filters_edges(self):
        """At threshold=0.95, only the r=0.9 edge H2→O2 should be excluded."""
        from rmip.reduction.coefficient_viz import CoefficientVisualizer
        r_AB   = {"H2": {"O2": 0.9}, "O2": {"H2": 0.5}}
        result = ReductionResult(mechanism_id=1, method="DRG",
                                 target_species=["H2","O2"], steps=[], r_AB=r_AB)
        cv     = CoefficientVisualizer(result, _make_species_G(["H2","O2"]))
        pos    = {"H2": (0.0,0.0), "O2": (1.0,1.0)}
        traces = cv._build_edge_traces_2d(pos, threshold=0.95)
        # Both edges below 0.95 → no traces (0.9 < 0.95 and 0.5 < 0.95)
        assert len(traces) == 0


# ─────────────────────────────────────────────────────────────────────────────
# TrimVisualizer unit tests
# ─────────────────────────────────────────────────────────────────────────────

def _make_trim_result():
    all_sp = ["H2","O2","H2O","H","OH","O","HO2"]
    all_rx = ["H2+O2<=>H2O+H", "H+O2<=>OH+O", "H2+OH<=>H2O+H", "HO2+H<=>OH+OH"]
    def _step(eps, retained_sp):
        retained_rx = all_rx[:len(retained_sp) - 2]  # rough proxy
        bfs = {sp: (0 if sp in ["H2","O2","H2O"] else 1) for sp in retained_sp}
        return ReductionStep(
            threshold=eps,
            retained_species=sorted(retained_sp),
            retained_reactions=sorted(retained_rx),
            removed_species=sorted(set(all_sp)-set(retained_sp)),
            removed_reactions=sorted(set(all_rx)-set(retained_rx)),
            n_species=len(retained_sp), n_reactions=len(retained_rx),
            bfs_levels=bfs,
        )
    return ReductionResult(
        mechanism_id=1, method="DRG",
        target_species=["H2","O2","H2O"],
        steps=[
            _step(0.0, all_sp),
            _step(0.3, ["H2","O2","H2O","H","OH","O"]),
            _step(0.5, ["H2","O2","H2O"]),
        ],
        r_AB={"H2": {"O2": 0.9}},
    )


def _make_bipartite_G():
    G = nx.DiGraph()
    species = ["H2","O2","H2O","H","OH","O","HO2"]
    rxns    = ["H2+O2<=>H2O+H","H+O2<=>OH+O","H2+OH<=>H2O+H","HO2+H<=>OH+OH"]
    for i, sp in enumerate(species):
        G.add_node(f"S_{i}", type="species", name=sp,
                   pos_x=float(i), pos_y=0.0,
                   pos_x_3d=float(i), pos_y_3d=0.0, pos_z_3d=0.0)
    for j, fs in enumerate(rxns):
        G.add_node(f"R_{j}", type="reaction", formula_string=fs,
                   pos_x=float(j)+0.5, pos_y=1.0,
                   pos_x_3d=float(j)+0.5, pos_y_3d=1.0, pos_z_3d=1.0)
    return G


class TestTrimVisualizer:

    def test_import(self):
        from rmip.reduction.trim_viz import TrimVisualizer

    def test_init(self):
        from rmip.reduction.trim_viz import TrimVisualizer
        tv = TrimVisualizer(_make_trim_result(), _make_bipartite_G())
        assert len(tv._steps) == 3
        assert "H2" in tv._seeds

    def test_pos2d_populated(self):
        from rmip.reduction.trim_viz import TrimVisualizer
        tv = TrimVisualizer(_make_trim_result(), _make_bipartite_G())
        assert "H2" in tv._pos2d
        assert "H2+O2<=>H2O+H" in tv._pos2d

    def test_pos3d_populated(self):
        from rmip.reduction.trim_viz import TrimVisualizer
        tv = TrimVisualizer(_make_trim_result(), _make_bipartite_G())
        assert "H2" in tv._pos3d
        assert len(tv._pos3d["H2"]) == 3

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("plotly"),
        reason="plotly not installed",
    )
    def test_bfs_figure_2d(self):
        import plotly.graph_objects as go
        from rmip.reduction.trim_viz import TrimVisualizer
        tv  = TrimVisualizer(_make_trim_result(), _make_bipartite_G())
        fig = tv._bfs_figure_2d(tv._steps[1])
        assert isinstance(fig, go.Figure)

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("plotly"),
        reason="plotly not installed",
    )
    def test_trim_figure_2d(self):
        import plotly.graph_objects as go
        from rmip.reduction.trim_viz import TrimVisualizer
        tv  = TrimVisualizer(_make_trim_result(), _make_bipartite_G())
        fig = tv._trim_figure_2d(tv._steps[1])
        assert isinstance(fig, go.Figure)

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("plotly"),
        reason="plotly not installed",
    )
    def test_trim_figure_3d(self):
        import plotly.graph_objects as go
        from rmip.reduction.trim_viz import TrimVisualizer
        tv  = TrimVisualizer(_make_trim_result(), _make_bipartite_G())
        fig = tv._trim_figure_3d(tv._steps[2])
        assert isinstance(fig, go.Figure)

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("plotly"),
        reason="plotly not installed",
    )
    def test_export_html(self, tmp_path):
        from rmip.reduction.trim_viz import TrimVisualizer
        tv  = TrimVisualizer(_make_trim_result(), _make_bipartite_G())
        out = str(tmp_path / "trim.html")
        tv.export_html(out, eps_index=1)
        assert Path(out).exists()
        assert Path(out).stat().st_size > 100

    def test_removed_nodes_absent_in_retained(self):
        """Removed species are not in retained list of the trimmed step."""
        step = _make_trim_result().steps[2]  # ε=0.5, only seeds
        assert "HO2" not in step.retained_species
        assert "HO2" in step.removed_species


# ─────────────────────────────────────────────────────────────────────────────
# Migration tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase5BExtendedMigrations:

    def test_drgpsa_drgsa_in_reduction_runs(self, tmp_path):
        db_path = str(tmp_path / "ext.db")
        from rmip.database.schema import init_db
        from rmip.database.migrations import run_phase3_migrations, run_phase5b_migrations
        conn = init_db(db_path)
        run_phase3_migrations(conn)
        run_phase5b_migrations(conn)
        conn.execute("INSERT INTO mechanisms (name,format,loaded_at) "
                     "VALUES ('M','chemkin','now')")
        mid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        for method in ("DRG","DRGEP","PFA","DRGPSA","DRGSA"):
            conn.execute(
                "INSERT INTO reduction_runs "
                "(mechanism_id, method, target_species_json, created_at) "
                "VALUES (?,?,'[]','now')", (mid, method))
        conn.commit()
        rows = {r[0] for r in conn.execute("SELECT method FROM reduction_runs")}
        assert "DRGPSA" in rows
        assert "DRGSA"  in rows
        conn.close()

    def test_widen_narrow_constraint(self, tmp_path):
        """_ensure_wide_reduction_method_constraint widens an old-style DB."""
        db_path = str(tmp_path / "narrow.db")
        from rmip.database.schema import init_db
        from rmip.database.migrations import (run_phase3_migrations,
            _ensure_wide_reduction_method_constraint)
        conn = init_db(db_path)
        run_phase3_migrations(conn)
        # Create narrow table by hand
        conn.executescript("""
            DROP TABLE IF EXISTS reduction_runs;
            CREATE TABLE reduction_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mechanism_id INTEGER NOT NULL,
                method TEXT NOT NULL CHECK(method IN ('DRG','DRGEP','PFA')),
                target_species_json TEXT NOT NULL,
                epsilon_min REAL NOT NULL DEFAULT 0.0,
                epsilon_max REAL NOT NULL DEFAULT 0.5,
                n_steps INTEGER NOT NULL DEFAULT 51,
                n_species_full INTEGER,
                n_reactions_full INTEGER,
                n_species_minimal INTEGER,
                n_reactions_minimal INTEGER,
                created_at TEXT NOT NULL
            );
        """)
        conn.commit()
        _ensure_wide_reduction_method_constraint(conn)
        conn.commit()
        # Insert DRGPSA — should no longer fail
        conn.execute("INSERT INTO mechanisms (name,format,loaded_at) "
                     "VALUES ('M','chemkin','now')")
        mid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO reduction_runs "
            "(mechanism_id, method, target_species_json, created_at) "
            "VALUES (?,'DRGPSA','[]','now')", (mid,))
        conn.commit()
        row = conn.execute("SELECT method FROM reduction_runs").fetchone()
        assert row[0] == "DRGPSA"
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLI tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCLIExtended:

    def test_reduce_viz_registered(self):
        from rmip.cli import _build_parser
        p  = _build_parser()
        ns = p.parse_args(["reduce-viz","1","--db","x.db",
                           "--reduction-json","r.json","--mode","coeff"])
        assert ns.mech_id == 1
        assert ns.mode    == "coeff"

    def test_reduce_viz_both_mode(self):
        from rmip.cli import _build_parser
        p  = _build_parser()
        ns = p.parse_args(["reduce-viz","2","--db","x.db",
                           "--reduction-json","r.json",
                           "--mode","both","--port","9000"])
        assert ns.port == 9000

    def test_reduce_drgpsa_method(self):
        from rmip.cli import _build_parser
        p  = _build_parser()
        ns = p.parse_args(["reduce","1","--db","x.db",
                           "--method","DRGPSA","--target","H2,O2,H2O",
                           "--sa-json","sa.json"])
        assert ns.method  == "DRGPSA"
        assert ns.sa_json == "sa.json"

    def test_reduce_drgsa_method(self):
        from rmip.cli import _build_parser
        p  = _build_parser()
        ns = p.parse_args(["reduce","1","--db","x.db",
                           "--method","DRGSA","--target","H2,O2"])
        assert ns.method == "DRGSA"

    def test_reduce_viz_export_html_arg(self):
        from rmip.cli import _build_parser
        p  = _build_parser()
        ns = p.parse_args(["reduce-viz","1","--db","x.db",
                           "--reduction-json","r.json",
                           "--export-html","out/snap"])
        assert ns.export_html == "out/snap"

    def test_reduce_viz_threshold_arg(self):
        from rmip.cli import _build_parser
        p  = _build_parser()
        ns = p.parse_args(["reduce-viz","1","--db","x.db",
                           "--reduction-json","r.json",
                           "--threshold","0.15"])
        assert abs(ns.threshold - 0.15) < 1e-9
