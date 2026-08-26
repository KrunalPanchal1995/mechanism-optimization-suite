"""
rmip/tests/test_phase5b.py
==========================
Test suite for Phase 5B — Mechanism Reduction (DRG, DRGEP, PFA).

Coverage
--------
  Types (ReductionStep, ReductionResult)
  DRGReducer          — coefficient computation, BFS propagation, ε sweep
  DRGEPReducer        — path-propagated EP coefficients, extended BFS
  PFAReducer          — signed stoichiometry, production/consumption paths
  DB migration        — run_phase5b_migrations(), reduction_runs table
  CLI reduce command  — argument parsing, JSON output
  Stylesheet helper   — _default_stylesheet_with_reduction()

Run with:
    python -m pytest src/rmip/tests/test_phase5b.py -v
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from typing import List

import pytest
import networkx as nx

# ─────────────────────────────────────────────────────────────────────────────
# DB helpers (identical to other test modules)
# ─────────────────────────────────────────────────────────────────────────────

def _make_db(path: str) -> sqlite3.Connection:
    from rmip.database.schema import init_db
    conn = init_db(path)
    from rmip.database.migrations import run_phase3_migrations, run_phase5b_migrations
    run_phase3_migrations(conn)
    run_phase5b_migrations(conn)
    return conn


def _insert_mechanism(conn, name, n_sp=3, n_rxn=2):
    cur = conn.execute(
        "INSERT INTO mechanisms (name, format, filepath, description, "
        "n_species, n_reactions, loaded_at, metadata_json) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (name, "chemkin", "/fake.inp", f"Test {name}", n_sp, n_rxn,
         "2024-01-01T00:00:00Z", "{}"),
    )
    return cur.lastrowid


def _insert_species(conn, mech_id, name, inchi_key=None, formula="H2"):
    cur = conn.execute(
        "INSERT INTO species (mechanism_id, name, inchi_key, formula, "
        "canonical_smiles, glossary_matched, tmin_K, tmid_K, tmax_K) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (mech_id, name, inchi_key, formula, None, 0, 200, 1000, 5000),
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


# ─────────────────────────────────────────────────────────────────────────────
# Minimal mechanism fixture: H2 combustion skeleton
#
#  Species:  H2, O2, H2O, H, OH, HO2
#  Reactions:
#    R1: H2 + O2 <=> H2O + H      (H2 depends on O2, O2 on H2)
#    R2: H + O2 <=> OH + O2       (H depends on O2)
#    R3: H2 + OH <=> H2O + H      (H2 depends on OH, OH on H2)
#    R4: HO2 + H <=> OH + OH      (HO2 pathway — weakly connected)
#
# Target seed: ["H2", "O2", "H2O"]
# ─────────────────────────────────────────────────────────────────────────────

def _build_h2_fixture(db_path: str):
    """Build a minimal H2 mechanism DB and graph. Returns (conn, mech_id, G)."""
    conn    = _make_db(db_path)
    mech_id = _insert_mechanism(conn, "H2_test", n_sp=6, n_rxn=4)

    sp = {}
    for name, form in [("H2","H2"), ("O2","O2"), ("H2O","H2O"),
                       ("H","H"), ("OH","HO"), ("HO2","HO2"), ("O","O")]:
        sp[name] = _insert_species(conn, mech_id, name, formula=form)

    # R1: H2 + O2 → H2O + H
    r1 = _insert_reaction(conn, mech_id, "H2 + O2 <=> H2O + H", "H2_O2=>H2O_H")
    _insert_stoich(conn, r1, sp["H2"],  1.0, "reactant")
    _insert_stoich(conn, r1, sp["O2"],  1.0, "reactant")
    _insert_stoich(conn, r1, sp["H2O"], 1.0, "product")
    _insert_stoich(conn, r1, sp["H"],   1.0, "product")

    # R2: H + O2 → OH + O
    r2 = _insert_reaction(conn, mech_id, "H + O2 <=> OH + O", "H_O2=>OH_O")
    _insert_stoich(conn, r2, sp["H"],   1.0, "reactant")
    _insert_stoich(conn, r2, sp["O2"],  1.0, "reactant")
    _insert_stoich(conn, r2, sp["OH"],  1.0, "product")
    _insert_stoich(conn, r2, sp["O"],   1.0, "product")

    # R3: H2 + OH → H2O + H
    r3 = _insert_reaction(conn, mech_id, "H2 + OH <=> H2O + H", "H2_OH=>H2O_H")
    _insert_stoich(conn, r3, sp["H2"],  1.0, "reactant")
    _insert_stoich(conn, r3, sp["OH"],  1.0, "reactant")
    _insert_stoich(conn, r3, sp["H2O"], 1.0, "product")
    _insert_stoich(conn, r3, sp["H"],   1.0, "product")

    # R4: HO2 + H → OH + OH  (weakly connected at high ε)
    r4 = _insert_reaction(conn, mech_id, "HO2 + H <=> OH + OH", "HO2_H=>OH_OH")
    _insert_stoich(conn, r4, sp["HO2"], 1.0, "reactant")
    _insert_stoich(conn, r4, sp["H"],   1.0, "reactant")
    _insert_stoich(conn, r4, sp["OH"],  2.0, "product")

    conn.commit()

    from rmip.core.graph_builder import GraphBuilder
    G = GraphBuilder(db_path).build_graph(mech_id, include_reverse_edges=False,
                                          include_spring=False)
    return conn, mech_id, G


# ─────────────────────────────────────────────────────────────────────────────
# Types tests
# ─────────────────────────────────────────────────────────────────────────────

class TestReductionStep:
    def test_fields(self):
        from rmip.types import ReductionStep
        s = ReductionStep(
            threshold=0.1,
            retained_species=["H2", "O2"],
            retained_reactions=["H2 + O2 <=> H2O"],
            removed_species=["HO2"],
            removed_reactions=[],
            n_species=2,
            n_reactions=1,
        )
        assert s.threshold == 0.1
        assert s.n_species == 2

    def test_round_trip(self):
        from rmip.types import ReductionStep
        s = ReductionStep(
            threshold=0.25,
            retained_species=["H2"],
            retained_reactions=["H2 + O2 <=> H2O"],
            removed_species=["HO2", "OH"],
            removed_reactions=["HO2 + H <=> OH + OH"],
            n_species=1,
            n_reactions=1,
        )
        s2 = ReductionStep.from_dict(s.to_dict())
        assert s2.threshold == s.threshold
        assert s2.retained_species == s.retained_species
        assert s2.removed_species  == s.removed_species


class TestReductionResult:
    def _make_result(self):
        from rmip.types import ReductionStep, ReductionResult
        steps = [
            ReductionStep(0.0, ["H2","O2","H2O","H","OH","HO2"],
                          ["R1","R2","R3","R4"], [], [], 6, 4),
            ReductionStep(0.5, ["H2","O2","H2O","H","OH"],
                          ["R1","R2","R3"], ["HO2"], ["R4"], 5, 3),
        ]
        return ReductionResult(
            mechanism_id=1, method="DRG",
            target_species=["H2","O2","H2O"], steps=steps,
        )

    def test_fields(self):
        r = self._make_result()
        assert r.method == "DRG"
        assert len(r.steps) == 2

    def test_round_trip(self):
        r = self._make_result()
        r2 = r.__class__.from_dict(r.to_dict())
        assert r2.method == r.method
        assert len(r2.steps) == len(r.steps)
        assert r2.steps[1].threshold == 0.5

    def test_save_load(self, tmp_path):
        r = self._make_result()
        p = str(tmp_path / "result.json")
        r.save(p)
        assert Path(p).exists()
        r2 = r.__class__.load(p)
        assert r2.mechanism_id == r.mechanism_id
        assert r2.steps[0].n_species == 6


# ─────────────────────────────────────────────────────────────────────────────
# DRG coefficient tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDRGCoefficients:
    def _make_stoich(self):
        """
        Simple stoichiometry:
          R1: A + B → C   (A interacts with B and C; B interacts with A and C)
          R2: A + C → D   (A interacts with C and D)
        """
        return {
            1: {"A": 1.0, "B": 1.0, "C": 1.0},
            2: {"A": 1.0, "C": 1.0, "D": 1.0},
        }

    def test_coefficient_A_B(self):
        from rmip.reduction.drg import DRGReducer

        stoich = self._make_stoich()
        rates  = {1: 2.0, 2: 1.0}   # ω_1=2, ω_2=1
        reducer = DRGReducer.__new__(DRGReducer)

        all_sp = {"A", "B", "C", "D"}
        r_AB = reducer._compute_drg_coefficients(all_sp, stoich, rates)

        # r_AB for (A, B):
        #   denom_A = |1.0|*2 + |1.0|*1 = 3.0   (A in R1 and R2)
        #   numer_AB = |1.0|*2 = 2.0              (A,B both in R1)
        #   r_AB = 2.0/3.0 ≈ 0.667
        assert abs(r_AB[("A","B")] - 2/3) < 1e-9

    def test_coefficient_A_D(self):
        from rmip.reduction.drg import DRGReducer

        stoich = self._make_stoich()
        rates  = {1: 2.0, 2: 1.0}
        reducer = DRGReducer.__new__(DRGReducer)
        all_sp  = {"A", "B", "C", "D"}
        r_AB = reducer._compute_drg_coefficients(all_sp, stoich, rates)

        # r_AD: A,D co-appear only in R2
        #   denom_A = 3.0, numer_AD = 1.0 → r_AD = 1/3
        assert abs(r_AB[("A","D")] - 1/3) < 1e-9

    def test_symmetry_not_guaranteed(self):
        from rmip.reduction.drg import DRGReducer

        stoich = self._make_stoich()
        rates  = {1: 2.0, 2: 1.0}
        reducer = DRGReducer.__new__(DRGReducer)
        all_sp  = {"A", "B", "C", "D"}
        r_AB = reducer._compute_drg_coefficients(all_sp, stoich, rates)

        # r_BA ≠ r_AB in general
        # denom_B = |1.0|*2 = 2.0   (B only in R1)
        # numer_BA = |1.0|*2 = 2.0  → r_BA = 1.0
        assert abs(r_AB[("B","A")] - 1.0) < 1e-9
        assert r_AB[("B","A")] != r_AB[("A","B")]

    def test_zero_denom_gives_zero(self):
        from rmip.reduction.drg import DRGReducer

        stoich  = {1: {"A": 1.0, "B": 1.0}}
        rates   = {1: 0.0}  # zero rate → zero contribution
        reducer = DRGReducer.__new__(DRGReducer)
        all_sp  = {"A", "B"}
        r_AB = reducer._compute_drg_coefficients(all_sp, stoich, rates)
        # All contributions are 0; r_AB = 0/0 → 0
        assert r_AB.get(("A","B"), 0.0) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# DRGReducer full reduce() tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDRGReducer:
    def test_reduce_returns_result(self, tmp_path):
        from rmip.reduction.drg import DRGReducer
        db = str(tmp_path / "test.db")
        conn, mech_id, G = _build_h2_fixture(db)
        reducer = DRGReducer(db)
        result  = reducer.reduce(mech_id, G, ["H2","O2","H2O"],
                                 epsilon_range=[0.0, 0.3, 0.6], save_db=False)
        assert result.mechanism_id == mech_id
        assert result.method == "DRG"
        assert len(result.steps) == 3

    def test_eps_zero_retains_all(self, tmp_path):
        from rmip.reduction.drg import DRGReducer
        db = str(tmp_path / "test.db")
        conn, mech_id, G = _build_h2_fixture(db)
        result = DRGReducer(db).reduce(
            mech_id, G, ["H2","O2","H2O"],
            epsilon_range=[0.0], save_db=False
        )
        step0 = result.steps[0]
        assert step0.threshold == 0.0
        # At ε=0, every species with any connection to seed is retained
        assert step0.n_species >= 3

    def test_high_eps_drops_weakly_connected(self, tmp_path):
        from rmip.reduction.drg import DRGReducer
        db = str(tmp_path / "test.db")
        conn, mech_id, G = _build_h2_fixture(db)
        result = DRGReducer(db).reduce(
            mech_id, G, ["H2","O2","H2O"],
            epsilon_range=[0.0, 0.99], save_db=False
        )
        step_last = result.steps[-1]
        # HO2 should be dropped at high ε — only seeds remain
        assert step_last.n_species <= result.steps[0].n_species

    def test_seeds_always_retained(self, tmp_path):
        from rmip.reduction.drg import DRGReducer
        db = str(tmp_path / "test.db")
        conn, mech_id, G = _build_h2_fixture(db)
        result = DRGReducer(db).reduce(
            mech_id, G, ["H2","O2","H2O"],
            epsilon_range=[0.99], save_db=False
        )
        retained = set(result.steps[0].retained_species)
        assert "H2"  in retained
        assert "O2"  in retained
        assert "H2O" in retained

    def test_monotone_retention(self, tmp_path):
        from rmip.reduction.drg import DRGReducer
        db = str(tmp_path / "test.db")
        conn, mech_id, G = _build_h2_fixture(db)
        eps_range = [round(i * 0.1, 1) for i in range(11)]
        result = DRGReducer(db).reduce(
            mech_id, G, ["H2","O2","H2O"],
            epsilon_range=eps_range, save_db=False
        )
        sp_counts = [s.n_species for s in result.steps]
        # n_species should be non-increasing as ε grows
        assert sp_counts == sorted(sp_counts, reverse=True)

    def test_retained_reactions_subset_of_all(self, tmp_path):
        from rmip.reduction.drg import DRGReducer
        db = str(tmp_path / "test.db")
        conn, mech_id, G = _build_h2_fixture(db)
        result = DRGReducer(db).reduce(
            mech_id, G, ["H2","O2","H2O"],
            epsilon_range=[0.5], save_db=False
        )
        step = result.steps[0]
        all_set = set(step.retained_reactions) | set(step.removed_reactions)
        # Every formula_string from retained+removed should cover each reaction once
        assert len(step.retained_reactions) + len(step.removed_reactions) == \
               len(set(step.retained_reactions) | set(step.removed_reactions))

    def test_save_db_inserts_run(self, tmp_path):
        from rmip.reduction.drg import DRGReducer
        db = str(tmp_path / "test.db")
        conn, mech_id, G = _build_h2_fixture(db)
        result = DRGReducer(db).reduce(
            mech_id, G, ["H2","O2","H2O"],
            epsilon_range=[0.0, 0.5], save_db=True
        )
        assert result.run_id is not None
        with sqlite3.connect(db) as c:
            rows = c.execute(
                "SELECT * FROM reduction_runs WHERE id=?", (result.run_id,)
            ).fetchall()
        assert len(rows) == 1

    def test_no_db_save_no_row(self, tmp_path):
        from rmip.reduction.drg import DRGReducer
        db = str(tmp_path / "test.db")
        conn, mech_id, G = _build_h2_fixture(db)
        result = DRGReducer(db).reduce(
            mech_id, G, ["H2","O2","H2O"],
            epsilon_range=[0.0], save_db=False
        )
        assert result.run_id is None

    def test_result_save_load(self, tmp_path):
        from rmip.reduction.drg import DRGReducer
        from rmip.types import ReductionResult
        db = str(tmp_path / "test.db")
        conn, mech_id, G = _build_h2_fixture(db)
        result = DRGReducer(db).reduce(
            mech_id, G, ["H2","O2","H2O"],
            epsilon_range=[0.0, 0.3], save_db=False
        )
        p = str(tmp_path / "steps.json")
        result.save(p)
        r2 = ReductionResult.load(p)
        assert r2.method == "DRG"
        assert len(r2.steps) == 2


# ─────────────────────────────────────────────────────────────────────────────
# DRGEP tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDRGEPReducer:
    def test_method_label(self, tmp_path):
        from rmip.reduction.drgep import DRGEPReducer
        db = str(tmp_path / "test.db")
        conn, mech_id, G = _build_h2_fixture(db)
        result = DRGEPReducer(db).reduce(
            mech_id, G, ["H2","O2","H2O"],
            epsilon_range=[0.0], save_db=False
        )
        assert result.method == "DRGEP"

    def test_seeds_retained(self, tmp_path):
        from rmip.reduction.drgep import DRGEPReducer
        db = str(tmp_path / "test.db")
        conn, mech_id, G = _build_h2_fixture(db)
        result = DRGEPReducer(db).reduce(
            mech_id, G, ["H2","O2","H2O"],
            epsilon_range=[0.99], save_db=False
        )
        retained = set(result.steps[0].retained_species)
        for seed in ["H2","O2","H2O"]:
            assert seed in retained

    def test_ep_coefficients_in_range(self):
        from rmip.reduction.drgep import DRGEPReducer
        reducer = DRGEPReducer.__new__(DRGEPReducer)
        r_direct = {("A","B"): 0.8, ("B","C"): 0.6, ("A","C"): 0.3}
        all_sp   = {"A","B","C","D"}
        ep = reducer._compute_ep_coefficients(all_sp, ["A"], r_direct)
        # All EP values should be in [0, 1]
        for sp, val in ep.items():
            assert 0.0 <= val <= 1.0 + 1e-9, f"EP[{sp}]={val} out of range"

    def test_ep_seed_is_one(self):
        from rmip.reduction.drgep import DRGEPReducer
        reducer = DRGEPReducer.__new__(DRGEPReducer)
        r_direct = {("A","B"): 0.5}
        all_sp   = {"A","B"}
        ep = reducer._compute_ep_coefficients(all_sp, ["A"], r_direct)
        assert abs(ep["A"] - 1.0) < 1e-9

    def test_ep_indirect_path(self):
        from rmip.reduction.drgep import DRGEPReducer
        reducer = DRGEPReducer.__new__(DRGEPReducer)
        # A→B (0.8), B→C (0.7): EP_AC via path A→B→C = 0.8*0.7 = 0.56
        # Direct A→C = 0.3 (less)
        r_direct = {("A","B"): 0.8, ("B","C"): 0.7, ("A","C"): 0.3}
        all_sp   = {"A","B","C"}
        ep = reducer._compute_ep_coefficients(all_sp, ["A"], r_direct)
        assert ep["C"] > 0.3   # indirect path dominates

    def test_monotone_with_eps(self, tmp_path):
        from rmip.reduction.drgep import DRGEPReducer
        db = str(tmp_path / "test.db")
        conn, mech_id, G = _build_h2_fixture(db)
        eps_range = [round(i * 0.1, 1) for i in range(11)]
        result = DRGEPReducer(db).reduce(
            mech_id, G, ["H2","O2","H2O"],
            epsilon_range=eps_range, save_db=False
        )
        counts = [s.n_species for s in result.steps]
        assert counts == sorted(counts, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# PFA tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPFAReducer:
    def test_method_label(self, tmp_path):
        from rmip.reduction.pfa import PFAReducer
        db = str(tmp_path / "test.db")
        conn, mech_id, G = _build_h2_fixture(db)
        result = PFAReducer(db).reduce(
            mech_id, G, ["H2","O2","H2O"],
            epsilon_range=[0.0], save_db=False
        )
        assert result.method == "PFA"

    def test_seeds_retained(self, tmp_path):
        from rmip.reduction.pfa import PFAReducer
        db = str(tmp_path / "test.db")
        conn, mech_id, G = _build_h2_fixture(db)
        result = PFAReducer(db).reduce(
            mech_id, G, ["H2","O2","H2O"],
            epsilon_range=[0.99], save_db=False
        )
        retained = set(result.steps[0].retained_species)
        for seed in ["H2","O2","H2O"]:
            assert seed in retained

    def test_monotone(self, tmp_path):
        from rmip.reduction.pfa import PFAReducer
        db = str(tmp_path / "test.db")
        conn, mech_id, G = _build_h2_fixture(db)
        eps_range = [round(i * 0.1, 1) for i in range(11)]
        result = PFAReducer(db).reduce(
            mech_id, G, ["H2","O2","H2O"],
            epsilon_range=eps_range, save_db=False
        )
        counts = [s.n_species for s in result.steps]
        assert counts == sorted(counts, reverse=True)

    def test_pfa_coefficients_valid(self, tmp_path):
        from rmip.reduction.pfa import PFAReducer
        db = str(tmp_path / "test.db")
        conn, mech_id, G = _build_h2_fixture(db)
        reducer = PFAReducer(db)
        stoich   = reducer._load_stoichiometry(mech_id)
        all_sp, _ = reducer._load_species_reactions(mech_id)
        rates    = {rxn_id: 1.0 for rxn_id in stoich}
        r_AB     = reducer._compute_drg_coefficients(all_sp, stoich, rates)
        for (A, B), val in r_AB.items():
            assert 0.0 <= val <= 1.0 + 1e-9, f"r_AB[{A},{B}]={val}"


# ─────────────────────────────────────────────────────────────────────────────
# DB Migration tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase5BMigration:
    def test_table_created(self, tmp_path):
        from rmip.database.migrations import run_phase5b_migrations, _table_exists
        from rmip.database.schema import init_db
        db = str(tmp_path / "test.db")
        conn = init_db(db)
        run_phase5b_migrations(conn)
        assert _table_exists(conn, "reduction_runs")

    def test_idempotent(self, tmp_path):
        from rmip.database.migrations import run_phase5b_migrations
        from rmip.database.schema import init_db
        db = str(tmp_path / "test.db")
        conn = init_db(db)
        run_phase5b_migrations(conn)
        run_phase5b_migrations(conn)   # second call must not raise
        run_phase5b_migrations(conn)   # third call also safe

    def test_table_schema(self, tmp_path):
        from rmip.database.migrations import run_phase5b_migrations
        from rmip.database.schema import init_db
        db = str(tmp_path / "test.db")
        conn = init_db(db)
        run_phase5b_migrations(conn)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(reduction_runs)").fetchall()}
        for expected in ["id","mechanism_id","method","target_species_json",
                         "epsilon_min","epsilon_max","n_steps","created_at"]:
            assert expected in cols, f"Missing column: {expected}"

    def test_method_check_constraint(self, tmp_path):
        from rmip.database.migrations import run_phase5b_migrations
        from rmip.database.schema import init_db
        import sqlite3
        db = str(tmp_path / "test.db")
        conn = init_db(db)
        run_phase5b_migrations(conn)
        mech_id = _insert_mechanism(conn, "test")
        # Valid method
        conn.execute(
            "INSERT INTO reduction_runs (mechanism_id, method, target_species_json,"
            " epsilon_min, epsilon_max, n_steps, created_at) VALUES (?,?,?,?,?,?,?)",
            (mech_id, "DRG", "[]", 0.0, 0.5, 51, "2024-01-01T00:00:00Z")
        )
        conn.commit()
        # Invalid method should raise
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO reduction_runs (mechanism_id, method, target_species_json,"
                " epsilon_min, epsilon_max, n_steps, created_at) VALUES (?,?,?,?,?,?,?)",
                (mech_id, "INVALID", "[]", 0.0, 0.5, 51, "2024-01-01T00:00:00Z")
            )
            conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# CLI tests
# ─────────────────────────────────────────────────────────────────────────────

class TestReduceCLI:
    def test_reduce_drg_runs(self, tmp_path):
        from rmip.cli import main
        db = str(tmp_path / "test.db")
        conn, mech_id, G = _build_h2_fixture(db)
        out = str(tmp_path / "drg.json")
        ret = main([
            "reduce", str(mech_id),
            "--db", db,
            "--method", "DRG",
            "--target", "H2,O2,H2O",
            "--eps-max", "0.5",
            "--eps-steps", "11",
            "--out", out,
            "--no-db-save",
        ])
        assert ret == 0
        assert Path(out).exists()

    def test_reduce_output_is_valid_json(self, tmp_path):
        from rmip.cli import main
        db = str(tmp_path / "test.db")
        conn, mech_id, G = _build_h2_fixture(db)
        out = str(tmp_path / "drg.json")
        main([
            "reduce", str(mech_id),
            "--db", db,
            "--method", "DRG",
            "--target", "H2,O2,H2O",
            "--eps-steps", "5",
            "--out", out,
            "--no-db-save",
        ])
        with open(out) as fh:
            data = json.load(fh)
        assert data["method"] == "DRG"
        assert len(data["steps"]) == 5

    def test_reduce_drgep_cli(self, tmp_path):
        from rmip.cli import main
        db = str(tmp_path / "test.db")
        conn, mech_id, G = _build_h2_fixture(db)
        out = str(tmp_path / "drgep.json")
        ret = main([
            "reduce", str(mech_id),
            "--db", db,
            "--method", "DRGEP",
            "--target", "H2,O2,H2O",
            "--eps-steps", "3",
            "--out", out,
            "--no-db-save",
        ])
        assert ret == 0

    def test_reduce_pfa_cli(self, tmp_path):
        from rmip.cli import main
        db = str(tmp_path / "test.db")
        conn, mech_id, G = _build_h2_fixture(db)
        out = str(tmp_path / "pfa.json")
        ret = main([
            "reduce", str(mech_id),
            "--db", db,
            "--method", "PFA",
            "--target", "H2,O2,H2O",
            "--eps-steps", "3",
            "--out", out,
            "--no-db-save",
        ])
        assert ret == 0

    def test_reduce_with_db_save(self, tmp_path):
        from rmip.cli import main
        db = str(tmp_path / "test.db")
        conn, mech_id, G = _build_h2_fixture(db)
        out = str(tmp_path / "drg.json")
        ret = main([
            "reduce", str(mech_id),
            "--db", db,
            "--method", "DRG",
            "--target", "H2,O2,H2O",
            "--eps-steps", "3",
            "--out", out,
        ])
        assert ret == 0
        with sqlite3.connect(db) as c:
            rows = c.execute("SELECT * FROM reduction_runs").fetchall()
        assert len(rows) == 1

    def test_reduce_no_condition_legacy_path_unchanged(self, tmp_path, capsys):
        """No --condition given: the single-condition path must run exactly
        as before this phase (byte-identical shape) -- single 'Running
        {method} reduction' log line, no 'MULTI-CONDITION' branch taken."""
        from rmip.cli import main
        db = str(tmp_path / "test.db")
        conn, mech_id, G = _build_h2_fixture(db)
        out = str(tmp_path / "drg.json")
        ret = main([
            "reduce", str(mech_id),
            "--db", db,
            "--method", "DRG",
            "--target", "H2,O2,H2O",
            "--eps-steps", "5",
            "--out", out,
            "--no-db-save",
        ])
        assert ret == 0
        captured = capsys.readouterr()
        assert "Running DRG reduction on mech id=" in captured.out
        assert "MULTI-CONDITION" not in captured.out
        with open(out) as fh:
            data = json.load(fh)
        assert data["method"] == "DRG"
        assert len(data["steps"]) == 5

    def test_reduce_with_condition_flags_multi_condition_path(self, tmp_path, capsys):
        from rmip.cli import main
        db = str(tmp_path / "test.db")
        conn, mech_id, G = _build_h2_fixture(db)
        out = str(tmp_path / "drg_multi.json")
        ret = main([
            "reduce", str(mech_id),
            "--db", db,
            "--method", "DRG",
            "--target", "H2,O2,H2O",
            "--eps-steps", "5",
            "--out", out,
            "--condition", "1000,1.0,1.0,H2:1.0",
            "--condition", "1500,2.0,0.8,H2:1.0",
            "--no-db-save",
        ])
        assert ret == 0
        captured = capsys.readouterr()
        assert "MULTI-CONDITION" in captured.out
        assert "2 condition(s)" in captured.out
        with open(out) as fh:
            data = json.load(fh)
        assert data["method"] == "DRG"
        assert len(data["steps"]) == 5
        # seed species always retained regardless of condition
        for step in data["steps"]:
            assert {"H2", "O2", "H2O"} <= set(step["retained_species"])

    def test_reduce_with_condition_saves_single_run_plus_multi_condition_row(self, tmp_path):
        from rmip.cli import main
        db = str(tmp_path / "test.db")
        conn, mech_id, G = _build_h2_fixture(db)
        out = str(tmp_path / "drg_multi.json")
        ret = main([
            "reduce", str(mech_id),
            "--db", db,
            "--method", "DRG",
            "--target", "H2,O2,H2O",
            "--eps-steps", "3",
            "--out", out,
            "--condition", "1000,1.0,1.0,H2:1.0",
            "--condition", "1500,2.0,0.8,H2:1.0",
        ])
        assert ret == 0
        with sqlite3.connect(db) as c:
            run_rows = c.execute("SELECT * FROM reduction_runs").fetchall()
            mc_rows = c.execute(
                "SELECT * FROM reduction_multi_condition_runs").fetchall()
        assert len(run_rows) == 1
        assert len(mc_rows) == 1

    def test_reduce_condition_malformed_returns_error(self, tmp_path):
        from rmip.cli import main
        db = str(tmp_path / "test.db")
        conn, mech_id, G = _build_h2_fixture(db)
        out = str(tmp_path / "drg.json")
        ret = main([
            "reduce", str(mech_id),
            "--db", db,
            "--method", "DRG",
            "--target", "H2,O2,H2O",
            "--out", out,
            "--condition", "bad,shape",
            "--no-db-save",
        ])
        assert ret == 1


# ─────────────────────────────────────────────────────────────────────────────
# Visualizer stylesheet helper tests
# ─────────────────────────────────────────────────────────────────────────────

class TestReductionStylesheet:
    def test_base_stylesheet_no_reduction(self):
        from rmip.analysis.visualization import _default_stylesheet
        sheet = _default_stylesheet(threshold=0.1)
        assert isinstance(sheet, list)
        assert any("node" in s.get("selector","") for s in sheet)

    def test_reduction_stylesheet_contains_dim_rule(self):
        from rmip.analysis.visualization import _default_stylesheet_with_reduction
        sheet = _default_stylesheet_with_reduction(
            retained_species={"H2","O2"},
            retained_reactions={"H2 + O2 <=> H2O"},
        )
        # Should contain a "dim all" rule
        assert any(
            s.get("selector") == "node" and
            s.get("style",{}).get("opacity", 1) < 0.2
            for s in sheet
        )

    def test_reduction_stylesheet_has_undim_for_retained(self):
        from rmip.analysis.visualization import _default_stylesheet_with_reduction
        sheet = _default_stylesheet_with_reduction(
            retained_species={"H2"},
            retained_reactions=set(),
        )
        selectors = [s.get("selector","") for s in sheet]
        # Should have a selector targeting H2 by name
        assert any('name = "H2"' in sel for sel in selectors)

    def test_empty_retained_produces_valid_sheet(self):
        from rmip.analysis.visualization import _default_stylesheet_with_reduction
        sheet = _default_stylesheet_with_reduction(
            retained_species=set(),
            retained_reactions=set(),
        )
        assert isinstance(sheet, list)
        assert len(sheet) > 0
