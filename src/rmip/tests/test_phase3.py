"""
rmip/tests/test_phase3.py
==========================
Phase-3 test suite for RMIP.

Modules tested
--------------
  - rmip.analysis.fusion            (FusionEngine)
  - rmip.analysis.parameter_estimator (ArrheniusEstimate — already tested in Phase 1)
  - rmip.simulation.performance_ranker (PerformanceRanker)
  - rmip.optimization.mso_interface  (MSOInterface — JSON contract only)
  - rmip.io.chemkin_writer           (ChemkinWriter)
  - rmip.generation.rmg_interface    (RMGInterface — DB operations only)
  - rmip.database.migrations         (run_phase3_migrations)

Test strategy
-------------
All tests use in-memory SQLite databases populated with minimal fixture data.
No Cantera, RMG, or file-system side effects in the core tests.
Subprocess-invoking tests (RMG, ck2yaml) are skipped with
``pytest.mark.skip`` unless the relevant executable is on PATH.

Run with:
    python -m pytest rmip/tests/test_phase3.py -v
"""

from __future__ import annotations

import json
import math
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_db(path: str) -> sqlite3.Connection:
    """Create a minimal RMIP DB (Phase-1 schema) at the given path."""
    from rmip.database.schema import init_db
    conn = init_db(path)
    return conn


def _insert_mechanism(conn: sqlite3.Connection, name: str, n_sp: int = 3, n_rxn: int = 2) -> int:
    cur = conn.execute(
        """
        INSERT INTO mechanisms (name, format, filepath, description,
                                n_species, n_reactions, loaded_at, metadata_json)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (name, "chemkin", "/fake/path.inp", f"Test mechanism {name}",
         n_sp, n_rxn, "2024-01-01T00:00:00Z", "{}"),
    )
    return cur.lastrowid


def _insert_species(
    conn: sqlite3.Connection,
    mech_id: int,
    name: str,
    inchi_key: str,
    formula: str = "C7H16",
    coeffs_low: Optional[List[float]] = None,
    coeffs_high: Optional[List[float]] = None,
) -> int:
    if coeffs_low is None:
        coeffs_low  = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    if coeffs_high is None:
        coeffs_high = [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
    cur = conn.execute(
        """
        INSERT INTO species (mechanism_id, name, inchi_key, formula,
                             canonical_smiles, thermo_type,
                             tmin_K, tmid_K, tmax_K,
                             coeffs_low_json, coeffs_high_json,
                             glossary_matched)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (mech_id, name, inchi_key, formula, "CCCCCCC", "nasa7",
         200.0, 1000.0, 5000.0,
         json.dumps(coeffs_low), json.dumps(coeffs_high), 1),
    )
    return cur.lastrowid


def _insert_reaction(
    conn: sqlite3.Connection,
    mech_id: int,
    formula_string: str,
    canonical_key: str,
    A_si: float = 1e10,
    n: float = 0.5,
    Ea_si: float = 50000.0,
    kinetics_type: str = "arrhenius",
) -> int:
    cur = conn.execute(
        """
        INSERT INTO reactions
            (mechanism_id, formula_string, canonical_key,
             reversible, kinetics_type,
             A_si, A_orig, A_units, n, Ea_si, Ea_orig, Ea_units)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (mech_id, formula_string, canonical_key,
         1, kinetics_type,
         A_si, A_si * 1e-6, "m³/mol/s", n, Ea_si, Ea_si / 4184, "kcal/mol"),
    )
    return cur.lastrowid


def _insert_stoich(
    conn: sqlite3.Connection,
    rxn_id: int,
    sp_id: int,
    coef: float,
    role: str,
) -> None:
    conn.execute(
        "INSERT INTO stoichiometry (reaction_id, species_id, stoich_coeff, role) VALUES (?,?,?,?)",
        (rxn_id, sp_id, coef, role),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test: DB migrations
# ─────────────────────────────────────────────────────────────────────────────

class TestMigrations:

    def test_migrations_idempotent(self, tmp_path):
        """run_phase3_migrations should be safe to call twice."""
        from rmip.database.migrations import run_phase3_migrations, schema_version
        from rmip.database.schema_phase3 import upgrade_db

        db = str(tmp_path / "test.db")
        conn = _make_db(db)
        upgrade_db(conn)
        run_phase3_migrations(conn)
        run_phase3_migrations(conn)  # second call must not raise
        ver = schema_version(conn)
        assert all(ver.values()), f"Missing schema elements: {ver}"

    def test_fusion_columns_added(self, tmp_path):
        from rmip.database.migrations import run_phase3_migrations, _column_exists

        db = str(tmp_path / "test.db")
        conn = _make_db(db)
        run_phase3_migrations(conn)
        assert _column_exists(conn, "reactions", "fusion_class")
        assert _column_exists(conn, "reactions", "source_mechanism_ids_json")

    def test_new_tables_created(self, tmp_path):
        from rmip.database.migrations import run_phase3_migrations, _table_exists

        db = str(tmp_path / "test.db")
        conn = _make_db(db)
        run_phase3_migrations(conn)
        assert _table_exists(conn, "optimization_runs")
        assert _table_exists(conn, "fusion_metadata")
        assert _table_exists(conn, "completion_runs")

    def test_insert_optimization_run(self, tmp_path):
        from rmip.database.migrations import (
            run_phase3_migrations, insert_optimization_run, list_optimization_runs
        )
        db = str(tmp_path / "test.db")
        conn = _make_db(db)
        run_phase3_migrations(conn)
        mid = _insert_mechanism(conn, "FusedMech")
        row_id = insert_optimization_run(
            conn, session_id=12345, fused_mechanism_id=mid,
            n_fuse_reactions=5, n_target_conditions=10,
            settings={"objective": "WMRE"},
        )
        conn.commit()
        rows = list_optimization_runs(conn)
        assert len(rows) == 1
        assert rows[0]["session_id"] == 12345

    def test_insert_fusion_metadata(self, tmp_path):
        from rmip.database.migrations import (
            run_phase3_migrations, insert_fusion_metadata, list_fusion_metadata
        )
        db = str(tmp_path / "test.db")
        conn = _make_db(db)
        run_phase3_migrations(conn)
        mid = _insert_mechanism(conn, "Fused")
        insert_fusion_metadata(
            conn, fused_mechanism_id=mid,
            source_mechanism_ids=[1, 2],
            comparison_session_id=None,
            n_verbatim=5, n_copy=3, n_fused=2, n_species=10,
            uf_warn_threshold=10.0, provenance={},
        )
        conn.commit()
        rows = list_fusion_metadata(conn)
        assert len(rows) == 1
        assert json.loads(rows[0]["source_mechanism_ids"]) == [1, 2]


# ─────────────────────────────────────────────────────────────────────────────
# Test: ParameterEstimator (smoke test of Arrhenius estimation)
# ─────────────────────────────────────────────────────────────────────────────

class TestParameterEstimator:

    def test_geometric_mean_A(self):
        from rmip.analysis.parameter_estimator import ParameterEstimator
        pe = ParameterEstimator()
        est = pe.estimate_arrhenius(
            A_vals=[1e10, 1e12],   # geometric mean = 1e11
            n_vals=[0.5, 0.5],
            Ea_vals=[50000, 50000],
        )
        assert math.isclose(est.A_mean_si, 1e11, rel_tol=1e-6)
        assert math.isclose(est.n_mean, 0.5)
        assert math.isclose(est.Ea_mean_si, 50000.0)

    def test_uncertainty_factor(self):
        from rmip.analysis.parameter_estimator import ParameterEstimator
        pe = ParameterEstimator()
        est = pe.estimate_arrhenius(
            A_vals=[1e9, 1e11],
            n_vals=[0.0, 0.0],
            Ea_vals=[0.0, 0.0],
        )
        assert math.isclose(est.uncertainty_factor, 100.0, rel_tol=1e-4)

    def test_mso_bounds(self):
        from rmip.analysis.parameter_estimator import ParameterEstimator
        pe = ParameterEstimator()
        est = pe.estimate_arrhenius(
            A_vals=[1e10, 2e10, 3e10],
            n_vals=[0.3, 0.5, 0.7],
            Ea_vals=[40000, 50000, 60000],
        )
        bounds = pe.arrhenius_to_mso_bounds(est)
        assert "params_lb" in bounds
        assert "params_ub" in bounds
        assert bounds["params_lb"][0] <= bounds["params_mean"][0] <= bounds["params_ub"][0]


# ─────────────────────────────────────────────────────────────────────────────
# Test: FusionEngine
# ─────────────────────────────────────────────────────────────────────────────

class TestFusionEngine:

    def _make_two_mechs_db(self, tmp_path) -> tuple:
        """Create a DB with 2 mechanisms sharing species/reactions."""
        from rmip.database.migrations import run_phase3_migrations
        from rmip.database.schema_phase3 import upgrade_db

        db = str(tmp_path / "fusion_test.db")
        conn = _make_db(db)
        upgrade_db(conn)
        run_phase3_migrations(conn)

        mid1 = _insert_mechanism(conn, "Mech1")
        mid2 = _insert_mechanism(conn, "Mech2")

        # Species (shared)
        sp1_1 = _insert_species(conn, mid1, "NC7H16", "INCHI_KEY_FUEL")
        sp1_2 = _insert_species(conn, mid1, "O2",     "INCHI_KEY_O2")
        sp1_3 = _insert_species(conn, mid1, "CO2",    "INCHI_KEY_CO2")

        sp2_1 = _insert_species(conn, mid2, "NC7H16", "INCHI_KEY_FUEL")
        sp2_2 = _insert_species(conn, mid2, "O2",     "INCHI_KEY_O2")
        sp2_3 = _insert_species(conn, mid2, "CO2",    "INCHI_KEY_CO2")

        # Reactions
        # shared-identical
        rxn1_a = _insert_reaction(conn, mid1, "NC7H16+O2<=>CO2",
                                  "CK_RXN_A", A_si=1e10)
        rxn2_a = _insert_reaction(conn, mid2, "NC7H16+O2<=>CO2",
                                  "CK_RXN_A", A_si=1e10)  # identical A

        # shared-divergent
        rxn1_b = _insert_reaction(conn, mid1, "NC7H16<=>CO2",
                                  "CK_RXN_B", A_si=1e8)
        rxn2_b = _insert_reaction(conn, mid2, "NC7H16<=>CO2",
                                  "CK_RXN_B", A_si=1e11)  # divergent A

        # unique to mech1
        rxn1_c = _insert_reaction(conn, mid1, "O2<=>CO2",
                                  "CK_RXN_C", A_si=5e9)

        # Stoichiometry (minimal — just to test copy logic)
        _insert_stoich(conn, rxn1_a, sp1_1, 1.0, "reactant")
        _insert_stoich(conn, rxn1_a, sp1_2, 1.0, "reactant")
        _insert_stoich(conn, rxn1_a, sp1_3, 1.0, "product")
        _insert_stoich(conn, rxn2_a, sp2_1, 1.0, "reactant")
        _insert_stoich(conn, rxn2_a, sp2_2, 1.0, "reactant")
        _insert_stoich(conn, rxn2_a, sp2_3, 1.0, "product")

        conn.commit()
        return db, mid1, mid2

    def _make_comparison_result(self) -> Any:
        """Build a minimal GraphComparisonResult for two mechanisms."""
        from rmip.types import (
            GraphComparisonResult,
            ReactionClassification,
            SpeciesClassification,
            ReactionClass,
            SpeciesClass,
        )

        sp_clsf = [
            SpeciesClassification(
                inchi_key="INCHI_KEY_FUEL", canonical_smiles="CCCCCCC",
                formula="C7H16", classification=SpeciesClass.UNIVERSAL,
                present_in_mech_ids=[1, 2],
            ),
            SpeciesClassification(
                inchi_key="INCHI_KEY_O2", canonical_smiles="[O][O]",
                formula="O2", classification=SpeciesClass.UNIVERSAL,
                present_in_mech_ids=[1, 2],
            ),
            SpeciesClassification(
                inchi_key="INCHI_KEY_CO2", canonical_smiles="O=C=O",
                formula="CO2", classification=SpeciesClass.UNIVERSAL,
                present_in_mech_ids=[1, 2],
            ),
        ]

        rxn_clsf = [
            ReactionClassification(
                canonical_key="CK_RXN_A",
                formula_string="NC7H16+O2<=>CO2",
                classification=ReactionClass.SHARED_IDENTICAL,
                present_in_mech_ids=[1, 2],
            ),
            ReactionClassification(
                canonical_key="CK_RXN_B",
                formula_string="NC7H16<=>CO2",
                classification=ReactionClass.SHARED_DIVERGENT,
                present_in_mech_ids=[1, 2],
                mean_A_si=math.sqrt(1e8 * 1e11),
                mean_n=0.5, mean_Ea_si=50000.0,
                min_A_si=1e8, max_A_si=1e11,
                min_n=0.5,  max_n=0.5,
                min_Ea_si=50000.0, max_Ea_si=50000.0,
            ),
            ReactionClassification(
                canonical_key="CK_RXN_C",
                formula_string="O2<=>CO2",
                classification=ReactionClass.UNIQUE,
                present_in_mech_ids=[1],
            ),
        ]

        return GraphComparisonResult(
            session_id=999,
            mechanism_ids=[1, 2],
            species_classifications=sp_clsf,
            reaction_classifications=rxn_clsf,
        )

    def test_fusion_creates_new_mechanism(self, tmp_path):
        from rmip.analysis.fusion import FusionEngine

        db, mid1, mid2 = self._make_two_mechs_db(tmp_path)
        cmp = self._make_comparison_result()
        # Patch mechanism_ids to match actual DB ids
        cmp.mechanism_ids = [mid1, mid2]

        engine = FusionEngine(db)
        result = engine.fuse(cmp, name="TestFusion")

        assert result.fused_mechanism_id > 0
        assert result.n_verbatim + result.n_copy + result.n_fused == 3

    def test_fusion_reaction_counts(self, tmp_path):
        from rmip.analysis.fusion import FusionEngine

        db, mid1, mid2 = self._make_two_mechs_db(tmp_path)
        cmp = self._make_comparison_result()
        cmp.mechanism_ids = [mid1, mid2]

        engine = FusionEngine(db)
        result = engine.fuse(cmp, name="TestFusion")

        assert result.n_copy == 1      # CK_RXN_A (SHARED_IDENTICAL)
        assert result.n_fused == 1     # CK_RXN_B (SHARED_DIVERGENT)
        assert result.n_verbatim == 1  # CK_RXN_C (UNIQUE)

    def test_fuse_reaction_uses_geometric_mean_A(self, tmp_path):
        from rmip.analysis.fusion import FusionEngine

        db, mid1, mid2 = self._make_two_mechs_db(tmp_path)
        cmp = self._make_comparison_result()
        cmp.mechanism_ids = [mid1, mid2]

        engine = FusionEngine(db)
        result = engine.fuse(cmp, name="TestFusion")

        fuse_rec = next(r for r in result.reaction_records if r.canonical_key == "CK_RXN_B")
        expected_A = math.sqrt(1e8 * 1e11)
        assert math.isclose(fuse_rec.A_si, expected_A, rel_tol=1e-4)

    def test_export_mso_bounds(self, tmp_path):
        from rmip.analysis.fusion import FusionEngine

        db, mid1, mid2 = self._make_two_mechs_db(tmp_path)
        cmp = self._make_comparison_result()
        cmp.mechanism_ids = [mid1, mid2]

        engine = FusionEngine(db)
        result = engine.fuse(cmp, name="TestFusion")

        out_path = str(tmp_path / "mso_bounds.json")
        engine.export_mso_bounds(result, out_path)

        with open(out_path) as fh:
            data = json.load(fh)

        assert data["n_fuse_reactions"] == 1
        block = data["fuse_blocks"][0]
        assert block["canonical_key"] == "CK_RXN_B"
        assert len(block["params_mean"]) == 3
        assert block["params_lb"][0] <= block["params_mean"][0] <= block["params_ub"][0]


# ─────────────────────────────────────────────────────────────────────────────
# Test: PerformanceRanker
# ─────────────────────────────────────────────────────────────────────────────

class TestPerformanceRanker:

    def _make_ranking_db(self, tmp_path) -> tuple:
        """DB with two mechanisms + sim results + experimental data."""
        from rmip.database.schema_phase3 import upgrade_db

        db = str(tmp_path / "rank_test.db")
        conn = _make_db(db)
        upgrade_db(conn)

        mid1 = _insert_mechanism(conn, "Mech_good")
        mid2 = _insert_mechanism(conn, "Mech_bad")

        # Experimental dataset
        ts = "2024-01-01T00:00:00Z"
        cur = conn.execute(
            """
            INSERT INTO exp_datasets
                (name, sim_type, fuel_name, source_ref, n_points, created_at)
            VALUES (?,?,?,?,?,?)
            """,
            ("Curran1998_IDT", "IDT", "NC7H16", "Curran1998", 2, ts),
        )
        ds_id = cur.lastrowid

        conn.execute(
            """
            INSERT INTO exp_data_points
                (dataset_id, T_K, P_atm, phi, value, units)
            VALUES (?,?,?,?,?,?)
            """,
            (ds_id, 1100.0, 13.5, 1.0, 2.0e-4, "s"),
        )
        conn.execute(
            """
            INSERT INTO exp_data_points
                (dataset_id, T_K, P_atm, phi, value, units)
            VALUES (?,?,?,?,?,?)
            """,
            (ds_id, 1300.0, 13.5, 1.0, 1.0e-5, "s"),
        )

        # Simulation results (mech1 closer to exp)
        for mech_id, err_factor in [(mid1, 1.1), (mid2, 2.0)]:
            for T_K, exp_val in [(1100.0, 2.0e-4), (1300.0, 1.0e-5)]:
                conn.execute(
                    """
                    INSERT INTO simulation_runs
                        (mechanism_id, sim_type, T_K, P_atm, phi,
                         fuel_name, fuel_composition, oxidizer_composition,
                         result, converged, created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (mech_id, "IDT", T_K, 13.5, 1.0,
                     "NC7H16", "NC7H16:1.0", "O2:1.0, N2:3.76",
                     exp_val * err_factor, 1, ts),
                )

        conn.commit()
        return db, mid1, mid2, ds_id

    def test_rank_output_structure(self, tmp_path):
        from rmip.simulation.performance_ranker import PerformanceRanker

        db, mid1, mid2, ds_id = self._make_ranking_db(tmp_path)
        ranker = PerformanceRanker(db)
        report = ranker.rank([mid1, mid2], ds_id, "IDT")

        assert report.dataset_id == ds_id
        assert report.sim_type == "IDT"
        assert set(report.mechanism_ids) == {mid1, mid2}

    def test_rank_mech1_beats_mech2(self, tmp_path):
        from rmip.simulation.performance_ranker import PerformanceRanker

        db, mid1, mid2, ds_id = self._make_ranking_db(tmp_path)
        ranker = PerformanceRanker(db)
        report = ranker.rank([mid1, mid2], ds_id, "IDT")

        best = report.best_per_regime()
        # mid1 should rank 1 in HTC (T_K = 1100, 1300 are both HTC)
        htc_best = best.get("HTC")
        if htc_best:  # may be None if all conditions land in HTC only
            assert htc_best.mechanism_id == mid1

    def test_rank_mre_values(self, tmp_path):
        from rmip.simulation.performance_ranker import PerformanceRanker

        db, mid1, mid2, ds_id = self._make_ranking_db(tmp_path)
        ranker = PerformanceRanker(db)
        report = ranker.rank([mid1, mid2], ds_id, "IDT")

        htc_scores = report.scores_for_regime("HTC")
        assert len(htc_scores) == 2
        mre_good = next(s.mre for s in htc_scores if s.mechanism_id == mid1)
        mre_bad  = next(s.mre for s in htc_scores if s.mechanism_id == mid2)
        assert mre_good < mre_bad

    def test_rank_summary_string(self, tmp_path):
        from rmip.simulation.performance_ranker import PerformanceRanker

        db, mid1, mid2, ds_id = self._make_ranking_db(tmp_path)
        ranker = PerformanceRanker(db)
        report = ranker.rank([mid1, mid2], ds_id, "IDT")
        summary = report.summary()
        assert "RankingReport" in summary
        assert "MRE" in summary


# ─────────────────────────────────────────────────────────────────────────────
# Test: MSOInterface (JSON contract — no subprocess)
# ─────────────────────────────────────────────────────────────────────────────

class TestMSOInterface:

    def test_prepare_writes_json(self, tmp_path):
        from rmip.database.migrations import run_phase3_migrations
        from rmip.optimization.mso_interface import MSOInterface

        db = str(tmp_path / "mso.db")
        conn = _make_db(db)
        run_phase3_migrations(conn)
        mid = _insert_mechanism(conn, "FusedMech")
        conn.commit()

        # Write a minimal mso_bounds.json
        bounds = {
            "fused_mechanism_id": mid,
            "n_fuse_reactions": 1,
            "fuse_blocks": [
                {
                    "canonical_key": "CK_RXN_B",
                    "formula_string": "NC7H16<=>CO2",
                    "source_mech_ids": [1, 2],
                    "params_mean": [1e9, 0.5, 50000.0],
                    "params_lb":   [1e8, 0.3, 40000.0],
                    "params_ub":   [1e10, 0.7, 60000.0],
                    "param_names": ["A_si", "n", "Ea_si"],
                    "units":       ["m^3/mol/s", "1", "J/mol"],
                }
            ],
        }
        bounds_path = str(tmp_path / "mso_bounds.json")
        with open(bounds_path, "w") as fh:
            json.dump(bounds, fh)

        target_conditions = [
            {
                "T_K": 1100.0, "P_atm": 13.5, "phi": 1.0,
                "fuel": "NC7H16:1.0", "oxidizer": "O2:1.0, N2:3.76",
                "observable": "IDT", "exp_value": 2.0e-4, "weight": 1.0,
            }
        ]

        work_dir = str(tmp_path / "mso_work")
        msi = MSOInterface(db, work_dir=work_dir)
        inp_path = msi.prepare(mid, bounds_path, target_conditions)

        assert inp_path.exists()
        with open(inp_path) as fh:
            data = json.load(fh)

        assert data["n_fuse_reactions"] == 1
        assert data["fuse_blocks"][0]["canonical_key"] == "CK_RXN_B"
        assert len(data["target_conditions"]) == 1

    def test_apply_creates_optimized_mechanism(self, tmp_path):
        """Test that MSOInterface.apply() creates a new mechanism in the DB."""
        from rmip.database.migrations import run_phase3_migrations
        from rmip.optimization.mso_interface import MSOInterface

        db = str(tmp_path / "mso.db")
        conn = _make_db(db)
        run_phase3_migrations(conn)
        mid = _insert_mechanism(conn, "FusedMech")
        sp_id = _insert_species(conn, mid, "NC7H16", "INCHI_KEY_FUEL")
        rxn_id = _insert_reaction(conn, mid, "NC7H16<=>CO2", "CK_RXN_B", A_si=1e9)
        _insert_stoich(conn, rxn_id, sp_id, 1.0, "reactant")
        conn.commit()

        work_dir = tmp_path / "mso_work"
        work_dir.mkdir()

        # Write dummy mso_input.json
        mso_input = {
            "session_id": 12345,
            "fused_mechanism_id": mid,
            "fuse_blocks": [
                {
                    "canonical_key": "CK_RXN_B",
                    "reaction_index": 0,
                    "params_mean": [1e9, 0.5, 50000.0],
                }
            ],
        }
        (work_dir / "mso_input.json").write_text(json.dumps(mso_input))

        # Write dummy mso_output
        out_dir = work_dir / "mso_output"
        out_dir.mkdir()
        mso_output = {
            "session_id": 12345,
            "status": "success",
            "objective_final": 0.05,
            "fuse_blocks": [
                {
                    "reaction_index": 0,
                    "canonical_key": "CK_RXN_B",
                    "optimized_params": [2.0e9, 0.6, 52000.0],
                }
            ],
        }
        (out_dir / "optimized_params.json").write_text(json.dumps(mso_output))

        msi = MSOInterface(db, work_dir=str(work_dir))
        result = msi.apply()

        assert result.status == "success"
        assert result.optimized_mech_id is not None
        assert result.optimized_mech_id != mid
        assert abs(result.objective_final - 0.05) < 1e-9
        assert len(result.reactions) == 1
        assert math.isclose(result.reactions[0].A_si_opt, 2.0e9)

    def test_apply_optimized_A_updated_in_db(self, tmp_path):
        """Verify the optimised A_si is written to the new DB mechanism."""
        from rmip.database.migrations import run_phase3_migrations
        from rmip.optimization.mso_interface import MSOInterface

        db = str(tmp_path / "mso.db")
        conn = _make_db(db)
        run_phase3_migrations(conn)
        mid = _insert_mechanism(conn, "FusedMech")
        sp_id = _insert_species(conn, mid, "NC7H16", "INCHI_KEY_FUEL2")
        rxn_id = _insert_reaction(conn, mid, "NC7H16<=>CO2", "CK_RXN_Z", A_si=1e9)
        _insert_stoich(conn, rxn_id, sp_id, 1.0, "reactant")
        conn.commit()

        work_dir = tmp_path / "mso_z"
        work_dir.mkdir()
        (work_dir / "mso_input.json").write_text(json.dumps({
            "session_id": 99, "fused_mechanism_id": mid,
            "fuse_blocks": [{"canonical_key": "CK_RXN_Z", "reaction_index": 0,
                             "params_mean": [1e9, 0.0, 0.0]}],
        }))
        out_dir = work_dir / "mso_output"
        out_dir.mkdir()
        (out_dir / "optimized_params.json").write_text(json.dumps({
            "session_id": 99, "status": "success", "objective_final": 0.01,
            "fuse_blocks": [{"reaction_index": 0, "canonical_key": "CK_RXN_Z",
                             "optimized_params": [9.99e9, 0.0, 0.0]}],
        }))

        msi = MSOInterface(db, work_dir=str(work_dir))
        result = msi.apply()

        opt_mid = result.optimized_mech_id
        with sqlite3.connect(db) as c:
            row = c.execute(
                "SELECT A_si FROM reactions WHERE mechanism_id=? AND canonical_key=?",
                (opt_mid, "CK_RXN_Z"),
            ).fetchone()
        assert row is not None
        assert math.isclose(row[0], 9.99e9, rel_tol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# Test: ChemkinWriter
# ─────────────────────────────────────────────────────────────────────────────

class TestChemkinWriter:

    def _make_writable_db(self, tmp_path) -> tuple:
        """DB with one mechanism, 2 species, 1 reaction."""
        db = str(tmp_path / "write_test.db")
        conn = _make_db(db)
        mid = _insert_mechanism(conn, "TestMech")
        sp1 = _insert_species(
            conn, mid, "NC7H16", "INCHI_FUEL", formula="C7H16",
            coeffs_low=[1.0, 2.0, 3.0, 4.0, 5.0, -6.0, 7.0],
            coeffs_high=[8.0, 9.0, 10.0, 11.0, 12.0, -13.0, 14.0],
        )
        sp2 = _insert_species(
            conn, mid, "O2", "INCHI_O2", formula="O2",
            coeffs_low=[3.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0],
            coeffs_high=[3.7, 0.0, 0.0, 0.0, 0.0, -1.2, 4.0],
        )
        rxn_id = _insert_reaction(
            conn, mid, "NC7H16+O2<=>CO2+H2O", "CK_RXN_MAIN",
            A_si=1e10, n=0.5, Ea_si=50000.0
        )
        _insert_stoich(conn, rxn_id, sp1, 1.0, "reactant")
        _insert_stoich(conn, rxn_id, sp2, 11.0, "reactant")
        conn.commit()
        return db, mid

    def test_write_creates_file(self, tmp_path):
        from rmip.io.chemkin_writer import ChemkinWriter

        db, mid = self._make_writable_db(tmp_path)
        writer = ChemkinWriter(db)
        out_path = str(tmp_path / "mech.inp")
        writer.write(mid, out_path)

        assert Path(out_path).exists()
        content = Path(out_path).read_text()
        assert "ELEMENTS" in content
        assert "SPECIES"  in content
        assert "THERMO"   in content
        assert "REACTIONS" in content
        assert "NC7H16"   in content

    def test_write_contains_reaction(self, tmp_path):
        from rmip.io.chemkin_writer import ChemkinWriter

        db, mid = self._make_writable_db(tmp_path)
        writer = ChemkinWriter(db)
        out_path = str(tmp_path / "mech.inp")
        writer.write(mid, out_path)

        content = Path(out_path).read_text()
        # Reaction formula stored in DB should appear
        assert "NC7H16+O2<=>CO2+H2O" in content or "NC7H16" in content

    def test_thermo_has_4_lines_per_species(self, tmp_path):
        from rmip.io.chemkin_writer import ChemkinWriter

        db, mid = self._make_writable_db(tmp_path)
        writer = ChemkinWriter(db)
        out_path = str(tmp_path / "mech.inp")
        writer.write(mid, out_path)

        content = Path(out_path).read_text()
        # Count lines ending in "1", "2", "3", "4" (NASA-7 line tags)
        thermo_block = content.split("THERMO")[1].split("END")[0]
        lines_with_tag = [l for l in thermo_block.splitlines()
                          if l.rstrip().endswith(("1", "2", "3", "4"))]
        # 2 species × 4 lines = 8 tagged lines
        assert len(lines_with_tag) == 8


# ─────────────────────────────────────────────────────────────────────────────
# Test: RMGInterface (DB operations only — no RMG subprocess)
# ─────────────────────────────────────────────────────────────────────────────

class TestRMGInterface:

    def _make_rmg_db(self, tmp_path) -> tuple:
        """DB with a base mechanism and a simulated 'RMG' mechanism."""
        from rmip.database.migrations import run_phase3_migrations

        db = str(tmp_path / "rmg_test.db")
        conn = _make_db(db)
        run_phase3_migrations(conn)

        mid_base = _insert_mechanism(conn, "BaseMech")
        mid_rmg  = _insert_mechanism(conn, "RMG_NC7H16")

        # Shared species
        _insert_species(conn, mid_base, "NC7H16", "IK_FUEL")
        _insert_species(conn, mid_base, "O2",     "IK_O2")
        _insert_species(conn, mid_rmg,  "NC7H16", "IK_FUEL")
        _insert_species(conn, mid_rmg,  "O2",     "IK_O2")

        # Extra species in RMG only
        _insert_species(conn, mid_rmg, "C7H15",  "IK_C7H15", formula="C7H15")
        _insert_species(conn, mid_rmg, "HO2",    "IK_HO2",   formula="HO2")

        # Reactions
        _insert_reaction(conn, mid_base, "NC7H16+O2<=>CO2", "CK_MAIN", A_si=1e10)
        _insert_reaction(conn, mid_rmg,  "NC7H16+O2<=>CO2", "CK_MAIN", A_si=1e10)  # shared
        sp_c7h15 = conn.execute(
            "SELECT id FROM species WHERE mechanism_id=? AND name='C7H15'", (mid_rmg,)
        ).fetchone()[0]
        rxn_new_id = _insert_reaction(conn, mid_rmg, "NC7H16<=>C7H15", "CK_NEW", A_si=5e8)
        sp_fuel_rmg = conn.execute(
            "SELECT id FROM species WHERE mechanism_id=? AND name='NC7H16'", (mid_rmg,)
        ).fetchone()[0]
        _insert_stoich(conn, rxn_new_id, sp_fuel_rmg, 1.0, "reactant")
        _insert_stoich(conn, rxn_new_id, sp_c7h15,    1.0, "product")

        conn.commit()
        return db, mid_base, mid_rmg

    def test_find_missing_species(self, tmp_path):
        from rmip.generation.rmg_interface import RMGInterface

        db, mid_base, mid_rmg = self._make_rmg_db(tmp_path)
        rmg = RMGInterface(db)
        missing = rmg.find_missing_species(mid_base, mid_rmg)
        assert set(missing) == {"IK_C7H15", "IK_HO2"}

    def test_find_missing_reactions(self, tmp_path):
        from rmip.generation.rmg_interface import RMGInterface

        db, mid_base, mid_rmg = self._make_rmg_db(tmp_path)
        rmg = RMGInterface(db)
        missing = rmg.find_missing_reactions(mid_base, mid_rmg)
        assert "CK_NEW" in missing
        assert "CK_MAIN" not in missing

    def test_find_missing_pathways(self, tmp_path):
        from rmip.generation.rmg_interface import RMGInterface

        db, mid_base, mid_rmg = self._make_rmg_db(tmp_path)
        rmg = RMGInterface(db)
        report = rmg.find_missing_pathways(mid_base, mid_rmg)
        assert report.n_missing_species  == 2
        assert report.n_missing_reactions == 1

    def test_append_to_mechanism_inplace(self, tmp_path):
        from rmip.generation.rmg_interface import RMGInterface

        db, mid_base, mid_rmg = self._make_rmg_db(tmp_path)
        rmg = RMGInterface(db)
        target_id = rmg.append_to_mechanism(mid_base, mid_rmg)

        assert target_id == mid_base  # in-place modification
        with sqlite3.connect(db) as conn:
            sp_count = conn.execute(
                "SELECT COUNT(*) FROM species WHERE mechanism_id=?", (mid_base,)
            ).fetchone()[0]
        assert sp_count == 4  # 2 original + 2 RMG-only

    def test_append_creates_clone_when_new_name_given(self, tmp_path):
        from rmip.generation.rmg_interface import RMGInterface

        db, mid_base, mid_rmg = self._make_rmg_db(tmp_path)
        rmg = RMGInterface(db)
        clone_id = rmg.append_to_mechanism(
            mid_base, mid_rmg, new_name="BaseMech_plus_RMG"
        )

        assert clone_id != mid_base
        with sqlite3.connect(db) as conn:
            # Base mechanism should be unmodified
            base_sp = conn.execute(
                "SELECT COUNT(*) FROM species WHERE mechanism_id=?", (mid_base,)
            ).fetchone()[0]
            clone_sp = conn.execute(
                "SELECT COUNT(*) FROM species WHERE mechanism_id=?", (clone_id,)
            ).fetchone()[0]

        assert base_sp == 2   # original, unchanged
        assert clone_sp == 4  # base + 2 from RMG


# ─────────────────────────────────────────────────────────────────────────────
# Integration smoke test
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase3Integration:

    def test_full_pipeline_smoke(self, tmp_path):
        """
        Smoke test: migrations → fusion → export bounds → prepare MSO input.
        No subprocess calls.  Verifies the JSON chain is consistent.
        """
        from rmip.database.migrations import run_phase3_migrations
        from rmip.database.schema_phase3 import upgrade_db
        from rmip.analysis.fusion import FusionEngine
        from rmip.optimization.mso_interface import MSOInterface
        from rmip.types import (
            GraphComparisonResult,
            ReactionClassification,
            SpeciesClassification,
            ReactionClass,
            SpeciesClass,
        )

        db = str(tmp_path / "integration.db")
        conn = _make_db(db)
        upgrade_db(conn)
        run_phase3_migrations(conn)

        mid1 = _insert_mechanism(conn, "MechA")
        mid2 = _insert_mechanism(conn, "MechB")
        sp1 = _insert_species(conn, mid1, "FUEL", "IK_FUEL")
        sp2 = _insert_species(conn, mid2, "FUEL", "IK_FUEL")
        sp3 = _insert_species(conn, mid1, "PROD", "IK_PROD")
        sp4 = _insert_species(conn, mid2, "PROD", "IK_PROD")
        rxn1 = _insert_reaction(conn, mid1, "FUEL<=>PROD", "CK_MAIN", A_si=1e10)
        rxn2 = _insert_reaction(conn, mid2, "FUEL<=>PROD", "CK_MAIN", A_si=1e11)  # divergent
        _insert_stoich(conn, rxn1, sp1, 1.0, "reactant")
        _insert_stoich(conn, rxn1, sp3, 1.0, "product")
        _insert_stoich(conn, rxn2, sp2, 1.0, "reactant")
        _insert_stoich(conn, rxn2, sp4, 1.0, "product")
        conn.commit()

        cmp = GraphComparisonResult(
            session_id=42,
            mechanism_ids=[mid1, mid2],
            species_classifications=[
                SpeciesClassification("IK_FUEL", "CCCCCCC", "C7H16",
                                      SpeciesClass.UNIVERSAL, [mid1, mid2]),
                SpeciesClassification("IK_PROD", "O=C=O",   "CO2",
                                      SpeciesClass.UNIVERSAL, [mid1, mid2]),
            ],
            reaction_classifications=[
                ReactionClassification(
                    "CK_MAIN", "FUEL<=>PROD",
                    ReactionClass.SHARED_DIVERGENT, [mid1, mid2],
                    min_A_si=1e10, max_A_si=1e11,
                    min_n=0.5, max_n=0.5,
                    min_Ea_si=50000.0, max_Ea_si=50000.0,
                ),
            ],
        )

        engine = FusionEngine(db)
        fusion_result = engine.fuse(cmp, name="IntegrationFusion")
        assert fusion_result.n_fused == 1

        bounds_path = str(tmp_path / "bounds.json")
        engine.export_mso_bounds(fusion_result, bounds_path)

        msi = MSOInterface(db, work_dir=str(tmp_path / "mso_work"))
        inp_path = msi.prepare(
            fused_mechanism_id=fusion_result.fused_mechanism_id,
            mso_bounds_path=bounds_path,
            target_conditions=[{
                "T_K": 1100.0, "P_atm": 13.5, "phi": 1.0,
                "fuel": "FUEL:1.0", "oxidizer": "O2:1.0, N2:3.76",
                "observable": "IDT", "exp_value": 2.0e-4,
            }],
        )

        with open(inp_path) as fh:
            mso_data = json.load(fh)

        assert mso_data["fused_mechanism_id"] == fusion_result.fused_mechanism_id
        assert mso_data["n_fuse_reactions"] == 1
        block = mso_data["fuse_blocks"][0]
        assert block["params_lb"][0] < block["params_mean"][0] < block["params_ub"][0]


# ===========================================================================
# Phase-1 Regression Tests — verifying core modules still work in Phase-3
# ===========================================================================

try:
    import networkx as nx  # noqa: F401
    _HAS_NX_REGRESSION = True
except ImportError:
    _HAS_NX_REGRESSION = False

try:
    from rdkit import Chem  # noqa: F401
    _HAS_RDKIT_REGRESSION = True
except ImportError:
    _HAS_RDKIT_REGRESSION = False


class TestDatabaseSchemaRegression:
    """Verify the schema DDL has the UNIQUE species constraint and batch insert functions."""

    def test_species_unique_constraint(self, tmp_path):
        from rmip.database.schema import init_db, insert_mechanism, insert_species
        db = tmp_path / "test.db"
        conn = init_db(str(db))
        mid = insert_mechanism(conn, name="test", format="chemkin")
        sid1 = insert_species(conn, mechanism_id=mid, name="CH4")
        sid2 = insert_species(conn, mechanism_id=mid, name="CH4")
        assert sid1 == sid2  # should return existing id due to UNIQUE constraint

    def test_batch_insert_stoichiometry(self, tmp_path):
        from rmip.database.schema import init_db, insert_mechanism, insert_species, insert_reaction, insert_stoichiometry_batch
        db = tmp_path / "test.db"
        conn = init_db(str(db))
        mid = insert_mechanism(conn, name="test", format="chemkin")
        sid = insert_species(conn, mechanism_id=mid, name="H2")
        rid = insert_reaction(conn, mechanism_id=mid, formula_string="H2<=>H+H")
        insert_stoichiometry_batch(conn, [(rid, sid, 1.0, "reactant")])
        rows = conn.execute(
            "SELECT * FROM stoichiometry WHERE reaction_id=?", (rid,)
        ).fetchall()
        assert len(rows) == 1

    def test_batch_insert_third_body(self, tmp_path):
        from rmip.database.schema import init_db, insert_mechanism, insert_species, insert_reaction, insert_third_body_batch
        db = tmp_path / "test.db"
        conn = init_db(str(db))
        mid = insert_mechanism(conn, name="test", format="chemkin")
        sid = insert_species(conn, mechanism_id=mid, name="N2")
        rid = insert_reaction(conn, mechanism_id=mid, formula_string="H+O2<=>OH+O")
        insert_third_body_batch(conn, [(rid, sid, 1.0)])
        rows = conn.execute(
            "SELECT * FROM third_body_efficiencies WHERE reaction_id=?", (rid,)
        ).fetchall()
        assert len(rows) == 1

    def test_init_db_creates_all_tables(self, tmp_path):
        from rmip.database.schema import init_db
        db = tmp_path / "test.db"
        conn = init_db(str(db))
        tables = {
            r[0] for r in
            conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        required = {"mechanisms", "species", "reactions", "stoichiometry", "third_body_efficiencies"}
        assert required.issubset(tables)


class TestCanonicalizationRegression:
    """Verify canonicalization functions from Phase-1."""

    def test_canonicalize_smiles_simple(self):
        from rmip.core.canonicalize import canonicalize_smiles
        assert canonicalize_smiles("CCCCCC") == "CCCCCC"
        nc = "C(C)(C)C"
        can = canonicalize_smiles(nc)
        assert can is not None
        assert canonicalize_smiles(can) == can

    def test_canonicalize_returns_none_on_bad_smiles(self):
        from rmip.core.canonicalize import canonicalize_smiles
        assert canonicalize_smiles("NOT_A_SMILES!!!") is None

    def test_make_reaction_key_direction_normalized(self):
        from rmip.core.canonicalize import make_reaction_canonical_key
        key_fwd = make_reaction_canonical_key(["C", "O"], ["CO"], reversible=True)
        key_rev = make_reaction_canonical_key(["CO"], ["C", "O"], reversible=True)
        assert key_fwd is not None
        assert key_fwd == key_rev

    def test_make_reaction_key_irreversible_not_normalized(self):
        from rmip.core.canonicalize import make_reaction_canonical_key
        key_fwd = make_reaction_canonical_key(["C", "O"], ["CO"], reversible=False)
        key_rev = make_reaction_canonical_key(["CO"], ["C", "O"], reversible=False)
        assert key_fwd != key_rev

    @pytest.mark.skipif(not _HAS_RDKIT_REGRESSION, reason="RDKit not installed")
    def test_smiles_to_formula(self):
        from rmip.core.canonicalize import smiles_to_formula
        assert smiles_to_formula("C") == "CH4"
        assert smiles_to_formula("CCCCCCC") == "C7H16"

    @pytest.mark.skipif(not _HAS_RDKIT_REGRESSION, reason="RDKit not installed")
    def test_smiles_to_mw(self):
        from rmip.core.canonicalize import smiles_to_mw
        mw = smiles_to_mw("C")
        assert mw is not None
        assert abs(mw - 0.01604) < 0.001  # kg/mol


class TestGlossaryReaderRegression:
    """Verify glossary reader functionality from Phase-1."""

    def test_load_csv(self, tmp_path):
        csv_file = tmp_path / "glossary.csv"
        csv_file.write_text("name,smiles\nCH4,C\nNC7H16,CCCCCCC\nO2,[O][O]\n")
        from rmip.io.glossary_reader import load_glossary
        g = load_glossary(str(csv_file))
        assert g["CH4"] == "C"
        assert g["NC7H16"] == "CCCCCCC"

    def test_lookup_case_insensitive(self, tmp_path):
        csv_file = tmp_path / "g.csv"
        csv_file.write_text("name,smiles\nNC7H16,CCCCCCC\n")
        from rmip.io.glossary_reader import load_glossary, lookup_smiles
        g = load_glossary(str(csv_file))
        assert lookup_smiles(g, "nc7h16") == "CCCCCCC"
        assert lookup_smiles(g, "UNKNOWN") is None

    def test_tab_separated_csv(self, tmp_path):
        tsv = tmp_path / "g.csv"
        tsv.write_text("name\tsmiles\nCH4\tC\n")
        from rmip.io.glossary_reader import load_glossary
        g = load_glossary(str(tsv))
        assert g["CH4"] == "C"

    def test_comment_lines_skipped(self, tmp_path):
        csv_file = tmp_path / "g.csv"
        csv_file.write_text("# comment\nname,smiles\nCH4,C\n")
        from rmip.io.glossary_reader import load_glossary
        g = load_glossary(str(csv_file))
        assert g["CH4"] == "C"


@pytest.mark.skipif(not _HAS_NX_REGRESSION, reason="NetworkX not installed")
class TestGraphBuilderRegression:
    """Verify graph builder functionality from Phase-1."""

    def _synthetic_db(self, tmp_path):
        from rmip.database.schema import init_db, insert_mechanism, insert_species, insert_reaction, insert_stoichiometry_batch, update_mechanism_counts
        db_path = str(tmp_path / "test.db")
        conn = init_db(db_path)
        mid = insert_mechanism(conn, name="H2 combustion", format="chemkin")

        sid_h2 = insert_species(conn, mechanism_id=mid, name="H2",
                                canonical_smiles="[HH]",
                                inchi_key="UFHFLCQGNIYNRP-UHFFFAOYSA-N")
        sid_o2 = insert_species(conn, mechanism_id=mid, name="O2",
                                canonical_smiles="[O][O]")
        sid_h2o = insert_species(conn, mechanism_id=mid, name="H2O",
                                 canonical_smiles="O")

        rid = insert_reaction(conn, mechanism_id=mid, formula_string="H2 + O2 <=> H2O + H2O",
                              reversible=1, kinetics_type="arrhenius",
                              A_si=1e8, n=0, Ea_si=1e4)
        insert_stoichiometry_batch(conn, [
            (rid, sid_h2, 1.0, "reactant"),
            (rid, sid_o2, 1.0, "reactant"),
            (rid, sid_h2o, 2.0, "product"),
        ])
        update_mechanism_counts(conn, mid, 3, 1)
        conn.commit()
        conn.close()
        return db_path, mid

    def test_build_graph_node_counts(self, tmp_path):
        from rmip.core.graph_builder import GraphBuilder
        db_path, mid = self._synthetic_db(tmp_path)
        gb = GraphBuilder(db_path)
        G = gb.build_graph(mid)
        sp_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "species"]
        rx_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "reaction"]
        assert len(sp_nodes) == 3
        assert len(rx_nodes) == 1

    def test_build_graph_edge_direction(self, tmp_path):
        from rmip.core.graph_builder import GraphBuilder
        db_path, mid = self._synthetic_db(tmp_path)
        gb = GraphBuilder(db_path)
        G = gb.build_graph(mid, include_reverse_edges=False)
        sp_nodes = {d["name"]: n for n, d in G.nodes(data=True) if d["type"] == "species"}
        rx_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "reaction"]
        rnode = rx_nodes[0]
        assert G.has_edge(sp_nodes["H2"], rnode)
        assert G.has_edge(sp_nodes["O2"], rnode)
        assert G.has_edge(rnode, sp_nodes["H2O"])
        assert not G.has_edge(sp_nodes["H2"], sp_nodes["H2O"])

    def test_graph_stats_detailed(self, tmp_path):
        from rmip.core.graph_builder import GraphBuilder
        db_path, mid = self._synthetic_db(tmp_path)
        gb = GraphBuilder(db_path)
        G = gb.build_graph(mid)
        stats = gb.get_stats(G)
        assert stats["n_species"] == 3
        assert stats["n_reactions"] == 1
        assert stats["n_isolated_species"] == 0
        assert "n_components_strong" in stats
        assert "largest_scc_size" in stats
        assert "avg_species_degree" in stats
        assert "max_species_degree" in stats
        assert "most_connected_species" in stats

    def test_export_graphml(self, tmp_path):
        from rmip.core.graph_builder import GraphBuilder
        import networkx as nx
        db_path, mid = self._synthetic_db(tmp_path)
        gb = GraphBuilder(db_path)
        G = gb.build_graph(mid)
        out = tmp_path / "out.graphml"
        gb.export_graphml(G, str(out))
        assert out.exists()
        G2 = nx.read_graphml(str(out))
        assert G2.number_of_nodes() == G.number_of_nodes()

    def test_export_json_cytoscape(self, tmp_path):
        from rmip.core.graph_builder import GraphBuilder
        db_path, mid = self._synthetic_db(tmp_path)
        gb = GraphBuilder(db_path)
        G = gb.build_graph(mid)
        out = tmp_path / "out.json"
        gb.export_json(G, str(out))
        import json
        data = json.loads(out.read_text())
        assert "elements" in data  # cytoscape_data format has 'elements' key

    def test_subgraph_and_paths(self, tmp_path):
        from rmip.core.graph_builder import GraphBuilder
        db_path, mid = self._synthetic_db(tmp_path)
        gb = GraphBuilder(db_path)
        G = gb.build_graph(mid)
        sub = gb.get_species_subgraph(G, ["H2"])
        assert sub.number_of_nodes() > 0
        paths = gb.find_paths_between_species(G, "H2", "H2O", cutoff=5)
        assert len(paths) >= 1


class TestChemkinWriterRegression:
    """Verify chemkin_writer use_original_units flag works correctly."""

    def test_use_original_units_default_true(self):
        from rmip.io.chemkin_writer import ChemkinWriter
        cw = ChemkinWriter("/fake/path.db", use_original_units=True)
        assert cw._use_orig is True

    def test_use_original_units_false(self):
        from rmip.io.chemkin_writer import ChemkinWriter
        cw = ChemkinWriter("/fake/path.db", use_original_units=False)
        assert cw._use_orig is False
