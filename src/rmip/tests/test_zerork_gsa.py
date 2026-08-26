"""
rmip/tests/test_zerork_gsa.py
==============================
Test suite for Zero-RK GSA (ZERORK_GSA_PLAN.md Phase 3).

Coverage
--------
  Types (GSARunConfig, SobolResult) — to_dict/from_dict/save/load round-trips,
    SobolResult.to_screening_dict()'s DoEFramework.run_screening()-shaped output.
  DB migration — run_gsa_migrations(), sensitivity_runs/sobol_indices/
    design_matrices tables, idempotency.
  ZerorkGSARunner pure-logic helpers (no external binary/venv needed):
    _write_gsa_matrix, _build_full_matrix, _parse_gsa_output.
  ZerorkGSARunner._all_reactions / _insert_run — needs Zero-RK installed
    (skipped otherwise) but not a live solver run.
  Full run_screening() round trip against the real perturbAFactorGSA_mpi.x
    binary + the dedicated SALib venv — skipped if either is absent.

Run with:
    python -m pytest src/rmip/tests/test_zerork_gsa.py -v
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

_ZERORK_INSTALL = "/home/krithika/zero-rk/build2/inst_dir"
_ZERORK_GSA_BIN = Path(_ZERORK_INSTALL) / "bin" / "perturbAFactorGSA_mpi.x"
_SALIB_PYTHON = Path("/home/krithika/.venvs/salib_gsa/bin/python3")
_MPIEXEC = Path("/usr/bin/mpiexec")

_needs_zerork = pytest.mark.skipif(
    not _ZERORK_GSA_BIN.exists(), reason="Zero-RK not installed at expected path",
)
_needs_salib_and_mpi = pytest.mark.skipif(
    not (_ZERORK_GSA_BIN.exists() and _SALIB_PYTHON.exists() and _MPIEXEC.exists()),
    reason="Zero-RK, the dedicated SALib venv, or mpiexec is missing",
)


# ─────────────────────────────────────────────────────────────────────────────
# Types: GSARunConfig / SobolResult
# ─────────────────────────────────────────────────────────────────────────────

class TestGSATypes:
    def test_gsa_run_config_round_trip(self):
        from rmip.types import GSARunConfig

        cfg = GSARunConfig(
            mechanism_id=8, method="sobol", n_samples=64, reaction_ids=[1, 2, 3],
            factor_bound=3.0, T_K=950.0, P_atm=5.0, phi=0.5, fuel="CH4:1.0",
            oxidizer="O2:1.0, N2:3.76", idt_temp_delta=500.0, n_mpi_ranks=8,
            seed=123, morris_num_levels=6,
        )
        d = cfg.to_dict()
        cfg2 = GSARunConfig.from_dict(d)
        assert cfg2.to_dict() == d

    def test_gsa_run_config_defaults(self):
        from rmip.types import GSARunConfig
        cfg = GSARunConfig(mechanism_id=1, method="morris", n_samples=32)
        assert cfg.reaction_ids is None
        assert cfg.factor_bound == 2.0
        assert cfg.n_mpi_ranks == 4

    def test_sobol_result_round_trip_sobol_method(self):
        from rmip.types import SobolResult
        r = SobolResult(
            mechanism_id=8, method="sobol", parameter_names=["R1", "R2"],
            n_samples=64, n_failed_samples=1,
            S1={"R1": 0.1, "R2": 0.2}, S1_conf={"R1": 0.01, "R2": 0.02},
            ST={"R1": 0.15, "R2": 0.25}, ST_conf={"R1": 0.02, "R2": 0.03},
            run_id=5,
        )
        d = r.to_dict()
        r2 = SobolResult.from_dict(d)
        assert r2.to_dict() == d
        assert r2.mu_star is None and r2.sigma is None

    def test_sobol_result_round_trip_morris_method(self):
        from rmip.types import SobolResult
        r = SobolResult(
            mechanism_id=8, method="morris", parameter_names=["R1"],
            n_samples=20, mu_star={"R1": 1.5}, mu_star_conf={"R1": 0.3},
            sigma={"R1": 0.7},
        )
        d = r.to_dict()
        assert SobolResult.from_dict(d).to_dict() == d
        assert r.S1 is None and r.ST is None

    def test_sobol_result_save_load(self):
        from rmip.types import SobolResult
        r = SobolResult(mechanism_id=8, method="sobol", parameter_names=["R1"],
                         n_samples=16, S1={"R1": 0.5}, S1_conf={"R1": 0.05},
                         ST={"R1": 0.6}, ST_conf={"R1": 0.06})
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "nested" / "result.json")
            r.save(path)
            loaded = SobolResult.load(path)
            assert loaded.to_dict() == r.to_dict()

    def test_to_screening_dict_matches_run_screening_contract(self):
        """DoEFramework.run_screening()'s documented return shape is
        {canonical_key: {"S1": float, "S1_conf": float, ...}} -- confirm
        SobolResult produces exactly that shape."""
        from rmip.types import SobolResult
        r = SobolResult(
            mechanism_id=8, method="sobol", parameter_names=["R1", "R2"],
            n_samples=64, S1={"R1": 0.1, "R2": 0.2}, S1_conf={"R1": 0.01, "R2": 0.02},
            ST={"R1": 0.15, "R2": 0.25}, ST_conf={"R1": 0.02, "R2": 0.03},
        )
        screening = r.to_screening_dict()
        assert set(screening.keys()) == {"R1", "R2"}
        assert screening["R1"] == {"S1": 0.1, "S1_conf": 0.01, "ST": 0.15, "ST_conf": 0.02}


# ─────────────────────────────────────────────────────────────────────────────
# DB migration
# ─────────────────────────────────────────────────────────────────────────────

class TestGSAMigrations:
    def test_tables_created(self):
        from rmip.database.schema import init_db
        from rmip.database.migrations import run_gsa_migrations, _table_exists

        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "p.db")
            conn = init_db(db_path)
            for t in ("sensitivity_runs", "sobol_indices", "design_matrices"):
                assert not _table_exists(conn, t)
            run_gsa_migrations(conn)
            for t in ("sensitivity_runs", "sobol_indices", "design_matrices"):
                assert _table_exists(conn, t)

    def test_idempotent(self):
        from rmip.database.schema import init_db
        from rmip.database.migrations import run_gsa_migrations

        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "p.db")
            conn = init_db(db_path)
            run_gsa_migrations(conn)
            run_gsa_migrations(conn)  # must not raise / must not duplicate schema


# ─────────────────────────────────────────────────────────────────────────────
# ZerorkGSARunner pure-logic helpers (no external binary/venv required)
# ─────────────────────────────────────────────────────────────────────────────

class TestGSAPureLogic:
    def test_write_gsa_matrix_format(self):
        """Zero-RK's gsaMatrixFile: header '{n_samples:6d} {n_reactions:6d}',
        then ONE value per line, row-major (confirmed against
        examples/perturbAFactorGSA/create_lognormal_mtx.py)."""
        from rmip.simulation.zerork_gsa import ZerorkGSARunner

        matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "gsa.mtx"
            ZerorkGSARunner._write_gsa_matrix(path, matrix, n_reactions=3)
            lines = path.read_text().splitlines()
        assert lines[0] == "     2      3"
        assert len(lines) == 1 + 2 * 3
        values = [float(l) for l in lines[1:]]
        assert values == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def test_build_full_matrix_expands_unselected_to_one(self):
        from rmip.simulation.zerork_gsa import ZerorkGSARunner
        import math

        all_ids = [10, 20, 30, 40]
        selected_ids = [20, 40]
        # two samples, log-multipliers for the 2 selected reactions
        X = [[math.log(2.0), math.log(0.5)], [0.0, math.log(3.0)]]
        full = ZerorkGSARunner._build_full_matrix(X, all_ids, selected_ids)
        assert len(full) == 2
        assert full[0] == [1.0, pytest.approx(2.0), 1.0, pytest.approx(0.5)]
        assert full[1] == [1.0, pytest.approx(1.0), 1.0, pytest.approx(3.0)]

    def test_parse_gsa_output_basic(self):
        """task_id=0 is perturbAFactorGSA_mpi.x's own unperturbed-baseline
        row (written before the sample loop, confirmed by reading
        applications/perturbAFactorGSA/perturbAFactor_mpi.cpp directly) --
        never one of the gsaMatrixFile's n_samples rows. Row j (0-indexed,
        matching the input matrix / SALib sample order) is written as
        task_id=j+1. This was a real, previously-undiscovered off-by-one
        bug (task_id read directly as the row index) found via
        ZERORK_GSA_PLAN.md Phase 4's cross-check against
        AnalyticalSobolPRS -- see gsa_prs_crosscheck.py."""
        from rmip.simulation.zerork_gsa import ZerorkGSARunner

        content = (
            "# comment line\n"
            "# number of reactions : 3\n"
            "        0    1.10e-02    9.9e-03\n"  # baseline -- must be ignored
            "        1    1.23e-02    5.5e-03\n"  # row 0
            "        2    1.30e-02    6.1e-03\n"  # row 1
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "gsa_output.dat"
            path.write_text(content)
            Y = ZerorkGSARunner._parse_gsa_output(path, n_samples=2)
        assert Y == [pytest.approx(5.5e-03), pytest.approx(6.1e-03)]

    def test_parse_gsa_output_failed_sample_becomes_nan(self):
        """Zero-RK writes `inf` (not 0.0) for a non-ignition sample -- see
        B11 in review.md. A bare `idt_s > 0` check lets `inf` through as a
        seemingly-valid value, poisoning mean-imputation and every
        downstream Sobol/Morris index; the parser must treat non-finite
        values as failures too."""
        from rmip.simulation.zerork_gsa import ZerorkGSARunner

        content = (
            "# comment\n"
            "        0    1.10e-02    9.9e-03\n"  # baseline -- must be ignored
            "        1    1.23e-02    inf\n"       # row 0: non-ignition -> NaN
            "        2    1.30e-02    6.1e-03\n"   # row 1
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "gsa_output.dat"
            path.write_text(content)
            Y = ZerorkGSARunner._parse_gsa_output(path, n_samples=2)
        assert Y[0] != Y[0]  # NaN
        assert Y[1] == pytest.approx(6.1e-03)

    def test_parse_gsa_output_missing_row_becomes_nan(self):
        """A task_id absent from the output entirely (e.g. MPI worker crash)
        must also map to NaN, not KeyError."""
        from rmip.simulation.zerork_gsa import ZerorkGSARunner

        content = (
            "# comment\n"
            "        0    1.10e-02    9.9e-03\n"  # baseline -- must be ignored
            "        1    1.23e-02    5.5e-03\n"  # row 0
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "gsa_output.dat"
            path.write_text(content)
            Y = ZerorkGSARunner._parse_gsa_output(path, n_samples=3)
        assert Y[0] == pytest.approx(5.5e-03)
        assert Y[1] != Y[1]
        assert Y[2] != Y[2]

    def test_parse_gsa_output_ignores_baseline_even_if_only_row(self):
        """Regression pin for the off-by-one fix itself: if task_id=0 (the
        baseline) were mistaken for row 0's result, this would return
        9.9e-03 instead of NaN."""
        from rmip.simulation.zerork_gsa import ZerorkGSARunner

        content = "# comment\n        0    1.10e-02    9.9e-03\n"
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "gsa_output.dat"
            path.write_text(content)
            Y = ZerorkGSARunner._parse_gsa_output(path, n_samples=1)
        assert Y[0] != Y[0]  # NaN -- row 0 (task_id=1) was never written


# ─────────────────────────────────────────────────────────────────────────────
# ZerorkGSARunner instance methods needing a real Zero-RK install
# ─────────────────────────────────────────────────────────────────────────────

def _make_db_with_reactions(db_path: str, n_reactions: int = 5) -> int:
    from rmip.database.schema import init_db, insert_mechanism, insert_reaction

    conn = init_db(db_path)
    mech_id = insert_mechanism(conn, name="fixture", format="chemkin",
                                filepath="/fake.inp", n_species=3, n_reactions=n_reactions)
    for i in range(n_reactions):
        conn.execute(
            "INSERT INTO reactions (mechanism_id, formula_string, reversible, "
            "A_si, n, Ea_si, kinetics_type, pressure_dependent, has_third_body, "
            "reaction_label) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (mech_id, f"A{i} <=> B{i}", 1, 1e10, 0.0, 5000.0, "arrhenius", 0, 0, f"R{i+1}"),
        )
    conn.commit()
    return mech_id


@_needs_zerork
class TestGSARunnerInstance:
    def test_all_reactions_ordered_by_id(self):
        from rmip.simulation.zerork_gsa import ZerorkGSARunner

        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "p.db")
            mech_id = _make_db_with_reactions(db_path, n_reactions=4)
            runner = ZerorkGSARunner(db_path, _ZERORK_INSTALL)
            reactions = runner._all_reactions(mech_id)
            assert [name for _, name in reactions] == [
                "R1:A0 <=> B0", "R2:A1 <=> B1", "R3:A2 <=> B2", "R4:A3 <=> B3",
            ]

    def test_insert_run_persists_rows(self):
        from rmip.simulation.zerork_gsa import ZerorkGSARunner
        from rmip.types import GSARunConfig, SobolResult

        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "p.db")
            mech_id = _make_db_with_reactions(db_path, n_reactions=2)
            runner = ZerorkGSARunner(db_path, _ZERORK_INSTALL)

            config = GSARunConfig(mechanism_id=mech_id, method="sobol", n_samples=8,
                                   T_K=1000.0, P_atm=1.0, phi=1.0, fuel="H2:1.0",
                                   oxidizer="O2:1.0, N2:3.76")
            result = SobolResult(
                mechanism_id=mech_id, method="sobol",
                parameter_names=["R1:A0 <=> B0", "R2:A1 <=> B1"], n_samples=8,
                S1={"R1:A0 <=> B0": 0.3, "R2:A1 <=> B1": 0.1},
                S1_conf={"R1:A0 <=> B0": 0.03, "R2:A1 <=> B1": 0.01},
                ST={"R1:A0 <=> B0": 0.4, "R2:A1 <=> B1": 0.2},
                ST_conf={"R1:A0 <=> B0": 0.04, "R2:A1 <=> B1": 0.02},
            )
            run_id = runner._insert_run(config, Path(td) / "matrix.mtx", result)
            assert run_id is not None

            with sqlite3.connect(db_path) as conn:
                run_row = conn.execute(
                    "SELECT mechanism_id, method, n_samples FROM sensitivity_runs WHERE id=?",
                    (run_id,),
                ).fetchone()
                assert run_row == (mech_id, "sobol", 8)

                idx_rows = conn.execute(
                    "SELECT parameter_name, S1, ST FROM sobol_indices WHERE run_id=? "
                    "ORDER BY parameter_name", (run_id,),
                ).fetchall()
                assert idx_rows == [
                    ("R1:A0 <=> B0", 0.3, 0.4),
                    ("R2:A1 <=> B1", 0.1, 0.2),
                ]

                dm_row = conn.execute(
                    "SELECT method, n_samples, n_parameters FROM design_matrices WHERE run_id=?",
                    (run_id,),
                ).fetchone()
                assert dm_row == ("sobol", 8, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Full end-to-end: SALib sampling + real Zero-RK MPI batch execution +
# SALib analysis, on a real mechanism already in outputs/project.db.
# ─────────────────────────────────────────────────────────────────────────────

@_needs_salib_and_mpi
class TestGSAEndToEnd:
    def test_run_screening_sobol_real_binary(self):
        import shutil
        from rmip.simulation.zerork_gsa import ZerorkGSARunner
        from rmip.types import GSARunConfig

        project_db = Path(__file__).resolve().parents[3] / "outputs" / "project.db"
        if not project_db.exists():
            pytest.skip("outputs/project.db not present")

        with tempfile.TemporaryDirectory() as td:
            db_copy = str(Path(td) / "p.db")
            shutil.copy(project_db, db_copy)

            with sqlite3.connect(db_copy) as conn:
                ids = [r[0] for r in conn.execute(
                    "SELECT id FROM reactions WHERE mechanism_id=8 ORDER BY id LIMIT 3"
                ).fetchall()]
            if len(ids) < 3:
                pytest.skip("mechanism_id=8 (GRI-3.0) not present in outputs/project.db")

            runner = ZerorkGSARunner(db_copy, _ZERORK_INSTALL)
            config = GSARunConfig(
                mechanism_id=8, method="sobol", n_samples=4, reaction_ids=ids,
                T_K=1000.0, P_atm=10.0, phi=1.0, fuel="H2:1.0",
                oxidizer="O2:1.0, N2:3.76", idt_temp_delta=400.0,
                n_mpi_ranks=4, seed=1,
            )
            result = runner.run_screening(config, work_dir=str(Path(td) / "work"), timeout=120)

            assert result.n_samples == 4 * (3 + 2)  # Saltelli first-order-only: N*(D+2)
            assert result.method == "sobol"
            assert set(result.parameter_names) == set(result.S1.keys())
            assert result.run_id is not None

    def test_run_screening_relative_work_dir_and_split_therm_file(self):
        """Regression test for the 2026-07-23 GUI stress-sweep finding: every
        real `rmip_suite_gui` GSA-tab invocation passes a *relative*
        `work_dir` (the GUI never sets --work-dir, and the CLI's own default
        is relative too), and `run_screening` used to hand that straight to
        `subprocess.run(..., cwd=str(work))` while also embedding the same
        relative `work` in every `gsa_input.yml` path value -- doubling the
        path from the child's perspective and crashing Zero-RK's parser on
        every real invocation. Separately, passing `mechFile == thermFile`
        (both pointed at the single self-contained ChemkinWriter output)
        also SIGABRTed the parser independently.

        Reproduces both preconditions for real: chdir's into a scratch
        directory and calls `run_screening(work_dir="gsa_work")` -- a bare
        relative name, matching the GUI's actual usage -- then asserts (a)
        the run completes against the real binary and (b) the written
        `gsa_input.yml` has `mechFile` != `thermFile`, pointing at two
        distinct on-disk files.
        """
        import shutil

        import yaml as _yaml

        from rmip.simulation.zerork_gsa import ZerorkGSARunner
        from rmip.types import GSARunConfig

        project_db = Path(__file__).resolve().parents[3] / "outputs" / "project.db"
        if not project_db.exists():
            pytest.skip("outputs/project.db not present")

        with tempfile.TemporaryDirectory() as td:
            db_copy = str(Path(td) / "p.db")
            shutil.copy(project_db, db_copy)

            with sqlite3.connect(db_copy) as conn:
                ids = [r[0] for r in conn.execute(
                    "SELECT id FROM reactions WHERE mechanism_id=8 ORDER BY id LIMIT 3"
                ).fetchall()]
            if len(ids) < 3:
                pytest.skip("mechanism_id=8 (GRI-3.0) not present in outputs/project.db")

            runner = ZerorkGSARunner(db_copy, _ZERORK_INSTALL)
            config = GSARunConfig(
                mechanism_id=8, method="sobol", n_samples=4, reaction_ids=ids,
                T_K=1000.0, P_atm=10.0, phi=1.0, fuel="H2:1.0",
                oxidizer="O2:1.0, N2:3.76", idt_temp_delta=400.0,
                n_mpi_ranks=4, seed=1,
            )

            old_cwd = os.getcwd()
            os.chdir(td)
            try:
                # Bare relative name, exactly how rmip_suite_gui's GSA tab /
                # the CLI's own default invoke this (never an absolute path).
                result = runner.run_screening(config, work_dir="gsa_work", timeout=120)
            finally:
                os.chdir(old_cwd)

            assert result.n_samples == 4 * (3 + 2)
            assert result.run_id is not None

            gsa_yml_path = Path(td) / "gsa_work" / "gsa_input.yml"
            assert gsa_yml_path.exists()
            with open(gsa_yml_path) as fh:
                written_cfg = _yaml.safe_load(fh)
            assert written_cfg["mechFile"] != written_cfg["thermFile"]
            assert Path(written_cfg["mechFile"]).exists()
            assert Path(written_cfg["thermFile"]).exists()
            # Both absolute -- the doubling bug only manifests with relative
            # values embedded alongside a relative subprocess cwd.
            assert Path(written_cfg["mechFile"]).is_absolute()
            assert Path(written_cfg["thermFile"]).is_absolute()
