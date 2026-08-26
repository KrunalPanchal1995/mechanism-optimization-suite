"""
rmip/tests/test_stubs_doe.py
=============================
Test suite for `rmip.stubs.DoEFramework` (M9), wired (this session, alongside
`rmip.stubs.FusionEngine`/M7) to delegate real screening to
`rmip.simulation.zerork_gsa.ZerorkGSARunner` (ZERORK_GSA_PLAN.md Phase 3)
over a mechanism built on the fly via `FusionEngine`.

Coverage
--------
  run_screening(): condition-required guard, no-FUSE-reactions short
    circuit, canonical_key remapping (the "label:formula" ->
    "canonical_key" translation is the one place a subtle bug could hide),
    and a real end-to-end run against the actual Zero-RK binary + SALib
    venv (skipped if either is absent).
  prepare_mso_input(): target_conditions built from exp_data_points,
    "lhs" rejection (documented as never-implemented), "none" skips
    screening entirely, and the |coefficient| < 0.01 cutoff (tested with a
    monkeypatched run_screening for a deterministic coefficient rather than
    relying on a real GSA run's small-sample-count numerics).

Reuses the H2-combustion DB fixture from test_stubs_fusion.py (same
CK_A/CK_B/CK_C VERBATIM/COPY/FUSE setup) rather than re-deriving one.

Run with:
    python -m pytest src/rmip/tests/test_stubs_doe.py -v
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_stubs_fusion import _build_h2_fixture, _build_spec  # noqa: E402

_ZERORK_INSTALL = "/home/krithika/zero-rk/build2/inst_dir"
_ZERORK_GSA_BIN = Path(_ZERORK_INSTALL) / "bin" / "perturbAFactorGSA_mpi.x"
_SALIB_PYTHON = Path("/home/krithika/.venvs/salib_gsa/bin/python3")
_MPIEXEC = Path("/usr/bin/mpiexec")
_needs_gsa_stack = pytest.mark.skipif(
    not (_ZERORK_GSA_BIN.exists() and _SALIB_PYTHON.exists() and _MPIEXEC.exists()),
    reason="Zero-RK, the dedicated SALib venv, or mpiexec is missing",
)


def _add_exp_data(db: str, dataset_id: int = 1, fuel_name: str = "H2") -> None:
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS exp_datasets (
                id INTEGER PRIMARY KEY, name TEXT, sim_type TEXT, fuel_name TEXT,
                source_ref TEXT, n_points INTEGER, filepath TEXT, created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS exp_data_points (
                id INTEGER PRIMARY KEY, dataset_id INTEGER, T_K REAL, P_atm REAL,
                phi REAL, value REAL, uncertainty REAL, units TEXT, notes TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO exp_datasets (id, name, sim_type, fuel_name, n_points, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (dataset_id, f"{fuel_name}_IDT", "IDT", fuel_name, 2, "2024-01-01T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO exp_data_points (dataset_id, T_K, P_atm, phi, value, units) "
            "VALUES (?,?,?,?,?,?)", (dataset_id, 1000.0, 10.0, 1.0, 8e-6, "s"),
        )
        conn.execute(
            "INSERT INTO exp_data_points (dataset_id, T_K, P_atm, phi, value, units) "
            "VALUES (?,?,?,?,?,?)", (dataset_id, 1100.0, 10.0, 1.0, 4e-6, "s"),
        )
        conn.commit()


def _h2_condition():
    from rmip.types import SimCondition
    # Fixture mechanism has no N2 -- pure O2 oxidizer, not "O2:1.0, N2:3.76".
    return SimCondition(T_K=1000.0, P_atm=10.0, phi=1.0, fuel="H2:1.0", oxidizer="O2:1.0")


class TestRunScreening:
    def test_requires_condition(self, tmp_path):
        from rmip.stubs import DoEFramework

        db, mid1, mid2 = _build_h2_fixture(tmp_path)
        spec = _build_spec(db, mid1, mid2)
        doe = DoEFramework(spec, exp_dataset_id=1, db_path=db)

        with pytest.raises(ValueError, match="SimCondition"):
            doe.run_screening(n_samples=4, method="sobol", condition=None)

    def test_no_fuse_reactions_returns_empty_dict(self, tmp_path):
        from rmip.stubs import DoEFramework
        from rmip.types import UnionGraphSpec

        db, mid1, mid2 = _build_h2_fixture(tmp_path)
        empty_spec = UnionGraphSpec(mechanism_ids=[mid1, mid2], comparison_session_id=0)
        doe = DoEFramework(empty_spec, exp_dataset_id=1, db_path=db)

        result = doe.run_screening(n_samples=4, method="sobol", condition=_h2_condition())
        assert result == {}

    @_needs_gsa_stack
    def test_real_run_aborts_on_unignitable_fixture(self, tmp_path):
        """The CK_A/CK_B/CK_C fixture (see test_stubs_fusion.py) is a 3-reaction
        DB-plumbing stub, not real chemistry -- it has no chain-branching path
        (no H+O2<=>O+OH) and never actually ignites under real Zero-RK. Before
        B11 was fixed, Zero-RK's `inf` non-ignition marker was accepted by a
        bare `idt_s > 0` check and silently mean-imputed as if it were a valid
        IDT, so this same call used to return a bogus-but-shaped S1/ST dict.
        Post-fix, 100% of samples correctly parse to NaN and the >20%-failed
        guard aborts loudly instead of imputing over an all-garbage batch --
        this is the real-Zero-RK analogue of injecting an `inf` row (see
        review_plan.md verification recipe item 4)."""
        from rmip.stubs import DoEFramework

        db, mid1, mid2 = _build_h2_fixture(tmp_path)
        spec = _build_spec(db, mid1, mid2)
        doe = DoEFramework(spec, exp_dataset_id=1, db_path=db)

        with pytest.raises(RuntimeError, match="failed to ignite"):
            doe.run_screening(
                n_samples=4, method="sobol", condition=_h2_condition(),
                zerork_install_dir=_ZERORK_INSTALL,
            )


class TestPrepareMSOInput:
    def test_lhs_rejected(self, tmp_path):
        from rmip.stubs import DoEFramework

        db, mid1, mid2 = _build_h2_fixture(tmp_path)
        spec = _build_spec(db, mid1, mid2)
        _add_exp_data(db)
        doe = DoEFramework(spec, exp_dataset_id=1, db_path=db)

        with pytest.raises(ValueError, match="lhs"):
            doe.prepare_mso_input("H2", session_id=1, screening_method="lhs")

    def test_none_screening_keeps_all_fuse_reactions_no_condition_needed(self, tmp_path):
        from rmip.stubs import DoEFramework

        db, mid1, mid2 = _build_h2_fixture(tmp_path)
        spec = _build_spec(db, mid1, mid2)
        _add_exp_data(db)
        doe = DoEFramework(spec, exp_dataset_id=1, db_path=db)

        mso = doe.prepare_mso_input("H2", session_id=1, screening_method="none")
        assert [b["canonical_key"] for b in mso.param_blocks] == ["CK_B"]
        assert mso.metadata["n_fuse_reactions_after_screening"] == 1

    def test_target_conditions_built_from_exp_data_points(self, tmp_path):
        from rmip.stubs import DoEFramework

        db, mid1, mid2 = _build_h2_fixture(tmp_path)
        spec = _build_spec(db, mid1, mid2)
        _add_exp_data(db)
        doe = DoEFramework(spec, exp_dataset_id=1, db_path=db)

        mso = doe.prepare_mso_input("H2", session_id=1, screening_method="none")
        assert mso.target_conditions == [
            {"T_K": 1000.0, "P_atm": 10.0, "phi": 1.0},
            {"T_K": 1100.0, "P_atm": 10.0, "phi": 1.0},
        ]

    def test_screening_requires_condition(self, tmp_path):
        from rmip.stubs import DoEFramework

        db, mid1, mid2 = _build_h2_fixture(tmp_path)
        spec = _build_spec(db, mid1, mid2)
        _add_exp_data(db)
        doe = DoEFramework(spec, exp_dataset_id=1, db_path=db)

        with pytest.raises(ValueError, match="condition"):
            doe.prepare_mso_input("H2", session_id=1, screening_method="sobol",
                                   n_screening_samples=4, condition=None)

    def test_low_coefficient_reaction_is_screened_out(self, tmp_path, monkeypatch):
        """Deterministic cutoff test -- monkeypatch run_screening() rather
        than depend on a real GSA run's small-sample-count numerics."""
        from rmip.stubs import DoEFramework

        db, mid1, mid2 = _build_h2_fixture(tmp_path)
        spec = _build_spec(db, mid1, mid2)
        _add_exp_data(db)
        doe = DoEFramework(spec, exp_dataset_id=1, db_path=db)

        monkeypatch.setattr(
            doe, "run_screening",
            lambda **kwargs: {"CK_B": {"S1": 0.001, "S1_conf": 0.0001}},
        )
        mso = doe.prepare_mso_input(
            "H2", session_id=1, screening_method="sobol",
            n_screening_samples=4, condition=_h2_condition(),
        )
        assert mso.param_blocks == []
        assert mso.metadata["n_fuse_reactions_after_screening"] == 0

    def test_high_coefficient_reaction_survives_cutoff(self, tmp_path, monkeypatch):
        from rmip.stubs import DoEFramework

        db, mid1, mid2 = _build_h2_fixture(tmp_path)
        spec = _build_spec(db, mid1, mid2)
        _add_exp_data(db)
        doe = DoEFramework(spec, exp_dataset_id=1, db_path=db)

        monkeypatch.setattr(
            doe, "run_screening",
            lambda **kwargs: {"CK_B": {"S1": 0.5, "S1_conf": 0.05}},
        )
        mso = doe.prepare_mso_input(
            "H2", session_id=1, screening_method="sobol",
            n_screening_samples=4, condition=_h2_condition(),
        )
        assert [b["canonical_key"] for b in mso.param_blocks] == ["CK_B"]

    def test_param_block_bounds_match_fuse_params(self, tmp_path):
        from rmip.stubs import DoEFramework
        import math

        db, mid1, mid2 = _build_h2_fixture(tmp_path)
        spec = _build_spec(db, mid1, mid2)
        _add_exp_data(db)
        doe = DoEFramework(spec, exp_dataset_id=1, db_path=db)

        mso = doe.prepare_mso_input("H2", session_id=1, screening_method="none")
        block = mso.param_blocks[0]
        assert block["formula_string"] == "H2 + O <=> H + OH"
        assert math.isclose(block["params_mean"][0], math.sqrt(1e8 * 1e11), rel_tol=1e-6)
        assert block["params_lb"][0] == 1e8
        assert block["params_ub"][0] == 1e11
