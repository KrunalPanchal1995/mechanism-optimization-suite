"""
rmip/tests/test_gsa_prs_crosscheck.py
======================================
Tests for rmip/simulation/gsa_prs_crosscheck.py (ZERORK_GSA_PLAN.md Phase 4
open question #4's validation cross-check: direct-simulation Sobol indices
vs. PRS-surrogate Sobol indices, for the same reactions).

Skip-marked consistently with test_zerork_gsa.py so the fast/no-toolchain
tests always run, and the full end-to-end test only runs where Zero-RK,
the SALib venv, mpiexec, the anaconda python3.9+cantera interpreter,
ck2yaml, and src/MUQ-SAC are all available.
"""
from __future__ import annotations

import shutil
import subprocess
import sqlite3
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

_ZERORK_INSTALL = "/home/krithika/zero-rk/build2/inst_dir"
_ZERORK_GSA_BIN = Path(_ZERORK_INSTALL) / "bin" / "perturbAFactorGSA_mpi.x"
_ZERORK_IDT_BIN = Path(_ZERORK_INSTALL) / "bin" / "constVolumeWSR.x"
_SALIB_PYTHON = Path("/home/krithika/.venvs/salib_gsa/bin/python3")
_MPIEXEC = Path("/usr/bin/mpiexec")
_PYTHON39 = "/home/krithika/anaconda3/bin/python3.9"
_CK2YAML = "/home/krithika/anaconda3/bin/ck2yaml"
_FIXTURE_XML = str(Path(__file__).parent / "fixtures" / "h2_o2_crosscheck_unsrt.xml")
_PROJECT_DB = Path(__file__).resolve().parents[3] / "outputs" / "project.db"


def _muqsac_importable() -> bool:
    try:
        import sys
        root = Path(__file__).resolve().parents[3]
        for p in (str(root / "src"), str(root / "src" / "MUQ-SAC")):
            if p not in sys.path:
                sys.path.insert(0, p)
        import Uncertainty  # noqa: F401
        import MechanismParser  # noqa: F401
        import MechManipulator2_0  # noqa: F401
        import DesignMatrix  # noqa: F401
        import ResponseSurface  # noqa: F401
        return True
    except Exception:
        return False


def _python39_has_cantera() -> bool:
    if not Path(_PYTHON39).exists():
        return False
    try:
        proc = subprocess.run(
            [_PYTHON39, "-c", "import cantera"], capture_output=True, timeout=30,
        )
        return proc.returncode == 0
    except Exception:
        return False


_needs_zerork = pytest.mark.skipif(
    not _ZERORK_GSA_BIN.exists(), reason="Zero-RK not installed at expected path",
)
_needs_zerork_idt = pytest.mark.skipif(
    not _ZERORK_IDT_BIN.exists(), reason="Zero-RK constVolumeWSR.x not installed",
)
_needs_salib_and_mpi = pytest.mark.skipif(
    not (_ZERORK_GSA_BIN.exists() and _SALIB_PYTHON.exists() and _MPIEXEC.exists()),
    reason="Zero-RK, the dedicated SALib venv, or mpiexec is missing",
)
_needs_muqsac = pytest.mark.skipif(
    not _muqsac_importable(), reason="src/MUQ-SAC modules not importable",
)
_needs_cantera_py39 = pytest.mark.skipif(
    not _python39_has_cantera(), reason="anaconda python3.9 + cantera not available",
)
_needs_ck2yaml = pytest.mark.skipif(
    shutil.which("ck2yaml") is None and not Path(_CK2YAML).exists(),
    reason="ck2yaml not on PATH",
)
_needs_project_db = pytest.mark.skipif(
    not _PROJECT_DB.exists(), reason="outputs/project.db not present",
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixture XML shape (regression guard against the blank-<class>/missing-no=
# bug found in the shipped DATASET/H2_O2_TEST fixture)
# ─────────────────────────────────────────────────────────────────────────────

class TestFixtureXML:
    def test_fixture_has_no_and_perturbation_type_and_nonblank_class(self):
        tree = ET.parse(_FIXTURE_XML)
        root = tree.getroot()
        reactions = list(root)
        assert len(reactions) == 3
        for r in reactions:
            assert r.attrib.get("no"), f"missing no= on {r.attrib.get('rxn')}"
            assert r.attrib.get("rxn"), "missing rxn="
            class_el = r.find("class")
            assert class_el is not None and class_el.text == "Elementary", (
                f"blank/missing <class> on {r.attrib.get('rxn')} would silently "
                "no-op MechManipulator2_0's perturbation dispatch"
            )
            pert_el = r.find("perturbation_type")
            assert pert_el is not None and pert_el.text == "factor"

    def test_fixture_temp_unsrt_arrays_same_length(self):
        tree = ET.parse(_FIXTURE_XML)
        for r in tree.getroot():
            temp = [float(x) for x in r.find("temp").text.split(",")]
            unsrt = [float(x) for x in r.find("unsrt").text.split(",")]
            assert len(temp) == len(unsrt) == 11


# ─────────────────────────────────────────────────────────────────────────────
# Design matrix shape (pure math, no toolchain -- needs MUQ-SAC importable
# only for DesignMatrix's own machinery)
# ─────────────────────────────────────────────────────────────────────────────

@_needs_muqsac
class TestDesignMatrixShape:
    def _fake_unsrt(self):
        """A minimal fake unsrt dict matching what DesignMatrix/Manipulator
        need: .activeParameters, .zeta.x, .data -- avoids needing a real
        mechanism YAML just to check the sampled matrix's shape/bounds."""
        import sys
        root = Path(__file__).resolve().parents[3]
        for p in (str(root / "src"), str(root / "src" / "MUQ-SAC")):
            if p not in sys.path:
                sys.path.insert(0, p)

        class _FakeZeta:
            x = 1.0

        class _FakeRxn:
            activeParameters = ["1_A"]
            zeta = _FakeZeta()
            data = {}

        return {"rxn1": _FakeRxn(), "rxn2": _FakeRxn(), "rxn3": _FakeRxn()}

    def test_build_design_matrix_shape_and_bounds(self):
        import rmip.simulation.gsa_prs_crosscheck as m

        unsrt = self._fake_unsrt()
        with tempfile.TemporaryDirectory() as td:
            X = m.build_design_matrix(unsrt, td, n_design=16, method="sobol", seed=42)
        assert X.shape[1] == 3
        assert X.shape[0] >= 16
        assert X.min() >= -1.0 - 1e-9
        assert X.max() <= 1.0 + 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Manipulator round-trip: perturbation math only, on a synthetic mechanism
# ─────────────────────────────────────────────────────────────────────────────

@_needs_muqsac
class TestManipulatorRoundTrip:
    def _synthetic_mech(self):
        return {
            "reactions": [
                {"equation": "A <=> B", "rate-constant": {"A": 1.0e13, "b": 0.0, "Ea": 20000.0}},
                {"equation": "C <=> D", "rate-constant": {"A": 2.0e12, "b": 0.5, "Ea": 15000.0}},
            ],
        }

    def _fake_unsrt(self, sigma_a=(0.15, 0.20)):
        class _FakeReaction:
            def __init__(self, index, sigma):
                self.index = index
                self.activeParameters = ["1_A"]
                self.nominal = np.array(
                    [np.log(1.0e13 if index == 0 else 2.0e12), 0.0, 1.0]
                )
                self.cholskyDeCorrelateMat = np.array([sigma])
                self.selection = [1.0, 0.0, 0.0]
                self.classification = "Elementary"

        return {
            "A <=> B": _FakeReaction(0, sigma_a[0]),
            "C <=> D": _FakeReaction(1, sigma_a[1]),
        }

    def test_only_targeted_reaction_changes(self):
        import sys
        root = Path(__file__).resolve().parents[3]
        for p in (str(root / "src"), str(root / "src" / "MUQ-SAC")):
            if p not in sys.path:
                sys.path.insert(0, p)
        from MechManipulator2_0 import Manipulator
        from copy import deepcopy

        mech = self._synthetic_mech()
        unsrt = self._fake_unsrt()
        zeta = np.array([1.0, 0.0])  # perturb only "A <=> B", at +1 sigma
        new_mech, _ = Manipulator(deepcopy(mech), unsrt, zeta).doPerturbation()

        a0_new = new_mech["reactions"][0]["rate-constant"]["A"]
        a1_new = new_mech["reactions"][1]["rate-constant"]["A"]

        assert a0_new == pytest.approx(1.0e13 * np.exp(0.15), rel=1e-9)
        assert a1_new == pytest.approx(2.0e12, rel=1e-9)  # unchanged


# ─────────────────────────────────────────────────────────────────────────────
# PRS + analytic Sobol glue, validated against a synthetic closed-form answer
# ─────────────────────────────────────────────────────────────────────────────

@_needs_muqsac
class TestPRSAnalyticSobol:
    def test_known_answer_linear_quadratic_function(self):
        """y = 3*z1 + 0.5*z2^2 (no z3 dependence, no interactions), z ~ U(-1,1).
        Closed-form (Sudret 2008): D_1 = 3^2/3 = 3.0, D_2 = 0.5^2 * 4/45,
        D_3 = 0, D = D_1+D_2. S1_1 = D_1/D (dominant), S1_2, S1_3 = D_3/D = 0."""
        import rmip.simulation.gsa_prs_crosscheck as m

        rng = np.random.default_rng(0)
        n = 400
        Z = rng.uniform(-1, 1, size=(n, 3))
        Y = 3.0 * Z[:, 0] + 0.5 * Z[:, 1] ** 2

        unsrt = {"rxn1": None, "rxn2": None, "rxn3": None}

        class _Fake:
            activeParameters = ["1_A"]

        unsrt = {"rxn1": _Fake(), "rxn2": _Fake(), "rxn3": _Fake()}

        with tempfile.TemporaryDirectory() as td:
            prs = m.fit_prs(list(Z), list(Y), td, criteria=100.0)
            sobol = m.compute_prs_sobol(prs, unsrt)

        D1 = 3.0 ** 2 / 3.0
        D2 = 0.5 ** 2 * 4.0 / 45.0
        D = D1 + D2
        expected_S1 = {"rxn1": D1 / D, "rxn2": D2 / D, "rxn3": 0.0}

        for rxn, expected in expected_S1.items():
            assert sobol[rxn]["S1"] == pytest.approx(expected, abs=0.02)


# ─────────────────────────────────────────────────────────────────────────────
# Regression pin: per-reaction-bounds composition vs. run_screening() itself
# ─────────────────────────────────────────────────────────────────────────────

@_needs_salib_and_mpi
@_needs_project_db
class TestPerReactionBoundsMatchesRunScreening:
    def test_equal_bounds_reproduces_run_screening_bit_for_bit(self):
        import rmip.simulation.gsa_prs_crosscheck as m
        from rmip.simulation.zerork_gsa import ZerorkGSARunner
        from rmip.types import GSARunConfig, SimCondition

        with tempfile.TemporaryDirectory() as td:
            db_copy = str(Path(td) / "p.db")
            shutil.copy(_PROJECT_DB, db_copy)

            with sqlite3.connect(db_copy) as conn:
                ids = [r[0] for r in conn.execute(
                    "SELECT id FROM reactions WHERE mechanism_id=8 ORDER BY id LIMIT 3"
                ).fetchall()]
            if len(ids) < 3:
                pytest.skip("mechanism_id=8 (GRI-3.0) not present")

            runner = ZerorkGSARunner(db_copy, _ZERORK_INSTALL)
            cond = SimCondition(T_K=1000.0, P_atm=10.0, phi=1.0, fuel="H2:1.0",
                                oxidizer="O2:1.0, N2:3.76")
            factor_bound = 1.5
            ln_bound = np.log(factor_bound)

            config = GSARunConfig(
                mechanism_id=8, method="sobol", n_samples=4, reaction_ids=ids,
                factor_bound=factor_bound, T_K=1000.0, P_atm=10.0, phi=1.0,
                fuel="H2:1.0", oxidizer="O2:1.0, N2:3.76", idt_temp_delta=400.0,
                n_mpi_ranks=4, seed=7,
            )
            expected = runner.run_screening(
                config, work_dir=str(Path(td) / "expected"), timeout=120, save_db=False,
            )

            sigma_A = {rid: ln_bound for rid in ids}
            actual = m.run_direct_sobol_per_reaction_bounds(
                runner, 8, ids, sigma_A, cond, n_base_samples=4,
                work_dir=str(Path(td) / "actual"), idt_temp_delta=400.0,
                seed=7, n_mpi_ranks=4,
            )

            assert actual.parameter_names == expected.parameter_names
            for name in expected.parameter_names:
                assert actual.S1[name] == pytest.approx(expected.S1[name], abs=1e-9)
                assert actual.ST[name] == pytest.approx(expected.ST[name], abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Full end-to-end: real GRI-3.0, fast settings -- pipeline-completion check,
# not a statistical-convergence check (see module docstring / RESUME.md)
# ─────────────────────────────────────────────────────────────────────────────

@_needs_zerork
@_needs_zerork_idt
@_needs_salib_and_mpi
@_needs_muqsac
@_needs_cantera_py39
@_needs_ck2yaml
@_needs_project_db
class TestFullCrossCheckEndToEnd:
    def test_run_crosscheck_completes_on_real_mechanism(self):
        import rmip.simulation.gsa_prs_crosscheck as m
        from rmip.types import SimCondition

        with sqlite3.connect(_PROJECT_DB) as conn:
            has_mech8 = conn.execute(
                "SELECT 1 FROM reactions WHERE mechanism_id=8 LIMIT 1"
            ).fetchone()
        if not has_mech8:
            pytest.skip("mechanism_id=8 (GRI-3.0) not present in outputs/project.db")

        cond = SimCondition(T_K=1000.0, P_atm=10.0, phi=1.0, fuel="H2:1.0",
                            oxidizer="O2:1.0, N2:3.76")
        with tempfile.TemporaryDirectory() as td:
            result = m.run_crosscheck(
                db_path=str(_PROJECT_DB),
                zerork_install_dir=_ZERORK_INSTALL,
                mechanism_id=8,
                xml_path=_FIXTURE_XML,
                condition=cond,
                work_dir=td,
                n_design=16, n_direct_base=32, seed=42, n_mpi_ranks=4,
            )
        assert result.mechanism_id == 8
        assert len(result.reaction_keys) == 3
        assert result.verdict in ("AGREE", "DISAGREE", "PRS_INVALID")
        print(f"\n[informational, not asserted -- fast settings, not "
              f"statistically converged] verdict={result.verdict}\n"
              f"{result.report_text()}")
