"""
rmip/tests/test_surrogate.py
============================
Tests for the gPC / active-subspace surrogate (Future.md pt. 7):
``rmip/analysis/surrogate.py`` + the chaospy venv backend
``rmip/simulation/_chaospy_backend.py``.

The backend runs in a dedicated venv (``/home/krithika/.venvs/chaospy_gpc``);
tests that need it are skipped when the venv is absent, so the suite still
passes on a machine without it. The pure-Python dataclass / migration tests
always run.

Run with:
    python -m pytest src/rmip/tests/test_surrogate.py -v
"""
from __future__ import annotations

import math
import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

from rmip.analysis.surrogate import (
    _CHAOSPY_PYTHON_DEFAULT,
    PCESurrogateBuilder,
)
from rmip.types import SimCondition, SurrogateResult

_VENV = _CHAOSPY_PYTHON_DEFAULT
_HAS_VENV = os.path.exists(_VENV)
venv_required = pytest.mark.skipif(
    not _HAS_VENV, reason=f"chaospy venv not present at {_VENV}")


# ---------------------------------------------------------------------------
# SurrogateResult dataclass (no venv, no cantera)
# ---------------------------------------------------------------------------

class TestSurrogateResult:
    def _make(self) -> SurrogateResult:
        return SurrogateResult(
            mechanism_id=1, qoi="idt", order=3,
            parameter_names=["A <=> B", "C <=> D"],
            n_samples=100, n_failed_samples=2, n_terms=10, r2_insample=0.97,
            qoi_mean=-3.0, qoi_variance=0.25,
            sobol_main={"A <=> B": 0.8, "C <=> D": 0.15},
            sobol_total={"A <=> B": 0.85, "C <=> D": 0.2},
            as_eigenvalues=[1.2, 0.05],
            as_eigenvectors=[[0.98, -0.2], [0.2, 0.98]],
            as_activity_scores={"A <=> B": 1.1, "C <=> D": 0.1},
            as_n_active=1,
            condition=SimCondition(T_K=1000, P_atm=1, phi=1.0, fuel="H2:1"),
        )

    def test_ranked_parameters_sorted_by_total(self):
        r = self._make()
        ranked = r.ranked_parameters()
        assert ranked[0][0] == "A <=> B"
        assert ranked[0][1] >= ranked[1][1]

    def test_to_dict_serialisable(self):
        import json
        d = self._make().to_dict()
        assert d["qoi"] == "idt"
        assert d["as_n_active"] == 1
        assert d["condition"]["T_K"] == 1000
        assert json.dumps(d)

    def test_save_roundtrip(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "r.json"
            self._make().save(str(p))
            d = json.loads(p.read_text())
            assert d["parameter_names"] == ["A <=> B", "C <=> D"]


# ---------------------------------------------------------------------------
# Distribution helper (no venv)
# ---------------------------------------------------------------------------

class TestDistributions:
    def test_uniform_logA(self):
        dists = PCESurrogateBuilder.uniform_logA_distributions(3, 2.0)
        assert len(dists) == 3
        assert dists[0]["type"] == "uniform"
        assert dists[0]["upper"] == pytest.approx(math.log(2.0))
        assert dists[0]["lower"] == pytest.approx(-math.log(2.0))


# ---------------------------------------------------------------------------
# Migration (no venv)
# ---------------------------------------------------------------------------

class TestSurrogateMigration:
    def test_creates_table_idempotent(self):
        from rmip.database.schema import init_db
        from rmip.database.migrations import run_surrogate_migrations, _table_exists
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "m.db")
            conn = init_db(db)
            assert not _table_exists(conn, "surrogate_runs")
            run_surrogate_migrations(conn)
            assert _table_exists(conn, "surrogate_runs")
            run_surrogate_migrations(conn)   # idempotent
            assert _table_exists(conn, "surrogate_runs")
            conn.close()


# ---------------------------------------------------------------------------
# Backend via builder — pure math (venv required, no cantera)
# ---------------------------------------------------------------------------

@venv_required
class TestBuilderMath:
    def test_linear_sobol_and_active_subspace(self):
        """
        f = 3*x0 + 1*x1 over i.i.d. uniform marginals.
        Analytic: Var = (9+1)*var → S1 = [0.9, 0.1], ST == S1 (no interaction),
        active subspace 1-D with dominant direction ∝ [3,1]/sqrt(10)=[0.949,0.316].
        """
        builder = PCESurrogateBuilder(chaospy_python=_VENV)
        dists = builder.uniform_logA_distributions(2, math.e)   # [-1, 1]
        names = ["x0", "x1"]

        def qoi(X):
            return [3.0 * row[0] + 1.0 * row[1] for row in X]

        with tempfile.TemporaryDirectory() as tmp:
            res = builder.build(dists, names, qoi, n_samples=200, order=2,
                                work_dir=tmp, seed=1, log_transformed=False)

        assert res.r2_insample > 0.999
        assert res.sobol_main["x0"] == pytest.approx(0.9, abs=0.05)
        assert res.sobol_main["x1"] == pytest.approx(0.1, abs=0.05)
        # total ≈ main for a purely additive function
        assert res.sobol_total["x0"] == pytest.approx(res.sobol_main["x0"], abs=0.05)
        # active subspace: one dominant direction
        assert res.as_n_active == 1
        assert res.as_eigenvalues[0] > 10 * max(res.as_eigenvalues[1], 1e-12)
        v0 = res.as_eigenvectors  # columns are directions; row i = param i
        dom = [abs(v0[0][0]), abs(v0[1][0])]
        assert dom[0] == pytest.approx(0.949, abs=0.05)
        assert dom[1] == pytest.approx(0.316, abs=0.05)
        # activity score ranks x0 above x1
        assert res.as_activity_scores["x0"] > res.as_activity_scores["x1"]

    def test_inert_parameter_has_zero_sobol(self):
        builder = PCESurrogateBuilder(chaospy_python=_VENV)
        dists = builder.uniform_logA_distributions(3, math.e)
        names = ["a", "b", "inert"]

        def qoi(X):
            return [row[0] + 0.5 * row[1] for row in X]     # 3rd var unused

        with tempfile.TemporaryDirectory() as tmp:
            res = builder.build(dists, names, qoi, n_samples=150, order=2,
                                work_dir=tmp, seed=3, log_transformed=False)
        assert res.sobol_total["inert"] == pytest.approx(0.0, abs=0.02)
        assert res.as_activity_scores["inert"] < res.as_activity_scores["a"]

    def test_nan_samples_dropped(self):
        builder = PCESurrogateBuilder(chaospy_python=_VENV)
        dists = builder.uniform_logA_distributions(2, math.e)

        calls = {"n": 0}

        def qoi(X):
            # fail the first 10 % of samples
            out = []
            for k, row in enumerate(X):
                calls["n"] += 1
                out.append(float("nan") if k < len(X) // 10
                           else row[0] + row[1])
            return out

        with tempfile.TemporaryDirectory() as tmp:
            res = builder.build(dists, ["x0", "x1"], qoi, n_samples=100,
                                order=2, work_dir=tmp, seed=5,
                                log_transformed=False)
        assert res.n_failed_samples > 0
        assert res.n_samples == 100 - res.n_failed_samples

    def test_too_many_failures_aborts(self):
        builder = PCESurrogateBuilder(chaospy_python=_VENV)
        dists = builder.uniform_logA_distributions(2, math.e)

        def qoi(X):
            return [float("nan")] * len(X)

        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(RuntimeError):
                builder.build(dists, ["x0", "x1"], qoi, n_samples=50, order=2,
                              work_dir=tmp, seed=7, log_transformed=False)


# ---------------------------------------------------------------------------
# Cantera IDT QoI (cantera required, no venv)
# ---------------------------------------------------------------------------

class TestCanteraIDTQoI:
    def test_multiplier_changes_idt(self):
        ct = pytest.importorskip("cantera")
        from rmip.analysis.surrogate import CanteraIDTMultiplierQoI

        gas = ct.Solution("h2o2.yaml")
        # reaction 10 = "H + O2 <=> O + OH", the master chain-branching step
        # (hugely sensitive for H2 ignition); 20 and 2 are also active.
        qoi = CanteraIDTMultiplierQoI(
            gas, reaction_indices=[10, 20, 2],
            T_K=1200.0, P_atm=1.0, phi=1.0,
            fuel="H2:1.0", oxidizer="O2:1.0, N2:3.76",
            t_end_s=1.0,
        )
        names = qoi.parameter_names()
        assert len(names) == 3 and all("=" in n or ">" in n for n in names)

        # baseline (all multipliers 1 → log-mult 0) vs a perturbation
        base = qoi([[0.0, 0.0, 0.0]])
        assert math.isfinite(base[0])          # H2 ignites at 1200 K, 1 atm
        # speeding up chain-branching 3x must shorten the IDT
        pert = qoi([[math.log(3.0), 0.0, 0.0]])
        assert math.isfinite(pert[0])
        assert pert[0] < base[0] - 1e-3        # log10(IDT) drops
        # and the multipliers were reset afterwards (no lingering state)
        base2 = qoi([[0.0, 0.0, 0.0]])
        assert base2[0] == pytest.approx(base[0], abs=1e-9)

    def test_bad_reaction_index_raises(self):
        ct = pytest.importorskip("cantera")
        from rmip.analysis.surrogate import CanteraIDTMultiplierQoI
        gas = ct.Solution("h2o2.yaml")
        with pytest.raises(ValueError):
            CanteraIDTMultiplierQoI(gas, [99999], 1200.0, 1.0, 1.0, "H2:1.0")


# ---------------------------------------------------------------------------
# Full stack: real IDT surrogate on h2o2 (venv + cantera) — small & slow-ish
# ---------------------------------------------------------------------------

@venv_required
class TestFullStackIDT:
    def test_h2o2_idt_surrogate(self):
        ct = pytest.importorskip("cantera")
        from rmip.analysis.surrogate import CanteraIDTMultiplierQoI

        gas = ct.Solution("h2o2.yaml")
        # sensitive H2/O2 reactions: chain-branch (10), H2+OH (20), H2+O (2),
        # H+HO2->2OH (17)
        qoi = CanteraIDTMultiplierQoI(
            gas, reaction_indices=[10, 20, 2, 17],
            T_K=1200.0, P_atm=1.0, phi=1.0,
            fuel="H2:1.0", oxidizer="O2:1.0, N2:3.76", t_end_s=1.0,
        )
        builder = PCESurrogateBuilder(chaospy_python=_VENV)
        dists = builder.uniform_logA_distributions(4, 2.0)

        with tempfile.TemporaryDirectory() as tmp:
            res = builder.build(
                dists, qoi.parameter_names(), qoi,
                n_samples=60, order=2, work_dir=tmp, qoi="idt", seed=11,
            )
        # a smooth IDT response over ±factor-2 should be captured well
        assert res.r2_insample > 0.9
        assert abs(sum(res.sobol_main.values()) - 1.0) < 0.4
        assert 1 <= res.as_n_active <= 4
        assert set(res.sobol_total) == set(qoi.parameter_names())
