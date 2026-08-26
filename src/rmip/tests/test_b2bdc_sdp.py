"""
rmip/tests/test_b2bdc_sdp.py
============================
Tests for the B2BDC SDP **outer** bounds (Phase-7 follow-on, Point 2b).

The PCEPoly→quadratic conversion is pure and always runs; the SDP path
(``B2BDCDataset.analyze_outer``) is guarded on the presence of the cvxpy venv,
mirroring how the emcee/MCMC tests guard on the chaospy venv.

Run with:
    python -m pytest src/rmip/tests/test_b2bdc_sdp.py -v
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from rmip.analysis.bayesian_calibration import PCEPoly
from rmip.analysis.b2bdc import B2BDCConstraint, B2BDCDataset, _CVXPY_PYTHON_DEFAULT

_HAS_CVXPY_VENV = os.path.exists(_CVXPY_PYTHON_DEFAULT)
_venv = pytest.mark.skipif(not _HAS_CVXPY_VENV, reason="cvxpy venv not present")


# ---------------------------------------------------------------------------
# PCEPoly.to_quadratic (pure)
# ---------------------------------------------------------------------------

def test_to_quadratic_matches_eval():
    # 3 + 2x0 - x1 + 4x0^2 + 5x0x1
    p = PCEPoly([[0, 0], [1, 0], [0, 1], [2, 0], [1, 1]], [3, 2, -1, 4, 5])
    assert p.is_quadratic()
    A, b, c = p.to_quadratic()
    assert np.allclose(A, A.T)                      # symmetric
    for x in (np.array([0.3, -0.7]), np.array([1.1, 0.4]), np.zeros(2)):
        assert p.eval(x) == pytest.approx(float(x @ A @ x + b @ x + c))


def test_to_quadratic_rejects_cubic():
    p = PCEPoly([[3, 0]], [1.0])
    assert not p.is_quadratic()
    with pytest.raises(ValueError):
        p.to_quadratic()


# ---------------------------------------------------------------------------
# SDP outer bounds (need the cvxpy venv)
# ---------------------------------------------------------------------------

@_venv
def test_outer_consistent_recovers_feasible_interval(tmp_path):
    # M(x)=x, band [0.2, 0.6] → feasible set x∈[0.2,0.6]
    c = B2BDCConstraint("exp1", PCEPoly([[1]], [1.0]), 0.2, 0.6)
    ds = B2BDCDataset([c], ["r0"], [(-1.0, 1.0)])
    res = ds.analyze_outer(work_dir=str(tmp_path / "sdp"))
    assert res.bound_type == "outer"
    assert res.consistent
    assert res.consistency_measure == pytest.approx(1.0, abs=1e-2)
    lo, hi = res.parameter_bounds["r0"]
    assert lo == pytest.approx(0.2, abs=1e-2)
    assert hi == pytest.approx(0.6, abs=1e-2)


@_venv
def test_outer_contains_inner(tmp_path):
    # two constraints on quadratic-ish models over 2 params
    c1 = B2BDCConstraint("e1", PCEPoly([[1, 0]], [1.0]), -0.3, 0.5)   # x0∈[-0.3,0.5]
    c2 = B2BDCConstraint("e2", PCEPoly([[0, 1]], [1.0]), 0.0, 0.4)    # x1∈[0.0,0.4]
    ds = B2BDCDataset([c1, c2], ["r0", "r1"], [(-1.0, 1.0), (-1.0, 1.0)])
    inner = ds.analyze(n_starts=10, seed=1)
    outer = ds.analyze_outer(work_dir=str(tmp_path / "sdp"))
    assert outer.consistent
    for name in ("r0", "r1"):
        ilo, ihi = inner.parameter_bounds[name]
        olo, ohi = outer.parameter_bounds[name]
        # outer interval contains the inner interval (with numerical slack)
        assert olo <= ilo + 1e-2
        assert ohi >= ihi - 1e-2


@_venv
def test_outer_certifies_inconsistency(tmp_path):
    # same model, two disjoint bands → feasible set empty
    surr = PCEPoly([[1]], [1.0])
    c1 = B2BDCConstraint("lo", surr, 0.2, 0.4)
    c2 = B2BDCConstraint("hi", surr, 0.6, 0.8)
    ds = B2BDCDataset([c1, c2], ["r0"], [(-1.0, 1.0)])
    res = ds.analyze_outer(work_dir=str(tmp_path / "sdp"))
    # C < 0 is a rigorous inconsistency certificate for the outer bound
    assert not res.consistent
    assert res.consistency_measure < 0.0


@_venv
def test_outer_rejects_nonquadratic(tmp_path):
    c = B2BDCConstraint("cubic", PCEPoly([[3]], [1.0]), 0.0, 1.0)
    ds = B2BDCDataset([c], ["r0"], [(-1.0, 1.0)])
    with pytest.raises(ValueError, match="quadratic"):
        ds.analyze_outer(work_dir=str(tmp_path / "sdp"))
