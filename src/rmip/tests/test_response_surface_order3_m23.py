"""
rmip/tests/test_response_surface_order3_m23.py
===============================================
Regression test for M23 (bug.md).

Both ``ResponseSurface`` implementations build their polynomial design matrix
(``MatPolyFitTransform``) up to order 5, but their *evaluation* half diverged:

* ``MUQ-SAC/ResponseSurface.py::evaluate_prs`` summed only order-1 and order-2
  terms, silently ignoring the order>=3 coefficients the fit produced.
* ``Sensitivity_Analysis/ResponseSurface.py::evaluate`` used ``row.index()`` /
  enumerate-vs-slice indexing for order>=3, producing the wrong monomials
  (and duplicates when zeta values repeat).

The invariant both must satisfy: evaluating at a point x must equal the dot
product of that point's fit design-matrix row with the coefficients. This test
loads each module by file path (they share the basename ``ResponseSurface``) and
checks that invariant for orders 2..5, including a deliberately repeated-value
input (the case that broke the old ``.index()`` code).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]
# Both ResponseSurface.py modules import sibling modules (StatisticalAnalysis,
# etc.) by bare name, so each engine dir must be importable. Append (never
# insert at 0) to keep stdlib precedence, per this repo's path rule.
for _d in ("MUQ-SAC", "Sensitivity_Analysis"):
    _p = str(_REPO / "src" / _d)
    if _p not in sys.path:
        sys.path.append(_p)
_CASES = {
    "muqsac": (_REPO / "src" / "MUQ-SAC" / "ResponseSurface.py", "evaluate_prs"),
    "sens":   (_REPO / "src" / "Sensitivity_Analysis" / "ResponseSurface.py", "evaluate"),
}


def _load_rs_class(path: Path, unique_name: str):
    spec = importlib.util.spec_from_file_location(unique_name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ResponseSurface


@pytest.mark.parametrize("which", list(_CASES.keys()))
@pytest.mark.parametrize("order", [2, 3, 4, 5])
@pytest.mark.parametrize("row", [
    [0.5, -0.3, 0.2],
    [0.5, 0.5, 0.5],       # repeated values: the case row.index() got wrong
    [0.1, 0.1, -0.4, 0.7],
])
def test_evaluate_matches_fit_transform(which, order, row):
    path, eval_method = _CASES[which]
    RS = _load_rs_class(path, f"rs_{which}")

    # Build an instance without the heavy __init__ (fit pipeline / cwd I/O);
    # we only exercise the pure monomial construction + evaluation.
    rs = RS.__new__(RS)
    rs.order = order
    rs.X = np.array([row], dtype=float)
    # feature_map=None reproduces the raw-zeta polynomial path this test
    # exercises (MUQ-SAC's ResponseSurface gained an opt-in identifiable-
    # subspace feature_map; Sensitivity_Analysis's has no such attribute and
    # ignores it -- see MUQ-SAC/ResponseSurface.py's class docstring).
    rs.feature_map = None

    # The fit design-matrix row for this point.
    design_row = np.asarray(rs.MatPolyFitTransform()[0], dtype=float)

    # Arbitrary but fixed coefficients of the matching length.
    rng = np.random.default_rng(0)
    rs.coeff = rng.standard_normal(len(design_row))

    expected = float(np.dot(design_row, rs.coeff))
    got = float(getattr(rs, eval_method)(row))

    assert got == pytest.approx(expected, rel=1e-9, abs=1e-9), (
        f"{which} order={order} row={row}: evaluate returned {got}, "
        f"fit-transform dot product is {expected}"
    )
