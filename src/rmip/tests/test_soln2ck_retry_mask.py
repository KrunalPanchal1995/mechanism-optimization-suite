"""
rmip/tests/test_soln2ck_retry_mask.py
======================================
Regression test for the 2026-07-17 soln2ck retry-mask defect (bug.md).

`data_management.run_sim` (the serial-retry path) used to *unconditionally*
scan the per-case `run` script for a literal ``soln2ck_2.py`` command and
raise ``RuntimeError("Could not find the soln2ck_2.py command in the run
script.")`` when it was absent. But make_input_file.py's ``solver: cantera``
branch generates a `run` script that runs ``cantera_.py`` directly and never
contains ``soln2ck_2.py`` by design -- so every retry of a cantera case failed
with that spurious error, masking whatever really caused the first attempt to
fail.

The fix branches Step 5 on the case's actual solver via
``_run_script_needs_soln2ck``. This test locks that predicate to the invariant
it must uphold, and guards it against make_input_file's templates drifting out
of sync.
"""
from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

import pytest

_MUQSAC = Path(__file__).resolve().parents[2] / "MUQ-SAC"


@pytest.fixture(scope="module")
def dm():
    sys.path.insert(0, str(_MUQSAC))
    import data_management as _dm
    return _dm


@pytest.fixture(scope="module")
def mif():
    sys.path.insert(0, str(_MUQSAC))
    import make_input_file as _mif
    return _mif


def test_predicate_matches_each_solver(dm):
    # The core regression: a cantera case must NOT require soln2ck_2.py.
    assert dm._run_script_needs_soln2ck("cantera") is False
    assert dm._run_script_needs_soln2ck("CHEMKIN_PRO") is True
    assert dm._run_script_needs_soln2ck("zerork") is True
    # Missing / blank solver must not require it either (no spurious raise).
    assert dm._run_script_needs_soln2ck("") is False
    assert dm._run_script_needs_soln2ck(None) is False


def _extract_s_run_block(src: str, solver_literal: str) -> str:
    """Return the source text of the s_run template for a given solver branch
    inside make_input_file.create_input_file."""
    # Grab from the branch header up to the next 'elif'/'else'/'return'.
    if solver_literal == "cantera":
        start = src.index('if solver == "cantera":')
    else:
        start = src.index(f'elif solver == "{solver_literal}":')
    tail = src[start:]
    m = re.search(r"\n\telif solver ==|\n\telse:|\n\treturn ", tail[10:])
    end = (m.start() + 10) if m else len(tail)
    return tail[:end]


def test_predicate_agrees_with_make_input_file_templates(dm, mif):
    """Whether the predicate says a solver needs soln2ck must match whether
    that solver's s_run template actually emits `soln2ck_2.py`."""
    src = inspect.getsource(mif.create_input_file)
    for solver in ("cantera", "CHEMKIN_PRO", "zerork"):
        block = _extract_s_run_block(src, solver)
        template_has_soln2ck = "soln2ck_2.py" in block
        assert dm._run_script_needs_soln2ck(solver) == template_has_soln2ck, (
            f"solver={solver!r}: predicate says "
            f"{dm._run_script_needs_soln2ck(solver)} but its s_run template "
            f"{'contains' if template_has_soln2ck else 'lacks'} soln2ck_2.py"
        )
    # Explicit: the cantera template must not contain soln2ck_2.py (root cause).
    assert "soln2ck_2.py" not in _extract_s_run_block(src, "cantera")
