"""
rmip/tests/test_ammonia_bug17_selection_mismatch.py
======================================================
Regression coverage for Ammonia campaign Bug 17 (AMMONIA_CAMPAIGN_2026-07-22.md,
bug.md): under correlation-based active-set reduction (the actively-evolving
partial-PRS "2_0" path), `self.selection` can arrive narrower than either
MechManipulator2_0 copy's fixed-stride assumption expects.

- `src/MUQ-SAC/MechManipulator2_0.py::getRxnPerturbationDict` already raised a
  clear ValueError in this case (`consumed N selection entries but
  self.selection has M`) -- confirmed still true.
- `src/Sensitivity_Analysis/MechManipulator2_0.py`'s copy used a HARDCODED
  stride of 3 per reaction with no sanity check at all -- a mismatch was
  silently misread instead of raising, which is WORSE (wrong statistics with
  no error, not just a crash). Fixed here to raise the same class of loud
  error up front.

Deliberately NOT fixed: the underlying correlation-reduction algorithm's
actual selection-array format/semantics -- see the dedicated bug.md note for
why a deeper fix wasn't attempted (insufficient context on the actively-
evolving partial-PRS subsystem's intended contract; the goal here is
converting a silent-wrong-data failure mode into a loud one, not redesigning
the format).

Runs via real subprocess isolation (matching the lesson from Phase 3.4's own
test, which broke other tests in the same pytest session by importing
MechManipulator2_0's bare name in-process) -- both MUQ-SAC's and
Sensitivity_Analysis's copies define an identically-named `Manipulator`
class importable as `MechManipulator2_0`.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_MUQSAC_SRC = _ROOT / "src" / "MUQ-SAC"
_SA_SRC = _ROOT / "src" / "Sensitivity_Analysis"
_PY = sys.executable

_FAKE_UNSRT_SETUP = """
class _FakeReaction:
    def __init__(self):
        self.activeParameters = ["1_A", "2_n", "3_Ea"]
unsrt = {"rxn1": _FakeReaction(), "rxn2": _FakeReaction(), "rxn3": _FakeReaction()}
mech = {"reactions": []}
"""


def _run_isolated(src_dir: Path, code: str) -> subprocess.CompletedProcess:
    full_code = f"import sys; sys.path.insert(0, {str(src_dir)!r})\n" + _FAKE_UNSRT_SETUP + code
    return subprocess.run(
        [_PY, "-c", full_code], capture_output=True, text=True, timeout=30,
    )


def _importable(src_dir: Path) -> bool:
    result = _run_isolated(src_dir, "import MechManipulator2_0")
    return result.returncode == 0


class TestSAMismatchNowRaisesLoudly:
    """The exact Bug 17 repro: 3 reactions, compressed-diagonal perturbation
    (len == N_rxns), but self.selection arrives with only 3 entries instead
    of the 9 (3 per reaction) either copy's fixed-stride convention needs."""

    @pytest.mark.skipif(not _importable(_SA_SRC), reason="Sensitivity_Analysis/MechManipulator2_0 not importable")
    def test_narrow_selection_raises_valueerror_not_silent_misindex(self):
        result = _run_isolated(_SA_SRC, """
import numpy as np
from MechManipulator2_0 import Manipulator
perturbation = np.zeros(3)   # compressed-diagonal format, len == N_rxns
selection = np.zeros(3)      # BUG 17 repro: only 3, not 3*3=9
m = Manipulator(mech, unsrt, perturbation, selection=selection)
try:
    m.getRxnPerturbationDict()
    print("NO_RAISE")
except ValueError as e:
    print("RAISED:" + str(e))
""")
        assert result.returncode == 0, result.stderr
        out = result.stdout.strip()
        assert out.startswith("RAISED:"), f"expected a loud ValueError, got: {out!r}"
        assert "self.selection has 3" in out
        assert "expects 9" in out

    @pytest.mark.skipif(not _importable(_SA_SRC), reason="Sensitivity_Analysis/MechManipulator2_0 not importable")
    def test_correctly_sized_selection_still_works(self):
        """Sanity check: the fix must not break the previously-working case
        (full 3-per-reaction width) -- no regression for real callers."""
        result = _run_isolated(_SA_SRC, """
import numpy as np
from MechManipulator2_0 import Manipulator
perturbation = np.array([1.0, 0.0, 0.0])   # only rxn1 active
selection = np.array([1.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0])  # 3*3=9, correct width
m = Manipulator(mech, unsrt, perturbation, selection=selection)
perturb, select_dict = m.getRxnPerturbationDict()
print("OK", len(perturb), len(select_dict))
print(list(perturb["rxn1"]))
""")
        assert result.returncode == 0, result.stderr
        lines = result.stdout.strip().splitlines()
        assert lines[0] == "OK 3 3"
        assert lines[1] == "[1.0, 0.0, 0.0]"

    @pytest.mark.skipif(not _importable(_MUQSAC_SRC), reason="MUQ-SAC/MechManipulator2_0 not importable")
    def test_muqsac_sibling_raises_the_analogous_error_for_parity(self):
        """MUQ-SAC's copy already raised for this class of mismatch --
        confirms the SA fix now produces the SAME class of protection, not
        just a superficially similar one."""
        result = _run_isolated(_MUQSAC_SRC, """
import numpy as np
from MechManipulator2_0 import Manipulator
perturbation = np.zeros(3)
selection = np.zeros(3)   # each reaction here has activeParameters len=3, so 9 expected
m = Manipulator(mech, unsrt, perturbation, selection=selection)
try:
    m.getRxnPerturbationDict()
    print("NO_RAISE")
except ValueError as e:
    print("RAISED:" + str(e))
""")
        assert result.returncode == 0, result.stderr
        out = result.stdout.strip()
        assert out.startswith("RAISED:")
        assert "consumed" in out and "self.selection has 3" in out
