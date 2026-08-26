"""
rmip/tests/test_muqsac_extended_designmatrix_rxn_len.py
==========================================================
Regression test for the self.rxn_len latent bug in
src/MUQ-SAC_EXTENDED/DesignMatrix.py (Phase 3.4, 2026-08 backlog program,
bug.md): getSA_samples/getNominal_samples both referenced self.rxn_len,
which was never assigned in __init__ -- the same class of bug already fixed
in src/MUQ-SAC/DesignMatrix.py (2026-07-22, ammonia campaign). This copy has
no live caller anywhere in the repo (confirmed via repo-wide grep), so this
guards against a latent AttributeError rather than a reported failure.

Runs via a real subprocess (matching Phase 3.1's own import-isolation
tests), NOT an in-process import: importing MUQ-SAC_EXTENDED's DesignMatrix/
Uncertainty bare names into the SAME pytest session poisons sys.modules for
every other test that expects the MUQ-SAC/ or Sensitivity_Analysis/ copies
of those same bare names -- confirmed the hard way, this file originally did
exactly that and broke test_gsa_prs_crosscheck.py/test_gsa_reaction_
prefilter.py when run in the same session (exactly the hazard
TestKnownShadowingHazard in test_muqsac_sa_import_isolation.py documents).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_EXT_SRC = _ROOT / "src" / "MUQ-SAC_EXTENDED"
_PY = sys.executable


def _run_isolated(code: str) -> subprocess.CompletedProcess:
    full_code = (
        f"import sys; sys.path.insert(0, {str(_EXT_SRC)!r})\n"
        + code
    )
    return subprocess.run(
        [_PY, "-c", full_code], capture_output=True, text=True, timeout=30,
    )


def _muqsac_extended_importable() -> bool:
    result = _run_isolated("import Uncertainty; import DesignMatrix")
    return result.returncode == 0


pytestmark = pytest.mark.skipif(
    not _muqsac_extended_importable(),
    reason="MUQ-SAC_EXTENDED/DesignMatrix.py's compiled deps (shuffle.so, Uncertainty) unavailable",
)


class TestRxnLenAssigned:
    def test_get_sa_samples_no_longer_raises_attributeerror(self):
        result = _run_isolated(
            "import DesignMatrix as DM\n"
            "d = DM.DesignMatrix({}, 'design', sample_length=3, ind=3)\n"
            "sa = d.getSA_samples(0.1)\n"
            "print(sa.shape)\n"
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "(3, 3)"

    def test_get_nominal_samples_no_longer_raises_attributeerror(self):
        result = _run_isolated(
            "import DesignMatrix as DM\n"
            "d = DM.DesignMatrix({}, 'design', sample_length=3, ind=3)\n"
            "print(d.getNominal_samples())\n"
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "[[0.0, 0.0, 0.0]]"

    def test_rxn_len_falls_back_to_sample_length_when_ind_none(self):
        result = _run_isolated(
            "import DesignMatrix as DM\n"
            "d = DM.DesignMatrix({}, 'design', sample_length=4)\n"
            "print(d.rxn_len)\n"
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "4"
