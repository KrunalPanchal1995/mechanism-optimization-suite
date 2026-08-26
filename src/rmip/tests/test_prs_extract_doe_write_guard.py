"""
rmip/tests/test_prs_extract_doe_write_guard.py
==================================================
Regression coverage for the read-only write guard added to
src/automation/prs_extract_doe.py (Phase 4, 2026-08 backlog program, bug.md).

MSO_PPRS_DATA_ROOT (default: this repo; in real use, set to
/data2/KRUNAL/FINALIZED_STAGE_OPTIMIZATION_2026/MSO_A_FACTOR/p_PRS by
src/mso_studio/core/automation.py, which sets it at IMPORT time via
os.environ.setdefault) points at the live external p_PRS campaign's own
directory. Nothing in prs_extract_doe.py currently writes there (write_csv's
out_dir defaults to the safe, in-repo OUT_DIR), but this guard makes that
invariant explicit and enforced -- so a future caller can't accidentally
write into the live campaign's directory.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@pytest.fixture()
def module(monkeypatch, tmp_path):
    """Reload prs_extract_doe with MSO_PPRS_DATA_ROOT pointed at a
    controlled temp directory, so tests don't depend on the real external
    campaign path existing (or not) on this machine."""
    fake_pprs_root = tmp_path / "fake_pprs_root"
    fake_pprs_root.mkdir()
    monkeypatch.setenv("MSO_PPRS_DATA_ROOT", str(fake_pprs_root))

    if "automation.prs_extract_doe" in sys.modules:
        del sys.modules["automation.prs_extract_doe"]
    import automation.prs_extract_doe as m
    importlib.reload(m)
    return m


class TestWriteGuard:
    def test_default_out_dir_still_works(self, module, tmp_path):
        rows = [{"p_start": 1, "p_final": 0.05, "gens": 1200,
                  "wmre": 12.1, "source": "test"}]
        out_path = module.write_csv("test_fuel", rows, out_dir=module.OUT_DIR)
        assert out_path.exists()
        out_path.unlink()  # cleanup -- OUT_DIR is real, in-repo

    def test_writing_under_pprs_root_raises(self, module):
        rows = [{"p_start": 1, "p_final": 0.05, "gens": 1200,
                  "wmre": 12.1, "source": "test"}]
        bad_dir = module.PPRS_DATA_ROOT / "MECHANISM_COMPARISON" / "should_not_write_here"
        with pytest.raises(PermissionError, match="PPRS_DATA_ROOT"):
            module.write_csv("test_fuel", rows, out_dir=bad_dir)
        assert not bad_dir.exists()

    def test_writing_to_pprs_root_itself_raises(self, module):
        rows = []
        with pytest.raises(PermissionError, match="PPRS_DATA_ROOT"):
            module.write_csv("test_fuel", rows, out_dir=module.PPRS_DATA_ROOT)

    def test_sibling_directory_outside_pprs_root_is_fine(self, module, tmp_path):
        """Sanity check the guard isn't overly broad -- a directory that
        merely shares a path PREFIX with PPRS_DATA_ROOT (not actually nested
        under it) must not be rejected."""
        sibling = module.PPRS_DATA_ROOT.parent / (module.PPRS_DATA_ROOT.name + "_sibling")
        sibling.mkdir(exist_ok=True)
        rows = [{"p_start": 1, "p_final": 0.05, "gens": 1200,
                  "wmre": 12.1, "source": "test"}]
        out_path = module.write_csv("test_fuel", rows, out_dir=sibling)
        assert out_path.exists()
