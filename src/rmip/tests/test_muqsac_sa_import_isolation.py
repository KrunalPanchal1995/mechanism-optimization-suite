"""
rmip/tests/test_muqsac_sa_import_isolation.py
================================================
Phase 3.1 (2026-08 backlog program, bug.md): pins down which of the two
diverged module trees -- ``src/MUQ-SAC/`` and ``src/Sensitivity_Analysis/``
-- each real entry point resolves its bare imports (``MechManipulator2_0``,
``DesignMatrix``/``DesignMatrix2_0``, ``data_management``,
``combustion_target_class``, ``simulation_manager2_0``) to.

Why this matters: both trees define several IDENTICALLY-NAMED top-level
modules (``MechManipulator2_0.py``, ``data_management.py``,
``combustion_target_class.py``, ``simulation_manager2_0.py`` in both;
``DesignMatrix.py`` in both, plus SA's own ``DesignMatrix2_0.py``), imported
everywhere by bare name (``import MechManipulator2_0``), never as a
qualified package (``MUQ-SAC`` isn't even a valid Python package name --
it has a hyphen). Which copy a bare import resolves to depends entirely on
``sys.path`` order at import time.

**Why this is safe in production today**: every real entry point
(``sens_3_params.py``, ``run_script.py``, ``nominal_sim_v3.py``, etc.) is
launched as its own OS subprocess (``science_command``/``PtyTerminal`` build
a fresh ``[python, <script>, ...]`` argv -- see ``mso_studio.core.backends.
science_command``, ``brute_force_sa_gui.py``), never ``exec``'d or
``runpy``'d in-process. Python's own standard behavior --the invoked
script's directory is auto-prepended to ``sys.path[0]``-- already
deterministically resolves every bare import to the correct tree, with zero
cross-contamination, because each subprocess starts with a fresh interpreter
and an empty ``sys.modules`` cache.

**The residual hazard** (what these tests guard against): if a future
change ever imports/execs one of these entry points' modules *in-process*
alongside the other tree's modules -- e.g. converting a subprocess call to
an in-process import "for performance", or a test file importing both trees'
same-named modules in one pytest session -- ``sys.modules`` caching means
whichever tree's module got imported FIRST silently wins for every later
bare import of the same name, even if the second import site explicitly
inserted its own directory at ``sys.path[0]`` first. ``TestKnownShadowingHazard``
below reproduces this concretely so it stays a documented, tested fact
instead of a rediscovered surprise.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_MUQSAC_DIR = _ROOT / "src" / "MUQ-SAC"
_SA_DIR = _ROOT / "src" / "Sensitivity_Analysis"

_PY = sys.executable


def _resolve_module_file(script_dir: Path, module_name: str) -> str:
    """Launch a fresh subprocess whose sys.path[0] is script_dir (mirroring
    `python <script_dir>/<script>.py`'s own auto-prepend behavior), import
    module_name, and return its __file__ -- exactly what a real entry point
    launched via subprocess.run([python, script_path, ...]) would resolve.
    """
    code = f"import sys; sys.path.insert(0, {str(script_dir)!r}); import {module_name}; print({module_name}.__file__)"
    result = subprocess.run(
        [_PY, "-c", code], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, (
        f"subprocess import of {module_name!r} from {script_dir} failed:\n{result.stderr}"
    )
    return result.stdout.strip()


class TestEntryPointResolution:
    """Each real entry point's own script directory (auto-prepended to
    sys.path[0] by the interpreter when run as `python <script>.py`) must
    resolve bare imports to that SAME tree, never the other one."""

    @pytest.mark.parametrize("module_name", [
        "MechManipulator2_0", "data_management", "combustion_target_class",
        "simulation_manager2_0",
    ])
    def test_muqsac_entry_points_resolve_to_muqsac_tree(self, module_name):
        resolved = _resolve_module_file(_MUQSAC_DIR, module_name)
        assert resolved.startswith(str(_MUQSAC_DIR)), (
            f"{module_name} resolved to {resolved!r}, expected under {_MUQSAC_DIR}"
        )

    @pytest.mark.parametrize("module_name", [
        "MechManipulator2_0", "data_management", "combustion_target_class",
        "simulation_manager2_0",
    ])
    def test_sa_entry_points_resolve_to_sa_tree(self, module_name):
        resolved = _resolve_module_file(_SA_DIR, module_name)
        assert resolved.startswith(str(_SA_DIR)), (
            f"{module_name} resolved to {resolved!r}, expected under {_SA_DIR}"
        )

    def test_muqsac_run_script_uses_bare_designmatrix(self):
        """run_script.py (MUQ-SAC) imports the base `DesignMatrix.py`, not
        SA's DesignMatrix2_0.py -- confirms these two specific entry points
        don't actually collide on THIS module name even without isolation,
        since the names themselves differ."""
        resolved = _resolve_module_file(_MUQSAC_DIR, "DesignMatrix")
        assert resolved == str(_MUQSAC_DIR / "DesignMatrix.py")

    def test_sens3p_uses_designmatrix2_0(self):
        """sens_3_params.py (Sensitivity_Analysis) imports DesignMatrix2_0,
        the module M30's chunk-width/index-mapping fix lives in -- confirmed
        distinct from run_script.py's bare `DesignMatrix` above."""
        resolved = _resolve_module_file(_SA_DIR, "DesignMatrix2_0")
        assert resolved == str(_SA_DIR / "DesignMatrix2_0.py")


class TestKnownShadowingHazard:
    """Documents the residual in-process hazard concretely: if MUQ-SAC's
    MechManipulator2_0 is imported FIRST in a process, a later attempt to
    reach Sensitivity_Analysis's copy (even after inserting SA's directory
    at sys.path[0]) silently returns the ALREADY-CACHED MUQ-SAC module
    instead -- sys.modules caching wins over sys.path order for a name
    that's already been imported once.

    This is a "confirmed dangerous" regression test, not a bug to fix here:
    it exists so a future change that moves one of these entry points from
    subprocess-isolated to in-process execution gets caught immediately
    instead of silently mis-resolving which tree's code actually ran.
    """

    def test_import_order_wins_over_later_sys_path_insert(self):
        code = (
            f"import sys\n"
            f"sys.path.insert(0, {str(_MUQSAC_DIR)!r})\n"
            f"import MechManipulator2_0\n"
            f"first_file = MechManipulator2_0.__file__\n"
            f"sys.path.insert(0, {str(_SA_DIR)!r})\n"
            f"import importlib\n"
            f"import MechManipulator2_0 as second_import\n"
            f"print(first_file)\n"
            f"print(second_import.__file__)\n"
        )
        result = subprocess.run(
            [_PY, "-c", code], capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
        first_file, second_file = result.stdout.strip().splitlines()
        assert first_file == str(_MUQSAC_DIR / "MechManipulator2_0.py")
        # The hazard: the "second" import, despite SA now being earlier on
        # sys.path, is STILL the MUQ-SAC copy -- sys.modules caching, not
        # sys.path order, decided this.
        assert second_file == first_file
        assert second_file != str(_SA_DIR / "MechManipulator2_0.py")
