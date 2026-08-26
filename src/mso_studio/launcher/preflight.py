"""
mso_studio.launcher.preflight
================================
Boot-time checks for the launcher splash screen and the main window's status
strip. Pure Python, no Qt (see ``__init__.py``) so it's directly unit
testable and reusable from both ``boot.py`` (splash) and ``window.py``
(status strip LEDs) without duplicating logic.

Each check has its own hard timeout so a wedged subprocess can never hang
the splash screen -- worst case a check reports ``fail`` and the launcher
still opens anyway.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import Iterator

from mso_studio.core.backends import SCIENCE_PYTHON
from mso_studio.launcher.catalog import CATALOG, REPO_ROOT, RUNNER_INTERNAL, script_path
from mso_studio.launcher.process import GUI_VENV_PYTHON

OK, WARN, FAIL = "ok", "warn", "fail"


@dataclass(frozen=True)
class BootCheck:
    label: str
    state: str    # OK | WARN | FAIL
    detail: str = ""


def _run_probe(argv, timeout: float) -> "subprocess.CompletedProcess[str] | None":
    try:
        return subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def check_repo_root() -> BootCheck:
    src = REPO_ROOT / "src"
    if src.is_dir():
        return BootCheck("Repository", OK, str(REPO_ROOT))
    return BootCheck("Repository", FAIL, f"src/ not found under {REPO_ROOT}")


def check_gui_venv() -> BootCheck:
    if not GUI_VENV_PYTHON.exists():
        return BootCheck(
            ".venv-gui / PySide6", FAIL,
            "missing -- run ./scripts/setup_gui_venv.sh",
        )
    result = _run_probe(
        [str(GUI_VENV_PYTHON), "-c", "import PySide6; print(PySide6.__version__)"],
        timeout=8.0,
    )
    if result is not None and result.returncode == 0:
        return BootCheck(".venv-gui / PySide6", OK, result.stdout.strip())
    return BootCheck(".venv-gui / PySide6", WARN, "python found but PySide6 import failed")


def check_science_python() -> BootCheck:
    result = _run_probe(
        [SCIENCE_PYTHON, "-c", "import sys; print(sys.version.split()[0])"],
        timeout=5.0,
    )
    if result is not None and result.returncode == 0:
        return BootCheck("Science python", OK, f"{SCIENCE_PYTHON} ({result.stdout.strip()})")
    return BootCheck("Science python", FAIL, f"could not run {SCIENCE_PYTHON}")


def check_cantera() -> BootCheck:
    result = _run_probe(
        [SCIENCE_PYTHON, "-c", "import cantera; print(cantera.__version__)"],
        timeout=10.0,
    )
    if result is not None and result.returncode == 0:
        return BootCheck("Cantera", OK, result.stdout.strip())
    return BootCheck("Cantera", WARN, "not importable in the science interpreter")


def check_modules() -> BootCheck:
    total = len(CATALOG)
    present = sum(
        1 for e in CATALOG
        if e.runner == RUNNER_INTERNAL or script_path(e).exists()
    )
    if present == total:
        return BootCheck("Modules", OK, f"{present}/{total} scripts found")
    return BootCheck("Modules", WARN, f"only {present}/{total} scripts found")


def check_disk() -> BootCheck:
    try:
        usage = shutil.disk_usage(REPO_ROOT)
    except OSError:
        return BootCheck("Disk space", WARN, "could not stat filesystem")
    free_gb = usage.free / (1024 ** 3)
    state = OK if free_gb > 2.0 else WARN
    return BootCheck("Disk space", state, f"{free_gb:.1f} GB free")


def run_boot_checks() -> Iterator[BootCheck]:
    """Yield each check's result as it completes, in display order."""
    yield check_repo_root()
    yield check_gui_venv()
    yield check_science_python()
    yield check_cantera()
    yield check_modules()
    yield check_disk()
