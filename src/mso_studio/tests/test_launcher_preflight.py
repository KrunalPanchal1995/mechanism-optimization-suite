"""
Preflight boot-check tests -- pure Python, no Qt. Each check must degrade to
warn/fail rather than raise when a probed interpreter/subprocess is missing
or hangs, since these checks feed the splash screen directly (an exception
here must never crash launcher_qt.py before its window even opens).
"""
import subprocess

import pytest

from mso_studio.launcher.preflight import (
    FAIL, OK, WARN, check_cantera, check_disk, check_gui_venv, check_modules,
    check_repo_root, check_science_python, run_boot_checks,
)

_VALID_STATES = {OK, WARN, FAIL}


def test_run_boot_checks_yields_all_six_in_order():
    labels = [c.label for c in run_boot_checks()]
    assert labels == [
        "Repository", ".venv-gui / PySide6", "Science python", "Cantera",
        "Modules", "Disk space",
    ]


def test_every_check_has_a_valid_state():
    for check in run_boot_checks():
        assert check.state in _VALID_STATES, f"{check.label}: invalid state {check.state!r}"


def test_check_repo_root_ok_on_the_real_repo():
    assert check_repo_root().state == OK


def test_check_modules_reports_all_scripts_found():
    check = check_modules()
    assert check.state == OK
    assert "/" in check.detail or check.detail.split("/")[0] == check.detail.split("/")[1]


def test_check_disk_never_raises():
    check = check_disk()
    assert check.state in _VALID_STATES


def test_check_science_python_degrades_to_fail_on_timeout(monkeypatch):
    def _timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout", 5))

    monkeypatch.setattr(subprocess, "run", _timeout)
    check = check_science_python()
    assert check.state == FAIL


def test_check_cantera_degrades_to_warn_on_timeout(monkeypatch):
    def _timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout", 10))

    monkeypatch.setattr(subprocess, "run", _timeout)
    check = check_cantera()
    assert check.state == WARN


def test_check_gui_venv_fails_when_venv_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("mso_studio.launcher.preflight.GUI_VENV_PYTHON", tmp_path / "nope")
    check = check_gui_venv()
    assert check.state == FAIL
    assert "setup_gui_venv.sh" in check.detail


def test_check_science_python_never_raises_on_missing_binary(monkeypatch):
    monkeypatch.setattr("mso_studio.launcher.preflight.SCIENCE_PYTHON", "/no/such/python-binary")
    check = check_science_python()  # must not raise
    assert check.state == FAIL
