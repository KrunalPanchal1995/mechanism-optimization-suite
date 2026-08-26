"""
LaunchManager tests -- pure Python, no Qt at all (mso_studio.launcher.process
imports nothing from PySide6, see the module docstring). Verifies the fix
for the two real bugs in the old Tkinter launcher_gui.py: science-runner
entries must launch under SCIENCE_PYTHON (not sys.executable, which lacks
Cantera in the GUI venv), and a second launch of an already-running entry
must not spawn a second process.
"""
import subprocess

import pytest

from mso_studio.core.backends import SCIENCE_PYTHON
from mso_studio.launcher.catalog import CATALOG_BY_ID, REPO_ROOT
from mso_studio.launcher.process import (
    GUI_VENV_PYTHON, LaunchError, LaunchManager, build_argv, missing_prerequisite,
)


class _FakeProcess:
    def __init__(self, pid: int = 4242):
        self.pid = pid
        self._alive = True
        self.terminated = False
        self.killed = False

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self.terminated = True
        self._alive = False

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.killed = True
        self._alive = False


@pytest.fixture
def fake_popen(monkeypatch):
    calls = []

    def _popen(argv, cwd=None, env=None):
        proc = _FakeProcess()
        calls.append({"argv": argv, "cwd": cwd, "env": env, "proc": proc})
        return proc

    monkeypatch.setattr(subprocess, "Popen", _popen)
    return calls


def test_build_argv_uses_science_python_not_sys_executable():
    entry = CATALOG_BY_ID["target_add_opt"]  # the one remaining RUNNER_SCIENCE entry
    argv = build_argv(entry)
    assert argv[0] == SCIENCE_PYTHON
    assert argv[0] != __import__("sys").executable
    assert argv[1].endswith("generate_target_opt.py")


def test_build_argv_uses_gui_venv_python_for_mso_studio():
    entry = CATALOG_BY_ID["mso_studio"]
    argv = build_argv(entry)
    assert argv[0] == str(GUI_VENV_PYTHON)
    assert argv[1].endswith("run_mso_studio.py")


def test_build_argv_appends_module_args_for_per_module_tiles():
    """Phase 1 (2026-08 GUI integration overhaul): a tile with a PySide6
    equivalent launches run_mso_studio.py --module <id> --standalone
    instead of the legacy Tkinter script of the same name."""
    entry = CATALOG_BY_ID["mechanism_check"]
    argv = build_argv(entry)
    assert argv[0] == str(GUI_VENV_PYTHON)
    assert argv[1].endswith("run_mso_studio.py")
    assert argv[2:] == ["--module", "mechanism_check", "--standalone"]


def test_launch_passes_cwd_and_env(fake_popen):
    manager = LaunchManager()
    entry = CATALOG_BY_ID["mechanism_check"]
    manager.launch(entry)

    assert len(fake_popen) == 1
    call = fake_popen[0]
    assert call["cwd"] == str(REPO_ROOT)
    assert call["env"] is not None
    assert str(REPO_ROOT / "src") in call["env"].get("PYTHONPATH", "")


def test_launch_twice_does_not_spawn_a_second_process(fake_popen):
    manager = LaunchManager()
    entry = CATALOG_BY_ID["mechanism_check"]
    pid1 = manager.launch(entry)
    pid2 = manager.launch(entry)
    assert pid1 == pid2
    assert len(fake_popen) == 1


def test_stop_terminates_and_deregisters(fake_popen):
    manager = LaunchManager()
    entry = CATALOG_BY_ID["mechanism_check"]
    manager.launch(entry)
    assert manager.is_running(entry.id)

    manager.stop(entry.id)
    assert not manager.is_running(entry.id)
    assert fake_popen[0]["proc"].terminated


def test_poll_reaps_exited_children(fake_popen):
    manager = LaunchManager()
    entry = CATALOG_BY_ID["mechanism_check"]
    manager.launch(entry)
    fake_popen[0]["proc"]._alive = False  # simulate the child exiting on its own

    finished = manager.poll()
    assert finished == [entry.id]
    assert not manager.is_running(entry.id)


def test_last_uptime_s_recorded_on_reap(fake_popen):
    """Phase 1 code-review fix: missing_prerequisite() can't detect a
    present-but-broken interpreter (e.g. .venv-gui exists but PySide6 was
    removed from it) -- Popen still succeeds; only the child's own startup
    then fails. last_uptime_s() is the signal LauncherWindow uses to notice
    a suspiciously-quick exit instead of the RUNNING badge just silently
    disappearing with no explanation."""
    manager = LaunchManager()
    entry = CATALOG_BY_ID["mechanism_check"]

    assert manager.last_uptime_s(entry.id) is None  # never launched yet

    manager.launch(entry)
    fake_popen[0]["proc"]._alive = False
    manager.poll()

    uptime = manager.last_uptime_s(entry.id)
    assert uptime is not None
    assert uptime >= 0.0


def test_last_uptime_s_updates_on_relaunch(fake_popen):
    manager = LaunchManager()
    entry = CATALOG_BY_ID["mechanism_check"]

    manager.launch(entry)
    fake_popen[0]["proc"]._alive = False
    manager.poll()
    first_uptime = manager.last_uptime_s(entry.id)

    manager.launch(entry)
    fake_popen[1]["proc"]._alive = False
    manager.poll()
    second_uptime = manager.last_uptime_s(entry.id)

    assert second_uptime is not None
    assert first_uptime is not None


def test_missing_prerequisite_flags_missing_gui_venv(monkeypatch, tmp_path):
    monkeypatch.setattr("mso_studio.launcher.process.GUI_VENV_PYTHON", tmp_path / "nope" / "python")
    entry = CATALOG_BY_ID["mso_studio"]
    reason = missing_prerequisite(entry)
    assert reason is not None
    assert "setup_gui_venv.sh" in reason


def test_missing_prerequisite_none_for_internal_runner():
    entry = CATALOG_BY_ID["doctor"]
    assert missing_prerequisite(entry) is None


def test_launch_raises_for_internal_runner():
    manager = LaunchManager()
    entry = CATALOG_BY_ID["doctor"]
    with pytest.raises(LaunchError):
        manager.launch(entry)


def test_launch_error_message_is_user_facing(monkeypatch):
    def _boom(argv, cwd=None, env=None):
        raise OSError("no such file")

    monkeypatch.setattr(subprocess, "Popen", _boom)
    manager = LaunchManager()
    entry = CATALOG_BY_ID["mechanism_check"]
    with pytest.raises(LaunchError, match="Could not launch"):
        manager.launch(entry)
