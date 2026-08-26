"""
LauncherWindow Phase-4 pause/resume UI tests: real subprocess (SIGSTOP
needs a real pid) + a scratch git repo monkeypatched in as REPO_ROOT,
matching test_launcher_pause.py's convention. Drives HeroPanel's
pause/resume signals and the window's handlers directly, not via click
simulation, matching this test module's usual convention.
"""
import subprocess

import pytest

from mso_studio.launcher.process import _RunningApp


def _init_scratch_repo(tmp_path):
    repo = tmp_path / "scratch_repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "README").write_text("scratch\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=repo, check=True)
    return repo


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    from PySide6.QtCore import QSettings
    ini_path = str(tmp_path / "launcher_test_settings.ini")

    def _factory(*_args, **_kwargs):
        return QSettings(ini_path, QSettings.Format.IniFormat)

    monkeypatch.setattr("mso_studio.launcher.window.QSettings", _factory)


@pytest.fixture
def scratch_repo(tmp_path, monkeypatch):
    repo = _init_scratch_repo(tmp_path)
    monkeypatch.setattr("mso_studio.launcher.process.REPO_ROOT", repo)
    return repo


@pytest.fixture(autouse=True)
def stub_information_dialog(monkeypatch):
    """_on_pause_requested always pops a real modal QMessageBox.information
    telling the user where to edit -- .exec() blocks forever under
    QT_QPA_PLATFORM=offscreen with nothing to click, so every test that
    reaches it needs this stubbed, not just the one test that asserts on
    it."""
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)


def _make_window_with_running_entry(qapp, entry_id="mechanism_check"):
    from mso_studio.launcher.process import LaunchManager
    from mso_studio.launcher.window import LauncherWindow

    manager = LaunchManager()
    proc = subprocess.Popen(["sleep", "30"])
    manager._running[entry_id] = _RunningApp(entry_id=entry_id, process=proc, argv=["sleep", "30"])

    window = LauncherWindow(manager, [])
    window.show()
    window._select_entry(entry_id)
    return window, proc


def test_pause_button_hidden_until_running(qapp, scratch_repo):
    from mso_studio.launcher.process import LaunchManager
    from mso_studio.launcher.window import LauncherWindow

    window = LauncherWindow(LaunchManager(), [])
    window.show()
    window._select_entry("mechanism_check")
    assert not window.hero.pause_button.isVisible()


def test_pause_requested_stops_process_and_shows_worktree(qapp, scratch_repo, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    window, proc = _make_window_with_running_entry(qapp)
    try:
        shown = []
        monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(a))

        window._on_pause_requested("mechanism_check")

        assert window._launch_manager.is_paused("mechanism_check")
        assert window.hero.pause_button.text() == "▶  Resume…"
        assert shown  # the "paused, edit here" dialog fired
        assert window._launch_manager.pause_session_for("mechanism_check") is not None
    finally:
        if window._launch_manager.is_paused("mechanism_check"):
            window._launch_manager.resume_unchanged("mechanism_check")
        proc.kill()
        proc.wait()


def test_pause_requested_warns_on_failure(qapp, scratch_repo, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from mso_studio.launcher.process import LaunchManager
    from mso_studio.launcher.window import LauncherWindow

    window = LauncherWindow(LaunchManager(), [])
    window.show()

    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))
    window._on_pause_requested("mechanism_check")  # nothing running -> PauseError

    assert warned


def test_resume_with_edited_flavor_relaunches(qapp, scratch_repo, monkeypatch):
    window, proc = _make_window_with_running_entry(qapp)
    try:
        window._on_pause_requested("mechanism_check")
        original_pid = window._launch_manager._running["mechanism_check"].process.pid

        monkeypatch.setattr(window, "_ask_resume_flavor", lambda *a, **k: "edited")
        window._on_resume_menu_requested("mechanism_check")

        assert not window._launch_manager.is_paused("mechanism_check")
        new_pid = window._launch_manager._running["mechanism_check"].process.pid
        assert new_pid != original_pid
    finally:
        window._launch_manager.stop("mechanism_check")


def test_resume_unchanged_flavor_keeps_same_process(qapp, scratch_repo, monkeypatch):
    window, proc = _make_window_with_running_entry(qapp)
    try:
        window._on_pause_requested("mechanism_check")

        monkeypatch.setattr(window, "_ask_resume_flavor", lambda *a, **k: "unchanged")
        window._on_resume_menu_requested("mechanism_check")

        assert not window._launch_manager.is_paused("mechanism_check")
        assert window._launch_manager._running["mechanism_check"].process.pid == proc.pid
    finally:
        proc.kill()
        proc.wait()


def test_resume_cancel_flavor_leaves_paused(qapp, scratch_repo, monkeypatch):
    window, proc = _make_window_with_running_entry(qapp)
    try:
        window._on_pause_requested("mechanism_check")

        monkeypatch.setattr(window, "_ask_resume_flavor", lambda *a, **k: "cancel")
        window._on_resume_menu_requested("mechanism_check")

        assert window._launch_manager.is_paused("mechanism_check")
    finally:
        window._launch_manager.resume_unchanged("mechanism_check")
        proc.kill()
        proc.wait()
