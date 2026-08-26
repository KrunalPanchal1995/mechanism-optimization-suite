"""
LauncherWindow tests: build offscreen, drive stage/search/favourite
filtering and the hero panel's launch/stop state, matching the "construct +
drive real methods, no click simulation" convention of test_converter_page.py.
A fake subprocess.Popen (module-scoped monkeypatch) stands in for real
processes -- these tests must never actually spawn a child GUI app.
"""
import subprocess

import pytest


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    """LauncherWindow persists favourites/recent/geometry to QSettings under
    the real "mechanism-optimization-suite"/"mso-launcher" scope, which
    writes to the user's actual ~/.config -- redirect every QSettings(...)
    call this module makes to a throwaway ini file per test, so the test
    suite never reads stale state from (or writes pollution into) real
    on-disk settings."""
    from PySide6.QtCore import QSettings
    ini_path = str(tmp_path / "launcher_test_settings.ini")

    def _factory(*_args, **_kwargs):
        return QSettings(ini_path, QSettings.Format.IniFormat)

    monkeypatch.setattr("mso_studio.launcher.window.QSettings", _factory)


class _FakeProcess:
    def __init__(self, pid: int = 4242):
        self.pid = pid
        self._alive = True

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self._alive = False

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self._alive = False


@pytest.fixture
def fake_popen(monkeypatch):
    def _popen(argv, cwd=None, env=None):
        return _FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", _popen)


def _make_window(qapp):
    from mso_studio.launcher.process import LaunchManager
    from mso_studio.launcher.window import LauncherWindow
    window = LauncherWindow(LaunchManager(), [])
    # QWidget.isVisible() reflects real visibility (all ancestors shown),
    # not just the internal "not explicitly hidden" flag -- show() is
    # required before any tile-visibility assertion means anything. Safe
    # under QT_QPA_PLATFORM=offscreen (no real display needed).
    window.show()
    return window


def test_window_builds_one_tile_per_catalog_entry(qapp):
    from mso_studio.launcher.catalog import CATALOG
    window = _make_window(qapp)
    assert len(window._tiles) == len(CATALOG)


def test_all_filter_shows_every_tile(qapp):
    from mso_studio.launcher.catalog import CATALOG
    window = _make_window(qapp)
    visible = [eid for eid, t in window._tiles.items() if t.isVisible()]
    assert set(visible) == {e.id for e in CATALOG}


def test_showevent_does_not_schedule_the_auto_welcome_timer_under_offscreen(qapp, monkeypatch):
    """Round 4 regression (WORKBENCH_AUDIT_2026-08-10.md): showEvent() used
    to unconditionally schedule QTimer.singleShot(0,
    _maybe_show_welcome_on_first_run), which -- for a "first run" window
    (the isolated_settings fixture always gives a fresh, never-seen
    QSettings) -- fires an UNMOCKED WelcomeDialog(...).exec() on some later
    event-loop pump. Under QT_QPA_PLATFORM=offscreen (every test invocation
    in this repo), there is no display and nothing can dismiss that modal
    dialog, so it blocks forever -- confirmed live: this exact scenario
    hung a real test run for over an hour before being root-caused with
    faulthandler.dump_traceback_later. The fix skips scheduling that timer
    entirely under offscreen; this test proves the skip by pumping events
    well past what a real singleShot(0, ...) would need and confirming
    _open_welcome was still never called."""
    from PySide6.QtWidgets import QApplication

    opened = []
    monkeypatch.setattr(
        "mso_studio.launcher.window.LauncherWindow._open_welcome",
        lambda self: opened.append(True),
    )
    window = _make_window(qapp)  # calls .show(), which fires showEvent()
    for _ in range(20):
        QApplication.processEvents()
    assert opened == [], (
        "the auto-welcome timer fired under the offscreen platform -- "
        "this would hang forever with a real, unmocked WelcomeDialog")


def test_stage_filter_narrows_the_visible_set(qapp):
    from mso_studio.launcher.catalog import entries_for_stage
    window = _make_window(qapp)
    window._set_filter("stage", "prepare")
    visible = {eid for eid, t in window._tiles.items() if t.isVisible()}
    assert visible == {e.id for e in entries_for_stage("prepare")}
    assert "idt_sensitivity" not in visible  # that's an "analyse"-stage entry


def test_search_matches_by_title(qapp):
    window = _make_window(qapp)
    window.search_box.setText("ignition")
    visible = {eid for eid, t in window._tiles.items() if t.isVisible()}
    assert visible == {"idt_sensitivity"}


def test_search_and_stage_filter_combine(qapp):
    window = _make_window(qapp)
    window._set_filter("stage", "prepare")
    window.search_box.setText("ignition")  # not in the prepare stage
    visible = {eid for eid, t in window._tiles.items() if t.isVisible()}
    assert visible == set()


def test_favourite_toggle_persists_and_filters(qapp):
    window = _make_window(qapp)
    assert "idt_sensitivity" not in window._favourites

    window._toggle_favourite("idt_sensitivity")
    assert "idt_sensitivity" in window._favourites
    assert window._tiles["idt_sensitivity"].star_label.text() == "★"

    window._set_filter("favourites", None)
    visible = {eid for eid, t in window._tiles.items() if t.isVisible()}
    assert visible == {"idt_sensitivity"}

    window._toggle_favourite("idt_sensitivity")
    assert "idt_sensitivity" not in window._favourites


def test_single_favourite_survives_a_fresh_qsettings_reload(qapp, tmp_path, monkeypatch):
    """Regression test: Qt's INI QSettings backend can collapse a
    single-element string list to a bare scalar string once it round-trips
    through disk to a *new* QSettings instance. set(that_scalar_string)
    then silently explodes into a set of individual characters instead of
    the one favourite it should be -- reproduced for real while building
    this launcher (saw "favourites=_, d, e, i, idt_sensitivity, n, s, t, v,
    y" on disk after exactly one favourite + a restart)."""
    from PySide6.QtCore import QSettings
    ini_path = str(tmp_path / "single_fav_settings.ini")

    def _factory(*_args, **_kwargs):
        return QSettings(ini_path, QSettings.Format.IniFormat)

    monkeypatch.setattr("mso_studio.launcher.window.QSettings", _factory)

    first = _make_window(qapp)
    first._toggle_favourite("idt_sensitivity")
    first.close()

    second = _make_window(qapp)  # simulates reopening the launcher fresh
    assert second._favourites == {"idt_sensitivity"}


def test_hero_shows_selected_entry(qapp):
    window = _make_window(qapp)
    window._select_entry("mechanism_check")
    assert window.hero.title_label.text() == "Mechanism Health-Check"


def test_launch_flips_hero_to_stop_and_tile_badge(qapp, fake_popen):
    window = _make_window(qapp)
    window._select_entry("mechanism_check")

    window._on_launch_requested("mechanism_check")

    assert window._launch_manager.is_running("mechanism_check")
    assert window.hero.launch_button.text() == "■  STOP"
    assert window._tiles["mechanism_check"].property("running") is True
    assert "mechanism_check" in window._recent


def test_stop_flips_hero_back_to_launch(qapp, fake_popen):
    window = _make_window(qapp)
    window._select_entry("mechanism_check")
    window._on_launch_requested("mechanism_check")

    window._on_stop_requested("mechanism_check")

    assert not window._launch_manager.is_running("mechanism_check")
    assert window.hero.launch_button.text() == "▶  LAUNCH"


def test_poll_updates_running_label(qapp, fake_popen):
    window = _make_window(qapp)
    window._on_launch_requested("mechanism_check")
    window._on_poll()
    assert window.running_label.text() == "1 running"


def test_quick_death_surfaces_a_status_message(qapp, fake_popen):
    """Phase 1 code-review fix: a process that exits within a couple
    seconds of launching (e.g. a broken-but-present .venv-gui interpreter)
    must not just silently disappear from the RUNNING badge -- it should
    say so in the status bar."""
    window = _make_window(qapp)
    window._on_launch_requested("mechanism_check")
    window._launch_manager._running["mechanism_check"].process._alive = False  # dies immediately

    window._on_poll()

    assert "mechanism_check" not in window._launch_manager.running_ids()
    assert "crashed on startup" in window.statusBar().currentMessage()


def test_normal_length_run_does_not_surface_a_crash_message(qapp, fake_popen, monkeypatch):
    window = _make_window(qapp)
    window._on_launch_requested("mechanism_check")
    monkeypatch.setattr(
        "mso_studio.launcher.process.LaunchManager.last_uptime_s",
        lambda self, entry_id: 45.0,
    )
    window._launch_manager._running["mechanism_check"].process._alive = False

    window._on_poll()

    assert "crashed on startup" not in window.statusBar().currentMessage()


def test_doctor_entry_opens_doctor_dialog_not_a_launch_error(qapp, monkeypatch):
    """Regression test (Phase 0 bug fix): the Doctor catalog entry has
    runner=RUNNER_INTERNAL and no subprocess to launch. It used to stay
    LAUNCH-enabled and raise a "no subprocess to launch" warning dialog when
    clicked; it must now open the Doctor panel instead, and never show the
    warning dialog at all."""
    from PySide6.QtWidgets import QMessageBox
    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: shown.append(a))

    window = _make_window(qapp)
    window._select_entry("doctor")
    assert window.hero.launch_button.text() == "⚙  OPEN"

    window._on_launch_requested("doctor")  # must not raise

    assert len(shown) == 0
    assert window._doctor_dialog is not None
    assert window._doctor_dialog.isVisible()
