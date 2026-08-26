"""
Pop-out tests for MainWindow (Phase 1, 2026-08 GUI integration overhaul):
double-clicking a nav entry (or Window > Open in new window) must detach
that module into its own ModuleWindow, tracked in _detached_windows and
untracked again once closed. Also covers ModuleWindow directly.
"""
import pytest


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    """Redirect every QSettings(...) call MainWindow/ModuleWindow/
    preferences make to a throwaway ini file, matching test_launcher_window.py's
    convention, so this test never reads or writes the user's real
    ~/.config state."""
    from PySide6.QtCore import QSettings
    ini_path = str(tmp_path / "main_window_test_settings.ini")

    def _factory(*_args, **_kwargs):
        return QSettings(ini_path, QSettings.Format.IniFormat)

    monkeypatch.setattr("mso_studio.gui.main_window.QSettings", _factory)
    monkeypatch.setattr("mso_studio.gui.module_window.QSettings", _factory)
    monkeypatch.setattr("mso_studio.gui.preferences.QSettings", _factory)


def _fake_page_factories(qapp):
    from PySide6.QtWidgets import QLabel

    built = {"a": 0, "b": 0}

    def make_a():
        built["a"] += 1
        return QLabel(f"page a #{built['a']}")

    def make_b():
        built["b"] += 1
        return QLabel(f"page b #{built['b']}")

    return [("mod_a", "Module A", make_a), ("mod_b", "Module B", make_b)], built


def test_double_click_detaches_into_module_window(qapp):
    from mso_studio.gui.main_window import MainWindow
    from mso_studio.gui.module_window import ModuleWindow

    factories, built = _fake_page_factories(qapp)
    window = MainWindow(factories)

    assert window._detached_windows == []
    item = window.nav_list.item(0)
    window._detach_item(item)

    assert len(window._detached_windows) == 1
    detached = window._detached_windows[0]
    assert isinstance(detached, ModuleWindow)
    assert detached.module_id == "mod_a"
    # 2, not 1: MainWindow.__init__ already built one copy for row 0 when it
    # auto-selected the first nav item on startup; _detach_item builds a
    # second, independent instance for the pop-out rather than reparenting.
    assert built["a"] == 2


def test_detached_window_is_forgotten_on_close(qapp):
    from mso_studio.gui.main_window import MainWindow

    factories, _built = _fake_page_factories(qapp)
    window = MainWindow(factories)

    window._detach_item(window.nav_list.item(0))
    assert len(window._detached_windows) == 1

    detached = window._detached_windows[0]
    detached.close()  # WA_DeleteOnClose schedules the C++ delete
    # Flush the DeferredDelete event -- a fixed processEvents() count was
    # observed flaky under heavy system load (deferred-delete not always
    # drained within 2 calls when the scheduler is starved), so poll with
    # a deadline instead of a fixed count.
    import time
    deadline = time.monotonic() + 5.0
    while window._detached_windows and time.monotonic() < deadline:
        qapp.processEvents()

    assert window._detached_windows == []


def test_open_in_new_window_menu_action_detaches_current_selection(qapp):
    from mso_studio.gui.main_window import MainWindow

    factories, _built = _fake_page_factories(qapp)
    window = MainWindow(factories)
    window.nav_list.setCurrentRow(1)

    window._detach_current()

    assert len(window._detached_windows) == 1
    assert window._detached_windows[0].module_id == "mod_b"


def test_module_window_persists_geometry_independently_per_module(qapp):
    from PySide6.QtWidgets import QLabel

    from mso_studio.gui.module_window import ModuleWindow

    w1 = ModuleWindow("mod_a", "Module A", QLabel("a"))
    w1.resize(555, 444)
    w1.close()

    w2 = ModuleWindow("mod_a", "Module A", QLabel("a again"))
    assert w2.size().width() == 555
    assert w2.size().height() == 444
