"""
WelcomeDialog + first-run behavior tests (Phase 4).
"""
import pytest


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_dialog_starts_on_first_page(qapp):
    from mso_studio.launcher.content import WELCOME_PAGES
    from mso_studio.launcher.welcome import WelcomeDialog

    dlg = WelcomeDialog("dark")
    assert dlg.title_label.text() == WELCOME_PAGES[0].title
    assert dlg.back_button.isEnabled() is False


def test_next_and_back_navigate(qapp):
    from mso_studio.launcher.content import WELCOME_PAGES
    from mso_studio.launcher.welcome import WelcomeDialog

    dlg = WelcomeDialog("dark")
    dlg._go_next()
    assert dlg.title_label.text() == WELCOME_PAGES[1].title
    assert dlg.back_button.isEnabled() is True

    dlg._go_back()
    assert dlg.title_label.text() == WELCOME_PAGES[0].title
    assert dlg.back_button.isEnabled() is False


def test_last_page_next_button_says_get_started_and_hides_skip(qapp):
    from mso_studio.launcher.content import WELCOME_PAGES
    from mso_studio.launcher.welcome import WelcomeDialog

    dlg = WelcomeDialog("dark")
    for _ in range(len(WELCOME_PAGES) - 1):
        dlg._go_next()
    assert dlg.next_button.text() == "Get started"
    assert not dlg.skip_button.isVisible()


def test_next_past_the_last_page_closes_the_dialog(qapp):
    from mso_studio.launcher.content import WELCOME_PAGES
    from mso_studio.launcher.welcome import WelcomeDialog
    from PySide6.QtWidgets import QDialog

    dlg = WelcomeDialog("dark")
    for _ in range(len(WELCOME_PAGES)):
        dlg._go_next()  # one extra call past the last page must not raise
    assert dlg.result() == QDialog.DialogCode.Accepted


def test_skip_accepts_immediately(qapp):
    from mso_studio.launcher.welcome import WelcomeDialog
    from PySide6.QtWidgets import QDialog

    dlg = WelcomeDialog("dark")
    dlg.skip_button.click()
    assert dlg.result() == QDialog.DialogCode.Accepted


class TestFirstRunIntegration:
    @pytest.fixture(autouse=True)
    def isolated_settings(self, monkeypatch, tmp_path):
        from PySide6.QtCore import QSettings
        ini_path = str(tmp_path / "welcome_test_settings.ini")

        def _factory(*_args, **_kwargs):
            return QSettings(ini_path, QSettings.Format.IniFormat)

        monkeypatch.setattr("mso_studio.launcher.window.QSettings", _factory)

    def _make_window(self, qapp):
        from mso_studio.launcher.process import LaunchManager
        from mso_studio.launcher.window import LauncherWindow
        return LauncherWindow(LaunchManager(), [])

    def test_first_run_marks_welcome_seen(self, qapp, monkeypatch):
        opened = []
        monkeypatch.setattr(
            "mso_studio.launcher.window.LauncherWindow._open_welcome",
            lambda self: opened.append(True),
        )
        window = self._make_window(qapp)
        assert window._settings.value("welcome_seen", False, type=bool) is False

        window._maybe_show_welcome_on_first_run()

        assert opened == [True]
        assert window._settings.value("welcome_seen", False, type=bool) is True

    def test_second_call_does_not_reopen(self, qapp, monkeypatch):
        opened = []
        monkeypatch.setattr(
            "mso_studio.launcher.window.LauncherWindow._open_welcome",
            lambda self: opened.append(True),
        )
        window = self._make_window(qapp)
        window._maybe_show_welcome_on_first_run()
        window._maybe_show_welcome_on_first_run()

        assert opened == [True]  # only the first call opened it

    def test_help_menu_welcome_tour_always_opens_regardless_of_seen_flag(self, qapp, monkeypatch):
        opened = []
        monkeypatch.setattr(
            "mso_studio.launcher.welcome.WelcomeDialog.exec",
            lambda self: opened.append(True),
        )
        window = self._make_window(qapp)
        window._settings.setValue("welcome_seen", True)

        window._open_welcome()

        assert opened == [True]
