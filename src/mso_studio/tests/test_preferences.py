"""
PreferencesDialog: construction, defaults, accent/wallpaper selection, and
that Apply actually persists to QSettings and calls theme.apply_theme --
without ever touching the real user's ~/.config (QSettings is redirected to
a tmp INI file, same pattern as test_launcher_window.py).
"""
import pytest


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def redirected_settings(tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings

    ini_path = str(tmp_path / "prefs.ini")

    def _factory(*_args, **_kwargs):
        return QSettings(ini_path, QSettings.Format.IniFormat)

    monkeypatch.setattr("mso_studio.gui.preferences.QSettings", _factory)
    return ini_path


def test_dialog_constructs_with_defaults(qapp, redirected_settings):
    from mso_studio.gui.preferences import PreferencesDialog, DEFAULTS

    dialog = PreferencesDialog()
    assert dialog._prefs["theme_mode"] == DEFAULTS["theme_mode"]
    assert dialog._prefs["accent_id"] == DEFAULTS["accent_id"]
    assert dialog._prefs["wallpaper_id"] == DEFAULTS["wallpaper_id"]


def test_apply_persists_theme_mode_and_accent(qapp, redirected_settings):
    from PySide6.QtCore import QSettings

    from mso_studio.gui.preferences import PreferencesDialog

    dialog = PreferencesDialog()
    dialog.theme_combo.setCurrentIndex(2)  # "Dark"
    dialog.accent_combo.setCurrentIndex(2)  # third preset
    dialog._on_apply()

    saved = QSettings(redirected_settings, QSettings.Format.IniFormat)
    assert saved.value("theme_mode") == "dark"
    assert saved.value("accent_id") == dialog.accent_combo.currentData()


def test_apply_calls_theme_apply_theme(qapp, redirected_settings, monkeypatch):
    from mso_studio.gui.preferences import PreferencesDialog

    calls = []
    monkeypatch.setattr(
        "mso_studio.gui.preferences.theme.apply_theme",
        lambda app, **kwargs: calls.append(kwargs),
    )
    dialog = PreferencesDialog()
    dialog._on_apply()
    assert len(calls) == 1
    assert "mode" in calls[0] and "accent_hex" in calls[0]


def test_reset_defaults_restores_widgets(qapp, redirected_settings):
    from mso_studio.gui.preferences import PreferencesDialog, DEFAULTS

    dialog = PreferencesDialog()
    dialog.theme_combo.setCurrentIndex(2)
    dialog._on_reset_defaults()
    assert dialog._prefs["theme_mode"] == DEFAULTS["theme_mode"]


def test_wallpaper_combo_lists_shipped_pack(qapp, redirected_settings):
    from mso_studio.gui.preferences import PreferencesDialog
    from mso_studio.launcher.wallpapers import list_wallpapers

    dialog = PreferencesDialog()
    shipped_names = {entry.name for entry in list_wallpapers()}
    combo_labels = {dialog.wallpaper_combo.itemText(i) for i in range(dialog.wallpaper_combo.count())}
    assert shipped_names.issubset(combo_labels)
    assert "None" in combo_labels
    assert "Choose image…" in combo_labels


def test_wallpaper_changed_signal_emits_on_apply(qapp, redirected_settings):
    from mso_studio.gui.preferences import PreferencesDialog

    dialog = PreferencesDialog()
    received = []
    dialog.wallpaperChanged.connect(received.append)
    dialog._on_apply()
    assert received == [dialog._prefs["wallpaper_id"]]


def test_resolve_accent_hex_prefers_custom_over_preset():
    from mso_studio.gui.preferences import resolve_accent_hex

    prefs = {"accent_id": "__custom__", "accent_custom_hex": "#123456"}
    assert resolve_accent_hex(prefs) == "#123456"


def test_resolve_accent_hex_falls_back_to_preset():
    from mso_studio.gui.preferences import resolve_accent_hex
    from mso_studio.gui.tokens import ACCENT_PRESETS

    prefs = {"accent_id": "green", "accent_custom_hex": ""}
    assert resolve_accent_hex(prefs) == ACCENT_PRESETS["green"]
