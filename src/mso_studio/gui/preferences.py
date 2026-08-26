"""
mso_studio.gui.preferences
=============================
``PreferencesDialog`` -- the app's first appearance control beyond the
3-item View > Theme menu: theme mode, accent color (6 presets + custom,
contrast-validated), font family (UI + monospace), font scale, density, and
wallpaper (pack picker + a user-supplied image + "None"), plus a live
preview pane and a "Reset to defaults" action.

Settings are persisted under the same ``QSettings(ORG_NAME, APP_NAME)``
namespace the launcher already uses for ``theme_mode`` (``launcher/
window.py``), so opening Preferences from either the launcher or a Studio
module page reads/writes the same values.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QSettings, Signal
from PySide6.QtGui import QColor, QFontDatabase
from PySide6.QtWidgets import (
    QColorDialog, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from mso_studio.gui import theme
from mso_studio.gui.tokens import (
    ACCENT_PRESETS, contrast_ratio, tokens_for, with_accent,
)
from mso_studio.launcher.wallpapers import NONE_ID, list_wallpapers, resolve_wallpaper_path

ORG_NAME = "mechanism-optimization-suite"
APP_NAME = "mso-launcher"

_THEME_MODES = [("System", "system"), ("Light", "light"), ("Dark", "dark")]
_FONT_SCALES = [
    ("Compact", "compact"), ("Normal", "normal"),
    ("Large", "large"), ("Extra Large", "xlarge"),
]
_DENSITIES = [("Comfortable", "comfortable"), ("Compact", "compact")]
_CUSTOM_ACCENT = "__custom__"

DEFAULTS = {
    "theme_mode": "system",
    "accent_id": "blue",
    "accent_custom_hex": "",
    "font_family": "",
    "mono_font_family": "",
    "font_scale": "normal",
    "density": "comfortable",
    "wallpaper_id": NONE_ID,
}


def _settings() -> QSettings:
    return QSettings(ORG_NAME, APP_NAME)


def load_preferences() -> dict:
    """Read every persisted preference, filling in defaults for anything
    never set. Used both by PreferencesDialog and by any window that wants
    to apply the saved appearance at startup (e.g. ``launcher_qt.py``)."""
    s = _settings()
    return {key: s.value(key, default) for key, default in DEFAULTS.items()}


def resolve_accent_hex(prefs: dict) -> Optional[str]:
    """None means "no override, use the theme's built-in accent"."""
    if prefs.get("accent_id") == _CUSTOM_ACCENT and prefs.get("accent_custom_hex"):
        return prefs["accent_custom_hex"]
    return ACCENT_PRESETS.get(prefs.get("accent_id", "blue"))


def apply_preferences_to_app(app, prefs: Optional[dict] = None) -> None:
    """Apply every persisted (or given) preference to *app* in one call --
    the same path both startup and PreferencesDialog.apply() use."""
    prefs = prefs or load_preferences()
    theme.apply_theme(
        app,
        mode=prefs.get("theme_mode", "system"),
        font_scale=prefs.get("font_scale", "normal"),
        accent_hex=resolve_accent_hex(prefs),
        font_family=prefs.get("font_family") or None,
        density=prefs.get("density", "comfortable"),
    )


class PreferencesDialog(QDialog):
    """Self-contained: reads/writes QSettings and calls
    ``theme.apply_theme`` directly on Apply/OK. Emits ``wallpaperChanged``
    so a host window (the launcher) can additionally update its own
    ``BackdropWidget`` -- wallpaper choice isn't part of ``apply_theme``
    since it's launcher-window chrome, not an application-wide QSS/QPalette
    concern."""

    wallpaperChanged = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.resize(560, 520)
        self._prefs = load_preferences()
        self._build_ui()
        self._load_into_widgets()
        self._update_preview()

    # -- construction -------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        form = QFormLayout()
        form.setSpacing(10)

        self.theme_combo = QComboBox(self)
        for label, _value in _THEME_MODES:
            self.theme_combo.addItem(label)
        self.theme_combo.currentIndexChanged.connect(self._update_preview)
        form.addRow("Theme", self.theme_combo)

        accent_row = QHBoxLayout()
        self.accent_combo = QComboBox(self)
        for name in ACCENT_PRESETS:
            self.accent_combo.addItem(name.capitalize(), name)
        self.accent_combo.addItem("Custom…", _CUSTOM_ACCENT)
        self.accent_combo.currentIndexChanged.connect(self._on_accent_combo_changed)
        accent_row.addWidget(self.accent_combo, 1)
        self.accent_swatch = QLabel(self)
        self.accent_swatch.setFixedSize(24, 24)
        accent_row.addWidget(self.accent_swatch)
        form.addRow("Accent color", _wrap(accent_row))

        self.ui_font_combo = QComboBox(self)
        self.ui_font_combo.addItem("Default", "")
        for family in QFontDatabase.families():
            self.ui_font_combo.addItem(family, family)
        self.ui_font_combo.currentIndexChanged.connect(self._update_preview)
        form.addRow("Font family (UI)", self.ui_font_combo)

        self.mono_font_combo = QComboBox(self)
        self.mono_font_combo.addItem("Default", "")
        for family in QFontDatabase.families():
            self.mono_font_combo.addItem(family, family)
        form.addRow("Font family (monospace)", self.mono_font_combo)

        self.font_scale_combo = QComboBox(self)
        for label, _value in _FONT_SCALES:
            self.font_scale_combo.addItem(label)
        self.font_scale_combo.currentIndexChanged.connect(self._update_preview)
        form.addRow("Font size", self.font_scale_combo)

        self.density_combo = QComboBox(self)
        for label, _value in _DENSITIES:
            self.density_combo.addItem(label)
        self.density_combo.currentIndexChanged.connect(self._update_preview)
        form.addRow("Density", self.density_combo)

        self.wallpaper_combo = QComboBox(self)
        self.wallpaper_combo.addItem("None", NONE_ID)
        for entry in list_wallpapers():
            self.wallpaper_combo.addItem(entry.name, entry.id)
        self.wallpaper_combo.addItem("Choose image…", "__choose__")
        self.wallpaper_combo.currentIndexChanged.connect(self._on_wallpaper_combo_changed)
        form.addRow("Wallpaper (launcher chrome only)", self.wallpaper_combo)

        self.scrim_note = QLabel(self)
        self.scrim_note.setObjectName("Caption")
        self.scrim_note.setWordWrap(True)
        form.addRow("", self.scrim_note)

        root.addLayout(form)

        self.preview_frame = QFrame(self)
        self.preview_frame.setObjectName("HeroPanel")
        preview_layout = QVBoxLayout(self.preview_frame)
        self.preview_title = QLabel("Preview", self.preview_frame)
        self.preview_title.setObjectName("SectionHeader")
        preview_layout.addWidget(self.preview_title)
        self.preview_body = QLabel(
            "The quick brown fox jumps over the lazy dog.", self.preview_frame,
        )
        preview_layout.addWidget(self.preview_body)
        preview_button_row = QHBoxLayout()
        self.preview_button = QPushButton("Primary action", self.preview_frame)
        self.preview_button.setObjectName("PrimaryAction")
        preview_button_row.addWidget(self.preview_button)
        preview_button_row.addStretch(1)
        preview_layout.addLayout(preview_button_row)
        root.addWidget(self.preview_frame, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.RestoreDefaults,
        )
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self._on_apply)
        buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(
            self._on_reset_defaults)
        root.addWidget(buttons)

    def _load_into_widgets(self) -> None:
        self._set_combo_by_data(self.theme_combo, self._prefs["theme_mode"],
                                 [v for _l, v in _THEME_MODES])
        accent_id = self._prefs["accent_id"]
        self._set_combo_by_userdata(self.accent_combo, accent_id)
        self._set_combo_by_userdata(self.ui_font_combo, self._prefs["font_family"])
        self._set_combo_by_userdata(self.mono_font_combo, self._prefs["mono_font_family"])
        self._set_combo_by_data(self.font_scale_combo, self._prefs["font_scale"],
                                 [v for _l, v in _FONT_SCALES])
        self._set_combo_by_data(self.density_combo, self._prefs["density"],
                                 [v for _l, v in _DENSITIES])
        self._set_combo_by_userdata(self.wallpaper_combo, self._prefs["wallpaper_id"])

    @staticmethod
    def _set_combo_by_data(combo: QComboBox, value: str, ordered_values) -> None:
        try:
            index = ordered_values.index(value)
        except ValueError:
            index = 0
        combo.setCurrentIndex(index)

    @staticmethod
    def _set_combo_by_userdata(combo: QComboBox, value: str) -> None:
        for i in range(combo.count()):
            if combo.itemData(i) == value:
                combo.setCurrentIndex(i)
                return

    # -- accent picking -------------------------------------------------
    def _on_accent_combo_changed(self) -> None:
        if self.accent_combo.currentData() == _CUSTOM_ACCENT:
            current = QColor(self._prefs.get("accent_custom_hex") or "#4f8ef7")
            color = QColorDialog.getColor(current, self, "Choose accent color")
            if color.isValid():
                self._prefs["accent_custom_hex"] = color.name()
            else:
                # Cancelled -- fall back to the previous preset rather than
                # leaving "Custom..." selected with no color behind it.
                self._set_combo_by_userdata(self.accent_combo, self._prefs.get("accent_id", "blue"))
        self._update_preview()

    def _current_accent_hex(self) -> str:
        data = self.accent_combo.currentData()
        if data == _CUSTOM_ACCENT:
            return self._prefs.get("accent_custom_hex") or "#4f8ef7"
        return ACCENT_PRESETS.get(data, "#4f8ef7")

    # -- wallpaper picking ------------------------------------------------
    def _on_wallpaper_combo_changed(self) -> None:
        if self.wallpaper_combo.currentData() == "__choose__":
            path, _filter = QFileDialog.getOpenFileName(
                self, "Choose wallpaper image", "",
                "Images (*.png *.jpg *.jpeg *.bmp)",
            )
            if path:
                self.wallpaper_combo.insertItem(
                    self.wallpaper_combo.count() - 1, path, path)
                self.wallpaper_combo.setCurrentIndex(self.wallpaper_combo.count() - 2)
            else:
                self._set_combo_by_userdata(self.wallpaper_combo, self._prefs.get("wallpaper_id", NONE_ID))
        self._update_preview()

    # -- live preview -------------------------------------------------
    def _update_preview(self) -> None:
        mode = _THEME_MODES[self.theme_combo.currentIndex()][1]
        font_scale = _FONT_SCALES[self.font_scale_combo.currentIndex()][1]
        density = _DENSITIES[self.density_combo.currentIndex()][1]
        accent_hex = self._current_accent_hex()
        font_family = self.ui_font_combo.currentData() or None

        qss = theme.build_qss(mode, font_scale, accent_hex, font_family, density)
        self.preview_frame.setStyleSheet(qss)
        self.accent_swatch.setStyleSheet(
            f"background: {accent_hex}; border: 1px solid #00000030; border-radius: 4px;")

        wallpaper_id = self.wallpaper_combo.currentData()
        if wallpaper_id in (None, NONE_ID, "__choose__"):
            self.scrim_note.setText(
                "No wallpaper: launcher panels stay on a flat surface color.")
        else:
            tokens = tokens_for(mode)
            path = resolve_wallpaper_path(wallpaper_id)
            if path is not None and path.exists():
                from mso_studio.gui.backdrop import average_color_hex, required_scrim_alpha
                from PySide6.QtGui import QImage
                image = QImage(str(path))
                if not image.isNull():
                    avg = average_color_hex(image)
                    alpha = required_scrim_alpha(avg, tokens.text, tokens.scrim)
                    self.scrim_note.setText(
                        f"Auto-scrim: {alpha * 100:.0f}% overlay applied to keep text "
                        f"readable over this wallpaper (computed floor, not adjustable "
                        f"below this -- shown behind launcher tiles only, never behind "
                        f"forms/tables/log views).")
                    return
            self.scrim_note.setText("Wallpaper preview unavailable for this image.")

    # -- persistence -------------------------------------------------
    def _collect_prefs(self) -> dict:
        accent_data = self.accent_combo.currentData()
        return {
            "theme_mode": _THEME_MODES[self.theme_combo.currentIndex()][1],
            "accent_id": accent_data,
            "accent_custom_hex": self._prefs.get("accent_custom_hex", ""),
            "font_family": self.ui_font_combo.currentData() or "",
            "mono_font_family": self.mono_font_combo.currentData() or "",
            "font_scale": _FONT_SCALES[self.font_scale_combo.currentIndex()][1],
            "density": _DENSITIES[self.density_combo.currentIndex()][1],
            "wallpaper_id": self.wallpaper_combo.currentData(),
        }

    def _on_apply(self) -> None:
        self._prefs = self._collect_prefs()
        s = _settings()
        for key, value in self._prefs.items():
            s.setValue(key, value)
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            apply_preferences_to_app(app, self._prefs)
        self.wallpaperChanged.emit(self._prefs["wallpaper_id"])

    def _on_ok(self) -> None:
        self._on_apply()
        self.accept()

    def _on_reset_defaults(self) -> None:
        self._prefs = dict(DEFAULTS)
        self._load_into_widgets()
        self._update_preview()


def _wrap(layout) -> QWidget:
    w = QWidget()
    w.setLayout(layout)
    return w
