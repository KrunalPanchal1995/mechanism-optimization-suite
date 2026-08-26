"""
Regression guard for the "text masked by black background" bug: verifies the
application QPalette is actually applied (not just a stylesheet), that every
required foreground/background token pair clears its WCAG contrast ratio in
both themes, and that the old blanket transparent-QWidget rule is gone.
"""
import re

import pytest

from mso_studio.gui.tokens import DARK, LIGHT, check_required_pairs


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.mark.parametrize("tokens,label", [(DARK, "dark"), (LIGHT, "light")])
def test_required_contrast_pairs_pass(tokens, label):
    failures = check_required_pairs(tokens)
    assert not failures, (
        f"{label} theme fails WCAG contrast for: "
        + ", ".join(f"{fg}/{bg} needs {mn}:1, got {ac:.2f}:1" for fg, bg, mn, ac in failures)
    )


@pytest.mark.parametrize("mode", ["dark", "light"])
def test_apply_theme_actually_sets_the_palette(qapp, mode):
    from PySide6.QtGui import QPalette

    from mso_studio.gui.theme import apply_theme
    from mso_studio.gui.tokens import tokens_for

    apply_theme(qapp, mode)
    tokens = tokens_for(mode)
    window_color = qapp.palette().color(QPalette.ColorRole.Window)
    assert window_color.name() == tokens.bg, (
        "app.palette() does not reflect the theme -- apply_theme() is not "
        "calling app.setPalette(), so unstyled widgets (menus/tooltips/"
        "popups/tables) will fall back to Qt's default light palette."
    )
    text_color = qapp.palette().color(QPalette.ColorRole.WindowText)
    assert text_color.name() == tokens.text


def test_no_global_transparent_widget_rule():
    from mso_studio.gui.theme import build_qss

    qss = build_qss("dark")
    qss_no_comments = re.sub(r"/\*.*?\*/", "", qss, flags=re.S)
    # A bare "QWidget { background: transparent }" rule (no object name, no
    # descendant selector, not part of a comma/">"-joined selector list) is
    # what made unstyled popups/frames paint through to raw black on X11
    # without a compositor. Scoped rules like
    # "QScrollArea, QScrollArea > QWidget { background: transparent }" are
    # fine and expected; only the completely generic bare selector is banned.
    assert not re.search(r"(?<![#\w>,])QWidget\s*\{\s*background:\s*transparent", qss_no_comments)


@pytest.mark.parametrize("mode", ["dark", "light"])
def test_previously_missing_selectors_now_present(mode):
    from mso_studio.gui.theme import build_qss

    qss = build_qss(mode)
    for selector in (
        "QMenuBar", "QMenu", "QToolTip", "QToolBar", "QToolButton",
        "QTableView", "QHeaderView::section", "QComboBox QAbstractItemView",
        "QCheckBox", "QRadioButton", "QScrollBar:vertical", "QSlider",
        "QMessageBox",
    ):
        assert selector in qss, f"{selector} has no QSS rule in {mode} mode"


def test_ui_family_is_wired_into_the_global_rule():
    from mso_studio.gui.theme import build_qss
    from mso_studio.gui.tokens import UI_FAMILY

    qss = build_qss("dark")
    assert UI_FAMILY in qss


@pytest.mark.parametrize("scale", ["compact", "normal", "large", "xlarge"])
def test_font_scale_changes_base_size(scale):
    from mso_studio.gui.theme import build_qss

    qss = build_qss("dark", font_scale=scale)
    match = re.search(r"\*\s*\{\s*font-family:[^}]*font-size:\s*(\d+)px", qss, re.S)
    assert match is not None
    size = int(match.group(1))
    if scale == "compact":
        assert size < 13
    elif scale == "normal":
        assert size == 13
    elif scale == "large":
        assert size > 13
    elif scale == "xlarge":
        assert size > 15


@pytest.mark.parametrize("accent_hex", ["#22c55e", "#ef4444", "#9333ea"])
def test_custom_accent_override_is_applied_and_stays_readable(accent_hex):
    from mso_studio.gui.theme import build_qss
    from mso_studio.gui.tokens import contrast_ratio, with_accent, DARK

    qss = build_qss("dark", accent_hex=accent_hex)
    assert accent_hex in qss
    overridden = with_accent(DARK, accent_hex)
    assert contrast_ratio(overridden.text_on_accent, overridden.accent) >= 4.5


def test_custom_font_family_overrides_ui_family():
    from mso_studio.gui.theme import build_qss

    qss = build_qss("dark", font_family="Comic Sans MS")
    match = re.search(r"\*\s*\{\s*font-family:\s*([^;]+);", qss)
    assert match is not None
    assert match.group(1).strip() == "Comic Sans MS"


def test_compact_density_reduces_button_padding():
    from mso_studio.gui.theme import build_qss

    comfortable = build_qss("dark", density="comfortable")
    compact = build_qss("dark", density="compact")

    def button_padding(qss: str) -> int:
        match = re.search(r"QPushButton\s*\{[^}]*padding:\s*(\d+)px", qss, re.S)
        assert match is not None
        return int(match.group(1))

    assert button_padding(compact) < button_padding(comfortable)


def test_apply_theme_accepts_preferences_overrides(qapp):
    from mso_studio.gui.theme import apply_theme

    # Must not raise -- this is the call path PreferencesDialog will use.
    apply_theme(qapp, "dark", font_scale="large", accent_hex="#22c55e",
                font_family="DejaVu Sans", density="compact")
    assert "#22c55e" in qapp.styleSheet()
