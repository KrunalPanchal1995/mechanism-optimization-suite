"""
mso_studio.gui.theme
======================
QSS + QPalette theming, built from the design tokens in
``mso_studio.gui.tokens`` (itself ported from the Tkinter monolith
``generate_target_opt.py`` via ``src/gui_common/theme.py``).

Historically this module only ever set a stylesheet
(``app.setStyleSheet(...)``) and never touched the application's
``QPalette`` or ``QStyle``. That is the root cause of the "text on a black
background" bug: Fusion's *default* palette is light (``Window #efefef``,
``WindowText #000000``), and any widget that paints from the palette rather
than from QSS -- ``QMenu``, ``QToolTip``, a combobox's popup
``QAbstractItemView``, ``QTableWidget``, disabled states -- rendered
near-white text from the ``*`` QSS rule on top of that light palette, or, for
the class of widget covered by the old blanket ``QWidget { background:
transparent }`` rule, on raw black (no compositor to show anything else
through). ``apply_theme`` now builds a real ``QPalette`` from the same
tokens and applies ``app.setStyle("Fusion")`` + ``app.setPalette(pal)``
*before* the stylesheet, so every unstyled widget already starts from a
correct base instead of relying on QSS to override every possible role.
"""
from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from mso_studio.gui.tokens import (
    DARK,
    DENSITY_SCALES,
    FONT_SCALES,
    FONT_SIZES,
    LIGHT,
    MONO_FAMILY,
    UI_FAMILY,
    Mode,
    Tokens,
    tokens_for,
    with_accent,
)

# Re-exported for backward compatibility: existing call sites
# (launcher/art.py, tiles.py, boot.py, welcome.py, gui/main_window.py) import
# `palette_for`, `DARK`, `LIGHT`, `MONO_FAMILY`, `FONT_SIZES`, `Mode` from
# this module and index the result with ["bg"], ["text"], etc.
__all__ = [
    "DARK", "LIGHT", "MONO_FAMILY", "UI_FAMILY", "FONT_SIZES", "Mode",
    "system_prefers_dark", "palette_for", "tokens_for", "build_qss",
    "build_qpalette", "apply_theme",
]


def system_prefers_dark() -> bool:
    """Best-effort OS dark-mode detection via Qt's palette (Qt 6.5+ follows it)."""
    app = QApplication.instance()
    if app is None:
        return True
    hint = app.styleHints()
    scheme = getattr(hint, "colorScheme", None)
    if scheme is not None:
        try:
            from PySide6.QtCore import Qt as _Qt
            return scheme() == _Qt.ColorScheme.Dark
        except Exception:
            pass
    # Fallback heuristic: dark if the window background is darker than mid-grey.
    color = app.palette().color(QPalette.ColorRole.Window)
    return color.lightness() < 128


def palette_for(mode: Mode) -> dict:
    """Dict form of the tokens for *mode* -- kept for existing ["key"] call sites."""
    return tokens_for(mode, system_prefers_dark()).as_dict()


def build_qpalette(tokens: Tokens) -> QPalette:
    """A full QPalette so unstyled Qt-native widgets (QMenu, QToolTip, popups,
    tables, disabled states) render correctly even where no QSS rule applies."""
    pal = QPalette()

    def c(hexval: str) -> QColor:
        return QColor(hexval)

    active = {
        QPalette.ColorRole.Window: tokens.bg,
        QPalette.ColorRole.WindowText: tokens.text,
        QPalette.ColorRole.Base: tokens.entry_bg,
        QPalette.ColorRole.AlternateBase: tokens.surface_raised,
        QPalette.ColorRole.Text: tokens.text,
        QPalette.ColorRole.Button: tokens.surface_raised,
        QPalette.ColorRole.ButtonText: tokens.text,
        QPalette.ColorRole.BrightText: tokens.error,
        QPalette.ColorRole.Highlight: tokens.selection_bg,
        QPalette.ColorRole.HighlightedText: tokens.selection_text,
        QPalette.ColorRole.ToolTipBase: tokens.surface_raised,
        QPalette.ColorRole.ToolTipText: tokens.text,
        QPalette.ColorRole.PlaceholderText: tokens.text_faint,
        QPalette.ColorRole.Link: tokens.accent,
    }
    for role, hexval in active.items():
        pal.setColor(QPalette.ColorGroup.Active, role, c(hexval))
        pal.setColor(QPalette.ColorGroup.Inactive, role, c(hexval))

    disabled = {
        QPalette.ColorRole.Window: tokens.bg,
        QPalette.ColorRole.WindowText: tokens.disabled_text,
        QPalette.ColorRole.Base: tokens.disabled_bg,
        QPalette.ColorRole.Text: tokens.disabled_text,
        QPalette.ColorRole.Button: tokens.disabled_bg,
        QPalette.ColorRole.ButtonText: tokens.disabled_text,
        QPalette.ColorRole.Highlight: tokens.disabled_bg,
        QPalette.ColorRole.HighlightedText: tokens.disabled_text,
        QPalette.ColorRole.ToolTipBase: tokens.disabled_bg,
        QPalette.ColorRole.ToolTipText: tokens.disabled_text,
    }
    for role, hexval in disabled.items():
        pal.setColor(QPalette.ColorGroup.Disabled, role, c(hexval))

    return pal


def build_qss(
    mode: Mode = "system", font_scale: str = "normal",
    accent_hex: str = None, font_family: str = None, density: str = "comfortable",
) -> str:
    """Render the full application stylesheet for *mode* at *font_scale*
    ("compact" | "normal" | "large" | "xlarge"). *accent_hex*, if given,
    overrides the theme's accent family (Preferences' custom color option);
    *font_family*, if given, overrides the UI font stack; *density*
    ("comfortable" | "compact") scales button/input/list-item padding."""
    tokens = tokens_for(mode, system_prefers_dark())
    if accent_hex:
        tokens = with_accent(tokens, accent_hex)
    p = tokens.as_dict()
    scale = FONT_SCALES.get(font_scale, 1.0)
    base_px = round(FONT_SIZES["body"] * scale) - 1  # body=14 -> base 13, unchanged at "normal"
    d_scale = DENSITY_SCALES.get(density, 1.0)
    family = font_family or UI_FAMILY

    def px(size_key: str) -> int:
        return max(9, round(FONT_SIZES[size_key] * scale))

    def dpx(value: float) -> int:
        return max(1, round(value * d_scale))

    return f"""
    * {{
        font-family: {family};
        font-size: {base_px}px;
        color: {p['text']};
    }}
    QMainWindow, QDialog {{
        background: {p['bg']};
    }}
    /* No blanket "QWidget {{ background: transparent }}" rule here (removed --
       it made every unstyled QWidget-derived popup/frame paint through to
       whatever is behind it, which on X11 without a compositor is raw black).
       Widgets that genuinely need a transparent background are opted in by
       objectName below. The QPalette built in build_qpalette() is now the
       correct base for everything else. */
    QWidget#Sidebar {{
        background: {p['surface']};
        border-right: 1px solid {p['border']};
    }}
    QScrollArea, QScrollArea > QWidget, QScrollArea > QWidget > QWidget#ScrollContent {{
        background: transparent;
        border: none;
    }}
    QListWidget#NavList {{
        background: {p['surface']};
        border: none;
        outline: none;
        padding: 8px;
    }}
    QListWidget#NavList::item {{
        padding: {dpx(10)}px {dpx(12)}px;
        border-radius: 6px;
        margin: 2px 0;
        color: {p['text_dim']};
    }}
    QListWidget#NavList::item:selected {{
        background: {p['surface_raised']};
        color: {p['accent']};
        font-weight: 600;
    }}
    QListWidget#NavList::item:hover:!selected {{
        background: {p['surface_raised']};
    }}
    QGroupBox {{
        border: none;
        margin-top: 18px;
        font-weight: 600;
        color: {p['text']};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 0px;
        padding: 0 0 6px 0;
        color: {p['accent2']};
    }}
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
        background: {p['entry_bg']};
        color: {p['text']};
        border: 1px solid {p['border']};
        border-radius: 4px;
        padding: {dpx(4)}px {dpx(6)}px;
        selection-background-color: {p['selection_bg']};
        selection-color: {p['selection_text']};
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
        border: 1px solid {p['accent']};
    }}
    QLineEdit[invalid="true"] {{
        border: 1px solid {p['error']};
    }}
    QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
        background: {p['disabled_bg']};
        color: {p['disabled_text']};
    }}
    QPushButton {{
        background: {p['surface_raised']};
        color: {p['text']};
        border: 1px solid {p['border']};
        border-radius: 4px;
        padding: {dpx(6)}px {dpx(14)}px;
    }}
    QPushButton:hover {{
        background: {p['border']};
    }}
    QPushButton:disabled {{
        color: {p['disabled_text']};
        background: {p['disabled_bg']};
    }}
    QPushButton#PrimaryAction {{
        background: {p['accent']};
        color: {p['text_on_accent']};
        font-weight: 600;
        padding: {dpx(10)}px {dpx(16)}px;
        border: none;
    }}
    QPushButton#PrimaryAction:hover {{
        background: {p['accent_hover']};
    }}
    QPushButton#PrimaryAction:pressed {{
        background: {p['accent_pressed']};
    }}
    QPushButton#PrimaryAction:disabled {{
        background: {p['surface_raised']};
        color: {p['text_dim']};
    }}
    QPushButton#DangerAction {{
        background: {p['error']};
        color: {p['text_on_accent']};
        border: none;
    }}
    QTabWidget::pane {{
        border: 1px solid {p['border']};
        border-radius: 4px;
        top: -1px;
    }}
    QTabBar::tab {{
        background: {p['surface']};
        color: {p['text_dim']};
        padding: 8px 16px;
        border: 1px solid {p['border']};
        border-bottom: none;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
    }}
    QTabBar::tab:selected {{
        background: {p['bg']};
        color: {p['accent']};
        font-weight: 600;
    }}
    QStatusBar {{
        background: {p['surface']};
        border-top: 1px solid {p['border']};
        color: {p['text_dim']};
    }}
    QSplitter::handle {{
        background: {p['border']};
    }}
    QLabel#SectionHeader {{
        color: {p['text']};
        font-weight: 600;
        font-size: {px('body')}px;
    }}
    QLabel#Caption {{
        color: {p['text_dim']};
        font-size: {px('small')}px;
    }}
    QLabel#Caption[clickable="true"] {{
        padding: 7px 4px;
    }}
    QLabel#StatusSuccess {{ color: {p['success']}; }}
    QLabel#StatusWarning {{ color: {p['warning']}; }}
    QLabel#StatusError   {{ color: {p['error']}; }}
    QPlainTextEdit#LogView, QTextEdit#LogView {{
        font-family: {MONO_FAMILY};
        background: {p['entry_bg']};
        color: {p['text']};
    }}
    QLabel#MonoSmall {{
        font-family: {MONO_FAMILY};
        font-size: {px('small')}px;
    }}
    QProgressBar {{
        border: 1px solid {p['border']};
        border-radius: 4px;
        background: {p['entry_bg']};
        color: {p['text']};
        text-align: center;
    }}
    QProgressBar::chunk {{
        background: {p['accent']};
    }}

    /* ── Previously-missing selectors: menus, tooltips, toolbars, tables,
       checkable inputs, scrollbars, sliders, message boxes ────────────── */
    QMenuBar {{
        background: {p['surface']};
        color: {p['text']};
        border-bottom: 1px solid {p['border']};
    }}
    QMenuBar::item {{
        background: transparent;
        padding: 4px 10px;
    }}
    QMenuBar::item:selected {{
        background: {p['surface_raised']};
        color: {p['accent']};
    }}
    QMenu {{
        background: {p['surface_raised']};
        color: {p['text']};
        border: 1px solid {p['border']};
        padding: 4px;
    }}
    QMenu::item {{
        padding: {dpx(6)}px {dpx(24)}px {dpx(6)}px {dpx(12)}px;
        border-radius: 4px;
    }}
    QMenu::item:selected {{
        background: {p['selection_bg']};
        color: {p['selection_text']};
    }}
    QMenu::item:disabled {{
        color: {p['disabled_text']};
    }}
    QMenu::separator {{
        height: 1px;
        background: {p['border']};
        margin: 4px 6px;
    }}
    QToolTip {{
        background: {p['surface_raised']};
        color: {p['text']};
        border: 1px solid {p['border_strong']};
        padding: 4px 8px;
    }}
    QToolBar {{
        background: {p['surface']};
        border: none;
        spacing: 4px;
    }}
    QToolButton {{
        background: transparent;
        color: {p['text']};
        border: 1px solid transparent;
        border-radius: 4px;
        padding: 4px 8px;
    }}
    QToolButton:hover {{
        background: {p['surface_raised']};
        border: 1px solid {p['border']};
    }}
    QToolButton:pressed, QToolButton:checked {{
        background: {p['accent']};
        color: {p['text_on_accent']};
    }}
    QTableView, QTableWidget, QListView, QTreeView {{
        background: {p['entry_bg']};
        alternate-background-color: {p['surface']};
        color: {p['text']};
        gridline-color: {p['border']};
        border: 1px solid {p['border']};
        selection-background-color: {p['selection_bg']};
        selection-color: {p['selection_text']};
    }}
    QTableView::item:selected, QTableWidget::item:selected,
    QListView::item:selected, QTreeView::item:selected {{
        background: {p['selection_bg']};
        color: {p['selection_text']};
    }}
    QHeaderView::section {{
        background: {p['surface_raised']};
        color: {p['text']};
        padding: 6px 8px;
        border: none;
        border-right: 1px solid {p['border']};
        border-bottom: 1px solid {p['border']};
        font-weight: 600;
    }}
    QComboBox QAbstractItemView {{
        background: {p['surface_raised']};
        color: {p['text']};
        border: 1px solid {p['border']};
        selection-background-color: {p['selection_bg']};
        selection-color: {p['selection_text']};
        outline: none;
    }}
    QCheckBox, QRadioButton {{
        color: {p['text']};
        spacing: 6px;
    }}
    QCheckBox::indicator, QRadioButton::indicator {{
        width: 15px;
        height: 15px;
        border: 1px solid {p['border_strong']};
        background: {p['entry_bg']};
    }}
    QCheckBox::indicator {{
        border-radius: 3px;
    }}
    QRadioButton::indicator {{
        border-radius: 8px;
    }}
    QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
        background: {p['accent']};
        border: 1px solid {p['accent']};
    }}
    QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
        background: {p['disabled_bg']};
        border: 1px solid {p['border']};
    }}
    QScrollBar:vertical {{
        background: transparent;
        width: 12px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {p['border_strong']};
        border-radius: 5px;
        min-height: 24px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {p['accent']};
    }}
    QScrollBar:horizontal {{
        background: transparent;
        height: 12px;
        margin: 0;
    }}
    QScrollBar::handle:horizontal {{
        background: {p['border_strong']};
        border-radius: 5px;
        min-width: 24px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: {p['accent']};
    }}
    QScrollBar::add-line, QScrollBar::sub-line {{
        height: 0px;
        width: 0px;
    }}
    QSlider::groove:horizontal {{
        height: 4px;
        background: {p['border']};
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: {p['accent']};
        width: 14px;
        margin: -6px 0;
        border-radius: 7px;
    }}
    QMessageBox {{
        background: {p['bg']};
    }}
    QMessageBox QLabel {{
        color: {p['text']};
    }}

    /* ── launcher_qt.py: Steam-style app library ─────────────────────── */
    QListWidget#StageList {{
        background: {p['surface']};
        border: none;
        outline: none;
        padding: 8px;
    }}
    QListWidget#StageList::item {{
        padding: {dpx(12)}px {dpx(12)}px;
        border-radius: 6px;
        margin: 1px 0;
        color: {p['text_dim']};
        font-size: {px('small')}px;
    }}
    QListWidget#StageList::item:selected {{
        background: {p['surface_raised']};
        color: {p['accent']};
        font-weight: 600;
    }}
    QListWidget#StageList::item:hover:!selected {{
        background: {p['surface_raised']};
    }}
    QLabel#SidebarSectionLabel {{
        color: {p['text_dim']};
        font-size: {px('micro')}px;
        font-weight: 700;
        letter-spacing: 1px;
        padding: 4px 6px 0 6px;
    }}
    QLineEdit#SearchBox {{
        background: {p['entry_bg']};
        border: 1px solid {p['border']};
        border-radius: 14px;
        padding: 9px 12px;
        font-size: {px('small')}px;
    }}
    QLineEdit#SearchBox:focus {{
        border: 1px solid {p['accent']};
    }}
    QFrame#HeroPanel {{
        background: {p['surface']};
        border: 1px solid {p['border']};
        border-radius: 10px;
    }}
    QFrame#AppTile, QFrame#StepCard {{
        background: {p['surface']};
        border: 1px solid {p['border']};
        border-radius: 10px;
    }}
    QFrame#AppTile:hover, QFrame#StepCard:hover {{
        border: 1px solid {p['accent']};
    }}
    QFrame#AppTile:focus, QFrame#StepCard:focus {{
        border: 2px solid {p['accent']};
    }}
    QFrame#AppTile[running="true"], QFrame#StepCard[running="true"] {{
        border: 1px solid {p['success']};
    }}
    QLabel#TileTitle {{
        color: {p['text']};
        font-weight: 600;
        font-size: {px('lead')}px;
        letter-spacing: 0.1px;
    }}
    QLabel#HeroTitle {{
        color: {p['text']};
        font-weight: 700;
        font-size: {px('hero')}px;
    }}
    QLabel#SectionHeader[spaced="true"] {{
        margin-top: 8px;
    }}
    QLabel#TileStage {{
        color: {p['accent']};
        font-weight: 700;
        font-size: {px('micro')}px;
        letter-spacing: 1.2px;
    }}
    QLabel#TileTagline, QLabel#TileMeta {{
        color: {p['text_dim']};
        font-size: {px('small')}px;
    }}
    QLabel#RunningBadge {{
        color: {p['success']};
        font-size: {px('micro')}px;
        font-weight: 600;
        background: {p['scrim']};
        border-radius: 3px;
        padding: 1px 5px;
    }}
    QLabel#FavouriteStar {{
        color: {p['warning']};
        font-size: 16px;
        background: {p['scrim']};
        border-radius: 8px;
        padding: 1px 3px;
    }}
    QWidget#StatusStrip {{
        background: {p['surface']};
        border-top: 1px solid {p['border']};
    }}
    QLabel#LedOk {{
        color: {p['success']};
    }}
    QLabel#LedWarn {{
        color: {p['warning']};
    }}
    QLabel#LedFail {{
        color: {p['error']};
    }}

    /* ── Splash screen ─────────────────────────────────────────────────── */
    QLabel#SplashSubtitle {{
        color: {p['text_dim']};
        font-size: {px('micro')}px;
        font-weight: 600;
        letter-spacing: 2px;
    }}
    QLabel#SplashLog {{
        font-family: {MONO_FAMILY};
        color: {p['text']};
        font-size: {px('small')}px;
    }}
    QProgressBar#SplashProgress {{
        border: none;
        border-radius: 3px;
        background: {p['surface_raised']};
    }}
    QProgressBar#SplashProgress::chunk {{
        border-radius: 3px;
        background: {p['accent']};
    }}
    QLabel#SplashHint {{
        color: {p['text_dim']};
        font-size: {px('micro') - 1}px;
        margin-top: 6px;
    }}

    /* ── Welcome dialog ───────────────────────────────────────────────── */
    QLabel#WelcomeTitle {{
        font-size: {px('lead') + 4}px;
        font-weight: 700;
        color: {p['text']};
    }}
    QLabel#WelcomeBody {{
        font-size: {px('body')}px;
        color: {p['text']};
    }}
    QLabel#PageDot {{
        color: {p['text_faint']};
    }}
    QLabel#PageDot[active="true"] {{
        color: {p['accent']};
    }}

    /* ── Workbench step cards (workbench_view.py) ─────────────────────── */
    QLabel#StepCardTitle {{
        color: {p['text']};
        font-weight: 600;
        font-size: {px('small')}px;
    }}
    QLabel#StepTitle {{
        font-size: {px('lead')}px;
        font-weight: 700;
        color: {p['text']};
    }}

    /* ── Workbench page (project step chain) ──────────────────────────── */
    QWidget#WorkbenchHeader {{
        background: {p['surface']};
        border: 1px solid {p['border']};
        border-radius: 8px;
    }}
    QLabel#WorkbenchProjectName {{
        color: {p['text']};
        font-weight: 700;
        font-size: {px('title')}px;
    }}
    QLabel#StepArrow {{
        color: {p['text_dim']};
        font-size: {px('title')}px;
        font-weight: 700;
    }}
    QLabel#StepIndex {{
        color: {p['text_dim']};
        font-size: {px('micro')}px;
        font-weight: 700;
    }}
    QLabel#StepStatusPending {{
        color: {p['text_dim']};
        font-size: {px('micro')}px;
        font-weight: 600;
    }}
    QLabel#StepStatusRunning {{
        color: {p['accent']};
        font-size: {px('micro')}px;
        font-weight: 600;
    }}
    QLabel#StepStatusDone {{
        color: {p['success']};
        font-size: {px('micro')}px;
        font-weight: 600;
    }}
    QLabel#StepStatusFailed {{
        color: {p['error']};
        font-size: {px('micro')}px;
        font-weight: 600;
    }}
    QLabel#StepStatusUnknown {{
        color: {p['text_dim']};
        font-size: {px('micro')}px;
        font-weight: 600;
    }}

    /* ── Module hero panel (title stays objectName "SectionHeader" --
       tests and RunSummaryCard locate the hero title by that name) ───── */
    QPushButton#LearnMoreToggle {{
        text-align: left;
        padding: 0;
        border: none;
        background: transparent;
        color: {p['accent']};
    }}
    QPushButton#LearnMoreToggle:hover {{
        color: {p['accent_hover']};
        text-decoration: underline;
    }}

    /* ── Module help panel (F1) ───────────────────────────────────────── */
    QLabel#HelpDetailTitle {{
        color: {p['text']};
        font-weight: 700;
        font-size: {px('title') + 5}px;
    }}
    QLabel#HelpSectionTitle {{
        color: {p['accent2']};
        font-weight: 700;
        font-size: {px('small')}px;
        letter-spacing: 1px;
    }}
    QLabel#HelpBody {{
        color: {p['text']};
        font-size: {px('body')}px;
    }}
    QLabel#HelpMeta {{
        color: {p['text_dim']};
        font-size: {px('small')}px;
    }}
    QPushButton#HelpLinkButton {{
        text-align: left;
        padding: 0;
        border: none;
        background: transparent;
        color: {p['accent']};
        font-size: {px('small')}px;
        text-decoration: underline;
    }}
    QPushButton#HelpLinkButton:hover {{
        color: {p['accent_hover']};
    }}
    """


def apply_theme(
    app: QApplication, mode: Mode = "system", font_scale: str = "normal",
    accent_hex: str = None, font_family: str = None, density: str = "comfortable",
) -> None:
    """Apply *mode* ("system" | "light" | "dark") to the whole application,
    with optional Preferences-dialog overrides: a custom *accent_hex*, a
    custom *font_family*, and *density* ("comfortable" | "compact").

    Sets the QStyle + a real QPalette *before* the stylesheet so every
    unstyled Qt-native widget (menus, tooltips, popups, tables, disabled
    controls) starts from the correct base colors rather than depending on
    QSS to override every possible palette role.
    """
    tokens = tokens_for(mode, system_prefers_dark())
    if accent_hex:
        tokens = with_accent(tokens, accent_hex)
    app.setStyle("Fusion")
    app.setPalette(build_qpalette(tokens))
    app.setStyleSheet(build_qss(mode, font_scale, accent_hex, font_family, density))
