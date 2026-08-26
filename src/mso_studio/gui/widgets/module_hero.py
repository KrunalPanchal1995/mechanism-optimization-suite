"""
mso_studio.gui.widgets.module_hero
=====================================
``ModuleHero`` -- the launcher's visual language (gradient art keyed to a
real generated result thumbnail, or a stage-colored glyph fallback) carried
into MSO Studio's module pages, replacing the plain title+description label
every page used to build inline. Single extension point: ``ModulePage``
(module_page.py) builds one of these instead of its old two-line header, so
all 9 ``ModulePage`` subclasses pick this up automatically; the 3
hand-rolled pages (automation.py, rmip_suite.py, tools.py) each swap one
call via ``make_module_hero()``.

Metadata (icon, tagline, which generated-art asset to reuse, the
plain-language "Learn more" paragraph) lives in
``mso_studio.launcher.content.MODULE_META``, keyed by Studio module id --
kept there rather than duplicated here since it's plain Python, no Qt.
"""
from __future__ import annotations

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from mso_studio.launcher.art import composed_art
from mso_studio.launcher.content import MODULE_META

_THUMB_SIZE = 72


class ModuleHero(QFrame):
    """Compact page-header banner: art thumbnail, title, tagline, the
    page's existing engineer-facing description, and an expandable
    "Learn more" paragraph in plain language."""

    def __init__(self, module_id: str, title: str, description: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("HeroPanel")
        meta = MODULE_META.get(module_id)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 12, 16, 12)
        outer.setSpacing(6)

        top_row = QHBoxLayout()
        top_row.setSpacing(12)

        self.thumb_label = QLabel(self)
        self.thumb_label.setFixedSize(_THUMB_SIZE, _THUMB_SIZE)
        if meta is not None:
            self.thumb_label.setPixmap(
                composed_art(meta.art_key, meta.icon, meta.stage, _THUMB_SIZE, _THUMB_SIZE)
            )
        top_row.addWidget(self.thumb_label)

        text_col = QVBoxLayout()
        text_col.setSpacing(2)
        title_label = QLabel(title, self)
        # Matches the plain QLabel(title) header ModulePage built inline
        # before this component existed -- some tests (and RunSummaryCard,
        # coincidentally) locate the page title by this objectName, so it
        # must keep carrying it.
        title_label.setObjectName("SectionHeader")
        text_col.addWidget(title_label)

        if meta is not None:
            tagline_label = QLabel(meta.tagline, self)
            tagline_label.setObjectName("Caption")
            tagline_label.setWordWrap(True)
            text_col.addWidget(tagline_label)

        if description:
            desc_label = QLabel(description, self)
            desc_label.setObjectName("Caption")
            desc_label.setWordWrap(True)
            text_col.addWidget(desc_label)

        text_col.addStretch(1)
        top_row.addLayout(text_col, 1)
        outer.addLayout(top_row)

        if meta is not None:
            self.learn_more_toggle = QPushButton("Learn more ▾", self)
            self.learn_more_toggle.setCheckable(True)
            self.learn_more_toggle.setFlat(True)
            self.learn_more_toggle.setObjectName("LearnMoreToggle")
            self.learn_more_toggle.clicked.connect(self._on_toggle_learn_more)
            outer.addWidget(self.learn_more_toggle)

            self.learn_more_label = QLabel(meta.learn_more, self)
            self.learn_more_label.setObjectName("Caption")
            self.learn_more_label.setWordWrap(True)
            self.learn_more_label.setVisible(False)
            outer.addWidget(self.learn_more_label)

    def _on_toggle_learn_more(self, checked: bool) -> None:
        self.learn_more_label.setVisible(checked)
        self.learn_more_toggle.setText("Learn more ▴" if checked else "Learn more ▾")


def make_module_hero(module_id: str, title: str, description: str = "", parent=None) -> ModuleHero:
    """Convenience wrapper for the 3 non-ModulePage pages (automation.py,
    rmip_suite.py, tools.py), which each hand-roll their own header rather
    than subclassing ModulePage -- one-line drop-in replacement for their
    existing 3-line QLabel block."""
    return ModuleHero(module_id, title, description, parent)
