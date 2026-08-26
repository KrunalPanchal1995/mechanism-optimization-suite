"""
mso_studio.launcher.help_panel
=================================
Full-page, per-module help -- Ansys/STAR-CCM+ style: every module documented
to the same fixed shape (Purpose / Inputs / Outputs / Prerequisites /
Typical use / Tips & pitfalls / See also), reachable by F1, from the hero
panel's "Full help" link, and from the Workbench step detail strip.

The content itself lives in ``launcher.content.MODULE_HELP`` (a plain
dataclass registry, no Qt); this module is purely the Qt presentation of it.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QScrollArea, QSizePolicy, QSplitter, QVBoxLayout, QWidget,
)

from mso_studio.launcher.art import composed_art
from mso_studio.launcher.catalog import CATALOG, CATALOG_BY_ID, STAGE_LABELS
from mso_studio.launcher.content import MODULE_HELP, MODULE_META

# The four MODULE_HELP/MODULE_META ids that exist only inside MSO Studio's
# module list (no launcher-catalog AppEntry) -- see content.py's note on the
# id mismatch between the launcher catalog and Studio's PAGE_FACTORIES.
_STUDIO_ONLY_TITLES: Dict[str, str] = {
    "sensitivity": "Sensitivity Analysis (Studio)",
    "optimization": "Optimization (Studio)",
    "automation": "Automation (Studio)",
    "tools": "Tools (Studio)",
}

# Every id MODULE_HELP documents, in a sensible display order: catalog
# entries in their declared (stage) order, then the Studio-only extras.
HELP_MODULE_IDS: List[str] = [entry.id for entry in CATALOG if entry.id in MODULE_HELP]
HELP_MODULE_IDS += [mid for mid in _STUDIO_ONLY_TITLES if mid in MODULE_HELP]


def module_title(module_id: str) -> str:
    entry = CATALOG_BY_ID.get(module_id)
    if entry is not None:
        return entry.title
    return _STUDIO_ONLY_TITLES.get(module_id, module_id)


def module_stage_label(module_id: str) -> str:
    entry = CATALOG_BY_ID.get(module_id)
    if entry is not None:
        return STAGE_LABELS.get(entry.stage, entry.stage).upper()
    return "STUDIO"


def _bulleted(values) -> str:
    return "\n".join(f"•  {v}" for v in values)


class ModuleHelpView(QWidget):
    """Renders one module's :class:`~mso_studio.launcher.content.ModuleHelp`
    record as a scrollable document. Clicking a "See also" entry emits
    ``navigateRequested`` rather than switching itself, so the dialog's
    module list (the single source of truth for "what's selected") stays in
    sync."""

    navigateRequested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll)

        body = QWidget(scroll)
        scroll.setWidget(body)
        self._layout = QVBoxLayout(body)
        self._layout.setContentsMargins(24, 20, 24, 24)
        self._layout.setSpacing(4)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.art_label = QLabel(body)
        self.art_label.setFixedHeight(140)
        self.art_label.setScaledContents(True)
        self.art_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._layout.addWidget(self.art_label)

        self.stage_label = QLabel("", body)
        self.stage_label.setObjectName("Caption")
        self._layout.addSpacing(12)
        self._layout.addWidget(self.stage_label)

        self.title_label = QLabel("", body)
        self.title_label.setObjectName("HelpDetailTitle")
        self._layout.addWidget(self.title_label)
        self._layout.addSpacing(8)

        self._module_id: Optional[str] = None
        self._section_widgets: List[QWidget] = []

    def set_module(self, module_id: str) -> None:
        help_ = MODULE_HELP.get(module_id)
        if help_ is None:
            return
        self._module_id = module_id

        entry = CATALOG_BY_ID.get(module_id)
        icon_glyph = entry.icon if entry is not None else (
            MODULE_META[module_id].icon if module_id in MODULE_META else "fa5s.cube")
        stage = entry.stage if entry is not None else "integrated"
        self.art_label.setPixmap(composed_art(
            entry.id if entry is not None else "", icon_glyph, stage,
            max(self.width(), 400), 140))

        self.stage_label.setText(module_stage_label(module_id))
        self.title_label.setText(module_title(module_id))

        for widget in self._section_widgets:
            widget.setParent(None)
        self._section_widgets.clear()

        self._add_section("PURPOSE", help_.purpose)
        if help_.inputs:
            self._add_section("INPUTS", _bulleted(help_.inputs))
        if help_.outputs:
            self._add_section("OUTPUTS", _bulleted(help_.outputs))
        if help_.prerequisites:
            self._add_section("PREREQUISITES", _bulleted(help_.prerequisites))
        self._add_section("TYPICAL USE", help_.typical_use)
        if help_.tips:
            self._add_section("TIPS & PITFALLS", _bulleted(help_.tips))
        if help_.see_also:
            self._add_see_also(help_.see_also)

    def _add_section(self, title: str, body_text: str) -> None:
        heading = QLabel(title, self)
        heading.setObjectName("HelpSectionTitle")
        self._layout.addSpacing(14)
        self._layout.addWidget(heading)
        self._section_widgets.append(heading)

        body = QLabel(body_text, self)
        body.setObjectName("HelpBody")
        body.setWordWrap(True)
        self._layout.addWidget(body)
        self._section_widgets.append(body)

    def _add_see_also(self, ids) -> None:
        heading = QLabel("SEE ALSO", self)
        heading.setObjectName("HelpSectionTitle")
        self._layout.addSpacing(14)
        self._layout.addWidget(heading)
        self._section_widgets.append(heading)

        row = QWidget(self)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(12)
        for other_id in ids:
            if other_id not in MODULE_HELP:
                continue
            link = QPushButton(module_title(other_id), row)
            link.setFlat(True)
            link.setCursor(Qt.CursorShape.PointingHandCursor)
            link.setObjectName("HelpLinkButton")
            link.clicked.connect(lambda checked=False, oid=other_id: self.navigateRequested.emit(oid))
            row_layout.addWidget(link)
        row_layout.addStretch(1)
        self._layout.addWidget(row)
        self._section_widgets.append(row)


class ModuleHelpDialog(QDialog):
    """F1 / "Full help" destination: a module picker on the left, the
    rendered :class:`ModuleHelpView` on the right. Cached and reused by the
    caller the same way ``LauncherWindow`` caches its Doctor dialog, so
    repeated F1 presses don't rebuild the widget tree."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Module Help")
        self.resize(900, 640)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.module_list = QListWidget(self)
        self.module_list.setObjectName("StageList")
        self.module_list.setFixedWidth(220)
        for module_id in HELP_MODULE_IDS:
            item = QListWidgetItem(module_title(module_id))
            item.setData(Qt.ItemDataRole.UserRole, module_id)
            self.module_list.addItem(item)
        self.module_list.currentItemChanged.connect(self._on_selection_changed)
        layout.addWidget(self.module_list)

        self.view = ModuleHelpView(self)
        self.view.navigateRequested.connect(self.show_module)
        layout.addWidget(self.view, 1)

        if HELP_MODULE_IDS:
            self.module_list.setCurrentRow(0)

    def show_module(self, module_id: str) -> None:
        for row in range(self.module_list.count()):
            item = self.module_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == module_id:
                self.module_list.setCurrentItem(item)
                return
        # Not in the list (shouldn't happen -- HELP_MODULE_IDS mirrors
        # MODULE_HELP) -- render it directly anyway rather than silently
        # doing nothing.
        self.view.set_module(module_id)

    def _on_selection_changed(self, current: Optional[QListWidgetItem], _previous) -> None:
        if current is None:
            return
        module_id = current.data(Qt.ItemDataRole.UserRole)
        self.view.set_module(module_id)
