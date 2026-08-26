"""
mso_studio.launcher.workbench_view
=====================================
Qt presentation of the Workbench: an Ansys-Fluent-schematic-style chain of
step cards (Convert -> Check -> Fuse -> ... -> Forward UQ) that the user
composes for a project, with each card's status derived read-only from the
project's run history via ``project_link``.

The data model (:mod:`mso_studio.launcher.workbench`) and the project
connection (:mod:`mso_studio.launcher.project_link`) are both Qt-free; this
module is purely the widgets on top of them, following the same split as
``catalog.py``/``tiles.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QLineEdit, QMenu, QPushButton,
    QScrollArea, QVBoxLayout, QWidget,
)

from mso_studio.launcher.art import composed_art
from mso_studio.launcher.catalog import CATALOG, CATALOG_BY_ID, STAGES, STAGE_LABELS
from mso_studio.launcher.workbench import (
    STATUS_DONE, STATUS_FAILED, STATUS_PENDING, STATUS_RUNNING, STATUS_UNKNOWN,
    Workbench, WorkbenchStep, status_for_steps,
)

_STEP_THUMB_SIZE = 56

_STATUS_LABEL = {
    STATUS_PENDING: "PENDING",
    STATUS_RUNNING: "RUNNING",
    STATUS_DONE: "DONE",
    STATUS_FAILED: "FAILED",
    STATUS_UNKNOWN: "UNKNOWN",
}
_STATUS_OBJECT_NAME = {
    STATUS_PENDING: "StepStatusPending",
    STATUS_RUNNING: "StepStatusRunning",
    STATUS_DONE: "StepStatusDone",
    STATUS_FAILED: "StepStatusFailed",
    STATUS_UNKNOWN: "StepStatusUnknown",
}


class StepCard(QFrame):
    """One Workbench step: step number, module art, title, stage, and a
    live status pill. Deliberately a separate class from ``AppTile`` (not a
    subclass) -- ``AppTile`` carries catalog-specific behaviour (favourites)
    and 18 tests are pinned to its exact shape; a step card only needs
    select/activate/remove."""

    activated = Signal(int)
    selected = Signal(int)
    removeRequested = Signal(int)

    def __init__(self, index: int, step: WorkbenchStep, mode: str = "dark", parent=None):
        super().__init__(parent)
        self.index = index
        self.step = step
        self._mode = mode

        self.setObjectName("StepCard")
        self.setFixedSize(200, 220)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)

        top_row = QHBoxLayout()
        self.index_label = QLabel(f"STEP {index + 1}", self)
        self.index_label.setObjectName("StepIndex")
        top_row.addWidget(self.index_label)
        top_row.addStretch(1)
        self.remove_button = QLabel("✕", self)
        self.remove_button.setObjectName("Caption")
        self.remove_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.remove_button.mousePressEvent = self._on_remove_clicked  # type: ignore[assignment]
        top_row.addWidget(self.remove_button)
        layout.addLayout(top_row)

        entry = CATALOG_BY_ID.get(step.entry_id)
        self.art_label = QLabel(self)
        self.art_label.setFixedSize(_STEP_THUMB_SIZE, _STEP_THUMB_SIZE)
        if entry is not None:
            self.art_label.setPixmap(composed_art(
                entry.id, entry.icon, entry.stage,
                _STEP_THUMB_SIZE, _STEP_THUMB_SIZE, dpr=self.devicePixelRatioF()))
        layout.addWidget(self.art_label, 0, Qt.AlignmentFlag.AlignHCenter)

        self.title_label = QLabel(step.display_label(), self)
        self.title_label.setObjectName("StepCardTitle")
        self.title_label.setWordWrap(True)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.title_label)

        stage_text = STAGE_LABELS.get(entry.stage, "") if entry is not None else ""
        self.stage_label = QLabel(stage_text.upper(), self)
        self.stage_label.setObjectName("TileStage")
        self.stage_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.stage_label)

        layout.addStretch(1)

        self.status_label = QLabel("", self)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.status_label)
        self.set_status(STATUS_PENDING)

    def set_status(self, status: str) -> None:
        self.status_label.setText(f"●  {_STATUS_LABEL.get(status, status.upper())}")
        self.status_label.setObjectName(_STATUS_OBJECT_NAME.get(status, "StepStatusUnknown"))
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def _on_remove_clicked(self, event) -> None:
        self.removeRequested.emit(self.index)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        self.selected.emit(self.index)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt override
        self.activated.emit(self.index)

    def focusInEvent(self, event) -> None:  # noqa: N802 - Qt override
        self.selected.emit(self.index)
        super().focusInEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.activated.emit(self.index)
        else:
            super().keyPressEvent(event)


class WorkbenchPage(QWidget):
    """The Workbench tab: project header, the step chain, an action row,
    and a detail strip for the selected step. Every mutation autosaves
    ``workbench.yaml`` immediately -- a handful of steps isn't worth a
    dirty-state prompt, the same reasoning STAR-CCM+'s simulation tree
    autosave uses for small edits."""

    launchRequested = Signal(str)
    helpRequested = Signal(str)
    changeProjectRequested = Signal()
    dirty = Signal()

    def __init__(self, mode: str = "dark", parent=None):
        super().__init__(parent)
        self._mode = mode
        self._project_dir: Optional[str] = None
        self._project_name = ""
        self._workbench = Workbench.default_template()
        self._selected_index: Optional[int] = None
        self._cards: List[StepCard] = []
        self._run_lookup: Optional[Callable[[], List[Dict]]] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 8)
        root.setSpacing(14)

        # -- project header --------------------------------------------
        header = QWidget(self)
        header.setObjectName("WorkbenchHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 12, 16, 12)
        header_col = QVBoxLayout()
        self.project_name_label = QLabel("No project open", header)
        self.project_name_label.setObjectName("WorkbenchProjectName")
        header_col.addWidget(self.project_name_label)
        self.project_path_label = QLabel("", header)
        self.project_path_label.setObjectName("Caption")
        header_col.addWidget(self.project_path_label)
        header_layout.addLayout(header_col, 1)
        self.change_project_button = QPushButton("Open / New Project…", header)
        self.change_project_button.clicked.connect(self.changeProjectRequested.emit)
        header_layout.addWidget(self.change_project_button)
        root.addWidget(header)

        # -- step chain ----------------------------------------------------
        self.chain_scroll = QScrollArea(self)
        self.chain_scroll.setWidgetResizable(True)
        self.chain_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.chain_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.chain_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chain_container = QWidget(self.chain_scroll)
        self.chain_layout = QHBoxLayout(self.chain_container)
        self.chain_layout.setContentsMargins(4, 4, 4, 4)
        self.chain_layout.setSpacing(4)
        self.chain_layout.addStretch(1)
        self.chain_scroll.setWidget(self.chain_container)
        root.addWidget(self.chain_scroll)

        # -- action row -----------------------------------------------------
        action_row = QHBoxLayout()
        self.add_step_button = QPushButton("+  Add step", self)
        self.add_step_button.clicked.connect(self._open_add_step_menu)
        action_row.addWidget(self.add_step_button)

        self.move_left_button = QPushButton("◀  Move", self)
        self.move_left_button.clicked.connect(lambda: self._move_selected(-1))
        action_row.addWidget(self.move_left_button)

        self.move_right_button = QPushButton("Move  ▶", self)
        self.move_right_button.clicked.connect(lambda: self._move_selected(1))
        action_row.addWidget(self.move_right_button)

        self.remove_button = QPushButton("Remove", self)
        self.remove_button.clicked.connect(self._remove_selected)
        action_row.addWidget(self.remove_button)

        self.run_button = QPushButton("▶  Run step", self)
        self.run_button.setObjectName("PrimaryAction")
        self.run_button.clicked.connect(self._run_selected)
        action_row.addWidget(self.run_button)

        action_row.addStretch(1)
        self.reset_button = QPushButton("Reset to template", self)
        self.reset_button.clicked.connect(self._reset_to_template)
        action_row.addWidget(self.reset_button)
        root.addLayout(action_row)

        # -- detail strip -----------------------------------------------
        detail = QWidget(self)
        detail.setObjectName("HeroPanel")
        detail_layout = QVBoxLayout(detail)
        detail_layout.setContentsMargins(16, 12, 16, 12)
        detail_top = QHBoxLayout()
        self.detail_title_label = QLabel("Select a step", detail)
        self.detail_title_label.setObjectName("StepTitle")
        detail_top.addWidget(self.detail_title_label)
        detail_top.addStretch(1)
        self.detail_status_label = QLabel("", detail)
        self.detail_status_label.setObjectName("Caption")
        detail_top.addWidget(self.detail_status_label)
        detail_layout.addLayout(detail_top)

        self.detail_meta_label = QLabel("", detail)
        self.detail_meta_label.setObjectName("Caption")
        detail_layout.addWidget(self.detail_meta_label)

        note_row = QHBoxLayout()
        note_row.addWidget(QLabel("Note:", detail))
        self.note_edit = QLineEdit(detail)
        self.note_edit.setPlaceholderText("Why is this step here? (optional)")
        self.note_edit.editingFinished.connect(self._on_note_edited)
        note_row.addWidget(self.note_edit, 1)
        detail_layout.addLayout(note_row)

        self.learn_more_button = QPushButton("Learn more →", detail)
        self.learn_more_button.setFlat(True)
        self.learn_more_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.learn_more_button.setObjectName("LearnMoreToggle")
        self.learn_more_button.clicked.connect(self._on_learn_more_clicked)
        detail_layout.addWidget(self.learn_more_button)

        root.addWidget(detail)
        self._set_detail_enabled(False)

        self._rebuild_cards()

    # -- project wiring -----------------------------------------------------
    def set_project(self, project_dir: Optional[str], project_name: str = "",
                     run_lookup: Optional[Callable[[], List[Dict]]] = None) -> None:
        """Point the page at *project_dir* (or ``None`` to show the no-
        project state) and load its ``workbench.yaml``. *run_lookup* is a
        zero-arg callable the page can call from :meth:`refresh_status` --
        supplied by the window, which owns the open ``Project`` handle, so
        this module never has to know how a run list is actually fetched."""
        self._project_dir = project_dir
        self._project_name = project_name
        self._run_lookup = run_lookup
        if project_dir is None:
            self.project_name_label.setText("No project open")
            self.project_path_label.setText("Open or create a project to build a Workbench.")
            self._workbench = Workbench.default_template()
        else:
            self.project_name_label.setText(project_name or Path(project_dir).name)
            self.project_path_label.setText(project_dir)
            workbench_path = Path(project_dir) / "workbench.yaml"
            self._workbench = Workbench.load(project_dir)
            if not workbench_path.exists():
                # First time this project has ever had a Workbench opened --
                # persist the seeded template immediately rather than only
                # on the user's first edit, so the file on disk always
                # matches what's on screen (the same reasoning Ansys uses:
                # a project's schematic exists on disk from the moment it's
                # created, not from the moment it's first touched).
                self._workbench.save(project_dir)
        self._selected_index = None
        self._rebuild_cards()
        self._update_detail_strip()

    def refresh_status(self, runs: List[Dict]) -> None:
        statuses = status_for_steps(self._workbench.steps, runs)
        for card in self._cards:
            card.set_status(statuses.get(card.step.entry_id, STATUS_UNKNOWN))
        self._update_detail_strip()

    # -- rebuilding the chain -----------------------------------------------
    def _rebuild_cards(self) -> None:
        for card in self._cards:
            card.setParent(None)
        self._cards = []
        # index 0 is the trailing stretch; insert before it.
        while self.chain_layout.count() > 1:
            item = self.chain_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

        for i, step in enumerate(self._workbench.steps):
            if i > 0:
                arrow = QLabel("→", self.chain_container)
                arrow.setObjectName("StepArrow")
                self.chain_layout.insertWidget(self.chain_layout.count() - 1, arrow)
            card = StepCard(i, step, self._mode, self.chain_container)
            card.selected.connect(self._on_card_selected)
            card.activated.connect(self._on_card_activated)
            card.removeRequested.connect(self._on_card_remove_requested)
            self.chain_layout.insertWidget(self.chain_layout.count() - 1, card)
            self._cards.append(card)

    def _selected_step(self) -> Optional[WorkbenchStep]:
        if self._selected_index is None:
            return None
        if not (0 <= self._selected_index < len(self._workbench.steps)):
            return None
        return self._workbench.steps[self._selected_index]

    # -- selection --------------------------------------------------------
    def _on_card_selected(self, index: int) -> None:
        self._selected_index = index
        self._update_detail_strip()

    def _on_card_activated(self, index: int) -> None:
        self._selected_index = index
        self._update_detail_strip()
        self._run_selected()

    def _on_card_remove_requested(self, index: int) -> None:
        self._workbench.remove(index)
        self._selected_index = None
        self._rebuild_cards()
        self._update_detail_strip()
        self._autosave()

    def _update_detail_strip(self) -> None:
        step = self._selected_step()
        if step is None:
            self._set_detail_enabled(False)
            return
        self._set_detail_enabled(True)
        entry = CATALOG_BY_ID.get(step.entry_id)
        self.detail_title_label.setText(step.display_label())
        self.note_edit.blockSignals(True)
        self.note_edit.setText(step.note)
        self.note_edit.blockSignals(False)
        if self._selected_index is not None and self._selected_index < len(self._cards):
            self.detail_status_label.setText(
                self._cards[self._selected_index].status_label.text())
        self.detail_meta_label.setText(
            entry.description if entry is not None else step.entry_id)

    def _set_detail_enabled(self, enabled: bool) -> None:
        for widget in (self.note_edit, self.learn_more_button, self.move_left_button,
                       self.move_right_button, self.remove_button, self.run_button):
            widget.setEnabled(enabled)
        if not enabled:
            self.detail_title_label.setText("Select a step")
            self.detail_meta_label.setText("")
            self.detail_status_label.setText("")
            self.note_edit.clear()

    # -- actions ---------------------------------------------------------
    def _open_add_step_menu(self) -> None:
        menu = QMenu(self)
        for stage_id, stage_label, _from, _to in STAGES:
            submenu = menu.addMenu(stage_label)
            for entry in CATALOG:
                if entry.stage != stage_id:
                    continue
                action = submenu.addAction(entry.title)
                action.triggered.connect(lambda checked=False, eid=entry.id: self._add_step(eid))
        menu.exec(QCursor.pos())

    def _add_step(self, entry_id: str) -> None:
        self._workbench.append(entry_id)
        self._rebuild_cards()
        self._selected_index = len(self._workbench.steps) - 1
        if self._cards:
            self._cards[-1].setFocus()
        self._update_detail_strip()
        self._autosave()

    def _move_selected(self, delta: int) -> None:
        if self._selected_index is None:
            return
        new_index = self._workbench.move(self._selected_index, delta)
        self._selected_index = new_index
        self._rebuild_cards()
        self._update_detail_strip()
        self._autosave()

    def _remove_selected(self) -> None:
        if self._selected_index is None:
            return
        self._on_card_remove_requested(self._selected_index)

    def _run_selected(self) -> None:
        step = self._selected_step()
        if step is not None:
            self.launchRequested.emit(step.entry_id)

    def _reset_to_template(self) -> None:
        self._workbench = Workbench.default_template()
        self._selected_index = None
        self._rebuild_cards()
        self._update_detail_strip()
        self._autosave()

    def _on_note_edited(self) -> None:
        step = self._selected_step()
        if step is None:
            return
        step.note = self.note_edit.text()
        self._autosave()

    def _on_learn_more_clicked(self) -> None:
        step = self._selected_step()
        if step is not None:
            self.helpRequested.emit(step.entry_id)

    def _autosave(self) -> None:
        if self._project_dir is not None:
            self._workbench.save(self._project_dir)
        self.dirty.emit()
