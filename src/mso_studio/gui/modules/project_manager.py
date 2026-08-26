"""
mso_studio.gui.modules.project_manager
=========================================
Project Manager: create/open an mso_project.Project, browse its run
history and registered artifacts ("the recorder"), add notes, resume a
run's artifacts into a fresh directory, and build the cross-pipeline LaTeX
report -- the first place to do any of this without the CLI
(``mso_project/cli.py``) or leaving MSO Studio. The launcher's own
Workbench panel (``launcher.project_link``) is deliberately read-only and
launcher-scoped; this page is where the mutating operations it declines to
expose actually live.

A plain QWidget page (like ToolsPage/RmipSuitePage), not a ModulePage
subclass -- several independent actions on one open project, not one
run-and-report backend job.
"""
from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea,
    QTabWidget, QVBoxLayout, QWidget,
)

from mso_studio.core import project_manager as pm_core
from mso_studio.gui.widgets.module_hero import make_module_hero
from mso_studio.gui.widgets.results_table import ResultsTable
from mso_studio.gui.workers import call_in_thread

_RUN_HEADERS = ("run_id", "pipeline", "label", "started_at", "finished_at", "status")
_ARTIFACT_HEADERS = ("artifact_id", "run_id", "role", "kind", "path", "size_bytes", "tags")


class ProjectManagerPage(QWidget):
    statusMessage = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._project_dir: Optional[str] = None
        self._runs: List[dict] = []
        self._threads = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.hero = make_module_hero(
            "project_manager", "Project Manager",
            "Create or open a project, browse its full run history and every "
            "registered artifact, add notes, resume a run into a fresh "
            "directory, and build the cross-pipeline LaTeX report.",
            self,
        )
        outer.addWidget(self.hero)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll, 1)
        body = QWidget(scroll)
        scroll.setWidget(body)
        root = QVBoxLayout(body)

        root.addWidget(self._build_project_group())

        tabs = QTabWidget(body)
        tabs.addTab(self._build_runs_tab(), "Runs (the recorder)")
        tabs.addTab(self._build_artifacts_tab(), "Artifacts")
        tabs.addTab(self._build_notes_tab(), "Notes")
        tabs.addTab(self._build_resume_tab(), "Resume a run")
        tabs.addTab(self._build_report_tab(), "Report")
        root.addWidget(tabs)

    # -- Project (create/open) -----------------------------------------

    def _build_project_group(self) -> QGroupBox:
        group = QGroupBox("Project", self)
        form = QFormLayout(group)

        row = QHBoxLayout()
        self.project_dir_edit = QLineEdit(group)
        row.addWidget(self.project_dir_edit)
        browse_button = QPushButton("Browse…", group)
        browse_button.clicked.connect(self._on_browse_dir)
        row.addWidget(browse_button)
        form.addRow("Directory", row)

        self.project_name_edit = QLineEdit(group)
        form.addRow("Name (for New)", self.project_name_edit)

        action_row = QHBoxLayout()
        self.new_project_button = QPushButton("New project", group)
        self.new_project_button.clicked.connect(self._on_new_project)
        action_row.addWidget(self.new_project_button)
        self.open_project_button = QPushButton("Open project", group)
        self.open_project_button.clicked.connect(self._on_open_project)
        action_row.addWidget(self.open_project_button)
        action_row.addStretch(1)
        form.addRow(action_row)

        self.project_status = QLabel("No project open.", group)
        self.project_status.setObjectName("Caption")
        form.addRow(self.project_status)
        return group

    def _on_browse_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Project directory")
        if path:
            self.project_dir_edit.setText(path)

    def _on_new_project(self) -> None:
        directory = self.project_dir_edit.text().strip()
        name = self.project_name_edit.text().strip()
        if not directory or not name:
            QMessageBox.warning(self, "Project Manager", "Directory and name are both required.")
            return
        thread, worker = call_in_thread(pm_core.create_project, directory, name)
        worker.finished.connect(self._on_project_opened)
        worker.error.connect(self._on_project_error)
        self._threads.append((thread, worker))

    def _on_open_project(self) -> None:
        directory = self.project_dir_edit.text().strip()
        if not directory:
            QMessageBox.warning(self, "Project Manager", "Pick a project directory first.")
            return
        thread, worker = call_in_thread(pm_core.open_project, directory)
        worker.finished.connect(self._on_project_opened)
        worker.error.connect(self._on_project_error)
        self._threads.append((thread, worker))

    def _on_project_opened(self, info: pm_core.ProjectInfo) -> None:
        self._project_dir = info.root
        self.project_status.setText(
            f"Open: {info.name!r} at {info.root} (created {info.created_at})")
        self.statusMessage.emit(f"Project Manager: opened {info.name!r}")
        self._on_refresh_runs()

    def _on_project_error(self, message: str) -> None:
        QMessageBox.critical(self, "Project Manager", message)

    def _require_project(self) -> Optional[str]:
        if not self._project_dir:
            QMessageBox.warning(self, "Project Manager", "Open a project first.")
            return None
        return self._project_dir

    # -- Runs -----------------------------------------------------

    def _build_runs_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        row = QHBoxLayout()
        self.refresh_runs_button = QPushButton("Refresh", tab)
        self.refresh_runs_button.clicked.connect(self._on_refresh_runs)
        row.addWidget(self.refresh_runs_button)
        row.addStretch(1)
        layout.addLayout(row)
        self.runs_table = ResultsTable(_RUN_HEADERS, tab)
        layout.addWidget(self.runs_table)
        return tab

    def _on_refresh_runs(self) -> None:
        project_dir = self._require_project()
        if project_dir is None:
            return
        thread, worker = call_in_thread(pm_core.list_runs, project_dir)
        worker.finished.connect(self._on_runs_loaded)
        worker.error.connect(self._on_project_error)
        self._threads.append((thread, worker))

    def _on_runs_loaded(self, runs: List[dict]) -> None:
        self._runs = runs
        rows = [(r.get("run_id"), r.get("pipeline"), r.get("label"),
                  r.get("started_at"), r.get("finished_at"), r.get("status")) for r in runs]
        self.runs_table.set_rows(rows)
        self.run_id_combo.clear()
        self.run_id_combo.addItems([str(r.get("run_id")) for r in runs])
        self.statusMessage.emit(f"Project Manager: {len(runs)} run(s)")

    # -- Artifacts -----------------------------------------------------

    def _build_artifacts_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        row = QHBoxLayout()
        self.artifacts_run_filter = QLineEdit(tab)
        self.artifacts_run_filter.setPlaceholderText("run_id filter (optional)")
        row.addWidget(self.artifacts_run_filter)
        self.refresh_artifacts_button = QPushButton("Refresh", tab)
        self.refresh_artifacts_button.clicked.connect(self._on_refresh_artifacts)
        row.addWidget(self.refresh_artifacts_button)
        row.addStretch(1)
        layout.addLayout(row)
        self.artifacts_table = ResultsTable(_ARTIFACT_HEADERS, tab)
        layout.addWidget(self.artifacts_table)
        return tab

    def _on_refresh_artifacts(self) -> None:
        project_dir = self._require_project()
        if project_dir is None:
            return
        run_id = self.artifacts_run_filter.text().strip() or None
        thread, worker = call_in_thread(pm_core.list_artifacts, project_dir, run_id)
        worker.finished.connect(self._on_artifacts_loaded)
        worker.error.connect(self._on_project_error)
        self._threads.append((thread, worker))

    def _on_artifacts_loaded(self, artifacts: List[dict]) -> None:
        rows = [(a.get("artifact_id"), a.get("run_id"), a.get("role"), a.get("kind"),
                  a.get("path"), a.get("size_bytes"), ",".join(a.get("tags") or [])) for a in artifacts]
        self.artifacts_table.set_rows(rows)
        self.statusMessage.emit(f"Project Manager: {len(artifacts)} artifact(s)")

    # -- Notes -----------------------------------------------------

    def _build_notes_tab(self) -> QWidget:
        tab = QWidget(self)
        form = QFormLayout(tab)
        self.note_title_edit = QLineEdit(tab)
        form.addRow("Title", self.note_title_edit)
        self.note_body_edit = QPlainTextEdit(tab)
        self.note_body_edit.setPlaceholderText("Markdown body…")
        form.addRow("Body", self.note_body_edit)
        self.add_note_button = QPushButton("Add note", tab)
        self.add_note_button.clicked.connect(self._on_add_note)
        form.addRow(self.add_note_button)
        self.note_status = QLabel("", tab)
        self.note_status.setObjectName("Caption")
        form.addRow(self.note_status)
        return tab

    def _on_add_note(self) -> None:
        project_dir = self._require_project()
        if project_dir is None:
            return
        title = self.note_title_edit.text().strip()
        if not title:
            QMessageBox.warning(self, "Project Manager", "A note needs a title.")
            return
        thread, worker = call_in_thread(
            pm_core.add_note, project_dir, title, self.note_body_edit.toPlainText())
        worker.finished.connect(self._on_note_added)
        worker.error.connect(self._on_project_error)
        self._threads.append((thread, worker))

    def _on_note_added(self, artifact_id: int) -> None:
        self.note_status.setText(f"Note added (artifact_id={artifact_id}).")
        self.note_title_edit.clear()
        self.note_body_edit.clear()

    # -- Resume -----------------------------------------------------

    def _build_resume_tab(self) -> QWidget:
        tab = QWidget(self)
        form = QFormLayout(tab)
        self.run_id_combo = QComboBox(tab)
        self.run_id_combo.setEditable(True)
        form.addRow("run_id", self.run_id_combo)
        row = QHBoxLayout()
        self.resume_target_edit = QLineEdit(tab)
        row.addWidget(self.resume_target_edit)
        browse_button = QPushButton("Browse…", tab)
        browse_button.clicked.connect(self._on_browse_resume_target)
        row.addWidget(browse_button)
        form.addRow("Target directory", row)
        self.resume_button = QPushButton("Resume (copy artifacts here)", tab)
        self.resume_button.clicked.connect(self._on_resume)
        form.addRow(self.resume_button)
        self.resume_status = QLabel("", tab)
        self.resume_status.setObjectName("Caption")
        self.resume_status.setWordWrap(True)
        form.addRow(self.resume_status)
        return tab

    def _on_browse_resume_target(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Resume target directory")
        if path:
            self.resume_target_edit.setText(path)

    def _on_resume(self) -> None:
        project_dir = self._require_project()
        if project_dir is None:
            return
        run_id = self.run_id_combo.currentText().strip()
        target_dir = self.resume_target_edit.text().strip()
        if not run_id or not target_dir:
            QMessageBox.warning(self, "Project Manager", "Pick both a run_id and a target directory.")
            return
        thread, worker = call_in_thread(pm_core.resume_run, project_dir, run_id, target_dir)
        worker.finished.connect(self._on_resumed)
        worker.error.connect(self._on_project_error)
        self._threads.append((thread, worker))

    def _on_resumed(self, materialized: dict) -> None:
        self.resume_status.setText(f"Materialized {len(materialized)} artifact(s) into the target directory.")

    # -- Report -----------------------------------------------------

    def _build_report_tab(self) -> QWidget:
        tab = QWidget(self)
        form = QFormLayout(tab)
        note = QLabel(
            "Builds the cross-pipeline LaTeX report (main + supplementary + "
            "notebook + manifest) from every registered artifact, routed by "
            "report_rules.yaml if present.", tab)
        note.setObjectName("Caption")
        note.setWordWrap(True)
        form.addRow(note)
        self.build_report_button = QPushButton("Build report", tab)
        self.build_report_button.clicked.connect(self._on_build_report)
        form.addRow(self.build_report_button)
        self.report_status = QLabel("", tab)
        self.report_status.setObjectName("Caption")
        self.report_status.setWordWrap(True)
        form.addRow(self.report_status)
        return tab

    def _on_build_report(self) -> None:
        project_dir = self._require_project()
        if project_dir is None:
            return
        self.build_report_button.setEnabled(False)
        thread, worker = call_in_thread(pm_core.build_project_report, project_dir)
        worker.finished.connect(self._on_report_built)
        worker.error.connect(self._on_report_error)
        self._threads.append((thread, worker))

    def _on_report_built(self, result: pm_core.ReportOutcome) -> None:
        self.build_report_button.setEnabled(True)
        compiled = "PDF compiled" if result.compiled else "LaTeX written, PDF compile skipped/failed"
        self.report_status.setText(f"{compiled}. Main: {result.main_tex}")
        self.statusMessage.emit(f"Project Manager: report built ({compiled})")

    def _on_report_error(self, message: str) -> None:
        self.build_report_button.setEnabled(True)
        QMessageBox.critical(self, "Project Manager", f"Report build failed: {message}")
