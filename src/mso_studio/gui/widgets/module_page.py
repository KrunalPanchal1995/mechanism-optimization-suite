"""
mso_studio.gui.widgets.module_page
=====================================
The standard module-page layout (guidelines doc, Part 6.1): setup on the
left (28-33% width), tabbed results on the right, a collapsible log strip
below the results, one full-width accent Run button anchored at the bottom
of the setup panel, and a :class:`RunSummaryCard` under it that owns the
"Generate LaTeX report" button.

Every capability page (nominal sim, compare, sensitivity, optimization, ...)
subclasses :class:`ModulePage` and only has to: build its setup fields with
:meth:`add_setup_row`/:meth:`add_advanced_row`, implement
:meth:`build_core_call` to return the ``core`` function + args to run, and
implement :meth:`populate_results` to fill its Plots/Tables tabs from a
finished :class:`RunRecord`. Threading, logging, cancellation, and the
report button are handled once, here.
"""
from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtWidgets import (
    QFormLayout, QFrame, QGroupBox, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QSplitter, QTabWidget, QToolButton, QVBoxLayout, QWidget,
)

from mso_studio.core.runrecord import RunRecord
from mso_studio.gui.widgets.log_view import LogView
from mso_studio.gui.widgets.module_hero import ModuleHero
from mso_studio.gui.widgets.run_summary import RunSummaryCard
from mso_studio.gui.workers import run_in_thread


class CollapsibleSection(QWidget):
    """A ``QToolButton``-driven disclosure triangle for the Advanced section
    (guidelines doc, Part 1.5 / Part 6.3 -- progressive disclosure, collapsed
    by default)."""

    def __init__(self, title: str = "Advanced", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.toggle = QToolButton(self)
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(False)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle.clicked.connect(self._on_toggled)
        layout.addWidget(self.toggle)

        self.content = QWidget(self)
        self.content_layout = QFormLayout(self.content)
        self.content.setVisible(False)
        layout.addWidget(self.content)

    def _on_toggled(self, checked: bool) -> None:
        self.content.setVisible(checked)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)

    def add_row(self, label: str, widget: QWidget) -> None:
        self.content_layout.addRow(label, widget)


class ModulePage(QWidget):
    """Base widget for one capability's GUI page.

    Signals
    -------
    statusMessage(str)
        Transient status-bar text (guidelines doc, Part 6.1: "never use
        pop-up dialogs for success messages").
    runFinished(object)
        Emitted with the finished :class:`RunRecord` (any terminal status).
    """

    statusMessage = Signal(str)
    runFinished = Signal(object)

    def __init__(self, module_id: str, title: str, description: str = "", parent=None):
        super().__init__(parent)
        self.module_id = module_id
        self._thread = None
        self._worker = None
        self._settings = QSettings("mechanism-optimization-suite", "mso-studio")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        self.hero = ModuleHero(module_id, title, description, self)
        root.addWidget(self.hero)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.main_splitter.setObjectName(f"{module_id}_main_splitter")
        root.addWidget(self.main_splitter, 1)

        self._build_setup_panel()
        self._build_results_panel()

        self.main_splitter.addWidget(self._setup_scroll)
        self.main_splitter.addWidget(self._results_splitter)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([380, 900])  # ~30:70, guidelines doc Part 6.2

        self._restore_splitter_state()
        self.main_splitter.splitterMoved.connect(lambda *_: self._save_splitter_state())

    # -- setup panel ---------------------------------------------------

    def _build_setup_panel(self) -> None:
        self._setup_scroll = QScrollArea(self)
        self._setup_scroll.setWidgetResizable(True)
        self._setup_scroll.setMinimumWidth(320)

        container = QWidget(self._setup_scroll)
        self._setup_layout = QVBoxLayout(container)
        self._setup_layout.setContentsMargins(16, 16, 16, 16)
        self._setup_layout.setSpacing(16)

        self._form_group = QGroupBox("", container)
        self._form = QFormLayout(self._form_group)
        self._form.setSpacing(8)
        self._form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._setup_layout.addWidget(self._form_group)

        self._advanced = CollapsibleSection("Advanced", container)
        self._setup_layout.addWidget(self._advanced)

        self._setup_layout.addStretch(1)

        run_row = QHBoxLayout()
        run_row.setSpacing(8)
        self.run_button = QPushButton("▶  Run", container)
        self.run_button.setObjectName("PrimaryAction")
        self.run_button.setMinimumHeight(36)
        self.run_button.clicked.connect(self._on_run_clicked)
        run_row.addWidget(self.run_button, 1)

        self.stop_button = QPushButton("■", container)
        self.stop_button.setToolTip("Stop the running job")
        self.stop_button.setFixedWidth(36)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._on_stop_clicked)
        run_row.addWidget(self.stop_button)
        self._setup_layout.addLayout(run_row)

        self.run_summary = RunSummaryCard(container)
        self.run_summary.reportRequested.connect(self._on_report_requested)
        self.run_summary.postprocessRequested.connect(self._on_postprocess_requested)
        self._setup_layout.addWidget(self.run_summary)

        self._setup_scroll.setWidget(container)

    def add_setup_row(self, label: str, widget: QWidget) -> None:
        self._form.addRow(label, widget)

    def add_advanced_row(self, label: str, widget: QWidget) -> None:
        self._advanced.add_row(label, widget)

    def set_setup_title(self, title: str) -> None:
        self._form_group.setTitle(title)

    # -- results panel ---------------------------------------------------

    def _build_results_panel(self) -> None:
        self._results_splitter = QSplitter(Qt.Orientation.Vertical, self)
        self._results_splitter.setObjectName(f"{self.module_id}_results_splitter")

        self.results_tabs = QTabWidget(self._results_splitter)
        self._results_splitter.addWidget(self.results_tabs)

        log_container = QWidget(self._results_splitter)
        log_layout = QVBoxLayout(log_container)
        log_layout.setContentsMargins(8, 4, 8, 8)
        log_layout.setSpacing(4)
        log_header = QLabel("Log", log_container)
        log_header.setObjectName("Caption")
        log_layout.addWidget(log_header)
        self.log_view = LogView(log_container)
        log_layout.addWidget(self.log_view)
        self._results_splitter.addWidget(log_container)

        self._results_splitter.setStretchFactor(0, 3)
        self._results_splitter.setStretchFactor(1, 1)
        # Log strip starts open at ~22% height (guidelines doc, Part 6.1's
        # own target ratio) so users can see live terminal output from the
        # backend subprocess (nominal sim, sensitivity, etc.) without first
        # having to discover and drag open a splitter collapsed to zero
        # height -- previously [700, 0] hid it completely by default.
        # Still user-adjustable/collapsible via the splitter handle, and
        # _restore_splitter_state() below preserves whatever size a user
        # has since chosen across restarts.
        self._results_splitter.setSizes([663, 187])

    def add_results_tab(self, title: str, widget: QWidget) -> None:
        self.results_tabs.addTab(widget, title)

    # -- persistence ---------------------------------------------------

    def _restore_splitter_state(self) -> None:
        state = self._settings.value(f"{self.module_id}/main_splitter")
        if state is not None:
            self.main_splitter.restoreState(state)

    def _save_splitter_state(self) -> None:
        self._settings.setValue(f"{self.module_id}/main_splitter", self.main_splitter.saveState())

    # -- run lifecycle ---------------------------------------------------
    # Subclasses implement these two hooks; everything else is generic.

    def build_core_call(self) -> tuple[Callable[..., RunRecord], tuple, dict]:
        """Return ``(core_fn, args, kwargs)`` for the current setup-panel state.

        Raise :class:`ValueError` with a user-facing message if the current
        form state is invalid -- it will be shown in the status bar rather
        than starting a run (guidelines doc, Part 1.6: prevent, don't just
        handle, errors).
        """
        raise NotImplementedError

    def populate_results(self, record: RunRecord) -> None:
        """Fill the results tabs from a finished RunRecord. Override."""
        pass

    def _on_run_clicked(self) -> None:
        try:
            core_fn, args, kwargs = self.build_core_call()
        except ValueError as exc:
            self.statusMessage.emit(str(exc))
            self.log_view.append_status(f"Not started: {exc}")
            return

        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.run_summary.show_running(getattr(self, "_current_label", self.module_id))
        self.log_view.append_status("Run started")
        self.statusMessage.emit("Running…")

        self._thread, self._worker = run_in_thread(core_fn, *args, **kwargs)
        self._worker.progress.connect(self.log_view.append_line)
        self._worker.finished.connect(self._on_run_finished)
        self._worker.error.connect(self._on_run_error)
        self._worker.cancelled.connect(self._on_run_cancelled)

    def _on_stop_clicked(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.log_view.append_status("Cancellation requested…")

    def _on_run_finished(self, record: RunRecord) -> None:
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.run_summary.show_record(record)
        self.log_view.append_status(f"Run {record.status} in {record.wall_time_s:.1f}s"
                                     if record.wall_time_s is not None else f"Run {record.status}")
        self.statusMessage.emit(f"Run complete — {record.config.label}")
        try:
            self.populate_results(record)
        except Exception as exc:  # noqa: BLE001 - never let a plot bug hide a successful run
            self.log_view.append_status(f"Results preview failed to render: {exc}")
        self.runFinished.emit(record)

    def _on_run_error(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.log_view.append_status(f"Failed: {message}")
        self.statusMessage.emit("Run failed — see log for details")

    def _on_run_cancelled(self) -> None:
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.log_view.append_status("Run cancelled")
        self.statusMessage.emit("Run cancelled")

    def _on_report_requested(self) -> None:
        record = self.run_summary.record
        if record is None:
            return
        self.statusMessage.emit("Generating LaTeX report…")
        self._generate_report(record)

    def _generate_report(self, record: RunRecord) -> None:
        from pathlib import Path

        from mso_studio.core.reporting import build_report

        out_dir = Path(record.run_dir()) / "report"
        try:
            result = build_report(record, out_dir)
        except Exception as exc:  # noqa: BLE001 - report must never crash the GUI
            self.log_view.append_status(f"Report generation failed: {exc}")
            self.statusMessage.emit("Report generation failed — see log")
            return

        if result.pdf_path is not None:
            self.log_view.append_status(f"Report PDF: {result.pdf_path}")
            self.statusMessage.emit(f"Report saved to {result.pdf_path}")
            _open_in_os_viewer(str(result.pdf_path))
        else:
            self.log_view.append_status(
                f"No LaTeX engine found — wrote a compile-ready bundle: {result.bundle_path}"
            )
            self.statusMessage.emit(f"Report bundle saved to {result.bundle_path}")

    # -- post-processing (2026-08 workbench audit, task 3) ----------------

    def _on_postprocess_requested(self) -> None:
        """Re-run this module's repo-side postprocess() (see
        mso_studio.core.postprocess_dispatch) over the finished run's own
        working_dir. Available for a failed/cancelled run too -- that run's
        partial working_dir is processed with --intermediate, which every
        postprocess() implementation is built to tolerate (see
        src/MUQ-SAC/VisualAid.py's module docstring)."""
        record = self.run_summary.record
        if record is None:
            return

        from mso_studio.core.postprocess_dispatch import has_postprocess, postprocess_command

        if not has_postprocess(self.module_id):
            self.log_view.append_status(
                f"No postprocess() entry point registered for '{self.module_id}' yet.")
            self.statusMessage.emit("Post-processing not available for this module.")
            return

        run_dir = record.config.working_dir
        intermediate = record.status != "done"
        command = postprocess_command(self.module_id, run_dir, intermediate=intermediate)
        if command is None:  # has_postprocess() already checked, but stay honest
            return

        self.log_view.append_status(
            f"Post-processing {run_dir}" + (" (intermediate/partial results)" if intermediate else "") + " …")
        self.statusMessage.emit("Post-processing results…")

        from mso_studio.core.backends import REPO_ROOT
        self._pp_thread, self._pp_worker = run_in_thread(
            _run_postprocess_backend, command, str(REPO_ROOT))
        self._pp_worker.progress.connect(self.log_view.append_line)
        self._pp_worker.finished.connect(self._on_postprocess_finished)
        self._pp_worker.error.connect(self._on_postprocess_error)

    def _on_postprocess_finished(self, result) -> None:
        if result.status == "done":
            self.log_view.append_status("Post-processing complete.")
            self.statusMessage.emit("Post-processing complete — see log.")
        else:
            self.log_view.append_status(f"Post-processing {result.status} — see log above.")
            self.statusMessage.emit(f"Post-processing {result.status}.")

    def _on_postprocess_error(self, message: str) -> None:
        self.log_view.append_status(f"Post-processing error: {message}")
        self.statusMessage.emit("Post-processing failed — see log.")


def _run_postprocess_backend(command, cwd, *, on_line=None, cancel_event=None):
    """core_fn for run_in_thread -- RunWorker injects on_line/cancel_event
    (see mso_studio.gui.workers.RunWorker.run), so this must accept exactly
    those keyword names. Streams the postprocess subprocess's output into
    the log the same way a regular run does; the returned BackendResult's
    ``.status`` duck-types as what RunWorker.run() checks
    (record.status == "cancelled") so no RunRecord wrapper is needed."""
    from mso_studio.core.backends import run_backend
    return run_backend(command, cwd=cwd, on_line=on_line, cancel_event=cancel_event)


def _open_in_os_viewer(path: str) -> None:
    import subprocess
    import sys

    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", path])
        elif sys.platform.startswith("linux"):
            subprocess.Popen(["xdg-open", path])
        elif sys.platform.startswith("win"):
            import os
            os.startfile(path)  # type: ignore[attr-defined]
    except OSError:
        pass  # opening is a convenience, not a hard requirement
