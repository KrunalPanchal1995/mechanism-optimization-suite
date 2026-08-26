"""
mso_studio.gui.widgets.run_summary
=====================================
The run-summary card (guidelines doc, Part 6.1's "signature visual idea"):
appears once a run finishes, shows status/timing/counts at a glance, and
hosts the "Generate LaTeX report" button -- disabled with an explanatory
tooltip until a run exists (Part 1.1 / Part 6.8.4). Also hosts "Export run
status…" (KRUNAL_JULY_23.md point 1) -- unlike the LaTeX report, this is
enabled for ANY finished run including failed ones, since seeing exactly
what inputs/options/error a failed run used is the whole point.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog, QFrame, QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout,
)

from mso_studio.core.runrecord import RunRecord
from mso_io.run_status import RunStatusReport

_STATUS_OBJECT_NAME = {
    "done": "StatusSuccess",
    "cancelled": "StatusWarning",
    "failed": "StatusError",
    "timeout": "StatusError",
}


class RunSummaryCard(QFrame):
    """Shows the last run's headline outcome; owns the report button."""

    reportRequested = Signal()
    postprocessRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._record: Optional[RunRecord] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)

        self.title_label = QLabel("No runs yet", self)
        self.title_label.setObjectName("SectionHeader")
        layout.addWidget(self.title_label)

        self.detail_label = QLabel("Run a simulation to see results here.", self)
        self.detail_label.setObjectName("Caption")
        self.detail_label.setWordWrap(True)
        layout.addWidget(self.detail_label)

        button_row = QHBoxLayout()
        button_row.setSpacing(8)
        self.report_button = QPushButton("Generate LaTeX report", self)
        self.report_button.setEnabled(False)
        self.report_button.setToolTip("Run a simulation first.")
        self.report_button.clicked.connect(self.reportRequested.emit)
        button_row.addWidget(self.report_button)

        # "process the results" flag (2026-08 workbench audit, task 3):
        # re-runs the matching repo's postprocess() -- always guaranteed to
        # emit CSVs, --journal on by default -- against this run's own
        # working_dir. Enabled for ANY finished record including a
        # cancelled/failed one (that's the --intermediate use case: partial
        # results are still worth processing) -- ModulePage decides
        # has_postprocess()/intermediate, this card just owns the button.
        self.postprocess_button = QPushButton("Post-process results…", self)
        self.postprocess_button.setEnabled(False)
        self.postprocess_button.setToolTip("Run a simulation first.")
        self.postprocess_button.clicked.connect(self.postprocessRequested.emit)
        button_row.addWidget(self.postprocess_button)

        self.export_status_button = QPushButton("Export run status…", self)
        self.export_status_button.setEnabled(False)
        self.export_status_button.setToolTip("Run a simulation first.")
        self.export_status_button.clicked.connect(self._on_export_status_clicked)
        button_row.addWidget(self.export_status_button)

        button_row.addStretch(1)
        layout.addLayout(button_row)

    def show_running(self, label: str) -> None:
        self._record = None
        self.title_label.setText(f"Running: {label} …")
        self.title_label.setObjectName("SectionHeader")
        self._refresh_style(self.title_label)
        self.detail_label.setText("This may take a while — the window stays responsive.")
        self.report_button.setEnabled(False)
        self.report_button.setToolTip("Run a simulation first.")
        self.postprocess_button.setEnabled(False)
        self.postprocess_button.setToolTip("Run a simulation first.")
        self.export_status_button.setEnabled(False)
        self.export_status_button.setToolTip("Run a simulation first.")

    def show_record(self, record: RunRecord) -> None:
        self._record = record
        status_name = _STATUS_OBJECT_NAME.get(record.status, "SectionHeader")
        wall = f"{record.wall_time_s:.1f} s" if record.wall_time_s is not None else "N/A"
        self.title_label.setText(f"{record.config.label} — {record.status}")
        self.title_label.setObjectName(status_name)
        self._refresh_style(self.title_label)

        detail_bits = [f"Wall time: {wall}"]
        if record.tables:
            detail_bits.append(f"{len(record.tables)} table(s)")
        if record.figures:
            detail_bits.append(f"{len(record.figures)} figure(s)")
        if record.warnings:
            detail_bits.append(f"{len(record.warnings)} warning(s)")
        self.detail_label.setText(" · ".join(detail_bits))

        can_report = record.status == "done"
        self.report_button.setEnabled(can_report)
        self.report_button.setToolTip(
            "Generate a LaTeX report from this run." if can_report
            else "Only available after a run completes successfully."
        )

        # Unlike the report button, post-processing is offered for ANY
        # finished record (done/failed/cancelled) -- a stopped or failed
        # run's partial working_dir is exactly the --intermediate use case.
        # ModulePage._on_postprocess_requested does the has_postprocess()
        # check and disables/hides this for pages with no postprocess()
        # entry point at all.
        self.postprocess_button.setEnabled(True)
        self.postprocess_button.setToolTip(
            "Re-run this pipeline's postprocess() over the run's own output "
            "directory: regenerates plots/CSVs, --journal on by default."
            + ("" if can_report else
               " This run didn't finish cleanly -- processed as partial/intermediate results.")
        )

        # unlike the LaTeX report, run-status export is useful for a FAILED
        # run too -- that's the whole point (KRUNAL_JULY_23.md point 1)
        self.export_status_button.setEnabled(True)
        self.export_status_button.setToolTip(
            "Export this run's inputs, options, and log to a text file.")

    def _on_export_status_clicked(self) -> None:
        if self._record is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export run status", "", "Text (*.txt)")
        if not path:
            return
        report = RunStatusReport.from_run_record(self._record)
        try:
            report.write_txt(path)
        except OSError as exc:
            QMessageBox.critical(self, "Export failed", f"Could not write {path}: {exc}")
            return
        QMessageBox.information(self, "Exported", f"Run status written to:\n{path}")

    def _refresh_style(self, widget) -> None:
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    @property
    def record(self) -> Optional[RunRecord]:
        return self._record
