"""
mso_studio.gui.modules.target_workbench
==========================================
Target Workbench: curate a .target/.add pair before it reaches Addendum --
plot, sanity-check, generate NTC-aware dummy points, prune for T/P/phi
coverage, then write a new versioned pair. Ported from
generate_target_opt.py's Tkinter ``_tab_target_workbench`` (previously
reachable only from the pre-split monolith). Part of the Optimize
Workbench (Phase 2, 2026-08 GUI integration overhaul) alongside
optimization/sensitivity/forward_uq.

A plain QWidget page, not a ``ModulePage`` subclass -- like ``ToolsPage``,
this is several independent interactive actions (load/check/generate/prune/
write), not one run-and-report backend job.
"""
from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPushButton, QScrollArea, QSpinBox,
    QTabWidget, QVBoxLayout, QWidget,
)

from mso_studio.core import target_workbench as twb_core
from mso_studio.gui.widgets.file_picker import FilePicker
from mso_studio.gui.widgets.module_hero import make_module_hero
from mso_studio.gui.widgets.results_table import ResultsTable
from mso_studio.gui.workers import call_in_thread

_ROW_HEADERS = ("line_no", "dataset_id", "target_type", "T", "P", "phi",
                 "observed", "deviation", "weight", "status")
_FINDING_HEADERS = ("severity", "line_no", "dataset_id", "check", "message")
_PLOT_KINDS = {
    "T-P-dataset": "t_p_dataset",
    "ln(tau) vs 1000/T": "lntau_vs_invT",
    "observed vs deviation": "observed_vs_deviation",
    "condition coverage": "condition_coverage",
}


class TargetWorkbenchPage(QWidget):
    statusMessage = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: List[twb_core.TargetRowInfo] = []
        self._dataset_ids: List[str] = []
        self._dropped: set[int] = set()
        self._dummy_rows: List[dict] = []          # {dataset_id, temperature_K, observed}
        self._findings: List[twb_core.Finding] = []
        self._suggested_drop: Optional[List[int]] = None
        self._threads = []  # keep (QThread, worker) pairs alive

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.hero = make_module_hero(
            "target_workbench", "Target Workbench",
            "Curate a .target/.add pair before it reaches Addendum: sanity-check, "
            "generate NTC-aware dummy points, prune for coverage, then write a new "
            "version. Nothing is written until you click Write below, and the "
            "source file is never modified.",
            self,
        )
        outer.addWidget(self.hero)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll, 1)
        body = QWidget(scroll)
        scroll.setWidget(body)
        root = QVBoxLayout(body)

        root.addWidget(self._build_load_group())
        root.addWidget(self._build_rows_group())

        tabs = QTabWidget(body)
        tabs.addTab(self._build_checks_tab(), "Checks")
        tabs.addTab(self._build_plot_tab(), "Plots")
        tabs.addTab(self._build_dummy_tab(), "Generate dummy points")
        tabs.addTab(self._build_prune_tab(), "Suggest prunable rows")
        root.addWidget(tabs)

        root.addWidget(self._build_write_group())

    # -- Load ---------------------------------------------------------

    def _build_load_group(self) -> QGroupBox:
        group = QGroupBox("Source .target / .add", self)
        form = QFormLayout(group)
        self.target_picker = FilePicker(mode="file", filter_="Target files (*.target)", parent=group)
        form.addRow(".target", self.target_picker)
        self.add_picker = FilePicker(mode="file", filter_="Add files (*.add)", must_exist=False, parent=group)
        form.addRow(".add (optional)", self.add_picker)

        load_row = QHBoxLayout()
        self.load_button = QPushButton("Load", group)
        self.load_button.clicked.connect(self._on_load)
        load_row.addWidget(self.load_button)
        load_row.addStretch(1)
        form.addRow(load_row)

        self.load_status = QLabel("Not loaded.", group)
        self.load_status.setObjectName("Caption")
        form.addRow(self.load_status)
        return group

    def _on_load(self) -> None:
        tpath = self.target_picker.path()
        if not tpath:
            QMessageBox.warning(self, "Target Workbench", "No .target file selected.")
            return
        apath = self.add_picker.path() or ""
        self.load_button.setEnabled(False)
        thread, worker = call_in_thread(twb_core.load_target_set, tpath, apath)
        worker.finished.connect(self._on_loaded)
        worker.error.connect(self._on_load_error)
        self._threads.append((thread, worker))

    def _on_loaded(self, result: twb_core.LoadResult) -> None:
        self.load_button.setEnabled(True)
        self._rows = result.rows
        self._dataset_ids = result.dataset_ids
        self._dropped = set()
        self._dummy_rows = []
        self._findings = []
        self._refresh_rows_table()
        self.load_status.setText(
            f"Loaded {len(self._rows)} rows, {len(self._dataset_ids)} dataset(s) from {self.target_picker.path()}")
        self.dummy_dataset_combo.clear()
        self.dummy_dataset_combo.addItems(self._dataset_ids)
        self.plot_dataset_combo.clear()
        self.plot_dataset_combo.addItems(self._dataset_ids)
        self.statusMessage.emit(f"Target Workbench: loaded {len(self._rows)} rows")

    def _on_load_error(self, message: str) -> None:
        self.load_button.setEnabled(True)
        QMessageBox.critical(self, "Target Workbench", f"Failed to load: {message}")

    # -- Rows table -----------------------------------------------------

    def _build_rows_group(self) -> QGroupBox:
        group = QGroupBox("Rows", self)
        layout = QVBoxLayout(group)

        ctrl = QHBoxLayout()
        self.mark_dropped_button = QPushButton("Mark selected as dropped", group)
        self.mark_dropped_button.clicked.connect(self._on_mark_dropped)
        ctrl.addWidget(self.mark_dropped_button)
        self.undrop_button = QPushButton("Undrop selected", group)
        self.undrop_button.clicked.connect(self._on_undrop)
        ctrl.addWidget(self.undrop_button)
        ctrl.addStretch(1)
        layout.addLayout(ctrl)

        self.rows_table = ResultsTable(_ROW_HEADERS, group)
        self.rows_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.rows_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.rows_table.setMaximumHeight(320)
        layout.addWidget(self.rows_table)
        return group

    def _refresh_rows_table(self) -> None:
        rows = []
        for r in self._rows:
            status = "dropped" if r.line_no in self._dropped else "ok"
            rows.append((r.line_no, r.dataset_id, r.target_type, r.temperature_K,
                          r.pressure_atm, r.phi_raw, r.observed, r.deviation,
                          r.data_weight, status))
        for d in self._dummy_rows:
            rows.append(("-", d["dataset_id"], "Tig", round(float(d["temperature_K"]), 1),
                         "-", "-", round(float(d["observed"]), 3), "-", "-", "dummy (staged)"))
        self.rows_table.set_rows(rows)

    def _selected_line_nos(self) -> List[int]:
        line_nos = []
        for item in self.rows_table.selectedItems():
            if item.column() != 0:
                continue
            text = item.text()
            if text.isdigit():
                line_nos.append(int(text))
        return line_nos

    def _on_mark_dropped(self) -> None:
        self._dropped.update(self._selected_line_nos())
        self._refresh_rows_table()

    def _on_undrop(self) -> None:
        for ln in self._selected_line_nos():
            self._dropped.discard(ln)
        self._refresh_rows_table()

    # -- Checks -----------------------------------------------------

    def _build_checks_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        row = QHBoxLayout()
        self.run_checks_button = QPushButton("Run checks", tab)
        self.run_checks_button.clicked.connect(self._on_run_checks)
        row.addWidget(self.run_checks_button)
        row.addStretch(1)
        layout.addLayout(row)
        self.findings_table = ResultsTable(_FINDING_HEADERS, tab)
        layout.addWidget(self.findings_table)
        self.checks_status = QLabel("", tab)
        self.checks_status.setObjectName("Caption")
        layout.addWidget(self.checks_status)
        return tab

    def _on_run_checks(self) -> None:
        tpath = self.target_picker.path()
        if not tpath or not self._rows:
            QMessageBox.warning(self, "Target Workbench", "Load a .target file first.")
            return
        self.run_checks_button.setEnabled(False)
        thread, worker = call_in_thread(twb_core.run_checks, tpath, self.add_picker.path() or "")
        worker.finished.connect(self._on_checks_finished)
        worker.error.connect(self._on_checks_error)
        self._threads.append((thread, worker))

    def _on_checks_finished(self, findings: List[twb_core.Finding]) -> None:
        self.run_checks_button.setEnabled(True)
        self._findings = findings
        rows = [(f.severity, f.line_no if f.line_no is not None else "-",
                  f.dataset_id, f.check, f.message) for f in findings]
        self.findings_table.set_rows(rows)
        n_err = sum(1 for f in findings if f.severity == "error")
        n_warn = sum(1 for f in findings if f.severity == "warning")
        self.checks_status.setText(
            f"Checks: {n_err} error(s), {n_warn} warning(s), {len(findings) - n_err - n_warn} info")

    def _on_checks_error(self, message: str) -> None:
        self.run_checks_button.setEnabled(True)
        QMessageBox.critical(self, "Target Workbench", f"Checks failed: {message}")

    # -- Plots -----------------------------------------------------

    def _build_plot_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        row = QHBoxLayout()
        self.plot_kind_combo = QComboBox(tab)
        self.plot_kind_combo.addItems(list(_PLOT_KINDS))
        row.addWidget(self.plot_kind_combo)
        self.plot_dataset_combo = QComboBox(tab)
        row.addWidget(self.plot_dataset_combo)
        self.show_plot_button = QPushButton("Show plot", tab)
        self.show_plot_button.clicked.connect(self._on_show_plot)
        row.addWidget(self.show_plot_button)
        row.addStretch(1)
        layout.addLayout(row)
        self.plot_label = QLabel(tab)
        self.plot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.plot_label, 1)
        return tab

    def _on_show_plot(self) -> None:
        tpath = self.target_picker.path()
        if not tpath or not self._rows:
            QMessageBox.warning(self, "Target Workbench", "Load a .target file first.")
            return
        import tempfile
        from pathlib import Path
        kind = _PLOT_KINDS[self.plot_kind_combo.currentText()]
        out_png = str(Path(tempfile.gettempdir()) / "mso_twb_plot_preview.png")
        self.show_plot_button.setEnabled(False)
        thread, worker = call_in_thread(
            twb_core.render_plot, tpath, self.add_picker.path() or "", kind, out_png,
            self.plot_dataset_combo.currentText(), sorted(self._dropped),
        )
        worker.finished.connect(self._on_plot_rendered)
        worker.error.connect(self._on_plot_error)
        self._threads.append((thread, worker))

    def _on_plot_rendered(self, png_path: str) -> None:
        self.show_plot_button.setEnabled(True)
        pixmap = QPixmap(png_path)
        if not pixmap.isNull():
            self.plot_label.setPixmap(pixmap.scaledToWidth(760, Qt.TransformationMode.SmoothTransformation))

    def _on_plot_error(self, message: str) -> None:
        self.show_plot_button.setEnabled(True)
        QMessageBox.critical(self, "Target Workbench", f"Plot failed: {message}")

    # -- Generate dummy points -----------------------------------------

    def _build_dummy_tab(self) -> QWidget:
        tab = QWidget(self)
        form = QFormLayout(tab)
        self.dummy_dataset_combo = QComboBox(tab)
        form.addRow("Dataset", self.dummy_dataset_combo)
        self.dummy_points_spin = QSpinBox(tab)
        self.dummy_points_spin.setRange(1, 200)
        self.dummy_points_spin.setValue(10)
        form.addRow("Points per zone (before/NTC/after)", self.dummy_points_spin)
        self.generate_dummy_button = QPushButton("Generate", tab)
        self.generate_dummy_button.clicked.connect(self._on_generate_dummy)
        form.addRow(self.generate_dummy_button)
        self.dummy_status = QLabel("No dummy points staged.", tab)
        self.dummy_status.setObjectName("Caption")
        form.addRow(self.dummy_status)
        return tab

    def _on_generate_dummy(self) -> None:
        tpath = self.target_picker.path()
        dataset_id = self.dummy_dataset_combo.currentText()
        if not tpath or not dataset_id:
            QMessageBox.warning(self, "Target Workbench", "Load a .target file and pick a dataset first.")
            return
        self.generate_dummy_button.setEnabled(False)
        thread, worker = call_in_thread(
            twb_core.generate_dummy_points, tpath, self.add_picker.path() or "",
            dataset_id, self.dummy_points_spin.value(),
        )
        worker.finished.connect(lambda result, ds=dataset_id: self._on_dummy_generated(result, ds))
        worker.error.connect(self._on_dummy_error)
        self._threads.append((thread, worker))

    def _on_dummy_generated(self, result: twb_core.DummyGenerationResult, dataset_id: str) -> None:
        self.generate_dummy_button.setEnabled(True)
        for row in result.rows:
            self._dummy_rows.append({
                "dataset_id": dataset_id,
                "temperature_K": row["Temperature_K"],
                "observed": row["observed"],
            })
        self.dummy_status.setText(f"{len(self._dummy_rows)} dummy point(s) staged.")
        self._refresh_rows_table()

    def _on_dummy_error(self, message: str) -> None:
        self.generate_dummy_button.setEnabled(True)
        QMessageBox.critical(self, "Target Workbench", f"Dummy generation failed: {message}")

    # -- Prune -----------------------------------------------------

    def _build_prune_tab(self) -> QWidget:
        tab = QWidget(self)
        form = QFormLayout(tab)
        self.prune_n_spin = QSpinBox(tab)
        self.prune_n_spin.setRange(1, 100000)
        self.prune_n_spin.setValue(1)
        form.addRow("Rows to drop", self.prune_n_spin)
        row = QHBoxLayout()
        self.suggest_prune_button = QPushButton("Suggest", tab)
        self.suggest_prune_button.clicked.connect(self._on_suggest_prune)
        row.addWidget(self.suggest_prune_button)
        self.apply_prune_button = QPushButton("Apply", tab)
        self.apply_prune_button.clicked.connect(self._on_apply_prune)
        self.apply_prune_button.setEnabled(False)
        row.addWidget(self.apply_prune_button)
        row.addStretch(1)
        form.addRow(row)
        self.prune_result_label = QLabel("", tab)
        self.prune_result_label.setObjectName("Caption")
        self.prune_result_label.setWordWrap(True)
        form.addRow(self.prune_result_label)
        return tab

    def _on_suggest_prune(self) -> None:
        tpath = self.target_picker.path()
        if not tpath or not self._rows:
            QMessageBox.warning(self, "Target Workbench", "Load a .target file first.")
            return
        n_drop = self.prune_n_spin.value()
        if n_drop >= len(self._rows):
            QMessageBox.warning(self, "Target Workbench", f"Rows to drop must be < {len(self._rows)}.")
            return
        self.suggest_prune_button.setEnabled(False)
        thread, worker = call_in_thread(
            twb_core.coverage_prune, tpath, self.add_picker.path() or "", n_drop, 0)
        worker.finished.connect(self._on_prune_suggested)
        worker.error.connect(self._on_prune_error)
        self._threads.append((thread, worker))

    def _on_prune_suggested(self, result: twb_core.PruneResult) -> None:
        self.suggest_prune_button.setEnabled(True)
        self._suggested_drop = result.drop_line_nos
        self.apply_prune_button.setEnabled(True)
        self.prune_result_label.setText(
            f"Suggests dropping {len(result.drop_line_nos)} row(s), keeping "
            f"coverage={result.coverage:.2%}. Click Apply to stage these drops "
            f"(nothing written yet).")

    def _on_prune_error(self, message: str) -> None:
        self.suggest_prune_button.setEnabled(True)
        QMessageBox.critical(self, "Target Workbench", f"Prune failed: {message}")

    def _on_apply_prune(self) -> None:
        if self._suggested_drop is None:
            QMessageBox.information(self, "Target Workbench", "Click Suggest first.")
            return
        self._dropped.update(self._suggested_drop)
        self._refresh_rows_table()

    # -- Write -----------------------------------------------------

    def _build_write_group(self) -> QGroupBox:
        group = QGroupBox("Write new version", self)
        form = QFormLayout(group)
        note = QLabel(
            "Writes a new <stem>_v<N>.target + .add (never overwrites the source). "
            "Runs the combustion_target parser gate immediately after writing and "
            "reports any row that fails to parse.", group)
        note.setObjectName("Caption")
        note.setWordWrap(True)
        form.addRow(note)
        self.out_stem_edit = QLineEdit("targets_curated", group)
        form.addRow("Output stem", self.out_stem_edit)
        self.write_button = QPushButton("Write new .target + .add", group)
        self.write_button.clicked.connect(self._on_write)
        form.addRow(self.write_button)
        self.write_status = QLabel("", group)
        self.write_status.setObjectName("Caption")
        self.write_status.setWordWrap(True)
        form.addRow(self.write_status)
        return group

    def _on_write(self) -> None:
        tpath = self.target_picker.path()
        if not tpath or not self._rows:
            QMessageBox.warning(self, "Target Workbench", "Load a .target file first.")
            return
        keep = [r.line_no for r in self._rows if r.line_no not in self._dropped]

        reason_log = [{"action": "drop", "line_no": ln} for ln in sorted(self._dropped)]
        reason_log += [{"action": "generate_dummy", "dataset_id": d["dataset_id"],
                         "temperature_K": float(d["temperature_K"])} for d in self._dummy_rows]
        for finding in self._findings:
            if finding.suggested_fix:
                reason_log.append({"action": "finding", "check": finding.check,
                                    "line_no": finding.line_no,
                                    "suggested_fix": finding.suggested_fix})

        self.write_button.setEnabled(False)
        thread, worker = call_in_thread(
            twb_core.write_curated, tpath, self.add_picker.path() or "",
            self.out_stem_edit.text() or "targets_curated", keep,
            self._dummy_rows, None, reason_log, None,
        )
        worker.finished.connect(self._on_write_finished)
        worker.error.connect(self._on_write_error)
        self._threads.append((thread, worker))

    def _on_write_finished(self, result: twb_core.WriteCuratedResult) -> None:
        self.write_button.setEnabled(True)
        if result.gate_errors:
            self.write_status.setText(
                f"Wrote {result.new_target_path} but {len(result.gate_errors)} row(s) "
                f"failed the combustion_target parser gate -- inspect before using it: " +
                "; ".join(str(e.get("message", e)) for e in result.gate_errors[:5]))
        else:
            self.write_status.setText(
                f"Wrote {result.new_target_path} + {result.new_add_path} "
                f"(log: {result.log_path}). All rows passed the parser gate.")
        self.statusMessage.emit(f"Target Workbench: wrote {result.new_target_path}")

    def _on_write_error(self, message: str) -> None:
        self.write_button.setEnabled(True)
        QMessageBox.critical(self, "Target Workbench", f"Write failed: {message}")
