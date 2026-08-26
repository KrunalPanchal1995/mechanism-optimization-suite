"""
mso_studio.gui.modules.rmip_suite
=====================================
RMIP Suite module page -- PySide6 port of ``rmip_suite_gui.py``: one shared
RMIP database field + 6 independent tabs (Graph Comparison, Mechanism
Reduction, Global Sensitivity Analysis, Active Parameter Selection, Target
Builder, ReSpecTh Import). A plain ``QWidget`` + ``QTabWidget`` (not
``ModulePage``) since each tab is its own independent action, matching
``ToolsPage``'s existing multi-action pattern in this codebase rather than
the single-action ``ModulePage`` contract.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox,
    QTabWidget, QVBoxLayout, QWidget,
)
from PySide6.QtCore import Qt, QTimer, Signal

from mso_io.run_status import RunStatusReport
from mso_studio.core import rmip_suite as core
from mso_studio.gui.actions import Action, register
from mso_studio.gui.widgets.file_picker import FilePicker
from mso_studio.gui.widgets.log_view import LogView
from mso_studio.gui.widgets.module_hero import make_module_hero
from mso_studio.gui.widgets.results_table import ResultsTable
from mso_studio.gui.workers import run_in_thread

register(Action(
    id="rmip_suite.compare_graph", label="Compare mechanism graphs",
    calls="core.rmip_suite.run_graph_compare", menu="Simulation",
))
register(Action(
    id="rmip_suite.reduce", label="Reduce mechanism",
    calls="core.rmip_suite.run_reduce", menu="Simulation",
))
register(Action(
    id="rmip_suite.gsa", label="Run global sensitivity analysis",
    calls="core.rmip_suite.run_gsa", menu="Simulation",
))


class _RmipTabBase(QWidget):
    """Shared run/log/worker plumbing for one RMIP Suite tab -- a lighter
    analogue of ``ModulePage`` (no setup/results-splitter chrome, since
    these are meant to be compact tabs within one page, not full pages)."""

    statusMessage = Signal(str)

    def __init__(self, db_picker: FilePicker, parent=None):
        super().__init__(parent)
        self.db_picker = db_picker
        self._thread = None
        self._worker = None
        self._last_record = None

        self._root = QVBoxLayout(self)
        self._fields_container = QWidget(self)
        self._fields_layout = QVBoxLayout(self._fields_container)
        self._fields_layout.setContentsMargins(0, 0, 0, 0)
        self._root.addWidget(self._fields_container)

        btn_row = QHBoxLayout()
        self._btn_row = btn_row
        self.run_button = QPushButton("▶  Run", self)
        self.run_button.clicked.connect(self._on_run_clicked)
        btn_row.addWidget(self.run_button)
        self.export_status_button = QPushButton("Export run status…", self)
        self.export_status_button.clicked.connect(self._on_export_status_clicked)
        btn_row.addWidget(self.export_status_button)
        self._extra_buttons(btn_row)
        btn_row.addStretch(1)
        self._root.addLayout(btn_row)

        self.log_view = LogView(self)
        self.log_view.setMaximumHeight(160)
        self._root.addWidget(self.log_view)

    def _field(self, label: str, widget: QWidget) -> None:
        row = QHBoxLayout()
        row.addWidget(QLabel(label, self), 0)
        row.addWidget(widget, 1)
        self._fields_layout.addLayout(row)

    def build_core_call(self):
        raise NotImplementedError

    def populate_results(self, record) -> None:
        pass

    def _extra_buttons(self, btn_row: QHBoxLayout) -> None:
        """Hook for a subclass to append its own buttons/widgets to the
        shared Run/Export-status row, called from __init__ right before the
        trailing stretch -- no-op by default. Mirrors rmip_suite_gui.py's
        (Tkinter) own `_TerminalTab._extra_buttons()` hook
        (GraphComparisonTab's "Launch graph viewer" button)."""
        pass

    def _on_run_clicked(self) -> None:
        try:
            core_fn, args, kwargs = self.build_core_call()
        except ValueError as exc:
            self.statusMessage.emit(str(exc))
            self.log_view.append_status(f"Not started: {exc}")
            return
        self.run_button.setEnabled(False)
        self.log_view.append_status("Run started")
        self.statusMessage.emit("Running…")

        self._thread, self._worker = run_in_thread(core_fn, *args, **kwargs)
        self._worker.progress.connect(self.log_view.append_line)
        self._worker.finished.connect(self._on_run_finished)
        self._worker.error.connect(self._on_run_error)

    def _on_run_finished(self, record) -> None:
        self.run_button.setEnabled(True)
        self._last_record = record
        self.log_view.append_status(f"Run {record.status}")
        self.statusMessage.emit(f"Run {record.status}")
        try:
            self.populate_results(record)
        except Exception as exc:  # noqa: BLE001 - never hide a successful run
            self.log_view.append_status(f"Results preview failed: {exc}")

    def _on_run_error(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self.log_view.append_status(f"Failed: {message}")
        self.statusMessage.emit("Run failed — see log")

    def _on_export_status_clicked(self) -> None:
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        if self._last_record is None:
            QMessageBox.information(self, "Nothing to export", "Run this tab first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export run status", "", "Text (*.txt)")
        if not path:
            return
        RunStatusReport.from_run_record(self._last_record).write_txt(path)
        QMessageBox.information(self, "Exported", f"Run status written to:\n{path}")


class GraphCompareTab(_RmipTabBase):
    def __init__(self, db_picker, parent=None):
        self._serve_proc = None  # set before super().__init__() -- _extra_buttons runs inside it
        super().__init__(db_picker, parent)
        self.mech_ids_edit = QLineEdit(self)
        self.mech_ids_edit.setPlaceholderText("mechanism IDs, comma-sep., e.g. 2,3,4")
        self._field("Mechanism IDs", self.mech_ids_edit)
        self.output_dir_picker = FilePicker(mode="directory", must_exist=False, parent=self)
        self._field("Output directory", self.output_dir_picker)
        self.tol_A_spin = QDoubleSpinBox(self)
        self.tol_A_spin.setDecimals(4); self.tol_A_spin.setValue(0.01)
        self._field("Relative tol. on A", self.tol_A_spin)
        self.tol_n_spin = QDoubleSpinBox(self)
        self.tol_n_spin.setDecimals(4); self.tol_n_spin.setValue(0.001)
        self._field("Absolute tol. on n", self.tol_n_spin)
        self.tol_Ea_spin = QDoubleSpinBox(self)
        self.tol_Ea_spin.setRange(0, 1e6); self.tol_Ea_spin.setValue(100.0)
        self._field("Absolute tol. on Ea (J/mol)", self.tol_Ea_spin)

    def _extra_buttons(self, btn_row: QHBoxLayout) -> None:
        """KRUNAL_JULY_23.md point 7 (Phase 5c, bug.md): mirrors
        rmip_suite_gui.py's (Tkinter) GraphComparisonTab -- a port spinbox,
        a formula/id label-mode combo, a launch button, and a clickable
        link label that lights up once the server is confirmed serving."""
        self.viewer_port_spin = QSpinBox(self)
        self.viewer_port_spin.setRange(1024, 65535)
        self.viewer_port_spin.setValue(8050)
        btn_row.addWidget(QLabel("Port:", self))
        btn_row.addWidget(self.viewer_port_spin)

        self.viewer_label_combo = QComboBox(self)
        self.viewer_label_combo.addItems(["formula", "id"])
        btn_row.addWidget(self.viewer_label_combo)

        self.launch_viewer_button = QPushButton("🌐  Launch graph viewer", self)
        self.launch_viewer_button.clicked.connect(self._on_launch_viewer_clicked)
        btn_row.addWidget(self.launch_viewer_button)

        self.viewer_link_label = QLabel("", self)
        self.viewer_link_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.viewer_link_label.setOpenExternalLinks(True)
        btn_row.addWidget(self.viewer_link_label)

    def build_core_call(self):
        if not self.db_picker.is_valid:
            raise ValueError("Choose an RMIP database first.")
        ids_raw = self.mech_ids_edit.text().strip()
        try:
            ids = [int(s.strip()) for s in ids_raw.split(",") if s.strip()]
        except ValueError:
            raise ValueError("Mechanism IDs must be comma-separated integers.")
        if len(ids) < 2:
            raise ValueError("Enter at least 2 mechanism IDs.")
        params = core.GraphCompareParams(
            db_path=self.db_picker.path(), mechanism_ids=ids,
            output_dir=self.output_dir_picker.path(),
            tol_A=self.tol_A_spin.value(), tol_n=self.tol_n_spin.value(),
            tol_Ea=self.tol_Ea_spin.value(),
        )
        return core.run_graph_compare, (params,), {}

    def _on_launch_viewer_clicked(self) -> None:
        try:
            _fn, (params,), _kwargs = self.build_core_call()
        except ValueError as exc:
            self.statusMessage.emit(str(exc))
            self.log_view.append_status(f"Viewer not started: {exc}")
            return

        self._stop_viewer()  # replace any previously-launched server on this tab

        port = self.viewer_port_spin.value()
        label_mode = self.viewer_label_combo.currentText()
        self._serve_proc = core.serve_graph_compare(params, port=port, label_mode=label_mode)
        url = f"http://localhost:{port}"
        self.viewer_link_label.setText(f'<a href="{url}">{url}</a>')
        self.log_view.append_status(f"Graph viewer starting at {url} (pid {self._serve_proc.pid})")
        self.statusMessage.emit(f"Graph viewer: {url}")

        QTimer.singleShot(2500, lambda: self._open_link(url))

    def _open_link(self, url: str) -> None:
        import webbrowser
        webbrowser.open(url)

    def _stop_viewer(self) -> None:
        if self._serve_proc is not None and self._serve_proc.poll() is None:
            self._serve_proc.terminate()
            try:
                self._serve_proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                self._serve_proc.kill()
        self._serve_proc = None

    def closeEvent(self, event) -> None:  # noqa: N802 -- Qt override
        self._stop_viewer()
        super().closeEvent(event)

    def __del__(self):
        # Best-effort -- Qt widget teardown doesn't reliably call
        # closeEvent for tabs removed from a QTabWidget without the whole
        # window closing, so stop the server here too rather than leak it.
        try:
            self._stop_viewer()
        except Exception:  # noqa: BLE001
            pass


class ReduceTab(_RmipTabBase):
    def __init__(self, db_picker, parent=None):
        super().__init__(db_picker, parent)
        self.mech_id_edit = QLineEdit(self)
        self._field("Mechanism ID", self.mech_id_edit)
        self.target_edit = QLineEdit(self)
        self.target_edit.setPlaceholderText("target species, comma-sep.")
        self._field("Target species", self.target_edit)
        self.eps_max_spin = QDoubleSpinBox(self)
        self.eps_max_spin.setValue(0.5)
        self._field("Max epsilon", self.eps_max_spin)
        self.eps_steps_spin = QSpinBox(self)
        self.eps_steps_spin.setRange(1, 1000); self.eps_steps_spin.setValue(51)
        self._field("Epsilon steps", self.eps_steps_spin)
        self.mech_path_picker = FilePicker(mode="file", must_exist=False, parent=self)
        self._field("Mechanism file (for Cantera rates)", self.mech_path_picker)
        self.output_dir_picker = FilePicker(mode="directory", must_exist=False, parent=self)
        self._field("Output directory", self.output_dir_picker)

    def build_core_call(self):
        if not self.db_picker.is_valid:
            raise ValueError("Choose an RMIP database first.")
        if not self.mech_id_edit.text().strip():
            raise ValueError("Enter a mechanism ID.")
        params = core.ReduceParams(
            db_path=self.db_picker.path(), mechanism_id=self.mech_id_edit.text().strip(),
            target_species=self.target_edit.text().strip(),
            eps_max=self.eps_max_spin.value(), eps_steps=self.eps_steps_spin.value(),
            mech_path=self.mech_path_picker.path(),
            output_dir=self.output_dir_picker.path(),
        )
        return core.run_reduce, (params,), {}


class GsaTab(_RmipTabBase):
    def __init__(self, db_picker, parent=None):
        super().__init__(db_picker, parent)
        self.mech_id_edit = QLineEdit(self)
        self._field("Mechanism ID", self.mech_id_edit)
        self.fuel_edit = QLineEdit(self)
        self.fuel_edit.setPlaceholderText("e.g. H2:1.0")
        self._field("Fuel", self.fuel_edit)
        self.zerork_picker = FilePicker(mode="directory", must_exist=False, parent=self)
        self._field("Zero-RK install prefix", self.zerork_picker)
        self.output_dir_picker = FilePicker(mode="directory", must_exist=False, parent=self)
        self._field("Output directory", self.output_dir_picker)

    def build_core_call(self):
        if not self.db_picker.is_valid:
            raise ValueError("Choose an RMIP database first.")
        if not self.mech_id_edit.text().strip():
            raise ValueError("Enter a mechanism ID.")
        params = core.GsaParams(
            db_path=self.db_picker.path(), mechanism_id=self.mech_id_edit.text().strip(),
            fuel=self.fuel_edit.text().strip(),
            zerork_install_prefix=self.zerork_picker.path(),
            output_dir=self.output_dir_picker.path(),
        )
        return core.run_gsa, (params,), {}


class ActiveParamSelectionTab(_RmipTabBase):
    def __init__(self, db_picker, parent=None):
        super().__init__(db_picker, parent)
        self.sens_dir_picker = FilePicker(mode="directory", parent=self)
        self._field("SA results directory", self.sens_dir_picker)
        self.mechanism_picker = FilePicker(mode="file", filter_="Cantera YAML (*.yaml *.yml)", parent=self)
        self._field("Mechanism YAML", self.mechanism_picker)
        self.threshold_spin = QDoubleSpinBox(self)
        self.threshold_spin.setDecimals(3); self.threshold_spin.setValue(0.05)
        self._field("Sensitivity threshold", self.threshold_spin)
        self.classification_picker = FilePicker(mode="file", must_exist=False,
                                                filter_="CSV (*.csv)", parent=self)
        self._field("Classification CSV (for XML/plots/LaTeX)", self.classification_picker)
        self.generate_xml_check = QCheckBox("Generate XML", self)
        self._field("", self.generate_xml_check)
        self.generate_plots_check = QCheckBox("Generate plots", self)
        self._field("", self.generate_plots_check)
        self.generate_latex_check = QCheckBox("Generate LaTeX tables", self)
        self._field("", self.generate_latex_check)
        self.results_dir_picker = FilePicker(mode="directory", must_exist=False, parent=self)
        self._field("Results directory", self.results_dir_picker)

    def build_core_call(self):
        if not self.mechanism_picker.is_valid:
            raise ValueError("Choose a mechanism YAML first.")
        params = core.ActiveParamSelectionParams(
            sens_dir=self.sens_dir_picker.path() or ".",
            mechanism=self.mechanism_picker.path(),
            threshold=self.threshold_spin.value(),
            classification_file=self.classification_picker.path(),
            generate_xml=self.generate_xml_check.isChecked(),
            generate_plots=self.generate_plots_check.isChecked(),
            generate_latex=self.generate_latex_check.isChecked(),
            results_dir=self.results_dir_picker.path(),
        )
        return core.run_active_param_selection, (params,), {}


class TargetBuilderTab(_RmipTabBase):
    def __init__(self, db_picker, parent=None):
        super().__init__(db_picker, parent)
        self.mech_id_edit = QLineEdit(self)
        self._field("Mechanism ID", self.mech_id_edit)
        self.base_opt_picker = FilePicker(mode="file", filter_="target.opt (*.opt)", parent=self)
        self._field("Base target.opt", self.base_opt_picker)
        self.work_dir_picker = FilePicker(mode="directory", must_exist=False, parent=self)
        self._field("Work directory", self.work_dir_picker)

        self.fuel_table = ResultsTable(["Species", "Mole fraction"], self)
        self._fields_layout.addWidget(QLabel("Fuel blend", self))
        self._fields_layout.addWidget(self.fuel_table)
        add_row = QHBoxLayout()
        self._fuel_sp_edit = QLineEdit(self)
        self._fuel_frac_edit = QLineEdit("1.0", self)
        add_btn = QPushButton("+ Add fuel", self)
        add_btn.clicked.connect(self._on_add_fuel_row)
        add_row.addWidget(self._fuel_sp_edit)
        add_row.addWidget(self._fuel_frac_edit)
        add_row.addWidget(add_btn)
        self._fields_layout.addLayout(add_row)
        self._fuel_rows: list = []

    def _on_add_fuel_row(self) -> None:
        sp = self._fuel_sp_edit.text().strip()
        if not sp:
            self.statusMessage.emit("Enter a fuel species name first.")
            return
        try:
            frac = float(self._fuel_frac_edit.text())
        except ValueError:
            self.statusMessage.emit("Mole fraction must be numeric.")
            return
        self._fuel_rows.append((sp, frac))
        self.fuel_table.set_rows(self._fuel_rows)
        self._fuel_sp_edit.clear()

    def build_core_call(self):
        if not self.db_picker.is_valid:
            raise ValueError("Choose an RMIP database first.")
        if not self.mech_id_edit.text().strip():
            raise ValueError("Enter a mechanism ID.")
        params = core.TargetBuilderParams(
            db_path=self.db_picker.path(), mechanism_id=self.mech_id_edit.text().strip(),
            base_target_opt=self.base_opt_picker.path(),
            fuel_blend=list(self._fuel_rows),
            work_dir=self.work_dir_picker.path(),
        )
        return core.run_build_target, (params,), {}


class RespecthImportTab(_RmipTabBase):
    def __init__(self, db_picker, parent=None):
        super().__init__(db_picker, parent)
        self.xml_picker = FilePicker(mode="file", filter_="XML (*.xml)", must_exist=False, parent=self)
        self._field("XML file or directory", self.xml_picker)
        self.fuel_edit = QLineEdit("NH3", self)
        self._field("Fuel (comma-sep. species)", self.fuel_edit)
        self.out_dir_picker = FilePicker(mode="directory", must_exist=False, parent=self)
        self._field("Output directory", self.out_dir_picker)
        self.no_db_check = QCheckBox("--no-db (legacy artefacts only)", self)
        self._field("", self.no_db_check)

    def build_core_call(self):
        if not self.db_picker.is_valid:
            raise ValueError("Choose an RMIP database first.")
        if not self.xml_picker.path():
            raise ValueError("Choose an XML file or directory.")
        params = core.RespecthImportParams(
            db_path=self.db_picker.path(), xml_path=self.xml_picker.path(),
            fuel=self.fuel_edit.text().strip() or "NH3",
            out_dir=self.out_dir_picker.path(), no_db=self.no_db_check.isChecked(),
        )
        return core.run_import_respecth, (params,), {}


class RMIPSuitePage(QWidget):
    statusMessage = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        self.hero = make_module_hero(
            "rmip_suite", "RMIP Suite",
            "Graph comparison, mechanism reduction, global sensitivity analysis, "
            "active parameter selection, target-file building, and ReSpecTh/RKD "
            "XML import (mechanism fusion has its own page, see Mechanism Fusion).",
            self,
        )
        root.addWidget(self.hero)

        db_row = QHBoxLayout()
        db_row.addWidget(QLabel("RMIP database (shared by all tabs)", self))
        self.db_picker = FilePicker(mode="file", filter_="SQLite DB (*.db)", parent=self)
        db_row.addWidget(self.db_picker, 1)
        root.addLayout(db_row)

        tabs = QTabWidget(self)
        root.addWidget(tabs, 1)
        self.graph_compare_tab = GraphCompareTab(self.db_picker, self)
        self.reduce_tab = ReduceTab(self.db_picker, self)
        self.gsa_tab = GsaTab(self.db_picker, self)
        self.active_param_tab = ActiveParamSelectionTab(self.db_picker, self)
        self.target_builder_tab = TargetBuilderTab(self.db_picker, self)
        self.respecth_import_tab = RespecthImportTab(self.db_picker, self)
        for tab, label in (
            (self.graph_compare_tab, "Graph Comparison"),
            (self.reduce_tab, "Mechanism Reduction"),
            (self.gsa_tab, "Global Sensitivity Analysis"),
            (self.active_param_tab, "Active Parameter Selection"),
            (self.target_builder_tab, "Target Builder"),
            (self.respecth_import_tab, "ReSpecTh Import"),
        ):
            tabs.addTab(tab, label)
            tab.statusMessage.connect(self.statusMessage.emit)
