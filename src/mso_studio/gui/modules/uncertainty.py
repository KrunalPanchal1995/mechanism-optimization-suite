"""
mso_studio.gui.modules.uncertainty
=====================================
Uncertainty Quantification and File Generator module page -- PySide6 port
of ``uncertainty_generator_gui.py`` (renamed together, KRUNAL_JULY_23.md
point 5). Primary workflow: SA data + mechanism + classification CSV
through src/ActiveParameterScreen.py (screening -> XML/plots/LaTeX/
filtered-mechanism). Manual XML-row entry and the symmetric-bounds
collapse (src/MUQ-SAC/Uncertainty.py) are kept as secondary sections, same
as the Tkinter app.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QPlainTextEdit, QPushButton,
    QVBoxLayout, QWidget,
)

from mso_studio.core.backends import REPO_ROOT
from mso_studio.gui.widgets.results_table import ResultsTable
from mso_studio.core.uncertainty import (
    ActiveParamScreenParams, GenerateXmlParams, ComputeBoundsParams,
    run_active_parameter_screen, run_generate_xml, run_compute_bounds,
)
from mso_studio.core.runrecord import RunRecord
from mso_studio.gui.actions import Action, register
from mso_studio.gui.widgets.file_picker import FilePicker
from mso_studio.gui.widgets.module_page import ModulePage

register(Action(
    id="uncertainty.active_param_screen", label="Run ActiveParameterScreen (from SA data)",
    calls="core.uncertainty.run_active_parameter_screen", menu="Simulation",
))
register(Action(
    id="uncertainty.generate_xml", label="Generate uncertainty XML",
    calls="core.uncertainty.run_generate_xml", menu="Simulation",
))
register(Action(
    id="uncertainty.compute_bounds", label="Compute symmetric bounds",
    calls="core.uncertainty.run_compute_bounds", shortcut="F5", menu="Simulation",
))

_DEMO_DIR = REPO_ROOT / "DATASET" / "METHYL_BUTANOATE_2026" / "ACTIVE_PARAMETER_SELECTION"
_DEMO = {
    "sens_dir": str(_DEMO_DIR / "METHYLBUTANOATE"),
    "mechanism": str(_DEMO_DIR / "MB_MB2D_LALIT_2024.yaml"),
    "classification_file": str(_DEMO_DIR / "results" / "master_active_reactions_CLASSIFIED.csv"),
    "fuel": "MB_MB2D_LALIT_2024",
    "threshold": "0.05",
    "carbon": "5",
}


class UncertaintyPage(ModulePage):
    def __init__(self, parent=None):
        super().__init__(
            "uncertainty", "Uncertainty Quantification and File Generator",
            "SA data + mechanism + classification CSV -> screening, uncertainty "
            "XML, plots, LaTeX tables, filtered mechanism "
            "(src/ActiveParameterScreen.py). Manual XML entry and symmetric "
            "A-factor bound collapse (src/MUQ-SAC/Uncertainty.py) are secondary.",
            parent,
        )
        self.set_setup_title("Compute symmetric bounds")
        self._build_sa_fields()
        self._build_fields()

        self._results_table = ResultsTable(
            ["Reaction", "A_nominal", "factor", "A_lb", "A_ub / error"], self)
        self.add_results_tab("Bounds", self._results_table)

        self._gen_log = QPlainTextEdit(self)
        self._gen_log.setReadOnly(True)
        self.add_results_tab("Generate XML log", self._gen_log)

        self._sa_log = QPlainTextEdit(self)
        self._sa_log.setReadOnly(True)
        self.add_results_tab("ActiveParameterScreen log", self._sa_log)

        self._gen_rows: list = []

    def _build_sa_fields(self) -> None:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.sens_dir_picker = FilePicker(mode="directory", parent=widget)
        layout.addWidget(QLabel("SA results dir (--sens-dir)"))
        layout.addWidget(self.sens_dir_picker)
        self.sa_mechanism_picker = FilePicker(
            mode="file", filter_="Mechanism files (*.yaml *.yml)", parent=widget)
        layout.addWidget(QLabel("Mechanism YAML (--mechanism)"))
        layout.addWidget(self.sa_mechanism_picker)
        self.classification_picker = FilePicker(
            mode="file", filter_="CSV (*.csv)", parent=widget)
        layout.addWidget(QLabel("Classification CSV (--classification-file)"))
        layout.addWidget(self.classification_picker)

        grid = QGridLayout()
        self.threshold_edit = QLineEdit("0.05", widget)
        grid.addWidget(QLabel("Threshold"), 0, 0)
        grid.addWidget(self.threshold_edit, 0, 1)
        self.carbon_edit = QLineEdit("0", widget)
        grid.addWidget(QLabel("Carbon cutoff"), 0, 2)
        grid.addWidget(self.carbon_edit, 0, 3)
        self.sa_fuel_edit = QLineEdit("Fuel", widget)
        grid.addWidget(QLabel("Fuel label"), 1, 0)
        grid.addWidget(self.sa_fuel_edit, 1, 1)
        self.results_dir_edit = QLineEdit("results", widget)
        grid.addWidget(QLabel("Results dir"), 1, 2)
        grid.addWidget(self.results_dir_edit, 1, 3)
        self.compare_temps_edit = QLineEdit(widget)
        self.compare_temps_edit.setPlaceholderText("space-sep K, e.g. 698 908 1287")
        grid.addWidget(QLabel("Compare temperatures"), 2, 0)
        grid.addWidget(self.compare_temps_edit, 2, 1, 1, 3)
        layout.addLayout(grid)

        check_grid = QGridLayout()
        self.generate_xml_check = QCheckBox("Generate XML", widget)
        check_grid.addWidget(self.generate_xml_check, 0, 0)
        self.generate_plots_check = QCheckBox("Generate plots", widget)
        check_grid.addWidget(self.generate_plots_check, 0, 1)
        self.per_case_plots_check = QCheckBox("Per-case plots", widget)
        check_grid.addWidget(self.per_case_plots_check, 0, 2)
        self.generate_latex_check = QCheckBox("Generate LaTeX", widget)
        check_grid.addWidget(self.generate_latex_check, 1, 0)
        self.longtable_check = QCheckBox("LaTeX longtable", widget)
        check_grid.addWidget(self.longtable_check, 1, 1)
        self.filtered_mech_check = QCheckBox("Filtered mechanism", widget)
        check_grid.addWidget(self.filtered_mech_check, 1, 2)
        layout.addLayout(check_grid)

        btn_row = QHBoxLayout()
        demo_btn = QPushButton("Load demo (Methyl Butanoate)", widget)
        demo_btn.clicked.connect(self._on_load_sa_demo)
        btn_row.addWidget(demo_btn)
        run_btn = QPushButton("Run ActiveParameterScreen", widget)
        run_btn.clicked.connect(self._on_run_sa_screen)
        btn_row.addWidget(run_btn)
        layout.addLayout(btn_row)

        self.add_setup_row("From SA data (ActiveParameterScreen)", widget)

    def _on_load_sa_demo(self) -> None:
        self.sens_dir_picker.set_path(_DEMO["sens_dir"])
        self.sa_mechanism_picker.set_path(_DEMO["mechanism"])
        self.classification_picker.set_path(_DEMO["classification_file"])
        self.sa_fuel_edit.setText(_DEMO["fuel"])
        self.threshold_edit.setText(_DEMO["threshold"])
        self.carbon_edit.setText(_DEMO["carbon"])
        self.compare_temps_edit.setText("698 908 1287")
        self.generate_xml_check.setChecked(True)
        self.generate_plots_check.setChecked(True)
        self.generate_latex_check.setChecked(True)
        self.statusMessage.emit("Loaded demo: DATASET/METHYL_BUTANOATE_2026/ACTIVE_PARAMETER_SELECTION")

    def _build_sa_params(self) -> "ActiveParamScreenParams":
        if not self.sa_mechanism_picker.is_valid:
            raise ValueError("Select a mechanism YAML first.")
        temps = [float(t) for t in self.compare_temps_edit.text().split()]
        return ActiveParamScreenParams(
            sens_dir=self.sens_dir_picker.path() or ".",
            mechanism=self.sa_mechanism_picker.path(),
            threshold=float(self.threshold_edit.text() or 0.05),
            carbon=int(self.carbon_edit.text() or 0),
            fuel=self.sa_fuel_edit.text().strip() or "Fuel",
            results_dir=self.results_dir_edit.text().strip() or "results",
            classification_file=self.classification_picker.path(),
            generate_xml=self.generate_xml_check.isChecked(),
            generate_plots=self.generate_plots_check.isChecked(),
            per_case_plots=self.per_case_plots_check.isChecked(),
            generate_latex=self.generate_latex_check.isChecked(),
            longtable=self.longtable_check.isChecked(),
            filtered_mechanism=self.filtered_mech_check.isChecked(),
            compare_temperatures=temps,
        )

    def _on_run_sa_screen(self) -> None:
        try:
            params = self._build_sa_params()
        except ValueError as e:
            self.statusMessage.emit(str(e))
            return
        self.statusMessage.emit("Running ActiveParameterScreen…")
        record = run_active_parameter_screen(params)
        self._sa_log.setPlainText(
            f"Status: {record.status}\n" +
            (f"Error: {record.error}\n" if record.error else "") +
            "\n".join(record.log_tail))
        if record.status == "done":
            self.statusMessage.emit(f"Done -> {record.results.get('results_dir', '')}")
        else:
            self.statusMessage.emit("ActiveParameterScreen failed — see log tab")

    def _build_fields(self) -> None:
        # -- Generate XML sub-section (its own button, doesn't use the page's
        # main Run button -- two independent actions, same as the Tk app) --
        gen_row_widget = QWidget(self)
        gen_layout = QVBoxLayout(gen_row_widget)
        gen_layout.setContentsMargins(0, 0, 0, 0)
        entry_row = QHBoxLayout()
        self._gen_rxn_edit = QLineEdit(gen_row_widget)
        self._gen_rxn_edit.setPlaceholderText("e.g. H+O2<=>O+OH")
        entry_row.addWidget(self._gen_rxn_edit, 2)
        self._gen_type_combo = QComboBox(gen_row_widget)
        self._gen_type_combo.addItems(["pressure_independent", "pressure_dependent"])
        entry_row.addWidget(self._gen_type_combo)
        self._gen_temps_edit = QLineEdit("500,2500", gen_row_widget)
        entry_row.addWidget(self._gen_temps_edit)
        self._gen_unsrt_edit = QLineEdit("0.3,0.3", gen_row_widget)
        entry_row.addWidget(self._gen_unsrt_edit)
        add_btn = QPushButton("+ Add row", gen_row_widget)
        add_btn.clicked.connect(self._on_add_gen_row)
        entry_row.addWidget(add_btn)
        gen_layout.addLayout(entry_row)

        self._gen_summary = QLabel("(no rows yet)", gen_row_widget)
        self._gen_summary.setObjectName("Caption")
        gen_layout.addWidget(self._gen_summary)

        self._gen_output_picker = FilePicker(mode="save", parent=gen_row_widget)
        self._gen_output_picker.set_path("outputs/uncertainty.xml")
        gen_layout.addWidget(self._gen_output_picker)

        gen_btn = QPushButton("Generate XML", gen_row_widget)
        gen_btn.clicked.connect(self._on_generate_xml_clicked)
        gen_layout.addWidget(gen_btn)
        self.add_setup_row("Generate uncertainty XML", gen_row_widget)

        # -- Compute bounds (drives the page's main Run button) --
        self.xml_picker = FilePicker(mode="file", filter_="XML (*.xml)", parent=self)
        self.add_setup_row("Uncertainty XML", self.xml_picker)
        self.mechanism_picker = FilePicker(
            mode="file", filter_="Mechanism files (*.yaml *.yml *.inp *.txt)", parent=self)
        self.add_setup_row("Mechanism file", self.mechanism_picker)
        self.output_csv_picker = FilePicker(mode="save", parent=self)
        self.output_csv_picker.set_path("outputs/uncertainty_bounds.csv")
        self.add_setup_row("Output CSV", self.output_csv_picker)

    def _on_add_gen_row(self) -> None:
        name = self._gen_rxn_edit.text().strip()
        if not name:
            self.statusMessage.emit("Enter a reaction string first.")
            return
        self._gen_rows.append({
            "name": name, "type": self._gen_type_combo.currentText(),
            "temperatures": self._gen_temps_edit.text().strip(),
            "uncertainties": self._gen_unsrt_edit.text().strip(),
        })
        self._gen_rxn_edit.clear()
        self._gen_summary.setText(f"{len(self._gen_rows)} row(s) added")

    def _on_generate_xml_clicked(self) -> None:
        if not self._gen_rows:
            self.statusMessage.emit("Add at least one reaction row first.")
            return
        out_path = self._gen_output_picker.path()
        if not out_path:
            self.statusMessage.emit("Set an output XML path first.")
            return
        params = GenerateXmlParams(reactions=list(self._gen_rows), output_path=out_path)
        record = run_generate_xml(params)
        self._gen_log.setPlainText(
            f"Status: {record.status}\n" +
            (f"Error: {record.error}\n" if record.error else "") +
            "\n".join(record.log_tail))
        if record.status == "done":
            self.xml_picker.set_path(out_path)
            self.statusMessage.emit(f"Wrote {len(self._gen_rows)} reaction(s) -> {out_path}")
        else:
            self.statusMessage.emit("XML generation failed — see 'Generate XML log' tab")

    def build_core_call(self):
        if not self.xml_picker.is_valid:
            raise ValueError("Select an uncertainty XML first.")
        if not self.mechanism_picker.is_valid:
            raise ValueError("Select a mechanism file first.")
        self._current_label = self.xml_picker.path().rsplit("/", 1)[-1]
        params = ComputeBoundsParams(
            xml_path=self.xml_picker.path(), mechanism=self.mechanism_picker.path(),
            output_csv=self.output_csv_picker.path(), label=self._current_label,
        )
        return run_compute_bounds, (params,), {}

    def populate_results(self, record: RunRecord) -> None:
        rows = record.results.get("rows", [])
        table = self._results_table
        out_rows = []
        for r in rows:
            if "error" in r:
                out_rows.append([r.get("rxn", ""), "", "", "", f"ERROR: {r['error']}"])
            else:
                out_rows.append([
                    r.get("rxn", ""), f"{r['A_nominal']:.4g}", f"{r['factor']:.4g}",
                    f"{r['A_lb']:.4g}", f"{r['A_ub']:.4g}",
                ])
        table.set_rows(out_rows)
        table.resizeColumnsToContents()
