"""
mso_studio.gui.modules.nominal
=================================
Nominal simulation module page -- wraps :mod:`mso_studio.core.nominal`
(itself wrapping ``src/mso_compare/run_nominal_sim.py``). This is the
reference implementation every other module page follows: fields in,
:class:`~mso_studio.core.nominal.NominalSimParams` out, threaded run via the
shared :class:`~mso_studio.gui.widgets.module_page.ModulePage` base.
"""
from __future__ import annotations

from PySide6.QtWidgets import QLineEdit, QSpinBox

from mso_studio.core.backends import REPO_ROOT
from mso_studio.core.nominal import NominalSimParams, run_nominal
from mso_studio.core.runrecord import RunRecord
from mso_studio.gui.actions import Action, register
from mso_studio.gui.widgets.file_picker import FilePicker
from mso_studio.gui.widgets.mpl_canvas import CsvTableView
from mso_studio.gui.widgets.module_page import ModulePage
from mso_studio.gui.widgets.target_tables_button import add_target_tables_button

register(Action(
    id="nominal_sim.run", label="Run nominal simulation",
    calls="core.nominal.run_nominal", shortcut="F5", menu="Simulation",
    tooltip="Run one nominal simulation against experimental targets.",
))
register(Action(
    id="nominal_sim.report", label="Generate LaTeX report",
    calls="core.reporting.build_report", shortcut="Ctrl+R", menu="Results",
))
register(Action(
    id="nominal_sim.target_tables", label="Generate LaTeX target tables",
    calls="core.mso_reports.generate_target_tables", menu="Results",
))


class NominalSimPage(ModulePage):
    def __init__(self, parent=None):
        super().__init__(
            "nominal_sim", "Nominal Simulation",
            "Run one nominal simulation of a mechanism against experimental "
            "ignition-delay/flame-speed targets (src/mso_compare).",
            parent,
        )
        self.set_setup_title("Run configuration")
        self._build_fields()

        self.tables_view = CsvTableView(self)
        self.add_results_tab("Tables", self.tables_view)
        self._dataset_keys: list[str] = []

    def _build_fields(self) -> None:
        self.label_edit = QLineEdit("nominal_htc", self)
        self.add_setup_row("Run label", self.label_edit)

        self.mechanism_picker = FilePicker(mode="file", filter_="Mechanism files (*.yaml *.yml *.inp)", parent=self)
        self.mechanism_picker.edit.setPlaceholderText("Path to mechanism (.yaml or CHEMKIN)")
        self.add_setup_row("Mechanism", self.mechanism_picker)

        self.targets_picker = FilePicker(mode="file", filter_="Target files (*.target)", parent=self)
        self.targets_picker.edit.setPlaceholderText("Path to .target file")
        self.add_setup_row("Targets", self.targets_picker)

        self.targets_count_spin = QSpinBox(self)
        self.targets_count_spin.setRange(1, 100000)
        self.targets_count_spin.setValue(52)
        self.targets_count_spin.setMaximumWidth(100)
        self.add_setup_row("Target count", self.targets_count_spin)

        self.thermo_picker = FilePicker(mode="file", parent=self)
        self.thermo_picker.edit.setPlaceholderText("Path to thermo file")
        self.add_setup_row("Thermo file", self.thermo_picker)

        self.trans_picker = FilePicker(mode="file", parent=self)
        self.trans_picker.edit.setPlaceholderText("Path to transport file")
        self.add_setup_row("Transport file", self.trans_picker)

        self.base_opt_picker = FilePicker(mode="file", filter_="Opt files (*.opt)", parent=self)
        self.base_opt_picker.edit.setPlaceholderText("Base .opt YAML template")
        self.add_setup_row("Base .opt template", self.base_opt_picker)

        self.muqsac_dir_picker = FilePicker(mode="directory", must_exist=True, parent=self)
        self.muqsac_dir_picker.set_path(str(REPO_ROOT / "src" / "MUQ-SAC"))
        self.add_advanced_row("MUQ-SAC source dir", self.muqsac_dir_picker)

        self.output_dir_picker = FilePicker(mode="directory", must_exist=False, parent=self)
        self.output_dir_picker.edit.setPlaceholderText(f"default: {REPO_ROOT}/SIM_RUNS/<label>")
        self.add_advanced_row("Output directory", self.output_dir_picker)

        self.uncertainty_picker = FilePicker(mode="file", must_exist=False, parent=self)
        self.uncertainty_picker.edit.setPlaceholderText("optional — defaults to the targets file")
        self.add_advanced_row("Uncertainty data (unused for nominal)", self.uncertainty_picker)

        target_tables_button = add_target_tables_button(
            self, lambda: self.targets_picker.path(), self.statusMessage, self.log_view,
        )
        self.add_advanced_row("", target_tables_button)

    def build_core_call(self):
        missing = []
        if not self.mechanism_picker.is_valid:
            missing.append("Mechanism")
        if not self.targets_picker.is_valid:
            missing.append("Targets")
        if not self.thermo_picker.is_valid:
            missing.append("Thermo file")
        if not self.trans_picker.is_valid:
            missing.append("Transport file")
        if not self.base_opt_picker.is_valid:
            missing.append("Base .opt template")
        if missing:
            raise ValueError(f"Missing or invalid: {', '.join(missing)}.")

        self._current_label = self.label_edit.text().strip() or "nominal_run"
        params = NominalSimParams(
            label=self._current_label,
            mechanism=self.mechanism_picker.path(),
            targets=self.targets_picker.path(),
            thermo_file=self.thermo_picker.path(),
            trans_file=self.trans_picker.path(),
            targets_count=self.targets_count_spin.value(),
            base_opt_file=self.base_opt_picker.path(),
            muqsac_src_dir=self.muqsac_dir_picker.path() or str(REPO_ROOT / "src" / "MUQ-SAC"),
            output_dir=self.output_dir_picker.path(),
            uncertainty_data=self.uncertainty_picker.path(),
        )
        return run_nominal, (params,), {}

    def populate_results(self, record: RunRecord) -> None:
        self._dataset_keys = sorted(record.tables.keys())
        if self._dataset_keys:
            self.tables_view.load_csv(record.tables[self._dataset_keys[0]])
