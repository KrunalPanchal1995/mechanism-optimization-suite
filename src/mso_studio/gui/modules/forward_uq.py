"""
mso_studio.gui.modules.forward_uq
=====================================
Forward Uncertainty Quantification module page -- PySide6 port of
``forward_uq_gui.py``. Propagates Arrhenius parameter uncertainties through
a trained PRS surrogate (Monte-Carlo/Sobol/LHS) -- requires an existing
target.opt plus ``run_script.py``'s cached PRS coefficients (``Opt/``) for
that mechanism.
"""
from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QLabel, QSpinBox

from mso_studio.core.forward_uq import ForwardUqParams, run_forward_uq
from mso_studio.core.runrecord import RunRecord
from mso_studio.gui.actions import Action, register
from mso_studio.gui.widgets.file_picker import FilePicker
from mso_studio.gui.widgets.mpl_canvas import CsvTableView
from mso_studio.gui.widgets.module_page import ModulePage

register(Action(
    id="forward_uq.run", label="Run forward UQ", shortcut="F5",
    calls="core.forward_uq.run_forward_uq", menu="Simulation",
    tooltip="Propagate Arrhenius uncertainties through a trained PRS surrogate.",
))


class ForwardUQPage(ModulePage):
    def __init__(self, parent=None):
        super().__init__(
            "forward_uq", "Forward Uncertainty Quantification",
            "Propagate Arrhenius parameter uncertainties through a trained PRS "
            "surrogate (Monte-Carlo/Sobol/LHS) to get the full distribution of "
            "combustion observables (forward_uq.py).",
            parent,
        )
        self.set_setup_title("Forward UQ inputs")
        self._build_fields()

        self.tables_view = CsvTableView(self)
        self.add_results_tab("Statistics", self.tables_view)

    def _build_fields(self) -> None:
        self.target_opt_picker = FilePicker(mode="file", filter_="Opt files (*.opt)", parent=self)
        self.add_setup_row("target.opt", self.target_opt_picker)

        note = QLabel(
            "Requires src/MUQ-SAC/run_script.py's cached PRS coefficients "
            "(an Opt/ directory) to already exist inside the output "
            "directory below for this mechanism.", self)
        note.setObjectName("Caption")
        note.setWordWrap(True)
        self.add_setup_row("", note)

        self.n_samples_spin = QSpinBox(self)
        self.n_samples_spin.setRange(100, 10_000_000)
        self.n_samples_spin.setSingleStep(1000)
        self.n_samples_spin.setValue(50_000)
        self.add_setup_row("QMC sample count", self.n_samples_spin)

        self.method_combo = QComboBox(self)
        self.method_combo.addItems(["sobol", "lhs", "numpy"])
        self.add_setup_row("Sampling method", self.method_combo)

        self.seed_spin = QSpinBox(self)
        self.seed_spin.setRange(0, 2**31 - 1)
        self.seed_spin.setValue(42)
        self.add_advanced_row("Random seed", self.seed_spin)

        self.output_dir_picker = FilePicker(mode="directory", must_exist=True, parent=self)
        self.output_dir_picker.edit.setPlaceholderText(
            "the directory containing the Opt/ PRS cache")
        self.add_setup_row("Output directory (= Opt/ location)", self.output_dir_picker)

    def build_core_call(self):
        if not self.target_opt_picker.is_valid:
            raise ValueError("Choose a target.opt file first.")
        if not self.output_dir_picker.is_valid:
            raise ValueError(
                "Choose the output directory containing the Opt/ PRS cache.")

        self._current_label = self.target_opt_picker.path().rsplit("/", 1)[-1].rsplit(".", 1)[0]
        params = ForwardUqParams(
            target_opt=self.target_opt_picker.path(),
            n_samples=self.n_samples_spin.value(),
            method=self.method_combo.currentText(),
            seed=self.seed_spin.value(),
            output_dir=self.output_dir_picker.path(),
            label=self._current_label,
        )
        return run_forward_uq, (params,), {}

    def populate_results(self, record: RunRecord) -> None:
        stats_path = record.results.get("statistics_path")
        if stats_path:
            self.tables_view.load_csv(stats_path)
