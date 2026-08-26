"""
mso_studio.gui.modules.optimization
======================================
Conventional optimization module page -- wraps
:mod:`mso_studio.core.optimization` (``src/MUQ-SAC/run_script.py``):
PRS-surrogate posterior optimization of Arrhenius A-factors. Can be a long
run (GA + L-BFGS-B/SLSQP over many PRS evaluations); the shared
:class:`~mso_studio.gui.widgets.module_page.ModulePage` threading makes that
safe -- the window stays responsive and Stop works.
"""
from __future__ import annotations

from mso_studio.core.optimization import OptimizationParams, run_optimization
from mso_studio.core.runrecord import RunRecord
from mso_studio.gui.actions import Action, register
from mso_studio.gui.widgets.file_picker import FilePicker
from mso_studio.gui.widgets.mpl_canvas import CsvTableView
from mso_studio.gui.widgets.module_page import ModulePage

register(Action(
    id="optimization.run", label="Run optimization",
    calls="core.optimization.run_optimization", shortcut="Ctrl+F5", menu="Simulation",
    tooltip="Run PRS-surrogate posterior optimization of the mechanism's A-factors.",
))


class OptimizationPage(ModulePage):
    def __init__(self, parent=None):
        super().__init__(
            "optimization", "Optimization",
            "Conventional mechanism optimization against ignition-delay/"
            "flame-speed targets (src/MUQ-SAC/run_script.py). This can run "
            "for a long time -- the window stays responsive and Stop works.",
            parent,
        )
        self.set_setup_title("Run configuration")
        self._build_fields()

        self.tables_view = CsvTableView(self)
        self.add_results_tab("Cost / errors", self.tables_view)

    def _build_fields(self) -> None:
        self.target_opt_picker = FilePicker(mode="file", filter_="Opt files (*.opt)", parent=self)
        self.target_opt_picker.edit.setPlaceholderText(
            "Fully-populated target.opt (mechanism, targets, uncertainty, Stats/PRS hyperparameters)"
        )
        self.add_setup_row("target.opt", self.target_opt_picker)

        self.output_dir_picker = FilePicker(mode="directory", must_exist=False, parent=self)
        self.output_dir_picker.edit.setPlaceholderText("default: alongside target.opt")
        self.add_advanced_row("Output / working directory", self.output_dir_picker)

    def build_core_call(self):
        if not self.target_opt_picker.is_valid:
            raise ValueError("Pick a fully-populated target.opt file first.")

        self._current_label = "optimization"
        params = OptimizationParams(
            target_opt_path=self.target_opt_picker.path(),
            output_dir=self.output_dir_picker.path(),
        )
        self._current_label = params.label
        return run_optimization, (params,), {}

    def populate_results(self, record: RunRecord) -> None:
        cost_path = record.tables.get("optimization_cost")
        if cost_path:
            self.tables_view.load_csv(cost_path)
        elif "prediction_errors" in record.tables:
            self.tables_view.load_csv(record.tables["prediction_errors"])
        if record.results.get("optimized_mechanism_path"):
            self.log_view.append_status(
                f"Optimized mechanism: {record.results['optimized_mechanism_path']}"
            )
