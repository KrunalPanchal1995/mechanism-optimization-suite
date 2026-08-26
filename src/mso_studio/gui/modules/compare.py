"""
mso_studio.gui.modules.compare
=================================
Mechanism-comparison module page -- wraps :mod:`mso_studio.core.compare`
(the full 4-phase ``run_mso_pipeline.py``: simulate -> merge -> error
metrics -> plots). Unlike nominal sim, this takes an existing
``mso_config.yaml`` rather than building one from scratch -- the schema
supports up to six run labels (nominal/stage1/stage2 x htc/ltc), which
belongs in a text config, not a wall of GUI fields.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from PySide6.QtWidgets import QCheckBox, QComboBox

from mso_studio.core.compare import ComparePipelineParams, run_compare
from mso_studio.core.runrecord import RunRecord
from mso_studio.gui.actions import Action, register
from mso_studio.gui.widgets.file_picker import FilePicker
from mso_studio.gui.widgets.mpl_canvas import CsvTableView
from mso_studio.gui.widgets.module_page import ModulePage
from mso_studio.gui.widgets.target_tables_button import add_target_tables_button

register(Action(
    id="mso_compare.run", label="Run mechanism comparison",
    calls="core.compare.run_compare", shortcut="Ctrl+Shift+R", menu="Simulation",
    tooltip="Run the 4-phase simulate/merge/error-metrics/plot pipeline.",
))
register(Action(
    id="mso_compare.target_tables", label="Generate LaTeX target tables",
    calls="core.mso_reports.generate_target_tables", menu="Results",
))


class ComparePage(ModulePage):
    def __init__(self, parent=None):
        super().__init__(
            "mso_compare", "Mechanism Comparison",
            "Run the full simulate -> merge -> error metrics -> plot pipeline "
            "(src/run_mso_pipeline.py) against an existing mso_config.yaml.",
            parent,
        )
        self.set_setup_title("Pipeline configuration")
        self._build_fields()

        self.tables_view = CsvTableView(self)
        self.add_results_tab("Error metrics", self.tables_view)
        self._table_keys: list[str] = []

    def _build_fields(self) -> None:
        self.config_picker = FilePicker(mode="file", filter_="YAML files (*.yaml *.yml)", parent=self)
        self.config_picker.edit.setPlaceholderText("Path to mso_config.yaml")
        self.add_setup_row("Pipeline config", self.config_picker)

        self.regime_combo = QComboBox(self)
        self.regime_combo.addItems(["both", "htc", "ltc"])
        self.add_setup_row("Regime", self.regime_combo)

        self.skip_sims_check = QCheckBox("Skip Phase 1 (use existing simulation outputs)", self)
        self.add_advanced_row("", self.skip_sims_check)
        self.skip_merge_check = QCheckBox("Skip Phase 2 (use existing merged outputs)", self)
        self.add_advanced_row("", self.skip_merge_check)
        self.skip_errors_check = QCheckBox("Skip Phase 3 (error metrics)", self)
        self.add_advanced_row("", self.skip_errors_check)
        self.skip_plots_check = QCheckBox("Skip Phase 4 (plots)", self)
        self.add_advanced_row("", self.skip_plots_check)
        self.dry_run_check = QCheckBox("Dry run (print plan only)", self)
        self.add_advanced_row("", self.dry_run_check)

        target_tables_button = add_target_tables_button(
            self, self._first_targets_path_in_config, self.statusMessage, self.log_view,
        )
        self.add_advanced_row("", target_tables_button)

    def _first_targets_path_in_config(self) -> str:
        """The .target file for whichever simulation label appears first in
        the loaded mso_config.yaml -- most configs share one targets file
        across regimes/stages, so this is a reasonable one-click default."""
        if not self.config_picker.is_valid:
            return ""
        try:
            raw = yaml.safe_load(Path(self.config_picker.path()).read_text())
            for sim in (raw.get("simulations") or {}).values():
                targets = sim.get("targets")
                if targets:
                    return targets
        except (OSError, yaml.YAMLError):
            pass
        return ""

    def build_core_call(self):
        if not self.config_picker.is_valid:
            raise ValueError("Pick an existing mso_config.yaml first.")

        self._current_label = "mso_compare"
        params = ComparePipelineParams(
            config_path=self.config_picker.path(),
            regime=self.regime_combo.currentText(),
            skip_simulations=self.skip_sims_check.isChecked(),
            skip_merge=self.skip_merge_check.isChecked(),
            skip_error_metrics=self.skip_errors_check.isChecked(),
            skip_plots=self.skip_plots_check.isChecked(),
            dry_run=self.dry_run_check.isChecked(),
        )
        self._current_label = params.label
        return run_compare, (params,), {}

    def populate_results(self, record: RunRecord) -> None:
        self._table_keys = sorted(record.tables.keys())
        if self._table_keys:
            self.tables_view.load_csv(record.tables[self._table_keys[0]])
