"""
mso_studio.gui.modules.fusion
================================
Mechanism Fusion module page -- PySide6 port of ``mean_value_fusion_gui.py``.
KRUNAL_JULY_23.md point 6's fix (automatic glossary backfill + the
degenerate-empty-union guard) lives in ``_run_fusion()``, called via
``mso_studio.core.fusion.run_fusion``'s ``--fuse`` CLI bridge, so it applies
here too.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QLabel, QLineEdit, QPlainTextEdit, QPushButton,
)

from mso_studio.core.fusion import FusionParams, run_fusion, _DEFAULT_GLOSSARY
from mso_studio.core.runrecord import RunRecord
from mso_studio.gui.actions import Action, register
from mso_studio.gui.widgets.file_picker import FilePicker
from mso_studio.gui.widgets.module_page import ModulePage

register(Action(
    id="fusion.run", label="Fuse mechanisms", shortcut="F5",
    calls="core.fusion.run_fusion", menu="Simulation",
    tooltip="Fuse >=2 mechanisms: geometric-mean/priority-pick kinetics, element-balance validation.",
))


class FusionPage(ModulePage):
    def __init__(self, parent=None):
        super().__init__(
            "mechanism_fusion", "Mechanism Fusion",
            "Merge >=2 mechanisms — geometric mean of A, arithmetic mean of n/Ea "
            "for shared-but-divergent reactions (or priority-pick, verbatim from "
            "the highest-priority mechanism). Species/reaction identity is "
            "backfilled from the glossary automatically before comparing.",
            parent,
        )
        self.set_setup_title("Fusion inputs")
        self._build_fields()

        self.result_view = QPlainTextEdit(self)
        self.result_view.setReadOnly(True)
        self.add_results_tab("Result", self.result_view)

    def _build_fields(self) -> None:
        self.db_picker = FilePicker(mode="file", filter_="SQLite DB (*.db)", parent=self)
        self.add_setup_row("RMIP database", self.db_picker)

        self.mech_ids_edit = QLineEdit(self)
        self.mech_ids_edit.setPlaceholderText("comma-separated mechanism IDs, e.g. 2,3,4")
        self.add_setup_row("Mechanism IDs", self.mech_ids_edit)

        self.glossary_picker = FilePicker(mode="file", must_exist=False, parent=self)
        self.glossary_picker.set_path(_DEFAULT_GLOSSARY)
        self.add_setup_row("Species glossary CSV", self.glossary_picker)
        note = QLabel(
            "Applied automatically before fusing -- fills in any species/reaction "
            "identity a mechanism was parsed without a glossary for (this is what "
            "caused 'Fused_Mech (fused, 0 sp / 0 rxn)'). Leave blank to skip.",
            self)
        note.setObjectName("Caption")
        note.setWordWrap(True)
        self.add_setup_row("", note)

        self.output_name_edit = QLineEdit("Fused_Mech", self)
        self.add_setup_row("Fused mechanism name", self.output_name_edit)
        self.output_dir_picker = FilePicker(mode="directory", must_exist=False, parent=self)
        self.output_dir_picker.edit.setPlaceholderText("default: outputs/fusion")
        self.add_setup_row("Output directory", self.output_dir_picker)
        self.write_cantera_check = QCheckBox("Also write Cantera YAML (via ck2yaml)", self)
        self.add_setup_row("", self.write_cantera_check)

        self.tol_A_spin = QDoubleSpinBox(self)
        self.tol_A_spin.setDecimals(4)
        self.tol_A_spin.setRange(0.0, 10.0)
        self.tol_A_spin.setValue(0.01)
        self.add_advanced_row("Relative tol. on A", self.tol_A_spin)
        self.tol_n_spin = QDoubleSpinBox(self)
        self.tol_n_spin.setDecimals(4)
        self.tol_n_spin.setRange(0.0, 10.0)
        self.tol_n_spin.setValue(0.001)
        self.add_advanced_row("Absolute tol. on n", self.tol_n_spin)
        self.tol_Ea_spin = QDoubleSpinBox(self)
        self.tol_Ea_spin.setRange(0.0, 1e6)
        self.tol_Ea_spin.setValue(100.0)
        self.add_advanced_row("Absolute tol. on Ea (J/mol)", self.tol_Ea_spin)

        self.priority_mode_check = QCheckBox(
            "Priority-pick mode (verbatim from the highest-priority mechanism)", self)
        self.add_advanced_row("Strategy", self.priority_mode_check)
        self.priority_order_edit = QLineEdit(self)
        self.priority_order_edit.setPlaceholderText(
            "priority order, comma-separated IDs, highest first")
        self.priority_order_edit.setEnabled(False)
        self.priority_mode_check.toggled.connect(self.priority_order_edit.setEnabled)
        self.add_advanced_row("", self.priority_order_edit)

    def build_core_call(self):
        if not self.db_picker.is_valid:
            raise ValueError("Choose an RMIP database first.")
        ids_raw = self.mech_ids_edit.text().strip()
        if not ids_raw:
            raise ValueError("Enter at least 2 mechanism IDs.")
        try:
            mech_ids = [int(s.strip()) for s in ids_raw.split(",") if s.strip()]
        except ValueError:
            raise ValueError("Mechanism IDs must be comma-separated integers.")
        if len(mech_ids) < 2:
            raise ValueError("Enter at least 2 mechanism IDs to fuse.")

        priority_order = None
        if self.priority_mode_check.isChecked():
            po_raw = self.priority_order_edit.text().strip()
            if not po_raw:
                raise ValueError("Priority-pick mode is checked but no priority order was entered.")
            try:
                priority_order = [int(s.strip()) for s in po_raw.split(",") if s.strip()]
            except ValueError:
                raise ValueError("Priority order must be comma-separated integers.")
            if set(priority_order) != set(mech_ids):
                raise ValueError(
                    "Priority order must list exactly the same mechanism IDs entered above.")

        self._current_label = self.output_name_edit.text().strip() or "Fused_Mech"
        params = FusionParams(
            mechanism_ids=mech_ids,
            db_path=self.db_picker.path(),
            tol_A=self.tol_A_spin.value(), tol_n=self.tol_n_spin.value(),
            tol_Ea=self.tol_Ea_spin.value(),
            output_name=self._current_label,
            output_dir=self.output_dir_picker.path(),
            write_cantera=self.write_cantera_check.isChecked(),
            priority_order=priority_order,
            glossary_path=self.glossary_picker.path() or None,
            label=self._current_label,
        )
        return run_fusion, (params,), {}

    def populate_results(self, record: RunRecord) -> None:
        lines = [f"Status: {record.status}"]
        if record.error:
            lines.append(f"Error: {record.error}")
        for key in ("fused_mech_id", "n_species", "n_verbatim", "n_copy",
                    "n_fuse", "balanced", "output"):
            if key in record.results:
                lines.append(f"{key}: {record.results[key]}")
        self.result_view.setPlainText("\n".join(lines))
