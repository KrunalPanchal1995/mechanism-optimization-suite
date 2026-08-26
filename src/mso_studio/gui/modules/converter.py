"""
mso_studio.gui.modules.converter
===================================
Mechanism Converter module page -- PySide6 port of ``mechanism_converter_gui.py``
(KRUNAL_JULY_23.md point 2's fix lives in the shared ``convert()`` this page
calls, via :mod:`mso_studio.core.converter`, so the species-name-corruption
bug is fixed here too, not just in the Tkinter original).
"""
from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QLabel, QPlainTextEdit

from mso_studio.core.converter import ConverterParams, run_convert
from mso_studio.core.runrecord import RunRecord
from mso_studio.gui.actions import Action, register
from mso_studio.gui.widgets.file_picker import FilePicker
from mso_studio.gui.widgets.module_page import ModulePage

register(Action(
    id="converter.run", label="Convert CHEMKIN mechanism", shortcut="F5",
    calls="core.converter.run_convert", menu="Simulation",
    tooltip="Convert a CHEMKIN mechanism to Cantera YAML / FlameMaster .pre / RMIP DB.",
))


class ConverterPage(ModulePage):
    def __init__(self, parent=None):
        super().__init__(
            "mechanism_converter", "Mechanism Converter",
            "CHEMKIN .inp/.mech + thermo + transport → Cantera YAML (ck2yaml), "
            "FlameMaster .pre (ScanMan), and/or RMIP database.",
            parent,
        )
        self.set_setup_title("Conversion inputs")
        self._build_fields()

        self.result_view = QPlainTextEdit(self)
        self.result_view.setReadOnly(True)
        self.add_results_tab("Result", self.result_view)

    def _build_fields(self) -> None:
        self.mechanism_picker = FilePicker(
            mode="file", filter_="CHEMKIN mechanism (*.inp *.mech);;All files (*)", parent=self)
        self.mechanism_picker.edit.setPlaceholderText("Path to CHEMKIN .inp/.mech")
        self.add_setup_row("Mechanism", self.mechanism_picker)

        self.thermo_picker = FilePicker(mode="file", must_exist=False, parent=self)
        self.thermo_picker.edit.setPlaceholderText("optional -- separate thermo file")
        self.add_setup_row("Thermo file", self.thermo_picker)

        self.transport_picker = FilePicker(mode="file", must_exist=False, parent=self)
        self.transport_picker.edit.setPlaceholderText("optional -- transport file")
        self.add_setup_row("Transport file", self.transport_picker)

        self.output_dir_picker = FilePicker(mode="directory", must_exist=False, parent=self)
        self.output_dir_picker.edit.setPlaceholderText("default: outputs/converted")
        self.add_setup_row("Output directory", self.output_dir_picker)

        self.make_yaml_check = QCheckBox("Cantera YAML", self)
        self.make_yaml_check.setChecked(True)
        self.add_setup_row("Outputs", self.make_yaml_check)
        self.make_pre_check = QCheckBox("FlameMaster .pre", self)
        self.add_setup_row("", self.make_pre_check)
        self.make_db_check = QCheckBox("Ingest into RMIP DB", self)
        self.add_setup_row("", self.make_db_check)

        self.permissive_check = QCheckBox("--permissive", self)
        self.permissive_check.setChecked(True)
        self.add_advanced_row("ck2yaml flags", self.permissive_check)
        self.no_validate_check = QCheckBox("--no-validate", self)
        self.add_advanced_row("", self.no_validate_check)

        self.fix_exp_check = QCheckBox("Auto-fix Fortran exponents", self)
        self.fix_exp_check.setChecked(True)
        self.fix_exp_check.setToolTip(
            "Repairs whitespace-delimited numeric-only tokens like '4.8749+04' -> "
            "'4.8749E+04'. Never touches species names (KRUNAL_JULY_23.md point 2 fix).")
        self.add_advanced_row("Auto-fixes", self.fix_exp_check)
        self.fix_crlf_check = QCheckBox("Normalize CRLF", self)
        self.fix_crlf_check.setChecked(True)
        self.add_advanced_row("", self.fix_crlf_check)
        self.opensmoke_check = QCheckBox("OpenSmoke: gas-only import", self)
        self.add_advanced_row("", self.opensmoke_check)

        self.scanman_picker = FilePicker(mode="file", must_exist=False, parent=self)
        self.add_advanced_row("ScanMan binary", self.scanman_picker)
        self.db_path_picker = FilePicker(mode="save", must_exist=False, parent=self)
        self.add_advanced_row("RMIP database (.db)", self.db_path_picker)
        self.glossary_picker = FilePicker(mode="file", must_exist=False, parent=self)
        self.add_advanced_row("Species glossary CSV", self.glossary_picker)

        note = QLabel(
            "CHEMKIN input is converted via ck2yaml automatically -- a convert "
            "failure surfaces here with the real ck2yaml error, not just 'failed'.",
            self)
        note.setObjectName("Caption")
        note.setWordWrap(True)
        self.add_setup_row("", note)

    def build_core_call(self):
        if not self.mechanism_picker.is_valid:
            raise ValueError("Choose a CHEMKIN mechanism file first.")
        if not (self.make_yaml_check.isChecked() or self.make_pre_check.isChecked()
                or self.make_db_check.isChecked()):
            raise ValueError("Select at least one output (YAML / .pre / DB).")

        mech_path = self.mechanism_picker.path()
        self._current_label = mech_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        params = ConverterParams(
            mechanism=mech_path,
            thermo=self.thermo_picker.path(),
            transport=self.transport_picker.path(),
            output_dir=self.output_dir_picker.path(),
            make_yaml=self.make_yaml_check.isChecked(),
            make_pre=self.make_pre_check.isChecked(),
            make_db=self.make_db_check.isChecked(),
            scanman=self.scanman_picker.path(),
            db_path=self.db_path_picker.path(),
            glossary=self.glossary_picker.path(),
            permissive=self.permissive_check.isChecked(),
            no_validate=self.no_validate_check.isChecked(),
            fix_exp=self.fix_exp_check.isChecked(),
            fix_crlf=self.fix_crlf_check.isChecked(),
            opensmoke_gas=self.opensmoke_check.isChecked(),
            label=self._current_label,
        )
        return run_convert, (params,), {}

    def populate_results(self, record: RunRecord) -> None:
        lines = [f"Status: {record.status}"]
        if record.error:
            lines.append(f"Error: {record.error}")
        for key, path in record.results.items():
            lines.append(f"{key}: {path}")
        self.result_view.setPlainText("\n".join(lines))
