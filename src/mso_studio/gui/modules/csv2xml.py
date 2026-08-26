"""
mso_studio.gui.modules.csv2xml
=================================
CSV -> ReSpecTh XML converter module page — "F1: CSV → ReSpecTh/RKD XML
converter" (WORKBENCH_AUDIT_2026-08-10.md, Part F) and the actual fix for
that audit's defect B11 (the plot digitizer's own minimal XML writer emits
a schema ``config_tools/converters_io/XML2CSV/xml2csv.py`` rejects
outright). Takes ANY small CSV — including the digitizer's own
``export_csv()`` output — plus a metadata form describing the experiment,
and produces a real ReSpecTh XML, then round-trip-checks it through the
actual ``xml2csv.py`` right here so the user sees "N target rows" (or the
real reason for 0) before leaving the page.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
)

from mso_studio.core.csv2xml import Csv2XmlParams, run_csv2xml
from mso_studio.core.runrecord import RunRecord
from mso_studio.gui.actions import Action, register
from mso_studio.gui.widgets.module_page import ModulePage

register(Action(
    id="csv2xml.run", label="Convert CSV to ReSpecTh XML", shortcut="F5",
    calls="core.csv2xml.run_csv2xml", menu="Convert",
    tooltip="Turn a digitized/tabulated CSV into a ReSpecTh XML that "
            "xml2csv.py can turn into a .target/.add pair.",
))


def _parse_fuel_field(text: str) -> dict:
    """'NH3:1.0, H2:0.5' -> {'NH3': 1.0, 'H2': 0.5}; a bare 'NH3' -> {'NH3': 1.0}."""
    fuel = {}
    for entry in text.split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, _, amount = entry.partition(":")
        fuel[name.strip()] = float(amount) if amount.strip() else 1.0
    return fuel


class Csv2XmlPage(ModulePage):
    def __init__(self, parent=None):
        super().__init__(
            "csv2xml", "CSV → ReSpecTh XML",
            "Convert a digitized/tabulated CSV into a ReSpecTh-schema XML "
            "that config_tools/converters_io/XML2CSV/xml2csv.py can turn "
            "into a .target/.add pair — closes the gap where the plot "
            "digitizer's own XML export was too minimal for xml2csv.py to accept.",
            parent,
        )
        self.set_setup_title("Input CSV + experiment metadata")
        self._build_fields()

        self._xml_preview = QPlainTextEdit(self)
        self._xml_preview.setReadOnly(True)
        self.add_results_tab("Produced XML", self._xml_preview)

        self._verify_view = QPlainTextEdit(self)
        self._verify_view.setReadOnly(True)
        self._verify_view.setPlainText(
            "Run a conversion to see whether xml2csv.py actually accepts "
            "the produced XML, and the .target row(s) it yields.")
        self.add_results_tab("Round-trip check", self._verify_view)

    def _build_fields(self) -> None:
        csv_row = QWidget(self)
        csv_layout = QHBoxLayout(csv_row)
        csv_layout.setContentsMargins(0, 0, 0, 0)
        self._csv_edit = QLineEdit(csv_row)
        self._csv_edit.setPlaceholderText("Path to a digitized/tabulated CSV (x, y[, y_plus, y_minus])")
        browse_btn = QPushButton("Browse…", csv_row)
        browse_btn.clicked.connect(self._on_browse_csv)
        csv_layout.addWidget(self._csv_edit)
        csv_layout.addWidget(browse_btn)
        self.add_setup_row("CSV file", csv_row)

        self._target_type_box = QComboBox(self)
        self._target_type_box.addItems(["Tig", "Fls", "Flf_a"])
        self.add_setup_row("Target type", self._target_type_box)

        self._x_quantity_box = QComboBox(self)
        self._x_quantity_box.addItems(["temperature", "phi"])
        self.add_setup_row("X-axis quantity", self._x_quantity_box)

        self._x_units_edit = QLineEdit("K", self)
        self.add_setup_row("X units", self._x_units_edit)
        self._y_units_edit = QLineEdit("ms", self)
        self.add_setup_row("Y units", self._y_units_edit)

        self._fuel_edit = QLineEdit("NH3:1.0", self)
        self._fuel_edit.setPlaceholderText("species:relative_amount, comma-separated, e.g. NH3:1.0")
        self.add_setup_row("Fuel composition", self._fuel_edit)

        self._pressure_spin = QDoubleSpinBox(self)
        self._pressure_spin.setRange(0.001, 1000.0)
        self._pressure_spin.setValue(1.0)
        self.add_setup_row("Pressure (atm)", self._pressure_spin)

        self._phi_spin = QDoubleSpinBox(self)
        self._phi_spin.setRange(0.01, 10.0)
        self._phi_spin.setValue(1.0)
        self.add_setup_row("Fixed φ (used if X = temperature)", self._phi_spin)

        self._temp_spin = QDoubleSpinBox(self)
        self._temp_spin.setRange(200.0, 5000.0)
        self._temp_spin.setValue(298.0)
        self.add_setup_row("Fixed T (K) (used if X = φ)", self._temp_spin)

        self._bath_ratio_spin = QDoubleSpinBox(self)
        self._bath_ratio_spin.setRange(0.0, 100.0)
        self._bath_ratio_spin.setValue(3.76)
        self.add_advanced_row("Bath gas : O2 ratio", self._bath_ratio_spin)

        self._bath_species_edit = QLineEdit("N2", self)
        self.add_advanced_row("Bath gas species", self._bath_species_edit)

        self._dataset_id_edit = QLineEdit(self)
        self._dataset_id_edit.setPlaceholderText("optional, default: CSV filename")
        self.add_advanced_row("Dataset id", self._dataset_id_edit)

        note = QLabel(
            "xml2csv.py only recognizes a fuel as fuel_species=[\"NH3\"] "
            "by default when run via its bare CLI/GUI import path; the "
            "round-trip check here always passes your own Fuel composition "
            "species through, so any fuel you name above is checked "
            "correctly regardless of that CLI default.", self)
        note.setObjectName("Caption")
        note.setWordWrap(True)
        self.add_advanced_row("", note)

    def _on_browse_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV files (*.csv);;All files (*)")
        if path:
            self._csv_edit.setText(path)

    def build_core_call(self):
        csv_path = self._csv_edit.text().strip()
        if not csv_path:
            raise ValueError("Select a CSV file.")
        fuel = _parse_fuel_field(self._fuel_edit.text())
        if not fuel:
            raise ValueError("Enter at least one fuel species, e.g. NH3:1.0.")

        dataset_id = self._dataset_id_edit.text().strip()
        import os
        base = dataset_id or os.path.splitext(os.path.basename(csv_path))[0]
        output_path = os.path.join(
            os.path.dirname(csv_path) or ".", f"{base}_respecth.xml")

        self._current_label = base
        params = Csv2XmlParams(
            csv_path=csv_path,
            output_path=output_path,
            target_type=self._target_type_box.currentText(),
            x_quantity=self._x_quantity_box.currentText(),
            x_units=self._x_units_edit.text().strip() or "K",
            y_units=self._y_units_edit.text().strip(),
            pressure_atm=self._pressure_spin.value(),
            phi=self._phi_spin.value(),
            temperature_K=self._temp_spin.value(),
            fuel=fuel,
            bath_gas_ratio=self._bath_ratio_spin.value(),
            bath_gas_species=self._bath_species_edit.text().strip() or "N2",
            dataset_id=dataset_id,
        )
        return run_csv2xml, (params,), {"fuel_list": list(fuel)}

    def populate_results(self, record: RunRecord) -> None:
        self._xml_preview.setPlainText(record.results.get("xml_preview", "") or "(no XML produced)")

        verify_ok = record.results.get("verify_ok")
        n_rows = record.results.get("verify_n_rows", 0)
        preview = record.results.get("verify_target_preview", "")
        lines = []
        if verify_ok:
            lines.append(f"✓ xml2csv.py accepted this file — {n_rows} target row(s) produced.")
        else:
            lines.append("✗ xml2csv.py did NOT accept this file — 0 target rows.")
        if record.warnings:
            lines.append("")
            lines.append("Warnings:")
            lines.extend(f"  {w}" for w in record.warnings)
        if preview:
            lines.append("")
            lines.append(".target preview:")
            lines.append(preview)
        self._verify_view.setPlainText("\n".join(lines))
