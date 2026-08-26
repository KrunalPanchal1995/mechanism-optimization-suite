"""
mso_studio.gui.modules.mechanism_check
=========================================
Mechanism Health-Check module page -- PySide6 port of
``mechanism_check_gui.py``. Findings render via
``mso_io.mechanism_check_fix_examples.build_detail_text_from_dict`` -- the
same descriptive-error + worked-fix-example text KRUNAL_JULY_23.md point 3
asked for, driven off the JSON report ``rmip check`` writes rather than the
live (Cantera-dependent) ``Finding`` objects.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QDoubleSpinBox, QFileDialog, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QPlainTextEdit, QPushButton, QSplitter,
    QVBoxLayout, QWidget,
)
from PySide6.QtCore import Qt

from mso_io.mechanism_check_fix_examples import build_detail_text_from_dict
from mso_studio.gui.widgets.results_table import ResultsTable
from mso_studio.core.mechanism_check import MechanismCheckParams, run_check
from mso_studio.core.runrecord import RunRecord
from mso_studio.gui.actions import Action, register
from mso_studio.gui.widgets.module_page import ModulePage

register(Action(
    id="mechanism_check.run", label="Health-check mechanism(s)", shortcut="F5",
    calls="core.mechanism_check.run_check", menu="Simulation",
    tooltip="Run rate/collision/thermo/balance/duplicate/pdep/completeness diagnostics.",
))

_ALL_CHECKS = (
    "load", "rate_sweep", "collision_limit", "thermo_discontinuity",
    "element_balance", "duplicates", "third_body_duplicates", "pdep_sanity",
    "completeness", "unit_sanity", "sim_setup_sanity", "csp_timescales",
    "syntax_lint",
)

_SEV_ORDER = {"error": 0, "warning": 1, "info": 2}


class MechanismCheckPage(ModulePage):
    def __init__(self, parent=None):
        super().__init__(
            "mechanism_check", "Mechanism Health-Check",
            "Diagnose combustion mechanisms — rate/collision/thermo/balance/"
            "duplicate/pdep/completeness/unit checks (src/rmip/analysis/mechanism_check.py).",
            parent,
        )
        self.set_setup_title("Mechanisms + diagnostics")
        self._build_fields()

        self._findings_table = ResultsTable(["Sev", "Check", "Message", "Location"], self)
        self._findings_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._findings_table.itemSelectionChanged.connect(self._on_selection_changed)
        self.add_results_tab("Findings", self._findings_table)

        self._detail_view = QPlainTextEdit(self)
        self._detail_view.setReadOnly(True)
        self._detail_view.setPlainText(
            "Select a finding above to see its full message, the underlying "
            "tool error (if any), and a worked example of how to fix it.")
        self.add_results_tab("Finding details", self._detail_view)

        self._current_findings: list = []   # [(mechanism, finding_dict), ...]

    def _build_fields(self) -> None:
        mech_row = QWidget(self)
        mech_layout = QVBoxLayout(mech_row)
        mech_layout.setContentsMargins(0, 0, 0, 0)
        self._mech_list = QListWidget(mech_row)
        self._mech_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        mech_layout.addWidget(self._mech_list)
        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add files…", mech_row)
        add_btn.clicked.connect(self._on_add_files)
        remove_btn = QPushButton("Remove", mech_row)
        remove_btn.clicked.connect(self._on_remove_selected)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(remove_btn)
        mech_layout.addLayout(btn_row)
        self.add_setup_row("Mechanisms", mech_row)

        note = QLabel(
            "Cantera YAML loads directly; CHEMKIN (.inp/.mech) is converted via "
            "ck2yaml first — a convert failure surfaces as a 'load' finding "
            "below, with the real ck2yaml error in the details tab.", self)
        note.setObjectName("Caption")
        note.setWordWrap(True)
        self.add_setup_row("", note)

        self._check_boxes: dict = {}
        checks_widget = QWidget(self)
        checks_layout = QVBoxLayout(checks_widget)
        checks_layout.setContentsMargins(0, 0, 0, 0)
        for name in _ALL_CHECKS:
            cb = QCheckBox(name, checks_widget)
            cb.setChecked(True)
            self._check_boxes[name] = cb
            checks_layout.addWidget(cb)
        self.add_advanced_row("Diagnostics", checks_widget)

        self._t_grid_edit = QLineEdit("300 800 1400 1800 2200 2500", self)
        self.add_advanced_row("T grid (K)", self._t_grid_edit)
        self._p_grid_edit = QLineEdit("0.1 1 10 50", self)
        self.add_advanced_row("P grid (atm)", self._p_grid_edit)

        self._collision_spin = QDoubleSpinBox(self)
        self._collision_spin.setRange(0.1, 1000.0)
        self._collision_spin.setValue(10.0)
        self.add_advanced_row("Collision exceedance ×", self._collision_spin)

        self._target_t_spin = QDoubleSpinBox(self)
        self._target_t_spin.setRange(300.0, 10000.0)
        self._target_t_spin.setValue(2500.0)
        self.add_advanced_row("Target T high (K)", self._target_t_spin)

    def _on_add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select mechanism file(s)", "",
            "Mechanisms (*.yaml *.yml *.inp *.mech *.ck *.cti);;All files (*)")
        existing = {self._mech_list.item(i).text() for i in range(self._mech_list.count())}
        for p in paths:
            if p not in existing:
                self._mech_list.addItem(p)

    def _on_remove_selected(self) -> None:
        for item in self._mech_list.selectedItems():
            self._mech_list.takeItem(self._mech_list.row(item))

    def _grid(self, text: str):
        raw = text.strip()
        if not raw:
            return []
        try:
            return [float(x) for x in raw.replace(",", " ").split()]
        except ValueError:
            return []

    def build_core_call(self):
        mechanisms = [self._mech_list.item(i).text() for i in range(self._mech_list.count())]
        if not mechanisms:
            raise ValueError("Add at least one mechanism file.")
        checks = [name for name, cb in self._check_boxes.items() if cb.isChecked()]
        if not checks:
            raise ValueError("Enable at least one diagnostic.")

        self._current_label = "+".join(m.rsplit("/", 1)[-1] for m in mechanisms[:3])
        params = MechanismCheckParams(
            mechanisms=mechanisms,
            checks=checks if len(checks) < len(_ALL_CHECKS) else [],
            T_grid=self._grid(self._t_grid_edit.text()),
            P_grid=self._grid(self._p_grid_edit.text()),
            collision_exceedance=self._collision_spin.value(),
            target_T_high=self._target_t_spin.value(),
            label=self._current_label,
        )
        return run_check, (params,), {}

    def populate_results(self, record: RunRecord) -> None:
        reports = record.results.get("reports", [])
        self._current_findings = []
        for report in reports:
            mech = report.get("mechanism", "")
            for finding in report.get("findings", []):
                self._current_findings.append((mech, finding))
        self._current_findings.sort(
            key=lambda mf: _SEV_ORDER.get(mf[1].get("severity"), 9))

        table = self._findings_table
        sev_icon = {"error": "✗", "warning": "▲", "info": "•"}
        rows = [
            [
                sev_icon.get(f.get("severity"), "•"),
                f"{mech.rsplit('/', 1)[-1]}: {f.get('check', '')}",
                f.get("message", ""),
                f.get("location", ""),
            ]
            for mech, f in self._current_findings
        ]
        table.set_rows(rows)
        table.resizeColumnsToContents()
        self._detail_view.setPlainText(
            "Select a finding above to see its full message, the underlying "
            "tool error (if any), and a worked example of how to fix it.")

    def _on_selection_changed(self) -> None:
        rows = self._findings_table.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        if not (0 <= row < len(self._current_findings)):
            return
        mech, finding = self._current_findings[row]
        self._detail_view.setPlainText(build_detail_text_from_dict(mech, finding))
