"""
mso_studio.gui.modules.sensitivity
=====================================
Sensitivity analysis module page -- hosts both backends as tabs within one
page (guidelines doc, Part 6.1's "Results area is tabbed, not windowed"
extended here to the setup side too, since the two techniques are
conceptually one "Sensitivity Analysis" capability with different inputs):

* Brute-force kinetics SA (``target.opt`` driven, 3-parameter Arrhenius).
* Ignition-delay sensitivity (scalar T/P/phi condition, Cantera mechanism).
"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QLineEdit, QPushButton, QSpinBox, QTabWidget,
    QVBoxLayout, QWidget, QFormLayout, QLabel,
)

_SRC = str(Path(__file__).resolve().parents[3])
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mso_studio.core.sensitivity import (
    BruteForceSAParams, IdtSensitivityParams, run_brute_force_sa, run_idt_sensitivity,
)
from mso_studio.core.runrecord import RunRecord
from mso_studio.gui.actions import Action, register
from mso_studio.gui.widgets.results_table import ResultsTable
from mso_studio.gui.widgets.file_picker import FilePicker
from mso_studio.gui.widgets.module_page import ModulePage, CollapsibleSection
from mso_io.plot_style import PlotStyle, EXPORT_FORMATS, LEGEND_LOCATIONS, GRID_AXES
from mso_io.target_add_io import preview_targets

register(Action(
    id="sensitivity.brute_force", label="Run brute-force sensitivity",
    calls="core.sensitivity.run_brute_force_sa", menu="Simulation",
))
register(Action(
    id="sensitivity.idt", label="Run IDT sensitivity",
    calls="core.sensitivity.run_idt_sensitivity", menu="Simulation",
))


class SensitivityPage(ModulePage):
    """Two techniques, one page: a sub-tab selector drives which core
    function :meth:`build_core_call` dispatches to."""

    def __init__(self, parent=None):
        super().__init__(
            "sensitivity", "Sensitivity Analysis",
            "Brute-force Arrhenius (A, n, Ea) kinetics SA, or ignition-delay "
            "sensitivity d ln(tau)/d ln(k) for one condition.",
            parent,
        )
        self.set_setup_title("Technique")
        self.technique_tabs = QTabWidget(self)
        self.add_setup_row("", self.technique_tabs)

        self._build_brute_force_tab()
        self._build_idt_tab()

        from mso_studio.gui.widgets.mpl_canvas import CsvTableView
        self.results_view = CsvTableView(self)
        self.add_results_tab("Results", self.results_view)

    def _build_brute_force_tab(self) -> None:
        tab = QWidget(self.technique_tabs)
        outer = QVBoxLayout(tab)
        form_holder = QWidget(tab)
        form = QFormLayout(form_holder)
        outer.addWidget(form_holder)

        self.bf_target_opt = FilePicker(mode="file", filter_="Opt files (*.opt)", parent=tab)
        form.addRow("target.opt", self.bf_target_opt)

        self.bf_output_dir = FilePicker(mode="directory", must_exist=False, parent=tab)
        self.bf_output_dir.edit.setPlaceholderText("default: alongside target.opt")
        form.addRow("Output directory", self.bf_output_dir)

        # KRUNAL_JULY_23.md point 4 -- preview every .target/.add case and
        # flag which ones the live src/Sensitivity_Analysis/ engine can
        # actually run (solver=zerork / target type=PFR / flux_analysis
        # are MUQ-SAC-only additions -- see src/mso_io/target_add_schema.py).
        preview_btn = QPushButton("Preview targets", tab)
        preview_btn.clicked.connect(self._preview_bf_targets)
        outer.addWidget(preview_btn)
        self.bf_targets_table = ResultsTable(
            ["ID", "Type", "Solver", "T [K]", "P [atm]", "Runnable"], tab)
        self.bf_targets_table.setMaximumHeight(220)
        outer.addWidget(self.bf_targets_table)
        self.bf_gate_label = QLabel("", tab)
        outer.addWidget(self.bf_gate_label)

        self._build_plot_style_section(outer, tab)

        self.technique_tabs.addTab(tab, "Brute-force (target.opt)")

    def _build_plot_style_section(self, outer: QVBoxLayout, tab: QWidget) -> None:
        section = CollapsibleSection("Plot style (sens_3_params.py output plots)", tab)
        self._style_widgets = {}

        def combo(label, key, values, default):
            w = QComboBox(tab)
            w.addItems(values)
            w.setCurrentText(default)
            self._style_widgets[key] = w
            section.add_row(label, w)

        def spin(label, key, default, lo=0, hi=999):
            w = QSpinBox(tab)
            w.setRange(lo, hi)
            w.setValue(default)
            self._style_widgets[key] = w
            section.add_row(label, w)

        def line(label, key, default=""):
            w = QLineEdit(default, tab)
            self._style_widgets[key] = w
            section.add_row(label, w)

        combo("Export format", "export_format", EXPORT_FORMATS, "png")
        spin("DPI", "dpi", 150, lo=50, hi=1200)
        line("Color scheme (colormap)", "color_scheme", "default")
        line("Explicit colors (comma-sep)", "colors", "")
        line("Font family", "font_family", "")
        spin("Title font size", "font_size_title", 12, lo=4, hi=48)
        spin("Axis label font size", "font_size_axis_label", 10, lo=4, hi=48)
        spin("Tick label font size", "font_size_tick_label", 8, lo=4, hi=48)
        spin("Legend font size", "font_size_legend", 9, lo=4, hi=48)
        line("Title", "title", "")
        line("X label", "xlabel", "")
        line("Y label", "ylabel", "")
        combo("Legend location", "legend_loc", LEGEND_LOCATIONS, "best")
        combo("Grid", "grid", GRID_AXES, "none")

        outer.addWidget(section)

    def _collect_plot_style(self) -> PlotStyle:
        w = self._style_widgets
        colors_raw = w["colors"].text().strip()
        colors = [c.strip() for c in colors_raw.split(",") if c.strip()] or None
        return PlotStyle(
            export_format=w["export_format"].currentText(),
            dpi=w["dpi"].value(),
            color_scheme=w["color_scheme"].text() or "default",
            colors=colors,
            font_family=w["font_family"].text(),
            font_size_title=w["font_size_title"].value(),
            font_size_axis_label=w["font_size_axis_label"].value(),
            font_size_tick_label=w["font_size_tick_label"].value(),
            font_size_legend=w["font_size_legend"].value(),
            title=w["title"].text(),
            xlabel=w["xlabel"].text(),
            ylabel=w["ylabel"].text(),
            legend_loc=w["legend_loc"].currentText(),
            grid=w["grid"].currentText(),
        )

    def _preview_bf_targets(self) -> None:
        if not self.bf_target_opt.is_valid:
            self.bf_gate_label.setText("Pick a target.opt first.")
            return
        import yaml
        with open(self.bf_target_opt.path()) as fh:
            cfg = yaml.safe_load(fh) or {}
        locations = cfg.get("Locations", {})
        base = Path(self.bf_target_opt.path()).resolve().parent
        targets_path = locations.get("targets")
        addendum_path = locations.get("addendum")
        if targets_path and not Path(targets_path).is_absolute():
            targets_path = str(base / targets_path)
        if addendum_path and not Path(addendum_path).is_absolute():
            addendum_path = str(base / addendum_path)
        if not targets_path or not Path(targets_path).exists():
            self.bf_gate_label.setText(f"Locations.targets not found: {targets_path}")
            self.bf_targets_table.setRowCount(0)
            return

        rows = preview_targets(targets_path, addendum_path)
        self.bf_targets_table.set_rows([
            [r.dataset_id, r.target_type, r.solver,
             f"{r.temperature_K:g}" if r.temperature_K is not None else "",
             f"{r.pressure_atm:g}" if r.pressure_atm is not None else "",
             "✓" if r.runnable else "⚠"]
            for r in rows
        ])
        n_blocked = sum(1 for r in rows if not r.runnable)
        self.bf_gate_label.setText(
            f"{n_blocked}/{len(rows)} case(s) use a solver/target-type only "
            "src/MUQ-SAC/'s make_input_file.py supports -- not runnable by the live "
            "engine yet." if n_blocked else f"All {len(rows)} case(s) runnable.")

    def _build_idt_tab(self) -> None:
        tab = QWidget(self.technique_tabs)
        form = QFormLayout(tab)

        self.idt_mechanism = FilePicker(mode="file", filter_="Cantera YAML (*.yaml *.yml)", parent=tab)
        form.addRow("Mechanism", self.idt_mechanism)

        self.idt_fuel = QLineEdit(tab)
        self.idt_fuel.setPlaceholderText("e.g. NC7H16")
        form.addRow("Fuel", self.idt_fuel)

        self.idt_oxidizer = QLineEdit("O2:1.0, N2:3.76", tab)
        form.addRow("Oxidizer", self.idt_oxidizer)

        self.idt_T = QDoubleSpinBox(tab); self.idt_T.setRange(200, 4000); self.idt_T.setValue(1200); self.idt_T.setSuffix(" K")
        form.addRow("Initial T", self.idt_T)
        self.idt_P = QDoubleSpinBox(tab); self.idt_P.setRange(0.01, 500); self.idt_P.setValue(13.5); self.idt_P.setSuffix(" atm")
        form.addRow("Pressure", self.idt_P)
        self.idt_phi = QDoubleSpinBox(tab); self.idt_phi.setRange(0.01, 10); self.idt_phi.setValue(1.0); self.idt_phi.setSingleStep(0.1)
        form.addRow("Equivalence ratio", self.idt_phi)

        self.idt_reactor = QComboBox(tab); self.idt_reactor.addItems(["constV", "constP"])
        form.addRow("Reactor", self.idt_reactor)
        self.idt_method = QComboBox(tab); self.idt_method.addItems(["brute", "cvodes"])
        form.addRow("Method", self.idt_method)
        self.idt_top_n = QSpinBox(tab); self.idt_top_n.setRange(1, 500); self.idt_top_n.setValue(20)
        form.addRow("Top-N reactions", self.idt_top_n)

        self.technique_tabs.addTab(tab, "Ignition-delay sensitivity")

    def build_core_call(self):
        if self.technique_tabs.currentIndex() == 0:
            if not self.bf_target_opt.is_valid:
                raise ValueError("Pick a target.opt file first.")
            self._current_label = "brute_force_sa"
            output_dir = self.bf_output_dir.path() or str(
                Path(self.bf_target_opt.path()).resolve().parent)
            style_dir = Path(output_dir)
            style_dir.mkdir(parents=True, exist_ok=True)
            style_path = style_dir / "plot_style.json"
            self._collect_plot_style().save(str(style_path))
            params = BruteForceSAParams(
                target_opt_path=self.bf_target_opt.path(),
                output_dir=output_dir,
                plot_style_json=str(style_path),
            )
            self._current_label = params.label
            return run_brute_force_sa, (params,), {}
        else:
            if not self.idt_mechanism.is_valid:
                raise ValueError("Pick a Cantera mechanism file first.")
            if not self.idt_fuel.text().strip():
                raise ValueError("Enter a fuel species.")
            self._current_label = "idt_sensitivity"
            params = IdtSensitivityParams(
                mechanism=self.idt_mechanism.path(),
                fuel=self.idt_fuel.text().strip(),
                T_K=self.idt_T.value(), P_atm=self.idt_P.value(), phi=self.idt_phi.value(),
                oxidizer=self.idt_oxidizer.text().strip() or "O2:1.0, N2:3.76",
                reactor=self.idt_reactor.currentText(), method=self.idt_method.currentText(),
                top_n=self.idt_top_n.value(),
            )
            self._current_label = params.label
            return run_idt_sensitivity, (params,), {}

    def populate_results(self, record: RunRecord) -> None:
        if "idt_sensitivity_json" in record.tables:
            # JSON, not CSV -- CsvTableView only understands CSV, so just
            # surface the path via the log for now (WS-D adds a proper
            # sortable-table renderer for the top-N reaction JSON).
            self.log_view.append_status(f"IDT sensitivity JSON: {record.tables['idt_sensitivity_json']}")
        elif record.results.get("sens_parameters_path"):
            self.log_view.append_status(f"Sensitivity pickle: {record.results['sens_parameters_path']}")
