"""
mso_studio.gui.modules.tools
===============================
Tools module page: curve/symbolic fitting, the plot digitizer, and Gemini
PDF Q&A / table extraction -- ported from the Tkinter ``src/notebook_gemini/``
panels (WS-D). Four sub-tabs, each its own widget class below; none are
``ModulePage`` subclasses since these are interactive tools, not
run-and-report backend jobs.
"""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget, QMessageBox,
    QPlainTextEdit, QPushButton, QRadioButton, QSpinBox, QSplitter, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from mso_studio.core import digitizer as digitizer_core
from mso_studio.core import rmg as rmg_core
from mso_studio.core import tools as tools_core
from mso_studio.gui.widgets.digitizer_view import DigitizerView
from mso_studio.gui.widgets.file_picker import FilePicker
from mso_studio.gui.widgets.log_view import LogView
from mso_studio.gui.widgets.module_hero import make_module_hero
from mso_studio.gui.widgets.mpl_canvas import MplCanvas
from mso_studio.gui.widgets.results_table import ResultsTable
from mso_studio.gui.workers import call_in_thread
from mso_io.notebook_gen import NotebookRequest, generate_mechanism_notebook


class ToolsPage(QWidget):
    statusMessage = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        self.hero = make_module_hero(
            "tools", "Tools",
            "Curve/symbolic fitting, the plot digitizer, and Gemini-backed PDF "
            "Q&A / table extraction -- ported from the Tkinter notebook_gemini "
            "panels, previously reachable only from the old monolith.",
            self,
        )
        root.addWidget(self.hero)

        tabs = QTabWidget(self)
        root.addWidget(tabs, 1)
        self.curve_fit_tab = CurveFitTab(self)
        self.digitizer_tab = DigitizerTab(self)
        self.pdf_qa_tab = PdfQaTab(self)
        self.pdf_tables_tab = PdfTablesTab(self)
        self.notebook_gen_tab = NotebookGenTab(self)
        self.rdkit_tab = RdkitTab(self)
        self.rmg_tab = RmgTab(self)
        self.native_extensions_tab = NativeExtensionsTab(self)
        for tab, label in (
            (self.curve_fit_tab, "Curve / Symbolic Fit"),
            (self.digitizer_tab, "Plot Digitizer"),
            (self.pdf_qa_tab, "PDF Q&&A"),
            (self.pdf_tables_tab, "PDF Tables"),
            (self.notebook_gen_tab, "Jupyter Notebook"),
            (self.rdkit_tab, "RDKit / SMILES"),
            (self.rmg_tab, "RMG (mechanism generation)"),
            (self.native_extensions_tab, "Native Extensions (.so)"),
        ):
            tabs.addTab(tab, label)
            if hasattr(tab, "statusMessage"):
                tab.statusMessage.connect(self.statusMessage.emit)


# ===========================================================================
# Curve / symbolic fit
# ===========================================================================

class CurveFitTab(QWidget):
    statusMessage = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._x_data: Optional[np.ndarray] = None
        self._y_data: Optional[np.ndarray] = None
        self._headers: List[str] = []
        self._numeric_columns: dict[str, np.ndarray] = {}
        self._threads = []

        layout = QHBoxLayout(self)
        left = QWidget(self)
        form = QFormLayout(left)
        form.setContentsMargins(16, 16, 16, 16)

        load_row = QHBoxLayout()
        self.load_button = QPushButton("Load CSV…", left)
        self.load_button.clicked.connect(self._on_load_csv)
        load_row.addWidget(self.load_button)
        form.addRow(load_row)

        self.x_combo = QComboBox(left)
        form.addRow("X column", self.x_combo)
        self.y_combo = QComboBox(left)
        form.addRow("Y column", self.y_combo)

        self.degree_spin = QSpinBox(left)
        self.degree_spin.setRange(1, 10)
        self.degree_spin.setValue(3)
        form.addRow("Poly degree", self.degree_spin)
        self.poly_fit_button = QPushButton("Poly fit", left)
        self.poly_fit_button.clicked.connect(self._on_poly_fit)
        form.addRow(self.poly_fit_button)

        self.pysr_iter_spin = QSpinBox(left)
        self.pysr_iter_spin.setRange(5, 500)
        self.pysr_iter_spin.setValue(40)
        form.addRow("PySR iterations", self.pysr_iter_spin)
        self.pysr_fit_button = QPushButton("PySR fit", left)
        self.pysr_fit_button.setToolTip(
            "Symbolic regression via PySR/Julia, run in the anaconda environment. "
            "May take a while; the window stays responsive."
        )
        self.pysr_fit_button.clicked.connect(self._on_pysr_fit)
        form.addRow(self.pysr_fit_button)

        self.export_button = QPushButton("Export equations…", left)
        self.export_button.clicked.connect(self._on_export)
        form.addRow(self.export_button)

        layout.addWidget(left, 0)

        right = QSplitter(self)
        self.canvas = MplCanvas(right)
        self._ax = self.canvas.figure.add_subplot(111)
        right.addWidget(self.canvas)
        self.equations_view = QPlainTextEdit(right)
        self.equations_view.setReadOnly(True)
        right.addWidget(self.equations_view)
        right.setSizes([500, 250])
        layout.addWidget(right, 1)

    def _on_load_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load data CSV", "", "CSV files (*.csv)")
        if not path:
            return
        try:
            with open(path, newline="") as fh:
                reader = csv.reader(fh)
                rows = list(reader)
            if not rows:
                raise ValueError("CSV is empty.")
            header = rows[0]
            body = rows[1:]
            columns: dict[str, list] = {h: [] for h in header}
            for row in body:
                for h, cell in zip(header, row):
                    try:
                        columns[h].append(float(cell))
                    except ValueError:
                        columns[h].append(float("nan"))
            numeric = {h: np.array(v) for h, v in columns.items() if np.isfinite(np.array(v)).any()}
            if not numeric:
                QMessageBox.warning(self, "No numeric columns", "CSV has no numeric columns to fit.")
                return
            self._numeric_columns = numeric
            self._headers = list(numeric.keys())
            self.x_combo.clear()
            self.y_combo.clear()
            self.x_combo.addItems(self._headers)
            self.y_combo.addItems(self._headers)
            if len(self._headers) > 1:
                self.y_combo.setCurrentIndex(1)
            self._log(f"Loaded: {Path(path).name} — {len(body)} rows, "
                       f"{len(self._headers)} numeric columns")
            self._replot()
        except Exception as exc:  # noqa: BLE001 - shown to the user, not raised
            QMessageBox.critical(self, "Load error", str(exc))

    def _get_xy(self):
        if not self._numeric_columns:
            QMessageBox.information(self, "No data", "Load a CSV first.")
            return None, None
        x_col = self.x_combo.currentText()
        y_col = self.y_combo.currentText()
        if x_col not in self._numeric_columns or y_col not in self._numeric_columns:
            QMessageBox.warning(self, "Column not found", "Select valid X and Y columns.")
            return None, None
        x = self._numeric_columns[x_col]
        y = self._numeric_columns[y_col]
        mask = np.isfinite(x) & np.isfinite(y)
        return x[mask], y[mask]

    def _on_poly_fit(self) -> None:
        x, y = self._get_xy()
        if x is None:
            return
        result = tools_core.poly_fit(x, y, self.degree_spin.value())
        self._log(
            f"\n--- Polynomial fit (degree {result.degree}) ---\n"
            f"RMSE = {result.rmse:.4g}\n"
            f"LaTeX: $y = {result.latex_equation}$"
        )
        self._replot(x_fit=result.x_fine, y_fit=result.y_fine, fit_label=f"Poly deg={result.degree}")

    def _on_pysr_fit(self) -> None:
        x, y = self._get_xy()
        if x is None:
            return
        self.pysr_fit_button.setEnabled(False)
        self.statusMessage.emit("Fitting with PySR (this can take a while)…")
        self._log("\n--- PySR symbolic regression starting ---")
        thread, worker = call_in_thread(tools_core.pysr_fit, x, y, self.pysr_iter_spin.value())
        worker.finished.connect(self._on_pysr_finished)
        worker.error.connect(self._on_pysr_error)
        self._threads.append((thread, worker))

    def _on_pysr_finished(self, result: tools_core.PySRFitResult) -> None:
        self.pysr_fit_button.setEnabled(True)
        lines = ["--- PySR results ---"]
        for eq in result.equations:
            lines.append(f"complexity={eq.complexity:>3}  loss={eq.loss:.4g}  {eq.equation}")
        lines.append(f"\nBest: {result.best_equation}")
        self._log("\n".join(lines))
        self._replot(x_fit=result.x_fine, y_fit=result.y_fine, fit_label="PySR best")
        self.statusMessage.emit("PySR fit complete")

    def _on_pysr_error(self, message: str) -> None:
        self.pysr_fit_button.setEnabled(True)
        self._log(f"\nPySR fit failed: {message}\n(needs pysr + Julia in the anaconda environment)")
        self.statusMessage.emit("PySR fit failed — see equations panel")

    def _on_export(self) -> None:
        content = self.equations_view.toPlainText().strip()
        if not content:
            QMessageBox.information(self, "Nothing to export", "Run a fit first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export equations", "", "LaTeX (*.tex);;Text (*.txt)")
        if not path:
            return
        Path(path).write_text(content)
        self.statusMessage.emit(f"Equations written to {path}")

    def _log(self, text: str) -> None:
        self.equations_view.appendPlainText(text)

    def _replot(self, x_fit=None, y_fit=None, fit_label: str = "Fit") -> None:
        x, y = self._get_xy()
        if x is None:
            return
        self._ax.clear()
        self._ax.scatter(x, y, s=20, label="Data", zorder=3)
        if x_fit is not None and y_fit is not None:
            self._ax.plot(x_fit, y_fit, lw=1.5, label=fit_label, color="tab:red")
        self._ax.set_xlabel(self.x_combo.currentText())
        self._ax.set_ylabel(self.y_combo.currentText())
        self._ax.legend()
        self.canvas.figure.tight_layout()
        self.canvas.draw_idle()


# ===========================================================================
# Digitizer
# ===========================================================================

class DigitizerTab(QWidget):
    statusMessage = Signal(str)

    _MODE_LABELS = {
        "x_cal": "Calibrate X axis — click 2 reference points",
        "y_cal": "Calibrate Y axis — click 2 reference points",
        "data": "Pick data points — click on the curve",
        "err_bar": "Pick error bars — click 1=upper, 2=lower of bar (applies to the last data point)",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._x_cal: List[digitizer_core.CalibPoint] = []
        self._y_cal: List[digitizer_core.CalibPoint] = []
        self._data_pts: List[digitizer_core.DataPoint] = []
        self._err_upper: Optional[float] = None
        self._threads = []

        layout = QHBoxLayout(self)
        left = QWidget(self)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(16, 16, 16, 16)

        open_row = QHBoxLayout()
        self.open_image_button = QPushButton("Open image…", left)
        self.open_image_button.clicked.connect(self._on_open_image)
        open_row.addWidget(self.open_image_button)
        self.open_pdf_button = QPushButton("Open PDF page…", left)
        self.open_pdf_button.clicked.connect(self._on_open_pdf_page)
        open_row.addWidget(self.open_pdf_button)
        left_layout.addLayout(open_row)

        self.mode_group = QButtonGroup(left)
        for mode, label in self._MODE_LABELS.items():
            radio = QRadioButton(mode.replace("_", " ").title(), left)
            radio.setProperty("mode", mode)
            if mode == "x_cal":
                radio.setChecked(True)
            radio.toggled.connect(self._on_mode_changed)
            self.mode_group.addButton(radio)
            left_layout.addWidget(radio)

        log_row = QHBoxLayout()
        self.log_x_check = QCheckBox("log X", left)
        self.log_y_check = QCheckBox("log Y", left)
        log_row.addWidget(self.log_x_check)
        log_row.addWidget(self.log_y_check)
        left_layout.addLayout(log_row)

        edit_row = QHBoxLayout()
        self.undo_button = QPushButton("Undo last", left)
        self.undo_button.clicked.connect(self._on_undo)
        edit_row.addWidget(self.undo_button)
        self.clear_button = QPushButton("Clear all", left)
        self.clear_button.clicked.connect(self._on_clear)
        edit_row.addWidget(self.clear_button)
        left_layout.addLayout(edit_row)

        self.export_csv_button = QPushButton("Export CSV…", left)
        self.export_csv_button.clicked.connect(self._on_export_csv)
        left_layout.addWidget(self.export_csv_button)
        self.export_xml_button = QPushButton("Export RKD XML…", left)
        self.export_xml_button.clicked.connect(self._on_export_xml)
        left_layout.addWidget(self.export_xml_button)
        self.export_yaml_button = QPushButton("Export YAML…", left)
        self.export_yaml_button.clicked.connect(self._on_export_yaml)
        left_layout.addWidget(self.export_yaml_button)

        self.gemini_axes_button = QPushButton("Gemini: read axes", left)
        self.gemini_axes_button.setToolTip("Ask Gemini to identify axis labels/units/scale from the image.")
        self.gemini_axes_button.clicked.connect(self._on_gemini_read_axes)
        left_layout.addWidget(self.gemini_axes_button)

        self.status_label = QLabel(self._MODE_LABELS["x_cal"], left)
        self.status_label.setObjectName("Caption")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)

        self.table_label = QLabel("(no points yet)", left)
        self.table_label.setObjectName("MonoSmall")
        left_layout.addWidget(self.table_label)
        left_layout.addStretch(1)

        layout.addWidget(left, 0)

        self.view = DigitizerView(self)
        self.view.pointClicked.connect(self._on_point_clicked)
        layout.addWidget(self.view, 1)

        self._image_path: Optional[str] = None

    def _current_mode(self) -> str:
        checked = self.mode_group.checkedButton()
        return checked.property("mode") if checked else "x_cal"

    def _on_mode_changed(self) -> None:
        self.status_label.setText(self._MODE_LABELS[self._current_mode()])

    def _on_open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if path:
            self._load_image(path)

    def _on_open_pdf_page(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open PDF", "", "PDF (*.pdf)")
        if not path:
            return
        self.statusMessage.emit("Reading PDF page count…")
        thread, worker = call_in_thread(tools_core.pdf_page_count, path)
        worker.finished.connect(lambda n, p=path: self._prompt_and_render_page(p, n))
        worker.error.connect(lambda msg: QMessageBox.critical(self, "PDF error", msg))
        self._threads.append((thread, worker))

    def _prompt_and_render_page(self, pdf_path: str, n_pages: int) -> None:
        page_no, ok = QInputDialog.getInt(
            self, "Page", f"PDF has {n_pages} page(s). Which page?", 1, 1, n_pages,
        )
        if not ok:
            return
        out_png = str(Path(tempfile.mkdtemp()) / f"page_{page_no}.png")
        thread, worker = call_in_thread(tools_core.pdf_render_page, pdf_path, page_no, out_png)
        worker.finished.connect(self._load_image)
        worker.error.connect(lambda msg: QMessageBox.critical(self, "PDF render error", msg))
        self._threads.append((thread, worker))

    def _load_image(self, path: str) -> None:
        self._image_path = path
        self.view.load_image(path)
        self._clear_points()
        self.statusMessage.emit(f"Image loaded: {Path(path).name}")

    def _on_point_clicked(self, px: float, py: float) -> None:
        if self._image_path is None:
            QMessageBox.information(self, "No image", "Load an image first.")
            return
        mode = self._current_mode()

        if mode in ("x_cal", "y_cal"):
            value, ok = QInputDialog.getDouble(
                self, f"{'X' if mode == 'x_cal' else 'Y'} calibration",
                f"Enter the data value at this point (px={px:.0f}, py={py:.0f}):",
                decimals=6,
            )
            if not ok:
                return
            cal_list = self._x_cal if mode == "x_cal" else self._y_cal
            cal_list.append(digitizer_core.CalibPoint(px, py, value))
            self.view.draw_marker(px, py, mode, f"{'X' if mode == 'x_cal' else 'Y'}={value:g}")
            if len(cal_list) == 2:
                self.status_label.setText(f"{'X' if mode=='x_cal' else 'Y'} axis calibrated. "
                                           + self._MODE_LABELS[mode])

        elif mode == "data":
            if len(self._x_cal) < 2 or len(self._y_cal) < 2:
                QMessageBox.warning(self, "Not calibrated", "Calibrate both X and Y axes first (2 points each).")
                return
            x, y = digitizer_core.pixel_to_data(px, py, tuple(self._x_cal[:2]), tuple(self._y_cal[:2]))
            if self.log_x_check.isChecked() and x > 0:
                x = 10 ** x
            if self.log_y_check.isChecked() and y > 0:
                y = 10 ** y
            self._data_pts.append(digitizer_core.DataPoint(px, py, x, y))
            self.view.draw_marker(px, py, mode, f"({x:.4g},{y:.4g})")
            self._refresh_table()

        elif mode == "err_bar":
            if len(self._x_cal) < 2 or len(self._y_cal) < 2:
                QMessageBox.warning(self, "Not calibrated", "Calibrate both X and Y axes first.")
                return
            _, y_raw = digitizer_core.pixel_to_data(px, py, tuple(self._x_cal[:2]), tuple(self._y_cal[:2]))
            y_val = 10 ** y_raw if (self.log_y_check.isChecked() and y_raw > 0) else y_raw
            if self._err_upper is None:
                self._err_upper = y_val
                self.view.draw_marker(px, py, mode, f"↑{y_val:.4g}")
                self.status_label.setText("Error bar: upper set — now click the LOWER bound")
            else:
                y_upper, y_lower = self._err_upper, y_val
                self._err_upper = None
                if not self._data_pts:
                    QMessageBox.warning(self, "No data point",
                                         "Add a data point in Data mode first, then assign its error bar.")
                    return
                dp = self._data_pts[-1]
                dp.dy_plus = abs(y_upper - dp.y)
                dp.dy_minus = abs(dp.y - y_lower)
                self.view.draw_marker(px, py, mode, f"↓{y_lower:.4g}")
                self.status_label.setText(f"Error bar set: +{dp.dy_plus:.4g} / -{dp.dy_minus:.4g}")
                self._refresh_table()

    def _refresh_table(self) -> None:
        if not self._data_pts:
            self.table_label.setText("(no points yet)")
            return
        lines = [f"{'x':>12}  {'y':>12}"]
        for dp in self._data_pts[-8:]:
            lines.append(f"{dp.x:>12.6g}  {dp.y:>12.6g}")
        if len(self._data_pts) > 8:
            lines.append(f"… ({len(self._data_pts)} total)")
        self.table_label.setText("\n".join(lines))

    def _on_undo(self) -> None:
        if self._data_pts:
            self._data_pts.pop()
        elif self._y_cal:
            self._y_cal.pop()
        elif self._x_cal:
            self._x_cal.pop()
        else:
            return
        self.view.pop_last_marker()
        self._refresh_table()

    def _on_clear(self) -> None:
        self._clear_points()

    def _clear_points(self) -> None:
        self._x_cal.clear()
        self._y_cal.clear()
        self._data_pts.clear()
        self._err_upper = None
        self.view.clear_markers()
        self._refresh_table()

    def _on_export_csv(self) -> None:
        if not self._data_pts:
            QMessageBox.information(self, "No data", "No data points to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save digitized data", "", "CSV (*.csv)")
        if not path:
            return
        x_label, ok1 = QInputDialog.getText(self, "X label", "X axis label:", QLineEdit.EchoMode.Normal, "T_K")
        y_label, ok2 = QInputDialog.getText(self, "Y label", "Y axis label:", QLineEdit.EchoMode.Normal, "tau_ms")
        digitizer_core.export_csv(self._data_pts, path, x_label or "x", y_label or "y")
        self.statusMessage.emit(f"Saved {len(self._data_pts)} points to {path}")

    def _on_export_xml(self) -> None:
        if not self._data_pts:
            QMessageBox.information(self, "No data", "No data points to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export RKD XML", "", "XML (*.xml)")
        if not path:
            return
        digitizer_core.export_rkd_xml(self._data_pts, path)
        self.statusMessage.emit(f"RKD XML saved to {path}")

    def _on_export_yaml(self) -> None:
        if not self._data_pts:
            QMessageBox.information(self, "No data", "No data points to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export YAML", "", "YAML (*.yaml *.yml)")
        if not path:
            return
        digitizer_core.export_yaml(self._data_pts, path)
        self.statusMessage.emit(f"YAML saved to {path}")

    def _on_gemini_read_axes(self) -> None:
        if self._image_path is None:
            QMessageBox.information(self, "No image", "Load an image first.")
            return
        if not tools_core.gemini_available():
            QMessageBox.information(self, "Gemini not available", "Set GOOGLE_API_KEY to use this feature.")
            return
        self.statusMessage.emit("Asking Gemini to read axes…")
        thread, worker = call_in_thread(
            tools_core.gemini_chat_with_image, self._image_path,
            "This is a combustion data plot. Identify:\n"
            "1. X axis: label, units, min/max, scale (linear/log)\n"
            "2. Y axis: label, units, min/max, scale (linear/log)\n"
            "3. Number of data series and legend labels (if any)\nBe concise.",
        )
        worker.finished.connect(lambda reply: QMessageBox.information(self, "Gemini axis reading", reply))
        worker.error.connect(lambda msg: QMessageBox.critical(self, "Gemini error", msg))
        self._threads.append((thread, worker))


# ===========================================================================
# PDF Q&A
# ===========================================================================

class PdfQaTab(QWidget):
    statusMessage = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pdf_path: Optional[str] = None
        self._threads = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)

        top = QHBoxLayout()
        self.path_label = QLabel("No PDF loaded", self)
        top.addWidget(self.path_label, 1)
        self.open_button = QPushButton("Open PDF…", self)
        self.open_button.clicked.connect(self._on_open)
        top.addWidget(self.open_button)
        self.summarise_button = QPushButton("Summarise", self)
        self.summarise_button.clicked.connect(self._on_summarise)
        top.addWidget(self.summarise_button)
        gemini_status = QLabel("Gemini: ready" if tools_core.gemini_available() else "Gemini: no API key", self)
        gemini_status.setObjectName("Caption")
        top.addWidget(gemini_status)
        layout.addLayout(top)

        self.chat_view = QPlainTextEdit(self)
        self.chat_view.setReadOnly(True)
        layout.addWidget(self.chat_view, 1)

        input_row = QHBoxLayout()
        self.question_edit = QLineEdit(self)
        self.question_edit.returnPressed.connect(self._on_ask)
        input_row.addWidget(self.question_edit, 1)
        self.ask_button = QPushButton("Ask", self)
        self.ask_button.clicked.connect(self._on_ask)
        input_row.addWidget(self.ask_button)
        layout.addLayout(input_row)

    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select PDF", "", "PDF files (*.pdf)")
        if not path:
            return
        self._pdf_path = path
        self.path_label.setText(Path(path).name)
        self._append_chat("system", "Loaded: " + Path(path).name)

    def _on_summarise(self) -> None:
        if not self._pdf_path:
            QMessageBox.information(self, "No PDF", "Load a PDF first.")
            return
        self._append_chat("you", "[Summarise this paper]")
        self._run(tools_core.gemini_summarise_pdf, self._pdf_path)

    def _on_ask(self) -> None:
        question = self.question_edit.text().strip()
        if not question:
            return
        self.question_edit.clear()
        self._append_chat("you", question)
        if not self._pdf_path:
            QMessageBox.information(self, "No PDF", "Load a PDF first.")
            return
        self._run(tools_core.gemini_chat_with_pdf, self._pdf_path, question)

    def _run(self, fn, *args) -> None:
        self.statusMessage.emit("Asking Gemini…")
        thread, worker = call_in_thread(fn, *args)
        worker.finished.connect(lambda reply: self._append_chat("gemini", reply))
        worker.error.connect(lambda msg: self._append_chat("error", msg))
        self._threads.append((thread, worker))

    def _append_chat(self, role: str, text: str) -> None:
        self.chat_view.appendPlainText(f"\n[{role}] {text}")
        self.statusMessage.emit("Ready")


# ===========================================================================
# PDF Tables
# ===========================================================================

class PdfTablesTab(QWidget):
    statusMessage = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pdf_path: Optional[str] = None
        self._tables: List[List[List[str]]] = []
        self._selected = -1
        self._threads = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)

        top = QHBoxLayout()
        self.path_label = QLabel("No PDF loaded", self)
        top.addWidget(self.path_label, 1)
        self.open_button = QPushButton("Open PDF…", self)
        self.open_button.clicked.connect(self._on_open)
        top.addWidget(self.open_button)
        self.extract_button = QPushButton("Extract tables", self)
        self.extract_button.clicked.connect(self._on_extract)
        top.addWidget(self.extract_button)
        self.export_button = QPushButton("Export CSV…", self)
        self.export_button.clicked.connect(self._on_export)
        top.addWidget(self.export_button)
        self.clean_button = QPushButton("Gemini: clean table", self)
        self.clean_button.clicked.connect(self._on_clean)
        top.addWidget(self.clean_button)
        layout.addLayout(top)

        splitter = QSplitter(self)
        self.table_list = QListWidget(splitter)
        self.table_list.currentRowChanged.connect(self._on_select)
        splitter.addWidget(self.table_list)
        self.preview = QTableWidget(splitter)
        self.preview.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        splitter.addWidget(self.preview)
        splitter.setSizes([200, 700])
        layout.addWidget(splitter, 1)

    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open PDF for table extraction", "", "PDF (*.pdf)")
        if not path:
            return
        self._pdf_path = path
        self.path_label.setText(Path(path).name)
        self._tables.clear()
        self.table_list.clear()
        self.preview.clear()
        self.statusMessage.emit("PDF loaded — click Extract tables.")

    def _on_extract(self) -> None:
        if not self._pdf_path:
            QMessageBox.information(self, "No PDF", "Load a PDF first.")
            return
        self.statusMessage.emit("Extracting tables…")
        thread, worker = call_in_thread(tools_core.pdf_extract_tables, self._pdf_path)
        worker.finished.connect(self._on_extracted)
        worker.error.connect(lambda msg: QMessageBox.critical(self, "Extraction error", msg))
        self._threads.append((thread, worker))

    def _on_extracted(self, result: tools_core.PdfTablesResult) -> None:
        self._tables = result.tables
        self.table_list.clear()
        if not self._tables:
            self.statusMessage.emit("No tables found in this PDF.")
            return
        for i, (tbl, page_no) in enumerate(zip(self._tables, result.page_nums)):
            nrow = len(tbl)
            ncol = max((len(r) for r in tbl), default=0)
            self.table_list.addItem(f"Table {i + 1}  (page {page_no})  {nrow}x{ncol}")
        self.table_list.setCurrentRow(0)
        self.statusMessage.emit(f"{len(self._tables)} table(s) extracted.")

    def _on_select(self, row: int) -> None:
        if row < 0 or row >= len(self._tables):
            return
        self._selected = row
        self._show_table(self._tables[row])

    def _show_table(self, table: List[List[str]]) -> None:
        self.preview.clear()
        if not table:
            self.preview.setRowCount(0)
            self.preview.setColumnCount(0)
            return
        ncol = max(len(r) for r in table)
        self.preview.setColumnCount(ncol)
        self.preview.setRowCount(len(table))
        for r, row in enumerate(table):
            for c in range(ncol):
                cell = row[c] if c < len(row) else ""
                self.preview.setItem(r, c, QTableWidgetItem(cell))
        self.preview.resizeColumnsToContents()

    def _on_export(self) -> None:
        if self._selected < 0 or not self._tables:
            QMessageBox.information(self, "No selection", "Extract tables and select one first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export table as CSV", "", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows(self._tables[self._selected])
        self.statusMessage.emit(f"Table saved to {path}")

    def _on_clean(self) -> None:
        if self._selected < 0 or not self._tables:
            QMessageBox.information(self, "No table", "Select a table first.")
            return
        if not tools_core.gemini_available():
            QMessageBox.information(self, "Gemini not available", "Set GOOGLE_API_KEY to use this feature.")
            return
        import io

        buf = io.StringIO()
        csv.writer(buf).writerows(self._tables[self._selected])
        self.statusMessage.emit("Asking Gemini to clean the table…")
        thread, worker = call_in_thread(tools_core.gemini_clean_table_csv, buf.getvalue())
        worker.finished.connect(self._on_cleaned)
        worker.error.connect(lambda msg: QMessageBox.critical(self, "Gemini error", msg))
        self._threads.append((thread, worker))

    def _on_cleaned(self, reply: str) -> None:
        try:
            reader = csv.reader(reply.strip().splitlines())
            new_table = [row for row in reader if row]
            self._tables[self._selected] = new_table
            self._show_table(new_table)
            self.statusMessage.emit("Table cleaned by Gemini — review and export.")
        except csv.Error as exc:
            QMessageBox.warning(self, "Parse error", f"Could not parse Gemini output as CSV:\n{exc}")


# ===========================================================================
# Jupyter notebook generation (KRUNAL_JULY_23.md point 8)
# ===========================================================================

class NotebookGenTab(QWidget):
    """Generates a ready-to-run .ipynb template (load mechanism, run a
    simulation, plot) against src/mso_io/notebook_gen.py -- no backend for
    this existed anywhere in the repo before."""

    statusMessage = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)

        form = QFormLayout()
        self.mechanism_picker = FilePicker(
            mode="file", filter_="Cantera YAML (*.yaml *.yml)", parent=self)
        form.addRow("Mechanism", self.mechanism_picker)

        self.kind_combo = QComboBox(self)
        self.kind_combo.addItem("Ignition delay (constant-volume)", "idt")
        self.kind_combo.addItem("Laminar flame speed", "flame_speed")
        self.kind_combo.addItem("Load + inspect only", "load_only")
        form.addRow("Notebook kind", self.kind_combo)

        self.fuel_edit = QLineEdit("H2", self)
        form.addRow("Fuel", self.fuel_edit)
        self.oxidizer_edit = QLineEdit("O2:1.0, N2:3.76", self)
        form.addRow("Oxidizer", self.oxidizer_edit)

        self.T_spin = QDoubleSpinBox(self)
        self.T_spin.setRange(200, 4000); self.T_spin.setValue(1000); self.T_spin.setSuffix(" K")
        form.addRow("Initial T", self.T_spin)
        self.P_spin = QDoubleSpinBox(self)
        self.P_spin.setRange(0.01, 500); self.P_spin.setValue(1.0); self.P_spin.setSuffix(" atm")
        form.addRow("Pressure", self.P_spin)
        self.phi_spin = QDoubleSpinBox(self)
        self.phi_spin.setRange(0.01, 10); self.phi_spin.setValue(1.0); self.phi_spin.setSingleStep(0.1)
        form.addRow("Equivalence ratio", self.phi_spin)

        layout.addLayout(form)

        btn_row = QHBoxLayout()
        self.generate_button = QPushButton("Generate notebook…", self)
        self.generate_button.clicked.connect(self._on_generate)
        btn_row.addWidget(self.generate_button)

        self.launch_button = QPushButton("Launch Jupyter…", self)
        self.launch_button.setToolTip(
            "Starts 'jupyter notebook' rooted at a folder you pick (e.g. the "
            "current project directory), as a background server -- the "
            "notebook opens in your browser and keeps running after this "
            "button click returns.")
        self.launch_button.clicked.connect(self._on_launch_jupyter)
        self.launch_button.setEnabled(tools_core.jupyter_available())
        btn_row.addWidget(self.launch_button)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self.result_label = QLabel("", self)
        self.result_label.setObjectName("Caption")
        layout.addWidget(self.result_label)
        layout.addStretch(1)

    def _on_generate(self) -> None:
        if not self.mechanism_picker.is_valid:
            QMessageBox.information(self, "No mechanism", "Pick a Cantera mechanism YAML first.")
            return
        default_name = Path(self.mechanism_picker.path()).stem + "_notebook.ipynb"
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save notebook", default_name, "Jupyter Notebook (*.ipynb)")
        if not out_path:
            return
        req = NotebookRequest(
            mechanism_path=self.mechanism_picker.path(),
            kind=self.kind_combo.currentData(),
            fuel=self.fuel_edit.text().strip() or "H2",
            oxidizer=self.oxidizer_edit.text().strip() or "O2:1.0, N2:3.76",
            T_K=self.T_spin.value(), P_atm=self.P_spin.value(), phi=self.phi_spin.value(),
        )
        try:
            path = generate_mechanism_notebook(out_path, req)
        except OSError as e:
            QMessageBox.critical(self, "Generate failed", str(e))
            return
        self.result_label.setText(f"Saved → {path}")
        self.statusMessage.emit(f"Notebook saved: {path}")

    def _on_launch_jupyter(self) -> None:
        root_dir = QFileDialog.getExistingDirectory(self, "Notebook root directory", "")
        if not root_dir:
            return
        try:
            pid = tools_core.launch_jupyter(root_dir)
        except OSError as e:
            QMessageBox.critical(self, "Launch failed", str(e))
            return
        self.result_label.setText(f"Jupyter started (pid {pid}) rooted at {root_dir}")
        self.statusMessage.emit(f"Jupyter notebook server started (pid {pid}).")


# ===========================================================================
# RDKit / SMILES (2026-08 workbench audit, helper-package gap #4)
# ===========================================================================

class RdkitTab(QWidget):
    """Validate a species/SMILES table (default: the repo's own
    species_smiles.csv, or any mechanism's species list a caller loads) via
    RDKit in the anaconda subprocess -- mso_studio.core.tools.rdkit_validate_smiles.
    A bad row never blocks the rest of the batch; the table shows exactly
    which rows failed and why."""

    statusMessage = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._threads = []
        self._rows: List[dict] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)

        btn_row = QHBoxLayout()
        self.load_default_button = QPushButton("Load species_smiles.csv", self)
        self.load_default_button.clicked.connect(self._on_load_default)
        btn_row.addWidget(self.load_default_button)
        self.load_csv_button = QPushButton("Load CSV…", self)
        self.load_csv_button.clicked.connect(self._on_load_csv)
        btn_row.addWidget(self.load_csv_button)
        self.validate_button = QPushButton("Validate all", self)
        self.validate_button.clicked.connect(self._on_validate)
        self.validate_button.setEnabled(False)
        btn_row.addWidget(self.validate_button)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        splitter = QSplitter(self)
        self.table = ResultsTable(
            ["Name", "SMILES", "Valid", "Canonical SMILES", "Formula", "MW", "Error"], splitter)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        splitter.addWidget(self.table)

        self.preview_label = QLabel("Select a row to preview its structure.", splitter)
        self.preview_label.setMinimumWidth(260)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setObjectName("Caption")
        splitter.addWidget(self.preview_label)
        splitter.setSizes([560, 260])
        layout.addWidget(splitter, 1)

    def _on_load_default(self) -> None:
        self._rows = tools_core.load_species_smiles_csv()
        self._populate_unvalidated()

    def _on_load_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load species/SMILES CSV", "", "CSV files (*.csv)")
        if not path:
            return
        self._rows = tools_core.load_species_smiles_csv(path)
        self._populate_unvalidated()

    def _populate_unvalidated(self) -> None:
        if not self._rows:
            QMessageBox.information(self, "No rows", "That CSV has no name/smiles columns, or is empty.")
            self.validate_button.setEnabled(False)
            return
        self.table.set_rows([[r["name"], r["smiles"], "", "", "", "", ""] for r in self._rows])
        self.validate_button.setEnabled(True)
        self.statusMessage.emit(f"Loaded {len(self._rows)} species — click Validate all.")

    def _on_validate(self) -> None:
        if not self._rows:
            return
        self.validate_button.setEnabled(False)
        self.statusMessage.emit("Validating SMILES via RDKit…")
        thread, worker = call_in_thread(tools_core.rdkit_validate_smiles, self._rows)
        worker.finished.connect(self._on_validated)
        worker.error.connect(self._on_validate_error)
        self._threads.append((thread, worker))

    def _on_validated(self, results: List[tools_core.SmilesValidation]) -> None:
        self.validate_button.setEnabled(True)
        self.table.set_rows([
            [r.name, r.smiles, "yes" if r.valid else "no",
             r.canonical_smiles or "", r.formula or "",
             r.mw if r.mw is not None else "", r.error or ""]
            for r in results
        ])
        n_bad = sum(1 for r in results if not r.valid)
        self.statusMessage.emit(
            f"Validated {len(results)} species — {n_bad} invalid." if n_bad
            else f"Validated {len(results)} species — all valid.")

    def _on_validate_error(self, message: str) -> None:
        self.validate_button.setEnabled(True)
        QMessageBox.critical(self, "Validation failed", message)
        self.statusMessage.emit("RDKit validation failed — is rdkit installed in the anaconda env?")

    def _on_row_selected(self) -> None:
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()})
        if not rows:
            return
        smiles_item = self.table.item(rows[0], 1)
        if smiles_item is None or not smiles_item.text():
            return
        try:
            png_path = tools_core.rdkit_render_structure(smiles_item.text())
        except Exception as exc:  # noqa: BLE001 - shown to the user, not fatal
            self.preview_label.setText(f"Render failed: {exc}")
            return
        pixmap = QPixmap(png_path)
        self.preview_label.setPixmap(pixmap)


# ===========================================================================
# RMG (RMG-Py mechanism generation, 2026-08 workbench audit, gap #4)
# ===========================================================================

class RmgTab(QWidget):
    """Run RMG-Py against an input.py config in its own conda environment
    (mso_studio.core.rmg) and surface the resulting chem_annotated.inp /
    species_dictionary.txt so it can flow into mechanism_converter /
    mechanism_check without hunting through chemkin/ by hand."""

    statusMessage = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._threads = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)

        available, reason = rmg_core.rmg_available()
        self.status_label = QLabel(("RMG-Py: " + reason), self)
        self.status_label.setObjectName("Caption")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        form = QFormLayout()
        self.input_picker = FilePicker(mode="file", filter_="RMG input (*.py)", parent=self)
        form.addRow("input.py", self.input_picker)
        self.out_dir_picker = FilePicker(mode="directory", must_exist=False, parent=self)
        self.out_dir_picker.edit.setPlaceholderText("default: alongside input.py")
        form.addRow("Output directory", self.out_dir_picker)
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        self.run_button = QPushButton("Run RMG…", self)
        self.run_button.setEnabled(available)
        self.run_button.clicked.connect(self._on_run)
        btn_row.addWidget(self.run_button)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self.log_view = LogView(self)
        layout.addWidget(self.log_view, 1)

        self.result_label = QLabel("", self)
        self.result_label.setObjectName("Caption")
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)

    def _on_run(self) -> None:
        if not self.input_picker.is_valid:
            QMessageBox.information(self, "No input.py", "Pick an RMG input.py config first.")
            return
        self.run_button.setEnabled(False)
        self.log_view.append_status("Starting RMG-Py (this can take a long time)…")
        self.statusMessage.emit("Running RMG…")
        out_dir = self.out_dir_picker.path() if self.out_dir_picker.is_valid else None
        thread, worker = call_in_thread(
            rmg_core.run_rmg_job, self.input_picker.path(), out_dir,
            on_line=self.log_view.append_line)
        worker.finished.connect(self._on_finished)
        worker.error.connect(self._on_error)
        self._threads.append((thread, worker))

    def _on_finished(self, result: rmg_core.RmgResult) -> None:
        self.run_button.setEnabled(True)
        lines = [f"Status: {result.status}", f"Output dir: {result.output_dir}"]
        if result.chem_annotated_path:
            lines.append(f"Mechanism (Chemkin): {result.chem_annotated_path}")
        if result.species_dictionary_path:
            lines.append(f"Species dictionary: {result.species_dictionary_path}")
        if result.cantera_yaml_path:
            lines.append(f"Cantera YAML: {result.cantera_yaml_path}")
        if not (result.chem_annotated_path or result.species_dictionary_path):
            lines.append("No chemkin/ output found -- the run may have stopped early; check the log above.")
        self.result_label.setText("\n".join(lines))
        self.statusMessage.emit(f"RMG run {result.status}.")

    def _on_error(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self.log_view.append_status(f"ERROR: {message}")
        self.statusMessage.emit("RMG run failed — see log.")


# ===========================================================================
# Native extension (.so) rebuild (KRUNAL_JULY_23.md point 8)
# ===========================================================================

class NativeExtensionsTab(QWidget):
    """Status + rebuild for this repo's pybind11 extensions (yamlwriter,
    parallel_yaml_writer, shuffle). See src/mso_io/native_extensions.py and
    mso_studio.core.tools.build_extension for why this is a real rebuild
    (source + build.sh/setup.py both exist under config_tools/), not a
    fabricated one."""

    statusMessage = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._threads = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)

        self.table = ResultsTable(["Extension", "Source found", "Installed copies"], self)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setMaximumHeight(140)
        layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh", self)
        self.refresh_button.clicked.connect(self._refresh)
        btn_row.addWidget(self.refresh_button)
        self.rebuild_button = QPushButton("Rebuild selected", self)
        self.rebuild_button.clicked.connect(self._on_rebuild)
        btn_row.addWidget(self.rebuild_button)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self.log_view = QPlainTextEdit(self)
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view, 1)

        self._infos = []
        self._refresh()

    def _refresh(self) -> None:
        from mso_io.native_extensions import inspect_all
        self._infos = inspect_all()
        self.table.set_rows([
            [info.name, "yes" if info.has_source else "no", len(info.installed_copies)]
            for info in self._infos
        ])

    def _on_rebuild(self) -> None:
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()})
        if not rows:
            QMessageBox.information(self, "Nothing selected", "Select an extension row first.")
            return
        name = self._infos[rows[0]].name
        self.log_view.appendPlainText(f"\n--- Rebuilding {name} ---")
        self.statusMessage.emit(f"Rebuilding {name}…")
        self.rebuild_button.setEnabled(False)

        from mso_studio.core.tools import build_extension
        thread, worker = call_in_thread(build_extension, name)
        worker.finished.connect(self._on_build_finished)
        worker.error.connect(self._on_build_error)
        self._threads.append((thread, worker))

    def _on_build_finished(self, result) -> None:
        self.rebuild_button.setEnabled(True)
        self.log_view.appendPlainText(result.log)
        if result.ok:
            self.log_view.appendPlainText(f"OK — built {result.built_artifact}")
            if result.copied_to:
                self.log_view.appendPlainText("Copied to:\n  " + "\n  ".join(result.copied_to))
            self.statusMessage.emit(f"{result.name} rebuilt.")
        else:
            self.log_view.appendPlainText(f"FAILED — see log above.")
            self.statusMessage.emit(f"{result.name} build failed.")
        self._refresh()

    def _on_build_error(self, message: str) -> None:
        self.rebuild_button.setEnabled(True)
        self.log_view.appendPlainText(f"ERROR: {message}")
        self.statusMessage.emit("Build error.")
