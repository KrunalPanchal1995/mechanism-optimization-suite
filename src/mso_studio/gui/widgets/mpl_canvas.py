"""
mso_studio.gui.widgets.mpl_canvas
====================================
Thin ``FigureCanvasQTAgg`` wrapper plus a CSV-table preview widget, used by
the Plots/Tables tabs of every module page's results panel.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget  # noqa: E402

from mso_io.mpl_theme import apply_global_rc_params, apply_rc_params, tokens_for_mode  # noqa: E402
from mso_studio.gui.theme import system_prefers_dark  # noqa: E402


class MplCanvas(FigureCanvasQTAgg):
    """A blank canvas ready for ``self.figure``. Colorblind-safe default cycle
    (matplotlib's ``tab10``) so figures match the LaTeX report's figures
    (guidelines doc, Part 6.5). Figure/axes colors are re-themed to match the
    app's current light/dark palette -- previously this canvas had no
    facecolor override at all, so every embedded plot was a plain white
    rectangle regardless of the surrounding dark shell."""

    def __init__(self, parent=None, width: float = 6.0, height: float = 4.0, dpi: int = 100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.figure)
        self.setParent(parent)
        self.apply_theme("system")

    def apply_theme(self, mode: str = "system") -> None:
        resolved = mode if mode in ("dark", "light") else ("dark" if system_prefers_dark() else "light")
        tokens = tokens_for_mode(resolved)
        apply_global_rc_params(tokens)
        apply_rc_params(self.figure, tokens)
        self.draw_idle()

    def clear(self) -> None:
        self.figure.clear()
        self.draw_idle()


class CsvTableView(QWidget):
    """Read-only table view of a CSV file, capped so a huge file doesn't
    freeze the GUI thread (Part 1.2) -- full data stays on disk, this is a
    preview only."""

    MAX_ROWS = 2000

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(self)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table)

    def load_csv(self, path: str) -> None:
        self.table.clear()
        p = Path(path)
        if not p.exists():
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return
        with p.open(newline="") as fh:
            reader = csv.reader(fh)
            rows = list(reader)
        if not rows:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return
        header, body = rows[0], rows[1:1 + self.MAX_ROWS]
        self.table.setColumnCount(len(header))
        self.table.setHorizontalHeaderLabels(header)
        self.table.setRowCount(len(body))
        for r, row in enumerate(body):
            for c, value in enumerate(row):
                self.table.setItem(r, c, QTableWidgetItem(value))
        self.table.resizeColumnsToContents()
