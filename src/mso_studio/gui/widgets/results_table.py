"""
mso_studio.gui.widgets.results_table
=======================================
``ResultsTable`` -- the read-only results table pattern that was hand-built
seven times across the module pages (mechanism_check's findings table,
uncertainty's results table, automation's task table, sensitivity's
brute-force targets table, rmip_suite's fuel table, and two in tools.py),
each repeating the same ``setColumnCount``/``setHorizontalHeaderLabels``/
``setEditTriggers(NoEditTriggers)`` setup and the same row-by-row
``QTableWidgetItem`` population loop, with no CSV export anywhere
(see ``MarkDownFiles/GUI_CONSOLIDATION_AUDIT.md``, Session B1).
"""
from __future__ import annotations

from typing import List, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QFileDialog, QMenu, QMessageBox, QTableWidget, QTableWidgetItem,
)


class ResultsTable(QTableWidget):
    """A read-only table: construct with column headers, populate rows with
    :meth:`set_rows`, export with :meth:`export_csv` or the right-click
    "Export CSV..." context-menu action every instance gets for free."""

    def __init__(self, headers: Sequence[str], parent=None):
        super().__init__(0, len(headers), parent)
        self.setHorizontalHeaderLabels(list(headers))
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)

    def set_rows(self, rows: Sequence[Sequence]) -> None:
        """Replace all rows. Each cell is ``str()``-ed, matching every
        existing call site's behavior (all seven pass in already-formatted
        strings or values whose ``str()`` is the display value)."""
        self.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                self.setItem(r, c, QTableWidgetItem(str(value)))

    def rows_as_lists(self) -> List[List[str]]:
        return [
            [self.item(r, c).text() if self.item(r, c) is not None else "" for c in range(self.columnCount())]
            for r in range(self.rowCount())
        ]

    def headers(self) -> List[str]:
        return [
            self.horizontalHeaderItem(c).text() if self.horizontalHeaderItem(c) is not None else ""
            for c in range(self.columnCount())
        ]

    def export_csv(self, path: str) -> None:
        import csv

        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(self.headers())
            writer.writerows(self.rows_as_lists())

    def _on_context_menu(self, pos) -> None:
        menu = QMenu(self)
        export_action = menu.addAction("Export CSV…")
        action = menu.exec(self.mapToGlobal(pos))
        if action is export_action:
            self._prompt_export()

    def _prompt_export(self) -> None:
        if self.rowCount() == 0:
            QMessageBox.information(self, "Nothing to export", "This table is empty.")
            return
        path, _filter = QFileDialog.getSaveFileName(self, "Export CSV", "", "CSV (*.csv)")
        if path:
            self.export_csv(path)
