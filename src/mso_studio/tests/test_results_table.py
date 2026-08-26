"""
ResultsTable: the shared read-only-table widget that replaces 7 hand-built
QTableWidget setups across the module pages (Session B1 of the
visual-system plan -- see MarkDownFiles/GUI_CONSOLIDATION_AUDIT.md).
"""
import csv

import pytest


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_construction_sets_headers_and_no_edit(qapp):
    from PySide6.QtWidgets import QAbstractItemView

    from mso_studio.gui.widgets.results_table import ResultsTable

    table = ResultsTable(["Sev", "Check", "Message", "Location"])
    assert table.columnCount() == 4
    assert table.headers() == ["Sev", "Check", "Message", "Location"]
    assert table.editTriggers() == QAbstractItemView.EditTrigger.NoEditTriggers


def test_set_rows_populates_and_replaces(qapp):
    from mso_studio.gui.widgets.results_table import ResultsTable

    table = ResultsTable(["A", "B"])
    table.set_rows([[1, "x"], [2, "y"]])
    assert table.rowCount() == 2
    assert table.rows_as_lists() == [["1", "x"], ["2", "y"]]

    table.set_rows([[3, "z"]])
    assert table.rowCount() == 1
    assert table.rows_as_lists() == [["3", "z"]]


def test_set_rows_empty_clears_table(qapp):
    from mso_studio.gui.widgets.results_table import ResultsTable

    table = ResultsTable(["A"])
    table.set_rows([[1], [2]])
    table.set_rows([])
    assert table.rowCount() == 0


def test_export_csv_round_trips(qapp, tmp_path):
    from mso_studio.gui.widgets.results_table import ResultsTable

    table = ResultsTable(["Species", "Mole fraction"])
    table.set_rows([["CH4", 0.5], ["O2", 1.5]])

    out = tmp_path / "out.csv"
    table.export_csv(str(out))

    with open(out, newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["Species", "Mole fraction"]
    assert rows[1] == ["CH4", "0.5"]
    assert rows[2] == ["O2", "1.5"]


def test_prompt_export_warns_on_empty_table_without_a_file_dialog(qapp, monkeypatch):
    """Doesn't invoke QMenu.exec() (would block a headless test waiting for
    a selection) -- exercises _prompt_export directly, the part that
    actually decides whether to open a save dialog."""
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    from mso_studio.gui.widgets.results_table import ResultsTable

    table = ResultsTable(["A"])  # no rows
    dialog_calls = []
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: dialog_calls.append(1) or ("", ""))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    table._prompt_export()
    assert dialog_calls == []  # empty table -> no dialog shown at all


def test_prompt_export_writes_file_when_path_chosen(qapp, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QFileDialog

    from mso_studio.gui.widgets.results_table import ResultsTable

    table = ResultsTable(["A"])
    table.set_rows([[1]])
    out = tmp_path / "chosen.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), "CSV (*.csv)"))
    table._prompt_export()
    assert out.exists()
