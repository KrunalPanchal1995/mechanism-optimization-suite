"""
Widget-level test for NotebookGenTab (KRUNAL_JULY_23.md point 8 -- Jupyter
notebook generation). Same minimal convention as test_converter_page.py:
construct + drive methods directly, no click simulation.
"""
import json

import pytest


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_tab_registered_on_tools_page(qapp):
    from mso_studio.gui.modules.tools import ToolsPage, NotebookGenTab
    page = ToolsPage()
    assert isinstance(page.notebook_gen_tab, NotebookGenTab)


def test_default_kind_is_idt(qapp):
    from mso_studio.gui.modules.tools import NotebookGenTab
    tab = NotebookGenTab()
    assert tab.kind_combo.currentData() == "idt"


def test_generate_writes_valid_notebook(qapp, tmp_path, monkeypatch):
    from mso_studio.gui.modules.tools import NotebookGenTab
    from PySide6.QtWidgets import QFileDialog

    mech = tmp_path / "mech.yaml"
    mech.write_text("phases: []\n")
    out_path = tmp_path / "out.ipynb"

    tab = NotebookGenTab()
    tab.mechanism_picker.set_path(str(mech))
    tab.fuel_edit.setText("H2")

    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out_path), ""))
    tab._on_generate()

    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["nbformat"] == 4
    assert str(mech) in "".join(
        "".join(c["source"]) for c in data["cells"] if c["cell_type"] == "code"
    )
    assert str(out_path) in tab.result_label.text()


def test_generate_without_mechanism_shows_message(qapp, monkeypatch):
    from mso_studio.gui.modules.tools import NotebookGenTab
    from PySide6.QtWidgets import QMessageBox

    tab = NotebookGenTab()
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(a))
    tab._on_generate()
    assert len(shown) == 1
