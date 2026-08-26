"""
Widget-level tests for MechanismCheckPage (KRUNAL_JULY_23.md: PySide6
conversion). Same minimal convention as test_converter_page.py.
"""
import pytest


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_page_registered_in_page_factories():
    from mso_studio.gui.modules import PAGE_FACTORIES
    ids = [mid for mid, _label, _factory in PAGE_FACTORIES]
    assert "mechanism_check" in ids


def test_build_core_call_requires_a_mechanism(qapp):
    from mso_studio.gui.modules.mechanism_check import MechanismCheckPage
    page = MechanismCheckPage()
    with pytest.raises(ValueError, match="mechanism"):
        page.build_core_call()


def test_build_core_call_requires_at_least_one_diagnostic(qapp):
    from mso_studio.gui.modules.mechanism_check import MechanismCheckPage
    page = MechanismCheckPage()
    page._mech_list.addItem("/a/m.yaml")
    for cb in page._check_boxes.values():
        cb.setChecked(False)
    with pytest.raises(ValueError, match="diagnostic"):
        page.build_core_call()


def test_build_core_call_returns_run_check_with_params(qapp):
    from mso_studio.core.mechanism_check import MechanismCheckParams, run_check
    from mso_studio.gui.modules.mechanism_check import MechanismCheckPage

    page = MechanismCheckPage()
    page._mech_list.addItem("/a/m1.yaml")
    page._mech_list.addItem("/a/m2.yaml")
    page._t_grid_edit.setText("300 1000")

    fn, args, kwargs = page.build_core_call()
    assert fn is run_check
    assert kwargs == {}
    (params,) = args
    assert isinstance(params, MechanismCheckParams)
    assert params.mechanisms == ["/a/m1.yaml", "/a/m2.yaml"]
    assert params.T_grid == [300.0, 1000.0]
    assert params.checks == []          # all boxes checked -> "run everything"


def test_populate_results_and_selection_shows_detail(qapp):
    from mso_studio.core.runrecord import RunConfig, new_record
    from mso_studio.gui.modules.mechanism_check import MechanismCheckPage

    page = MechanismCheckPage()
    cfg = RunConfig(module="mechanism_check", label="m")
    record = new_record(cfg)
    record.results["reports"] = [{
        "mechanism": "/a/128-957.inp",
        "findings": [{
            "check": "load", "severity": "error", "category": "load-failure",
            "message": "Mechanism failed to load in Cantera — see error detail.",
            "location": "128-957.inp",
            "details": {"error": 'Unexpected token "C2H4O1-2+"'},
            "fix_options": [],
        }],
    }]
    page.populate_results(record)
    assert page._findings_table.rowCount() == 1

    page._findings_table.selectRow(0)
    text = page._detail_view.toPlainText()
    assert "128-957.inp" in text
    assert 'Unexpected token "C2H4O1-2+"' in text
    assert "How to fix this" in text
