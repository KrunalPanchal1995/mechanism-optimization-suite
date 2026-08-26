"""Widget-level tests for UncertaintyPage (KRUNAL_JULY_23.md PySide6 conversion)."""
import pytest


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_page_registered_in_page_factories():
    from mso_studio.gui.modules import PAGE_FACTORIES
    ids = [mid for mid, _label, _factory in PAGE_FACTORIES]
    assert "uncertainty" in ids


def test_add_gen_row_accumulates_rows(qapp):
    from mso_studio.gui.modules.uncertainty import UncertaintyPage
    page = UncertaintyPage()
    page._gen_rxn_edit.setText("H+O2<=>O+OH")
    page._on_add_gen_row()
    assert len(page._gen_rows) == 1
    assert page._gen_rows[0]["name"] == "H+O2<=>O+OH"
    assert page._gen_rxn_edit.text() == ""   # cleared after add


def test_add_gen_row_requires_name(qapp):
    from mso_studio.gui.modules.uncertainty import UncertaintyPage
    page = UncertaintyPage()
    page._gen_rxn_edit.setText("")
    page._on_add_gen_row()
    assert len(page._gen_rows) == 0


def test_build_core_call_requires_xml_and_mechanism(qapp):
    from mso_studio.gui.modules.uncertainty import UncertaintyPage
    page = UncertaintyPage()
    with pytest.raises(ValueError, match="XML"):
        page.build_core_call()


def test_build_core_call_returns_run_compute_bounds(qapp, tmp_path):
    from mso_studio.core.uncertainty import ComputeBoundsParams, run_compute_bounds
    from mso_studio.gui.modules.uncertainty import UncertaintyPage

    xml = tmp_path / "u.xml"
    xml.write_text("<uncertainty/>")
    mech = tmp_path / "m.yaml"
    mech.write_text("phases: []\n")
    page = UncertaintyPage()
    page.xml_picker.set_path(str(xml))
    page.mechanism_picker.set_path(str(mech))

    fn, args, kwargs = page.build_core_call()
    assert fn is run_compute_bounds
    assert kwargs == {}
    (params,) = args
    assert isinstance(params, ComputeBoundsParams)
    assert params.xml_path == str(xml)
    assert params.mechanism == str(mech)


def test_populate_results_shows_ok_and_error_rows(qapp):
    from mso_studio.core.runrecord import RunConfig, new_record
    from mso_studio.gui.modules.uncertainty import UncertaintyPage

    page = UncertaintyPage()
    cfg = RunConfig(module="uncertainty_compute_bounds", label="u")
    record = new_record(cfg)
    record.results["rows"] = [
        {"rxn": "H+O2<=>O+OH", "A_nominal": 9.841e13, "factor": 1.49, "A_lb": 6.6e13, "A_ub": 1.47e14},
        {"rxn": "BAD+X<=>Y", "error": "no thermo"},
    ]
    page.populate_results(record)
    assert page._results_table.rowCount() == 2
    assert page._results_table.item(0, 0).text() == "H+O2<=>O+OH"
    assert "ERROR" in page._results_table.item(1, 4).text()


def test_page_title_renamed(qapp):
    """KRUNAL_JULY_23.md point 5 -- rename to match what the tool does."""
    from PySide6.QtWidgets import QLabel
    from mso_studio.gui.modules.uncertainty import UncertaintyPage
    page = UncertaintyPage()
    header = next(w for w in page.findChildren(QLabel) if w.objectName() == "SectionHeader")
    assert header.text() == "Uncertainty Quantification and File Generator"


def test_load_sa_demo_populates_fields(qapp):
    from mso_studio.gui.modules.uncertainty import UncertaintyPage, _DEMO
    page = UncertaintyPage()
    page._on_load_sa_demo()
    assert page.sa_mechanism_picker.path() == _DEMO["mechanism"]
    assert page.sa_fuel_edit.text() == "MB_MB2D_LALIT_2024"
    assert page.generate_xml_check.isChecked()
    assert page.generate_plots_check.isChecked()
    assert page.generate_latex_check.isChecked()


def test_demo_paths_exist_in_this_checkout(qapp):
    import os
    from mso_studio.gui.modules.uncertainty import UncertaintyPage
    page = UncertaintyPage()
    page._on_load_sa_demo()
    assert os.path.isdir(page.sens_dir_picker.path())
    assert os.path.isfile(page.sa_mechanism_picker.path())
    assert os.path.isfile(page.classification_picker.path())


def test_build_sa_params_requires_mechanism(qapp):
    from mso_studio.gui.modules.uncertainty import UncertaintyPage
    page = UncertaintyPage()
    with pytest.raises(ValueError, match="mechanism"):
        page._build_sa_params()


def test_build_sa_params_matches_demo_invocation(qapp):
    from mso_studio.gui.modules.uncertainty import UncertaintyPage, _DEMO
    page = UncertaintyPage()
    page._on_load_sa_demo()
    params = page._build_sa_params()
    assert params.mechanism == _DEMO["mechanism"]
    assert params.threshold == 0.05
    assert params.carbon == 5
    assert params.fuel == "MB_MB2D_LALIT_2024"
    assert params.generate_xml and params.generate_plots and params.generate_latex
    assert params.compare_temperatures == [698.0, 908.0, 1287.0]


def test_run_sa_screen_invokes_core_and_populates_log(qapp, monkeypatch):
    from mso_studio.gui.modules import uncertainty as mod
    from mso_studio.gui.modules.uncertainty import UncertaintyPage
    from mso_studio.core.runrecord import RunConfig, new_record

    page = UncertaintyPage()
    page._on_load_sa_demo()

    captured = {}

    def fake_run(params):
        captured["params"] = params
        cfg = RunConfig(module="active_parameter_screen", label="t")
        record = new_record(cfg)
        record.results["results_dir"] = "results"
        record.mark_finished("done", 0)
        return record

    monkeypatch.setattr(mod, "run_active_parameter_screen", fake_run)
    page._on_run_sa_screen()
    assert "params" in captured
    assert "Status: done" in page._sa_log.toPlainText()
