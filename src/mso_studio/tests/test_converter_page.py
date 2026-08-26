"""
Widget-level test for ConverterPage (KRUNAL_JULY_23.md: user asked to
convert all Tkinter GUIs to PySide6). Same minimal convention as
test_run_summary.py: construct + drive build_core_call(), no click
simulation, no real conversion run.
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
    assert "mechanism_converter" in ids


def test_build_core_call_requires_mechanism(qapp):
    from mso_studio.gui.modules.converter import ConverterPage
    page = ConverterPage()
    with pytest.raises(ValueError, match="mechanism"):
        page.build_core_call()


def test_build_core_call_requires_at_least_one_output(qapp, tmp_path):
    from mso_studio.gui.modules.converter import ConverterPage
    mech = tmp_path / "m.inp"
    mech.write_text("ELEMENTS\nH\nEND\nSPECIES\nH2\nEND\n")
    page = ConverterPage()
    page.mechanism_picker.set_path(str(mech))
    page.make_yaml_check.setChecked(False)
    with pytest.raises(ValueError, match="output"):
        page.build_core_call()


def test_build_core_call_returns_run_convert_with_params(qapp, tmp_path):
    from mso_studio.core.converter import ConverterParams, run_convert
    from mso_studio.gui.modules.converter import ConverterPage

    mech = tmp_path / "128-957.inp"
    mech.write_text("ELEMENTS\nH\nEND\nSPECIES\nH2\nEND\n")
    page = ConverterPage()
    page.mechanism_picker.set_path(str(mech))
    page.thermo_picker.set_path(str(tmp_path / "128-957.dat"))
    page.output_dir_picker.set_path(str(tmp_path / "out"))
    page.permissive_check.setChecked(True)

    fn, args, kwargs = page.build_core_call()
    assert fn is run_convert
    assert kwargs == {}
    (params,) = args
    assert isinstance(params, ConverterParams)
    assert params.mechanism == str(mech)
    assert params.output_dir == str(tmp_path / "out")
    assert params.permissive is True
    assert page._current_label == "128-957"


def test_populate_results_shows_status_and_error(qapp):
    from mso_studio.core.runrecord import RunConfig, new_record
    from mso_studio.gui.modules.converter import ConverterPage

    page = ConverterPage()
    cfg = RunConfig(module="mechanism_converter", label="x")
    record = new_record(cfg)
    record.mark_finished("failed", return_code=1, error="ck2yaml failed")
    page.populate_results(record)
    text = page.result_view.toPlainText()
    assert "failed" in text
    assert "ck2yaml failed" in text
