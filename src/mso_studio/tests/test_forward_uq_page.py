"""Widget-level tests for ForwardUQPage (KRUNAL_JULY_23.md PySide6 conversion)."""
import pytest


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_page_registered_in_page_factories():
    from mso_studio.gui.modules import PAGE_FACTORIES
    ids = [mid for mid, _label, _factory in PAGE_FACTORIES]
    assert "forward_uq" in ids


def test_build_core_call_requires_target_opt(qapp):
    from mso_studio.gui.modules.forward_uq import ForwardUQPage
    page = ForwardUQPage()
    with pytest.raises(ValueError, match="target.opt"):
        page.build_core_call()


def test_build_core_call_requires_output_dir(qapp, tmp_path):
    from mso_studio.gui.modules.forward_uq import ForwardUQPage
    target_opt = tmp_path / "case1.opt"
    target_opt.write_text("Locations: {}\n")
    page = ForwardUQPage()
    page.target_opt_picker.set_path(str(target_opt))
    with pytest.raises(ValueError, match="Opt/"):
        page.build_core_call()


def test_build_core_call_returns_run_forward_uq_with_params(qapp, tmp_path):
    from mso_studio.core.forward_uq import ForwardUqParams, run_forward_uq
    from mso_studio.gui.modules.forward_uq import ForwardUQPage

    target_opt = tmp_path / "case1.opt"
    target_opt.write_text("Locations: {}\n")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    page = ForwardUQPage()
    page.target_opt_picker.set_path(str(target_opt))
    page.output_dir_picker.set_path(str(out_dir))
    page.n_samples_spin.setValue(1000)
    page.method_combo.setCurrentText("lhs")
    page.seed_spin.setValue(7)

    fn, args, kwargs = page.build_core_call()
    assert fn is run_forward_uq
    assert kwargs == {}
    (params,) = args
    assert isinstance(params, ForwardUqParams)
    assert params.target_opt == str(target_opt)
    assert params.n_samples == 1000
    assert params.method == "lhs"
    assert params.seed == 7
    assert page._current_label == "case1"


def test_populate_results_loads_statistics_csv(qapp, tmp_path):
    from mso_studio.core.runrecord import RunConfig, new_record
    from mso_studio.gui.modules.forward_uq import ForwardUQPage

    stats = tmp_path / "forward_uq_statistics.csv"
    stats.write_text("case_id,eta_mean\ncase-0,1.2\n")
    page = ForwardUQPage()
    cfg = RunConfig(module="forward_uq", label="x")
    record = new_record(cfg)
    record.results["statistics_path"] = str(stats)
    page.populate_results(record)   # must not raise
