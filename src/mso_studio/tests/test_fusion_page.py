"""Widget-level tests for FusionPage (KRUNAL_JULY_23.md PySide6 conversion)."""
import pytest


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_page_registered_in_page_factories():
    from mso_studio.gui.modules import PAGE_FACTORIES
    ids = [mid for mid, _label, _factory in PAGE_FACTORIES]
    assert "mechanism_fusion" in ids


def test_build_core_call_requires_database(qapp):
    from mso_studio.gui.modules.fusion import FusionPage
    page = FusionPage()
    with pytest.raises(ValueError, match="database"):
        page.build_core_call()


def test_build_core_call_requires_two_mechanism_ids(qapp, tmp_path):
    from mso_studio.gui.modules.fusion import FusionPage
    db = tmp_path / "p.db"
    db.write_text("")
    page = FusionPage()
    page.db_picker.set_path(str(db))
    page.mech_ids_edit.setText("2")
    with pytest.raises(ValueError, match="at least 2"):
        page.build_core_call()


def test_priority_mode_requires_matching_order(qapp, tmp_path):
    from mso_studio.gui.modules.fusion import FusionPage
    db = tmp_path / "p.db"
    db.write_text("")
    page = FusionPage()
    page.db_picker.set_path(str(db))
    page.mech_ids_edit.setText("2,3,4")
    page.priority_mode_check.setChecked(True)
    page.priority_order_edit.setText("2,3")   # missing 4
    with pytest.raises(ValueError, match="exactly the same"):
        page.build_core_call()


def test_build_core_call_returns_run_fusion_with_params(qapp, tmp_path):
    from mso_studio.core.fusion import FusionParams, run_fusion
    from mso_studio.gui.modules.fusion import FusionPage

    db = tmp_path / "p.db"
    db.write_text("")
    page = FusionPage()
    page.db_picker.set_path(str(db))
    page.mech_ids_edit.setText("2, 3, 4")
    page.output_name_edit.setText("MyFusion")
    page.output_dir_picker.set_path(str(tmp_path / "out"))
    page.priority_mode_check.setChecked(True)
    page.priority_order_edit.setText("4,3,2")

    fn, args, kwargs = page.build_core_call()
    assert fn is run_fusion
    assert kwargs == {}
    (params,) = args
    assert isinstance(params, FusionParams)
    assert params.mechanism_ids == [2, 3, 4]
    assert params.priority_order == [4, 3, 2]
    assert params.output_name == "MyFusion"
    assert page._current_label == "MyFusion"


def test_populate_results_shows_counts(qapp):
    from mso_studio.core.runrecord import RunConfig, new_record
    from mso_studio.gui.modules.fusion import FusionPage

    page = FusionPage()
    cfg = RunConfig(module="mechanism_fusion", label="x")
    record = new_record(cfg)
    record.results.update({"fused_mech_id": 19, "n_species": 14, "n_verbatim": 3,
                           "n_copy": 8, "n_fuse": 0, "balanced": True})
    record.mark_finished("done", 0)
    page.populate_results(record)
    text = page.result_view.toPlainText()
    assert "fused_mech_id: 19" in text
    assert "n_species: 14" in text
