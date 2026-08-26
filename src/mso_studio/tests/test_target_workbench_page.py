"""
Widget-level tests for TargetWorkbenchPage: construct + drive the
``_on_*``/``_refresh_rows_table`` methods directly with fake result
objects, matching test_sensitivity_page.py / test_converter_page.py's
convention -- no click simulation, no real subprocess bridge (that's
test_target_workbench.py's job).
"""
import pytest

from mso_studio.core import target_workbench as twb_core


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def _fake_load_result():
    rows = [
        twb_core.TargetRowInfo(
            line_no=1, dataset_id="ds_a", target_type="Tig",
            temperature_K=1500.0, pressure_atm=10.0, phi_raw="1.0",
            observed=500.0, deviation=50.0, data_weight=1.0,
            composition_sum=1.0, is_dummy=False),
        twb_core.TargetRowInfo(
            line_no=2, dataset_id="ds_a", target_type="Tig",
            temperature_K=1600.0, pressure_atm=10.0, phi_raw="1.0",
            observed=400.0, deviation=40.0, data_weight=1.0,
            composition_sum=1.0, is_dummy=False),
        twb_core.TargetRowInfo(
            line_no=3, dataset_id="ds_b", target_type="Tig",
            temperature_K=1400.0, pressure_atm=5.0, phi_raw="0.5",
            observed=800.0, deviation=80.0, data_weight=1.0,
            composition_sum=1.0, is_dummy=False),
    ]
    return twb_core.LoadResult(rows=rows, dataset_ids=["ds_a", "ds_b"])


def test_on_loaded_populates_rows_table_and_combos(qapp):
    from mso_studio.gui.modules.target_workbench import TargetWorkbenchPage

    page = TargetWorkbenchPage()
    page.target_picker.edit.setText("/fake/path.target")
    page._on_loaded(_fake_load_result())

    assert page.rows_table.rowCount() == 3
    assert [page.dummy_dataset_combo.itemText(i) for i in range(page.dummy_dataset_combo.count())] == ["ds_a", "ds_b"]
    assert [page.plot_dataset_combo.itemText(i) for i in range(page.plot_dataset_combo.count())] == ["ds_a", "ds_b"]
    assert "Loaded 3 rows" in page.load_status.text()


def test_mark_and_undrop_updates_status_column(qapp):
    from mso_studio.gui.modules.target_workbench import TargetWorkbenchPage

    page = TargetWorkbenchPage()
    page._on_loaded(_fake_load_result())

    page.rows_table.selectRow(0)
    page._on_mark_dropped()
    assert 1 in page._dropped
    row0 = [page.rows_table.item(0, c).text() for c in range(page.rows_table.columnCount())]
    assert row0[-1] == "dropped"

    page._on_undrop()  # row 0 still selected
    assert 1 not in page._dropped
    row0 = [page.rows_table.item(0, c).text() for c in range(page.rows_table.columnCount())]
    assert row0[-1] == "ok"


def test_on_checks_finished_populates_findings_table_and_status(qapp):
    from mso_studio.gui.modules.target_workbench import TargetWorkbenchPage

    page = TargetWorkbenchPage()
    findings = [
        twb_core.Finding(severity="error", line_no=1, dataset_id="ds_a",
                         check="parses", message="boom", suggested_fix=None),
        twb_core.Finding(severity="warning", line_no=2, dataset_id="ds_a",
                         check="units", message="odd T", suggested_fix=None),
        twb_core.Finding(severity="info", line_no=None, dataset_id="ds_b",
                         check="ntc_crossing", message="crosses NTC", suggested_fix=None),
    ]
    page._on_checks_finished(findings)
    assert page.findings_table.rowCount() == 3
    assert page._findings == findings
    assert "1 error(s), 1 warning(s), 1 info" in page.checks_status.text()


def test_generate_dummy_stages_rows_and_refreshes_table(qapp):
    from mso_studio.gui.modules.target_workbench import TargetWorkbenchPage

    page = TargetWorkbenchPage()
    page._on_loaded(_fake_load_result())

    result = twb_core.DummyGenerationResult(
        rows=[{"Temperature_K": 1550.0, "observed": 450.0},
              {"Temperature_K": 1560.0, "observed": 440.0}],
        template_dataset_id="ds_a",
    )
    page._on_dummy_generated(result, "ds_a")

    assert len(page._dummy_rows) == 2
    assert page._dummy_rows[0]["dataset_id"] == "ds_a"
    assert "2 dummy point(s) staged" in page.dummy_status.text()
    # 3 real rows + 2 staged dummy rows
    assert page.rows_table.rowCount() == 5


def test_apply_prune_requires_suggest_first(qapp):
    from mso_studio.gui.modules.target_workbench import TargetWorkbenchPage

    page = TargetWorkbenchPage()
    page._on_loaded(_fake_load_result())
    assert page._suggested_drop is None

    prune_result = twb_core.PruneResult(keep_line_nos=[1, 3], drop_line_nos=[2], coverage=0.75)
    page._on_prune_suggested(prune_result)
    assert page._suggested_drop == [2]
    assert page.apply_prune_button.isEnabled()

    page._on_apply_prune()
    assert 2 in page._dropped


def test_write_finished_reports_gate_errors(qapp):
    from mso_studio.gui.modules.target_workbench import TargetWorkbenchPage

    page = TargetWorkbenchPage()
    result = twb_core.WriteCuratedResult(
        new_target_path="/tmp/out_v1.target", new_add_path="/tmp/out_v1.add",
        log_path="/tmp/out_v1_curation_log.yaml",
        gate_errors=[{"message": "line 5 failed to parse"}],
    )
    page._on_write_finished(result)
    assert "1 row(s)" in page.write_status.text()
    assert "failed the combustion_target parser gate" in page.write_status.text()

    result_clean = twb_core.WriteCuratedResult(
        new_target_path="/tmp/out_v2.target", new_add_path="/tmp/out_v2.add",
        log_path="/tmp/out_v2_curation_log.yaml", gate_errors=[],
    )
    page._on_write_finished(result_clean)
    assert "All rows passed the parser gate" in page.write_status.text()
