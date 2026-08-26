"""
Tests for the Workbench data model (mso_studio.launcher.workbench) --
pure Python, no Qt. Covers round-trip persistence, graceful degradation on
malformed/unknown data (the launcher must never fail to open a project
because of a stale workbench.yaml), and the pure run<->step status join.
"""
import importlib.util

import pytest

from mso_studio.launcher.catalog import CATALOG_BY_ID
from mso_studio.launcher.workbench import (
    DEFAULT_TEMPLATE_IDS, PIPELINE_FOR_ENTRY, STATUS_DONE, STATUS_FAILED,
    STATUS_PENDING, STATUS_RUNNING, STATUS_UNKNOWN, Workbench, WorkbenchStep,
    pipeline_for_entry, status_for_steps,
)

_HAS_MSO_PROJECT = importlib.util.find_spec("mso_project") is not None


def test_default_template_ids_all_exist_in_catalog():
    for entry_id in DEFAULT_TEMPLATE_IDS:
        assert entry_id in CATALOG_BY_ID


def test_default_template_builds_steps_in_order():
    wb = Workbench.default_template()
    assert [s.entry_id for s in wb.steps] == DEFAULT_TEMPLATE_IDS


def test_save_and_load_round_trip(tmp_path):
    wb = Workbench.default_template()
    wb.steps[0].note = "start here"
    wb.steps[1].optional = True
    wb.save(str(tmp_path))

    loaded = Workbench.load(str(tmp_path))
    assert [s.entry_id for s in loaded.steps] == [s.entry_id for s in wb.steps]
    assert loaded.steps[0].note == "start here"
    assert loaded.steps[1].optional is True


def test_load_missing_file_returns_default_template(tmp_path):
    wb = Workbench.load(str(tmp_path))
    assert [s.entry_id for s in wb.steps] == DEFAULT_TEMPLATE_IDS


def test_load_malformed_yaml_falls_back_to_default(tmp_path):
    (tmp_path / "workbench.yaml").write_text(":: not: valid: yaml: [[[")
    wb = Workbench.load(str(tmp_path))
    assert [s.entry_id for s in wb.steps] == DEFAULT_TEMPLATE_IDS


def test_load_drops_unknown_entry_id_without_raising(tmp_path):
    (tmp_path / "workbench.yaml").write_text(
        "schema_version: 1\n"
        "steps:\n"
        "  - entry_id: mechanism_converter\n"
        "  - entry_id: some_removed_module\n"
    )
    wb = Workbench.load(str(tmp_path))
    assert [s.entry_id for s in wb.steps] == ["mechanism_converter"]


def test_load_future_schema_version_still_loads_known_fields(tmp_path):
    (tmp_path / "workbench.yaml").write_text(
        "schema_version: 99\n"
        "steps:\n"
        "  - entry_id: mechanism_converter\n"
        "    label: custom label\n"
    )
    wb = Workbench.load(str(tmp_path))
    assert wb.schema_version == 99
    assert wb.steps[0].label == "custom label"


def test_insert_rejects_unknown_entry_id():
    wb = Workbench(steps=[])
    with pytest.raises(ValueError):
        wb.insert(0, "not_a_real_module")


def test_move_clamps_at_bounds():
    wb = Workbench.default_template()
    n = len(wb.steps)
    first_id = wb.steps[0].entry_id
    # Moving the first step left is a no-op (clamped, not an error).
    new_index = wb.move(0, -1)
    assert new_index == 0
    assert wb.steps[0].entry_id == first_id
    # Moving the last step right is likewise clamped.
    last_id = wb.steps[-1].entry_id
    new_index = wb.move(n - 1, +1)
    assert new_index == n - 1
    assert wb.steps[-1].entry_id == last_id


def test_move_reorders_steps():
    wb = Workbench.default_template()
    second_id = wb.steps[1].entry_id
    new_index = wb.move(1, -1)
    assert new_index == 0
    assert wb.steps[0].entry_id == second_id


def test_remove_and_append():
    wb = Workbench(steps=[])
    wb.append("mechanism_converter")
    wb.append("mechanism_check")
    assert len(wb.steps) == 2
    wb.remove(0)
    assert [s.entry_id for s in wb.steps] == ["mechanism_check"]


def test_display_label_falls_back_to_catalog_title():
    step = WorkbenchStep(entry_id="mechanism_converter")
    assert step.display_label() == CATALOG_BY_ID["mechanism_converter"].title


def test_display_label_prefers_custom_label():
    step = WorkbenchStep(entry_id="mechanism_converter", label="My step")
    assert step.display_label() == "My step"


def test_pipeline_for_entry_matches_id_by_default():
    assert pipeline_for_entry("nominal_sim") == "nominal_sim"


def test_pipeline_for_entry_uses_override_table():
    assert pipeline_for_entry("mechanism_fusion") == "mean_value_fusion"


def test_pipeline_for_entry_empty_for_entries_with_no_footprint():
    assert pipeline_for_entry("target_add_opt") == ""
    assert pipeline_for_entry("mso_studio") == ""
    assert pipeline_for_entry("doctor") == ""


@pytest.mark.skipif(not _HAS_MSO_PROJECT, reason="mso_project not importable")
def test_pipeline_overrides_match_real_footprint_keys():
    from mso_project.footprints import FOOTPRINTS

    for entry_id, pipeline in PIPELINE_FOR_ENTRY.items():
        if pipeline:
            assert pipeline in FOOTPRINTS, (entry_id, pipeline)


def test_status_for_steps_pending_with_no_runs():
    steps = [WorkbenchStep(entry_id="nominal_sim")]
    assert status_for_steps(steps, []) == {"nominal_sim": STATUS_PENDING}


def test_status_for_steps_reads_run_status():
    steps = [WorkbenchStep(entry_id="nominal_sim")]
    runs = [{"pipeline": "nominal_sim", "status": "done", "started_at": "2026-01-01T00:00:00"}]
    assert status_for_steps(steps, runs) == {"nominal_sim": STATUS_DONE}


def test_status_for_steps_latest_run_wins():
    steps = [WorkbenchStep(entry_id="nominal_sim")]
    runs = [
        {"pipeline": "nominal_sim", "status": "failed", "started_at": "2026-01-01T00:00:00"},
        {"pipeline": "nominal_sim", "status": "done", "started_at": "2026-01-02T00:00:00"},
    ]
    assert status_for_steps(steps, runs) == {"nominal_sim": STATUS_DONE}


def test_status_for_steps_unknown_pipeline_mapping():
    steps = [WorkbenchStep(entry_id="doctor")]
    runs = [{"pipeline": "doctor", "status": "done", "started_at": "2026-01-01T00:00:00"}]
    # doctor has no pipeline mapping (RUNNER_INTERNAL, never recorded as a
    # run) -- status is honestly UNKNOWN, not guessed from an unrelated run.
    assert status_for_steps(steps, runs) == {"doctor": STATUS_UNKNOWN}


def test_status_for_steps_running():
    steps = [WorkbenchStep(entry_id="nominal_sim")]
    runs = [{"pipeline": "nominal_sim", "status": "running", "started_at": "2026-01-01T00:00:00"}]
    assert status_for_steps(steps, runs) == {"nominal_sim": STATUS_RUNNING}


def test_status_for_steps_ignores_runs_with_no_pipeline():
    steps = [WorkbenchStep(entry_id="nominal_sim")]
    runs = [{"pipeline": None, "status": "done", "started_at": "2026-01-01T00:00:00"}]
    assert status_for_steps(steps, runs) == {"nominal_sim": STATUS_PENDING}
