"""
Tests for the Workbench Qt widgets (workbench_view.StepCard / WorkbenchPage).
Constructs real widgets under QT_QPA_PLATFORM=offscreen and drives real
methods, per this test package's convention (see test_launcher_window.py's
docstring) -- no click simulation, call the same methods a click would.
"""
import pytest

from mso_studio.launcher.workbench import STATUS_DONE, STATUS_FAILED, Workbench


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def page(qapp):
    from mso_studio.launcher.workbench_view import WorkbenchPage
    return WorkbenchPage("dark")


def test_page_starts_with_default_template(page):
    assert len(page._cards) == len(Workbench.default_template().steps)


def test_no_project_state_shows_placeholder(page):
    page.set_project(None)
    assert "No project open" in page.project_name_label.text()


def test_set_project_loads_workbench_yaml(page, tmp_path):
    wb = Workbench.default_template()
    wb.save(str(tmp_path))
    page.set_project(str(tmp_path), "My Project")
    assert page.project_name_label.text() == "My Project"
    assert len(page._cards) == len(wb.steps)


def test_add_step_appends_and_autosaves(page, tmp_path):
    page.set_project(str(tmp_path), "P")
    before = len(page._cards)
    page._add_step("mechanism_converter")
    assert len(page._cards) == before + 1
    reloaded = Workbench.load(str(tmp_path))
    assert len(reloaded.steps) == before + 1


def test_remove_selected_step(page, tmp_path):
    page.set_project(str(tmp_path), "P")
    initial = len(page._cards)
    page._on_card_selected(0)
    page._remove_selected()
    assert len(page._cards) == initial - 1
    reloaded = Workbench.load(str(tmp_path))
    assert len(reloaded.steps) == initial - 1


def test_move_selected_reorders(page, tmp_path):
    page.set_project(str(tmp_path), "P")
    first_id = page._workbench.steps[0].entry_id
    second_id = page._workbench.steps[1].entry_id
    page._on_card_selected(0)
    page._move_selected(1)
    assert page._workbench.steps[0].entry_id == second_id
    assert page._workbench.steps[1].entry_id == first_id


def test_double_click_emits_launch_requested(page, tmp_path):
    page.set_project(str(tmp_path), "P")
    received = []
    page.launchRequested.connect(received.append)
    entry_id = page._workbench.steps[0].entry_id
    page._on_card_activated(0)
    assert received == [entry_id]


def test_learn_more_emits_help_requested(page, tmp_path):
    page.set_project(str(tmp_path), "P")
    received = []
    page.helpRequested.connect(received.append)
    entry_id = page._workbench.steps[0].entry_id
    page._on_card_selected(0)
    page._on_learn_more_clicked()
    assert received == [entry_id]


def test_reset_to_template_restores_default(page, tmp_path):
    page.set_project(str(tmp_path), "P")
    page._on_card_selected(0)
    page._remove_selected()
    page._reset_to_template()
    assert [s.entry_id for s in page._workbench.steps] == [
        s.entry_id for s in Workbench.default_template().steps
    ]


def test_refresh_status_updates_card_pills(page, tmp_path):
    page.set_project(str(tmp_path), "P")
    entry_id = page._workbench.steps[0].entry_id
    from mso_studio.launcher.workbench import pipeline_for_entry
    pipeline = pipeline_for_entry(entry_id)
    runs = [{"pipeline": pipeline, "status": "done", "started_at": "2026-01-01T00:00:00"}] if pipeline else []
    page.refresh_status(runs)
    if pipeline:
        assert page._cards[0].status_label.objectName() == "StepStatusDone"


def test_note_edit_persists_to_step(page, tmp_path):
    page.set_project(str(tmp_path), "P")
    page._on_card_selected(0)
    page.note_edit.setText("test note")
    page._on_note_edited()
    assert page._workbench.steps[0].note == "test note"
    reloaded = Workbench.load(str(tmp_path))
    assert reloaded.steps[0].note == "test note"


def test_detail_strip_disabled_with_no_selection(page, tmp_path):
    page.set_project(str(tmp_path), "P")
    page._selected_index = None
    page._update_detail_strip()
    assert not page.run_button.isEnabled()


def test_detail_strip_enabled_with_selection(page, tmp_path):
    page.set_project(str(tmp_path), "P")
    page._on_card_selected(0)
    assert page.run_button.isEnabled()
