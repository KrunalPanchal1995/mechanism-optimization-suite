"""
Widget-level tests for ProjectManagerPage: construct + drive the ``_on_*``
handlers directly with fake core results, matching
test_target_workbench_page.py's convention.
"""
import pytest

from mso_studio.core import project_manager as pm_core


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_on_project_opened_sets_state_and_status(qapp, monkeypatch):
    """_on_project_opened chains into _on_refresh_runs(), which spawns a
    real QThread (call_in_thread) -- every other test in this module (and
    test_target_workbench_page.py's convention) calls a worker.finished
    handler directly, never the button handler that starts the thread, to
    stay off real threading entirely. This is the one test that needs the
    button handler itself, so stub out the chained thread-spawning call:
    a bare script reproduction confirmed even an explicit thread.wait()
    doesn't avoid "QThread: Destroyed while thread '' is still running"
    aborting the process when the target path doesn't exist and the
    worker thread finishes before the test can synchronize with it."""
    from mso_studio.gui.modules.project_manager import ProjectManagerPage

    page = ProjectManagerPage()
    monkeypatch.setattr(page, "_on_refresh_runs", lambda: None)
    info = pm_core.ProjectInfo(
        root="/tmp/fake_project", name="Ammonia", description="",
        tags=["ammonia"], created_at="2026-08-06T00:00:00Z")
    page._on_project_opened(info)

    assert page._project_dir == "/tmp/fake_project"
    assert "Ammonia" in page.project_status.text()


def test_on_runs_loaded_populates_table_and_run_id_combo(qapp):
    from mso_studio.gui.modules.project_manager import ProjectManagerPage

    page = ProjectManagerPage()
    runs = [
        {"run_id": "r1", "pipeline": "nominal_sim", "label": "case1",
         "started_at": "t0", "finished_at": "t1", "status": "done"},
        {"run_id": "r2", "pipeline": "forward_uq", "label": "case2",
         "started_at": "t2", "finished_at": None, "status": "running"},
    ]
    page._on_runs_loaded(runs)

    assert page.runs_table.rowCount() == 2
    assert [page.run_id_combo.itemText(i) for i in range(page.run_id_combo.count())] == ["r1", "r2"]


def test_on_artifacts_loaded_populates_table(qapp):
    from mso_studio.gui.modules.project_manager import ProjectManagerPage

    page = ProjectManagerPage()
    artifacts = [{"artifact_id": 1, "run_id": "r1", "role": "results", "kind": "csv",
                  "path": "/a/b.csv", "size_bytes": 100, "tags": ["x", "y"]}]
    page._on_artifacts_loaded(artifacts)

    assert page.artifacts_table.rowCount() == 1
    row = [page.artifacts_table.item(0, c).text() for c in range(page.artifacts_table.columnCount())]
    assert row[-1] == "x,y"


def test_require_project_warns_when_none_open(qapp, monkeypatch):
    from mso_studio.gui.modules.project_manager import ProjectManagerPage

    page = ProjectManagerPage()
    warned = []
    monkeypatch.setattr(
        "mso_studio.gui.modules.project_manager.QMessageBox.warning",
        lambda *a, **k: warned.append(a))
    result = page._require_project()
    assert result is None
    assert warned


def test_on_note_added_clears_form(qapp):
    from mso_studio.gui.modules.project_manager import ProjectManagerPage

    page = ProjectManagerPage()
    page.note_title_edit.setText("Some title")
    page.note_body_edit.setPlainText("Some body")
    page._on_note_added(42)

    assert "42" in page.note_status.text()
    assert page.note_title_edit.text() == ""
    assert page.note_body_edit.toPlainText() == ""


def test_on_report_built_reports_compile_status(qapp):
    from mso_studio.gui.modules.project_manager import ProjectManagerPage

    page = ProjectManagerPage()
    result = pm_core.ReportOutcome(
        main_tex="/p/main.tex", main_pdf="/p/main.pdf",
        supplementary_tex="/p/supp.tex", supplementary_pdf="/p/supp.pdf",
        notebook_path="/p/nb.ipynb", manifest_path="/p/manifest.yaml",
        readme_path="/p/README.md", compiled=True,
    )
    page._on_report_built(result)
    assert "PDF compiled" in page.report_status.text()
    assert page.build_report_button.isEnabled()
