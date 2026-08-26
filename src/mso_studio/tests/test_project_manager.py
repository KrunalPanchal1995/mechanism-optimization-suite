"""
Tests for mso_studio.core.project_manager -- real mso_project.Project
objects in tmp_path (pure stdlib + PyYAML, safe in-process in the GUI venv,
confirmed importable -- no subprocess bridge, unlike core.target_workbench).
"""
import pytest

from mso_studio.core import project_manager as pm_core


def test_create_and_open_project(tmp_path):
    project_dir = str(tmp_path / "proj1")
    info = pm_core.create_project(project_dir, "Ammonia campaign", "desc", ["ammonia"])
    assert info.name == "Ammonia campaign"
    assert info.tags == ["ammonia"]

    reopened = pm_core.open_project(project_dir)
    assert reopened.name == "Ammonia campaign"
    assert reopened.root == info.root


def test_list_runs_empty_for_fresh_project(tmp_path):
    project_dir = str(tmp_path / "proj2")
    pm_core.create_project(project_dir, "Fresh")
    assert pm_core.list_runs(project_dir) == []


def test_add_note_and_list_artifacts(tmp_path):
    project_dir = str(tmp_path / "proj3")
    pm_core.create_project(project_dir, "Notes test")
    artifact_id = pm_core.add_note(project_dir, "Decision log", "Chose CMA-ES over GA.")
    assert artifact_id > 0

    artifacts = pm_core.list_artifacts(project_dir)
    assert len(artifacts) == 1
    assert artifacts[0]["role"] == "user_note"
    assert artifacts[0]["kind"] == "doc"


def test_list_artifacts_filters_by_role(tmp_path):
    project_dir = str(tmp_path / "proj4")
    pm_core.create_project(project_dir, "Filter test")
    pm_core.add_note(project_dir, "Note A", "body a")
    pm_core.add_note(project_dir, "Note B", "body b")

    all_artifacts = pm_core.list_artifacts(project_dir)
    assert len(all_artifacts) == 2
    only_notes = pm_core.list_artifacts(project_dir, role="user_note")
    assert len(only_notes) == 2
    none_matching = pm_core.list_artifacts(project_dir, role="nonexistent_role")
    assert none_matching == []


def test_resume_materializes_registered_artifact(tmp_path):
    from pathlib import Path

    from mso_project import Project

    project_dir = str(tmp_path / "proj5")
    project = Project.create(project_dir, "Resume test")
    working_dir = tmp_path / "work"
    working_dir.mkdir()
    with project.capture("test_pipeline", "case1", str(working_dir)) as run_id:
        (working_dir / "result.csv").write_text("a,b\n1,2\n")
    project.close()

    runs = pm_core.list_runs(project_dir)
    assert len(runs) == 1
    assert runs[0]["run_id"] == run_id

    target_dir = str(tmp_path / "resumed")
    materialized = pm_core.resume_run(project_dir, run_id, target_dir)
    assert len(materialized) == 1
    materialized_path = list(materialized.values())[0]
    assert Path(materialized_path).exists()
    assert Path(materialized_path).read_text() == "a,b\n1,2\n"


def test_open_nonexistent_project_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        pm_core.open_project(str(tmp_path / "nope"))
