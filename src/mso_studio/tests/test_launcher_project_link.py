"""
Tests for mso_studio.launcher.project_link -- the launcher's sole,
firewalled connection to the (separately, concurrently developed, and as of
this writing entirely untracked) mso_project package. Every accessor must
degrade to a safe default rather than raise, and nothing here should ever
import mso_project at module scope -- these tests monkeypatch the import to
fail and confirm the launcher still behaves.
"""
import builtins
import importlib.util

import pytest

from mso_studio.launcher import project_link

_HAS_MSO_PROJECT = importlib.util.find_spec("mso_project") is not None


def test_is_available_never_raises():
    available, reason = project_link.is_available()
    assert isinstance(available, bool)
    assert isinstance(reason, str)


def test_is_project_dir_false_for_empty_directory(tmp_path):
    assert project_link.is_project_dir(str(tmp_path)) is False


def test_is_project_dir_true_when_manifest_present(tmp_path):
    (tmp_path / "project.yaml").write_text("name: test\n")
    assert project_link.is_project_dir(str(tmp_path)) is True


def test_is_project_dir_handles_nonexistent_path():
    assert project_link.is_project_dir("/no/such/path/at/all") is False


def test_read_manifest_parses_yaml(tmp_path):
    (tmp_path / "project.yaml").write_text("name: My Project\ndescription: hi\n")
    manifest = project_link.read_manifest(str(tmp_path))
    assert manifest == {"name": "My Project", "description": "hi"}


def test_read_manifest_returns_none_when_missing(tmp_path):
    assert project_link.read_manifest(str(tmp_path)) is None


def test_read_manifest_returns_none_on_malformed_yaml(tmp_path):
    (tmp_path / "project.yaml").write_text("{,")
    assert project_link.read_manifest(str(tmp_path)) is None


def test_read_manifest_does_not_create_artifacts_db(tmp_path):
    (tmp_path / "project.yaml").write_text("name: test\n")
    project_link.read_manifest(str(tmp_path))
    assert not (tmp_path / "artifacts.db").exists()


def test_list_runs_returns_empty_list_on_broken_project():
    class BrokenProject:
        def list_runs(self):
            raise RuntimeError("boom")

    assert project_link.list_runs(BrokenProject()) == []


def test_list_artifacts_returns_empty_list_on_broken_project():
    class BrokenProject:
        def list_artifacts(self, run_id=None):
            raise RuntimeError("boom")

    assert project_link.list_artifacts(BrokenProject()) == []


def test_artifact_count_zero_on_broken_project():
    class BrokenProject:
        def list_artifacts(self, run_id=None):
            raise RuntimeError("boom")

    assert project_link.artifact_count(BrokenProject()) == 0


def test_open_project_raises_project_link_error_when_import_fails(monkeypatch, tmp_path):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mso_project" or name.startswith("mso_project."):
            raise ImportError("simulated: mso_project mid-edit")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(project_link.ProjectLinkError):
        project_link.open_project(str(tmp_path))


def test_create_project_raises_project_link_error_when_import_fails(monkeypatch, tmp_path):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mso_project" or name.startswith("mso_project."):
            raise ImportError("simulated: mso_project mid-edit")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(project_link.ProjectLinkError):
        project_link.create_project(str(tmp_path), name="x")


@pytest.mark.skipif(not _HAS_MSO_PROJECT, reason="mso_project not importable")
def test_create_then_open_real_project_round_trips(tmp_path):
    directory = str(tmp_path / "proj")
    project = project_link.create_project(directory, name="Round Trip", description="d")
    assert project.info["name"] == "Round Trip"

    manifest = project_link.read_manifest(directory)
    assert manifest["name"] == "Round Trip"

    reopened = project_link.open_project(directory)
    assert reopened.info["name"] == "Round Trip"
    assert project_link.list_runs(reopened) == []
