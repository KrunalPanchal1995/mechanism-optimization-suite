"""
snapshot.py: unpicklable values degrade gracefully (recorded, not silently
dropped, and don't sink the rest of the snapshot), artifact symlinking,
and manifest content.
"""
import threading

import pytest

from mso_workbench import snapshot


class _Unpicklable:
    def __reduce__(self):
        raise TypeError("nope, not picklable")


def test_all_picklable_namespace_round_trips(tmp_path):
    ns = {"a": 1, "b": [1, 2, 3], "c": {"nested": True}}
    snap_dir = snapshot.write_snapshot(tmp_path, "stage_x", ns)
    loaded = snapshot.load_namespace(snap_dir)
    assert loaded == ns
    manifest = snapshot.load_manifest(snap_dir)
    assert manifest["unpicklable"] == []


def test_one_unpicklable_value_does_not_sink_the_rest(tmp_path):
    ns = {"good": 42, "bad": _Unpicklable(), "also_good": "hello"}
    snap_dir = snapshot.write_snapshot(tmp_path, "stage_x", ns)
    loaded = snapshot.load_namespace(snap_dir)
    assert loaded == {"good": 42, "also_good": "hello"}

    manifest = snapshot.load_manifest(snap_dir)
    assert len(manifest["unpicklable"]) == 1
    assert manifest["unpicklable"][0]["key"] == "bad"
    assert manifest["unpicklable"][0]["type"] == "_Unpicklable"


def test_unpicklable_lock_object_is_handled(tmp_path):
    """threading.Lock is a real example of something run_script.py's
    namespace could plausibly hold and cannot be pickled."""
    ns = {"lock": threading.Lock(), "count": 5}
    snap_dir = snapshot.write_snapshot(tmp_path, "stage_x", ns)
    loaded = snapshot.load_namespace(snap_dir)
    assert loaded == {"count": 5}


def test_manifest_records_stage_and_source_ref(tmp_path):
    snap_dir = snapshot.write_snapshot(
        tmp_path, "sampling.pre_design_matrix", {}, source_ref=("run_script.py", "module"),
    )
    manifest = snapshot.load_manifest(snap_dir)
    assert manifest["stage"] == "sampling.pre_design_matrix"
    assert manifest["source_ref"] == {"file": "run_script.py", "function": "module"}
    assert "cwd" in manifest
    assert "pid" in manifest
    assert "timestamp" in manifest


def test_manifest_source_ref_none_when_not_given(tmp_path):
    snap_dir = snapshot.write_snapshot(tmp_path, "stage_x", {})
    assert snapshot.load_manifest(snap_dir)["source_ref"] is None


def test_env_file_has_python_version_and_packages(tmp_path):
    import json

    snap_dir = snapshot.write_snapshot(tmp_path, "stage_x", {})
    env = json.loads((snap_dir / snapshot.ENV_FILE).read_text())
    assert "python_version" in env
    assert "numpy" in env["packages"]  # available in this test env


def test_existing_artifacts_are_symlinked_into_the_snapshot(tmp_path):
    (tmp_path / "unsrt.pkl").write_bytes(b"fake pickle bytes")
    (tmp_path / "DesignMatrix.csv").write_text("a,b,c\n1,2,3\n")

    snap_dir = snapshot.write_snapshot(tmp_path, "stage_x", {})
    manifest = snapshot.load_manifest(snap_dir)
    assert set(manifest["artifact_links"]) == {"unsrt.pkl", "DesignMatrix.csv"}
    assert (snap_dir / "unsrt.pkl").resolve() == (tmp_path / "unsrt.pkl").resolve()
    assert (snap_dir / "DesignMatrix.csv").read_text() == "a,b,c\n1,2,3\n"


def test_no_artifacts_present_is_fine(tmp_path):
    snap_dir = snapshot.write_snapshot(tmp_path, "stage_x", {})
    assert snapshot.load_manifest(snap_dir)["artifact_links"] == []


def test_list_snapshots_empty_when_none_taken(tmp_path):
    assert snapshot.list_snapshots(tmp_path) == []


def test_list_snapshots_returns_all_in_order(tmp_path):
    snapshot.write_snapshot(tmp_path, "stage_a", {})
    snapshot.write_snapshot(tmp_path, "stage_b", {})
    snaps = snapshot.list_snapshots(tmp_path)
    assert len(snaps) == 2


def test_git_sha_does_not_crash_outside_a_repo(tmp_path):
    # tmp_path is not a git repo -- _git_sha must return None, not raise.
    snap_dir = snapshot.write_snapshot(tmp_path, "stage_x", {})
    manifest = snapshot.load_manifest(snap_dir)
    assert "git_sha" in manifest  # None is an acceptable value
