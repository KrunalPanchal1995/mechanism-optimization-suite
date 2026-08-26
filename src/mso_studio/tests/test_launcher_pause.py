"""
Tests for LaunchManager's pause/resume-with-new-code (Phase 4). Uses real
subprocesses (SIGSTOP/SIGCONT/kill need a real pid) and a real throwaway
scratch git repo (tmp_path, monkeypatched in as REPO_ROOT) -- never touches
this suite's own repo, matching test_git_pause.py's convention.
"""
import subprocess
import sys
import time

import pytest

from mso_studio.launcher.process import LaunchManager, PauseError, _RunningApp
from mso_workbench import git_pause


def _init_scratch_repo(tmp_path):
    repo = tmp_path / "scratch_repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "src").mkdir()
    (repo / "src" / "module_a.py").write_text("VALUE = 'original'\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=repo, check=True)
    return repo


@pytest.fixture
def scratch_repo(tmp_path, monkeypatch):
    repo = _init_scratch_repo(tmp_path)
    monkeypatch.setattr("mso_studio.launcher.process.REPO_ROOT", repo)
    return repo


def _proc_state(pid: int) -> str:
    with open(f"/proc/{pid}/status") as fh:
        for line in fh:
            if line.startswith("State:"):
                return line
    return ""


def _spawn_sleeper() -> subprocess.Popen:
    return subprocess.Popen(["sleep", "30"])


def test_pause_for_edit_stops_process_and_creates_session(scratch_repo):
    manager = LaunchManager()
    proc = _spawn_sleeper()
    manager._running["mechanism_check"] = _RunningApp(
        entry_id="mechanism_check", process=proc, argv=["sleep", "30"])
    try:
        session = manager.pause_for_edit("mechanism_check")

        assert session.branch.startswith("mso-pause/mechanism_check/")
        time.sleep(0.2)
        assert "T" in _proc_state(proc.pid)  # SIGSTOPped
        assert manager.is_paused("mechanism_check")
        assert manager.pause_session_for("mechanism_check") is session
    finally:
        proc.kill()
        proc.wait()


def test_pause_for_edit_raises_if_not_running(scratch_repo):
    manager = LaunchManager()
    with pytest.raises(PauseError):
        manager.pause_for_edit("nope")


def test_pause_for_edit_raises_if_already_paused(scratch_repo):
    manager = LaunchManager()
    proc = _spawn_sleeper()
    manager._running["x"] = _RunningApp(entry_id="x", process=proc, argv=["sleep", "30"])
    try:
        manager.pause_for_edit("x")
        with pytest.raises(PauseError, match="already paused"):
            manager.pause_for_edit("x")
    finally:
        proc.kill()
        proc.wait()


def test_resume_unchanged_continues_same_process(scratch_repo):
    manager = LaunchManager()
    proc = _spawn_sleeper()
    manager._running["x"] = _RunningApp(entry_id="x", process=proc, argv=["sleep", "30"])
    try:
        manager.pause_for_edit("x")
        manager.resume_unchanged("x")
        time.sleep(0.2)

        assert not manager.is_paused("x")
        assert proc.pid == manager._running["x"].process.pid  # same process, not relaunched
        assert proc.poll() is None  # still alive
    finally:
        proc.kill()
        proc.wait()


def test_resume_unchanged_raises_if_not_paused(scratch_repo):
    manager = LaunchManager()
    proc = _spawn_sleeper()
    manager._running["x"] = _RunningApp(entry_id="x", process=proc, argv=["sleep", "30"])
    try:
        with pytest.raises(PauseError, match="not paused"):
            manager.resume_unchanged("x")
    finally:
        proc.kill()
        proc.wait()


def test_resume_with_new_code_relaunches_using_edited_worktree_module(scratch_repo, tmp_path):
    """The core Phase-4 promise: pause, edit code in the worktree, resume,
    and the NEW process actually picks up the edit -- via PYTHONPATH, not
    an in-process hot-reload (see git_pause's module docstring)."""
    manager = LaunchManager()
    proc = _spawn_sleeper()
    marker = tmp_path / "marker.txt"
    driver_argv = [sys.executable, "-c",
                   f"import module_a; open({str(marker)!r}, 'w').write(module_a.VALUE)"]
    manager._running["x"] = _RunningApp(entry_id="x", process=proc, argv=driver_argv)

    session = manager.pause_for_edit("x")
    original_pid = proc.pid

    module_path = session.worktree_path + "/src/module_a.py"
    with open(module_path, "w") as fh:
        fh.write("VALUE = 'edited'\n")

    new_pid = manager.resume_with_new_code("x")
    assert new_pid != original_pid
    assert not manager.is_paused("x")

    new_proc = manager._running["x"].process
    new_proc.wait(timeout=10)
    assert marker.read_text() == "edited"


def test_resume_with_new_code_raises_without_pause_session(scratch_repo):
    manager = LaunchManager()
    proc = _spawn_sleeper()
    manager._running["x"] = _RunningApp(entry_id="x", process=proc, argv=["sleep", "30"])
    try:
        with pytest.raises(PauseError):
            manager.resume_with_new_code("x")
    finally:
        proc.kill()
        proc.wait()


def test_commit_pause_results_stores_files_on_branch(scratch_repo, tmp_path):
    manager = LaunchManager()
    proc = _spawn_sleeper()
    manager._running["x"] = _RunningApp(entry_id="x", process=proc, argv=["sleep", "30"])
    try:
        manager.pause_for_edit("x")
        result_file = tmp_path / "output.csv"
        result_file.write_text("a,b\n1,2\n")

        sha = manager.commit_pause_results("x", [str(result_file)], "store results")
        assert sha is not None
    finally:
        proc.kill()
        proc.wait()


def test_commit_pause_results_raises_without_session(scratch_repo):
    manager = LaunchManager()
    proc = _spawn_sleeper()
    manager._running["x"] = _RunningApp(entry_id="x", process=proc, argv=["sleep", "30"])
    try:
        with pytest.raises(PauseError):
            manager.commit_pause_results("x", ["/nonexistent"], "message")
    finally:
        proc.kill()
        proc.wait()


def test_discard_pause_worktree_removes_checkout(scratch_repo):
    manager = LaunchManager()
    proc = _spawn_sleeper()
    manager._running["x"] = _RunningApp(entry_id="x", process=proc, argv=["sleep", "30"])
    try:
        session = manager.pause_for_edit("x")
        assert git_pause.list_pause_sessions(str(scratch_repo)) != []

        manager.discard_pause_worktree("x")

        assert manager.pause_session_for("x") is None
        import os
        assert not os.path.isdir(session.worktree_path)
    finally:
        manager.resume_unchanged("x") if manager.is_paused("x") else None
        proc.kill()
        proc.wait()
