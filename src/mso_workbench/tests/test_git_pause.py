"""
Tests for mso_workbench.git_pause -- real git operations against a
throwaway scratch repo in tmp_path (never touches this suite's own repo),
matching the "run it for real" convention the rest of this codebase's test
suite uses for subprocess-backed adapters.
"""
import os
import subprocess
import time

import pytest

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


def test_create_pause_branch_checks_out_a_second_working_directory(tmp_path):
    repo = _init_scratch_repo(tmp_path)
    session = git_pause.create_pause_branch(str(repo), "mechanism_check")

    assert session.branch.startswith("mso-pause/mechanism_check/")
    assert os.path.isdir(session.worktree_path)
    assert (repo / ".git" / "worktrees").exists()
    # the worktree has its own full checkout, independent of the main tree
    worktree_module = os.path.join(session.worktree_path, "src", "module_a.py")
    assert os.path.isfile(worktree_module)
    with open(worktree_module) as fh:
        assert "original" in fh.read()


def test_create_pause_branch_does_not_touch_main_working_tree(tmp_path):
    repo = _init_scratch_repo(tmp_path)
    session = git_pause.create_pause_branch(str(repo), "nominal_sim")

    # editing the worktree's copy must not affect the main checkout
    with open(os.path.join(session.worktree_path, "src", "module_a.py"), "w") as fh:
        fh.write("VALUE = 'edited'\n")

    with open(repo / "src" / "module_a.py") as fh:
        assert "original" in fh.read()

    # current branch of the main repo is unaffected
    result = subprocess.run(
        ["git", "branch", "--show-current"], cwd=repo, capture_output=True, text=True, check=True)
    assert result.stdout.strip() != session.branch


def test_two_pause_sessions_get_distinct_branches_and_worktrees(tmp_path):
    repo = _init_scratch_repo(tmp_path)
    session1 = git_pause.create_pause_branch(str(repo), "mechanism_check")
    time.sleep(1.1)  # branch/dir names are timestamp-suffixed at 1s resolution
    session2 = git_pause.create_pause_branch(str(repo), "mechanism_check")

    assert session1.branch != session2.branch
    assert session1.worktree_path != session2.worktree_path
    assert os.path.isdir(session1.worktree_path)
    assert os.path.isdir(session2.worktree_path)


def test_remove_pause_worktree_removes_checkout_but_keeps_branch(tmp_path):
    repo = _init_scratch_repo(tmp_path)
    session = git_pause.create_pause_branch(str(repo), "forward_uq")
    assert os.path.isdir(session.worktree_path)

    git_pause.remove_pause_worktree(session)

    assert not os.path.isdir(session.worktree_path)
    branches = subprocess.run(
        ["git", "branch", "--list", session.branch], cwd=repo,
        capture_output=True, text=True, check=True).stdout
    assert session.branch in branches


def test_list_pause_sessions_reports_only_pause_worktrees(tmp_path):
    repo = _init_scratch_repo(tmp_path)
    assert git_pause.list_pause_sessions(str(repo)) == []

    session = git_pause.create_pause_branch(str(repo), "rmip_suite")
    sessions = git_pause.list_pause_sessions(str(repo))
    assert len(sessions) == 1
    assert sessions[0]["worktree"] == session.worktree_path
    assert session.branch in sessions[0]["branch"]


def test_build_reload_env_prepends_worktree_src_to_pythonpath(tmp_path):
    repo = _init_scratch_repo(tmp_path)
    session = git_pause.create_pause_branch(str(repo), "sensitivity")

    env = git_pause.build_reload_env(session, {"PYTHONPATH": "/other/path", "FOO": "bar"})
    assert env["PYTHONPATH"].startswith(os.path.join(session.worktree_path, "src"))
    assert env["PYTHONPATH"].endswith("/other/path")
    assert env["FOO"] == "bar"

    env_no_existing = git_pause.build_reload_env(session, {})
    assert env_no_existing["PYTHONPATH"] == os.path.join(session.worktree_path, "src")


def test_commit_results_to_branch_copies_and_commits(tmp_path):
    repo = _init_scratch_repo(tmp_path)
    session = git_pause.create_pause_branch(str(repo), "mso_compare")

    result_file = tmp_path / "run_output" / "comparison.csv"
    result_file.parent.mkdir()
    result_file.write_text("a,b\n1,2\n")

    sha = git_pause.commit_results_to_branch(session, [str(result_file)], "store comparison results")
    assert sha is not None

    log = subprocess.run(
        ["git", "log", "--oneline", "-1"], cwd=session.worktree_path,
        capture_output=True, text=True, check=True).stdout
    assert "store comparison results" in log

    dest = os.path.join(session.worktree_path, ".mso_pause_results", "mso_compare", "comparison.csv")
    assert os.path.isfile(dest)
    with open(dest) as fh:
        assert fh.read() == "a,b\n1,2\n"


def test_commit_results_to_branch_returns_none_when_nothing_exists(tmp_path):
    repo = _init_scratch_repo(tmp_path)
    session = git_pause.create_pause_branch(str(repo), "uncertainty")

    sha = git_pause.commit_results_to_branch(session, ["/nonexistent/path"], "nothing to commit")
    assert sha is None


def test_pause_and_resume_process_signals(tmp_path):
    proc = subprocess.Popen(["sleep", "5"])
    try:
        git_pause.pause_process(proc.pid)
        time.sleep(0.2)
        # a SIGSTOPped process's status shows as stopped in /proc
        with open(f"/proc/{proc.pid}/status") as fh:
            status_text = fh.read()
        assert "State:\tT " in status_text or "State:\tT" in status_text

        git_pause.resume_process_unchanged(proc.pid)
        time.sleep(0.2)
        with open(f"/proc/{proc.pid}/status") as fh:
            status_text = fh.read()
        assert "State:\tS" in status_text or "State:\tR" in status_text
    finally:
        proc.kill()
        proc.wait()


def test_create_pause_branch_raises_git_pause_error_for_bad_repo(tmp_path):
    not_a_repo = tmp_path / "not_a_repo"
    not_a_repo.mkdir()
    with pytest.raises(git_pause.GitPauseError):
        git_pause.create_pause_branch(str(not_a_repo), "whatever")


class TestRewriteTargetOptForWorktree:
    """Round-2 audit fix, B5: a target.opt's Bin: SA_tool/SA_tool_3p/
    yaml_writer entries are literal absolute paths into repo_root, which
    PYTHONPATH-prepending (build_reload_env) never reaches -- see B1/B5 in
    WORKBENCH_AUDIT_2026-08-10.md."""

    def _target_opt(self, repo, **bin_overrides):
        bin_section = {
            "bin": "/home/krithika/MUQ-SAC/V3.0",
            "solver_bin": "/home/krithika/FlameMaster/Bin/bin",
            "yaml_writer": str(repo / "src" / "MUQ-SAC" / "parallel_yaml_writer.so"),
            "SA_tool": str(repo / "src" / "Sensitivity_Analysis" / "sens.py"),
            "SA_tool_3p": str(repo / "src" / "Sensitivity_Analysis" / "sens_3_params.py"),
        }
        bin_section.update(bin_overrides)
        path = repo / "target.opt"
        import yaml
        path.write_text(yaml.safe_dump({"Bin": bin_section, "project": {"name": "test"}}))
        return str(path)

    def test_repo_internal_bin_paths_are_repointed_into_the_worktree(self, tmp_path):
        repo = _init_scratch_repo(tmp_path)
        session = git_pause.create_pause_branch(str(repo), "sa_stage")
        target_opt = self._target_opt(repo)

        new_path = git_pause.rewrite_target_opt_for_worktree(session, target_opt)

        assert new_path != target_opt
        import yaml
        with open(new_path) as fh:
            rewritten = yaml.safe_load(fh)
        assert rewritten["Bin"]["SA_tool"] == os.path.join(
            session.worktree_path, "src", "Sensitivity_Analysis", "sens.py")
        assert rewritten["Bin"]["SA_tool_3p"] == os.path.join(
            session.worktree_path, "src", "Sensitivity_Analysis", "sens_3_params.py")
        assert rewritten["Bin"]["yaml_writer"] == os.path.join(
            session.worktree_path, "src", "MUQ-SAC", "parallel_yaml_writer.so")
        # External (non-repo) tool paths are left untouched.
        assert rewritten["Bin"]["bin"] == "/home/krithika/MUQ-SAC/V3.0"
        assert rewritten["Bin"]["solver_bin"] == "/home/krithika/FlameMaster/Bin/bin"
        # The original file on disk is never modified.
        with open(target_opt) as fh:
            original = yaml.safe_load(fh)
        assert original["Bin"]["SA_tool"] == str(repo / "src" / "Sensitivity_Analysis" / "sens.py")

    def test_already_external_bin_paths_are_left_alone(self, tmp_path):
        repo = _init_scratch_repo(tmp_path)
        session = git_pause.create_pause_branch(str(repo), "sa_stage")
        target_opt = self._target_opt(
            repo,
            SA_tool="/some/other/machine/sens.py",
            SA_tool_3p="/some/other/machine/sens_3_params.py",
            yaml_writer="/some/other/machine/parallel_yaml_writer.so",
        )

        result = git_pause.rewrite_target_opt_for_worktree(session, target_opt)

        assert result == target_opt  # no rewrite needed -- returned unchanged
