"""
mso_workbench.git_pause
==========================
Phase 4 of the 2026-08 GUI integration overhaul: "pause a run, edit the
source, resume with the change" -- implemented at the git level, per the
user's explicit instruction to use a branch (never the main working tree)
rather than merging into master, with results committed onto that branch.

Two pause flavors, because "every science-tool run" (the user's own scope
answer) includes dozens of pre-existing CLI scripts that were never
instrumented with :mod:`mso_workbench.breakpoint` and never will be
individually -- this module works uniformly at the OS-process level so it
needs zero cooperation from the target script:

* **Pause / resume-unchanged** -- ``SIGSTOP``/``SIGCONT`` on the process.
  Works for literally any subprocess; the process's own code never changes.
* **Resume with a code change** -- there is no safe way to hot-patch
  arbitrary already-running foreign-process bytecode from outside (that's
  what ptrace-based injection tools like gdb are for, and they are not
  appropriate here). Instead: create a git branch + a real ``git worktree``
  checkout (a second, independent working directory -- never touches the
  main checkout, so nothing else running against this repo is disturbed),
  let the user edit code there, then kill the stopped process and relaunch
  the same argv with ``PYTHONPATH`` pointed at the worktree's ``src/`` --
  a cold restart using the edited code, not an in-place hot-reload. Honest
  about the tradeoff: in-process state from before the pause is not
  preserved unless the pipeline itself checkpoints to disk (many of this
  suite's do, via RunRecord/mso_project's capture()).

The one pipeline already instrumented with ``mso_workbench.breakpoint``
(``src/MUQ-SAC/run_script.py``) gets a strictly richer THIRD option on top
of these two: true in-process hot-reload at its next stage boundary,
implemented separately in :mod:`mso_workbench.breakpoint`'s resume path
(swaps ``sys.modules`` entries under the paused process's own interpreter,
no restart needed) -- not duplicated here.
"""
from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

#: Where pause worktrees live, relative to the repo root -- inside the repo
#: (matching .venv-gui/, outputs/ etc.'s existing convention of living
#: in-tree but gitignored) rather than an unfamiliar external location.
WORKTREES_DIRNAME = ".mso_pause_worktrees"

BRANCH_PREFIX = "mso-pause"


class GitPauseError(RuntimeError):
    """Raised for any git-worktree operation failure; message is user-facing."""


@dataclass
class PauseSession:
    label: str                 # sanitized identifier, e.g. the launcher entry id
    branch: str                 # e.g. "mso-pause/mechanism_check/20260806-142233"
    worktree_path: str           # absolute path to the second working directory
    repo_root: str                # the main repo's root (source of the branch)
    created_at: float


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")
    return slug or "run"


def _run_git(args: Sequence[str], cwd: str, timeout: float = 30.0) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GitPauseError(f"git {' '.join(args)} failed to start: {exc}") from exc
    if result.returncode != 0:
        raise GitPauseError(
            f"git {' '.join(args)} failed (exit {result.returncode}):\n{result.stderr.strip()}")
    return result


def create_pause_branch(repo_root: str, label: str) -> PauseSession:
    """Create ``mso-pause/<label>/<timestamp>`` off the repo's current HEAD
    and check it out into a fresh ``git worktree`` -- never touches the
    caller's own working tree. Raises :class:`GitPauseError` if git itself
    fails (dirty submodule state, an existing worktree at the target path,
    etc.) -- the caller must not treat a partially-created worktree as a
    success."""
    slug = _slugify(label)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    branch = f"{BRANCH_PREFIX}/{slug}/{timestamp}"
    worktrees_root = Path(repo_root) / WORKTREES_DIRNAME
    worktrees_root.mkdir(parents=True, exist_ok=True)
    worktree_path = worktrees_root / f"{slug}_{timestamp}"

    _run_git(["worktree", "add", "-b", branch, str(worktree_path), "HEAD"], cwd=repo_root)

    return PauseSession(
        label=label, branch=branch, worktree_path=str(worktree_path),
        repo_root=str(Path(repo_root).resolve()), created_at=time.time(),
    )


def remove_pause_worktree(session: PauseSession, *, force: bool = True) -> None:
    """Removes the worktree CHECKOUT only -- the branch itself, and any
    commits made on it, are never deleted. Safe to call even if the
    directory was already removed by hand (git prunes it either way)."""
    args = ["worktree", "remove"]
    if force:
        args.append("--force")
    args.append(session.worktree_path)
    try:
        _run_git(args, cwd=session.repo_root)
    except GitPauseError:
        # Directory already gone / never fully created -- prune the stale
        # metadata rather than leaving `git worktree list` referencing it.
        _run_git(["worktree", "prune"], cwd=session.repo_root)


def list_pause_sessions(repo_root: str) -> List[dict]:
    """Every currently-registered worktree under WORKTREES_DIRNAME, as
    ``git worktree list --porcelain`` reports it -- for a GUI to show what
    pause sessions still have a live checkout."""
    result = _run_git(["worktree", "list", "--porcelain"], cwd=repo_root)
    sessions = []
    current: dict = {}
    prefix = str(Path(repo_root) / WORKTREES_DIRNAME)
    for line in result.stdout.splitlines():
        if line.startswith("worktree "):
            if current.get("worktree", "").startswith(prefix):
                sessions.append(current)
            current = {"worktree": line[len("worktree "):]}
        elif line.startswith("branch "):
            current["branch"] = line[len("branch "):]
        elif line == "":
            continue
    if current.get("worktree", "").startswith(prefix):
        sessions.append(current)
    return sessions


def pause_process(pid: int) -> None:
    """SIGSTOP -- works for any process, no cooperation required."""
    os.kill(pid, signal.SIGSTOP)


def resume_process_unchanged(pid: int) -> None:
    """SIGCONT -- the process continues running whatever code it already
    has loaded in memory."""
    os.kill(pid, signal.SIGCONT)


def build_reload_env(session: PauseSession, base_env: dict) -> dict:
    """Returns *base_env* with the pause worktree's ``src/`` prepended to
    ``PYTHONPATH``, so a freshly-launched process picks up whatever was
    edited there ahead of the main checkout's copy."""
    env = dict(base_env)
    worktree_src = str(Path(session.worktree_path) / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = worktree_src + (os.pathsep + existing if existing else "")
    return env


def rewrite_target_opt_for_worktree(session: PauseSession, target_opt_path: str) -> str:
    """(Round-2 audit fix, B5) A ``target.opt``'s ``Bin:`` section stores
    ``SA_tool``/``SA_tool_3p``/``yaml_writer`` as literal absolute paths
    (see B1). ``resume_with_new_code``'s ``PYTHONPATH``-prepend trick never
    reaches those: Python sets a subprocess's ``sys.path[0]`` to its own
    invoked script's directory, which wins over ``PYTHONPATH`` for that
    script's bare-name sibling imports -- so a subprocess launched with
    ``sens.py``'s *original* absolute path keeps loading sibling modules
    from the main checkout even when the edited copy lives in the pause
    worktree. This is B1's fix's counterpart at resume time: given a
    ``target.opt`` whose ``Bin:`` entries point somewhere under
    *session.repo_root*, write a working copy with those three entries
    repointed at the equivalent path under *session.worktree_path*, and
    return the new file's path. The caller is responsible for substituting
    that path into the argv actually used to relaunch (this function has no
    side effect on the original file, and does not know how the config path
    reaches the target process).

    Entries that do NOT point under *repo_root* (``bin``, ``solver_bin`` --
    genuinely external installs, not part of this repo) are left untouched.

    Returns *target_opt_path* unchanged (no copy made) if nothing needed
    rewriting -- e.g. a target.opt that was already regenerated with B1's
    fix and happens to already match the worktree, or one with no Bin
    section at all.

    Note: the working copy is written via ``yaml.safe_load``/``safe_dump``,
    which reformats the file and drops comments -- acceptable for a
    transient, resume-time-only artifact (never the checked-in source
    file), but not a general-purpose "edit in place" tool.
    """
    import yaml

    with open(target_opt_path) as fh:
        config = yaml.safe_load(fh)

    bin_section = (config or {}).get("Bin") or {}
    repo_root = str(Path(session.repo_root).resolve())
    worktree_root = str(Path(session.worktree_path).resolve())

    changed = False
    for key in ("SA_tool", "SA_tool_3p", "yaml_writer"):
        value = bin_section.get(key)
        if not value:
            continue
        try:
            resolved = str(Path(value).resolve())
        except OSError:
            continue
        if resolved == repo_root or resolved.startswith(repo_root + os.sep):
            rel = os.path.relpath(resolved, repo_root)
            bin_section[key] = str(Path(worktree_root) / rel)
            changed = True

    if not changed:
        return target_opt_path

    dest_dir = Path(session.worktree_path) / ".mso_pause_configs"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / Path(target_opt_path).name
    with open(dest_path, "w") as fh:
        yaml.safe_dump(config, fh, sort_keys=False, default_flow_style=False)
    return str(dest_path)


def commit_results_to_branch(
    session: PauseSession, result_paths: Sequence[str], message: str,
) -> Optional[str]:
    """Copies each path in *result_paths* (files or directories) into
    ``<worktree>/.mso_pause_results/<label>/`` and commits them on the
    pause branch -- "the results must also be stored in the project
    branch" per the user's own instruction. Returns the new commit sha, or
    ``None`` if there was nothing to commit (e.g. every path was missing).
    Never touches the main working tree or any other branch."""
    dest_root = Path(session.worktree_path) / ".mso_pause_results" / _slugify(session.label)
    dest_root.mkdir(parents=True, exist_ok=True)

    copied = []
    for raw_path in result_paths:
        src = Path(raw_path)
        if not src.exists():
            continue
        dest = dest_root / src.name
        if src.is_dir():
            shutil.copytree(src, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dest)
        copied.append(str(dest))

    if not copied:
        return None

    _run_git(["add", "-A", str(dest_root)], cwd=session.worktree_path)
    _run_git(["commit", "-m", message], cwd=session.worktree_path)
    sha = _run_git(["rev-parse", "HEAD"], cwd=session.worktree_path).stdout.strip()
    return sha
