"""
mso_studio.launcher.process
==============================
``LaunchManager`` -- builds the argv for a catalog entry and tracks the
``Popen`` handles the launcher spawns, so the window can show a live
``RUNNING`` badge, refuse to spawn a second copy of something already
running, and offer a Stop button. Pure Python, no Qt (see ``__init__.py``).

Replaces the two launch code paths in ``launcher_gui.py``
(``LauncherApp._launch`` / ``._launch_mso_studio``) with one that is correct
for *this* launcher's interpreter: ``launcher_qt.py`` runs under
``.venv-gui``, which has PySide6 but not Cantera, so every ``RUNNER_SCIENCE``
entry must be launched with the anaconda ``SCIENCE_PYTHON`` interpreter (via
``mso_studio.core.backends``), never ``sys.executable`` -- using
``sys.executable`` here would try to run e.g. ``mechanism_check_gui.py``
under ``.venv-gui`` and fail on the first ``import cantera``.

Also fixes two real bugs in the old launcher along the way: no ``cwd`` was
passed (children broke on relative paths if the launcher was started from
elsewhere), and there was no de-duplication (double-clicking spawned two
copies of the same app).

Phase 4 (2026-08 GUI integration overhaul) adds pause/resume-with-new-code:
every app this class tracks can be SIGSTOPped, given a git-worktree branch
to edit code in, and either resumed unchanged (SIGCONT) or relaunched
against the edited code -- see ``mso_workbench.git_pause`` for the actual
git mechanics and the tradeoffs of process-level (not in-process) pausing.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from mso_studio.core.backends import SCIENCE_PYTHON, sanitized_subprocess_env
from mso_studio.launcher.catalog import (
    AppEntry,
    REPO_ROOT,
    RUNNER_GUI_VENV,
    RUNNER_INTERNAL,
    RUNNER_SCIENCE,
    script_path,
)
from mso_workbench import git_pause

GUI_VENV_PYTHON = REPO_ROOT / ".venv-gui" / "bin" / "python"

GUI_VENV_MISSING_MESSAGE = (
    "MSO Studio's dedicated virtual environment doesn't exist yet.\n\n"
    "Run this once from a terminal:\n"
    "    ./scripts/setup_gui_venv.sh"
)


class LaunchError(RuntimeError):
    """Raised by :meth:`LaunchManager.launch` when a prerequisite is missing
    or ``Popen`` itself fails. ``str(exc)`` is user-facing."""


def build_argv(entry: AppEntry) -> List[str]:
    """Build the argv that will run *entry*, without spawning anything."""
    if entry.runner == RUNNER_SCIENCE:
        return [SCIENCE_PYTHON, str(script_path(entry)), *entry.args]
    if entry.runner == RUNNER_GUI_VENV:
        return [str(GUI_VENV_PYTHON), str(script_path(entry)), *entry.args]
    raise ValueError(f"{entry.id}: runner {entry.runner!r} has no subprocess argv")


def missing_prerequisite(entry: AppEntry) -> Optional[str]:
    """Return a user-facing reason *entry* can't be launched right now, or
    ``None`` if it's ready."""
    if entry.runner == RUNNER_INTERNAL:
        return None
    sp = script_path(entry)
    if not sp.exists():
        return f"{entry.script} not found at {sp}"
    if entry.runner == RUNNER_GUI_VENV and not GUI_VENV_PYTHON.exists():
        return GUI_VENV_MISSING_MESSAGE
    return None


@dataclass
class _RunningApp:
    entry_id: str
    process: subprocess.Popen
    argv: List[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.monotonic)
    pause_session: Optional[git_pause.PauseSession] = None
    paused: bool = False


class PauseError(RuntimeError):
    """Raised by pause/resume operations; ``str(exc)`` is user-facing."""


class LaunchManager:
    """Tracks running subprocesses spawned for catalog entries."""

    def __init__(self) -> None:
        self._running: Dict[str, _RunningApp] = {}
        # Uptime at death for the most recently reaped copy of each entry --
        # a broken-but-present GUI_VENV_PYTHON (partial/corrupted
        # .venv-gui, e.g. PySide6 uninstalled after setup) isn't caught by
        # missing_prerequisite() (the interpreter file exists) or by
        # launch()'s Popen call (starting the process succeeds; only the
        # *child's own code* then fails). Popen gives no synchronous signal
        # for that -- but a process that exits within a couple seconds of
        # being launched is a strong sign it crashed on startup rather than
        # being closed normally, so callers (LauncherWindow's poll timer)
        # can surface that via last_uptime_s() after poll() reaps it.
        self._last_uptime: Dict[str, float] = {}

    # -- launching ---------------------------------------------------
    def launch(self, entry: AppEntry) -> int:
        """Start *entry* (or return the pid of the already-running copy).

        Raises :class:`LaunchError` if a prerequisite is missing or the
        subprocess fails to start.
        """
        if entry.runner == RUNNER_INTERNAL:
            raise LaunchError(f"{entry.title} has no subprocess to launch.")

        existing = self._running.get(entry.id)
        if existing is not None and existing.process.poll() is None:
            return existing.process.pid

        reason = missing_prerequisite(entry)
        if reason is not None:
            raise LaunchError(reason)

        argv = build_argv(entry)
        try:
            proc = subprocess.Popen(
                argv,
                cwd=str(REPO_ROOT),
                env=sanitized_subprocess_env(),
            )
        except OSError as exc:
            raise LaunchError(f"Could not launch {entry.title}:\n{exc}") from exc

        self._running[entry.id] = _RunningApp(entry_id=entry.id, process=proc, argv=argv)
        return proc.pid

    # -- status --------------------------------------------------------
    def poll(self) -> List[str]:
        """Reap any children that have exited; return the ids that just died."""
        finished = []
        for entry_id, running in list(self._running.items()):
            if running.process.poll() is not None:
                finished.append(entry_id)
                self._last_uptime[entry_id] = time.monotonic() - running.started_at
                del self._running[entry_id]
        return finished

    def last_uptime_s(self, entry_id: str) -> Optional[float]:
        """Uptime of the most recent reaped copy of *entry_id*, or ``None``
        if it's never been launched-and-reaped. A short value here (a few
        seconds) after ``poll()`` reports the id as finished is the best
        available signal that it crashed on startup rather than being
        closed normally -- see the note in ``__init__``."""
        return self._last_uptime.get(entry_id)

    def is_running(self, entry_id: str) -> bool:
        running = self._running.get(entry_id)
        return running is not None and running.process.poll() is None

    def running_ids(self) -> List[str]:
        return [eid for eid, r in self._running.items() if r.process.poll() is None]

    def pid(self, entry_id: str) -> Optional[int]:
        running = self._running.get(entry_id)
        return running.process.pid if running else None

    def uptime_s(self, entry_id: str) -> Optional[float]:
        running = self._running.get(entry_id)
        if running is None:
            return None
        return time.monotonic() - running.started_at

    # -- stopping --------------------------------------------------------
    def stop(self, entry_id: str, *, grace_s: float = 3.0) -> None:
        running = self._running.get(entry_id)
        if running is None:
            return
        proc = running.process
        if proc.poll() is not None:
            del self._running[entry_id]
            return
        proc.terminate()
        try:
            proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        del self._running[entry_id]

    def stop_all(self, *, grace_s: float = 3.0) -> None:
        for entry_id in list(self._running.keys()):
            self.stop(entry_id, grace_s=grace_s)

    # -- pause / resume-with-new-code (Phase 4) ---------------------------
    # Process-level, not in-process -- works uniformly for every entry this
    # class launches, with no cooperation required from the target script.
    # See mso_workbench.git_pause's module docstring for the full rationale
    # and the honest tradeoff (a code-change resume is a cold restart, not
    # a hot in-process reload).

    def is_paused(self, entry_id: str) -> bool:
        running = self._running.get(entry_id)
        return running is not None and running.paused

    def pause_session_for(self, entry_id: str) -> Optional[git_pause.PauseSession]:
        running = self._running.get(entry_id)
        return running.pause_session if running else None

    def pause_for_edit(self, entry_id: str) -> git_pause.PauseSession:
        """SIGSTOP the running process and check out a fresh git-worktree
        branch for editing. Raises :class:`PauseError` if nothing is
        running for *entry_id*, it's already paused, or the git-worktree
        creation itself fails (the process is left running in that case --
        never SIGSTOP before the branch/worktree exists)."""
        running = self._running.get(entry_id)
        if running is None or running.process.poll() is not None:
            raise PauseError(f"{entry_id} is not running.")
        if running.paused:
            raise PauseError(f"{entry_id} is already paused.")

        try:
            session = git_pause.create_pause_branch(str(REPO_ROOT), entry_id)
        except git_pause.GitPauseError as exc:
            raise PauseError(str(exc)) from exc

        git_pause.pause_process(running.process.pid)
        running.paused = True
        running.pause_session = session
        return session

    def resume_unchanged(self, entry_id: str) -> None:
        """SIGCONT -- the process continues with whatever code it already
        has loaded. The pause branch/worktree (if any) is left in place;
        callers can still commit_pause_results() against it later, or
        clean it up explicitly."""
        running = self._running.get(entry_id)
        if running is None or not running.paused:
            raise PauseError(f"{entry_id} is not paused.")
        git_pause.resume_process_unchanged(running.process.pid)
        running.paused = False

    def resume_with_new_code(self, entry_id: str) -> int:
        """Kills the stopped process and relaunches the same argv with
        PYTHONPATH pointed at the pause worktree's src/, so the new process
        picks up whatever was edited there. Raises :class:`PauseError` if
        *entry_id* isn't currently paused via :meth:`pause_for_edit`
        (there is no worktree to reload from otherwise).

        (Round-2 audit fix, B5) PYTHONPATH alone does not reach a
        subprocessed pipeline stage launched via a ``target.opt``-configured
        absolute path (the SA stage, see B1/B5 in
        WORKBENCH_AUDIT_2026-08-10.md) -- Python sets a subprocess's
        ``sys.path[0]`` to its own script's directory, which wins over
        PYTHONPATH for that script's sibling imports. If *argv* contains a
        literal ``.opt`` config file argument (the CLI-driven pipelines --
        e.g. ``run_script.py target.opt`` -- pass one; GUI-driven entries
        that pick a target.opt at runtime instead of via argv do not), it is
        rewritten via :func:`git_pause.rewrite_target_opt_for_worktree` and
        the rewritten copy's path is substituted before relaunching, so the
        SA stage's ``Bin:`` paths point into the worktree too."""
        running = self._running.get(entry_id)
        if running is None or not running.paused or running.pause_session is None:
            raise PauseError(f"{entry_id} is not paused with an active edit session.")

        session = running.pause_session
        running.process.kill()  # safe -- it's SIGSTOPped, not doing anything
        running.process.wait(timeout=5)

        argv = list(running.argv)
        for i, arg in enumerate(argv):
            if arg.endswith(".opt") and os.path.isfile(arg):
                argv[i] = git_pause.rewrite_target_opt_for_worktree(session, arg)

        env = git_pause.build_reload_env(session, sanitized_subprocess_env())
        try:
            proc = subprocess.Popen(argv, cwd=str(REPO_ROOT), env=env)
        except OSError as exc:
            raise LaunchError(f"Could not relaunch {entry_id} with the edited code:\n{exc}") from exc

        self._running[entry_id] = _RunningApp(
            entry_id=entry_id, process=proc, argv=argv, pause_session=session)
        return proc.pid

    def commit_pause_results(
        self, entry_id: str, result_paths: List[str], message: str,
    ) -> Optional[str]:
        """Copies *result_paths* into the entry's pause worktree and
        commits them onto its mso-pause/* branch -- "the results must also
        be stored in the project branch" per the user's own instruction.
        Returns the commit sha, or None if nothing existed to commit.
        Raises :class:`PauseError` if entry_id never had a pause session."""
        running = self._running.get(entry_id)
        if running is None or running.pause_session is None:
            raise PauseError(f"{entry_id} has no pause session to commit results to.")
        try:
            return git_pause.commit_results_to_branch(running.pause_session, result_paths, message)
        except git_pause.GitPauseError as exc:
            raise PauseError(str(exc)) from exc

    def discard_pause_worktree(self, entry_id: str) -> None:
        """Removes the worktree checkout (never the branch/its commits) and
        clears the tracked session -- for when the user is done editing and
        doesn't need the checkout directory anymore."""
        running = self._running.get(entry_id)
        if running is None or running.pause_session is None:
            return
        git_pause.remove_pause_worktree(running.pause_session)
        running.pause_session = None
