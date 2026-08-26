"""
rmip/tests/test_pty_terminal.py
================================
Tests for gui_common/pty_terminal.py's PtyTerminal widget covering the four
bugs fixed there:

  1. fd + zombie leak in stop() (SIGTERM -> bounded waitpid poll -> SIGKILL
     escalation -> final blocking waitpid; always close self._fd).
  2. reader-thread restart race, fixed via a monotonically-increasing
     generation counter (self._gen) passed explicitly into _read_loop.
  3. scrollback (pyte.HistoryScreen) -- covered by an availability/
     construction check only; verifying exact rendered scrollback content
     pixel-for-pixel isn't practical headlessly.
  4. window-close guard: <Destroy> on the frame itself calls stop().

Skips cleanly:
  - on non-POSIX platforms (no `pty` module).
  - if a Tk root truly cannot be constructed in this environment (no
    display, no Xvfb).

Every Tk window created here is immediately withdraw()n so nothing pops up
visibly on a (possibly shared) X display -- see bug.md / GUI_STRESS_TEST_PLAN.md
re: the :52.0 shared-display click hazard from 2026-07-17. None of these
tests send synthetic mouse/keyboard input at the OS level (no xdotool/
event_generate-as-a-real-click); they only call Python methods directly, so
they don't carry that risk.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("pty")
tk = pytest.importorskip("tkinter")
pytest.importorskip("pyte")

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))


@pytest.fixture(scope="module")
def tk_root():
    try:
        root = tk.Tk()
        root.withdraw()
    except tk.TclError as exc:
        pytest.skip(f"Tk cannot be initialized in this environment: {exc}")
    yield root
    try:
        root.destroy()
    except tk.TclError:
        pass


def _pump(root, seconds):
    """Drive the Tk event loop for a bit so PtyTerminal's after()-based
    refresh timer and the pty reader thread both get a chance to run,
    without calling mainloop() (which would block)."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        root.update()
        time.sleep(0.02)


def _wait_until(root, predicate, timeout=3.0, interval=0.02):
    deadline = time.time() + timeout
    ok = False
    while time.time() < deadline:
        root.update()
        if predicate():
            ok = True
            break
        time.sleep(interval)
    return ok or predicate()


def test_stop_reaps_child_and_clears_fd(tk_root):
    """Bug 1: stop() must reap the child (no zombie) and clear self._fd."""
    from gui_common.pty_terminal import PtyTerminal

    term = PtyTerminal(tk_root)
    term.run(["/bin/echo", "hello"])
    _pump(tk_root, 0.5)
    pid = term._pid
    assert pid is not None

    term.stop()

    assert term._fd is None
    assert term._pid is None
    # The child must already be reaped: a second waitpid on the same pid
    # must fail because the OS no longer considers it our child.
    with pytest.raises(ChildProcessError):
        os.waitpid(pid, os.WNOHANG)

    term.destroy()
    tk_root.update()


def test_run_twice_no_cross_session_contamination(tk_root):
    """Bug 2: a slow second run() must not have its rendered output
    corrupted by stray bytes fed in by the first run()'s reader thread
    still draining the old (already-superseded) fd."""
    from gui_common.pty_terminal import PtyTerminal

    term = PtyTerminal(tk_root)
    term.run(["/bin/sh", "-c", "echo first-session-marker"])
    _pump(tk_root, 0.4)

    term.run(["/bin/sh", "-c", "sleep 0.3; echo second-session-marker"])
    got_marker = _wait_until(
        tk_root,
        lambda: "second-session-marker" in "\n".join(term._screen.display),
        timeout=3.0,
    )
    assert got_marker, "second session's output never appeared"

    rendered = "\n".join(term._screen.display)
    assert "second-session-marker" in rendered
    assert "first-session-marker" not in rendered

    term.stop()
    term.destroy()
    tk_root.update()


def test_destroy_toplevel_reaps_child(tk_root):
    """Bug 4: destroying the widget's window must terminate + reap the
    child promptly instead of leaking it."""
    from gui_common.pty_terminal import PtyTerminal

    top = tk.Toplevel(tk_root)
    top.withdraw()
    term = PtyTerminal(top)
    term.run(["/bin/sleep", "5"])
    _pump(tk_root, 0.3)
    pid = term._pid
    assert pid is not None

    top.destroy()
    tk_root.update()

    deadline = time.time() + 2.0
    reaped = False
    while time.time() < deadline:
        try:
            wpid, _status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            reaped = True
            break
        if wpid == pid:
            reaped = True
            break
        time.sleep(0.05)
    assert reaped, "child process was not reaped promptly after Toplevel.destroy()"


def test_history_screen_scrollback_available(tk_root):
    """Bug 3 (construction-level check): run() must build a HistoryScreen
    with scrollback, and the prev_page/next_page scroll bindings must be
    wired up and callable without error."""
    import pyte
    from gui_common.pty_terminal import PtyTerminal

    term = PtyTerminal(tk_root)
    term.run(["/bin/sh", "-c", "for i in $(seq 1 200); do echo line-$i; done"])
    _wait_until(tk_root, lambda: not term._alive, timeout=3.0)

    assert isinstance(term._screen, pyte.HistoryScreen)
    assert term._screen.history.size > 0

    # Exercise the scroll bindings directly (no synthetic OS-level input).
    term._on_scroll_up()
    term._on_scroll_down()

    term.stop()
    term.destroy()
    tk_root.update()
