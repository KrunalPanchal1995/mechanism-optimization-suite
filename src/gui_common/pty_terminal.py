"""
gui_common/pty_terminal.py
===========================
Embedded real-terminal widget, extracted verbatim from
generate_target_opt.py's PtyTerminal class.

Standardized as the ONE long-job launch mechanism across all split-out GUI
apps (see GUI_SEPARATION_AND_IMPROVEMENT_PLAN_2026-07-12_142230.md §1.4) --
the monolith used this in exactly one place (Reaction Screening's SA button)
while other tabs shelled out to an external OS terminal window or blocked
the Tk mainloop with a plain subprocess.run(). Every new app's "run a long
external job" button should use this widget, not those other two patterns.
"""
from __future__ import annotations

import os
import time

try:
    import tkinter as tk
    from tkinter import ttk
    _HAS_TK = True
except ImportError:
    _HAS_TK = False

try:
    import pty
    import signal
    import threading
    import pyte
    _HAS_PTY_TERMINAL = True
except ImportError:
    _HAS_PTY_TERMINAL = False


class PtyTerminal(ttk.Frame):
    """A real embedded terminal: spawns a subprocess on a pseudo-terminal
    (stdlib pty.fork()) and renders its live ANSI/VT100 output — colors,
    cursor movement, in-place progress updates — via pyte's screen-buffer
    emulator, refreshed into a read-only Tk Text widget. Unlike a plain
    log-streaming text box, this behaves like a real terminal window because
    the child process actually has a tty (isatty() is True), so tools that
    render progress bars / colorize output based on that behave normally.
    """

    _FG_COLORS = {
        "black": "#45475a", "red": "#f38ba8", "green": "#a6e3a1", "brown": "#f9e2af",
        "blue": "#89b4fa", "magenta": "#cba6f7", "cyan": "#94e2d5", "white": "#cdd6f4",
        "default": "#cdd6f4",
    }

    def __init__(self, parent, rows=24, cols=100, font=("Courier", 10)):
        super().__init__(parent)
        self._rows, self._cols = rows, cols
        self._screen = None
        self._stream = None
        self._pid = None
        self._fd = None
        self._alive = False
        self._refresh_job = None
        # Monotonically-increasing generation counter. Each run() bumps this
        # and hands the new value to its reader thread as an explicit `gen`
        # argument; the reader checks `gen == self._gen` before touching
        # self._stream/self._screen so a stale reader thread from a
        # just-superseded session (still blocked in os.read() on the old fd)
        # can never feed bytes into the NEW session's Stream/Screen.
        self._gen = 0

        self.text = tk.Text(self, width=cols, height=rows, bg="#1e1e2e", fg="#cdd6f4",
                             insertbackground="#cdd6f4", font=font, wrap=tk.NONE, state="disabled")
        self.text.pack(fill=tk.BOTH, expand=True)
        for name, color in self._FG_COLORS.items():
            self.text.tag_configure(f"fg_{name}", foreground=color)

        # Scrollback interaction: Page Up/Down and mouse wheel (Windows/macOS
        # deliver <MouseWheel>; X11 delivers <Button-4>/<Button-5>).
        self.text.bind("<Prior>", self._on_scroll_up)
        self.text.bind("<Next>", self._on_scroll_down)
        self.text.bind("<MouseWheel>", self._on_mouse_wheel)
        self.text.bind("<Button-4>", self._on_scroll_up)
        self.text.bind("<Button-5>", self._on_scroll_down)

        # Window-close guard: closing/destroying the frame (or the app
        # window it lives in) must not leak the child process. <Destroy>
        # bubbles up from child widgets too (e.g. self.text), so only react
        # when the event's widget is this frame itself.
        self.bind("<Destroy>", self._on_destroy)

    def is_available(self) -> bool:
        return _HAS_PTY_TERMINAL

    def run(self, cmd: list, cwd: str = None):
        """Spawn cmd on a real pty and start streaming its live output."""
        if not _HAS_PTY_TERMINAL:
            self.text.configure(state="normal")
            self.text.delete("1.0", tk.END)
            self.text.insert(tk.END, "pyte is not installed — run: pip install pyte")
            self.text.configure(state="disabled")
            return
        self.stop()
        self._gen += 1
        my_gen = self._gen
        # HistoryScreen keeps a scrollback buffer (2000 lines, split evenly
        # between the top/bottom history queues per `ratio`) so output that
        # scrolls off the visible rows is still reachable via Page Up/Down
        # or the mouse wheel instead of being lost forever.
        self._screen = pyte.HistoryScreen(self._cols, self._rows, history=2000, ratio=0.5)
        self._stream = pyte.ByteStream(self._screen)
        pid, fd = pty.fork()
        if pid == 0:
            # Child: replace this process image entirely with cmd.
            try:
                if cwd:
                    os.chdir(cwd)
                os.execvp(cmd[0], cmd)
            except Exception:
                os._exit(1)
        self._pid = pid
        self._fd = fd
        self._alive = True
        threading.Thread(target=self._read_loop, args=(my_gen, fd), daemon=True).start()
        self._refresh()

    def _read_loop(self, gen, fd):
        """Reader loop for one run() "generation". `gen` and `fd` are taken
        as explicit arguments (not read from self.* inside the loop)
        precisely because self._fd/self._gen get reassigned by the NEXT
        run() call while this thread might still be blocked in os.read() on
        the OLD fd -- it must keep operating on the fd/generation it was
        started with, and must never feed a newer generation's Stream."""
        while True:
            if gen != self._gen:
                # Superseded by a newer run() before we even attempted a
                # read (or while sleeping between reads) -- exit quietly.
                return
            try:
                data = os.read(fd, 4096)
            except OSError:
                break
            if not data:
                break
            if gen != self._gen:
                # Became stale while blocked in os.read(); this data
                # belongs to a child that's already being torn down --
                # discard it rather than feeding the new session's stream.
                return
            self._stream.feed(data)
        # Only the CURRENT generation's reader is allowed to mark the
        # terminal not-alive; a stale/superseded reader exiting must not
        # stomp on a newer session's state.
        if gen == self._gen:
            self._alive = False

    def _refresh_once(self):
        """The actual screen-drawing body, factored out of the periodic
        _refresh() timer so both that timer AND the on-demand scrollback
        bindings (Page Up/Down, mouse wheel) can force an immediate
        re-render without waiting for the next 150ms tick."""
        if self._screen is not None:
            self.text.configure(state="normal")
            self.text.delete("1.0", tk.END)
            for row_idx in range(self._rows):
                line = self._screen.display[row_idx]
                row_buf = self._screen.buffer.get(row_idx, {})
                col, n = 0, len(line)
                while col < n:
                    ch = row_buf.get(col)
                    fg = ch.fg if (ch and ch.fg in self._FG_COLORS) else "default"
                    start = col
                    while col < n:
                        c2 = row_buf.get(col)
                        fg2 = c2.fg if (c2 and c2.fg in self._FG_COLORS) else "default"
                        if fg2 != fg:
                            break
                        col += 1
                    self.text.insert(tk.END, line[start:col], (f"fg_{fg}",))
                self.text.insert(tk.END, "\n")
            if not self._alive:
                self.text.insert(tk.END, "\n[process exited]\n")
            self.text.configure(state="disabled")
            self.text.see(tk.END)

    def _refresh(self):
        self._refresh_once()
        if self._alive:
            self._refresh_job = self.after(150, self._refresh)

    def _on_scroll_up(self, event=None):
        if self._screen is not None:
            self._screen.prev_page()
            self._refresh_once()
        return "break"

    def _on_scroll_down(self, event=None):
        if self._screen is not None:
            self._screen.next_page()
            self._refresh_once()
        return "break"

    def _on_mouse_wheel(self, event):
        # Windows delivers multiples of 120, macOS small integers; either
        # way, sign of event.delta indicates scroll direction.
        if getattr(event, "delta", 0) > 0:
            return self._on_scroll_up(event)
        return self._on_scroll_down(event)

    def _on_destroy(self, event):
        # <Destroy> bubbles up from every child widget (self.text, etc.) --
        # only react to the terminal frame's OWN destruction.
        if event.widget is self:
            self.stop()

    def stop(self):
        self._alive = False
        if self._refresh_job:
            self.after_cancel(self._refresh_job)
            self._refresh_job = None
        if self._pid:
            pid = self._pid
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass
            reaped = False
            # Bounded grace period (~1s in 50ms increments) for the child to
            # exit after SIGTERM before we escalate.
            for _ in range(20):
                try:
                    wpid, _status = os.waitpid(pid, os.WNOHANG)
                except (ChildProcessError, OSError):
                    # Already reaped (e.g. by the OS, or a previous stop()).
                    reaped = True
                    break
                if wpid == pid:
                    reaped = True
                    break
                time.sleep(0.05)
            if not reaped:
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
                try:
                    os.waitpid(pid, 0)
                except (ChildProcessError, OSError):
                    pass
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
        self._pid = None
        self._fd = None
