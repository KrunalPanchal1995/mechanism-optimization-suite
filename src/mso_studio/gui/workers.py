"""
mso_studio.gui.workers
========================
The one threading pattern every module page uses (guidelines doc, Part 2 /
Part 1.2): the GUI creates a worker object, moves it to a ``QThread``,
connects ``progress``/``finished(RunRecord)``/``error(str)`` signals to UI
slots, and starts it. The worker calls only ``core/`` functions -- never Qt.

No computation longer than ~100 ms may run on the GUI thread; this is the
only place in the codebase that is allowed to bridge a blocking ``core``
call onto a background thread.
"""
from __future__ import annotations

import threading
import traceback
from typing import Any, Callable

from PySide6.QtCore import QObject, QThread, Signal

from mso_studio.core.runrecord import RunRecord


class RunWorker(QObject):
    """Runs ``core_fn(*args, on_line=..., cancel_event=..., **kwargs)`` off-thread.

    ``core_fn`` must return a :class:`~mso_studio.core.runrecord.RunRecord`
    (or raise). Any core exception is caught here and re-raised to the GUI
    thread via the ``error`` signal with its ``str(exc)`` message intact --
    core exceptions are written to be user-facing (guidelines doc, Part 1.3
    / Part 3.1), so the GUI never needs to translate them further.
    """

    progress = Signal(str)          # one log line
    finished = Signal(object)       # RunRecord
    error = Signal(str)             # user-facing message
    cancelled = Signal()

    def __init__(self, core_fn: Callable[..., RunRecord], *args: Any, **kwargs: Any):
        super().__init__()
        self._core_fn = core_fn
        self._args = args
        self._kwargs = kwargs
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        """Request cancellation. The running backend process is terminated;
        the run's RunRecord (if one was already started) is marked cancelled."""
        self._cancel_event.set()

    def run(self) -> None:
        try:
            record = self._core_fn(
                *self._args,
                on_line=self.progress.emit,
                cancel_event=self._cancel_event,
                **self._kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - surfacing to the GUI is the point
            self.error.emit(str(exc) or f"{type(exc).__name__}: (no message)\n{traceback.format_exc()[-2000:]}")
            return
        if record.status == "cancelled":
            self.cancelled.emit()
        else:
            self.finished.emit(record)


def run_in_thread(core_fn: Callable[..., RunRecord], *args: Any, **kwargs: Any) -> tuple[QThread, RunWorker]:
    """Build and start a ``QThread``/``RunWorker`` pair for *core_fn*.

    Returns ``(thread, worker)`` -- the caller must keep both alive (e.g. as
    attributes on the module page) until ``finished``/``error``/``cancelled``
    fires, then call ``thread.quit(); thread.wait()``.
    """
    thread = QThread()
    worker = RunWorker(core_fn, *args, **kwargs)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    for sig in (worker.finished, worker.error, worker.cancelled):
        sig.connect(thread.quit)
    thread.start()
    return thread, worker


class CallWorker(QObject):
    """Runs any plain ``core_fn(*args, **kwargs) -> object`` off-thread.

    For quick, non-``RunRecord`` operations (validate/status/preflight-style
    checks) that still must not block the GUI thread but don't fit the
    run/progress/cancel shape of :class:`RunWorker` -- no ``on_line``/
    ``cancel_event`` are injected here.
    """

    finished = Signal(object)
    error = Signal(str)

    def __init__(self, core_fn: Callable[..., Any], *args: Any, **kwargs: Any):
        super().__init__()
        self._core_fn = core_fn
        self._args = args
        self._kwargs = kwargs

    def run(self) -> None:
        try:
            result = self._core_fn(*self._args, **self._kwargs)
        except Exception as exc:  # noqa: BLE001 - surfacing to the GUI is the point
            self.error.emit(str(exc) or f"{type(exc).__name__}: (no message)\n{traceback.format_exc()[-2000:]}")
            return
        self.finished.emit(result)


def call_in_thread(core_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[QThread, CallWorker]:
    """Build and start a ``QThread``/``CallWorker`` pair for *core_fn*. Same
    lifecycle contract as :func:`run_in_thread`."""
    thread = QThread()
    worker = CallWorker(core_fn, *args, **kwargs)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    for sig in (worker.finished, worker.error):
        sig.connect(thread.quit)
    thread.start()
    return thread, worker
