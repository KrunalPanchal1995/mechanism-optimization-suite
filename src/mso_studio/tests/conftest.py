import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@pytest.fixture(autouse=True)
def _close_leaked_top_level_widgets():
    """(Round-2 audit fix, B10 -- WORKBENCH_AUDIT_2026-08-10.md) Several
    test files construct a real top-level QWidget/QMainWindow per test
    (LauncherWindow, ModuleHero, ...) via ``.show()`` and never
    ``.close()`` it. ``QApplication`` is a process-wide singleton
    (``QApplication.instance() or QApplication([])``), so those leaked
    windows accumulate ACROSS EVERY test file in the whole pytest session,
    not just within the one file that created them -- and every
    ``apply_theme()``'s ``app.setStyleSheet(...)`` call re-polishes every
    live top-level widget the QApplication still knows about, so the cost
    grows with how many tests have already run.

    In a full ``src/mso_studio/tests/`` run this eventually makes some
    later test's window construction take minutes, which pytest's
    dot-per-test progress output makes look like it "hung" on whichever
    test's dot happened to print next -- the original B10 investigation
    blamed ``test_main_window_popout.py`` for exactly this reason (it
    reliably hung once progress reached ~57%, but 57% is a function of how
    many leaked windows had accumulated by then, not of that file itself).
    Reproducing under ``faulthandler`` with a SIGABRT timeout instead
    pointed at a completely different, much earlier test --
    ``test_launcher_window.py::test_doctor_entry_opens_doctor_dialog_not_a_launch_error``
    -- stuck inside ``apply_theme -> QApplication.setStyleSheet``.

    Fix: get rid of every top-level widget after each test, so the live set
    a QApplication carries into the next test never grows past what that
    one test itself created. Deliberately ``.hide()`` + ``.deleteLater()``
    rather than ``.close()``: a first attempt using ``.close()`` here traded
    one hang for a WORSE one -- ``.close()`` invokes the widget's own
    ``closeEvent`` override, and ``LauncherWindow.closeEvent`` can pop a
    modal ``QMessageBox``/``WelcomeDialog.exec()`` (real app behavior, e.g.
    "apps still running, leave them running?" / first-run welcome), which
    pumps a NESTED Qt event loop with nothing under ``offscreen`` able to
    dismiss it -- a genuine, reproducible hang, confirmed via the same
    faulthandler technique (stuck in
    ``LauncherWindow.closeEvent -> _maybe_show_welcome_on_first_run ->
    _open_welcome -> WelcomeDialog.exec()``). A generic, cross-cutting test
    fixture has no business invoking application-specific close-time
    business logic anyway; ``.hide()`` removes the widget from
    ``topLevelWidgets()``/style-repolish consideration exactly as well as
    ``.close()`` does, without running arbitrary app code.
    """
    yield
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        return
    app = QApplication.instance()
    if app is None:
        return
    def _pump():
        try:
            app.processEvents()
        except RuntimeError:
            # A handful of tests (test_module_hero.py,
            # test_native_extensions_tab.py, test_notebook_gen_tab.py)
            # construct and discard more than one page/canvas within a
            # single test body; matplotlib's Qt backend schedules a
            # deferred `QTimer.singleShot(0, self._draw_idle)` on canvas
            # show/resize that can outlive the canvas itself if the test
            # replaced it before that timer ever fired. That stale
            # callback firing here and raising "RuntimeError: Internal
            # C++ object (MplCanvas) already deleted" is a PRE-EXISTING
            # widget-lifecycle bug in those pages/tests -- unrelated to
            # this fixture's own cleanup, simply invisible before this
            # fixture existed because nothing previously called
            # processEvents() during teardown to give the stale timer a
            # chance to run at all. Recorded as a defect in
            # WORKBENCH_AUDIT_2026-08-10.md rather than fixed here (out of
            # scope for the pause/hang fix this fixture exists for) --
            # swallow it so it doesn't regress tests that passed before.
            pass

    # Drain events already queued BY THE TEST ITSELF first, so legitimate
    # pending callbacks (not just the stale-timer case above) run against
    # still-live objects before we start hiding/deleting anything below.
    _pump()
    for widget in list(app.topLevelWidgets()):
        widget.hide()
        widget.deleteLater()
    # Flush the DeferredDelete events queued by deleteLater() above so the
    # C++ objects are actually gone before the next test runs, not just
    # scheduled for some later, unspecified point.
    for _ in range(10):
        _pump()
