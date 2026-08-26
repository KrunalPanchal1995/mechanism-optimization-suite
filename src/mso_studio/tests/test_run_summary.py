"""
KRUNAL_JULY_23.md point 1: "Export run status…" on RunSummaryCard.

This is the first widget-level (not core-only) test in this suite -- kept
deliberately minimal (construct + read button state, no click simulation,
no file dialog) since QT_QPA_PLATFORM=offscreen is the only headless
guarantee this repo relies on for PySide6 (see the GUI test campaign's
TESTING_CONVENTIONS.md: never drive a live X display). A QApplication is
required for any QWidget to exist at all; it's created once per test-session
by the module-scoped fixture below.
"""
import pytest

from mso_studio.core.runrecord import RunConfig, new_record


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_export_button_disabled_before_any_run(qapp):
    from mso_studio.gui.widgets.run_summary import RunSummaryCard
    card = RunSummaryCard()
    assert not card.export_status_button.isEnabled()


def test_export_button_enabled_after_successful_run(qapp):
    from mso_studio.gui.widgets.run_summary import RunSummaryCard
    card = RunSummaryCard()
    cfg = RunConfig(module="nominal_sim", label="htc_case")
    record = new_record(cfg)
    record.mark_finished("done", return_code=0)
    card.show_record(record)
    assert card.export_status_button.isEnabled()


def test_export_button_enabled_after_failed_run_too(qapp):
    """Unlike the LaTeX-report button, export-run-status must stay enabled
    for a FAILED run -- seeing exactly what a failed run's inputs/options/
    error were is the whole point of this feature."""
    from mso_studio.gui.widgets.run_summary import RunSummaryCard
    card = RunSummaryCard()
    cfg = RunConfig(module="sensitivity", label="case1")
    record = new_record(cfg)
    record.mark_finished("failed", return_code=1, error="solver crashed")
    card.show_record(record)
    assert card.export_status_button.isEnabled()
    assert not card.report_button.isEnabled()      # LaTeX report stays gated on "done"


def test_export_button_disabled_while_running(qapp):
    from mso_studio.gui.widgets.run_summary import RunSummaryCard
    card = RunSummaryCard()
    cfg = RunConfig(module="nominal_sim", label="htc_case")
    record = new_record(cfg)
    record.mark_finished("done", return_code=0)
    card.show_record(record)
    assert card.export_status_button.isEnabled()

    card.show_running("htc_case_2")
    assert not card.export_status_button.isEnabled()
