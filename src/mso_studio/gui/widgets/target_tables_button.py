"""
mso_studio.gui.widgets.target_tables_button
==============================================
A small reusable "Generate LaTeX target tables..." button -- the richer,
module-specific LaTeX export (guidelines doc, Part 4: "Export options...
next to the one-click default") available on any module page that already
has a ``.target`` file selected. Wraps
:func:`mso_studio.core.mso_reports.generate_target_tables`.
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtWidgets import QMessageBox, QPushButton, QWidget

from mso_studio.core.mso_reports import TargetTablesParams, generate_target_tables
from mso_studio.gui.workers import call_in_thread


def add_target_tables_button(
    parent: QWidget, get_target_csv: Callable[[], str], status_signal, log_view,
) -> QPushButton:
    """Build and return a "Generate LaTeX target tables..." button. The
    caller adds it to their own layout (``add_advanced_row`` etc.);
    *get_target_csv* is called at click time to fetch the current
    ``.target`` path from whatever picker the page already has.
    """
    button = QPushButton("Generate LaTeX target tables…", parent)
    button.setToolTip(
        "Per-target-type LaTeX tables (Tig/Fls/JSR/Flf/RCM) straight from "
        "the .target file, via src/target_tables.py."
    )
    threads: list = []  # keep worker pairs alive on the button itself

    def _on_click() -> None:
        target_csv = get_target_csv()
        if not target_csv:
            QMessageBox.information(parent, "No targets file", "Select a .target file first.")
            return
        status_signal.emit("Generating LaTeX target tables…")
        params = TargetTablesParams(target_csv=target_csv, compile_pdf=True)
        thread, worker = call_in_thread(generate_target_tables, params)

        def _finished(tex_path):
            log_view.append_status(f"LaTeX target tables: {tex_path}")
            status_signal.emit(f"Target tables written to {tex_path}")

        def _error(message):
            log_view.append_status(f"Target tables failed: {message}")
            status_signal.emit("Target tables generation failed — see log")

        worker.finished.connect(_finished)
        worker.error.connect(_error)
        threads.append((thread, worker))

    button.clicked.connect(_on_click)
    button._keep_alive_threads = threads  # avoid GC of the list itself
    return button
