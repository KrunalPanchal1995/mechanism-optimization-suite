"""
mso_studio.gui.widgets.log_view
=================================
The collapsible log/messages strip (guidelines doc, Part 6.1: "Log /
messages, collapsible, ~22%") that every module page's worker streams
backend stdout/stderr into.
"""
from __future__ import annotations

from datetime import datetime

from PySide6.QtWidgets import QPlainTextEdit


class LogView(QPlainTextEdit):
    """Append-only, auto-scrolling log surface for streamed backend output."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("LogView")
        self.setReadOnly(True)
        self.setMaximumBlockCount(5000)  # bound memory on very chatty backends

    def append_line(self, line: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.appendPlainText(f"{stamp}  {line}")
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())

    def append_status(self, message: str) -> None:
        """Append a GUI-generated status line (run started/finished/failed),
        visually the same as backend lines -- the log is one continuous
        narrative of what happened (guidelines doc, Part 1.1)."""
        self.append_line(f"— {message} —")
