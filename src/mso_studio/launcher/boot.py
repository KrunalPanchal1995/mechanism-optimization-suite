"""
mso_studio.launcher.boot
===========================
``BootSplash`` -- the "loading a game" moment. Frameless, centered, painted
gradient background with the wordmark; runs the real preflight checks from
``preflight.py`` off the GUI thread (via the existing
``mso_studio.gui.workers.call_in_thread`` pattern) and reveals each result
line by line with a small stagger, so what looks like a loading animation is
actually reporting genuine facts about the environment (venv present,
science python resolves, Cantera importable, all 13 scripts found, free
disk). Click anywhere, or wait, and it hands off to the main window.
"""
from __future__ import annotations

from typing import List

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter
from PySide6.QtWidgets import (
    QApplication, QGraphicsOpacityEffect, QLabel, QProgressBar, QVBoxLayout, QWidget,
)

from mso_studio.gui.theme import palette_for
from mso_studio.gui.workers import call_in_thread
from mso_studio.launcher.art import wordmark
from mso_studio.launcher.preflight import BootCheck, FAIL, OK, WARN, run_boot_checks

_STATE_GLYPH = {OK: "✔", WARN: "⚠", FAIL: "✖"}
_REVEAL_INTERVAL_MS = 150
_MIN_VISIBLE_MS = 700


def _collect_checks() -> List[BootCheck]:
    """core_fn for call_in_thread: run every preflight check to completion."""
    return list(run_boot_checks())


class BootSplash(QWidget):
    """Shows itself, runs preflight checks, emits ``finished(list[BootCheck])``."""

    finished = Signal(list)

    def __init__(self, mode: str = "dark") -> None:
        super().__init__(
            None,
            Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self._mode = mode
        self._palette = palette_for(mode)
        self._results: List[BootCheck] = []
        self._revealed = 0
        self._done_collecting = False
        self._emitted = False
        self._thread = None
        self._worker = None

        self.setFixedSize(520, 360)
        self._center_on_screen()
        self._build_ui()

        self._reveal_timer = QTimer(self)
        self._reveal_timer.setInterval(_REVEAL_INTERVAL_MS)
        self._reveal_timer.timeout.connect(self._reveal_next)

        self._min_timer = QTimer(self)
        self._min_timer.setSingleShot(True)
        self._min_timer.setInterval(_MIN_VISIBLE_MS)
        self._min_timer.timeout.connect(self._maybe_finish)

    # -- construction -----------------------------------------------------
    def _center_on_screen(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        self.move(
            geo.center().x() - self.width() // 2,
            geo.center().y() - self.height() // 2,
        )

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 36, 40, 28)
        layout.setSpacing(6)
        layout.addStretch(1)

        self.logo_label = QLabel(self)
        self.logo_label.setPixmap(wordmark(52, self._mode))
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.logo_label)

        subtitle = QLabel("MECHANISM OPTIMIZATION SUITE", self)
        subtitle.setObjectName("SplashSubtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(subtitle)
        layout.addSpacing(18)

        self.log_label = QLabel("", self)
        self.log_label.setObjectName("SplashLog")
        self.log_label.setMinimumHeight(140)
        self.log_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.log_label)

        layout.addStretch(1)

        self.progress = QProgressBar(self)
        self.progress.setObjectName("SplashProgress")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        layout.addWidget(self.progress)

        hint = QLabel("click to skip", self)
        hint.setObjectName("SplashHint")
        hint.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(hint)

    # -- painting -----------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        grad = QLinearGradient(rect.topLeft(), rect.bottomRight())
        grad.setColorAt(0.0, QColor(self._palette["bg"]))
        grad.setColorAt(1.0, QColor(self._palette["surface"]))
        painter.fillRect(rect, grad)
        painter.setPen(QColor(self._palette["border"]))
        painter.drawRect(rect.adjusted(0, 0, -1, -1))

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        self.show()
        self._min_timer.start()
        self._thread, self._worker = call_in_thread(_collect_checks)
        self._worker.finished.connect(self._on_checks_collected)
        self._worker.error.connect(self._on_checks_errored)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override, click-to-skip
        self._skip()
        super().mousePressEvent(event)

    def _skip(self) -> None:
        self._reveal_timer.stop()
        self._revealed = len(self._results)
        self._render_log()
        self.progress.setValue(100)
        self._maybe_finish(force=True)

    # -- check collection / reveal -----------------------------------------------
    def _on_checks_collected(self, results: List[BootCheck]) -> None:
        self._results = results
        self._done_collecting = True
        self._reveal_timer.start()

    def _on_checks_errored(self, message: str) -> None:
        # Preflight checks are defensive and shouldn't normally raise, but if
        # something unexpected happens, don't block the launcher on it.
        self._results = [BootCheck("Preflight", WARN, message.splitlines()[0] if message else "error")]
        self._done_collecting = True
        self._reveal_timer.start()

    def _reveal_next(self) -> None:
        if self._revealed >= len(self._results):
            self._reveal_timer.stop()
            self._maybe_finish()
            return
        self._revealed += 1
        self._render_log()

    def _render_log(self) -> None:
        lines = []
        for check in self._results[: self._revealed]:
            glyph = _STATE_GLYPH.get(check.state, "•")
            lines.append(f"{glyph}  {check.label}")
        self.log_label.setText("<br>".join(lines))
        total = max(len(self._results), 1)
        self.progress.setValue(int(100 * self._revealed / total))

    # -- handoff -----------------------------------------------------------
    def _maybe_finish(self, force: bool = False) -> None:
        if self._emitted:
            return
        all_revealed = self._done_collecting and self._revealed >= len(self._results)
        min_elapsed = not self._min_timer.isActive()
        if force or (all_revealed and min_elapsed):
            self._emitted = True
            self._reveal_timer.stop()
            self.finished.emit(self._results)
