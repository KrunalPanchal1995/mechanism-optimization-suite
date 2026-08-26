"""
mso_studio.launcher.welcome
==============================
``WelcomeDialog`` -- the first-run (and Help-menu-reachable) tour: a short,
paginated walkthrough of what the suite is for and how to use the
launcher, using the plain-language copy in ``content.WELCOME_PAGES``.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from mso_studio.launcher.art import wordmark
from mso_studio.launcher.content import WELCOME_PAGES


class WelcomeDialog(QDialog):
    def __init__(self, mode: str = "dark", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Welcome")
        self.setModal(True)
        self.resize(560, 420)
        self._mode = mode
        self._page = 0

        # QDialog paints its Window background from the application QPalette
        # by default; now that theme.apply_theme() sets a real QPalette (not
        # just a stylesheet), no manual per-dialog override is needed here.

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 20)
        layout.setSpacing(12)

        logo = QLabel(self)
        logo.setPixmap(wordmark(36, mode))
        layout.addWidget(logo)

        self.title_label = QLabel(self)
        self.title_label.setObjectName("WelcomeTitle")
        self.title_label.setWordWrap(True)
        layout.addWidget(self.title_label)

        self.body_label = QLabel(self)
        self.body_label.setWordWrap(True)
        self.body_label.setObjectName("WelcomeBody")
        layout.addWidget(self.body_label, 1)

        self.dots_row = QHBoxLayout()
        self.dots_row.setSpacing(6)
        self.dots_row.addStretch(1)
        self._dot_labels = []
        for _ in WELCOME_PAGES:
            dot = QLabel("●", self)
            dot.setObjectName("PageDot")
            self.dots_row.addWidget(dot)
            self._dot_labels.append(dot)
        self.dots_row.addStretch(1)
        layout.addLayout(self.dots_row)

        button_row = QHBoxLayout()
        button_row.setSpacing(8)
        self.skip_button = QPushButton("Skip", self)
        self.skip_button.clicked.connect(self.accept)
        button_row.addWidget(self.skip_button)
        button_row.addStretch(1)
        self.back_button = QPushButton("Back", self)
        self.back_button.clicked.connect(self._go_back)
        button_row.addWidget(self.back_button)
        self.next_button = QPushButton("Next", self)
        self.next_button.setObjectName("PrimaryAction")
        self.next_button.clicked.connect(self._go_next)
        button_row.addWidget(self.next_button)
        layout.addLayout(button_row)

        self._render_page()

    def _render_page(self) -> None:
        page = WELCOME_PAGES[self._page]
        self.title_label.setText(page.title)
        self.body_label.setText(page.body)
        self.back_button.setEnabled(self._page > 0)
        is_last = self._page == len(WELCOME_PAGES) - 1
        self.next_button.setText("Get started" if is_last else "Next")
        self.skip_button.setVisible(not is_last)
        for i, dot in enumerate(self._dot_labels):
            dot.setProperty("active", i == self._page)
            dot.style().unpolish(dot)
            dot.style().polish(dot)

    def _go_next(self) -> None:
        if self._page >= len(WELCOME_PAGES) - 1:
            self.accept()
            return
        self._page += 1
        self._render_page()

    def _go_back(self) -> None:
        if self._page > 0:
            self._page -= 1
            self._render_page()
