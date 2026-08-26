"""
mso_studio.gui.widgets.file_picker
=====================================
A labeled file/directory picker: ``QLineEdit`` + "Browse..." button, with
inline validation (guidelines doc, Part 1.6 -- "validate as typed, red
outline + inline message, not after Run is pressed"). Optionally remembers
the last few paths chosen (per *recent_key*) across sessions, with a small
dropdown next to Browse to reuse one.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QSettings, Signal
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLineEdit, QMenu, QPushButton, QToolButton, QWidget,
)

_RECENT_MAX = 8


class FilePicker(QWidget):
    """A path field with a native-dialog browse button and live existence check."""

    pathChanged = Signal(str)

    def __init__(self, *, mode: str = "file", filter_: str = "All files (*)",
                 must_exist: bool = True, recent_key: Optional[str] = None,
                 parent: Optional[QWidget] = None):
        """
        Parameters
        ----------
        mode : {"file", "directory", "save"}
        filter_ : str
            Qt file-dialog filter string (ignored for ``mode="directory"``).
        must_exist : bool
            If True, an empty or non-existent path is flagged red
            (see :attr:`is_valid`); ``mode="save"`` implies ``False``.
        recent_key : str, optional
            If given, the last (up to 8) distinct paths chosen through this
            picker are persisted under this key (shared QSettings namespace
            with the rest of Studio) and offered in a small dropdown next
            to Browse -- opt-in, since a one-off picker (e.g. "output
            directory") gains nothing from a recency list.
        """
        super().__init__(parent)
        self._mode = mode
        self._filter = filter_
        self._must_exist = must_exist and mode != "save"
        self._recent_key = recent_key
        self._settings = QSettings("mechanism-optimization-suite", "mso-studio")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.edit = QLineEdit(self)
        self.edit.textChanged.connect(self._on_text_changed)
        layout.addWidget(self.edit, 1)

        self.browse_button = QPushButton("Browse…", self)
        self.browse_button.clicked.connect(self._browse)
        layout.addWidget(self.browse_button)

        self.recent_button: Optional[QToolButton] = None
        if recent_key is not None:
            self.recent_button = QToolButton(self)
            self.recent_button.setText("▾")
            self.recent_button.setToolTip("Recent")
            self.recent_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            self.recent_button.setMenu(self._build_recent_menu())
            layout.addWidget(self.recent_button)

        self._on_text_changed(self.edit.text())

    def _recent_paths(self) -> List[str]:
        if self._recent_key is None:
            return []
        value = self._settings.value(f"recent_paths/{self._recent_key}", [])
        return list(value) if value else []

    def _remember_path(self, path: str) -> None:
        if self._recent_key is None or not path:
            return
        recents = [p for p in self._recent_paths() if p != path]
        recents.insert(0, path)
        self._settings.setValue(f"recent_paths/{self._recent_key}", recents[:_RECENT_MAX])
        if self.recent_button is not None:
            self.recent_button.setMenu(self._build_recent_menu())

    def _build_recent_menu(self) -> QMenu:
        menu = QMenu(self)
        recents = self._recent_paths()
        if not recents:
            action = menu.addAction("(no recent paths)")
            action.setEnabled(False)
            return menu
        for path in recents:
            action = menu.addAction(path)
            action.triggered.connect(lambda checked=False, p=path: self.edit.setText(p))
        return menu

    def _browse(self) -> None:
        start = self.edit.text() or str(Path.home())
        if self._mode == "file":
            path, _ = QFileDialog.getOpenFileName(self, "Select file", start, self._filter)
        elif self._mode == "save":
            path, _ = QFileDialog.getSaveFileName(self, "Select destination", start, self._filter)
        else:
            path = QFileDialog.getExistingDirectory(self, "Select directory", start)
        if path:
            self.edit.setText(path)
            self._remember_path(path)

    def _on_text_changed(self, text: str) -> None:
        valid = self.is_valid
        self.edit.setProperty("invalid", not valid)
        self.edit.style().unpolish(self.edit)
        self.edit.style().polish(self.edit)
        self.pathChanged.emit(text)

    @property
    def is_valid(self) -> bool:
        text = self.edit.text().strip()
        if not text:
            return False
        if not self._must_exist:
            return True
        return Path(text).exists()

    def path(self) -> str:
        return self.edit.text().strip()

    def set_path(self, value: str) -> None:
        self.edit.setText(value)
