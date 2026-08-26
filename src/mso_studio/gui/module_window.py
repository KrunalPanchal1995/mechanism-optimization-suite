"""
mso_studio.gui.module_window
===============================
A standalone top-level window around exactly one module page. Two things
share this: MainWindow's "pop-out" (double-click a nav entry / Window >
Open in new window, detaching a page that's currently living inside the
shell) and ``run_mso_studio.py --module <id> --standalone`` (a launcher
tile opening straight into one module, with no shell/sidebar at all --
Phase 1 of the 2026-08 GUI integration overhaul: every launcher tile with a
PySide6 equivalent must open that, not the legacy Tkinter script of the
same name).

Kept deliberately thin: theming, log folder, and preferences are
one-liners delegating to the same helpers MainWindow uses, so the two
window kinds never drift out of sync on appearance.
"""
from __future__ import annotations

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QLabel, QMainWindow, QMessageBox, QStatusBar, QWidget

from mso_studio import __version__

ORG_NAME = "mechanism-optimization-suite"


class ModuleWindow(QMainWindow):
    """Top-level window hosting one already-built module page."""

    def __init__(self, module_id: str, label: str, page: QWidget, parent=None):
        super().__init__(parent)
        self.module_id = module_id
        self.page = page
        self.setWindowTitle(f"{label} — MSO Studio v{__version__}")
        # Standalone tiles/pop-outs are throwaway windows; let Qt actually
        # destroy (and emit `destroyed`) on close instead of leaving a
        # hidden zombie only Python gc would eventually clean up -- callers
        # (MainWindow's detached-window tracking) rely on `destroyed` firing.
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._settings = QSettings(ORG_NAME, f"mso-studio-{module_id}")

        self.setCentralWidget(page)
        if hasattr(page, "statusMessage"):
            page.statusMessage.connect(lambda msg: self.statusBar().showMessage(msg, 5000))

        self._build_menu()
        self._build_status_bar(label)
        self._restore_geometry()
        self._apply_theme()

    # -- menu / status bar -------------------------------------------------

    def _build_menu(self) -> None:
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction("Preferences…", self._open_preferences)
        file_menu.addSeparator()
        exit_action = file_menu.addAction("Exit")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)

        help_menu = menu_bar.addMenu("&Help")
        help_menu.addAction("Open log folder", self._open_log_folder)
        help_menu.addAction("About", self._open_about)

    def _build_status_bar(self, label: str) -> None:
        bar = QStatusBar(self)
        self.setStatusBar(bar)
        version_label = QLabel(f"mso-studio v{__version__}", self)
        version_label.setObjectName("Caption")
        bar.addPermanentWidget(version_label)
        bar.showMessage(f"{label} ready", 3000)

    # -- theme ---------------------------------------------------------

    def _apply_theme(self) -> None:
        from PySide6.QtWidgets import QApplication

        from mso_studio.gui.preferences import apply_preferences_to_app
        app = QApplication.instance()
        if app is not None:
            apply_preferences_to_app(app)

    def _open_preferences(self) -> None:
        from mso_studio.gui.preferences import PreferencesDialog

        dialog = PreferencesDialog(self)
        if dialog.exec():
            self._apply_theme()

    def _open_log_folder(self) -> None:
        import subprocess
        import sys

        from mso_studio.core.runrecord import runs_root
        path = str(runs_root())
        try:
            if sys.platform.startswith("linux"):
                subprocess.Popen(["xdg-open", path])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            elif sys.platform.startswith("win"):
                import os
                os.startfile(path)  # type: ignore[attr-defined]
        except OSError:
            QMessageBox.information(self, "Log folder", path)

    def _open_about(self) -> None:
        QMessageBox.about(
            self, "About MSO Studio",
            f"<b>MSO Studio</b> v{__version__}<br>"
            "Unified PySide6 GUI for the Mechanism Optimization Suite.<br><br>"
            "Backends run in the anaconda python3.9 environment; "
            "the GUI runs in a dedicated .venv-gui.",
        )

    # -- geometry persistence ---------------------------------------------

    def _restore_geometry(self) -> None:
        geometry = self._settings.value("window_geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        else:
            self.resize(1100, 760)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt override
        self._settings.setValue("window_geometry", self.saveGeometry())
        super().closeEvent(event)
