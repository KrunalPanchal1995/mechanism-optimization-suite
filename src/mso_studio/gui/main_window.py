"""
mso_studio.gui.main_window
=============================
The integrator shell (guidelines doc, Part 6.1/6.7): a left navigation list
of module pages, a menu bar + toolbar built from the action registry, a
persistent status bar, and window geometry/splitter state that survives a
restart. Each module page is also "pop-out capable": double-clicking its
nav entry (or Window > Open in new window) detaches it into its own
top-level window, for the "hybrid shell + pop-out" architecture.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QSettings, QSize, Qt
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent, QDragEnterEvent, QDropEvent, QIcon
from PySide6.QtWidgets import (
    QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QSplitter,
    QStackedWidget, QStatusBar, QToolBar, QVBoxLayout, QWidget,
)

from mso_studio import __version__
from mso_studio.gui import theme
from mso_studio.gui.actions import ACTIONS
from mso_studio.gui.module_window import ModuleWindow

ORG_NAME = "mechanism-optimization-suite"
APP_NAME = "mso-studio"


class MainWindow(QMainWindow):
    def __init__(self, page_factories: "list[tuple[str, str, callable]]"):
        """
        Parameters
        ----------
        page_factories : list of (module_id, nav_label, factory)
            *factory* is a zero-arg callable returning a freshly-built
            ``ModulePage`` (or any ``QWidget``) instance. Pages are built
            lazily, on first navigation, so app start-up doesn't pay for
            every module's setup widgets up front.
        """
        super().__init__()
        self.setWindowTitle(f"MSO Studio — {APP_NAME} v{__version__}")
        self.setAcceptDrops(True)
        self._settings = QSettings(ORG_NAME, APP_NAME)
        self._page_factories = {mid: factory for mid, _label, factory in page_factories}
        self._page_labels = {mid: label for mid, label, _factory in page_factories}
        self._pages: dict[str, QWidget] = {}
        self._detached_windows: list[QMainWindow] = []
        # Appearance (theme_mode, accent, font, density, wallpaper) is
        # shared app-wide via mso_studio.gui.preferences' own QSettings
        # namespace, not this window's self._settings (which is scoped to
        # "mso-studio" for geometry/splitter state) -- so Preferences stays
        # in sync whether it's opened from the launcher or from here.
        from mso_studio.gui.preferences import load_preferences
        self._theme_mode = load_preferences()["theme_mode"]

        self._build_nav_and_stack(page_factories)
        self._build_menu_and_toolbar()
        self._build_status_bar()
        self._restore_geometry()
        self._apply_theme()

        if self.nav_list.count() > 0:
            self.nav_list.setCurrentRow(0)

    # -- layout ---------------------------------------------------------

    def _build_nav_and_stack(self, page_factories) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal, central)

        sidebar = QWidget(splitter)
        sidebar.setObjectName("Sidebar")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        self.nav_list = QListWidget(sidebar)
        self.nav_list.setObjectName("NavList")
        self.nav_list.setMinimumWidth(200)
        self.nav_list.setMaximumWidth(260)
        for module_id, label, _factory in page_factories:
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, module_id)
            self.nav_list.addItem(item)
        self.nav_list.currentRowChanged.connect(self._on_nav_changed)
        self.nav_list.itemDoubleClicked.connect(self._detach_item)
        sidebar_layout.addWidget(self.nav_list)
        splitter.addWidget(sidebar)

        self.stack = QStackedWidget(splitter)
        self._empty_state = self._build_empty_state()
        self.stack.addWidget(self._empty_state)
        splitter.addWidget(self.stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([220, 1060])

        layout.addWidget(splitter)
        self.setCentralWidget(central)

    def _build_empty_state(self) -> QWidget:
        w = QWidget(self)
        v = QVBoxLayout(w)
        v.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label = QLabel("Select a module on the left to get started.", w)
        label.setObjectName("Caption")
        v.addWidget(label)
        return w

    def _get_or_build_page(self, module_id: str) -> QWidget:
        if module_id not in self._pages:
            page = self._page_factories[module_id]()
            if hasattr(page, "statusMessage"):
                page.statusMessage.connect(lambda msg: self.statusBar().showMessage(msg, 5000))
            self._pages[module_id] = page
            self.stack.addWidget(page)
        return self._pages[module_id]

    def _on_nav_changed(self, row: int) -> None:
        if row < 0:
            self.stack.setCurrentWidget(self._empty_state)
            return
        item = self.nav_list.item(row)
        module_id = item.data(Qt.ItemDataRole.UserRole)
        page = self._get_or_build_page(module_id)
        self.stack.setCurrentWidget(page)

    # -- pop-out (hybrid shell + pop-out architecture) ---------------------

    def _detach_current(self) -> None:
        row = self.nav_list.currentRow()
        if row < 0:
            return
        self._detach_item(self.nav_list.item(row))

    def _detach_item(self, item: QListWidgetItem) -> None:
        """Open *item*'s module in its own top-level ``ModuleWindow``.

        Builds a fresh page instance (via the same factory the shell uses)
        rather than reparenting the embedded one -- Qt widgets have exactly
        one parent, and reparenting would leave the shell's nav entry
        pointing at an empty stack page until the detached window closes.
        The detached copy and the shell's own copy run independently, each
        with its own setup/results state, which is the simpler and more
        predictable of the two behaviors for a long-running module page.
        """
        module_id = item.data(Qt.ItemDataRole.UserRole)
        label = self._page_labels.get(module_id, module_id)
        page = self._page_factories[module_id]()
        window = ModuleWindow(module_id, label, page, parent=None)
        self._detached_windows.append(window)
        window.destroyed.connect(lambda *_: self._forget_detached(window))
        window.show()

    def _forget_detached(self, window) -> None:
        if window in self._detached_windows:
            self._detached_windows.remove(window)

    # -- menu / toolbar ---------------------------------------------------

    def _build_menu_and_toolbar(self) -> None:
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction(self._make_action("Preferences…", self._open_preferences))
        file_menu.addSeparator()
        exit_action = self._make_action("Exit", self.close, shortcut="Ctrl+Q")
        file_menu.addAction(exit_action)

        view_menu = menu_bar.addMenu("&View")
        theme_menu = view_menu.addMenu("Theme")
        theme_group = QActionGroup(self)
        theme_group.setExclusive(True)
        for mode, label in (("system", "System"), ("light", "Light"), ("dark", "Dark")):
            action = QAction(label, self, checkable=True)
            action.setChecked(mode == self._theme_mode)
            action.triggered.connect(lambda checked, m=mode: self._set_theme(m))
            theme_group.addAction(action)
            theme_menu.addAction(action)

        window_menu = menu_bar.addMenu("&Window")
        window_menu.addAction(self._make_action(
            "Open in new window", self._detach_current, shortcut="Ctrl+Shift+N",
        ))

        # Module pages are reached via the sidebar; &Tools is reserved for
        # cross-cutting tools (PySR/digitizer/PDF panels, WS-D) once wired.
        menu_bar.addMenu("&Tools")

        help_menu = menu_bar.addMenu("&Help")
        help_menu.addAction(self._make_action("Documentation", self._open_docs, shortcut="F1"))
        help_menu.addAction(self._make_action("Open log folder", self._open_log_folder))
        help_menu.addAction(self._make_action("About", self._open_about))

        toolbar = QToolBar("Main", self)
        toolbar.setObjectName("MainToolbar")
        toolbar.setIconSize(QSize(18, 18))
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

    def _make_action(self, label: str, handler, shortcut: Optional[str] = None) -> QAction:
        action = QAction(label, self)
        if shortcut:
            action.setShortcut(shortcut)
        action.triggered.connect(handler)
        return action

    def _build_status_bar(self) -> None:
        bar = QStatusBar(self)
        self.setStatusBar(bar)
        version_label = QLabel(f"mso-studio v{__version__}", self)
        version_label.setObjectName("Caption")
        bar.addPermanentWidget(version_label)
        bar.showMessage("Ready", 3000)

    # -- theme ---------------------------------------------------------

    def _set_theme(self, mode: str) -> None:
        self._theme_mode = mode
        self._settings.setValue("theme_mode", mode)
        self._apply_theme()

    def _apply_theme(self) -> None:
        from PySide6.QtWidgets import QApplication

        from mso_studio.gui.preferences import apply_preferences_to_app
        app = QApplication.instance()
        if app is not None:
            # apply_preferences_to_app reads QSettings (theme_mode was just
            # written there by _set_theme before this call), so it picks up
            # accent/font/density from Preferences too, not just the mode.
            apply_preferences_to_app(app)

    # -- menu handlers ---------------------------------------------------

    def _open_preferences(self) -> None:
        from mso_studio.gui.preferences import PreferencesDialog, load_preferences

        dialog = PreferencesDialog(self)
        if dialog.exec():
            # PreferencesDialog persists under its own shared appearance
            # namespace (mso-launcher) so the launcher and every Studio
            # window agree on one appearance regardless of which one opened
            # Preferences -- read that back, not this window's own
            # self._settings (mso-studio), which has no such key.
            self._theme_mode = load_preferences()["theme_mode"]

    def _open_docs(self) -> None:
        import webbrowser
        docs_index = Path(__file__).resolve().parents[3] / "docs" / "_build" / "html" / "index.html"
        if docs_index.exists():
            webbrowser.open(docs_index.as_uri())
        else:
            QMessageBox.information(
                self, "Documentation",
                "The docs site hasn't been built yet.\nRun: sphinx-build -b html docs docs/_build/html",
            )

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

    # -- geometry persistence ---------------------------------------------------

    def _restore_geometry(self) -> None:
        geometry = self._settings.value("window_geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        else:
            self.resize(1280, 800)  # guidelines doc Part 6.2 default

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt override
        self._settings.setValue("window_geometry", self.saveGeometry())
        super().closeEvent(event)

    # -- drag and drop (Part 6.8.2) ---------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        self.statusBar().showMessage(f"Dropped: {path} — open the relevant module to load it.", 6000)
