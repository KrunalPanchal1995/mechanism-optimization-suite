"""
mso_studio.launcher.window
=============================
``LauncherWindow`` -- the Steam-library main window: search + stage rail on
the left, a hero panel + reflowing tile grid on the right, a live
running-process status strip at the bottom. Ties together every other
module in this package: ``catalog`` (what to show), ``process`` (what's
running), ``preflight``/``doctor`` (environment status), ``art``/``tiles``
(how it looks).
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import (
    QAction, QActionGroup, QCloseEvent, QColor, QKeySequence, QPalette, QShortcut,
)
from PySide6.QtWidgets import (
    QApplication, QDialog, QFileDialog, QFrame, QHBoxLayout, QInputDialog, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QScrollArea, QSplitter, QStackedWidget, QStatusBar, QVBoxLayout, QWidget,
)

from mso_studio import __version__
from mso_studio.gui import theme
from mso_studio.gui.backdrop import BackdropWidget
from mso_studio.gui.tokens import tokens_for
from mso_studio.launcher.wallpapers import resolve_wallpaper_path
from mso_studio.launcher.catalog import (
    CATALOG, CATALOG_BY_ID, RUNNER_INTERNAL, STAGES, STAGE_LABELS, AppEntry,
    entries_for_stage, search,
)
from mso_studio.launcher.doctor import DoctorPanel
from mso_studio.launcher.help_panel import ModuleHelpDialog
from mso_studio.launcher.preflight import BootCheck, FAIL, OK, WARN
from mso_studio.launcher.process import LaunchError, LaunchManager, PauseError
from mso_studio.launcher.project_link import ProjectLinkError
from mso_studio.launcher.tiles import FlowLayout, HeroPanel, AppTile
from mso_studio.launcher.workbench_view import WorkbenchPage
from mso_studio.launcher import project_link

ORG_NAME = "mechanism-optimization-suite"
APP_NAME = "mso-launcher"

_FILTER_ALL = "all"
_FILTER_STAGE = "stage"
_FILTER_FAVOURITES = "favourites"
_FILTER_RECENT = "recent"

_MAX_RECENT = 5
_MAX_RECENT_PROJECTS = 5

# Main-panel stack indices.
_STACK_LIBRARY = 0
_STACK_WORKBENCH = 1

# Don't re-query the project database on every 1 Hz poll tick -- SQLite
# reads are cheap but there's no reason to hit disk faster than a human can
# notice a status pill change.
_WORKBENCH_STATUS_REFRESH_INTERVAL_S = 5.0


def _settings_list(value) -> List[str]:
    """Normalize a QSettings stringlist read-back.

    Qt's INI backend collapses a single-element string list to a bare
    scalar string once it round-trips through disk to a fresh QSettings
    instance (confirmed while building this launcher: saving exactly one
    favourite, then reopening, turned ``["idt_sensitivity"]`` into the bare
    string ``"idt_sensitivity"``). Without this guard, ``set(that_string)``
    silently explodes into a set of individual characters instead of the
    one entry it should be -- and the next save then writes that character
    soup back to disk, corrupting favourites/recent permanently.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    return list(value)


class LauncherWindow(QMainWindow):
    def __init__(
        self,
        launch_manager: Optional[LaunchManager] = None,
        boot_results: Optional[List[BootCheck]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"MSO Launcher — v{__version__}")
        self._settings = QSettings(ORG_NAME, APP_NAME)
        self._launch_manager = launch_manager or LaunchManager()
        self._boot_results = boot_results or []
        self._theme_mode = self._settings.value("theme_mode", "system")

        self._favourites = set(_settings_list(self._settings.value("favourites", [])))
        self._recent = _settings_list(self._settings.value("recent", []))

        self._tiles: Dict[str, AppTile] = {}
        self._filter = (_FILTER_ALL, None)
        self._search_text = ""
        self._selected_id: Optional[str] = None
        self._doctor_dialog: Optional[QDialog] = None
        self._help_dialog: Optional[ModuleHelpDialog] = None
        self._entrance_played = False
        self._welcome_checked = False

        # -- project / Workbench state --------------------------------------
        # `self._project` is a live mso_project.Project handle (or None if no
        # project is open); everything that touches it goes through
        # project_link so a problem in that in-flight package degrades
        # gracefully instead of taking the launcher down. Recent projects
        # persist the same way recent/favourites do (see _settings_list).
        self._project = None
        self._project_dir: Optional[str] = None
        self._recent_projects = _settings_list(self._settings.value("recent_projects", []))
        self._last_status_refresh = 0.0

        self._build_ui()
        self._build_menu()
        self._build_shortcuts()
        self._restore_geometry()
        self._apply_theme()
        self._rebuild_tiles()
        self._select_entry(CATALOG[0].id if CATALOG else None)

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(1000)
        self._poll_timer.timeout.connect(self._on_poll)
        self._poll_timer.start()

    # -- layout ---------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget(self)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal, central)
        splitter.addWidget(self._build_sidebar(splitter))

        # Main panel is a stack: index 0 is the existing hero+tile-grid
        # library (untouched, so every test pinned to self.hero/self._tiles/
        # self.grid_layout keeps working exactly as before), index 1 is the
        # Workbench page. Selecting a library filter (stage/search/etc.)
        # switches back to index 0; the sidebar's "Workbench" entry switches
        # to index 1. The two views never fight over sidebar selection
        # because only one of stage_list/the Workbench button drives the
        # stack at a time.
        self.main_stack = QStackedWidget(splitter)
        self.main_stack.addWidget(self._build_main_panel(self.main_stack))

        self.workbench_page = WorkbenchPage(self._theme_mode, self.main_stack)
        self.workbench_page.launchRequested.connect(self._on_launch_requested)
        self.workbench_page.helpRequested.connect(self._open_help)
        self.workbench_page.changeProjectRequested.connect(self._open_project_picker)
        self.main_stack.addWidget(self.workbench_page)

        splitter.addWidget(self.main_stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([260, 1120])
        root.addWidget(splitter, 1)

        root.addWidget(self._build_status_strip(central))

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Ready", 3000)

    def _build_sidebar(self, parent) -> QWidget:
        sidebar = QWidget(parent)
        sidebar.setObjectName("Sidebar")
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        layout.addWidget(self._build_project_section(sidebar))

        self.search_box = QLineEdit(sidebar)
        self.search_box.setObjectName("SearchBox")
        self.search_box.setPlaceholderText("Search modules…")
        self.search_box.textChanged.connect(self._on_search_changed)
        layout.addWidget(self.search_box)

        self.stage_list = QListWidget(sidebar)
        self.stage_list.setObjectName("StageList")
        self.stage_list.currentItemChanged.connect(self._on_stage_changed)
        layout.addWidget(self.stage_list, 1)

        doctor_button = QLabel("⚙  Doctor / Recipes", sidebar)
        doctor_button.setObjectName("Caption")
        doctor_button.setProperty("clickable", True)
        doctor_button.setCursor(Qt.CursorShape.PointingHandCursor)
        doctor_button.mousePressEvent = lambda event: self._open_doctor()  # type: ignore[assignment]
        layout.addWidget(doctor_button)

        self._rebuild_stage_list()
        return sidebar

    def _build_project_section(self, parent) -> QWidget:
        """The PROJECT block above the module library: New/Open project,
        a Workbench switch, and up to 5 recent projects. Separate widget
        (not folded into stage_list) because it drives the main_stack
        instead of the library filter, and its item count changes shape
        (0-5 recents) independently of the fixed stage list."""
        section = QWidget(parent)
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 8)
        layout.setSpacing(2)

        header = QLabel("PROJECT", section)
        header.setObjectName("SidebarSectionLabel")
        layout.addWidget(header)

        self.workbench_nav_label = self._sidebar_action_label(
            section, "▣  Workbench", self._show_workbench)
        layout.addWidget(self.workbench_nav_label)
        layout.addWidget(self._sidebar_action_label(
            section, "＋  New project…", self._new_project))
        layout.addWidget(self._sidebar_action_label(
            section, "▸  Open project…", self._open_project_dialog))

        self.recent_projects_container = QWidget(section)
        self.recent_projects_layout = QVBoxLayout(self.recent_projects_container)
        self.recent_projects_layout.setContentsMargins(0, 0, 0, 0)
        self.recent_projects_layout.setSpacing(2)
        layout.addWidget(self.recent_projects_container)
        self._rebuild_recent_projects()

        return section

    def _sidebar_action_label(self, parent, text: str, on_click) -> QLabel:
        label = QLabel(text, parent)
        label.setObjectName("Caption")
        label.setProperty("clickable", True)
        label.setCursor(Qt.CursorShape.PointingHandCursor)
        label.mousePressEvent = lambda event: on_click()  # type: ignore[assignment]
        return label

    def _rebuild_recent_projects(self) -> None:
        while self.recent_projects_layout.count():
            item = self.recent_projects_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        for path in self._recent_projects[:_MAX_RECENT_PROJECTS]:
            manifest = project_link.read_manifest(path)
            name = (manifest or {}).get("name") or path.rsplit("/", 1)[-1]
            label = self._sidebar_action_label(
                self.recent_projects_container, f"    {name}",
                lambda p=path: self._open_project(p))
            label.setToolTip(path)
            self.recent_projects_layout.addWidget(label)

    def _rebuild_stage_list(self) -> None:
        self.stage_list.blockSignals(True)
        self.stage_list.clear()

        all_item = QListWidgetItem(f"ALL  ({len(CATALOG)})")
        all_item.setData(Qt.ItemDataRole.UserRole, (_FILTER_ALL, None))
        self.stage_list.addItem(all_item)

        for stage_id, label, _from, _to in STAGES:
            count = len(entries_for_stage(stage_id))
            item = QListWidgetItem(f"▸ {label}  ({count})")
            item.setData(Qt.ItemDataRole.UserRole, (_FILTER_STAGE, stage_id))
            self.stage_list.addItem(item)

        fav_item = QListWidgetItem(f"★ Favourites  ({len(self._favourites)})")
        fav_item.setData(Qt.ItemDataRole.UserRole, (_FILTER_FAVOURITES, None))
        self.stage_list.addItem(fav_item)

        recent_item = QListWidgetItem(f"Recent  ({len(self._recent)})")
        recent_item.setData(Qt.ItemDataRole.UserRole, (_FILTER_RECENT, None))
        self.stage_list.addItem(recent_item)

        self.stage_list.blockSignals(False)
        self.stage_list.setCurrentRow(0)

    def _build_main_panel(self, parent) -> QWidget:
        panel = QWidget(parent)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 8)
        layout.setSpacing(14)

        self.hero = HeroPanel(self._theme_mode, panel)
        self.hero.launchRequested.connect(self._on_launch_requested)
        self.hero.stopRequested.connect(self._on_stop_requested)
        self.hero.helpRequested.connect(self._open_help)
        self.hero.pauseRequested.connect(self._on_pause_requested)
        self.hero.resumeMenuRequested.connect(self._on_resume_menu_requested)
        layout.addWidget(self.hero)

        self.grid_scroll = QScrollArea(panel)
        self.grid_scroll.setWidgetResizable(True)
        self.grid_scroll.setFrameShape(QFrame.Shape.NoFrame)

        # BackdropWidget so an optional wallpaper can show in the gaps
        # between tiles -- chrome only, per the "wallpaper never behind
        # text" rule; each AppTile paints its own opaque surface on top.
        self.grid_container = BackdropWidget(tokens_for(self._theme_mode), parent=self.grid_scroll)
        self.grid_layout = FlowLayout(self.grid_container, margin=4, h_spacing=14, v_spacing=14)
        self.grid_scroll.setWidget(self.grid_container)
        layout.addWidget(self.grid_scroll, 1)
        self._apply_scroll_palette()
        self._apply_wallpaper(self._settings.value("wallpaper_id", "none"))

        return panel

    def _build_status_strip(self, parent) -> QWidget:
        strip = QWidget(parent)
        strip.setObjectName("StatusStrip")
        strip.setCursor(Qt.CursorShape.PointingHandCursor)
        layout = QHBoxLayout(strip)
        layout.setContentsMargins(16, 6, 16, 6)
        layout.setSpacing(18)

        self._led_labels: Dict[str, QLabel] = {}
        for check in self._boot_results:
            led = QLabel(f"●  {check.label}", strip)
            led.setObjectName({OK: "LedOk", WARN: "LedWarn", FAIL: "LedFail"}.get(check.state, "LedWarn"))
            led.setToolTip(check.detail)
            layout.addWidget(led)
            self._led_labels[check.label] = led

        layout.addStretch(1)
        self.running_label = QLabel("0 running", strip)
        self.running_label.setObjectName("Caption")
        layout.addWidget(self.running_label)

        strip.mousePressEvent = lambda event: self._open_doctor()  # type: ignore[assignment]
        return strip

    # -- menu ---------------------------------------------------------
    def _build_menu(self) -> None:
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")
        new_project_action = QAction("New Project…", self)
        new_project_action.triggered.connect(self._new_project)
        file_menu.addAction(new_project_action)
        open_project_action = QAction("Open Project…", self)
        open_project_action.triggered.connect(self._open_project_dialog)
        file_menu.addAction(open_project_action)
        self._close_project_action = QAction("Close Project", self)
        self._close_project_action.setEnabled(False)
        self._close_project_action.triggered.connect(self._close_project)
        file_menu.addAction(self._close_project_action)
        file_menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

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
        view_menu.addSeparator()
        preferences_action = QAction("Preferences…", self)
        preferences_action.triggered.connect(self._open_preferences)
        view_menu.addAction(preferences_action)

        help_menu = menu_bar.addMenu("&Help")
        module_help_action = QAction("Module help", self)
        module_help_action.setShortcut("F1")
        module_help_action.triggered.connect(lambda: self._open_help(self._selected_id))
        help_menu.addAction(module_help_action)
        welcome_action = QAction("Welcome tour", self)
        welcome_action.triggered.connect(self._open_welcome)
        help_menu.addAction(welcome_action)
        about_action = QAction("About", self)
        about_action.triggered.connect(self._open_about)
        help_menu.addAction(about_action)

    def _build_shortcuts(self) -> None:
        # F1 is registered once, as the Help menu's "Module help" action
        # shortcut (see _build_menu) -- not duplicated as a QShortcut here,
        # which would otherwise register the same key sequence twice and
        # trigger Qt's "ambiguous shortcut" warning.
        QShortcut(QKeySequence("Ctrl+F"), self, activated=self._focus_search)
        QShortcut(QKeySequence("/"), self, activated=self._focus_search)
        for i, (stage_id, *_rest) in enumerate(STAGES, start=1):
            QShortcut(QKeySequence(f"Ctrl+{i}"), self,
                      activated=lambda sid=stage_id: self._set_filter(_FILTER_STAGE, sid))

    def _focus_search(self) -> None:
        self.search_box.setFocus()
        self.search_box.selectAll()

    # -- tiles / filtering ---------------------------------------------------
    def _rebuild_tiles(self) -> None:
        for tile in self._tiles.values():
            tile.setParent(None)
        self._tiles.clear()

        for entry in CATALOG:
            tile = AppTile(entry, self._theme_mode, self.grid_container)
            tile.selected.connect(self._select_entry)
            tile.activated.connect(self._on_tile_activated)
            tile.favouriteToggled.connect(self._toggle_favourite)
            tile.set_favourite(entry.id in self._favourites)
            tile.set_running(self._launch_manager.is_running(entry.id))
            self._tiles[entry.id] = tile
            self.grid_layout.addWidget(tile)

        self._apply_filter()

    def _visible_entries(self) -> List[AppEntry]:
        kind, value = self._filter
        if kind == _FILTER_STAGE:
            base = entries_for_stage(value)
        elif kind == _FILTER_FAVOURITES:
            base = [e for e in CATALOG if e.id in self._favourites]
        elif kind == _FILTER_RECENT:
            base = [CATALOG_BY_ID[eid] for eid in self._recent if eid in CATALOG_BY_ID]
        else:
            base = list(CATALOG)

        if self._search_text:
            matched_ids = {e.id for e in search(self._search_text)}
            base = [e for e in base if e.id in matched_ids]
        return base

    def _apply_filter(self) -> None:
        visible = {e.id for e in self._visible_entries()}
        for entry_id, tile in self._tiles.items():
            tile.setVisible(entry_id in visible)
        self._play_entrance_once()

    def _on_search_changed(self, text: str) -> None:
        self._search_text = text.strip()
        self._apply_filter()

    def _on_stage_changed(self, current: Optional[QListWidgetItem], _previous) -> None:
        if current is None:
            return
        # Any library filter selection means the user wants the tile grid,
        # not the Workbench -- switch back so the two views never fight
        # over what the sidebar is currently driving. Guarded because the
        # sidebar (and its initial stage_list.setCurrentRow(0), which fires
        # this handler) is built in _build_sidebar(), before main_stack
        # exists -- _build_ui() constructs the sidebar first, then the
        # stacked main panel.
        if hasattr(self, "main_stack"):
            self.main_stack.setCurrentIndex(_STACK_LIBRARY)
        kind, value = current.data(Qt.ItemDataRole.UserRole)
        self._filter = (kind, value)
        self._apply_filter()

    def _set_filter(self, kind: str, value) -> None:
        for i in range(self.stage_list.count()):
            item = self.stage_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == (kind, value):
                self.stage_list.setCurrentItem(item)
                return

    # -- entrance animation -----------------------------------------------------
    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        # Tiles report isVisible()==False until the window itself has
        # actually been shown, so _apply_filter()'s calls to
        # _play_entrance_once() during __init__ are always no-ops (every
        # tile skipped) -- this is the one call that happens after show(),
        # so the entrance animation actually plays.
        super().showEvent(event)
        self._play_entrance_once()
        if not self._welcome_checked:
            self._welcome_checked = True
            # Deferred to the next event-loop turn: showEvent can fire more
            # than once (e.g. window state changes), and starting a modal
            # QDialog.exec() synchronously from inside showEvent nests an
            # event loop before this window has finished painting.
            #
            # Never scheduled under the offscreen platform: there is no
            # real display and no user able to dismiss a modal dialog, so
            # an unmocked _maybe_show_welcome_on_first_run() -> _open_welcome()
            # -> WelcomeDialog(...).exec() firing from this deferred timer
            # blocks forever. This platform is used exclusively for
            # automated GUI tests (every test invocation in this repo sets
            # QT_QPA_PLATFORM=offscreen); a real interactive user never
            # runs under it. Guarded HERE (not inside
            # _maybe_show_welcome_on_first_run or _open_welcome themselves)
            # because both of those are real, directly-tested business
            # logic several existing tests exercise on purpose (some via a
            # mocked _open_welcome, one via a mocked WelcomeDialog.exec) --
            # guarding either of those methods broke those tests, since an
            # unconditional early-return runs before any mock even gets a
            # chance to record the call. This is the one call site that's
            # genuinely automatic/unattended (fires from a generic
            # app.processEvents() call during ANY test's teardown, not from
            # anything the test itself asked for), so it's the only one
            # that needs the guard.
            #
            # Round 4 (WORKBENCH_AUDIT_2026-08-10.md): found via a genuine
            # ~1-hour full-suite hang, root-caused with
            # faulthandler.dump_traceback_later to exactly this path.
            # Separate from (and not covered by) the earlier Round-2 B10
            # fix in tests/conftest.py, which only guarded the
            # closeEvent-triggered path (switching that fixture's cleanup
            # from .close() to .hide()) -- this path fires non-
            # deterministically depending on exact event-loop timing,
            # which is why it manifested as an intermittent hang on a
            # different test than the one Round 2 investigated, not a
            # reliably reproducible per-test failure.
            if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
                QTimer.singleShot(0, self._maybe_show_welcome_on_first_run)

    def _play_entrance_once(self) -> None:
        if self._entrance_played:
            return
        self._entrance_played = True
        delay = 0
        for tile in self._tiles.values():
            if not tile.isVisible():
                continue
            tile.play_entrance(delay)
            delay = min(delay + 20, 400)

    # -- selection / launch ---------------------------------------------------
    def _select_entry(self, entry_id: Optional[str]) -> None:
        if entry_id is None:
            return
        self._selected_id = entry_id
        entry = CATALOG_BY_ID.get(entry_id)
        last_run = self._settings.value(f"last_run/{entry_id}", "")
        last_run_text = f"Last run: {last_run}" if last_run else "Never run"
        self.hero.set_entry(entry, last_run_text)
        self.hero.set_running(self._launch_manager.is_running(entry_id))
        self.hero.set_paused(self._launch_manager.is_paused(entry_id))

    def _on_tile_activated(self, entry_id: str) -> None:
        self._select_entry(entry_id)
        self._on_launch_requested(entry_id)

    def _on_launch_requested(self, entry_id: str) -> None:
        entry = CATALOG_BY_ID.get(entry_id)
        if entry is None:
            return
        if entry.runner == RUNNER_INTERNAL:
            # Doctor/Recipes-style entries (see HeroPanel.set_running's
            # OPEN-button branch) have no subprocess to launch -- route to
            # the in-window panel instead of LaunchManager.launch(), which
            # would always raise LaunchError for these.
            self._open_doctor()
            return
        try:
            self._launch_manager.launch(entry)
        except LaunchError as exc:
            QMessageBox.warning(self, "Launch failed", str(exc))
            return
        self._record_recent(entry_id)
        self._refresh_running_state()
        self.statusBar().showMessage(f"Launched {entry.title}", 4000)

    def _on_stop_requested(self, entry_id: str) -> None:
        self._launch_manager.stop(entry_id)
        self._refresh_running_state()
        entry = CATALOG_BY_ID.get(entry_id)
        if entry is not None:
            self.statusBar().showMessage(f"Stopped {entry.title}", 4000)

    def _on_pause_requested(self, entry_id: str) -> None:
        """Phase 4: SIGSTOP the running app and check out a fresh
        mso-pause/* branch worktree for editing -- see
        mso_workbench.git_pause for the git mechanics."""
        entry = CATALOG_BY_ID.get(entry_id)
        try:
            session = self._launch_manager.pause_for_edit(entry_id)
        except PauseError as exc:
            QMessageBox.warning(self, "Pause failed", str(exc))
            return
        self._refresh_running_state()
        title = entry.title if entry is not None else entry_id
        QMessageBox.information(
            self, f"{title} paused",
            f"Paused. Edit the code checked out at:\n\n{session.worktree_path}\n\n"
            f"(branch {session.branch})\n\n"
            "When you're done, click Resume… on this panel and choose "
            "whether to resume with your edits or leave the run unchanged.",
        )
        self.statusBar().showMessage(f"Paused {title} — editing on {session.branch}", 6000)

    def _ask_resume_flavor(self, title: str, session) -> str:
        """Returns "edited" / "unchanged" / "cancel". Split out from
        _on_resume_menu_requested so tests can drive the decision directly
        instead of simulating a click on a modal QMessageBox."""
        box = QMessageBox(self)
        box.setWindowTitle(f"Resume {title}")
        text = "How should this run resume?"
        if session is not None:
            text += f"\n\nEdit branch: {session.branch}\nWorktree: {session.worktree_path}"
        box.setText(text)
        edited_button = box.addButton("Resume with edited code", QMessageBox.ButtonRole.AcceptRole)
        unchanged_button = box.addButton("Resume unchanged", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked is edited_button:
            return "edited"
        if clicked is unchanged_button:
            return "unchanged"
        return "cancel"

    def _on_resume_menu_requested(self, entry_id: str) -> None:
        entry = CATALOG_BY_ID.get(entry_id)
        title = entry.title if entry is not None else entry_id
        session = self._launch_manager.pause_session_for(entry_id)

        flavor = self._ask_resume_flavor(title, session)
        try:
            if flavor == "edited":
                self._launch_manager.resume_with_new_code(entry_id)
                self.statusBar().showMessage(f"Resumed {title} with the edited code", 6000)
            elif flavor == "unchanged":
                self._launch_manager.resume_unchanged(entry_id)
                self.statusBar().showMessage(f"Resumed {title} unchanged", 6000)
            else:
                return
        except (PauseError, LaunchError) as exc:
            QMessageBox.warning(self, "Resume failed", str(exc))
            return
        self._refresh_running_state()

    def _record_recent(self, entry_id: str) -> None:
        import datetime

        self._settings.setValue(
            f"last_run/{entry_id}", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        )
        self._recent = [entry_id] + [e for e in self._recent if e != entry_id]
        self._recent = self._recent[:_MAX_RECENT]
        self._settings.setValue("recent", self._recent)
        self._rebuild_stage_list()

    def _toggle_favourite(self, entry_id: str) -> None:
        if entry_id in self._favourites:
            self._favourites.discard(entry_id)
        else:
            self._favourites.add(entry_id)
        self._settings.setValue("favourites", sorted(self._favourites))
        # The window is the single source of truth for favourite state
        # (persisted in QSettings); push it back to the tile here rather
        # than letting AppTile flip its own star on click -- that way this
        # method stays correct no matter what triggers a toggle.
        tile = self._tiles.get(entry_id)
        if tile is not None:
            tile.set_favourite(entry_id in self._favourites)
        self._rebuild_stage_list()
        if self._filter[0] == _FILTER_FAVOURITES:
            self._apply_filter()

    # -- polling -----------------------------------------------------------
    _QUICK_DEATH_THRESHOLD_S = 3.0

    def _on_poll(self) -> None:
        finished = self._launch_manager.poll()
        for entry_id in finished:
            uptime = self._launch_manager.last_uptime_s(entry_id)
            entry = CATALOG_BY_ID.get(entry_id)
            if entry is None or uptime is None or uptime >= self._QUICK_DEATH_THRESHOLD_S:
                continue
            # missing_prerequisite() can't detect a present-but-broken
            # interpreter (e.g. .venv-gui exists but PySide6 got removed
            # from it) -- Popen succeeds regardless, only the child's own
            # startup code then fails. Exiting within a couple seconds of
            # launch is the best available signal of that, so surface it
            # rather than let the RUNNING badge just quietly disappear.
            self.statusBar().showMessage(
                f"{entry.title} exited after only {uptime:.1f}s — it may have crashed on startup. "
                "Check Doctor / Recipes if this keeps happening.",
                8000,
            )
        self._refresh_running_state()
        self._maybe_refresh_workbench_status()

    def _maybe_refresh_workbench_status(self) -> None:
        if self._project is None:
            return
        now = time.monotonic()
        if now - self._last_status_refresh < _WORKBENCH_STATUS_REFRESH_INTERVAL_S:
            return
        self._last_status_refresh = now
        self.workbench_page.refresh_status(self._current_project_runs())

    def _refresh_running_state(self) -> None:
        running_ids = set(self._launch_manager.running_ids())
        for entry_id, tile in self._tiles.items():
            tile.set_running(entry_id in running_ids, self._launch_manager.uptime_s(entry_id))
        if self._selected_id is not None:
            self.hero.set_running(self._selected_id in running_ids)
            self.hero.set_paused(self._launch_manager.is_paused(self._selected_id))
        self.running_label.setText(f"{len(running_ids)} running")

    # -- project / Workbench -------------------------------------------------
    def _show_workbench(self) -> None:
        self.main_stack.setCurrentIndex(_STACK_WORKBENCH)

    def _open_project_picker(self) -> None:
        """Entry point for the Workbench page's "Open / New Project…"
        button -- offers the same two choices as the File menu items,
        without duplicating their logic. A small QMenu rather than
        QMessageBox because Qt's StandardButton set has no "New" button to
        pair with "Open"."""
        from PySide6.QtWidgets import QMenu

        menu = QMenu(self)
        menu.addAction("Open existing project…", self._open_project_dialog)
        menu.addAction("New project…", self._new_project)
        menu.exec(self.workbench_page.change_project_button.mapToGlobal(
            self.workbench_page.change_project_button.rect().bottomLeft()))

    def _new_project(self) -> None:
        available, reason = project_link.is_available()
        if not available:
            QMessageBox.warning(
                self, "Project features unavailable",
                f"Project storage (mso_project) could not be loaded:\n{reason}",
            )
            return
        directory = QFileDialog.getExistingDirectory(self, "Choose a location for the new project")
        if not directory:
            return
        name, ok = QInputDialog.getText(self, "New Project", "Project name:")
        if not ok or not name.strip():
            return
        try:
            project = project_link.create_project(directory, name=name.strip())
        except ProjectLinkError as exc:
            QMessageBox.warning(self, "Could not create project", str(exc))
            return
        self._set_active_project(directory, project)

    def _open_project_dialog(self) -> None:
        available, reason = project_link.is_available()
        if not available:
            QMessageBox.warning(
                self, "Project features unavailable",
                f"Project storage (mso_project) could not be loaded:\n{reason}",
            )
            return
        directory = QFileDialog.getExistingDirectory(self, "Open Project")
        if not directory:
            return
        self._open_project(directory)

    def _open_project(self, directory: str) -> None:
        if not project_link.is_project_dir(directory):
            QMessageBox.warning(
                self, "Not a project",
                f"No project.yaml found at:\n{directory}",
            )
            return
        try:
            project = project_link.open_project(directory)
        except ProjectLinkError as exc:
            QMessageBox.warning(self, "Could not open project", str(exc))
            return
        self._set_active_project(directory, project)

    def _set_active_project(self, directory: str, project) -> None:
        self._project = project
        self._project_dir = directory
        name = (getattr(project, "info", None) or {}).get("name", "")
        self.workbench_page.set_project(directory, name, run_lookup=self._current_project_runs)
        self._close_project_action.setEnabled(True)
        self._record_recent_project(directory)
        self._last_status_refresh = 0.0  # force an immediate status refresh
        self._show_workbench()
        self.statusBar().showMessage(f"Opened project: {name or directory}", 4000)

    def _close_project(self) -> None:
        self._project = None
        self._project_dir = None
        self._close_project_action.setEnabled(False)
        self.workbench_page.set_project(None)

    def _current_project_runs(self) -> List[Dict]:
        if self._project is None:
            return []
        return project_link.list_runs(self._project)

    def _record_recent_project(self, directory: str) -> None:
        self._recent_projects = [directory] + [p for p in self._recent_projects if p != directory]
        self._recent_projects = self._recent_projects[:_MAX_RECENT_PROJECTS]
        self._settings.setValue("recent_projects", self._recent_projects)
        self._rebuild_recent_projects()

    # -- module help --------------------------------------------------------
    def _open_help(self, module_id: Optional[str]) -> None:
        if self._help_dialog is None:
            self._help_dialog = ModuleHelpDialog(self)
        if module_id is not None:
            self._help_dialog.show_module(module_id)
        self._help_dialog.show()
        self._help_dialog.raise_()
        self._help_dialog.activateWindow()

    # -- doctor -----------------------------------------------------------
    def _open_doctor(self) -> None:
        if self._doctor_dialog is None:
            dialog = QDialog(self)
            dialog.setWindowTitle("Doctor / Recipes")
            dialog.resize(760, 520)
            layout = QVBoxLayout(dialog)
            layout.addWidget(DoctorPanel(dialog))
            self._doctor_dialog = dialog
        self._doctor_dialog.show()
        self._doctor_dialog.raise_()
        self._doctor_dialog.activateWindow()

    # -- preferences -----------------------------------------------------
    def _open_preferences(self) -> None:
        from mso_studio.gui.preferences import PreferencesDialog

        dialog = PreferencesDialog(self)
        dialog.wallpaperChanged.connect(self._apply_wallpaper)
        if dialog.exec():
            # Preferences persists theme_mode itself; keep this window's
            # cached copy (drives the Theme menu's checkmarks + tile art)
            # in sync so a later _set_theme() call doesn't clobber it.
            self._theme_mode = self._settings.value("theme_mode", self._theme_mode)
            self._apply_scroll_palette()
            self.grid_container.set_tokens(tokens_for(self._theme_mode))
            self._rebuild_tiles()

    # -- theme ---------------------------------------------------------
    def _set_theme(self, mode: str) -> None:
        self._theme_mode = mode
        self._settings.setValue("theme_mode", mode)
        self._apply_theme()
        self._apply_scroll_palette()
        self.grid_container.set_tokens(tokens_for(mode))
        from mso_studio.launcher.art import clear_cache
        clear_cache()
        self._rebuild_tiles()
        if self._selected_id:
            self._select_entry(self._selected_id)

    def _apply_wallpaper(self, wallpaper_id: str) -> None:
        """*wallpaper_id* is one of the shipped pack's ids, an absolute
        path to a user-supplied image, or "none". Persisted so it survives
        a restart; called at startup and from PreferencesDialog."""
        self._settings.setValue("wallpaper_id", wallpaper_id)
        self.grid_container.set_wallpaper(resolve_wallpaper_path(wallpaper_id))

    def _apply_theme(self) -> None:
        app = QApplication.instance()
        if app is not None:
            theme.apply_theme(app, self._theme_mode)

    def _apply_scroll_palette(self) -> None:
        # QAbstractScrollArea's internal viewport paints with the
        # QPalette Base/Window color and doesn't reliably pick up the
        # "background: transparent" QSS rule for QWidget across Qt
        # versions/styles -- setting the palette directly on the
        # scroll area, its viewport, and the tile-grid container is the
        # robust fix, so the theme's background shows through instead of
        # a bright rectangle behind the tile grid.
        bg = QColor(theme.palette_for(self._theme_mode)["bg"])
        for widget in (self.grid_scroll, self.grid_scroll.viewport(), self.grid_container):
            palette = widget.palette()
            palette.setColor(QPalette.ColorRole.Window, bg)
            palette.setColor(QPalette.ColorRole.Base, bg)
            widget.setPalette(palette)
            widget.setAutoFillBackground(True)

    def _open_about(self) -> None:
        QMessageBox.about(
            self, "About MSO Launcher",
            f"<b>MSO Launcher</b> v{__version__}<br>"
            "Game-style entry point for the Mechanism Optimization Suite.<br><br>"
            f"{len(CATALOG)} modules catalogued.",
        )

    def _open_welcome(self) -> None:
        from mso_studio.launcher.welcome import WelcomeDialog
        WelcomeDialog(self._theme_mode, self).exec()

    def _maybe_show_welcome_on_first_run(self) -> None:
        if self._settings.value("welcome_seen", False, type=bool):
            return
        self._settings.setValue("welcome_seen", True)
        self._open_welcome()

    # -- geometry / shutdown -----------------------------------------------------
    def _restore_geometry(self) -> None:
        geometry = self._settings.value("window_geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        else:
            self.resize(1180, 760)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt override
        running = self._launch_manager.running_ids()
        if running:
            titles = ", ".join(CATALOG_BY_ID[eid].title for eid in running if eid in CATALOG_BY_ID)
            reply = QMessageBox.question(
                self, "Apps still running",
                f"{len(running)} app(s) still running ({titles}).\n\n"
                "Leave them running when the launcher closes?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.No:
                self._launch_manager.stop_all()
        self._settings.setValue("window_geometry", self.saveGeometry())
        super().closeEvent(event)
