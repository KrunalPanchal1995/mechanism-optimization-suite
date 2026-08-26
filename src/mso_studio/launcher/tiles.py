"""
mso_studio.launcher.tiles
============================
The Steam-library visual pieces: a reflowing tile grid layout, the tile
widget itself (artwork + title + running badge + favourite star, with a
hover lift/glow), and the hero panel above the grid (big art, description,
the LAUNCH/STOP button, and a details drawer built on the existing
``CollapsibleSection`` from ``mso_studio.gui.widgets.module_page``).
"""
from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import (
    QEasingCurve, QPropertyAnimation, QRect, QSize, Qt, QTimer, Property, Signal,
)
from PySide6.QtGui import QColor, QCursor
from PySide6.QtWidgets import (
    QFrame, QGraphicsDropShadowEffect, QGraphicsOpacityEffect, QHBoxLayout,
    QLabel, QLayout, QLayoutItem, QPushButton, QSizePolicy, QVBoxLayout,
    QWidget, QWidgetItem,
)

from mso_studio.gui.theme import palette_for
from mso_studio.gui.widgets.module_page import CollapsibleSection
from mso_studio.launcher.art import tile_art
from mso_studio.launcher.catalog import AppEntry, RUNNER_INTERNAL, STAGE_LABELS
from mso_studio.launcher.process import build_argv, missing_prerequisite

# Sized so title/stage/description are legible at a glance (Ansys/STAR-CCM+
# module-tile scale) rather than requiring a hover or click to find out what
# a module does. TILE_HEIGHT is a real fixed height (see AppTile.sizeHint
# below) -- previously only width was fixed, so tiles with longer titles
# wrapped to more lines and rows went ragged; a uniform height fixes that
# and is part of what makes the larger tiles read as a deliberate grid
# rather than just "the same tiles, bigger."
TILE_WIDTH = 260
TILE_ART_HEIGHT = 132
TILE_HEIGHT = 300


class FlowLayout(QLayout):
    """Standard reflowing layout (Qt's documented FlowLayout example,
    adapted): wraps child widgets left-to-right, top-to-bottom, so the tile
    grid re-packs itself as the window is resized instead of clipping or
    scrolling horizontally."""

    def __init__(self, parent=None, margin: int = 0, h_spacing: int = 12, v_spacing: int = 12):
        super().__init__(parent)
        self._items: List[QLayoutItem] = []
        self._h_spacing = h_spacing
        self._v_spacing = v_spacing
        self.setContentsMargins(margin, margin, margin, margin)

    def addItem(self, item: QLayoutItem) -> None:  # noqa: N802 - Qt override
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> Optional[QLayoutItem]:  # noqa: N802
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int) -> Optional[QLayoutItem]:  # noqa: N802
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self) -> Qt.Orientations:  # noqa: N802
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:  # noqa: N802
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QSize(margins.left() + margins.right(), margins.top() + margins.bottom())
        return size

    def _do_layout(self, rect: QRect, test_only: bool) -> int:
        margins = self.contentsMargins()
        effective = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        x, y = effective.x(), effective.y()
        line_height = 0

        for item in self._items:
            widget = item.widget()
            space_x, space_y = self._h_spacing, self._v_spacing
            next_x = x + item.sizeHint().width() + space_x
            if next_x - space_x > effective.right() and line_height > 0:
                x = effective.x()
                y = y + line_height + space_y
                next_x = x + item.sizeHint().width() + space_x
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(x, y, item.sizeHint().width(), item.sizeHint().height()))
            x = next_x
            line_height = max(line_height, item.sizeHint().height())

        return y + line_height - rect.y() + margins.bottom()


class AppTile(QFrame):
    """One catalog entry in the grid: artwork, title, tagline, favourite
    star, and a live RUNNING badge. Hover lifts the tile and brightens its
    shadow toward the entry's stage accent -- hover is purely visual and
    never changes what's selected (a prior version emitted ``selected`` on
    hover too, which meant moving the mouse toward the hero panel's LAUNCH
    button silently retargeted it to whatever tile the cursor last crossed;
    fixed by making selection click/focus-driven only)."""

    activated = Signal(str)          # double-click / Enter / Space -> launch
    selected = Signal(str)           # click or keyboard focus -> show in hero panel
    favouriteToggled = Signal(str)

    def __init__(self, entry: AppEntry, mode: str = "dark", parent=None):
        super().__init__(parent)
        self.entry = entry
        self._mode = mode
        self._favourite = False
        self._lift = 0.0

        self.setObjectName("AppTile")
        self.setFixedSize(TILE_WIDTH, TILE_HEIGHT)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setProperty("running", False)
        # Tab-focusable + click-focusable, so keyboard users can reach every
        # tile (the `activated` signal above always claimed "Enter -> launch"
        # but nothing made tiles focusable or handled key events until now).
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 10)
        layout.setSpacing(4)

        self.art_label = QLabel(self)
        self.art_label.setFixedSize(TILE_WIDTH, TILE_ART_HEIGHT)
        self.art_label.setPixmap(
            tile_art(entry, TILE_WIDTH, TILE_ART_HEIGHT, dpr=self.devicePixelRatioF()))
        layout.addWidget(self.art_label)

        self.badge_label = QLabel("", self.art_label)
        self.badge_label.setObjectName("RunningBadge")
        self.badge_label.move(10, 8)
        self.badge_label.hide()

        self.star_label = QLabel("☆", self.art_label)
        self.star_label.setObjectName("FavouriteStar")
        self.star_label.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.star_label.setFixedSize(24, 24)
        self.star_label.move(TILE_WIDTH - 32, 6)
        self.star_label.mousePressEvent = self._on_star_clicked  # type: ignore[assignment]

        title = QLabel(entry.title, self)
        title.setObjectName("TileTitle")
        title.setWordWrap(True)
        title.setContentsMargins(10, 8, 10, 0)
        layout.addWidget(title)

        stage = QLabel(STAGE_LABELS.get(entry.stage, entry.stage).upper(), self)
        stage.setObjectName("TileStage")
        stage.setContentsMargins(10, 0, 10, 0)
        layout.addWidget(stage)

        # The longer `description` (not the short tile `tagline`) so a bigger
        # tile actually earns its size -- title + stage + a real sentence of
        # what the module does, legible without a hover or click. `tagline`
        # remains the short string used in HeroPanel/search; only the tile
        # body text changed. objectName stays TileTagline so no QSS rule is
        # orphaned by this swap.
        body = QLabel(entry.description, self)
        body.setObjectName("TileTagline")
        body.setWordWrap(True)
        body.setContentsMargins(10, 2, 10, 0)
        layout.addWidget(body, 1)

        self._shadow: Optional[QGraphicsDropShadowEffect] = None
        self._install_shadow()

        self._lift_anim = QPropertyAnimation(self, b"lift", self)
        self._lift_anim.setDuration(140)
        self._lift_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._entrance_anim: Optional[QPropertyAnimation] = None

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return QSize(TILE_WIDTH, TILE_HEIGHT)

    # -- running / favourite state -----------------------------------------
    def set_running(self, running: bool, uptime_s: Optional[float] = None) -> None:
        self.setProperty("running", running)
        self.style().unpolish(self)
        self.style().polish(self)
        if running:
            mins = int((uptime_s or 0) // 60)
            secs = int((uptime_s or 0) % 60)
            self.badge_label.setText(f"● RUNNING · {mins}m{secs:02d}s")
            self.badge_label.adjustSize()
            self.badge_label.show()
        else:
            self.badge_label.hide()

    def set_favourite(self, favourite: bool) -> None:
        self._favourite = favourite
        self.star_label.setText("★" if favourite else "☆")

    def _on_star_clicked(self, event) -> None:
        # Only emit -- don't flip the star locally. The window owns
        # favourites (persisted in QSettings) and is the single source of
        # truth; it calls set_favourite() back once the toggle is applied,
        # so the same code path updates the tile whether the toggle came
        # from clicking this star or from anywhere else in the window.
        self.favouriteToggled.emit(self.entry.id)

    # -- hover lift/glow -----------------------------------------------------
    def _install_shadow(self) -> None:
        """(Re)install the hover drop-shadow as this tile's graphics effect.

        Qt widgets support exactly one QGraphicsEffect at a time, so
        anything that temporarily swaps in a different effect (currently
        only ``play_entrance`` below, for the one-shot fade-in) must call
        this again afterward -- otherwise ``self._shadow`` is left pointing
        at a C++ object Qt already deleted when the effect was replaced,
        and the next hover crashes with "Internal C++ object already
        deleted" inside ``set_lift``.
        """
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(0)
        shadow.setOffset(0, 0)
        shadow.setColor(QColor(0, 0, 0, 0))
        self._shadow = shadow
        self.setGraphicsEffect(shadow)

    def get_lift(self) -> float:
        return self._lift

    def set_lift(self, value: float) -> None:
        self._lift = value
        if self._shadow is None:
            return  # mid-swap during play_entrance(); nothing to update yet
        self._shadow.setBlurRadius(4 + value * 4)
        self._shadow.setOffset(0, value * 0.6)
        palette = palette_for(self._mode)
        color = QColor(palette["glow"])
        color.setAlpha(int(min(160, value * 24)))
        self._shadow.setColor(color)

    lift = Property(float, get_lift, set_lift)

    def play_entrance(self, delay_ms: int, duration_ms: int = 260) -> None:
        """One-shot staggered fade-in, played once when the grid first
        becomes visible (see LauncherWindow.showEvent). Temporarily swaps
        the hover drop-shadow for a QGraphicsOpacityEffect and reinstalls a
        fresh shadow once the fade completes -- see _install_shadow's
        docstring for why the swap-back is required."""
        self._shadow = None
        effect = QGraphicsOpacityEffect(self)
        effect.setOpacity(0.0)
        self.setGraphicsEffect(effect)

        anim = QPropertyAnimation(effect, b"opacity", self)
        anim.setDuration(duration_ms)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.finished.connect(self._install_shadow)
        QTimer.singleShot(delay_ms, anim.start)
        self._entrance_anim = anim  # keep alive until it finishes

    def enterEvent(self, event) -> None:  # noqa: N802 - Qt override
        # Hover is purely visual (lift + glow) -- it must NOT change
        # self.selected, or moving the pointer toward the hero panel's
        # LAUNCH button (which necessarily crosses other tiles) silently
        # retargets what LAUNCH will act on. See the class docstring.
        self._lift_anim.stop()
        self._lift_anim.setStartValue(self._lift)
        self._lift_anim.setEndValue(6.0)
        self._lift_anim.start()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._lift_anim.stop()
        self._lift_anim.setStartValue(self._lift)
        self._lift_anim.setEndValue(0.0)
        self._lift_anim.start()
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        self.selected.emit(self.entry.id)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt override
        self.activated.emit(self.entry.id)
        super().mouseDoubleClickEvent(event)

    def focusInEvent(self, event) -> None:  # noqa: N802 - Qt override
        # Keyboard focus is an explicit user action (Tab / click-to-focus),
        # unlike hover -- treating it the same as a click keeps the tile
        # fully usable without a mouse: Tab through tiles, hero panel
        # follows focus, Enter/Space launches (see keyPressEvent).
        self.selected.emit(self.entry.id)
        super().focusInEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.activated.emit(self.entry.id)
            return
        super().keyPressEvent(event)


class HeroPanel(QFrame):
    """Large "now selected" panel: art, title, description, LAUNCH/STOP, and
    a details drawer with the exact command that will run."""

    launchRequested = Signal(str)
    stopRequested = Signal(str)
    helpRequested = Signal(str)
    pauseRequested = Signal(str)       # Phase 4: pause & check out an edit branch
    resumeMenuRequested = Signal(str)  # Phase 4: ask how to resume (window owns the dialog)

    def __init__(self, mode: str = "dark", parent=None):
        super().__init__(parent)
        self.setObjectName("HeroPanel")
        self._mode = mode
        self._entry: Optional[AppEntry] = None
        self._running = False
        self._paused = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 12)
        layout.setSpacing(0)

        self.art_label = QLabel(self)
        self.art_label.setFixedHeight(180)
        self.art_label.setScaledContents(True)
        # Ignored horizontal policy: the pixmap is regenerated at the
        # panel's current width in resizeEvent below, so the label's own
        # sizeHint must never feed back into the parent's requested size --
        # without this, setPixmap(wider pixmap) -> larger sizeHint ->
        # layout grows HeroPanel -> resizeEvent fires again -> infinite
        # relayout loop (each generating a fresh pixmap every frame).
        self.art_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.art_label)
        self._art_width = -1
        self._art_entry_id: Optional[str] = None

        body = QVBoxLayout()
        body.setContentsMargins(20, 14, 20, 0)
        body.setSpacing(4)

        self.stage_label = QLabel("", self)
        self.stage_label.setObjectName("Caption")
        body.addWidget(self.stage_label)

        self.title_label = QLabel("Select a module", self)
        self.title_label.setObjectName("HeroTitle")
        body.addWidget(self.title_label)

        self.description_label = QLabel("", self)
        self.description_label.setObjectName("Caption")
        self.description_label.setWordWrap(True)
        body.addWidget(self.description_label)

        button_row = QHBoxLayout()
        button_row.setSpacing(10)
        button_row.setContentsMargins(0, 12, 0, 0)

        self.launch_button = QPushButton("▶  LAUNCH", self)
        self.launch_button.setObjectName("PrimaryAction")
        self.launch_button.setMinimumHeight(44)
        self.launch_button.setMinimumWidth(160)
        self.launch_button.clicked.connect(self._on_launch_clicked)
        button_row.addWidget(self.launch_button)

        self.details_toggle = QPushButton("ⓘ  Details", self)
        self.details_toggle.setCheckable(True)
        self.details_toggle.clicked.connect(self._on_details_toggled)
        button_row.addWidget(self.details_toggle)

        # Phase 4: pause a running app, edit its code on a fresh git-branch
        # worktree, then resume unchanged or resume using the edit. Hidden
        # unless the entry is actually running (see set_running/set_paused).
        self.pause_button = QPushButton("⏸  Pause && edit", self)
        self.pause_button.clicked.connect(self._on_pause_clicked)
        self.pause_button.hide()
        button_row.addWidget(self.pause_button)

        self.last_run_label = QLabel("", self)
        self.last_run_label.setObjectName("Caption")
        button_row.addWidget(self.last_run_label)
        button_row.addStretch(1)
        body.addLayout(button_row)

        self.details = CollapsibleSection("", self)
        self.details.toggle.hide()
        self.details.content.setVisible(False)
        body.addWidget(self.details)

        layout.addLayout(body)

        self.set_entry(None)

    # -- state -----------------------------------------------------------
    def set_entry(self, entry: Optional[AppEntry], last_run_text: str = "") -> None:
        self._entry = entry
        if entry is None:
            self.art_label.clear()
            self._art_width = -1
            self._art_entry_id = None
            self.stage_label.setText("")
            self.title_label.setText("Select a module on the left to get started")
            self.description_label.setText("")
            self.launch_button.setEnabled(False)
            self.last_run_label.setText("")
            self.pause_button.hide()
            self._clear_details()
            return

        self._update_art()
        self.stage_label.setText(STAGE_LABELS.get(entry.stage, entry.stage).upper())
        self.title_label.setText(entry.title)
        self.description_label.setText(entry.description)
        self.last_run_label.setText(last_run_text)
        self.set_running(self._running)
        self._render_details()

    def set_paused(self, paused: bool) -> None:
        self._paused = paused
        self._refresh_pause_button()

    def _refresh_pause_button(self) -> None:
        if self._entry is None or self._entry.runner == RUNNER_INTERNAL or not self._running:
            self.pause_button.hide()
            return
        self.pause_button.show()
        if self._paused:
            self.pause_button.setText("▶  Resume…")
        else:
            self.pause_button.setText("⏸  Pause && edit")

    def set_running(self, running: bool) -> None:
        self._running = running
        if self._entry is None:
            return
        self._refresh_pause_button()
        if self._entry.runner == RUNNER_INTERNAL:
            # Doctor/Recipes-style entries have no subprocess to run or
            # stop -- they open a panel in this window instead. Previously
            # this button stayed labelled LAUNCH and enabled for these too,
            # so clicking it always raised a "no subprocess to launch"
            # warning; OPEN routes to LauncherWindow._open_doctor() instead
            # (see _on_launch_requested there).
            self.launch_button.setText("⚙  OPEN")
            self.launch_button.setObjectName("PrimaryAction")
        elif running:
            self.launch_button.setText("■  STOP")
            self.launch_button.setObjectName("DangerAction")
        else:
            self.launch_button.setText("▶  LAUNCH")
            self.launch_button.setObjectName("PrimaryAction")
        self.launch_button.setEnabled(True)
        self._refresh_style(self.launch_button)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._update_art()
        super().resizeEvent(event)

    def _update_art(self) -> None:
        if self._entry is None:
            return
        width = max(self.width(), 200)
        entry_changed = self._entry.id != self._art_entry_id
        # Guard against regenerating (and re-setting, which would re-trigger
        # a resize under some layouts) the same pixmap every frame -- but
        # the guard must key on the ENTRY too, not just width: two
        # different entries selected back to back at the same window width
        # both hash to "width unchanged", which used to skip repainting the
        # art entirely -- title/description would update to the new entry
        # while the artwork silently kept showing the previous one.
        if width == self._art_width and not entry_changed:
            return
        self._art_width = width
        self._art_entry_id = self._entry.id
        self.art_label.setPixmap(
            tile_art(self._entry, width, 180, dpr=self.devicePixelRatioF()))
        if entry_changed:
            self._play_art_reveal()

    def _play_art_reveal(self) -> None:
        """One-shot cross-fade when the hero's art actually changes (a
        different entry selected) -- not on every resize, and not a
        looping animation (this launcher deliberately avoids perpetual
        background animation so it costs ~0 CPU at idle)."""
        effect = QGraphicsOpacityEffect(self.art_label)
        self.art_label.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity", self.art_label)
        anim.setDuration(220)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.finished.connect(lambda: self.art_label.setGraphicsEffect(None))
        anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
        self._art_anim = anim  # keep alive until it finishes

    def _refresh_style(self, widget: QWidget) -> None:
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def _clear_details(self) -> None:
        while self.details.content_layout.rowCount():
            self.details.content_layout.removeRow(0)

    def _render_details(self) -> None:
        self._clear_details()
        entry = self._entry
        if entry is None:
            return
        self.details.content_layout.addRow("Script", QLabel(entry.script or "(none)"))
        try:
            argv = " ".join(build_argv(entry)) if entry.script else "(runs in-window)"
        except ValueError:
            argv = "(runs in-window)"
        argv_label = QLabel(argv)
        argv_label.setWordWrap(True)
        self.details.content_layout.addRow("Command", argv_label)
        if entry.runner == RUNNER_INTERNAL:
            status_text, status_name = "Opens in this window — no separate process", "StatusSuccess"
        else:
            reason = missing_prerequisite(entry)
            status_text, status_name = (reason, "StatusError") if reason else ("Ready", "StatusSuccess")
        status_label = QLabel(status_text)
        status_label.setObjectName(status_name)
        self.details.content_layout.addRow("Status", status_label)
        self._append_help_summary(entry)
        self._append_learn_more(entry)
        self._append_full_help_button(entry)

    def _append_help_summary(self, entry: AppEntry) -> None:
        """Condensed Inputs/Outputs/Prerequisites rows from the structured
        ``MODULE_HELP`` record, Ansys/STAR-CCM+ style -- the full record
        (plus Typical use / Tips / See also) lives in the F1 help panel;
        this is just enough to answer "what does this need and what does it
        produce" without leaving the hero."""
        from mso_studio.launcher.content import MODULE_HELP

        help_ = MODULE_HELP.get(entry.id)
        if help_ is None:
            return
        for label, values in (
            ("Inputs", help_.inputs),
            ("Outputs", help_.outputs),
            ("Prerequisites", help_.prerequisites),
        ):
            if not values:
                continue
            value_label = QLabel(" · ".join(values), self)
            value_label.setWordWrap(True)
            value_label.setObjectName("Caption")
            self.details.content_layout.addRow(label, value_label)

    def _append_full_help_button(self, entry: AppEntry) -> None:
        from mso_studio.launcher.content import MODULE_HELP

        if entry.id not in MODULE_HELP:
            return
        button = QPushButton("Full help (F1) →", self)
        button.setFlat(True)
        button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        button.setObjectName("LearnMoreToggle")
        button.clicked.connect(lambda: self.helpRequested.emit(entry.id))
        self.details.content_layout.addRow(button)

    def _append_learn_more(self, entry: AppEntry) -> None:
        from mso_studio.launcher.content import LEARN_MORE

        text = LEARN_MORE.get(entry.id)
        if not text:
            return
        heading = QLabel("Learn more", self)
        heading.setObjectName("SectionHeader")
        heading.setProperty("spaced", True)
        self.details.content_layout.addRow(heading)

        body = QLabel(text, self)
        body.setWordWrap(True)
        body.setObjectName("Caption")
        self.details.content_layout.addRow(body)

    def _on_launch_clicked(self) -> None:
        if self._entry is None:
            return
        if self._running:
            self.stopRequested.emit(self._entry.id)
        else:
            # Also used for RUNNER_INTERNAL entries (the OPEN button) --
            # window._on_launch_requested is responsible for routing those
            # to _open_doctor() instead of LaunchManager.launch().
            self.launchRequested.emit(self._entry.id)

    def _on_details_toggled(self, checked: bool) -> None:
        self.details.content.setVisible(checked)

    def _on_pause_clicked(self) -> None:
        if self._entry is None:
            return
        if self._paused:
            self.resumeMenuRequested.emit(self._entry.id)
        else:
            self.pauseRequested.emit(self._entry.id)
