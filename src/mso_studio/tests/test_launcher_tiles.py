"""
AppTile/HeroPanel interaction tests (Phase 0 bug-fix regression suite).

Covers three real bugs found and fixed while building on top of the
launcher: (1) hovering a tile used to mutate the global selection, so
moving the mouse toward the hero panel's LAUNCH button silently retargeted
it; (2) the Doctor/Recipes tile (runner=RUNNER_INTERNAL, no subprocess)
stayed LAUNCH-enabled and always raised a "no subprocess to launch" warning
when clicked; (3) the entrance-fade animation swapped out the hover
drop-shadow effect for an opacity effect and never reliably restored it,
so a hover during/after that window threw "Internal C++ object already
deleted". Same convention as test_run_summary.py: construct + drive real
Qt event handlers directly, no click simulation via QTest.
"""
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QEnterEvent, QFocusEvent, QKeyEvent, QMouseEvent


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def _make_tile(qapp, entry_id="mechanism_check"):
    from mso_studio.launcher.catalog import CATALOG_BY_ID
    from mso_studio.launcher.tiles import AppTile
    return AppTile(CATALOG_BY_ID[entry_id], "dark")


# -- Bug 1: hover must not change selection --------------------------------

def test_hover_does_not_emit_selected(qapp):
    tile = _make_tile(qapp)
    selected_ids = []
    tile.selected.connect(selected_ids.append)

    enter_event = QEnterEvent(QPointF(5, 5), QPointF(5, 5), QPointF(5, 5))
    tile.enterEvent(enter_event)
    tile.leaveEvent(QEvent(QEvent.Type.Leave))

    assert selected_ids == []


def test_click_still_emits_selected(qapp):
    """Selection stays click-driven -- only the hover path was removed."""
    tile = _make_tile(qapp)
    selected_ids = []
    tile.selected.connect(selected_ids.append)

    event = QMouseEvent(
        QEvent.Type.MouseButtonPress, QPointF(5, 5), QPointF(5, 5),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    )
    tile.mousePressEvent(event)

    assert selected_ids == [tile.entry.id]


def test_double_click_emits_activated(qapp):
    tile = _make_tile(qapp)
    activated_ids = []
    tile.activated.connect(activated_ids.append)

    event = QMouseEvent(
        QEvent.Type.MouseButtonDblClick, QPointF(5, 5), QPointF(5, 5),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    )
    tile.mouseDoubleClickEvent(event)

    assert activated_ids == [tile.entry.id]


# -- Keyboard reachability ----------------------------------------------------

def test_tile_is_tab_focusable(qapp):
    tile = _make_tile(qapp)
    assert tile.focusPolicy() == Qt.FocusPolicy.StrongFocus


def test_focus_in_emits_selected(qapp):
    tile = _make_tile(qapp)
    selected_ids = []
    tile.selected.connect(selected_ids.append)

    tile.focusInEvent(QFocusEvent(QEvent.Type.FocusIn))

    assert selected_ids == [tile.entry.id]


@pytest.mark.parametrize("key", [Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space])
def test_enter_and_space_emit_activated(qapp, key):
    tile = _make_tile(qapp)
    activated_ids = []
    tile.activated.connect(activated_ids.append)

    event = QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)
    tile.keyPressEvent(event)

    assert activated_ids == [tile.entry.id]


def test_other_keys_do_not_activate(qapp):
    tile = _make_tile(qapp)
    activated_ids = []
    tile.activated.connect(activated_ids.append)

    event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_A, Qt.KeyboardModifier.NoModifier)
    tile.keyPressEvent(event)

    assert activated_ids == []


# -- Bug 3: entrance-fade must not break the hover shadow afterward ----------

def test_play_entrance_swaps_to_opacity_effect(qapp):
    from PySide6.QtWidgets import QGraphicsOpacityEffect
    tile = _make_tile(qapp)

    tile.play_entrance(delay_ms=0)

    assert isinstance(tile.graphicsEffect(), QGraphicsOpacityEffect)
    assert tile._shadow is None


def test_hover_during_entrance_window_does_not_raise(qapp):
    """This is the exact crash scenario: play_entrance() has swapped the
    shadow out (self._shadow is None) but the fade hasn't finished yet, and
    the user hovers. Must be a guarded no-op, not a crash."""
    tile = _make_tile(qapp)
    tile.play_entrance(delay_ms=0)

    tile.set_lift(6.0)  # must not raise


def test_shadow_is_restored_after_entrance_finishes(qapp):
    from PySide6.QtWidgets import QGraphicsDropShadowEffect
    tile = _make_tile(qapp)
    tile.play_entrance(delay_ms=0)

    tile._install_shadow()  # what anim.finished calls

    assert isinstance(tile._shadow, QGraphicsDropShadowEffect)
    assert tile.graphicsEffect() is tile._shadow


def test_hover_works_normally_after_entrance_restores_shadow(qapp):
    tile = _make_tile(qapp)
    tile.play_entrance(delay_ms=0)
    tile._install_shadow()

    tile.set_lift(6.0)  # must not raise, and must actually update the shadow

    assert tile._shadow.blurRadius() > 0


# -- Bug 2: Doctor (RUNNER_INTERNAL) tile must not offer a dead LAUNCH ------

def test_hero_shows_open_button_for_internal_runner(qapp):
    from mso_studio.launcher.catalog import CATALOG_BY_ID
    from mso_studio.launcher.tiles import HeroPanel

    hero = HeroPanel("dark")
    hero.set_entry(CATALOG_BY_ID["doctor"])

    assert hero.launch_button.text() == "⚙  OPEN"
    assert hero.launch_button.isEnabled()


def test_hero_launch_button_for_normal_entry_unaffected(qapp):
    from mso_studio.launcher.catalog import CATALOG_BY_ID
    from mso_studio.launcher.tiles import HeroPanel

    hero = HeroPanel("dark")
    hero.set_entry(CATALOG_BY_ID["mechanism_check"])

    assert hero.launch_button.text() == "▶  LAUNCH"


def test_hero_details_status_for_internal_runner_is_not_misleading(qapp):
    from mso_studio.launcher.catalog import CATALOG_BY_ID
    from mso_studio.launcher.tiles import HeroPanel

    hero = HeroPanel("dark")
    hero.set_entry(CATALOG_BY_ID["doctor"])
    hero.details_toggle.setChecked(True)
    hero._on_details_toggled(True)

    # Find the status row's label by its object name -- position alone
    # isn't stable now that a "Learn more" paragraph is appended after it.
    status_labels = [
        w for w in hero.details.content.findChildren(type(hero.title_label))
        if w.objectName() in ("StatusSuccess", "StatusError")
    ]
    assert len(status_labels) == 1
    assert "window" in status_labels[0].text().lower()
    assert status_labels[0].objectName() == "StatusSuccess"


def test_every_catalog_entry_has_learn_more_text(qapp):
    from mso_studio.launcher.catalog import CATALOG
    from mso_studio.launcher.content import LEARN_MORE

    missing = [e.id for e in CATALOG if not LEARN_MORE.get(e.id)]
    assert missing == []


def test_hero_details_shows_learn_more_text(qapp):
    from mso_studio.launcher.catalog import CATALOG_BY_ID
    from mso_studio.launcher.content import LEARN_MORE
    from mso_studio.launcher.tiles import HeroPanel

    hero = HeroPanel("dark")
    entry = CATALOG_BY_ID["idt_sensitivity"]
    hero.set_entry(entry)
    hero.details_toggle.setChecked(True)
    hero._on_details_toggled(True)

    body_texts = [
        w.text() for w in hero.details.content.findChildren(type(hero.title_label))
        if w.objectName() == "Caption"
    ]
    assert any(LEARN_MORE[entry.id] == t for t in body_texts)


# -- Phase 6: hero art must update on entry change, not just resize --------

def test_hero_art_updates_when_entry_changes_at_the_same_width(qapp):
    """Regression test: _update_art()'s cache guard used to key only on
    width, so selecting a different entry without resizing the window left
    the OLD entry's artwork on screen while title/description correctly
    updated to the new one -- a real, easily-hit stale-art bug (the common
    case is clicking between tiles without ever resizing)."""
    from mso_studio.launcher.catalog import CATALOG_BY_ID
    from mso_studio.launcher.tiles import HeroPanel

    hero = HeroPanel("dark")
    hero.resize(900, 200)
    hero.set_entry(CATALOG_BY_ID["idt_sensitivity"])
    first_image = hero.art_label.pixmap().toImage()

    hero.set_entry(CATALOG_BY_ID["forward_uq"])  # different entry, same width
    second_image = hero.art_label.pixmap().toImage()

    assert hero.title_label.text() == "Forward UQ"
    assert first_image != second_image


def test_hero_art_reveal_plays_only_on_real_entry_change(qapp):
    from mso_studio.launcher.catalog import CATALOG_BY_ID
    from mso_studio.launcher.tiles import HeroPanel

    hero = HeroPanel("dark")
    hero.resize(900, 200)
    hero.set_entry(CATALOG_BY_ID["idt_sensitivity"])
    assert hero.art_label.graphicsEffect() is not None  # first selection also "changes" (None -> entry)

    hero.art_label.setGraphicsEffect(None)
    hero.resize(950, 200)  # same entry, width-only change
    assert hero.art_label.graphicsEffect() is None  # no reveal replay for a plain resize

    hero.set_entry(CATALOG_BY_ID["forward_uq"])  # real entry change
    assert hero.art_label.graphicsEffect() is not None
