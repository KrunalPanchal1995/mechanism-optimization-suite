"""
ModuleHero tests (Phase 5): the launcher's visual language rolled out to
MSO Studio's 12 module pages. Same construct-and-drive convention as
test_run_summary.py.
"""
import pytest


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_every_page_factories_module_id_has_hero_metadata(qapp):
    from mso_studio.gui.modules import PAGE_FACTORIES
    from mso_studio.launcher.content import MODULE_META

    missing = [mid for mid, _label, _factory in PAGE_FACTORIES if mid not in MODULE_META]
    assert missing == []


def test_hero_shows_title_and_tagline(qapp):
    from mso_studio.gui.widgets.module_hero import ModuleHero
    from mso_studio.launcher.content import MODULE_META

    hero = ModuleHero("mechanism_check", "Mechanism Health-Check", "some engineer text")
    labels = [w.text() for w in hero.findChildren(type(hero.learn_more_label))]
    assert MODULE_META["mechanism_check"].tagline in labels
    assert "some engineer text" in labels
    assert hero.learn_more_toggle is not None
    assert not hero.thumb_label.pixmap().isNull()


def test_hero_thumbnail_uses_real_asset_when_art_key_present(qapp):
    from mso_studio.gui.widgets.module_hero import ModuleHero
    from mso_studio.launcher.art import _load_real_asset

    hero = ModuleHero("mechanism_check", "Mechanism Health-Check")
    # mechanism_check has a real generated asset (art_key="mechanism_check")
    assert _load_real_asset("mechanism_check") is not None
    assert not hero.thumb_label.pixmap().isNull()


def test_hero_falls_back_gracefully_for_modules_with_no_asset(qapp):
    from mso_studio.gui.widgets.module_hero import ModuleHero
    from mso_studio.launcher.content import MODULE_META

    assert MODULE_META["tools"].art_key == ""
    hero = ModuleHero("tools", "Tools")
    assert not hero.thumb_label.pixmap().isNull()  # glyph fallback, still non-null


def test_learn_more_starts_collapsed_and_toggles(qapp):
    from mso_studio.gui.widgets.module_hero import ModuleHero

    hero = ModuleHero("forward_uq", "Forward UQ")
    hero.show()  # isVisible() reflects real visibility, requires a shown ancestor chain
    assert not hero.learn_more_label.isVisible()
    assert "▾" in hero.learn_more_toggle.text()

    hero.learn_more_toggle.setChecked(True)
    hero._on_toggle_learn_more(True)
    assert hero.learn_more_label.isVisible()
    assert "▴" in hero.learn_more_toggle.text()


def test_learn_more_text_matches_module_meta(qapp):
    from mso_studio.gui.widgets.module_hero import ModuleHero
    from mso_studio.launcher.content import MODULE_META

    hero = ModuleHero("optimization", "Optimization")
    assert hero.learn_more_label.text() == MODULE_META["optimization"].learn_more


def test_unknown_module_id_does_not_crash(qapp):
    """Defensive: a module id with no MODULE_META entry (e.g. added to
    PAGE_FACTORIES without updating content.py) must not crash the page --
    it just gets a title-only hero with no tagline/learn-more."""
    from mso_studio.gui.widgets.module_hero import ModuleHero

    hero = ModuleHero("no_such_module", "Mystery Module")
    assert not hasattr(hero, "learn_more_toggle")


def test_make_module_hero_matches_direct_construction(qapp):
    from mso_studio.gui.widgets.module_hero import make_module_hero

    hero = make_module_hero("rmip_suite", "RMIP Suite", "desc text")
    assert hero.learn_more_toggle is not None


def test_module_page_subclass_exposes_hero_attribute(qapp):
    from mso_studio.gui.modules.converter import ConverterPage

    page = ConverterPage()
    assert page.hero is not None
    assert page.hero.learn_more_toggle is not None


def test_hand_rolled_pages_expose_hero_attribute(qapp):
    from mso_studio.gui.modules.tools import ToolsPage

    page = ToolsPage()
    assert page.hero is not None


def test_hero_gradient_follows_module_stage_not_one_fixed_gradient(qapp):
    """Regression guard: every ModuleHero used to render the same fixed
    "integrated"-stage cyan/blue gradient regardless of module (module_id
    was never threaded into composed_art's stage argument). Two modules in
    different MODULE_META stages, both with no dedicated art asset (so the
    gradient itself, not a photo/plot, is what's on screen), must now
    produce visibly different thumbnails."""
    from mso_studio.gui.widgets.module_hero import ModuleHero
    from mso_studio.launcher.content import MODULE_META

    assert MODULE_META["automation"].stage != MODULE_META["tools"].stage
    assert MODULE_META["automation"].art_key == ""
    assert MODULE_META["tools"].art_key == ""

    hero_a = ModuleHero("automation", "Automation")
    hero_b = ModuleHero("tools", "Tools")
    assert hero_a.thumb_label.pixmap().toImage() != hero_b.thumb_label.pixmap().toImage()


def test_every_module_meta_stage_is_a_known_stage():
    from mso_studio.launcher.catalog import STAGE_ORDER
    from mso_studio.launcher.content import MODULE_META

    unknown = [mid for mid, meta in MODULE_META.items() if meta.stage not in STAGE_ORDER]
    assert unknown == []
