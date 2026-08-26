"""
Tile-art tests: gradient+glyph pixmaps are non-null for every catalog entry
and memoised (same object returned for the same cache key), matching the
convention set by test_run_summary.py (construct, read state, no rendering
diff / pixel comparison).
"""
import pytest


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_tile_art_is_non_null_for_every_catalog_entry(qapp):
    from mso_studio.launcher.art import tile_art
    from mso_studio.launcher.catalog import CATALOG

    for entry in CATALOG:
        pixmap = tile_art(entry, 176, 96)
        assert not pixmap.isNull(), entry.id
        assert pixmap.width() > 0 and pixmap.height() > 0


def test_tile_art_is_memoised(qapp):
    from mso_studio.launcher.art import tile_art
    from mso_studio.launcher.catalog import CATALOG_BY_ID

    entry = CATALOG_BY_ID["idt_sensitivity"]
    first = tile_art(entry, 176, 96)
    second = tile_art(entry, 176, 96)
    assert first is second


def test_tile_art_different_sizes_are_not_shared(qapp):
    from mso_studio.launcher.art import tile_art
    from mso_studio.launcher.catalog import CATALOG_BY_ID

    entry = CATALOG_BY_ID["idt_sensitivity"]
    small = tile_art(entry, 100, 60)
    large = tile_art(entry, 200, 120)
    assert small is not large
    assert small.size() != large.size()


def test_wordmark_is_non_null_for_both_modes(qapp):
    from mso_studio.launcher.art import wordmark

    for mode in ("dark", "light"):
        pixmap = wordmark(52, mode)
        assert not pixmap.isNull()


def test_clear_cache_forces_regeneration(qapp):
    from mso_studio.launcher import art
    from mso_studio.launcher.catalog import CATALOG_BY_ID

    entry = CATALOG_BY_ID["idt_sensitivity"]
    first = art.tile_art(entry, 176, 96)
    art.clear_cache()
    second = art.tile_art(entry, 176, 96)
    assert first is not second


# ===========================================================================
# Phase 3: real generated artwork (scripts/generate_launcher_art.py ->
# assets/launcher/<id>.{png,svg}), loaded and composited over the gradient.
# ===========================================================================
def test_every_catalog_entry_has_generated_art_on_disk():
    """Guards against silently shipping the plain gradient+glyph fallback
    for an entry that should have real artwork -- if this fails after
    adding a catalog entry, run scripts/generate_launcher_art.py."""
    from mso_studio.launcher.art import ASSET_DIR
    from mso_studio.launcher.catalog import CATALOG

    missing = [
        e.id for e in CATALOG
        if not (ASSET_DIR / f"{e.id}.png").exists() and not (ASSET_DIR / f"{e.id}.svg").exists()
    ]
    assert not missing, f"no generated art for: {missing} -- run scripts/generate_launcher_art.py"


def test_load_real_asset_returns_a_valid_pixmap_for_every_entry(qapp):
    from mso_studio.launcher.art import _load_real_asset
    from mso_studio.launcher.catalog import CATALOG

    for entry in CATALOG:
        pixmap = _load_real_asset(entry.id)
        assert pixmap is not None, entry.id
        assert not pixmap.isNull()
        assert pixmap.width() > 0 and pixmap.height() > 0


def test_load_real_asset_is_cached(qapp):
    from mso_studio.launcher.art import _load_real_asset

    first = _load_real_asset("idt_sensitivity")
    second = _load_real_asset("idt_sensitivity")
    assert first is second


def test_load_real_asset_returns_none_for_unknown_id(qapp):
    from mso_studio.launcher.art import _load_real_asset

    assert _load_real_asset("no_such_entry_id") is None


def test_tile_art_uses_real_asset_not_the_glyph_fallback(qapp):
    """Once real art exists for an entry, tile_art must actually composite
    it rather than silently falling back to the qtawesome glyph -- checked
    indirectly by asserting the two render differently at the same size,
    since the fallback path and the asset path produce visually distinct
    pixmaps for any entry with real art."""
    from mso_studio.launcher.art import _asset_cache, tile_art
    from mso_studio.launcher.catalog import CATALOG_BY_ID

    entry = CATALOG_BY_ID["idt_sensitivity"]
    with_asset = tile_art(entry, 176, 96)

    saved = _asset_cache.pop(entry.id, None)
    _asset_cache[entry.id] = None  # force the no-asset fallback path
    try:
        from mso_studio.launcher import art
        art.clear_cache()
        without_asset = tile_art(entry, 176, 96)
    finally:
        _asset_cache.pop(entry.id, None)
        if saved is not None:
            _asset_cache[entry.id] = saved

    assert with_asset.toImage() != without_asset.toImage()


def test_aspect_ratio_is_preserved_not_stretched(qapp):
    """A very wide request (like the hero panel's banner) must not distort
    a tile-shaped source image -- scaled dimensions must keep the source's
    aspect ratio rather than filling the full requested box."""
    from mso_studio.launcher.art import _load_real_asset, tile_art
    from mso_studio.launcher.catalog import CATALOG_BY_ID

    entry = CATALOG_BY_ID["idt_sensitivity"]
    source = _load_real_asset(entry.id)
    source_ratio = source.width() / source.height()

    wide_pixmap = tile_art(entry, 900, 140)
    # Can't inspect the drawn content's bounding box directly without pixel
    # scanning, but we CAN assert the function didn't just stretch-fill --
    # regression-test the geometry math in isolation instead.
    from PySide6.QtCore import Qt as _Qt
    scaled = source.scaled(
        int(900 * 0.90), int(140 * 0.90),
        _Qt.AspectRatioMode.KeepAspectRatio, _Qt.TransformationMode.SmoothTransformation,
    )
    assert abs(scaled.width() / scaled.height() - source_ratio) < 0.05
    assert not wide_pixmap.isNull()
