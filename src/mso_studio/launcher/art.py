"""
mso_studio.launcher.art
==========================
Generated tile/hero artwork and the launcher wordmark.

Each entry's real artwork -- an actual tornado sensitivity chart, an actual
Arrhenius uncertainty band, an actual reaction-network graph, etc, rendered
once offline by ``scripts/generate_launcher_art.py`` from real repo data --
lives at ``assets/launcher/<id>.png`` (or ``.svg``). When present, it's
scaled (aspect-preserved) and centered on top of the gradient below. When
absent (asset deleted, or a catalog entry added without running the
generator yet), this falls back to the original design: a 45-degree
gradient between the entry's stage colors plus a ``qtawesome`` glyph, so
the launcher never breaks because an asset is missing. Both the gradient
and the glyph fallback are composited against the entry's *stage* colors
(``STAGE_GRADIENT``), not the app's light/dark theme -- the stage palette
doesn't change with theme mode, so this artwork is theme-independent by
design and is not (and doesn't need to be) keyed by mode. The glyph tint is
chosen for contrast against each stage's actual gradient rather than a
hardcoded white, which read fine on darker stage colors (blue, purple) but
washed out on brighter ones (green, orange). Results are memoised so
scrolling/resizing the tile grid doesn't repaint from scratch every frame.
Only ``wordmark()`` depends on the app theme (it paints on the neutral
window chrome, not a stage gradient) and is cache-keyed by mode.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import qtawesome as qta
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPixmap

from mso_studio.gui.theme import Mode, palette_for
from mso_studio.gui.tokens import relative_luminance
from mso_studio.launcher.catalog import REPO_ROOT, AppEntry, STAGE_GRADIENT

ASSET_DIR = REPO_ROOT / "assets" / "launcher"

_cache: Dict[Tuple, QPixmap] = {}
_asset_cache: Dict[str, Optional[QPixmap]] = {}


def _load_real_asset(entry_id: str) -> Optional[QPixmap]:
    """Load (and cache) the native-resolution generated artwork for
    *entry_id*, or ``None`` if none exists. Loaded once at whatever
    resolution the generator produced -- tile_art() below scales a copy of
    this per requested size rather than reloading from disk each time."""
    if entry_id in _asset_cache:
        return _asset_cache[entry_id]

    pixmap = None
    png_path = ASSET_DIR / f"{entry_id}.png"
    svg_path = ASSET_DIR / f"{entry_id}.svg"
    if png_path.exists():
        loaded = QPixmap(str(png_path))
        pixmap = loaded if not loaded.isNull() else None
    elif svg_path.exists():
        from PySide6.QtSvg import QSvgRenderer
        renderer = QSvgRenderer(str(svg_path))
        if renderer.isValid():
            size = renderer.defaultSize()
            if size.width() <= 0 or size.height() <= 0:
                size = size.__class__(800, 450)
            loaded = QPixmap(size)
            loaded.fill(Qt.GlobalColor.transparent)
            svg_painter = QPainter(loaded)
            renderer.render(svg_painter)
            svg_painter.end()
            pixmap = loaded

    _asset_cache[entry_id] = pixmap
    return pixmap


def _gradient(size, from_hex: str, to_hex: str) -> QLinearGradient:
    grad = QLinearGradient(QPointF(0, 0), QPointF(size.width(), size.height()))
    grad.setColorAt(0.0, QColor(from_hex))
    grad.setColorAt(1.0, QColor(to_hex))
    return grad


def composed_art(
    art_id: str, icon: str, stage: str, width: int, height: int, dpr: float = 1.0,
) -> QPixmap:
    """Gradient + (real asset or glyph fallback) artwork, sized
    *width*x*height* (logical px). The shared rendering path behind both
    ``tile_art()`` (launcher tiles/hero, keyed by an ``AppEntry``) and
    ``mso_studio.gui.widgets.module_hero.ModuleHero`` (Studio page headers,
    which have no ``AppEntry`` to key off of) -- takes raw parameters
    instead of an ``AppEntry`` so both callers can share one implementation.

    *art_id* keys both the memoisation cache and the generated-asset lookup
    (``assets/launcher/<art_id>.{png,svg}``); pass ``""`` to always use the
    glyph fallback (e.g. a module with no dedicated generated asset yet).
    """
    key = ("composed", art_id, icon, stage, width, height, round(dpr, 2))
    cached = _cache.get(key)
    if cached is not None:
        return cached

    pixmap = QPixmap(int(width * dpr), int(height * dpr))
    pixmap.setDevicePixelRatio(dpr)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    rect = QRectF(0, 0, width, height)

    from_hex, to_hex = STAGE_GRADIENT.get(stage, ("#4f8ef7", "#38bdf8"))
    painter.fillRect(rect, _gradient(rect.size(), from_hex, to_hex))

    real_art = _load_real_asset(art_id) if art_id else None
    if real_art is not None:
        # Scale aspect-preserved to fit within the tile with a margin, and
        # center it -- never stretched, so a wide hero call and a small
        # square tile call both get a correctly-proportioned figure rather
        # than a distorted one. QPainter on a dpr-aware QPixmap already
        # operates in logical coordinates (Qt handles the device-pixel
        # scaling internally, same as every other drawPixmap/drawText call
        # in this module), so no manual dpr arithmetic is needed here.
        margin = 0.90
        target_w = max(1, int(width * margin))
        target_h = max(1, int(height * margin))
        scaled = real_art.scaled(
            target_w, target_h,
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
        )
        x = (width - scaled.width()) / 2
        y = (height - scaled.height()) / 2
        painter.drawPixmap(QPointF(x, y), scaled)
    else:
        glyph_size = int(min(width, height) * 0.42)
        # Tinted for contrast against *this stage's actual gradient*
        # (which doesn't change with light/dark mode) rather than a
        # hardcoded white that read fine on some stage colors and washed
        # out on brighter ones.
        avg_luminance = (relative_luminance(from_hex) + relative_luminance(to_hex)) / 2
        glyph_hex = "#0b1220" if avg_luminance > 0.55 else "#ffffff"
        qicon = qta.icon(icon, color=QColor(glyph_hex))
        pix = qicon.pixmap(glyph_size, glyph_size)
        painter.drawPixmap(
            int((width - glyph_size) / 2), int((height - glyph_size) / 2) - 4, pix,
        )
    painter.end()

    _cache[key] = pixmap
    return pixmap


def tile_art(entry: AppEntry, width: int, height: int, dpr: float = 1.0) -> QPixmap:
    """Gradient + glyph artwork for *entry*, sized *width*x*height* (logical px)."""
    return composed_art(entry.id, entry.icon, entry.stage, width, height, dpr)


def wordmark(height: int, mode: Mode = "dark") -> QPixmap:
    """The ``◆ MSO`` mark used on the splash screen and window header."""
    key = ("wordmark", height, mode)
    cached = _cache.get(key)
    if cached is not None:
        return cached

    palette = palette_for(mode)
    width = int(height * 3.4)
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    diamond_size = int(height * 0.6)
    grad = _gradient(
        QRectF(0, 0, diamond_size, diamond_size).size(),
        palette["accent"], palette["accent2"],
    )
    painter.setBrush(grad)
    painter.setPen(Qt.PenStyle.NoPen)
    cx, cy = diamond_size * 0.55, height / 2
    half = diamond_size * 0.42
    diamond = [
        QPointF(cx, cy - half), QPointF(cx + half, cy),
        QPointF(cx, cy + half), QPointF(cx - half, cy),
    ]
    painter.drawPolygon(diamond)

    font = QFont()
    font.setBold(True)
    font.setPixelSize(int(height * 0.62))
    font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 108)
    painter.setFont(font)
    painter.setPen(QColor(palette["text"]))
    text_rect = QRectF(diamond_size + 8, 0, width - diamond_size - 8, height)
    painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, "MSO")
    painter.end()

    _cache[key] = pixmap
    return pixmap


def clear_cache() -> None:
    """Drop all memoised art -- call when the theme mode changes."""
    _cache.clear()
