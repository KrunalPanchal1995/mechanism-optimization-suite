"""
mso_studio.gui.backdrop
==========================
``BackdropWidget`` -- wallpaper backgrounds for chrome only (launcher tile
area, hero panels, the boot splash), per the locked design decision:
photographic/procedural wallpapers may sit behind decorative chrome, but
never behind body text, forms, tables, or log views -- those stay on an
opaque ``tokens.surface``.

The one hazard a wallpaper introduces -- unreadable text if the image is
too bright/dark/busy under a label -- is closed structurally here:
``required_scrim_alpha()`` computes the minimum scrim opacity that gets
``tokens.text`` back above a WCAG contrast floor against the wallpaper's
average color, and ``BackdropWidget`` always applies at least that much.
A user-supplied image therefore cannot make text unreadable; worst case it
gets heavily scrimmed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QWidget

from mso_studio.gui.tokens import Tokens, contrast_ratio

MIN_TEXT_CONTRAST = 4.5
_ALPHA_STEPS = 51  # 0.00, 0.02, ..., 1.00


def _hex_to_rgb01(hex_color: str):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _rgb01_to_hex(rgb) -> str:
    return "#" + "".join(f"{max(0, min(255, round(c * 255))):02x}" for c in rgb)


def required_scrim_alpha(
    wallpaper_avg_hex: str, text_hex: str, scrim_hex: str,
    target_ratio: float = MIN_TEXT_CONTRAST,
) -> float:
    """Smallest scrim opacity in [0, 1] (a flat ``scrim_hex`` layer over the
    wallpaper) such that *text_hex* clears *target_ratio* contrast against
    the resulting blend. Scans rather than bisects, since alpha-compositing
    contrast isn't guaranteed monotonic for an arbitrary wallpaper/scrim/text
    combination -- a linear scan is cheap (at most 51 contrast_ratio calls)
    and always finds the true minimum if one exists at all. Falls back to
    1.0 (fully opaque -- the wallpaper is entirely hidden) if even that
    doesn't clear the ratio, which cannot happen for any scrim/text pair
    that itself passes ``tokens.py``'s ``REQUIRED_PAIRS`` check.
    """
    wallpaper = _hex_to_rgb01(wallpaper_avg_hex)
    scrim = _hex_to_rgb01(scrim_hex)
    for step in range(_ALPHA_STEPS):
        alpha = step / (_ALPHA_STEPS - 1)
        blended = tuple(wallpaper[i] * (1 - alpha) + scrim[i] * alpha for i in range(3))
        if contrast_ratio(text_hex, _rgb01_to_hex(blended)) >= target_ratio:
            return alpha
    return 1.0


def average_color_hex(image: QImage, sample_w: int = 32, sample_h: int = 18) -> str:
    """Cheap average color of *image* via a tiny nearest-neighbor downsample."""
    small = image.scaled(
        sample_w, sample_h,
        Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.FastTransformation,
    )
    r = g = b = n = 0
    for y in range(small.height()):
        for x in range(small.width()):
            c = small.pixelColor(x, y)
            r += c.red()
            g += c.green()
            b += c.blue()
            n += 1
    n = max(n, 1)
    return _rgb01_to_hex((r / n / 255.0, g / n / 255.0, b / n / 255.0))


def _blur_qimage(image: QImage, radius: int) -> QImage:
    """Gaussian-blur *image* via Pillow if available; returns the original
    image unchanged if Pillow can't be imported (a missing blur is a
    cosmetic degradation, not a legibility hazard -- the scrim floor still
    holds regardless)."""
    if radius <= 0:
        return image
    try:
        from PIL import Image as PILImage
        from PIL import ImageFilter
    except ImportError:
        return image

    rgba = image.convertToFormat(QImage.Format.Format_RGBA8888)
    w, h = rgba.width(), rgba.height()
    # PySide6's QImage.bits() returns a memoryview already sized to
    # sizeInBytes() -- the .setsize() call is leftover from the older
    # sip.voidptr API and doesn't exist on a memoryview, so every wallpaper
    # application crashed here with an uncaught AttributeError.
    pil_img = PILImage.frombuffer("RGBA", (w, h), bytes(rgba.bits()), "raw", "RGBA", 0, 1)
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius))
    data = blurred.tobytes("raw", "RGBA")
    out = QImage(data, w, h, QImage.Format.Format_RGBA8888)
    return out.copy()  # copy: `data` must outlive the QImage view over it


class BackdropWidget(QWidget):
    """Paints, in order: the wallpaper (scaled to fill, center-cropped),
    an optional pre-rendered blur, and a contrast-safe scrim. Falls back to
    a flat ``tokens.bg`` fill when no wallpaper is set or the image fails
    to load."""

    def __init__(
        self, tokens: Tokens, wallpaper_path: Optional[Path] = None,
        blur_radius: int = 24, parent=None,
    ):
        super().__init__(parent)
        self._tokens = tokens
        self._blur_radius = blur_radius
        self._wallpaper_path: Optional[Path] = None
        self._avg_hex = tokens.bg
        self._scrim_alpha = 0.0
        self._base_pixmap: Optional[QPixmap] = None
        self._scaled_cache: Optional[QPixmap] = None
        self._scaled_cache_size = None
        if wallpaper_path is not None:
            self.set_wallpaper(wallpaper_path)

    def set_tokens(self, tokens: Tokens) -> None:
        self._tokens = tokens
        self._recompute_scrim()
        self.update()

    def scrim_alpha(self) -> float:
        return self._scrim_alpha

    def set_wallpaper(self, path: Optional[Path]) -> None:
        self._scaled_cache = None
        if path is None:
            self._wallpaper_path = None
            self._base_pixmap = None
            self._scrim_alpha = 0.0
            self.update()
            return

        image = QImage(str(path))
        if image.isNull():
            self._wallpaper_path = None
            self._base_pixmap = None
            self._scrim_alpha = 0.0
            self.update()
            return

        self._wallpaper_path = Path(path)
        self._avg_hex = average_color_hex(image)
        blurred = _blur_qimage(image, self._blur_radius)
        self._base_pixmap = QPixmap.fromImage(blurred)
        self._recompute_scrim()
        self.update()

    def _recompute_scrim(self) -> None:
        if self._wallpaper_path is None:
            self._scrim_alpha = 0.0
            return
        self._scrim_alpha = required_scrim_alpha(
            self._avg_hex, self._tokens.text, self._tokens.scrim,
        )

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._scaled_cache = None
        super().resizeEvent(event)

    def _ensure_scaled(self) -> Optional[QPixmap]:
        if self._base_pixmap is None:
            return None
        size = self.size()
        if size.width() <= 0 or size.height() <= 0:
            return None
        if self._scaled_cache is not None and self._scaled_cache_size == size:
            return self._scaled_cache
        scaled = self._base_pixmap.scaled(
            size, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = max(0, (scaled.width() - size.width()) // 2)
        y = max(0, (scaled.height() - size.height()) // 2)
        cropped = scaled.copy(x, y, size.width(), size.height())
        self._scaled_cache = cropped
        self._scaled_cache_size = size
        return cropped

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        pixmap = self._ensure_scaled()
        if pixmap is not None:
            painter.drawPixmap(0, 0, pixmap)
            if self._scrim_alpha > 0:
                color = QColor(self._tokens.scrim)
                color.setAlphaF(self._scrim_alpha)
                painter.fillRect(self.rect(), color)
        else:
            painter.fillRect(self.rect(), QColor(self._tokens.bg))
        painter.end()
