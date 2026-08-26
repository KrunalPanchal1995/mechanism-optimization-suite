"""
Wallpaper legibility guard: for every shipped wallpaper (plus synthetic
pure-white and pure-black extremes standing in for the worst-case
user-supplied image), the computed scrim opacity must bring body text back
above the WCAG floor -- a wallpaper can never make text unreadable, only
heavily scrimmed in the worst case.
"""
from pathlib import Path

import pytest

WALLPAPER_DIR = Path(__file__).resolve().parents[3] / "assets" / "wallpapers"


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def _wallpaper_pngs():
    if not WALLPAPER_DIR.exists():
        return []
    return sorted(WALLPAPER_DIR.glob("*.png"))


@pytest.mark.parametrize("png_path", _wallpaper_pngs(), ids=lambda p: p.stem)
@pytest.mark.parametrize("mode_label,tokens_attr", [("dark", "DARK"), ("light", "LIGHT")])
def test_shipped_wallpaper_scrim_clears_contrast_floor(qapp, png_path, mode_label, tokens_attr):
    from mso_studio.gui.backdrop import average_color_hex, required_scrim_alpha
    from mso_studio.gui import tokens as tokens_mod
    from mso_studio.gui.tokens import contrast_ratio
    from PySide6.QtGui import QImage

    tokens = getattr(tokens_mod, tokens_attr)
    image = QImage(str(png_path))
    assert not image.isNull(), f"failed to load {png_path}"
    avg_hex = average_color_hex(image)

    alpha = required_scrim_alpha(avg_hex, tokens.text, tokens.scrim)
    blended = _blend(avg_hex, tokens.scrim, alpha)
    ratio = contrast_ratio(tokens.text, blended)
    assert ratio >= 4.5 - 1e-6, (
        f"{png_path.name} in {mode_label} mode: scrim alpha {alpha:.2f} only "
        f"reaches {ratio:.2f}:1 contrast for text, need 4.5:1"
    )


@pytest.mark.parametrize("synthetic_hex", ["#ffffff", "#000000"])
@pytest.mark.parametrize("mode_label,tokens_attr", [("dark", "DARK"), ("light", "LIGHT")])
def test_synthetic_extreme_wallpaper_scrim_clears_contrast_floor(synthetic_hex, mode_label, tokens_attr):
    from mso_studio.gui import tokens as tokens_mod
    from mso_studio.gui.tokens import contrast_ratio
    from mso_studio.gui.backdrop import required_scrim_alpha

    tokens = getattr(tokens_mod, tokens_attr)
    alpha = required_scrim_alpha(synthetic_hex, tokens.text, tokens.scrim)
    blended = _blend(synthetic_hex, tokens.scrim, alpha)
    ratio = contrast_ratio(tokens.text, blended)
    assert ratio >= 4.5 - 1e-6, (
        f"synthetic {synthetic_hex} wallpaper in {mode_label} mode: scrim alpha "
        f"{alpha:.2f} only reaches {ratio:.2f}:1, need 4.5:1"
    )


def _blend(wallpaper_hex: str, scrim_hex: str, alpha: float) -> str:
    from mso_studio.gui.backdrop import _hex_to_rgb01, _rgb01_to_hex

    w = _hex_to_rgb01(wallpaper_hex)
    s = _hex_to_rgb01(scrim_hex)
    return _rgb01_to_hex(tuple(w[i] * (1 - alpha) + s[i] * alpha for i in range(3)))


def test_no_wallpaper_falls_back_to_flat_surface(qapp):
    from mso_studio.gui.backdrop import BackdropWidget
    from mso_studio.gui.tokens import DARK

    widget = BackdropWidget(DARK)
    assert widget.scrim_alpha() == 0.0


def test_missing_image_path_does_not_crash(qapp):
    from mso_studio.gui.backdrop import BackdropWidget
    from mso_studio.gui.tokens import DARK

    widget = BackdropWidget(DARK, wallpaper_path=Path("/nonexistent/path.png"))
    assert widget.scrim_alpha() == 0.0
