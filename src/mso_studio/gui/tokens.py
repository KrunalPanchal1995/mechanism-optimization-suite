"""
mso_studio.gui.tokens
======================
Single source of truth for the visual design tokens used across MSO Studio:
colors, the type scale, and the WCAG contrast rules that must hold between
them. Pure Python, no Qt import -- so the Tkinter side of the suite
(``src/gui_common/theme.py``) can consume the exact same values instead of
maintaining its own copy of the palette.

``theme.py`` turns these tokens into QSS + a QPalette; ``backdrop.py`` (the
wallpaper subsystem) uses ``contrast_ratio`` to keep scrims legible over an
arbitrary user-supplied image. Nothing here should import from either.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, List, Tuple

Mode = str  # "system" | "light" | "dark"


@dataclass(frozen=True)
class Tokens:
    # Surfaces, lowest to highest elevation.
    bg: str
    surface: str
    surface_raised: str
    surface_sunken: str
    overlay: str

    # Structure.
    border: str
    border_strong: str

    # Text.
    text: str
    text_dim: str
    text_faint: str
    text_on_accent: str

    # Brand / semantic.
    accent: str
    accent_hover: str
    accent_pressed: str
    accent2: str
    accent3: str
    success: str
    warning: str
    error: str
    info: str

    # Interactive states.
    selection_bg: str
    selection_text: str
    entry_bg: str
    disabled_bg: str
    disabled_text: str

    # Wallpaper chrome only (Session A3) -- never used behind body text.
    scrim: str
    glow: str

    def as_dict(self) -> Dict[str, str]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


DARK = Tokens(
    bg="#1e2433",
    surface="#252b3b",
    surface_raised="#2e3650",
    surface_sunken="#171c29",
    overlay="#2e3650",
    border="#374151",
    border_strong="#4b5563",
    text="#e2e8f0",
    text_dim="#94a3b8",
    text_faint="#6b7d99",
    text_on_accent="#0b1220",
    accent="#4f8ef7",
    accent_hover="#6ba3ff",
    accent_pressed="#3b7ae0",
    accent2="#38bdf8",
    accent3="#a855f7",
    success="#22c55e",
    warning="#f59e0b",
    error="#ef4444",
    info="#38bdf8",
    selection_bg="#4f8ef7",
    selection_text="#0b1220",
    entry_bg="#111827",
    disabled_bg="#252b3b",
    disabled_text="#5b6577",
    scrim="#0b0e17",
    glow="#4f8ef7",
)

LIGHT = Tokens(
    bg="#f4f5f7",
    surface="#ffffff",
    surface_raised="#eef0f4",
    surface_sunken="#e5e7eb",
    overlay="#eef0f4",
    border="#bcc2cc",
    border_strong="#9ca3af",
    text="#1f2937",
    text_dim="#6b7280",
    text_faint="#75808f",
    text_on_accent="#ffffff",
    accent="#2563eb",
    accent_hover="#3b76f0",
    accent_pressed="#1d4fc4",
    accent2="#0ea5e9",
    accent3="#9333ea",
    success="#16a34a",
    warning="#d97706",
    error="#dc2626",
    info="#0ea5e9",
    selection_bg="#2563eb",
    selection_text="#ffffff",
    entry_bg="#ffffff",
    disabled_bg="#eef0f4",
    disabled_text="#9ca3af",
    scrim="#e5e7eb",
    glow="#2563eb",
)

MONO_FAMILY = "DejaVu Sans Mono, Consolas, monospace"
UI_FAMILY = "Inter, Segoe UI, Cantarell, DejaVu Sans, sans-serif"

FONT_SIZES = {
    "micro": 11,
    "small": 12,
    "body": 14,
    "lead": 15,
    "title": 17,
    "hero": 24,
    "display": 30,
}

# Compact / Normal / Large / Extra Large -- Session A3's Preferences dialog.
FONT_SCALES = {
    "compact": 0.85,
    "normal": 1.0,
    "large": 1.15,
    "xlarge": 1.3,
}

# Comfortable / Compact density -- scales the handful of paddings that
# most affect perceived density (buttons, inputs, list/menu items).
DENSITY_SCALES = {
    "comfortable": 1.0,
    "compact": 0.65,
}


def hex_to_rgb01(hex_color: str) -> Tuple[float, float, float]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def rgb01_to_hex(rgb: Tuple[float, float, float]) -> str:
    return "#" + "".join(f"{max(0, min(255, round(c * 255))):02x}" for c in rgb)


def lighten(hex_color: str, amount: float) -> str:
    """Blend *hex_color* toward white by *amount* in [0, 1]."""
    r, g, b = hex_to_rgb01(hex_color)
    return rgb01_to_hex((r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount))


def darken(hex_color: str, amount: float) -> str:
    """Blend *hex_color* toward black by *amount* in [0, 1]."""
    r, g, b = hex_to_rgb01(hex_color)
    return rgb01_to_hex((r * (1 - amount), g * (1 - amount), b * (1 - amount)))


def readable_text_for(bg_hex: str, light_hex: str = "#ffffff", dark_hex: str = "#0b1220") -> str:
    """Whichever of *light_hex*/*dark_hex* clears a higher contrast ratio
    against *bg_hex* -- used to pick a legible label color for an
    arbitrary user-chosen accent."""
    if contrast_ratio(light_hex, bg_hex) >= contrast_ratio(dark_hex, bg_hex):
        return light_hex
    return dark_hex


def _srgb_channel(c: float) -> float:
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(hex_color: str) -> float:
    """WCAG 2.1 relative luminance of a "#rrggbb" color, in [0, 1]."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    r, g, b = (_srgb_channel(c) for c in (r, g, b))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(fg_hex: str, bg_hex: str) -> float:
    """WCAG 2.1 contrast ratio between two colors, in [1, 21]."""
    l1 = relative_luminance(fg_hex) + 0.05
    l2 = relative_luminance(bg_hex) + 0.05
    return max(l1, l2) / min(l1, l2)


# (fg_token, bg_token, minimum_ratio) -- every pair here is asserted by
# test_theme_contrast.py, in both DARK and LIGHT, so a future edit that
# breaks legibility fails the build instead of shipping.
REQUIRED_PAIRS: List[Tuple[str, str, float]] = [
    ("text", "bg", 4.5),
    ("text", "surface", 4.5),
    ("text", "surface_raised", 4.5),
    ("text", "entry_bg", 4.5),
    ("text_dim", "bg", 3.0),
    ("text_dim", "surface", 3.0),
    ("text_faint", "surface", 3.0),
    ("text_on_accent", "accent", 4.5),
    ("selection_text", "selection_bg", 4.5),
    ("accent", "bg", 3.0),
    ("accent", "surface", 3.0),
    ("success", "surface", 3.0),
    ("warning", "surface", 3.0),
    ("error", "surface", 3.0),
    ("border", "bg", 1.5),
    ("border", "surface", 1.2),
]


def check_required_pairs(tokens: Tokens) -> List[Tuple[str, str, float, float]]:
    """Return failures as (fg_token, bg_token, required, actual); empty if all pass."""
    d = tokens.as_dict()
    failures = []
    for fg, bg, minimum in REQUIRED_PAIRS:
        actual = contrast_ratio(d[fg], d[bg])
        if actual < minimum:
            failures.append((fg, bg, minimum, actual))
    return failures


def tokens_for(mode: Mode, system_prefers_dark: bool = True) -> Tokens:
    if mode == "light":
        return LIGHT
    if mode == "dark":
        return DARK
    return DARK if system_prefers_dark else LIGHT


# Preferences dialog accent presets -- 6 curated hues plus "custom" (any
# color the user picks, subject to with_accent()'s contrast validation).
ACCENT_PRESETS = {
    "blue": "#4f8ef7",
    "cyan": "#38bdf8",
    "violet": "#a855f7",
    "green": "#22c55e",
    "amber": "#f59e0b",
    "rose": "#ef4444",
}


def with_accent(tokens: Tokens, accent_hex: str) -> Tokens:
    """A copy of *tokens* with the accent family (accent/accent_hover/
    accent_pressed/selection_bg/selection_text/glow) re-derived from a
    user-chosen *accent_hex*, everything else unchanged. Hover/pressed are
    lightened/darkened by a fixed amount rather than left to the caller, so
    every accent choice gets a consistent 3-step ramp. selection_text is
    picked (not fixed) so an arbitrarily bright or dark custom accent still
    keeps its highlighted-text contrast -- the actual validation that a
    color is usable happens in the Preferences dialog via
    ``tokens.check_required_pairs``, not here.
    """
    from dataclasses import replace

    return replace(
        tokens,
        accent=accent_hex,
        accent_hover=lighten(accent_hex, 0.18),
        accent_pressed=darken(accent_hex, 0.18),
        text_on_accent=readable_text_for(accent_hex),
        selection_bg=accent_hex,
        selection_text=readable_text_for(accent_hex),
        glow=accent_hex,
    )
