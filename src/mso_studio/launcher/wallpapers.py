"""
mso_studio.launcher.wallpapers
=================================
Small pure-Python catalog over ``assets/wallpapers/`` -- lists the shipped
pack (each a procedurally generated PNG with a sidecar JSON, see
``scripts/generate_wallpapers.py``) and resolves a wallpaper id (or a
user-supplied absolute path, or ``"none"``) to a concrete file path for
``BackdropWidget``. No Qt import, so it's usable from the Preferences
dialog and from tests without a QApplication.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
WALLPAPER_DIR = REPO_ROOT / "assets" / "wallpapers"

NONE_ID = "none"


@dataclass(frozen=True)
class WallpaperEntry:
    id: str
    name: str
    path: Path
    license: str
    dominant_luminance: float
    recommended_scrim_alpha: float


def list_wallpapers() -> List[WallpaperEntry]:
    """Every wallpaper with both a ``.png`` and a sidecar ``.json`` present."""
    if not WALLPAPER_DIR.exists():
        return []
    entries = []
    for json_path in sorted(WALLPAPER_DIR.glob("*.json")):
        png_path = json_path.with_suffix(".png")
        if not png_path.exists():
            continue
        try:
            meta = json.loads(json_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        entries.append(WallpaperEntry(
            id=meta.get("id", json_path.stem),
            name=meta.get("name", json_path.stem),
            path=png_path,
            license=meta.get("license", "unknown"),
            dominant_luminance=meta.get("dominant_luminance", 0.5),
            recommended_scrim_alpha=meta.get("recommended_scrim_alpha", 0.5),
        ))
    return entries


def resolve_wallpaper_path(wallpaper_id: Optional[str]) -> Optional[Path]:
    """*wallpaper_id* may be ``None``/``"none"`` (no wallpaper), one of the
    shipped pack ids, or an absolute path to a user-supplied image
    (Preferences' "Choose image..." option persists the raw path as the
    "id"). Returns ``None`` for no-wallpaper; does not itself verify a
    user-supplied path exists -- ``BackdropWidget.set_wallpaper`` already
    degrades gracefully (falls back to a flat fill) if the image fails to
    load."""
    if not wallpaper_id or wallpaper_id == NONE_ID:
        return None
    candidate = Path(wallpaper_id)
    if candidate.is_absolute():
        return candidate
    for entry in list_wallpapers():
        if entry.id == wallpaper_id:
            return entry.path
    return None
