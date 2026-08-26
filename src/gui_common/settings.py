"""
gui_common/settings.py
========================
Per-app GUI settings persistence.

The monolith (generate_target_opt.py) used a single global
``.rmip_gui_settings.json`` (one file, next to the script, one key in
practice: "checkpoint_path"). With multiple split-out apps now importing
this same helper, a single shared file would collide -- each app gets its
own file under ``.gui_settings/<app_name>.json`` instead.
"""
from __future__ import annotations

import json
from pathlib import Path

_SETTINGS_DIR = Path(__file__).resolve().parents[2] / ".gui_settings"


def load_settings(app_name: str) -> dict:
    """Load this app's small persisted GUI settings (e.g. checkpoint path)."""
    path = _SETTINGS_DIR / f"{app_name}.json"
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def save_settings(app_name: str, settings: dict) -> None:
    """Persist this app's small GUI settings."""
    path = _SETTINGS_DIR / f"{app_name}.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(settings, indent=2))
    except Exception as e:
        print(f"Could not save GUI settings for {app_name!r}: {e}")
