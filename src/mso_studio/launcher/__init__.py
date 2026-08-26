"""
mso_studio.launcher
=====================
The game-style catalog/process/preflight backend for the PySide6 launcher
(``launcher_qt.py`` at the repo root). Split the same way the rest of the
codebase splits GUI code from logic: everything importable here is plain
Python (``catalog.py``, ``process.py``, ``preflight.py``) so it can be
exercised by tests under the anaconda interpreter too, without PySide6. The
Qt widgets (``art.py``, ``boot.py``, ``tiles.py``, ``doctor.py``,
``window.py``) live in this same package but do import Qt -- callers that
need to stay Qt-free (tests, the anaconda interpreter) should only import
``mso_studio.launcher.catalog``, ``.process``, or ``.preflight`` directly,
never the bare package (this ``__init__`` stays empty on purpose so
``import mso_studio.launcher`` never pulls in Qt as a side effect).
"""
from __future__ import annotations
