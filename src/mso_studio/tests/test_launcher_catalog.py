"""
Catalog tests for the game-style launcher (launcher_qt.py). Pure-Python
checks (no Qt) plus one icon-resolution check that needs a QApplication for
qtawesome. Also verifies launcher_gui.py's Tkinter _APPS list is generated
FROM this catalog (not merely checked against it, per Session A4 of the
visual-system plan -- CATALOG is now the single source of truth, so drift
between the two lists is structurally impossible rather than just detected).
"""
import importlib.util
import sys
from pathlib import Path

import pytest

from mso_studio.launcher.catalog import (
    CATALOG, CATALOG_BY_ID, STAGE_ORDER, script_path,
)


def _load_tkinter_apps():
    """Import launcher_gui.py's module-level _APPS list without requiring
    tkinter to actually be importable (this repo's test env may lack a
    display, but _APPS is defined before any Tk object is constructed)."""
    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / "launcher_gui.py"
    spec = importlib.util.spec_from_file_location("launcher_gui", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("launcher_gui", module)
    spec.loader.exec_module(module)
    return module._APPS


def test_catalog_scripts_exist_on_disk():
    missing = [e.id for e in CATALOG if e.script and not script_path(e).exists()]
    assert not missing, f"catalog entries with missing scripts: {missing}"


def test_catalog_ids_are_unique():
    ids = [e.id for e in CATALOG]
    assert len(ids) == len(set(ids))


def test_catalog_stages_are_all_known():
    unknown = [e.id for e in CATALOG if e.stage not in STAGE_ORDER]
    assert not unknown, f"entries with unrecognised stage: {unknown}"


def test_catalog_by_id_matches_catalog():
    assert set(CATALOG_BY_ID) == {e.id for e in CATALOG}
    for entry_id, entry in CATALOG_BY_ID.items():
        assert entry.id == entry_id


def test_tkinter_apps_are_generated_from_the_catalog():
    """launcher_gui.py's _APPS is built from every script-backed CATALOG
    entry (any runner), excluding "mso_studio" (which has its own dedicated
    button in that window instead), so this can only fail if that
    filter/derivation itself is broken -- not from someone forgetting to
    update a second, independently hand-maintained list (the old failure
    mode this test used to guard against).

    Round 4 (WORKBENCH_AUDIT_2026-08-10.md, defect R2): this test's own
    `expected` used to hardcode the OLD RUNNER_SCIENCE-only filter, which
    the Phase-1 GUI integration overhaul had already made stale (18 of 20
    catalog entries are RUNNER_GUI_VENV) -- so this test kept "passing"
    against a comparison that was itself wrong, right up until
    launcher_gui.py's own filter was fixed to match its real 5-tuple
    `_APPS` shape (title, script, description, runner, args), which is
    when this test's stale 3-tuple `expected` finally surfaced as a
    mismatch.
    """
    expected = [
        (e.title, e.script, e.description, e.runner, e.args)
        for e in CATALOG
        if e.script and e.id != "mso_studio"
    ]
    assert _load_tkinter_apps() == expected


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_every_catalog_icon_resolves(qapp):
    import qtawesome as qta
    for entry in CATALOG:
        icon = qta.icon(entry.icon)
        assert not icon.isNull(), f"{entry.id}: qtawesome icon {entry.icon!r} failed to resolve"
