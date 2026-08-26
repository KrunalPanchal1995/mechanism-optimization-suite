"""
mso_studio.launcher.project_link
===================================
The launcher's ONLY connection to ``mso_project`` (the project-storage
package: ``project.yaml`` manifest + ``artifacts.db`` SQLite index of runs
and artifacts, keyed by pipeline/footprint). ``mso_project`` is being built
in a separate, parallel session and, as of this module's writing, is
entirely untracked in git -- so treat its API as unstable ground and never
let a change there take the launcher down with it.

Every public function here:
  * imports ``mso_project`` lazily, inside the function body, wrapped in
    ``try/except`` -- never at module import time;
  * returns a safe default (``None``, ``[]``, ``False``) rather than
    raising, for every failure *except* the explicit ``*_project`` actions
    the user directly asked for (create/open), which raise
    :class:`ProjectLinkError` with a message a dialog can show as-is;
  * touches only the documented, read-mostly surface of ``Project``:
    ``open``, ``create``, ``list_runs``, ``list_artifacts``, ``.root``,
    ``.info``. It deliberately never calls ``capture()``, ``scan()``,
    ``register()``, ``resume()``, or anything under ``mso_project.report`` --
    those are mutating or process-spawning operations that belong to
    whatever originates a run, not to a launcher that only wants to *show*
    project state.

This module makes no changes to any file under ``src/mso_project`` and
never will -- see the launcher Workbench plan for why.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


class ProjectLinkError(RuntimeError):
    """Raised only by the explicit create/open actions below, with a
    message suitable for showing the user directly."""


def is_available() -> Tuple[bool, str]:
    """Whether ``mso_project`` can be imported right now, and if not, why.
    Call this before offering New/Open Project in the UI so a broken or
    mid-edit checkout of the other session's package degrades to "project
    features unavailable" instead of a crash on first use."""
    try:
        import mso_project  # noqa: F401
    except Exception as exc:  # pragma: no cover - exact exception type varies
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def is_project_dir(path: str) -> bool:
    """True if *path* looks like an ``mso_project`` project directory --
    checked by looking for ``project.yaml`` on disk, not by importing
    ``mso_project`` at all, so this works even if the package itself can't
    currently be imported."""
    try:
        return (Path(path) / "project.yaml").is_file()
    except OSError:
        return False


def read_manifest(path: str) -> Optional[Dict[str, Any]]:
    """Parse ``<path>/project.yaml`` directly with ``yaml.safe_load`` --
    deliberately NOT via ``Project.open()``, which would also touch (and,
    if missing, create) ``artifacts.db`` as a side effect. Used for cheap
    listings (e.g. the sidebar's recent-projects list) that only need a
    name, not a live database connection. Returns ``None`` on any error."""
    manifest_path = Path(path) / "project.yaml"
    try:
        raw = yaml.safe_load(manifest_path.read_text())
    except (OSError, yaml.YAMLError):
        return None
    return raw if isinstance(raw, dict) else None


def open_project(path: str):
    """Open an existing project at *path*. Raises :class:`ProjectLinkError`
    with a user-facing message on any failure -- missing ``mso_project``,
    missing ``project.yaml``, a corrupt manifest, or a schema this launcher
    doesn't understand yet."""
    try:
        from mso_project import Project
    except Exception as exc:
        raise ProjectLinkError(
            "Project features are unavailable (mso_project could not be "
            f"imported): {exc}"
        ) from exc
    try:
        return Project.open(path)
    except Exception as exc:
        raise ProjectLinkError(f"Could not open project at {path}: {exc}") from exc


def create_project(path: str, name: str, description: str = ""):
    """Create a new project at *path*. Raises :class:`ProjectLinkError` on
    any failure, including "a project already exists here"."""
    try:
        from mso_project import Project
    except Exception as exc:
        raise ProjectLinkError(
            "Project features are unavailable (mso_project could not be "
            f"imported): {exc}"
        ) from exc
    try:
        return Project.create(path, name=name, description=description)
    except Exception as exc:
        raise ProjectLinkError(f"Could not create project at {path}: {exc}") from exc


def list_runs(project: Any) -> List[Dict[str, Any]]:
    """``project.list_runs()``, or ``[]`` if that raises for any reason
    (e.g. the DB schema changed underneath us). Never lets a status refresh
    take down the poll loop."""
    try:
        return list(project.list_runs())
    except Exception:
        return []


def list_artifacts(project: Any, run_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """``project.list_artifacts(run_id=...)``, or ``[]`` on any error."""
    try:
        return list(project.list_artifacts(run_id=run_id))
    except Exception:
        return []


def artifact_count(project: Any, run_id: Optional[str] = None) -> int:
    return len(list_artifacts(project, run_id=run_id))
