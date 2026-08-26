"""
mso_studio.core.project_manager
==================================
Adapter for the Project Manager module page: create/open an
``mso_project.Project`` (project.yaml + artifacts.db), browse its runs and
artifacts ("the recorder"), add notes, resume a run's artifacts into a
fresh directory, and build the cross-pipeline LaTeX report.

``mso_project`` is pure stdlib + PyYAML (see its own module docstring --
"safe outside .venv-gui"), confirmed importable in the GUI venv, so this
runs entirely in-process; no subprocess bridge needed, unlike
``core.target_workbench``.

Previously CLI-only (``python -m mso_project ...`` / ``mso_project/cli.py``)
with a read-only, launcher-only viewer (``launcher.project_link``, which
deliberately never calls the mutating operations this module wraps). This
is the first place a user can create/open a project, browse its full run
history, add notes, resume a run, and generate its report without leaving
MSO Studio.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from mso_studio.core.backends import REPO_ROOT

_SRC = str(REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.append(_SRC)


@dataclass
class ProjectInfo:
    root: str
    name: str
    description: str
    tags: List[str]
    created_at: str


def _to_info(project) -> ProjectInfo:
    info = project.info
    return ProjectInfo(
        root=str(project.root), name=info.get("name", ""),
        description=info.get("description", ""),
        tags=list(info.get("tags", [])), created_at=info.get("created_at", ""),
    )


def create_project(project_dir: str, name: str, description: str = "",
                    tags: Optional[List[str]] = None) -> ProjectInfo:
    from mso_project import Project
    project = Project.create(project_dir, name, description, tags)
    info = _to_info(project)
    project.close()
    return info


def open_project(project_dir: str) -> ProjectInfo:
    from mso_project import Project
    project = Project.open(project_dir)
    info = _to_info(project)
    project.close()
    return info


def list_runs(project_dir: str, pipeline: Optional[str] = None) -> List[Dict[str, Any]]:
    from mso_project import Project
    project = Project.open(project_dir)
    try:
        return project.list_runs(pipeline=pipeline)
    finally:
        project.close()


def list_artifacts(
    project_dir: str, run_id: Optional[str] = None, role: Optional[str] = None,
) -> List[Dict[str, Any]]:
    from mso_project import Project
    project = Project.open(project_dir)
    try:
        return project.list_artifacts(run_id=run_id, role=role)
    finally:
        project.close()


def add_note(project_dir: str, title: str, body_markdown: str) -> int:
    from mso_project import Project
    project = Project.open(project_dir)
    try:
        return project.add_note(title, body_markdown)
    finally:
        project.close()


def resume_run(project_dir: str, run_id: str, target_dir: str, mode: str = "copy") -> Dict[str, str]:
    from mso_project import Project
    project = Project.open(project_dir)
    try:
        return project.resume(run_id, target_dir, mode=mode)
    finally:
        project.close()


@dataclass
class ReportOutcome:
    main_tex: str
    main_pdf: Optional[str]
    supplementary_tex: str
    supplementary_pdf: Optional[str]
    notebook_path: str
    manifest_path: str
    readme_path: str
    compiled: bool


def build_project_report(project_dir: str, out_dir: Optional[str] = None) -> ReportOutcome:
    from mso_project import Project
    from mso_project.report.builder import build_report

    project = Project.open(project_dir)
    try:
        result = build_report(project, out_dir=out_dir)
    finally:
        project.close()
    return ReportOutcome(
        main_tex=result.main_tex, main_pdf=result.main_pdf,
        supplementary_tex=result.supplementary_tex, supplementary_pdf=result.supplementary_pdf,
        notebook_path=result.notebook_path, manifest_path=result.manifest_path,
        readme_path=result.readme_path, compiled=result.compiled,
    )
