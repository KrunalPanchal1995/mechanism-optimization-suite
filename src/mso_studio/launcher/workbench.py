"""
mso_studio.launcher.workbench
================================
An Ansys-Workbench-style pipeline: a user-composed, ordered sequence of
catalog modules ("steps") that lives with the project, the same way
Fluent's project schematic stores Geometry -> Mesh -> Setup -> Run ->
Postprocess as an explicit chain rather than leaving the order implicit.

Pure Python, no Qt -- mirrors ``catalog.py``/``process.py`` so this module
stays importable (and unit-testable) without a display. It is also
deliberately independent of ``mso_project`` at import time: the on-disk
format here (``workbench.yaml``) is owned entirely by the launcher, and
status is joined against the project database only through the read-only
functions in ``project_link.py``, never by importing ``mso_project``
directly from here. That keeps this module usable even on a checkout where
``src/mso_project`` doesn't exist yet or is mid-edit in another session.

``workbench.yaml`` lives at ``<project_dir>/workbench.yaml`` -- a sibling of
``mso_project``'s own ``project.yaml`` and ``artifacts.db``, but a distinct
file that package does not read, write, or know about.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from mso_studio.launcher.catalog import CATALOG_BY_ID

SCHEMA_VERSION = 1
WORKBENCH_FILENAME = "workbench.yaml"

# Catalog entry id -> the `pipeline` string mso_project's `runs` table (and
# its footprints/ registry) uses to classify a run's output files. Most
# catalog ids already match a footprint name exactly; this only lists the
# exceptions. An empty string means "no footprint exists for this module
# yet" -- status_for_steps() then always reports STATUS_UNKNOWN for it,
# which is honest (we have no way to tell) rather than guessing.
PIPELINE_FOR_ENTRY: Dict[str, str] = {
    "mechanism_fusion": "mean_value_fusion",
    "target_add_opt": "",
    "mso_studio": "",
    "doctor": "",
}


def pipeline_for_entry(entry_id: str) -> str:
    """The ``runs.pipeline`` value a step's launches would be recorded
    under, or ``""`` if none is known."""
    if entry_id in PIPELINE_FOR_ENTRY:
        return PIPELINE_FOR_ENTRY[entry_id]
    return entry_id


# -- step status -------------------------------------------------------
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_UNKNOWN = "unknown"

_RUN_STATUS_MAP = {
    "running": STATUS_RUNNING,
    "done": STATUS_DONE,
    "failed": STATUS_FAILED,
    "unknown": STATUS_UNKNOWN,
}

# The README-documented end-to-end sequence (MarkDownFiles/README.md's
# "Conference paper workflow", steps 1-8), expressed as catalog ids. Seeded
# into every new project so a Workbench starts as a useful checklist rather
# than an empty canvas -- the same way a fresh Fluent project already has
# five empty-but-labelled schematic cells.
DEFAULT_TEMPLATE_IDS: List[str] = [
    "mechanism_converter",
    "mechanism_check",
    "mechanism_fusion",
    "target_add_opt",
    "nominal_sim",
    "brute_force_sa",
    "uncertainty_generator",
    "mso_compare",
    "forward_uq",
]
DEFAULT_TEMPLATE_OPTIONAL = {"mechanism_fusion", "forward_uq"}


@dataclass
class WorkbenchStep:
    entry_id: str
    label: str = ""
    note: str = ""
    optional: bool = False

    def display_label(self) -> str:
        if self.label:
            return self.label
        entry = CATALOG_BY_ID.get(self.entry_id)
        return entry.title if entry is not None else self.entry_id

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkbenchStep":
        return cls(
            entry_id=str(data.get("entry_id", "")),
            label=str(data.get("label", "") or ""),
            note=str(data.get("note", "") or ""),
            optional=bool(data.get("optional", False)),
        )


@dataclass
class Workbench:
    steps: List[WorkbenchStep] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    # -- construction -----------------------------------------------
    @classmethod
    def default_template(cls) -> "Workbench":
        return cls(steps=[
            WorkbenchStep(entry_id=entry_id, optional=entry_id in DEFAULT_TEMPLATE_OPTIONAL)
            for entry_id in DEFAULT_TEMPLATE_IDS
            if entry_id in CATALOG_BY_ID
        ])

    @classmethod
    def load(cls, project_dir: str) -> "Workbench":
        """Load ``workbench.yaml`` from *project_dir*, or a fresh
        :meth:`default_template` if the file doesn't exist yet. Never
        raises on a malformed or half-written file, and never lets an
        unrecognised entry id (a catalog entry renamed or removed since the
        file was written) break the whole load -- that step is just
        dropped. The launcher must always be able to open a project."""
        path = Path(project_dir) / WORKBENCH_FILENAME
        if not path.is_file():
            return cls.default_template()
        try:
            raw = yaml.safe_load(path.read_text()) or {}
        except (yaml.YAMLError, OSError):
            return cls.default_template()
        if not isinstance(raw, dict):
            return cls.default_template()

        schema_version = int(raw.get("schema_version", SCHEMA_VERSION) or SCHEMA_VERSION)
        steps_raw = raw.get("steps") or []
        steps: List[WorkbenchStep] = []
        for item in steps_raw:
            if not isinstance(item, dict):
                continue
            step = WorkbenchStep.from_dict(item)
            if step.entry_id not in CATALOG_BY_ID:
                continue
            steps.append(step)
        return cls(steps=steps, schema_version=schema_version)

    # -- persistence --------------------------------------------------
    def save(self, project_dir: str) -> None:
        path = Path(project_dir) / WORKBENCH_FILENAME
        payload = {
            "schema_version": self.schema_version,
            "steps": [step.to_dict() for step in self.steps],
        }
        path.write_text(yaml.safe_dump(payload, default_flow_style=False, sort_keys=False))

    # -- mutation -------------------------------------------------------
    def insert(self, index: int, entry_id: str) -> WorkbenchStep:
        if entry_id not in CATALOG_BY_ID:
            raise ValueError(f"unknown catalog entry id: {entry_id!r}")
        step = WorkbenchStep(entry_id=entry_id)
        index = max(0, min(index, len(self.steps)))
        self.steps.insert(index, step)
        return step

    def append(self, entry_id: str) -> WorkbenchStep:
        return self.insert(len(self.steps), entry_id)

    def remove(self, index: int) -> None:
        if 0 <= index < len(self.steps):
            del self.steps[index]

    def move(self, index: int, delta: int) -> int:
        """Move the step at *index* by *delta* positions (-1 = left/earlier,
        +1 = right/later). Returns the step's new index; a move past either
        end is clamped, not an error, so callers can wire "move left"/"move
        right" buttons without bounds-checking themselves."""
        if not (0 <= index < len(self.steps)):
            return index
        new_index = max(0, min(index + delta, len(self.steps) - 1))
        if new_index != index:
            step = self.steps.pop(index)
            self.steps.insert(new_index, step)
        return new_index


def status_for_steps(steps: List[WorkbenchStep], runs: List[Dict[str, Any]]) -> Dict[str, str]:
    """Pure join between a Workbench's steps and ``Project.list_runs()``'s
    output. Takes plain dicts (not a live ``Project``) so this has no
    dependency on ``mso_project`` being importable and is trivially testable
    with fixture data. For each step's pipeline, the *latest* matching run
    (by ``started_at``) decides the status; a step with no matching runs is
    PENDING, and a step whose module has no known pipeline mapping is
    UNKNOWN (not guessed)."""
    result: Dict[str, str] = {}
    runs_by_pipeline: Dict[str, List[Dict[str, Any]]] = {}
    for run in runs:
        pipeline = run.get("pipeline")
        if not pipeline:
            continue
        runs_by_pipeline.setdefault(pipeline, []).append(run)

    for step in steps:
        pipeline = pipeline_for_entry(step.entry_id)
        if not pipeline:
            result[step.entry_id] = STATUS_UNKNOWN
            continue
        candidates = runs_by_pipeline.get(pipeline) or []
        if not candidates:
            result[step.entry_id] = STATUS_PENDING
            continue
        latest = max(candidates, key=lambda r: r.get("started_at") or "")
        result[step.entry_id] = _RUN_STATUS_MAP.get(latest.get("status"), STATUS_UNKNOWN)
    return result
