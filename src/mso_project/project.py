"""
mso_project.project
====================
``Project`` -- the top-level object this package builds around: a directory
that remembers every artifact location a pipeline run produced, wherever it
was produced, and reloads that memory on restart.

Design (see the "mso_project" plan for the full rationale):

* **Index, don't relocate.** Registering a file records its absolute path;
  the file itself never moves. Non-invasive, and safe with read-only/live
  external run directories this repo already has conventions against
  touching (the ``p_PRS`` automation campaign, live MUQ-SAC studies).
* **Two files on disk, both human-inspectable, neither a proprietary
  format:** ``project.yaml`` (name/description/tags -- hand-editable) and
  ``artifacts.db`` (a SQLite index -- see ``schema.py``), mirroring this
  repo's own conventions (``target.opt``'s YAML, ``rmip/database``'s
  SQLite) rather than an opaque commercial project-file format.
* **Runs reuse ``mso_studio.core.runrecord.RunRecord``/``RunConfig``**
  (pure stdlib + optional ``platformdirs``, no Qt import, safe outside
  ``.venv-gui``) instead of inventing a parallel run-metadata schema --
  but saved under ``<project_dir>/runs/`` rather than that module's
  machine-global ``platformdirs`` location, via this module's own
  ``runs_root()``. ``artifacts.db`` is the file-level index RunRecord
  alone doesn't provide at scale (hundreds of files per run).

Canonical ``src/MUQ-SAC/run_script.py`` capture + resume recipe
-----------------------------------------------------------------
::

    from mso_project import Project

    proj = Project.open("/data/PROJECTS/AMMONIA_2026")
    with proj.capture("run_script", "case1", working_dir=run_root,
                       backend_command=[py, "src/MUQ-SAC/run_script.py", "target.opt"]):
        subprocess.run([py, "src/MUQ-SAC/run_script.py", "target.opt"], cwd=run_root)

    # ... later, possibly after a restart, in a fresh directory:
    proj = Project.open("/data/PROJECTS/AMMONIA_2026")
    run_id = proj.list_runs(pipeline="run_script")[-1]["run_id"]
    proj.resume(run_id, target_dir="/scratch/resumed_case1")   # copies, by default
    subprocess.run([py, "src/MUQ-SAC/run_script.py", "target.opt"], cwd="/scratch/resumed_case1")
    # existence-checks like `if "unsrt.pkl" not in os.listdir():` now pass,
    # so run_script.py picks up exactly where it left off. Wrap that second
    # subprocess call in its own proj.capture(...) to register the new
    # generation -- mirroring run_script.py's own archive_previous_study /
    # .study_manifest.json per-study convention rather than fighting it.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from .schema import get_connection, init_db

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.normpath(os.path.join(_THIS_DIR, ".."))
if _SRC_DIR not in sys.path:
    # Append, never insert at 0 -- src/mso_io is a stdlib-shadowing risk the
    # same way src/io used to be; appending keeps stdlib precedence (see
    # feedback_env_isolation / conftest.py's own note on this).
    sys.path.append(_SRC_DIR)

try:
    from mso_studio.core.runrecord import RunConfig, RunRecord, sha256_of_file
    _HAS_RUNRECORD = True
except ImportError:
    _HAS_RUNRECORD = False

from .footprints import classify as _classify_artifact

_KIND_BY_EXT = {
    ".csv": "csv", ".tsv": "csv",
    ".png": "png", ".jpg": "png", ".jpeg": "png", ".svg": "png",
    ".pdf": "pdf",
    ".pkl": "pkl", ".pickle": "pkl",
    ".yaml": "yaml", ".yml": "yaml",
    ".json": "json",
    ".txt": "txt", ".save": "txt",
    ".log": "log", ".out": "log", ".lst": "log",
    ".md": "doc", ".rst": "doc", ".docx": "doc", ".tex": "doc",
    ".ipynb": "notebook",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


def _kind_of(path: str) -> str:
    return _KIND_BY_EXT.get(Path(path).suffix.lower(), "other")


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip()).strip("_")
    return slug or "untitled"


def _snapshot_dir(directory: str) -> Dict[str, Any]:
    """{relpath: (size, mtime)} for every file under directory. Cheap
    (stat only, no hashing) so this is safe to run before AND after a
    capture() block even over a Perturbed_Mech/-sized tree."""
    out: Dict[str, Any] = {}
    root = Path(directory)
    if not root.is_dir():
        return out
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            p = Path(dirpath) / fname
            try:
                st = p.stat()
            except OSError:
                continue
            out[str(p.relative_to(root))] = (st.st_size, st.st_mtime)
    return out


def _run_dir_for(project_root: Path, pipeline: str, label: str, started_at: str) -> Path:
    """mso_project's own run directory -- deliberately NOT
    mso_studio.core.runrecord.runs_root() (that's a machine-global,
    platformdirs-based location shared across every mso_studio consumer;
    this project's runs live under its own directory so the project stays
    self-contained and portable)."""
    safe_label = _slugify(label)
    stamp = started_at.replace(":", "-").split(".")[0]
    return project_root / "runs" / pipeline / f"{stamp}_{safe_label}"


class Project:
    """A project directory: ``project.yaml`` + ``artifacts.db`` + ``runs/`` +
    ``notes/``. Construct via :meth:`create` or :meth:`open`, not directly."""

    def __init__(self, project_dir: str, conn: sqlite3.Connection, info: dict):
        self.root = Path(project_dir).resolve()
        self._conn = conn
        self.info = info  # {"name", "description", "tags", "created_at", ...}

    # ------------------------------------------------------------------
    # Create / open (the "remember on restart" contract)
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, project_dir: str, name: str, description: str = "",
               tags: Optional[List[str]] = None) -> "Project":
        """Create a new project directory. Raises FileExistsError if
        ``project.yaml`` already exists there (use :meth:`open` instead)."""
        root = Path(project_dir).resolve()
        yaml_path = root / "project.yaml"
        if yaml_path.exists():
            raise FileExistsError(
                f"{yaml_path} already exists -- use Project.open() to reopen "
                f"an existing project, not create() again.")
        root.mkdir(parents=True, exist_ok=True)
        (root / "notes").mkdir(exist_ok=True)
        (root / "runs").mkdir(exist_ok=True)
        (root / "reports").mkdir(exist_ok=True)

        created_at = _now_iso()
        info = {
            "schema_version": 1,
            "name": name,
            "description": description,
            "created_at": created_at,
            "tags": list(tags or []),
        }
        yaml_path.write_text(yaml.safe_dump(info, default_flow_style=False, sort_keys=False))

        conn = init_db(str(root / "artifacts.db"))
        conn.execute(
            "INSERT INTO project_info (id, name, description, tags_json, created_at, schema_version) "
            "VALUES (1, ?, ?, ?, ?, ?)",
            (name, description, json.dumps(list(tags or [])), created_at, 1),
        )
        conn.commit()
        return cls(str(root), conn, info)

    @classmethod
    def open(cls, project_dir: str) -> "Project":
        """Reopen an existing project -- this is the "remembers on restart"
        entry point. Raises FileNotFoundError if no project.yaml is there."""
        root = Path(project_dir).resolve()
        yaml_path = root / "project.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"No project.yaml at {root} -- use Project.create() to make "
                f"a new project first.")
        info = yaml.safe_load(yaml_path.read_text()) or {}
        db_path = root / "artifacts.db"
        conn = init_db(str(db_path)) if not db_path.exists() else get_connection(str(db_path))
        return cls(str(root), conn, info)

    def _save_manifest(self) -> None:
        (self.root / "project.yaml").write_text(
            yaml.safe_dump(self.info, default_flow_style=False, sort_keys=False))

    def close(self) -> None:
        self._conn.commit()
        self._conn.close()

    def __enter__(self) -> "Project":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, path: str, role: Optional[str] = None,
                 run_id: Optional[str] = None, tags: Optional[List[str]] = None,
                 kind: Optional[str] = None, report_include: str = "auto",
                 compute_checksum: bool = False) -> int:
        """Register one file. Idempotent for the same (run_id, path) pair --
        re-registering updates size/mtime/role rather than duplicating a row.

        Returns the artifact_id. Raises FileNotFoundError if path doesn't
        exist (register a file only once it's actually been written --
        callers wanting to track an EXPECTED-but-not-yet-written path should
        use capture(), which diffs after the run completes)."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"register(): not a file: {path}")
        abs_path = str(p.resolve())
        size = p.stat().st_size
        mtime = p.stat().st_mtime
        checksum = sha256_of_file(abs_path) if (compute_checksum and _HAS_RUNRECORD) else None
        role = role or "unclassified"
        kind = kind or _kind_of(abs_path)
        registered_at = _now_iso()
        tags_json = json.dumps(list(tags or []))

        existing = self._conn.execute(
            "SELECT artifact_id FROM artifacts WHERE path = ? AND "
            "(run_id IS ? OR run_id = ?)", (abs_path, run_id, run_id)
        ).fetchone()
        if existing:
            self._conn.execute(
                "UPDATE artifacts SET role=?, kind=?, size_bytes=?, "
                "checksum_sha256=COALESCE(?, checksum_sha256), mtime=?, "
                "registered_at=?, tags_json=?, report_include=? "
                "WHERE artifact_id=?",
                (role, kind, size, checksum, mtime, registered_at, tags_json,
                 report_include, existing["artifact_id"]))
            self._conn.commit()
            return existing["artifact_id"]

        cur = self._conn.execute(
            "INSERT INTO artifacts (run_id, path, role, kind, size_bytes, "
            "checksum_sha256, mtime, registered_at, tags_json, report_include) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, abs_path, role, kind, size, checksum, mtime,
             registered_at, tags_json, report_include))
        self._conn.commit()
        return cur.lastrowid

    def register_many(self, paths: Iterable[str], **kw) -> List[int]:
        return [self.register(p, **kw) for p in paths if Path(p).is_file()]

    def set_report_include(self, artifact_id: int, value: str) -> None:
        if value not in ("auto", "main", "supplementary", "exclude"):
            raise ValueError(f"invalid report_include: {value!r}")
        self._conn.execute("UPDATE artifacts SET report_include=? WHERE artifact_id=?",
                            (value, artifact_id))
        self._conn.commit()

    # ------------------------------------------------------------------
    # Notes (freeform user documentation, tracked like any other artifact)
    # ------------------------------------------------------------------

    def add_note(self, title: str, body_markdown: str) -> int:
        notes_dir = self.root / "notes"
        notes_dir.mkdir(exist_ok=True)
        slug = _slugify(title)
        path = notes_dir / f"{slug}.md"
        n = 2
        while path.exists():
            path = notes_dir / f"{slug}-{n}.md"
            n += 1
        path.write_text(f"# {title}\n\n{body_markdown}\n")
        return self.register(str(path), role="user_note", kind="doc")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def list_runs(self, pipeline: Optional[str] = None) -> List[Dict[str, Any]]:
        if pipeline:
            rows = self._conn.execute(
                "SELECT * FROM runs WHERE pipeline=? ORDER BY started_at", (pipeline,)).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM runs ORDER BY started_at").fetchall()
        return [dict(r) for r in rows]

    def list_artifacts(self, run_id: Optional[str] = None, role: Optional[str] = None,
                        kind: Optional[str] = None) -> List[Dict[str, Any]]:
        clauses, params = [], []
        if run_id is not None:
            clauses.append("run_id = ?"); params.append(run_id)
        if role is not None:
            clauses.append("role = ?"); params.append(role)
        if kind is not None:
            clauses.append("kind = ?"); params.append(kind)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM artifacts {where} ORDER BY registered_at", params).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["tags"] = json.loads(d.pop("tags_json") or "[]")
            out.append(d)
        return out

    # ------------------------------------------------------------------
    # Capture -- directory-diff auto-registration around an arbitrary
    # pipeline invocation, requiring zero code changes in that pipeline.
    # ------------------------------------------------------------------

    @contextmanager
    def capture(self, pipeline: str, label: str, working_dir: str,
                backend_command: Optional[List[str]] = None,
                inputs: Optional[Dict[str, str]] = None,
                parameters: Optional[Dict[str, Any]] = None):
        """Snapshot ``working_dir`` before and after the ``with`` block,
        register every new/changed file, and save a
        ``mso_studio.core.runrecord.RunRecord`` describing the run --
        reused as-is for run-level metadata rather than inventing a
        parallel schema (see this module's docstring).

        Works for ANY pipeline invoked inside the block (a direct Python
        call or a ``subprocess.run(..., cwd=working_dir)``) with no code
        changes to that pipeline. Roles are assigned via
        ``FOOTPRINTS[pipeline]`` where a path matches a known pattern, else
        the generic extension-based fallback (``role="unclassified"`` --
        never silently dropped).

        On an exception inside the block: whatever files WERE produced
        before the failure are still registered (partial results are
        useful), the run is marked ``status="failed"``, and the exception
        is re-raised unchanged.
        """
        working_dir = str(Path(working_dir).resolve())
        started_at = _now_iso()
        before = _snapshot_dir(working_dir)

        config = None
        if _HAS_RUNRECORD:
            config = RunConfig(
                module=pipeline, label=label,
                backend_command=list(backend_command or []),
                working_dir=working_dir,
                inputs=dict(inputs or {}), parameters=dict(parameters or {}))

        run_dir = _run_dir_for(self.root, pipeline, label, started_at)
        run_id = str(run_dir)
        self._conn.execute(
            "INSERT INTO runs (run_id, pipeline, label, working_dir, started_at, status) "
            "VALUES (?, ?, ?, ?, ?, 'running')",
            (run_id, pipeline, label, working_dir, started_at))
        self._conn.commit()

        status = "done"
        error: Optional[str] = None
        try:
            yield run_id
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            after = _snapshot_dir(working_dir)
            changed = {rel for rel, meta in after.items() if before.get(rel) != meta}
            record_path = None
            if _HAS_RUNRECORD and config is not None:
                record = RunRecord(config=config, started_at=started_at)
                record.capture_input_hashes()
                for rel in sorted(changed):
                    abs_path = str(Path(working_dir) / rel)
                    kind = _classify_artifact(rel, pipeline, abs_path)[1]
                    if kind == "csv":
                        record.tables[rel] = abs_path
                    elif kind in ("png", "pdf"):
                        record.figures[rel] = abs_path
                if error:
                    record.mark_finished("failed", error=error)
                else:
                    record.mark_finished("done")
                record_path = str(record.save(run_dir=run_dir))

            finished_at = _now_iso()
            self._conn.execute(
                "UPDATE runs SET finished_at=?, status=?, record_path=? WHERE run_id=?",
                (finished_at, status, record_path, run_id))
            self._conn.commit()

            for rel in sorted(changed):
                abs_path = str(Path(working_dir) / rel)
                role, kind = _classify_artifact(rel, pipeline, abs_path)
                if Path(abs_path).is_file():
                    self.register(abs_path, role=role, run_id=run_id, kind=kind)

    def scan(self, directory: str, pipeline: Optional[str] = None,
             run_id: Optional[str] = None) -> List[int]:
        """Retro-import an existing directory (e.g. a run from before this
        project existed, or one produced by a tool run outside a
        capture() block). Registers every file under ``directory``,
        classified the same way capture() does.

        If ``run_id`` is omitted, a new ``runs`` row is created with
        ``status="unknown"`` (its completion status was never observed) so
        it still shows up in list_runs(); pass an existing run_id to attach
        these files to a run capture() already created instead.
        """
        directory = str(Path(directory).resolve())
        if run_id is None:
            started_at = _now_iso()
            run_dir = _run_dir_for(self.root, pipeline or "scan", Path(directory).name, started_at)
            run_id = str(run_dir)
            self._conn.execute(
                "INSERT INTO runs (run_id, pipeline, label, working_dir, started_at, "
                "finished_at, status) VALUES (?, ?, ?, ?, ?, ?, 'unknown')",
                (run_id, pipeline or "scan", Path(directory).name, directory,
                 started_at, started_at))
            self._conn.commit()

        artifact_ids = []
        for rel, _meta in _snapshot_dir(directory).items():
            abs_path = str(Path(directory) / rel)
            role, kind = _classify_artifact(rel, pipeline, abs_path)
            artifact_ids.append(self.register(abs_path, role=role, run_id=run_id, kind=kind))
        return artifact_ids

    # ------------------------------------------------------------------
    # Resume -- materialize a run's tracked artifacts into a working dir.
    # ------------------------------------------------------------------

    def resume(self, run_id: str, target_dir: str, mode: str = "copy") -> Dict[str, str]:
        """Materialize every artifact of ``run_id`` into ``target_dir`` at
        the same relative layout it was captured from, so a pipeline can be
        re-invoked with ``cwd=target_dir`` and pick up exactly where it
        left off. Returns ``{relpath: materialized_path}`` (relpath, not
        role, since many artifacts commonly share one role). See
        ``restore.resume`` for the copy-vs-link safety rationale (default
        ``mode="copy"``).

        **Contract: pause / finished / crashed sources only.** Resuming
        against a *still actively writing* source is racy (WORKBENCH_AUDIT_
        2026-08-10.md, defect B6) -- confirmed live: `scan()`/`resume()`
        against a still-running `good_ffcm1` campaign correctly recovered
        `unsrt.pkl`, but the resumed copy's SA subprocess then failed
        differently than the source would have, because a directory the
        source created moments after the snapshot (`nominal/`) didn't exist
        yet in either copy at snapshot time. There is no code-level fix for
        this -- it is an inherent limitation of snapshotting a moving
        target -- so the caller's own responsibility is to only resume a
        source that is paused (SIGSTOP), finished, or crashed. A quick
        tell that a source is still writing: any matched artifact whose
        mtime is within the last few seconds of `scan()`/`resume()` being
        called.
        """
        from .restore import resume as _resume
        return _resume(self, run_id, target_dir, mode=mode)
