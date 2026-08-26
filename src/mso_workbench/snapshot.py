"""
mso_workbench.snapshot
=========================
Writes the on-disk record of a pause: the real in-memory state at that
point (whatever's picklable), a manifest describing exactly where/when/
what, and symlinks to the on-disk artifacts already produced by that stage
-- so a notebook opened against this snapshot can rebuild the exact
situation the pipeline was in, not an approximation of it.

Per-value (not whole-dict) pickling: ``run_script.py`` holds live objects
at some stages -- ``SSM`` (a simulation manager with open subprocess
handles), pool references, etc. -- that cannot be pickled at all. Pickling
the whole namespace dict in one call means one such value fails the entire
snapshot; pickling each value independently means only that one value is
skipped (recorded, not silently dropped) and everything else is preserved.
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from mso_workbench.control import snapshots_dir

NAMESPACE_FILE = "namespace.pkl"
MANIFEST_FILE = "manifest.json"
ENV_FILE = "env.json"

# On-disk artifacts worth symlinking into a snapshot when present at
# run_dir's root -- the ones a notebook rebuilding this stage will want
# without re-deriving them (see the run-directory layout audited in
# src/mso_project/footprints/run_script.py).
_CANDIDATE_ARTIFACTS = (
    "unsrt.pkl",
    "DesignMatrix.csv",
    "a_type_samples_numpy.pkl",
    "b_type_samples_numpy.pkl",
    "c_type_samples_numpy.pkl",
    "sens_parameters.pkl",
    "sens_3p_parameters.pkl",
    ".study_manifest.json",
)


def _git_sha(cwd: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
            text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _pickle_namespace(namespace: Dict[str, Any], out_path: Path) -> list:
    """Pickle each value independently; return a list of
    {"key", "type", "repr"} for anything that couldn't be pickled."""
    picklable: Dict[str, Any] = {}
    failures = []
    for key, value in namespace.items():
        try:
            pickle.dumps(value)
        except Exception as exc:  # noqa: BLE001 - genuinely any object type can fail here
            failures.append({
                "key": key,
                "type": type(value).__name__,
                "repr": _safe_repr(value),
                "error": str(exc),
            })
            continue
        picklable[key] = value
    with open(out_path, "wb") as fh:
        pickle.dump(picklable, fh)
    return failures


def _safe_repr(value: Any) -> str:
    try:
        text = repr(value)
    except Exception:  # noqa: BLE001
        return f"<{type(value).__name__}: repr() failed>"
    return text if len(text) <= 500 else text[:500] + "...(truncated)"


def write_snapshot(
    run_dir, stage_id: str, namespace: Dict[str, Any],
    source_ref: Optional[Tuple[str, str]] = None,
) -> Path:
    """Write a full snapshot for a pause at *stage_id*. Returns the
    snapshot directory. *source_ref* is ``(file, function)`` for the
    breakpoint call site, purely descriptive."""
    run_dir = Path(run_dir)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    snap_dir = snapshots_dir(run_dir) / f"{stage_id.replace('.', '_')}_{timestamp}"
    snap_dir.mkdir(parents=True, exist_ok=True)

    unpicklable = _pickle_namespace(namespace, snap_dir / NAMESPACE_FILE)

    artifact_links = []
    for name in _CANDIDATE_ARTIFACTS:
        src = run_dir / name
        if src.exists():
            link = snap_dir / name
            try:
                if not link.exists():
                    os.symlink(src.resolve(), link)
                artifact_links.append(name)
            except OSError:
                pass  # symlinking is a convenience; a failure here must not lose the snapshot

    manifest = {
        "stage": stage_id,
        "timestamp": timestamp,
        "run_dir": str(run_dir.resolve()),
        "cwd": os.getcwd(),
        "pid": os.getpid(),
        "argv": sys.argv,
        "git_sha": _git_sha(str(run_dir)),
        "source_ref": {"file": source_ref[0], "function": source_ref[1]} if source_ref else None,
        "namespace_keys": list(namespace.keys()),
        "unpicklable": unpicklable,
        "artifact_links": artifact_links,
    }
    (snap_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))

    env = {
        "python_version": sys.version,
        "platform": sys.platform,
        "sys_path": sys.path,
        "packages": _best_effort_package_versions(),
    }
    (snap_dir / ENV_FILE).write_text(json.dumps(env, indent=2))

    return snap_dir


def _best_effort_package_versions() -> Dict[str, str]:
    versions = {}
    for name in ("numpy", "scipy", "pandas", "yaml", "sklearn", "matplotlib"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            continue
    return versions


def load_manifest(snapshot_dir) -> dict:
    return json.loads((Path(snapshot_dir) / MANIFEST_FILE).read_text())


def load_namespace(snapshot_dir) -> Dict[str, Any]:
    with open(Path(snapshot_dir) / NAMESPACE_FILE, "rb") as fh:
        return pickle.load(fh)


def list_snapshots(run_dir) -> list:
    """All snapshot directories for *run_dir*, oldest first."""
    base = snapshots_dir(run_dir)
    if not base.exists():
        return []
    return sorted(p for p in base.iterdir() if p.is_dir())
