"""
mso_project.restore
====================
Materialize a captured run's tracked artifacts back into a working
directory, at the same relative layout they were captured from, so a
pipeline can be re-invoked with ``cwd=target_dir`` and resume exactly where
it left off.

Copy vs. link
-------------
``src/MUQ-SAC/run_script.py`` mutates several of its own artifacts IN
PLACE on a subsequent run: it can ``shutil.rmtree("Perturbed_Mech")`` and
regenerate it (Bug-4 completeness check), and it deletes+rewrites
``Opt/Data/ResponseSurface/responsecoef_case-*.csv`` on every refit. If
those paths were symlinked back into ``target_dir``, that rmtree/rewrite
would delete or corrupt the PROJECT's own stored originals through the
link -- not just the working copy. Default ``mode="copy"`` makes that
structurally impossible; ``mode="link"`` is available (fast, space-free)
for callers who understand the tradeoff, and this module still warns (via
the returned ``warnings`` list, not a hard failure) whenever it links an
artifact whose role is known to be rewritten in place.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from .project import Project

# Roles known to be deleted/rewritten in place by their producing pipeline
# (see run_script.py's create_response_surface cache-bust and the Bug-4
# Perturbed_Mech completeness check) -- unsafe to symlink back.
UNSAFE_TO_LINK_ROLES = {
    "response_surface_coeff",
    "perturbed_mechanism",
    "outlier_perturbed_mechanism",
}


def resume(project: "Project", run_id: str, target_dir: str,
           mode: str = "copy") -> Dict[str, str]:
    """Materialize every artifact of ``run_id`` into ``target_dir``.

    Returns ``{relpath: materialized_absolute_path}`` -- keyed by the
    artifact's path relative to the run's original working_dir, NOT by
    role (a run commonly has many artifacts sharing one role, e.g. every
    ``Perturbed_Mech/mechanism_*.yaml`` is ``role="perturbed_mechanism"``,
    so a role-keyed mapping would silently drop all but one).

    Raises ValueError for an unknown run_id or an invalid mode.
    """
    if mode not in ("copy", "link"):
        raise ValueError(f"mode must be 'copy' or 'link', got {mode!r}")

    runs = {r["run_id"]: r for r in project.list_runs()}
    if run_id not in runs:
        raise ValueError(f"unknown run_id: {run_id!r}")
    working_dir = Path(runs[run_id]["working_dir"])

    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    materialized: Dict[str, str] = {}
    warnings_: List[str] = []

    for artifact in project.list_artifacts(run_id=run_id):
        src = Path(artifact["path"])
        if not src.is_file():
            warnings_.append(f"skipped (source missing on disk): {src}")
            continue
        try:
            rel = src.relative_to(working_dir)
        except ValueError:
            # Registered under a different root than this run's working_dir
            # (shouldn't happen via capture()/scan(), but stay defensive
            # rather than silently mis-placing a file) -- fall back to the
            # basename directly under target_dir.
            rel = Path(src.name)
            warnings_.append(f"{src} is not under working_dir {working_dir}; "
                              f"restored to target root as {rel}")

        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            dest.unlink()

        if mode == "copy":
            shutil.copy2(src, dest)
        else:
            if artifact["role"] in UNSAFE_TO_LINK_ROLES:
                warnings_.append(
                    f"linked {rel} (role={artifact['role']!r}) -- this role is "
                    f"rewritten/deleted in place by its producing pipeline; the "
                    f"project's stored original may be corrupted by the next run.")
            os.symlink(src.resolve(), dest)

        materialized[str(rel)] = str(dest)

    if warnings_:
        import sys
        for w in warnings_:
            print(f"[mso_project.restore] warning: {w}", file=sys.stderr)

    return materialized
