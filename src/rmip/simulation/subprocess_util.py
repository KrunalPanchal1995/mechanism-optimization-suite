"""
rmip/simulation/subprocess_util.py
===================================
Shared helpers for the native-solver subprocess bridges (Zero-RK, MUQ-SAC,
RMG).

Two hazards this consolidates (M18b, bug.md):

1. **Environment leakage.** Zero-RK's ``constVolumeWSR.x`` / ``*_mpi.x`` and the
   MUQ-SAC pipeline are dynamically linked against the system ``libstdc++``. If
   the parent process inherited an ``LD_LIBRARY_PATH`` pointing at a *foreign*
   toolchain (a classic offender: an ANSYS install that ships its own, older
   ``libstdc++.so.6``), that foreign lib is picked up first and the solver
   fails at load time with a ``GLIBCXX_… not found`` / symbol error that has
   nothing to do with the mechanism. ``sanitized_subprocess_env`` strips those
   foreign entries out of a copy of the environment before launch.

   The same ANSYS setup also globally exports ``LD_PRELOAD=libstdc++.so.6.0.28``
   — do **not** strip this here. Investigated 2026-08-06: on this machine the
   system-wide ``ld.so.cache`` itself resolves the plain ``libstdc++.so.6``
   SONAME to an even older, broken ANSYS symlink target
   (``libstdc++.so.6.0.25``, missing ``GLIBCXX_3.4.26``) *regardless* of
   ``LD_LIBRARY_PATH`` — that global ``LD_PRELOAD`` is what steers Zero-RK's
   MPI binaries away from that broken default and onto a working copy.
   Stripping it here broke `test_zerork_gsa.py`'s real-binary tests. The RMG
   backend has the opposite need (rdkit requires ``GLIBCXX_3.4.29+``, newer
   than what that preload provides) — that is handled by the RMG-specific
   ``rmg_subprocess_env`` below, not by changing this shared default.

2. **Unbounded timeouts.** The bridge ``run*`` methods used to default
   ``timeout=None``, so a hung solver (a non-converging flame solve, a wedged
   MPI rank) would block the calling thread forever. ``DEFAULT_SUBPROCESS_TIMEOUT_S``
   gives every bridge a finite default; callers can still pass an explicit
   value (including a deliberately larger one) when they know a run is slow.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

# Finite default so a wedged native solver can't block a bridge call forever.
# 30 minutes comfortably covers IDT/JSR/PFR/GSA batch runs on the mechanism
# sizes this suite handles; callers with a genuinely longer run pass an explicit
# larger timeout.
DEFAULT_SUBPROCESS_TIMEOUT_S: float = 1800.0

# Substrings (case-insensitive) that mark an LD_LIBRARY_PATH entry as belonging
# to a foreign toolchain whose libstdc++ must not shadow the system one.
_FOREIGN_LIBPATH_MARKERS = ("ansys", "/ansys_inc/", "fluent", "matlab")


def _is_foreign_libpath(entry: str) -> bool:
    low = entry.lower()
    return any(marker in low for marker in _FOREIGN_LIBPATH_MARKERS)


def sanitized_subprocess_env(
    extra: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Return a copy of ``os.environ`` with foreign ``LD_LIBRARY_PATH`` entries
    removed, so a native solver links against the system ``libstdc++`` (M18b).

    Parameters
    ----------
    extra : dict, optional
        Additional environment variables to set/override on top of the
        sanitized copy (e.g. ``{"NUMEXPR_MAX_THREADS": "1"}``).
    """
    env = os.environ.copy()

    ld = env.get("LD_LIBRARY_PATH")
    if ld:
        kept = [p for p in ld.split(os.pathsep) if p and not _is_foreign_libpath(p)]
        if kept:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(kept)
        else:
            env.pop("LD_LIBRARY_PATH", None)

    if extra:
        env.update(extra)
    return env


def rmg_subprocess_env(rmg_python: str) -> Dict[str, str]:
    """Environment for launching a subprocess inside the RMG conda env.

    rdkit (an RMG dependency) requires ``GLIBCXX_3.4.29+``. Starting from
    :func:`sanitized_subprocess_env` isn't enough here: this machine's global
    ``LD_PRELOAD=libstdc++.so.6.0.28`` (needed by Zero-RK, see the module
    docstring — do not strip it there) forces a libstdc++ that satisfies
    Zero-RK's older ``GLIBCXX_3.4.26`` requirement but *not* rdkit's newer
    one. Unset it and put the RMG env's own ``lib/`` (which ships a
    sufficiently new libstdc++) first on ``LD_LIBRARY_PATH`` so rdkit's
    import succeeds regardless of what the parent process happens to export.
    """
    env = sanitized_subprocess_env()
    env.pop("LD_PRELOAD", None)

    rmg_lib = str(Path(rmg_python).resolve().parent.parent / "lib")
    existing = env.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = (
        rmg_lib if not existing else rmg_lib + os.pathsep + existing
    )
    return env


def resolve_timeout(timeout: Optional[float]) -> Optional[float]:
    """Map a caller's ``timeout`` to the finite default when unset.

    ``None`` means "use the finite default" (the safe behaviour). A caller that
    truly wants no timeout can pass a very large number explicitly.
    """
    return DEFAULT_SUBPROCESS_TIMEOUT_S if timeout is None else timeout
