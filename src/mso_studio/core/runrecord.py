"""
mso_studio.core.runrecord
==========================
The data contract between :mod:`mso_studio.core` and :mod:`mso_studio.gui`.

Every module (nominal simulation, mechanism comparison, sensitivity analysis,
optimization, mechanism-comparison automation, ...) produces a
:class:`RunRecord` when it finishes. The GUI never inspects backend output
files directly -- it reads the ``RunRecord`` the ``core`` adapter built. This
is also, verbatim, the report data source: ``reporting.build_report(record,
out_dir)`` needs nothing else.

Design rules (see the project's GUI guidelines doc):

* ``RunConfig`` captures *everything* needed to reproduce a run and is plain,
  JSON-serializable data -- no Qt, no numpy arrays with exotic dtypes.
* ``RunRecord`` captures the config plus provenance (versions, host, hash,
  timing) plus results (scalars/tables/figures) plus the full log.
* Every finished run is serialized to its own timestamped folder under
  :func:`runs_root`, so "the last run" is recoverable after a restart or a
  crash and reports remain generatable without re-running anything.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import platformdirs
    _HAS_PLATFORMDIRS = True
except ImportError:  # pragma: no cover - platformdirs is a light dependency
    _HAS_PLATFORMDIRS = False

APP_NAME = "mso-studio"
APP_AUTHOR = "mechanism-optimization-suite"

#: Bumped whenever RunRecord's on-disk schema changes in a way that could
#: break older report/GUI code trying to read a newer record (or vice versa).
SCHEMA_VERSION = 1


def runs_root() -> Path:
    """Return (creating if needed) the directory all RunRecords are stored under.

    Uses :mod:`platformdirs` so runs never live next to the executable (which
    may be read-only once frozen/installed); falls back to
    ``~/.mso-studio/runs`` if platformdirs is unavailable.
    """
    if _HAS_PLATFORMDIRS:
        base = Path(platformdirs.user_data_dir(APP_NAME, APP_AUTHOR))
    else:
        base = Path.home() / f".{APP_NAME}"
    root = base / "runs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def sha256_of_file(path: str | os.PathLike) -> Optional[str]:
    """SHA-256 hex digest of a file, or ``None`` if it doesn't exist / can't be read."""
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    try:
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def software_versions() -> Dict[str, str]:
    """Best-effort version snapshot stamped into every RunRecord and report."""
    versions = {
        "mso_studio": _studio_version(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for mod_name in ("PySide6", "matplotlib", "numpy", "yaml", "jinja2"):
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass
    return versions


def _studio_version() -> str:
    try:
        from mso_studio import __version__
        return __version__
    except ImportError:
        return "0.dev"


@dataclass
class RunConfig:
    """Everything needed to reproduce a run. Serializable to JSON/YAML.

    Parameters
    ----------
    module : str
        Which capability produced this config, e.g. ``"nominal_sim"``,
        ``"mso_compare"``, ``"sensitivity"``, ``"optimization"``,
        ``"automation"``.
    label : str
        Human-readable run label shown in the GUI and report title
        (e.g. ``"nominal_htc"``).
    backend_command : list of str
        The exact argv used to invoke the backend subprocess. Kept so the
        report's appendix can show precisely how to reproduce the run from a
        terminal.
    working_dir : str
        Directory the backend subprocess was launched from.
    inputs : dict
        Free-form key -> path/value map of the inputs that mattered (mechanism
        file, targets file, thermo/trans files, ``target.opt`` path, ...).
        Every path-looking value is hashed into ``input_hashes`` at record
        time (see :meth:`RunRecord.capture_input_hashes`).
    parameters : dict
        Free-form key -> value map of scalar/string settings the user chose
        in the GUI form (thresholds, algorithm names, thread counts, ...).
    extra : dict
        Anything module-specific that doesn't fit the above.
    """

    module: str
    label: str
    backend_command: List[str] = field(default_factory=list)
    working_dir: str = ""
    inputs: Dict[str, str] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class RunRecord:
    """The complete, self-describing result of one run.

    This is the sole input to :func:`mso_studio.core.reporting.build_report`
    and the sole thing the GUI holds onto as ``self.last_run`` -- see the
    project's GUI guidelines doc, Part 2 and Part 4.1.
    """

    config: RunConfig
    started_at: str  # ISO-8601, UTC
    finished_at: Optional[str] = None
    wall_time_s: Optional[float] = None
    status: str = "running"  # running | done | failed | cancelled
    return_code: Optional[int] = None
    hostname: str = field(default_factory=socket.gethostname)
    software_versions: Dict[str, str] = field(default_factory=software_versions)
    input_hashes: Dict[str, Optional[str]] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    tables: Dict[str, str] = field(default_factory=dict)   # name -> CSV path
    figures: Dict[str, str] = field(default_factory=dict)  # name -> PDF/PNG path
    log_tail: List[str] = field(default_factory=list)      # last N lines, full log stays on disk
    log_path: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    schema_version: int = SCHEMA_VERSION

    # -- provenance capture -------------------------------------------------

    def capture_input_hashes(self) -> None:
        """Hash every path-shaped value in ``config.inputs`` (Part 4.1)."""
        for key, value in self.config.inputs.items():
            if isinstance(value, str) and value:
                self.input_hashes[key] = sha256_of_file(value)

    def mark_finished(self, status: str, return_code: Optional[int] = None,
                       error: Optional[str] = None) -> None:
        self.finished_at = _now_iso()
        self.status = status
        self.return_code = return_code
        self.error = error
        try:
            started = datetime.fromisoformat(self.started_at)
            finished = datetime.fromisoformat(self.finished_at)
            self.wall_time_s = (finished - started).total_seconds()
        except ValueError:
            self.wall_time_s = None

    # -- serialization --------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunRecord":
        data = dict(data)
        data["config"] = RunConfig.from_dict(data["config"])
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def run_dir(self) -> Path:
        """The timestamped folder this record belongs to (created if needed)."""
        safe_label = "".join(c if c.isalnum() or c in "-_." else "_" for c in self.config.label)
        stamp = self.started_at.replace(":", "-").split(".")[0]
        d = runs_root() / self.config.module / f"{stamp}_{safe_label}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(self, run_dir: Optional[Path] = None) -> Path:
        """Write ``record.json`` into *run_dir* (default: :meth:`run_dir`).

        Returns the path written. Called as soon as a run completes so "the
        last run" survives a restart/crash (guidelines doc, Part 4.1).
        """
        d = run_dir or self.run_dir()
        d.mkdir(parents=True, exist_ok=True)
        path = d / "record.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return path

    @classmethod
    def load(cls, path: str | os.PathLike) -> "RunRecord":
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)

    @classmethod
    def latest(cls, module: Optional[str] = None) -> Optional["RunRecord"]:
        """Return the most recently *started* saved RunRecord, or ``None``.

        Parameters
        ----------
        module : str, optional
            Restrict to one module's runs (e.g. ``"nominal_sim"``). If
            omitted, searches across all modules.
        """
        root = runs_root()
        search_dirs = [root / module] if module else list(root.glob("*"))
        candidates: List[Path] = []
        for d in search_dirs:
            if d.is_dir():
                candidates.extend(d.glob("*/record.json"))
        if not candidates:
            return None
        newest = max(candidates, key=lambda p: p.stat().st_mtime)
        return cls.load(newest)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


def new_record(config: RunConfig) -> RunRecord:
    """Create a fresh, ``running`` RunRecord for *config* with the start time stamped."""
    return RunRecord(config=config, started_at=_now_iso())
