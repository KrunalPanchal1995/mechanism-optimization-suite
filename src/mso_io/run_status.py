"""
mso_io/run_status.py
=====================
Framework-agnostic "what did this run just do" report.

KRUNAL_JULY_23.md point 1: "Add option of exporting the status of the run
from any GUI into a txt file (for eg: which file locations were used, what
was the output location, then options selected and error or output
printed)". No GUI in this repo had anything like it before this module --
the closest analogues were MSO Studio's per-run ``record.json`` (machine
JSON, not a human-readable status export) and the mechanism health-check's
JSON findings export (findings only, not settings/inputs/output-location).

This module is the one place that decides what "run status" means and how
it renders to plain text, so every GUI (11 standalone Tkinter apps + MSO
Studio) produces the same shape of report:

* Tkinter apps build a :class:`RunStatusReport` directly from their own
  current field values + captured log text (see
  ``gui_common.run_status_export.add_export_button`` for the shared
  button/save-dialog wiring every app uses).
* MSO Studio (PySide6) builds one from its existing ``RunRecord``
  (``mso_studio.core.runrecord``) via :meth:`RunStatusReport.from_run_record`
  -- duck-typed, not a hard import, so this module has zero Qt/mso_studio
  dependency and stays importable from the plain-anaconda-python Tkinter
  apps too.

Usage::

    from mso_io.run_status import RunStatusReport

    report = RunStatusReport(
        tool="Mechanism Converter",
        inputs={"mechanism (.inp)": "/path/128-957.inp",
                "thermo (.dat)": "/path/128-957.dat"},
        output_dir="/path/outputs/converted",
        options={"Auto-fix Fortran exponents": True, "--permissive": True},
        status="failed", summary="ck2yaml failed (exit 1)",
        log=captured_log_text,
    )
    report.write_txt("run_status.txt")
"""
from __future__ import annotations

import socket
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


@dataclass
class RunStatusReport:
    """Everything a user would want in a "what just happened" export: which
    files were used, where output went, which options were picked, and the
    full captured output/error text.

    Every field is plain data (str/dict/bool/number) -- no framework types --
    so the same report shape works whether it was built by a Tkinter app's
    own widget values or from an MSO Studio ``RunRecord``.
    """
    tool: str
    generated_at: str = field(default_factory=_now_iso)
    inputs: Dict[str, str] = field(default_factory=dict)
    output_dir: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)
    status: str = "unknown"           # "success" | "failed" | "unknown"
    summary: Optional[str] = None     # one-line outcome, e.g. "2 error(s)"
    log: str = ""                     # full captured stdout/stderr/log text

    def to_text(self) -> str:
        title = f"{self.tool} — run status"
        lines = [
            title,
            "=" * len(title),
            f"Generated : {self.generated_at}",
            f"Host      : {socket.gethostname()}",
            f"Status    : {self.status}" + (f" — {self.summary}" if self.summary else ""),
            "",
        ]
        if self.inputs:
            lines.append("Inputs / file locations")
            lines.append("-" * 24)
            width = max((len(k) for k in self.inputs), default=0)
            for k, v in self.inputs.items():
                lines.append(f"  {k.ljust(width)} : {v}")
            lines.append("")
        if self.output_dir:
            lines += ["Output location", "-" * 16, f"  {self.output_dir}", ""]
        if self.options:
            lines.append("Options selected")
            lines.append("-" * 17)
            width = max((len(k) for k in self.options), default=0)
            for k, v in self.options.items():
                lines.append(f"  {k.ljust(width)} : {v}")
            lines.append("")
        if self.log:
            lines += ["Output / errors", "-" * 16, self.log.rstrip(), ""]
        return "\n".join(lines)

    def write_txt(self, path: str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_text())
        return p

    @classmethod
    def from_run_record(cls, record: Any) -> "RunStatusReport":
        """Build a RunStatusReport from an ``mso_studio.core.runrecord.RunRecord``.

        Duck-typed (accesses ``record.config``/``.status``/etc. directly)
        rather than importing ``mso_studio`` -- this module has to stay
        importable from the plain-anaconda-python Tkinter apps, which never
        have PySide6/mso_studio on their path (see feedback_env_isolation:
        the two GUI stacks use separate interpreters).
        """
        cfg = record.config
        options = dict(cfg.parameters)
        if getattr(record, "wall_time_s", None) is not None:
            options["wall_time_s"] = round(record.wall_time_s, 2)
        summary = record.error if record.status == "failed" and record.error else None
        return cls(
            tool=cfg.module,
            generated_at=record.finished_at or record.started_at,
            inputs=dict(cfg.inputs),
            output_dir=cfg.working_dir or None,
            options=options,
            status=record.status,
            summary=summary,
            log="\n".join(getattr(record, "log_tail", None) or []),
        )
