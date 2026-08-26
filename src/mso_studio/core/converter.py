"""
mso_studio.core.converter
===========================
Adapter over ``mechanism_converter_gui.py``'s CHEMKIN -> Cantera YAML / .pre /
RMIP-DB conversion (the same fix confirmed in KRUNAL_JULY_23.md point 2:
``convert()`` no longer corrupts species names like KONNOV_2019's
``C2H4O1-2``, and auto-inserts genuinely-undeclared species).

Shells out to ``mechanism_converter_gui.py --convert`` in the anaconda
interpreter rather than importing ``convert()`` directly -- ``convert()``
pulls in ``rmip.io.chemkin_preflight`` (Cantera-adjacent) and calls
``ck2yaml``/``ScanMan``, none of which exist in the PySide6-only GUI venv
(see ``backends.py``'s module docstring for the env-isolation rationale).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from mso_studio.core.backends import science_command, run_backend, REPO_ROOT
from mso_studio.core.runrecord import RunConfig, RunRecord, new_record

_CONVERTER_SCRIPT = REPO_ROOT / "mechanism_converter_gui.py"


@dataclass
class ConverterParams:
    """User-facing inputs for one CHEMKIN -> Cantera/RMIP conversion.

    Parameters mirror ``mechanism_converter_gui.convert()``'s keyword
    arguments 1:1 -- see that function's docstring for what each output/
    flag actually does.
    """

    mechanism: str
    thermo: str = ""
    transport: str = ""
    output_dir: str = ""
    make_yaml: bool = True
    make_pre: bool = False
    make_db: bool = False
    scanman: str = ""
    db_path: str = ""
    glossary: str = ""
    permissive: bool = True
    no_validate: bool = False
    fix_exp: bool = True
    fix_crlf: bool = True
    opensmoke_gas: bool = False
    label: str = ""
    extra_overrides: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.output_dir:
            self.output_dir = str(REPO_ROOT / "outputs" / "converted")
        if not self.label:
            self.label = Path(self.mechanism).stem


def run_convert(
    params: ConverterParams,
    *,
    on_line: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> RunRecord:
    """Run one CHEMKIN conversion and return its :class:`RunRecord`.

    Unlike ``BackendError``-raising adapters (e.g. ``nominal.run_nominal``),
    a non-zero exit here is an expected, common outcome (a real conversion
    failure -- bad species names, missing thermo, etc.) rather than an
    infrastructure error, so this does NOT raise on failure: it always
    returns a RunRecord, with ``status="failed"`` and ``error`` set to the
    tool's own reported error line(s) -- exactly the descriptive detail
    KRUNAL_JULY_23.md asked every GUI to surface, not a generic
    "backend exited non-zero" message.
    """
    out_dir = Path(params.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    args = ["--convert", "--input", params.mechanism, "--out-dir", str(out_dir)]
    if params.thermo:
        args += ["--thermo", params.thermo]
    if params.transport:
        args += ["--transport", params.transport]
    args.append("--make-yaml" if params.make_yaml else "--no-make-yaml")
    if params.make_pre:
        args.append("--make-pre")
    if params.make_db:
        args.append("--make-db")
    if params.scanman:
        args += ["--scanman", params.scanman]
    if params.db_path:
        args += ["--db-path", params.db_path]
    if params.glossary:
        args += ["--glossary", params.glossary]
    if params.permissive:
        args.append("--permissive")
    if params.no_validate:
        args.append("--no-validate")
    args.append("--fix-exp" if params.fix_exp else "--no-fix-exp")
    args.append("--fix-crlf" if params.fix_crlf else "--no-fix-crlf")
    if params.opensmoke_gas:
        args.append("--opensmoke-gas")

    command = science_command(str(_CONVERTER_SCRIPT), *args)

    run_config = RunConfig(
        module="mechanism_converter",
        label=params.label,
        backend_command=command,
        working_dir=str(out_dir),
        inputs={
            "mechanism": params.mechanism,
            "thermo": params.thermo,
            "transport": params.transport,
            "scanman": params.scanman,
            "db_path": params.db_path,
            "glossary": params.glossary,
        },
        parameters={
            "make_yaml": params.make_yaml, "make_pre": params.make_pre,
            "make_db": params.make_db, "permissive": params.permissive,
            "no_validate": params.no_validate, "fix_exp": params.fix_exp,
            "fix_crlf": params.fix_crlf, "opensmoke_gas": params.opensmoke_gas,
        },
        extra=dict(params.extra_overrides),
    )
    record = new_record(run_config)
    record.capture_input_hashes()

    log_path = out_dir / f"mso_studio_{params.label}_convert.log"
    result = run_backend(
        command, cwd=str(out_dir), on_line=on_line, cancel_event=cancel_event,
        log_path=log_path,
    )
    record.log_path = str(log_path)
    record.log_tail = result.tail[-200:]

    if result.status == "cancelled":
        record.mark_finished("cancelled", result.return_code)
        record.save()
        return record

    if result.status != "done":
        error_lines = [ln for ln in result.tail if ln.startswith("ERROR:")]
        error = error_lines[-1][len("ERROR: "):] if error_lines else \
            f"conversion failed (exit {result.return_code}) -- see log"
        record.mark_finished("failed", result.return_code, error=error)
        record.save()
        return record

    _collect_results(record, out_dir, params)
    record.mark_finished("done", result.return_code)
    record.save()
    return record


def _collect_results(record: RunRecord, out_dir: Path, params: ConverterParams) -> None:
    stem = Path(params.mechanism).stem
    for suffix, key in ((".yaml", "cantera_yaml"), (".pre", "flamemaster_pre")):
        candidate = out_dir / f"{stem}{suffix}"
        if candidate.exists():
            record.results[key] = str(candidate)
    record.results["output_dir"] = str(out_dir)
