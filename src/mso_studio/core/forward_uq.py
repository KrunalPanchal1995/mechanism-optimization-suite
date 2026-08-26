"""
mso_studio.core.forward_uq
=============================
Adapter over ``forward_uq.py`` (PRS-surrogate Monte-Carlo/Sobol/LHS
uncertainty propagation). Unlike the converter/health-check/fusion/
uncertainty-generator ports, no new headless CLI mode was needed --
``forward_uq.py`` already has a full ``argparse`` contract
(``forward_uq.py target.opt --n_samples ... --method ... --seed ...
--output_dir ...``), the same one ``forward_uq_gui.py``'s ``_run_uq()``
already shells out to via the Tk PtyTerminal.

Shells out via ``science_command``/``run_backend`` since ``forward_uq.py``
needs the MUQ-SAC/Cantera-adjacent PRS stack -- not installed in the
PySide6-only GUI venv (see ``backends.py``'s docstring).
"""
from __future__ import annotations

import glob
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from mso_studio.core.backends import science_command, run_backend, REPO_ROOT
from mso_studio.core.runrecord import RunConfig, RunRecord, new_record

_SCRIPT = REPO_ROOT / "forward_uq.py"


@dataclass
class ForwardUqParams:
    """User-facing inputs for one forward-UQ propagation run.

    ``target_opt`` must be an existing target.opt-style YAML with
    ``Locations.mechanism``/``Locations.uncertainty_data`` already set, and
    ``src/MUQ-SAC/run_script.py``'s cached PRS coefficients must already
    exist for that mechanism (see ``forward_uq.py``'s own docstring) --
    this adapter does not build either of those, matching
    ``forward_uq_gui.py``'s own scope (it only loads an existing config).

    IMPORTANT -- ``output_dir`` must be the SAME directory the ``Opt/``
    PRS-coefficient cache already lives in, not a fresh empty directory:
    ``forward_uq.py`` resolves ``Opt/`` as ``os.path.join(os.getcwd(),
    "Opt")`` (its own ``workdir`` concept), and this adapter launches with
    ``cwd=output_dir`` -- exactly matching ``forward_uq_gui.py``'s own
    ``_run_uq()`` (``self._terminal.run(cmd, cwd=str(out_dir_abs))``), so
    the constraint is inherited from the original Tk app, not introduced
    here. forward_uq's own results (``forward_uq_statistics.csv``, a
    ``results/`` subfolder of plots) get added alongside the existing
    ``Opt/`` cache in that same directory.
    """

    target_opt: str
    n_samples: int = 50_000
    method: str = "sobol"
    seed: int = 42
    output_dir: str = ""
    label: str = ""
    extra_overrides: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.output_dir:
            self.output_dir = str(REPO_ROOT / "outputs" / "forward_uq")
        if not self.label:
            self.label = Path(self.target_opt).stem


def run_forward_uq(
    params: ForwardUqParams,
    *,
    on_line: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> RunRecord:
    """Run one forward-UQ propagation and return its :class:`RunRecord`.

    Unlike the converter/check/fusion adapters, a non-zero exit here IS
    treated as an infrastructure-style failure (missing PRS cache, bad
    config, ...) since there's no meaningful "partial success" concept for
    a single Monte-Carlo propagation run -- it either produces
    ``forward_uq_statistics.csv`` or it doesn't.
    """
    out_dir = Path(params.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    target_opt_abs = str(Path(params.target_opt).resolve())

    command = science_command(
        str(_SCRIPT), target_opt_abs,
        "--n_samples", str(params.n_samples), "--method", params.method,
        "--seed", str(params.seed), "--output_dir", str(out_dir))

    run_config = RunConfig(
        module="forward_uq", label=params.label,
        backend_command=command, working_dir=str(out_dir),
        inputs={"target.opt": params.target_opt},
        parameters={"n_samples": params.n_samples, "method": params.method,
                   "seed": params.seed, "output_dir": str(out_dir)},
        extra=dict(params.extra_overrides),
    )
    record = new_record(run_config)
    record.capture_input_hashes()

    log_path = out_dir / f"mso_studio_{params.label}_forward_uq.log"
    result = run_backend(
        command, cwd=str(out_dir), on_line=on_line, cancel_event=cancel_event,
        log_path=log_path, timeout=3600.0,
    )
    record.log_path = str(log_path)
    record.log_tail = result.tail[-200:]

    if result.status == "cancelled":
        record.mark_finished("cancelled", result.return_code)
        record.save()
        return record

    if result.status != "done":
        from mso_studio.core.backends import BackendError
        err = BackendError(result.command, result.return_code, result.log_path, result.tail)
        record.mark_finished("failed", result.return_code, error=str(err))
        record.save()
        raise err

    _collect_results(record, out_dir)
    record.mark_finished("done", result.return_code)
    record.save()
    return record


def _collect_results(record: RunRecord, out_dir: Path) -> None:
    hits = sorted(
        glob.glob(str(out_dir / "**" / "forward_uq_statistics.csv"), recursive=True),
        key=os.path.getmtime, reverse=True,
    )
    if hits:
        record.tables["forward_uq_statistics"] = hits[0]
        record.results["statistics_path"] = hits[0]
    record.results["output_dir"] = str(out_dir)
