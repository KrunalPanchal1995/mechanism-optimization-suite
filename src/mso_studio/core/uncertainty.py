"""
mso_studio.core.uncertainty
==============================
Adapter over ``uncertainty_generator_gui.py``'s two headless CLI modes:
``--generate-xml`` (build a MUQ-SAC uncertainty XML from reaction rows) and
``--compute-bounds`` (collapse a temperature-dependent uncertainty XML to
one symmetric A-factor bound per reaction, via src/MUQ-SAC/Uncertainty.py).

Shells out via ``science_command``/``run_backend`` since
``compute_symmetric_bounds`` needs src/MUQ-SAC's scipy/numpy stack -- not
installed in the PySide6-only GUI venv (see ``backends.py``'s docstring).
``generate_uncertainty_xml`` itself needs only stdlib, but goes through the
same bridge for one consistent recipe.
"""
from __future__ import annotations

import csv
import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from mso_studio.core.backends import science_command, run_backend, REPO_ROOT
from mso_studio.core.runrecord import RunConfig, RunRecord, new_record

_SCRIPT = REPO_ROOT / "uncertainty_generator_gui.py"
_ACTIVE_PARAM_SCRIPT = REPO_ROOT / "src" / "ActiveParameterScreen.py"


@dataclass
class GenerateXmlParams:
    """One row per reaction: {name, type, temperatures, uncertainties,
    [classification], [perturbation_type], [data_mode]} -- see
    ``uncertainty_generator_gui.generate_uncertainty_xml``'s docstring."""

    reactions: List[Dict[str, Any]]
    output_path: str
    label: str = "generate_xml"


def run_generate_xml(
    params: GenerateXmlParams,
    *,
    on_line: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> RunRecord:
    out_path = Path(params.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reactions_json = out_path.with_suffix(".reactions.json")
    reactions_json.write_text(json.dumps(params.reactions))

    command = science_command(
        str(_SCRIPT), "--generate-xml",
        "--reactions-json", str(reactions_json), "--output", str(out_path))

    run_config = RunConfig(
        module="uncertainty_generate_xml", label=params.label,
        backend_command=command, working_dir=str(out_path.parent),
        parameters={"n_reactions": len(params.reactions)},
    )
    record = new_record(run_config)

    log_path = out_path.parent / f"mso_studio_{params.label}.log"
    result = run_backend(command, cwd=str(out_path.parent), on_line=on_line,
                         cancel_event=cancel_event, log_path=log_path)
    record.log_path = str(log_path)
    record.log_tail = result.tail[-200:]

    if result.status == "cancelled":
        record.mark_finished("cancelled", result.return_code)
        record.save()
        return record

    if result.status != "done":
        record.mark_finished(
            "failed", result.return_code,
            error=f"XML generation failed (exit {result.return_code}) -- see log")
        record.save()
        return record

    record.results["output_path"] = str(out_path)
    record.mark_finished("done", result.return_code)
    record.save()
    return record


@dataclass
class ComputeBoundsParams:
    xml_path: str
    mechanism: str
    output_csv: str = ""
    label: str = ""

    def __post_init__(self) -> None:
        if not self.output_csv:
            self.output_csv = str(REPO_ROOT / "outputs" / "uncertainty_bounds.csv")
        if not self.label:
            self.label = Path(self.xml_path).stem


_OK_LINE = re.compile(
    r"^OK (.+): A_nominal=([\d.eE+-]+) factor=([\d.eE+-]+) "
    r"A_lb=([\d.eE+-]+) A_ub=([\d.eE+-]+)$")
_ERR_LINE = re.compile(r"^ERROR (.+?): (.+)$")


def run_compute_bounds(
    params: ComputeBoundsParams,
    *,
    on_line: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> RunRecord:
    """Does NOT raise when individual reactions fail to resolve (an expected,
    per-reaction outcome the tool itself tolerates) -- ``status="failed"``
    is reserved for the CLI itself not completing."""
    out_csv = Path(params.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    command = science_command(
        str(_SCRIPT), "--compute-bounds", "--xml", params.xml_path,
        "--mechanism", params.mechanism, "--output-csv", str(out_csv))

    run_config = RunConfig(
        module="uncertainty_compute_bounds", label=params.label,
        backend_command=command, working_dir=str(out_csv.parent),
        inputs={"uncertainty XML": params.xml_path, "mechanism": params.mechanism},
    )
    record = new_record(run_config)
    record.capture_input_hashes()

    log_path = out_csv.parent / f"mso_studio_{params.label}_bounds.log"
    result = run_backend(command, cwd=str(out_csv.parent), on_line=on_line,
                         cancel_event=cancel_event, log_path=log_path)
    record.log_path = str(log_path)
    record.log_tail = result.tail[-200:]

    if result.status == "cancelled":
        record.mark_finished("cancelled", result.return_code)
        record.save()
        return record

    rows = []
    for line in result.tail:
        m = _OK_LINE.match(line)
        if m:
            rows.append({"rxn": m.group(1), "A_nominal": float(m.group(2)),
                        "factor": float(m.group(3)), "A_lb": float(m.group(4)),
                        "A_ub": float(m.group(5))})
            continue
        m = _ERR_LINE.match(line)
        if m:
            rows.append({"rxn": m.group(1), "error": m.group(2)})

    if not out_csv.exists() and not rows:
        # neither the CSV nor any parseable per-reaction line -- the CLI
        # itself didn't run to completion (bad argv, import error, ...)
        record.mark_finished(
            "failed", result.return_code,
            error=f"compute-bounds produced no output (exit {result.return_code}) -- see log")
        record.save()
        return record

    n_ok = sum(1 for r in rows if "error" not in r)
    n_err = len(rows) - n_ok
    record.results["rows"] = rows
    record.results["n_ok"] = n_ok
    record.results["n_err"] = n_err
    if out_csv.exists():
        record.tables["uncertainty_bounds"] = str(out_csv)

    record.mark_finished("done", result.return_code)
    record.save()
    return record


@dataclass
class ActiveParamScreenParams:
    """Every option src/ActiveParameterScreen.py accepts (KRUNAL_JULY_23.md
    point 5's primary "SA data -> screening -> XML/plots/LaTeX/filtered-
    mechanism" workflow -- mirrors uncertainty_generator_gui.py's "From SA
    data" tab and rmip_suite_gui.py's ActiveParamSelectionTab)."""

    sens_dir: str
    mechanism: str
    threshold: float = 0.05
    carbon: int = 0
    fuel: str = "Fuel"
    results_dir: str = "results"
    classification_file: str = ""
    generate_xml: bool = False
    generate_plots: bool = False
    per_case_plots: bool = False
    generate_latex: bool = False
    longtable: bool = False
    filtered_mechanism: bool = False
    compare_temperatures: List[float] = field(default_factory=list)
    compare_colors: List[str] = field(default_factory=list)
    label: str = "active_param_screen"


def run_active_parameter_screen(
    params: ActiveParamScreenParams,
    *,
    on_line: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> RunRecord:
    needs_class = (params.generate_xml or params.generate_plots
                   or params.generate_latex or params.filtered_mechanism)
    if needs_class and not params.classification_file:
        raise ValueError("XML/plots/LaTeX/filtered-mechanism require a classification CSV.")

    args = [
        "--sens-dir", params.sens_dir, "--mechanism", params.mechanism,
        "--threshold", str(params.threshold), "--carbon", str(params.carbon),
        "--fuel", params.fuel, "--results-dir", params.results_dir,
    ]
    if params.classification_file:
        args += ["--classification-file", params.classification_file]
    if params.generate_xml:
        args.append("--generate-xml")
    if params.generate_plots:
        args.append("--generate-plots")
    if params.per_case_plots:
        args.append("--per-case-plots")
    if params.generate_latex:
        args.append("--generate-latex")
    if params.longtable:
        args.append("--longtable")
    if params.filtered_mechanism:
        args.append("--filtered-mechanism")
    if params.compare_temperatures:
        args += ["--compare-temperatures", *[str(t) for t in params.compare_temperatures]]
    if params.compare_colors:
        args += ["--compare-colors", *params.compare_colors]

    command = science_command(str(_ACTIVE_PARAM_SCRIPT), *args)
    out_dir = Path(params.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_config = RunConfig(
        module="active_parameter_screen", label=params.label,
        backend_command=command, working_dir=str(out_dir.resolve()),
        inputs={"sens_dir": params.sens_dir, "mechanism": params.mechanism,
                "classification_file": params.classification_file},
    )
    record = new_record(run_config)
    record.capture_input_hashes()

    log_path = out_dir / f"mso_studio_{params.label}.log"
    result = run_backend(command, cwd=str(out_dir.resolve()), on_line=on_line,
                         cancel_event=cancel_event, log_path=log_path)
    record.log_path = str(log_path)
    record.log_tail = result.tail[-200:]

    if result.status == "cancelled":
        record.mark_finished("cancelled", result.return_code)
        record.save()
        return record
    if result.status != "done":
        record.mark_finished(
            "failed", result.return_code,
            error=f"ActiveParameterScreen failed (exit {result.return_code}) -- see log")
        record.save()
        return record

    if params.generate_xml:
        for stem in ("HTC_all", "HTC_factor", "LTC_all", "LTC_factor"):
            p = out_dir / "xml" / f"{stem}.xml"
            if p.exists():
                record.tables[f"xml_{stem}"] = str(p)
    record.results["results_dir"] = str(out_dir.resolve())
    record.mark_finished("done", result.return_code)
    record.save()
    return record
