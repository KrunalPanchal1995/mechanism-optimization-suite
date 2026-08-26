"""
mso_studio.core.rmip_suite
=============================
Adapters over the 6 tools ``rmip_suite_gui.py`` groups under one umbrella
(Graph Comparison, Mechanism Reduction, Global Sensitivity Analysis, Active
Parameter Selection, Target Builder, ReSpecTh Import). Every one of these
already had a full headless CLI before this port (5 via ``rmip_cli.py``
subcommands, 1 via ``src/ActiveParameterScreen.py`` directly) -- unlike the
converter/fusion/uncertainty-generator ports, no new CLI mode was needed
anywhere in this module.

Shells out via ``science_command``/``run_backend`` since these all need
Cantera/rmip/RDKit -- not installed in the PySide6-only GUI venv (see
``backends.py``'s docstring).
"""
from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from mso_studio.core.backends import (
    science_command, run_backend, sanitized_subprocess_env, REPO_ROOT,
)
from mso_studio.core.runrecord import RunConfig, RunRecord, new_record

_RMIP_CLI = REPO_ROOT / "rmip_cli.py"
_ACTIVE_PARAM_SCRIPT = REPO_ROOT / "src" / "ActiveParameterScreen.py"


def _run(module: str, label: str, command: List[str], working_dir: str,
        inputs: dict, parameters: dict, *,
        on_line: Optional[Callable[[str], None]],
        cancel_event: Optional[threading.Event]) -> RunRecord:
    """Shared run/record/log plumbing every one of this module's 6 adapters
    uses identically -- does NOT raise on a non-zero exit (each tool's own
    stdout/stderr is the descriptive error; the log tail carries it)."""
    Path(working_dir).mkdir(parents=True, exist_ok=True)
    run_config = RunConfig(
        module=module, label=label, backend_command=command,
        working_dir=working_dir, inputs=inputs, parameters=parameters,
    )
    record = new_record(run_config)
    record.capture_input_hashes()

    log_path = Path(working_dir) / f"mso_studio_{label}_{module}.log"
    result = run_backend(command, cwd=working_dir, on_line=on_line,
                         cancel_event=cancel_event, log_path=log_path)
    record.log_path = str(log_path)
    record.log_tail = result.tail[-200:]

    if result.status == "cancelled":
        record.mark_finished("cancelled", result.return_code)
    elif result.status != "done":
        error_lines = [ln for ln in result.tail if ln.strip()]
        error = error_lines[-1] if error_lines else \
            f"{module} failed (exit {result.return_code}) -- see log"
        record.mark_finished("failed", result.return_code, error=error)
    else:
        record.mark_finished("done", result.return_code)
    record.save()
    return record


# ===========================================================================
# 1. Graph Comparison  (rmip compare-graph)
# ===========================================================================
@dataclass
class GraphCompareParams:
    db_path: str
    mechanism_ids: List[int]
    output_dir: str = ""
    tol_A: float = 0.01
    tol_n: float = 0.001
    tol_Ea: float = 100.0
    label: str = "graph_compare"

    def __post_init__(self) -> None:
        if not self.output_dir:
            self.output_dir = str(REPO_ROOT / "outputs" / "graph_cmp")


def run_graph_compare(params: GraphCompareParams, *, on_line=None, cancel_event=None) -> RunRecord:
    command = science_command(
        str(_RMIP_CLI), "compare-graph", "--mechs",
        *[str(m) for m in params.mechanism_ids], "--db", params.db_path,
        "--out", params.output_dir, "--tol-A", str(params.tol_A),
        "--tol-n", str(params.tol_n), "--tol-Ea", str(params.tol_Ea))
    record = _run("graph_compare", params.label, command, params.output_dir,
                  {"RMIP database": params.db_path,
                   "mechanism IDs": ", ".join(str(m) for m in params.mechanism_ids)},
                  {"tol_A": params.tol_A, "tol_n": params.tol_n, "tol_Ea": params.tol_Ea},
                  on_line=on_line, cancel_event=cancel_event)
    if record.status == "done":
        cls_path = Path(params.output_dir) / "graph_classification.json"
        spec_path = Path(params.output_dir) / "union_graph_spec.json"
        if cls_path.exists():
            record.results["classification_json"] = str(cls_path)
        if spec_path.exists():
            record.results["union_graph_spec_json"] = str(spec_path)
    return record


def serve_graph_compare(
    params: GraphCompareParams, *, port: int = 8050, label_mode: str = "formula",
) -> subprocess.Popen:
    """Launch ``rmip compare-graph --serve`` as a DETACHED subprocess and
    return the handle immediately -- does NOT wait for it to exit.

    KRUNAL_JULY_23.md point 7's PySide6 link-surfacing (Phase 5c, 2026-08
    backlog program, bug.md): a Dash dev server never exits on its own, so
    routing this through ``run_backend`` (used by every other adapter in
    this module) would block forever until ``DEFAULT_TIMEOUT_S`` kills it --
    ``run_backend`` iterates the child's stdout until EOF, which only
    happens at process exit. The caller owns the returned ``Popen`` and is
    responsible for its lifecycle (store it, terminate it when the tab/app
    closes or before starting a replacement server on the same tab).
    """
    command = science_command(
        str(_RMIP_CLI), "compare-graph", "--mechs",
        *[str(m) for m in params.mechanism_ids], "--db", params.db_path,
        "--out", params.output_dir, "--tol-A", str(params.tol_A),
        "--tol-n", str(params.tol_n), "--tol-Ea", str(params.tol_Ea),
        "--serve", "--port", str(port), "--label", label_mode,
    )
    Path(params.output_dir).mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        command, cwd=params.output_dir, env=sanitized_subprocess_env(),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


# ===========================================================================
# 2. Mechanism Reduction  (rmip reduce)
# ===========================================================================
@dataclass
class ReduceParams:
    db_path: str
    mechanism_id: str
    method: str = "DRG"
    target_species: str = ""
    eps_max: float = 0.5
    eps_steps: int = 51
    out_json: str = ""
    mech_path: str = ""
    conditions: List[Tuple[float, float, float, str]] = field(default_factory=list)
    output_dir: str = ""
    label: str = ""

    def __post_init__(self) -> None:
        if not self.output_dir:
            self.output_dir = str(REPO_ROOT / "outputs" / "reduce")
        if not self.label:
            self.label = f"reduce_{self.mechanism_id}"


def run_reduce(params: ReduceParams, *, on_line=None, cancel_event=None) -> RunRecord:
    if not params.target_species:
        raise ValueError("target_species is required.")
    command = science_command(
        str(_RMIP_CLI), "reduce", params.mechanism_id, "--db", params.db_path,
        "--method", params.method, "--target", params.target_species,
        "--eps-max", str(params.eps_max), "--eps-steps", str(params.eps_steps))
    if params.out_json:
        command += ["--out", params.out_json]
    if params.mech_path:
        command += ["--mech-path", params.mech_path]
    for T_K, P_atm, phi, fuel in params.conditions:
        command += ["--condition", f"{T_K},{P_atm},{phi},{fuel}"]
    return _run("reduce", params.label, command, params.output_dir,
               {"RMIP database": params.db_path, "mechanism ID": params.mechanism_id,
                "mechanism file": params.mech_path},
               {"method": params.method, "target": params.target_species,
                "eps_max": params.eps_max, "eps_steps": params.eps_steps,
                "n_conditions": len(params.conditions)},
               on_line=on_line, cancel_event=cancel_event)


# ===========================================================================
# 3. Global Sensitivity Analysis  (rmip gsa-run)
# ===========================================================================
@dataclass
class GsaParams:
    db_path: str
    mechanism_id: str
    fuel: str
    method: str = "sobol"
    n_samples: int = 256
    reactions: str = "all"
    zerork_install_prefix: str = ""
    T_K: float = 1000.0
    P_atm: float = 1.0
    phi: float = 1.0
    oxidizer: str = "O2:1.0, N2:3.76"
    output_dir: str = ""
    label: str = ""

    def __post_init__(self) -> None:
        if not self.output_dir:
            self.output_dir = str(REPO_ROOT / "outputs" / "gsa")
        if not self.label:
            self.label = f"gsa_{self.mechanism_id}"


def run_gsa(params: GsaParams, *, on_line=None, cancel_event=None) -> RunRecord:
    if not params.fuel:
        raise ValueError("fuel is required.")
    command = science_command(
        str(_RMIP_CLI), "gsa-run", params.mechanism_id, "--db", params.db_path,
        "--method", params.method, "--n-samples", str(params.n_samples),
        "--reactions", params.reactions,
        "--zerork-install-prefix", params.zerork_install_prefix,
        "--T", str(params.T_K), "--P", str(params.P_atm), "--phi", str(params.phi),
        "--fuel", params.fuel, "--oxidizer", params.oxidizer)
    return _run("gsa", params.label, command, params.output_dir,
               {"RMIP database": params.db_path, "mechanism ID": params.mechanism_id,
                "Zero-RK install prefix": params.zerork_install_prefix},
               {"method": params.method, "n_samples": params.n_samples,
                "reactions": params.reactions, "T_K": params.T_K,
                "P_atm": params.P_atm, "phi": params.phi, "fuel": params.fuel},
               on_line=on_line, cancel_event=cancel_event)


# ===========================================================================
# 4. Active Parameter Selection  (src/ActiveParameterScreen.py)
# ===========================================================================
@dataclass
class ActiveParamSelectionParams:
    sens_dir: str
    mechanism: str
    threshold: float = 0.05
    carbon: int = 0
    fuel: str = "Fuel"
    results_dir: str = ""
    classification_file: str = ""
    generate_xml: bool = False
    generate_plots: bool = False
    generate_latex: bool = False
    filtered_mechanism: bool = False
    label: str = "active_param_selection"

    def __post_init__(self) -> None:
        if not self.results_dir:
            self.results_dir = str(REPO_ROOT / "outputs" / "active_param_selection" / "results")


def run_active_param_selection(
    params: ActiveParamSelectionParams, *, on_line=None, cancel_event=None,
) -> RunRecord:
    needs_class = params.generate_xml or params.generate_plots or params.generate_latex
    if needs_class and not params.classification_file:
        raise ValueError("XML/plots/LaTeX require a classification CSV.")
    command = science_command(
        str(_ACTIVE_PARAM_SCRIPT),
        "--sens-dir", params.sens_dir, "--mechanism", params.mechanism,
        "--threshold", str(params.threshold), "--carbon", str(params.carbon),
        "--fuel", params.fuel, "--results-dir", params.results_dir,
    )
    if needs_class:
        command += ["--classification-file", params.classification_file]
    if params.generate_xml:
        command.append("--generate-xml")
    if params.generate_plots:
        command.append("--generate-plots")
    if params.generate_latex:
        command.append("--generate-latex")
    if params.filtered_mechanism:
        command.append("--filtered-mechanism")

    record = _run("active_param_selection", params.label, command, params.results_dir,
                 {"SA results directory": params.sens_dir, "mechanism": params.mechanism,
                  "classification CSV": params.classification_file},
                 {"threshold": params.threshold, "carbon": params.carbon,
                  "fuel": params.fuel, "generate_xml": params.generate_xml,
                  "generate_plots": params.generate_plots,
                  "generate_latex": params.generate_latex},
                 on_line=on_line, cancel_event=cancel_event)
    if record.status == "done":
        results_dir = Path(params.results_dir)
        for csv_name in ("active_reactions_detailed.csv", "master_active_reactions.csv",
                         "active_species.csv", "active_reaction_summary.csv"):
            p = results_dir / csv_name
            if p.exists():
                record.tables[csv_name] = str(p)
    return record


# ===========================================================================
# 5. Target Builder  (rmip build-target)
# ===========================================================================
@dataclass
class TargetBuilderParams:
    db_path: str
    mechanism_id: str
    base_target_opt: str
    fuel_blend: List[Tuple[str, float]]
    target_type: str = "JSR"
    solver: str = "cantera"
    T_K: float = 1000.0
    P_atm: float = 1.0
    phi: float = 1.0
    oxidizer: str = "O2:1.0, N2:3.76"
    work_dir: str = ""
    dataset_id: str = ""
    extra_fields: dict = field(default_factory=dict)   # {flag: value_or_bool}
    label: str = ""

    def __post_init__(self) -> None:
        if not self.work_dir:
            self.work_dir = str(REPO_ROOT / "outputs" / "target_builder")
        if not self.label:
            self.label = f"target_builder_{self.mechanism_id}"


def run_build_target(params: TargetBuilderParams, *, on_line=None, cancel_event=None) -> RunRecord:
    if not params.fuel_blend:
        raise ValueError("Add at least one fuel-blend row (species + mole fraction).")
    fuel = ", ".join(f"{sp}:{frac}" for sp, frac in params.fuel_blend)
    command = [
        *science_command(str(_RMIP_CLI), "build-target", params.mechanism_id,
                         "--db", params.db_path, "--base-target-opt", params.base_target_opt,
                         "--target-type", params.target_type, "--solver", params.solver,
                         "--fuel", fuel, "--oxidizer", params.oxidizer,
                         "--T", str(params.T_K), "--P", str(params.P_atm),
                         "--phi", str(params.phi), "--work-dir", params.work_dir),
    ]
    if params.dataset_id:
        command += ["--dataset-id", params.dataset_id]
    for flag, value in params.extra_fields.items():
        if isinstance(value, bool):
            if value:
                command.append(flag)
        elif str(value).strip():
            command += [flag, str(value).strip()]
    return _run("target_builder", params.label, command, params.work_dir,
               {"RMIP database": params.db_path, "mechanism ID": params.mechanism_id,
                "base target.opt": params.base_target_opt},
               {"target_type": params.target_type, "solver": params.solver,
                "fuel": fuel, "T_K": params.T_K, "P_atm": params.P_atm, "phi": params.phi},
               on_line=on_line, cancel_event=cancel_event)


# ===========================================================================
# 6. ReSpecTh / RKD XML Import  (rmip import-respecth)
# ===========================================================================
@dataclass
class RespecthImportParams:
    db_path: str
    xml_path: str
    fuel: str = "NH3"
    solver: str = "cantera"
    out_dir: str = ""
    dataset_prefix: str = ""
    no_db: bool = False
    label: str = "respecth_import"


def run_import_respecth(params: RespecthImportParams, *, on_line=None, cancel_event=None) -> RunRecord:
    if not params.xml_path:
        raise ValueError("An XML file/directory is required.")
    command = science_command(
        str(_RMIP_CLI), "import-respecth", params.xml_path, "--db", params.db_path,
        "--fuel", params.fuel, "--solver", params.solver)
    if params.out_dir:
        command += ["--out-dir", params.out_dir]
    if params.dataset_prefix:
        command += ["--dataset-prefix", params.dataset_prefix]
    if params.no_db:
        command.append("--no-db")
    working_dir = params.out_dir or str(REPO_ROOT / "outputs" / "respecth_import")
    return _run("respecth_import", params.label, command, working_dir,
               {"RMIP database": params.db_path, "XML path": params.xml_path},
               {"fuel": params.fuel, "solver": params.solver, "no_db": params.no_db},
               on_line=on_line, cancel_event=cancel_event)
