"""
sim_runner.py
-------------
Rewritten nominal simulation runner for the mso_compare framework.

Replaces the original ``run_nominal_sim.py`` with the following improvements:

Bug fixes
~~~~~~~~~
1. **Progress file location** — the progress/locations files now always land
   inside ``NOMINAL/``, not in the parent directory.  This is achieved by
   saving ``nominalDir`` as an absolute :class:`~pathlib.Path` and using it
   for all file-existence checks before and after calling ``SM()``.
2. **Plot directory creation** — ``Plot/Dataset/{Tig,RCM,JSR,Fls}/`` are
   created explicitly with ``Path.mkdir(parents=True, exist_ok=True)``
   before any CSV write attempt.
3. **YAML import conflict** — ``ruamel_yaml`` / ``PyYAML`` dual-import
   resolved; only ``PyYAML`` (``yaml.safe_load``) is used here.
4. **Directory changes** — all ``os.chdir()`` calls are wrapped in
   try/finally blocks so the original working directory is always restored,
   even if an exception occurs.

New features
~~~~~~~~~~~~
* Importable as a module function: ``run_nominal_simulation(cfg, ...)``
* Accepts a :class:`~mso_compare.config.SimRunConfig` + base ``.opt`` path;
  builds the merged ``optInputs`` dict internally.
* Optional ``dry_run=True`` flag prints the plan without executing.
* Structured logging to both console and a per-run ``.log`` file.
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from .config import SimRunConfig
from .utils  import (
    build_run_config,
    load_opt_file,
    setup_logging,
    setup_muqsac_path,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Target/addendum preview (Phase 5b, 2026-08 backlog program, bug.md)
# ---------------------------------------------------------------------------

def preview_targets_for_run(sim_run_cfg: SimRunConfig, base_opt_file: str) -> list:
    """Preview the targets a nominal-sim run will actually simulate.

    Resolves ``Locations.targets``/``Locations.addendum`` the exact same
    way :func:`run_nominal_simulation` does -- via :func:`build_run_config`
    layering *sim_run_cfg* over *base_opt_file* -- so this reflects any
    per-run ``addendum`` override (``SimRunConfig.addendum``), not just
    whatever the base template happens to specify. Read-only, does not
    launch anything.

    Closes the gap KRUNAL_JULY_23.md point 4's "same for nominal
    simulations" follow-up flagged: nominal_sim_gui.py wraps this
    mso_compare pipeline (not MUQ-SAC's target.opt/.target/.add system
    directly), and had no way to preview which targets/solver a run would
    actually use before launching it.

    Returns a list of :class:`mso_io.target_add_io.TargetPreview`.
    """
    from mso_io.target_add_io import preview_targets

    base_opt  = load_opt_file(base_opt_file)
    optInputs = build_run_config(base_opt, sim_run_cfg)
    locations = optInputs["Locations"]

    targets_path  = locations["targets"]
    addendum_path = locations.get("addendum") or None
    return preview_targets(targets_path, addendum_path)


# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

def _create_sim_directories(output_dir: Path) -> Dict[str, Path]:
    """
    Create the full directory tree for one simulation run.

    Layout::

        output_dir/
        ├── NOMINAL/
        │   └── Data/
        │       └── Simulations/
        └── Plot/
            └── Dataset/
                ├── Tig/
                ├── RCM/
                ├── JSR/
                └── Fls/

    Returns
    -------
    dict
        Named :class:`~pathlib.Path` objects for frequently-accessed dirs.
    """
    dirs: Dict[str, Path] = {
        "nominal":  output_dir / "NOMINAL",
        "data":     output_dir / "NOMINAL" / "Data",
        "sims":     output_dir / "NOMINAL" / "Data" / "Simulations",
        "plot":     output_dir / "Plot",
        "dataset":  output_dir / "Plot" / "Dataset",
        "tig":      output_dir / "Plot" / "Dataset" / "Tig",
        "rcm":      output_dir / "Plot" / "Dataset" / "RCM",
        "jsr":      output_dir / "Plot" / "Dataset" / "JSR",
        "fls":      output_dir / "Plot" / "Dataset" / "Fls",
        "pfr":      output_dir / "Plot" / "Dataset" / "PFR",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    logger.info("Directory structure ready under: %s", output_dir)
    return dirs


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
from typing import Optional

def _locate_progress_files(nominal_dir: Path, output_dir: Path):
    """
    Search only this run's own directories for an existing progress/locations
    pair. original_cwd is deliberately excluded — it's shared across every
    run label, so searching it (especially recursively) can pick up a
    sibling run's leftover files instead of this run's own.
    """
    candidates = (nominal_dir, output_dir)

    def _find(filename: str) -> Optional[Path]:
        for cand in candidates:
            p = cand / filename
            if p.exists():
                return p
        # fallback: recurse, but never leave this run's own output_dir
        return next(output_dir.rglob(filename), None)

    return _find("progress"), _find("locations")


def _sweep_missing_simulations(
    data_management,
    exe_locations: List[str],
    target_list: List[Any],
    fuel: Any,
    optInputs: Dict[str, Any],
) -> None:
    """
    Detect every simulation across ALL cases that failed to produce output,
    then re-run all of them in a single multiprocessing pool.

    ``data_management.generate_target_value_tables`` (called once per
    case, right after this) already does exactly this detect-then-rerun
    dance, but only within its own case -- so a compare run with e.g. 68
    cases spins up 68 separate multiprocessing pools, most of them idle
    apart from the handful of cases that actually had a failure. Sweeping
    across cases first collapses that into one pool, sized off this study's
    own ``Counts.parallel_threads`` (never hardcoded -- the per-case
    ``Worker(100)`` at the call site below ignores the configured thread
    budget; that is a separate, pre-existing issue in the shared module).
    The per-case loop that follows becomes a fast no-op re-check once this
    sweep has filled in the missing output.
    """
    missing: List[tuple] = []  # (case_idx, target, file_path)
    for case_idx, t in enumerate(target_list):
        case_tag = f"case-{case_idx}"
        case_locs = [
            loc.strip("\n") for loc in exe_locations
            if case_tag in loc.strip("\n").split("/")
        ]
        for i, loc in enumerate(case_locs):
            eta_, _ETA, file_path = data_management.extract_output(
                t, fuel, loc[:-3] + "output/", i,
                input_=optInputs, caseID=case_idx,
            )
            if "N/A" in str(eta_):
                missing.append((case_idx, t, file_path))

    if not missing:
        return

    workers = int(
        optInputs.get("Counts", {}).get("rerun_workers")
        or optInputs["Counts"]["parallel_threads"]
    )
    n_cases = len({c for c, _, _ in missing})
    logger.info(
        "Cross-case sweep: %d simulation(s) across %d case(s) did not "
        "return expected results — re-running in one pool (%d workers) …",
        len(missing), n_cases, workers,
    )

    W = data_management.Worker(workers)
    for idx, (case_idx, t, file_path) in enumerate(missing):
        W.pool.apply_async(
            data_management.run_sim,
            args=(idx, t, optInputs, case_idx, file_path, len(missing)),
            callback=W.callback_run,
            error_callback=W.callback_error,
        )
    W.pool.close()
    W.pool.join()
    W.pool.terminate()
    del W

    logger.info("Cross-case sweep complete — %d simulation(s) re-run.", len(missing))


def run_nominal_simulation(
    sim_run_cfg: SimRunConfig,
    base_opt_file: str,
    muqsac_dir:  str  = "",
    dry_run:     bool = False,
    raw: dict = {},
) -> Dict[str, Any]:
    """
    Run one nominal simulation.

    Parameters
    ----------
    sim_run_cfg : SimRunConfig
        Configuration for this run (mechanism, targets, file paths, etc.).
    base_opt_file : str
        Path to the base ``.opt`` YAML template.
    muqsac_dir : str
        Path to the ``src/MUQ-SAC`` directory.  Added to ``sys.path``
        before MUQ-SAC modules are imported.
    dry_run : bool
        If ``True``, print a summary of what would run but do not execute.

    Returns
    -------
    dict
        ``{"label", "output_dir", "plot_dir", "status"}``

    Raises
    ------
    ImportError
        If MUQ-SAC modules cannot be imported.
    FileNotFoundError
        If required input files are missing (raised by ``sim_run_cfg.validate()``).
    """
    label      = sim_run_cfg.label
    output_dir = Path(sim_run_cfg.output_dir).resolve()

    # ── Per-run log file ───────────────────────────────────────────────────
    log_path = output_dir / f"mso_{label}.log"
    output_dir.mkdir(parents=True, exist_ok=True)   # ensure dir exists for log
    setup_logging(f"sim_runner.{label}", log_file=str(log_path))

    logger.info("=" * 62)
    logger.info("  Nominal simulation: %s", label)
    logger.info("  Mechanism    : %s", sim_run_cfg.mechanism)
    logger.info("  Targets      : %s  (%d)", sim_run_cfg.targets, sim_run_cfg.targets_count)
    logger.info("  Output dir   : %s", output_dir)
    logger.info("=" * 62)

    if dry_run:
        logger.info("[DRY RUN] No files written.")
        return {
            "label":      label,
            "output_dir": str(output_dir),
            "plot_dir":   str(output_dir / "Plot" / "Dataset"),
            "status":     "dry_run",
        }

    # ── Validate inputs ────────────────────────────────────────────────────
    sim_run_cfg.validate()

    # ── Set up MUQ-SAC sys.path ────────────────────────────────────────────
    if muqsac_dir:
        setup_muqsac_path(muqsac_dir)

    # ── Import MUQ-SAC modules (deferred so sys.path is ready) ────────────
    try:
        import combustion_target_class       # noqa: PLC0415
        import data_management               # noqa: PLC0415  (data_management_nominal does not exist)
        import simulation_manager2_0 as simulator   # noqa: PLC0415
    except ImportError as exc:
        logger.error(
            "Failed to import MUQ-SAC modules: %s\n"
            "Ensure 'muqsac_src_dir' points to src/MUQ-SAC in your config.",
            exc,
        )
        raise

    # ── Load and override .opt config ──────────────────────────────────────
    base_opt   = load_opt_file(base_opt_file)
    optInputs  = build_run_config(base_opt, sim_run_cfg)

    # ── Create directory structure ─────────────────────────────────────────
    dirs        = _create_sim_directories(output_dir)
    nominal_dir = dirs["nominal"]   # absolute Path
    #print(nominal_dir)
    #print(output_dir)
    # ── Read target file ───────────────────────────────────────────────────
    locations     = optInputs["Locations"]
    addendum_path = locations.get("addendum", "")

    addendum: dict = {}
    if addendum_path and Path(addendum_path).exists():
        with open(addendum_path, "r") as fh:
            addendum = yaml.safe_load(fh) or {}
    elif addendum_path:
        logger.warning("Addendum file not found: %s  (continuing without it)", addendum_path)

    targets_count = int(optInputs["Counts"]["targets_count"])
    target_lines  = Path(locations["targets"]).read_text().splitlines()

    target_list:  List[Any] = []
    string_target = ""
    for c_index, raw_line in enumerate(target_lines[:targets_count]):
        # Strip inline comments
        line = raw_line.split("#")[0].strip()
        if not line:
            continue
        add = deepcopy(addendum)
        t   = combustion_target_class.combustion_target(line, add, c_index)
        string_target += (
            f"{t.dataSet_id}|{t.target}|{t.species_dict}|"
            f"{t.temperature}|{t.pressure}|{t.phi}|"
            f"{t.observed}|{t.std_dvtn}\n"
        )
        target_list.append(t)

    logger.info("Loaded %d targets from: %s", len(target_list), locations["targets"])

    # Write target summary (in output_dir, not the old home_dir)
    (output_dir / "target_data.txt").write_text(string_target)

    # ── Run simulations ────────────────────────────────────────────────────
    # Critical fix: change into nominal_dir BEFORE calling SM() so that the
    # progress and locations files are written inside NOMINAL/, not outside.
    progress_file  = nominal_dir / "progress"
    locations_file = nominal_dir / "locations"

    
    
    original_cwd = Path.cwd()
    progress_path, locations_path = _locate_progress_files(nominal_dir, output_dir)

    os.chdir(nominal_dir)
    
    try:
        if progress_path is None or locations_path is None:
           logger.info("Progress/locations pair incomplete (progress=%s, locations=%s) — "
                       "launching simulations …", progress_path, locations_path)
           exe_locations: List[str] = (
               simulator.SM(target_list, optInputs, {}, [])
               .make_nominal_dir_in_parallel()
           )
        else:
           logger.info("Reusing progress=%s, locations=%s", progress_path, locations_path)
           exe_locations = locations_path.read_text().splitlines()
    finally:
        os.chdir(original_cwd)
    

    # ── Extract simulation results ─────────────────────────────────────────
    print(nominal_dir)
    logger.info("Extracting simulation results …")
    fuel            = optInputs["Inputs"].get("fuel", {})

    # ── Cross-case re-run sweep ─────────────────────────────────────────────
    # Catch every case's missing simulations up front, in one pool, before
    # the per-case extraction loop below does its own (much smaller) retry.
    _sweep_missing_simulations(data_management, exe_locations, target_list, fuel, optInputs)

    temp_sim_opt:   Dict[str, List[float]] = {}

    for case_idx in range(len(target_list)):
        t = target_list[case_idx]                       # NEW
        cached_file = dirs["sims"] / f"sim_data_case-{case_idx}.lst"

        if cached_file.exists():
            # Same None-guard as the fresh-compute branch below: a case that
            # failed after its retry can have a literal "None" second field
            # in its cached row, which float() rejects (ValueError) and used
            # to abort extraction for the whole run, discarding every other
            # case's already-cached result.
            rows = [r for r in cached_file.read_text().splitlines() if r.strip()]
            eta  = [np.exp(float(r.split("\t")[1])) / 10.0 for r in rows
                    if r.split("\t")[1].strip() not in ("None", "")]
            temp_sim_opt[str(t.index)] = eta             # was case_idx
        else:
            case_dir_path = nominal_dir / f"case-{case_idx}"
            os.chdir(case_dir_path)
            try:
                data_sheet, failed_sim, eta_arr = (
                   data_management.generate_target_value_tables(
                       exe_locations, target_list, case_idx, fuel,
                   input_= optInputs)
                )
                # H2/O2 extreme-condition sweep fix: generate_target_value_tables()
                # can return eta_arr=None, or a list containing None entries,
                # for a case where some/all rows failed (a legitimate outcome
                # when deliberately sweeping to physical/numerical extremes,
                # not just a transient error) -- np.exp() on either raised
                # TypeError and aborted extraction for the WHOLE run,
                # discarding every other case's already-computed results.
                if eta_arr is None:
                    eta_arr = []
                else:
                    eta_arr = [v for v in eta_arr if v is not None]
                temp_sim_opt[str(t.index)] = list(np.exp(eta_arr) / 10.0)  # was case_idx
            finally:
                os.chdir(original_cwd)

            (dirs["sims"] / f"sim_data_case-{case_idx}.lst").write_text(data_sheet)
            (dirs["sims"] / f"failed_sim_data_case-{case_idx}.lst").write_text(failed_sim)

            if not temp_sim_opt[str(t.index)]:
                logger.warning("Case %d (target index %s) produced no result — "
                               "check failed_sim_data_case-%d.lst", case_idx, t.index, case_idx)

    # ── Write per-dataset CSV files ────────────────────────────────────────
    logger.info("Writing dataset CSV files …")
    dataset_ids = set(t.dataSet_id for t in target_list)
    for d_set in sorted(dataset_ids):
        _write_dataset_csv(d_set, target_list, temp_sim_opt, dirs)

    logger.info("✓  Nominal simulation complete: %s", label)
    return {
        "label":      label,
        "output_dir": str(output_dir),
        "plot_dir":   str(dirs["dataset"]),
        "status":     "done",
    }


# ---------------------------------------------------------------------------
# CSV writer (internal)
# ---------------------------------------------------------------------------

def _write_dataset_csv(
    d_set:        str,
    target_list:  List[Any],
    temp_sim_opt: Dict[str, List[float]],
    dirs:         Dict[str, Path],
) -> None:
    """
    Write a single per-dataset CSV file to the appropriate Plot/Dataset sub-folder.

    Ignition delay (IDT) targets (``Tig``, ``RCM``, ``JSR``) produce::

        DS_ID, T, Obs(us), Nominal

    Flame-speed targets (``Fls``) produce::

        DS_ID, T, P, Phi, Fuel, Ox, BathGas, Obs(us), Nominal

    Plug-flow-reactor targets (``PFR``) produce::

        DS_ID, T, P, species, Obs(us), Nominal

    (closes the pipeline gap noted in H2/review/flag.md -- PFR cases ran to
    completion and wrote their own output/pfr_full.json per case, but this
    writer previously had no branch for the target type, so no
    Plot/Dataset/PFR/*.csv was ever produced.)
    """
    matching = [t for t in target_list if t.dataSet_id == d_set]
    if not matching:
        return

    target_type = matching[0].target   # "Tig", "RCM", "JSR", "Fls", "PFR"

    if target_type in ("Tig", "RCM", "JSR"):
        lines = ["DS_ID,T,Obs(us),Nominal\n"]
        for t in matching:
            vals = temp_sim_opt.get(str(t.index))
            if not vals:
               logger.warning("No simulated value for target index %s (dataset %s) — writing NaN.",
                            t.index, d_set)
               sim_val = float("nan")
            else:
               sim_val = vals[0]
            lines.append(f"{t.uniqueID},{t.temperature},{t.observed},{sim_val}\n")

        out_dir = dirs.get(target_type.lower(), dirs["tig"])
        (out_dir / f"{d_set}.csv").write_text("".join(lines))
        logger.debug("  CSV → %s / %s.csv", out_dir.name, d_set)

    elif target_type == "Fls":
        lines = ["DS_ID,T,P,Phi,Fuel,Ox,BathGas,Obs(us),Nominal\n"]
        for t in matching:
            # Task 4B: fix double-index — vals is already the list; take first element directly
            vals = temp_sim_opt.get(str(t.index), [])
            sim_val = vals[0] if vals else float("nan")
            lines.append(
                f"{t.uniqueID},{t.temperature},{t.pressure},{t.phi},"
                f"{t.fuel_dict},{t.oxidizer_x},{t.BG_dict},"
                f"{t.observed},{sim_val}\n"
            )
        (dirs["fls"] / f"{d_set}.csv").write_text("".join(lines))
        logger.debug("  CSV → fls / %s.csv", d_set)

    elif target_type == "PFR":
        lines = ["DS_ID,T,P,species,Obs(us),Nominal\n"]
        for t in matching:
            vals = temp_sim_opt.get(str(t.index), [])
            sim_val = vals[0] if vals else float("nan")
            species = t.add.get("species", "") if getattr(t, "add", None) else ""
            lines.append(f"{t.uniqueID},{t.temperature},{t.pressure},{species},"
                         f"{t.observed},{sim_val}\n")
        (dirs["pfr"] / f"{d_set}.csv").write_text("".join(lines))
        logger.debug("  CSV → pfr / %s.csv", d_set)

    else:
        logger.warning(
            "Unknown target type '%s' for dataset '%s' — CSV not written.",
            target_type, d_set,
        )
