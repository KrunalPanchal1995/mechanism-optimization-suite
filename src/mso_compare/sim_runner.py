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
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    logger.info("Directory structure ready under: %s", output_dir)
    return dirs


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_nominal_simulation(
    sim_run_cfg: SimRunConfig,
    base_opt_file: str,
    muqsac_dir:  str  = "",
    dry_run:     bool = False,
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
        import data_management               # noqa: PLC0415
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
    os.chdir(nominal_dir)

    try:
        if not progress_file.exists():
            logger.info("No progress file found — launching simulations …")
            exe_locations: List[str] = (
                simulator.SM(target_list, optInputs, {}, [])
                .make_nominal_dir_in_parallel()
            )
        else:
            logger.info("Progress file detected — loading existing simulation locations …")
            exe_locations = locations_file.read_text().splitlines()
    finally:
        # Always restore the original working directory
        os.chdir(original_cwd)

    # ── Extract simulation results ─────────────────────────────────────────
    logger.info("Extracting simulation results …")
    fuel            = optInputs["Inputs"].get("fuel", {})
    temp_sim_opt:   Dict[str, List[float]] = {}

    for case_idx in range(len(target_list)):
        cached_file = dirs["sims"] / f"sim_data_case-{case_idx}.lst"

        if cached_file.exists():
            # Re-use previously extracted results
            rows = [r for r in cached_file.read_text().splitlines() if r.strip()]
            eta  = [np.exp(float(r.split("\t")[1])) / 10.0 for r in rows]
            temp_sim_opt[str(case_idx)] = eta
        else:
            # Extract from simulation case directory
            case_dir_path = nominal_dir / f"case-{case_idx}"
            os.chdir(case_dir_path)
            try:
                data_sheet, failed_sim, eta_arr = (
                    data_management.generate_target_value_tables(
                        exe_locations, target_list, case_idx, fuel
                    )
                )
                temp_sim_opt[str(case_idx)] = list(np.exp(eta_arr) / 10.0)
            finally:
                os.chdir(original_cwd)

            # Cache results
            (dirs["sims"] / f"sim_data_case-{case_idx}.lst").write_text(data_sheet)
            (dirs["sims"] / f"failed_sim_data_case-{case_idx}.lst").write_text(failed_sim)

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
    """
    matching = [t for t in target_list if t.dataSet_id == d_set]
    if not matching:
        return

    target_type = matching[0].target   # "Tig", "RCM", "JSR", "Fls"

    if target_type in ("Tig", "RCM", "JSR"):
        lines = ["DS_ID,T,Obs(us),Nominal\n"]
        for t in matching:
            sim_val = temp_sim_opt.get(str(t.index), [None])[0]
            lines.append(f"{t.uniqueID},{t.temperature},{t.observed},{sim_val}\n")

        out_dir = dirs.get(target_type.lower(), dirs["tig"])
        (out_dir / f"{d_set}.csv").write_text("".join(lines))
        logger.debug("  CSV → %s / %s.csv", out_dir.name, d_set)

    elif target_type == "Fls":
        lines = ["DS_ID,T,P,Phi,Fuel,Ox,BathGas,Obs(us),Nominal\n"]
        for t in matching:
            sim_val = temp_sim_opt.get(str(t.index), [None])[0]
            lines.append(
                f"{t.uniqueID},{t.temperature},{t.pressure},{t.phi},"
                f"{t.fuel_dict},{t.oxidizer_x},{t.BG_dict},"
                f"{t.observed},{sim_val}\n"
            )
        (dirs["fls"] / f"{d_set}.csv").write_text("".join(lines))
        logger.debug("  CSV → fls / %s.csv", d_set)

    else:
        logger.warning(
            "Unknown target type '%s' for dataset '%s' — CSV not written.",
            target_type, d_set,
        )
