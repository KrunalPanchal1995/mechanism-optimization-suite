#!/usr/bin/env python
"""
run_nominal_sim.py
------------------
Standalone CLI for launching a single nominal simulation run.

Usage modes
-----------
1. YAML config mode (recommended)::

       python scripts/run_nominal_sim.py \\
           --config mso_config.yaml \\
           --run nominal_htc

2. Dry-run to verify paths without executing::

       python scripts/run_nominal_sim.py \\
           --config mso_config.yaml \\
           --run nominal_htc \\
           --dry-run

3. Interactive mode (prompts for all inputs)::

       python scripts/run_nominal_sim.py --interactive

Available run labels (set in your YAML config)
-----------------------------------------------
  nominal_htc   nominal_ltc
  stage1_htc    stage1_ltc
  stage2_htc    stage2_ltc
  (or any custom label you define)
"""

import argparse
import sys
from pathlib import Path

# ── Make the src/ package importable when run as a script ─────────────────
_SCRIPTS_DIR = Path(__file__).resolve().parent
_SRC_DIR     = _SCRIPTS_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from mso_compare.config     import load_config, SimRunConfig
from mso_compare.sim_runner import run_nominal_simulation
from mso_compare.utils      import setup_logging

logger = setup_logging("run_nominal_sim")


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def _interactive() -> None:
    """Prompt the user for all required inputs and run one nominal simulation."""
    print()
    print("=" * 62)
    print("  mso_compare — Nominal Simulation (Interactive Mode)")
    print("=" * 62)

    print("\nSimulation type:")
    print("  1)  High-Temperature (HTC) targets")
    print("  2)  Low-Temperature  (LTC) targets")
    print("  3)  Custom (specify target file manually)")
    choice = input("Select [1/2/3, default 3]: ").strip() or "3"

    label_map = {"1": "nominal_htc", "2": "nominal_ltc", "3": "nominal_custom"}
    label     = label_map.get(choice, "nominal_custom")

    base_opt  = input("\nPath to base .opt template file: ").strip()
    mechanism = input("Path to mechanism file (.yaml / chemkin): ").strip()
    targets   = input("Path to targets file (.target): ").strip()
    t_count   = int(input("Number of targets to read: ").strip())
    thermo    = input("Path to thermo file: ").strip()
    trans     = input("Path to trans  file: ").strip()
    uncert    = input(
        "Path to uncertainty file (press Enter to use dummy): "
    ).strip() or targets  # dummy = targets path (not used in nominal)
    default_out = f"./SIM_RUNS/{label}"
    output    = input(f"Output directory [{default_out}]: ").strip() or default_out
    muqsac    = input("Path to src/MUQ-SAC directory: ").strip()

    sim_cfg = SimRunConfig(
        label            = label,
        mechanism        = mechanism,
        targets          = targets,
        targets_count    = t_count,
        thermo_file      = thermo,
        trans_file       = trans,
        uncertainty_data = uncert,
        output_dir       = output,
    )

    print(f"\nLaunching '{label}' …\n")
    result = run_nominal_simulation(
        sim_run_cfg   = sim_cfg,
        base_opt_file = base_opt,
        muqsac_dir    = muqsac,
    )
    print(f"\n✓  Done.  Status  : {result['status']}")
    print(f"   Plots written to: {result.get('plot_dir', 'N/A')}")


# ---------------------------------------------------------------------------
# Config-file mode
# ---------------------------------------------------------------------------

def _from_config(yaml_path: str, run_label: str, dry_run: bool = False) -> None:
    """Run one simulation using the pipeline YAML config."""
    cfg = load_config(yaml_path)

    if run_label not in cfg.simulations:
        logger.error(
            "Run label '%s' not found in config.  Available: %s",
            run_label,
            list(cfg.simulations.keys()),
        )
        sys.exit(1)

    sim_cfg = cfg.simulations[run_label]
    result  = run_nominal_simulation(
        sim_run_cfg   = sim_cfg,
        base_opt_file = cfg.base_opt_file,
        muqsac_dir    = cfg.muqsac_src_dir,
        dry_run       = dry_run,
    )
    logger.info("Status: %s", result["status"])
    if result["status"] == "done":
        logger.info("Plot output: %s", result.get("plot_dir", "N/A"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="run_nominal_sim.py",
        description="Launch a single nominal simulation run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to the YAML pipeline config file (mso_config.yaml).",
    )
    parser.add_argument(
        "--run", "-r",
        type=str,
        help="Run label to execute, e.g. nominal_htc.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without executing.",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Use interactive prompt mode.",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List all run labels defined in the config and exit.",
    )

    args = parser.parse_args()

    if args.interactive:
        _interactive()
        return

    if args.list_runs:
        if not args.config:
            parser.error("--list-runs requires --config.")
        cfg = load_config(args.config)
        print("Configured run labels:")
        for lbl in cfg.simulations:
            print(f"  {lbl}")
        return

    if args.config and args.run:
        _from_config(args.config, args.run, dry_run=args.dry_run)
        return

    parser.print_help()
    print(
        "\nError: Provide --interactive, or both --config and --run.\n"
        "       Use --list-runs to see available run labels.\n"
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
