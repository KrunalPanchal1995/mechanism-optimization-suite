#!/usr/bin/env python
"""
run_merge.py
------------
Standalone CLI for merging nominal simulation CSV outputs into an organised
(φ, P) folder structure.

This script corresponds to Phase 2 of the MSO pipeline.  It reads the
``Plot/Dataset/Tig/`` directories from each configured simulation run,
merges matching files across Nominal / Stage-1 / Stage-2 runs, and writes
standardised CSV files organised by φ sub-folder.

Usage
-----
Process both HTC and LTC::

    python scripts/run_merge.py --config mso_config.yaml

Process only HTC::

    python scripts/run_merge.py --config mso_config.yaml --regime htc

Dry run (show paths without writing)::

    python scripts/run_merge.py --config mso_config.yaml --dry-run

Output structure
----------------
::

    MERGE_OUTPUT/
    ├── HTC/
    │   ├── all/           ← Nominal + Stage-1 + Stage-2
    │   │   ├── phi_0_5/
    │   │   │   ├── Phi_0_5_P_10.csv
    │   │   │   └── Phi_0_5_P_20.csv
    │   │   ├── phi_1_0/
    │   │   └── phi_2_0/
    │   ├── nominal_s1/    ← Nominal + Stage-1 only
    │   │   └── phi_*/...
    │   └── nominal_s2/    ← Nominal + Stage-2 only
    │       └── phi_*/...
    └── LTC/
        └── ...  (same structure)
"""

import argparse
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_SRC_DIR     = _SCRIPTS_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from mso_compare.config import load_config
from mso_compare.merge  import merge_simulation_results
from mso_compare.utils  import setup_logging

logger = setup_logging("run_merge")


def _resolve_regime(regime: str):
    if regime == "both":
        return ["htc", "ltc"]
    if regime in ("htc", "ltc"):
        return [regime]
    raise ValueError(f"Invalid regime '{regime}'. Use 'htc', 'ltc', or 'both'.")


def _merge_regime(cfg, prefix: str, dry_run: bool = False) -> None:
    """Run the merge step for one regime (htc or ltc)."""

    def _plot_dir(label: str):
        """Return Plot/Dataset/Tig directory for a run label, or None."""
        sim = cfg.simulations.get(label)
        if sim is None:
            return None
        return Path(sim.output_dir) / "Plot" / "Dataset" / "Tig"

    nominal_dir = _plot_dir(f"nominal_{prefix}")
    stage1_dir  = _plot_dir(f"stage1_{prefix}")
    stage2_dir  = _plot_dir(f"stage2_{prefix}")
    out_root    = Path(getattr(cfg.merge, f"{prefix}_output_dir")).resolve()

    if nominal_dir is None or not nominal_dir.exists():
        logger.warning(
            "Nominal Plot/Dataset/Tig directory not found for %s.  "
            "Run the nominal simulation first, or check output_dir in the config.",
            prefix.upper(),
        )
        return

    logger.info("Merging %s regime:", prefix.upper())
    logger.info("  Nominal  : %s", nominal_dir)
    logger.info("  Stage-1  : %s", stage1_dir  or "(not configured)")
    logger.info("  Stage-2  : %s", stage2_dir  or "(not configured)")
    logger.info("  Output   : %s", out_root)

    if dry_run:
        logger.info("[DRY RUN] Merge would write to: %s", out_root)
        return

    merge_simulation_results(
        nominal_folder = nominal_dir,
        stage1_folder  = stage1_dir,
        stage2_folder  = stage2_dir,
        output_root    = out_root,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="run_merge.py",
        description="Merge nominal simulation CSV outputs by φ and P.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        type=str,
        help="Path to the YAML pipeline config file.",
    )
    parser.add_argument(
        "--regime",
        choices=["htc", "ltc", "both"],
        default="both",
        help="Which regime to merge (default: both).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print paths without writing files.",
    )

    args   = parser.parse_args()
    cfg    = load_config(args.config)

    for prefix in _resolve_regime(args.regime):
        _merge_regime(cfg, prefix, dry_run=args.dry_run)

    logger.info("✓  Merge step complete.")


if __name__ == "__main__":
    main()
