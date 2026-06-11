#!/usr/bin/env python
"""
run_mso_pipeline.py
-------------------
Full end-to-end MSO pipeline CLI.

Executes all four phases in sequence:

    Phase 1 — Nominal simulations  (run_nominal_sim logic)
    Phase 2 — Merge CSV outputs    (run_merge logic)
    Phase 3 — Error metrics        (chi-squared CSVs)
    Phase 4 — Plots                (per-φ PDFs + combined panels)

Any phase can be skipped with the corresponding ``--skip-*`` flag, which is
useful when re-running the pipeline after making changes to a later phase
(e.g. regenerating plots without re-running simulations).

Usage
-----
Full run::

    python scripts/run_mso_pipeline.py --config mso_config.yaml

Inspect the plan without executing::

    python scripts/run_mso_pipeline.py --config mso_config.yaml --dry-run

Skip simulations (use existing results)::

    python scripts/run_mso_pipeline.py --config mso_config.yaml \\
        --skip-simulations

Re-run only plots::

    python scripts/run_mso_pipeline.py --config mso_config.yaml \\
        --skip-simulations --skip-merge --skip-error-metrics

HTC regime only::

    python scripts/run_mso_pipeline.py --config mso_config.yaml \\
        --regime htc

Config file
-----------
The config file must be a YAML file that follows the schema shown in
``configs/template_mso_pipeline.yaml``.  It should live in your working
directory.  A minimal skeleton::

    project:
      name: Butadiene_MSO
      working_dir: /path/to/project
      fuel: C4H6

    base_opt_file: /path/to/base.opt
    muqsac_src_dir: /path/to/src/MUQ-SAC

    simulations:
      nominal_htc: { mechanism: ..., targets: ..., ... }
      stage1_htc:  { mechanism: ..., targets: ..., ... }
      stage2_htc:  { mechanism: ..., targets: ..., ... }
      nominal_ltc: { ... }
      ...

    merge:
      htc_output_dir: ./MERGE_OUTPUT/HTC
      ltc_output_dir: ./MERGE_OUTPUT/LTC

    error_metrics:
      htc_output_dir: ./ERROR_METRICS/HTC
      ltc_output_dir: ./ERROR_METRICS/LTC

    plots:
      htc_output_dir: ./PLOTS/HTC
      ltc_output_dir: ./PLOTS/LTC
      mode: auto
      plot_per_phi:  true
      plot_combined: true

See ``configs/template_mso_pipeline.yaml`` for the fully annotated schema.
"""

import argparse
import logging
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_SRC_DIR     = _SCRIPTS_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from mso_compare.pipeline import MSO_Pipeline
from mso_compare.utils    import setup_logging

logger = setup_logging("run_mso_pipeline")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="run_mso_pipeline.py",
        description="Full MSO pipeline: simulate → merge → errors → plot.",
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
        "--dry-run",
        action="store_true",
        help="Print the pipeline plan without executing.",
    )
    parser.add_argument(
        "--skip-simulations",
        action="store_true",
        help="Skip Phase 1 (use existing simulation outputs).",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip Phase 2 (use existing merged outputs).",
    )
    parser.add_argument(
        "--skip-error-metrics",
        action="store_true",
        help="Skip Phase 3.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip Phase 4.",
    )
    parser.add_argument(
        "--regime",
        choices=["htc", "ltc", "both"],
        default="both",
        help="Which regime to process in Phases 2–4 (default: both).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    pipeline = MSO_Pipeline.from_yaml(args.config)

    pipeline.run(
        dry_run            = args.dry_run,
        skip_simulations   = args.skip_simulations,
        skip_merge         = args.skip_merge,
        skip_error_metrics = args.skip_error_metrics,
        skip_plots         = args.skip_plots,
        regime             = args.regime,
    )


if __name__ == "__main__":
    main()
