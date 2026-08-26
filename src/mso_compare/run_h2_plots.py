#!/usr/bin/env python
"""
run_h2_plots.py
----------------
Standalone CLI for regenerating the H2/O2 extreme-condition sweep's full
figure set (``H2/plots/*.png``) -- the same figures embedded in
``H2/paper/main.tex`` -- so a future run doesn't require knowing to invoke
``src/MUQ-SAC/VisualAid.py``'s ``H2SweepPlotter`` and
``H2/standalone/explosion_limit_map.py`` by hand from inside ``H2/``.

This wraps two independent pieces, run in order:

1. **Data generation** (optional, ``--regen-data``):
   ``H2/standalone/explosion_limit_map.py`` -- the analytic 2nd-limit
   crossover locus, the traced (bisected) numeric ignitability boundary
   per key phi, the dense 20-level-phi ignitable-volume and
   stiffness-volume grids (H2/review/flag.md §0), and the key-temperature
   comparison table. Takes on the order of 10-15 minutes (the dense
   stiffness volume alone is ~17,600 Jacobian evaluations); skip with
   ``--no-regen-data`` (the default) to just re-plot from whatever CSVs
   already exist in ``H2/results/``.
2. **Plotting**: ``src/MUQ-SAC/VisualAid.py``'s ``H2SweepPlotter.run_all()``
   -- every figure in ``H2/plots/``, from the CSVs in ``H2/results/`` and
   the pipeline's own ``H2/SIM_RUNS/<sim_run_label>/`` output.

Usage
-----
Re-plot only, from existing H2/results/ CSVs (fast, the common case)::

    python src/mso_compare/run_h2_plots.py

Regenerate the underlying data first, then plot (slow, ~10-15 min)::

    python src/mso_compare/run_h2_plots.py --regen-data

Point at a differently-located H2 study directory, or a different
pipeline-driven nominal-sim run label::

    python src/mso_compare/run_h2_plots.py --h2-dir /path/to/H2 \\
        --sim-run-label nominal_chemkin_pro
"""

import argparse
import sys
from pathlib import Path

# ── Make the src/ package importable when run as a script ─────────────────
_SCRIPTS_DIR = Path(__file__).resolve().parent
_SRC_DIR     = _SCRIPTS_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
# H2SweepPlotter lives in src/MUQ-SAC/VisualAid.py -- "MUQ-SAC" has a
# hyphen, so it can't be a dotted package import; add it to sys.path
# directly, same as H2/standalone/*.py's own ad-hoc convention.
_MUQSAC_DIR = _SRC_DIR / "MUQ-SAC"
if str(_MUQSAC_DIR) not in sys.path:
    sys.path.insert(0, str(_MUQSAC_DIR))

from mso_compare.utils import setup_logging  # noqa: E402

logger = setup_logging("run_h2_plots")


def _regen_data(h2_dir: Path) -> None:
    """Run H2/standalone/explosion_limit_map.py's full data-generation
    pipeline (crossover locus, traced boundary, dense ignitable-volume and
    stiffness-volume grids, key-temperature table) against *h2_dir*."""
    standalone_dir = h2_dir / "standalone"
    if str(standalone_dir) not in sys.path:
        sys.path.insert(0, str(standalone_dir))
    import explosion_limit_map as elm  # noqa
    logger.info("Regenerating H2 sweep data via %s ...", standalone_dir / "explosion_limit_map.py")
    crossover_rows = elm.build_crossover_locus()
    elm.build_numeric_boundary()
    elm.build_numeric_volume_dense()
    stiffness_rows = elm.build_dense_stiffness_field()
    elm.build_stiffness_volume_dense()
    elm.build_key_temperature_comparison(crossover_rows, stiffness_rows)
    logger.info("✓  Data regeneration complete.")


def generate_h2_plots(h2_dir: str = "H2", sim_run_label: str = "nominal_cantera",
                      regen_data: bool = False) -> dict:
    """Regenerate the full H2/O2 extreme-condition sweep figure set.
    Returns the same {figure_key: path_or_list} dict H2SweepPlotter.run_all()
    does. Importable directly (e.g. from a notebook or another script), not
    just via this file's CLI."""
    import VisualAid  # H2SweepPlotter (src/MUQ-SAC/VisualAid.py)

    h2_path = Path(h2_dir).resolve()
    if not h2_path.is_dir():
        raise FileNotFoundError(f"H2 study directory not found: {h2_path}")

    if regen_data:
        _regen_data(h2_path)

    logger.info("Plotting H2 sweep figures: h2_dir=%s, sim_run_label=%s", h2_path, sim_run_label)
    plotter = VisualAid.H2SweepPlotter(str(h2_path))
    results = plotter.run_all(sim_run_label=sim_run_label)

    n_ok, n_err = 0, 0
    for key, val in results.items():
        if key.endswith("_error"):
            n_err += 1
            logger.warning("  %s: %s", key, val)
            continue
        if val:
            n_ok += 1 if isinstance(val, str) else len(val)
    logger.info("✓  Wrote %d figure(s) to %s (%d plot function(s) raised).",
                n_ok, h2_path / "plots", n_err)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="run_h2_plots.py",
        description="Regenerate the H2/O2 extreme-condition sweep's full figure set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--h2-dir", default="H2",
                        help="Path to the H2 study directory (default: H2, i.e. run from the repo root).")
    parser.add_argument("--sim-run-label", default="nominal_cantera",
                        help="Pipeline-driven nominal-sim run label to read IDT/species-profile "
                             "data from, under H2/SIM_RUNS/ (default: nominal_cantera).")
    parser.add_argument("--regen-data", action="store_true",
                        help="Regenerate H2/results/*.csv first via "
                             "H2/standalone/explosion_limit_map.py (slow, ~10-15 min: the dense "
                             "stiffness-volume grid alone is ~17,600 Jacobian evaluations). "
                             "Default: off -- just re-plot from whatever CSVs already exist.")
    args = parser.parse_args()

    generate_h2_plots(h2_dir=args.h2_dir, sim_run_label=args.sim_run_label,
                      regen_data=args.regen_data)


if __name__ == "__main__":
    main()
