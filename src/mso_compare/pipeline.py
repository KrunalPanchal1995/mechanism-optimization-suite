"""
pipeline.py
-----------
End-to-end MSO pipeline orchestrator for the mso_compare framework.

The :class:`MSO_Pipeline` class drives all four phases of the workflow:

1. **Simulate**     — Run up to six nominal simulations (nominal/Stage-1/Stage-2
                       × HTC/LTC) using the configured ``.opt`` overrides.
2. **Merge**        — Align per-run CSV outputs into ``all/``, ``nominal_s1/``,
                       ``nominal_s2/`` sub-folders organised by φ.
3. **Error metrics** — Compute χ² error CSVs that mirror the merge structure.
4. **Plot**          — Generate per-φ PDFs and combined multi-panel PDFs.

Each phase is skippable (``skip_*=True``) so the pipeline can be re-run
from any point, e.g. to regenerate plots after a style change.

The pipeline gracefully handles **partial configurations**:
* A single-run config (only ``nominal_htc`` defined) is treated as a
  *nominal-only* workflow — no merge is attempted.
* If Stage-1 or Stage-2 folders are absent the merge step emits a warning
  and skips those comparison sets.

Usage (Python)
--------------
    from mso_compare.pipeline import MSO_Pipeline
    pipe = MSO_Pipeline.from_yaml("mso_config.yaml")
    pipe.run(dry_run=True)          # inspect plan
    pipe.run()                      # execute

Usage (CLI)
-----------
    python scripts/run_mso_pipeline.py --config mso_config.yaml
    python scripts/run_mso_pipeline.py --config mso_config.yaml --dry-run
    python scripts/run_mso_pipeline.py --config mso_config.yaml \\
        --skip-simulations --regime htc
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .config       import PipelineConfig, load_config
from .sim_runner   import run_nominal_simulation
from .merge        import merge_simulation_results
from .error_metrics import generate_error_csvs
from .plotter      import plot_comparison
from .utils        import setup_logging

logger = logging.getLogger(__name__)

# Run-label tuples for each regime
_HTC_RUNS = ("nominal_htc", "stage1_htc", "stage2_htc")
_LTC_RUNS = ("nominal_ltc", "stage1_ltc", "stage2_ltc")


class MSO_Pipeline:
    """
    Full MSO pipeline orchestrator.

    Parameters
    ----------
    config : PipelineConfig
        Loaded and validated pipeline configuration.
    """

    def __init__(self, config: PipelineConfig):
        self.cfg = config

    # ── Constructor ────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "MSO_Pipeline":
        """
        Build a pipeline from a YAML config file.

        Parameters
        ----------
        yaml_path : str
            Path to ``mso_config.yaml``.
        """
        cfg = load_config(yaml_path)
        return cls(cfg)

    # ── Plan / dry-run ─────────────────────────────────────────────────────

    def print_plan(self) -> None:
        """Print a human-readable summary of the pipeline plan."""
        cfg = self.cfg
        width = 62
        sep   = "=" * width

        print(f"\n{sep}")
        print(f"  MSO Pipeline — {cfg.project_name}")
        print(sep)
        print(f"  Working dir     : {cfg.working_dir}")
        print(f"  Base .opt file  : {cfg.base_opt_file}")
        print(f"  MUQ-SAC dir     : {cfg.muqsac_src_dir or '(not set)'}")
        print(f"  Pipeline mode   : {'nominal-only' if cfg.is_nominal_only() else 'full MSO' if cfg.is_mso_complete() else 'partial'}")

        print(f"\n  Configured simulation runs ({len(cfg.simulations)}):")
        for label, sim in cfg.simulations.items():
            print(f"    [{label}]")
            print(f"      Mechanism     : {sim.mechanism}")
            print(f"      Targets file  : {sim.targets}")
            print(f"      Targets count : {sim.targets_count}")
            print(f"      Output dir    : {sim.output_dir}")

        print(f"\n  Merge output:")
        print(f"    HTC  →  {cfg.merge.htc_output_dir}")
        print(f"    LTC  →  {cfg.merge.ltc_output_dir}")

        print(f"\n  Error metrics:")
        print(f"    HTC  →  {cfg.error_metrics.htc_output_dir}")
        print(f"    LTC  →  {cfg.error_metrics.ltc_output_dir}")
        print(f"    σ = {cfg.error_metrics.uncertainty_fraction * 100:.0f} % of observed")

        print(f"\n  Plots:")
        print(f"    HTC  →  {cfg.plots.htc_output_dir}")
        print(f"    LTC  →  {cfg.plots.ltc_output_dir}")
        print(f"    Mode         : {cfg.plots.mode}")
        print(f"    Per-φ PDF    : {cfg.plots.plot_per_phi}")
        print(f"    Combined PDF : {cfg.plots.plot_combined}")
        print(sep + "\n")

    # ── Phase 1: Simulations ───────────────────────────────────────────────

    def run_simulations(self, dry_run: bool = False) -> None:
        """
        Run all configured nominal simulations.

        Parameters
        ----------
        dry_run : bool
            If ``True``, print what would run without executing.
        """
        logger.info("── Phase 1: Nominal Simulations ─────────────────────────")
        for label, sim_cfg in self.cfg.simulations.items():
            logger.info("Launching run: %s", label)
            result = run_nominal_simulation(
                sim_run_cfg  = sim_cfg,
                base_opt_file= self.cfg.base_opt_file,
                muqsac_dir   = self.cfg.muqsac_src_dir,
                dry_run      = dry_run,
            )
            logger.info("  → %s  [status: %s]", label, result["status"])

    # ── Phase 2: Merge ─────────────────────────────────────────────────────

    def run_merge(self, regime: str = "both") -> None:
        """
        Merge per-run CSV outputs into the structured output folders.

        Parameters
        ----------
        regime : 'htc' | 'ltc' | 'both'
        """
        logger.info("── Phase 2: Merging Results ─────────────────────────────")

        regimes = _resolve_regime(regime)
        for pfx in regimes:
            self._merge_regime(pfx)

    def _merge_regime(self, prefix: str) -> None:
        """Run merge for one regime (``htc`` or ``ltc``)."""
        nominal_dir = self._sim_plot_dir(f"nominal_{prefix}")
        stage1_dir  = self._sim_plot_dir(f"stage1_{prefix}")
        stage2_dir  = self._sim_plot_dir(f"stage2_{prefix}")
        out_root    = Path(
            getattr(self.cfg.merge, f"{prefix}_output_dir")
        ).resolve()

        if nominal_dir is None or not nominal_dir.exists():
            logger.warning(
                "Nominal Plot/Dataset/Tig directory not found for %s — merge skipped.",
                prefix.upper(),
            )
            return

        logger.info("Merging %s → %s", prefix.upper(), out_root)
        merge_simulation_results(
            nominal_folder = nominal_dir,
            stage1_folder  = stage1_dir,
            stage2_folder  = stage2_dir,
            output_root    = out_root,
        )

    # ── Phase 3: Error metrics ─────────────────────────────────────────────

    def run_error_metrics(self, regime: str = "both") -> None:
        """
        Compute and write chi-squared error CSVs.

        Parameters
        ----------
        regime : 'htc' | 'ltc' | 'both'
        """
        logger.info("── Phase 3: Error Metrics ───────────────────────────────")

        for pfx in _resolve_regime(regime):
            merge_root = Path(getattr(self.cfg.merge, f"{pfx}_output_dir")).resolve()
            error_root = Path(getattr(self.cfg.error_metrics, f"{pfx}_output_dir")).resolve()

            if not merge_root.exists():
                logger.warning(
                    "Merge root not found for %s — error metrics skipped.", pfx.upper()
                )
                continue

            logger.info(
                "Error metrics %s: %s → %s", pfx.upper(), merge_root, error_root
            )
            generate_error_csvs(
                merge_root           = merge_root,
                error_root           = error_root,
                uncertainty_fraction = self.cfg.error_metrics.uncertainty_fraction,
            )

    # ── Phase 4: Plots ─────────────────────────────────────────────────────

    def run_plots(self, regime: str = "both") -> None:
        """
        Generate all IDT comparison plots.

        Iterates over every comparison sub-folder under each merge root
        (``all/``, ``nominal_s1/``, ``nominal_s2/``, ``nominal_only/``)
        and calls :func:`~mso_compare.plotter.plot_comparison` for each.

        Parameters
        ----------
        regime : 'htc' | 'ltc' | 'both'
        """
        logger.info("── Phase 4: Plots ───────────────────────────────────────")

        for pfx in _resolve_regime(regime):
            merge_root = Path(getattr(self.cfg.merge,  f"{pfx}_output_dir")).resolve()
            plot_root  = Path(getattr(self.cfg.plots,  f"{pfx}_output_dir")).resolve()

            if not merge_root.exists():
                logger.warning(
                    "Merge root not found for %s — plots skipped.", pfx.upper()
                )
                continue

            # Each immediate sub-folder is one comparison set
            for sub_dir in sorted(
                d for d in merge_root.iterdir() if d.is_dir()
            ):
                out_sub = plot_root / sub_dir.name
                logger.info("  Plotting: %s / %s", pfx.upper(), sub_dir.name)
                plot_comparison(
                    data_dir      = sub_dir,
                    plot_dir      = out_sub,
                    mode          = self.cfg.plots.mode,
                    unc_frac      = self.cfg.error_metrics.uncertainty_fraction,
                    plot_per_phi  = self.cfg.plots.plot_per_phi,
                    plot_combined = self.cfg.plots.plot_combined,
                )

    # ── Full pipeline ──────────────────────────────────────────────────────

    def run(
        self,
        dry_run:             bool = False,
        skip_simulations:    bool = False,
        skip_merge:          bool = False,
        skip_error_metrics:  bool = False,
        skip_plots:          bool = False,
        regime:              str  = "both",
    ) -> None:
        """
        Execute the full pipeline end-to-end.

        Parameters
        ----------
        dry_run : bool
            Print a plan without executing any simulations.
        skip_simulations : bool
            Skip Phase 1 (use existing simulation outputs).
        skip_merge : bool
            Skip Phase 2 (use existing merged outputs).
        skip_error_metrics : bool
            Skip Phase 3.
        skip_plots : bool
            Skip Phase 4.
        regime : 'htc' | 'ltc' | 'both'
            Which regime to process in Phases 2–4.
        """
        self.print_plan()

        if dry_run:
            logger.info("[DRY RUN] Plan printed — no execution.")
            return

        if not skip_simulations:
            self.run_simulations()

        if not skip_merge:
            self.run_merge(regime=regime)

        if not skip_error_metrics:
            self.run_error_metrics(regime=regime)

        if not skip_plots:
            self.run_plots(regime=regime)

        logger.info("✓  Pipeline complete.")

    # ── Internal helpers ───────────────────────────────────────────────────

    def _sim_plot_dir(self, label: str) -> Optional[Path]:
        """
        Return the ``Plot/Dataset/Tig`` directory for a simulation-run label,
        or ``None`` if that label is not in the configuration.
        """
        sim = self.cfg.simulations.get(label)
        if sim is None:
            return None
        return Path(sim.output_dir) / "Plot" / "Dataset" / "Tig"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_regime(regime: str):
    """Return a list of prefix strings from a regime specifier."""
    if regime == "both":
        return ["htc", "ltc"]
    if regime in ("htc", "ltc"):
        return [regime]
    raise ValueError(f"Invalid regime '{regime}'. Use 'htc', 'ltc', or 'both'.")
