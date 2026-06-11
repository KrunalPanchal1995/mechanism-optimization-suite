"""
config.py
---------
Configuration dataclasses and YAML config loader for the mso_compare framework.

The pipeline is driven by a single YAML file (mso_config.yaml) that lives in
the user's working directory.  See configs/template_mso_pipeline.yaml for a
fully annotated example.

Dataclasses
-----------
SimRunConfig     — one nominal simulation run (mechanism + targets + paths)
MergeConfig      — output roots for HTC / LTC merged CSVs
ErrorMetricsConfig — output roots + uncertainty fraction for error CSVs
PlotConfig       — output roots + plot style options
PipelineConfig   — top-level container for all of the above

Public API
----------
load_config(yaml_path) -> PipelineConfig
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simulation-run configuration
# ---------------------------------------------------------------------------

@dataclass
class SimRunConfig:
    """
    Configuration for one nominal simulation run.

    Parameters
    ----------
    label : str
        Human-readable identifier, e.g. ``nominal_htc``, ``stage1_ltc``.
    mechanism : str
        Absolute path to the mechanism file (YAML or CHEMKIN format).
    targets : str
        Absolute path to the ``.target`` file for this run.
    targets_count : int
        Number of targets to read from the targets file.
    thermo_file : str
        Absolute path to the thermodynamics data file.
    trans_file : str
        Absolute path to the transport data file.
    uncertainty_data : str
        Absolute path to the uncertainty XML file.
        For nominal simulations this field is not actually used;
        a dummy path is acceptable.
    output_dir : str
        Directory where all outputs for this run will be written.
        Created automatically if it does not exist.
    extra_overrides : dict
        Optional additional key→value pairs to override in the ``.opt``
        config before the simulation is launched.  The dict is structured
        as ``{section: {key: value}}``, mirroring the ``.opt`` YAML layout.
    """

    label:            str
    mechanism:        str
    targets:          str
    targets_count:    int
    thermo_file:      str
    trans_file:       str
    uncertainty_data: str
    output_dir:       str
    extra_overrides:  Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """
        Check that required file paths exist on disk.

        Raises
        ------
        FileNotFoundError
            If any required file (mechanism, targets, thermo, trans) is missing.
        """
        required = {
            "mechanism":  self.mechanism,
            "targets":    self.targets,
            "thermo_file": self.thermo_file,
            "trans_file": self.trans_file,
        }
        missing = [f"{k} = {v}" for k, v in required.items()
                   if not Path(v).exists()]
        if missing:
            raise FileNotFoundError(
                f"[{self.label}] The following files were not found:\n"
                + "\n".join(f"  {m}" for m in missing)
            )

        # Uncertainty file may be dummy for nominal runs — only warn
        if self.uncertainty_data and not Path(self.uncertainty_data).exists():
            logger.warning(
                "[%s] uncertainty_data not found: %s  "
                "(acceptable for nominal simulations)",
                self.label, self.uncertainty_data,
            )


# ---------------------------------------------------------------------------
# Sub-configuration blocks
# ---------------------------------------------------------------------------

@dataclass
class MergeConfig:
    """Output directories for merged CSV results."""
    htc_output_dir: str = "./MERGE_OUTPUT/HTC"
    ltc_output_dir: str = "./MERGE_OUTPUT/LTC"


@dataclass
class ErrorMetricsConfig:
    """Output directories and uncertainty fraction for error-metric CSVs."""
    htc_output_dir:       str   = "./ERROR_METRICS/HTC"
    ltc_output_dir:       str   = "./ERROR_METRICS/LTC"
    #: σ = uncertainty_fraction × obs  (default 10 %)
    uncertainty_fraction: float = 0.10


@dataclass
class PlotConfig:
    """Output directories and rendering options for comparison plots."""
    htc_output_dir: str  = "./PLOTS/HTC"
    ltc_output_dir: str  = "./PLOTS/LTC"
    plot_per_phi:   bool = True   # one PDF per φ sub-folder
    plot_combined:  bool = True   # one multi-panel PDF for all φ values
    #: 'auto' detects from CSV columns; other valid values:
    #: 'nominal_only' | 'nominal_vs_s1' | 'nominal_vs_s2' | 'all_stages'
    mode:           str  = "auto"


# ---------------------------------------------------------------------------
# Top-level pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    Top-level container for the entire MSO pipeline configuration.

    Populated by :func:`load_config` from a YAML file.
    """

    project_name:   str
    working_dir:    str
    fuel:           str
    base_opt_file:  str

    #: Dict keyed by run label, e.g. 'nominal_htc', 'stage1_htc', …
    simulations:   Dict[str, SimRunConfig]

    merge:          MergeConfig
    error_metrics:  ErrorMetricsConfig
    plots:          PlotConfig

    #: Path to the src/MUQ-SAC directory (added to sys.path at runtime).
    muqsac_src_dir: str = ""

    # ── Convenience helpers ────────────────────────────────────────────────

    def available_runs(self) -> List[str]:
        """Return list of configured run labels."""
        return list(self.simulations.keys())

    def has_htc_runs(self) -> bool:
        return any("htc" in k for k in self.simulations)

    def has_ltc_runs(self) -> bool:
        return any("ltc" in k for k in self.simulations)

    def is_mso_complete(self) -> bool:
        """True if all six MSO runs are configured."""
        required = {
            "nominal_htc", "nominal_ltc",
            "stage1_htc",  "stage1_ltc",
            "stage2_htc",  "stage2_ltc",
        }
        return required.issubset(set(self.simulations))

    def is_nominal_only(self) -> bool:
        """True if no stage runs are configured (single nominal comparison)."""
        return not any("stage" in k for k in self.simulations)


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def load_config(yaml_path: str) -> PipelineConfig:
    """
    Load a pipeline YAML configuration file and return a :class:`PipelineConfig`.

    Parameters
    ----------
    yaml_path : str
        Path to the YAML config file.  See ``configs/template_mso_pipeline.yaml``
        for the full schema.

    Raises
    ------
    FileNotFoundError
        If *yaml_path* does not exist.
    KeyError
        If a required field is missing in the YAML.
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(path, "r") as fh:
        raw: dict = yaml.safe_load(fh)

    # ── Project section ────────────────────────────────────────────────────
    proj = raw.get("project", {})

    # ── Simulation runs ────────────────────────────────────────────────────
    sim_configs: Dict[str, SimRunConfig] = {}
    for label, sim_raw in raw.get("simulations", {}).items():
        sim_configs[label] = SimRunConfig(
            label            = label,
            mechanism        = sim_raw["mechanism"],
            targets          = sim_raw["targets"],
            targets_count    = int(sim_raw["targets_count"]),
            thermo_file      = sim_raw["thermo_file"],
            trans_file       = sim_raw["trans_file"],
            uncertainty_data = sim_raw.get("uncertainty_data", ""),
            output_dir       = sim_raw["output_dir"],
            extra_overrides  = sim_raw.get("extra_overrides", {}),
        )

    # ── Sub-configs ────────────────────────────────────────────────────────
    merge_raw = raw.get("merge", {})
    err_raw   = raw.get("error_metrics", {})
    plot_raw  = raw.get("plots", {})

    return PipelineConfig(
        project_name   = proj.get("name",        "MSO_Project"),
        working_dir    = proj.get("working_dir",  str(Path.cwd())),
        fuel           = proj.get("fuel",         ""),
        base_opt_file  = raw["base_opt_file"],
        simulations    = sim_configs,
        merge          = MergeConfig(
            htc_output_dir = merge_raw.get("htc_output_dir", "./MERGE_OUTPUT/HTC"),
            ltc_output_dir = merge_raw.get("ltc_output_dir", "./MERGE_OUTPUT/LTC"),
        ),
        error_metrics  = ErrorMetricsConfig(
            htc_output_dir       = err_raw.get("htc_output_dir",       "./ERROR_METRICS/HTC"),
            ltc_output_dir       = err_raw.get("ltc_output_dir",       "./ERROR_METRICS/LTC"),
            uncertainty_fraction = float(err_raw.get("uncertainty_fraction", 0.10)),
        ),
        plots          = PlotConfig(
            htc_output_dir = plot_raw.get("htc_output_dir", "./PLOTS/HTC"),
            ltc_output_dir = plot_raw.get("ltc_output_dir", "./PLOTS/LTC"),
            plot_per_phi   = bool(plot_raw.get("plot_per_phi",   True)),
            plot_combined  = bool(plot_raw.get("plot_combined",  True)),
            mode           = plot_raw.get("mode", "auto"),
        ),
        muqsac_src_dir = raw.get("muqsac_src_dir", ""),
    )
