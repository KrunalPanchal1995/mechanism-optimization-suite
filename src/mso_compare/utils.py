"""
utils.py
--------
Shared utility functions for the mso_compare framework.

Contents
--------
setup_logging          — Configure and return a timestamped logger
parse_phi_pressure     — Extract φ and P from dataset IDs / filenames
standardise_filename   — Convert dataset ID to standard CSV filename
phi_folder_name        — Return the sub-folder name for a φ value
load_opt_file          — Load a .opt YAML file
apply_overrides        — Deep-merge override dict into a base config dict
build_run_config       — Build the optInputs dict for one SimRunConfig
setup_muqsac_path      — Add MUQ-SAC directory to sys.path
"""

from __future__ import annotations

import copy
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(
    name:     str,
    log_file: Optional[str] = None,
    level:    int = logging.INFO,
) -> logging.Logger:
    """
    Create and configure a logger.

    A StreamHandler to stdout is always added.  If *log_file* is given,
    a FileHandler is also added (parent directories are created automatically).

    Parameters
    ----------
    name : str
        Logger name (typically ``__name__`` of the calling module).
    log_file : str | None
        Optional path to a log file.
    level : int
        Logging level (default ``logging.INFO``).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls (e.g., in interactive use)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Optional file handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Dataset-ID / filename parsing
# ---------------------------------------------------------------------------

# Matches:  ...phi0_5_p10...   phi1_p20   phi2_0_p40  (case-insensitive)
# Groups:   (phi_digits)       (pressure_digits)
_DSID_RE = re.compile(
    r"phi(\d+(?:_\d+)?)_p(\d+)",
    re.IGNORECASE,
)


def parse_phi_pressure(name: str) -> Tuple[str, str]:
    """
    Parse equivalence ratio (φ) and pressure (P) from a dataset ID or filename.

    The function looks for the pattern ``phi<value>_p<value>`` (case-insensitive).
    Underscores in the φ value are interpreted as decimal separators:
    ``phi0_5`` → φ = 0.5.

    Parameters
    ----------
    name : str
        Dataset ID (e.g. ``ing_BUTADIENE_HTC_phi0_5_p10``) or
        filename (e.g. ``Phi_0_5_P_10.csv``).

    Returns
    -------
    (phi_str, pressure_str)
        Both use underscore notation, e.g. ``("0_5", "10")``.

    Raises
    ------
    ValueError
        If the pattern cannot be found in *name*.

    Examples
    --------
    >>> parse_phi_pressure("ing_BUTADIENE_HTC_phi0_5_p10")
    ('0_5', '10')
    >>> parse_phi_pressure("Phi_1_0_P_20.csv")
    ('1_0', '20')
    >>> parse_phi_pressure("phi2_p40")
    ('2', '40')
    """
    # Normalise underscored patterns like "Phi_0_5_P_10" → "phi0_5_p10"
    normalised = re.sub(r"_+", "_", name)           # collapse runs of _
    normalised = re.sub(r"(?i)phi_", "phi", normalised)  # remove _ after Phi
    normalised = re.sub(r"(?i)_p_",  "_p",  normalised)  # remove _ in _P_

    m = _DSID_RE.search(normalised)
    if not m:
        raise ValueError(
            f"Cannot parse φ/P from '{name}'.  "
            "Expected pattern: phi<digits>[_<digits>]_p<digits>"
        )
    return m.group(1), m.group(2)


def standardise_filename(dataset_id: str) -> str:
    """
    Convert a dataset ID or original filename to the standard CSV filename.

    Examples
    --------
    >>> standardise_filename("ing_BUTADIENE_HTC_phi0_5_p10")
    'Phi_0_5_P_10.csv'
    >>> standardise_filename("phi2_p40.csv")
    'Phi_2_P_40.csv'
    """
    phi_str, pres_str = parse_phi_pressure(dataset_id)
    return f"Phi_{phi_str}_P_{pres_str}.csv"


def phi_folder_name(phi_str: str) -> str:
    """
    Return the sub-folder name for a given φ string.

    Examples
    --------
    >>> phi_folder_name("0_5")
    'phi_0_5'
    >>> phi_folder_name("1")
    'phi_1'
    """
    return f"phi_{phi_str}"


# ---------------------------------------------------------------------------
# .opt file helpers
# ---------------------------------------------------------------------------

def load_opt_file(opt_path: str) -> dict:
    """
    Load a ``.opt`` YAML configuration file.

    Parameters
    ----------
    opt_path : str
        Path to the ``.opt`` file.

    Returns
    -------
    dict
        Parsed YAML content.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    p = Path(opt_path)
    if not p.exists():
        raise FileNotFoundError(f".opt file not found: {opt_path}")
    with open(p, "r") as fh:
        return yaml.safe_load(fh)


def apply_overrides(base_config: dict, overrides: dict) -> dict:
    """
    Deep-merge *overrides* into *base_config* and return a new dict.

    For each key in *overrides*:
    * If the key exists in *base_config* and both values are dicts,
      the override dict is merged into the base dict (existing keys updated,
      new keys added).
    * Otherwise the entire value is replaced.

    The original *base_config* is never mutated.

    Parameters
    ----------
    base_config : dict
        The original ``.opt`` config.
    overrides : dict
        Mapping of ``{section: {key: value}}`` to apply.

    Returns
    -------
    dict
        A new dict with overrides applied.
    """
    cfg = copy.deepcopy(base_config)
    for section, values in overrides.items():
        if (
            isinstance(values, dict)
            and section in cfg
            and isinstance(cfg[section], dict)
        ):
            cfg[section].update(values)
        else:
            cfg[section] = values
    return cfg


def build_run_config(base_opt: dict, sim_run_cfg: Any) -> dict:
    """
    Build the ``optInputs`` dict for one simulation run.

    Applies the :class:`~mso_compare.config.SimRunConfig` fields as overrides
    to *base_opt*, then applies any ``extra_overrides`` on top.

    Parameters
    ----------
    base_opt : dict
        Base ``.opt`` config dict (loaded by :func:`load_opt_file`).
    sim_run_cfg : SimRunConfig
        Run-specific configuration.

    Returns
    -------
    dict
        The merged ``optInputs`` dict ready to pass to MUQ-SAC modules.
    """
    overrides: Dict[str, Any] = {
        "Locations": {
            "mechanism":        sim_run_cfg.mechanism,
            "targets":          sim_run_cfg.targets,
            "uncertainty_data": sim_run_cfg.uncertainty_data,
            "thermo_file":      sim_run_cfg.thermo_file,
            "trans_file":       sim_run_cfg.trans_file,
        },
        "Counts": {
            "targets_count": sim_run_cfg.targets_count,
        },
    }
    # Apply extra overrides last so they take precedence
    for section, vals in sim_run_cfg.extra_overrides.items():
        if section in overrides and isinstance(vals, dict):
            overrides[section].update(vals)
        else:
            overrides[section] = vals

    return apply_overrides(base_opt, overrides)


# ---------------------------------------------------------------------------
# sys.path helper
# ---------------------------------------------------------------------------

def setup_muqsac_path(muqsac_dir: str) -> None:
    """
    Prepend the MUQ-SAC source directory to ``sys.path``.

    Idempotent — calling multiple times with the same path has no effect.

    Parameters
    ----------
    muqsac_dir : str
        Path to the ``src/MUQ-SAC`` directory.
    """
    resolved = str(Path(muqsac_dir).resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
        logging.getLogger(__name__).debug("Added to sys.path: %s", resolved)
