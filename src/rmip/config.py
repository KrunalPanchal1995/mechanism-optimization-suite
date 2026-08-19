"""
rmip/config.py
==============
Central configuration dataclass and logging initialiser for RMIP.

All modules receive an RMIPConfig instance (or its attributes) at
construction time.  Never import config from within io/ or core/
readers — pass the db_path as a plain string instead to keep those
modules independent.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class RMIPConfig:
    """
    Project-level configuration.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.  Created if it does not exist.
    log_level : str
        Python logging level string: 'DEBUG'|'INFO'|'WARNING'|'ERROR'.
    cantera_cache_dir : str | None
        Directory for cached Cantera YAML files produced from CHEMKIN input.
        Defaults to ``<db_parent>/.rmip_cantera_cache/``.
    regime_thresholds : dict
        Configurable LTC/NTC/HTC boundary temperatures (K).
        Keys: ``LTC_MAX_K`` (default 850), ``NTC_MAX_K`` (default 1000).
    tol_A : float
        Relative tolerance on pre-exponential factor A for kinetic identity.
    tol_n : float
        Absolute tolerance on temperature exponent n.
    tol_Ea : float
        Absolute tolerance on activation energy Ea [J/mol].
    n_sim_workers : int
        Number of parallel workers for batch simulation runs.
    """

    db_path:            str
    log_level:          str   = "INFO"
    cantera_cache_dir:  Optional[str] = None
    regime_thresholds:  Dict[str, float] = field(default_factory=lambda: {
        "LTC_MAX_K": 850.0,
        "NTC_MAX_K": 1000.0,
    })
    # Kinetic identity tolerances (GraphComparator defaults)
    tol_A:              float = 0.01   # 1 % relative
    tol_n:              float = 0.001
    tol_Ea:             float = 100.0  # J/mol
    n_sim_workers:      int   = 1

    def __post_init__(self) -> None:
        if self.cantera_cache_dir is None:
            db_parent = Path(self.db_path).parent
            self.cantera_cache_dir = str(db_parent / ".rmip_cantera_cache")
        setup_logging(self.log_level)

    @property
    def LTC_MAX_K(self) -> float:
        return self.regime_thresholds["LTC_MAX_K"]

    @property
    def NTC_MAX_K(self) -> float:
        return self.regime_thresholds["NTC_MAX_K"]


def setup_logging(level: str = "INFO") -> None:
    """
    Configure the root logger for RMIP.

    Safe to call multiple times — subsequent calls update the level only
    if a handler has already been attached.
    """
    numeric = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger("rmip")
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        root.addHandler(handler)
    root.setLevel(numeric)
