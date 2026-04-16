"""
dataset_tools.py

High-level utilities to:
- load `.target` + `.add` into `combustion_target` objects
- export a uniform CSV with dynamic fuel/bath-gas columns
- visualize the (T, P, phi) space (with optional dilution coloring)
- build a sparse subset (~100 points) without losing coverage

This is written to work with the *refactored* `combustion_target` parser:
    from combustion_target_class_refactored import combustion_target
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Exporter:
    """
    Export a uniform CSV from a list of `combustion_target` objects.

    The exporter builds a *global* list of fuel species and bath-gas species
    present across the whole dataset, and creates one mole-fraction column per
    species. Any species missing for a point is set to 0.0.
    """

    def __init__(self, targets: Sequence[combustion_target]):
        self.targets = list(targets)

        # Collect species sets
        self.fuel_species = sorted({sp for t in self.targets for sp in (t.fuel_dict or {}).keys()})
        self.bg_species = sorted({sp for t in self.targets for sp in (t.BG_dict or {}).keys()})
        self.oxidizer_species = sorted({t.oxidizer for t in self.targets if getattr(t, "oxidizer", None)})

        # We'll create one column per species with prefix 'X_' (mole fraction)
        # This stays unambiguous and uniform.
        self.species_columns = sorted({*self.fuel_species, *self.bg_species, *self.oxidizer_species})
        self.species_columns = [f"X_{sp}" for sp in self.species_columns]

    def to_dataframe(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        # map species col -> species name
        sp_from_col = {col: col.replace("X_", "", 1) for col in self.species_columns}

        for i, t in enumerate(self.targets, start=1):
            row: Dict[str, Any] = {
                "sr_no": i,
                "dataset_ID": t.dataSet_id,
                "Temperature_K": t.temperature,
                "Pressure_Pa": t.pressure,
                "Phi": t.phi,
                "Fuel_list": ",".join(sorted((t.fuel_dict or {}).keys())),
                "Fuel_total": float(sum((t.fuel_dict or {}).values())) if t.fuel_dict else 0.0,
                "Bathgas_list": ",".join(sorted((t.BG_dict or {}).keys())),
                "Bathgas_total": float(sum((t.BG_dict or {}).values())) if t.BG_dict else 0.0,
                "target_type": t.target,
                "observed": t.observed,
                "obs_unit": t.units["observed"] if t.units and "observed" in t.units else None,
            }

            # Fill species mole fractions (fuels + bathgas + oxidizer)
            species_mix = {}
            if t.species_dict:
                species_mix.update(t.species_dict)

            for col, sp in sp_from_col.items():
                row[col] = float(species_mix.get(sp, 0.0))

            rows.append(row)

        # Stable column ordering
        base_cols = [
            "sr_no", "dataset_ID",
            "Temperature_K", "Pressure_Pa", "Phi",
            "Fuel_list", "Fuel_total",
            "Bathgas_list", "Bathgas_total",
            "target_type", "observed", "obs_unit",
        ]
        df = pd.DataFrame(rows)
        # Add species columns at the end
        ordered = [c for c in base_cols if c in df.columns] + [c for c in self.species_columns if c in df.columns]
        return df[ordered]

    def to_csv(self, out_path: str | Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(out_path, index=False)
        return out_path