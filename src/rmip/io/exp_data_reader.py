"""
rmip/io/exp_data_reader.py
==========================
Reads experimental IDT / LFS data from CSV (primary) or YAML into the
RMIP database via schema_phase3 CRUD helpers.

Expected CSV columns (case-insensitive, order-independent):
  Required : T_K, value
  Contextual: P_atm, phi, uncertainty, units, notes, source_ref

If `units` column is absent, the caller must pass `default_units`.
If `phi` / `P_atm` are absent, they default to the values supplied
in the load call (useful for iso-phi or iso-pressure datasets).

Examples
--------
>>> reader = ExpDataReader("project.db")
>>> ds_id = reader.load_csv(
...     "curran1998_idt.csv",
...     dataset_name="Curran1998_IDT_nC7H16",
...     sim_type="IDT",
...     fuel_name="NC7H16",
...     source_ref="Curran et al. Combust. Flame 1998",
...     default_units="s",
...     default_P_atm=13.5,
...     default_phi=1.0,
... )
>>> points = reader.get_points(ds_id)
"""

from __future__ import annotations

import csv
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml  # PyYAML

from rmip.database.schema_phase3 import (
    insert_exp_dataset,
    insert_exp_data_points,
    get_exp_data_points,
    list_exp_datasets,
)

logger = logging.getLogger(__name__)


class ExpDataReader:
    """
    Reads experimental IDT / LFS data and loads it into the RMIP database.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database (must already be initialised by
        schema.init_db + schema_phase3.upgrade_db).
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_csv(
        self,
        filepath: str,
        *,
        dataset_name:   str,
        sim_type:       str,                  # 'IDT' | 'LFS'
        fuel_name:      Optional[str] = None,
        source_ref:     Optional[str] = None,
        default_units:  str           = "s",  # 's' for IDT, 'm/s' for LFS
        default_T_K:    Optional[float] = None,
        default_P_atm:  Optional[float] = None,
        default_phi:    Optional[float] = None,
        comment_char:   str           = "#",
    ) -> int:
        """
        Load an experimental CSV file.  Returns the dataset_id.
        Existing dataset with the same name is re-loaded (old points kept).

        CSV may contain a header comment block prefixed with `comment_char`.

        Parameters
        ----------
        filepath : str
            Path to the CSV file.
        dataset_name : str
            Unique label for this dataset in the DB.
        sim_type : str
            'IDT' or 'LFS'.
        fuel_name : str, optional
            Fuel label (informational only, stored with dataset).
        source_ref : str, optional
            Literature reference string.
        default_units : str
            Fallback if 'units' column absent.  'IDT' → 's', 'LFS' → 'm/s'.
        default_T_K / default_P_atm / default_phi : float, optional
            Column-level defaults used when a column is missing.
        comment_char : str
            Lines starting with this character are ignored.
        """
        sim_type = sim_type.upper()
        if sim_type not in ("IDT", "LFS"):
            raise ValueError(f"sim_type must be 'IDT' or 'LFS', got {sim_type!r}")

        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Experimental data file not found: {filepath}")

        rows = self._parse_csv(path, comment_char)
        points = self._normalise_rows(
            rows,
            default_units  = default_units,
            default_T_K    = default_T_K,
            default_P_atm  = default_P_atm,
            default_phi    = default_phi,
        )

        conn = self._connect()
        try:
            ds_id = insert_exp_dataset(
                conn,
                name       = dataset_name,
                sim_type   = sim_type,
                fuel_name  = fuel_name,
                source_ref = source_ref,
                filepath   = str(path.resolve()),
            )
            insert_exp_data_points(conn, ds_id, points)
            logger.info(
                "Loaded %d experimental points into dataset '%s' (id=%d)",
                len(points), dataset_name, ds_id,
            )
            return ds_id
        finally:
            conn.close()

    def load_yaml(
        self,
        filepath: str,
        *,
        dataset_name: str,
        sim_type:     str,
        fuel_name:    Optional[str] = None,
        source_ref:   Optional[str] = None,
        default_units: str          = "s",
    ) -> int:
        """
        Load a YAML experimental data file.
        Expected structure::

            metadata:
              sim_type: IDT
              fuel_name: NC7H16
              source_ref: "..."
            data:
              - {T_K: 800, P_atm: 13.5, phi: 1.0, value: 2.3e-3, units: s}
              - ...
        """
        path = Path(filepath)
        with open(path) as fh:
            doc = yaml.safe_load(fh)

        meta  = doc.get("metadata", {})
        rows  = doc.get("data", [])
        sim_type   = sim_type.upper()
        fuel_name  = fuel_name  or meta.get("fuel_name")
        source_ref = source_ref or meta.get("source_ref")

        points = self._normalise_rows(rows, default_units=default_units)

        conn = self._connect()
        try:
            ds_id = insert_exp_dataset(
                conn,
                name       = dataset_name,
                sim_type   = sim_type,
                fuel_name  = fuel_name,
                source_ref = source_ref,
                filepath   = str(path.resolve()),
            )
            insert_exp_data_points(conn, ds_id, points)
            logger.info("Loaded %d points from YAML into dataset '%s'",
                        len(points), dataset_name)
            return ds_id
        finally:
            conn.close()

    def get_points(self, dataset_id: int) -> List[Dict[str, Any]]:
        """Return all experimental data points for a dataset."""
        conn = self._connect()
        try:
            return get_exp_data_points(conn, dataset_id)
        finally:
            conn.close()

    def list_datasets(self) -> List[Dict[str, Any]]:
        """Return all datasets registered in the DB."""
        conn = self._connect()
        try:
            return list_exp_datasets(conn)
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _parse_csv(
        path: Path,
        comment_char: str,
    ) -> List[Dict[str, str]]:
        """Parse CSV, stripping comment lines, return list of dicts (all str)."""
        lines = []
        with open(path, newline="") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped or stripped.startswith(comment_char):
                    continue
                lines.append(line)

        reader = csv.DictReader(lines)
        # Normalise header keys to lowercase
        rows = []
        for row in reader:
            rows.append({k.strip().lower(): v.strip() for k, v in row.items()})
        return rows

    @staticmethod
    def _normalise_rows(
        rows: List[Dict[str, Any]],
        *,
        default_units: str            = "s",
        default_T_K:   Optional[float] = None,
        default_P_atm: Optional[float] = None,
        default_phi:   Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convert raw row dicts into normalised point dicts ready for DB insert.
        Handles both lowercase-string keys (from CSV) and direct float dicts
        (from YAML).
        """
        points = []
        for i, row in enumerate(rows):
            # Case-insensitive key normalisation for YAML dicts
            r = {str(k).lower(): v for k, v in row.items()}

            def _get(key: str, default=None):
                v = r.get(key, default)
                if v is None:
                    return default
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return v   # return as-is for string fields

            T_K   = _get("t_k",   default_T_K)
            P_atm = _get("p_atm", default_P_atm)
            phi   = _get("phi",   default_phi)
            value = _get("value")

            if value is None:
                logger.warning("Row %d: missing 'value', skipping", i)
                continue

            points.append({
                "T_K":         T_K,
                "P_atm":       P_atm,
                "phi":         phi,
                "value":       float(value),
                "uncertainty": _get("uncertainty"),
                "units":       str(r.get("units", default_units)).strip(),
                "notes":       r.get("notes") or r.get("source_ref") or None,
            })

        return points
