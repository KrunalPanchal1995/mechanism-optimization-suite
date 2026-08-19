"""
rmip/analysis/comparison.py
============================
Module 6: Mechanism Comparison Engine.

Compares two or more parsed mechanisms on:
  1. Species coverage (matched by inchi_key — D1)
  2. Reaction coverage (matched by canonical_key — D2)
  3. Kinetic rate ratio k_A/k_B at a temperature grid
  4. NASA-7 Cp/H/S polynomial differences at a temperature grid

This is the first module that crosses mechanism boundaries.
GraphComparator (Phase 3) extends this with classification + union spec.

Usage::

    from rmip.analysis.comparison import ComparisonEngine

    eng = ComparisonEngine("project.db")
    report = eng.compare([1, 2, 3])
    eng.save_report(report, output_dir="comparison_out/")
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from rmip.database.schema import init_db

log = logging.getLogger("rmip.analysis.comparison")

# Temperature grid for kinetic rate comparison [K]
_T_GRID_K = [300.0, 500.0, 700.0, 1000.0, 1200.0, 1500.0, 2000.0]
_R_SI     = 8.314462618  # J/mol/K


class ComparisonEngine:
    """
    Compare N ≥ 2 mechanisms.

    Parameters
    ----------
    db_path : str
        Path to the project SQLite database.
    """

    def __init__(self, db_path: str) -> None:
        self._db   = db_path
        self._conn = init_db(db_path)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def compare(self, mechanism_ids: List[int]) -> Dict[str, Any]:
        """
        Perform a full comparison.

        Returns
        -------
        dict with keys:
            mechanism_ids, mechanism_names,
            species_comparison, reaction_comparison,
            kinetic_diffs, thermo_diffs
        """
        if len(mechanism_ids) < 2:
            raise ValueError("Need at least 2 mechanism IDs to compare.")

        names = self._get_names(mechanism_ids)
        log.info("Comparing mechanisms: %s", list(zip(mechanism_ids, names)))

        sp_report   = self._compare_species(mechanism_ids)
        rxn_report  = self._compare_reactions(mechanism_ids)
        kin_diffs   = self._compute_kinetic_diffs(rxn_report["matched"], mechanism_ids)
        thermo_diffs = self._compute_thermo_diffs(sp_report["matched_by_inchi"], mechanism_ids)

        return {
            "mechanism_ids":       mechanism_ids,
            "mechanism_names":     names,
            "species_comparison":  sp_report,
            "reaction_comparison": rxn_report,
            "kinetic_diffs":       kin_diffs,
            "thermo_diffs":        thermo_diffs,
        }

    # ------------------------------------------------------------------
    # Species comparison
    # ------------------------------------------------------------------

    def _compare_species(
        self, mech_ids: List[int]
    ) -> Dict[str, Any]:
        """
        Cross-reference species by inchi_key.
        Returns:
          universal  — inchi_keys present in ALL mechanisms
          partial    — inchi_keys present in ≥2 but not all
          unique     — {mech_id: [inchi_keys only in that mech]}
          matched_by_inchi — {inchi_key: {mech_id: sp_row}}
        """
        # Load species for each mechanism
        sp_sets: Dict[int, Dict[str, sqlite3.Row]] = {}
        for mid in mech_ids:
            rows = self._conn.execute(
                "SELECT * FROM species WHERE mechanism_id=? AND inchi_key IS NOT NULL",
                (mid,),
            ).fetchall()
            sp_sets[mid] = {r["inchi_key"]: r for r in rows}

        all_keys = set().union(*[set(s.keys()) for s in sp_sets.values()])

        universal:  List[str] = []
        partial:    List[str] = []
        unique:     Dict[int, List[str]] = {mid: [] for mid in mech_ids}
        matched:    Dict[str, Dict[int, Any]] = {}

        for key in all_keys:
            present = [mid for mid in mech_ids if key in sp_sets[mid]]
            entry   = {mid: dict(sp_sets[mid][mid]) for mid in present if mid in sp_sets and key in sp_sets[mid]}
            matched[key] = entry
            if len(present) == len(mech_ids):
                universal.append(key)
            elif len(present) == 1:
                unique[present[0]].append(key)
            else:
                partial.append(key)

        return {
            "n_total_unique_species": len(all_keys),
            "n_universal":  len(universal),
            "n_partial":    len(partial),
            "n_unique":     {str(mid): len(v) for mid, v in unique.items()},
            "universal":    universal,
            "partial":      partial,
            "unique":       {str(mid): v for mid, v in unique.items()},
            "matched_by_inchi": matched,
        }

    # ------------------------------------------------------------------
    # Reaction comparison
    # ------------------------------------------------------------------

    def _compare_reactions(
        self, mech_ids: List[int]
    ) -> Dict[str, Any]:
        """
        Cross-reference reactions by canonical_key.
        """
        rx_sets: Dict[int, Dict[str, sqlite3.Row]] = {}
        for mid in mech_ids:
            rows = self._conn.execute(
                "SELECT * FROM reactions WHERE mechanism_id=? AND canonical_key IS NOT NULL",
                (mid,),
            ).fetchall()
            rx_sets[mid] = {r["canonical_key"]: r for r in rows}

        all_keys = set().union(*[set(s.keys()) for s in rx_sets.values()])

        shared:  List[str] = []
        unique:  Dict[int, List[str]] = {mid: [] for mid in mech_ids}
        matched: Dict[str, Dict[int, Any]] = {}

        for key in all_keys:
            present = [mid for mid in mech_ids if key in rx_sets[mid]]
            entry   = {mid: dict(rx_sets[mid][key]) for mid in present}
            matched[key] = entry
            if len(present) > 1:
                shared.append(key)
            else:
                unique[present[0]].append(key)

        return {
            "n_total_unique_reactions": len(all_keys),
            "n_shared":  len(shared),
            "n_unique":  {str(mid): len(v) for mid, v in unique.items()},
            "shared":    shared,
            "unique":    {str(mid): v for mid, v in unique.items()},
            "matched":   matched,
        }

    # ------------------------------------------------------------------
    # Kinetic rate comparison
    # ------------------------------------------------------------------

    def _compute_kinetic_diffs(
        self,
        matched:   Dict[str, Dict[int, Any]],
        mech_ids:  List[int],
    ) -> List[Dict[str, Any]]:
        """
        For each shared reaction, compute k at the T grid and relative difference.
        Returns list of dicts with comparison data.
        """
        diffs = []
        mids  = mech_ids

        for canonical_key, mech_data in matched.items():
            present_ids = [mid for mid in mids if mid in mech_data]
            if len(present_ids) < 2:
                continue

            # Reference = first mechanism
            ref_mid  = present_ids[0]
            ref_rx   = mech_data[ref_mid]
            formula  = ref_rx["formula_string"]
            k_ref    = self._k_at_grid(ref_rx)

            comparisons = []
            for mid in present_ids[1:]:
                rx    = mech_data[mid]
                k_cmp = self._k_at_grid(rx)
                ratios = []
                for i, T in enumerate(_T_GRID_K):
                    k_r = k_ref[i]
                    k_c = k_cmp[i]
                    if k_r is not None and k_c is not None and k_r > 0:
                        ratios.append(k_c / k_r)
                    else:
                        ratios.append(None)
                comparisons.append({
                    "mech_ref":    ref_mid,
                    "mech_cmp":    mid,
                    "T_grid_K":    _T_GRID_K,
                    "k_ref":       k_ref,
                    "k_cmp":       k_cmp,
                    "k_ratio":     ratios,
                    "max_ratio":   max((r for r in ratios if r is not None), default=None),
                })

            diffs.append({
                "canonical_key": canonical_key,
                "formula_string": formula,
                "present_in_mechs": present_ids,
                "params": {
                    str(mid): {
                        "A_si":  mech_data[mid]["A_si"],
                        "n":     mech_data[mid]["n"],
                        "Ea_si": mech_data[mid]["Ea_si"],
                    }
                    for mid in present_ids
                },
                "comparisons": comparisons,
            })
        return diffs

    @staticmethod
    def _k_at_grid(rx: Dict[str, Any]) -> List[Optional[float]]:
        """Compute k = A * T^n * exp(-Ea/RT) at each T in the grid."""
        A  = rx.get("A_si")
        n  = rx.get("n")
        Ea = rx.get("Ea_si")
        if A is None or n is None or Ea is None:
            return [None] * len(_T_GRID_K)
        try:
            return [A * (T ** n) * math.exp(-Ea / (_R_SI * T)) for T in _T_GRID_K]
        except (ValueError, OverflowError):
            return [None] * len(_T_GRID_K)

    # ------------------------------------------------------------------
    # Thermo comparison
    # ------------------------------------------------------------------

    def _compute_thermo_diffs(
        self,
        matched_by_inchi: Dict[str, Dict[int, Any]],
        mech_ids: List[int],
    ) -> List[Dict[str, Any]]:
        """
        For each universally-matched species, compare Cp/R at a T grid.
        """
        T_grid = [300.0, 500.0, 700.0, 1000.0, 1500.0, 2000.0]
        diffs  = []

        for inchi_key, mech_data in matched_by_inchi.items():
            present_ids = [mid for mid in mech_ids if mid in mech_data]
            if len(present_ids) < 2:
                continue

            ref_mid  = present_ids[0]
            ref_sp   = mech_data[ref_mid]
            name     = ref_sp.get("name")
            cp_ref   = self._cp_at_grid(ref_sp, T_grid)

            comparisons = []
            for mid in present_ids[1:]:
                sp    = mech_data[mid]
                cp_c  = self._cp_at_grid(sp, T_grid)
                diffs_cp = [
                    abs(a - b) if a is not None and b is not None else None
                    for a, b in zip(cp_ref, cp_c)
                ]
                comparisons.append({
                    "mech_ref":  ref_mid,
                    "mech_cmp":  mid,
                    "T_grid_K":  T_grid,
                    "Cp_R_ref":  cp_ref,
                    "Cp_R_cmp":  cp_c,
                    "abs_diff":  diffs_cp,
                })

            diffs.append({
                "inchi_key":       inchi_key,
                "species_name":    name,
                "present_in_mechs": present_ids,
                "comparisons":     comparisons,
            })
        return diffs

    @staticmethod
    def _cp_at_grid(
        sp: Dict[str, Any], T_grid: List[float]
    ) -> List[Optional[float]]:
        """Compute Cp/R from NASA-7 coefficients at each T in T_grid."""
        def nasa7_cp(coeffs: List[float], T: float) -> float:
            a = coeffs
            return a[0] + a[1]*T + a[2]*T**2 + a[3]*T**3 + a[4]*T**4

        lo_raw = sp.get("coeffs_low_json") or sp.get("coeffs_low")
        hi_raw = sp.get("coeffs_high_json") or sp.get("coeffs_high")
        tmid   = sp.get("tmid_K") or 1000.0

        lo = json.loads(lo_raw) if isinstance(lo_raw, str) else lo_raw
        hi = json.loads(hi_raw) if isinstance(hi_raw, str) else hi_raw

        if lo is None or hi is None:
            return [None] * len(T_grid)

        results = []
        for T in T_grid:
            try:
                coeffs = lo if T <= tmid else hi
                results.append(nasa7_cp(coeffs, T))
            except Exception:
                results.append(None)
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_names(self, mech_ids: List[int]) -> List[str]:
        names = []
        for mid in mech_ids:
            row = self._conn.execute(
                "SELECT name FROM mechanisms WHERE id=?", (mid,)
            ).fetchone()
            names.append(row["name"] if row else f"mech_{mid}")
        return names

    # ------------------------------------------------------------------
    # Save / export
    # ------------------------------------------------------------------

    def save_report(
        self,
        report:     Dict[str, Any],
        output_dir: str = ".",
    ) -> None:
        """
        Save comparison report as JSON + two CSV files.

        Files written:
          ``comparison_report.json``
          ``kinetic_diff.csv``
          ``thermo_diff.csv``
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Full JSON report (omit the large matched dicts to keep it readable)
        summary = {
            k: v for k, v in report.items()
            if k not in ("species_comparison", "reaction_comparison")
        }
        summary["species_summary"] = {
            k: v for k, v in report["species_comparison"].items()
            if k not in ("matched_by_inchi", "universal", "partial")
        }
        summary["reaction_summary"] = {
            k: v for k, v in report["reaction_comparison"].items()
            if k not in ("matched", "shared")
        }
        with open(out / "comparison_report.json", "w") as fh:
            json.dump(summary, fh, indent=2, default=str)

        # Kinetic diff CSV
        kin_rows = []
        for entry in report["kinetic_diffs"]:
            for cmp in entry["comparisons"]:
                for i, T in enumerate(_T_GRID_K):
                    kin_rows.append({
                        "canonical_key": entry["canonical_key"],
                        "formula":       entry["formula_string"],
                        "mech_ref":      cmp["mech_ref"],
                        "mech_cmp":      cmp["mech_cmp"],
                        "T_K":           T,
                        "k_ref":         cmp["k_ref"][i],
                        "k_cmp":         cmp["k_cmp"][i],
                        "k_ratio":       cmp["k_ratio"][i],
                    })
        _write_csv(out / "kinetic_diff.csv", kin_rows)

        # Thermo diff CSV
        thermo_rows = []
        for entry in report["thermo_diffs"]:
            for cmp in entry["comparisons"]:
                for i, T in enumerate(cmp["T_grid_K"]):
                    thermo_rows.append({
                        "inchi_key":    entry["inchi_key"],
                        "species_name": entry["species_name"],
                        "mech_ref":     cmp["mech_ref"],
                        "mech_cmp":     cmp["mech_cmp"],
                        "T_K":          T,
                        "CpR_ref":      cmp["Cp_R_ref"][i],
                        "CpR_cmp":      cmp["Cp_R_cmp"][i],
                        "abs_diff":     cmp["abs_diff"][i],
                    })
        _write_csv(out / "thermo_diff.csv", thermo_rows)

        log.info("Saved comparison report to '%s/'", output_dir)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    import csv
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
