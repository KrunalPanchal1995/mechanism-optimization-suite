"""
rmip/analysis/benchmarking.py
==============================
Mechanism Performance Benchmarker

Workflow
--------
1. Load experimental IDT / LFS data (via ExpDataReader or from DB).
2. For each mechanism in the comparison set, run Cantera simulations
   at the experimental conditions using IDTRunner / LFSRunner.
3. Compute per-mechanism error metrics vs. experiment:
     MRE      = mean |sim - exp| / exp           (Mean Relative Error)
     RMSE     = sqrt(mean((sim - exp)²))         (in native units)
     log_RMSE = sqrt(mean((log10(sim/exp))²))    (log-space, for IDT)
4. Split by temperature regime (LTC / NTC / HTC) for IDT.
5. Rank mechanisms by MRE per regime.
6. Store results in DB (simulation_runs + mechanism_scores tables).
7. Generate comparison plots and ranking tables.

The benchmarker answers:
  "Which mechanism works best for IDT in the LTC regime?"
  "Which mechanism predicts LFS most accurately at phi=1.0?"

Usage
-----
>>> bench = MechanismBenchmarker("project.db")

>>> # Load experimental data
>>> ds_id = bench.reader.load_csv(
...     "curran1998_idt.csv",
...     dataset_name="Curran1998_nC7H16_IDT",
...     sim_type="IDT", fuel_name="NC7H16",
...     default_units="s", default_P_atm=13.5, default_phi=1.0,
... )

>>> # Run simulations and compute scores (conditions from exp dataset)
>>> scores_df = bench.run_and_score(
...     mechanism_ids=[1, 2, 3],
...     dataset_id=ds_id,
...     fuel="NC7H16:1.0",
...     n_workers=4,
... )
>>> print(scores_df)

>>> # Plot
>>> bench.plot_IDT_comparison([1, 2, 3], ds_id, output_dir="bench_plots/")
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from rmip.database.schema_phase3 import (
    get_exp_data_points,
    get_mechanism_scores,
    get_simulation_runs,
    insert_simulation_run,
    upsert_mechanism_score,
)
from rmip.io.exp_data_reader import ExpDataReader
from rmip.simulation.idt_runner import IDTRunner
from rmip.simulation.lfs_runner import LFSRunner
from rmip.types import SimCondition, SimType, Regime

logger = logging.getLogger(__name__)


class MechanismBenchmarker:
    """
    Benchmarks mechanism performance against experimental IDT / LFS data.

    Parameters
    ----------
    db_path : str
        Path to the RMIP SQLite database.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self.reader   = ExpDataReader(db_path)

    # ------------------------------------------------------------------
    # Primary workflow
    # ------------------------------------------------------------------

    def run_and_score(
        self,
        mechanism_ids: List[int],
        dataset_id:    int,
        fuel:          str,
        oxidizer:      str = "O2:1.0, N2:3.76",
        n_workers:     int = 1,
    ) -> "pd.DataFrame":
        """
        Convenience method: run simulations at experimental conditions and
        compute MRE / RMSE scores.

        Parameters
        ----------
        mechanism_ids : list[int]
            Mechanisms to benchmark.
        dataset_id : int
            Experimental dataset id (from exp_datasets table).
        fuel : str
            Cantera fuel composition string, e.g. ``"NC7H16:1.0"``.
        oxidizer : str
            Cantera oxidizer composition string.
        n_workers : int
            Parallel Cantera workers per mechanism.

        Returns
        -------
        pd.DataFrame  — score table with columns:
            mechanism_id, mechanism_name, dataset_id, regime, n_points,
            mre, rmse, log_rmse, rank_position
        """
        # Load dataset metadata
        ds_info  = self._get_dataset_info(dataset_id)
        sim_type = ds_info["sim_type"]

        # Build simulation conditions from experimental data
        exp_points = self.reader.get_points(dataset_id)
        conditions = [
            SimCondition(
                T_K      = p["T_K"]   or 1000.0,
                P_atm    = p["P_atm"] or 1.0,
                phi      = p["phi"]   or 1.0,
                fuel     = fuel,
                oxidizer = oxidizer,
            )
            for p in exp_points
        ]

        # Run simulations for each mechanism
        all_sim_results: Dict[int, List] = {}
        for mid in mechanism_ids:
            logger.info("Running %s simulations for mechanism %d ...", sim_type, mid)
            if sim_type == SimType.IDT:
                runner  = IDTRunner(mid, self._db_path)
                sim_df  = runner.run_batch(conditions, n_workers=n_workers)
                results = sim_df["idt_s"].tolist()
            else:
                runner  = LFSRunner(mid, self._db_path)
                sim_df  = runner.run_batch(conditions, n_workers=n_workers)
                results = sim_df["lfs_m_s"].tolist()

            all_sim_results[mid] = results

            # Persist to DB
            self._store_sim_results(
                mid, sim_type, conditions, exp_points, results
            )

        # Compute scores
        scores = self._compute_scores_all(
            mechanism_ids, dataset_id, exp_points,
            all_sim_results, sim_type
        )
        return scores

    def run_custom_conditions(
        self,
        mechanism_ids: List[int],
        sim_type:      str,
        conditions:    List[SimCondition],
        n_workers:     int = 1,
    ) -> "pd.DataFrame":
        """
        Run simulations for user-supplied conditions (no exp dataset required).

        Returns a DataFrame with mechanism_id + condition columns + result.
        """
        import pandas as pd

        all_rows = []
        for mid in mechanism_ids:
            logger.info("Running %s for mechanism %d ...", sim_type, mid)
            if sim_type.upper() == "IDT":
                runner = IDTRunner(mid, self._db_path)
                df     = runner.run_batch(conditions, n_workers=n_workers)
            else:
                runner = LFSRunner(mid, self._db_path)
                df     = runner.run_batch(conditions, n_workers=n_workers)
            df["mechanism_id"] = mid
            all_rows.append(df)

        return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

    def get_scores(
        self,
        dataset_id:     int,
        regime:         str = "all",
    ) -> "pd.DataFrame":
        """
        Retrieve stored mechanism scores from the DB.

        Returns
        -------
        pd.DataFrame  sorted by rank_position (ascending) then mre.
        """
        import pandas as pd

        conn = self._connect()
        try:
            rows = get_mechanism_scores(conn, dataset_id, regime)
        finally:
            conn.close()

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_IDT_comparison(
        self,
        mechanism_ids: List[int],
        dataset_id:    int,
        output_dir:    str,
        mech_names:    Optional[Dict[int, str]] = None,
    ) -> None:
        """
        Generate IDT comparison plots (Arrhenius-style: log(IDT) vs 1000/T).

        One plot per (P_atm, phi) combination found in the dataset.
        """
        import matplotlib.pyplot as plt
        import pandas as pd

        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        exp_points = self.reader.get_points(dataset_id)
        names      = mech_names or self._get_mechanism_names(mechanism_ids)

        # Group experimental data by (P_atm, phi)
        groups: Dict[Tuple, List] = {}
        for p in exp_points:
            key = (round(p["P_atm"] or 1.0, 2), round(p["phi"] or 1.0, 2))
            groups.setdefault(key, []).append(p)

        for (P_atm, phi), pts in groups.items():
            fig, ax = plt.subplots(figsize=(7, 5))

            # Experimental data
            T_arr   = [p["T_K"]   for p in pts if p["T_K"]   and p["value"]]
            idt_arr = [p["value"] for p in pts if p["T_K"]   and p["value"]]
            unc_arr = [p["uncertainty"] or 0 for p in pts if p["T_K"] and p["value"]]
            inv_T   = [1000.0 / T for T in T_arr]

            ax.semilogy(inv_T, idt_arr, "ko", ms=6, label="Experiment", zorder=5)
            if any(u > 0 for u in unc_arr):
                ax.fill_between(
                    inv_T,
                    [v - u for v, u in zip(idt_arr, unc_arr)],
                    [v + u for v, u in zip(idt_arr, unc_arr)],
                    color="gray", alpha=0.3, label="Exp. uncertainty",
                )

            # Simulation curves
            colours = plt.cm.tab10.colors
            conn = self._connect()
            try:
                for i, mid in enumerate(mechanism_ids):
                    rows = get_simulation_runs(conn, mid, sim_type="IDT")
                    sim_rows = [
                        r for r in rows
                        if abs((r["P_atm"] or 0) - P_atm) < 0.05
                        and abs((r["phi"]   or 0) - phi)   < 0.05
                        and r["result"] is not None
                    ]
                    if not sim_rows:
                        continue
                    sim_rows.sort(key=lambda r: r["T_K"])
                    T_sim   = [r["T_K"]    for r in sim_rows]
                    idt_sim = [r["result"] for r in sim_rows]
                    ax.semilogy(
                        [1000 / T for T in T_sim], idt_sim,
                        color=colours[i % 10],
                        label=names.get(mid, f"Mech {mid}"),
                    )
            finally:
                conn.close()

            ax.set_xlabel("1000/T  [1/K]", fontsize=12)
            ax.set_ylabel("IDT  [s]",      fontsize=12)
            ax.set_title(f"IDT  —  P={P_atm} atm, φ={phi}", fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, which="both", ls="--", alpha=0.4)

            fname = outdir / f"IDT_P{P_atm:.1f}atm_phi{phi:.2f}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved IDT plot: %s", fname)

    def plot_LFS_comparison(
        self,
        mechanism_ids: List[int],
        dataset_id:    int,
        output_dir:    str,
        mech_names:    Optional[Dict[int, str]] = None,
    ) -> None:
        """
        Generate LFS comparison plots (Su vs phi).
        """
        import matplotlib.pyplot as plt

        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        exp_points = self.reader.get_points(dataset_id)
        names      = mech_names or self._get_mechanism_names(mechanism_ids)

        # Group by (T_K, P_atm)
        groups: Dict[Tuple, List] = {}
        for p in exp_points:
            key = (round(p["T_K"] or 300, 0), round(p["P_atm"] or 1.0, 2))
            groups.setdefault(key, []).append(p)

        for (T_K, P_atm), pts in groups.items():
            fig, ax = plt.subplots(figsize=(7, 5))

            phi_arr = [p["phi"]   for p in pts if p["phi"]  and p["value"]]
            lfs_arr = [p["value"] for p in pts if p["phi"]  and p["value"]]

            ax.plot(phi_arr, [v * 100 for v in lfs_arr],
                    "ko", ms=6, label="Experiment")

            colours = plt.cm.tab10.colors
            conn = self._connect()
            try:
                for i, mid in enumerate(mechanism_ids):
                    rows = get_simulation_runs(conn, mid, sim_type="LFS")
                    sim_rows = [
                        r for r in rows
                        if abs((r["T_K"]   or 0) - T_K)   < 5.0
                        and abs((r["P_atm"] or 0) - P_atm) < 0.05
                        and r["result"] is not None
                    ]
                    if not sim_rows:
                        continue
                    sim_rows.sort(key=lambda r: r["phi"])
                    phi_sim = [r["phi"]    for r in sim_rows]
                    lfs_sim = [r["result"] * 100 for r in sim_rows]
                    ax.plot(phi_sim, lfs_sim,
                            color=colours[i % 10],
                            label=names.get(mid, f"Mech {mid}"))
            finally:
                conn.close()

            ax.set_xlabel("Equivalence ratio φ", fontsize=12)
            ax.set_ylabel("Su  [cm/s]",           fontsize=12)
            ax.set_title(f"LFS  —  T={T_K} K, P={P_atm} atm", fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, ls="--", alpha=0.4)

            fname = outdir / f"LFS_T{T_K:.0f}K_P{P_atm:.1f}atm.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved LFS plot: %s", fname)

    def print_ranking_table(
        self,
        dataset_id:    int,
        regime:        str = "all",
    ) -> None:
        """Print a formatted ranking table to stdout."""
        df = self.get_scores(dataset_id, regime)
        if df.empty:
            print("No scores found for dataset_id=%d, regime=%s" % (dataset_id, regime))
            return

        print(f"\n{'='*70}")
        print(f"  MECHANISM RANKING  |  dataset_id={dataset_id}  |  regime={regime}")
        print(f"{'='*70}")
        print(f"{'Rank':<5} {'Mechanism':<30} {'n_pts':<7} {'MRE%':<10} {'log_RMSE':<10}")
        print(f"{'-'*70}")
        for _, row in df.iterrows():
            mre_pct  = f"{row['mre']*100:.2f}%" if row.get("mre") is not None else "N/A"
            log_r    = f"{row['log_rmse']:.3f}"  if row.get("log_rmse") is not None else "N/A"
            rank     = row.get("rank_position") or "—"
            name     = str(row.get("mechanism_name", f"Mech {row['mechanism_id']}"))[:28]
            print(f"{rank:<5} {name:<30} {row.get('n_points',0):<7} {mre_pct:<10} {log_r:<10}")
        print(f"{'='*70}\n")

    # ------------------------------------------------------------------
    # Score computation
    # ------------------------------------------------------------------

    def _compute_scores_all(
        self,
        mechanism_ids:   List[int],
        dataset_id:      int,
        exp_points:      List[Dict],
        all_sim_results: Dict[int, List],
        sim_type:        str,
    ) -> "pd.DataFrame":
        import pandas as pd

        exp_vals = [p["value"] for p in exp_points]

        all_score_rows = []
        all_regimes_seen: List[Regime] = []
        conn = self._connect()
        try:
            for mid in mechanism_ids:
                sim_vals = all_sim_results.get(mid, [])

                # Prefer the mechanism's own detected NTC onset (Future.md pt. 7:
                # HTT/LTT replaces the fixed LTC/NTC/HTC thresholds) when available;
                # fall back to the fixed 850/1000 K split otherwise. Each mechanism
                # can have its own onset, so regimes are computed per-mechanism.
                regimes, regime_set = self._regimes_for_mechanism(
                    conn, mid, exp_points, sim_type
                )
                all_regimes_seen.extend(regimes)

                for regime in regime_set:
                    if regime == Regime.ALL:
                        idx = list(range(len(exp_vals)))
                    else:
                        idx = [i for i, r in enumerate(regimes) if r == regime]

                    if not idx:
                        continue

                    mre, rmse, log_rmse = self._error_metrics(
                        [exp_vals[i]  for i in idx],
                        [sim_vals[i]  if i < len(sim_vals) else None for i in idx],
                    )
                    rank = upsert_mechanism_score(
                        conn, mid, dataset_id, regime.value,
                        n_points=len(idx),
                        mre=mre, rmse=rmse, log_rmse=log_rmse,
                    )
                    all_score_rows.append({
                        "mechanism_id": mid,
                        "dataset_id":   dataset_id,
                        "regime":       regime.value,
                        "n_points":     len(idx),
                        "mre":          mre,
                        "rmse":         rmse,
                        "log_rmse":     log_rmse,
                    })

            # Compute ranks within each regime (union across all mechanisms,
            # since HTT/LTT-detected mechanisms and fixed-threshold mechanisms
            # may contribute different regime labels to the same dataset).
            self._update_ranks(conn, dataset_id, mechanism_ids, all_regimes_seen, sim_type)
            conn.commit()
        finally:
            conn.close()

        return pd.DataFrame(all_score_rows)

    def _regimes_for_mechanism(
        self,
        conn:       sqlite3.Connection,
        mechanism_id: int,
        exp_points: List[Dict],
        sim_type:   str,
    ) -> Tuple[List[Regime], List[Regime]]:
        """
        Classify each exp point into a regime for one mechanism.

        Returns (per_point_regimes, regime_buckets_to_score). Prefers the
        mechanism's own detected NTC onset (``ntc_runs``, Future.md pt. 7 —
        HTT/LTT) over the fixed 850/1000 K LTC/NTC/HTC thresholds; falls back
        to the fixed thresholds when no NTC run has been stored.
        """
        if sim_type != "IDT":
            return [Regime.ALL] * len(exp_points), [Regime.ALL]

        onset = self._ntc_onset_for_mechanism(conn, mechanism_id)
        if onset is not None:
            regimes = [
                Regime.HTT if (p["T_K"] or 0.0) > onset else Regime.LTT
                for p in exp_points
            ]
            return regimes, [Regime.HTT, Regime.LTT, Regime.ALL]

        regimes = [
            Regime.LTC if (p["T_K"] or 1500) < 850 else
            (Regime.NTC if (p["T_K"] or 1500) < 1000 else Regime.HTC)
            for p in exp_points
        ]
        return regimes, [Regime.LTC, Regime.NTC, Regime.HTC, Regime.ALL]

    @staticmethod
    def _ntc_onset_for_mechanism(
        conn:         sqlite3.Connection,
        mechanism_id: int,
    ) -> Optional[float]:
        """Return the most recent T_NTC_onset for a mechanism, or None."""
        try:
            from rmip.database.schema_phase6 import get_latest_ntc_run
            run = get_latest_ntc_run(conn, mechanism_id)
        except sqlite3.OperationalError:
            return None   # ntc_runs table not migrated into this DB yet
        return run["T_NTC_onset"] if run and run.get("T_NTC_onset") is not None else None

    @staticmethod
    def _error_metrics(
        exp:  List[Optional[float]],
        sim:  List[Optional[float]],
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Compute MRE, RMSE, log_RMSE ignoring None (non-ignition) pairs."""
        pairs = [
            (e, s) for e, s in zip(exp, sim)
            if e is not None and s is not None and e > 0 and s > 0
        ]
        if not pairs:
            return None, None, None

        e_arr = np.array([p[0] for p in pairs])
        s_arr = np.array([p[1] for p in pairs])

        mre      = float(np.mean(np.abs(s_arr - e_arr) / e_arr))
        rmse     = float(np.sqrt(np.mean((s_arr - e_arr) ** 2)))
        log_rmse = float(np.sqrt(np.mean((np.log10(s_arr / e_arr)) ** 2)))
        return mre, rmse, log_rmse

    def _update_ranks(
        self,
        conn:          sqlite3.Connection,
        dataset_id:    int,
        mechanism_ids: List[int],
        regimes:       List[Regime],
        sim_type:      str,
    ) -> None:
        """Assign rank_position to each mechanism within each regime."""
        unique_regimes = set(regimes) | {Regime.ALL}
        for regime in unique_regimes:
            rows = get_mechanism_scores(conn, dataset_id, regime.value)
            rows_with_mre = [(r["mechanism_id"], r["mre"])
                             for r in rows if r.get("mre") is not None]
            rows_with_mre.sort(key=lambda x: x[1])
            for rank, (mid, _) in enumerate(rows_with_mre, start=1):
                conn.execute(
                    """UPDATE mechanism_scores
                       SET rank_position=?
                       WHERE mechanism_id=? AND dataset_id=? AND regime=?""",
                    (rank, mid, dataset_id, regime.value),
                )

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _store_sim_results(
        self,
        mechanism_id: int,
        sim_type:     str,
        conditions:   List[SimCondition],
        exp_points:   List[Dict],
        results:      List[Optional[float]],
    ) -> None:
        """Persist simulation results to simulation_runs table."""
        conn = self._connect()
        try:
            for cond, result in zip(conditions, results):
                fuel_name = cond.fuel.split(":")[0]
                insert_simulation_run(
                    conn,
                    mechanism_id         = mechanism_id,
                    sim_type             = sim_type,
                    T_K                  = cond.T_K,
                    P_atm                = cond.P_atm,
                    phi                  = cond.phi,
                    fuel_name            = fuel_name,
                    fuel_composition     = cond.fuel,
                    oxidizer_composition = cond.oxidizer,
                    result               = result,
                    converged            = result is not None,
                )
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _get_dataset_info(self, dataset_id: int) -> Dict[str, Any]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM exp_datasets WHERE id=?", (dataset_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"No experimental dataset with id={dataset_id}")
            return dict(row)
        finally:
            conn.close()

    def _get_mechanism_names(
        self, mechanism_ids: List[int]
    ) -> Dict[int, str]:
        conn = self._connect()
        try:
            names = {}
            for mid in mechanism_ids:
                row = conn.execute(
                    "SELECT name FROM mechanisms WHERE id=?", (mid,)
                ).fetchone()
                names[mid] = row["name"] if row else f"Mechanism {mid}"
            return names
        finally:
            conn.close()
