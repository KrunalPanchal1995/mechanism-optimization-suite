"""
rmip/simulation/performance_ranker.py
======================================
Ranks N mechanisms by simulated performance vs. experimental data.

What it does
------------
1. Reads simulation_runs from the DB (IDT or LFS) for the requested mechanisms.
2. Reads experimental data points from exp_data_points.
3. Computes Mean Relative Error (MRE) and RMSE per mechanism per regime.
4. Produces a ranked DataFrame and a RankingReport.

Regime definitions (IDT only — LFS is treated as regime ALL):
  LTC : T_K < 850                (fixed-threshold fallback)
  NTC : 850 ≤ T_K < 1000         (fixed-threshold fallback)
  HTC : T_K ≥ 1000               (fixed-threshold fallback)
  HTT : T_K > T_NTC_onset        (Future.md pt. 7 — used instead of LTC/NTC/HTC
  LTT : T_K < T_NTC_onset          whenever the mechanism has a stored NTC run)

MRE formula (per regime)
-------------------------
    MRE = (1/N) Σ |sim_i − exp_i| / exp_i   where exp_i > 0

Matching policy
---------------
A simulation point is matched to an experimental point if:
  |T_sim − T_exp|  < tol_T_K   (default: 20 K)
  |P_sim − P_exp|  < tol_P_atm (default: 0.5 atm)
  |phi_sim − phi_exp| < tol_phi (default: 0.05)

If no experimental point matches a sim point, the point is skipped (not
counted in MRE denominator).

Usage
-----
>>> from rmip.simulation.performance_ranker import PerformanceRanker
>>> ranker = PerformanceRanker("project.db")
>>> report = ranker.rank(
...     mechanism_ids=[1, 2, 3],
...     dataset_id=1,
...     sim_type="IDT",
... )
>>> print(report.summary())
>>> report.save("ranking_report.json")
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("rmip.simulation.performance_ranker")


# ─────────────────────────────────────────────────────────────────────────────
# Regime definitions
# ─────────────────────────────────────────────────────────────────────────────

REGIME_DEFINITIONS: Dict[str, Tuple[float, float]] = {
    "LTC": (0.0,    850.0),
    "NTC": (850.0, 1000.0),
    "HTC": (1000.0, 1e9),
    "ALL": (0.0,    1e9),
}

# All regime labels that can appear in a RankingReport (fixed-threshold set
# plus the detected-onset HTT/LTT set — at most one set is used per mechanism).
_ALL_REGIME_LABELS: Tuple[str, ...] = ("LTC", "NTC", "HTC", "HTT", "LTT", "ALL")


def _classify_regime(T_K: float) -> str:
    if T_K < 850.0:
        return "LTC"
    elif T_K < 1000.0:
        return "NTC"
    else:
        return "HTC"


def _ntc_onset_for_mechanism(conn: sqlite3.Connection, mechanism_id: int) -> Optional[float]:
    """Return the most recent T_NTC_onset for a mechanism, or None (Future.md pt. 7)."""
    try:
        from rmip.database.schema_phase6 import get_latest_ntc_run
        run = get_latest_ntc_run(conn, mechanism_id)
    except sqlite3.OperationalError:
        return None   # ntc_runs table not migrated into this DB yet
    return run["T_NTC_onset"] if run and run.get("T_NTC_onset") is not None else None


def _regimes_and_bounds_for_mechanism(
    conn: sqlite3.Connection, mechanism_id: int, sim_type: str,
) -> Tuple[List[str], Dict[str, Tuple[float, float]]]:
    """
    Return (regime_labels, bounds) for one mechanism. Prefers the detected
    NTC onset (HTT/LTT) over the fixed LTC/NTC/HTC thresholds when a
    ``ntc_runs`` row exists for this mechanism.
    """
    if sim_type != "IDT":
        return ["ALL"], {"ALL": (0.0, 1e9)}

    onset = _ntc_onset_for_mechanism(conn, mechanism_id)
    if onset is not None:
        bounds = {"HTT": (onset, 1e9), "LTT": (0.0, onset), "ALL": (0.0, 1e9)}
        return ["HTT", "LTT", "ALL"], bounds

    return ["LTC", "NTC", "HTC", "ALL"], dict(REGIME_DEFINITIONS)


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RegimeScore:
    """MRE and RMSE for one mechanism in one regime."""
    mechanism_id:  int
    mechanism_name: str
    regime:        str          # "LTC" | "NTC" | "HTC" | "ALL"
    sim_type:      str          # "IDT" | "LFS"
    n_matched:     int          # number of (sim, exp) pairs used
    mre:           Optional[float]   # None if n_matched == 0
    rmse:          Optional[float]   # same
    rank:          Optional[int] = None   # set after sorting all mechanisms

    def to_dict(self) -> dict:
        return {
            "mechanism_id":   self.mechanism_id,
            "mechanism_name": self.mechanism_name,
            "regime":         self.regime,
            "sim_type":       self.sim_type,
            "n_matched":      self.n_matched,
            "mre":            self.mre,
            "rmse":           self.rmse,
            "rank":           self.rank,
        }


@dataclass
class RankingReport:
    """Full ranking output for a single (dataset_id, sim_type) call."""
    dataset_id:     int
    dataset_name:   str
    sim_type:       str
    mechanism_ids:  List[int]
    scores:         List[RegimeScore]    # all regime × mechanism combinations
    created_at:     str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ── Convenience accessors ─────────────────────────────────────────────

    def best_per_regime(self) -> Dict[str, Optional[RegimeScore]]:
        """Return the rank-1 mechanism for each regime actually present in this report."""
        out: Dict[str, Optional[RegimeScore]] = {}
        regimes_present = {s.regime for s in self.scores}
        for regime in _ALL_REGIME_LABELS:
            if regime not in regimes_present:
                continue
            regime_scores = [s for s in self.scores if s.regime == regime and s.rank is not None]
            out[regime] = min(regime_scores, key=lambda s: s.rank) if regime_scores else None
        return out

    def scores_for_regime(self, regime: str) -> List[RegimeScore]:
        return sorted(
            [s for s in self.scores if s.regime == regime],
            key=lambda s: (s.rank is None, s.rank),
        )

    def scores_for_mechanism(self, mech_id: int) -> List[RegimeScore]:
        return [s for s in self.scores if s.mechanism_id == mech_id]

    def summary(self) -> str:
        lines = [
            f"RankingReport: dataset='{self.dataset_name}' sim_type={self.sim_type}",
            f"  Mechanisms: {self.mechanism_ids}",
            "",
        ]
        for regime in _ALL_REGIME_LABELS:
            regime_scores = self.scores_for_regime(regime)
            if not regime_scores:
                continue
            lines.append(f"  [{regime}]")
            for s in regime_scores:
                mre_str  = f"{s.mre*100:.2f}%" if s.mre is not None else "N/A"
                rmse_str = f"{s.rmse:.3E}"     if s.rmse is not None else "N/A"
                lines.append(
                    f"    Rank {s.rank:2d}  mech={s.mechanism_id} ({s.mechanism_name})"
                    f"  MRE={mre_str}  RMSE={rmse_str}  n={s.n_matched}"
                )
            lines.append("")
        return "\n".join(lines)

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        log.info("RankingReport saved to %s", path)

    def to_dict(self) -> dict:
        return {
            "dataset_id":    self.dataset_id,
            "dataset_name":  self.dataset_name,
            "sim_type":      self.sim_type,
            "mechanism_ids": self.mechanism_ids,
            "created_at":    self.created_at,
            "best_per_regime": {
                k: v.to_dict() if v else None
                for k, v in self.best_per_regime().items()
            },
            "scores": [s.to_dict() for s in self.scores],
        }


# ─────────────────────────────────────────────────────────────────────────────
# PerformanceRanker
# ─────────────────────────────────────────────────────────────────────────────

class PerformanceRanker:
    """
    Rank N mechanisms against an experimental dataset.

    Parameters
    ----------
    db_path : str | Path
    tol_T_K : float   Temperature matching tolerance (K). Default: 20.
    tol_P_atm : float Pressure matching tolerance (atm). Default: 0.5.
    tol_phi : float   Equivalence ratio matching tolerance. Default: 0.05.
    """

    def __init__(
        self,
        db_path: str | Path,
        tol_T_K:   float = 20.0,
        tol_P_atm: float = 0.5,
        tol_phi:   float = 0.05,
    ) -> None:
        self._db_path = str(db_path)
        self._tol_T   = tol_T_K
        self._tol_P   = tol_P_atm
        self._tol_phi = tol_phi

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def rank(
        self,
        mechanism_ids: List[int],
        dataset_id:    int,
        sim_type:      str,
    ) -> RankingReport:
        """
        Compute MRE / RMSE for each mechanism in each regime.

        Parameters
        ----------
        mechanism_ids : list[int]
        dataset_id    : int  — id in exp_datasets table
        sim_type      : "IDT" or "LFS"

        Returns
        -------
        RankingReport with per-regime ranking.
        """
        sim_type = sim_type.upper()
        if sim_type not in ("IDT", "LFS"):
            raise ValueError(f"sim_type must be 'IDT' or 'LFS', got '{sim_type}'")

        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            dataset_row = self._load_dataset(conn, dataset_id)
            exp_points  = self._load_exp_points(conn, dataset_id)
            mech_names  = self._load_mech_names(conn, mechanism_ids)
            sim_rows_by_mech = {
                mid: self._load_sim_rows(conn, mid, sim_type)
                for mid in mechanism_ids
            }

        log.info(
            "Ranking %d mechanisms vs. dataset '%s' (%d exp points) sim_type=%s",
            len(mechanism_ids), dataset_row["name"], len(exp_points), sim_type,
        )

        all_scores: List[RegimeScore] = []

        with sqlite3.connect(self._db_path) as conn2:
            conn2.row_factory = sqlite3.Row
            for mid in mechanism_ids:
                sim_rows = sim_rows_by_mech.get(mid, [])
                if not sim_rows:
                    log.warning("No simulation_runs found for mechanism_id=%d sim_type=%s", mid, sim_type)

                # Prefer this mechanism's detected NTC onset (HTT/LTT) over the
                # fixed LTC/NTC/HTC thresholds — Future.md pt. 7.
                regimes, bounds = _regimes_and_bounds_for_mechanism(conn2, mid, sim_type)

                for regime in regimes:
                    score = self._compute_score(
                        mechanism_id=mid,
                        mechanism_name=mech_names.get(mid, f"mech_{mid}"),
                        regime=regime,
                        sim_type=sim_type,
                        sim_rows=sim_rows,
                        exp_points=exp_points,
                        bounds=bounds[regime],
                    )
                    all_scores.append(score)

        # Rank within each regime (lower MRE = better)
        self._assign_ranks(all_scores)

        return RankingReport(
            dataset_id=dataset_id,
            dataset_name=dataset_row["name"],
            sim_type=sim_type,
            mechanism_ids=mechanism_ids,
            scores=all_scores,
        )

    def rank_all_datasets(
        self,
        mechanism_ids: List[int],
        sim_type: str = "IDT",
    ) -> List[RankingReport]:
        """Rank against every exp_dataset in the DB whose sim_type matches."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            datasets = conn.execute(
                "SELECT id FROM exp_datasets WHERE sim_type=?", (sim_type.upper(),)
            ).fetchall()
        return [
            self.rank(mechanism_ids, ds["id"], sim_type)
            for ds in datasets
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # Internal — scoring
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_score(
        self,
        mechanism_id:   int,
        mechanism_name: str,
        regime:         str,
        sim_type:       str,
        sim_rows:       List[dict],
        exp_points:     List[dict],
        bounds:         Optional[Tuple[float, float]] = None,
    ) -> RegimeScore:
        """Match sim rows to exp points and compute MRE/RMSE for one regime."""
        T_lo, T_hi = bounds if bounds is not None else REGIME_DEFINITIONS[regime]
        # Filter sim rows to this regime
        sim_in_regime = [
            r for r in sim_rows
            if T_lo <= r["T_K"] < T_hi and r["converged"] and r["result"] is not None
        ]
        # Filter exp points to this regime
        exp_in_regime = [p for p in exp_points if T_lo <= p["T_K"] < T_hi]

        if not sim_in_regime or not exp_in_regime:
            return RegimeScore(
                mechanism_id=mechanism_id,
                mechanism_name=mechanism_name,
                regime=regime,
                sim_type=sim_type,
                n_matched=0,
                mre=None,
                rmse=None,
            )

        rel_errors: List[float] = []
        sq_errors:  List[float] = []

        for exp in exp_in_regime:
            sim_match = self._find_best_match(exp, sim_in_regime)
            if sim_match is None:
                continue
            exp_val = exp["value"]
            sim_val = sim_match["result"]
            if exp_val == 0.0:
                continue
            rel_err = abs(sim_val - exp_val) / abs(exp_val)
            rel_errors.append(rel_err)
            sq_errors.append((sim_val - exp_val) ** 2)

        n = len(rel_errors)
        if n == 0:
            return RegimeScore(
                mechanism_id=mechanism_id,
                mechanism_name=mechanism_name,
                regime=regime,
                sim_type=sim_type,
                n_matched=0,
                mre=None,
                rmse=None,
            )

        mre  = sum(rel_errors) / n
        rmse = math.sqrt(sum(sq_errors) / n)
        return RegimeScore(
            mechanism_id=mechanism_id,
            mechanism_name=mechanism_name,
            regime=regime,
            sim_type=sim_type,
            n_matched=n,
            mre=mre,
            rmse=rmse,
        )

    def _find_best_match(
        self,
        exp: dict,
        sim_rows: List[dict],
    ) -> Optional[dict]:
        """Find closest simulation point to this experimental point."""
        best = None
        best_dist = float("inf")
        for s in sim_rows:
            if (
                abs(s["T_K"]   - exp["T_K"])   < self._tol_T
                and abs(s["P_atm"] - exp["P_atm"]) < self._tol_P
                and abs(s["phi"]   - exp["phi"])   < self._tol_phi
            ):
                dist = abs(s["T_K"] - exp["T_K"])
                if dist < best_dist:
                    best_dist = dist
                    best = s
        return best

    @staticmethod
    def _assign_ranks(scores: List[RegimeScore]) -> None:
        """In-place: assign rank within each regime (lower MRE = rank 1)."""
        from itertools import groupby
        # Group by regime
        groups: Dict[str, List[RegimeScore]] = {}
        for s in scores:
            groups.setdefault(s.regime, []).append(s)

        for regime, group in groups.items():
            # Only rank those with actual MRE values
            has_mre = [s for s in group if s.mre is not None]
            no_mre  = [s for s in group if s.mre is None]
            sorted_group = sorted(has_mre, key=lambda s: s.mre)
            for rank, score in enumerate(sorted_group, start=1):
                score.rank = rank
            for score in no_mre:
                score.rank = len(has_mre) + 1   # unranked last

    # ─────────────────────────────────────────────────────────────────────────
    # Internal — DB loading
    # ─────────────────────────────────────────────────────────────────────────

    def _load_dataset(self, conn: sqlite3.Connection, dataset_id: int) -> dict:
        row = conn.execute(
            "SELECT * FROM exp_datasets WHERE id=?", (dataset_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"exp_dataset id={dataset_id} not found")
        return dict(row)

    def _load_exp_points(self, conn: sqlite3.Connection, dataset_id: int) -> List[dict]:
        rows = conn.execute(
            "SELECT * FROM exp_data_points WHERE dataset_id=?", (dataset_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def _load_mech_names(
        self, conn: sqlite3.Connection, mech_ids: List[int]
    ) -> Dict[int, str]:
        placeholders = ",".join("?" * len(mech_ids))
        rows = conn.execute(
            f"SELECT id, name FROM mechanisms WHERE id IN ({placeholders})", mech_ids
        ).fetchall()
        return {row["id"]: row["name"] for row in rows}

    def _load_sim_rows(
        self, conn: sqlite3.Connection, mech_id: int, sim_type: str
    ) -> List[dict]:
        rows = conn.execute(
            """
            SELECT T_K, P_atm, phi, result, converged
              FROM simulation_runs
             WHERE mechanism_id=? AND sim_type=?
            """,
            (mech_id, sim_type),
        ).fetchall()
        return [dict(r) for r in rows]
