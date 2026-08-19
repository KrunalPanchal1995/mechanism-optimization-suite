"""
rmip/optimization/mso_interface.py
===================================
Module: MSO (Multi-Stage Optimization) Interface

Design decision D7
------------------
RMIP does NOT import OptimizationTool_v4 directly.
The interface is a JSON file contract:

  Step 1.  MSOInterface.prepare()  → writes  <work_dir>/mso_input.json
  Step 2.  User (or MSOInterface.run()) calls KP's OptimizationTool_v4.py
           as a subprocess.  The tool reads mso_input.json and writes
           <work_dir>/mso_output/optimized_params.json
  Step 3.  MSOInterface.apply()  → reads the output JSON, updates the
           fused mechanism in the RMIP DB, and returns an OptimizationResult.

mso_input.json schema (consumed by OptimizationTool_v4.py)
------------------------------------------------------------
{
  "session_id":         int,
  "fused_mechanism_id": int,
  "fuel_name":          str,
  "n_fuse_reactions":   int,
  "fuse_blocks": [
    {
      "reaction_index":  int,       // 0-based, for OptimizationTool indexing
      "canonical_key":   str,
      "formula_string":  str,
      "params_mean":     [A_si, n, Ea_si],
      "params_lb":       [A_min, n_min, Ea_min],
      "params_ub":       [A_max, n_max, Ea_max],
      "param_names":     ["A_si", "n", "Ea_si"],
      "units":           ["m^3/mol/s", "1", "J/mol"]
    }, ...
  ],
  "target_conditions": [
    {"T_K": 1100, "P_atm": 13.5, "phi": 1.0, "fuel": "NC7H16:1.0",
     "oxidizer": "O2:1.0, N2:3.76", "observable": "IDT",
     "exp_value": 2.3e-4, "weight": 1.0}, ...
  ],
  "optimization_settings": {
    "objective":          "WMRE",       // or "logRMSE"
    "n_stages":           2,
    "ga_num_generations": 500,
    "ga_population_size": 50,
    "lbfgsb_maxiter":     200,
    "random_seed":        42
  }
}

mso_output/optimized_params.json schema (written by OptimizationTool_v4.py)
----------------------------------------------------------------------------
{
  "session_id":     int,
  "status":         "success" | "failed",
  "objective_final": float,
  "fuse_blocks": [
    {
      "reaction_index":    int,
      "canonical_key":     str,
      "optimized_params":  [A_si, n, Ea_si]
    }, ...
  ]
}

Usage
-----
>>> from rmip.optimization.mso_interface import MSOInterface
>>> msi = MSOInterface("project.db", work_dir="mso_run_001/")
>>> # Prepare from FusionResult + target conditions
>>> msi.prepare(fusion_result, target_conditions, mso_bounds_path="fusion_params.json")
>>> # Run (calls OptimizationTool_v4.py as subprocess)
>>> opt_result = msi.run(tool_path="path/to/OptimizationTool_v4.py")
>>> # Or, if user ran the tool externally:
>>> opt_result = msi.apply()
>>> print(opt_result.objective_final)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("rmip.optimization.mso_interface")


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OptimizedReaction:
    """A single reaction's optimised parameters from MSO output."""
    reaction_index: int
    canonical_key:  str
    A_si_opt:       float
    n_opt:          float
    Ea_si_opt:      float

    # Change vs. pre-optimisation mean
    A_si_prior:  Optional[float] = None
    n_prior:     Optional[float] = None
    Ea_si_prior: Optional[float] = None

    @property
    def delta_logA(self) -> Optional[float]:
        import math
        if self.A_si_prior and self.A_si_opt:
            return math.log10(self.A_si_opt) - math.log10(self.A_si_prior)
        return None

    def to_dict(self) -> dict:
        return {
            "reaction_index":  self.reaction_index,
            "canonical_key":   self.canonical_key,
            "A_si_opt":        self.A_si_opt,
            "n_opt":           self.n_opt,
            "Ea_si_opt":       self.Ea_si_opt,
            "A_si_prior":      self.A_si_prior,
            "n_prior":         self.n_prior,
            "Ea_si_prior":     self.Ea_si_prior,
            "delta_logA":      self.delta_logA,
        }


@dataclass
class OptimizationResult:
    """Full output of an MSO run."""
    session_id:      int
    status:          str          # "success" | "failed"
    objective_final: Optional[float]
    fused_mechanism_id: int
    optimized_mech_id:  Optional[int]  # new DB entry for optimised mechanism
    reactions:       List[OptimizedReaction]
    runtime_s:       Optional[float] = None
    error_msg:       Optional[str]   = None

    def to_dict(self) -> dict:
        return {
            "session_id":         self.session_id,
            "status":             self.status,
            "objective_final":    self.objective_final,
            "fused_mechanism_id": self.fused_mechanism_id,
            "optimized_mech_id":  self.optimized_mech_id,
            "runtime_s":          self.runtime_s,
            "error_msg":          self.error_msg,
            "reactions":          [r.to_dict() for r in self.reactions],
        }

    def summary(self) -> str:
        lines = [
            f"OptimizationResult: status={self.status}",
            f"  Session id        : {self.session_id}",
            f"  Fused mech id     : {self.fused_mechanism_id}",
            f"  Optimised mech id : {self.optimized_mech_id}",
            f"  Objective (final) : {self.objective_final}",
            f"  # reactions opt'd : {len(self.reactions)}",
            f"  Runtime           : {self.runtime_s:.1f} s" if self.runtime_s else "",
        ]
        return "\n".join(l for l in lines if l)


# ─────────────────────────────────────────────────────────────────────────────
# MSOInterface
# ─────────────────────────────────────────────────────────────────────────────

class MSOInterface:
    """
    Bridges RMIP fusion results to KP's OptimizationTool_v4.py.

    Parameters
    ----------
    db_path   : str | Path — RMIP SQLite database
    work_dir  : str | Path — directory for mso_input.json, mso_output/
    settings  : dict       — override default optimization settings
    """

    DEFAULT_SETTINGS: Dict[str, Any] = {
        "objective":          "WMRE",
        "n_stages":           2,
        "ga_num_generations": 500,
        "ga_population_size": 50,
        "lbfgsb_maxiter":     200,
        "random_seed":        42,
    }

    def __init__(
        self,
        db_path:  str | Path,
        work_dir: str | Path = "mso_run",
        settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._db_path = str(db_path)
        self._work_dir = Path(work_dir)
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._settings = {**self.DEFAULT_SETTINGS, **(settings or {})}
        self._session_id = int(datetime.now(timezone.utc).timestamp())

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Prepare — write mso_input.json
    # ─────────────────────────────────────────────────────────────────────────

    def prepare(
        self,
        fused_mechanism_id: int,
        mso_bounds_path:    str,
        target_conditions:  List[Dict[str, Any]],
        settings_override:  Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Write mso_input.json from the MSO bounds file (fusion_params.json) and
        a list of target conditions with experimental values.

        Parameters
        ----------
        fused_mechanism_id : int
            DB id of the fused mechanism (written by FusionEngine).
        mso_bounds_path : str
            Path to the JSON written by FusionEngine.export_mso_bounds().
        target_conditions : list[dict]
            Each dict must have:
              T_K, P_atm, phi, fuel, oxidizer, observable, exp_value
            Optional: weight (default 1.0)
        settings_override : dict | None
            Override any key in self._settings for this run.

        Returns
        -------
        Path to the written mso_input.json.
        """
        with open(mso_bounds_path) as fh:
            bounds_data = json.load(fh)

        settings = {**self._settings, **(settings_override or {})}

        # Validate target_conditions
        required_keys = {"T_K", "P_atm", "phi", "fuel", "oxidizer", "observable", "exp_value"}
        for i, tc in enumerate(target_conditions):
            missing = required_keys - set(tc.keys())
            if missing:
                raise ValueError(f"target_conditions[{i}] missing keys: {missing}")
            tc.setdefault("weight", 1.0)

        mso_input = {
            "session_id":         self._session_id,
            "fused_mechanism_id": fused_mechanism_id,
            "fuel_name":          bounds_data.get("fuel_name", ""),
            "n_fuse_reactions":   bounds_data.get("n_fuse_reactions", 0),
            "fuse_blocks": [
                {**block, "reaction_index": i}
                for i, block in enumerate(bounds_data.get("fuse_blocks", []))
            ],
            "target_conditions":    target_conditions,
            "optimization_settings": settings,
        }

        out_path = self._work_dir / "mso_input.json"
        with open(out_path, "w") as fh:
            json.dump(mso_input, fh, indent=2)

        log.info(
            "mso_input.json written: %d FUSE reactions, %d target conditions → %s",
            mso_input["n_fuse_reactions"], len(target_conditions), out_path,
        )
        return out_path

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Run — subprocess call to OptimizationTool_v4.py
    # ─────────────────────────────────────────────────────────────────────────

    def run(
        self,
        tool_path: str,
        python_exe: str = sys.executable,
        timeout_s: Optional[float] = None,
        extra_args: Optional[List[str]] = None,
    ) -> "OptimizationResult":
        """
        Call KP's OptimizationTool_v4.py as a subprocess.

        The tool is expected to:
          - read  <work_dir>/mso_input.json
          - write <work_dir>/mso_output/optimized_params.json

        If the subprocess exits non-zero or times out, an OptimizationResult
        with status='failed' is returned (not an exception), so the caller can
        still call apply() on a partial output if one exists.

        Parameters
        ----------
        tool_path   : str  — path to OptimizationTool_v4.py (or equivalent)
        python_exe  : str  — Python interpreter to use (default: current)
        timeout_s   : float | None — wall-clock timeout in seconds
        extra_args  : list[str] | None — additional CLI args passed to the tool

        Returns
        -------
        OptimizationResult (status='success' or 'failed')
        """
        import time
        input_path = self._work_dir / "mso_input.json"
        if not input_path.exists():
            raise FileNotFoundError(
                f"mso_input.json not found at {input_path}. Call prepare() first."
            )

        cmd = [
            python_exe,
            str(tool_path),
            "--input",   str(input_path),
            "--work-dir", str(self._work_dir),
        ]
        if extra_args:
            cmd.extend(extra_args)

        log.info("Launching MSO tool: %s", " ".join(cmd))
        t0 = time.perf_counter()

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            runtime_s = time.perf_counter() - t0

            if proc.returncode != 0:
                log.error("MSO tool exited with code %d:\n%s", proc.returncode, proc.stderr)
                return OptimizationResult(
                    session_id=self._session_id,
                    status="failed",
                    objective_final=None,
                    fused_mechanism_id=self._read_fused_mech_id(),
                    optimized_mech_id=None,
                    reactions=[],
                    runtime_s=runtime_s,
                    error_msg=f"Process exited {proc.returncode}:\n{proc.stderr[:2000]}",
                )
        except subprocess.TimeoutExpired as exc:
            log.error("MSO tool timed out after %.0f s", timeout_s)
            return OptimizationResult(
                session_id=self._session_id,
                status="failed",
                objective_final=None,
                fused_mechanism_id=self._read_fused_mech_id(),
                optimized_mech_id=None,
                reactions=[],
                runtime_s=timeout_s,
                error_msg=f"Timeout after {timeout_s} s",
            )

        log.info("MSO tool completed in %.1f s", runtime_s)
        return self.apply(runtime_s=runtime_s)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: Apply — read output, update DB, return result
    # ─────────────────────────────────────────────────────────────────────────

    def apply(self, runtime_s: Optional[float] = None) -> OptimizationResult:
        """
        Read mso_output/optimized_params.json and apply the optimised
        parameters to a new mechanism DB entry (a copy of the fused mechanism
        with updated A/n/Ea for FUSE reactions).

        Returns
        -------
        OptimizationResult
        """
        output_path = self._work_dir / "mso_output" / "optimized_params.json"
        if not output_path.exists():
            raise FileNotFoundError(
                f"MSO output not found: {output_path}\n"
                "Run the optimization tool first (via run() or externally)."
            )

        with open(output_path) as fh:
            out_data = json.load(fh)

        fused_mech_id = self._read_fused_mech_id()
        status        = out_data.get("status", "failed")
        obj_final     = out_data.get("objective_final")

        if status != "success":
            return OptimizationResult(
                session_id=self._session_id,
                status=status,
                objective_final=obj_final,
                fused_mechanism_id=fused_mech_id,
                optimized_mech_id=None,
                reactions=[],
                runtime_s=runtime_s,
                error_msg=out_data.get("error_msg"),
            )

        # Load prior params from mso_input.json for delta computation
        prior_map = self._load_prior_params()

        # Build OptimizedReaction list
        opt_reactions: List[OptimizedReaction] = []
        for block in out_data.get("fuse_blocks", []):
            ck  = block["canonical_key"]
            idx = block["reaction_index"]
            prm = block["optimized_params"]  # [A_si, n, Ea_si]
            prior = prior_map.get(ck, {})
            opt_reactions.append(OptimizedReaction(
                reaction_index=idx,
                canonical_key=ck,
                A_si_opt=prm[0],
                n_opt=prm[1],
                Ea_si_opt=prm[2],
                A_si_prior=prior.get("A_si"),
                n_prior=prior.get("n"),
                Ea_si_prior=prior.get("Ea_si"),
            ))

        # Clone fused mechanism into a new "optimized" mechanism row
        opt_mech_id = self._create_optimized_mechanism(fused_mech_id, opt_reactions)

        result = OptimizationResult(
            session_id=self._session_id,
            status=status,
            objective_final=obj_final,
            fused_mechanism_id=fused_mech_id,
            optimized_mech_id=opt_mech_id,
            reactions=opt_reactions,
            runtime_s=runtime_s,
        )

        # Persist result summary
        result_path = self._work_dir / "optimization_result.json"
        with open(result_path, "w") as fh:
            json.dump(result.to_dict(), fh, indent=2)
        log.info("Optimization result saved to %s", result_path)
        log.info(result.summary())

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _read_fused_mech_id(self) -> int:
        """Read fused_mechanism_id from mso_input.json if it exists."""
        p = self._work_dir / "mso_input.json"
        if p.exists():
            with open(p) as fh:
                return json.load(fh).get("fused_mechanism_id", -1)
        return -1

    def _load_prior_params(self) -> Dict[str, Dict[str, float]]:
        """Return {canonical_key: {A_si, n, Ea_si}} from mso_input.json fuse_blocks."""
        p = self._work_dir / "mso_input.json"
        if not p.exists():
            return {}
        with open(p) as fh:
            data = json.load(fh)
        out: Dict[str, Dict[str, float]] = {}
        for block in data.get("fuse_blocks", []):
            ck = block["canonical_key"]
            pm = block.get("params_mean", [None, None, None])
            out[ck] = {"A_si": pm[0], "n": pm[1], "Ea_si": pm[2]}
        return out

    def _create_optimized_mechanism(
        self,
        fused_mech_id: int,
        opt_reactions: List[OptimizedReaction],
    ) -> int:
        """
        Clone the fused mechanism into a new DB entry with updated kinetics
        for the FUSE reactions.  Returns new mechanism_id.
        """
        opt_map = {r.canonical_key: r for r in opt_reactions}
        ts = datetime.now(timezone.utc).isoformat()

        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Fetch fused mechanism metadata
            fused_row = conn.execute(
                "SELECT * FROM mechanisms WHERE id=?", (fused_mech_id,)
            ).fetchone()
            if fused_row is None:
                raise ValueError(f"Fused mechanism id={fused_mech_id} not found in DB")

            fused_meta = json.loads(fused_row["metadata_json"] or "{}")
            new_meta = {
                **fused_meta,
                "optimized_from":   fused_mech_id,
                "optimization_session": self._session_id,
                "optimized_at":     ts,
            }

            # Create new mechanism row
            cur = conn.execute(
                """
                INSERT INTO mechanisms
                    (name, format, filepath, description,
                     n_species, n_reactions, loaded_at, metadata_json)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    f"{fused_row['name']}_MSO_opt",
                    "optimized",
                    None,
                    f"MSO-optimized from mechanism id={fused_mech_id}",
                    fused_row["n_species"],
                    fused_row["n_reactions"],
                    ts,
                    json.dumps(new_meta),
                ),
            )
            new_mech_id = cur.lastrowid

            # Copy species rows verbatim
            species_rows = conn.execute(
                "SELECT * FROM species WHERE mechanism_id=?", (fused_mech_id,)
            ).fetchall()
            old_new_sp: Dict[int, int] = {}
            for sp in species_rows:
                sp_dict = dict(sp)
                old_id = sp_dict.pop("id")
                sp_dict["mechanism_id"] = new_mech_id
                cols = list(sp_dict.keys())
                vals = [sp_dict[c] for c in cols]
                cur2 = conn.execute(
                    f"INSERT INTO species ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})",
                    vals,
                )
                old_new_sp[old_id] = cur2.lastrowid

            # Copy reaction rows, updating optimised reactions
            rxn_rows = conn.execute(
                "SELECT * FROM reactions WHERE mechanism_id=?", (fused_mech_id,)
            ).fetchall()
            old_new_rxn: Dict[int, int] = {}
            for rxn in rxn_rows:
                rxn_dict = dict(rxn)
                old_rxn_id = rxn_dict.pop("id")
                rxn_dict["mechanism_id"] = new_mech_id

                ck = rxn_dict.get("canonical_key") or ""
                if ck in opt_map:
                    opt_r = opt_map[ck]
                    rxn_dict["A_si"]    = opt_r.A_si_opt
                    rxn_dict["A_orig"]  = opt_r.A_si_opt
                    rxn_dict["n"]       = opt_r.n_opt
                    rxn_dict["Ea_si"]   = opt_r.Ea_si_opt
                    rxn_dict["Ea_orig"] = opt_r.Ea_si_opt
                    rxn_dict["rmg_comment"] = (
                        f"MSO-optimised (session {self._session_id})"
                    )

                cols = list(rxn_dict.keys())
                vals = [rxn_dict[c] for c in cols]
                cur3 = conn.execute(
                    f"INSERT INTO reactions ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})",
                    vals,
                )
                old_new_rxn[old_rxn_id] = cur3.lastrowid

            # Copy stoichiometry with remapped ids
            stoich_rows = conn.execute(
                """
                SELECT s.*
                  FROM stoichiometry s
                  JOIN reactions r ON r.id = s.reaction_id
                 WHERE r.mechanism_id = ?
                """,
                (fused_mech_id,),
            ).fetchall()
            for srow in stoich_rows:
                new_rxn_id = old_new_rxn.get(srow["reaction_id"])
                new_sp_id  = old_new_sp.get(srow["species_id"])
                if new_rxn_id and new_sp_id:
                    conn.execute(
                        "INSERT INTO stoichiometry (reaction_id, species_id, stoich_coeff, role) VALUES (?,?,?,?)",
                        (new_rxn_id, new_sp_id, srow["stoich_coeff"], srow["role"]),
                    )

            conn.commit()

        log.info(
            "Optimized mechanism written to DB: id=%d ('%s_MSO_opt')",
            new_mech_id, fused_row["name"],
        )
        return new_mech_id
