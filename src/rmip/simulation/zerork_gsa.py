"""
rmip/simulation/zerork_gsa.py
==============================
Phase 3 of MarkDownFiles/ZERORK_GSA_PLAN.md -- real Sobol/Morris GSA:
SALib sampling + index computation, executed via Zero-RK's
``perturbAFactorGSA_mpi.x`` MPI batch runner (which itself only does raw
parallel IDT sampling, never computes an index -- see the plan's section
1.1.2). This module fills that gap.

Scope decision (confirmed with KP, 2026-07-12): operates standalone on a
single ``mechanism_id`` + selected reaction A-factors, NOT yet wired to
``rmip.stubs.DoEFramework.run_screening()``'s documented
``UnionGraphSpec.fuse_params`` interface. That interface targets a merged
mechanism produced by ``FusionEngine`` (M7), which is itself an
unimplemented stub in this codebase -- wiring ``run_screening()`` to
delegate here is a small follow-up once M7 exists, not attempted now.
``SobolResult.to_screening_dict()`` already matches ``run_screening()``'s
documented per-parameter return shape for that future hookup.

Subprocess boundaries (D7 principle, same as ``zerork_bridge.py``/
``muqsac_bridge.py``: never import the external tool, only call it):

1. Zero-RK's ``perturbAFactorGSA_mpi.x``, invoked via ``/usr/bin/mpiexec``
   (NOT anaconda's ``mpirun``, which is a different, incompatible MPI
   implementation -- Zero-RK was built against the system OpenMPI via
   ``/usr/bin/mpicc``, see ZERORK_GSA_PLAN.md Phase 0).
2. SALib, invoked via a dedicated venv
   (``/home/krithika/.venvs/salib_gsa``), not imported directly. SALib
   1.5.2 requires numpy>=2.0/scipy>=1.9.3/pandas>=2.0; this repo's shared
   anaconda environment is pinned to numpy 1.24.4 for scikit-learn/numba
   compatibility. Installing SALib into that shared environment once
   upgraded numpy to 2.0.2 and broke scikit-learn's import outright (a
   real incident during this work, immediately rolled back) -- this
   subprocess boundary exists specifically so that can't happen again, not
   as a hypothetical precaution. See ``_salib_backend.py`` for the
   request/response JSON contract.

Zero-RK's ``gsaMatrixFile`` format (confirmed from
``examples/perturbAFactorGSA/create_lognormal_mtx.py``): a header line
``"{n_samples:6d} {n_reactions:6d}\\n"``, then ONE value per line (not one
row per line), row-major -- ``n_reactions`` MUST equal the mechanism's
total reaction count exactly (the app cross-checks this against the parsed
``mechFile``), so reactions not in the caller's selected subset are still
written as columns, held at multiplier 1.0 for every sample.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from rmip.simulation.subprocess_util import resolve_timeout, sanitized_subprocess_env
from rmip.types import GSARunConfig, SobolResult, SimCondition

log = logging.getLogger("rmip.simulation.zerork_gsa")

_SALIB_PYTHON_DEFAULT = "/home/krithika/.venvs/salib_gsa/bin/python3"
_MPIEXEC_DEFAULT = "/usr/bin/mpiexec"
_MAX_FAILED_SAMPLE_FRACTION = 0.20


class ZerorkGSARunner:
    """
    Sobol/Morris screening over a mechanism's reaction A-factors.

    Parameters
    ----------
    db_path : str
        RMIP SQLite database path.
    zerork_install_dir : str
        Zero-RK install prefix (must contain ``bin/perturbAFactorGSA_mpi.x``).
    salib_python : str, optional
        Path to the dedicated SALib venv's python3. Defaults to
        ``/home/krithika/.venvs/salib_gsa/bin/python3``.
    mpiexec : str, optional
        Path to the ``mpiexec`` matching the MPI Zero-RK was built against.
        Defaults to ``/usr/bin/mpiexec`` (system OpenMPI).
    """

    def __init__(
        self,
        db_path: str,
        zerork_install_dir: str,
        salib_python: Optional[str] = None,
        mpiexec: Optional[str] = None,
    ) -> None:
        from rmip.simulation.zerork_bridge import ZerorkBridge

        self._db = db_path
        self._bridge = ZerorkBridge(db_path, zerork_install_dir)
        self._bin_dir = Path(zerork_install_dir) / "bin"
        exe = self._bin_dir / "perturbAFactorGSA_mpi.x"
        if not exe.exists():
            raise FileNotFoundError(f"perturbAFactorGSA_mpi.x not found at {exe}")
        self._salib_python = salib_python or _SALIB_PYTHON_DEFAULT
        self._mpiexec = mpiexec or _MPIEXEC_DEFAULT
        self._backend_script = str(Path(__file__).parent / "_salib_backend.py")

    # ------------------------------------------------------------------
    # Reaction enumeration
    # ------------------------------------------------------------------

    def _all_reactions(self, mechanism_id: int) -> List[Tuple[int, str]]:
        """
        ``[(db_id, "R3:O + H2 <=> H + OH"), ...]`` ordered by id.

        Must match ``ChemkinWriter``'s reaction file order exactly (both
        query ``ORDER BY id`` -- confirmed by direct inspection) since
        Zero-RK's ``gsaMatrixFile`` columns are positional, not name-keyed.
        ``reaction_label`` (``R1``, ``R2``, ...) guarantees a unique name
        even for mechanisms with duplicate ``formula_string`` entries
        (DUPLICATE-tagged reactions); ``formula_string`` is appended for
        human readability in the persisted results.
        """
        with sqlite3.connect(self._db) as conn:
            rows = conn.execute(
                "SELECT id, reaction_label, formula_string FROM reactions "
                "WHERE mechanism_id=? ORDER BY id",
                (mechanism_id,),
            ).fetchall()
        if not rows:
            raise ValueError(f"No reactions found for mechanism_id={mechanism_id}")
        return [(rid, f"{label}:{formula}") for rid, label, formula in rows]

    # ------------------------------------------------------------------
    # SALib subprocess calls (see _salib_backend.py for the contract)
    # ------------------------------------------------------------------

    def _salib_call(self, cmd: str, request: dict, work_dir: Path) -> dict:
        req_path = work_dir / f"salib_{cmd}_request.json"
        resp_path = work_dir / f"salib_{cmd}_response.json"
        # M19: remove any stale response from a previous call so that, if the
        # backend exits 0 without writing a fresh one, we fail loudly rather
        # than silently returning last run's SALib result.
        if resp_path.exists():
            resp_path.unlink()
        with open(req_path, "w") as fh:
            json.dump(request, fh)
        proc = subprocess.run(
            [self._salib_python, self._backend_script, cmd, str(req_path), str(resp_path)],
            capture_output=True, text=True,
            timeout=resolve_timeout(None), env=sanitized_subprocess_env(),  # M18b
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"SALib backend {cmd!r} failed (rc={proc.returncode}):\n{proc.stderr[-4000:]}"
            )
        if not resp_path.exists():
            raise RuntimeError(
                f"SALib backend {cmd!r} exited 0 but wrote no response at {resp_path}"
            )
        with open(resp_path) as fh:
            return json.load(fh)

    # ------------------------------------------------------------------
    # gsaMatrixFile
    # ------------------------------------------------------------------

    @staticmethod
    def _write_gsa_matrix(path: Path, matrix: List[List[float]], n_reactions: int) -> None:
        n_samples = len(matrix)
        with open(path, "w") as fh:
            fh.write("{0:6d} {1:6d}\n".format(n_samples, n_reactions))
            for row in matrix:
                for value in row:
                    fh.write("{0:24.16e}\n".format(value))

    @staticmethod
    def _build_full_matrix(
        X_selected: List[List[float]],
        all_reaction_ids: List[int],
        selected_reaction_ids: List[int],
    ) -> List[List[float]]:
        """
        Expand a SALib sample matrix (n_rows x n_selected, log-multipliers)
        to Zero-RK's full-mechanism-width matrix (n_rows x n_all_reactions,
        actual multipliers = exp(log-multiplier)). Reactions outside the
        caller's selection are held at multiplier 1.0 for every sample.
        """
        idx_of = {rid: i for i, rid in enumerate(selected_reaction_ids)}
        full: List[List[float]] = []
        for row in X_selected:
            full_row = [1.0] * len(all_reaction_ids)
            for j, rid in enumerate(all_reaction_ids):
                sel_idx = idx_of.get(rid)
                if sel_idx is not None:
                    full_row[j] = math.exp(row[sel_idx])
            full.append(full_row)
        return full

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_gsa_output(path: Path, n_samples: int) -> List[float]:
        """
        Parse ``perturbAFactorGSA_mpi.x``'s ``outFile``: comment lines start
        with ``#``, data rows are ``task_id  cpu_time[s]  idt_col_1
        [idt_col_2 ...]``. This runner always requests exactly one
        ``idtTemps`` threshold, so there is exactly one IDT column (index
        2). A non-ignition/failed sample is written as a non-positive
        value -- mapped to NaN here, mean-imputed by the caller before
        handing Y to SALib's analyzer (SALib can't handle NaN).

        task_id offset (found via ZERORK_GSA_PLAN.md Phase 4's cross-check
        against AnalyticalSobolPRS, confirmed by reading
        applications/perturbAFactorGSA/perturbAFactor_mpi.cpp directly):
        task_id=0 is always the app's own unperturbed-baseline run, written
        BEFORE the sample loop ("report the unperturbed IDT calculation as
        task 0") -- it is never one of the gsaMatrixFile's ``n_samples``
        rows. Row ``j`` (0-indexed, matching the input matrix / SALib
        sample order) is written as ``task_id = j+1``
        (``fprintf(outFilePtr, "%8d ...", j+1, ...)`` inside the
        ``j=0..nTask-1`` loop). Reading task_id directly as the row index
        (the original code here) silently shifted every sample's Y by one
        row relative to X -- corrupting the Saltelli pick-freeze pairing for
        every Sobol/Morris index this module has ever computed, discovered
        only because Phase 4's independent PRS-surrogate comparison
        disagreed sharply with the direct-simulation indices for two
        genuinely near-zero-sensitivity reactions until this offset was
        applied (see gsa_prs_crosscheck.py / test_gsa_prs_crosscheck.py).
        """
        rows: Dict[int, float] = {}
        with open(path) as fh:
            for line in fh:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                fields = s.split()
                task_id = int(float(fields[0]))
                idt_s = float(fields[2])
                rows[task_id] = idt_s if (math.isfinite(idt_s) and idt_s > 0) else float("nan")
        return [rows.get(i + 1, float("nan")) for i in range(n_samples)]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_screening(
        self,
        config: GSARunConfig,
        work_dir: str,
        timeout: Optional[float] = None,
        save_db: bool = True,
    ) -> SobolResult:
        # Resolve to absolute up front (same fix pattern as B13/forward_uq_gui,
        # bug.md): step 4 below launches perturbAFactorGSA_mpi.x with
        # cwd=str(work), while gsa_cfg (written a few lines down) embeds
        # str(work / "...") into every path value inside gsa_input.yml. If
        # `work_dir` arrives relative (the CLI's own default, and every real
        # GSATab invocation -- the GUI never sets --work-dir), those two
        # relative uses compound: the child process's cwd is already inside
        # `work`, so a config value like "gsa_work_1_sobol/mech_1.inp" doubles
        # to a nonexistent nested path from the child's perspective. 100%
        # reproducible with a real Zero-RK binary (confirmed against both a
        # 9-species dummy fixture and a real 32-species ammonia mechanism,
        # 2026-07-23 GUI stress sweep): perturbAFactorGSA_mpi.x SIGABRTs with
        # "could not parse mech ... and thermo ... files" on every invocation.
        work = Path(work_dir).resolve()
        work.mkdir(parents=True, exist_ok=True)

        all_reactions = self._all_reactions(config.mechanism_id)
        all_ids = [rid for rid, _ in all_reactions]
        name_of = dict(all_reactions)

        selected_ids = config.reaction_ids if config.reaction_ids else all_ids
        unknown = set(selected_ids) - set(all_ids)
        if unknown:
            raise ValueError(
                f"reaction_ids not found in mechanism_id={config.mechanism_id}: {unknown}"
            )
        selected_names = [name_of[rid] for rid in selected_ids]

        ln_bound = math.log(config.factor_bound)
        problem = {
            "num_vars": len(selected_ids),
            "names": selected_names,
            "bounds": [[-ln_bound, ln_bound]] * len(selected_ids),
        }

        # 1. Sample (SALib, in its venv) -- log-multipliers per selected reaction.
        sample_resp = self._salib_call(
            "sample",
            {
                "method": config.method, "problem": problem,
                "n_samples": config.n_samples, "seed": config.seed,
                "morris_num_levels": config.morris_num_levels,
            },
            work,
        )
        X = sample_resp["X"]
        n_rows = len(X)

        # 2. Expand to Zero-RK's full-mechanism-width matrix, write gsaMatrixFile.
        full_matrix = self._build_full_matrix(X, all_ids, selected_ids)
        matrix_path = work / "gsa_matrix.mtx"
        self._write_gsa_matrix(matrix_path, full_matrix, len(all_ids))

        # 3. Mechanism export + composition resolution (reuse ZerorkBridge --
        #    same case-sensitive species lookup / stream-ratio convention
        #    Phase 1 already validated for IDT/JSR).
        mech_path = self._bridge.export_mechanism(config.mechanism_id, str(work))
        condition = SimCondition(
            T_K=config.T_K, P_atm=config.P_atm, phi=config.phi,
            fuel=config.fuel, oxidizer=config.oxidizer,
        )
        fuel_dict, oxidizer_dict = self._bridge._resolve_composition(config.mechanism_id, condition)

        out_path = work / "gsa_output.dat"
        check_path = work / "gsa_check.dat"
        # Zero-RK's own Chemkin parser (perturbAFactorGSA_mpi.x, "derived from
        # Cantera CKreader 1.0") cannot handle mechFile and thermFile pointing
        # at the identical path -- even though ChemkinWriter's output is a
        # single self-contained file with an inline THERMO ALL block (same
        # shape ck2yaml handles fine). Passing the same path for both keys
        # reproducibly aborts the parser (SIGABRT, "could not parse mech and
        # thermo files") on every mechanism tried, from a 9-species dummy
        # fixture up to a real 32-species ammonia mechanism -- confirmed via
        # isolated repro: splitting into two on-disk copies of the identical
        # content lets the parser succeed and reach its next (unrelated) step.
        # Write a second, separate copy so the two config keys never alias.
        therm_path = work / f"{mech_path.stem}_therm{mech_path.suffix}"
        shutil.copy(str(mech_path), str(therm_path))
        gsa_cfg = {
            "mechFile": str(mech_path),
            "thermFile": str(therm_path),
            "mechLogFile": str(work / "mech.clog"),
            "gsaMatrixFile": str(matrix_path),
            "outFile": str(out_path),
            "checkFile": str(check_path),
            "initTemp": config.T_K,
            "initPres": config.P_atm * 101325.0,
            "initPhi": config.phi,
            "fuelComp": fuel_dict,
            "oxidizerComp": oxidizer_dict,
            "maxTime": 1.0,
            "idtTemps": [config.idt_temp_delta],
            "relTol": 1e-8,
            "absTol": 1e-20,
        }
        gsa_yml_path = work / "gsa_input.yml"
        with open(gsa_yml_path, "w") as fh:
            yaml.safe_dump(gsa_cfg, fh, default_flow_style=False)

        # 4. Execute the real Zero-RK MPI batch run.
        exe = self._bin_dir / "perturbAFactorGSA_mpi.x"
        n_ranks = max(2, min(config.n_mpi_ranks, n_rows + 1))
        cmd = [self._mpiexec, "-n", str(n_ranks), str(exe), gsa_yml_path.name]
        log.info("Running Zero-RK GSA batch: %s (cwd=%s, %d samples)",
                  " ".join(cmd), work, n_rows)
        proc = subprocess.run(
            cmd, cwd=str(work), capture_output=True, text=True,
            timeout=resolve_timeout(timeout), env=sanitized_subprocess_env(),  # M18b
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"perturbAFactorGSA_mpi.x failed (rc={proc.returncode}):\n"
                f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
            )

        # 5. Parse Y, mean-impute any failed (non-ignition/non-finite) samples.
        Y = self._parse_gsa_output(out_path, n_rows)
        n_failed = sum(1 for y in Y if y != y)
        if n_failed:
            log.warning(
                "%d/%d GSA samples failed to ignite -- mean-imputed before analysis",
                n_failed, n_rows,
            )
        if n_failed / n_rows > _MAX_FAILED_SAMPLE_FRACTION:
            raise RuntimeError(
                f"{n_failed}/{n_rows} GSA samples failed to ignite "
                f"(>{_MAX_FAILED_SAMPLE_FRACTION:.0%}) -- aborting instead of "
                f"imputing over an unreliable majority of samples"
            )
        Y_clean = [y for y in Y if y == y]
        if not Y_clean:
            raise RuntimeError("All GSA samples failed to ignite -- no valid IDT values to analyze")
        y_mean = sum(Y_clean) / len(Y_clean)
        Y_filled = [y if y == y else y_mean for y in Y]

        # 6. Analyze (SALib, in its venv).
        analyze_req = {
            "method": config.method, "problem": problem, "Y": Y_filled,
            "seed": config.seed, "morris_num_levels": config.morris_num_levels,
        }
        if config.method == "morris":
            analyze_req["X"] = X
        analyze_resp = self._salib_call("analyze", analyze_req, work)

        result = SobolResult(
            mechanism_id=config.mechanism_id, method=config.method,
            parameter_names=selected_names, n_samples=n_rows, n_failed_samples=n_failed,
        )
        if config.method == "sobol":
            result.S1 = dict(zip(selected_names, analyze_resp["S1"]))
            result.S1_conf = dict(zip(selected_names, analyze_resp["S1_conf"]))
            result.ST = dict(zip(selected_names, analyze_resp["ST"]))
            result.ST_conf = dict(zip(selected_names, analyze_resp["ST_conf"]))
        else:
            result.mu_star = dict(zip(selected_names, analyze_resp["mu_star"]))
            result.mu_star_conf = dict(zip(selected_names, analyze_resp["mu_star_conf"]))
            result.sigma = dict(zip(selected_names, analyze_resp["sigma"]))

        # 7. Persist (DB metadata/indices + the design matrix file path).
        if save_db:
            result.run_id = self._insert_run(config, matrix_path, result)

        return result

    # ------------------------------------------------------------------
    # DB persistence
    # ------------------------------------------------------------------

    def _insert_run(
        self, config: GSARunConfig, matrix_path: Path, result: SobolResult,
    ) -> Optional[int]:
        """Insert a sensitivity_runs row + its sobol_indices/design_matrices
        children, mirroring reduction/drg.py's ``_insert_run`` pattern."""
        from rmip.database.migrations import run_gsa_migrations, _table_exists

        with sqlite3.connect(self._db) as conn:
            if not _table_exists(conn, "sensitivity_runs"):
                log.info("sensitivity_runs table missing — running GSA migrations")
                run_gsa_migrations(conn)

            ts = datetime.now(timezone.utc).isoformat()
            cur = conn.execute(
                """
                INSERT INTO sensitivity_runs
                    (mechanism_id, method, n_samples, n_failed_samples,
                     reaction_ids_json, factor_bound, T_K, P_atm, phi,
                     fuel, oxidizer, idt_temp_delta, seed, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    config.mechanism_id, config.method, result.n_samples,
                    result.n_failed_samples, json.dumps(config.reaction_ids),
                    config.factor_bound, config.T_K, config.P_atm, config.phi,
                    config.fuel, config.oxidizer, config.idt_temp_delta,
                    config.seed, ts,
                ),
            )
            run_id = cur.lastrowid

            name_to_reaction_id = {
                f"{label}:{formula}": rid
                for rid, label, formula in conn.execute(
                    "SELECT id, reaction_label, formula_string FROM reactions "
                    "WHERE mechanism_id=?",
                    (config.mechanism_id,),
                ).fetchall()
            }

            for name in result.parameter_names:
                conn.execute(
                    """
                    INSERT INTO sobol_indices
                        (run_id, reaction_id, parameter_name, S1, S1_conf, ST, ST_conf,
                         mu_star, mu_star_conf, sigma)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        run_id, name_to_reaction_id.get(name), name,
                        (result.S1 or {}).get(name), (result.S1_conf or {}).get(name),
                        (result.ST or {}).get(name), (result.ST_conf or {}).get(name),
                        (result.mu_star or {}).get(name), (result.mu_star_conf or {}).get(name),
                        (result.sigma or {}).get(name),
                    ),
                )

            conn.execute(
                """
                INSERT INTO design_matrices
                    (run_id, method, n_samples, n_parameters, seed, matrix_path, created_at)
                VALUES (?,?,?,?,?,?,?)
                """,
                (
                    run_id, config.method, result.n_samples,
                    len(result.parameter_names), config.seed, str(matrix_path), ts,
                ),
            )
            conn.commit()
        return run_id
