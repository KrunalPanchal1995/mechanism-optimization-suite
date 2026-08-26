"""
rmip/simulation/zerork_bridge.py
==================================
Phase 1 of MarkDownFiles/ZERORK_GSA_PLAN.md -- RMIP <-> Zero-RK bridge.

Subprocess + YAML file contract only (D7 principle, same as
``muqsac_bridge.py``): RMIP never imports Zero-RK internals, only calls its
compiled ``.x`` binaries and reads the text files they write.

Reactor coverage this phase: ``constVolumeWSR`` (IDT) and ``constVolumePSR``
(JSR). See ZERORK_GSA_PLAN.md's Phase 0 section for the verified install
location, the build process, and three empirically-discovered quirks this
module works around:

1. Species names in ``fuel_mole_fracs``/``oxidizer_mole_fracs`` must match
   the mechanism file's *exact* casing (case-sensitive lookup in Zero-RK's
   parser). ``_zerork_species_names`` resolves this against the DB's
   ``species.name`` values -- the same source of truth
   ``rmip.io.chemkin_writer.ChemkinWriter`` used to write the mechanism
   file Zero-RK actually reads, so casing is guaranteed to match regardless
   of what casing Cantera's own species names happen to use.
2. ``constVolumeWSR``/``constVolumePSR``'s YAML schemas require several
   fields as **vectors**, even for a single condition (``initial_temperatures``,
   ``initial_pressures``, ``initial_phis``, ``initial_egrs``,
   ``preconditioner_thresholds``), plus a few fields with no documented
   default (``eps_lin``, ``nonlinear_convergence_coefficient``) that abort
   immediately if omitted. Defaults for the latter are hardcoded here from
   Zero-RK's own shipped examples.
3. IDT definition differs from ``rmip.simulation.idt_runner.IDTRunner``:
   Zero-RK's ``constVolumeWSR`` computes ignition delay via a fixed
   temperature-rise threshold (``temperature_deltas``), while ``IDTRunner``
   (Cantera-based) uses max-dT/dt with a threshold fallback. The two agree
   closely in practice -- cross-validated to 0.00004% relative difference
   on a real ``project.db`` mechanism in Phase 0 -- but are not the same
   algorithm. Both produce ``IDTResult``, so callers comparing across
   solvers should be aware the underlying definition differs slightly.
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from rmip.simulation.subprocess_util import resolve_timeout, sanitized_subprocess_env
from rmip.types import IDTResult, JSRResult, SimCondition

log = logging.getLogger("rmip.simulation.zerork_bridge")

# Defaults matching Zero-RK's own shipped examples (constVolumeWSR/hydrogen_TSweep.yml,
# constVolumePSR/input.yml.in) -- these fields have no documented default and abort
# the run immediately if omitted, so hardcoding known-good values here is deliberate,
# not a guess.
_DEFAULT_PRECONDITIONER_THRESHOLD = 2.048e-3
_DEFAULT_EPS_LIN = 0.05
_DEFAULT_NONLINEAR_CONV_COEFF = 0.05


class ZerorkBridge:
    """
    Bridge between RMIP and Zero-RK's ``constVolumeWSR`` (IDT) /
    ``constVolumePSR`` (JSR) solvers.

    Parameters
    ----------
    db_path : str
        RMIP SQLite database path.
    zerork_install_dir : str
        Path to Zero-RK's install prefix (must contain ``bin/``), e.g.
        ``"/home/krithika/zero-rk/build2/inst_dir"``.
    """

    def __init__(self, db_path: str, zerork_install_dir: str) -> None:
        self._db = db_path
        self._bin_dir = Path(zerork_install_dir) / "bin"
        if not self._bin_dir.exists():
            raise FileNotFoundError(
                f"Zero-RK bin/ not found at {self._bin_dir} -- check "
                "zerork_install_dir points at the install prefix (contains bin/)."
            )

    # ------------------------------------------------------------------
    # Mechanism export (DB -> CHEMKIN-ASCII, Zero-RK's required input format)
    # ------------------------------------------------------------------

    def export_mechanism(self, mechanism_id: int, work_dir: str) -> Path:
        """
        Write the DB mechanism to a single CHEMKIN-ASCII file (embedded
        THERMO block) that Zero-RK reads as both ``mechFile`` and
        ``thermFile``. Reuses ``ChemkinWriter``, which is DB-driven (not
        format-of-origin-driven) -- works whether the mechanism was
        originally loaded from CHEMKIN or Cantera YAML (Phase 0 confirmed
        no new converter is needed for this).
        """
        from rmip.io.chemkin_writer import ChemkinWriter

        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        mech_path = work / f"mech_{mechanism_id}.inp"

        writer = ChemkinWriter(self._db, use_original_units=True)
        writer.write(mechanism_id, str(mech_path), include_thermo_in_mech=True)
        return mech_path

    def _zerork_species_names(self, mechanism_id: int) -> Dict[str, str]:
        """
        Map UPPER(name) -> exact DB-stored casing for every species in this
        mechanism, so composition dicts can be safely re-keyed to match
        whatever casing ``export_mechanism`` wrote into the mechanism file
        Zero-RK will actually parse.
        """
        import sqlite3

        with sqlite3.connect(self._db) as conn:
            rows = conn.execute(
                "SELECT name FROM species WHERE mechanism_id=?", (mechanism_id,)
            ).fetchall()
        return {name.upper(): name for (name,) in rows}

    def _resolve_composition(
        self, mechanism_id: int, condition: SimCondition,
    ) -> tuple[Dict[str, float], Dict[str, float]]:
        """
        Parse condition.fuel/condition.oxidizer directly into
        (fuel_mole_fracs, oxidizer_mole_fracs) dicts, re-keyed to the DB's
        exact species-name casing so Zero-RK's case-sensitive parser accepts
        them.

        NOTE: this deliberately does NOT go through Cantera's
        ``set_equivalence_ratio`` + ``mole_fraction_dict()`` the way
        ``MuqsacBridge._resolve_composition`` does. Zero-RK's
        ``fuel_mole_fracs``/``oxidizer_mole_fracs`` fields are *stream-
        relative* ratios (moles of species per mole of fuel / per mole of
        oxidizer -- the same semantics as Cantera's own
        ``set_equivalence_ratio(phi, fuel=..., oxidizer=...)`` *input*
        strings, e.g. ``oxidizer="O2:1.0, N2:3.76"`` need not sum to 1) and
        it applies ``initial_phis`` internally to combine them into the
        actual mixture. Feeding it pre-mixed absolute mole fractions (as an
        earlier version of this method did, by calling
        ``set_equivalence_ratio`` here and also passing ``initial_phis``)
        double-applies the phi mixing and produces a physically invalid
        initial state -- confirmed empirically: CVode failed at t=0 with
        NaN state for every species on every reaction condition tried.
        """
        casing = self._zerork_species_names(mechanism_id)

        def _parse(spec: str) -> Dict[str, float]:
            out: Dict[str, float] = {}
            for token in spec.split(","):
                token = token.strip()
                if not token:
                    continue
                sp, _, x_str = token.partition(":")
                sp = sp.strip()
                x = float(x_str) if x_str else 1.0
                db_name = casing.get(sp.upper())
                if db_name is None:
                    raise ValueError(
                        f"Species '{sp}' not found in mechanism_id={mechanism_id} "
                        "(case-insensitive match failed against DB species table)."
                    )
                out[db_name] = x
            return out

        fuel_dict = _parse(condition.fuel)
        oxidizer_dict = _parse(condition.oxidizer)
        if not fuel_dict:
            raise ValueError(f"No fuel species resolved for {condition.fuel!r}")
        return fuel_dict, oxidizer_dict

    # ------------------------------------------------------------------
    # IDT (constVolumeWSR)
    # ------------------------------------------------------------------

    def build_idt_input(
        self,
        mech_path: Path,
        mechanism_id: int,
        condition: SimCondition,
        work_dir: str,
        idt_temp_delta: float = 400.0,
        stop_time_s: float = 1.0,
        rtol: float = 1e-8,
        atol: float = 1e-20,
    ) -> Path:
        """Write a constVolumeWSR YAML input for a single SimCondition."""
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)

        fuel_dict, oxidizer_dict = self._resolve_composition(mechanism_id, condition)

        cfg = {
            "mechFile": str(mech_path),
            "thermFile": str(mech_path),
            "logFile": "mech.clog",
            "idtFile": "idt.dat",
            "thistFile": "idt.thist",
            "fuel_mole_fracs": fuel_dict,
            "oxidizer_mole_fracs": oxidizer_dict,
            "trace_mole_fracs": {},
            "temperature_deltas": [idt_temp_delta],
            "stop_time": stop_time_s,
            "print_time": stop_time_s,
            "relative_tolerance": rtol,
            "absolute_tolerance": atol,
            "initial_temperatures": [condition.T_K],
            "initial_pressures": [condition.P_atm * 101325.0],
            "initial_phis": [condition.phi],
            "initial_egrs": [0.0],
            "preconditioner_thresholds": [_DEFAULT_PRECONDITIONER_THRESHOLD],
            "eps_lin": _DEFAULT_EPS_LIN,
            "nonlinear_convergence_coefficient": _DEFAULT_NONLINEAR_CONV_COEFF,
            "long_output": 1,
        }
        input_path = work / "idt_input.yml"
        with open(input_path, "w") as fh:
            yaml.safe_dump(cfg, fh, default_flow_style=False)
        return input_path

    def run_idt(self, input_yml: Path, timeout: Optional[float] = None) -> subprocess.CompletedProcess:
        """Invoke ``constVolumeWSR.x <input_yml>`` as a subprocess (serial, no MPI)."""
        exe = self._bin_dir / "constVolumeWSR.x"
        work_dir = Path(input_yml).parent
        cmd = [str(exe), input_yml.name]
        log.info("Running Zero-RK IDT: %s (cwd=%s)", " ".join(cmd), work_dir)
        return subprocess.run(
            cmd, cwd=str(work_dir), capture_output=True, text=True,
            timeout=resolve_timeout(timeout), env=sanitized_subprocess_env(),  # M18b
        )

    def parse_idt_output(
        self, work_dir: str, mechanism_id: int, condition: SimCondition,
    ) -> IDTResult:
        """
        Parse constVolumeWSR's ``idt.dat``.

        Fixed-position data columns (verified empirically in Phase 0,
        stable regardless of species/mole-fraction column count which comes
        after): 0=run_id 1=T 2=P 3=phi 4=EGR 5=P_thresh 6=density
        7=IDT[s] 8=cpu_time[s] 9=cvode_err_count.
        """
        path = Path(work_dir) / "idt.dat"
        if not path.exists():
            return IDTResult(
                condition=condition, mechanism_id=mechanism_id,
                idt_s=None, converged=False, cpu_time_s=0.0,
                error_msg=f"Output file not found: {path}",
            )
        data_line = self._first_data_row(path)
        if data_line is None:
            return IDTResult(
                condition=condition, mechanism_id=mechanism_id,
                idt_s=None, converged=False, cpu_time_s=0.0,
                error_msg="No data row found in idt.dat",
            )
        fields = data_line.split()
        idt_s = float(fields[7])
        cpu_time_s = float(fields[8])
        cvode_err = int(fields[9])
        return IDTResult(
            condition=condition, mechanism_id=mechanism_id,
            idt_s=idt_s if idt_s > 0 else None,
            converged=(cvode_err == 0 and idt_s > 0),
            cpu_time_s=cpu_time_s,
            error_msg=None if cvode_err == 0 else f"{cvode_err} CVode error(s)",
        )

    # ------------------------------------------------------------------
    # JSR (constVolumePSR)
    # ------------------------------------------------------------------

    def build_jsr_input(
        self,
        mech_path: Path,
        mechanism_id: int,
        condition: SimCondition,
        residence_time_s: float,
        work_dir: str,
        tracked_species: Optional[List[str]] = None,
        stop_time_s: float = 10.0,
        rtol: float = 1e-8,
        atol: float = 1e-20,
    ) -> Path:
        """
        Write a constVolumePSR YAML input for a single SimCondition.

        ``tracked_species`` defaults to every species in the mechanism if
        not given -- constVolumePSR only reports mole fractions for
        species listed in ``tracked_species_names``, so the default keeps
        ``parse_jsr_output`` able to return a full composition (matching
        MuqsacBridge/JSRResult's `species_mole_fractions` contract) rather
        than a partial one.
        """
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)

        fuel_dict, oxidizer_dict = self._resolve_composition(mechanism_id, condition)

        if tracked_species is None:
            casing = self._zerork_species_names(mechanism_id)
            tracked_species = sorted(casing.values())

        cfg = {
            "mechFile": str(mech_path),
            "thermFile": str(mech_path),
            "logFile": "mech.clog",
            "idtFile": "jsr.dat",
            "thistFile": "jsr.thist",
            "fuel_mole_fracs": fuel_dict,
            "oxidizer_mole_fracs": oxidizer_dict,
            "tracked_species_names": list(tracked_species),
            "steplimiter": 1e16,
            "stop_time": stop_time_s,
            "energy_enabled": 1,
            "residence_times": [residence_time_s],
            "print_time": stop_time_s,
            "max_internal_dt": 1.0,
            "relative_tolerance": rtol,
            "absolute_tolerance": atol,
            "initial_temperatures": [condition.T_K],
            "initial_pressures": [condition.P_atm * 101325.0],
            "initial_phis": [condition.phi],
            "initial_egrs": [0.0],
            "preconditioner_thresholds": [1.0e-6],
            "eps_lin": _DEFAULT_EPS_LIN,
            "nonlinear_convergence_coefficient": _DEFAULT_NONLINEAR_CONV_COEFF,
        }
        input_path = work / "jsr_input.yml"
        with open(input_path, "w") as fh:
            yaml.safe_dump(cfg, fh, default_flow_style=False)
        # stash the tracked-species order alongside the input for the parser
        (work / "jsr_tracked_species.yml").write_text(
            yaml.safe_dump(list(tracked_species))
        )
        return input_path

    def run_jsr(self, input_yml: Path, timeout: Optional[float] = None) -> subprocess.CompletedProcess:
        """Invoke ``constVolumePSR.x <input_yml>`` as a subprocess (serial, no MPI)."""
        exe = self._bin_dir / "constVolumePSR.x"
        work_dir = Path(input_yml).parent
        cmd = [str(exe), input_yml.name]
        log.info("Running Zero-RK JSR: %s (cwd=%s)", " ".join(cmd), work_dir)
        return subprocess.run(
            cmd, cwd=str(work_dir), capture_output=True, text=True,
            timeout=resolve_timeout(timeout), env=sanitized_subprocess_env(),  # M18b
        )

    def parse_jsr_output(
        self, work_dir: str, mechanism_id: int, condition: SimCondition,
    ) -> JSRResult:
        """
        Parse constVolumePSR's ``jsr.dat``.

        Fixed-position columns (verified empirically): 0=run_id 1=T 2=P
        3=phi 4=EGR 5=P_thresh 6=density 7=resid_time[s] 8=sim_time[s]
        9=cvode_err_count, followed by N "init mlfrc <sp>" then N
        "final mlfrc <sp>" columns (N = len(tracked_species), order taken
        from the sidecar file ``build_jsr_input`` wrote, not re-derived).
        """
        path = Path(work_dir) / "jsr.dat"
        tracked_path = Path(work_dir) / "jsr_tracked_species.yml"
        if not path.exists():
            return JSRResult(
                condition=condition, mechanism_id=mechanism_id, solver="zerork",
                residence_time_s=0.0, species_mole_fractions={}, converged=False,
                error_msg=f"Output file not found: {path}",
            )
        tracked_species = yaml.safe_load(tracked_path.read_text()) if tracked_path.exists() else []
        n = len(tracked_species)

        data_line = self._first_data_row(path)
        if data_line is None:
            return JSRResult(
                condition=condition, mechanism_id=mechanism_id, solver="zerork",
                residence_time_s=0.0, species_mole_fractions={}, converged=False,
                error_msg="No data row found in jsr.dat",
            )
        fields = data_line.split()
        resid_time_s = float(fields[7])
        cpu_time_s = float(fields[8])
        cvode_err = int(fields[9])

        final_start = 10 + n  # 10 fixed columns (0-9) + N "init mlfrc" columns
        final_fracs = {
            sp: float(fields[final_start + i])
            for i, sp in enumerate(tracked_species)
        } if n > 0 else {}

        return JSRResult(
            condition=condition, mechanism_id=mechanism_id, solver="zerork",
            residence_time_s=resid_time_s,
            species_mole_fractions=final_fracs,
            converged=(cvode_err == 0),
            cpu_time_s=cpu_time_s,
            error_msg=None if cvode_err == 0 else f"{cvode_err} CVode error(s)",
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _first_data_row(path: Path) -> Optional[str]:
        """
        Return the first non-comment, non-blank line of a Zero-RK output
        file -- the sweep-results table's data row(s) come before a blank
        line and a second (stats-summary) table, so for a single-condition
        run this is exactly the one row we want.
        """
        with open(path) as fh:
            for line in fh:
                s = line.strip()
                if s and not s.startswith("#"):
                    return s
        return None
