"""
rmip/generation/rmg_supervisor.py
=================================
W7b — RMG generation supervisor (P30, P16).

RMG jobs are notorious for never terminating (P30) and for pruning by
species *count* rather than memory, so a run can quietly eat all RAM (P16 —
unfixable from outside RMG except by supervision).  ``RmgSupervisor`` wraps
the RMG subprocess with the guards RMG itself lacks:

- **wall-clock timeout** — the whole job is killed (process group) at the
  budget, leaving the partial chemkin output inspectable;
- **memory budget** — the RSS of the whole process tree is polled via
  ``/proc`` (no psutil dependency); exceeding the budget stops the job
  *before* the OOM killer takes the machine down;
- **stall detection** — if ``RMG.log`` stops growing for ``stall_timeout_s``
  the job is declared stalled (the P30 signature: alive but going nowhere);
- **automatic tolerance relaxation** — optionally, a stalled/timed-out run
  is retried with ``toleranceMoveToCore`` multiplied by ``relax_factor``
  (bounded retries; each stage recorded), the standard manual recovery for
  non-converging enlargement;
- **staged conditions** — ``run_staged([...])`` runs a list of input files
  (e.g. one T/P point each) sequentially under one shared supervisor
  configuration, so a single pathological condition can't sink the batch.

Every early stop writes ``supervisor_report.json`` + ``.txt`` into the
output directory, so a runaway job yields a partial, *explained* result
instead of hanging for days.  Reports link into W7a: point
``rmip rmg-explain`` at the same directory for the promotion story so far.

The subprocess environment is sanitized via ``simulation/subprocess_util``.
Tests drive the supervisor with fake subprocesses (``command_template``),
so no RMG env is needed.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

_DEFAULT_RMG_PYTHON = "/home/krithika/anaconda3/envs/rmg_env/bin/python"
_DEFAULT_RMG_SCRIPT = "/data/OPEN_CODE/RMG_py/rmg.py"

_TOL_RE = re.compile(r"(toleranceMoveToCore\s*=\s*)([\d.eE+-]+)")
_CORE_SUMMARY_RE = re.compile(r"The model core has (\d+) species")


# ---------------------------------------------------------------------------
@dataclass
class StageReport:
    outcome: str          # completed | wall-timeout | memory-budget | stalled | error
    reason: str
    elapsed_s: float
    peak_rss_mb: float
    iterations_completed: int
    returncode: Optional[int]
    tolerance_move_to_core: Optional[float] = None


@dataclass
class SupervisorReport:
    input_file: str
    output_dir: str
    outcome: str
    reason: str
    elapsed_s: float
    peak_rss_mb: float
    iterations_completed: int
    stages: List[StageReport] = field(default_factory=list)
    partial_outputs: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.outcome == "completed"

    def to_dict(self) -> Dict:
        return {
            "input_file": self.input_file,
            "output_dir": self.output_dir,
            "outcome": self.outcome,
            "reason": self.reason,
            "elapsed_s": self.elapsed_s,
            "peak_rss_mb": self.peak_rss_mb,
            "iterations_completed": self.iterations_completed,
            "stages": [vars(s) for s in self.stages],
            "partial_outputs": self.partial_outputs,
        }

    def render_text(self) -> str:
        lines = [
            f"RMG supervisor report — {self.input_file}",
            f"  outcome  : {self.outcome} ({self.reason})",
            f"  elapsed  : {self.elapsed_s:.1f} s over "
            f"{len(self.stages)} stage(s)",
            f"  peak RSS : {self.peak_rss_mb:.0f} MB",
            f"  iterations completed: {self.iterations_completed}",
        ]
        for i, st in enumerate(self.stages, 1):
            tol = (f", toleranceMoveToCore={st.tolerance_move_to_core:g}"
                   if st.tolerance_move_to_core is not None else "")
            lines.append(f"    stage {i}: {st.outcome} ({st.reason}) after "
                         f"{st.elapsed_s:.1f} s, {st.iterations_completed} "
                         f"iteration(s){tol}")
        if self.partial_outputs:
            lines.append("  partial outputs available:")
            lines += [f"    - {p}" for p in self.partial_outputs]
            lines.append("  (run `rmip rmg-explain <output_dir>` for the "
                         "promotion story so far)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
def _tree_rss_mb(pid: int) -> float:
    """RSS of a process and all descendants, in MB, via /proc (Linux)."""
    total_kb = 0
    stack = [pid]
    seen = set()
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        try:
            with open(f"/proc/{p}/status") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        total_kb += int(line.split()[1])
                        break
        except OSError:
            continue
        task_dir = Path(f"/proc/{p}/task")
        try:
            for tid in task_dir.iterdir():
                child_file = tid / "children"
                try:
                    stack.extend(int(c) for c in
                                 child_file.read_text().split())
                except OSError:
                    pass
        except OSError:
            pass
    return total_kb / 1024.0


class RmgSupervisor:
    """Supervised RMG execution. See the module docstring for the guards.

    Parameters
    ----------
    rmg_python, rmg_script : the RMG env interpreter and rmg.py (defaults
        match reaction_family_kb).
    command_template : optional list with ``{input}``/``{output}``
        placeholders, overriding the RMG command entirely (used by tests and
        by anyone wrapping a different generator).
    mem_budget_mb : kill the job when the process tree's RSS exceeds this.
    wall_timeout_s : kill the job at this wall-clock budget.
    stall_timeout_s : declare a stall when the log grows by nothing for this
        long (None disables).
    relax_on_stall : retry a stalled/timed-out stage with
        ``toleranceMoveToCore * relax_factor`` (up to ``max_relaxations``).
    """

    def __init__(
        self,
        rmg_python: Optional[str] = None,
        rmg_script: Optional[str] = None,
        command_template: Optional[Sequence[str]] = None,
        mem_budget_mb: Optional[float] = None,
        wall_timeout_s: Optional[float] = None,
        stall_timeout_s: Optional[float] = None,
        poll_interval_s: float = 2.0,
        relax_on_stall: bool = False,
        relax_factor: float = 10.0,
        max_relaxations: int = 2,
    ) -> None:
        self.rmg_python = rmg_python or _DEFAULT_RMG_PYTHON
        self.rmg_script = rmg_script or _DEFAULT_RMG_SCRIPT
        self.command_template = list(command_template) if command_template \
            else None
        self.mem_budget_mb = mem_budget_mb
        self.wall_timeout_s = wall_timeout_s
        self.stall_timeout_s = stall_timeout_s
        self.poll_interval_s = float(poll_interval_s)
        self.relax_on_stall = bool(relax_on_stall)
        self.relax_factor = float(relax_factor)
        self.max_relaxations = int(max_relaxations)

    # ------------------------------------------------------------------
    def _build_command(self, input_py: str, output_dir: str) -> List[str]:
        if self.command_template:
            return [arg.replace("{input}", str(input_py))
                       .replace("{output}", str(output_dir))
                    for arg in self.command_template]
        if not Path(self.rmg_python).exists():
            raise RuntimeError(
                f"RMG interpreter not found at {self.rmg_python}. Pass "
                "rmg_python= for your RMG env (or command_template= for a "
                "custom generator).")
        return [self.rmg_python, self.rmg_script, "-o", str(output_dir),
                str(input_py)]

    @staticmethod
    def _read_tolerance(input_py: str) -> Optional[float]:
        try:
            m = _TOL_RE.search(Path(input_py).read_text())
            return float(m.group(2)) if m else None
        except OSError:
            return None

    @staticmethod
    def _relax_tolerance(input_py: str, factor: float, out_path: str) -> float:
        """Write a copy of the input with toleranceMoveToCore scaled up."""
        text = Path(input_py).read_text()
        m = _TOL_RE.search(text)
        if not m:
            raise RuntimeError(
                f"Cannot relax tolerance: no toleranceMoveToCore found in "
                f"{input_py}")
        new_tol = float(m.group(2)) * factor
        Path(out_path).write_text(
            _TOL_RE.sub(lambda mm: f"{mm.group(1)}{new_tol:g}", text,
                        count=1))
        return new_tol

    # ------------------------------------------------------------------
    def _run_stage(self, input_py: str, output_dir: str) -> StageReport:
        from rmip.simulation.subprocess_util import sanitized_subprocess_env

        os.makedirs(output_dir, exist_ok=True)
        cmd = self._build_command(input_py, output_dir)
        log_path = Path(output_dir) / "RMG.log"
        stdout_path = Path(output_dir) / "supervisor_stdout.log"

        t0 = time.monotonic()
        peak_rss = 0.0
        outcome, reason = "error", "unknown"
        with open(stdout_path, "w") as out_fh:
            proc = subprocess.Popen(
                cmd, stdout=out_fh, stderr=subprocess.STDOUT,
                env=sanitized_subprocess_env(), start_new_session=True)
            last_log_size = -1
            last_growth_t = time.monotonic()
            try:
                while True:
                    rc = proc.poll()
                    if rc is not None:
                        outcome = "completed" if rc == 0 else "error"
                        reason = (f"process exited rc={rc}" if rc
                                  else "process finished")
                        break

                    now = time.monotonic()
                    peak_rss = max(peak_rss, _tree_rss_mb(proc.pid))

                    if (self.wall_timeout_s is not None
                            and now - t0 > self.wall_timeout_s):
                        outcome, reason = "wall-timeout", (
                            f"wall-clock budget {self.wall_timeout_s:g} s "
                            "exceeded")
                        self._kill(proc)
                        break
                    if (self.mem_budget_mb is not None
                            and peak_rss > self.mem_budget_mb):
                        outcome, reason = "memory-budget", (
                            f"process-tree RSS {peak_rss:.0f} MB exceeded "
                            f"budget {self.mem_budget_mb:g} MB (P16)")
                        self._kill(proc)
                        break
                    if self.stall_timeout_s is not None:
                        size = (log_path.stat().st_size
                                if log_path.exists()
                                else stdout_path.stat().st_size)
                        if size != last_log_size:
                            last_log_size = size
                            last_growth_t = now
                        elif now - last_growth_t > self.stall_timeout_s:
                            outcome, reason = "stalled", (
                                f"no log growth for {self.stall_timeout_s:g}"
                                " s (P30 signature)")
                            self._kill(proc)
                            break
                    time.sleep(self.poll_interval_s)
            finally:
                if proc.poll() is None:
                    self._kill(proc)

        iterations = self._count_iterations(log_path, stdout_path)
        return StageReport(
            outcome=outcome, reason=reason,
            elapsed_s=time.monotonic() - t0, peak_rss_mb=peak_rss,
            iterations_completed=iterations, returncode=proc.poll(),
            tolerance_move_to_core=self._read_tolerance(input_py))

    @staticmethod
    def _kill(proc: subprocess.Popen) -> None:
        """Terminate the whole process group, escalating to SIGKILL."""
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            proc.wait()

    @staticmethod
    def _count_iterations(*log_paths: Path) -> int:
        for p in log_paths:
            try:
                return len(_CORE_SUMMARY_RE.findall(
                    Path(p).read_text(errors="replace")))
            except OSError:
                continue
        return 0

    # ------------------------------------------------------------------
    def run(self, input_py: str, output_dir: str) -> SupervisorReport:
        """Run one supervised RMG job (with bounded tolerance-relaxation
        retries if configured); write the report into ``output_dir``."""
        stages: List[StageReport] = []
        current_input = str(input_py)
        for attempt in range(1 + (self.max_relaxations
                                  if self.relax_on_stall else 0)):
            stage = self._run_stage(current_input, output_dir)
            stages.append(stage)
            if stage.outcome == "completed" or not self.relax_on_stall:
                break
            if stage.outcome not in ("stalled", "wall-timeout"):
                break
            if attempt >= self.max_relaxations:
                break
            relaxed = str(Path(output_dir)
                          / f"input_relaxed_{attempt + 1}.py")
            try:
                new_tol = self._relax_tolerance(
                    current_input, self.relax_factor, relaxed)
            except RuntimeError as exc:
                stage.reason += f"; relaxation impossible ({exc})"
                break
            logger.info("Stage %d %s — retrying with toleranceMoveToCore="
                        "%g", attempt + 1, stage.outcome, new_tol)
            current_input = relaxed

        last = stages[-1]
        report = SupervisorReport(
            input_file=str(input_py), output_dir=str(output_dir),
            outcome=last.outcome, reason=last.reason,
            elapsed_s=sum(s.elapsed_s for s in stages),
            peak_rss_mb=max(s.peak_rss_mb for s in stages),
            iterations_completed=last.iterations_completed,
            stages=stages,
            partial_outputs=self._partial_outputs(output_dir))
        self._write_report(report)
        return report

    def run_staged(self, inputs: Sequence[str],
                   output_root: str) -> List[SupervisorReport]:
        """Run several input files (e.g. one T/P condition each)
        sequentially; one pathological stage cannot sink the batch."""
        reports = []
        for i, inp in enumerate(inputs, 1):
            out = str(Path(output_root) / f"stage_{i:02d}")
            reports.append(self.run(inp, out))
        return reports

    # ------------------------------------------------------------------
    @staticmethod
    def _partial_outputs(output_dir: str) -> List[str]:
        found = []
        for rel in ("chemkin/chem_annotated.inp", "chemkin/chem.inp",
                    "chemkin/species_dictionary.txt", "RMG.log"):
            p = Path(output_dir) / rel
            if p.is_file():
                found.append(str(p))
        return found

    @staticmethod
    def _write_report(report: SupervisorReport) -> None:
        base = Path(report.output_dir) / "supervisor_report"
        with open(base.with_suffix(".json"), "w") as fh:
            json.dump(report.to_dict(), fh, indent=2)
        base.with_suffix(".txt").write_text(report.render_text() + "\n")
