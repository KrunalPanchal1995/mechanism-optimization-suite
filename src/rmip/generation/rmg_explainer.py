"""
rmip/generation/rmg_explainer.py
================================
W7a — edge→core promotion explainer + edge inspector (P17, P26).

RMG gives no after-the-fact visibility into *why* each species ended up in
the model core, and no way to inspect the edge species it considered but
rejected (the P26 ask, unmet since 2007).  All of that information is in
fact already in ``RMG.log`` — RMG's solver logs every promotion with the
flux value and tolerance that triggered it — it has just never been
collected into a report.  This module parses the log (read-only; no RMG
internals touched, no RMG env needed) and attributes each core species to
the event that promoted it.

Log grammar (verbatim from the RMG-Py source, ``rmgpy/solver/base.pyx`` and
``rmgpy/rmg/model.py`` — see the regexes below for the exact lines):

- flux promotion: ``At time T s, species X at rate ratio V exceeded the
  minimum rate for moving to model core of TOL``
- resurrection:   ``… was added to model core in model resurrection process``
- singularity:    ``… was added to model core to avoid singularity``
- confirmation:   ``Adding species X to model core``
- edge sighting:  ``… exceeded the minimum rate for simulation interruption
  of TOL`` (a species that got *close* but was not promoted at that point)
- pruning:        ``Pruning species X to make num_edge_species smaller …``
- forbidden:      ``Species X was Forbidden and not added to Core…``
- iteration end:  ``The model core has N species and M reactions``

Iterations are counted by the enlargement summaries, so every event is
attributed to the iteration in which it occurred.

If the run saved edge chemkin output (``chemkin/chem_edge_annotated.inp``),
its SPECIES block is merged into the edge inventory so edge species that
never showed measurable flux are listed too.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Log line grammar (kept verbatim-compatible with RMG-Py's format strings)
# ---------------------------------------------------------------------------
_NUM = r"[\d.eE+-]+"
_RE_FLUX_PROMOTION = re.compile(
    rf"At time\s+({_NUM}) s, species (.+?) at rate ratio ({_NUM}) exceeded "
    rf"the minimum rate for moving to model core of ({_NUM})")
_RE_RESURRECTION = re.compile(
    rf"At time\s+({_NUM}) s, species (.+?) at rate ratio ({_NUM}) was added "
    rf"to model core in model resurrection process")
_RE_SINGULARITY = re.compile(
    rf"At time\s+({_NUM}) s, species (.+?) was added to model core to avoid "
    rf"singularity")
_RE_ADDING_CORE = re.compile(r"Adding species (.+?) to model core\s*$")
_RE_EDGE_SIGHTING = re.compile(
    rf"At time\s+({_NUM}) s, species (.+?) at ({_NUM}) exceeded the minimum "
    rf"rate for simulation interruption of ({_NUM})")
_RE_PRUNING = re.compile(
    r"Pruning species (.+?) to make num_edge_species smaller")
_RE_FORBIDDEN = re.compile(
    r"Species (.+?) was Forbidden and not added to Core")
_RE_CORE_SUMMARY = re.compile(
    r"The model core has (\d+) species and (\d+) reactions")
_RE_EDGE_SUMMARY = re.compile(
    r"The model edge has (\d+) species and (\d+) reactions")
_RE_COMPLETED = re.compile(r"MODEL GENERATION COMPLETED")


@dataclass
class PromotionEvent:
    """Why one species entered the model core."""
    species: str
    mechanism: str            # "flux" | "resurrection" | "singularity" | "seed/initial"
    iteration: int
    time_s: Optional[float] = None
    rate_ratio: Optional[float] = None
    tolerance: Optional[float] = None

    def why(self) -> str:
        if self.mechanism == "flux":
            return (f"rate ratio {self.rate_ratio:g} exceeded the "
                    f"move-to-core tolerance {self.tolerance:g} at "
                    f"t = {self.time_s:g} s (iteration {self.iteration})")
        if self.mechanism == "resurrection":
            return (f"added during model resurrection (highest-flux edge "
                    f"species, rate ratio {self.rate_ratio:g}) at "
                    f"t = {self.time_s:g} s (iteration {self.iteration})")
        if self.mechanism == "singularity":
            return (f"added to avoid a zero-flux singularity at "
                    f"t = {self.time_s:g} s (iteration {self.iteration})")
        return ("present from the start (seed mechanism / initial species / "
                "library) — never flux-promoted")


@dataclass
class EdgeRecord:
    """One species RMG considered but did not keep in the core."""
    species: str
    fate: str                     # "pruned" | "forbidden" | "remained-edge"
    max_rate_ratio: Optional[float] = None   # best flux it ever showed
    last_seen_iteration: Optional[int] = None


@dataclass
class RunExplanation:
    run_dir: str
    promotions: List[PromotionEvent] = field(default_factory=list)
    edge: List[EdgeRecord] = field(default_factory=list)
    iterations: int = 0
    completed: bool = False
    final_core: Optional[Dict[str, int]] = None      # {"species": n, "reactions": m}
    final_edge: Optional[Dict[str, int]] = None

    def to_dict(self) -> Dict:
        return {
            "run_dir": self.run_dir,
            "iterations": self.iterations,
            "completed": self.completed,
            "final_core": self.final_core,
            "final_edge": self.final_edge,
            "promotions": [
                {"species": p.species, "mechanism": p.mechanism,
                 "iteration": p.iteration, "time_s": p.time_s,
                 "rate_ratio": p.rate_ratio, "tolerance": p.tolerance,
                 "why": p.why()}
                for p in self.promotions],
            "edge": [
                {"species": e.species, "fate": e.fate,
                 "max_rate_ratio": e.max_rate_ratio,
                 "last_seen_iteration": e.last_seen_iteration}
                for e in self.edge],
        }

    def render_text(self, top_edge: int = 30) -> str:
        lines = [
            f"RMG run explanation — {self.run_dir}",
            f"  iterations: {self.iterations}   "
            f"completed: {self.completed}",
        ]
        if self.final_core:
            lines.append(f"  final core: {self.final_core['species']} species"
                         f" / {self.final_core['reactions']} reactions")
        if self.final_edge:
            lines.append(f"  final edge: {self.final_edge['species']} species"
                         f" / {self.final_edge['reactions']} reactions")
        lines += ["", "Why each core species is here "
                      "(flux-ranked; P17):"]
        for p in self.promotions:
            lines.append(f"  {p.species:<24} {p.why()}")
        lines += ["", f"Edge species RMG considered but rejected (P26), "
                      f"top {top_edge} by flux:"]
        for e in self.edge[:top_edge]:
            rr = (f"max rate ratio {e.max_rate_ratio:g}"
                  if e.max_rate_ratio is not None else "no measurable flux")
            lines.append(f"  {e.species:<24} {e.fate:<14} {rr}")
        if not self.edge:
            lines.append("  (none recorded in the log; run with "
                         "saveEdgeSpecies=True for the full edge)")
        return "\n".join(lines)


class RmgRunExplainer:
    """Parse a finished (or interrupted) RMG run directory.

    Parameters
    ----------
    run_dir : str — directory containing ``RMG.log`` and (optionally)
        ``chemkin/chem_edge_annotated.inp``.
    """

    def __init__(self, run_dir: str) -> None:
        self.run_dir = str(run_dir)
        self.log_path = self._find_log()

    def _find_log(self) -> str:
        for cand in ("RMG.log", "rmg.log"):
            p = Path(self.run_dir) / cand
            if p.is_file():
                return str(p)
        raise FileNotFoundError(
            f"No RMG.log found in {self.run_dir} — point run_dir at an RMG "
            "output directory.")

    # ------------------------------------------------------------------
    def explain(self) -> RunExplanation:
        promotions: Dict[str, PromotionEvent] = {}     # first event wins
        sightings: Dict[str, EdgeRecord] = {}
        iteration = 1
        n_summaries = 0
        completed = False
        final_core = final_edge = None

        def see_edge(name: str, ratio: Optional[float]) -> None:
            rec = sightings.setdefault(name, EdgeRecord(species=name,
                                                        fate="remained-edge"))
            if ratio is not None and (rec.max_rate_ratio is None
                                      or ratio > rec.max_rate_ratio):
                rec.max_rate_ratio = ratio
            rec.last_seen_iteration = iteration

        with open(self.log_path, errors="replace") as fh:
            for line in fh:
                m = _RE_FLUX_PROMOTION.search(line)
                if m:
                    t, sp, ratio, tol = m.groups()
                    promotions.setdefault(sp, PromotionEvent(
                        species=sp, mechanism="flux", iteration=iteration,
                        time_s=float(t), rate_ratio=float(ratio),
                        tolerance=float(tol)))
                    continue
                m = _RE_RESURRECTION.search(line)
                if m:
                    t, sp, ratio = m.groups()
                    promotions.setdefault(sp, PromotionEvent(
                        species=sp, mechanism="resurrection",
                        iteration=iteration, time_s=float(t),
                        rate_ratio=float(ratio)))
                    continue
                m = _RE_SINGULARITY.search(line)
                if m:
                    t, sp = m.groups()
                    promotions.setdefault(sp, PromotionEvent(
                        species=sp, mechanism="singularity",
                        iteration=iteration, time_s=float(t)))
                    continue
                m = _RE_ADDING_CORE.search(line)
                if m:
                    sp = m.group(1)
                    promotions.setdefault(sp, PromotionEvent(
                        species=sp, mechanism="seed/initial",
                        iteration=iteration))
                    sightings.pop(sp, None)     # it made it — not edge
                    continue
                m = _RE_EDGE_SIGHTING.search(line)
                if m:
                    _t, sp, ratio, _tol = m.groups()
                    if sp not in promotions:
                        see_edge(sp, float(ratio))
                    continue
                m = _RE_PRUNING.search(line)
                if m:
                    see_edge(m.group(1), None)
                    sightings[m.group(1)].fate = "pruned"
                    continue
                m = _RE_FORBIDDEN.search(line)
                if m:
                    see_edge(m.group(1), None)
                    sightings[m.group(1)].fate = "forbidden"
                    continue
                m = _RE_CORE_SUMMARY.search(line)
                if m:
                    final_core = {"species": int(m.group(1)),
                                  "reactions": int(m.group(2))}
                    n_summaries += 1
                    iteration = n_summaries + 1
                    continue
                m = _RE_EDGE_SUMMARY.search(line)
                if m:
                    final_edge = {"species": int(m.group(1)),
                                  "reactions": int(m.group(2))}
                    continue
                if _RE_COMPLETED.search(line):
                    completed = True

        # Promoted species can't also be edge records (late sightings).
        for sp in promotions:
            sightings.pop(sp, None)

        # Merge the saved edge chemkin species list, if present (P26).
        for name in self._edge_chemkin_species():
            if name not in promotions:
                sightings.setdefault(name, EdgeRecord(species=name,
                                                      fate="remained-edge"))

        # Rank: promotions by flux (seed/initial last), edge by flux.
        promo_list = sorted(
            promotions.values(),
            key=lambda p: (p.mechanism == "seed/initial",
                           -(p.rate_ratio or 0.0)))
        edge_list = sorted(
            sightings.values(),
            key=lambda e: -(e.max_rate_ratio
                            if e.max_rate_ratio is not None else -1.0))

        return RunExplanation(
            run_dir=self.run_dir, promotions=promo_list, edge=edge_list,
            iterations=max(n_summaries, 1) if (promo_list or n_summaries)
            else 0,
            completed=completed, final_core=final_core, final_edge=final_edge)

    # ------------------------------------------------------------------
    def _edge_chemkin_species(self) -> List[str]:
        """SPECIES block of chemkin/chem_edge_annotated.inp, if saved."""
        for cand in ("chemkin/chem_edge_annotated.inp",
                     "chemkin/chem_edge.inp"):
            p = Path(self.run_dir) / cand
            if p.is_file():
                return _chemkin_species_block(str(p))
        return []


def _chemkin_species_block(path: str) -> List[str]:
    """Names in a CHEMKIN file's SPECIES…END block (comments stripped)."""
    names: List[str] = []
    in_block = False
    with open(path, errors="replace") as fh:
        for raw in fh:
            line = raw.split("!", 1)[0].strip()
            if not line:
                continue
            up = line.upper()
            if up.startswith("SPECIES"):
                in_block = True
                line = line[len("SPECIES"):].strip()
                if not line:
                    continue
                up = line.upper()
            if in_block:
                if up.startswith("END"):
                    break
                names.extend(line.split())
    return names
