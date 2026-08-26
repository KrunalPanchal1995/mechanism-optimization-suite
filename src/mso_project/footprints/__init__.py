"""
mso_project.footprints
=======================
Declarative artifact-role definitions, one module per pipeline, used by
``Project.capture()``/``Project.scan()`` to assign a meaningful ``role`` to
a discovered file instead of leaving everything ``"unclassified"``.

A footprint is a list of :class:`FootprintEntry` -- ``pattern`` is a glob
relative to the run's ``working_dir`` (``fnmatch``-style, ``**`` matches
across directory separators via ``pathlib.Path.glob``'s own recursive-glob
support). The FIRST matching entry wins; entries are checked in list order,
so put more specific patterns before broader ones within a footprint.

Adding a new pipeline: create ``footprints/<name>.py`` with a module-level
``FOOTPRINT: List[FootprintEntry]``, then add it to ``FOOTPRINTS`` below.
Nothing else needs to change -- ``Project.capture(pipeline=<name>, ...)``
picks it up automatically, and anything a footprint doesn't cover still gets
registered (never silently dropped) via ``_generic``'s extension-based
fallback with ``role="unclassified"``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class FootprintEntry:
    pattern: str            # glob relative to working_dir, e.g. "Opt/Data/Simulations/sim_data_case-*.lst"
    role: str                # e.g. "simulation_log"
    kind: Optional[str] = None  # override _generic's extension-based kind guess if set
    required: bool = False   # informational only today -- reserved for a future "footprint completeness" check


from . import (  # noqa: E402  (after FootprintEntry is defined, which these modules import)
    brute_force_sa,
    forward_uq,
    fuse_kld,
    graph_sa,
    idt_sensitivity,
    mean_value_fusion,
    mechanism_check,
    mechanism_converter,
    mso_compare,
    nominal_sim,
    rmip_suite,
    run_script,
    uncertainty_generator,
)

FOOTPRINTS = {
    "run_script": run_script.FOOTPRINT,
    "forward_uq": forward_uq.FOOTPRINT,
    "graph_sa": graph_sa.FOOTPRINT,
    "mso_compare": mso_compare.FOOTPRINT,
    "fuse_kld": fuse_kld.FOOTPRINT,
    "mean_value_fusion": mean_value_fusion.FOOTPRINT,
    "brute_force_sa": brute_force_sa.FOOTPRINT,
    "idt_sensitivity": idt_sensitivity.FOOTPRINT,
    "nominal_sim": nominal_sim.FOOTPRINT,
    "uncertainty_generator": uncertainty_generator.FOOTPRINT,
    "mechanism_converter": mechanism_converter.FOOTPRINT,
    "mechanism_check": mechanism_check.FOOTPRINT,
    "rmip_suite": rmip_suite.FOOTPRINT,
}


def classify(rel_path: str, pipeline: Optional[str], abs_path: str) -> "tuple[str, str]":
    """Return (role, kind) for a path already known to be relative to a
    capture's working_dir. Tries the named pipeline's footprint first (if
    given and known), then falls back to extension-based classification --
    every file gets SOME role/kind, never a hard failure."""
    from . import _generic
    if pipeline and pipeline in FOOTPRINTS:
        for entry in FOOTPRINTS[pipeline]:
            if _generic.glob_match(rel_path, entry.pattern):
                return entry.role, (entry.kind or _generic.kind_of(abs_path))
    return "unclassified", _generic.kind_of(abs_path)
