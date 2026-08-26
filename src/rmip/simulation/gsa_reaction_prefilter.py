"""
rmip/simulation/gsa_reaction_prefilter.py
==========================================
MarkDownFiles/ZERORK_GSA_PLAN.md Phase 4's deferred "wire GSA screening into
the PRS/Forward-UQ pipeline" item: an opt-in prefilter that restricts
``src/MUQ-SAC/run_script.py``'s ``unsrt_data`` dict (reaction-formula string
-> ``Uncertainty.reaction``) to the subset of reactions an
``rmip gsa-run`` Sobol/Morris screening judged "important", BEFORE that dict
is handed to ``DesignMatrix``/the PRS training loop.

Hard requirement (confirmed with KP): default OFF, provably a no-op when not
configured. See ``apply_gsa_prefilter`` -- it is the single seam
``run_script.py`` calls through (see its ``_maybe_apply_gsa_prefilter``), and
it returns ``unsrt_data`` completely unchanged whenever the caller's config
dict is falsy/empty or explicitly ``enabled: false``.

Reaction-key formats reconciled here
-------------------------------------
- ``unsrt_data`` (MUQ-SAC) is keyed by bare Cantera equation strings, e.g.
  ``"H + O2 <=> O + OH"`` (see ``Uncertainty.py``'s ``reaction.__init__``,
  which reads the XML ``rxn=`` attribute verbatim).
- An ``rmip gsa-run`` JSON result (``SobolResult.save()``, see
  ``rmip/types.py``) is keyed by ``ZerorkGSARunner._all_reactions()``'s
  ``f"{reaction_label}:{formula_string}"`` (e.g. ``"R3:O + H2 <=> H + OH"``)
  -- the label prefix guarantees uniqueness even for DUPLICATE-tagged
  reactions sharing one ``formula_string``. ``select_gsa_important_reactions``
  strips that ``R<n>:`` prefix so its returned set matches ``unsrt_data``'s
  bare-formula keys.
- ``build_reaction_id_maps``'s DB mode mirrors
  ``rmip/simulation/gsa_prs_crosscheck.py::run_crosscheck``'s exact query
  (``SELECT id, formula_string FROM reactions WHERE mechanism_id=?``) --
  that proof-of-concept is the pattern this generalizes, not reinvents. File
  mode exists because production ``run_script.py`` has no rmip DB at all,
  only a ``Locations.mechanism`` file path (see that key's use in
  ``run_script.py`` around its ``mech_file_location`` assignment) -- it loads
  a Cantera ``Solution`` directly and assigns synthetic 1-based ids in
  ``Solution.reactions()`` order. This numbering is NOT guaranteed to match a
  real DB's ``id`` column in general (DB ids are global-autoincrement across
  every mechanism in the database, not reset to 1 per mechanism) -- it is
  only guaranteed to match a DB whose reactions for that particular
  mechanism_id were inserted in file order starting at id=1 (true for a
  freshly-populated single-mechanism DB; see
  ``test_gsa_reaction_prefilter.py``'s DB-vs-file-mode equivalence test for
  the exact construction this holds for).
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set

log = logging.getLogger("rmip.simulation.gsa_reaction_prefilter")

_VALID_METRICS = ("S1", "ST", "mu_star")
_DEFAULT_THRESHOLD = 0.01
_LABEL_PREFIX_RE = re.compile(r"^R\d+:")


@dataclass
class ReactionIdMaps:
    id_of_formula: Dict[str, int]
    formula_of_id: Dict[int, str]


def _strip_label_prefix(name: str) -> str:
    """``"R3:O + H2 <=> H + OH"`` -> ``"O + H2 <=> H + OH"``. Names that
    already lack the ``R<n>:`` label prefix (e.g. a caller who hands in bare
    formula strings directly) pass through unchanged."""
    return _LABEL_PREFIX_RE.sub("", name, count=1)


def build_reaction_id_maps(
    db_path: Optional[str] = None,
    mechanism_id: Optional[int] = None,
    mechanism_path: Optional[str] = None,
) -> ReactionIdMaps:
    """Build a bidirectional reaction-id <-> formula-string map.

    Exactly one of ``(db_path AND mechanism_id)`` or ``mechanism_path`` must
    be given.

    DB mode mirrors ``gsa_prs_crosscheck.py::run_crosscheck``'s exact query.
    File mode loads a Cantera ``Solution`` from ``mechanism_path`` directly
    (no rmip DB involved) and assigns 1-based ids in ``Solution.reactions()``
    order -- see the module docstring for the id-numbering caveat.
    """
    db_mode = db_path is not None and mechanism_id is not None
    file_mode = mechanism_path is not None
    if db_mode and file_mode:
        raise ValueError(
            "build_reaction_id_maps: provide exactly one of "
            "(db_path AND mechanism_id) or mechanism_path -- not both"
        )
    if not db_mode and not file_mode:
        if (db_path is None) != (mechanism_id is None):
            raise ValueError(
                "build_reaction_id_maps: db_path and mechanism_id must be "
                "given together"
            )
        raise ValueError(
            "build_reaction_id_maps: provide exactly one of "
            "(db_path AND mechanism_id) or mechanism_path"
        )

    if file_mode:
        import cantera as ct

        sol = ct.Solution(mechanism_path)
        id_of_formula: Dict[str, int] = {}
        formula_of_id: Dict[int, str] = {}
        for i, rxn in enumerate(sol.reactions()):
            rid = i + 1
            id_of_formula[rxn.equation] = rid
            formula_of_id[rid] = rxn.equation
        return ReactionIdMaps(id_of_formula, formula_of_id)

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, formula_string FROM reactions WHERE mechanism_id=?",
            (mechanism_id,),
        ).fetchall()
    id_of_formula = {formula: rid for rid, formula in rows}
    formula_of_id = {rid: formula for rid, formula in rows}
    return ReactionIdMaps(id_of_formula, formula_of_id)


def select_gsa_important_reactions(
    gsa_result_path: str,
    metric: str = "ST",
    threshold: Optional[float] = None,
    top_n: Optional[int] = None,
) -> Set[str]:
    """Read an ``rmip gsa-run`` JSON output (``SobolResult.save()``'s shape,
    see ``rmip/types.py``) and return the set of reaction-formula keys
    (label prefix stripped, see ``_strip_label_prefix``) judged important.

    ``threshold`` and ``top_n`` are mutually exclusive: passing both
    explicitly (both not None) raises ``ValueError`` rather than silently
    picking one -- a caller who set both almost certainly meant only one and
    silently ignoring the other would be a worse failure mode than a loud
    error. Passing neither (both None, the default) uses the default
    threshold (0.01) -- NOT a default of ``threshold=0.01`` in the function
    signature itself, precisely so that a caller who passes only ``top_n``
    doesn't spuriously trip the "both given" check above.
    """
    if threshold is not None and top_n is not None:
        raise ValueError(
            "select_gsa_important_reactions: pass only one of threshold or "
            "top_n, not both"
        )
    if metric not in _VALID_METRICS:
        raise ValueError(
            f"select_gsa_important_reactions: metric must be one of "
            f"{_VALID_METRICS}, got {metric!r}"
        )

    with open(gsa_result_path) as fh:
        data = json.load(fh)
    values = data.get(metric)
    if not values:
        raise ValueError(
            f"GSA result at {gsa_result_path!r} has no {metric!r} values "
            f"(method={data.get('method')!r}) -- Sobol results only "
            f"populate S1/ST, Morris results only populate mu_star"
        )

    stripped = {_strip_label_prefix(name): float(v) for name, v in values.items()}

    if top_n is not None:
        ranked = sorted(stripped.items(), key=lambda kv: kv[1], reverse=True)
        return {name for name, _ in ranked[:top_n]}

    thr = threshold if threshold is not None else _DEFAULT_THRESHOLD
    return {name for name, v in stripped.items() if v >= thr}


def apply_gsa_prefilter(unsrt_data: dict, prefilter_cfg: dict) -> dict:
    """THE opt-in seam ``run_script.py`` calls through.

    Returns ``unsrt_data`` completely UNCHANGED (the same object) whenever
    ``prefilter_cfg`` is falsy/empty, or truthy but has ``enabled: false`` --
    this is the default/disabled path and is provably a no-op: every branch
    that would filter/read a GSA result file is unreachable when this
    condition is true.

    When enabled, reads the GSA result at ``prefilter_cfg["gsa_result_json"]``
    (metric = ``prefilter_cfg.get("metric", "ST")``, selection =
    ``threshold``/``top_n`` from the config, see
    ``select_gsa_important_reactions`` for the default/mutual-exclusion
    rules) and returns a NEW dict containing only the ``unsrt_data`` entries
    whose key survived selection.

    Raises ``ValueError`` (not a silent empty dict) if the GSA result's
    reaction keys have ZERO overlap with ``unsrt_data``'s keys at all -- a
    guard against a reaction-formula format mismatch between the GSA result
    and the uncertainty XML (e.g. wrong mechanism, or a GSA result keyed by
    canonical_key while unsrt_data is keyed by formula_string). A threshold
    that legitimately excludes every reaction (overlap exists, but nothing
    clears the bar) is a valid, if unusual, configuration choice and returns
    an empty dict rather than raising -- that is a user's explicit choice,
    not a format bug.
    """
    if not prefilter_cfg or not prefilter_cfg.get("enabled", True):
        return unsrt_data

    gsa_result_path = prefilter_cfg["gsa_result_json"]
    metric = prefilter_cfg.get("metric", "ST")
    threshold = prefilter_cfg.get("threshold")
    top_n = prefilter_cfg.get("top_n")
    if threshold is None and top_n is None:
        threshold = _DEFAULT_THRESHOLD

    with open(gsa_result_path) as fh:
        gsa_data = json.load(fh)
    values = gsa_data.get(metric) or {}
    all_gsa_keys = {_strip_label_prefix(name) for name in values.keys()}
    if not (all_gsa_keys & set(unsrt_data.keys())):
        raise ValueError(
            f"GSA prefilter: zero overlap between the GSA result's reaction "
            f"keys and unsrt_data's keys -- likely a reaction-formula format "
            f"mismatch (wrong mechanism, or a GSA result keyed differently "
            f"than the uncertainty XML). GSA keys sample: "
            f"{sorted(all_gsa_keys)[:3]!r}; unsrt_data keys sample: "
            f"{sorted(unsrt_data.keys())[:3]!r}"
        )

    important = select_gsa_important_reactions(
        gsa_result_path, metric=metric, threshold=threshold, top_n=top_n,
    )
    return {k: v for k, v in unsrt_data.items() if k in important}
