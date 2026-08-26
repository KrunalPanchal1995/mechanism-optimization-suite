"""
rmip/generation/rmg_lint.py
===========================
W7c — static linter for RMG job input files (P39).

RMG silently accepts input files that later ruin a (potentially days-long)
run: a library listed in both ``reactionLibraries`` and ``seedMechanisms``
duplicates every one of its reactions in the generated model (RMG-Py issues
#2979 / #2882); duplicate entries within one list (#2485) do the same; and
inverted interrupt/move-to-core tolerances quietly change the enlargement
behaviour.  ``lint_rmg_input`` catches these **before** the run starts.

The input file is Python, so it is parsed with ``ast`` — never executed —
and the ``database(...)`` / ``model(...)`` call keywords are read
statically.  Only literal entries can be checked; dynamically-built lists
are reported as unlintable rather than silently passed.

Emits mechanism_check-style ``Finding``s (check id ``rmg_lint``).
"""

from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional


def _literal(node: ast.AST) -> Optional[Any]:
    """Literal value of an AST node, or None if it isn't a literal."""
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None


def _library_names(value: Any) -> Optional[List[str]]:
    """Normalise a libraries list — entries may be strings or
    ``('name', bool)`` tuples — to a list of names.  None if not a list of
    literals."""
    if not isinstance(value, (list, tuple)):
        return None
    names = []
    for entry in value:
        if isinstance(entry, str):
            names.append(entry)
        elif (isinstance(entry, (list, tuple)) and entry
              and isinstance(entry[0], str)):
            names.append(entry[0])
        else:
            return None
    return names


def _calls_named(tree: ast.AST, name: str) -> List[ast.Call]:
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == name]


def lint_rmg_input(path: str) -> List[Any]:
    """Statically lint an RMG ``input.py``. Returns a list of ``Finding``s
    (empty = clean)."""
    from rmip.analysis.mechanism_check import Finding

    findings: List[Finding] = []
    with open(path) as fh:
        source = fh.read()

    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        return [Finding(
            check="rmg_lint", severity="error", category="syntax-error",
            message=f"RMG input file is not valid Python: {exc}",
            location=f"{path}:{exc.lineno}")]

    # ---- database(...) ---------------------------------------------------
    db_calls = _calls_named(tree, "database")
    if not db_calls:
        findings.append(Finding(
            check="rmg_lint", severity="warning", category="missing-block",
            message="No database(...) call found — RMG will fail at startup.",
            location=path))
    list_kwargs = ("thermoLibraries", "reactionLibraries", "seedMechanisms",
                   "kineticsDepositories")
    lib_lists: Dict[str, List[str]] = {}
    for call in db_calls:
        for kw in call.keywords:
            if kw.arg not in list_kwargs:
                continue
            value = _literal(kw.value)
            names = _library_names(value)
            if names is None:
                if value is None:
                    findings.append(Finding(
                        check="rmg_lint", severity="info",
                        category="unlintable",
                        message=f"database({kw.arg}=...) is built dynamically "
                                "— cannot lint its entries statically.",
                        location=f"{path}:{kw.value.lineno}"))
                continue
            lib_lists[kw.arg] = names

            # -- duplicates within one list (#2485) ------------------------
            seen: Dict[str, int] = {}
            for n in names:
                seen[n] = seen.get(n, 0) + 1
            for n, count in seen.items():
                if count > 1:
                    findings.append(Finding(
                        check="rmg_lint", severity="error",
                        category="duplicate-library",
                        message=(f"Library '{n}' is listed {count}x in "
                                 f"{kw.arg} — its content will be duplicated "
                                 "in the generated model (RMG-Py #2485)."),
                        location=f"{path}:{kw.value.lineno}",
                        details={"library": n, "list": kw.arg,
                                 "count": count}))

    # ---- reactionLibraries ∩ seedMechanisms (#2979 / #2882) --------------
    overlap = set(lib_lists.get("reactionLibraries", [])) \
        & set(lib_lists.get("seedMechanisms", []))
    for n in sorted(overlap):
        findings.append(Finding(
            check="rmg_lint", severity="error", category="dual-listed-library",
            message=(f"'{n}' is listed in BOTH reactionLibraries and "
                     "seedMechanisms — every reaction in it will appear "
                     "twice in the generated mechanism (RMG-Py #2979/#2882). "
                     "Keep it in one list only."),
            location=path, details={"library": n}))

    # ---- model(...) tolerances -------------------------------------------
    for call in _calls_named(tree, "model"):
        tol: Dict[str, float] = {}
        for kw in call.keywords:
            v = _literal(kw.value)
            if isinstance(v, (int, float)):
                tol[kw.arg] = float(v)
        move = tol.get("toleranceMoveToCore")
        interrupt = tol.get("toleranceInterruptSimulation")
        if move is not None and move <= 0:
            findings.append(Finding(
                check="rmg_lint", severity="error", category="bad-tolerance",
                message=f"toleranceMoveToCore = {move:g} must be positive.",
                location=f"{path}:{call.lineno}"))
        if (move is not None and interrupt is not None
                and interrupt < move):
            findings.append(Finding(
                check="rmg_lint", severity="warning",
                category="tolerance-order",
                message=(f"toleranceInterruptSimulation ({interrupt:g}) < "
                         f"toleranceMoveToCore ({move:g}) — simulations "
                         "will be interrupted before species can exceed the "
                         "move-to-core threshold, which usually stalls "
                         "enlargement. Conventionally interrupt ≥ move."),
                location=f"{path}:{call.lineno}",
                details={"toleranceMoveToCore": move,
                         "toleranceInterruptSimulation": interrupt}))
    if not _calls_named(tree, "model"):
        findings.append(Finding(
            check="rmg_lint", severity="warning", category="missing-block",
            message="No model(...) call found — RMG will fail at startup.",
            location=path))

    return findings
