"""
sys.path shim so every module under validation/ (and validation/search_history/,
validation/report_figures/, and tests/) can `import ResponseSurface`,
`import IdentifiablePRS`, `import VisualAid`, etc. -- the "production" modules
that live one level up, directly in the MUQ-SAC tree root -- regardless of the
caller's cwd or how deep in this package it lives.

This file is resolved from ITS OWN __file__ (never from cwd), so it works
identically:
  - run directly:            python3.9 validation/kfold_cv.py
  - run from a foreign cwd:  cd /; python3.9 /path/to/validation/kfold_cv.py
  - imported by a sibling:   validation/search_history/kfold_robust_search.py
  - imported by tests/:      tests/conftest.py

Layout assumption (true both in test_build/ today and in src/MUQ-SAC/ once
this tree is mirrored there):

    <MUQ-SAC-root>/                <- ResponseSurface.py, IdentifiablePRS.py, ... live here
    <MUQ-SAC-root>/validation/_bootstrap.py                (this file)
    <MUQ-SAC-root>/validation/search_history/*.py
    <MUQ-SAC-root>/validation/report_figures/*.py
    <MUQ-SAC-root>/tests/*.py

so the MUQ-SAC root is: two levels up from search_history/report_figures/tests
files that import this module via `validation/_bootstrap.py`, but since this
shim only ever needs to know ITS OWN location, the rule is simply "the parent
directory of the directory this file lives in" -- i.e. os.path.dirname twice
from _bootstrap.py's own path, no matter who imports it.
"""
import os
import sys

# Directory this file lives in: <MUQ-SAC-root>/validation
_VALIDATION_DIR = os.path.dirname(os.path.abspath(__file__))
# The MUQ-SAC root itself: <MUQ-SAC-root>  (== test_build/ today, src/MUQ-SAC/
# once this tree is mirrored there per the plan's Part 1 final step)
MUQSAC_ROOT = os.path.dirname(_VALIDATION_DIR)

# As of 2026-07-31, NOT every production module is mirrored between
# test_build/ and src/MUQ-SAC/ -- specifically VisualAid.py currently exists
# ONLY under src/MUQ-SAC/ (the original test_build/plot_band_violation_samples.py
# had TWO hardcoded sys.path.insert calls for exactly this reason: one for
# test_build/, one for src/MUQ-SAC/, so it could `import VisualAid`). This
# task's scope explicitly excludes writing/modifying anything under
# src/MUQ-SAC/, so the fix here is READ-ONLY: also make the sibling
# src/MUQ-SAC/ directory importable (never written to), the same way the
# original script's second sys.path.insert did. When MUQSAC_ROOT is itself
# test_build/, that sibling is <repo_root>/src/MUQ-SAC; when this tree is
# later mirrored so MUQSAC_ROOT IS src/MUQ-SAC, the computed "sibling" is
# identical to MUQSAC_ROOT and the extra entry is simply a no-op duplicate
# (skipped by the `not in sys.path` guard).
_REPO_ROOT_GUESS = os.path.dirname(MUQSAC_ROOT)
_SRC_MUQSAC_SIBLING = os.path.join(_REPO_ROOT_GUESS, "src", "MUQ-SAC")


def add_muqsac_to_path():
    """Idempotently APPEND the MUQ-SAC root (and, if it exists and differs,
    the src/MUQ-SAC/ sibling -- see module docstring) to sys.path so `import
    ResponseSurface`, `import IdentifiablePRS`, `import AdaptiveResampler`,
    `import VisualAid`, etc. resolve to the production modules -- not some
    other copy that might also be importable.

    Deliberately appended, not prepended: MUQ-SAC-root (test_build/, or
    src/MUQ-SAC/ once mirrored) ALSO contains flat, same-named files
    (kfold_cv.py, etc.) that predate this validation/ package. If this were
    inserted at sys.path[0], it would take priority over the calling
    script's own directory and silently shadow validation/kfold_cv.py,
    validation/case_data.py, validation/config.py, etc. with the old flat
    originals -- exactly the "wrong tree" bug the plan's import-hygiene
    verification step is designed to catch. Appending guarantees the
    caller's own directory (already at sys.path[0] for a directly-run
    script, or wherever conftest.py/pytest put it) wins for any name that
    exists in both places, while production-only names (ResponseSurface,
    IdentifiablePRS, VisualAid, AdaptiveResampler, ...) still resolve fine
    since they don't exist under validation/ at all."""
    if MUQSAC_ROOT not in sys.path:
        sys.path.append(MUQSAC_ROOT)
    if os.path.isdir(_SRC_MUQSAC_SIBLING) and _SRC_MUQSAC_SIBLING not in sys.path:
        sys.path.append(_SRC_MUQSAC_SIBLING)
    return MUQSAC_ROOT


# Convenience: importing this module is enough to get the path wired, so
# `import _bootstrap` alone (no explicit call) also works for callers that
# prefer that style.
add_muqsac_to_path()
