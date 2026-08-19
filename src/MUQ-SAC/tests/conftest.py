"""
pytest bootstrap: adds the MUQ-SAC root (the directory this tests/ package
lives directly under -- test_build/ today, src/MUQ-SAC/ once mirrored) to
sys.path, so both `pytest tests/` and `python3.9 tests/test_doptimal.py`
work identically, from any cwd.

Resolved from THIS file's own __file__ (never from cwd), same pattern as
validation/_bootstrap.py.
"""
import os
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
MUQSAC_ROOT = os.path.dirname(_TESTS_DIR)

if MUQSAC_ROOT not in sys.path:
    sys.path.append(MUQSAC_ROOT)
