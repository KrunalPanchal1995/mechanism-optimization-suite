"""
Root conftest for the consolidated mechanism-optimization suite.

The RMIP package now lives at ``src/rmip`` but still imports itself internally as
``rmip`` (e.g. ``from rmip.core.parser import MechanismParser``). To make those
imports resolve when running ``pytest`` from the repository root, we put ``src``
on ``sys.path``.

IMPORTANT — append, never insert at index 0:
    ``src/io`` is a top-level package literally named ``io``. If ``src`` were
    inserted at the *front* of ``sys.path`` it would shadow the standard-library
    ``io`` module and break unrelated code. Appending keeps stdlib precedence
    while still making ``rmip`` importable (nothing else on the path is named
    ``rmip``).
"""

import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.append(_SRC)
