#!/usr/bin/env python3
"""
Repo-root launcher for the RMIP command-line interface.

RMIP lives at ``src/rmip`` but imports itself as ``rmip``. Run it through this
launcher from the repository root:

    python rmip_cli.py list --db project.db
    python rmip_cli.py visualize 1 2 --union --db project.db

Do NOT ``cd src`` and run ``python -m rmip.cli`` — ``src/io`` is a package named
``io`` and, as the current working directory, it shadows the standard-library
``io`` module and crashes the interpreter at start-up. This launcher *appends*
``src`` to ``sys.path`` so ``rmip`` resolves while stdlib keeps precedence.
"""

import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.append(_SRC)

from rmip.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
