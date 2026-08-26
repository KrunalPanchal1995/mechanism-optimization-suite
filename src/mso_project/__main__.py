"""Entry point for ``python -m mso_project``."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
