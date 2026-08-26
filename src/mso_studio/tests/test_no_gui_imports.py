"""Mechanically enforces the core/gui split from the project's GUI guidelines
doc, Part 2: "core/ never imports the GUI toolkit, and gui/ never contains
an equation." This test covers the first half.
"""
import re
from pathlib import Path

CORE_DIR = Path(__file__).resolve().parents[1] / "core"
# Only real `import X` / `from X import ...` statements count -- mentions of
# "PySide6" in docstrings/comments/string literals (e.g. the optional
# version-probe in runrecord.software_versions) are not a layering violation.
_FORBIDDEN_IMPORT = re.compile(
    r"^\s*(import|from)\s+(PySide6|PyQt5|PyQt6)\b"
)


def test_core_has_no_qt_imports():
    offenders = []
    for py_file in CORE_DIR.rglob("*.py"):
        for lineno, line in enumerate(py_file.read_text().splitlines(), start=1):
            if _FORBIDDEN_IMPORT.match(line):
                offenders.append(f"{py_file.relative_to(CORE_DIR.parent)}:{lineno}: {line.strip()}")
    assert not offenders, "core/ must not import Qt:\n" + "\n".join(offenders)
