"""Footprint for mechanism_check_gui.py. Both outputs are user-driven
dialogs (askdirectory / asksaveasfilename) with no fixed default location
-- as with idt_sensitivity, prefer an explicit Project.register() call
right after the GUI's own dialog returns a path when possible."""
from __future__ import annotations

from . import FootprintEntry

FOOTPRINT = [
    FootprintEntry("**/*.fixed.yaml", "mechanism_check_fixed"),
    FootprintEntry("**/*.json", "mechanism_check_report"),
]
