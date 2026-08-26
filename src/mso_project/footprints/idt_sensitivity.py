"""Footprint for idt_sensitivity_gui.py / rmip.simulation.idt_sensitivity.

Unlike every other tool here, this one has NO fixed output directory --
results are saved via an interactive asksaveasfilename dialog to wherever
the user pointed it. There is nothing precise to glob for, so this
footprint is deliberately broad (any .json/.png/.pdf under the directory a
caller explicitly scan()s with pipeline="idt_sensitivity") rather than
tied to a specific subpath. Prefer Project.register() with an explicit role
for this pipeline's outputs when the exact save path is known at call time
(e.g. right after the GUI's own save dialog returns a path) instead of
relying on this fallback footprint."""
from __future__ import annotations

from . import FootprintEntry

FOOTPRINT = [
    FootprintEntry("*.json", "idt_sensitivity_result"),
    FootprintEntry("*.png", "idt_sensitivity_figure"),
    FootprintEntry("*.pdf", "idt_sensitivity_figure"),
]
