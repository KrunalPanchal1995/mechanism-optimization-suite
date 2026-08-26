"""
mso_io/mechanism_check_fix_examples.py
========================================
Worked "how do I fix this" examples per mechanism health-check finding
category (KRUNAL_JULY_23.md point 3: "make the errors thrown more
descriptive[.] Also give examples on how they can fix the issue raised.").

Factored out of ``mechanism_check_gui.py`` (the Tkinter original) so the
PySide6 port (``mso_studio.gui.modules.mechanism_check``) can render the
identical detail text without importing that module -- which pulls in
``rmip.analysis.mechanism_check`` (Cantera-dependent) at module level, never
available in the PySide6-only GUI venv. This module has zero framework/
Cantera dependencies: pure dict + string formatting, operating on the plain
``dict`` shape ``Finding.to_dict()``/JSON round-tripping produces, not the
``Finding`` dataclass itself.

Not exhaustive -- categories missing from ``FIX_EXAMPLES`` still render
fine, just without a canned example.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

FIX_EXAMPLES: Dict[str, str] = {
    "load-failure": (
        "The full ck2yaml/Cantera error is printed above under 'Underlying "
        "tool error'. Common causes and fixes:\n"
        "  • Fortran-style exponent missing 'E' — e.g. '1.7-13' should read "
        "'1.7E-13'. Run the Mechanism Converter with 'Auto-fix Fortran "
        "exponents' enabled, then re-check that output.\n"
        "  • 'Unexpected token' inside a reaction equation, e.g. "
        "'C2H4O1-2+OH' — usually a species name containing a hyphen got "
        "mis-split; check the SPECIES block still declares that exact "
        "spelling.\n"
        "  • 'undeclared participant'/'not in the SPECIES block' — add the "
        "missing species as its own line in SPECIES, or run the Mechanism "
        "Converter (it now auto-inserts genuinely-missing species).\n"
        "  • This file may not be Cantera YAML — CHEMKIN '.inp'/'.mech' "
        "files are converted via ck2yaml automatically before loading, so a "
        "convert failure surfaces here too."),
    "species-case-mismatch": (
        "e.g. SPECIES declares 'AR' but a reaction uses 'Ar' — CHEMKIN/"
        "Cantera treat these as different species. Rewrite every reaction "
        "usage to match the SPECIES-block spelling exactly (the 'auto' fix "
        "option below does this for you)."),
    "species-case-twins": (
        "Two SPECIES-block entries differ only in case (e.g. 'CH2' vs "
        "'ch2') — Cantera will silently treat them as two species. Rename "
        "one, or confirm they are genuinely distinct."),
    "undeclared-species-ref": (
        "A reaction references a species that's never declared. Add it as "
        "its own line in the SPECIES block (right before that block's END), "
        "or drop it from the reaction if it's a bookkeeping placeholder "
        "(e.g. chemiluminescence 'HV') rather than a real species."),
    "syntax": (
        "Line/column given above — open the file at that position (use "
        "'Open in editor' once available, or your own editor) and compare "
        "against a working reaction/thermo line nearby for the expected "
        "column layout."),
    "thermo-discontinuity": (
        "The low-T and high-T NASA-7 polynomial pieces disagree at Tmid "
        "(Cp/H/S jump). Refit the polynomial with a continuity constraint "
        "at Tmid, or move Tmid to a temperature both pieces already agree "
        "at."),
    "thermo-trange": (
        "A NASA-7 entry's [Tlow, Thigh] doesn't cover the screening grid — "
        "extrapolation risk. Get a wider-range fit, or restrict simulations "
        "to inside the declared range."),
    "unmarked-duplicate": (
        "Two reactions have the same rate-determining participants but "
        "neither is flagged DUPLICATE. Add 'DUPLICATE' under both entries "
        "if this is intentional (e.g. two pressure regimes), or remove the "
        "accidental copy."),
    "third-body-implied-duplicate": (
        "Two reactions look textually different but become the SAME "
        "physical reaction once the generic third body 'M' is instantiated "
        "-- e.g. a bare-M form ('H2+M<=>2H+M') and an explicit self-collider "
        "form of the same step ('2H+H2<=>2H2'). Cantera rejects this at "
        "load as an 'Undeclared duplicate reactions' error. Either mark "
        "both DUPLICATE (if intentional, matching Cantera's own semantics -- "
        "their rates get summed), or reconcile the source mechanisms so "
        "only one form is used."),
    "ea-magnitude": (
        "An activation energy is implausibly large/small for the declared "
        "units — the classic cal/mol vs J/kmol mixup. Check the REACTIONS "
        "header's unit keyword against how Ea was actually entered."),
    "element-imbalance": (
        "A reaction's element counts don't balance across the arrow. Fix "
        "the stoichiometric coefficients, or check for a typo'd species "
        "formula."),
    "charge-imbalance": (
        "Net charge differs between reactants and products — check for a "
        "missing/extra electron 'E' species or a mis-typed ion."),
    "missing-transport": (
        "A species has no transport (viscosity/diffusion) entry — fine for "
        "0-D kinetics, but flame/transport-coupled simulations will fail. "
        "Add a TRANSPORT entry, or borrow one from a chemically similar "
        "species as a placeholder."),
    "plog-negative-A": (
        "A PLOG pressure point has a negative pre-exponential factor — "
        "usually a sign error transcribing the rate expression from the "
        "source paper/supplement."),
    "plog-unsorted": (
        "PLOG pressure points must increase monotonically — reorder the "
        "lines so pressure increases top-to-bottom."),
    "troe-params": (
        "A Troe falloff's parameters are outside their physically-valid "
        "range (e.g. a=0 or a negative T***) — re-check against the "
        "source; some vendors omit the optional 4th (T2) parameter, which "
        "is fine, but the first 3 must still be sane."),
    "collision-limit": (
        "The fitted rate exceeds the gas-kinetic collision limit at some "
        "grid point — refit with a lower A-factor, or confirm the rate "
        "expression's units (a units error inflates A by orders of "
        "magnitude)."),
    "dead-reaction": (
        "This reaction's rate is ~0 everywhere on the screening grid — "
        "likely a zeroed/placeholder fit. Remove it, or supply a real rate "
        "if it's meant to contribute."),
    "decoupled-species": (
        "This species never appears as a product of any reaction (or "
        "vice versa) — it can never be formed/consumed. Check for a typo'd "
        "species name elsewhere, or remove it if it's genuinely unused."),
    "unused-species": (
        "Declared in SPECIES but never appears in any reaction. Harmless, "
        "but usually indicates a typo elsewhere (compare spellings) or "
        "leftover cruft from mechanism merging."),
}


def fix_example_for(category: str) -> Optional[str]:
    return FIX_EXAMPLES.get(category)


def build_detail_text_from_dict(mechanism: Optional[str], finding: Dict[str, Any]) -> str:
    """Same rendering as ``mechanism_check_gui.build_detail_text``, but
    operating on the plain-dict shape ``Finding.to_dict()``/JSON
    round-tripping produces, instead of the ``Finding`` dataclass -- for
    callers (the PySide6 port) that only ever see JSON, never the live
    Cantera-dependent objects."""
    lines = [
        f"Mechanism : {mechanism or '(unknown)'}",
        f"Check     : {finding.get('check', '')}   "
        f"Category: {finding.get('category', '')}   "
        f"Severity: {str(finding.get('severity', '')).upper()}",
        f"Location  : {finding.get('location', '')}",
        "",
        finding.get("message", ""),
    ]
    err = (finding.get("details") or {}).get("error")
    if err:
        lines += ["", "Underlying tool error", "-" * 22, str(err).rstrip()]
    example = fix_example_for(finding.get("category", ""))
    if example:
        lines += ["", "How to fix this", "-" * 16, example]
    fix_options = finding.get("fix_options") or []
    if fix_options:
        lines += ["", "Available auto/choice fixes:"]
        for o in fix_options:
            tag = "auto" if o.get("kind") == "auto" else o.get("kind", "")
            lines.append(f"  • [{tag}] {o.get('label', '')} — {o.get('description', '')}")
    return "\n".join(lines)
