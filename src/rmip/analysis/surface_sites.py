"""
rmip/analysis/surface_sites.py
==============================
P4 — surface-site balancing (shared by W2 converter preflight and W5).

Cantera (>= 3.0) *rejects* a surface mechanism at load when a reaction's
surface-site count differs between the two sides ("Number of surface sites
not balanced in reaction …").  The standard repair — what mechanism authors
do by hand — is to add the *empty site* species to the deficient side, which
balances sites without touching element balance (the empty site has no
elemental composition).

``balance_surface_sites`` performs that repair on the **parsed YAML dict**
(pre-load, like the other raw-YAML fixes in mechanism_fixer), so an
unloadable mechanism becomes loadable:

- site count per side = Σ stoich_coeff × sites(species) over the surface
  phase's species (``sites:`` key of the species entry, default 1);
- the deficit is appended as ``n <empty-site>`` to the lighter side;
- the empty site is auto-detected as a surface species with an empty
  ``composition: {}``; if none exists, a placeholder ``(v)`` (vacant site,
  empty composition, zero constant-cp thermo, 1 site) is synthesized and
  added to the phase.

Only reactions whose *sites* are unbalanced are touched; element balance is
never affected (the inserted species has no elements).  Equations are
assumed to use Cantera's spaced form (``A + 2 B => C``), which is what
ck2yaml and all YAML writers emit.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional, Tuple

_ARROW_RE = re.compile(r"(<=>|=>|(?<![<>=])=(?![<>=]))")
_COEFF_RE = re.compile(r"^(\d+(?:\.\d+)?)\s+(.+)$")


def _split_equation(eq: str) -> Tuple[str, str, str]:
    """-> (lhs, arrow, rhs). Raises ValueError on an unparseable equation."""
    m = _ARROW_RE.search(eq)
    if not m:
        raise ValueError(f"No reaction arrow found in equation: {eq!r}")
    arrow = m.group(1)
    lhs, rhs = eq.split(arrow, 1)
    return lhs.strip(), arrow, rhs.strip()


def _side_terms(side: str) -> List[Tuple[float, str]]:
    """Parse one side into [(coeff, species), ...] (spaced-'+' separated)."""
    out = []
    for term in re.split(r"\s\+\s|(?<=\s)\+\s|\s\+(?=\s)", side):
        term = term.strip()
        if not term:
            continue
        m = _COEFF_RE.match(term)
        if m:
            out.append((float(m.group(1)), m.group(2).strip()))
        else:
            out.append((1.0, term))
    return out


def _fmt_coeff(c: float) -> str:
    return str(int(c)) if float(c).is_integer() else f"{c:g}"


def _site_count(terms, sites_by_species: Dict[str, float]) -> float:
    return sum(c * sites_by_species[name]
               for c, name in terms if name in sites_by_species)


def balance_surface_sites(
    doc: Dict[str, Any],
    surface_phase: Optional[str] = None,
    empty_site: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Balance surface-reaction site counts in a parsed mechanism YAML dict.

    Returns ``(new_doc, notes)`` — the input dict is not modified; ``notes``
    describes every change (one entry per rebalanced reaction, plus one if a
    placeholder empty-site species was synthesized).  An already-balanced
    mechanism comes back unchanged with empty notes.
    """
    doc = copy.deepcopy(doc)
    notes: List[str] = []

    phases = doc.get("phases", []) or []
    surf = None
    if surface_phase:
        surf = next((p for p in phases if p.get("name") == surface_phase), None)
        if surf is None:
            raise ValueError(f"No phase named {surface_phase!r} in the document")
    else:
        cands = [p for p in phases
                 if "surface" in str(p.get("thermo", ""))
                 or "edge" in str(p.get("thermo", ""))]
        if len(cands) != 1:
            raise ValueError(
                f"Cannot auto-detect the surface phase (candidates: "
                f"{[p.get('name') for p in cands] or 'none'}); pass "
                "surface_phase= explicitly.")
        surf = cands[0]

    surf_species = list(surf.get("species", []) or [])
    species_entries = {s.get("name"): s for s in doc.get("species", []) or []}
    sites_by_species = {
        name: float(species_entries.get(name, {}).get("sites", 1) or 1)
        for name in surf_species}

    # ---- pick / synthesize the empty site ------------------------------
    if empty_site is None:
        empties = [n for n in surf_species
                   if not (species_entries.get(n, {}).get("composition") or {})]
        empty_site = empties[0] if empties else None
    synthesized = False
    if empty_site is None:
        empty_site = "(v)"
        synthesized = True   # added lazily, only if a repair actually needs it
    elif empty_site not in sites_by_species:
        raise ValueError(f"Empty-site species {empty_site!r} is not in the "
                         f"surface phase's species list")
    empty_sites_per = sites_by_species.get(empty_site, 1.0)

    # ---- reaction sections of this phase -------------------------------
    sections = surf.get("reactions", ["reactions"])
    if isinstance(sections, str):
        sections = [sections]
    if sections in (["all"], ["declared-species"]):
        sections = ["reactions"]

    changed = False
    for sec in sections:
        for rxn in doc.get(sec, []) or []:
            eq = rxn.get("equation")
            if not eq:
                continue
            try:
                lhs, arrow, rhs = _split_equation(eq)
            except ValueError:
                continue
            lterms, rterms = _side_terms(lhs), _side_terms(rhs)
            nl = _site_count(lterms, sites_by_species)
            nr = _site_count(rterms, sites_by_species)
            if nl == nr:
                continue
            deficit = abs(nl - nr) / empty_sites_per
            add = f"{_fmt_coeff(deficit)} {empty_site}" if deficit != 1.0 \
                else empty_site
            if nl < nr:
                lhs = f"{lhs} + {add}"
            else:
                rhs = f"{rhs} + {add}"
            new_eq = f"{lhs} {arrow} {rhs}"
            rxn["equation"] = new_eq
            changed = True
            notes.append(
                f"site balance {_fmt_coeff(nl)} vs {_fmt_coeff(nr)}: "
                f"'{eq}' -> '{new_eq}'")

    if changed and synthesized:
        doc.setdefault("species", []).append({
            "name": empty_site,
            "composition": {},
            "sites": 1,
            "thermo": {"model": "constant-cp", "T0": 298.15,
                       "h0": "0.0 kJ/mol", "s0": "0.0 J/mol/K",
                       "cp0": "0.0 J/mol/K"},
        })
        surf.setdefault("species", []).append(empty_site)
        notes.append(
            f"synthesized empty-site placeholder species '{empty_site}' "
            "(empty composition, zero thermo, 1 site) in phase "
            f"'{surf.get('name')}'")

    return doc, notes
