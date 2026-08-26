"""
rmip/generation/rmg_diff.py
===========================
W7d — RMG reproducibility diff (P15, shared with W8).

"RMG is not reproducible across versions" is only actionable once it is a
*concrete diff*: which species/reactions appeared, which vanished, which
rates changed.  This module aligns two RMG output directories and reports
exactly that.

The hard part is identity: RMG's species labels carry run-dependent indices
(``C2H5(3)`` in one run may be ``C2H5(7)`` in the next), so label equality
is meaningless across runs.  Like ``analysis/graph_comparator`` (which
matches by InChI through the RMIP DB), identity here is **structural**, but
computed directly from the run artifacts with no RMG or RDKit dependency:
each species' adjacency list in ``species_dictionary.txt`` is canonicalised
with a Morgan-style iterative-refinement hash (element/valence-electron
colors refined over bond-typed neighborhoods), so isomorphic graphs — and
only those — compare equal.  Species that have no dictionary entry (inerts
like N2 injected by the reactor block) fall back to their label.

Reactions are aligned by the multiset of canonical reactant/product keys
(direction-insensitive for reversible reactions), then their Arrhenius
parameters from the CHEMKIN output are compared.

Inputs are RMG run directories (``chemkin/chem_annotated.inp`` +
``chemkin/species_dictionary.txt`` auto-located) or direct file paths.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# species_dictionary.txt parsing + canonical graph key
# ---------------------------------------------------------------------------
def parse_species_dictionary(path: str) -> Dict[str, str]:
    """-> {label: adjacency-list text block} (blocks separated by blanks)."""
    out: Dict[str, str] = {}
    block: List[str] = []
    with open(path, errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.strip():
                block.append(line)
            elif block:
                out[block[0].strip()] = "\n".join(block)
                block = []
    if block:
        out[block[0].strip()] = "\n".join(block)
    return out


_ATOM_RE = re.compile(
    r"^(\d+)\s+\*?\d*\s*([A-Z][a-z]?)\s+(u\d+)\s+(p\d+)\s+(c[+-]?\d+)(.*)$")
_BOND_RE = re.compile(r"\{(\d+),([SDTBQ])\}")


def canonical_graph_key(adjlist: str) -> str:
    """Morgan-refinement canonical hash of an RMG adjacency list.

    Isomorphic molecular graphs (same atoms, bonds, radical/lone-pair/charge
    state, multiplicity) hash equal regardless of atom numbering or label.
    """
    multiplicity = ""
    atoms: Dict[int, Tuple[str, str, str, str]] = {}
    bonds: Dict[int, List[Tuple[int, str]]] = {}
    for line in adjlist.splitlines()[1:]:                # skip the label line
        s = line.strip()
        if s.startswith("multiplicity"):
            multiplicity = s.split()[-1]
            continue
        m = _ATOM_RE.match(s)
        if not m:
            continue
        idx = int(m.group(1))
        atoms[idx] = (m.group(2), m.group(3), m.group(4), m.group(5))
        bonds[idx] = [(int(b), o) for b, o in _BOND_RE.findall(m.group(6))]

    if not atoms:
        # No parseable atoms — hash the raw text minus the label.
        body = "\n".join(adjlist.splitlines()[1:])
        return "raw:" + hashlib.md5(body.encode()).hexdigest()[:16]

    # Initial colors: intrinsic atom state.
    color = {i: hashlib.md5(("|".join(atoms[i])).encode()).hexdigest()
             for i in atoms}
    # Refine over bond-typed neighborhoods; len(atoms) rounds guarantees
    # stabilisation for these molecule sizes.
    for _ in range(max(3, min(len(atoms), 12))):
        color = {
            i: hashlib.md5((
                color[i] + "&" + ",".join(sorted(
                    f"{order}:{color[j]}" for j, order in bonds[i]))
            ).encode()).hexdigest()
            for i in atoms}
    fingerprint = multiplicity + ";" + ",".join(sorted(color.values()))
    return hashlib.md5(fingerprint.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# CHEMKIN output parsing (RMG-generated files are regular)
# ---------------------------------------------------------------------------
@dataclass
class ChemkinReaction:
    equation: str
    reactants: Tuple[str, ...]        # species labels, expanded by coeff
    products: Tuple[str, ...]
    reversible: bool
    A: float
    n: float
    Ea: float                          # in the file's units (compared as-is)


_RXN_LINE_RE = re.compile(
    r"^(.+?)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*$")


def _parse_side(side: str) -> Tuple[List[str], bool]:
    """One side of a CHEMKIN equation -> (expanded species labels, had_M)."""
    labels: List[str] = []
    had_m = False
    side = re.sub(r"\(\+M\)", "", side, flags=re.IGNORECASE)
    for term in side.split("+"):
        term = term.strip()
        if not term:
            continue
        if term.upper() == "M":
            had_m = True
            continue
        m = re.match(r"^(\d+)\s*(.+)$", term)
        if m and not re.match(r"^\d", m.group(2)):
            labels.extend([m.group(2).strip()] * int(m.group(1)))
        else:
            labels.append(term)
    return labels, had_m


def parse_chemkin(path: str) -> Tuple[List[str], List[ChemkinReaction]]:
    """-> (species labels, reactions) from a CHEMKIN mechanism file.

    Only top-level reaction lines are read (LOW/TROE/PLOG/DUPLICATE
    auxiliary lines are skipped) — enough to align reactions and compare
    their lead Arrhenius parameters.
    """
    species: List[str] = []
    reactions: List[ChemkinReaction] = []
    section = None
    with open(path, errors="replace") as fh:
        for raw in fh:
            line = raw.split("!", 1)[0].rstrip()
            s = line.strip()
            if not s:
                continue
            up = s.upper()
            if up.startswith("SPECIES"):
                section = "species"
                rest = s[len("SPECIES"):].strip()
                if rest:
                    species.extend(rest.split())
                continue
            if up.startswith("REACTIONS"):
                section = "reactions"
                continue
            if up.startswith("END"):
                section = None
                continue
            if section == "species":
                species.extend(s.split())
            elif section == "reactions":
                if "=" not in s:
                    continue                     # auxiliary line
                if re.match(r"^(LOW|TROE|PLOG|SRI|DUP|CHEB|TCHEB|PCHEB|REV)",
                            up):
                    continue
                m = _RXN_LINE_RE.match(s)
                if not m:
                    continue
                eq, A, n, Ea = m.groups()
                eq = eq.strip()
                if "<=>" in eq:
                    lhs, rhs = eq.split("<=>", 1)
                    rev = True
                elif "=>" in eq:
                    lhs, rhs = eq.split("=>", 1)
                    rev = False
                else:
                    lhs, rhs = eq.split("=", 1)
                    rev = True
                reac, _ = _parse_side(lhs)
                prod, _ = _parse_side(rhs)
                reactions.append(ChemkinReaction(
                    equation=eq, reactants=tuple(sorted(reac)),
                    products=tuple(sorted(prod)), reversible=rev,
                    A=float(A), n=float(n), Ea=float(Ea)))
    return species, reactions


# ---------------------------------------------------------------------------
# The diff
# ---------------------------------------------------------------------------
@dataclass
class RateChange:
    equation_a: str
    equation_b: str
    dlog10_A: float
    dn: float
    dEa: float


@dataclass
class ReproducibilityDiff:
    run_a: str
    run_b: str
    common_species: int = 0
    renamed: List[Tuple[str, str]] = field(default_factory=list)   # (a, b)
    only_in_a: List[str] = field(default_factory=list)
    only_in_b: List[str] = field(default_factory=list)
    common_reactions: int = 0
    reactions_only_in_a: List[str] = field(default_factory=list)
    reactions_only_in_b: List[str] = field(default_factory=list)
    rate_changes: List[RateChange] = field(default_factory=list)

    @property
    def identical(self) -> bool:
        return not (self.only_in_a or self.only_in_b
                    or self.reactions_only_in_a or self.reactions_only_in_b
                    or self.rate_changes)

    def to_dict(self) -> Dict:
        return {
            "run_a": self.run_a, "run_b": self.run_b,
            "identical": self.identical,
            "species": {
                "common": self.common_species,
                "renamed": [{"a": a, "b": b} for a, b in self.renamed],
                "only_in_a": self.only_in_a, "only_in_b": self.only_in_b,
            },
            "reactions": {
                "common": self.common_reactions,
                "only_in_a": self.reactions_only_in_a,
                "only_in_b": self.reactions_only_in_b,
                "rate_changes": [
                    {"equation_a": r.equation_a, "equation_b": r.equation_b,
                     "dlog10_A": r.dlog10_A, "dn": r.dn, "dEa": r.dEa}
                    for r in self.rate_changes],
            },
        }

    def render_text(self) -> str:
        lines = [
            f"RMG reproducibility diff — A: {self.run_a}   B: {self.run_b}",
            f"  verdict: {'IDENTICAL' if self.identical else 'DIVERGED'}",
            "",
            f"  species  : {self.common_species} matched "
            f"({len(self.renamed)} under different labels), "
            f"{len(self.only_in_a)} only in A, "
            f"{len(self.only_in_b)} only in B",
            f"  reactions: {self.common_reactions} matched, "
            f"{len(self.reactions_only_in_a)} only in A, "
            f"{len(self.reactions_only_in_b)} only in B, "
            f"{len(self.rate_changes)} with changed rates",
        ]
        if self.renamed:
            lines.append("")
            lines.append("  same structure, different label:")
            lines += [f"    {a}  ->  {b}" for a, b in self.renamed]
        for grp, label in ((self.only_in_a, "species only in A"),
                           (self.only_in_b, "species only in B"),
                           (self.reactions_only_in_a, "reactions only in A"),
                           (self.reactions_only_in_b, "reactions only in B")):
            if grp:
                lines.append("")
                lines.append(f"  {label}:")
                lines += [f"    - {x}" for x in grp]
        if self.rate_changes:
            lines.append("")
            lines.append(f"  {'Δlog10A':>8} {'Δn':>7} {'ΔEa':>10}  reaction")
            for r in sorted(self.rate_changes,
                            key=lambda r: -abs(r.dlog10_A)):
                lines.append(f"  {r.dlog10_A:>+8.3f} {r.dn:>+7.3f} "
                             f"{r.dEa:>+10.3g}  {r.equation_a}")
        return "\n".join(lines)


def _locate(run: str, names: Tuple[str, ...]) -> Optional[str]:
    p = Path(run)
    if p.is_file():
        return str(p)
    for n in names:
        cand = p / n
        if cand.is_file():
            return str(cand)
    return None


def reproducibility_diff(
    run_a: str,
    run_b: str,
    tol_log10_A: float = 0.1,
    tol_n: float = 0.05,
    tol_Ea: float = 100.0,
) -> ReproducibilityDiff:
    """Diff two RMG output directories (same input, e.g. different RMG
    versions).  ``run_a``/``run_b`` may be run dirs or direct chemkin paths.
    """
    chem_names = ("chemkin/chem_annotated.inp", "chemkin/chem.inp",
                  "chem_annotated.inp", "chem.inp")
    dict_names = ("chemkin/species_dictionary.txt", "species_dictionary.txt")

    diff = ReproducibilityDiff(run_a=run_a, run_b=run_b)
    sides = {}
    for tag, run in (("a", run_a), ("b", run_b)):
        chem = _locate(run, chem_names)
        if chem is None:
            raise FileNotFoundError(
                f"No CHEMKIN output found under {run} (looked for "
                f"{', '.join(chem_names)})")
        sdict_path = _locate(run, dict_names)
        adj = parse_species_dictionary(sdict_path) if sdict_path else {}
        species, reactions = parse_chemkin(chem)
        key_of = {label: (canonical_graph_key(adj[label])
                          if label in adj else f"label:{label}")
                  for label in species}
        sides[tag] = {"species": species, "reactions": reactions,
                      "key_of": key_of}

    # ---- species alignment by canonical key ------------------------------
    a_by_key = {sides["a"]["key_of"][s]: s for s in sides["a"]["species"]}
    b_by_key = {sides["b"]["key_of"][s]: s for s in sides["b"]["species"]}
    for key, sa in a_by_key.items():
        sb = b_by_key.get(key)
        if sb is None:
            diff.only_in_a.append(sa)
        else:
            diff.common_species += 1
            if sa != sb:
                diff.renamed.append((sa, sb))
    diff.only_in_b = [sb for key, sb in b_by_key.items()
                      if key not in a_by_key]

    # ---- reaction alignment by canonical species multisets ---------------
    def rxn_key(r: ChemkinReaction, key_of) -> Tuple:
        lhs = tuple(sorted(key_of.get(s, f"label:{s}") for s in r.reactants))
        rhs = tuple(sorted(key_of.get(s, f"label:{s}") for s in r.products))
        if r.reversible:
            return ("rev",) + tuple(sorted((lhs, rhs)))
        return ("irr", lhs, rhs)

    a_rxns: Dict[Tuple, ChemkinReaction] = {}
    for r in sides["a"]["reactions"]:
        a_rxns.setdefault(rxn_key(r, sides["a"]["key_of"]), r)
    b_rxns: Dict[Tuple, ChemkinReaction] = {}
    for r in sides["b"]["reactions"]:
        b_rxns.setdefault(rxn_key(r, sides["b"]["key_of"]), r)

    for key, ra in a_rxns.items():
        rb = b_rxns.get(key)
        if rb is None:
            diff.reactions_only_in_a.append(ra.equation)
            continue
        diff.common_reactions += 1
        d_log_a = (math.log10(rb.A) - math.log10(ra.A)
                   if ra.A > 0 and rb.A > 0 else float("inf"))
        d_n = rb.n - ra.n
        d_ea = rb.Ea - ra.Ea
        if (abs(d_log_a) > tol_log10_A or abs(d_n) > tol_n
                or abs(d_ea) > tol_Ea):
            diff.rate_changes.append(RateChange(
                equation_a=ra.equation, equation_b=rb.equation,
                dlog10_A=d_log_a, dn=d_n, dEa=d_ea))
    diff.reactions_only_in_b = [rb.equation for key, rb in b_rxns.items()
                                if key not in a_rxns]
    return diff
