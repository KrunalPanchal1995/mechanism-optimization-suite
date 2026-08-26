"""
rmip/core/canonicalize.py
=========================
RDKit-based canonicalization helpers.

All cross-mechanism species and reaction identity is based on:
  - InChI key     (D1): 25-character hash from RDKit InChI
  - Canonical key (D2): sorted SMILES frozenset for reactions

Graceful degradation: if RDKit is unavailable the raw SMILES is
returned as-is and inchi_key is set to None. A warning is emitted once.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Dict, Optional, Tuple

log = logging.getLogger("rmip.core.canonicalize")
_RDKIT_WARNED = False


def _rdkit_available() -> bool:
    global _RDKIT_WARNED
    try:
        from rdkit import Chem  # noqa: F401
        return True
    except ImportError:
        if not _RDKIT_WARNED:
            log.warning(
                "RDKit not available — canonical SMILES and InChI keys will not "
                "be computed.  Cross-mechanism matching will be degraded."
            )
            _RDKIT_WARNED = True
        return False


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    Return RDKit canonical SMILES, or the original string if RDKit unavailable.
    Returns None if the SMILES is invalid.
    """
    if not smiles:
        return None
    if not _rdkit_available():
        return smiles
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        log.debug("Invalid SMILES: %s", smiles)
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def smiles_to_inchi(smiles: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Convert SMILES to (InChI, InChI_key).
    Returns (None, None) on failure.
    """
    if not smiles:
        return None, None
    if not _rdkit_available():
        # Fallback: hash the raw SMILES deterministically
        h = hashlib.sha256(smiles.encode()).hexdigest()[:25].upper()
        return None, h
    from rdkit import Chem
    from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    try:
        inchi = MolToInchi(mol)
        key   = InchiToInchiKey(inchi) if inchi else None
        return inchi, key
    except Exception as e:
        log.debug("InChI conversion failed for '%s': %s", smiles, e)
        return None, None


def _cancel_self_third_body(
    reactant_smiles: list[str],
    reactant_coeffs: list[float],
    product_smiles:  list[str],
    product_coeffs:  list[float],
) -> Tuple[list[str], list[str]]:
    """
    Cancel the shared (min) count of any species appearing on BOTH sides of
    a reaction, expanding the rest into repeated-per-coefficient entries.

    Why: a reaction written with an explicit self-third-body, e.g.
    ``2H+H2<=>2H2`` (H2 acting as its own collider: 2H + M(=H2) -> H2 + M),
    is the same elementary step as the generic-M form ``H2+M<=>2H+M`` --
    Cantera treats them as equivalent, but this suite's canonicalizer
    previously did not (different canonical_key), found stress-testing
    fusion on Keromnes/Li/Davis (2026-07-14, bug.md). Generic "M" never
    appears in a reaction's reactant/product SMILES lists at all (stripped
    at parse time -- see _chemkin_parser.py's bare/parenthesized-M
    handling), so the fix here is symmetric: for a self-third-body form,
    remove the shared (min of reactant-count, product-count) portion of
    the overlapping species from BOTH sides -- that portion is the
    collider passing through unreacted -- leaving only the residual "core"
    reaction, exactly matching what the generic-M form already produces.
    """
    r_counts: Dict[str, float] = {}
    for s, c in zip(reactant_smiles, reactant_coeffs):
        r_counts[s] = r_counts.get(s, 0.0) + c
    p_counts: Dict[str, float] = {}
    for s, c in zip(product_smiles, product_coeffs):
        p_counts[s] = p_counts.get(s, 0.0) + c

    for s in set(r_counts) & set(p_counts):
        shared = min(r_counts[s], p_counts[s])
        if shared > 0:
            r_counts[s] -= shared
            p_counts[s] -= shared

    def _expand(counts: Dict[str, float]) -> list[str]:
        out: list[str] = []
        for s, c in counts.items():
            n = int(round(c))
            out.extend([s] * n)
        return out

    return _expand(r_counts), _expand(p_counts)


def make_reaction_canonical_key(
    reactant_smiles: list[str],
    product_smiles:  list[str],
    reversible:      bool = True,
    reactant_coeffs: Optional[list[float]] = None,
    product_coeffs:  Optional[list[float]] = None,
    cancel_self_third_body: bool = False,
) -> Optional[str]:
    """
    Build a canonical reaction key (design decision D2).

    Algorithm
    ---------
    1. Canonicalise each species SMILES with RDKit.
    2. Sort each side lexicographically.
    3. For reversible reactions: normalise direction so that
       ``min(reactants_str, products_str)`` is always on the left.
    4. Join: ``"A,B>>C,D"``.

    Parameters
    ----------
    reactant_coeffs, product_coeffs : optional per-entry stoichiometric
        coefficients, parallel to reactant_smiles/product_smiles. Only
        consulted when cancel_self_third_body=True; omitted entries default
        to 1.0 each, matching prior (pre-M-equivalence-fix) behavior for
        every existing caller that doesn't opt in.
    cancel_self_third_body : if True, apply _cancel_self_third_body() before
        building the key -- opt-in (default False) since this changes which
        reactions canonically match and the blast radius on mechanisms other
        than the one that surfaced the bug wasn't fully characterized.
        core/parser.py's DB-build path opts in; leave off elsewhere unless
        deliberately testing this behavior.

    Returns None if any SMILES cannot be canonicalised.
    """
    if cancel_self_third_body:
        r_coeffs = reactant_coeffs if reactant_coeffs is not None else [1.0] * len(reactant_smiles)
        p_coeffs = product_coeffs if product_coeffs is not None else [1.0] * len(product_smiles)
        reactant_smiles, product_smiles = _cancel_self_third_body(
            reactant_smiles, r_coeffs, product_smiles, p_coeffs,
        )

    r_can = [canonicalize_smiles(s) for s in reactant_smiles]
    p_can = [canonicalize_smiles(s) for s in product_smiles]

    if None in r_can or None in p_can:
        return None

    r_str = ",".join(sorted(r_can))
    p_str = ",".join(sorted(p_can))

    if reversible and r_str > p_str:
        r_str, p_str = p_str, r_str

    return f"{r_str}>>{p_str}"


class Canonicalizer:
    """
    Stateless helper that wraps the module-level functions
    for use in the parser and comparison engine.
    """

    def canonicalize(self, smiles: str) -> Optional[str]:
        return canonicalize_smiles(smiles)

    def to_inchi(self, smiles: str) -> Tuple[Optional[str], Optional[str]]:
        return smiles_to_inchi(smiles)

    def reaction_key(
        self,
        reactant_smiles: list[str],
        product_smiles:  list[str],
        reversible:      bool = True,
        reactant_coeffs: Optional[list[float]] = None,
        product_coeffs:  Optional[list[float]] = None,
        cancel_self_third_body: bool = False,
    ) -> Optional[str]:
        return make_reaction_canonical_key(
            reactant_smiles, product_smiles, reversible,
            reactant_coeffs=reactant_coeffs, product_coeffs=product_coeffs,
            cancel_self_third_body=cancel_self_third_body,
        )


# ---------------------------------------------------------------------------
# Additional helper functions restored from Phase-1
# ---------------------------------------------------------------------------

def smiles_to_formula(smiles: str) -> Optional[str]:
    """Return molecular formula from SMILES (e.g. 'C7H16'). Returns None on failure."""
    if not smiles:
        return None
    if not _rdkit_available():
        return None
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol_h = Chem.AddHs(mol)
        return rdMolDescriptors.CalcMolFormula(mol_h)
    except Exception as exc:
        log.debug("smiles_to_formula error for '%s': %s", smiles, exc)
        return None


def smiles_to_mw(smiles: str) -> Optional[float]:
    """Return molecular weight in kg/mol from SMILES. Returns None on failure."""
    if not smiles:
        return None
    if not _rdkit_available():
        return None
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol_h = Chem.AddHs(mol)
        mw_g_mol = Descriptors.MolWt(mol_h)
        return mw_g_mol / 1000.0  # g/mol → kg/mol
    except Exception as exc:
        log.debug("smiles_to_mw error for '%s': %s", smiles, exc)
        return None
