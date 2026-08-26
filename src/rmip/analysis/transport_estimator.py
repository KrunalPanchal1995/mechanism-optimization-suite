"""
rmip/analysis/transport_estimator.py
====================================
Estimate Lennard-Jones transport parameters (geometry, well-depth, collision
diameter, polarizability) for a species from its molecular structure (SMILES),
using RDKit.  This fills the gap that made fused mechanisms silently break
laminar-flame-speed simulations: species assembled from a donor mechanism that
carried no transport data can now be given a physically reasonable estimate
instead of nothing.

Method (documented, and validated against literature LJ values in the tests)
---------------------------------------------------------------------------
1. **Radical saturation.**  Combustion mechanisms are dominated by radicals
   (CH3•, C2H5•, …).  A radical's LJ parameters are, to good approximation,
   those of its *saturated parent* (e.g. C2H5 ≈ C2H6).  We add one H per radical
   electron before estimating size/polarizability.

2. **Collision diameter σ [Å]** from the van-der-Waals molecular volume via the
   VABC method (Zhao, Abraham & Zissimos, *J. Org. Chem.* 2003):

       V_vdw = Σ (4/3)π·r_vdw(atom)³  − 5.92·N_bonds − 14.7·R_arom − 3.8·R_aliph

   then treat the molecule as a sphere of that volume:  σ = 2·(3·V_vdw/4π)^⅓.
   (Validated: CH4 → 3.67 Å vs 3.758 Å literature; C2H6 → 4.35 vs 4.44.)

3. **Well-depth ε/kB [K]** from polarizability α (from Crippen molar
   refractivity, α[Å³] = 0.3964·MR) via the fitted correlation
   ``ε/kB ≈ 90·α^0.53`` — an engineering estimate good to roughly ±25 % for
   non-polar species (CH4 → 149 K vs 148.6; C2H6 → 198 vs 215.7).

Limitations (honest)
--------------------
* The dipole moment is **not** estimated (set to 0) — strongly polar species
  (H2O, alcohols) will have their ε underestimated.  Prefer measured transport
  when available; estimation is a fallback for otherwise-missing species.
* ``rotational-relaxation`` defaults to 1.0 for molecules (0 for single atoms).
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from rmip.io.transport_io import TransportData

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import Crippen
    _HAS_RDKIT = True
except ImportError:  # pragma: no cover - rdkit is a suite dependency
    Chem = None       # type: ignore[assignment]
    Crippen = None    # type: ignore[assignment]
    _HAS_RDKIT = False

# Well-depth correlation ε/kB = _EPS_A · α**_EPS_B  (α in Å³, ε/kB in K).
_EPS_A = 90.0
_EPS_B = 0.53

# Small set of common linear polyatomics (RDKit-canonical SMILES) whose
# geometry the atom-count heuristic would otherwise mislabel as non-linear.
_LINEAR_POLYATOMICS = {
    "O=C=O",     # CO2
    "C#C",       # acetylene
    "N#C",       # HCN (canonical)
    "S=C=S",     # CS2
    "[N-]=[N+]=O",  # N2O (one canonical form)
}


def _saturate_radicals(mol):
    """Return a copy of ``mol`` with each radical electron replaced by an H."""
    rw = Chem.RWMol(Chem.AddHs(mol))
    # Snapshot radical centres before mutating (adding atoms invalidates the
    # live atom iterator with "size changed during iteration").
    radicals = [(a.GetIdx(), a.GetNumRadicalElectrons())
                for a in rw.GetAtoms() if a.GetNumRadicalElectrons()]
    for idx, nr in radicals:
        for _ in range(nr):
            h_idx = rw.AddAtom(Chem.Atom(1))
            rw.AddBond(idx, h_idx, Chem.BondType.SINGLE)
        rw.GetAtomWithIdx(idx).SetNumRadicalElectrons(0)
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out


def _vdw_volume(mol_with_H) -> float:
    """van-der-Waals molecular volume [Å³] via the VABC additive method."""
    pt = Chem.GetPeriodicTable()
    v_atoms = 0.0
    for atom in mol_with_H.GetAtoms():
        r = pt.GetRvdw(atom.GetAtomicNum())
        v_atoms += (4.0 / 3.0) * math.pi * r ** 3
    n_bonds = mol_with_H.GetNumBonds()
    ri = mol_with_H.GetRingInfo()
    n_arom = 0
    n_aliph = 0
    for ring in ri.AtomRings():
        if all(mol_with_H.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            n_arom += 1
        else:
            n_aliph += 1
    return v_atoms - 5.92 * n_bonds - 14.7 * n_arom - 3.8 * n_aliph


def _geometry(orig_mol) -> str:
    molH = Chem.AddHs(orig_mol)
    n = molH.GetNumAtoms()
    if n == 1:
        return "atom"
    if n == 2:
        return "linear"
    canon = Chem.MolToSmiles(Chem.RemoveHs(orig_mol))
    if canon in _LINEAR_POLYATOMICS:
        return "linear"
    return "nonlinear"


def estimate_transport(smiles: str, name: Optional[str] = None) -> Optional[TransportData]:
    """
    Estimate :class:`TransportData` from a SMILES string, or ``None`` if RDKit
    is unavailable or the SMILES cannot be parsed / sized.
    """
    if not _HAS_RDKIT:
        logger.warning("RDKit unavailable — cannot estimate transport for %s", name or smiles)
        return None
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning("Unparseable SMILES %r (species %s) — no transport estimate",
                       smiles, name)
        return None

    geometry = _geometry(mol)
    parent = _saturate_radicals(mol)

    v_vdw = _vdw_volume(parent)
    if v_vdw <= 0.0:
        logger.warning("Non-positive VdW volume for %s (%s) — no estimate",
                       name, smiles)
        return None
    diameter_A = 2.0 * (3.0 * v_vdw / (4.0 * math.pi)) ** (1.0 / 3.0)

    alpha = 0.3964 * Crippen.MolMR(parent)      # polarizability [Å³]
    well_depth_K = _EPS_A * (alpha ** _EPS_B) if alpha > 0 else _EPS_A

    rot_relax = 0.0 if geometry == "atom" else 1.0

    return TransportData(
        geometry          = geometry,
        well_depth_K      = round(well_depth_K, 4),
        diameter_A        = round(diameter_A, 4),
        dipole_D          = 0.0,
        polarizability_A3 = round(max(alpha, 0.0), 4),
        rot_relax         = rot_relax,
        source            = "estimated:vabc-polarizability",
    )
