"""
rmip/io/transport_io.py
=======================
Transport-property (Lennard-Jones) I/O for CHEMKIN transport files and the
shared :class:`TransportData` record used across the suite.

Until now the suite parsed thermo but **never** transport (design decision
"D6"): fused mechanisms lost their Lennard-Jones data and silently broke
laminar-flame-speed simulations.  This module + :mod:`rmip.analysis.
transport_estimator` close that gap.

CHEMKIN transport-file format
-----------------------------
A flat, one-line-per-species database (no block delimiters)::

    ! name        geom   eps/kB[K]  sigma[Å]  dipole[D]  polar[Å³]  Zrot
    AR              0     136.500    3.330     0.000      0.000      0.000
    H2O             2     572.400    2.605     1.844      0.000      4.000

* ``geom``  : 0 = single atom, 1 = linear molecule, 2 = non-linear molecule
* ``eps/kB``: Lennard-Jones potential well depth / Boltzmann constant [K]
* ``sigma`` : Lennard-Jones collision diameter [Å]
* ``dipole``: dipole moment [Debye]
* ``polar`` : polarizability [Å³]
* ``Zrot``  : rotational relaxation collision number at 298 K
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

# CHEMKIN integer geometry code <-> Cantera geometry string.
_GEOM_INT_TO_STR = {0: "atom", 1: "linear", 2: "nonlinear"}
_GEOM_STR_TO_INT = {v: k for k, v in _GEOM_INT_TO_STR.items()}


@dataclass
class TransportData:
    """
    Lennard-Jones transport parameters for one species.

    Units mirror the CHEMKIN transport database (and Cantera's ``transport``
    block): ``well_depth_K`` [K], ``diameter_A`` [Å], ``dipole_D`` [Debye],
    ``polarizability_A3`` [Å³], ``rot_relax`` dimensionless.
    """
    geometry:          str            # "atom" | "linear" | "nonlinear"
    well_depth_K:      float          # eps/kB
    diameter_A:        float          # sigma
    dipole_D:          float = 0.0
    polarizability_A3: float = 0.0
    rot_relax:         float = 0.0
    source:            str = ""        # "chemkin" | "cantera" | "estimated:<method>"

    # ------------------------------------------------------------------
    def geom_int(self) -> int:
        return _GEOM_STR_TO_INT.get(self.geometry, 2)

    def to_dict(self) -> Dict[str, Union[str, float]]:
        return {
            "geometry":          self.geometry,
            "well_depth_K":      self.well_depth_K,
            "diameter_A":        self.diameter_A,
            "dipole_D":          self.dipole_D,
            "polarizability_A3": self.polarizability_A3,
            "rot_relax":         self.rot_relax,
            "source":            self.source,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "TransportData":
        return cls(
            geometry          = d["geometry"],
            well_depth_K      = float(d["well_depth_K"]),
            diameter_A        = float(d["diameter_A"]),
            dipole_D          = float(d.get("dipole_D", 0.0) or 0.0),
            polarizability_A3 = float(d.get("polarizability_A3", 0.0) or 0.0),
            rot_relax         = float(d.get("rot_relax", 0.0) or 0.0),
            source            = d.get("source", ""),
        )

    @classmethod
    def from_cantera_dict(cls, d: Dict, source: str = "cantera") -> "TransportData":
        """Build from a Cantera-YAML ``transport`` mapping (Å, K, Debye, Å³)."""
        return cls(
            geometry          = d.get("geometry", "nonlinear"),
            well_depth_K      = float(d["well-depth"]),
            diameter_A        = float(d["diameter"]),
            dipole_D          = float(d.get("dipole", 0.0) or 0.0),
            polarizability_A3 = float(d.get("polarizability", 0.0) or 0.0),
            rot_relax         = float(d.get("rotational-relaxation", 0.0) or 0.0),
            source            = source,
        )

    def to_chemkin_line(self, name: str) -> str:
        """Format one fixed-width CHEMKIN transport line."""
        return (
            f"{name:<18s}{self.geom_int():<4d}"
            f"{self.well_depth_K:>10.3f}{self.diameter_A:>10.3f}"
            f"{self.dipole_D:>10.3f}{self.polarizability_A3:>10.3f}"
            f"{self.rot_relax:>10.3f}"
        )

    def to_cantera_dict(self) -> Dict[str, Union[str, float]]:
        """
        Build a Cantera-YAML ``transport`` mapping.  Optional fields are only
        emitted when non-zero (matching Cantera's own writer conventions).
        """
        d: Dict[str, Union[str, float]] = {
            "model":      "gas",
            "geometry":   self.geometry,
            "well-depth": round(self.well_depth_K, 4),
            "diameter":   round(self.diameter_A, 4),
        }
        if self.dipole_D:
            d["dipole"] = round(self.dipole_D, 4)
        if self.polarizability_A3:
            d["polarizability"] = round(self.polarizability_A3, 4)
        if self.rot_relax:
            d["rotational-relaxation"] = round(self.rot_relax, 4)
        return d


# ---------------------------------------------------------------------------
# CHEMKIN transport file parsing / writing
# ---------------------------------------------------------------------------

_SKIP_TOKENS = {"TRANSPORT", "TRANS", "ALL", "END"}


def parse_chemkin_transport(source: str) -> Dict[str, TransportData]:
    """
    Parse a CHEMKIN transport database into ``{species_name: TransportData}``.

    ``source`` may be a path to a transport file or the file's text content.
    Comment text after ``!`` is ignored; blank lines and the
    ``TRANSPORT``/``TRANS ALL``/``END`` keyword lines are skipped.  Lines with
    fewer than the 6 required numeric columns are skipped (logged by callers).
    """
    if os.path.exists(source):
        with open(source, "r", errors="replace") as fh:
            text = fh.read()
    else:
        text = source

    result: Dict[str, TransportData] = {}
    for raw in text.splitlines():
        line = raw.split("!", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        # skip standalone keyword lines (TRANSPORT / TRANS ALL / END)
        if len(parts) == 1 and parts[0].upper() in _SKIP_TOKENS:
            continue
        if parts[0].upper() in ("TRANSPORT", "TRANS") and len(parts) <= 2:
            continue
        if len(parts) < 7:
            continue
        name = parts[0]
        try:
            geom  = int(float(parts[1]))
            eps   = float(parts[2])
            sigma = float(parts[3])
            dip   = float(parts[4])
            pol   = float(parts[5])
            zrot  = float(parts[6])
        except ValueError:
            continue
        result[name] = TransportData(
            geometry          = _GEOM_INT_TO_STR.get(geom, "nonlinear"),
            well_depth_K      = eps,
            diameter_A        = sigma,
            dipole_D          = dip,
            polarizability_A3 = pol,
            rot_relax         = zrot,
            source            = "chemkin",
        )
    return result


def write_chemkin_transport(
    data: Dict[str, TransportData],
    header: Optional[str] = None,
) -> str:
    """Serialise ``{name: TransportData}`` to a CHEMKIN transport-file string."""
    lines: List[str] = []
    if header:
        for hl in header.splitlines():
            lines.append(f"! {hl}")
    lines.append(
        "! name           geom  eps/kB[K]  sigma[A] dipole[D] polar[A3]  Zrot"
    )
    for name, td in data.items():
        lines.append(td.to_chemkin_line(name))
    return "\n".join(lines) + "\n"
