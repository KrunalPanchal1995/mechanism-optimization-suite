"""Backend for surface_crosscheck.py — runs INSIDE the RMG conda env.

Invoked as:  <rmg_python> _rmg_surface_backend.py '<json config>'

Config:  {"chemkin_gas": path, "chemkin_surf": path, "T_K": float}
Output:  JSON on stdout:
         {"reactions": [{"equation": str, "log10_keq": float}, ...],
          "coverages": {}}

The RMG side reports per-reaction dimensionless standard-state equilibrium
constants computed from RMG's *own* thermo of the loaded CHEMKIN species:
log10 K = -ΔG°_rxn/(RT ln 10).  This is the P21 Keq-inconsistency probe.
Coverages are left empty (running RMG's SurfaceReactor requires a full core
configuration; users who have one can export coverages into the offline
JSON instead).

This script imports rmgpy and therefore only runs in the RMG env; the
calling module treats any failure here as a clean, explained error.
"""
from __future__ import annotations

import json
import math
import sys


def main() -> int:
    cfg = json.loads(sys.argv[1])
    T = float(cfg["T_K"])

    try:
        from rmgpy.chemkin import load_chemkin_file
    except ImportError as exc:                            # pragma: no cover
        print(f"rmgpy not importable in this interpreter: {exc}",
              file=sys.stderr)
        return 3

    # Surface CHEMKIN files reference the gas species; RMG's loader accepts
    # a list of files loaded in sequence sharing one species pool.
    species, reactions = load_chemkin_file(
        cfg["chemkin_surf"], check_duplicates=False)
    gas_species, gas_reactions = load_chemkin_file(
        cfg["chemkin_gas"], check_duplicates=False)
    thermo_by_label = {s.label: s for s in list(species) + list(gas_species)}

    R = 8.314462618          # J/mol/K
    out = {"reactions": [], "coverages": {}}
    for rxn in reactions:
        try:
            dG = 0.0
            for sp in rxn.products:
                dG += thermo_by_label[sp.label].get_free_energy(T)
            for sp in rxn.reactants:
                dG -= thermo_by_label[sp.label].get_free_energy(T)
            log10_keq = -dG / (R * T * math.log(10.0))
        except Exception as exc:                          # noqa: BLE001
            print(f"skipping {rxn}: {exc}", file=sys.stderr)
            continue
        out["reactions"].append({"equation": str(rxn),
                                 "log10_keq": log10_keq})

    json.dump(out, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
