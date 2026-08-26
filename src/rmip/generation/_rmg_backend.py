"""
rmip/generation/_rmg_backend.py
===============================
Dedicated-env backend for **targeted** RMG reaction-family enumeration
(FEATURE_ROADMAP #2).  Runs *inside* the RMG conda env and is invoked by the
shared-env front-end as::

    <rmg_env>/bin/python <_rmg_backend.py> <cmd> <request.json> <response.json>

exactly like the other D7 backends.  ``rmgpy`` is imported *inside* the command
functions so the shared env can introspect this file without RMG installed.

Unlike ``rmg_interface.py::generate`` (a FULL, flux-driven model-generation run),
this loads the reaction-family database once and enumerates **one generation** of
reactions for a *given* species set — the "just the nodes/edges for these
species / just these reactions" operation the user asked for, at a fraction of a
full run's cost.

Commands
--------
- ``families`` : list the available reaction-family names.
- ``react``    : given a list of ``{label, smiles}`` species, enumerate one
  generation of family reactions among/from them.  Response reactions carry the
  participating SMILES, the family, an estimated Arrhenius rate, and reversibility;
  ``new_species`` lists product species whose SMILES were not in the input set.

Request/response are plain JSON (SMILES strings), so the shared-env front-end
never needs an rmgpy object.
"""
from __future__ import annotations

import json
import sys


def _load_database(request):
    """Load only the kinetics families (+thermo for rate estimation) from RMG."""
    import os
    from rmgpy import settings
    from rmgpy.data.rmg import RMGDatabase

    db_path = request.get("rmg_database") or os.environ.get("RMG_DATABASE_PATH") \
        or settings["database.directory"]
    families = request.get("families", "default")
    database = RMGDatabase()
    database.load(
        path=db_path,
        thermo_libraries=request.get("thermo_libraries", ["primaryThermoLibrary"]),
        reaction_libraries=[],
        seed_mechanisms=[],
        kinetics_families=families,
        kinetics_depositories=[],
    )
    return database


def cmd_families(request: dict) -> dict:
    database = _load_database(request)
    return {"families": sorted(database.kinetics.families.keys())}


def cmd_react(request: dict) -> dict:
    from rmgpy.species import Species

    database = _load_database(request)
    only = request.get("families") if isinstance(request.get("families"), list) else None

    species = []
    in_smiles = set()
    for spec in request["species"]:
        sp = Species(label=spec.get("label", "")).from_smiles(spec["smiles"])
        sp.generate_resonance_structures()
        species.append(sp)
        in_smiles.add(sp.molecule[0].to_smiles())

    # generate_reactions_from_families() takes *one reactant combo* (1-3
    # species) per call, not the whole species pool at once — passing all N
    # species in a single call raises "Cannot check for duplicate reactants
    # if provided number of reactants is greater than 3" for any N > 3.
    # Mirror RMG's own react_all() (rmgpy/rmg/react.py): build the unimolecular
    # tuples (each species alone) and bimolecular tuples (every pair, incl.
    # self-pairs) and call once per tuple. Trimolecular is intentionally
    # skipped — out of scope for this "one generation, targeted" scan.
    n = len(species)
    reactant_tuples = [(species[i],) for i in range(n)]
    for i in range(n):
        for j in range(i, n):
            reactant_tuples.append((species[i], species[j]))

    reactions = []
    for combo in reactant_tuples:
        reactions.extend(database.kinetics.generate_reactions_from_families(
            combo, only_families=only))

    out_rxns = []
    new_smiles = {}
    for rxn in reactions:
        try:
            database.kinetics.estimate_kinetics(rxn) if hasattr(
                database.kinetics, "estimate_kinetics") else None
        except Exception:                       # noqa: BLE001
            pass
        r_sm = [m.molecule[0].to_smiles() for m in rxn.reactants]
        p_sm = [m.molecule[0].to_smiles() for m in rxn.products]
        for m in rxn.products:
            sm = m.molecule[0].to_smiles()
            if sm not in in_smiles:
                new_smiles[sm] = getattr(m, "label", "") or sm
        arr = _arrhenius(rxn)
        out_rxns.append({
            "reactants": r_sm, "products": p_sm,
            "family": getattr(rxn, "family", None),
            "reversible": bool(getattr(rxn, "reversible", True)),
            "arrhenius": arr,
        })
    return {"reactions": out_rxns,
            "new_species": [{"smiles": s, "label": lbl} for s, lbl in new_smiles.items()]}


def _arrhenius(rxn):
    """Extract A/n/Ea (SI) from a reaction's kinetics if available, else None."""
    k = getattr(rxn, "kinetics", None)
    if k is None:
        return None
    try:
        return {"A": float(k.A.value_si), "n": float(k.n.value_si),
                "Ea": float(k.Ea.value_si)}
    except Exception:                           # noqa: BLE001
        return None


_COMMANDS = {"families": cmd_families, "react": cmd_react}


def main() -> int:
    if len(sys.argv) != 4 or sys.argv[1] not in _COMMANDS:
        sys.stderr.write(
            f"usage: {sys.argv[0]} <{'|'.join(_COMMANDS)}> <request.json> <response.json>\n")
        return 2
    cmd, req_path, resp_path = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(req_path) as fh:
        request = json.load(fh)
    result = _COMMANDS[cmd](request)
    with open(resp_path, "w") as fh:
        json.dump(result, fh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
