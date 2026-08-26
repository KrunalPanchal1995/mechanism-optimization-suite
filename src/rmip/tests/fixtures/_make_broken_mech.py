"""
Generator for the mechanism-check test fixtures.

Produces two deliberately-broken mechanisms derived from Cantera's shipped
``h2o2.yaml`` (so all thermo/transport records are otherwise valid):

  broken_mech.yaml   — LOADS in Cantera (with warnings) and carries four fault
                       classes screened on the live Solution: collision-limit
                       violation, thermo Tmid discontinuity, an unused species,
                       and an activation-energy unit trap.
  broken_dup.yaml    — FAILS to load: an unmarked duplicate reaction (Cantera
                       refuses it). Exercises the load-error + raw duplicate scan.
  broken_plog.yaml   — FAILS to load: a PLOG reaction with a negative pressure
                       piece (Cantera 3.x rejects it at PlogRate::validate).
                       Exercises the raw pdep-sanity scan on a non-loadable file.

Run:  python src/rmip/tests/fixtures/_make_broken_mech.py
(kept as a script so the fixtures are inspectable/committed, not regenerated at
import time; the test module can also call build_broken_dicts() directly.)
"""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, Tuple


def _base() -> Dict[str, Any]:
    import cantera as ct
    import yaml
    with open(os.path.join(ct.__path__[0], "data", "h2o2.yaml")) as fh:
        return yaml.safe_load(fh)


def build_broken_dicts() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Return (loadable_broken, unmarked_duplicate, negative_plog) mech dicts."""
    base = _base()

    # =========================== broken_mech.yaml ===========================
    mech = copy.deepcopy(base)
    mech["description"] = ("Deliberately broken H2/O2 mechanism for RMIP "
                           "mechanism-check tests. DO NOT use for simulation.")

    # -- Fault 1: collision-limit violation --------------------------------
    # O + HO2 <=> OH + O2 is bimolecular; blow its A up by 1e6 (cm3/mol/s).
    for r in mech["reactions"]:
        if r["equation"] == "O + HO2 <=> OH + O2":
            r["rate-constant"]["A"] *= 1.0e6           # ~5 orders over the limit
            break

    # -- Fault 2: activation-energy unit trap ------------------------------
    # O + H2 <=> H + OH: a ~60 kcal/mol barrier entered as J/kmol (2.51e8) into
    # a cal/mol field — the classic silent 1000x error. ÷4184 restores it.
    for r in mech["reactions"]:
        if r["equation"] == "O + H2 <=> H + OH":
            r["rate-constant"]["Ea"] = 2.51e8
            break

    # -- Fault 3: NASA-7 Tmid discontinuity in OH --------------------------
    # Offset the high-T a0 coefficient so Cp/H/S jump at Tmid.
    for sp in mech["species"]:
        if sp["name"] == "OH":
            data = sp["thermo"]["data"]                 # [low(7), high(7)]
            data[1] = list(data[1])
            data[1][0] += 3.0                           # big Cp offset
            break

    # -- Fault 4: unused species -------------------------------------------
    # Clone AR's thermo/transport under a new name never referenced anywhere.
    for sp in mech["species"]:
        if sp["name"] == "AR":
            ghost = copy.deepcopy(sp)
            ghost["name"] = "XX"
            ghost["composition"] = {"Ar": 1}
            mech["species"].append(ghost)
            break
    mech["phases"][0]["species"].append("XX")

    # ========================== broken_dup.yaml ============================
    dup = copy.deepcopy(base)
    dup["description"] = ("H2/O2 mechanism with an UNMARKED duplicate reaction — "
                          "Cantera refuses to load this. RMIP check should detect "
                          "and offer to mark it duplicate.")
    # Duplicate the first simple reaction without the 'duplicate' flag.
    first = None
    for r in dup["reactions"]:
        if r.get("type") is None and "rate-constant" in r:
            first = r
            break
    clone = copy.deepcopy(first)
    clone["rate-constant"]["A"] *= 0.5                  # a different rate, same eq
    dup["reactions"].insert(dup["reactions"].index(first) + 1, clone)

    # ========================== broken_plog.yaml ===========================
    plog = copy.deepcopy(base)
    plog["description"] = ("H2/O2 mechanism with a PLOG reaction carrying a "
                           "negative pressure piece — Cantera 3.x rejects it at "
                           "load. RMIP check should detect it via the raw scan.")
    plog["reactions"].append({
        "equation": "H2O2 <=> 2 OH",                    # unimolecular decomposition
        "type": "pressure-dependent-Arrhenius",
        "rate-constants": [
            {"P": "0.1 atm", "A": 1.0e12, "b": 0.0, "Ea": 45000.0},
            {"P": "1.0 atm", "A": -5.0e11, "b": 0.0, "Ea": 45000.0},   # NEGATIVE
            {"P": "10.0 atm", "A": 3.0e12, "b": 0.0, "Ea": 48000.0},
        ],
    })

    return mech, dup, plog


def build_deprecated_dict() -> Dict[str, Any]:
    """A mechanism (W1) carrying deprecated CTI→YAML keywords Cantera rejects.

    Faults: phase ``thermo: ideal_gas`` (should be ``ideal-gas``), a reaction
    ``type: PLOG`` and one ``type: three_body`` (should be
    ``pressure-dependent-Arrhenius`` / ``three-body``). After migration the file
    loads cleanly.
    """
    dep = copy.deepcopy(_base())
    dep["description"] = ("H2/O2 mechanism with deprecated CTI-era keywords "
                          "(ideal_gas / PLOG / three_body) — Cantera refuses it "
                          "until migrated. Tests the W1 migration linter.")
    dep["phases"][0]["thermo"] = "ideal_gas"                 # → ideal-gas
    for r in dep["reactions"]:
        if r.get("type") == "three-body":
            r["type"] = "three_body"                        # → three-body
            break
    dep["reactions"].append({
        "equation": "H2O2 <=> 2 OH",
        "type": "PLOG",                                     # → pressure-dependent-Arrhenius
        "rate-constants": [
            {"P": "0.1 atm", "A": 1.0e12, "b": 0.0, "Ea": 45000.0},
            {"P": "1.0 atm", "A": 2.0e12, "b": 0.0, "Ea": 45500.0},
        ],
    })
    return dep


def build_integrator_failure_dict() -> Dict[str, Any]:
    """A mechanism (P7) engineered to make CVODES itself fail mid-integration.

    LOADS fine in Cantera (only the usual NasaPoly2 discontinuity warning) —
    unlike ``broken_mech``'s A-factor boost, this is not caught by any rate/
    collision-limit screen; it only surfaces once you actually try to
    *integrate* a reactor, which is exactly what ``check_sim_setup_sanity``
    (P7) exists to catch.

    Fault: OH's high-T NASA-7 quartic coefficient (a4, the T^4 term) is set
    to an absurd, deeply-negative value. At combustion temperatures this
    drives Cp/R (and hence dT/dt) to NaN, so ``ReactorNet.advance``/``step``
    raises a genuine ``CanteraError`` ("ydot contains non-finite elements")
    on the very first RHS evaluation at high T — verified interactively
    against Cantera before being encoded here as a fixture.
    """
    base = _base()
    mech = copy.deepcopy(base)
    mech["description"] = ("H2/O2 mechanism with an absurd NASA-7 high-T "
                           "quartic coefficient on OH — loads fine in "
                           "Cantera, but drives CVODES to a non-finite RHS "
                           "at combustion temperatures. RMIP check's "
                           "sim_setup_sanity (P7) should flag the "
                           "integrator failure.")
    for sp in mech["species"]:
        if sp["name"] == "OH":
            data = sp["thermo"]["data"]          # [low(7), high(7)]
            data[1] = list(data[1])
            data[1][4] = -5.0e5                  # a4 (T^4 coeff): sane ~1e-14
            break
    return mech


def main() -> int:
    import yaml
    here = os.path.dirname(os.path.abspath(__file__))
    mech, dup, plog = build_broken_dicts()
    outputs = (("broken_mech.yaml", mech),
               ("broken_dup.yaml", dup),
               ("broken_plog.yaml", plog),
               ("broken_deprecated.yaml", build_deprecated_dict()),
               ("broken_integrator.yaml", build_integrator_failure_dict()))
    for name, data in outputs:
        with open(os.path.join(here, name), "w") as fh:
            yaml.safe_dump(data, fh, sort_keys=False)
    print("wrote " + ", ".join(n for n, _ in outputs) + " to", here)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
