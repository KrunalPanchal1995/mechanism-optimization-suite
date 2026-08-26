#!/usr/bin/env python3
"""
rmip/analysis/_regression_worker.py
===================================
Self-contained worker for the W8b version-behaviour regression harness.

DELIBERATELY imports nothing from ``rmip`` — only the standard library,
``cantera`` and (transitively) numpy — so it can be executed by **any**
Python that has Cantera installed, including an isolated venv holding a
different Cantera version (the env-isolation subprocess pattern):

    <some/venv/bin/python> _regression_worker.py cases.json results.json

``cases.json``::

    {"cases": [{"id": "...", "mechanism": "h2o2.yaml", "task": "idt", ...}]}

``results.json``::

    {"meta": {"cantera_version": "...", "python": "..."},
     "results": {"<case id>": {"ok": true, "observables": {...},
                               "error": null, "wall_s": 0.42}}}

Tasks
-----
equilibrium : HP-equilibrate from (T, P, phi) → ``T_ad_K`` + ``X_<sp>`` for
              each species in ``track_species``.
idt         : constant-volume adiabatic ignition, max-dT/dt → ``idt_s``.
lfs         : freely-propagating premixed flame, mixture-averaged transport
              → ``lfs_m_per_s``.
"""
from __future__ import annotations

import json
import sys
import time


def _make_gas(case):
    import cantera as ct
    gas = ct.Solution(case["mechanism"])
    phi = case.get("phi")
    if phi is not None:
        gas.set_equivalence_ratio(phi, fuel=case["fuel"],
                                  oxidizer=case.get("oxidizer",
                                                    "O2:1.0, N2:3.76"))
        gas.TP = case["T_K"], case["P_atm"] * ct.one_atm
    else:
        gas.TPX = case["T_K"], case["P_atm"] * ct.one_atm, case["X"]
    return gas


def run_equilibrium(case):
    gas = _make_gas(case)
    gas.equilibrate("HP")
    obs = {"T_ad_K": float(gas.T)}
    for sp in case.get("track_species", []):
        obs[f"X_{sp}"] = float(gas.X[gas.species_index(sp)])
    return obs


def run_idt(case):
    import cantera as ct
    gas = _make_gas(case)
    reactor = ct.IdealGasReactor(gas)
    sim = ct.ReactorNet([reactor])
    sim.rtol, sim.atol = 1e-10, 1e-16
    t_end = case.get("t_end_s", 1.0)
    n_steps = int(case.get("n_steps", 4000))
    times, temps = [0.0], [reactor.T]
    dt = t_end / n_steps
    for i in range(1, n_steps + 1):
        sim.advance(i * dt)
        times.append(i * dt)
        temps.append(reactor.T)
        # stop shortly after ignition so the max-dT/dt point is bracketed
        if reactor.T > case["T_K"] + 500.0 and i > 10:
            for j in range(i + 1, min(i + 200, n_steps) + 1):
                sim.advance(j * dt)
                times.append(j * dt)
                temps.append(reactor.T)
            break
    dTdt_max, idt = 0.0, None
    for k in range(1, len(times)):
        dT = (temps[k] - temps[k - 1]) / (times[k] - times[k - 1])
        if dT > dTdt_max:
            dTdt_max, idt = dT, 0.5 * (times[k] + times[k - 1])
    if idt is None or temps[-1] < case["T_K"] + 400.0:
        raise RuntimeError(f"no ignition within t_end={t_end} s "
                           f"(T_final={temps[-1]:.1f} K)")
    return {"idt_s": float(idt)}


def run_lfs(case):
    import cantera as ct
    gas = _make_gas(case)
    flame = ct.FreeFlame(gas, width=case.get("width_m", 0.03))
    flame.set_refine_criteria(ratio=3.0, slope=0.07, curve=0.14)
    try:
        flame.transport_model = "mixture-averaged"
    except Exception:                                      # noqa: BLE001
        flame.transport_model = "Mix"        # pre-3.0 spelling
    flame.solve(loglevel=0, auto=True)
    return {"lfs_m_per_s": float(flame.velocity[0])}


_TASKS = {"equilibrium": run_equilibrium, "idt": run_idt, "lfs": run_lfs}


def run_cases(cases):
    results = {}
    for case in cases:
        t0 = time.perf_counter()
        entry = {"ok": False, "observables": {}, "error": None}
        try:
            entry["observables"] = _TASKS[case["task"]](case)
            entry["ok"] = True
        except Exception as exc:                           # noqa: BLE001
            entry["error"] = f"{type(exc).__name__}: {exc}"
        entry["wall_s"] = round(time.perf_counter() - t0, 3)
        results[case["id"]] = entry
    return results


def main(argv):
    if len(argv) != 3:
        print("usage: _regression_worker.py cases.json results.json",
              file=sys.stderr)
        return 2
    with open(argv[1]) as fh:
        cases = json.load(fh)["cases"]
    import cantera as ct
    payload = {
        "meta": {"cantera_version": ct.__version__,
                 "python": sys.version.split()[0],
                 "executable": sys.executable},
        "results": run_cases(cases),
    }
    with open(argv[2], "w") as fh:
        json.dump(payload, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
