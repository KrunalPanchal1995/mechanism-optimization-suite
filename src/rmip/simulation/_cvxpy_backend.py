"""
rmip/simulation/_cvxpy_backend.py
=================================
Dedicated-venv backend for the B2BDC **SDP outer bounds** (FEATURE_ROADMAP #5
follow-on).  Runs *inside* the ``cvxpy`` venv and is invoked by the shared-env
front-end as::

    <venv>/bin/python3 _cvxpy_backend.py <cmd> <request.json> <response.json>

exactly like ``_salib_backend.py`` / ``_chaospy_backend.py``.  Heavy imports
(cvxpy, numpy) live inside the command functions so the shared env can import the
file for introspection without needing cvxpy.

SDP formulation (S-procedure / QCQP relaxation)
-----------------------------------------------
Lift ``x → z = [1; x]`` and relax ``Z = z zᵀ`` to ``Z ⪰ 0, Z₀₀ = 1`` (dropping the
rank-1 constraint).  Any quadratic ``q(x) = xᵀA x + bᵀx + c`` equals ``⟨M_q, Z⟩``
with ``M_q = [[c, bᵀ/2],[b/2, A]]``.  The box ``lb ≤ x ≤ ub`` enters as the
concave constraints ``(x_i−lb_i)(ub_i−x_i) ≥ 0``, and each experiment as
``L_e ≤ ⟨M_e, Z⟩ ≤ U_e``.  Because the relaxation is an *outer* approximation of
the feasible set, the optimum bounds the true one from outside — a rigorous
certificate (a negative consistency ⇒ provably inconsistent; every
parameter/prediction interval *contains* the true feasible range).
"""
from __future__ import annotations

import json
import sys


def _mq(A, b, c, d, np):
    """(d+1, d+1) symmetric matrix with zᵀ M z = xᵀA x + bᵀx + c, z=[1;x]."""
    M = np.zeros((d + 1, d + 1))
    M[0, 0] = c
    M[0, 1:] = np.asarray(b) / 2.0
    M[1:, 0] = np.asarray(b) / 2.0
    M[1:, 1:] = np.asarray(A)
    return M


def _common_constraints(Z, box, models, d, cp, np):
    cons = [Z >> 0, Z[0, 0] == 1]
    lb, ub = box["lb"], box["ub"]
    for i in range(d):
        A = np.zeros((d, d)); A[i, i] = -1.0
        b = np.zeros(d); b[i] = lb[i] + ub[i]
        c = -lb[i] * ub[i]
        cons.append(cp.trace(_mq(A, b, c, d, np) @ Z) >= 0)
    for m in models:
        Mm = _mq(m["A"], m["b"], m["c"], d, np)
        cons.append(cp.trace(Mm @ Z) >= m["L"])
        cons.append(cp.trace(Mm @ Z) <= m["U"])
    return cons


def _extremum(box, models, target, sense, d, cp, np):
    Z = cp.Variable((d + 1, d + 1), symmetric=True)
    cons = _common_constraints(Z, box, models, d, cp, np)
    obj = cp.trace(_mq(target["A"], target["b"], target["c"], d, np) @ Z)
    prob = cp.Problem(cp.Maximize(obj) if sense == "max" else cp.Minimize(obj), cons)
    prob.solve(solver=cp.SCS, verbose=False)
    return (None if prob.value is None else float(prob.value)), prob.status


def cmd_solve(request: dict) -> dict:
    """Consistency measure + parameter/prediction outer bounds in one request."""
    import cvxpy as cp
    import numpy as np

    d = int(request["d"])
    box = request["box"]
    models = request["models"]

    out: dict = {"solver": "SCS", "statuses": []}

    # ── consistency: max γ with |⟨M_e,Z⟩ − mid_e| ≤ (1−γ) half_e ─────────────
    Z = cp.Variable((d + 1, d + 1), symmetric=True)
    g = cp.Variable()
    cons = [Z >> 0, Z[0, 0] == 1]
    lb, ub = box["lb"], box["ub"]
    for i in range(d):
        A = np.zeros((d, d)); A[i, i] = -1.0
        b = np.zeros(d); b[i] = lb[i] + ub[i]
        cons.append(cp.trace(_mq(A, b, -lb[i] * ub[i], d, np) @ Z) >= 0)
    for m in models:
        mid = 0.5 * (m["L"] + m["U"])
        half = 0.5 * (m["U"] - m["L"])
        v = cp.trace(_mq(m["A"], m["b"], m["c"], d, np) @ Z)
        cons.append(v - mid <= half * (1 - g))
        cons.append(mid - v <= half * (1 - g))
    prob = cp.Problem(cp.Maximize(g), cons)
    prob.solve(solver=cp.SCS, verbose=False)
    out["consistency_measure"] = (None if prob.value is None else float(prob.value))
    out["consistent"] = (out["consistency_measure"] is not None
                         and out["consistency_measure"] >= 0.0)
    out["statuses"].append(prob.status)

    # ── parameter outer bounds: min/max of each x_i over the feasible set ────
    param_bounds = {}
    for i in range(d):
        A = np.zeros((d, d)); b = np.zeros(d); b[i] = 1.0
        lo, s1 = _extremum(box, models, {"A": A, "b": b, "c": 0.0}, "min", d, cp, np)
        hi, s2 = _extremum(box, models, {"A": A, "b": b, "c": 0.0}, "max", d, cp, np)
        param_bounds[str(i)] = [lo, hi]
        out["statuses"] += [s1, s2]
    out["parameter_bounds"] = param_bounds

    # ── prediction outer bounds: min/max of each target quadratic ────────────
    preds = []
    for t in request.get("predictions", []):
        lo, s1 = _extremum(box, models, t, "min", d, cp, np)
        hi, s2 = _extremum(box, models, t, "max", d, cp, np)
        preds.append({"name": t.get("name", "prediction"), "lower": lo, "upper": hi})
        out["statuses"] += [s1, s2]
    out["predictions"] = preds
    return out


_COMMANDS = {"solve": cmd_solve}


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
