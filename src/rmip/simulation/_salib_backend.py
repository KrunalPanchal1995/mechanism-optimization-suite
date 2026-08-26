"""
Standalone SALib sampling/analysis backend for ZerorkGSARunner (Phase 3,
ZERORK_GSA_PLAN.md), invoked as a subprocess under a dedicated venv --
NOT importable from the main rmip environment.

Why a separate venv/subprocess instead of a normal import: SALib 1.5.2
requires numpy>=2.0/scipy>=1.9.3/pandas>=2.0. This repo's main anaconda
environment is pinned to numpy 1.24.4 (scikit-learn, numba, and other
already-installed packages depend on the numpy 1.x ABI) -- installing
SALib directly into that environment once upgraded numpy to 2.0.2 and broke
scikit-learn's import (a real incident during this work, not a hypothetical
precaution). Isolating SALib in its own venv and talking to it only via
subprocess + JSON files follows the same D7 "subprocess + file contract,
never import" principle already used everywhere else in this codebase for
external tools (Zero-RK itself, CHEMKIN_PRO, FlameMaster).

Usage:
    <venv>/bin/python3 _salib_backend.py sample  <request.json> <response.json>
    <venv>/bin/python3 _salib_backend.py analyze <request.json> <response.json>

request.json (sample):
    {"method": "sobol"|"morris",
     "problem": {"num_vars": N, "names": [...], "bounds": [[lo,hi], ...]},
     "n_samples": int, "seed": int|null, "morris_num_levels": int}
response.json (sample):
    {"X": [[...], ...]}   -- sample matrix, row-major, shape (n_rows, N)

request.json (analyze):
    {"method": "sobol"|"morris", "problem": {...}, "Y": [...],
     "X": [[...], ...] (morris only), "seed": int|null,
     "morris_num_levels": int}
response.json (analyze):
    sobol:  {"S1": [...], "S1_conf": [...], "ST": [...], "ST_conf": [...]}
    morris: {"mu_star": [...], "mu_star_conf": [...], "sigma": [...]}
"""
import json
import sys


def cmd_sample(req: dict) -> dict:
    problem = req["problem"]
    method = req["method"]
    n_samples = req["n_samples"]
    seed = req.get("seed")

    if method == "sobol":
        from SALib.sample import sobol as sobol_sample
        X = sobol_sample.sample(problem, n_samples, calc_second_order=False, seed=seed)
    elif method == "morris":
        from SALib.sample import morris as morris_sample
        X = morris_sample.sample(
            problem, n_samples, num_levels=req.get("morris_num_levels", 4), seed=seed,
        )
    else:
        raise ValueError(f"unknown GSA method: {method!r}")
    return {"X": X.tolist()}


def cmd_analyze(req: dict) -> dict:
    import numpy as np

    problem = req["problem"]
    method = req["method"]
    Y = np.array(req["Y"], dtype=float)
    seed = req.get("seed")

    if method == "sobol":
        from SALib.analyze import sobol as sobol_analyze
        Si = sobol_analyze.analyze(problem, Y, calc_second_order=False, seed=seed)
        return {
            "S1": list(Si["S1"]), "S1_conf": list(Si["S1_conf"]),
            "ST": list(Si["ST"]), "ST_conf": list(Si["ST_conf"]),
        }
    elif method == "morris":
        from SALib.analyze import morris as morris_analyze
        X = np.array(req["X"], dtype=float)
        Si = morris_analyze.analyze(
            problem, X, Y, num_levels=req.get("morris_num_levels", 4), seed=seed,
        )
        return {
            "mu_star": list(Si["mu_star"]), "mu_star_conf": list(Si["mu_star_conf"]),
            "sigma": list(Si["sigma"]),
        }
    else:
        raise ValueError(f"unknown GSA method: {method!r}")


def main() -> None:
    if len(sys.argv) != 4 or sys.argv[1] not in ("sample", "analyze"):
        print("Usage: _salib_backend.py sample|analyze <request.json> <response.json>",
              file=sys.stderr)
        sys.exit(2)
    cmd, req_path, resp_path = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(req_path) as fh:
        req = json.load(fh)
    result = cmd_sample(req) if cmd == "sample" else cmd_analyze(req)
    with open(resp_path, "w") as fh:
        json.dump(result, fh)


if __name__ == "__main__":
    main()
