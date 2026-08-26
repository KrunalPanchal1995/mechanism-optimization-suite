#!/usr/bin/env python3
"""
bootstrap_forward_uq_cache.py — synthesize the minimal Opt/ + DM_FOR_PARTIAL_PRS/
cache forward_uq.py needs to run past its Opt/-not-found guard and actually
compute forward_uq_statistics.csv, WITHOUT running the full multi-hour
MUQ-SAC optimization pipeline (src/MUQ-SAC/run_script.py).

Call `build(workdir, target_opt_path, n_cases=2)` with the directory
forward_uq.py will be run from (its cwd == its `workdir`) and the target.opt
it will load. Derives the active-parameter list from the SAME uncertainty
XML forward_uq.py itself loads (via build_index_maps over Uncertainty.py's
uncertaintyData), rather than hard-coding '1_A'/'2_A'/'3_A', so it can't
drift out of sync with rIndex-format changes.

File shapes (verified against forward_uq.py:236-313,
MUQ-SAC/ResponseSurface.py:56-77,248-277,391-421 -- see
GUI_STRESS_TEST_PLAN.md / progress.md for the derivation):

  Opt/Data/ResponseSurface/responsecoef_case-{i}.csv
      "Coefficients\n" + one float per line. Count = 1 + n_comp + n_comp*
      (n_comp+1)/2 for a responseOrder=2 PRS (constant + linear + upper-
      triangular quadratic/cross terms, INCLUDING self-products) -- with
      n_comp=3 (all 3 reactions selected) that's 1+3+6=10.

  Opt/Data/Simulations/sim_data_case-{i}.lst
      Tab-delimited, >=2 columns; forward_uq.py's _read_sim_data() reads
      column index 1 (0-based) as the eta (ln-scale IDT) value.

  DM_FOR_PARTIAL_PRS/case-{i}/pDesignMatrix.csv
      Comma-delimited rows, n_comp columns each, values in [-1, 1] (compressed
      zeta design matrix). Row count must match sim_data's row count (after
      forward_uq.py's own min-length trim); ResponseSurface's internal
      train_test_split(test_size=0.1) + a `min_len>=4` guard mean >=12 rows
      gives a comfortable 10 train / 2 test.

  DM_FOR_PARTIAL_PRS/case-{i}/selected_parameters.csv
      One active-parameter string per line (must exactly match entries of
      flat_params_list, e.g. '1_A').
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


def _flat_params_list(uncertainty_xml: str, mechanism_yaml: str) -> list:
    """Derive the exact same flat_params_list forward_uq.py's own
    build_index_maps() would produce for this XML/mechanism pair, instead
    of hard-coding '1_A'/'2_A'/'3_A' (avoids rIndex-format drift)."""
    _repo = Path(__file__).resolve().parents[5]
    for p in (str(_repo / "src" / "MUQ-SAC"), str(_repo / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import Uncertainty as uncertainty_mod

    locations = {
        "uncertainty_data": uncertainty_xml,
        "mechanism": mechanism_yaml,
        "thermo_file": "",
        "trans_file": "",
    }
    dataset = uncertainty_mod.uncertaintyData(locations, {})
    unsrt_data = dataset.extract_uncertainty()
    flat_params_list = []
    for rxn_key, rxn_obj in unsrt_data.items():
        flat_params_list.extend(rxn_obj.activeParameters)
    return flat_params_list


def build(workdir: Path, uncertainty_xml: str, mechanism_yaml: str,
          n_cases: int = 2, n_samples: int = 16, seed: int = 0) -> list:
    workdir = Path(workdir)
    flat_params_list = _flat_params_list(uncertainty_xml, mechanism_yaml)
    n_comp = len(flat_params_list)
    n_coeffs = 1 + n_comp + n_comp * (n_comp + 1) // 2   # responseOrder=2

    rng = np.random.default_rng(seed)
    opt_dir = workdir / "Opt"
    (opt_dir / "Data" / "ResponseSurface").mkdir(parents=True, exist_ok=True)
    (opt_dir / "Data" / "Simulations").mkdir(parents=True, exist_ok=True)

    for i in range(n_cases):
        # -- ResponseSurface coefficients: small, finite, arbitrary but
        # deterministic (seeded) -- the surrogate just needs to evaluate.
        coeffs = rng.normal(loc=6.9, scale=0.05, size=n_coeffs)   # ~ln(962 us)
        coef_path = opt_dir / "Data" / "ResponseSurface" / f"responsecoef_case-{i}.csv"
        coef_path.write_text("Coefficients\n" + "\n".join(f"{c}" for c in coeffs) + "\n")

        # -- Simulation eta values + design matrix (compressed zeta space) --
        dm = rng.uniform(-1.0, 1.0, size=(n_samples, n_comp))
        etas = 6.9 + 0.05 * dm.sum(axis=1)   # mild dependence, still finite/sane
        sim_path = opt_dir / "Data" / "Simulations" / f"sim_data_case-{i}.lst"
        sim_path.write_text(
            "\n".join(f"case-{i}-{j}\t{eta}" for j, eta in enumerate(etas)) + "\n"
        )

        case_dir = workdir / "DM_FOR_PARTIAL_PRS" / f"case-{i}"
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "pDesignMatrix.csv").write_text(
            "\n".join(",".join(f"{v}" for v in row) for row in dm) + "\n"
        )
        (case_dir / "selected_parameters.csv").write_text(
            "\n".join(flat_params_list) + "\n"
        )

    return flat_params_list


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("workdir")
    p.add_argument("uncertainty_xml")
    p.add_argument("mechanism_yaml")
    p.add_argument("--n-cases", type=int, default=2)
    args = p.parse_args()
    params = build(Path(args.workdir), args.uncertainty_xml, args.mechanism_yaml,
                    n_cases=args.n_cases)
    print(f"Built synthetic PRS cache for {args.n_cases} case(s), "
          f"{len(params)} active params: {params}")
