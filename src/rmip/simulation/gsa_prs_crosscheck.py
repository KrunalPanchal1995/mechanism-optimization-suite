"""
rmip/simulation/gsa_prs_crosscheck.py
======================================
Standalone validation harness for MarkDownFiles/ZERORK_GSA_PLAN.md's Phase 4
open question #4: compare ZerorkGSARunner's direct-simulation Sobol indices
against AnalyticalSobolPRS's PRS-surrogate Sobol indices, for the same
reactions on a real mechanism -- "worth doing first" (per that plan) before
trusting Phase 3's GSA indices as a pre-filter for PRS training.

Scope (confirmed with KP): validation only. Does NOT modify
src/MUQ-SAC/run_script.py or wire a pre-filter into the PRS training
pipeline -- that is separate, deliberately deferred follow-up work. Every
production module this composes (ZerorkBridge, ZerorkGSARunner, ChemkinWriter,
Uncertainty, MechanismParser, MechManipulator2_0, DesignMatrix, ResponseSurface,
AnalyticalSobolPRS) is used unmodified, as a library.

Import notes
------------
- MUQ-SAC modules (Uncertainty/MechanismParser/MechManipulator2_0/
  DesignMatrix/ResponseSurface) are only importable with
  ``src/MUQ-SAC`` on ``sys.path`` (they cross-import each other by bare
  module name, e.g. ``import Uncertainty`` inside DesignMatrix.py).
  ``_ensure_muqsac_importable()`` does this lazily (inside functions that
  need it, not at module import time) so importing this module doesn't
  silently mutate ``sys.path`` for every other rmip caller.
- ``AnalyticalSobolPRS`` lives at ``src/forward_uq/sobol_prs.py`` -- imported
  via ``importlib.util.spec_from_file_location`` rather than
  ``from forward_uq.sobol_prs import ...``, because the repo-root
  ``forward_uq.py`` *script* and the ``src/forward_uq/`` *package* share the
  same module name; whichever sys.path order wins would shadow the other.
  Loading the file directly by path sidesteps that collision (the module
  only imports numpy/pandas, no further cross-imports, so this is safe).

Key finding this module encodes (see ZERORK_GSA_PLAN.md open question #4 for
the full writeup): ``Manipulator``'s A-factor-only perturbation uses
``sigma_A = min(reaction.uncertainties)`` per reaction (~0.13-0.16 for the
H2/O2 trio used below) -- a much narrower domain than
``ZerorkGSARunner.run_screening()``'s own default ``factor_bound=2.0``
(``ln(2) ~= 0.69``). An apples-to-apples comparison needs the direct-sim
Sobol run to sample the SAME per-reaction bound the PRS was trained on --
see ``run_direct_sobol_per_reaction_bounds()``, which composes
``ZerorkGSARunner``'s private sampling/matrix/parsing helpers with a
per-reaction bounds list instead of one shared bound.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from rmip.types import SimCondition, SobolResult

log = logging.getLogger("rmip.simulation.gsa_prs_crosscheck")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MUQSAC_SRC = _REPO_ROOT / "src" / "MUQ-SAC"
_SOBOL_PRS_PATH = _REPO_ROOT / "src" / "forward_uq" / "sobol_prs.py"

_DEFAULT_PYTHON39 = "/home/krithika/anaconda3/bin/python3.9"
_DEFAULT_CK2YAML = "/home/krithika/anaconda3/bin/ck2yaml"

# PRS validity gate -- same MRE_criteria default used in every real
# target.opt Stats block in this repo, so "the PRS is good enough" means the
# same thing here as everywhere else.
_DEFAULT_MRE_CRITERIA = 5.0


def _ensure_muqsac_importable() -> None:
    for p in (str(_REPO_ROOT / "src"), str(_MUQSAC_SRC)):
        if p not in sys.path:
            sys.path.insert(0, p)


_AnalyticalSobolPRS = None


def _load_analytical_sobol_prs_class():
    """Load AnalyticalSobolPRS directly by file path (see module docstring
    for why this avoids ``import forward_uq``)."""
    global _AnalyticalSobolPRS
    if _AnalyticalSobolPRS is None:
        spec = importlib.util.spec_from_file_location(
            "rmip_gsa_crosscheck_sobol_prs", str(_SOBOL_PRS_PATH)
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _AnalyticalSobolPRS = module.AnalyticalSobolPRS
    return _AnalyticalSobolPRS


# ---------------------------------------------------------------------------
# 1. Nominal mechanism preparation
# ---------------------------------------------------------------------------

def prepare_nominal_cantera_yaml(
    db_path: str, mechanism_id: int, work_dir: str, ck2yaml_exe: str = _DEFAULT_CK2YAML,
) -> Path:
    """DB -> CHEMKIN (ChemkinWriter) -> Cantera YAML (ck2yaml). Both the
    direct-sim and PRS-training sides perturb copies of this SAME base
    mechanism, so the comparison is apples-to-apples."""
    from rmip.simulation.zerork_bridge import ZerorkBridge
    from rmip.io.chemkin_writer import ChemkinWriter

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    bridge = ZerorkBridge(db_path, "/home/krithika/zero-rk/build2/inst_dir")
    chemkin_path = bridge.export_mechanism(mechanism_id, str(work))

    writer = ChemkinWriter(db_path)
    yaml_path = work / f"mech_{mechanism_id}.yaml"
    writer.convert_to_cantera_yaml(
        str(chemkin_path), output_yaml=str(yaml_path), ck2yaml_exe=ck2yaml_exe,
    )
    return yaml_path


# ---------------------------------------------------------------------------
# 2. Uncertainty subset (fixture XML already scoped to just the target reactions)
# ---------------------------------------------------------------------------

def build_unsrt_subset(xml_path: str, cantera_yaml_path: str) -> dict:
    """{rxn_key: Uncertainty.reaction}. DesignMatrix always iterates every
    key handed to it -- this is why the fixture XML is scoped down to just
    the target reactions rather than filtering a full-mechanism dict."""
    _ensure_muqsac_importable()
    import Uncertainty

    paths = {
        "uncertainty_data": xml_path,
        "mechanism": cantera_yaml_path,
        "thermo_file": "",
        "trans_file": "",
    }
    return Uncertainty.uncertaintyData(paths, {}).extract_uncertainty()


# ---------------------------------------------------------------------------
# 3. Design matrix (ζ in [-1,1]^n, A-factor-only)
# ---------------------------------------------------------------------------

def build_design_matrix(
    unsrt_subset: dict, work_dir: str, n_design: int = 40,
    method: str = "sobol", seed: int = 42,
) -> np.ndarray:
    """DesignMatrix writes DesignMatrix.csv to cwd -- run inside work_dir to
    keep that a scoped side effect, not a repo-root pollutant."""
    _ensure_muqsac_importable()
    from DesignMatrix import DesignMatrix

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    original_dir = Path.cwd()
    try:
        import os
        os.chdir(work)
        dm = DesignMatrix(
            unsrt_subset, "A-facto", n_design, ind=len(unsrt_subset),
            method=method, seed=seed,
        )
        return dm.getSamples()
    finally:
        import os
        os.chdir(original_dir)


# ---------------------------------------------------------------------------
# 4. Training data: perturb + convert + run Zero-RK IDT, per design row
# ---------------------------------------------------------------------------

def run_training_point(
    nominal_yaml_path: str, unsrt_subset: dict, zeta_row: np.ndarray,
    mechanism_id: int, condition: SimCondition, row_work_dir: str,
    db_path: str, zerork_install_dir: str,
    python39: str = _DEFAULT_PYTHON39, timeout: float = 60.0,
) -> Optional[float]:
    """One training point: perturb the nominal mechanism at zeta_row via
    MechManipulator2_0.Manipulator, convert to CHEMKIN-ASCII via
    soln2ck_2.py (the same converter make_input_file.py's zerork Tig branch
    already uses for this exact conversion), run Zero-RK IDT. Returns idt_s,
    or None if the point failed to converge (dropped -- one fewer training
    sample, not a fatal error)."""
    _ensure_muqsac_importable()
    from MechanismParser import Parser
    from MechManipulator2_0 import Manipulator
    from rmip.simulation.zerork_bridge import ZerorkBridge

    row = Path(row_work_dir)
    row.mkdir(parents=True, exist_ok=True)

    mech = Parser(nominal_yaml_path).mech
    new_mech, _ = Manipulator(deepcopy(mech), unsrt_subset, np.asarray(zeta_row)).doPerturbation()
    perturbed_yaml = row / "perturbed.yaml"
    perturbed_yaml.write_text(yaml.safe_dump(new_mech, default_flow_style=False))

    proc = subprocess.run(
        [python39, str(_MUQSAC_SRC / "soln2ck_2.py"), str(perturbed_yaml),
         "--mechanism=mechanism.inp"],
        cwd=str(row), capture_output=True, text=True, timeout=timeout,
    )
    mech_inp = row / "mechanism.inp"
    if proc.returncode != 0 or not mech_inp.exists():
        log.warning("soln2ck_2.py failed for row in %s (rc=%s): %s",
                    row, proc.returncode, proc.stderr[-500:])
        return None

    bridge = ZerorkBridge(db_path, zerork_install_dir)
    try:
        idt_input = bridge.build_idt_input(mech_inp, mechanism_id, condition, str(row / "idt"))
        idt_proc = bridge.run_idt(idt_input, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    if idt_proc.returncode != 0:
        log.warning("Zero-RK IDT failed for row in %s: %s", row, idt_proc.stdout[-500:])
        return None

    result = bridge.parse_idt_output(str(row / "idt"), mechanism_id, condition)
    return result.idt_s if result.converged else None


def collect_training_data(
    nominal_yaml_path: str, unsrt_subset: dict, design_matrix: np.ndarray,
    mechanism_id: int, condition: SimCondition, work_dir: str,
    db_path: str, zerork_install_dir: str, python39: str = _DEFAULT_PYTHON39,
) -> Tuple[List[List[float]], List[float]]:
    """Runs every design-matrix row, drops non-converged rows. Returns
    (zeta_rows, idt_s_values) -- same length, positionally aligned."""
    work = Path(work_dir)
    zeta_rows: List[List[float]] = []
    idt_values: List[float] = []
    n_dropped = 0
    for i, zeta_row in enumerate(design_matrix):
        idt_s = run_training_point(
            nominal_yaml_path, unsrt_subset, zeta_row, mechanism_id, condition,
            str(work / f"train_row_{i:03d}"), db_path, zerork_install_dir, python39,
        )
        if idt_s is None:
            n_dropped += 1
            continue
        zeta_rows.append(list(zeta_row))
        idt_values.append(idt_s)
    if n_dropped:
        log.warning("%d/%d training rows dropped (non-converged)", n_dropped, len(design_matrix))

    # M19: the fit is a 2nd-order "Full" PRS over p = design_matrix.shape[1]
    # parameters, which has n_coeff = 1 + 2p + C(p,2) coefficients. A fit with
    # fewer converged rows than coefficients is underdetermined, so its Sobol
    # indices -- and the AGREE/DISAGREE verdict built on them -- are meaningless.
    # The old `< 4` gate let e.g. a 3-parameter (10-coefficient) fit through on
    # 4 rows. Require enough converged rows that the training half (after the
    # 10% test split in fit_prs) still over-determines the fit.
    p = int(design_matrix.shape[1]) if design_matrix.ndim == 2 else len(zeta_rows[0])
    n_coeff = 1 + 2 * p + p * (p - 1) // 2
    min_rows = math.ceil((n_coeff + 1) / 0.9)  # +1 margin, then undo the 10% test split
    if len(zeta_rows) < min_rows:
        raise RuntimeError(
            f"Only {len(zeta_rows)} training rows converged, but an order-2 "
            f"'Full' PRS over {p} parameters has {n_coeff} coefficients. "
            f"Need >= {min_rows} converged rows (accounting for the 10% test "
            f"split) for a well-posed fit; below this the PRS Sobol indices and "
            f"the AGREE/DISAGREE crosscheck verdict are not meaningful."
        )
    return zeta_rows, idt_values


# ---------------------------------------------------------------------------
# 5. Fit the PRS (order 2, "Full" -- our subset dict already IS the intended scope)
# ---------------------------------------------------------------------------

def fit_prs(zeta_rows: List[List[float]], idt_values: List[float], work_dir: str,
            criteria: float = _DEFAULT_MRE_CRITERIA):
    """Fits a 2nd-order ResponseSurface. Must run with a `Data/ResponseSurface/`
    directory under cwd (ResponseSurface.create_response_surface()/test()
    write cache/diagnostic files there unconditionally) -- work_dir is used
    as a scratch cwd, chdir'd into and back out of, mirroring forward_uq.py's
    own load_prs_from_cache() pattern."""
    _ensure_muqsac_importable()
    import ResponseSurface as PRS_module
    from sklearn.model_selection import train_test_split
    import os

    work = Path(work_dir)
    (work / "Data" / "ResponseSurface").mkdir(parents=True, exist_ok=True)

    X = np.asarray(zeta_rows, dtype=float)
    Y = np.asarray(idt_values, dtype=float)
    xTrain, xTest, yTrain, yTest = train_test_split(
        X, Y, random_state=104, test_size=0.1, shuffle=True,
    )

    original_dir = Path.cwd()
    try:
        os.chdir(work)
        prs = PRS_module.ResponseSurface(
            xTrain, yTrain, "case-crosscheck", "crosscheck",
            responseOrder=2, prs_type="Full", criteria=criteria,
        )
        prs.create_response_surface()
        prs.xTest, prs.yTest = xTest, yTest
        prs.test(xTest, yTest)
    finally:
        os.chdir(original_dir)
    return prs


# ---------------------------------------------------------------------------
# 6. PRS-surrogate Sobol (analytic, from PRS coefficients)
# ---------------------------------------------------------------------------

def _build_index_maps(unsrt_data: dict) -> Tuple[List[str], Dict[int, Tuple[str, str]], Dict[str, List[int]], int]:
    """Reimplemented locally (not imported from root forward_uq.py -- see
    module docstring's Import notes for why): flat ζ-vector <-> reaction/
    parameter index maps. Logic mirrors forward_uq.py's build_index_maps()
    exactly."""
    flat_params_list: List[str] = []
    rxn_for_flat: Dict[int, Tuple[str, str]] = {}
    flat_for_rxn: Dict[str, List[int]] = {}
    flat_idx = 0
    for rxn_key, rxn_obj in unsrt_data.items():
        flat_for_rxn[rxn_key] = []
        for param_str in rxn_obj.activeParameters:
            flat_params_list.append(param_str)
            rxn_for_flat[flat_idx] = (rxn_key, param_str)
            flat_for_rxn[rxn_key].append(flat_idx)
            flat_idx += 1
    return flat_params_list, rxn_for_flat, flat_for_rxn, flat_idx


def _build_comp_to_rxn(prs, rxn_for_flat: Dict[int, Tuple[str, str]]) -> Dict[int, Tuple[str, str]]:
    """Reimplemented locally, mirrors forward_uq.py's build_comp_to_rxn()
    exactly: prs._active_idx lists full-space indices of selected params."""
    return {
        comp_i: rxn_for_flat[full_i]
        for comp_i, full_i in enumerate(prs._active_idx)
        if full_i in rxn_for_flat
    }


def compute_prs_sobol(prs, unsrt_subset: dict) -> Dict[str, Dict]:
    """{rxn_key: {"S1", "ST", "V_i", "V_Ti", "active"}} via analytic ANOVA
    decomposition of the fitted PRS's coefficients -- no extra simulations."""
    AnalyticalSobolPRS = _load_analytical_sobol_prs_class()
    _, rxn_for_flat, _, _ = _build_index_maps(unsrt_subset)
    comp_to_rxn = _build_comp_to_rxn(prs, rxn_for_flat)
    rxn_keys = list(unsrt_subset.keys())
    return AnalyticalSobolPRS(prs, comp_to_rxn, rxn_keys).compute()


# ---------------------------------------------------------------------------
# 7. Direct-sim Sobol via Zero-RK, per-reaction bounds (not run_screening()'s
#    single shared factor_bound -- see module docstring)
# ---------------------------------------------------------------------------

def run_direct_sobol_per_reaction_bounds(
    runner, mechanism_id: int, reaction_ids: List[int], sigma_A: Dict[int, float],
    condition: SimCondition, n_base_samples: int, work_dir: str,
    idt_temp_delta: float = 400.0, seed: Optional[int] = None,
    n_mpi_ranks: int = 8, method: str = "sobol", morris_num_levels: int = 4,
    timeout: Optional[float] = None,
) -> SobolResult:
    """
    Mirrors ZerorkGSARunner.run_screening()'s body (sample -> expand matrix
    -> write gsaMatrixFile -> run perturbAFactorGSA_mpi.x -> parse ->
    analyze), composed from its private helpers, but with
    problem['bounds'] = [[-sigma_A[rid], sigma_A[rid]] for rid in
    reaction_ids] instead of one shared ln(factor_bound) -- so the direct-sim
    side samples the SAME per-reaction domain the PRS was trained on.

    Reuses runner._all_reactions / runner._salib_call / runner._build_full_matrix
    / runner._write_gsa_matrix / runner._parse_gsa_output / runner._bridge /
    runner._bin_dir / runner._mpiexec directly -- private-but-stable reuse
    within the same package. See test_gsa_prs_crosscheck.py's
    TestPerReactionBoundsMatchesRunScreening for a regression pin against
    run_screening() itself: if zerork_gsa.py's internals ever change
    underneath this composition, that test fails loudly.
    """
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    all_reactions = runner._all_reactions(mechanism_id)
    all_ids = [rid for rid, _ in all_reactions]
    name_of = dict(all_reactions)

    unknown = set(reaction_ids) - set(all_ids)
    if unknown:
        raise ValueError(f"reaction_ids not found in mechanism_id={mechanism_id}: {unknown}")
    selected_names = [name_of[rid] for rid in reaction_ids]

    problem = {
        "num_vars": len(reaction_ids),
        "names": selected_names,
        "bounds": [[-sigma_A[rid], sigma_A[rid]] for rid in reaction_ids],
    }

    sample_resp = runner._salib_call(
        "sample",
        {"method": method, "problem": problem, "n_samples": n_base_samples,
         "seed": seed, "morris_num_levels": morris_num_levels},
        work,
    )
    X = sample_resp["X"]
    n_rows = len(X)

    full_matrix = runner._build_full_matrix(X, all_ids, reaction_ids)
    matrix_path = work / "gsa_matrix.mtx"
    runner._write_gsa_matrix(matrix_path, full_matrix, len(all_ids))

    mech_path = runner._bridge.export_mechanism(mechanism_id, str(work))
    fuel_dict, oxidizer_dict = runner._bridge._resolve_composition(mechanism_id, condition)

    out_path = work / "gsa_output.dat"
    check_path = work / "gsa_check.dat"
    gsa_cfg = {
        "mechFile": str(mech_path),
        "thermFile": str(mech_path),
        "mechLogFile": str(work / "mech.clog"),
        "gsaMatrixFile": str(matrix_path),
        "outFile": str(out_path),
        "checkFile": str(check_path),
        "initTemp": condition.T_K,
        "initPres": condition.P_atm * 101325.0,
        "initPhi": condition.phi,
        "fuelComp": fuel_dict,
        "oxidizerComp": oxidizer_dict,
        "maxTime": 1.0,
        "idtTemps": [idt_temp_delta],
        "relTol": 1e-8,
        "absTol": 1e-20,
    }
    gsa_yml_path = work / "gsa_input.yml"
    with open(gsa_yml_path, "w") as fh:
        yaml.safe_dump(gsa_cfg, fh, default_flow_style=False)

    exe = runner._bin_dir / "perturbAFactorGSA_mpi.x"
    n_ranks = max(2, min(n_mpi_ranks, n_rows + 1))
    cmd = [runner._mpiexec, "-n", str(n_ranks), str(exe), gsa_yml_path.name]
    log.info("Running per-reaction-bounds direct Sobol: %s (cwd=%s, %d samples)",
              " ".join(cmd), work, n_rows)
    proc = subprocess.run(cmd, cwd=str(work), capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"perturbAFactorGSA_mpi.x failed (rc={proc.returncode}):\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )

    Y = runner._parse_gsa_output(out_path, n_rows)
    n_failed = sum(1 for y in Y if y != y)
    if n_failed:
        log.warning("%d/%d direct-sim GSA samples failed to ignite -- mean-imputed",
                    n_failed, n_rows)
    Y_clean = [y for y in Y if y == y]
    if not Y_clean:
        raise RuntimeError("All direct-sim GSA samples failed to ignite")
    y_mean = sum(Y_clean) / len(Y_clean)
    Y_filled = [y if y == y else y_mean for y in Y]

    analyze_req = {"method": method, "problem": problem, "Y": Y_filled,
                    "seed": seed, "morris_num_levels": morris_num_levels}
    if method == "morris":
        analyze_req["X"] = X
    analyze_resp = runner._salib_call("analyze", analyze_req, work)

    result = SobolResult(
        mechanism_id=mechanism_id, method=method, parameter_names=selected_names,
        n_samples=n_rows, n_failed_samples=n_failed,
    )
    if method == "sobol":
        result.S1 = dict(zip(selected_names, analyze_resp["S1"]))
        result.S1_conf = dict(zip(selected_names, analyze_resp["S1_conf"]))
        result.ST = dict(zip(selected_names, analyze_resp["ST"]))
        result.ST_conf = dict(zip(selected_names, analyze_resp["ST_conf"]))
    else:
        result.mu_star = dict(zip(selected_names, analyze_resp["mu_star"]))
        result.mu_star_conf = dict(zip(selected_names, analyze_resp["mu_star_conf"]))
        result.sigma = dict(zip(selected_names, analyze_resp["sigma"]))
    return result


# ---------------------------------------------------------------------------
# 8. Compare + report
# ---------------------------------------------------------------------------

@dataclass
class CrossCheckResult:
    mechanism_id: int
    reaction_keys: List[str]
    prs_test_error_pct: float
    prs_valid: bool
    S1_prs: Dict[str, float] = field(default_factory=dict)
    ST_prs: Dict[str, float] = field(default_factory=dict)
    S1_direct: Dict[str, float] = field(default_factory=dict)
    S1_direct_conf: Dict[str, float] = field(default_factory=dict)
    ST_direct: Dict[str, float] = field(default_factory=dict)
    ST_direct_conf: Dict[str, float] = field(default_factory=dict)
    rank_agrees: bool = False
    max_abs_delta_S1: float = 0.0
    verdict: str = "PRS_INVALID"

    def to_dict(self) -> dict:
        return {
            "mechanism_id": self.mechanism_id, "reaction_keys": self.reaction_keys,
            "prs_test_error_pct": self.prs_test_error_pct, "prs_valid": self.prs_valid,
            "S1_prs": self.S1_prs, "ST_prs": self.ST_prs,
            "S1_direct": self.S1_direct, "S1_direct_conf": self.S1_direct_conf,
            "ST_direct": self.ST_direct, "ST_direct_conf": self.ST_direct_conf,
            "rank_agrees": self.rank_agrees, "max_abs_delta_S1": self.max_abs_delta_S1,
            "verdict": self.verdict,
        }

    def report_text(self) -> str:
        lines = [
            f"GSA/PRS Sobol cross-check -- mechanism_id={self.mechanism_id}",
            f"PRS test error: {self.prs_test_error_pct:.2f}% (valid={self.prs_valid})",
            f"Verdict: {self.verdict}",
            "",
            f"{'reaction':<30} {'S1_prs':>10} {'S1_direct':>12} {'delta':>10}",
        ]
        for rxn in self.reaction_keys:
            s1p = self.S1_prs.get(rxn, float("nan"))
            s1d = self.S1_direct.get(rxn, float("nan"))
            lines.append(f"{rxn:<30} {s1p:>10.4f} {s1d:>12.4f} {abs(s1p - s1d):>10.4f}")
        return "\n".join(lines)


def compare(
    reaction_keys: List[str], prs_test_error_pct: float, mre_criteria: float,
    S1_prs: Dict[str, float], S1_direct: Dict[str, float],
    ST_prs: Dict[str, float], ST_direct: Dict[str, float],
    S1_direct_conf: Dict[str, float], ST_direct_conf: Dict[str, float],
    mechanism_id: int,
) -> CrossCheckResult:
    prs_valid = bool(prs_test_error_pct <= mre_criteria)
    result = CrossCheckResult(
        mechanism_id=mechanism_id, reaction_keys=reaction_keys,
        prs_test_error_pct=float(prs_test_error_pct), prs_valid=prs_valid,
        S1_prs=S1_prs, ST_prs=ST_prs, S1_direct=S1_direct, S1_direct_conf=S1_direct_conf,
        ST_direct=ST_direct, ST_direct_conf=ST_direct_conf,
    )
    if not prs_valid:
        result.verdict = "PRS_INVALID"
        return result

    argmax_prs = max(reaction_keys, key=lambda r: S1_prs.get(r, 0.0))
    argmax_direct = max(reaction_keys, key=lambda r: S1_direct.get(r, 0.0))
    result.rank_agrees = bool(argmax_prs == argmax_direct)

    deltas = [abs(S1_direct.get(r, 0.0) - S1_prs.get(r, 0.0)) for r in reaction_keys]
    result.max_abs_delta_S1 = float(max(deltas)) if deltas else 0.0

    magnitude_ok = all(
        abs(S1_direct.get(r, 0.0) - S1_prs.get(r, 0.0))
        <= max(0.10, 2 * S1_direct_conf.get(r, 0.0))
        for r in reaction_keys
    )
    result.verdict = "AGREE" if (result.rank_agrees and magnitude_ok) else "DISAGREE"
    return result


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run_crosscheck(
    db_path: str, zerork_install_dir: str, mechanism_id: int,
    xml_path: str, condition: SimCondition, work_dir: str,
    n_design: int = 40, n_direct_base: int = 512, seed: int = 42,
    mre_criteria: float = _DEFAULT_MRE_CRITERIA,
    python39: str = _DEFAULT_PYTHON39, ck2yaml_exe: str = _DEFAULT_CK2YAML,
    n_mpi_ranks: int = 8,
) -> CrossCheckResult:
    """Runs the full pipeline end to end and writes
    work_dir/crosscheck_report.json + crosscheck_report.txt."""
    from rmip.simulation.zerork_gsa import ZerorkGSARunner
    import sqlite3

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    nominal_yaml = prepare_nominal_cantera_yaml(
        db_path, mechanism_id, str(work / "nominal"), ck2yaml_exe=ck2yaml_exe,
    )
    unsrt_subset = build_unsrt_subset(xml_path, str(nominal_yaml))
    reaction_keys = list(unsrt_subset.keys())

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, formula_string FROM reactions WHERE mechanism_id=?",
            (mechanism_id,),
        ).fetchall()
    id_of_formula = {formula: rid for rid, formula in rows}
    reaction_ids = [id_of_formula[rxn] for rxn in reaction_keys]
    sigma_A = {
        id_of_formula[rxn]: float(np.asarray(unsrt_subset[rxn].cholskyDeCorrelateMat).ravel()[0])
        for rxn in reaction_keys
    }

    design_matrix = build_design_matrix(
        unsrt_subset, str(work / "design"), n_design=n_design, method="sobol", seed=seed,
    )
    zeta_rows, idt_values = collect_training_data(
        str(nominal_yaml), unsrt_subset, design_matrix, mechanism_id, condition,
        str(work / "training"), db_path, zerork_install_dir, python39,
    )
    prs = fit_prs(zeta_rows, idt_values, str(work / "prs"), criteria=mre_criteria)
    prs_sobol = compute_prs_sobol(prs, unsrt_subset)

    runner = ZerorkGSARunner(db_path, zerork_install_dir)
    direct_result = run_direct_sobol_per_reaction_bounds(
        runner, mechanism_id, reaction_ids, sigma_A, condition, n_direct_base,
        str(work / "direct"), seed=seed, n_mpi_ranks=n_mpi_ranks,
    )
    name_of_id = dict(runner._all_reactions(mechanism_id))
    direct_name_for_rxn = {rxn: name_of_id[rid] for rxn, rid in zip(reaction_keys, reaction_ids)}

    S1_prs = {rxn: prs_sobol[rxn]["S1"] for rxn in reaction_keys}
    ST_prs = {rxn: prs_sobol[rxn]["ST"] for rxn in reaction_keys}
    S1_direct = {rxn: direct_result.S1[direct_name_for_rxn[rxn]] for rxn in reaction_keys}
    ST_direct = {rxn: direct_result.ST[direct_name_for_rxn[rxn]] for rxn in reaction_keys}
    S1_direct_conf = {rxn: direct_result.S1_conf[direct_name_for_rxn[rxn]] for rxn in reaction_keys}
    ST_direct_conf = {rxn: direct_result.ST_conf[direct_name_for_rxn[rxn]] for rxn in reaction_keys}

    result = compare(
        reaction_keys, prs.ytestMaxError, mre_criteria,
        S1_prs, S1_direct, ST_prs, ST_direct, S1_direct_conf, ST_direct_conf,
        mechanism_id,
    )
    (work / "crosscheck_report.json").write_text(json.dumps(result.to_dict(), indent=2))
    (work / "crosscheck_report.txt").write_text(result.report_text())
    return result
