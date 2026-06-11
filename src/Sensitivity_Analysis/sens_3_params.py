"""
sens_3_params.py
===================
3-Parameter (ln(A), n, Ea/R) brute-force sensitivity analysis using
forward-difference or central-difference finite-difference schemes
applied to Arrhenius kinetic parameters.

Sensitivity coefficient definition
-----------------------------------
For a generic observable η (e.g. ignition delay time) and reduced
parameter ζ_x (x ∈ {A, n, Ea}):

  S_x  =  f_norm_x  ×  Δη / η₀

where the finite-difference quotient Δη/η₀ is:

  Forward  (1st-order):   (η⁺ − η₀) / η₀
  Central  (2nd-order):   (η⁺ − η⁻) / (2 · η₀)

and the normalization factors are:
  A  :  f_norm_A  = ln(A₀) / δ_A          [A₀ = P_o[0], δ_A = Δln(A)]
  n  :  f_norm_n  = 1 / δ_n               [δ_n = Δn]
  Ea :  f_norm_Ea = (Ea₀/R) / δ_Ea        [Ea₀/R = P_o[2], δ_Ea = Δ(Ea/R)]

Usage
-----
    python sens_3_params.py <input_file.yaml>

Required YAML key (under Stats):
    SA_scheme: "forward_difference"   # or "central_difference"
    (Defaults to "forward_difference" if absent.)
"""

# ============================================================
# SECTION 1 — IMPORTS
# ============================================================
import numpy as np
import scipy as sp
import scipy.stats as stats
from scipy.optimize import minimize
import os, sys, re, threading, subprocess, time
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('figure', max_open_warning=0)
from scipy.linalg import block_diag
from scipy import optimize as spopt
import json
import multiprocessing
import concurrent.futures
import pickle
try:
    import ruamel_yaml as yaml
except ImportError:
    from ruamel import yaml
import pandas as pd
import yaml
sys.path.append('/parallel_yaml_writer.so')
import parallel_yaml_writer

import reaction_selection as rs
from MechManipulator2_0 import Manipulator
from copy import deepcopy

import combustion_target_class
import data_management
import simulation_manager2_0 as simulator
import Uncertainty as uncertainty
import DesignMatrix2_0 as DM
import ResponseSurface as PRS
import VisualAid as VA

# ============================================================
# SECTION 2 — LOGGING SETUP
# ============================================================
import logging
from datetime import datetime

_log_filename = f"SA_3P_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(_log_filename),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)
log.info(f"{'='*64}")
log.info(f"  SA_3P_BruteForce.py — Session start")
log.info(f"  Log file : {_log_filename}")
log.info(f"{'='*64}")

# ============================================================
# SECTION 3 — INPUT FILE & KEY-WORD CONSTANTS
# ============================================================
log.info("SECTION 3 — Loading input file")

# ── keyword constants ──────────────────────────────────────
optType         = "optimization_type"
targets         = "targets"
mech            = "mechanism"
pre_file        = "Initial_pre_file"
count           = "Counts"
countTar        = "targets_count"
home_dir        = os.getcwd()
fuel            = "fuel"
fuelClass       = "fuelClass"
bin_solve       = "solver_bin"
bin_opt         = "bin"
globRxn         = "global_reaction"
countThreads    = "parallel_threads"
unsrt           = "uncertainty_data"
thermoF         = "thermo_file"
transF          = "trans_file"
order           = "Order_of_PRS"
startProfile    = "StartProfilesData"
design          = "Design_of_PRS"
countRxn        = "total_reactions"
fT              = "fileType"
add             = "addendum"

if len(sys.argv) > 1:
    input_file = open(sys.argv[1], 'r')
    optInputs  = yaml.safe_load(input_file)
    log.info(f"  Input file '{sys.argv[1]}' loaded successfully.")
else:
    log.error("No input file provided.")
    log.error("Usage: python SA_3P_BruteForce.py <input_file.yaml>")
    sys.exit(1)

iFile                   = str(os.getcwd()) + "/" + str(sys.argv[1])
dataCounts              = optInputs[count]
binLoc                  = optInputs["Bin"]
inputs                  = optInputs["Inputs"]
locations               = optInputs["Locations"]
startProfile_location   = optInputs[startProfile]
stats_                  = optInputs["Stats"]
global A_fact_samples
A_fact_samples          = stats_["Sampling_of_PRS"]

# ── optional key defaults ──────────────────────────────────
if "sensitive_parameters" not in stats_:
    stats_["sensitive_parameters"] = "Principle_SubMatrix"
    optInputs["Stats"]["sensitive_parameters"] = "Principle_SubMatrix"
if "Arrhenius_Selection_Type" not in stats_:
    stats_["Arrhenius_Selection_Type"] = "some"
    optInputs["Stats"]["Arrhenius_Selection_Type"] = "some"

# ── SA scheme ──────────────────────────────────────────────
sa_scheme = stats_.get("SA_scheme", "forward_difference")
if sa_scheme not in ("forward_difference", "central_difference"):
    log.warning(f"  Unknown SA_scheme='{sa_scheme}'. Defaulting to 'forward_difference'.")
    sa_scheme = "forward_difference"
log.info(f"  SA scheme          : {sa_scheme.upper()}")

unsrt_location       = locations[unsrt]
mech_file_location   = locations[mech]
thermo_file_location = locations[thermoF]
trans_file_location  = locations[transF]
fileType             = inputs[fT]
samap_executable     = optInputs["Bin"]["samap_executable"]
jpdap_executable     = optInputs["Bin"]["jpdap_executable"]

file_specific_input  = "-f chemkin" if fileType == "chemkin" else ""
fuel                 = inputs[fuel]
gr                   = inputs[globRxn]
global_reaction      = gr

design_type      = stats_[design]
parallel_threads = dataCounts[countThreads]
targets_count    = int(dataCounts["targets_count"])
rps_order        = stats_[order]
PRS_type         = stats_["PRS_type"]

log.info(f"  Parallel threads   : {parallel_threads}")
log.info(f"  Targets count      : {targets_count}")

# ============================================================
# SECTION 4 — TARGET LOADING
# ============================================================
log.info(f"{'='*64}")
log.info("SECTION 4 — Loading combustion targets")

targetLines = open(locations[targets], 'r').readlines()
addendum    = yaml.safe_load(open(locations[add], 'r').read())

target_list   = []
c_index       = 0
string_target = ""
for target in targetLines[:targets_count]:
    if "#" in target:
        target = target[:target.index('#')]
    add_copy = deepcopy(addendum)
    t = combustion_target_class.combustion_target(target, add_copy, c_index)
    string_target += (
        f"{t.dataSet_id}|{t.target}|{t.species_dict}|"
        f"{t.temperature}|{t.pressure}|{t.phi}|"
        f"{t.observed}|{t.std_dvtn}\n"
    )
    c_index += 1
    target_list.append(t)

case_dir = range(0, len(target_list))
log.info(f"  Targets loaded     : {len(target_list)}")
log.info(f"  Case indices       : {list(case_dir)}")

# ============================================================
# SECTION 5 — UNCERTAINTY DATA
# ============================================================
log.info(f"{'='*64}")
log.info("SECTION 5 — Uncertainty quantification")

if "unsrt.pkl" not in os.listdir():
    log.info("  Running uncertainty analysis (this may take a while) ...")
    UncertDataSet = uncertainty.uncertaintyData(locations, binLoc)
    unsrt_data    = UncertDataSet.extract_uncertainty()
    with open('unsrt.pkl', 'wb') as f_:
        pickle.dump(unsrt_data, f_)
    log.info("  Uncertainty analysis complete — saved to 'unsrt.pkl'.")
else:
    with open('unsrt.pkl', 'rb') as f_:
        unsrt_data = pickle.load(f_)
    log.info("  Uncertainty data loaded from 'unsrt.pkl'.")

# ============================================================
# SECTION 6 — MECHANISM PARSING
# ============================================================
log.info(f"{'='*64}")
log.info("SECTION 6 — Parsing mechanism and building reaction index")

with open(mech_file_location, 'r') as f_:
    yaml_mech = f_.read()
mechanism    = yaml.safe_load(yaml_mech)
species      = mechanism['phases'][0]["species"]
species_data = mechanism["species"]
reactions    = mechanism["reactions"]

selected_reactions = [rxn for rxn in unsrt_data]
reaction_dict      = rs.reaction_index(selected_reactions, reactions)
rxn_type           = rs.getRxnType(mechanism, selected_reactions)

string_f   = ""
string_g   = ""
index_dict = {}
for index in reaction_dict:
    index_dict[reaction_dict[index]] = index
    string_f += f"{index}\t{reaction_dict[index]}\n"
for rxn in rxn_type:
    string_g += f"{rxn}\t{rxn_type[rxn]}\n"
open("Reaction_dict.txt", "w").write(string_f)
open("Reaction_type.txt", "w").write(string_g)

rxn_dict             = {}
rxn_dict["reaction"] = reaction_dict
rxn_dict["type"]     = rxn_type
rxn_dict["data"]     = rs.getRxnDetails(mechanism, selected_reactions)

string_reaction = ""
for index in reaction_dict:
    string_reaction += f"{index}\t{reaction_dict[index]}\n"
open("selected_rxn.txt", "+w").write(string_reaction)

rxn_list         = []
activeParameters = []
for rxn in unsrt_data:
    rxn_list.append(rxn)
    activeParameters.extend(unsrt_data[rxn].activeParameters)
ap = len(activeParameters)

log.info(f"  Selected reactions : {len(rxn_list)}")
log.info(f"  Active parameters  : {ap}")

# ── Temperature grid for rate evaluation ──────────────────
global T, theta
T     = np.linspace(300, 2500, 100)
theta = np.array([T / T, np.log(T), -1.0 / T])

def getUnsrtLimit(Po, P_u, P_l):
    K_o = np.asarray([i.dot(Po) for i in theta.T]).flatten()
    K_u = np.asarray([i.dot(P_u) for i in theta.T]).flatten()
    K_l = np.asarray([i.dot(P_l) for i in theta.T]).flatten()
    return K_o, K_u, K_l

def getKappa(P):
    K = np.asarray([i.dot(P) for i in theta.T]).flatten()
    return np.exp(K)

# ============================================================
# SECTION 7 — DESIGN MATRIX GENERATION
# ============================================================
log.info(f"{'='*64}")
log.info("SECTION 7 — Generating design matrices")

# ── Parameter selection masks (one per Arrhenius param) ───
# Each mask is a flat array of length 3*N_rxns flagging which
# parameter in the full [ln(A), n, Ea/R] × N_rxns space to perturb.
select_param_a  = np.asarray([1, 0, 0] * len(rxn_list))
select_param_n  = np.asarray([0, 1, 0] * len(rxn_list))
select_param_ea = np.asarray([0, 0, 1] * len(rxn_list))
perturb_fact    = 0.1

# ── Helpers ────────────────────────────────────────────────
def _load_matrix_csv(filepath):
    """Load a matrix from a CSV file (one row per line, comma-separated)."""
    rows = []
    for line in open(filepath).readlines():
        rows.append([float(v) for v in line.strip("\n").strip(",").split(",")])
    return rows

def _save_dm_csv(mat, filepath):
    s = ""
    for row in mat:
        for element in row:
            s += f"{element},"
        s += "\n"
    open(filepath, 'w').write(s)

# ──── 7a. Nominal design matrix ────────────────────────────
log.info("  [DM 7a] Nominal ...")
if "DesignMatrix_x0_3P.csv" not in os.listdir():
    design_matrix_x0_3P = DM.DesignMatrix(
        unsrt_data, design_type, 1, ind=len(activeParameters)
    ).getNominal_samples()
    _save_dm_csv(design_matrix_x0_3P, "DesignMatrix_x0_3P.csv")
    log.info("         Generated and saved 'DesignMatrix_x0_3P.csv'.")
else:
    design_matrix_x0_3P = _load_matrix_csv("DesignMatrix_x0_3P.csv")
    log.info("         Loaded 'DesignMatrix_x0_3P.csv'.")

# ──── 7b. MULTIPLY design matrices (sign=+1) ───────────────
log.info("  [DM 7b] Multiply (sign=+1) ...")
for param_type, select_mask, dm_attr, sel_attr in [
    ("A",  select_param_a,  "design_matrix_A",  "selection_matrix_A"),
    ("n",  select_param_n,  "design_matrix_n",  "selection_matrix_n"),
    ("Ea", select_param_ea, "design_matrix_Ea", "selection_matrix_Ea"),
]:
    csv_dm  = f"DesignMatrix_{param_type}.csv"
    csv_sel = f"pSelectionMatrix_{param_type}.csv"
    if csv_dm not in os.listdir():
        sel_mat, dm = DM.DesignMatrix(
            unsrt_data, design_type, len(reaction_dict)
        ).getSA_3P_samples(select_mask, param_type=param_type,
                           perturb_fact=perturb_fact, sign=1)
        log.info(f"         Generated DesignMatrix_{param_type}.csv  "
                 f"shape={np.asarray(dm).shape}.")
    else:
        dm  = _load_matrix_csv(csv_dm)
        sel_mat = _load_matrix_csv(csv_sel)
        log.info(f"         Loaded    DesignMatrix_{param_type}.csv.")
    # bind to named variables used throughout the rest of the file
    if param_type == "A":
        design_matrix_A,  selection_matrix_A  = dm, sel_mat
    elif param_type == "n":
        design_matrix_n,  selection_matrix_n  = dm, sel_mat
    elif param_type == "Ea":
        design_matrix_Ea, selection_matrix_Ea = dm, sel_mat

# ──── 7c. DIVIDE design matrices (sign=-1, central only) ───
if sa_scheme == "central_difference":
    log.info("  [DM 7c] Divide (sign=-1) — required for central difference ...")
    for param_type, select_mask in [
        ("A",  select_param_a),
        ("n",  select_param_n),
        ("Ea", select_param_ea),
    ]:
        csv_dm  = f"DesignMatrix_{param_type}_neg.csv"
        csv_sel = f"pSelectionMatrix_{param_type}_neg.csv"
        if csv_dm not in os.listdir():
            sel_mat_neg, dm_neg = DM.DesignMatrix(
                unsrt_data, design_type, len(reaction_dict)
            ).getSA_3P_samples(select_mask, param_type=param_type,
                               perturb_fact=perturb_fact, sign=-1)
            log.info(f"         Generated DesignMatrix_{param_type}_neg.csv  "
                     f"shape={np.asarray(dm_neg).shape}.")
        else:
            dm_neg  = _load_matrix_csv(csv_dm)
            sel_mat_neg = _load_matrix_csv(csv_sel)
            log.info(f"         Loaded    DesignMatrix_{param_type}_neg.csv.")
        if param_type == "A":
            design_matrix_A_neg,  selection_matrix_A_neg  = dm_neg, sel_mat_neg
        elif param_type == "n":
            design_matrix_n_neg,  selection_matrix_n_neg  = dm_neg, sel_mat_neg
        elif param_type == "Ea":
            design_matrix_Ea_neg, selection_matrix_Ea_neg = dm_neg, sel_mat_neg

# ── Convert to numpy arrays ────────────────────────────────
design_matrix_A   = np.asarray(design_matrix_A,   dtype=float)
design_matrix_n   = np.asarray(design_matrix_n,   dtype=float)
design_matrix_Ea  = np.asarray(design_matrix_Ea,  dtype=float)
if sa_scheme == "central_difference":
    design_matrix_A_neg  = np.asarray(design_matrix_A_neg,  dtype=float)
    design_matrix_n_neg  = np.asarray(design_matrix_n_neg,  dtype=float)
    design_matrix_Ea_neg = np.asarray(design_matrix_Ea_neg, dtype=float)

# ============================================================
# SECTION 8 — delta_dict COMPUTATION
# ============================================================
log.info(f"{'='*64}")
log.info("SECTION 8 — Computing delta_dict (parameter perturbation vectors)")
log.info(
    "  Each reaction gets multiply and (optionally) divide perturbed "
    "parameter vectors derived from its reduced Cholesky L_r and the "
    "diagonal ζ-value extracted from the SA design matrix."
)

delta_dict = {}

for i, rxn in enumerate(rxn_list):
    # ── Nominal parameter vector: P_o = [ln(A₀), n₀, Ea₀/R] ─────────────
    P_o = np.asarray(unsrt_data[rxn].nominal, dtype=float).flatten()   # (3,)
    L   = unsrt_data[rxn].cov

    # ── Reduced Cholesky matrices (1×1 each) ─────────────────────────────
    # For a single selected parameter (m=1), get_reduced_cholesky returns
    #   Σ_r = L_r @ L_r.T   where L_r is (1×1) = sqrt(Σ[param, param]).
    # This projects a scalar reduced-space sample ζ_r to the perturbation:
    #   δ_param = L_r · ζ_r   (scalar)
    _, L_r_A  = unsrt_data[rxn].get_reduced_cholesky((0,))   # A-factor
    _, L_r_n  = unsrt_data[rxn].get_reduced_cholesky((1,))   # n
    _, L_r_Ea = unsrt_data[rxn].get_reduced_cholesky((2,))   # Ea/R

    # ──────────────────────────────────────────────────────────────────────
    # MULTIPLY entries  (η⁺, sign=+1 perturbation)
    # ──────────────────────────────────────────────────────────────────────
    # zr_X  : scalar ζ extracted from the diagonal of the multiply DM.
    #          design_matrix_X is (N_rxns × N_rxns) diagonal; row i perturbs
    #          only reaction i, so [i,i] is the Class-A ζ for that reaction.
    zr_A  = np.array([[design_matrix_A [i, i]]])   # (1,1)
    zr_n  = np.array([[design_matrix_n [i, i]]])   # (1,1)
    zr_Ea = np.array([[design_matrix_Ea[i, i]]])   # (1,1)

    # delta_multiply = L_r · zr  →  scalar perturbation in parameter space
    delta_A_mul  = float((L_r_A  @ zr_A ).item())   # Δln(A)
    delta_n_mul  = float((L_r_n  @ zr_n ).item())   # Δn
    delta_Ea_mul = float((L_r_Ea @ zr_Ea).item())   # Δ(Ea/R)

    # P_multiply = P_o + perturbation vector (only selected index is shifted)
    P_mul_A  = P_o + np.array([delta_A_mul,  0.0,         0.0        ])
    P_mul_n  = P_o + np.array([0.0,          delta_n_mul, 0.0        ])
    P_mul_Ea = P_o + np.array([0.0,          0.0,         delta_Ea_mul])

    entry = {
        # ── multiply ζ scalars ─────────────────────────────────────────────
        "zr_A"    : float(zr_A.item()),
        "zr_n"    : float(zr_n.item()),
        "zr_Ea"   : float(zr_Ea.item()),
        # ── reduced Cholesky (1×1) ─────────────────────────────────────────
        "L"       : L,
        "L_r_A"   : L_r_A,
        "L_r_n"   : L_r_n,
        "L_r_Ea"  : L_r_Ea,
        # ── multiply perturbation scalars ──────────────────────────────────
        "delta_A"    : delta_A_mul,
        "delta_n"    : delta_n_mul,
        "delta_Ea"   : delta_Ea_mul,
        # ── nominal and multiply perturbed parameter vectors ───────────────
        "P_o"     : P_o,
        "P_A"     : P_mul_A,
        "P_n"     : P_mul_n,
        "P_Ea"    : P_mul_Ea,
    }

    # ──────────────────────────────────────────────────────────────────────
    # DIVIDE entries  (η⁻, sign=-1 perturbation) — central difference only
    # ──────────────────────────────────────────────────────────────────────
    if sa_scheme == "central_difference":
        # zr_neg : scalar ζ extracted from the diagonal of the DIVIDE DM.
        #          By construction (sign=-1 in getSA_3P_samples), the divide
        #          DM is the negated multiply DM, so zr_neg = -zr.
        #          We read it explicitly from design_matrix_X_neg so the
        #          values are fully traceable to the actual perturbed YAML
        #          mechanisms that will be simulated.
        zr_A_neg  = np.array([[design_matrix_A_neg [i, i]]])   # (1,1)
        zr_n_neg  = np.array([[design_matrix_n_neg [i, i]]])   # (1,1)
        zr_Ea_neg = np.array([[design_matrix_Ea_neg[i, i]]])   # (1,1)

        # delta_divide = L_r · zr_neg
        # Since zr_neg = -zr_mul, delta_divide = -delta_multiply, which
        # means the rate coefficient is divided by the same factor it was
        # multiplied by — exactly symmetric in log-parameter space.
        delta_A_div  = float((L_r_A  @ zr_A_neg ).item())   # = -delta_A_mul
        delta_n_div  = float((L_r_n  @ zr_n_neg ).item())   # = -delta_n_mul
        delta_Ea_div = float((L_r_Ea @ zr_Ea_neg).item())   # = -delta_Ea_mul

        # P_divide = P_o + [delta_div, 0, 0]  (≡  P_o - [delta_mul, 0, 0])
        P_div_A  = P_o + np.array([delta_A_div,  0.0,         0.0        ])
        P_div_n  = P_o + np.array([0.0,          delta_n_div, 0.0        ])
        P_div_Ea = P_o + np.array([0.0,          0.0,         delta_Ea_div])

        entry.update({
            # ── divide ζ scalars ───────────────────────────────────────────
            "zr_A_neg"     : float(zr_A_neg.item()),
            "zr_n_neg"     : float(zr_n_neg.item()),
            "zr_Ea_neg"    : float(zr_Ea_neg.item()),
            # ── divide perturbation scalars ────────────────────────────────
            "delta_A_neg"  : delta_A_div,
            "delta_n_neg"  : delta_n_div,
            "delta_Ea_neg" : delta_Ea_div,
            # ── divide perturbed parameter vectors ─────────────────────────
            "P_div_A"      : P_div_A,
            "P_div_n"      : P_div_n,
            "P_div_Ea"     : P_div_Ea,
        })

    delta_dict[rxn] = entry

log.info(f"  delta_dict built for {len(delta_dict)} reactions.")
_r = rxn_list[1]
log.info(f"  Sample reaction  : {_r}")
log.info(f"    P_o            : {delta_dict[_r]['P_o']}")
log.info(f"    L_r_A  (1×1)   : {delta_dict[_r]['L_r_A']}  "
         f"zr_A={delta_dict[_r]['zr_A']:+.6f}  "
         f"→ delta_A  (mul)={delta_dict[_r]['delta_A']:+.6f}")
log.info(f"    L_r_n  (1×1)   : {delta_dict[_r]['L_r_n']}  "
         f"zr_n={delta_dict[_r]['zr_n']:+.6f}  "
         f"→ delta_n  (mul)={delta_dict[_r]['delta_n']:+.6f}")
log.info(f"    L_r_Ea (1×1)   : {delta_dict[_r]['L_r_Ea']}  "
         f"zr_Ea={delta_dict[_r]['zr_Ea']:+.6f}  "
         f"→ delta_Ea (mul)={delta_dict[_r]['delta_Ea']:+.6f}")
if sa_scheme == "central_difference":
    log.info(f"    zr_A_neg       : {delta_dict[_r]['zr_A_neg']:+.6f}  "
             f"→ delta_A  (div)={delta_dict[_r]['delta_A_neg']:+.6f}")
    log.info(f"    zr_n_neg       : {delta_dict[_r]['zr_n_neg']:+.6f}  "
             f"→ delta_n  (div)={delta_dict[_r]['delta_n_neg']:+.6f}")
    log.info(f"    zr_Ea_neg      : {delta_dict[_r]['zr_Ea_neg']:+.6f}  "
             f"→ delta_Ea (div)={delta_dict[_r]['delta_Ea_neg']:+.6f}")
    # Symmetry check: divide deltas should be exact negatives of multiply
    sym_ok = all(
        abs(delta_dict[_r][f"delta_{p}_neg"] + delta_dict[_r][f"delta_{p}"]) < 1e-12
        for p in ("A", "n", "Ea")
    )
    log.info(f"    Symmetry check (div = -mul): {'PASS' if sym_ok else 'FAIL'}")

# ============================================================
# SECTION 9 — PLOT DESIGN MATRIX SAMPLES
# ============================================================
log.info(f"{'='*64}")
log.info("SECTION 9 — Plotting design matrix samples")
VA.DesignMatrixPlotter(unsrt_data).plot_dm_samples()
log.info("  Done.")

# ============================================================
# SECTION 10 — PERTURBED YAML FILE GENERATION
# ============================================================
log.info(f"{'='*64}")
log.info("SECTION 10 — Generating perturbed mechanism YAML files")

yaml_loc_nominal = [mech_file_location]
SSM = simulator.SM(target_list, optInputs, unsrt_data, design_matrix_A)


def _generate_perturbed_yamls(SSM, design_matrix, selection_matrix, subdir, label):
    """
    Generate perturbed mechanism YAML files in chunks.

    Parameters
    ----------
    SSM            : simulation manager instance
    design_matrix  : 2-D array (N_rxns × params) — p_design_matrix
    selection_matrix : 2-D array — p_selection_matrix
    subdir         : subdirectory name under Perturbed_Mech_SA_3P_BruteForce/
    label          : short label for log messages (e.g. 'A', 'n_neg')

    Returns
    -------
    yaml_loc : list of full YAML file paths
    """
    chunk_size   = 500
    chunks_dm    = [design_matrix[i:i+chunk_size]   for i in range(0, len(design_matrix),   chunk_size)]
    chunks_sel   = [selection_matrix[i:i+chunk_size] for i in range(0, len(selection_matrix), chunk_size)]
    yaml_loc     = []
    total_count  = 0
    base_dir     = os.getcwd() + f"/Perturbed_Mech_SA_3P_BruteForce/{subdir}"
    for chunk_dm, chunk_sel in zip(chunks_dm, chunks_sel):
        yaml_list     = SSM.getYAML_List(chunk_dm, chunk_sel)
        location_mech = [base_dir] * len(yaml_list)
        index_list    = [str(total_count + j) for j in range(len(yaml_list))]
        for j in range(len(yaml_list)):
            yaml_loc.append(f"{base_dir}/mechanism_{total_count + j}.yaml")
        total_count += len(yaml_list)
        SSM.getPerturbedMechLocation(yaml_list, location_mech, index_list)
        log.info(f"    [{label}]  {total_count} YAML files generated so far ...")
    log.info(f"    [{label}]  Total: {total_count} YAML files written to '{subdir}/'.")
    return yaml_loc


def _reconstruct_yaml_locs(design_matrix, subdir):
    """Reconstruct YAML path list from existing directory (no regeneration)."""
    base_dir = os.getcwd() + f"/Perturbed_Mech_SA_3P_BruteForce/{subdir}"
    return [f"{base_dir}/mechanism_{i}.yaml" for i in range(len(design_matrix))]


# ── Create/check top-level YAML directory ─────────────────
perturb_base = "Perturbed_Mech_SA_3P_BruteForce"
if perturb_base not in os.listdir():
    os.mkdir(perturb_base)
    for sd in ("A_factor", "n", "Ea"):
        os.mkdir(f"{perturb_base}/{sd}")
    if sa_scheme == "central_difference":
        for sd in ("A_factor_neg", "n_neg", "Ea_neg"):
            os.mkdir(f"{perturb_base}/{sd}")
    log.info(f"  Created '{perturb_base}/' directory tree.")
    need_gen_mul = True
    need_gen_div = (sa_scheme == "central_difference")
else:
    log.info(f"  '{perturb_base}/' already exists.")
    need_gen_mul = False
    if sa_scheme == "central_difference":
        # Create divide subdirs if they are missing (e.g. forward run existed before)
        need_gen_div = False
        for sd in ("A_factor_neg", "n_neg", "Ea_neg"):
            full_sd = f"{perturb_base}/{sd}"
            if not os.path.isdir(full_sd):
                os.makedirs(full_sd)
                log.info(f"    Created missing subdir: {full_sd}")
                need_gen_div = True
    else:
        need_gen_div = False

# ── MULTIPLY YAMLs ─────────────────────────────────────────
log.info("  [YAML] Multiply perturbed mechanisms ...")
if need_gen_mul:
    yaml_loc_A  = _generate_perturbed_yamls(SSM, design_matrix_A,  selection_matrix_A,  "A_factor", "A")
    yaml_loc_n  = _generate_perturbed_yamls(SSM, design_matrix_n,  selection_matrix_n,  "n",        "n")
    yaml_loc_Ea = _generate_perturbed_yamls(SSM, design_matrix_Ea, selection_matrix_Ea, "Ea",       "Ea")
else:
    yaml_loc_A  = _reconstruct_yaml_locs(design_matrix_A,  "A_factor")
    yaml_loc_n  = _reconstruct_yaml_locs(design_matrix_n,  "n")
    yaml_loc_Ea = _reconstruct_yaml_locs(design_matrix_Ea, "Ea")
    log.info("    Loaded multiply YAML paths from existing directory.")

# ── DIVIDE YAMLs (central difference only) ─────────────────
if sa_scheme == "central_difference":
    log.info("  [YAML] Divide perturbed mechanisms (central difference) ...")
    if need_gen_div:
        yaml_loc_A_neg  = _generate_perturbed_yamls(
            SSM, design_matrix_A_neg,  selection_matrix_A_neg,  "A_factor_neg", "A_neg")
        yaml_loc_n_neg  = _generate_perturbed_yamls(
            SSM, design_matrix_n_neg,  selection_matrix_n_neg,  "n_neg",        "n_neg")
        yaml_loc_Ea_neg = _generate_perturbed_yamls(
            SSM, design_matrix_Ea_neg, selection_matrix_Ea_neg, "Ea_neg",       "Ea_neg")
    else:
        yaml_loc_A_neg  = _reconstruct_yaml_locs(design_matrix_A_neg,  "A_factor_neg")
        yaml_loc_n_neg  = _reconstruct_yaml_locs(design_matrix_n_neg,  "n_neg")
        yaml_loc_Ea_neg = _reconstruct_yaml_locs(design_matrix_Ea_neg, "Ea_neg")
        log.info("    Loaded divide YAML paths from existing directory.")

# ── Per-case YAML location dicts ──────────────────────────
yaml_loc_nominal_case = {case: yaml_loc_nominal for case in case_dir}
yaml_loc_A_case       = {case: yaml_loc_A       for case in case_dir}
yaml_loc_n_case       = {case: yaml_loc_n       for case in case_dir}
yaml_loc_Ea_case      = {case: yaml_loc_Ea      for case in case_dir}
if sa_scheme == "central_difference":
    yaml_loc_A_neg_case  = {case: yaml_loc_A_neg  for case in case_dir}
    yaml_loc_n_neg_case  = {case: yaml_loc_n_neg  for case in case_dir}
    yaml_loc_Ea_neg_case = {case: yaml_loc_Ea_neg for case in case_dir}

# ============================================================
# SECTION 11 — SIMULATION FIELD CREATION  (SA_3P/ tree)
# ============================================================
log.info(f"{'='*64}")
log.info("SECTION 11 — Creating simulation field under SA_3P/")

if "SA_3P" not in os.listdir():
    os.mkdir("SA_3P")
    os.chdir("SA_3P")
    # ── multiply simulation dirs ───────────────────────────
    for d in ("multiply_A", "multiply_n", "multiply_Ea",
              "multiply",   "divide",     "nominal"):
        os.mkdir(d)
    # ── data tree ─────────────────────────────────────────
    os.makedirs("Data/Simulations/Multiply_A")
    os.makedirs("Data/Simulations/Multiply_n")
    os.makedirs("Data/Simulations/Multiply_Ea")
    os.makedirs("Data/Simulations/Multiply")
    os.makedirs("Data/Simulations/Divide")
    os.makedirs("Data/Simulations/Nominal")
    os.makedirs("Data/ResponseSurface")
    # ── central-difference divide dirs ────────────────────
    if sa_scheme == "central_difference":
        for d in ("divide_A", "divide_n", "divide_Ea"):
            os.mkdir(d)
        for d in ("Data/Simulations/Divide_A",
                  "Data/Simulations/Divide_n",
                  "Data/Simulations/Divide_Ea"):
            os.makedirs(d)
    log.info("  SA_3P/ directory tree created.")
    os.chdir("multiply_A")
    SADir = os.getcwd()
else:
    log.info("  SA_3P/ already exists.")
    os.chdir("SA_3P")
    # Ensure central-difference dirs exist if scheme was changed
    if sa_scheme == "central_difference":
        for d in ("divide_A", "divide_n", "divide_Ea"):
            if not os.path.isdir(d):
                os.mkdir(d)
                log.info(f"    Created missing simulation dir: {d}")
        for d in ("Data/Simulations/Divide_A",
                  "Data/Simulations/Divide_n",
                  "Data/Simulations/Divide_Ea"):
            if not os.path.isdir(d):
                os.makedirs(d)
                log.info(f"    Created missing data dir: {d}")
    os.chdir("multiply_A")
    SADir = os.getcwd()

# ============================================================
# SECTION 12 — RUNNING SIMULATIONS
# ============================================================
log.info(f"{'='*64}")
log.info("SECTION 12 — Running simulations")


def _run_or_load_sim(SM_obj, yaml_case_dict, design_mat, sim_dir, label):
    """
    Run simulations or load previously saved locations.

    Returns
    -------
    locations : list of execution location paths
    """
    if not os.path.isfile("progress"):
        log.info(f"  [SIM] Running {label} simulations ...")
        locs = SM_obj.make_dir_in_parallel(yaml_case_dict)
        log.info(f"  [SIM] {label} simulations complete.")
    else:
        log.info(f"  [SIM] {label}: progress file detected — loading saved locations.")
        with open(os.getcwd() + "/locations") as inf:
            locs = list(inf)
    return locs


# ──── 12a. Multiply-A ──────────────────────────────────────
log.info(f"  Entering multiply_A/  ({os.getcwd()})")
FlameMaster_Execution_location_A = _run_or_load_sim(
    simulator.SM(target_list, optInputs, rxn_dict, design_matrix_A),
    yaml_loc_A_case, design_matrix_A, SADir, "MULTIPLY-A"
)

# ──── 12b. Multiply-n ──────────────────────────────────────
os.chdir("../multiply_n");  SADir = os.getcwd()
log.info(f"  Entering multiply_n/  ({os.getcwd()})")
FlameMaster_Execution_location_n = _run_or_load_sim(
    simulator.SM(target_list, optInputs, rxn_dict, design_matrix_n),
    yaml_loc_n_case, design_matrix_n, SADir, "MULTIPLY-n"
)

# ──── 12c. Multiply-Ea ─────────────────────────────────────
os.chdir("../multiply_Ea"); SADir = os.getcwd()
log.info(f"  Entering multiply_Ea/  ({os.getcwd()})")
FlameMaster_Execution_location_Ea = _run_or_load_sim(
    simulator.SM(target_list, optInputs, rxn_dict, design_matrix_Ea),
    yaml_loc_Ea_case, design_matrix_Ea, SADir, "MULTIPLY-Ea"
)

# ──── 12d. Divide simulations (central difference only) ────
if sa_scheme == "central_difference":
    os.chdir("../divide_A");  SADir = os.getcwd()
    log.info(f"  Entering divide_A/  ({os.getcwd()})")
    FlameMaster_Execution_location_A_neg = _run_or_load_sim(
        simulator.SM(target_list, optInputs, rxn_dict, design_matrix_A_neg),
        yaml_loc_A_neg_case, design_matrix_A_neg, SADir, "DIVIDE-A"
    )

    os.chdir("../divide_n");  SADir = os.getcwd()
    log.info(f"  Entering divide_n/  ({os.getcwd()})")
    FlameMaster_Execution_location_n_neg = _run_or_load_sim(
        simulator.SM(target_list, optInputs, rxn_dict, design_matrix_n_neg),
        yaml_loc_n_neg_case, design_matrix_n_neg, SADir, "DIVIDE-n"
    )

    os.chdir("../divide_Ea"); SADir = os.getcwd()
    log.info(f"  Entering divide_Ea/  ({os.getcwd()})")
    FlameMaster_Execution_location_Ea_neg = _run_or_load_sim(
        simulator.SM(target_list, optInputs, rxn_dict, design_matrix_Ea_neg),
        yaml_loc_Ea_neg_case, design_matrix_Ea_neg, SADir, "DIVIDE-Ea"
    )

# ──── 12e. Nominal ─────────────────────────────────────────
os.chdir("../nominal");  SADir = os.getcwd()
log.info(f"  Entering nominal/  ({os.getcwd()})")
FlameMaster_Execution_location_x0 = _run_or_load_sim(
    simulator.SM(target_list, optInputs, rxn_dict, design_matrix_x0_3P),
    yaml_loc_nominal_case, design_matrix_x0_3P, SADir, "NOMINAL"
)

os.chdir("..")
SAdir = os.getcwd()   # ← points to SA_3P/

# ============================================================
# SECTION 13 — COLLECTING SIMULATION DATA
# ============================================================
log.info(f"{'='*64}")
log.info("SECTION 13 — Collecting simulation ETA values")


def _collect_case_data(optInputs,case_dir, SAdir, data_subdir, sim_subdir,
                       exec_locs, target_list, fuel, label):
    """
    Collect ETA simulation results for one perturbation type.

    Looks for cached .lst files in Data/Simulations/<data_subdir>/;
    if missing, runs data_management.generate_SA_target_value_tables
    from <sim_subdir>/case-<n>/ and caches the result.

    Returns
    -------
    dict  {str(case): {"ETA": [...], "index": [...]}}
    """
    result  = {}
    data_dir = f"{SAdir}/Data/Simulations/{data_subdir}"
    for case in case_dir:
        lst_file = f"{data_dir}/sim_data_case-{case}.lst"
        if os.path.isfile(lst_file):
            lines = open(lst_file).readlines()
            try:
                ETA    = [float(l.split("\t")[1]) for l in lines]
                folder = [float(l.split("\t")[0]) for l in lines]
            except (ValueError, IndexError) as exc:
                log.warning(f"  [{label}] Case {case}: parse error ({exc}) — ETA set to [].")
                ETA, folder = [], []
            result[str(case)] = {"ETA": ETA, "index": folder}
            log.info(f"  [{label}] Case {case}: loaded {len(ETA)} values from cache.")
        else:
            log.info(f"  [{label}] Case {case}: Collecting the data")
            os.chdir(f"{SAdir}/{sim_subdir}/case-{case}")
            data_sheet, failed_sim, index, ETA, eta = \
                data_management.generate_SA_target_value_tables(
                    exec_locs, target_list, case, fuel, input_ = optInputs
                )
            result[str(case)] = {"ETA": ETA, "index": index}
            open(f"{data_dir}/sim_data_case-{case}.lst",        'w').write(data_sheet)
            open(f"{data_dir}/failed_sim_data_case-{case}.lst", 'w').write(failed_sim)
            os.chdir(SAdir)
            log.info(f"  [{label}] Case {case}: computed {len(ETA)} values.")
    return result


log.info("  [DATA] Nominal ...")
temp_sim_opt_x0 = _collect_case_data(
    optInputs, case_dir, SAdir, "Nominal", "nominal",
    FlameMaster_Execution_location_x0, target_list, fuel, "Nominal"
)

log.info("  [DATA] Multiply-A ...")
temp_sim_opt_A = _collect_case_data(
    optInputs, case_dir, SAdir, "Multiply_A", "multiply_A",
    FlameMaster_Execution_location_A, target_list, fuel, "Multiply_A"
)

log.info("  [DATA] Multiply-n ...")
temp_sim_opt_n = _collect_case_data(
    optInputs, case_dir, SAdir, "Multiply_n", "multiply_n",
    FlameMaster_Execution_location_n, target_list, fuel, "Multiply_n"
)

log.info("  [DATA] Multiply-Ea ...")
temp_sim_opt_Ea = _collect_case_data(
    optInputs, case_dir, SAdir, "Multiply_Ea", "multiply_Ea",
    FlameMaster_Execution_location_Ea, target_list, fuel, "Multiply_Ea"
)

if sa_scheme == "central_difference":
    log.info("  [DATA] Divide-A ...")
    temp_sim_opt_A_neg = _collect_case_data(
        optInputs, case_dir, SAdir, "Divide_A", "divide_A",
        FlameMaster_Execution_location_A_neg, target_list, fuel, "Divide_A"
    )
    log.info("  [DATA] Divide-n ...")
    temp_sim_opt_n_neg = _collect_case_data(
        optInputs, case_dir, SAdir, "Divide_n", "divide_n",
        FlameMaster_Execution_location_n_neg, target_list, fuel, "Divide_n"
    )
    log.info("  [DATA] Divide-Ea ...")
    temp_sim_opt_Ea_neg = _collect_case_data(
        optInputs, case_dir, SAdir, "Divide_Ea", "divide_Ea",
        FlameMaster_Execution_location_Ea_neg, target_list, fuel, "Divide_Ea"
    )

log.info("  All simulation data collected.")

# ============================================================
# SECTION 14 — SA COEFFICIENT COMPUTATION
# ============================================================
log.info(f"{'='*64}")
log.info(f"SECTION 14 — Computing SA coefficients  [{sa_scheme.upper()}]")

os.makedirs("../Plots_SA",           exist_ok=True)
os.makedirs("Data/SensitivityCoeffs", exist_ok=True)

scheme_tag = "FD" if sa_scheme == "forward_difference" else "CD"
selected_BRUTE_FORCE_PARAMETERS = {}

for case_index, case in enumerate(temp_sim_opt_A):
    log.info(f"  [SA] Case {case_index}  (key='{case}') ...")

    # ── Extract ETA arrays for this case ─────────────────────────────────
    multiply_A  = np.asarray(temp_sim_opt_A [str(case)]["ETA"], dtype=float)
    multiply_n  = np.asarray(temp_sim_opt_n [str(case)]["ETA"], dtype=float)
    multiply_Ea = np.asarray(temp_sim_opt_Ea[str(case)]["ETA"], dtype=float)
    nominal     = float(np.asarray(temp_sim_opt_x0[str(case)]["ETA"], dtype=float).flatten()[0])
    log.info(f"    η₀ (nominal IDT) = {nominal:.6e}")

    if sa_scheme == "central_difference":
        divide_A  = np.asarray(temp_sim_opt_A_neg [str(case)]["ETA"], dtype=float)
        divide_n  = np.asarray(temp_sim_opt_n_neg [str(case)]["ETA"], dtype=float)
        divide_Ea = np.asarray(temp_sim_opt_Ea_neg[str(case)]["ETA"], dtype=float)

    T_ = float(target_list[case_index].temperature)

    SA_coeff_A               = []
    SA_coeff_n               = []
    SA_coeff_Ea              = []
    SA_coeff_without_k_perturbed = []   # raw forward (η⁺−η₀)/η₀ for diagnostics

    for rxn_index, rxn in enumerate(unsrt_data):

        # ── Normalization factors ─────────────────────────────────────────
        # These convert the finite-difference ratio Δη/η₀ into a proper
        # sensitivity coefficient dln(η)/dζ_x in reduced-parameter space.
        #
        # A  : f_norm = ln(A₀) / δ_A
        #        because  dln(A)/dζ_A = δ_A, so  dln(A)/1 = ln(A₀) is the
        #        reference scale → multiply by (ln(A₀) / δ_A)
        # n  : f_norm = 1 / δ_n
        #        n is additive (not log-scaled), so the scale is just 1/δ_n
        # Ea : f_norm = (Ea₀/R) / δ_Ea
        #        same as A: Ea/R appears as a multiplicative factor in
        #        ln(k), so the reference scale is Ea₀/R
        delta_a      = float(delta_dict[rxn]["delta_A"])
        A_o          = float(delta_dict[rxn]["P_o"][0])   # = ln(A₀)
        normalized_A = A_o / delta_a

        delta_n_val  = float(delta_dict[rxn]["delta_n"])
        normalized_n = 1.0 / delta_n_val              # n is not log-scaled

        delta_ea     = float(delta_dict[rxn]["delta_Ea"])
        Ea_o         = float(delta_dict[rxn]["P_o"][2])   # = Ea₀/R
        normalized_ea = Ea_o / delta_ea

        # ── Finite-difference ratios ──────────────────────────────────────
        if sa_scheme == "forward_difference":
            # ─────────────────────────────────────────────────────────────
            # FORWARD DIFFERENCE (1st-order accurate):
            #
            #   fact_X  =  (η⁺ − η₀) / η₀
            #
            # where η⁺ is the response with parameter X shifted by +δ_X,
            # and η₀ is the nominal (unperturbed) response.
            # ─────────────────────────────────────────────────────────────
            fact_A  = float((multiply_A [rxn_index] - nominal) / nominal)
            fact_n  = float((multiply_n [rxn_index] - nominal) / nominal)
            fact_ea = float((multiply_Ea[rxn_index] - nominal) / nominal)

        elif sa_scheme == "central_difference":
            # ─────────────────────────────────────────────────────────────
            # CENTRAL DIFFERENCE (2nd-order accurate):
            #
            #   fact_X  =  (η⁺ − η⁻) / (2 · η₀)
            #
            # where η⁺ is the response with parameter X shifted by +δ_X,
            # and η⁻ is the response with parameter X shifted by −δ_X
            # (i.e. the divide case, same magnitude but opposite sign).
            # The factor of 2 in the denominator accounts for the full
            # 2·δ_X step width used in the central scheme.
            # ─────────────────────────────────────────────────────────────
            fact_A  = float((multiply_A [rxn_index] - divide_A [rxn_index]) / (2.0 * nominal))
            fact_n  = float((multiply_n [rxn_index] - divide_n [rxn_index]) / (2.0 * nominal))
            fact_ea = float((multiply_Ea[rxn_index] - divide_Ea[rxn_index]) / (2.0 * nominal))

        # ── SA coefficients:  S_X = f_norm_X · fact_X ────────────────────
        SA_coeff_A .append(normalized_A  * fact_A)
        SA_coeff_n .append(normalized_n  * fact_n)
        SA_coeff_Ea.append(normalized_ea * fact_ea)

        # Diagnostic: raw forward (η⁺−η₀)/η₀ regardless of scheme
        SA_coeff_without_k_perturbed.append(
            float((multiply_A[rxn_index] - nominal) / nominal)
        )

    log.info(f"    SA computed for {len(SA_coeff_A)} reactions.")

    # ── Assemble rxn_Sa dicts ─────────────────────────────────────────────
    rxn_Sa   = {}
    rxn_Sa_1 = {}
    for count_, rxn in enumerate(rxn_list):
        rxn_Sa  [rxn] = [SA_coeff_A[count_], SA_coeff_n[count_], SA_coeff_Ea[count_]]
        rxn_Sa_1[rxn] = [SA_coeff_without_k_perturbed[count_]]

    # ── Sort descending by |S_A| ──────────────────────────────────────────
    SA_dict   = dict(sorted(rxn_Sa.items(),   key=lambda item: abs(item[1][0]), reverse=True))
    SA_dict_1 = dict(sorted(rxn_Sa_1.items(), key=lambda item: abs(item[1][0]), reverse=True))

    selected_BRUTE_FORCE_PARAMETERS[str(case_index)] = rxn_Sa

    sort_rlist  = list(SA_dict.keys())
    sort_alist  = [SA_dict[r][0]   for r in sort_rlist]
    sort_nlist  = [SA_dict[r][1]   for r in sort_rlist]
    sort_ealist = [SA_dict[r][2]   for r in sort_rlist]
    sort_alist_1 = [SA_dict_1[r][0] for r in sort_rlist]
    ticks        = list(range(len(sort_alist)))

    log.info(f"    Top-5 sensitive reactions (|S_A|):")
    for rank, rxn_name in enumerate(sort_rlist[:5]):
        log.info(f"      {rank+1}. {rxn_name:40s}  S_A={sort_alist[rank]:+.5f}")

    # ── Plots ─────────────────────────────────────────────────────────────
    log.info(f"    Generating SA plots ...")

    fig = plt.figure()
    y_pos = range(len(sort_alist))
    plt.barh(y_pos, sorted(sort_alist, key=abs), alpha=0.51)
    plt.yticks(y_pos, sort_rlist)
    plt.xlabel(r'$S_i = \partial\ln(\eta)\,/\,\partial\zeta_\alpha$'
               f'  [{scheme_tag}]')
    plt.savefig(f'../Plots_SA/sensitivity_A_{case_index}_{scheme_tag}.png',
                bbox_inches="tight")
    plt.close()

    fake_data = pd.DataFrame({
        "index": sort_rlist,
        0: sort_alist,
        1: sort_nlist,
        2: np.asarray(sort_ealist) * 10,
    })
    fake_data.set_index("index", drop=False)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True,
                                         figsize=(8, 8), frameon=False)
    fake_data[0].plot.barh(ax=ax1)
    fake_data[1].plot.barh(ax=ax2)
    fake_data[2].plot.barh(ax=ax3)
    ax1.set_yticks(ticks, sort_rlist)
    ax1.set_xlabel(r'$\partial\ln(\eta)\,/\,\partial\zeta_\alpha$')
    ax2.set_xlabel(r'$\partial\ln(\eta)\,/\,\partial\zeta_n$')
    ax3.set_xlabel(r'$\partial\ln(\eta)\,/\,\partial\zeta_\epsilon\ (\times10^{-1})$')
    fig.savefig(f'../Plots_SA/sensitivity_{case_index}_{scheme_tag}.png',
                bbox_inches="tight")
    plt.close()

    # ── Text output files ─────────────────────────────────────────────────
    def _write_sens(rlist, coeff_list):
        s = f"Sensitivity Analysis (Cantera, scheme={sa_scheme}), Tig, T={T_} K:\n"
        for ind, rxn in enumerate(rlist):
            s += (f"\t{coeff_list[ind]:.8f}"
                  f"\t{index_dict[rxn.split(':')[0]]}"
                  f"\t{rxn}\n")
        return s

    tag = f"T_{T_}_case_{case_index}_{scheme_tag}"
    open(f"Data/SensitivityCoeffs/FM_sensitivity_{tag}.txt",    'w').write(_write_sens(sort_rlist, sort_alist))
    open(f"Data/SensitivityCoeffs/FM_sensitivity_{tag}_1.txt",  'w').write(_write_sens(sort_rlist, sort_alist_1))
    open(f"Data/SensitivityCoeffs/FM_sensitivity_{tag}_n.txt",  'w').write(_write_sens(sort_rlist, sort_nlist))
    open(f"Data/SensitivityCoeffs/FM_sensitivity_{tag}_ea.txt", 'w').write(_write_sens(sort_rlist, sort_ealist))
    log.info(f"    Sensitivity files written (tag='{tag}').")

log.info("  All SA coefficients computed and saved.")

# ============================================================
# SECTION 15 — SAVE RESULTS & EXIT
# ============================================================
log.info(f"{'='*64}")
log.info("SECTION 15 — Saving global results")

os.chdir("..")   # back to run directory

if "sens_3p_parameters.pkl" not in os.listdir():
    with open('sens_3p_parameters.pkl', 'wb') as f_:
        pickle.dump(selected_BRUTE_FORCE_PARAMETERS, f_)
    log.info("  Saved 'sens_3p_parameters.pkl'.")
else:
    log.info("  'sens_3p_parameters.pkl' already exists — skipping overwrite.")

log.info(f"{'='*64}")
log.info(f"  3-PARAMETER SENSITIVITY ANALYSIS COMPLETE")
log.info(f"  Scheme : {sa_scheme.upper()}")
log.info(f"  Log    : {_log_filename}")
log.info(f"{'='*64}")

raise AssertionError(f"3-PARAM SA DONE  (scheme={sa_scheme})")
