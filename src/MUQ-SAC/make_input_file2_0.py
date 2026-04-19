"""
make_input_file.py  —  Simulation input-file generator for MUQ-SAC
====================================================================
Supported solvers: Cantera | CHEMKIN-Pro | FlameMaster
Supported experiment types: JSR, Flf, RCM, Tig, Fls, Flf_a, Flw

Entry point
-----------
create_input_file(case, opt_dict, target, mech_file=None)
    Returns (instring, s_convert, s_run, extract)

Bugs fixed vs original
----------------------
1. subprocess.call("./run_profile")        → subprocess.call(["./run_profile"])
2. raise Assertionerror(...)               → raise AssertionError(...)  (7 occurrences)
3. doNonReactive VolumeProfile signature   → keywords-dict form
4. FlameMaster Tig-BL positional {} mixed with keyword args → removed stray {}
5. `copy` module used but never imported   → added from copy import deepcopy
6. FlameMaster extract `outfile` undefined if no match → guarded with default
"""

import os
import subprocess
from copy import deepcopy

import yaml
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


# ════════════════════════════════════════════════════════════════════════════
#  Shared embedded Python code fragments
#  These strings are written verbatim into the generated simulation scripts.
#  They use {{ / }} to produce literal { / } in the output file.
# ════════════════════════════════════════════════════════════════════════════

_CANTERA_HELPERS = """
def find_nearest(array, value):
    array = np.asarray(np.abs(np.log(array)))
    value = np.abs(np.log(value))
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def find_nearest_value(xi, x, y):
    return y[min(range(len(x)), key=lambda i: abs(x[i] - xi))]

def fast_nearest_interp(xi, x, y):
    spacing = np.diff(x) / 2
    x = x + np.hstack([spacing, spacing[-1]])
    y = np.hstack([y, y[-1]])
    return y[np.searchsorted(x, xi)]
"""

# ignitionDelay that works with Cantera timeHistory (column "temperature")
_IGN_DELAY_CANTERA = """
def ignitionDelay(df, pList, species, cond="max",
                  specified_value="None;None", exp_conc="None"):
    if cond == "max":
        time  = df.index.to_numpy()
        conc  = df[species].to_numpy()
        slope = np.diff(df["temperature"].to_numpy()) / np.diff(time)
        peaks, _ = find_peaks(conc)
        if len(peaks) > 1:
            peak_ind = conc[peaks].argmax()
        elif len(peaks) == 1:
            peak_ind = 0
        else:
            raise AssertionError("No ignition detected!")
        tau = time[peaks[peak_ind]]
    elif cond in ("onset", "dt-max"):
        time = df.index.to_numpy()
        arr  = np.asarray(pList) if species == "p" else df[species].to_numpy()
        slope = np.diff(arr) / np.diff(time)
        if cond == "onset":
            tau = time[int(np.diff(slope).argmax())]
        else:
            tau = time[int(slope.argmax())]
    elif cond == "specific":
        if specified_value.split(";")[0] is None:
            raise AssertionError("Input required for specified_value")
        tgt  = float(specified_value.split(";")[0])
        unit = specified_value.split(";")[1]
        t_a  = df.index.to_numpy()
        c_a  = df[species].to_numpy()
        AVOG = 6.02214e23
        if unit == "molecule":
            tgt = tgt / AVOG * gas.atomic_weight(species)
            tau = t_a[find_nearest(c_a, tgt)[0]]
        elif unit == "molecule/cm3":
            tgt = (tgt / AVOG) * 1e6
            if exp_conc != "":
                tgt /= (exp_conc / AVOG) * 1e6
            f   = Akima1DInterpolator(t_a, c_a)
            tn  = np.arange(min(t_a), max(t_a), 1e-8)
            tau = fast_nearest_interp(tgt, f(tn), tn)
        elif unit == "mole/cm3":
            if exp_conc != "":
                tgt /= exp_conc
            f   = Akima1DInterpolator(t_a, c_a)
            tn  = np.arange(min(t_a), max(t_a), 1e-8)
            tau = fast_nearest_interp(tgt, f(tn), tn)
    return tau
"""

# ignitionDelay for CHEMKIN-Pro output (column "Temperature  (K)")
_IGN_DELAY_CHEMKIN = """
def ignitionDelay(df, pList, species, cond="max",
                  specified_value="None;None", exp_conc="None"):
    if cond == "max":
        time  = df.index.to_numpy()
        conc  = df[species].to_numpy()
        slope = np.diff(df["Temperature  (K)"].to_numpy()) / np.diff(time)
        peaks, _ = find_peaks(conc)
        if len(peaks) > 1:
            peak_ind = conc[peaks].argmax()
        elif len(peaks) == 1:
            peak_ind = 0
        else:
            raise AssertionError("No ignition detected!")
        tau = time[peaks[peak_ind]]
    elif cond in ("onset", "dt-max"):
        time  = df.index.to_numpy()
        arr   = np.asarray(pList) if species == "p" else df[species].to_numpy()
        slope = np.diff(arr) / np.diff(time)
        if cond == "onset":
            tau = time[int(np.diff(slope).argmax())]
        else:
            tau = time[int(slope.argmax())]
    elif cond == "specific":
        if specified_value.split(";")[0] is None:
            raise AssertionError("Input required for specified_value")
        tgt  = float(specified_value.split(";")[0])
        unit = specified_value.split(";")[1]
        t_a  = df.index.to_numpy()
        c_a  = df[species].to_numpy()
        AVOG = 6.02214e23
        if unit == "molecule":
            tgt = tgt / AVOG * gas.atomic_weight(species)
            tau = t_a[find_nearest(c_a, tgt)[0]]
        elif unit == "molecule/cm3":
            tgt = (tgt / AVOG) * 1e6
            if exp_conc != "":
                tgt /= (exp_conc / AVOG) * 1e6
            f   = Akima1DInterpolator(t_a, c_a)
            tn  = np.arange(min(t_a), max(t_a), 1e-8)
            tau = fast_nearest_interp(tgt, f(tn), tn)
        elif unit == "mole/cm3":
            if exp_conc != "":
                tgt /= exp_conc
            f   = Akima1DInterpolator(t_a, c_a)
            tn  = np.arange(min(t_a), max(t_a), 1e-8)
            tau = fast_nearest_interp(tgt, f(tn), tn)
    return tau
"""

# VolumeProfile class — keywords-dict interface
_VOLUME_PROFILE = """
class VolumeProfile(object):
    def __init__(self, keywords):
        self.time     = np.array(keywords["vproTime"])
        vol0          = keywords["vproVol"][0]
        self.volume   = np.array(keywords["vproVol"]) / vol0
        self.velocity = np.append(np.diff(self.volume) / np.diff(self.time), 0)
    def __call__(self, t):
        if t < self.time[-1]:
            prev = self.time[self.time <= t][-1]
            idx  = np.where(self.time == prev)[0][0]
            return self.velocity[idx]
        return 0
"""

# VolumeProfile class — positional (time, volume) interface for non-reactive
_VOLUME_PROFILE_POS = """
class VolumeProfile(object):
    def __init__(self, time, volume):
        self.time     = np.array(time)
        self.volume   = np.array(volume) / volume[0]
        self.velocity = np.append(np.diff(self.volume) / np.diff(self.time), 0)
    def __call__(self, t):
        if t < self.time[-1]:
            prev = self.time[self.time <= t][-1]
            idx  = np.where(self.time == prev)[0][0]
            return self.velocity[idx]
        return 0
"""

# BUG FIX: original called VolumeProfile(inp_time, inp_vol) but class takes keywords dict
_DO_NON_REACTIVE = """
def doNonReactive(gas, inp_time, inp_vol):
    gas.set_multiplier(0)
    r    = ct.IdealGasReactor(gas)
    netw = ct.ReactorNet([r])
    env  = ct.Reservoir(ct.Solution('air.yaml'))
    ct.Wall(r, env, A=1.0,
            velocity=VolumeProfile({{"vproTime": inp_time, "vproVol": inp_vol}}))
    cols = [r.component_name(i) for i in range(r.n_vars)]
    th   = pd.DataFrame(columns=cols)
    t = 0; pList = []; vList = []; counter = 1
    while t < 1.0:
        t = netw.step()
        if counter % 1 == 0:
            th.loc[t] = netw.get_state().astype("float64")
            pList.append(gas.P); vList.append(r.volume)
        counter += 1
    return th, pList, vList
"""

_CONVERT_CKCSV = """
def convert_and_transpose_ckcsv(input_filepath, output_filepath):
    import csv as _csv
    with open(input_filepath, "r") as f:
        rows = list(_csv.reader(f))
    data_rows = rows[1:-2]
    headers   = [f"{{r[0]}} {{r[1]}}" for r in data_rows]
    transposed = list(zip(*[r[2:] for r in data_rows]))
    with open(output_filepath, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(headers)
        w.writerows(transposed)
    print(f"Transposition complete → {{output_filepath}}")
"""

_DELETE_OUTPUT_FILES = """
def _delete_output_files(directory, keep):
    import glob
    for pat in ("*.out", "*.asc", "*.zip"):
        for fpath in glob.glob(os.path.join(directory, pat)):
            if os.path.basename(fpath).lower() != keep.lower():
                try:
                    os.remove(fpath)
                except Exception as e:
                    print(f"Error deleting {{fpath}}: {{e}}")
"""

# Ignition criteria — uses reactorTemperature Python variable (not a format placeholder)
_CRITERIA_BLOCK = """
global criteria
if reactorTemperature >= 1800:
    criteria = reactorTemperature + 7
elif reactorTemperature >= 1400:
    criteria = reactorTemperature + 50
else:
    criteria = reactorTemperature + 100
"""

# Flow-reactor / JSR extraction helpers
_FLW_EXTRACT_HELPERS = """
def interpolation2D(d, x):
    return d[0][1] + (x - d[0][0]) * (d[1][1] - d[0][1]) / (d[1][0] - d[0][0])

def getNearest(value, array):
    return int(np.abs(np.asarray(array) - value).argmin())

def populate(target, array, x):
    array = list(array)
    return np.asarray(array[:x+1] + [target] + array[x+1:])

def getTimeStamp(target, array):
    if target in array:
        array = list(array); x = array.index(target)
        return np.asarray(array), x, len(array[x+1:])
    idx = getNearest(target, array)
    x   = idx - 1 if array[idx] > target else idx
    y   = len(array[x:])
    return populate(target, array, x), x, y

def timeScaleShift(target, array, x, y):
    if target in array:
        array = list(array); mid = array.index(target)
        return np.asarray(array[mid-x:mid] + [array[mid]] + array[mid+1:mid+y+1])
    idx = getNearest(target, array)
    new = populate(target, array, idx-1 if array[idx] > target else idx)
    return timeScaleShift(target, new, x, y)

def rateDefination(g1, g2):
    rate = np.abs((g2[1] - g1[1]) / ((g1[0] - g2[0]) * 1000))
    return rate, abs(g2[0] - g1[0]) * 1000

def getDataPoints(arr_t, arr_x, X):
    arr_x = list(arr_x)
    if X in arr_x:
        i = arr_x.index(X); return arr_t[i], X
    idx = getNearest(X, arr_x)
    if idx >= len(arr_t): idx = len(arr_t) - 1
    a, b = (idx-1, idx) if arr_x[idx] > X else (idx, min(idx+1, len(arr_t)-1))
    t_p = interpolation2D([[arr_x[a], arr_t[a]], [arr_x[b], arr_t[b]]], X)
    return t_p, X

def getRate(xo, t_half, X, t, add):
    fact = 100 if "percentage" in add["unit"] else 1
    y1, y2, anc = float(add["range_"][0])/fact, float(add["range_"][1])/fact, float(add["anchor"])/fact
    ni_t, lx, ly = getTimeStamp(t_half, t)
    ni_x = timeScaleShift(float(xo)*anc, X, lx, ly)
    rate, dt = rateDefination(getDataPoints(t, ni_x, float(xo)*y1),
                              getDataPoints(t, ni_x, float(xo)*y2))
    return ni_t, ni_x, rate, dt

def load_species(species, string):
    for item in str(string).strip("{{}}").split(","):
        k = item.split(":")
        if str(species) == k[0].strip('"').strip("'"):
            return float(k[1])
    raise KeyError(f"Species {{species}} not found")
"""

# FlameMaster ignition-delay extract script (shared between BL and non-BL)
_FM_EXTRACT_TIG = """#!/usr/bin/python
import numpy as np
import os
import pandas as pd
from scipy.interpolate import CubicSpline

os.chdir("output")
ignitionDelayDefination = "{ign_def}"
ign_cond = "{cond}"
specific = "{specific}"
atomic_weight = {molecular_weight}

def find_nearest(array, value):
    array = np.asarray(np.abs(np.log(array)))
    value = np.abs(np.log(value))
    return (np.abs(array - value)).argmin(), array[(np.abs(array - value)).argmin()]

def ignitionDelay(df, species, cond="max", specified_value="None;None"):
    Y, X = None, None
    for col in df:
        a = col.strip()
        if a == "X-"+species or species.upper() in a:
            Y = list(df[col])
        elif "P" in a and Y is None:
            Y = list(df[col])
        if a == "t[ms]":
            X = list(df[col])
    if Y is None or X is None:
        raise AssertionError(f"Could not find columns for species={{species}}")
    time, profile, t = np.asarray(X), np.asarray(Y), np.asarray(X)
    if cond == "max":
        return time[list(profile).index(max(profile))]
    elif cond == "onset":
        slope = np.diff(profile) / np.diff(time)
        return t[int(np.diff(slope).argmax())]
    elif cond == "dt-max":
        slope = np.diff(profile) / np.diff(time)
        return t[int(slope.argmax())]
    elif cond == "specific":
        tgt  = float(specified_value.split(";")[0])
        unit = specified_value.split(";")[1]
        if unit == "molecule":
            tgt = tgt / 6.02214e23 * atomic_weight
            idx = find_nearest(profile, tgt)[0]
            return time[idx]
        f    = CubicSpline(time, profile, bc_type="natural")
        tn   = np.arange(min(time), max(time), 1e-8)
        idx  = find_nearest(f(tn), tgt)[0]
        return time[idx]

list_files = os.listdir()
outfile = next((f for f in list_files if f.startswith("Y" if "p" in ignitionDelayDefination else "X")), None)
if outfile is None:
    raise FileNotFoundError("No output file found in ./output")

data = [line.strip("\\n") for line in open(outfile).readlines()[1:]]
open("modified.csv", "w+").write("\\n".join(data))
df   = pd.read_csv("modified.csv", sep="\\t")

tau = ignitionDelay(df, ignitionDelayDefination, ign_cond, specific)
open("tau.out", "w+").write("tau\\t{{}}\\tms".format(tau))
"""


# ════════════════════════════════════════════════════════════════════════════
#  Public helper functions
# ════════════════════════════════════════════════════════════════════════════

def create_JPDAP_input(input_dict):
    return (
        "* Uncertain Arrhenius parameters [A/An/AE/AnE]\n"
        f"{input_dict['uncertain_parameters']}\n"
        "* Uncertainty type [3slog10k/2slog10k/1slog10k/1slnk/2slnk/3slnk]\n"
        f"{input_dict['uncertainty_type']}\n"
        "* Number of data (1st row),data in rows: temperature uncertainty\n"
        f"{input_dict['len_temp_data']}\n"
        f"{input_dict['temperature_unsrt_data']}\n"
        "* Test sets: number of sets (1st row),sets in rows "
        "[sa,sn,se,ran,rae,rne, omit missing] a=lnA,e=E/R\n"
        f"{input_dict['L']}"
    )


def create_SAMAP_input(input_dict):
    return (
        "* Uncertain Arrhenius parameters [A/An/AE/AnE]\n"
        f"{input_dict['uncertain_parameters']}\n"
        "* Uncertainty type [3slog10k/2slog10k/1slog10k/1slnk/2slnk/3slnk]\n"
        f"{input_dict['uncertainty_type']}\n"
        "* Mean values of the uncertain Arrhenius parameters (A,n,E/R[K])\n"
        f"{input_dict['alpha']} {input_dict['n']} {input_dict['epsilon']}\n"
        "* Covariance matrix [(a,n,e)x(a,n,e)]\n"
        f" {input_dict['covariance_matrix']}\n"
        "* n limits: n_min n_max\n"
        f"{input_dict['n_min']} {input_dict['n_max']}\n"
        "* Temperature range: Tmin Tmax\n"
        f"{input_dict['T_begin']} {input_dict['T_end']}\n"
        "* Number of equidistant T points (>=10)\n"
        f"{input_dict['equidistant_T']}\n"
        "* Distribution (UNIFORM/NORMAL)\n"
        f"{input_dict['sampling_distribution']}\n"
        "* Sampling method\n"
        f"{input_dict['sampling_method']}\n"
        "* Random seed\n"
        f"{input_dict['Random_seed']}\n"
        "* Number of samples and skipped (SOBOL)\n"
        f"{input_dict['samples']} {input_dict['samples_skipped']}"
    )


def create_start_profile_input(inputs, target):
    """
    Generate profile_generator.input and run_profile for FlameMaster start-profile generation.
    BUG FIX: subprocess.call now uses list form.
    """
    sp = inputs["StartProfilesData"][target.target]
    units_str = yaml.dump(
        yaml.load(str(sp["units"]), Loader=Loader),
        default_flow_style=True
    )
    instring = f"""#########
# Input #
#########

Inputs:

 bin: {inputs["Bin"]["bin"]}

 Flame: {sp["Flame"]}

 MechanismFile: {inputs["Locations"]["Initial_pre_file"]}

 StartProfilesFile: {sp["StartProfilesFile"]}

 ComputeWithRadiation: {target.add["ComputeWithRadiation"]}

 Thermodiffusion: {target.add["Thermodiffusion"]}

 ExpTempFile: {target.add["ExpTempFile"]}

 CopyTo: {sp["CopyTo"]}

 UniqueID: {target.uniqueID}

#######################
# Boundary conditions #
#######################

Boundary:

 fuel: {sp["fuel"]}

 oxidizer: {sp["oxidizer"]}

 bathGas: {sp["bathGas"]}

 globalReaction: {sp["globalReaction"]}

 pressure: {target.pressure}

 temperature: {target.temperature}

 flowRate: {target.add["flow_rate"]}

 units: {units_str}"""

    with open("profile_generator.input", "w") as f:
        f.write(instring)

    run_src = inputs["Bin"]["bin"] + "/profileGenerator.py"
    with open("run_profile", "w") as f:
        f.write(f"#!/bin/bash\npython3 {run_src} profile_generator.input &> profile.log\n")

    # BUG FIX: must be a list, not a bare string
    subprocess.call(["chmod", "+x", "run_profile"])


def create_inputs_to_start(inputs, phi, fuel, oxidizer, bathgas, globeRxn, P, T, controls):
    """Generate FlameMaster .input file and run_generate script."""
    exp = inputs["exp_type"]["exp"]

    if exp == "Fls":
        instring = """############
# Numerics #
############

TimeDepFlag = {time_flag}
DeltaTStart = 1.0e-8
DeltaTMax = 1.0e5
UseNumericalJac is TRUE
UseSecondOrdJac is TRUE
UseModifiedNewton = TRUE
DampFlag = TRUE
LambdaMin = 1.0e-2
MaxIter = 5000
TolRes = 1.0e-15
TolDy = 1e-4

DeltaNewGrid = 25
OneSolutionOneGrid = TRUE
initialgridpoints = {init_grid}
maxgridpoints = {max_grid}
q = -0.25
R = 60

OutputPath is ./output
StartProfilesFile is {s_p_loc}

MechanismFile is {p_f}
globalReaction is {g_r};

{fuel_is}
oxidizer is {oxidizer}

Flame is {flame}
ExactBackward is TRUE

{phi}

pressure = {pressure}
ComputeWithRadiation is {radiation_tag}
Thermodiffusion is {thermodiffusion_tag}

Unburnt Side {{
    dirichlet {{
        t = {temperature}
        {dirichlet}
    }}
}}

{bounds}
""".format(
            time_flag=inputs["TimeFlag"], init_grid=inputs["initialgridpoints"],
            max_grid=inputs["maxgridpoints"], s_p_loc=inputs["StartProfilesFile"],
            p_f=inputs["MechanismFile"], g_r=globeRxn, fuel_is=fuel["fuelIs"],
            oxidizer=oxidizer["oxidizerIs"], flame=inputs["Flame"], phi=phi,
            pressure=P, radiation_tag=inputs["ComputeWithRadiation"],
            thermodiffusion_tag=inputs["Thermodiffusion"], temperature=T,
            dirichlet=controls["dirichlet"], bounds=controls["conc_bounds"]
        )

    elif exp == "Flf":
        instring = """############
# Numerics #
############

UseNumericalJac is TRUE
UseSecondOrdJac is TRUE
UseModifiedNewton = TRUE
DampFlag = TRUE
LambdaMin = 1.0e-2
TimeDepFlag is {time_flag}
DeltaTStart = 1e-5
MaxIter = 5000
TolRes = 1.0e-15
TolDy = 1e-4

DeltaNewGrid = 25
OneSolutionOneGrid = TRUE
initialgridpoints = {init_grid}
maxgridpoints = {max_grid}
q = -0.25
R = 60

WriteEverySolution is TRUE
OutputPath is ./output
StartProfilesFile is {s_p_loc}

MechanismFile is {p_f}
globalReaction is {g_r};

{fuel_is}
oxidizer is {oxidizer}

Flame is {flame}
ExactBackward is TRUE

pressure = {pressure}
ComputeWithRadiation is {radiation_tag}
Thermodiffusion is {thermodiffusion_tag}
TransModel is MonoAtomic
ExpTempFile is {etp}

ConstLewisNumber is TRUE
ConstMassFlux is TRUE
MassFlowRate = {flow_rate}

Unburnt Side {{
    dirichlet {{
        t = {temperature}
        {dirichlet}
    }}
}}
""".format(
            time_flag=inputs["TimeFlag"], init_grid=inputs["initialgridpoints"],
            max_grid=inputs["maxgridpoints"], s_p_loc=inputs["StartProfilesFile"],
            p_f=inputs["MechanismFile"], g_r=globeRxn, fuel_is=fuel["fuelIs"],
            oxidizer=oxidizer["oxidizerIs"], flame=inputs["Flame"], pressure=P,
            radiation_tag=inputs["ComputeWithRadiation"],
            thermodiffusion_tag=inputs["Thermodiffusion"],
            etp=inputs["ExpTempFile"], flow_rate=inputs["flowRate"],
            temperature=T, dirichlet=controls["dirichlet"],
            bounds=controls["conc_bounds"]
        )
    else:
        raise ValueError(f"create_inputs_to_start: unknown exp type {exp!r}")

    with open("FlameMaster.input", "w") as f:
        f.write(instring)

    with open("run_generate", "w") as f:
        f.write(f"#!/bin/bash\n{inputs['bin']}/FlameMan &> Flame.log\n")
    subprocess.call(["chmod", "+x", "run_generate"])


# ════════════════════════════════════════════════════════════════════════════
#  Internal: start-profile generation helper
# ════════════════════════════════════════════════════════════════════════════

def _generate_start_profile(opt_dict, target, startProfile_location, file_specific_command):
    """
    Generate and run the FlameMaster start-profile if it hasn't been cached yet.
    Sets target.add["StartProfilesFile"] in both branches.
    """
    sp_dir = startProfile_location[target.target]["CopyTo"]
    if target.uniqueID not in os.listdir(sp_dir):
        od = deepcopy(opt_dict)
        tt = target.target

        def _yml(d):
            return yaml.dump(yaml.load(str(d), Loader=Loader), default_flow_style=True)

        od["StartProfilesData"][tt]["fuel"]["key"] = (
            str(od["StartProfilesData"][tt]["fuel"]["key"]) + "-" + str(target.fuel_type))
        od["StartProfilesData"][tt]["fuel"]["To"]       = target.fuel_id
        od["StartProfilesData"][tt]["fuel"]["ToConc"]   = target.fuel_x
        od["StartProfilesData"][tt]["oxidizer"]["To"]   = target.oxidizer
        od["StartProfilesData"][tt]["oxidizer"]["ToConc"] = target.oxidizer_x
        od["StartProfilesData"][tt]["bathGas"]["key"]   = (
            str(od["StartProfilesData"][tt]["bathGas"]["key"]) + "-" + str(target.bath_gas_id))
        od["StartProfilesData"][tt]["bathGas"]["To"]    = target.bath_gas
        od["StartProfilesData"][tt]["bathGas"]["ToConc"] = target.bath_gas_x
        od["StartProfilesData"][tt]["globalReaction"]["ToRxn"] = od["Inputs"]["global_reaction"]

        od["StartProfilesData"][tt]["fuel"]            = _yml(od["StartProfilesData"][tt]["fuel"])
        od["StartProfilesData"][tt]["oxidizer"]        = _yml(od["StartProfilesData"][tt]["oxidizer"])
        od["StartProfilesData"][tt]["bathGas"]         = _yml(od["StartProfilesData"][tt]["bathGas"])
        od["StartProfilesData"][tt]["globalReaction"]  = _yml(od["StartProfilesData"][tt]["globalReaction"])

        target.add["Fsc"] = file_specific_command
        create_start_profile_input(od, target)
        os.makedirs("output", exist_ok=True)
        # BUG FIX: subprocess.call requires list form
        subprocess.call(["./run_profile"])

    target.add["StartProfilesFile"] = sp_dir + "/" + str(target.uniqueID)


# ════════════════════════════════════════════════════════════════════════════
#  Main entry point
# ════════════════════════════════════════════════════════════════════════════

def create_input_file(case, opt_dict, target, mech_file=None):
    """
    Generate simulation input, conversion, run, and extraction scripts.

    Returns
    -------
    (instring, s_convert, s_run, extract)
    """
    thermo_file_location   = opt_dict["Locations"]["thermo_file"]
    trans_file_location    = opt_dict["Locations"]["trans_file"]
    startProfile_location  = opt_dict["StartProfilesData"]
    file_specific_command  = "-f chemkin"
    mech_file              = mech_file or "mechanism.yaml"
    global_reaction        = opt_dict["Inputs"]["global_reaction"]

    instring = ""
    extract  = ""

    # ── custom input file overrides everything ────────────────────────────
    if target.input_file is not None:
        instring = open(target.input_file, "r").read()

    # ── JSR ───────────────────────────────────────────────────────────────
    elif "JSR" in target.target:
        instring = """import pandas as pd
import time
import cantera as ct

gas = ct.Solution("{mech}")
reactor_temperature = {temperature}
reactor_pressure    = {pressure}
inlet_concentrations = {species_conc}
gas.TPX = reactor_temperature, reactor_pressure, inlet_concentrations
species = "{species}"
residence_time   = {residenceTime}
reactor_volume   = 1
max_simulation_time = {maxSimTime}
fuel_air_mixture_tank = ct.Reservoir(gas)
exhaust = ct.Reservoir(gas)
stirred_reactor = ct.IdealGasReactor(gas, energy="off", volume=reactor_volume)
mass_flow_controller = ct.MassFlowController(
    upstream=fuel_air_mixture_tank, downstream=stirred_reactor,
    mdot=stirred_reactor.mass / residence_time)
pressure_regulator = ct.PressureController(
    upstream=stirred_reactor, downstream=exhaust, master=mass_flow_controller)
reactor_network = ct.ReactorNet([stirred_reactor])
time_history = ct.SolutionArray(gas, extra=["t"])
t = 0; counter = 1
while t < max_simulation_time:
    t = reactor_network.step()
    if counter % 10 == 0:
        time_history.append(stirred_reactor.thermo.state, t=t)
    counter += 1
MF = time_history(species).X[-1][0]
with open("output/jsr.out", "w") as f:
    f.write("#T(K)\\tmole fraction\\n{{}}\\t{{}}".format(reactor_temperature, MF))
""".format(mech=mech_file, temperature=target.temperature, pressure=target.pressure,
           species_conc=target.species_dict, species=str(target.add["species"]),
           residenceTime=target.add["residenceTime"],
           maxSimTime=target.add["maxSimulationTime"])

    # ── Flf (cantera counterflow / flame stabilised) ──────────────────────
    elif "Flf" in target.target and "cantera" in target.add["solver"]:
        instring = """#!/usr/bin/python
import cantera as ct
import numpy as np
import pandas as pd
import time

gas = ct.Solution('{mech}')
gas.TPX = {temperature}, {pressure}, {species_conc}
stirredReactor = ct.IdealGasReactor(gas, energy='off', volume=1)
massFlowController = ct.MassFlowController(
    upstream=ct.Reservoir(gas), downstream=stirredReactor,
    mdot=stirredReactor.mass/{residenceTime})
pressureRegulator = ct.Valve(upstream=stirredReactor,
    downstream=ct.Reservoir(gas), K=0.01)
reactorNetwork = ct.ReactorNet([stirredReactor])
t = 0
while t < {maxSimTime}:
    t = reactorNetwork.step()
Flf = stirredReactor.thermo['{species}'].X[0]
with open("output/flf.out", "w") as f:
    f.write("#T(K)\\tflf(conc.)\\n{{}}\\t{{}}".format({temperature}, Flf))
""".format(mech=mech_file, temperature=target.temperature, pressure=target.pressure,
           species_conc=target.species_dict, species=target.add["species"],
           residenceTime=target.add["residenceTime"],
           maxSimTime=target.add["maxSimulationTime"])

    # ── RCM ───────────────────────────────────────────────────────────────
    elif "RCM" in target.target:
        if "cantera" in target.add["solver"] and "non_reactive" not in target.add["RCM_type"]:
            # ---- Reactive RCM (cantera) -----------------------------------
            _CHEMKIN_RCM_HELPERS = """
import os, csv
import numpy as np
from chemkin import ChemkinJob
import pandas as pd
from scipy.interpolate import CubicSpline, Akima1DInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def find_first_slope_transition(time, temp, tmin, tmax):
    df = pd.DataFrame({{"time": time, "temperature": temp}})
    df = df[(df.temperature >= tmin) & (df.temperature <= tmax)].reset_index(drop=True)
    df["slope"] = np.gradient(df.temperature, df.time)
    for i in range(1, len(df)-1):
        if df.slope.iloc[i-1] > 0 and df.slope.iloc[i] < df.slope.iloc[i-1] and df.slope.iloc[i] > 0:
            return {{"time": df.time.iloc[i], "temperature": df.temperature.iloc[i], "slope": df.slope.iloc[i]}}
    return None

def find_local_peak_in_range(time, temp, tmin, tmax):
    df = pd.DataFrame({{"time": time, "temperature": temp}})
    df = df[(df.time >= tmin) & (df.time <= tmax)].reset_index(drop=True)
    df["slope"] = np.gradient(df.temperature, df.time)
    for i in range(1, len(df)-1):
        if df.slope.iloc[i-1] > 0 and df.slope.iloc[i+1] < 0:
            return {{"time": df.time.iloc[i], "temperature": df.temperature.iloc[i], "slope": df.slope.iloc[i]}}
    return None

def find_critical_point(time, temp, tmin, tmax):
    df = pd.DataFrame({{"time": time, "temperature": temp}})
    df = df[(df.time >= tmin) & (df.time <= tmax)].reset_index(drop=True)
    df["slope"] = np.gradient(df.temperature, df.time)
    for i in range(1, len(df)):
        if df.slope.iloc[i] < 0 and df.slope.iloc[i-1] > 0:
            return {{"type": "slope_decrease", "time": df.time.iloc[i], "temperature": df.temperature.iloc[i]}}
    peak_idx = df.temperature.idxmax()
    return {{"type": "peak", "time": df.time.iloc[peak_idx], "temperature": df.temperature.iloc[peak_idx]}}

def find_slope_change_points(time, temp, tmin, tmax):
    df = pd.DataFrame({{"time": time, "temperature": temp}})
    df = df[(df.time >= tmin) & (df.time <= tmax)].reset_index(drop=True)
    df["slope"] = np.gradient(df.temperature, df.time)
    first = None
    for i in range(1, len(df)):
        if df.slope.iloc[i] < df.slope.iloc[i-1] and df.slope.iloc[i] > 0:
            first = {{"time": df.time.iloc[i], "temperature": df.temperature.iloc[i]}}; break
    d2   = np.gradient(df.slope, df.time)
    idx  = np.argmax(np.abs(d2))
    return {{"first_slope_decrease_point": first,
             "first_major_slope_change_point": {{"time": df.time.iloc[idx], "temperature": df.temperature.iloc[idx]}}}}

def find_nearest_temperature(time_list, temp_list, target_time):
    idx = np.abs(np.array(time_list) - target_time).argmin()
    return {{"time": time_list[idx], "temperature": temp_list[idx]}}

def find_peak_point(time, temp):
    df = pd.DataFrame({{"time": time, "temperature": temp}})
    df["slope"] = np.gradient(df.temperature, df.time)
    idx = df.temperature.idxmax()
    return {{"time": df.time.iloc[idx], "temperature": df.temperature.iloc[idx], "slope": df.slope.iloc[idx]}}
"""
            instring = (_CHEMKIN_RCM_HELPERS + """
from scipy.signal import find_peaks
""" + _CANTERA_HELPERS + _IGN_DELAY_CANTERA + _VOLUME_PROFILE + _DO_NON_REACTIVE + """
from pyked import ChemKED
import cantera as ct, yaml, time, matplotlib.pyplot as plt

def find_nearest(array, value):
    array = np.asarray(np.abs(np.log(array)))
    value = np.abs(np.log(value))
    return (np.abs(array - value)).argmin(), array[(np.abs(array - value)).argmin()]

target_Temperature = {target_Temperature}
reactorTemperature = {temperature_K}
reactorPressure    = {pressure}
""" + _CRITERIA_BLOCK + """
spec     = {species_conc}
conc_X   = list(spec.items())
gas = ct.Solution('{mech}')
gas.TPX = reactorTemperature, reactorPressure, spec
currentDir = os.path.dirname(__file__).strip("\\n")
chemFile   = os.path.join(currentDir, 'mechanism.inp')
tempDir    = os.path.join(currentDir, 'output')
estimatedIgnitionDelayTime = {estimateTIG}
VP_type  = "{VP_FILE_TYPE}"
if VP_type == "ck_file":
    VP = "{volume_profile}"
elif VP_type in ("Dict", "csv_file"):
    df = pd.read_csv("{volume_profile}")
    VP = VolumeProfile({{"vproTime": df["time(s)"], "vproVol": df["volume(cm3)"]}})
else:
    VP = "{volume_profile}"
job = ChemkinJob(name=conc_X[0][0], chemFile=chemFile, tempDir=tempDir)
job.preprocess()
input_file = job.writeInputHomogeneousBatch(
    problemType="constrainVandSolveE", reactants=conc_X,
    temperature=reactorTemperature, pressure=reactorPressure,
    endTime=estimatedIgnitionDelayTime, variableVolume=True,
    variableVolumeProfile=VP, variableVolumeProfileType=VP_type)
job.run(input_file, model="CKReactorGenericClosed", pro=True)
job.postprocess(sens=False, rop=False, all=True, transpose=False)
""" + _CONVERT_CKCSV + """
saveAll = {saveAll}
if saveAll:
    convert_and_transpose_ckcsv(job.ckcsvFile, "time_history.csv")
timeHistory = pd.read_csv("time_history.csv", index_col=0)
tempe = timeHistory["Temperature  (K)"].to_numpy()
time_ = timeHistory.index.to_numpy()
smooth = gaussian_filter1d(tempe, sigma=5)
sc1 = find_first_slope_transition(time_, smooth, 0.8*target_Temperature, target_Temperature+10)
t1  = sc1["time"] if sc1 else time_[0]
sc2 = find_peak_point(time_, tempe)
t2  = sc2["time"]
mx1 = find_slope_change_points(time_, tempe, t1, 0.9*t2)
if mx1["first_slope_decrease_point"]:
    t1d = mx1["first_slope_decrease_point"]["time"]
    T1d = mx1["first_slope_decrease_point"]["temperature"]
else:
    lp  = find_local_peak_in_range(time_, tempe, t1, 1.06*t1) or {{"time": t1, "temperature": tempe[0]}}
    t1d = lp["time"]; T1d = lp["temperature"]
mx2 = find_slope_change_points(time_, smooth, t2, 1.5*t2)
if mx2["first_slope_decrease_point"]:
    t2d = mx2["first_slope_decrease_point"]["time"]
else:
    cp  = find_critical_point(time_, smooth, t2, 1.5*t2)
    t2d = cp["time"]
t3 = find_nearest_temperature(list(time_), list(tempe), (t2+t2d)/2)["time"]
tau = t3 - t1d
fig = plt.figure(); plt.plot(t1d, T1d, "o"); plt.plot(t3, tempe[list(time_).index(t3)] if t3 in time_ else T1d, "o"); plt.plot(time_, tempe, "-"); plt.savefig("Ignition_def.pdf")
with open("output/RCM.out", "w") as tau_file:
    tau_file.write(f"#T(K)\\ttau(us)\\n{{T1d}}\\t{{tau * 1e6}}\\n")
""" + _DELETE_OUTPUT_FILES + """
_delete_output_files("output/", "RCM.out")
""").format(
                mech=mech_file,
                target_Temperature=target.temperature,
                temperature_K=target.temperature_i,
                pressure=target.pressure_i,
                species_conc=target.species_dict,
                exp_conc=target.add["exp_conc"][float(target.temperature)],
                volume_profile=target.add["volume_profile"][float(target.temperature)],
                VP_FILE_TYPE=target.add["volume_profile_type"],
                estimateTIG=float(target.add["estimateTIG"]),
                delay_def=target.add["ign_delay_def"],
                delay_cond=target.add["ign_cond"],
                specific_cond=target.add["specific_cond"],
                saveAll=target.add["saveAll"])

        elif "cantera" in target.add["solver"] and "non_reactive" in target.add["RCM_type"]:
            # ---- Non-reactive RCM (cantera) ------------------------------
            _VP_pos_str = _VOLUME_PROFILE_POS  # this variant takes (time, volume)
            instring = ("""#!/usr/bin/python
from __future__ import division, print_function
import pandas as pd, numpy as np, time, cantera as ct
from scipy.interpolate import CubicSpline, Akima1DInterpolator
import matplotlib.pyplot as plt
from scipy.integrate import ode
from scipy.signal import find_peaks

class TemperatureFromPressure(object):
    def __init__(self, pressure, T_initial, chem_file="species.cti",
                 cti_source=None, init_X=None):
        gas = ct.Solution(source=cti_source) if cti_source else ct.Solution(chem_file)
        gas.TP = T_initial, pressure[0]*1e5 if init_X is None else None
        if init_X: gas.TPX = T_initial, pressure[0]*1e5, init_X
        initial_entropy = gas.entropy_mass
        self.temperature = np.zeros(len(pressure))
        for i, p in enumerate(pressure):
            gas.SP = initial_entropy, p*1e5
            self.temperature[i] = gas.T
""" + _VP_pos_str + _CANTERA_HELPERS + _IGN_DELAY_CANTERA + _DO_NON_REACTIVE + """
VP_file_type = "{VP_FILE_TYPE}"
if VP_file_type == "ck_file":
    df = pd.read_csv("{volumeProfile}", delim_whitespace=True, header=None)
    df.columns = ["Tag", "Time", "Volume"]
    inp_time = df["Time"].to_numpy(); inp_vol = df["Volume"].to_numpy()
else:
    df = pd.read_csv("{volumeProfile}", sep=",")
    inp_time = df["time(s)"].to_numpy(); inp_vol = df["volume(cm3)"].to_numpy()

TDC_time = inp_time[list(inp_vol).index(min(inp_vol))]
gas    = ct.Solution("{mech}")
gas_nr = ct.Solution("{mech}")
reactorTemperature = {temperature}
reactorPressure    = {pressure}
""" + _CRITERIA_BLOCK + """
gas.TPX    = reactorTemperature, reactorPressure, {species_conc}
gas_nr.TPX = reactorTemperature, reactorPressure, {species_conc}

timeHistory_nonreactive, pList_nr, vol_nr = doNonReactive(gas_nr, inp_time, inp_vol)
fig = plt.figure()
plt.plot(list(timeHistory_nonreactive.index), pList_nr, "r-", label="Non-reactive")
plt.savefig("p_profile.png", bbox_inches="tight")
""").format(
                mech=mech_file,
                volumeProfile=target.add["volume_profile"][float(target.temperature)],
                temperature=target.temperature, pressure=target.pressure,
                species_conc=target.species_dict,
                exp_conc=target.add["exp_conc"][float(target.temperature)],
                VP_FILE_TYPE=target.add["volume_profile_type"],
                delay_def=target.add["ign_delay_def"],
                delay_cond=target.add["ign_cond"],
                specific_cond=target.add["specific_cond"],
                saveAll=target.add["saveAll"])

        elif "CHEMKIN_PRO" in target.add["solver"]:
            raise AssertionError("CHEMKIN_PRO RCM: use the reactive cantera branch")
        else:
            raise AssertionError(f"Invalid solver for RCM: {target.add['solver']!r}")

    # ── Tig (ignition delay) ──────────────────────────────────────────────
    elif "Tig" in target.target:
        if target.ig_mode == "RCM" and target.simulation == "VTIM":
            if "cantera" in target.add["solver"]:
                # VTIM with dp/dt profile
                instring = ("""#!/usr/bin/python
from __future__ import division, print_function
import pandas as pd, numpy as np, time, cantera as ct
from scipy.interpolate import CubicSpline, Akima1DInterpolator
import matplotlib.pyplot as plt
from scipy.integrate import ode
from scipy.signal import find_peaks
""" + _CANTERA_HELPERS + _IGN_DELAY_CANTERA + _VOLUME_PROFILE + """
def getTimeProfile(t1, t2): return np.arange(t1, t2, {dt})

def getPressureProfile(gas, time, dpdt):
    p = gas.P; PressureProfile = []
    for t in time:
        PressureProfile.append(p); p += {dt} * (0.01 * dpdt * p * 1000)
    return np.asarray(PressureProfile)

def getVolumeProfile_From_PressureProfile(gas, PP):
    rho_o, Po = gas.DP; gamma = gas.cp / gas.cv
    return np.asarray([(1/rho_o)*(P/Po)**(-1/gamma) for P in PP])

gas = ct.Solution("{mech}")
reactorTemperature = {temperature}
reactorPressure    = {pressure}
""" + _CRITERIA_BLOCK + """
gas.TPX = reactorTemperature, reactorPressure, {species_conc}
r = ct.IdealGasReactor(contents=gas, name="Batch Reactor")
env = ct.Reservoir(ct.Solution("{mech}"))
dpdt = {dpdt}
estimatedIgnitionDelayTime = {estimateTIG}
time_arr = getTimeProfile(0, estimatedIgnitionDelayTime)
pressure_profile = getPressureProfile(gas, time_arr, dpdt)
volume_profile   = getVolumeProfile_From_PressureProfile(gas, pressure_profile)
string = "time(s),volume(cm3)\\n" + "".join(f"{{t}},{{v}}\\n" for t,v in zip(time_arr, volume_profile))
open(f"VTIM_P_{{int(reactorPressure/100000)}}_T_{{int(reactorTemperature)}}.csv","w").write(string)
keywords = {{"vproTime": time_arr, "vproVol": volume_profile}}
ct.Wall(r, env, velocity=VolumeProfile(keywords))
reactorNetwork = ct.ReactorNet([r])
reactorNetwork.max_time_step = 0.0001
cols = [r.component_name(i) for i in range(r.n_vars)]
timeHistory = pd.DataFrame(columns=cols)
t = 0; pressureList = []; counter = 1
while t < estimatedIgnitionDelayTime:
    t = reactorNetwork.step()
    if counter % 1 == 0:
        timeHistory.loc[t] = reactorNetwork.get_state().astype("float64")
        pressureList.append(gas.P)
    counter += 1
tau = ignitionDelay(timeHistory, pressureList, "{delay_def}", "{delay_cond}", "{specific_cond}", {exp_conc})
with open("output/tau.out", "w") as f:
    f.write("#T(K)\\ttau(us)\\n{{}}\\t{{}}".format(reactorTemperature, tau*1e6))
species = "{delay_def}"
time_out = timeHistory.index.to_numpy()
conc_out = pressureList if species == "p" else timeHistory[species].to_numpy()
fig = plt.figure(); plt.plot(time_out, conc_out, "b-"); plt.savefig("profile.pdf")
saveAll = {saveAll}
if saveAll: timeHistory.to_csv("time_history.csv")
""").format(mech=mech_file, temperature=target.temperature, pressure=target.pressure,
                            species_conc=target.species_dict,
                            exp_conc=target.add["exp_conc"][float(target.temperature)],
                            dpdt=target.add["dpdt"], dt=float(target.add["dt"]),
                            estimateTIG=float(target.add["estimateTIG"]),
                            delay_def=target.add["ign_delay_def"],
                            delay_cond=target.add["ign_cond"],
                            specific_cond=target.add["specific_cond"],
                            saveAll=target.add["saveAll"])

            elif "cantera" in target.add["solver"] and not target.add.get("BoundaryLayer"):
                # Standard constant-pressure batch reactor (cantera)
                instring = ("""#!/usr/bin/python
from __future__ import division, print_function
import pandas as pd, numpy as np, time, cantera as ct
from scipy.interpolate import CubicSpline, Akima1DInterpolator
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
""" + _CANTERA_HELPERS + _IGN_DELAY_CANTERA + """
gas = ct.Solution("{mech}")
reactorTemperature = {temperature}
reactorPressure    = {pressure}
""" + _CRITERIA_BLOCK + """
gas.TPX = reactorTemperature, reactorPressure, {species_conc}
r = ct.IdealGasReactor(contents=gas, name="Batch Reactor")
reactorNetwork = ct.ReactorNet([r])
reactorNetwork.max_time_step = 0.0001
cols = [r.component_name(i) for i in range(r.n_vars)]
timeHistory = pd.DataFrame(columns=cols)
estimatedIgnitionDelayTime = {estimateTIG}
t = 0; pressureList = []; counter = 1
while t < estimatedIgnitionDelayTime:
    t = reactorNetwork.step()
    if counter % 1 == 0:
        timeHistory.loc[t] = reactorNetwork.get_state().astype("float64")
        pressureList.append(gas.P)
    counter += 1
tau = ignitionDelay(timeHistory, pressureList, "{delay_def}", "{delay_cond}", "{specific_cond}", {exp_conc})
with open("output/tau.out", "w") as f:
    f.write("#T(K)\\ttau(us)\\n{{}}\\t{{}}".format(reactorTemperature, tau*1e6))
saveAll = {saveAll}
if saveAll: timeHistory.to_csv("time_history.csv")
""").format(mech=mech_file, temperature=target.temperature, pressure=target.pressure,
                            species_conc=target.species_dict,
                            exp_conc=target.add["exp_conc"][float(target.temperature)],
                            delay_def=target.add["ign_delay_def"],
                            delay_cond=target.add["ign_cond"],
                            estimateTIG=target.add["estimateTIG"],
                            specific_cond=target.add["specific_cond"],
                            saveAll=target.add["saveAll"])

            elif "CHEMKIN_PRO" in target.add["solver"] and not target.add.get("BoundaryLayer"):
                # CHEMKIN-Pro standard Tig
                instring = ("""#!/usr/bin/env python
import os, csv, numpy as np
from chemkin import ChemkinJob
import pandas as pd
from scipy.interpolate import CubicSpline, Akima1DInterpolator
from scipy.signal import find_peaks
""" + _CANTERA_HELPERS + _IGN_DELAY_CHEMKIN + _CONVERT_CKCSV + """
gas = None  # not needed for CHEMKIN path
reactorTemperature = {temperature_K}
reactorPressure    = {pressure}
spec   = {species_conc}
conc_X = list(spec.items())
currentDir = os.path.dirname(__file__).strip("\\n")
chemFile   = os.path.join(currentDir, "mechanism.inp")
tempDir    = os.path.join(currentDir, "output")
""" + _CRITERIA_BLOCK + """
job = ChemkinJob(name=conc_X[0][0], chemFile=chemFile, tempDir=tempDir)
job.preprocess()
input_file = job.writeInputHomogeneousBatch(
    problemType="constrainVandSolveE", reactants=conc_X,
    temperature=reactorTemperature, pressure=reactorPressure,
    endTime={End_Time})
job.run(input_file, model="CKReactorGenericClosed", pro=True)
job.postprocess(sens=False, rop=False, all=True, transpose=False)
convert_and_transpose_ckcsv(job.ckcsvFile, "time_history.csv")
timeHistory  = pd.read_csv("time_history.csv", index_col=0)
pressureList = timeHistory["Pressure  (bar)"]
species_col  = f"Mole_fraction_{{'{delay_def}'}}  ()" if "{delay_def}" != "p" else "Pressure  (bar)"
tau = ignitionDelay(timeHistory, pressureList, species_col, "{delay_cond}", "{specific_cond}", {exp_conc})
if tau:
    with open("output/tau.out", "w") as f:
        f.write(f"#T(K)\\ttau(us)\\n{{reactorTemperature}}\\t{{tau * 1e6}}\\n")
""" + _DELETE_OUTPUT_FILES + """
_delete_output_files("output/", "tau.out")
""").format(mech=mech_file, temperature_K=target.temperature, pressure=target.pressure,
                            species_conc=target.species_dict,
                            exp_conc=target.add["exp_conc"][float(target.temperature)],
                            End_Time=target.add["estimateTIG"],
                            delay_def=target.add["ign_delay_def"],
                            delay_cond=target.add["ign_cond"],
                            specific_cond=target.add["specific_cond"],
                            saveAll=target.add["saveAll"])

            elif "CHEMKIN_PRO" in target.add["solver"] and target.add.get("BoundaryLayer"):
                # CHEMKIN-Pro with boundary-layer / dpdt profile
                instring = ("""#!/usr/bin/env python
import os, csv, numpy as np
from chemkin import ChemkinJob
import pandas as pd, cantera as ct
from scipy.interpolate import CubicSpline, Akima1DInterpolator
from scipy.signal import find_peaks
global dt
dt = {dt}
""" + _CANTERA_HELPERS + _IGN_DELAY_CHEMKIN + _VOLUME_PROFILE + _CONVERT_CKCSV + """
def getTimeProfile(t1, t2): return np.arange(t1, t2, {dt})
def getPressureProfile(gas, time, dpdt):
    p = gas.P; PP = []
    for t in time:
        PP.append(p); p += {dt}*(0.01*dpdt*p*1000)
    return np.asarray(PP)
def getVolumeProfile_From_PressureProfile(gas, PP):
    rho_o, Po = gas.DP; gamma = gas.cp/gas.cv
    return np.asarray([(1/rho_o)*(P/Po)**(-1/gamma) for P in PP])

gas = ct.Solution("{mech}")
reactorTemperature = {temperature_K}
reactorPressure    = {pressure}
spec   = {species_conc}
conc_X = list(spec.items())
gas.TPX = reactorTemperature, reactorPressure, spec
""" + _CRITERIA_BLOCK + """
dpdt = {dpdt}
estimatedIgnitionDelayTime = {estimateTIG}
time_arr = getTimeProfile(0, estimatedIgnitionDelayTime)
PP  = getPressureProfile(gas, time_arr, dpdt)
VP  = getVolumeProfile_From_PressureProfile(gas, PP)
keywords = {{"vproTime": time_arr, "vproVol": VP}}
vp_obj = VolumeProfile(keywords)
currentDir = os.path.dirname(__file__).strip("\\n")
chemFile   = os.path.join(currentDir, "mechanism.inp")
tempDir    = os.path.join(currentDir, "output")
job = ChemkinJob(name=conc_X[0][0], chemFile=chemFile, tempDir=tempDir)
job.preprocess()
input_file = job.writeInputHomogeneousBatch(
    problemType="constrainVandSolveE", reactants=conc_X,
    temperature=reactorTemperature, pressure=reactorPressure,
    endTime=estimatedIgnitionDelayTime, variableVolume=True,
    variableVolumeProfile=vp_obj, variableVolumeProfileType="Dict")
job.run(input_file, model="CKReactorGenericClosed", pro=True)
job.postprocess(sens=False, rop=False, all=True, transpose=False)
convert_and_transpose_ckcsv(job.ckcsvFile, "time_history.csv")
timeHistory  = pd.read_csv("time_history.csv", index_col=0)
pressureList = timeHistory["Pressure  (bar)"]
delay_def    = "{delay_def}"
species_col  = f"Mole_fraction_{{delay_def}}  ()" if delay_def != "p" else "Pressure  (bar)"
tau = ignitionDelay(timeHistory, pressureList, species_col, "{delay_cond}", "{specific_cond}", {exp_conc})
if tau:
    with open("output/tau.out", "w") as f:
        f.write(f"#T(K)\\ttau(us)\\n{{reactorTemperature}}\\t{{tau * 1e6}}\\n")
""" + _DELETE_OUTPUT_FILES + """
_delete_output_files("output/", "tau.out")
""").format(mech=mech_file, temperature_K=target.temperature, pressure=target.pressure,
                            species_conc=target.species_dict,
                            exp_conc=target.add["exp_conc"][float(target.temperature)],
                            dpdt=target.add["dpdt"], dt=float(target.add["dt"]),
                            estimateTIG=float(target.add["estimateTIG"]),
                            delay_def=target.add["ign_delay_def"],
                            delay_cond=target.add["ign_cond"],
                            specific_cond=target.add["specific_cond"],
                            saveAll=target.add["saveAll"])

            elif "FlameMaster" in target.add["solver"] and target.add.get("BoundaryLayer"):
                # FlameMaster Tig with dp/dt boundary layer
                # BUG FIX: removed stray positional {} that mixed with keyword args
                instring = """############
# Numerics #
############

RelTol = 1.0e-9
AbsTol = 1.0e-12
TStart = 0.0
TEnd = 0.05

WriteEverySolution is TRUE
PrintMolarFractions is TRUE
OutputPath is ./output
NOutputs = 2000

MechanismFile is mechanism.pre
globalReaction is {g_r};

{fuel_is}
oxidizer is O2

Flame is {simulation}
ExactBackward is TRUE
Pressure = {pressure}
PressureChange = {dpdt}

InitialCond {{
    t = {temperature}
    {init_condition}
}}
""".format(pressure=target.pressure, init_condition=target.initialCond,
           temperature=target.temperature, phi=target.phi,
           simulation=target.simulation, fuel_is=target.fuel_is,
           g_r=global_reaction,
           dpdt=target.add["dpdt"][target.temperature] / 100)
                extract = _FM_EXTRACT_TIG.format(
                    ign_def=target.add["ign_delay_def"],
                    cond=target.add["ign_cond"],
                    specific=target.add["specific_cond"],
                    molecular_weight=target.add["mol_wt"])

            else:
                # FlameMaster standard Tig (homogeneous)
                instring = """############
# Numerics #
############

RelTol = 1.0e-9
AbsTol = 1.0e-12
TStart = 0.0
TEnd = {tend}

WriteEverySolution is TRUE
PrintMolarFractions is TRUE
OutputPath is ./output
NOutputs = 1500

MechanismFile is mechanism.pre
globalReaction is {g_r};

{fuel_is}
oxidizer is O2

Flame is {simulation}
ExactBackward is TRUE
Pressure = {pressure}

InitialCond {{
    t = {temperature}
    {init_condition}
}}
""".format(pressure=target.pressure, init_condition=target.initialCond,
           temperature=target.temperature, phi=target.phi,
           simulation=target.simulation, fuel_is=target.fuel_is,
           g_r=global_reaction, tend=float(target.add["estimateTIG"]))
                extract = _FM_EXTRACT_TIG.format(
                    ign_def=target.add["ign_delay_def"],
                    cond=target.add["ign_cond"],
                    specific=target.add["specific_cond"],
                    molecular_weight=target.add["mol_wt"])
        else:
            raise AssertionError(
                f"Unknown ignition mode / simulation combo: {target.ig_mode!r} / {target.simulation!r}")

    # ── Fls (laminar flame speed) ─────────────────────────────────────────
    elif "Fls" in target.target:
        if "cantera" in target.add["solver"] and "phi" not in target.add.get("type", ""):
            instring = """#!/usr/bin/python
from __future__ import print_function, division
import cantera as ct, numpy as np, pandas as pd
To = {temperature}
Po = {pressure}
gas = ct.Solution("{mech}")
gas.TPX = To, Po, {species_conc}
width = {width}
flame = ct.FreeFlame(gas, width=width)
flame.set_refine_criteria(ratio={ratio}, slope={slope}, curve={curve}, prune=0.003)
flame.soret_enabled = True
flame.energy_enabled = True
flame.transport_model = "Multi"
flame.solve(loglevel={loglevel}, refine_grid=True, auto={auto})
Su0 = flame.velocity[0]
with open("output/Su.out", "w") as f:
    f.write("#T(K)\\tSu(cm/s)\\n{{}}\\t{{}}".format(To, Su0*100))
""".format(mech=mech_file, temperature=target.temperature, pressure=target.pressure,
           species_conc=target.species_dict, width=target.add["width"],
           ratio=target.add["ratio"], slope=target.add["slope"],
           curve=target.add["curve"], loglevel=target.add["loglevel"],
           auto=target.add["auto"])

        elif "cantera" in target.add["solver"] and "phi" in target.add.get("type", ""):
            instring = """#!/usr/bin/python
from __future__ import print_function, division
import cantera as ct, numpy as np, pandas as pd
To = {temperature}
Po = {pressure}
gas = ct.Solution("{mech}")
gas.set_equivalence_ratio({phi}, {fuel}, {{"O2": 1.0, "N2": 3.76}})
width = {width}
flame = ct.FreeFlame(gas, width=width)
flame.set_refine_criteria(ratio={ratio}, slope={slope}, curve={curve}, prune=0.003)
flame.soret_enabled = True
flame.energy_enabled = True
flame.transport_model = "Multi"
flame.solve(loglevel={loglevel}, refine_grid=True, auto={auto})
Su0 = flame.velocity[0]
with open("output/Su.out", "w") as f:
    f.write("#T(K)\\tSu(cm/s)\\n{{}}\\t{{}}".format(To, Su0*100))
""".format(mech=mech_file, temperature=target.temperature, phi=target.phi,
           fuel=target.add["fuel"], pressure=target.pressure,
           species_conc=target.species_dict, width=target.add["width"],
           ratio=target.add["ratio"], slope=target.add["slope"],
           curve=target.add["curve"], loglevel=target.add["loglevel"],
           auto=target.add["auto"])

        else:
            # FlameMaster free-flame
            _generate_start_profile(opt_dict, target, startProfile_location,
                                    file_specific_command)
            instring = """############
# Numerics #
############

DeltaTStart = 1.0e-8
DeltaTMax = 1.0e5
UseNumericalJac is TRUE
UseSecondOrdJac is TRUE
UseModifiedNewton = TRUE
DampFlag = TRUE
LambdaMin = 1.0e-2
MaxIter = 5000
TolRes = 1.0e-15
TolDy = 1e-4

DeltaNewGrid = 25
OneSolutionOneGrid = TRUE
initialgridpoints = 89
maxgridpoints = 700
q = -0.25
R = 60

WriteEverySolution is TRUE
PrintMolarFractions is TRUE
AdditionalOutput is TRUE
OutputPath is ./output
StartProfilesFile is {s_p_loc}

MechanismFile is mechanism.pre
globalReaction is {g_r};

{fuel_is}
oxidizer is {oxidizer}

Flame is {simulation}
ExactBackward is TRUE
pressure = {pressure}
ComputeWithRadiation is {radiation_tag}
Thermodiffusion is {thermodiffusion_tag}

Unburnt Side {{
    dirichlet {{
        t = {temperature}
        {dirichlet}
    }}
}}
""".format(pressure=target.pressure, dirichlet=target.initialCond,
           temperature=target.temperature, phi=target.phi,
           simulation=target.simulation, fuel_is=target.fuel_is,
           oxidizer=target.oxidizer, g_r=global_reaction,
           s_p_loc=target.add["StartProfilesFile"],
           radiation_tag=target.add["ComputeWithRadiation"],
           thermodiffusion_tag=target.add["Thermodiffusion"])

    # ── Flf_a (burner-stabilised flame) ───────────────────────────────────
    elif "Flf_a" in target.target:
        if "cantera" in target.add["solver"]:
            instring = """#!/usr/bin/python
import cantera as ct, numpy as np, pandas as pd, os

def printSolution(df, species, criteria):
    for col in df:
        if species == col and "max" in criteria:
            return f"{{criteria}}({{species}})\\t{{max(df[col])}}"

p      = {pressure}
tburner = {temperature}
mdot   = {mdot}
reactants = {species_conc}
width  = {width_bsf} / 100
gas    = ct.Solution("{mech}")
gas.TPX = tburner, p, reactants
os.chdir("output")
f = ct.BurnerFlame(gas=gas, width=width)
f.burner.mdot = mdot
zloc, tvals = np.genfromtxt("{data_file}", delimiter=",", comments="#").T
zloc /= max(zloc)
f.set_refine_criteria(ratio={ratio}, slope={slope_bsf}, curve={curve})
f.flame.set_fixed_temp_profile(zloc, tvals)
f.energy_enabled = False
f.transport_model = "{transport_model}"
f.solve({solve_bsf})
try:
    f.write_hdf("burner_flame.h5", group="{group}", mode="w", description="{description}")
except ImportError:
    f.save("burner_flame.xml", "{group}", "{description}")
f.write_csv("burner_flame.csv", quiet=False)
df = pd.read_csv("burner_flame.csv")
string = printSolution(df, "X_{target}", "{criteria}")
open("result.dout", "w+").write(string)
""".format(mech=mech_file, temperature=target.burner_temp, pressure=target.pressure,
           species_conc=target.species_dict, width_bsf=target.add["flf_grid"],
           ratio=target.add["ratio"], slope_bsf=target.add["slope_bsf"],
           curve=target.add["curve"], loglevel=target.add["loglevel"],
           solve_bsf=target.add["solve_bsf"],
           transport_model=target.add["transport_model"],
           group=target.add["group"], description=target.add["description"],
           data_file=target.add["ExpTempFile"], mdot=target.add["flow_rate"],
           target=target.add["flf_target"], criteria=target.add["flf_cond"])
        else:
            # FlameMaster Flf_a
            _generate_start_profile(opt_dict, target, startProfile_location,
                                    file_specific_command)
            instring = """############
# Numerics #
############

UseNumericalJac is TRUE
UseSecondOrdJac is TRUE
UseModifiedNewton = TRUE
DampFlag = TRUE
LambdaMin = 1.0e-2
DeltaTStart = 1e-5
MaxIter = 5000
TolRes = 1.0e-15
TolDy = 1e-4

DeltaNewGrid = 25
OneSolutionOneGrid = TRUE
initialgridpoints = 89
maxgridpoints = 300
q = -0.25
R = 60

AdditionalOutput is TRUE
WriteEverySolution is TRUE
PrintMolarFractions is TRUE
OutputPath is ./output
StartProfilesFile is {s_p_loc}

MechanismFile is mechanism.pre
globalReaction is {g_r};

{fuel_is}
oxidizer is {oxidizer}

Flame is {simulation}
ExactBackward is TRUE
pressure = {pressure}
ComputeWithRadiation is {radiation_tag}
Thermodiffusion is {thermodiffusion_tag}
TransModel is MonoAtomic
ExpTempFile is {etp}
ConstLewisNumber is TRUE
ConstMassFlux is TRUE
MassFlowRate = {flow_rate}

Unburnt Side {{
    dirichlet {{
        t = {temperature}
        {dirichlet}
    }}
}}
""".format(pressure=target.pressure, oxidizer=target.oxidizer,
           dirichlet=target.initialCond, temperature=target.temperature,
           phi=target.phi, simulation=target.simulation, fuel_is=target.fuel_is,
           g_r=global_reaction, s_p_loc=target.add["StartProfilesFile"],
           thermodiffusion_tag=target.add["Thermodiffusion"],
           radiation_tag=target.add["ComputeWithRadiation"],
           flow_rate=target.add["flow_rate"], etp=target.add["ExpTempFile"])

    # ── Flw (flow reactor / JSR) ──────────────────────────────────────────
    elif "Flw" in target.target:
        if target.reactor in ("JSR", "PSR", "FlowReactor"):
            if "cantera" in target.add["solver"]:
                instring = ("""#!/usr/bin/python
import cantera as ct, numpy as np, matplotlib.pyplot as plt, pandas as pd, json, os
""" + _FLW_EXTRACT_HELPERS + """
T_0          = {temperature}
pressure     = {pressure}
composition_0 = {species_conc}
length        = {flowReactorLength}
area          = {crossSectionalArea}
reactorVolume = {reactorVolume}
reaction_mechanism = "{mech}"
residenceTime = {residence_time}
species       = "{species_in_investigation}"
n_steps       = {time_step}
t_total       = {total_time}
gas1 = ct.Solution(reaction_mechanism)
gas1.TPX = T_0, pressure, composition_0
r1   = ct.IdealGasConstPressureReactor(gas1)
mdot = r1.mass / residenceTime
sim1 = ct.ReactorNet([r1])
dt   = t_total / n_steps
t1   = (np.arange(n_steps) + 1) * dt
z1 = np.zeros_like(t1); u1 = np.zeros_like(t1)
states1 = ct.SolutionArray(r1.thermo)
for n1, t_i in enumerate(t1):
    sim1.advance(t_i)
    u1[n1] = mdot / area / r1.thermo.density
    z1[n1] = z1[n1-1] + u1[n1] * dt
    states1.append(r1.thermo.state)
os.chdir("output")
pd.DataFrame(t1).to_csv("states_time.csv")
pd.DataFrame(states1.X[:, gas1.species_index(species)], columns=[species]).to_csv("states_X.csv")
pd.DataFrame(states1.T, columns=["T"]).to_csv("states_T.csv")
time_range = np.asarray([t for t in t1 if t <= residenceTime])
X    = pd.DataFrame(states1.X[:, gas1.species_index(species)], columns=[species]).to_numpy().flatten()
xo   = load_species(species, composition_0)
t_half = {anchor_time}
add  = dict(method="{method}", range_={limits}, anchor={anchor}, unit="{unit}")
ni_t, ni_x, rate, dt_r = getRate(xo, t_half, X, time_range, add)
open("rate.csv","w").write("rate\\t{{}}\\tppm/ms".format(float(rate*1e6)))
open("time.csv","w").write("time\\t{{}}\\tms".format(dt_r))
os.chdir("..")
""").format(mech=mech_file, temperature=target.temperature, pressure=target.pressure,
                            species_conc=target.species_dict,
                            flowReactorLength=target.add["flw_length"],
                            flow_velocity=target.add["flow_velocity"],
                            crossSectionalArea=target.add["crossSectionalArea"],
                            reactorVolume=target.add["reactorVolume"],
                            residence_time=target.add["residenceTime"],
                            total_time=target.add["total_time"],
                            time_step=target.add["time_step"],
                            anchor_time=target.add["anchor_time"],
                            flow_rate=target.flow_rate,
                            species_in_investigation=target.add["flw_species"],
                            method=target.add["flw_method"],
                            limits=target.add["flw_limits"],
                            anchor=target.add["anchor"],
                            unit=target.add["limit_units"])

            else:
                # FlameMaster Flw
                instring = """############
# Numerics #
############

RelTol = 1.0e-10
AbsTol = 1.0e-12
TStart = 0.0
TEnd = {tend}
TRes = {tres}

WriteEverySolution is TRUE
PrintMolarFractions is TRUE
SensObjALL is TRUE
OutputPath is ./output
NOutputs = 1000

MechanismFile is mechanism.pre
globalReaction is {g_r};

{fuel_is}
oxidizer is O2

Flame is {simulation}
{isotherm}
{heat}
AmbientTemp is {T_amb}
ExactBackward is TRUE
Pressure = {pressure}

InitialCond {{
    t = {temperature}
{init_condition}
}}
""".format(tend=target.add["total_time"], tres=target.add["residenceTime"],
           pressure=target.pressure, init_condition=target.initialCond,
           temperature=target.temperature, phi=target.phi,
           simulation=target.simulation, fuel_is=target.fuel_is,
           g_r=global_reaction, heat=target.add["heat"],
           T_amb=target.add["T_amb"], isotherm=target.add["isIsotherm"])

                extract = ("""#!/usr/bin/python
import numpy as np, os, pandas as pd
""" + _FLW_EXTRACT_HELPERS + """
flw_target    = "{flw_target}"
anchor        = {anchor}
limit         = {limit}
flw_method    = "{flw_method}"
residenceTime = {residenceTime}
anchor_time   = {anchor_time}
limit_units   = "{limit_units}"
initial_species_conc = {xo}

os.chdir("output")
for fname in os.listdir():
    if fname.startswith("X"):
        outfile = fname; break
else:
    raise FileNotFoundError("No X* output file found")

data = [line.strip("\\n") for line in open(outfile).readlines() if "*" not in line]
open("modified.csv","w+").write("\\n".join(data))
df = pd.read_csv("modified.csv", sep="\\t")

for col in df:
    if "X-"+flw_target+" " in col:
        profile = df[col]
    if col.strip().startswith("t"):
        time = df[col] / 1000

time_range = np.asarray([t for t in time if t <= residenceTime])
add = dict(method=flw_method, range_=(40,60), anchor=50, unit=limit_units)
ni_t, ni_x, rate, dt = getRate(initial_species_conc, anchor_time, profile.to_numpy(), time_range, add)
open("rate.csv","w").write("rate\\t{{}}\\tppm/ms".format(float(rate*1e6)))
open("time.csv","w").write("time\\t{{}}\\tms".format(dt))
""").format(flw_target=target.add["flw_species"], anchor=target.add["anchor"],
                           limit=target.add["flw_limits"],
                           flw_method=target.add["flw_method"],
                           residenceTime=target.add["residenceTime"],
                           anchor_time=target.add["anchor_time"],
                           limit_units=target.add["limit_units"],
                           xo=target.species_dict[str(target.add["flw_species"])])
    else:
        raise AssertionError(f"Unknown experiment type: {target.target!r}")

    # ════════════════════════════════════════════════════════════════════
    # Build conversion, run, and extract scripts based on solver
    # ════════════════════════════════════════════════════════════════════
    solver = target.add.get("solver", "FlameMaster")

    if solver == "cantera":
        s_convert = (
            "#!/bin/bash\n"
            "ck2yaml --input=mechanism.mech --thermo=thermo.therm "
            "--transport=transport.trans &> out\n"
        )
        s_run = "#!/bin/bash\npython3.9 cantera_.py &> solve\n"
        extract = ""

    elif solver == "CHEMKIN_PRO":
        s_convert = (
            "#!/bin/bash\n"
            "ck2yaml --input=mechanism.mech --thermo=thermo.therm "
            "--transport=transport.trans &> out\n"
        )
        bin_path  = opt_dict["Bin"]["bin"]
        s_run = (
            "#!/bin/bash\n"
            f"export NUMEXPR_MAX_THREADS=1\n"
            f"python3.9 {bin_path}/soln2ck_2.py --mechanism=mechanism.inp "
            f"--thermo=thermo.dat {mech_file} &> soln2ck.out\n"
            "python3.9 cantera_.py &> solve\n"
            "find ./output -type f ! -name 'RCM.out' ! -name 'tau.out' "
            "-delete &> delete.log\n"
            "find . -type f \\( -name 'time_history.csv' -o -name 'ScanMan.log' \\) "
            "-delete &> delete2.log\n"
        )
        extract = ""

    else:
        # FlameMaster
        bin_dir = opt_dict["Bin"]["solver_bin"]
        muqsac  = opt_dict["Bin"]["bin"]
        s_convert = (
            "#!/bin/bash\n"
            f"python3.9 {muqsac}/soln2ck_FM.py "
            f"--mechanism=mechanism.inp --thermo=therm.dat --transport=trans.dat "
            f"{mech_file} &> soln2ck.log\n"
            f"{bin_dir}/ScanMan -i mechanism.inp -t {thermo_file_location} "
            f"-m {trans_file_location} {file_specific_command} -3sr -N 0.05 -E "
            f"-o mechanism.pre &> ScanMan.log\n"
        )
        s_run = (
            "#!/bin/bash\n"
            f"{bin_dir}/FlameMan &> Flame.log && python3.9 extract.py &> extract.log\n"
            "find ./output -type f ! -name 'modified.csv' ! -name 'tau.out' "
            "-delete &> delete.log\n"
            "find . -type f \\( -name 'mechanism.pre' -o -name 'ScanMan.log' \\) "
            "-delete &> delete2.log\n"
        )

    return instring, s_convert, s_run, extract
