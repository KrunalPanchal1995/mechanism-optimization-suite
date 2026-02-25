import os
import sys
import yaml
import numpy as np
import pandas as pd
from collections import defaultdict

# ----------------------------
# Helpers
# ----------------------------
def filter_list(lst):
    return [x for x in lst if x != ""]

def read_header_yaml(header_file: str) -> dict:
    with open(header_file, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh.read()) or {}

def read_csv_clean(csv_file: str) -> pd.DataFrame:
    """
    Reads CSV robustly:
    - ignores comment lines starting with '#'
    - trims/normalizes column names (removes quotes/spaces)
    """
    df = pd.read_csv(csv_file, sep=",", comment="#", engine="python")

    def norm_col(c):
        c = str(c).strip()
        if len(c) >= 2 and ((c[0] == c[-1] == '"') or (c[0] == c[-1] == "'")):
            c = c[1:-1]
        return c.strip()

    df.columns = [norm_col(c) for c in df.columns]
    return df

def require_columns(df: pd.DataFrame, required: list, where: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing columns {missing} in {where}. "
            f"Detected columns: {list(df.columns)}"
        )

def normalize_species_list(value):
    """
    Normalize header fields that can be:
      - ["N2","AR"]
      - {"N2":0.78,"AR":0.22}
      - [{"N2":0.78},{"AR":0.22}]
      - "N2"
    into a list of species strings.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return list(value.keys())
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                out.extend(list(item.keys()))
        return out
    return []

def get_move_number_from_fen(fen: str):
    try:
        return int(fen.split()[5])
    except Exception:
        return None

def get_last_move_color_from_fen(fen: str):
    try:
        return "black" if fen.split()[1] == "w" else "white"
    except Exception:
        return None

# ----------------------------
# Entry: input directory
# ----------------------------
if len(sys.argv) > 1:
    os.chdir(sys.argv[1])
    print("Input directory found\n")
else:
    print(
        "Please enter a valid input directory as argument.\n"
        "For details of preparing the input file, please see the UserManual.\n\n"
        "Program exiting."
    )
    sys.exit(1)

path = os.getcwd()
input_files = [f for f in os.listdir(path) if f.lower().endswith(".csv")]
input_headers = [f for f in os.listdir(path) if f.lower().endswith(".header")]

os.makedirs("output", exist_ok=True)

# ----------------------------
# Main loop
# ----------------------------
for csv_file in input_files:
    base = csv_file.rsplit(".", 1)[0]
    header_name = f"{base}.header"
    if header_name not in input_headers:
        continue

    head = read_header_yaml(header_name)
    data = read_csv_clean(csv_file)

    ds = head.get("Data_structure", [])
    if not ds or not isinstance(ds, (list, tuple)):
        raise ValueError(f"{header_name}: 'Data_structure' missing/invalid.")

    require_columns(data, [ds[0]], csv_file)
    n = len(data[ds[0]])

    fuel_type = [head.get("Fuel_type")] * n
    oxidizer = [head.get("Oxidizer")] * n
    target = head.get("Target")

    # ---- Phi ----
    if "phi" in ds and "phi" in data.columns:
        phi = data["phi"]
    elif "Phi" in ds and "Phi" in data.columns:
        phi = data["Phi"]
    elif "Phi" in head:
        phi = np.ones(n) * float(head["Phi"])
    else:
        phi = ["N/A"] * n

    # ---- res ----
    res = None
    if "res" in ds and "res" in data.columns:
        res = data["res"]

    # ---- Pressure ----
    pressure_i = None
    if "Pi" in ds and "Pi" in data.columns:
        pressure_i = data["Pi"]

    if "P" in ds and "P" in data.columns:
        pressure = data["P"]
    else:
        pressure = np.ones(n) * float(head["Pressure"])

    # ---- Temperature ----
    temperature_i = None
    if "Ti" in ds and "Ti" in data.columns:
        temperature_i = data["Ti"]

    if "T" in ds and "T" in data.columns:
        temperature = data["T"]
    else:
        temperature = np.ones(n) * float(head["Temperature"])

    # ---- Dataset index ----
    if "Sr" in ds:
        require_columns(data, ["Sr"], csv_file)
        serial_id = data["Sr"]
        dataSetIndex = [f"{head.get('DataSet','DATASET')}_{x}" for x in serial_id]
    else:
        dataSetIndex = [head.get("DataSet", "DATASET")] * n

    # ---- Fuel ----
    fuel = [head.get("Fuel")] * n
    if fuel_type and isinstance(fuel_type[0], str) and "Multi" in fuel_type[0]:
        if "a" in ds:
            fuel_x = []
            for row_i in range(n):
                temp = {}
                fuel_species = normalize_species_list(fuel[0])
                for spec in fuel_species:
                    if spec not in data.columns:
                        raise KeyError(f"{csv_file}: missing fuel species column '{spec}'")
                    temp[spec] = data.loc[row_i, spec]
                fuel_x.append(temp)
        else:
            fuel_x = np.ones(n) * float(head["Fuel_x"])
    else:
        if "Fuel" in ds and "Fuel" in data.columns:
            fuel_x = data["Fuel"]
        else:
            fuel_x = np.ones(n) * float(head["Fuel_x"])

    # ---- Oxidizer fraction ----
    if "Ox" in ds and "Ox" in data.columns:
        oxidizer_x = data["Ox"]
    else:
        oxidizer_x = np.ones(n) * float(head["Oxidizer_x"])

    # ---- Bath gas fractions ----
    if "bg1" in ds:
        bath_gas_species = normalize_species_list(head.get("Bath_gas"))
        bath_gas_x = []
        for row_i in range(n):
            temp = {}
            for spec in bath_gas_species:
                if spec not in data.columns:
                    raise KeyError(
                        f"{csv_file}: missing bath gas column '{spec}'. "
                        f"Detected columns: {list(data.columns)}"
                    )
                temp[spec] = data.loc[row_i, spec]
            bath_gas_x.append(temp)
        bath_gas_out = [bath_gas_species] * n
    else:
        bath_gas_x = [head.get("Bath_gas_x")] * n
        bath_gas_out = [head.get("Bath_gas")] * n

    # ---- target value ----
    if target in ds and target in data.columns:
        target_value = data[target]
    else:
        target_value = np.ones(n) * float(head[target])

    # ---- sim type ----
    if "Sim" in ds and "Sim" in data.columns:
        sim_type = data["Sim"]
    else:
        sim_type = [head.get("Simulation_type")] * n

    # ---- measurement type ----
    if "Measure_type" in ds and "Measure_type" in data.columns:
        Measure_type = data["Measure_type"]
    else:
        Measure_type = [head.get("Measurnment_type")] * n

    def get_series_or_head(key_ds, key_head, default="N/A", cast_float=False):
        if key_ds in ds and key_ds in data.columns:
            return data[key_ds]
        if key_head in head:
            return np.ones(n) * float(head[key_head]) if cast_float else [head[key_head]] * n
        return [default] * n

    flame_type = get_series_or_head("flame_type", "Flame_type", "N/A")
    reactor_type = get_series_or_head("reactor_type", "Reactor_type", "N/A")
    ign_type = get_series_or_head("ign_mode", "Ignition_mode", "N/A")
    start_pro = get_series_or_head("start_pro", "startprofile", "N/A")
    flow_rate = get_series_or_head("flow_rate", "Flow_rate", "N/A", cast_float=True)

    # ---- uncertainty ----
    if "sigma" in ds and "sigma" in data.columns:
        sigma = data["sigma"]
        unsrt_cfg = head.get("Unsrt", {})
        unsrt_type = (unsrt_cfg.get("type") if isinstance(unsrt_cfg, dict) else "absolute") or "absolute"
        unsrt = []
        for row_i in range(n):
            s = float(sigma.iloc[row_i])
            if unsrt_type == "absolute":
                unsrt.append(s)
            else:
                unsrt.append(float(target_value.iloc[row_i]) * s)
    else:
        unsrt = [head.get("Unsrt")] * n

    # ---- units ----
    if "obs_unit" in ds and "obs_unit" in data.columns:
        obs_unit = data["obs_unit"]
        units = [head.get("Unit")] * n
    else:
        obs_unit = [""] * n
        units = [head.get("Unit")] * n

    # ---- Build output ----
    string = ""
    weight = n
    target_out = [target] * n

    for row_i in range(n):
        if "Tig" in str(target):
            string += (
                f"{row_i+1}\t|{dataSetIndex[row_i]}"
                f"\t| target -- {target_out[row_i]}"
                f"\t| simulation -- {sim_type[row_i]}"
                f"\t| measurnment_type -- {Measure_type[row_i]}"
                f"\t|Ignition_mode -- {ign_type[row_i]}"
                f"\t|Flame_type -- {flame_type[row_i]}"
                f"\t|Reactor_type -- {reactor_type[row_i]}"
                f"\t|Fuel_type -- {fuel_type[row_i]}"
                f"\t| Fuel -- x->{fuel[row_i]} = {fuel_x[row_i]}"
                f"\t| Oxidizer -- x->{oxidizer[row_i]} = {oxidizer_x[row_i]}"
                f"\t| Bath_gas -- x->{bath_gas_out[row_i]} = {bath_gas_x[row_i]}"
                f"\t|T -- {temperature[row_i]}"
                f"\t| P -- {pressure[row_i]}"
                f"\t| flow_rate -- {flow_rate[row_i]}"
                f"\t|Phi -- {phi[row_i]}"
                f"\t| observed -- {target_value[row_i]}"
                f"\t|obs_unit -- {obs_unit[row_i]}"
                f"\t| deviation -- {unsrt[row_i]}"
                f"\t|data_weight -- {weight}"
                f"\t|units -- {units[row_i]}\n"
            )
        elif "RCM" in str(target):
            string += (
                f"{row_i+1}\t|{dataSetIndex[row_i]}"
                f"\t| target -- {target_out[row_i]}"
                f"\t| simulation -- {sim_type[row_i]}"
                f"\t| measurnment_type -- {Measure_type[row_i]}"
                f"\t|Ignition_mode -- {ign_type[row_i]}"
                f"\t|Flame_type -- {flame_type[row_i]}"
                f"\t|Reactor_type -- {reactor_type[row_i]}"
                f"\t|Fuel_type -- {fuel_type[row_i]}"
                f"\t| Fuel -- x->{fuel[row_i]} = {fuel_x[row_i]}"
                f"\t| Oxidizer -- x->{oxidizer[row_i]} = {oxidizer_x[row_i]}"
                f"\t| Bath_gas -- x->{bath_gas_out[row_i]} = {bath_gas_x[row_i]}"
                f"\t| Ti -- {temperature_i[row_i] if temperature_i is not None else 'N/A'}"
                f"\t| T -- {temperature[row_i]}"
                f"\t| Pi -- {pressure_i[row_i] if pressure_i is not None else 'N/A'}"
                f"\t| P -- {pressure[row_i]}"
                f"\t| flow_rate -- {flow_rate[row_i]}"
                f"\t|Phi -- {phi[row_i]}"
                f"\t| observed -- {target_value[row_i]}"
                f"\t|obs_unit -- {obs_unit[row_i]}"
                f"\t| deviation -- {unsrt[row_i]}"
                f"\t|data_weight -- {weight}"
                f"\t|units -- {units[row_i]}\n"
            )
        elif "Fls" in str(target):
            string += (
                f"{row_i+1}\t|{dataSetIndex[row_i]}"
                f"\t| target -- {target_out[row_i]}"
                f"\t| simulation -- {sim_type[row_i]}"
                f"\t| measurnment_type -- {Measure_type[row_i]}"
                f"\t|Ignition_mode -- {ign_type[row_i]}"
                f"\t|Flame_type -- {flame_type[row_i]}"
                f"\t|Reactor_type -- {reactor_type[row_i]}"
                f"\t|Fuel_type -- {fuel_type[row_i]}"
                f"\t| Fuel -- x->{fuel[row_i]} = {fuel_x[row_i]}"
                f"\t| Oxidizer -- x->{oxidizer[row_i]} = {oxidizer_x[row_i]}"
                f"\t| Bath_gas -- x->{bath_gas_out[row_i]} = {bath_gas_x[row_i]}"
                f"\t|T -- {temperature[row_i]}"
                f"\t| P -- {pressure[row_i]}"
                f"\t| flow_rate -- {flow_rate[row_i]}"
                f"\t|Phi -- {phi[row_i]}"
                f"\t| observed -- {target_value[row_i]}"
                f"\t|obs_unit -- {obs_unit[row_i]}"
                f"\t| deviation -- {unsrt[row_i]}"
                f"\t|data_weight -- {weight}"
                f"\t|units -- {units[row_i]}\n"
            )
        else:
            string += (
                f"{row_i+1}\t|{dataSetIndex[row_i]}"
                f"\t| target -- {target_out[row_i]}"
                f"\t| simulation -- {sim_type[row_i]}"
                f"\t| measurnment_type -- {Measure_type[row_i]}"
                f"\t|Ignition_mode -- {ign_type[row_i]}"
                f"\t|Flame_type -- {flame_type[row_i]}"
                f"\t|Reactor_type -- {reactor_type[row_i]}"
                f"\t|Fuel_type -- {fuel_type[row_i]}"
                f"\t| Fuel -- x->{fuel[row_i]} = {fuel_x[row_i]}"
                f"\t| Oxidizer -- x->{oxidizer[row_i]} = {oxidizer_x[row_i]}"
                f"\t| Bath_gas -- x->{bath_gas_out[row_i]} = {bath_gas_x[row_i]}"
                f"\t|T -- {temperature[row_i]}"
                f"\t| P -- {pressure[row_i]}"
                f"\t| flow_rate -- {flow_rate[row_i]}"
                f"\t|Phi -- {phi[row_i]}"
                f"\t| res -- {res[row_i] if res is not None else 'N/A'}"
                f"\t| observed -- {target_value[row_i]}"
                f"\t|obs_unit -- {obs_unit[row_i]}"
                f"\t| deviation -- {unsrt[row_i]}"
                f"\t|data_weight -- {weight}"
                f"\t|units -- {units[row_i]}\n"
            )

    out_path = os.path.join("output", f"{base}.out")
    with open(out_path, "w", encoding="utf-8") as outFile:
        outFile.write(string)

print("Done. Outputs are in ./output/")