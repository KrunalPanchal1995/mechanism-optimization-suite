========================================================
XLX2CSV Converter Documentation (Python-commented format)
========================================================

This folder contains sample files for converting Excel/CSV-based datasets into
standardized target `.out` files.

Supported dataset examples in this folder:
    - Ignition delay files
    - Flame speed files
    - JSR simulation files
    - RCM simulations


How to Run
----------

From terminal / PowerShell:

    python xlx2csv.py <input_directory>

Example:

    python xlx2csv.py Tig/


What the script does (high level):
    1) Changes into the given input directory
    2) Finds all `.csv` files
    3) For each `.csv`, searches for a matching `.header` file with the same base name
    4) Converts each CSV row into a standardized text record
    5) Writes an output file to:
           <input_directory>/output/<csv_name_without_ext>.out


------------------------------------------------------------------------------
1. Purpose of this Tool
------------------------------------------------------------------------------

`xlx2csv.py` is a data-format conversion utility designed for combustion-kinetics
datasets.

It converts experimental or simulated data tables (stored as CSV files) into a
standardized `.out` format that can be used for:
    - mechanism optimization
    - uncertainty analysis
    - surrogate modeling
    - validation against ignition delay / flame speed targets
    - automated workflows (e.g., Cantera / Chemkin based solvers)

Key idea:
    The converter DOES NOT assume a fixed dataset structure.
    The user defines the dataset structure explicitly using a YAML `.header` file.

This enables:
    - maximum flexibility
    - heterogeneous datasets
    - mixed constant + per-row variables
    - multi-fuel / multi-bath-gas cases


------------------------------------------------------------------------------
2. High-Level Concept
------------------------------------------------------------------------------

Each dataset is defined by TWO files:

    1) *.csv
       - tabular numerical data
       - one row = one experiment or one simulation condition

    2) *.header
       - metadata + rules that describe how to interpret the CSV
       - written in YAML format

Conversion workflow:
    - The converter reads both files.
    - For each row in the CSV, it uses the header rules to interpret columns,
      fill constants, compute derived values, and finally write a standardized
      `.out` line.

Result:
    - A human- and machine-readable `.out` file suitable for downstream tools.


------------------------------------------------------------------------------
3. Folder Structure
------------------------------------------------------------------------------

All input files must be placed in ONE directory.

Example layout:

    XLX2CSV/
    ├── xlx2csv.py
    └── Tig/
        ├── dataset1.csv
        ├── dataset1.header
        ├── dataset2.csv
        └── dataset2.header

The script will automatically create:

    Tig/output/


------------------------------------------------------------------------------
4. File Naming Rules (Strict)
------------------------------------------------------------------------------

Each CSV must have a corresponding header file with the SAME base name:

    example.csv
    example.header

Strict requirements:
    - Column names must match EXACTLY what is referenced in `Data_structure`
    - CSV lines beginning with `#` are ignored (treated as comments)


------------------------------------------------------------------------------
5. CSV File Format
------------------------------------------------------------------------------

Rules:
    - Comma-separated
    - Must include a header row (column names)
    - Column names must match what you declare in the header's `Data_structure`

Example CSV:

    #Data_structure:
    Sr,a,Ox,bg1,T,P,Tig,sigma
    ing_Ethyl_Butanoate,0.0313,0.2035,0.7652,684.32,30.31,120978,60489
    ing_Ethyl_Butanoate,0.0313,0.2035,0.7652,698.71,30.24,85520,42760

Interpretation:
    - Each row represents one experiment/simulation condition.


------------------------------------------------------------------------------
6. Header File Format (*.header) - YAML
------------------------------------------------------------------------------

The `.header` file is written in YAML and may include:

    - dataset metadata
    - constant values (used when not present in CSV)
    - variable definitions (mapped through `Data_structure`)
    - rules for uncertainty interpretation
    - units metadata

Full example header (Shock-Tube Ignition Delay):

    #######################################
    ##   Shock-tube Ignition Delay       ##
    #######################################

    DataSet: ign_BUTANOATE_phi_1

    Target: Tig

    Fuel_type: Multi
    Fuel: {a: MB-C5H10O2}

    Oxidizer: O2

    Bath_gas: {bg1: AR}

    Simulation_type: Isochor Homo Reactor
    Measurnment_type: timeDelay

    Ignition_mode: reflected

    Unsrt: {type: absolute}

    Unit: {conc: mol, P: bar, T: K, observed: us}

    Data_structure:
     - Sr
     - a
     - Ox
     - bg1
     - T
     - P
     - Tig
     - sigma


------------------------------------------------------------------------------
7. Explanation of Each Header Field
------------------------------------------------------------------------------

7.1 DataSet
-----------
A dataset identifier string.

If `Sr` appears in `Data_structure`, the final dataset index becomes:

    <DataSet>_<Sr_value>

Example:
    DataSet: ign_BUTANOATE_phi_1
    Sr row value: ing_Ethyl_Butanoate

Final dataset index:
    ign_BUTANOATE_phi_1_ing_Ethyl_Butanoate


7.2 Target
----------
The observable being modeled or measured.

Common examples:
    - Tig  -> ignition delay
    - Fls  -> laminar flame speed
    - RCM  -> rapid compression machine ignition

Rule:
    If `Target` appears in `Data_structure`, then the CSV MUST contain a column
    with that name.


7.3 Fuel_type
-------------
Controls how fuel composition is interpreted.

Typical values:
    - Multi: fuel composition depends on CSV columns (variable fuel composition)
    - anything else: single-fuel case (often constant composition)


7.4 Fuel
--------
Defines how CSV variables map to fuel species.

Example:
    Fuel: {a: MB-C5H10O2}

Meaning:
    - column `a` exists in CSV
    - its numeric value is the mole fraction of MB-C5H10O2

Multiple fuels example:
    Fuel: {a: CH4, b: C2H6}


7.5 Oxidizer
------------
Oxidizer species name.

Rule:
    - If "Ox" is in Data_structure, the CSV column `Ox` supplies the oxidizer mole fraction
    - Otherwise, provide a constant oxidizer fraction in the header:

        Oxidizer_x: 0.21


7.6 Bath_gas
------------
Maps CSV columns to bath gas species.

Example:
    Bath_gas: {bg1: AR}

Meaning:
    - column `bg1` exists in CSV
    - its value = mole fraction of AR

Multiple bath gases example:
    Bath_gas: {bg1: N2, bg2: AR}

Constant bath gas example (if not using CSV columns):
    Bath_gas_x: {N2: 0.79, AR: 0.21}


7.7 Simulation_type / Measurnment_type / Ignition_mode
------------------------------------------------------
These are metadata fields used for labeling and interpretation.
They are typically replicated for every row in output.


7.8 Unsrt (Uncertainty)
-----------------------
Controls how `sigma` is interpreted.

Example:
    Unsrt: {type: absolute}

Options:
    - absolute: sigma is already absolute
    - otherwise: sigma is treated as relative (fraction) of observed value

Example:
    observed = 1000
    sigma    = 0.1
    => absolute uncertainty = 100


7.9 Unit
--------
Human-readable units metadata used for output labeling.

Example:
    Unit: {conc: mol, P: bar, T: K, observed: us}


------------------------------------------------------------------------------
8. Data_structure (Most Important Section)
------------------------------------------------------------------------------

`Data_structure` defines the contract between header and CSV.

Rule:
    If a variable appears in `Data_structure`, it MUST exist as a CSV column.

Anything NOT listed in Data_structure is assumed constant and must be provided
in the header.

Example:
    Data_structure:
      - Sr
      - a
      - Ox
      - bg1
      - T
      - P
      - Tig
      - sigma

Therefore, CSV must contain columns:
    Sr, a, Ox, bg1, T, P, Tig, sigma


Constants example:
    If P is NOT in Data_structure:
        Pressure: 30

    If T is NOT in Data_structure:
        Temperature: 700

    If Phi is NOT in Data_structure:
        Phi: 1.0


------------------------------------------------------------------------------
9. Output Format
------------------------------------------------------------------------------

The script writes:

    output/<csv_name>.out

Each line corresponds to one CSV row and includes fields such as:
    - dataset index
    - target
    - simulation & measurement type
    - fuel / oxidizer / bath gas composition
    - T, P, Phi
    - observed value
    - uncertainty
    - units

This `.out` format is intended for:
    - direct inspection
    - downstream optimization codes
    - reproducibility and archiving


