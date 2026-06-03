#!/usr/bin/env python3
"""
modules/xml_generator.py

MUQ uncertainty XML generator (legacy Data_Extraction.py format).

Generates, from a classification DataFrame:
    HTC_all.xml, HTC_factor.xml, LTC_all.xml, LTC_factor.xml

CHANGES vs original:
- generate_xml_files() is the entry point the driver calls (the old
  driver imported this name but the module only defined generate_xml()).
- RxnCount default is now 2 *interconnected* rows, so int(rxn_count)-2
  no longer silently produces 0 for plain PLOG reactions. If the
  classification file supplies RxnCount it is used verbatim; otherwise
  PLOG/PLOG-Duplicate default to the High+Low pair (count = 2).
  >>> If your legacy MUQ reader expects a different RxnCount semantics,
      set it explicitly in the classification CSV. <<<
- f value read from canonical 'f' column; data_type defaulting handled
  by interfaces.load_classification_file before this runs.
"""

import numpy as np
import pandas as pd

from .interfaces import DEFAULT_DATA_TYPE, normalize_group


def xml_escape_reaction(rxn):
    rxn = str(rxn)
    rxn = rxn.replace("=", "&#61;")
    rxn = rxn.replace("<", "&#60;")
    rxn = rxn.replace(">", "&#62;")
    rxn = rxn.replace(" ", "&#032;")
    return rxn


def build_unsrt_string(f_value, data_type=DEFAULT_DATA_TYPE):
    ln_f = np.log(float(f_value))
    if data_type == "constant;end_points":
        unsrt = f"{ln_f},{ln_f}"
    else:
        unsrt = str(ln_f)
    return unsrt, "300,2500"


# ------------------------------------------------------------------
# Per-class block builders
# ------------------------------------------------------------------

def getElementaryRxn(rxn, rxn_id, f_value,
                     perturbation_type="all", data_type=DEFAULT_DATA_TYPE):
    unsrt, temp = build_unsrt_string(f_value, data_type)
    rxn = xml_escape_reaction(rxn)
    return f"""
\t<reaction rxn = '{rxn}' no = 'R{rxn_id}'>
\t <class>Elementary</class>
\t\t<type>pressure_independent</type>
\t\t<sub_type name = "forward">
\t\t\t<multiple> False </multiple>
\t\t\t<branches>N/A</branches>
\t\t\t<pressure_limit>N/A</pressure_limit>
\t\t\t<common_temp>N/A</common_temp>
\t\t\t<temp_limit>N/A</temp_limit>
\t\t</sub_type>
\t\t<perturbation_type>{perturbation_type}</perturbation_type>
\t\t<data_type>{data_type}</data_type>
\t\t<unsrt>{unsrt}</unsrt>
\t\t<temp>{temp}</temp>
\t\t<file></file>
\t</reaction>
"""


def getDuplicateRxn(rxn, rxn_id, f_value,
                    perturbation_type="all", data_type=DEFAULT_DATA_TYPE):
    unsrt, temp = build_unsrt_string(f_value, data_type)
    rxn = xml_escape_reaction(rxn)
    blocks = ""
    for branch in ("A", "B"):
        suffix = "a" if branch == "A" else "b"
        blocks += f"""
\t<reaction rxn = '{rxn}' no = 'R{rxn_id}{suffix}'>
\t <class>Duplicate</class>
\t\t<type>pressure_independent</type>
\t\t<sub_type name = "duplicate">
\t\t\t<multiple> True </multiple>
\t\t\t<branches>{branch}</branches>
\t\t</sub_type>
\t\t<perturbation_type>{perturbation_type}</perturbation_type>
\t\t<data_type>{data_type}</data_type>
\t\t<unsrt>{unsrt}</unsrt>
\t\t<temp>{temp}</temp>
\t\t<file></file>
\t</reaction>
"""
    return blocks


def getPLOGRxn(rxn, rxn_id, f_value, rxn_count,
               perturbation_type="all", data_type=DEFAULT_DATA_TYPE):
    unsrt, temp = build_unsrt_string(f_value, data_type)
    rxn = xml_escape_reaction(rxn)
    return f"""
\t<PLOG rxn = '{rxn}' no = 'R{rxn_id}:High'>
\t <class>PLOG</class>
\t\t<type>pressure_dependent</type>
\t\t<sub_type name = "forward">
\t\t\t<multiple> True </multiple>
\t\t\t<branches></branches>
\t\t\t<pressure_limit>High</pressure_limit>
\t\t</sub_type>
\t\t<perturbation_type>{perturbation_type}</perturbation_type>
\t\t<data_type>{data_type}</data_type>
\t\t<unsrt>{unsrt}</unsrt>
\t\t<temp>{temp}</temp>
\t\t<file></file>
\t</PLOG>

\t<PLOG rxn = '{rxn}' no = 'R{rxn_id}:Low'>
\t <class>PLOG</class>
\t\t<type>pressure_dependent</type>
\t\t<sub_type name = "forward">
\t\t\t<multiple> True </multiple>
\t\t\t<branches></branches>
\t\t\t<pressure_limit>Low</pressure_limit>
\t\t</sub_type>
\t\t<perturbation_type>{perturbation_type}</perturbation_type>
\t\t<data_type>{data_type}</data_type>
\t\t<unsrt>{unsrt}</unsrt>
\t\t<temp>{temp}</temp>
\t\t<file></file>
\t</PLOG>

\t<PLOG-Interconnectedness>
\t  <class>PLOG</class>
\t  <InterConnectedRxns>
\t     R{rxn_id}:High,R{rxn_id}:Low
\t  </InterConnectedRxns>
\t  <RxnCount>{int(rxn_count)}</RxnCount>
\t  <unsrtQuantType>Interpolation</unsrtQuantType>
\t</PLOG-Interconnectedness>
"""


def getPLOG_DuplicateRxn(rxn, rxn_id, f_value, rxn_count,
                         perturbation_type="all",
                         data_type=DEFAULT_DATA_TYPE):
    unsrt, temp = build_unsrt_string(f_value, data_type)
    rxn = xml_escape_reaction(rxn)
    txt = ""
    for branch in ("a", "b"):
        B = branch.upper()
        txt += f"""
\t<PLOG rxn = '{rxn}' no = 'R{rxn_id}{branch}:High'>
\t <class>PLOG-Duplicate</class>
\t\t<type>pressure_dependent</type>
\t\t<sub_type name = "forward">
\t\t\t<multiple>True</multiple>
\t\t\t<branches>{B}</branches>
\t\t\t<pressure_limit>High</pressure_limit>
\t\t</sub_type>
\t\t<perturbation_type>{perturbation_type}</perturbation_type>
\t\t<data_type>{data_type}</data_type>
\t\t<unsrt>{unsrt}</unsrt>
\t\t<temp>{temp}</temp>
\t</PLOG>

\t<PLOG rxn = '{rxn}' no = 'R{rxn_id}{branch}:Low'>
\t <class>PLOG-Duplicate</class>
\t\t<type>pressure_dependent</type>
\t\t<sub_type name = "forward">
\t\t\t<multiple>True</multiple>
\t\t\t<branches>{B}</branches>
\t\t\t<pressure_limit>Low</pressure_limit>
\t\t</sub_type>
\t\t<perturbation_type>{perturbation_type}</perturbation_type>
\t\t<data_type>{data_type}</data_type>
\t\t<unsrt>{unsrt}</unsrt>
\t\t<temp>{temp}</temp>
\t</PLOG>

\t<PLOG-Interconnectedness>
\t  <class>PLOG-Duplicate</class>
\t  <InterConnectedRxns>
\t     R{rxn_id}{branch}:High,R{rxn_id}{branch}:Low
\t  </InterConnectedRxns>
\t  <RxnCount>{int(rxn_count)}</RxnCount>
\t  <unsrtQuantType>Interpolation</unsrtQuantType>
\t</PLOG-Interconnectedness>
"""
    return txt


# ------------------------------------------------------------------
# Writers
# ------------------------------------------------------------------

def _reaction_block(row, perturbation_type):
    rxn_id = int(row["Reaction_ID"])
    rxn = row["Reaction"]
    rxn_type = row.get("Reaction_Type", "Elementary")
    f_value = row["f"]
    data_type = row.get("data_type", DEFAULT_DATA_TYPE)
    if pd.isna(data_type):
        data_type = DEFAULT_DATA_TYPE

    # RxnCount here is the TOTAL number of pressure points (rate-constants)
    # read from the mechanism. The interconnectedness count is total - 2,
    # because the top rate (k_inf / High limit) and the bottom rate
    # (Low limit) are counted separately. Falls back to 2 -> 0 extra rows
    # if the mechanism count is unavailable.
    total_rates = row.get("RxnCount", 2)
    if pd.isna(total_rates) or total_rates is None:
        total_rates = 2
    interconnected = max(int(total_rates) - 2, 0)

    if rxn_type in ("Elementary", "ThirdBody", "Falloff"):
        return getElementaryRxn(rxn, rxn_id, f_value,
                                perturbation_type, data_type)
    if rxn_type == "Duplicate":
        return getDuplicateRxn(rxn, rxn_id, f_value,
                               perturbation_type, data_type)
    if rxn_type == "PLOG":
        return getPLOGRxn(rxn, rxn_id, f_value, interconnected,
                          perturbation_type, data_type)
    if rxn_type == "PLOG-Duplicate":
        return getPLOG_DuplicateRxn(rxn, rxn_id, f_value, interconnected,
                                    perturbation_type, data_type)
    # unknown type: treat as elementary but keep going
    return getElementaryRxn(rxn, rxn_id, f_value,
                            perturbation_type, data_type)


def generate_xml(df, output_xml, perturbation_type="all"):
    """Write a single XML file from a DataFrame of reactions."""
    xml_text = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<uncertainty>\n'
    )
    for _, row in df.iterrows():
        xml_text += _reaction_block(row, perturbation_type)
    xml_text += "\n</uncertainty>\n"
    with open(output_xml, "w") as f:
        f.write(xml_text)
    print(f"[XML] Written -> {output_xml}")


def _enrich_and_dedup(df, reaction_lookup):
    """Fill Reaction_Type / RxnCount from the mechanism lookup and drop
    duplicate-group copies, keeping only the lowest ID per equation.

    Without a lookup this is a no-op apart from defaulting Reaction_Type.
    With a lookup:
      - Reaction_Type is taken from the YAML (overriding the CSV) so PLOG /
        Duplicate / ThirdBody are correctly classified;
      - RxnCount is filled from the mechanism for PLOG variants, so the
        writer's (RxnCount - 2) interconnectedness count is accurate;
      - a same-equation duplicate pair is reduced to the single lowest ID
        (your downstream code expands the a/b internals).
    """
    df = df.copy()

    if reaction_lookup is None:
        if "Reaction_Type" not in df.columns:
            df["Reaction_Type"] = "Elementary"
        return df

    types, counts, keep_mask = [], [], []
    seen_groups = set()
    for _, row in df.iterrows():
        rid = int(row["Reaction_ID"])
        entry = reaction_lookup.get(rid)
        if entry is None:
            types.append(row.get("Reaction_Type", "Elementary"))
            counts.append(row.get("RxnCount"))
            keep_mask.append(True)
            continue

        types.append(entry["Reaction_Type"])
        counts.append(entry.get("RxnCount"))

        group = entry.get("Duplicate_Group", (rid,))
        if len(group) > 1:
            # keep only the lowest-ID copy of a duplicate group
            if group in seen_groups or rid != min(group):
                keep_mask.append(False)
            else:
                seen_groups.add(group)
                keep_mask.append(True)
        else:
            keep_mask.append(True)

    df["Reaction_Type"] = types
    # only overwrite RxnCount where the mechanism actually supplied one
    mech_counts = [c if c is not None else df.iloc[i].get("RxnCount")
                   for i, c in enumerate(counts)]
    df["RxnCount"] = mech_counts
    df = df[keep_mask].reset_index(drop=True)
    return df


def generate_xml_files(classification_df, output_dir, reaction_lookup=None):
    """Generate HTC/LTC * all/factor XML files.

    classification_df must already be loaded via
    interfaces.load_classification_file (so it has 'Group_norm', 'f',
    'data_type').

    reaction_lookup (optional, from mechanism_utils.build_reaction_lookup):
    when supplied, reaction types and PLOG rate counts are taken from the
    mechanism and duplicate pairs are collapsed to the lowest ID. Strongly
    recommended — without it, every reaction is treated as Elementary and
    no PLOG counting / dedup happens.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    if "Group_norm" not in classification_df.columns:
        classification_df = classification_df.copy()
        classification_df["Group_norm"] = (
            classification_df["Group"].apply(normalize_group)
        )

    written = []
    for group in ("HTC", "LTC"):
        gdf = classification_df[classification_df["Group_norm"] == group]
        if gdf.empty:
            continue
        gdf = _enrich_and_dedup(gdf, reaction_lookup)
        for ptype in ("all", "factor"):
            out = os.path.join(output_dir, f"{group}_{ptype}.xml")
            generate_xml(gdf, out, perturbation_type=ptype)
            written.append(out)
    return written
