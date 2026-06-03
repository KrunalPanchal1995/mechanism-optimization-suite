#!/usr/bin/env python3
"""
modules/latex_tables.py

Publication-quality LaTeX tables for HTC/LTC active reaction lists.

CHANGES vs original latex_table_generator.py:
- Exposes generate_latex_tables(classification_df, output_dir, ...) so the
  driver can import it. The original only had a CLI main().
- Reuses interfaces.normalize_group and the canonical 'f' column.
"""

import os

import pandas as pd

from .interfaces import normalize_group


def latex_escape(text):
    if pd.isna(text):
        return ""
    text = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Reaction arrows in math mode. Order matters: handle the reversible
    # '<=>' first, then the forward '=>' and bare '='.
    text = text.replace("<=>", r"$\rightleftharpoons$")
    text = text.replace("=>", r"$\Rightarrow$")
    text = text.replace("=", r"$\Rightarrow$")
    return text


def _rid_label(rid):
    s = str(rid).strip()
    return s if s.upper().startswith("R") else f"R{s}"


def build_table(df, group_name, fuel_name, longtable=False):
    """Build a single-header `tabular` table.

    The `longtable` argument is accepted for backward compatibility but
    ignored: output is always a `tabular` with exactly one header row.
    """
    lines = []
    lines.append(r"\begin{table}[!ht]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{List of active reactions classified as "
        rf"{group_name} for {fuel_name}}}")
    lines.append(rf"\label{{table:{fuel_name}_{group_name}}}")
    lines.append(r"\begin{tabular}{ccc}")
    lines.append(r"\hline")
    lines.append(r"\textbf{ID} & \textbf{Reaction} & \textbf{f} \\")
    lines.append(r"\hline")

    for _, row in df.iterrows():
        rid = _rid_label(row["Reaction_ID"])
        reaction = latex_escape(row["Reaction"])
        f_value = row["f"]
        lines.append(rf"\textbf{{{rid}}} & {reaction} & {f_value} \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_latex_tables(classification_df, output_dir,
                          fuel_name="Fuel", longtable=False,
                          sort_by_id=True):
    """Write HTC/LTC LaTeX tables.

    sort_by_id : if True (default), rows within each table are sorted by
    Reaction_ID ascending. Set False to keep classification-file order.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = classification_df.copy()
    if "Group_norm" not in df.columns:
        df["Group_norm"] = df["Group"].apply(normalize_group)
    if "f" not in df.columns and "Unsrt" in df.columns:
        df["f"] = df["Unsrt"]

    written = []
    for group in ("HTC", "LTC"):
        gdf = df[df["Group_norm"] == group]
        if gdf.empty:
            continue
        if sort_by_id:
            # numeric sort regardless of dtype (handles any stray strings)
            gdf = gdf.copy()
            gdf["_rid_sort"] = pd.to_numeric(gdf["Reaction_ID"],
                                             errors="coerce")
            gdf = gdf.sort_values("_rid_sort").drop(columns="_rid_sort")
        out = os.path.join(output_dir, f"{group}_table.tex")
        with open(out, "w") as f:
            f.write(build_table(gdf, group, fuel_name))
        written.append(out)
        print(f"[LaTeX] Written -> {out}")
    return written
