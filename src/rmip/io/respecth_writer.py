"""
rmip.io.respecth_writer
==========================
Writer half of the ReSpecTh/RKD XML bridge — the inverse of
``respecth_reader.py`` (which itself bridges TO ``xml2csv.py``). Builds a
ReSpecTh-schema XML from a CSV of digitized/tabulated ``(x, y[, y_plus,
y_minus])`` points plus a small ``metadata`` dict describing the experiment
(target type, fixed conditions, composition), satisfying every HARD
requirement ``config_tools/converters_io/XML2CSV/xml2csv.py``'s
``_parse_xml_file()`` enforces before it will accept a file at all, rather
than skip it with "Missing commonProperties or dataGroup" (see
WORKBENCH_AUDIT_2026-08-10.md, defect B11 / Part F proposal F1).

Every requirement below was confirmed by reading ``xml2csv.py`` directly
(not guessed), and cross-checked against a real, valid, minimal fixture:
``DATASET/AMMONIA/FLAME_SPEED/x20101002.xml``.

  * ``<experimentType>`` text and ``<apparatus><kind>/<mode></apparatus>``
    — ``_classify_target()`` matches ``kind`` (with ``mode`` only consulted
    for the ``kind == "flame"`` case, which this writer never emits, so
    ``mode`` content is not load-bearing here).
  * ``<dataGroup id="dg1">`` literally, with ≥1 ``<dataPoint>``.
  * Composition with a nonzero O2 mole fraction, matching a **known**
    fuel/stoichiometry lookup (``chem_data.py`` — the same module
    ``xml2csv.py`` itself imports) so the equivalence ratio ``xml2csv.py``
    RE-DERIVES from the written mole fractions matches the value this
    writer was asked to encode, rather than merely emitting a plausible
    but silently-wrong composition. ``xml2csv.py`` never reads an
    "equivalence ratio" property directly — φ is always recomputed from
    composition — confirmed by grepping the module.
  * The observable property's ``name`` attribute must CONTAIN
    ``"ignition delay"`` or ``"laminar burning velocity"``.
  * ``<ignitionType target=... type=.../>`` (Tig only) is OPTIONAL:
    ``xml2csv.py`` only requires it to parse successfully if the element
    is present at all (confirmed: the parsing loop is gated on finding the
    element as a direct child of ``<experiment>``); omitting it entirely
    is valid and simpler than guessing a species-based ignition
    definition the caller didn't supply.
  * ``<bibliographyLink>``/``<commonProperties>`` uncertainty are optional
    everywhere — ``xml2csv.py`` degrades gracefully without them.

This module is deliberately CSV+metadata generic, not tied to the
digitizer specifically — the same "F1: CSV → ReSpecTh XML converter" the
audit proposed, usable by any hand-prepared CSV (the digitizer's own
``export_csv`` output is just one such CSV, in the exact ``x, y[, y_plus,
y_minus]`` shape this module reads).
"""
from __future__ import annotations

import csv
import os
import sys
from typing import Dict, List, Optional, Tuple
from xml.dom.minidom import parseString
from xml.etree import ElementTree as ET

# Same stoichiometric-O2 lookup xml2csv.py itself imports (chem_data.py, at
# the repo root) — reusing it (not reimplementing a second guess-table)
# means a phi computed here and RE-derived by xml2csv.py's own
# composition -> phi calculation matches exactly, rather than drifting.
_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
try:
    from chem_data import DEFAULT_STOICH_O2 as _STOICH_O2, compute_nu_o2 as _compute_nu_o2
    _HAS_CHEM_DATA = True
except ImportError:  # pragma: no cover - chem_data.py is part of this repo
    _STOICH_O2 = {}
    _HAS_CHEM_DATA = False

    def _compute_nu_o2(_formula: str) -> float:
        return 0.0


def _nu_o2(species: str) -> float:
    v = _STOICH_O2.get(species)
    if v is not None:
        return v
    return _compute_nu_o2(species)


# Recognized <ignitionType target=...> keys xml2csv.py's _IGN_SPECIES_MW
# table matches against — "OH" is the most common, widely-supported
# ReSpecTh convention for shock-tube/RCM ignition-delay data.
DEFAULT_IGNITION = {"target": "OH", "type": "max", "amount": "", "units": ""}

# (Target_kind, Target_mode) pairs _classify_target() (xml2csv.py) accepts,
# one representative literal per target_type this writer supports.
_APPARATUS_BY_TARGET_TYPE = {
    "Tig": ("shock tube", "reflected shock"),
    "Fls": ("outwardly propagating spherical flame", "spherical"),
    "Flf_a": ("heat flux burner", "premixed"),
}

_EXPERIMENT_TYPE_BY_TARGET_TYPE = {
    "Tig": "ignition delay measurement",
    "Fls": "laminar burning velocity measurement",
    "Flf_a": "laminar burning velocity measurement",
}

# Substring xml2csv.py's classifier expects the observable dataGroup
# property's `name` attribute to CONTAIN — confirmed at xml2csv.py's
# `if exp in var_info.get("name", ""):` check.
_OBSERVABLE_NAME_BY_TARGET_TYPE = {
    "Tig": "ignition delay",
    "Fls": "laminar burning velocity",
    "Flf_a": "laminar burning velocity",
}

_SUPPORTED_TARGET_TYPES = tuple(_APPARATUS_BY_TARGET_TYPE)


class RespecthWriterError(ValueError):
    """Raised for a metadata/CSV combination this writer cannot encode."""


def _looks_numeric(cell: str) -> bool:
    try:
        float(cell)
        return True
    except (TypeError, ValueError):
        return False


def _read_csv_points(csv_path: str) -> Tuple[List[str], List[List[str]]]:
    """Returns ``(header, rows)`` read POSITIONALLY, not by column name.

    ``mso_studio.core.digitizer.export_csv``'s real recorded output
    (``Test_run_23_JULY/mso_studio/gui_widget_evidence/digitized_points.csv``)
    has duplicate column headers (``T_K,T_K`` -- an x/y label defaulted or
    typed identically at export time). ``csv.DictReader`` silently
    collapses duplicate keys to whichever column comes last, losing the
    first column's data entirely -- confirmed: ``DictReader`` on that exact
    file yields ``{'T_K': '25.0'}``, discarding the `50.0`. Reading
    positionally instead means this writer works on that file (and any
    other CSV with non-unique or missing headers) exactly as well as one
    with clean, distinct column names.

    Also tolerates a CSV with NO header row at all -- module docstring
    advertises this writer as "usable by any hand-prepared CSV", and a
    headerless CSV is a completely ordinary hand-prepared shape. Without
    this check, a 3-row headerless numeric CSV silently produced only 2
    ``<dataPoint>`` elements (row 0 consumed as a phantom header) -- the
    worst failure mode this tool can have, since it drops a real digitized
    point with no error or warning. Detected by checking whether row 0's
    first two fields both parse as floats; if so, row 0 is data, not a
    header, and placeholder column names are synthesized (their *content*
    is never read downstream -- only ``len(header)``, for the "has
    y_plus/y_minus columns" check).
    """
    # "utf-8-sig", not "utf-8": a BOM-prefixed CSV (routine from Excel/
    # Windows-authored exports) otherwise leaves a literal U+FEFF glued to
    # the first cell of row 0 ("﻿1000"), which fails the headerless-CSV
    # numeric-detection check above and silently resurrects the exact
    # first-point-dropped bug that check exists to prevent. utf-8-sig
    # strips a leading BOM if present and is a no-op otherwise, so this is
    # strictly more permissive, never less.
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(row for row in fh if not row.lstrip().startswith("#"))
        # A truly blank line parses as [] and is already dropped by `if r`
        # below, but a line that's only whitespace (e.g. a stray blank row
        # padded with a trailing comma-less space, common from a
        # hand-edited or spreadsheet-exported CSV) parses as a non-empty
        # list like ['   '] and used to reach the data-row loop, where it
        # failed float() and raised "not valid (x, y) data" -- a real row
        # count is still 0 there, so this should be silently skipped like
        # any other blank line, not treated as a malformed data point.
        rows = [r for r in reader if r and any(cell.strip() for cell in r)]
    if not rows:
        return [], []
    first = rows[0]
    if len(first) >= 2 and _looks_numeric(first[0]) and _looks_numeric(first[1]):
        header = [f"col{i}" for i in range(len(first))]
        return header, rows
    return first, rows[1:]


def _composition_for_phi(
    phi: float, fuel: Dict[str, float], bath_gas_ratio: float, bath_gas_species: str = "N2",
) -> Tuple[Dict[str, float], float]:
    """Mole fractions ``{species: x}`` for *fuel* (species -> relative
    amount within the fuel blend, e.g. ``{"NH3": 1.0}``) plus O2 plus an
    implicit bath gas, at equivalence ratio *phi*, holding
    ``bath_gas:O2 == bath_gas_ratio`` (3.76 for air's N2:O2). Returns
    ``(mole_fractions_without_bath_gas, bath_gas_mole_fraction)``.

    Uses :func:`_nu_o2` — the SAME stoichiometric-O2 lookup ``xml2csv.py``
    itself uses — so the phi ``xml2csv.py`` re-derives from this
    composition matches the requested *phi* exactly (both sides compute
    from the identical table), not merely approximately.
    """
    fuel_total = sum(fuel.values())
    if fuel_total <= 0:
        raise RespecthWriterError(f"Fuel composition {fuel!r} has zero total amount.")
    nu_o2 = sum(_nu_o2(sp) * frac for sp, frac in fuel.items()) / fuel_total
    if nu_o2 <= 0:
        raise RespecthWriterError(
            f"No known stoichiometric O2 demand for fuel {fuel!r} — "
            f"add it to chem_data.DEFAULT_STOICH_O2 or pass nu_o2 explicitly."
        )
    f_a_st = 1.0 / nu_o2  # stoichiometric fuel:O2 mole ratio (1 mole fuel-equivalent : nu_o2 moles O2)
    o2_moles = 1.0
    fuel_moles_total = phi * f_a_st * o2_moles
    bath_moles = bath_gas_ratio * o2_moles
    total = fuel_moles_total + o2_moles + bath_moles
    mole_fractions = {sp: (fuel_moles_total * frac / fuel_total) / total for sp, frac in fuel.items()}
    mole_fractions["O2"] = o2_moles / total
    return mole_fractions, bath_moles / total


def _merge_bath_gas(mole_fracs: Dict[str, float], bath_gas_species: str, bath_x: float) -> Dict[str, float]:
    """Merge the bath-gas mole fraction into *mole_fracs*, ADDING to any
    existing fuel-species contribution of the same name rather than
    overwriting it. A fuel blend that itself contains the bath-gas species
    -- e.g. a diluted syngas fuel specified as {"H2": 0.4, "CO": 0.4, "N2":
    0.2} paired with the default air bath gas (also N2) -- is a real,
    physically ordinary combustion case, not an edge case; the naive
    ``{**mole_fracs, bath_gas_species: bath_x}`` dict-merge this replaces
    silently discarded the fuel-side N2 mole fraction entirely rather than
    combining it with the bath-gas contribution."""
    merged = dict(mole_fracs)
    merged[bath_gas_species] = merged.get(bath_gas_species, 0.0) + bath_x
    return merged


def _species_link(parent: ET.Element, preferred_key: str) -> None:
    ET.SubElement(parent, "speciesLink", {"preferredKey": preferred_key})


def write_respecth_xml(csv_path: str, metadata: dict, output_path: str) -> str:
    """Build a ReSpecTh-schema XML from a digitized/tabulated CSV (the
    exact ``x, y[, y_plus, y_minus]`` shape
    :func:`mso_studio.core.digitizer.export_csv` already produces) plus
    *metadata*, and write it to *output_path*. Returns the XML string.

    ``metadata`` shape::

        {
            "target_type": "Tig" | "Fls" | "Flf_a",
            "x_quantity": "temperature" | "phi",
            "x_units": "K" | "unitless" | ...,
            "y_units": "ms" | "cm/s" | ...,
            "fixed": {
                "pressure_atm": float,
                # whichever of "temperature_K" / "phi" is NOT x_quantity:
                "temperature_K": float,   # required if x_quantity == "phi"
                "phi": float,             # required if x_quantity == "temperature"
            },
            "composition": {
                "fuel": {"NH3": 1.0},          # species -> relative amount
                "bath_gas_ratio": 3.76,        # bath_gas : O2 molar ratio
                "bath_gas_species": "N2",      # optional, default "N2"
            },
            "ignition": {"target": "OH", "type": "max", ...},  # Tig only, optional
            "bibliography": {"author": ..., "doi": ..., "description": ...},  # optional
            "dataset_id": "my_digitized_series",  # optional, default derived from csv filename
        }
    """
    target_type = metadata.get("target_type")
    if target_type not in _SUPPORTED_TARGET_TYPES:
        raise RespecthWriterError(
            f"target_type must be one of {_SUPPORTED_TARGET_TYPES}, got {target_type!r}"
        )
    x_quantity = metadata.get("x_quantity")
    if x_quantity not in ("temperature", "phi"):
        raise RespecthWriterError(f"x_quantity must be 'temperature' or 'phi', got {x_quantity!r}")

    header, rows = _read_csv_points(csv_path)
    if not rows:
        raise RespecthWriterError(f"{csv_path}: no data rows.")
    if len(header) < 2:
        raise RespecthWriterError(f"{csv_path}: need at least 2 columns (x, y), found {header!r}.")
    # 2 columns: (x, y). 3 columns: (x, y, dy) -- a single symmetric
    # uncertainty. 4+ columns: (x, y, y_plus, y_minus) -- asymmetric.
    # Previously gated at `>= 4` only, which silently discarded a 3rd
    # column entirely for a (x, y, dy)-shaped CSV: no error, no warning,
    # the uncertainty a user explicitly provided just vanished. The
    # per-row loop below already handles a missing row[3] via its own
    # `len(row) > 3` check (y_minus defaults to 0.0, so dy = max(y_plus,
    # 0.0) = the single dy value) -- the only fix needed is not gating the
    # whole property out at 3 columns.
    has_err = len(header) >= 3

    fixed = metadata.get("fixed", {})
    composition = metadata.get("composition", {})
    fuel = composition.get("fuel")
    if not fuel:
        raise RespecthWriterError("metadata['composition']['fuel'] is required.")
    bath_gas_ratio = float(composition.get("bath_gas_ratio", 3.76))
    bath_gas_species = composition.get("bath_gas_species", "N2")

    dataset_id = metadata.get("dataset_id") or os.path.splitext(os.path.basename(csv_path))[0]

    root = ET.Element("experiment")
    bib = metadata.get("bibliography")
    if bib:
        bib_el = ET.SubElement(root, "bibliographyLink")
        if bib.get("description"):
            ET.SubElement(bib_el, "description").text = bib["description"]
        if bib.get("doi"):
            ET.SubElement(bib_el, "referenceDOI").text = bib["doi"]
        if bib.get("author") or bib.get("year"):
            details = ET.SubElement(bib_el, "details")
            if bib.get("author"):
                ET.SubElement(details, "author").text = bib["author"]
            if bib.get("year"):
                ET.SubElement(details, "year").text = str(bib["year"])
    else:
        ET.SubElement(root, "fileAuthor").text = "MSO Studio csv2xml"

    ET.SubElement(root, "experimentType").text = _EXPERIMENT_TYPE_BY_TARGET_TYPE[target_type]

    kind, mode = _APPARATUS_BY_TARGET_TYPE[target_type]
    apparatus = ET.SubElement(root, "apparatus")
    ET.SubElement(apparatus, "kind").text = kind
    ET.SubElement(apparatus, "mode").text = mode

    if target_type == "Tig":
        ign = metadata.get("ignition")
        if ign:
            ET.SubElement(root, "ignitionType", {
                "target": ign.get("target", DEFAULT_IGNITION["target"]),
                "type": ign.get("type", DEFAULT_IGNITION["type"]),
                **({"amount": str(ign["amount"])} if ign.get("amount") not in (None, "") else {}),
                **({"units": ign["units"]} if ign.get("units") else {}),
            })

    common = ET.SubElement(root, "commonProperties")
    ET.SubElement(common, "property", {
        "name": "pressure", "label": "p", "sourcetype": "reported", "units": "atm",
    }).append(_value_el(fixed.get("pressure_atm", 1.0)))

    if x_quantity == "phi":
        # Composition varies per data point (each row's phi differs) --
        # written into dataGroup below, not commonProperties. Temperature
        # is fixed here since it isn't the digitized axis.
        temp_el = ET.SubElement(common, "property", {
            "name": "temperature", "label": "T", "sourcetype": "reported", "units": "K",
        })
        temp_el.append(_value_el(fixed.get("temperature_K", 298.0)))
    else:
        # x_quantity == "temperature": composition is CONSTANT across all
        # rows (goes into commonProperties as "initial composition"),
        # phi is fixed.
        phi_fixed = float(fixed.get("phi", 1.0))
        mole_fracs, bath_x = _composition_for_phi(phi_fixed, fuel, bath_gas_ratio, bath_gas_species)
        comp_el = ET.SubElement(common, "property", {
            "name": "initial composition", "sourcetype": "calculated",
        })
        for sp, x in _merge_bath_gas(mole_fracs, bath_gas_species, bath_x).items():
            entry = ET.SubElement(comp_el, "component")
            _species_link(entry, sp)
            amount = ET.SubElement(entry, "amount", {"units": "mole fraction"})
            amount.text = f"{x:.6g}"

    dg = ET.SubElement(root, "dataGroup", {"id": "dg1"})
    prop_ids: List[str] = []

    def _next_id() -> str:
        prop_ids.append(f"x{len(prop_ids) + 1}")
        return prop_ids[-1]

    x_id = _next_id()
    if x_quantity == "temperature":
        ET.SubElement(dg, "property", {
            "id": x_id, "name": "temperature", "label": "T",
            "sourcetype": "reported", "units": metadata.get("x_units", "K"),
        })
    else:
        ET.SubElement(dg, "property", {
            "id": x_id, "name": "equivalence ratio", "label": "phi",
            "sourcetype": "reported", "units": metadata.get("x_units", "unitless"),
        })
        # Per-row composition properties, one per species (fuel + O2 + bath
        # gas) -- xml2csv.py's "varying composition per data point" branch
        # (the `else` of its own commonProperties-composition check) reads
        # these by `specification/preferredKey`, not by a fixed id scheme.
        species_ids: Dict[str, str] = {}
        # De-duplicate while preserving order (dict.fromkeys keeps first
        # occurrence) -- if bath_gas_species collides with a fuel species
        # name or "O2", this used to declare the SAME species name as two
        # separate <property> elements with two different ids, and the
        # second (bath-gas) id silently won every per-row value write
        # below, leaving the first id's <property> declared but never
        # populated in any <dataPoint> -- a structurally invalid dataGroup
        # that a strict XML consumer could reject.
        species_names = list(dict.fromkeys(list(fuel) + ["O2", bath_gas_species]))
        for sp in species_names:
            sid = _next_id()
            species_ids[sp] = sid
            comp_prop = ET.SubElement(dg, "property", {
                "id": sid, "name": "composition", "label": f"x_{sp}",
                "sourcetype": "calculated", "units": "mole fraction",
            })
            _species_link(comp_prop, sp)

    y_id = _next_id()
    ET.SubElement(dg, "property", {
        "id": y_id, "name": _OBSERVABLE_NAME_BY_TARGET_TYPE[target_type], "label": "y",
        "sourcetype": "reported", "units": metadata.get("y_units", ""),
    })

    if has_err:
        dy_id = _next_id()
        ET.SubElement(dg, "property", {
            "id": dy_id, "name": "evaluated standard deviation", "label": "dy",
            "sourcetype": "estimated", "units": metadata.get("y_units", ""),
            "reference": _OBSERVABLE_NAME_BY_TARGET_TYPE[target_type], "kind": "absolute",
        })

    for row_num, row in enumerate(rows, start=1):
        try:
            x_val = float(row[0])
            y_val = float(row[1])
        except (IndexError, ValueError) as exc:
            raise RespecthWriterError(
                f"{csv_path}: data row {row_num} ({row!r}) is not valid "
                f"(x, y) data: {exc}"
            ) from exc
        dp = ET.SubElement(dg, "dataPoint")
        ET.SubElement(dp, x_id).text = f"{x_val:.6g}"
        if x_quantity == "phi":
            mole_fracs, bath_x = _composition_for_phi(x_val, fuel, bath_gas_ratio, bath_gas_species)
            for sp, x in _merge_bath_gas(mole_fracs, bath_gas_species, bath_x).items():
                ET.SubElement(dp, species_ids[sp]).text = f"{x:.6g}"
        ET.SubElement(dp, y_id).text = f"{y_val:.6g}"
        if has_err:
            try:
                y_plus = float(row[2]) if len(row) > 2 and row[2] not in ("", None) else 0.0
                y_minus = float(row[3]) if len(row) > 3 and row[3] not in ("", None) else 0.0
            except ValueError as exc:
                raise RespecthWriterError(
                    f"{csv_path}: data row {row_num} ({row!r}) has a non-numeric "
                    f"y_plus/y_minus value: {exc}"
                ) from exc
            ET.SubElement(dp, dy_id).text = f"{max(y_plus, y_minus):.6g}"

    tree = ET.ElementTree(root)
    raw = ET.tostring(root, encoding="unicode")
    pretty = parseString(raw).toprettyxml(indent="  ")
    # toprettyxml() emits its own XML declaration line; keep exactly one.
    pretty = "\n".join(line for line in pretty.splitlines() if line.strip())
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(pretty + "\n")
    return pretty


def _value_el(value) -> ET.Element:
    el = ET.Element("value")
    el.text = f"{float(value):.6g}"
    return el
