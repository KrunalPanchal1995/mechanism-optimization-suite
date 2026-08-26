"""
rmip/io/respecth_reader.py
==========================
ReSpecTh / PrIMe (RKD XML) experimental-data ingestion — **bridges both**
experimental-data subsystems of this suite:

  * **Legacy core** (endpoint A): the mature standalone converter
    ``config_tools/converters_io/XML2CSV/xml2csv.py`` already turns a directory
    of ReSpecTh RKD XML files into the pipe-delimited ``.target`` + ``.add``
    (+ ``.bib`` + flat ``.csv``) files that ``src/MUQ-SAC/run_script.py`` and
    the GUI Experimental-Data tab consume.  We call it in-process here so XML
    ingestion becomes a first-class, importable step instead of a detached CLI.

  * **Modern rmip DB** (endpoint B): the same converted ``.target`` file is
    re-parsed into rmip-canonical experimental points and loaded into the
    ``exp_datasets`` / ``exp_data_points`` SQLite tables (see
    :class:`rmip.io.exp_data_reader.ExpDataReader`).

Why re-parse rather than copy legacy values?  The two subsystems use *different
unit conventions*: the legacy ``combustion_target`` keeps IDT internally in
microseconds and pressure in Pascals, whereas the rmip DB is canonicalised to
**IDT in seconds, LFS in m/s, pressure in atm**.  So this module reads the raw
value + its source unit (carried in each ``.target`` line's ``units`` JSON) and
converts to the rmip canon explicitly.

This module is stdlib-only on the rmip side; ``xml2csv`` itself only needs
``numpy`` (already a pinned dependency), so importing it in-process is safe.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Locate + import the standalone xml2csv converter (endpoint A)
# ---------------------------------------------------------------------------

_XML2CSV_DIR = (
    Path(__file__).resolve().parents[3]
    / "config_tools" / "converters_io" / "XML2CSV"
)


def _load_xml2csv():
    """
    Import ``xml2csv.py`` from ``config_tools`` by file path (it is a script,
    not an installed package).  Its directory is appended to ``sys.path`` so
    its sibling ``chem_data`` import resolves — appended, never inserted at 0,
    to preserve stdlib precedence (same rule as the repo's root conftest).
    """
    if not _XML2CSV_DIR.exists():
        raise FileNotFoundError(
            f"xml2csv converter directory not found: {_XML2CSV_DIR}"
        )
    if str(_XML2CSV_DIR) not in sys.path:
        sys.path.append(str(_XML2CSV_DIR))
    spec = importlib.util.spec_from_file_location(
        "rmip_vendored_xml2csv", str(_XML2CSV_DIR / "xml2csv.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Unit maps + experiment-type mapping
# ---------------------------------------------------------------------------

# ReSpecTh internal target tag (from xml2csv._classify_target) -> rmip sim_type.
RESPECTH_SIM_TYPE_MAP: Dict[str, str] = {
    "Tig":   "IDT",   # shock tube / ignition-delay measurement
    "RCM":   "IDT",   # rapid compression machine (also an ignition delay)
    "Fls":   "LFS",   # laminar burning velocity
    "Flf":   "LFS",   # heat-flux / burner-stabilised flame speed
    "Flf_a": "LFS",   # heat-flux burner (imposed T-profile) flame speed
    "JSR":   "JSR",   # jet-stirred / flow-reactor speciation
    "PFR":   "PFR",   # plug-flow reactor speciation
    "Flw":   "PFR",   # flow-reactor species decay
}

# Multiplicative factors to the rmip canonical unit.
_TIME_TO_S = {
    "s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0,
    "ms": 1e-3, "millisecond": 1e-3,
    "us": 1e-6, "µs": 1e-6, "microsecond": 1e-6,
    "ns": 1e-9,
}
_VEL_TO_MS = {
    "m/s": 1.0, "ms-1": 1.0,
    "cm/s": 1e-2, "cms-1": 1e-2,
    "mm/s": 1e-3,
}
_P_TO_ATM = {
    "atm": 1.0,
    "bar": 1.0 / 1.01325,
    "mbar": 1.0 / 1013.25,
    "torr": 1.0 / 760.0, "mmhg": 1.0 / 760.0,
    "pa": 1.0 / 101325.0,
    "kpa": 1.0 / 101.325,
}


def _observed_factor(sim_type: str, unit: Optional[str]) -> Tuple[float, str]:
    """
    Return ``(factor, canonical_unit)`` for the observed quantity: multiply the
    raw value (and its absolute uncertainty) by ``factor`` to get the rmip canon.
    Unknown units fall back to factor 1.0 with a warning.
    """
    u = (unit or "").strip().lower()
    if sim_type == "IDT":
        if u and u not in _TIME_TO_S:
            logger.warning("Unknown IDT time unit %r — treating as seconds", unit)
        return _TIME_TO_S.get(u, 1.0), "s"
    if sim_type == "LFS":
        if u and u not in _VEL_TO_MS:
            logger.warning("Unknown LFS velocity unit %r — treating as m/s", unit)
        return _VEL_TO_MS.get(u, 1.0), "m/s"
    # speciation (JSR/PFR): canonicalise to mole fraction
    if u == "ppm":
        return 1e-6, "mole_fraction"
    return 1.0, (u or "mole_fraction")


def _pressure_to_atm(value: float, unit: Optional[str]) -> float:
    u = (unit or "atm").strip().lower()
    if u not in _P_TO_ATM:
        logger.warning("Unknown pressure unit %r — treating as atm", unit)
    return value * _P_TO_ATM.get(u, 1.0)


# ---------------------------------------------------------------------------
# .target line tokeniser (mirrors combustion_target.__init__'s | / -- split)
# ---------------------------------------------------------------------------

_FUEL_SPECIES_RE = re.compile(r"'([^']+)'\s*:")


def _tokenize_target_line(line: str) -> Optional[Tuple[str, Dict[str, str]]]:
    """
    Split one ``.target`` line into ``(dataset_id, {token_key: value})`` using
    the same ``|`` then ``--`` convention the legacy ``combustion_target``
    parser uses.  Returns ``None`` for blank lines.
    """
    line = line.rstrip("\n")
    if not line.strip():
        return None
    parts = [p.strip() for p in line.split("|")]
    # parts[0] = running index, parts[1] = dataset id, remainder = "key -- value"
    dataset_id = parts[1] if len(parts) > 1 else "dataset"
    fields: Dict[str, str] = {}
    for tok in parts[2:]:
        if "--" in tok:
            key, val = tok.split("--", 1)
            fields[key.strip()] = val.strip()
    return dataset_id, fields


def _fuel_name_from_field(fuel_field: Optional[str]) -> Optional[str]:
    """Best-effort fuel label: first species name in the Fuel token dict."""
    if not fuel_field:
        return None
    names = _FUEL_SPECIES_RE.findall(fuel_field)
    if names:
        return "+".join(dict.fromkeys(names))  # de-dup, preserve order
    return None


class ParsedDataset:
    """A ReSpecTh dataset normalised to rmip-canonical points."""

    __slots__ = ("dataset_id", "sim_type", "target_tag", "fuel_name", "points")

    def __init__(self, dataset_id: str, sim_type: str, target_tag: str,
                 fuel_name: Optional[str]) -> None:
        self.dataset_id = dataset_id
        self.sim_type = sim_type
        self.target_tag = target_tag
        self.fuel_name = fuel_name
        self.points: List[Dict[str, Any]] = []


def datasets_from_target_file(
    target_path: str,
    default_fuel_name: Optional[str] = None,
) -> List[ParsedDataset]:
    """
    Parse a converted ``.target`` file into a list of :class:`ParsedDataset`,
    grouping lines by dataset id and converting every value to the rmip canon
    (IDT [s], LFS [m/s], speciation [mole fraction]; P [atm]; T [K]).

    ``default_fuel_name`` (e.g. the fuel(s) declared at import) is used as the
    authoritative dataset fuel label; the ``.target`` Fuel token only carries
    xml2csv's internal letter keys, so it is a poor source for a species name.

    Lines whose ``target`` tag maps to no known rmip sim_type, or whose numeric
    fields are unparseable, are skipped with a warning (the legacy ``.target``
    file itself is left intact — this only affects DB ingestion).
    """
    grouped: Dict[str, ParsedDataset] = {}
    with open(target_path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            tok = _tokenize_target_line(line)
            if tok is None:
                continue
            dataset_id, f = tok
            tag = f.get("target", "").strip()
            sim_type = RESPECTH_SIM_TYPE_MAP.get(tag)
            if sim_type is None:
                logger.warning("Line %d: unknown target tag %r — skipping", lineno, tag)
                continue

            try:
                units = json.loads(f.get("units", "{}").replace("'", '"'))
            except (json.JSONDecodeError, AttributeError):
                units = {}

            try:
                value_raw = float(f["observed"])
                T_K = float(f["T"])
                P_raw = float(f["P"])
            except (KeyError, ValueError) as exc:
                logger.warning("Line %d: unparseable numeric field (%s) — skipping",
                               lineno, exc)
                continue

            factor, canon_unit = _observed_factor(sim_type, units.get("observed"))
            value = value_raw * factor

            dev_raw = f.get("deviation", "")
            try:
                uncertainty = float(dev_raw) * factor if dev_raw not in ("", "0") else None
            except ValueError:
                uncertainty = None

            phi_raw = f.get("Phi", "")
            try:
                phi = float(phi_raw)
            except (TypeError, ValueError):
                phi = None

            P_atm = _pressure_to_atm(P_raw, units.get("P"))

            key = f"{Path(target_path).stem}:{dataset_id}:{sim_type}"
            ds = grouped.get(key)
            if ds is None:
                fuel_name = default_fuel_name or _fuel_name_from_field(f.get("Fuel"))
                ds = ParsedDataset(dataset_id, sim_type, tag, fuel_name)
                grouped[key] = ds
            ds.points.append({
                "T_K": T_K,
                "P_atm": P_atm,
                "phi": phi,
                "value": value,
                "uncertainty": uncertainty,
                "units": canon_unit,
                "notes": f"ReSpecTh {tag} (dataset {dataset_id})",
            })

    return list(grouped.values())


# ---------------------------------------------------------------------------
# Endpoint A: run the legacy converter in-process
# ---------------------------------------------------------------------------

def _resolve_xml_dir(xml_path: str) -> Tuple[str, Optional[str]]:
    """
    ``xml2csv`` operates on a *directory* of ``*.xml``.  If ``xml_path`` is a
    single file, stage it into a fresh temp dir.  Returns ``(dir, tmp_or_None)``
    where ``tmp`` (if not None) must be removed by the caller.
    """
    p = Path(xml_path)
    if p.is_dir():
        return str(p), None
    if p.is_file():
        tmp = tempfile.mkdtemp(prefix="rmip_respecth_")
        shutil.copy2(str(p), os.path.join(tmp, p.name))
        return tmp, tmp
    raise FileNotFoundError(f"ReSpecTh XML path not found: {xml_path}")


def convert_respecth(
    xml_path: str,
    out_dir: Optional[str] = None,
    *,
    fuel_list: Optional[List[str]] = None,
    input_fuel: Optional[Dict[str, str]] = None,
    stoichiometry: Optional[Dict[str, float]] = None,
    solver: str = "cantera",
) -> Dict[str, Any]:
    """
    Endpoint A: convert ReSpecTh XML (a file or a directory) to the legacy
    ``.target`` / ``.add`` / ``.bib`` / ``.csv`` artefacts by calling the
    existing ``xml2csv.main`` in-process.  Returns its summary dict
    (``n_files_ok``, ``n_files_skipped``, ``n_rows``, ``target_path``,
    ``add_path``, ``bib_path``, ``csv_path``, ``warnings``).

    If ``out_dir`` is None it defaults to the XML directory (matching
    ``xml2csv``); pass one explicitly when ingesting a single file so the
    artefacts don't land next to the source.
    """
    xml2csv = _load_xml2csv()
    xml_dir, tmp = _resolve_xml_dir(xml_path)
    # When a single file was staged into a temp dir, artefacts must go elsewhere
    # or they'd be lost with the temp dir on cleanup.
    effective_out = out_dir or (xml_dir if tmp is None else os.getcwd())
    try:
        return xml2csv.main(
            xml_dir=xml_dir,
            out_dir=effective_out,
            fuel_list=fuel_list,
            input_fuel=input_fuel,
            stoichiometry=stoichiometry,
            solver=solver,
        )
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
