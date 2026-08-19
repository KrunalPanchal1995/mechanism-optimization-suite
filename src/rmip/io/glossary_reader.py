"""
rmip/io/glossary_reader.py
==========================
Load a species-SMILES glossary from CSV or YAML and provide a fast
case-insensitive lookup.

CSV format (minimum required columns, header case-insensitive)::

    name,smiles
    NC7H16,CCCCCCC
    O2,[O][O]

YAML format (flat or list)::

    NC7H16: CCCCCCC
    O2: "[O][O]"

Extra columns / keys are silently ignored (design decision D5).

**Phase 5D — species variant overrides** (optional ``variants_path``)::

    variant_name,canonical_name,reason
    OHEX,OH,excited_state
    HNO2,HONO,tautomer

Usage::

    from rmip.io.glossary_reader import load_glossary, lookup_smiles, load_variants

    glossary = load_glossary("species_smiles.csv")
    smiles   = lookup_smiles(glossary, "NC7H16")   # → "CCCCCCC"

    variants = load_variants("species_variants.csv")
    # → {"OHEX": {"canonical_name": "OH", "reason": "excited_state"}, ...}
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, Optional

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

log = logging.getLogger("rmip.io.glossary")

# Type alias: upper-cased species name → raw SMILES string
Glossary = Dict[str, str]

# Type alias: upper-cased variant name → {canonical_name, reason}
Variants = Dict[str, Dict[str, str]]


def load_glossary(path: str) -> Glossary:
    """
    Load a species glossary from *path* (.csv or .yaml/.yml).

    Returns
    -------
    dict
        ``{UPPER_CASE_NAME: smiles_string}``.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".yaml", ".yml"):
        glossary = _load_yaml(p)
    elif suffix == ".csv":
        glossary = _load_csv(p)
    else:
        # Fallback: try CSV first, then YAML
        try:
            glossary = _load_csv(p)
        except Exception:
            glossary = _load_yaml(p)
    log.info("Loaded glossary: %d entries from '%s'", len(glossary), path)
    return glossary


def lookup_smiles(glossary: Glossary, species_name: str) -> Optional[str]:
    """
    Return the SMILES string for *species_name* (case-insensitive).

    Returns ``None`` if the species is not in the glossary.
    """
    return glossary.get(species_name.upper())


def load_variants(path: str) -> Variants:
    """
    Load a Phase-5D species-variant override CSV.

    CSV must have columns ``variant_name``, ``canonical_name``, ``reason``
    (header case-insensitive).  Comment lines starting with ``#`` are ignored.

    Returns
    -------
    dict
        ``{UPPER_CASE_VARIANT_NAME: {"canonical_name": str, "reason": str}}``.
    """
    p = Path(path)
    result: Variants = {}
    with p.open(newline="", encoding="utf-8") as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    if reader.fieldnames is None:
        log.warning("Empty or headerless variants CSV: %s", path)
        return result

    header_map = {h.strip().lower(): h.strip() for h in reader.fieldnames if h}
    vcol  = header_map.get("variant_name")
    ccol  = header_map.get("canonical_name")
    rcol  = header_map.get("reason")
    if vcol is None or ccol is None:
        raise ValueError(
            f"Variants CSV '{path}' must have 'variant_name' and "
            f"'canonical_name' columns. Found: {list(reader.fieldnames)}"
        )

    for row in reader:
        vname  = (row.get(vcol) or "").strip()
        cname  = (row.get(ccol) or "").strip()
        reason = (row.get(rcol) or "").strip() if rcol else ""
        if vname and cname:
            result[vname.upper()] = {"canonical_name": cname, "reason": reason}

    log.info("Loaded %d species variant overrides from '%s'", len(result), path)
    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> Glossary:
    result: Glossary = {}
    with path.open(newline="", encoding="utf-8") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        # Skip comment lines starting with '#'
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
    reader = csv.DictReader(lines, dialect=dialect)
    if reader.fieldnames is None:
        raise ValueError(f"Empty or headerless CSV: {path}")

    # Case-insensitive header mapping
    header_map = {h.strip().lower(): h.strip() for h in reader.fieldnames if h}
    name_col   = header_map.get("name")
    smiles_col = header_map.get("smiles")
    if name_col is None or smiles_col is None:
        raise ValueError(
            f"Glossary CSV '{path}' must have 'name' and 'smiles' columns. "
            f"Found: {list(reader.fieldnames)}"
        )

    for row in reader:
        name   = (row.get(name_col)   or "").strip()
        smiles = (row.get(smiles_col) or "").strip()
        if name and smiles:
            result[name.upper()] = smiles
    return result


def _load_yaml(path: Path) -> Glossary:
    if not _HAS_YAML:
        raise ImportError(
            "PyYAML is required to load YAML glossary files. "
            "Install with: pip install pyyaml"
        )
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    result: Glossary = {}

    if isinstance(raw, dict):
        # flat: {name: smiles}
        for name, smiles in raw.items():
            if name and smiles:
                result[str(name).upper()] = str(smiles)
    elif isinstance(raw, list):
        # list of {name: ..., smiles: ...}
        for entry in raw:
            name   = entry.get("name")   or entry.get("Name")
            smiles = entry.get("smiles") or entry.get("SMILES")
            if name and smiles:
                result[str(name).upper()] = str(smiles)
    else:
        raise ValueError(f"Unsupported YAML glossary structure in '{path}'")
    return result
