"""target_add_io.py -- framework-free read/write helpers for the MUQ-SAC
``.target`` (pipe-delimited experimental cases) and ``.add`` (per-case YAML
option overrides) file pair, plus a lightweight preview parser for GUI
tables.

The authoritative parser for these files is
``src/MUQ-SAC/combustion_target_class.py::combustion_target.__init__`` (also
duplicated, slightly behind, in ``src/Sensitivity_Analysis/``) -- that class
is what ``sens_3_params.py`` actually instantiates at run time, so this
module's ``preview_targets()`` is a *read-only, display-only* lightweight
re-implementation (regex-based, no numpy/sklearn/pandas dependency) and
``format_target_line()`` is written to produce output that class's parser
accepts byte-for-byte (verified against the real pipe/``--``/``->``/``=``
grammar it uses, and covered by tests). Neither of those two functions calls
into ``combustion_target_class`` itself, so this module stays importable in
the lightweight PySide6 ``.venv-gui`` environment too.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .target_add_schema import NewCaseFields, ADD_DEFAULTS, is_runnable


# ---------------------------------------------------------------------------
# .add (YAML dict keyed by dataset ID)
# ---------------------------------------------------------------------------

def load_add(path: str) -> Dict[str, dict]:
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path!r} does not contain a YAML mapping at the top level")
    return data


def save_add(path: str, data: Dict[str, dict]) -> None:
    with open(path, "w") as fh:
        yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False)


def set_case_options(add_data: Dict[str, dict], dataset_id: str,
                      overrides: Dict[str, object]) -> Dict[str, dict]:
    """Return a copy of add_data with dataset_id's entry updated with
    overrides (only overrides -- unset keys keep falling back to
    combustion_target_class's own ADD_DEFAULTS cascade at parse time)."""
    out = dict(add_data)
    entry = dict(out.get(dataset_id) or {})
    entry.update(overrides)
    out[dataset_id] = entry
    return out


# ---------------------------------------------------------------------------
# .target (pipe-delimited) -- lightweight preview parser
# ---------------------------------------------------------------------------

@dataclass
class TargetPreview:
    line_no: int
    dataset_id: str
    target_type: str = ""
    simulation: str = ""
    temperature_K: Optional[float] = None
    pressure_atm: Optional[float] = None
    phi: str = ""
    species: str = ""
    solver: str = ""              # from the paired .add, "" if not resolvable
    runnable: bool = True
    runnable_reason: str = ""
    raw: str = ""


def _field(parts: List[str], key: str) -> Optional[str]:
    for p in parts:
        kv = p.split("--", 1)
        if len(kv) != 2:
            continue
        if kv[0].strip() == key:
            return kv[1].strip()
    return None


def preview_targets(target_path: str, add_path: Optional[str] = None) -> List[TargetPreview]:
    """Display-only parse of a .target file. Tolerant of malformed lines
    (skips them rather than raising, since this only feeds a GUI table --
    the real run still goes through combustion_target_class, which will
    raise its own precise error on a genuinely bad line)."""
    add_data: Dict[str, dict] = {}
    if add_path:
        try:
            add_data = load_add(add_path)
        except Exception:
            add_data = {}

    out: List[TargetPreview] = []
    with open(target_path) as fh:
        for i, raw in enumerate(fh):
            line = raw.rstrip("\n")
            if not line.strip() or line.strip().startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            dataset_id = parts[1].strip().strip("\t")
            if not dataset_id:
                continue
            target_type = (_field(parts, "target") or "").strip()
            t_raw = _field(parts, "T")
            p_raw = _field(parts, "P")
            try:
                t_val = float(t_raw) if t_raw not in (None, "") else None
            except ValueError:
                t_val = None
            try:
                p_val = float(p_raw) if p_raw not in (None, "") else None
            except ValueError:
                p_val = None

            add_entry = add_data.get(dataset_id) or {}
            solver = add_entry.get("solver", ADD_DEFAULTS["solver"])
            flux = bool(add_entry.get("flux_analysis", ADD_DEFAULTS["flux_analysis"]))
            ok, reason = is_runnable(solver, target_type or "Tig", flux)

            out.append(TargetPreview(
                line_no=i + 1,
                dataset_id=dataset_id,
                target_type=target_type,
                simulation=(_field(parts, "simulation") or "").strip(),
                temperature_K=t_val,
                pressure_atm=p_val,
                phi=(_field(parts, "Phi") or "").strip(),
                species=(_field(parts, "species") or "").strip(),
                solver=solver,
                runnable=ok,
                runnable_reason=reason,
                raw=line,
            ))
    return out


# ---------------------------------------------------------------------------
# .target -- manual-entry serializer (writer, matches the parser's grammar)
# ---------------------------------------------------------------------------

def _fuel_multi_repr(species_fracs: Dict[str, float]) -> str:
    keys = "abcdefghijklmnopqrstuvwxyz"
    ids = {keys[i]: sp for i, sp in enumerate(species_fracs)}
    xs = {keys[i]: frac for i, (sp, frac) in enumerate(species_fracs.items())}
    return f"x->{ids} = {xs}"


def _bath_gas_repr(species_fracs: Dict[str, float]) -> str:
    keys = "abcdefghijklmnopqrstuvwxyz"
    ids = {keys[i]: sp for i, sp in enumerate(species_fracs)}
    xs = {keys[i]: frac for i, (sp, frac) in enumerate(species_fracs.items())}
    return f"x->{ids}={xs}"


def format_target_line(index: int, fields: NewCaseFields) -> str:
    """Serialize a NewCaseFields into one .target pipe-delimited line,
    matching combustion_target_class.combustion_target.__init__'s parser
    (src/MUQ-SAC/combustion_target_class.py:90-340) field-for-field."""
    units = {"conc": "mol", "P": "atm", "T": "K", "observed": "unitless", "flow": ""}
    segments = [
        f"{index} ",
        f"{fields.dataset_id}",
        f" target -- {fields.target_type}",
        f" simulation -- {fields.simulation}",
        f" Fuel_type -- Multi",
        f" Fuel -- {_fuel_multi_repr(fields.fuel)}",
        f" Oxidizer -- x->{fields.oxidizer} = {fields.oxidizer_x}",
        f" Bath_gas -- {_bath_gas_repr(fields.bath_gas)}",
        f" T -- {fields.temperature_K}",
        f" P -- {fields.pressure_atm}",
        f" Phi -- {fields.phi}",
        f" observed -- {fields.observed}",
        f" deviation -- {fields.deviation}",
        f" data_weight -- {fields.data_weight}",
        f" species -- {fields.species}",
        f" units -- {units}",
    ]
    return "|".join(segments)


def append_case(target_path: str, add_path: str, fields: NewCaseFields) -> int:
    """Append a manually-authored case to an existing .target file and its
    paired .add YAML. Returns the 1-based line index used. Creates both
    files (with the target file's leading counter starting at 1) if they
    don't exist yet."""
    tpath = Path(target_path)
    existing_lines: List[str] = []
    if tpath.exists():
        existing_lines = [ln for ln in tpath.read_text().splitlines() if ln.strip()]
    index = len(existing_lines) + 1

    line = format_target_line(index, fields)
    with open(tpath, "a") as fh:
        if existing_lines and not tpath.read_text().endswith("\n"):
            fh.write("\n")
        fh.write(line + "\n")

    add_data = load_add(add_path) if Path(add_path).exists() else {}
    overrides = dict(fields.add_overrides)
    overrides.setdefault("solver", fields.solver)
    add_data = set_case_options(add_data, fields.dataset_id, overrides)
    save_add(add_path, add_data)
    return index


def update_line(target_path: str, add_path: str, line_no: int, fields: NewCaseFields) -> None:
    """Replace the .target line at 1-based ``line_no`` with ``fields``, and
    update the paired .add entry for ``fields.dataset_id`` the same way
    ``append_case`` does (merge overrides, not a full replace -- keys not
    covered by ``fields.add_overrides``/``solver`` keep their existing
    values, falling back to ``combustion_target_class``'s own ADD_DEFAULTS
    cascade at parse time).

    Phase 5a (2026-08 backlog program, bug.md): in-place row editing was
    genuinely missing machinery before this -- append_case/
    write_filtered_target_file only ever add or filter, neither can modify
    an existing row.

    Raises IndexError if line_no is out of range (matches Python's own
    list-index convention rather than silently no-op'ing on a bad index).
    """
    tpath = Path(target_path)
    lines = [ln for ln in tpath.read_text().splitlines() if ln.strip()]
    if not (1 <= line_no <= len(lines)):
        raise IndexError(f"line_no {line_no} out of range (file has {len(lines)} lines)")

    lines[line_no - 1] = format_target_line(line_no, fields)
    tpath.write_text("\n".join(lines) + "\n")

    add_data = load_add(add_path) if Path(add_path).exists() else {}
    overrides = dict(fields.add_overrides)
    overrides.setdefault("solver", fields.solver)
    add_data = set_case_options(add_data, fields.dataset_id, overrides)
    save_add(add_path, add_data)


def delete_line(target_path: str, line_no: int) -> None:
    """Remove the .target line at 1-based ``line_no``.

    Does NOT touch the paired .add file -- .add entries are looked up by
    dataset ID, not line position, and other .target lines may still
    reference the same ID (one dataset ID commonly spans many T/P rows), so
    there's no dangling-reference risk to clean up; an orphaned .add entry
    for an ID no longer present in .target is harmless (simply unused).

    Raises IndexError if line_no is out of range.
    """
    tpath = Path(target_path)
    lines = [ln for ln in tpath.read_text().splitlines() if ln.strip()]
    if not (1 <= line_no <= len(lines)):
        raise IndexError(f"line_no {line_no} out of range (file has {len(lines)} lines)")

    del lines[line_no - 1]
    tpath.write_text("\n".join(lines) + ("\n" if lines else ""))


def write_filtered_target_file(target_path: str, line_numbers: List[int], out_path: str) -> int:
    """Write a copy of target_path containing only the given 1-based line
    numbers (for "run just these selected rows" without mutating the
    source file). Line numbers, not dataset IDs, because one dataset ID
    commonly spans many lines (one experimental series at several T/P
    conditions) -- filtering by ID would pull in every sibling condition
    even when the user selected just one row. Returns lines written."""
    keep = set(int(n) for n in line_numbers)
    written = 0
    with open(target_path) as src, open(out_path, "w") as dst:
        for i, raw in enumerate(src):
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            if (i + 1) in keep:
                dst.write(line + "\n")
                written += 1
    return written


# ---------------------------------------------------------------------------
# Target Workbench curation output (2026-08-04)
# ---------------------------------------------------------------------------

def _next_version(out_dir: Path, out_stem: str) -> int:
    """1-based next unused version for <out_stem>_v<N>.target in out_dir --
    curation output is always versioned, never overwrites a prior version."""
    existing = list(out_dir.glob(f"{out_stem}_v*.target"))
    used = []
    for p in existing:
        suffix = p.stem[len(out_stem) + 2:]  # strip "<out_stem>_v"
        if suffix.isdigit():
            used.append(int(suffix))
    return (max(used) + 1) if used else 1


def write_curated_set(
    target_path: str,
    add_path: str,
    out_stem: str,
    keep_line_numbers: List[int],
    new_cases: Optional[List[NewCaseFields]] = None,
    add_updates: Optional[Dict[str, dict]] = None,
    reason_log: Optional[List[dict]] = None,
    out_dir: Optional[str] = None,
) -> "tuple[str, str, str]":
    """Write a new, versioned ``.target`` + ``.add`` pair from a Target
    Workbench curation session: keep only ``keep_line_numbers`` (1-based,
    from the ORIGINAL ``target_path``), optionally append ``new_cases``
    (e.g. dummy points -- one ``NewCaseFields`` per new row, with its
    composition/solver/target_type already filled in by the caller, since
    this module stays framework-free and does not know about
    ``VisualAid.DummyPointGenerator``'s DataFrame shape), optionally apply
    ``add_updates`` (``{dataset_id: {overrides}}``, e.g. a Phi backfill from
    ``TargetChecker.check_parses``'s ``suggested_fix``), and write a
    ``<out_stem>_v<N>_curation_log.yaml`` recording ``reason_log`` (a list of
    plain dicts -- add/drop/backfill entries with a reason string) alongside.

    NEVER writes to ``target_path``/``add_path`` -- always a new
    ``<out_stem>_v<N>.target``/``.add`` in ``out_dir`` (default: the
    directory containing ``target_path``), where N is the next unused
    integer for ``out_stem`` there, so no prior version is ever clobbered.

    Returns (new_target_path, new_add_path, log_path).
    """
    src_dir = Path(target_path).parent
    dest_dir = Path(out_dir) if out_dir else src_dir
    dest_dir.mkdir(parents=True, exist_ok=True)

    version = _next_version(dest_dir, out_stem)
    new_target_path = dest_dir / f"{out_stem}_v{version}.target"
    new_add_path = dest_dir / f"{out_stem}_v{version}.add"
    log_path = dest_dir / f"{out_stem}_v{version}_curation_log.yaml"

    write_filtered_target_file(target_path, keep_line_numbers, str(new_target_path))

    add_data = load_add(add_path) if Path(add_path).exists() else {}
    save_add(str(new_add_path), add_data)

    for fields in (new_cases or []):
        append_case(str(new_target_path), str(new_add_path), fields)

    if add_updates:
        add_data = load_add(str(new_add_path))
        for dataset_id, overrides in add_updates.items():
            add_data = set_case_options(add_data, dataset_id, overrides)
        save_add(str(new_add_path), add_data)

    log_entries = list(reason_log or [])
    log_entries.append({
        "kept_line_numbers": list(keep_line_numbers),
        "dropped_count_from_source": None,  # caller fills in if it wants a diff count
        "new_cases_added": len(new_cases or []),
        "add_updates_applied": list((add_updates or {}).keys()),
        "source_target": str(target_path),
        "source_add": str(add_path),
    })
    with open(log_path, "w") as fh:
        yaml.safe_dump({"version": version, "entries": log_entries},
                        fh, default_flow_style=False, sort_keys=False)

    return str(new_target_path), str(new_add_path), str(log_path)
