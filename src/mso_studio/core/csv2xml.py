"""
mso_studio.core.csv2xml
==========================
Adapter over ``rmip.io.respecth_writer.write_respecth_xml`` — "F1: CSV →
ReSpecTh/RKD XML converter" from WORKBENCH_AUDIT_2026-08-10.md, Part F,
and the actual fix for that same audit's defect B11 (the plot digitizer's
own minimal XML writer emits a schema
``config_tools/converters_io/XML2CSV/xml2csv.py`` rejects outright).

Unlike most ``core/*.py`` adapters, this one runs entirely IN-PROCESS, not
via ``science_command``/``run_backend``: ``respecth_writer`` is pure
stdlib+``chem_data.py`` (no Cantera), and the "round-trip check" below
imports ``xml2csv.py`` the same by-file-path way
``rmip.io.respecth_reader._load_xml2csv`` already does — its only real
dependency, ``numpy``, is present in ``.venv-gui`` (confirmed: this is the
same interpreter this module itself runs under), so no anaconda subprocess
bridge is needed here the way it is for Cantera-dependent adapters.

Deliberately lighter-weight than the ``RunRecord``/``RunConfig`` pattern
other adapters use: converting one CSV is an instant, deterministic
transform, not a multi-minute campaign worth its own run-history entry —
matches ``mso_studio.core.digitizer``'s own plain-function style.
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from mso_studio.core.backends import REPO_ROOT
from mso_studio.core.runrecord import RunConfig, RunRecord, new_record

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from rmip.io.respecth_writer import write_respecth_xml, RespecthWriterError  # noqa: E402

_XML2CSV_DIR = REPO_ROOT / "config_tools" / "converters_io" / "XML2CSV"


@dataclass
class Csv2XmlParams:
    csv_path: str
    output_path: str
    target_type: str  # "Tig" | "Fls" | "Flf_a"
    x_quantity: str   # "temperature" | "phi"
    x_units: str = "K"
    y_units: str = ""
    pressure_atm: float = 1.0
    phi: Optional[float] = None            # required if x_quantity == "temperature"
    temperature_K: Optional[float] = None  # required if x_quantity == "phi"
    fuel: Dict[str, float] = field(default_factory=dict)
    bath_gas_ratio: float = 3.76
    bath_gas_species: str = "N2"
    dataset_id: str = ""
    ignition_target: str = ""
    ignition_type: str = "max"
    bibliography: Optional[Dict[str, str]] = None

    def to_metadata(self) -> dict:
        fixed = {"pressure_atm": self.pressure_atm}
        if self.phi is not None:
            fixed["phi"] = self.phi
        if self.temperature_K is not None:
            fixed["temperature_K"] = self.temperature_K
        metadata = {
            "target_type": self.target_type,
            "x_quantity": self.x_quantity,
            "x_units": self.x_units,
            "y_units": self.y_units,
            "fixed": fixed,
            "composition": {
                "fuel": self.fuel,
                "bath_gas_ratio": self.bath_gas_ratio,
                "bath_gas_species": self.bath_gas_species,
            },
        }
        if self.dataset_id:
            metadata["dataset_id"] = self.dataset_id
        if self.ignition_target:
            metadata["ignition"] = {"target": self.ignition_target, "type": self.ignition_type}
        if self.bibliography:
            metadata["bibliography"] = self.bibliography
        return metadata


@dataclass
class ConvertResult:
    ok: bool
    xml_path: str = ""
    xml_preview: str = ""
    error: str = ""


def convert(params: Csv2XmlParams) -> ConvertResult:
    """Write the ReSpecTh XML for *params*. Never raises -- a bad
    metadata/CSV combination (e.g. an unrecognized fuel with no known
    stoichiometric O2 demand) is a normal, expected user-facing outcome,
    not an infrastructure failure, matching ``converter.run_convert``'s and
    ``fusion.run_fusion``'s own "return a result, don't raise" convention."""
    try:
        xml_str = write_respecth_xml(params.csv_path, params.to_metadata(), params.output_path)
    except (RespecthWriterError, OSError, ValueError) as exc:
        return ConvertResult(ok=False, error=str(exc))
    return ConvertResult(ok=True, xml_path=params.output_path, xml_preview=xml_str)


@dataclass
class VerifyResult:
    ok: bool
    n_files_ok: int = 0
    n_rows: int = 0
    target_preview: str = ""
    warnings: List[str] = field(default_factory=list)
    error: str = ""


def _load_xml2csv():
    """Import xml2csv.py by file path -- same pattern
    ``rmip.io.respecth_reader._load_xml2csv`` already uses for the same
    module, kept as a separate copy here since that function isn't a
    public API and this adapter has no other reason to depend on
    ``rmip.io.respecth_reader``."""
    if str(_XML2CSV_DIR) not in sys.path:
        sys.path.append(str(_XML2CSV_DIR))
    spec = importlib.util.spec_from_file_location(
        "mso_studio_vendored_xml2csv", str(_XML2CSV_DIR / "xml2csv.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def verify_with_xml2csv(xml_path: str, fuel_list: Optional[List[str]] = None) -> VerifyResult:
    """Run the produced XML through the REAL xml2csv.py converter and
    report whether it actually parses into target rows -- proving the file
    isn't just well-formed XML but genuinely consumable by the rest of the
    pipeline, before the user leaves this page. *fuel_list* should include
    whatever fuel species the XML's composition actually names (default
    ``["NH3"]``, matching xml2csv.py's own CLI default) -- xml2csv.py's
    composition parser only recognizes species in this list as "the fuel",
    silently miscategorizing anything else as bath gas otherwise."""
    import tempfile

    try:
        mod = _load_xml2csv()
    except Exception as exc:  # pragma: no cover - missing/broken vendored module
        return VerifyResult(ok=False, error=f"could not load xml2csv.py: {exc}")

    xml_path = Path(xml_path).resolve()
    with tempfile.TemporaryDirectory() as tmp_xml_dir, tempfile.TemporaryDirectory() as tmp_out_dir:
        (Path(tmp_xml_dir) / xml_path.name).write_bytes(xml_path.read_bytes())
        try:
            result = mod.main(tmp_xml_dir, tmp_out_dir, fuel_list=fuel_list or ["NH3"])
        except Exception as exc:
            return VerifyResult(ok=False, error=f"xml2csv.py raised: {exc}")

        target_path = result.get("target_path") if isinstance(result, dict) else None
        n_rows = result.get("n_rows", 0) if isinstance(result, dict) else 0
        n_files_ok = result.get("n_files_ok", 0) if isinstance(result, dict) else 0
        warnings = result.get("warnings", []) if isinstance(result, dict) else []
        preview = ""
        if target_path and Path(target_path).exists():
            preview = Path(target_path).read_text()[:2000]
        return VerifyResult(
            ok=bool(n_files_ok) and bool(n_rows),
            n_files_ok=n_files_ok, n_rows=n_rows,
            target_preview=preview, warnings=list(warnings),
        )


def run_csv2xml(params: Csv2XmlParams, *, verify: bool = True,
                 fuel_list: Optional[List[str]] = None,
                 on_line=None, cancel_event=None) -> RunRecord:
    """``ModulePage``-compatible entry point: wraps :func:`convert` (and, if
    *verify* is set, :func:`verify_with_xml2csv`) in a :class:`RunRecord`,
    matching every other module page's ``build_core_call()`` contract
    (``ModulePage`` always runs this via a background ``QThread`` regardless
    of whether it's a subprocess or not — see ``module_page.py``'s
    ``_on_run_clicked``, ``run_in_thread(core_fn, ...)`` — so running
    in-process here needs no special-casing).

    *on_line*/*cancel_event* are accepted for interface parity with every
    other ``core.*.run_x`` adapter but unused: this conversion is a single
    fast, synchronous, non-cancellable operation, not a long subprocess.
    """
    run_config = RunConfig(
        module="csv2xml",
        label=params.dataset_id or Path(params.csv_path).stem,
        backend_command=["(in-process)", "rmip.io.respecth_writer.write_respecth_xml"],
        working_dir=str(Path(params.output_path).parent),
        inputs={"csv_path": params.csv_path},
        parameters={
            "target_type": params.target_type, "x_quantity": params.x_quantity,
            "fuel": params.fuel, "bath_gas_ratio": params.bath_gas_ratio,
        },
    )
    record = new_record(run_config)
    record.capture_input_hashes()

    convert_result = convert(params)
    if not convert_result.ok:
        record.mark_finished("failed", 1, error=convert_result.error)
        record.save()
        return record

    record.results["xml_path"] = convert_result.xml_path
    record.results["xml_preview"] = convert_result.xml_preview

    if verify:
        vresult = verify_with_xml2csv(convert_result.xml_path, fuel_list=fuel_list)
        record.results["verify_ok"] = vresult.ok
        record.results["verify_n_rows"] = vresult.n_rows
        record.results["verify_target_preview"] = vresult.target_preview
        record.warnings = vresult.warnings
        if not vresult.ok and not vresult.error:
            record.warnings.append(
                "XML written, but xml2csv.py produced 0 target rows from it — "
                "check that 'fuel' names a species in fuel_list.")

    record.mark_finished("done", 0)
    record.save()
    return record
