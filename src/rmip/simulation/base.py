"""
rmip/simulation/base.py
=======================
Base infrastructure for all Cantera-based simulation runners.

Responsibilities
----------------
1. Load a Cantera `Solution` object from a mechanism stored in the RMIP DB.
   - cantera_yaml  : direct ct.Solution(path)
   - chemkin       : auto-convert with ck2yaml to a temp YAML, then load
   - cti           : auto-convert with ctml2yaml, then load
2. Provide `_load_solution()` as a protected method for subclasses.
3. Define the `AbstractSimRunner` protocol that IDTRunner and LFSRunner follow.

Design decisions
----------------
- We store the original mechanism file path in `mechanisms.filepath`.
  For CHEMKIN files the thermo path (if separate) is stored in metadata_json
  under the key "thermo_filepath".
- Converted YAML files are cached in a subdirectory of the DB's parent folder
  (`<db_dir>/.rmip_cantera_cache/`) to avoid redundant conversions.
- Transport data is deliberately not passed to Cantera (design decision D6).
  Cantera will load with `transport_model=None` for CHEMKIN mechanisms.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cantera import with informative error
# ---------------------------------------------------------------------------

try:
    import cantera as ct
    _HAS_CANTERA = True
except ImportError:
    ct = None  # type: ignore[assignment]
    _HAS_CANTERA = False


def _require_cantera() -> None:
    if not _HAS_CANTERA:
        raise ImportError(
            "Cantera is required for simulation modules. "
            "Install it via:  conda install -c cantera cantera"
        )


# ---------------------------------------------------------------------------
# Mechanism loader
# ---------------------------------------------------------------------------

class MechanismLoader:
    """
    Loads a Cantera `Solution` from a mechanism record in the RMIP database.

    Parameters
    ----------
    db_path : str
        Path to the RMIP SQLite database.
    cache_dir : str | None
        Directory for converted YAML files.  Defaults to ``<db_dir>/.rmip_cantera_cache``.
    """

    def __init__(self, db_path: str, cache_dir: Optional[str] = None) -> None:
        self._db_path  = db_path
        self._cache_dir = Path(cache_dir) if cache_dir else (
            Path(db_path).parent / ".rmip_cantera_cache"
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, mechanism_id: int) -> "ct.Solution":
        """
        Return a Cantera `Solution` for the given mechanism_id.
        Conversion from CHEMKIN/CTI is performed automatically on first call
        and cached for subsequent calls.
        """
        _require_cantera()
        rec = self._fetch_mechanism_record(mechanism_id)
        fmt      = rec["format"]          # 'chemkin' | 'cantera_yaml' | 'cti'
        filepath = rec["filepath"]
        meta     = json.loads(rec["metadata_json"] or "{}")

        if fmt in ("cantera_yaml", "reduced"):
            # 'reduced' = a materialize_subset() skeletal/lumped/QSSA mechanism;
            # its filepath points at the companion YAML written at materialise time.
            logger.debug("Loading Cantera YAML: %s", filepath)
            return ct.Solution(filepath)

        if fmt == "cti":
            yaml_path = self._convert_cti_to_yaml(filepath)
            return ct.Solution(str(yaml_path))

        if fmt == "chemkin":
            thermo_path = meta.get("thermo_filepath")
            yaml_path   = self._convert_chemkin_to_yaml(filepath, thermo_path)
            return ct.Solution(str(yaml_path))

        raise ValueError(f"Unknown mechanism format '{fmt}' for mechanism_id={mechanism_id}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_mechanism_record(self, mechanism_id: int) -> Dict[str, Any]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT id, name, format, filepath, metadata_json "
                "FROM mechanisms WHERE id=?",
                (mechanism_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"No mechanism with id={mechanism_id} in DB")
            return dict(row)
        finally:
            conn.close()

    @staticmethod
    def _ck2yaml_convert_mech(mech_path: str, thermo_path: Optional[str], yaml_path: Path) -> None:
        """cantera.ck2yaml.convert_mech -- confirmed present on Cantera 3.0.1."""
        from cantera import ck2yaml
        ck2yaml.convert_mech(
            mech_path, thermo_file=thermo_path, out_name=str(yaml_path), quiet=True,
        )

    @staticmethod
    def _ck2yaml_convert(mech_path: str, thermo_path: Optional[str], yaml_path: Path) -> None:
        """cantera.ck2yaml.convert -- seen on some other Cantera 3.x builds."""
        from cantera.ck2yaml import convert
        convert(
            input=mech_path, thermo=thermo_path, transport=None, surface=None,
            out_name=str(yaml_path), quiet=True,
        )

    @staticmethod
    def _ck2yaml_module_convert(mech_path: str, thermo_path: Optional[str], yaml_path: Path) -> None:
        """Older Cantera 2.x module-level ck2yaml.convert."""
        from cantera import ck2yaml  # type: ignore[attr-defined]
        ck2yaml.convert(input=mech_path, thermo_file=thermo_path, out_name=str(yaml_path))

    def _convert_chemkin_to_yaml(
        self,
        mech_path:   str,
        thermo_path: Optional[str],
    ) -> Path:
        """
        Convert a CHEMKIN mechanism to Cantera YAML.
        The result is cached by a hash of the input file(s).
        """
        # M20: the cache key must capture (a) the mechanism content, (b) the
        # thermo content AND whether a thermo path was requested at all (a
        # requested-but-missing thermo is a different input than no thermo), and
        # (c) the Cantera version -- ck2yaml output format changes across
        # versions, so a cache built by another Cantera must not be reused.
        import cantera as _ct
        hash_src = _file_hash(mech_path)
        hash_src += f"|thermo_arg={thermo_path or ''}"
        if thermo_path and os.path.exists(thermo_path):
            hash_src += "|thermo=" + _file_hash(thermo_path)
        hash_src += f"|cantera={_ct.__version__}"
        cache_key  = hashlib.md5(hash_src.encode()).hexdigest()[:12]
        yaml_path  = self._cache_dir / f"chemkin_{cache_key}.yaml"

        if yaml_path.exists():
            logger.debug("Using cached YAML conversion: %s", yaml_path)
            return yaml_path

        logger.info("Converting CHEMKIN → YAML: %s", mech_path)
        # M20: convert into a unique temp file and os.replace() into place only
        # on success, so a crash / kill mid-conversion can never leave a
        # truncated YAML permanently cached at the final path.
        tmp_yaml = yaml_path.with_name(f".{yaml_path.stem}.{os.getpid()}.tmp.yaml")
        # Cantera's ck2yaml entry point has changed name across versions
        # (convert_mech on 3.0.1, convert on some other 3.x builds, module-path
        # ck2yaml.convert on older 2.x). Try each in turn rather than assuming
        # one -- confirmed empirically that only convert_mech exists here.
        last_exc: Optional[Exception] = None
        try:
            for attempt in (self._ck2yaml_convert_mech, self._ck2yaml_convert,
                            self._ck2yaml_module_convert):
                try:
                    attempt(mech_path, thermo_path, tmp_yaml)
                    last_exc = None
                    break
                except (ImportError, AttributeError) as exc:
                    last_exc = exc
                    continue
                except Exception as exc:
                    raise RuntimeError(
                        f"CHEMKIN→YAML conversion failed: {exc}\n"
                        f"  mechanism : {mech_path}\n"
                        f"  thermo    : {thermo_path}"
                    ) from exc
            if last_exc is not None:
                raise RuntimeError(
                    f"ck2yaml conversion failed for {mech_path}.\n"
                    "None of the known cantera.ck2yaml entry points "
                    "(convert_mech/convert/module-convert) exist on this Cantera "
                    "install. Please convert to Cantera YAML manually with:\n"
                    f"  ck2yaml --input={mech_path} "
                    f"{'--thermo=' + thermo_path if thermo_path else ''} "
                    f"--output={yaml_path}\n"
                    f"Last error: {last_exc}"
                ) from last_exc

            os.replace(tmp_yaml, yaml_path)
        finally:
            if tmp_yaml.exists():
                try:
                    tmp_yaml.unlink()
                except OSError:
                    pass

        logger.info("Saved converted YAML: %s", yaml_path)
        return yaml_path

    def _convert_cti_to_yaml(self, cti_path: str) -> Path:
        """Convert a CTI file to Cantera YAML."""
        import cantera as _ct
        hash_src = _file_hash(cti_path) + f"|cantera={_ct.__version__}"
        cache_key = hashlib.md5(hash_src.encode()).hexdigest()[:12]
        yaml_path = self._cache_dir / f"cti_{cache_key}.yaml"

        if yaml_path.exists():
            return yaml_path

        logger.info("Converting CTI → YAML: %s", cti_path)
        # M20: atomic write -- convert to a temp file and os.replace() on success
        # so a crash mid-conversion can't cache a truncated YAML forever.
        tmp_yaml = yaml_path.with_name(f".{yaml_path.stem}.{os.getpid()}.tmp.yaml")
        try:
            from cantera.ctml2yaml import convert as ctml_convert  # type: ignore
            ctml_convert(cti_path, str(tmp_yaml))
            os.replace(tmp_yaml, yaml_path)
        except Exception as exc:
            raise RuntimeError(f"CTI→YAML conversion failed: {exc}") from exc
        finally:
            if tmp_yaml.exists():
                try:
                    tmp_yaml.unlink()
                except OSError:
                    pass

        return yaml_path


def _file_hash(path: str) -> str:
    """Return hex MD5 of a file."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Abstract simulation runner
# ---------------------------------------------------------------------------

class AbstractSimRunner(ABC):
    """
    Abstract base class for IDTRunner and LFSRunner.

    Subclasses must implement `run_single` and `_detect_result`.
    """

    def __init__(
        self,
        mechanism_id: int,
        db_path:      str,
        cache_dir:    Optional[str] = None,
    ) -> None:
        _require_cantera()
        self.mechanism_id = mechanism_id
        self._db_path     = db_path
        self._loader      = MechanismLoader(db_path, cache_dir)
        self._gas: Optional["ct.Solution"] = None

    @property
    def gas(self) -> "ct.Solution":
        """Lazily-loaded Cantera Solution (cached after first load)."""
        if self._gas is None:
            self._gas = self._loader.load(self.mechanism_id)
        return self._gas

    def reset_gas(self) -> None:
        """Force a fresh gas object (use after a failed simulation)."""
        self._gas = None

    @abstractmethod
    def run_single(self, *args: Any, **kwargs: Any) -> Any:
        """Run one simulation and return the result dataclass."""
        ...
