"""
rmip.io.importers — W3 foreign-format mechanism importers (P5).

Pluggable importer framework: each backend implements
:class:`~rmip.io.importers.base.MechanismImporter` (``sniff`` + ``import_``);
:mod:`~rmip.io.importers.registry` dispatches by format detection.

Backends:
  * :mod:`.opensmoke` — OpenSmoke-flavoured CHEMKIN with LIQUID/MATERIAL/SOLID
    condensed-phase extensions (the P5 polyethylene-pyrolysis case).
"""
from rmip.io.importers.registry import detect_format, import_mechanism  # noqa: F401
