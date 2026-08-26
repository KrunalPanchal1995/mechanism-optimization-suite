"""
rmip/analysis/yaml_safe.py
===========================
A PyYAML loader that will not silently corrupt combustion-mechanism species
names.

Bug this exists to prevent (found 2026-07-22 stress-testing Glarborg 2018
health-check + auto-fix): PyYAML's default (``SafeLoader``) resolver follows
YAML **1.1** ``bool`` semantics, where the bare tokens ``y|Y|yes|Yes|YES|``
``n|N|no|No|NO|on|On|ON|off|Off|OFF|true|false|...`` all implicitly resolve to
a Python ``bool`` instead of ``str``. Nitric oxide's species name is exactly
``NO`` — an extremely common species in any NOx-containing mechanism (every
ammonia/hydrogen mechanism in this repo has one). ``yaml.safe_load`` on a
perfectly valid Cantera YAML (itself written by ``ck2yaml``, which emits the
plain unquoted token ``NO`` in flow-style species lists) silently turns that
one species name into Python ``False``. Anything downstream that re-serializes
the parsed dict (:mod:`rmip.analysis.mechanism_fixer`) then writes a literal
YAML ``false`` into what should be a list of species-name strings, which
Cantera's own (stricter, YAML-1.2-ish) reader rejects outright with a
confusing ``Could not parse species declaration of type 'vector<AnyValue>'``
error that has nothing obviously to do with the real cause.

Cantera's own YAML reader never resolves bare ``NO``/``YES``/``ON``/``OFF``/
``Y``/``N`` as booleans — only the exact lowercase/titlecase spellings of
``true``/``false`` are treated as booleans in Cantera-generated YAML (e.g.
``duplicate: true``). This module's loader matches that narrower behaviour:
only ``true``/``True``/``TRUE``/``false``/``False``/``FALSE`` resolve to
``bool``; every other YAML-1.1 boolean alias (``yes``, ``no``, ``on``, ``off``,
``y``, ``n`` and their case variants) is left as a plain string.

Use :func:`safe_load` / :func:`safe_load_path` anywhere a Cantera-format YAML
(mechanism file, or a dict round-tripped through one) is parsed with PyYAML —
in place of ``yaml.safe_load``.
"""
from __future__ import annotations

import re
from typing import Any, IO, Optional, Union

import yaml

__all__ = ["CanteraSafeLoader", "safe_load", "safe_load_path"]

# Only the Cantera/JSON-style spellings resolve as bool; everything else in
# PyYAML's stock YAML-1.1 bool set (yes/no/on/off/y/n, any case) stays a str.
_BOOL_RE = re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$")


class CanteraSafeLoader(yaml.SafeLoader):
    """``yaml.SafeLoader`` with the bool implicit-resolver narrowed so bare
    ``NO``/``YES``/``ON``/``OFF``/``Y``/``N`` tokens stay strings."""


CanteraSafeLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    _BOOL_RE,
    list("tTfF"),
)
# Drop the stock SafeLoader's broader bool resolver (registered under the
# same tag by PyYAML's base Resolver) so ours is the only one in effect.
CanteraSafeLoader.yaml_implicit_resolvers = {
    key: [(tag, regexp) for tag, regexp in resolvers
          if tag != "tag:yaml.org,2002:bool"] + (
              [("tag:yaml.org,2002:bool", _BOOL_RE)] if key in list("tTfF") else []
          )
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


def safe_load(stream: Union[str, bytes, IO]) -> Any:
    """Drop-in replacement for ``yaml.safe_load`` that will not mangle a
    species literally named ``NO`` (or ``YES``/``ON``/``OFF``/``Y``/``N``)
    into a Python ``bool``."""
    return yaml.load(stream, Loader=CanteraSafeLoader)


def safe_load_path(path: Optional[str]) -> Optional[Any]:
    """``safe_load`` a file path; returns ``None`` on any error, mirroring the
    best-effort ``_try_parse_yaml``/``_load_yaml`` helpers this replaces."""
    if not path:
        return None
    try:
        with open(path, "r") as fh:
            return safe_load(fh)
    except Exception:                                      # noqa: BLE001
        return None
