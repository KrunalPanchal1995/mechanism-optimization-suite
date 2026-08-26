"""
mso_project.footprints._generic
================================
Shared helpers used by every footprint module and by the classifier in
``footprints/__init__.py``: glob matching (including a ``**`` recursive
form, since ``fnmatch`` alone treats ``**`` no differently from ``*``) and
extension-based ``kind`` inference for files no footprint pattern covers.
"""
from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Optional

_KIND_BY_EXT = {
    ".csv": "csv", ".tsv": "csv",
    ".png": "png", ".jpg": "png", ".jpeg": "png", ".svg": "png",
    ".pdf": "pdf",
    ".pkl": "pkl", ".pickle": "pkl",
    ".yaml": "yaml", ".yml": "yaml",
    ".json": "json",
    ".txt": "txt", ".save": "txt",
    ".log": "log", ".out": "log", ".lst": "log",
    ".md": "doc", ".rst": "doc", ".docx": "doc", ".tex": "doc",
    ".ipynb": "notebook",
    ".inp": "other", ".dat": "other", ".mech": "other",
}


def kind_of(path: str) -> str:
    return _KIND_BY_EXT.get(Path(path).suffix.lower(), "other")


def glob_match(rel_path: str, pattern: str) -> bool:
    """Match a path relative to a run's working_dir against a footprint
    pattern. Supports a standalone ``**`` PATH SEGMENT (matches zero or
    more whole path components, standard globstar semantics -- e.g.
    ``a/**/b`` matches ``a/b``, ``a/x/b``, and ``a/x/y/b`` alike, unlike
    bare ``fnmatch`` where ``**`` behaves identically to ``*`` and refuses
    to cross a ``/`` at all) in addition to ordinary single-segment
    ``*``/``?``/``[...]`` glob syntax within each segment.
    """
    rel_path = rel_path.replace("\\", "/")
    pattern = pattern.replace("\\", "/")
    if "**" not in pattern.split("/"):
        # No standalone "**" segment (a literal like "foo**bar" isn't
        # treated specially -- only a whole "**" path component is) ->
        # plain fnmatch is exact and simpler.
        return fnmatch.fnmatch(rel_path, pattern)

    # Segment-by-segment translation. A "**" segment becomes "(?:.*/)?" --
    # NOT "(?:[^/]+/)*" -- so that the slash separating it from its
    # PRECEDING literal segment (added explicitly below, once, when that
    # segment isn't the last) is the only mandatory one, and "**" plus its
    # own trailing slash collapses to optional as a unit. This is what
    # makes the zero-intervening-directories case work: "a/**/b" must
    # match "a/b" (one slash total) as well as "a/x/b" (two slashes) --
    # gluing a separate mandatory "/" on BOTH sides of "**" (this
    # function's first, buggy implementation) can never produce the
    # one-slash form.
    segments = pattern.split("/")
    n = len(segments)
    regex = ""
    for i, seg in enumerate(segments):
        if seg == "**":
            regex += ".*" if i == n - 1 else "(?:.*/)?"
        else:
            regex += fnmatch.translate(seg)[:-2]
            if i < n - 1:
                regex += "/"
    regex += r"\Z"
    return re.match(regex, rel_path) is not None
