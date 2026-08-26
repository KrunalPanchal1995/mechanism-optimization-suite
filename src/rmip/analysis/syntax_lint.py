"""
rmip/analysis/syntax_lint.py
=============================
ChemCheck-style syntax linter (P9 in ``MECHANISM_CHECK_RESEARCH.md`` §4) — a
raw, line-by-line text scan for common CHEMKIN ``.inp`` and Cantera-YAML
syntax defects, each reported with an accurate 1-indexed line number and
(where determinable) a 1-indexed column.

This is deliberately **not** a parser: it is a plain-text scan, so it works
even when the file fails to load in Cantera/ck2yaml (undeclared species, a
malformed block, ...) — exactly the situation where ``MechanismChecker``'s
only existing signal is ck2yaml's raw, sometimes noisy, stderr passthrough
(``_clean_cantera_error``). A handful of precise, well-scoped heuristics:

  (a) missing-exponent scientific notation — the classic CHEMKIN Fortran
      "1.7-13" (meant "1.7E-13") gotcha: a decimal-point mantissa immediately
      followed by a bare +/- and 1-3 more digits, with no e/E in between.
      Requiring the decimal point and a non-word delimiter right after the
      exponent digits (so "2500.0" or a "300-500" integer range never
      match) keeps this heuristic very low false-positive.
  (b) a stray tab character on a non-comment line.
  (c) a numeric field truncated mid-token: a trailing bare '.', or a
      trailing bare 'E'/'e'/'D'/'d' (with or without a lone following sign)
      and no digits after it.
  (d) a suspicious/illegal character in a species-name token of what looks
      like a reaction-equation line (one containing '<=>', '=>' or '=').

Comments are skipped: CHEMKIN '!' and YAML '#' (and anything after them on
a line, since CHEMKIN allows trailing inline comments).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class LintIssue:
    line: int
    column: Optional[int]
    severity: str                      # "error" | "warning"
    message: str


# ---------------------------------------------------------------------------
# (a) missing-exponent scientific notation
# ---------------------------------------------------------------------------
# A decimal mantissa (needs the '.' — an integer range like "300-500" is not
# a rate-constant literal) immediately followed by a bare sign and 1-3
# digits, with no 'e'/'E' in between, and not immediately followed by
# another digit or '.' (which would mean it's part of a longer number, e.g.
# the "2500.0" in a T-range and not a truncated exponent).
_MISSING_EXP_RE = re.compile(r'(?<![\w.])(\d+\.\d+)([+-]\d{1,3})(?![\w.])')
# whole-token form of the same shape, used by _equation_part below so a
# malformed-exponent rate constant is still recognised as "a numeric field"
# and stripped from the equation text (otherwise it leaks into the species-
# token scan and trips heuristic (d) as a spurious side effect of (a)).
_MISSING_EXP_TOKEN_RE = re.compile(r'^\d+\.\d+[+-]\d{1,3}$')

# (c) truncated numeric tokens
_TRAILING_DOT_RE = re.compile(r'^\d+\.$')
_TRAILING_EXP_RE = re.compile(r'^\d*\.?\d+[eEdD]$')
_TRAILING_EXP_SIGN_RE = re.compile(r'^\d*\.?\d+[eEdD][+-]$')

# A mapping key whose value opens a YAML literal (|) or folded (>) block
# scalar, optionally with chomping/indentation indicators (-, +, digits),
# e.g. 'note: |-', '  description: |2-', 'note: >-'. Group 1 captures the
# leading indentation so the body's indent threshold can be derived.
_YAML_BLOCK_SCALAR_KEY_RE = re.compile(r'^(\s*)[\w.\-]+:\s*[|>][0-9+\-]*\s*$')

# 'note:'/'description:' with an inline value on the same line (not a block
# scalar -- that's the pattern above): free-form bibliography text, never
# mechanism data. Matches e.g. 'note: Tsang and Hampson ... (1986)' or
# "description: 'quoted text'".
_YAML_NOTE_KEY_RE = re.compile(r'^\s*(note|description):\s+\S')

# (d) reaction-equation detection + species-token charset
_ARROW_RE = re.compile(r'<=>|=>|=')
_NUMERIC_TOKEN_RE = re.compile(r'^[+-]?(\d+\.?\d*|\.\d+)([eEdD][+-]?\d+)?$')
# Species names in CHEMKIN/Cantera commonly use letters, digits, and a small
# set of punctuation (radicals '*', isomers '#', locants/parens, '-', '_').
_SPECIES_TOKEN_RE = re.compile(r'^[A-Za-z0-9()*#\-_]+$')


def _strip_comment(line: str) -> str:
    for marker in ("!", "#"):
        idx = line.find(marker)
        if idx != -1:
            return line[:idx]
    return line


class SyntaxLinter:
    """Raw line-by-line text scan for CHEMKIN ``.inp`` / Cantera YAML syntax
    defects. Works even when the file cannot be parsed at all."""

    def __init__(self, path: str) -> None:
        self.path = path

    def lint(self) -> List[LintIssue]:
        issues: List[LintIssue] = []
        try:
            with open(self.path, "r", errors="replace") as fh:
                raw_lines = fh.readlines()
        except OSError:
            return issues

        # A Cantera YAML's free-form bibliography/description text lives in
        # `note: |-`-style literal/folded block scalars, whose continuation
        # lines carry no '#' comment marker of their own (YAML doesn't need
        # one — indentation alone marks them as part of the block). Without
        # tracking that, journal citations like "J Phys Chem A 2013." or
        # "17:2069-2082, 2011." get scanned as if they were mechanism data
        # and misfire heuristic (c)/(a) ("truncated numeric field" / missing
        # exponent) hundreds of times over on a real mechanism (found
        # 2026-07-22 sweeping Glarborg 2018 and Konnov 2019's health-check
        # reports — the false positives dwarfed the real findings). Only
        # relevant for .yaml/.yml input; a CHEMKIN .inp has no block-scalar
        # syntax so this state machine simply never triggers there.
        is_yaml = self.path.lower().endswith((".yaml", ".yml"))
        block_indent: Optional[int] = None   # indent of the body once inside one

        for i, raw_line in enumerate(raw_lines):
            lineno = i + 1
            line = raw_line.rstrip("\n").rstrip("\r")
            lead = line.lstrip()
            indent = len(line) - len(lead)

            if is_yaml and block_indent is not None:
                if lead and indent < block_indent:
                    block_indent = None          # dedented back out: block over
                else:
                    continue                     # still inside the block body

            if not lead or lead.startswith(("!", "#")):
                continue

            if is_yaml:
                m = _YAML_BLOCK_SCALAR_KEY_RE.match(line)
                if m:
                    key_indent = len(m.group(1))
                    # The body's indent isn't known until the first body line
                    # appears; any indent strictly greater than the key's own
                    # counts (YAML requires >=1 extra level, exact amount is
                    # the block's own business). Use key_indent+1 as the
                    # threshold so a body indented by just one space still
                    # counts as "inside".
                    block_indent = key_indent + 1
                    continue
                if _YAML_NOTE_KEY_RE.match(line):
                    # An INLINE 'note: <free-form citation text>' (as opposed
                    # to the block-scalar 'note: |' form above) is just as
                    # much free-form bibliography as a block body -- e.g.
                    # 'note: Tsang and Hampson, J. Phys. Chem. Ref. Data,
                    # 15:1087 (1986).' ends in a bare "1986)." that heuristic
                    # (c) reads as a truncated numeric field, and citation
                    # text routinely contains '=' or reads like a reaction
                    # equation to heuristic (d). The value is never mechanism
                    # data, so skip the whole line outright.
                    continue

            # (b) stray tab — checked on the *original* line, before comment
            # stripping, so a tab preceding an inline comment is still seen.
            issues.extend(self._check_stray_tab(line, lineno))

            code = _strip_comment(line)
            if not code.strip():
                continue

            issues.extend(self._check_missing_exponent(code, lineno))
            issues.extend(self._check_truncated_numeric(code, lineno))
            issues.extend(self._check_species_charset(code, lineno))
        return issues

    # -- (a) ----------------------------------------------------------
    @staticmethod
    def _check_missing_exponent(code: str, lineno: int) -> List[LintIssue]:
        out = []
        for m in _MISSING_EXP_RE.finditer(code):
            out.append(LintIssue(
                line=lineno, column=m.start() + 1, severity="error",
                message=(f"'{m.group(0)}' looks like a scientific-notation "
                         "literal missing its 'e'/'E' exponent marker (did "
                         f"you mean '{m.group(1)}e{m.group(2)}'?)."),
            ))
        return out

    # -- (b) ------------------------------------------------------------
    @staticmethod
    def _check_stray_tab(line: str, lineno: int) -> List[LintIssue]:
        idx = line.find("\t")
        if idx == -1:
            return []
        return [LintIssue(
            line=lineno, column=idx + 1, severity="warning",
            message="Stray tab character on a data line (use spaces — some "
                    "CHEMKIN parsers mis-tokenize tabs).",
        )]

    # -- (c) --------------------------------------------------------------
    @staticmethod
    def _check_truncated_numeric(code: str, lineno: int) -> List[LintIssue]:
        out = []
        for m in re.finditer(r'\S+', code):
            tok = m.group(0).rstrip(",;")
            if (_TRAILING_DOT_RE.match(tok) or _TRAILING_EXP_RE.match(tok)
                    or _TRAILING_EXP_SIGN_RE.match(tok)):
                out.append(LintIssue(
                    line=lineno, column=m.start() + 1, severity="error",
                    message=f"Numeric field '{tok}' looks truncated (ends "
                            "mid-token).",
                ))
        return out

    # -- (d) ----------------------------------------------------------
    def _check_species_charset(self, code: str, lineno: int) -> List[LintIssue]:
        arrow_m = _ARROW_RE.search(code)
        if arrow_m is None:
            return []
        # YAML writes reaction equations as 'key: value' (e.g.
        # '- equation: 2 O + M <=> O2 + M'); only the text after the last
        # colon *before* the arrow is chemistry -- the key name itself
        # ('- equation', 'third-body-efficiency', ...) is not a species
        # token and must not be scanned. A bare CHEMKIN .inp line has no
        # such colon, so scan_start stays 0 and the whole line is used.
        prefix = code[:arrow_m.start()]
        colon_idx = prefix.rfind(":")
        scan_start = colon_idx + 1 if colon_idx != -1 else 0
        scan_code = code[scan_start:]

        eq_part = self._equation_part(scan_code)
        if not eq_part:
            return []
        out = []
        search_from = scan_start
        for side in re.split(r'<=>|=>|=', eq_part):
            # Falloff third-body notation '(+M)' / '(+AR)' carries its own
            # '+' *inside* the parens, which would otherwise corrupt the
            # plain '+'-split below (e.g. 'OH (+M)' -> 'OH (', 'M)'). Drop
            # the whole wrapper first; it is legitimate syntax, not a
            # species token to validate.
            side_clean = re.sub(r'\(\+[^)]*\)', ' ', side)
            for term in side_clean.split("+"):
                t = term.strip()
                if not t:
                    continue
                # drop a leading numeric stoichiometric coefficient
                parts = t.split(None, 1)
                if len(parts) == 2 and _NUMERIC_TOKEN_RE.match(parts[0]):
                    t = parts[1].strip()
                if not t or t.upper() == "M":
                    continue
                if _NUMERIC_TOKEN_RE.match(t) or _SPECIES_TOKEN_RE.match(t):
                    continue
                idx = code.find(t, search_from)
                col = idx + 1 if idx != -1 else None
                if idx != -1:
                    search_from = idx + len(t)
                out.append(LintIssue(
                    line=lineno, column=col, severity="warning",
                    message=(f"Species-name token '{t}' contains suspicious "
                             "character(s) for a CHEMKIN/YAML species name."),
                ))
        return out

    @staticmethod
    def _equation_part(code: str) -> str:
        """Strip trailing whitespace-separated numeric rate-constant fields
        (CHEMKIN ``.inp`` puts A, b, Ea right after the equation on the same
        line) so only the equation text is scanned for species tokens."""
        tokens = code.split()
        while tokens:
            last = tokens[-1].rstrip(",;")
            if _NUMERIC_TOKEN_RE.match(last) or _MISSING_EXP_TOKEN_RE.match(last):
                tokens.pop()
            else:
                break
        return " ".join(tokens)
