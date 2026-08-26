"""
rmip/io/chemkin_preflight.py
============================
W2 — CHEMKIN structural preflight linter.

Closes community problems **P2** (``ck2yaml`` dies with cryptic Python
tracebacks on odd whitespace / Fortran exponents), **P3** (``NONCON`` surface
keyword has no Cantera semantics), **P13** (``bulk`` phase keyword unrecognised),
**P18** (no "is my file even structured correctly?" pre-flight), and **P35**
(the same species written two ways across sections — ``AR`` vs ``Ar`` — which
CHEMKIN then rejects). See ``Cantera google groups/W2_chemkin_preflight_converter.md``.

This is a **pure text-level lexer** — it does *not* depend on a successful parse,
so it can explain *why* a file will not convert, with a **line number**, instead
of surfacing a `ck2yaml` traceback. It reuses the shipped `Finding`/`CheckReport`
model (`rmip.analysis.mechanism_check`) so its output renders as JSON + coloured
text and plugs into the same tooling.

Checks
------
- **Block structure** — ELEMENTS/SPECIES/THERMO/REACTIONS presence & `END` closure.
- **Species name consistency (P35)** — case-variant spellings of the same species
  across the SPECIES block, thermo records, and reaction equations (ERROR); tokens
  used in reactions but never declared (WARNING).
- **Whitespace / exponent hazards (P2)** — CRLF, tab characters, and Fortran-style
  exponents (``1.7-13`` missing the ``E``), each locatable and auto-fixable.
- **Ambiguous surface constructs (P3, P13)** — ``NONCON``, ``BULK``, ``SITE``,
  ``MATERIAL`` flagged explicitly rather than left to crash `ck2yaml`.
- **Ea unit declaration** — the REACTIONS header's unit keywords (the cal/mol vs
  J/kmol trap at source).

The ``--fix`` path writes a cleaned ``.inp`` (CRLF→LF, Fortran exponents, and —
opt-in — case-normalised species) that then converts cleanly.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Optional, Set, Tuple

from rmip.analysis.mechanism_check import Finding, FixOption, CheckReport

log = logging.getLogger("rmip.io.chemkin_preflight")

# section-opening keywords (first token, case-insensitive)
_ELEM = {"ELEMENTS", "ELEM"}
_SPEC = {"SPECIES", "SPEC", "SPECS"}
_THERMO = {"THERMO", "THERM", "THER"}
_REAC = {"REACTIONS", "REAC"}
_SURFACE = {"SITE", "BULK", "MATERIAL"}          # surface/heterogeneous openers
_OPENERS = _ELEM | _SPEC | _THERMO | _REAC | _SURFACE

# reaction auxiliary keywords / lines that are NOT reaction equations
_AUX_KEYWORDS = {"DUPLICATE", "DUP", "LOW", "HIGH", "TROE", "SRI", "PLOG",
                 "REV", "FORD", "RORD", "CHEB", "TCHEB", "PCHEB", "UNITS",
                 "END", "LT", "RLT", "JAN", "FIT1", "EXCI", "MOME", "XSMI"}

# recognised REACTIONS-header unit tokens
_EA_UNITS = {"CAL/MOLE", "KCAL/MOLE", "JOULES/MOLE", "KJOULES/MOLE",
             "KELVINS", "EVOLTS"}
_QTY_UNITS = {"MOLES", "MOLECULES"}

# Reaction-equation tokens that look like a species (start with a letter,
# not an aux keyword) but are actually bookkeeping placeholders with no real
# thermo data -- e.g. "HV" for a chemiluminescence photon. Auto-inserting
# these into the SPECIES block would trade "undeclared species" for a new
# ck2yaml failure (missing thermo for HV). Left as a plain warning instead of
# an auto-fix; see rmip.io.chemkin_parser's own note about dropping these
# via allow_pseudo_species=True.
_PSEUDO_SPECIES = {"HV"}

# Fortran exponent missing the E:  "1.86-04" / "4.87+13"  (not "1.86E-04").
# Matched only against a *whole whitespace-delimited token* (see
# _iter_fortran_exp_tokens / _fix_fortran) that is entirely numeric --
# digits and at most one '.' -- plus the bare exponent sign at the end.
# This must NOT be a char-class/lookaround match against the raw line: doing
# so matches *inside* species/element identifiers that contain digits and a
# hyphen, e.g. the real ring-locant species name "C2H4O1-2" (KONNOV_2019
# ammonia/hydrogen mechanism) -- rewriting its SPECIES-block declaration to
# "C2H4O1E-2" while leaving reaction-equation usages of "C2H4O1-2" untouched,
# which desyncs the two and makes ck2yaml report the (correctly-spelled)
# usage as undeclared. Confirmed against the real mechanism: the old
# lookaround regex flagged/would-fix 52 occurrences in 128-957.inp/.dat, all
# of them this species-name false positive, none a genuine exponent. See
# mechanism_converter_gui.py's twin fix (_FORTRAN_EXP_TOKEN) and
# KRUNAL_JULY_23.md.
_FORTRAN_EXP_TOKEN = re.compile(r"^([+-]?(?:\d+\.\d*|\.\d+|\d+))([+-])(\d{1,3})$")


class ChemkinPreflight:
    """Line-numbered structural linter for a CHEMKIN ``.inp`` (+ optional
    separate thermo/transport files)."""

    def __init__(self, path: str, thermo: Optional[str] = None,
                 transport: Optional[str] = None) -> None:
        self.path = path
        self.thermo = thermo
        self.transport = transport
        self._raw = _read_bytes(path)
        self.text = self._raw.decode("utf-8", errors="replace")
        self.lines = self.text.split("\n")
        self.declared: List[str] = []            # SPECIES-block order, first spelling
        self._blocks: Dict[str, Tuple[int, Optional[int]]] = {}
        # reaction participants seen but never declared in SPECIES -- real
        # species names only (excludes _PSEUDO_SPECIES); dedup, first-seen
        # order. Populated by _check_species_consistency; consumed by
        # _insert_missing_species (the "auto-repair genuinely-undeclared
        # species without user input" fix).
        self._undeclared_species: List[str] = []
        self._species_scanned = False

    # ------------------------------------------------------------------
    def run(self) -> CheckReport:
        report = CheckReport(mechanism=self.path, ran_checks=["chemkin_preflight"])
        findings: List[Finding] = []
        findings += self._check_encoding_whitespace()
        block_findings, sections = self._scan_blocks()
        findings += block_findings
        findings += self._check_surface_constructs(sections)
        findings += self._check_species_consistency(sections)
        findings += self._check_ea_units(sections)
        report.findings = findings
        report.loaded = True                     # "preflight ran"
        report.meta = {"n_lines": len(self.lines),
                       "n_declared_species": len(self.declared),
                       "sections": {k: v[0] for k, v in self._blocks.items()}}
        return report

    # ------------------------------------------------------------------
    # Encoding / whitespace / Fortran exponents (P2)
    # ------------------------------------------------------------------
    def _check_encoding_whitespace(self) -> List[Finding]:
        out: List[Finding] = []
        if b"\r" in self._raw:
            out.append(Finding(
                check="chemkin_preflight", severity="warning", category="crlf",
                message="File uses CRLF / CR line endings — normalise to LF "
                        "(some CHEMKIN tokenizers mis-handle the stray CR).",
                location="whole file",
                details={"line": 0},
                fix_options=[_normalize_fix()]))
        for i, line in enumerate(self.lines, start=1):
            code = line.split("!", 1)[0]
            if "\t" in code:
                out.append(Finding(
                    check="chemkin_preflight", severity="warning",
                    category="tab-char",
                    message="Tab character in a data line — CHEMKIN is column/"
                            "whitespace sensitive; replace tabs with spaces.",
                    location=f"line {i}",
                    details={"line": i, "column": code.index("\t") + 1},
                    fix_options=[_normalize_fix()]))
            if line.lstrip().startswith("!"):
                continue
            offset, tok = next(_iter_fortran_exp_matches(code), (None, None))
            if tok is not None:
                out.append(Finding(
                    check="chemkin_preflight", severity="error",
                    category="fortran-exponent",
                    message=(f"Fortran-style exponent '{tok}' missing the "
                             "'E' (e.g. 1.7-13 → 1.7E-13) — a top cause of "
                             "ck2yaml parse failures."),
                    location=f"line {i}",
                    details={"line": i, "column": offset + 1},
                    fix_options=[_normalize_fix()]))
        return out

    # ------------------------------------------------------------------
    # Block structure  (returns per-section {name: (start,end)} line ranges)
    # ------------------------------------------------------------------
    def _scan_blocks(self) -> Tuple[List[Finding], Dict[str, Tuple[int, int]]]:
        out: List[Finding] = []
        sections: Dict[str, Tuple[int, int]] = {}
        current: Optional[str] = None
        start_line = 0
        for i, line in enumerate(self.lines, start=1):
            stripped = line.split("!", 1)[0].strip()
            if not stripped:
                continue
            first = stripped.split()[0].upper()
            if current is None and first in _OPENERS:
                current = _canon_section(first)
                start_line = i
                self._blocks[current] = (i, None)
                continue
            if current is not None and first == "END":
                sections[current] = (start_line, i)
                self._blocks[current] = (start_line, i)
                current = None
                continue
            # a new opener while a block is still open → previous block missing END
            if current is not None and first in _OPENERS \
                    and _canon_section(first) != current:
                out.append(Finding(
                    check="chemkin_preflight", severity="error",
                    category="missing-end",
                    message=(f"Section '{current}' (opened at line {start_line}) is "
                             f"not closed by END before '{first}' begins."),
                    location=f"line {i}", details={"line": i, "section": current}))
                sections[current] = (start_line, i - 1)
                current = _canon_section(first)
                start_line = i
                self._blocks[current] = (i, None)
        if current is not None:
            sections[current] = (start_line, len(self.lines))
            out.append(Finding(
                check="chemkin_preflight", severity="error", category="missing-end",
                message=(f"Section '{current}' (opened at line {start_line}) is "
                         "never closed by END (reached end of file)."),
                location=f"line {start_line}",
                details={"line": start_line, "section": current}))

        if "REACTIONS" not in sections:
            out.append(Finding(
                check="chemkin_preflight", severity="info", category="no-reactions",
                message="No REACTIONS block found — is this a thermo/transport-only "
                        "file?", location="whole file", details={"line": 0}))
        return out, sections

    # ------------------------------------------------------------------
    # Surface / heterogeneous constructs (P3, P13)
    # ------------------------------------------------------------------
    def _check_surface_constructs(self, sections) -> List[Finding]:
        out: List[Finding] = []
        for i, line in enumerate(self.lines, start=1):
            code = line.split("!", 1)[0]
            up = code.upper()
            for kw, (cat, msg) in {
                "NONCON": ("noncon", "The NONCON (non-site-conserving surface "
                           "reaction) keyword has no defined Cantera semantics — "
                           "this reaction will not convert (see W5)."),
                "MATERIAL": ("material", "OpenSmoke/surface 'MATERIAL' keyword — "
                             "Cantera uses a site/interface model; needs the W3 "
                             "importer, not ck2yaml."),
            }.items():
                if re.search(rf"\b{kw}\b", up):
                    out.append(Finding(
                        check="chemkin_preflight", severity="warning", category=cat,
                        message=msg, location=f"line {i}", details={"line": i}))
        # BULK / SITE section openers
        for name in ("BULK", "SITE"):
            if name in self._blocks:
                ln = self._blocks[name][0]
                out.append(Finding(
                    check="chemkin_preflight", severity="warning",
                    category="surface-phase",
                    message=(f"'{name}' phase block (line {ln}) — ck2yaml surface "
                             "support is limited; verify the converted interface "
                             "phase and site balancing (see W5)."),
                    location=f"line {ln}", details={"line": ln}))
        return out

    # ------------------------------------------------------------------
    # Species name consistency (P35)
    # ------------------------------------------------------------------
    def _check_species_consistency(self, sections) -> List[Finding]:
        out: List[Finding] = []
        self._species_scanned = True
        if "SPECIES" not in sections:
            return out
        s0, s1 = sections["SPECIES"]
        # collect declared species (preserve first spelling), keyed upper-case
        canon: Dict[str, str] = {}
        for i in range(s0, s1):
            stripped = self.lines[i].split("!", 1)[0].strip()
            if not stripped or stripped.split()[0].upper() in _SPEC:
                continue
            for tok in stripped.split():
                if tok.upper() == "END":
                    continue
                canon.setdefault(tok.upper(), tok)
                if tok not in self.declared:
                    self.declared.append(tok)

        # scan reaction equations for case-variants / undeclared participants
        if "REACTIONS" in sections:
            r0, r1 = sections["REACTIONS"]
            seen_variant: Set[Tuple[str, str, int]] = set()
            for i in range(r0, r1):
                lineno = i + 1
                toks = self._reaction_species(self.lines[i])
                for tok in toks:
                    up = tok.upper()
                    if up in canon and canon[up] != tok:
                        key = (canon[up], tok, lineno)
                        if key in seen_variant:
                            continue
                        seen_variant.add(key)
                        out.append(Finding(
                            check="chemkin_preflight", severity="error",
                            category="species-case-mismatch",
                            message=(f"Species '{tok}' is declared as '{canon[up]}' "
                                     "in the SPECIES block — CHEMKIN/Cantera treat "
                                     "these as different species (the AR/Ar trap)."),
                            location=f"line {lineno}",
                            details={"line": lineno, "used": tok,
                                     "declared": canon[up]},
                            fix_options=[FixOption(
                                id="normalize_species_case", kind="auto",
                                recommended=True,
                                label=f"Rewrite '{tok}' → '{canon[up]}'",
                                description="Match the SPECIES-block spelling "
                                            "everywhere.",
                                params={"used": tok, "declared": canon[up]})]))
                    elif up not in canon and _looks_like_species(tok):
                        fix_options = []
                        if up in _PSEUDO_SPECIES:
                            msg = (f"Reaction participant '{tok}' is not in the "
                                   "SPECIES block, and looks like a bookkeeping "
                                   "placeholder (e.g. chemiluminescence 'HV') "
                                   "rather than a real species -- adding it would "
                                   "just trade this error for a missing-thermo "
                                   "one. Drop it from the reaction, or declare it "
                                   "with real thermo data if it is genuine.")
                        else:
                            msg = (f"Reaction participant '{tok}' is not in the "
                                   f"SPECIES block (e.g. add '{tok}' as its own "
                                   "line in the SPECIES block, right before its "
                                   "END).")
                            if tok not in self._undeclared_species:
                                self._undeclared_species.append(tok)
                            fix_options = [FixOption(
                                id="insert_missing_species", kind="auto",
                                recommended=True,
                                label=f"Add '{tok}' to the SPECIES block",
                                description="Insert the missing species "
                                            "declaration so ck2yaml stops "
                                            "treating it as undeclared.",
                                params={"species": tok})]
                        out.append(Finding(
                            check="chemkin_preflight", severity="warning",
                            category="undeclared-species",
                            message=msg,
                            location=f"line {lineno}",
                            details={"line": lineno, "token": tok},
                            fix_options=fix_options))
        return out

    def _reaction_species(self, line: str) -> List[str]:
        """Best-effort extraction of species tokens from a reaction equation line."""
        code = line.split("!", 1)[0].strip()
        if not code or "=" not in code:
            return []
        first = code.split()[0].upper().rstrip("/")
        if first in _AUX_KEYWORDS:
            return []
        if "/" in code and "=" not in code.split("/")[0]:
            return []                            # third-body efficiency / aux line
        # peel up to 3 trailing numeric Arrhenius tokens
        toks = code.split()
        while toks and _is_number(toks[-1]) and len(toks) > 1:
            # stop once we'd strip into the equation (need at least the eq token)
            if "=" in "".join(toks[:-1]):
                toks.pop()
            else:
                break
        eq = " ".join(toks)
        eq = re.sub(r"\(\+\s*[A-Za-z0-9]+\s*\)", " ", eq)   # (+M), (+N2)
        eq = eq.replace("<=>", " ").replace("=>", " ").replace("<=", " ")
        eq = eq.replace("=", " ").replace("+", " ")
        out: List[str] = []
        for t in eq.split():
            t = t.strip()
            if not t or t.upper() == "M":
                continue
            t = re.sub(r"^\d+", "", t)           # strip leading stoich coeff (2H2→H2)
            if t and not _is_number(t):
                out.append(t)
        return out

    # ------------------------------------------------------------------
    # Ea unit declaration
    # ------------------------------------------------------------------
    def _check_ea_units(self, sections) -> List[Finding]:
        out: List[Finding] = []
        if "REACTIONS" not in sections:
            return out
        r0 = sections["REACTIONS"][0]
        header = self.lines[r0 - 1].split("!", 1)[0].strip()
        toks = [t.upper() for t in header.split()[1:]]      # drop the REACTIONS word
        if not toks:
            out.append(Finding(
                check="chemkin_preflight", severity="info", category="ea-units",
                message="REACTIONS header declares no units — CHEMKIN defaults to "
                        "cal/mole & moles; confirm that matches the rate data "
                        "(the J/kmol vs cal/mol trap).",
                location=f"line {r0}", details={"line": r0}))
            return out
        unknown = [t for t in toks if t not in _EA_UNITS | _QTY_UNITS]
        if unknown:
            out.append(Finding(
                check="chemkin_preflight", severity="warning", category="ea-units",
                message=(f"Unrecognised REACTIONS unit keyword(s) {unknown} — "
                         f"expected one of {sorted(_EA_UNITS)} + "
                         f"{sorted(_QTY_UNITS)}."),
                location=f"line {r0}", details={"line": r0, "unknown": unknown}))
        return out

    # ------------------------------------------------------------------
    # --fix : write a cleaned .inp
    # ------------------------------------------------------------------
    def clean_text(self, normalize_case: bool = False) -> Tuple[str, List[str]]:
        """Return (cleaned_text, notes) applying the safe auto-normalisations.

        CRLF, tabs, Fortran exponents, and undeclared-species insertion are
        always safe/automatic (no user input needed -- KRUNAL_JULY_23.md's
        "deploy a code that does fix this automatically" ask). Species-case
        normalisation is **opt-in** (``normalize_case=True``) and is applied
        **only within reaction-equation lines** — never inside thermo records,
        element symbols, or comments — so it cannot corrupt data that legitimately
        reuses a species spelling.
        """
        notes: List[str] = []
        raw = self._raw
        if b"\r" in raw:
            raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            notes.append("normalised CRLF→LF")
        text = raw.decode("utf-8", errors="replace")
        text, n_tab = _detab(text)
        if n_tab:
            notes.append(f"replaced tabs on {n_tab} line(s)")
        text, n_exp = _fix_fortran(text)
        if n_exp:
            notes.append(f"fixed {n_exp} Fortran exponent(s)")
        if not self._species_scanned:
            # clean_text()/write_clean() may be called on a fresh instance
            # without .run() having scanned SPECIES/REACTIONS first -- do it
            # now so undeclared species are known either way.
            _, sections = self._scan_blocks()
            self._check_species_consistency(sections)
        text, n_undeclared = self._insert_missing_species(text)
        if n_undeclared:
            notes.append(
                f"added {n_undeclared} undeclared species to the SPECIES "
                f"block: {', '.join(self._undeclared_species)}")
        if normalize_case and self.declared:
            text, n_case = self._normalize_species_case(text)
            if n_case:
                notes.append(f"case-normalised {n_case} species token(s) in "
                             "reaction equations")
        return text, notes

    def _insert_missing_species(self, text: str) -> Tuple[str, int]:
        """Insert reaction participants that were never declared in the
        SPECIES block (the ``undeclared-species`` warning from
        ``_check_species_consistency``, minus ``_PSEUDO_SPECIES``) as new
        SPECIES-block entries, deduplicated and in first-seen order.

        CRLF/tab/Fortran-exponent fixes never insert or delete lines, so the
        1-based line numbers recorded in ``sections["SPECIES"]`` (from the
        *original* file, via ``self.lines``) still line up with ``text``'s
        line positions at this point in ``clean_text()``.
        """
        if not self._undeclared_species:
            return text, 0
        _, sections = self._scan_blocks()
        if "SPECIES" not in sections:
            return text, 0
        _, end_line = sections["SPECIES"]          # 1-based line# of the END line
        lines = text.split("\n")
        insert_idx = end_line - 1                  # 0-based index of that END line
        if not (0 <= insert_idx <= len(lines)):
            return text, 0
        lines[insert_idx:insert_idx] = list(self._undeclared_species)
        return "\n".join(lines), len(self._undeclared_species)

    def _normalize_species_case(self, text: str) -> Tuple[str, int]:
        """Rewrite species tokens to the SPECIES-block spelling — reactions only."""
        canon = {s.upper(): s for s in self.declared}
        pattern = re.compile(r"\b[A-Za-z][A-Za-z0-9()\-\*]*\b")
        n = 0

        def repl(m):
            nonlocal n
            tok = m.group(0)
            fixed = canon.get(tok.upper())
            if fixed and fixed != tok:
                n += 1
                return fixed
            return tok

        # locate the REACTIONS block in the (already cleaned) text and touch only it
        lines = text.split("\n")
        in_reac = False
        for i, line in enumerate(lines):
            first = line.split("!", 1)[0].strip().split()[:1]
            head = first[0].upper() if first else ""
            if head in _REAC:
                in_reac = True
                continue
            if in_reac and head == "END":
                in_reac = False
                continue
            if not in_reac:
                continue
            if "=" not in line.split("!", 1)[0]:
                continue                         # skip aux/efficiency lines
            code, bang, comment = line.partition("!")
            lines[i] = pattern.sub(repl, code) + bang + comment
        return "\n".join(lines), n

    def write_clean(self, out_path: str, normalize_case: bool = False) -> List[str]:
        text, notes = self.clean_text(normalize_case=normalize_case)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        with open(out_path, "w") as fh:
            fh.write(text if text.endswith("\n") else text + "\n")
        return notes


# ===========================================================================
# helpers
# ===========================================================================
def _normalize_fix() -> FixOption:
    return FixOption(
        id="chemkin_normalize", kind="auto", recommended=True,
        label="Auto-normalise (CRLF, tabs, Fortran exponents)",
        description="Write a cleaned .inp that ck2yaml can parse.",
        params={})


def _canon_section(first: str) -> str:
    if first in _ELEM:
        return "ELEMENTS"
    if first in _SPEC:
        return "SPECIES"
    if first in _THERMO:
        return "THERMO"
    if first in _REAC:
        return "REACTIONS"
    return first.upper()


def _looks_like_species(tok: str) -> bool:
    # a species token starts with a letter and isn't an aux keyword
    return bool(tok) and tok[0].isalpha() and tok.upper() not in _AUX_KEYWORDS


def _is_number(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def _detab(text: str) -> Tuple[str, int]:
    n = sum(1 for ln in text.split("\n") if "\t" in ln.split("!", 1)[0])
    return text.replace("\t", "    "), n


def _iter_fortran_exp_matches(code: str):
    """Yield (start_offset_in_code, matched_token) for each whitespace-
    delimited token in `code` that is a genuine Fortran-style exponent
    missing the 'E' (see _FORTRAN_EXP_TOKEN's docstring/comment above) --
    i.e. a token that is entirely numeric aside from the bare trailing
    exponent sign. A species token like "C2H4O1-2" is never yielded because
    it contains letters."""
    for tok_m in re.finditer(r"\S+", code):
        tok = tok_m.group(0)
        if _FORTRAN_EXP_TOKEN.match(tok):
            yield tok_m.start(), tok


def _fix_fortran(text: str) -> Tuple[str, int]:
    out, n = [], 0
    for line in text.split("\n"):
        if line.lstrip().startswith("!"):
            out.append(line)
            continue
        code, bang, comment = line.partition("!")
        parts = re.split(r"(\s+)", code)
        for i, part in enumerate(parts):
            fixed, k = _FORTRAN_EXP_TOKEN.subn(r"\1E\2\3", part)
            if k:
                parts[i] = fixed
                n += k
        out.append("".join(parts) + bang + comment)
    return "\n".join(out), n


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


# ===========================================================================
# module-level convenience
# ===========================================================================
def preflight(path: str, thermo: Optional[str] = None,
              transport: Optional[str] = None) -> CheckReport:
    """Run the CHEMKIN preflight linter on ``path`` and return a CheckReport."""
    return ChemkinPreflight(path, thermo=thermo, transport=transport).run()
