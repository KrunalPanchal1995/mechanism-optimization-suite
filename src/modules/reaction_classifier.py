"""
reaction_classifier.py
======================
Automatic HTC / LTC / NTC classifier for CHEMKIN-format reaction strings.

Classification is based on:
  - Species naming conventions (Curran 1998, NUIGMech, LLNL nomenclature)
  - Reaction stoichiometry patterns
  - Curran's 25 reaction class framework (HTC classes 1-9, LTC classes 10-25)
  - Sarathy 2014 extension (20 LTC sub-classes for branched alkanes)
  - Waddington mechanism species (for unsaturated fuels, Attarde 2024, Dong 2023)

Usage:
    from reaction_classifier import classify_reaction, classify_file
    result = classify_reaction("nc7h16 + oh => c7h15-2 + h2o")
    classify_file("mechanism.inp", output_csv="classified.csv")

Returns:
    dict with keys:
        'regime'    : 'HTC' | 'LTC' | 'NTC' | 'BOUNDARY' | 'UNCLASSIFIED'
        'class_id'  : int (Curran class number, 0 if unknown)
        'class_name': str (human-readable class name)
        'confidence': 'HIGH' | 'MEDIUM' | 'LOW'
        'flags'     : list of str (notes / warnings)
        'reactants' : list of str
        'products'  : list of str

Author: KP / IIT Madras Thermodynamics & Combustion Lab
Reference: Curran et al. (1998) Combust. Flame 114:149-177
           Sarathy et al. (2014) Combust. Flame 161:1444-1459
           Dong et al. (2023) Proc. Combust. Inst. 39:365-373
           Attarde & Narayanaswamy (2024) Combust. Flame 260:113213
"""

import re
import csv
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    regime: str = "UNCLASSIFIED"
    class_id: int = 0
    class_name: str = ""
    confidence: str = "LOW"
    flags: list = field(default_factory=list)
    reactants: list = field(default_factory=list)
    products: list = field(default_factory=list)
    raw_reaction: str = ""

    def to_dict(self):
        return {
            "regime": self.regime,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "flags": "; ".join(self.flags),
            "reactants": " + ".join(self.reactants),
            "products": " + ".join(self.products),
            "raw_reaction": self.raw_reaction,
        }

# ─────────────────────────────────────────────────────────────────────────────
# Species pattern library
# These patterns cover Curran 1998 (n-heptane), Dooley 2008 (MB),
# NUIGMech (hexenes, butadiene), and LLNL conventions.
# ─────────────────────────────────────────────────────────────────────────────

# ------- LTC-exclusive species patterns -------
# These appear ONLY in low-temperature peroxy / QOOH chemistry

LTC_SPECIES_PATTERNS = [
    # 1. Alkylperoxy radicals (RO2 / ROO): heptyl, MB, generic
    (re.compile(r'\bc\d+h\d+o2[-_]?\d*\b', re.I),     "RO2 species (R+O2 addition)"),
    (re.compile(r'\b\w+o2\b', re.I),                   "Generic RO2 species"),
    (re.compile(r'\b\w+oo\b', re.I),                   "Peroxy radical (ROO notation)"),
    # 2. Hydroperoxy-alkyl radicals (QOOH)
    (re.compile(r'\bc\d+h\d+ooh[-_]?\d+[-_]?\d*\b', re.I), "QOOH radical (RO2 isomerization product)"),
    (re.compile(r'\b\w+ooh\w*\b', re.I),               "QOOH species"),
    # 3. Peroxyalkylhydroperoxide (O2QOOH): second O2 addition
    (re.compile(r'\bc\d+h\d+ooh[-_]?\d+o2\b', re.I),  "O2QOOH (2nd O2 addition)"),
    (re.compile(r'\b\w+ooho\d*\b', re.I),              "O2QOOH species"),
    # 4. Ketohydroperoxide (KET) species
    (re.compile(r'\bnc\d+ket\d+\b', re.I),             "Ketohydroperoxide (chain branching)"),
    (re.compile(r'\b\w+ket\d*\b', re.I),               "Ketohydroperoxide species"),
    (re.compile(r'\bmbket\w+\b', re.I),                "MB-ketohydroperoxide (MB LTC)"),
    # 5. Cyclic ether products from QOOH
    (re.compile(r'\bc\d+h\d+o[-_]\d+[-_]\d+\b', re.I),"Cyclic ether (QOOH cyclization product)"),
    # 6. Waddington pathway species (unsaturated fuels, OH+alkene)
    (re.compile(r'\bc\d+h\d+oh[-_]*\d*oo\b', re.I),   "Hydroxy-peroxy (Waddington intermediate)"),
    (re.compile(r'\bc\d+h\d+oh[-_]*\d*ooh\b', re.I),  "Hydroxy-QOOH (Waddington product)"),
    (re.compile(r'\bc2h3chohch2oo\b', re.I),           "Butadiene Waddington: 1-buten-4-ol-3-yl-O2"),
    (re.compile(r'\bc4h6[-_]*.*oh.*oo\b', re.I),       "Butadiene hydroxy-peroxy species"),
    # 7. MB-specific peroxy/QOOH
    (re.compile(r'\bmb[2-4m]oo\b', re.I),              "MB-alkylperoxy radical"),
    (re.compile(r'\bmb[2-4m]ooh\w+\b', re.I),         "MB-QOOH radical"),
    (re.compile(r'\bmbmoo\b', re.I),                   "MB methyl-site peroxy radical"),
    (re.compile(r'\bmbmooh\w*\b', re.I),               "MB methyl-site QOOH"),
]

# ------- HTC-exclusive species patterns -------
# These appear predominantly in high-temperature decomposition and abstraction

HTC_SPECIES_PATTERNS = [
    # Radicals from direct H-abstraction (no oxygen in name)
    (re.compile(r'\bc\d+h\d+[-_]\d+\b', re.I),  "Alkyl radical (H-abstraction product)"),
    (re.compile(r'\bnc\d+h\d+\b', re.I),         "Linear alkyl radical"),
    (re.compile(r'\bpc\d+h\d+\b', re.I),         "Primary alkyl radical"),
    # Beta-scission olefin products
    (re.compile(r'\bc\d+h\d+[-_]\d+\b', re.I),  "Olefin (beta-scission product)"),
]

# ------- Small oxidizer / chain carrier species (boundary/ambiguous) -------
OXIDIZER_SPECIES = {"oh", "ho2", "h", "o", "h2o2", "o2", "h2o", "ch3", "ch3o2",
                    "ch3o", "c2h5", "c2h3", "hco", "co", "co2", "h2", "n2", "ar"}

NTC_INDICATOR_SPECIES = {
    "h2o2",  # H2O2 buildup/decomposition is the NTC→HTC transition marker
}

# ─────────────────────────────────────────────────────────────────────────────
# Reaction pattern library (based on stoichiometry shape, not species names)
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (class_id, regime, confidence, class_name, match_function)
# match_function receives (reactant_list, product_list, all_species)

def _has(species_list, pattern):
    """Check if any species in list matches re pattern."""
    return any(pattern.search(sp) for sp in species_list)

def _any_ltc(species_list):
    return any(pat.search(sp) for sp, _ in [(sp, None) for sp in species_list]
               for pat, _ in LTC_SPECIES_PATTERNS)

def _ltc_species_match(species_list):
    """Return first matching LTC pattern description, or None."""
    for sp in species_list:
        for pat, desc in LTC_SPECIES_PATTERNS:
            if pat.search(sp):
                return desc
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Core reaction-type detectors
# Ordered by specificity: most specific first
# ─────────────────────────────────────────────────────────────────────────────

def _detect_reaction_type(reactants, products, all_species, flags):
    """
    Returns (class_id, regime, confidence, class_name).
    Implements the Curran 1998 25-class framework with Sarathy 2014 extensions.
    """
    r_set = set(reactants)
    p_set = set(products)
    r_low = [s.lower() for s in reactants]
    p_low = [s.lower() for s in products]
    all_low = [s.lower() for s in all_species]

    # ── LTC CLASS 22: 2nd O2 addition  Q˙OOH + O2 = O˙2QOOH ──────────────
    o2qooh_pat = re.compile(r'\w+ooh[-_]?\d*o2|\w+o2.*ooh', re.I)
    if any(o2qooh_pat.search(s) for s in p_low):
        if "o2" in r_low:
            return (22, "LTC", "HIGH",
                    "Class 22: Addition of Q˙OOH to O2 → O˙2QOOH")

    # ── LTC CLASS 23: O˙2QOOH isomerization → ketohydroperoxide + O˙H ──────
    ket_pat = re.compile(r'(nc\d+ket|mbket|\w+ket)\d*', re.I)
    o2qooh_r_pat = re.compile(r'\w+(ooh.*o2|o2.*ooh)\w*', re.I)
    if (any(ket_pat.search(s) for s in p_low) and
            "oh" in p_low):
        return (23, "LTC", "HIGH",
                "Class 23: O˙2QOOH isomerization → ketohydroperoxide + O˙H")

    # ── LTC CLASS 24: Ketohydroperoxide decomposition ─────────────────────
    if (any(ket_pat.search(s) for s in r_low) and
            "oh" in p_low):
        return (24, "LTC", "HIGH",
                "Class 24: Ketohydroperoxide decomposition → oxygenated radical + O˙H")

    # ── LTC CLASS 10: R˙ + O2 = RO˙2 (first O2 addition) ────────────────
    ro2_pat = re.compile(r'\b(\w+)o2[-_]?\d*\b|\b(\w+)oo\b', re.I)
    ket_or_ooh_in_products = any(
        re.search(r'ooh|ket', s, re.I) for s in p_low
    )
    if (any(ro2_pat.search(s) for s in p_low) and
            "o2" in r_low and
            not ket_or_ooh_in_products):
        return (10, "LTC", "HIGH",
                "Class 10: R˙ + O2 = RO˙2 (alkylperoxy radical formation)")

    # ── LTC CLASS 10 reverse: RO˙2 = R˙ + O2 ─────────────────────────────
    if (any(ro2_pat.search(s) for s in r_low) and
            "o2" in p_low and
            not any(re.search(r'ooh', s, re.I) for s in p_low)):
        return (10, "LTC", "HIGH",
                "Class 10 (rev): RO˙2 ⇌ R˙ + O2 (equilibrium)")

    # ── LTC CLASS 12: RO˙2 isomerization → Q˙OOH ─────────────────────────
    qooh_pat = re.compile(r'\w+ooh[-_]?\d', re.I)
    if (any(ro2_pat.search(s) for s in r_low) and
            any(qooh_pat.search(s) for s in p_low) and
            "o2" not in p_low):
        return (12, "LTC", "HIGH",
                "Class 12: RO˙2 isomerization → Q˙OOH (internal H-shift)")

    # ── NTC / LTC CLASS 20: Q˙OOH → olefin + HO˙2 (KEY NTC inhibitor) ─────
    # NOTE: must be BEFORE Class 12 reverse to avoid misclassification
    if (len(r_low) == 1 and
            any(qooh_pat.search(s) for s in r_low) and
            "ho2" in p_low and
            not any(re.search(r"\bo2\b", s, re.I) for s in p_low if s != "ho2") and
            not any(re.search(r"o2[-_]", s, re.I) for s in p_low)):
        return (20, "NTC", "HIGH",
                "Class 20: Q˙OOH → olefin + HO˙2 (KEY NTC inhibitor, chain propagation)")

    # ── LTC CLASS 22: QOOH + O2 → O2QOOH (2nd O2 addition) ──────────────
    if ("o2" in r_low and
            any(qooh_pat.search(s) for s in r_low) and
            len(r_low) == 2):
        return (22, "LTC", "HIGH",
                "Class 22: Q˙OOH + O2 → O˙2QOOH (2nd O2 addition, chain branching)")

    # ── NTC CLASS 19: Q˙OOH → cyclic ether + O˙H ────────────────────────
    # Heuristic: QOOH in reactant, OH in products, and a positional cyclic ether
    # pattern in products (CnH(2n-2)Ox-a-b notation, fewer H than saturated RO2)
    # This check MUST come before Class 12 reverse because the cyclic ether
    # species (e.g. c7h14o2-4) also matches the ro2_pat pattern.
    cyc_prod_pat2 = re.compile(r'\bc\d+h\d+o\d*[-_]\d+[-_]?\d*\b', re.I)
    if (any(qooh_pat.search(s) for s in r_low) and
            "oh" in p_low and
            any(cyc_prod_pat2.search(s) for s in p_low if s != "oh")):
        return (19, "NTC", "HIGH",
                "Class 19: Q\u02d9OOH → cyclic ether + O\u02d9H "
                "(chain propagation, NTC inhibitor)")

    # ── LTC CLASS 12 reverse: Q˙OOH → RO˙2 ──────────────────────────────
    if (any(qooh_pat.search(s) for s in r_low) and
            any(ro2_pat.search(s) for s in p_low) and
            "o2" not in r_low):
        return (12, "LTC", "HIGH",
                "Class 12 (rev): Q\u02d9OOH → RO\u02d9 2 (reverse isomerization)")

    # ── NTC / LTC CLASS 20 (duplicate guard) ─────────────────────────────
    if (any(qooh_pat.search(s) for s in r_low) and
            "ho2" in p_low):
        return (20, "NTC", "HIGH",
                "Class 20: Q\u02d9OOH → olefin + HO\u02d9 2 (KEY NTC inhibitor)")

    # ── LTC CLASS 21: Q˙OOH → olefin + carbonyl + O˙H ───────────────────
    if (any(qooh_pat.search(s) for s in r_low) and
            "oh" in p_low and
            len(p_low) >= 3):
        return (21, "LTC", "MEDIUM",
                "Class 21: Q\u02d9OOH \u03b2-scission → olefin + carbonyl + O\u02d9H")

    # ── LTC Waddington: hydroxy-RO2 isomerization (unsaturated fuels) ─────
    wad_pat = re.compile(r'\b(c\d+h\d+.*oh.*oo|c2h3chohch2oo|c4h6.*oh.*oo)\b', re.I)
    wad_prod_pat = re.compile(r'\b(c\d+h\d+.*oh.*ooh|c4h5oh|enol|c2h3cho)\b', re.I)
    if (any(wad_pat.search(s) for s in r_low) or
            any(wad_pat.search(s) for s in p_low)):
        return (12, "LTC", "HIGH",
                "Class 12 (Waddington variant): hydroxy-RO˙2 isomerization "
                "→ hydroxy-Q˙OOH (unsaturated fuel LTC entry)")

    # ── HTC / NTC boundary: RO˙2 + HO˙2 → ROOH + O2 ──────────────────────
    rooh_pat = re.compile(r'\b\w+o2h\b|\b\w+ooh\b', re.I)
    if (any(ro2_pat.search(s) for s in r_low) and
            "ho2" in r_low and
            any(rooh_pat.search(s) for s in p_low)):
        return (13, "BOUNDARY", "MEDIUM",
                "Class 13: RO˙2 + HO˙2 → ROOH + O2 (boundary: active near NTC region)")

    # ── NTC indicator: H2O2 decomposition (HTC/NTC transition) ────────────
    if "h2o2" in r_low and p_low.count("oh") >= 1:
        return (0, "NTC", "HIGH",
                "H2O2 = 2OH (NTC→HTC transition: key branching at ~800-900K)")

    # ── HTC CLASS 1: Unimolecular fuel decomposition ──────────────────────
    # Large fuel → two radical fragments (no small oxidizers)
    fuel_pat = re.compile(r'\bnc\d{1,2}h\d+\b|\bmb[-_c]?\d+h\d+\b|\b[a-z]{2,4}-c\d+h\d+o\d*\b',
                          re.I)
    if (len(r_low) == 1 and
            any(fuel_pat.search(s) for s in r_low) and
            all(sp not in OXIDIZER_SPECIES for sp in p_low) and
            not any(re.search(r'o2|ooh|ket', s, re.I) for s in p_low)):
        return (1, "HTC", "HIGH",
                "Class 1: Unimolecular fuel decomposition → two alkyl radicals")

    # ── HTC CLASS 2: H-atom abstraction from fuel ─────────────────────────
    abstractors = {"oh", "h", "o", "ch3", "ho2", "ch3o2", "c2h3", "c2h5",
                   "o2", "ch3o"}
    radical_prod_pat = re.compile(r'\b\w+[-_]\d+\b|\b\w+j\b|\bmb\d+j\b|\bmbmj\b',
                                  re.I)
    h_transfer_prods = {"h2", "h2o", "ch4", "h2o2", "ch3oh", "ch3o2h"}
    if (any(sp in abstractors for sp in r_low) and
            any(radical_prod_pat.search(s) for s in p_low) and
            any(sp in h_transfer_prods for sp in p_low) and
            len(r_low) == 2):
        # HO2 and CH3O2 abstraction are borderline (active at intermediate T)
        if "ho2" in r_low:
            flags.append("HO2 abstraction active at intermediate T (700-950K); "
                         "boundary HTC/NTC")
            return (2, "HTC", "MEDIUM",
                    "Class 2: H-atom abstraction by HO˙2 "
                    "(boundary: important in NTC region)")
        if "ch3o2" in r_low:
            flags.append("CH3O2 abstraction borderline HTC/NTC")
            return (2, "HTC", "MEDIUM",
                    "Class 2: H-atom abstraction by CH3O˙2")
        return (2, "HTC", "HIGH",
                "Class 2: H-atom abstraction from fuel by radical")

    # ── HTC CLASS 3: Alkyl radical decomposition (β-scission) ─────────────
    alkyl_r_pat = re.compile(r'\b(c\d+h\d+[-_]\d+|nc\d+h\d+|pc\d+h\d+|'
                             r'mb\d+j|mbmj|mb[2-4]j)\b', re.I)
    olefin_pat  = re.compile(r'\b(c\d+h\d+[-_]\d+|c\d+h\d+)\b', re.I)
    if (len(r_low) == 1 and
            any(alkyl_r_pat.search(s) for s in r_low) and
            not any(re.search(r'oo|ooh|ket', s, re.I) for s in p_low) and
            len(p_low) == 2):
        return (3, "HTC", "HIGH",
                "Class 3: Alkyl radical β-scission → olefin + smaller radical")

    # ── HTC CLASS 4/6 (NTC boundary): concerted elimination RO˙2 → olefin + HO˙2 ─
    if (any(ro2_pat.search(s) for s in r_low) and
            "ho2" in p_low and
            not any(re.search(r'ooh', s, re.I) for s in p_low)):
        flags.append("Concerted elimination: active in NTC region, competes with "
                     "RO2 isomerization; classified as LTC Class 6 "
                     "(Sarathy 2014 convention)")
        return (6, "LTC", "HIGH",
                "Class 6: Concerted elimination RO˙2 → olefin + HO˙2 "
                "(NTC chain propagation inhibitor)")

    # ── HTC CLASS 5: Alkyl radical isomerization ──────────────────────────
    # R → R' (no oxygen, single species → single species)
    if (len(r_low) == 1 and len(p_low) == 1 and
            any(alkyl_r_pat.search(s) for s in r_low) and
            any(alkyl_r_pat.search(s) for s in p_low) and
            not any(re.search(r'oo|ooh|o2', s, re.I) for s in all_low)):
        return (5, "HTC", "HIGH",
                "Class 5: Alkyl radical isomerization (H-shift; HTC)")

    # ── HTC CLASS 7a/b/c: Addition of radical to alkene ──────────────────
    # Subclass 7a: H, O, CH3 addition → HTC
    # Subclass 7b: OH addition to alkene → Dual (Waddington LTC or HTC β-scission)
    # Subclass 7c: HO2 addition → LTC/NTC (concerted or non-concerted)
    alkene_pat = re.compile(r'\bc\d+h\d+[-_]\d+\b|\bc4h6\b|\bc\d+h\d+\b', re.I)
    if len(r_low) == 2:
        if "h" in r_low and any(alkene_pat.search(s) for s in r_low):
            return (7, "HTC", "HIGH",
                    "Class 7a: H˙ + alkene → alkyl radical (HTC only)")
        if "o" in r_low and any(alkene_pat.search(s) for s in r_low):
            return (7, "HTC", "HIGH",
                    "Class 7a: O˙ + alkene → alkoxy + radical (HTC)")
        if "ch3" in r_low and any(alkene_pat.search(s) for s in r_low):
            return (7, "HTC", "HIGH",
                    "Class 7a: CH˙3 + alkene → adduct (HTC)")
        if "oh" in r_low and any(alkene_pat.search(s) for s in r_low):
            # Check if products show Waddington path (hydroxy-alkyl radical → LTC)
            hydroxy_prod = any(re.search(r'oh|hoch|ch.*oh', s, re.I) for s in p_low
                               if s not in OXIDIZER_SPECIES)
            if hydroxy_prod:
                flags.append("OH + alkene forms hydroxy-alkyl radical → feeds "
                             "Waddington LTC pathway (Class 7b dual-regime)")
                return (7, "BOUNDARY", "HIGH",
                        "Class 7b: OH˙ + alkene → hydroxy-alkyl radical "
                        "(DUAL REGIME: HTC via β-scission OR LTC via Waddington)")
            return (7, "HTC", "MEDIUM",
                    "Class 7b: OH˙ + alkene → direct substitution products (HTC pathway)")
        if "ho2" in r_low and any(alkene_pat.search(s) for s in r_low):
            flags.append("HO2 + alkene: concerted → cyclic ether + OH (LTC), "
                         "or non-concerted → QOOH (LTC/NTC)")
            return (7, "LTC", "HIGH",
                    "Class 7c: HO˙2 + alkene → cyclic ether + O˙H or Q˙OOH "
                    "(LTC/NTC; key for unsaturated fuel low-T chemistry)")

    # ── HTC CLASSES 8-9: Alkenyl radical decomposition / olefin decomposition ─
    alkenyl_pat = re.compile(r'\bc\d+h\d+[-_][in]\b|\bc4h5[-_][ni]\b|\bc3h5[-_]a\b',
                             re.I)
    if (len(r_low) == 1 and
            any(alkenyl_pat.search(s) for s in r_low)):
        return (8, "HTC", "HIGH",
                "Class 8: Alkenyl radical decomposition (HTC)")

    # ── LTC CLASS 17: ROOH = RO˙ + O˙H ──────────────────────────────────
    ro_rad_pat = re.compile(r'\b\w+o\b', re.I)
    if (any(rooh_pat.search(s) for s in r_low) and
            "oh" in p_low and
            any(ro_rad_pat.search(s) for s in p_low)):
        return (17, "LTC", "MEDIUM",
                "Class 17: ROOH → RO˙ + O˙H (alkyl hydroperoxide decomposition)")

    # ── LTC CLASS 18: RO˙ decomposition → aldehyde + alkyl ────────────────
    if (len(r_low) == 1 and
            any(ro_rad_pat.search(s) for s in r_low) and
            not any(re.search(r'o2|ooh', s, re.I) for s in all_low)):
        return (18, "LTC", "MEDIUM",
                "Class 18: RO˙ decomposition → aldehyde/ketone + alkyl radical")

    # ── LTC CLASS 25: Cyclic ether + OH/HO2 ──────────────────────────────
    if (any(cyclic_ether_pat.search(s) for s in r_low) and
            ("oh" in r_low or "ho2" in r_low)):
        return (25, "LTC", "MEDIUM",
                "Class 25: Cyclic ether reactions with O˙H or HO˙2")

    # ── Fallback: check if any species is LTC-characteristic ────────────
    ltc_match = _ltc_species_match(all_low)
    if ltc_match:
        flags.append(f"LTC species detected: {ltc_match}")
        return (0, "LTC", "MEDIUM", f"LTC by species signature: {ltc_match}")

    return (0, "UNCLASSIFIED", "LOW", "")


# ─────────────────────────────────────────────────────────────────────────────
# Reaction string parser
# Handles: A + B => C + D, A + B <=> C + D, A(+M) => ..., A = B
# ─────────────────────────────────────────────────────────────────────────────

def _parse_reaction(reaction_str: str):
    """
    Parse a CHEMKIN-format reaction string into reactants and products.
    Strips stoichiometric coefficients and third-body notations.

    Returns (reactants, products) as lists of lowercase strings.
    """
    # Strip Arrhenius parameters at end of line (three floats)
    line = re.sub(r'[\d.eE+\-]+\s+[\d.eE+\-]+\s+[\d.eE+\-]+\s*$', '',
                  reaction_str.strip())
    # Strip comments
    line = re.sub(r'!.*$', '', line).strip()
    if not line:
        return [], []

    # Identify arrow: =>, <=>, =, =>
    arrow_pat = re.compile(r'<=>|=>|=')
    match = arrow_pat.search(line)
    if not match:
        return [], []

    lhs = line[:match.start()].strip()
    rhs = line[match.end():].strip()

    def parse_side(side_str):
        # Remove third-body: (+M), +M
        side_str = re.sub(r'\(\+m\)|\+m\b', '', side_str, flags=re.I)
        # Split by +
        parts = [p.strip() for p in side_str.split('+') if p.strip()]
        species = []
        for part in parts:
            # Remove leading stoichiometric integer or float
            sp = re.sub(r'^\d+\.?\d*\s*', '', part).strip().lower()
            if sp:
                species.append(sp)
        return species

    return parse_side(lhs), parse_side(rhs)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def classify_reaction(reaction_str: str) -> ClassificationResult:
    """
    Classify a single CHEMKIN reaction string into HTC / LTC / NTC / BOUNDARY.

    Parameters
    ----------
    reaction_str : str
        A CHEMKIN-format reaction line, e.g.
        "nc7h16 + oh => c7h15-2 + h2o"
        "c7h15o2-2 => c7h14ooh2-4"
        "C4H6 + HO2 <=> C4H61-3OOH4"

    Returns
    -------
    ClassificationResult
    """
    result = ClassificationResult(raw_reaction=reaction_str.strip())
    reactants, products = _parse_reaction(reaction_str)

    if not reactants:
        result.flags.append("Could not parse reaction string")
        return result

    result.reactants = reactants
    result.products = products
    all_species = reactants + products

    class_id, regime, confidence, class_name = _detect_reaction_type(
        reactants, products, all_species, result.flags
    )

    result.class_id  = class_id
    result.regime    = regime
    result.confidence = confidence
    result.class_name = class_name
    return result


def classify_reactions(reactions: list) -> list:
    """Classify a list of reaction strings. Returns list of ClassificationResult."""
    return [classify_reaction(r) for r in reactions]


def classify_file(input_path: str, output_csv: Optional[str] = None,
                  skip_comments: bool = True) -> list:
    """
    Read a CHEMKIN mechanism file and classify all reactions.

    Parameters
    ----------
    input_path  : path to CHEMKIN .inp / .dat / .mech file
    output_csv  : if given, write results to this CSV path
    skip_comments: skip lines starting with '!' or '%'

    Returns
    -------
    List of ClassificationResult objects.
    """
    results = []
    with open(input_path, 'r', errors='replace') as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            if skip_comments and stripped.startswith(('!', '%', '//')):
                continue
            # Skip keyword lines
            if re.match(r'^(REACTIONS|END|SPECIES|THERMO|ELEMENTS)\b',
                        stripped, re.I):
                continue
            # Must contain an arrow to be a reaction
            if not re.search(r'<=>|=>|=', stripped):
                continue
            results.append(classify_reaction(stripped))

    if output_csv:
        _write_csv(results, output_csv)
        print(f"Results written to {output_csv}")

    return results


def _write_csv(results: list, path: str):
    """Write classification results to CSV."""
    if not results:
        return
    fieldnames = list(results[0].to_dict().keys())
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())


def summarize(results: list) -> dict:
    """Print and return a summary of classification results."""
    from collections import Counter
    regime_counts = Counter(r.regime for r in results)
    class_counts  = Counter(r.class_name for r in results if r.class_name)
    total = len(results)

    print(f"\n{'='*60}")
    print(f"  Reaction Classification Summary  ({total} reactions)")
    print(f"{'='*60}")
    for regime in ["HTC", "LTC", "NTC", "BOUNDARY", "UNCLASSIFIED"]:
        count = regime_counts.get(regime, 0)
        pct   = 100 * count / total if total else 0
        bar   = '█' * int(pct / 2)
        print(f"  {regime:<14} {count:>5}  ({pct:5.1f}%)  {bar}")
    print(f"{'='*60}")
    print("\n  Top 10 reaction classes:")
    for name, count in class_counts.most_common(10):
        print(f"    [{count:>4}]  {name}")
    print()
    return {"by_regime": dict(regime_counts), "by_class": dict(class_counts)}


# ─────────────────────────────────────────────────────────────────────────────
# Command-line interface
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # ── Demo mode: classify the n-Heptane reactions from KP's tables ──────
    demo_reactions = [
        # HTC — n-Heptane
        "nc7h16 => c5h11-1 + c2h5",
        "nc7h16 + h => c7h15-1 + h2",
        "nc7h16 + oh => c7h15-2 + h2o",
        "nc7h16 + ho2 => c7h15-1 + h2o2",
        "nc7h16 + ch3o2 => c7h15-2 + ch3o2h",
        "c7h15-2 => pc4h9 + c3h6",
        "c7h15-3 => c4h8-1 + nc3h7",
        # LTC — n-Heptane
        "c7h15-2 + o2 => c7h15o2-2",
        "c7h15o2-2 => c7h15-2 + o2",
        "c7h15o2-2 => c7h14ooh2-4",
        "c7h14ooh2-4 => c7h14o2-4 + oh",
        "c7h14ooh2-4 => oh + ch3cho + c5h10-1",
        "c7h14ooh2-4 + o2 => c7h14ooh2-4o2",
        "c7h14ooh2-4o2 => nc7ket24 + oh",
        "nc7ket24 => nc3h7cho + ch3coch2 + oh",
        # NTC boundary
        "c7h14ooh2-4 => c7h14 + ho2",  # Class 20: KEY NTC inhibitor
        "h2o2 => oh + oh",
        # MB — HTC
        "mb-c5h10o2 + oh => h2o + mbmj",
        "mb4j => c2h4 + me2j",
        # MB — LTC
        "mb2oo => mb2ooh4j",
        "mb4ooh2j + o2 => mb4ooh2o2",
        "mb4ooh2o2 => mbket42 + oh",
        "mbket42 => oh + ch2cho + me2do",
        "mb2oo => mb2d + ho2",
        # MB — misclassified: should be HTC
        "mbmj => mb3j",  # alkyl radical isomerization → should be HTC
        # 1,3-Butadiene HTC
        "c4h6 + oh => c4h5-i + h2o",
        "c4h6 + h => c4h71-4",
        "c4h71-3 <=> c4h71-4",
        # 1,3-Butadiene LTC (Waddington)
        "c2h3chohch2oo <=> c4h51,3oh2 + ho2",
        "c2h3chohch2oo <=> c4h6o1-3ooh4",
        "c4h6o1-3ooh4 <=> c2h3cho + ch2o + oh",
        "c4h5oh1-4ooh + o2 <=> c4h5oh1-4ooh-2oo",
        # Boundary
        "c4h6 + ho2 <=> c4h61-3ooh4",       # Class 7c
        "c4h6 + oh => hoch2-ch=ch-ch2-dot",  # Class 7b dual-regime
    ]

    print("\n" + "="*60)
    print("  Reaction Classifier Demo — KP / IIT Madras")
    print("="*60)

    results = classify_reactions(demo_reactions)
    for res in results:
        flag_str = f"  ⚠ {res.flags[0]}" if res.flags else ""
        print(f"\n  Reaction : {res.raw_reaction}")
        print(f"  Regime   : {res.regime:<12} Confidence: {res.confidence}")
        print(f"  Class    : [{res.class_id:>2}] {res.class_name}")
        if flag_str:
            print(f"  Flag     :{flag_str}")

    summarize(results)

    # CLI: python reaction_classifier.py mechanism.inp output.csv
    if len(sys.argv) >= 2:
        in_file = sys.argv[1]
        out_file = sys.argv[2] if len(sys.argv) >= 3 else "classified_reactions.csv"
        print(f"\nClassifying file: {in_file}")
        file_results = classify_file(in_file, out_file)
        summarize(file_results)
