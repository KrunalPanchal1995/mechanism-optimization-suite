#!/usr/bin/env python3.9
"""
merge_hyperparams.py -- makes the round1 (+) round2 (+) round3 hand-merge
that historically produced chosen_hyperparams_merged.json and
kfold_final_hyperparams.json reproducible. Previously that merge was done ad
hoc with no producing script.

Merge rule: for each case listed in --cases (or every case key present in
OVERLAY, if --cases is omitted), OVERLAY's entry wins outright and replaces
BASE's entry for that case. This is a plain override, not a "keep whichever
scores better" comparison -- by construction, every search round only
touched cases it intended to improve, so the overlay is trusted unconditionally
for the cases it names. Cases not in the overlay's winning set keep BASE's
entry untouched.

Usage:
    merge_hyperparams.py BASE.json OVERLAY.json -o OUT.json [--cases 3,4]

Output JSON is BASE's per-case entries with OVERLAY's substituted in for the
selected cases, plus a "_provenance" block recording:
  - base/overlay file paths and their md5s
  - which cases came from which side

Run: python3.9 merge_hyperparams.py --help
"""
import argparse
import hashlib
import json
import os


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(path):
    with open(path) as f:
        return json.load(f)


def merge(base, overlay, cases=None):
    """base, overlay: dict[str-or-int case -> hyperparam dict].
    cases: iterable of case ids (int or str) to take from overlay, or None
    to take every case key present in overlay.

    Returns (merged_dict, provenance_cases) where provenance_cases maps
    case-id-as-str -> "base" or "overlay"."""
    # Normalize all keys to str (JSON dict keys are always str once loaded,
    # but --cases may be passed as ints), and drop any pre-existing
    # "_provenance" block -- BASE or OVERLAY may themselves be the output of
    # an earlier merge_hyperparams.py run (chained merges, e.g. round1+round2
    # then +round3), so their own provenance must not be treated as a case.
    base = {str(k): v for k, v in base.items() if k != "_provenance"}
    overlay = {str(k): v for k, v in overlay.items() if k != "_provenance"}

    if cases is None:
        overlay_cases = set(overlay.keys())
    else:
        overlay_cases = {str(c) for c in cases}
        missing = overlay_cases - set(overlay.keys())
        if missing:
            raise ValueError(f"--cases named {sorted(missing)} not present in overlay file")

    merged = dict(base)
    provenance_cases = {}
    all_cases = sorted(set(base.keys()) | overlay_cases, key=lambda c: int(c))
    for c in all_cases:
        if c in overlay_cases:
            merged[c] = overlay[c]
            provenance_cases[c] = "overlay"
        elif c in base:
            provenance_cases[c] = "base"
        else:
            # case named in --cases but absent from base too -- still take overlay
            merged[c] = overlay[c]
            provenance_cases[c] = "overlay"
    return merged, provenance_cases


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", help="Base hyperparameters JSON (e.g. round N)")
    ap.add_argument("overlay", help="Overlay hyperparameters JSON (e.g. round N+1); wins per --cases")
    ap.add_argument("-o", "--output", required=True, help="Output merged JSON path")
    ap.add_argument("--cases", default=None,
                     help="Comma-separated case ids that overlay wins for (default: every "
                          "case key present in overlay)")
    args = ap.parse_args()

    base_path = os.path.abspath(args.base)
    overlay_path = os.path.abspath(args.overlay)

    base = _load(base_path)
    overlay = _load(overlay_path)

    cases = None
    if args.cases is not None:
        cases = [c.strip() for c in args.cases.split(",") if c.strip() != ""]

    merged, provenance_cases = merge(base, overlay, cases=cases)

    merged["_provenance"] = {
        "base_path": base_path,
        "base_md5": _md5(base_path),
        "overlay_path": overlay_path,
        "overlay_md5": _md5(overlay_path),
        "cases_from": provenance_cases,
    }

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)

    n_overlay = sum(1 for v in provenance_cases.values() if v == "overlay")
    n_base = sum(1 for v in provenance_cases.values() if v == "base")
    print(f"merged {len(provenance_cases)} cases ({n_overlay} from overlay, {n_base} from base) -> {out_path}")
    for c, src in sorted(provenance_cases.items(), key=lambda kv: int(kv[0])):
        print(f"  case {c}: {src}")


if __name__ == "__main__":
    main()
