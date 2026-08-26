"""
rmip/config/recipes.py
======================
W9b — task → recipe router: ``rmip how-do-i "<task>"`` (P36).

The recurring onboarding question is not "does the tool exist" but "which of
the fifty subcommands / example scripts covers MY task". This module holds a
curated table mapping plain-language tasks to the exact command, and a fuzzy
matcher over it. All commands are shown through the ``rmip_cli.py`` launcher
(never ``python src/rmip/cli.py`` — the ``io`` package shadowing rule).
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Dict, List

_CLI = "python3 rmip_cli.py"


@dataclass
class Recipe:
    task: str                       # plain-language task phrase
    command: str                    # the exact command to run
    description: str                # one line: what it does
    keywords: List[str] = field(default_factory=list)
    gui: str = ""                   # matching GUI app, when one exists

    def to_dict(self) -> Dict[str, str]:
        return {"task": self.task, "command": self.command,
                "description": self.description, "gui": self.gui}


RECIPES: List[Recipe] = [
    Recipe("check my mechanism file for errors",
           f"{_CLI} check mech.yaml",
           "Full health battery: rate NaN/zero, collision limit, thermo "
           "discontinuity, balance, duplicates, PLOG, units, name consistency.",
           ["check", "health", "diagnose", "validate", "broken", "errors",
            "lint", "mechanism"],
           gui="mechanism_check_gui.py"),
    Recipe("automatically fix a broken mechanism",
           f"{_CLI} check mech.yaml --fix auto",
           "Runs the health battery, applies the unambiguous repairs and "
           "writes <name>.fixed.yaml.",
           ["fix", "repair", "broken", "auto", "correct"],
           gui="mechanism_check_gui.py"),
    Recipe("convert a CHEMKIN mechanism to Cantera YAML",
           f"{_CLI} preflight mech.inp --thermo therm.dat --fix && "
           f"{_CLI} check mech.yaml",
           "Preflight cleans the classic ck2yaml blockers (CRLF, Fortran "
           "exponents, case-variant species), then converts.",
           ["convert", "chemkin", "ck2yaml", "yaml", "inp", "translate",
            "import"],
           gui="mechanism_converter_gui.py"),
    Recipe("migrate an old CTI-era YAML to current Cantera",
           f"{_CLI} migrate mech.yaml",
           "Rewrites deprecated CTI→YAML keywords (type: PLOG, "
           "stoichiometric, …) to the current schema.",
           ["migrate", "deprecated", "cti", "upgrade", "cantera 3",
            "keywords", "schema"]),
    Recipe("ignition delay sensitivity analysis",
           f"{_CLI} idt-sens mech.yaml --T 1000 --P 20 --phi 1.0 --fuel NC7H16:1.0",
           "d ln(tau)/d ln(k) ranking — brute-force or CVODES, selectable "
           "IDT definition, tornado output.",
           ["idt", "ignition", "delay", "sensitivity", "tau", "brute force",
            "reactions matter", "ranking"],
           gui="idt_sensitivity_gui.py"),
    Recipe("reduce a mechanism (DRG / DRGEP / PFA)",
           f"{_CLI} reduce --mech 1 --db project.db --method drgep "
           "--targets NC7H16 OH --out reduced/",
           "Skeletal reduction with error control against IDT targets.",
           ["reduce", "reduction", "skeletal", "drg", "drgep", "pfa",
            "smaller", "species removal"]),
    Recipe("reaction path / flux diagram",
           f"{_CLI} flux mech.yaml --T 1200 --P 10 --phi 1.0 --fuel CH4:1.0 "
           "--element C --out flux.dot",
           "Integrated element-flux graph at a reactor state (Graphviz .dot).",
           ["reaction path", "flux", "diagram", "pathway", "graphviz",
            "element flux", "network"]),
    Recipe("global sensitivity analysis (Sobol / Morris)",
           f"{_CLI} gsa-run --mech 1 --db project.db --backend salib",
           "Sobol/Morris GSA over rate uncertainties (SALib venv backend; "
           "optional Zero-RK native solver).",
           ["gsa", "sobol", "morris", "global sensitivity", "salib",
            "variance", "screening"]),
    Recipe("forward uncertainty propagation (surrogate UQ)",
           f"{_CLI} surrogate --mech 1 --db project.db --order 2",
           "gPC/PRS surrogate + Monte-Carlo propagation of rate "
           "uncertainties onto targets (chaospy venv backend).",
           ["uq", "uncertainty", "propagation", "forward", "monte carlo",
            "pce", "chaospy", "surrogate", "error bars"],
           gui="forward_uq_gui.py"),
    Recipe("calibrate / optimize rate parameters against experiments",
           f"{_CLI} calibrate --mech 1 --db project.db --dataset 1",
           "Bayesian MCMC calibration of active rate parameters against "
           "loaded experimental targets.",
           ["calibrate", "optimize", "mcmc", "bayesian", "fit", "tune",
            "experiments", "posterior"]),
    Recipe("bound-to-bound consistency analysis (B2BDC)",
           f"{_CLI} b2bdc --mech 1 --db project.db --dataset 1",
           "Dataset consistency + feasible-set parameter bounds (SDP outer "
           "bounds via the cvxpy venv).",
           ["b2bdc", "bound to bound", "consistency", "feasible", "sdp",
            "dataset"]),
    Recipe("compare two or more mechanisms",
           f"{_CLI} compare --mechs 1 2 --db project.db --out comparison/",
           "Kinetics/thermo side-by-side diff of stored mechanisms.",
           ["compare", "difference", "diff", "versus", "two mechanisms"]),
    Recipe("check my RMG input file before a long run",
           f"{_CLI} rmg-lint input.py",
           "Static lint: duplicate libraries, seed/reactionLibraries "
           "dual-listing, inverted tolerances — before wasting a multi-day run.",
           ["rmg", "input", "lint", "libraries", "seed", "duplicate",
            "tolerance"]),
    Recipe("diff two RMG-generated mechanisms",
           f"{_CLI} rmg-diff runA/chem.inp runB/chem.inp",
           "Graph-canonical alignment of two RMG outputs: species/reactions "
           "added, removed, renamed (P15 structural half).",
           ["rmg", "diff", "reproducible", "versions", "drift", "outputs"]),
    Recipe("check for behaviour drift after a Cantera/version upgrade",
           f"{_CLI} regression",
           "Runs the canonical IDT/LFS/equilibrium suite and diffs against "
           "committed goldens; --env <venv python> tests another Cantera.",
           ["regression", "version", "upgrade", "cantera", "drift", "golden",
            "flame speed changed", "different results"]),
    Recipe("verify my toolchain before starting a task",
           f"{_CLI} doctor --task rmg",
           "Preflights every prerequisite (venvs, binaries, versions) with "
           "the exact fix for anything missing.",
           ["doctor", "prerequisites", "setup", "install", "environment",
            "missing", "toolchain", "onboarding"]),
    Recipe("simulate a jet-stirred reactor (JSR)",
           f"{_CLI} jsr mech.yaml --T 800 --P 1 --tau 1.0 --X 'CH4:0.01, O2:0.1, N2:0.89'",
           "Steady-state JSR species profiles over a temperature ramp.",
           ["jsr", "jet stirred", "reactor", "speciation", "profiles"]),
    Recipe("simulate a plug-flow reactor (PFR)",
           f"{_CLI} pfr mech.yaml --T 900 --P 1 --X 'CH4:0.05, O2:0.2, N2:0.75'",
           "PFR axial species/temperature profiles.",
           ["pfr", "plug flow", "reactor", "tubular", "profiles"]),
    Recipe("map NTC behaviour of a fuel",
           f"{_CLI} ntc --mech 1 --db project.db --fuel NC7H16:1.0",
           "IDT vs 1000/T sweep with automatic NTC-region detection.",
           ["ntc", "negative temperature coefficient", "cool flame",
            "low temperature", "idt curve"]),
    Recipe("check thermo data quality / provenance",
           f"{_CLI} thermo-check mech.yaml",
           "Cross-checks NASA-7 fits against reference data; flags "
           "discontinuities and outliers.",
           ["thermo", "nasa", "polynomial", "enthalpy", "provenance",
            "discontinuity", "cp"]),
    Recipe("design the most informative next experiments (DoE)",
           f"{_CLI} design-experiments --mech 1 --db project.db",
           "Optimal experiment design over the gPC surrogate.",
           ["doe", "design", "experiments", "optimal", "informative"]),
    Recipe("multi-objective Pareto mechanism optimization",
           f"{_CLI} pareto --mech 1 --db project.db",
           "NSGA-II Pareto front over competing targets.",
           ["pareto", "multi objective", "nsga", "trade off"]),
]


def find_recipes(query: str, limit: int = 3) -> List[Recipe]:
    """Rank recipes against a plain-language query.

    Scoring: exact keyword/task token hits dominate; a difflib similarity
    over the task phrase breaks ties, so near-miss wording still routes.
    """
    q = query.lower().strip()
    q_tokens = set(q.replace("-", " ").replace("_", " ").split())
    scored = []
    for r in RECIPES:
        hay_tokens = set(r.task.lower().split())
        for kw in r.keywords:
            hay_tokens.update(kw.lower().split())
        token_hits = len(q_tokens & hay_tokens)
        phrase_hits = sum(1 for kw in r.keywords if kw in q)
        fuzz = difflib.SequenceMatcher(None, q, r.task.lower()).ratio()
        score = 2.0 * token_hits + 3.0 * phrase_hits + fuzz
        scored.append((score, r))
    scored.sort(key=lambda t: -t[0])
    top = [r for score, r in scored[:limit] if score >= 2.0]
    if not top and scored:
        top = [scored[0][1]]        # always answer with the closest match
    return top


def render_recipes(recipes: List[Recipe], query: str = "") -> str:
    if not recipes:
        return "No matching recipe."
    lines = []
    if query:
        lines.append(f'How do I… "{query}"')
        lines.append("")
    for i, r in enumerate(recipes, 1):
        lines.append(f"{i}. {r.task}")
        lines.append(f"     $ {r.command}")
        lines.append(f"     {r.description}")
        if r.gui:
            lines.append(f"     GUI: python3 {r.gui} (or via launcher_gui.py)")
        lines.append("")
    return "\n".join(lines).rstrip()
