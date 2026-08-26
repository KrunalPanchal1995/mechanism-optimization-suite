"""
mso_studio.launcher.content
==============================
Plain-language "Learn more" text per catalog entry, and the Welcome tour's
page content. Pure Python (plain strings/dataclasses), no Qt -- kept
separate from ``catalog.py`` so the short tile/hero copy (taglines,
one-liners) and this longer explanatory copy can each be edited without
touching the other.

Every paragraph below follows the same shape: what it does, why a
combustion-mechanism researcher would reach for it, and what the result
physically means -- not just engineer-facing mechanics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

LEARN_MORE: dict[str, str] = {
    "mechanism_converter": (
        "A combustion mechanism is a list of chemical species and reactions with "
        "rate parameters -- the raw ingredients a simulator needs to predict how a "
        "fuel burns. Most published mechanisms ship in CHEMKIN's decades-old text "
        "format; Cantera, the solver this suite runs on, needs its own YAML format "
        "instead. This tool does that translation -- species, thermodynamic "
        "polynomials, transport properties, and every reaction's rate law -- while "
        "automatically fixing the handful of formatting quirks (Fortran-style "
        "exponents, Windows line endings, case-sensitivity slips) that make raw "
        "CHEMKIN files fail to load otherwise. Skip this and nothing downstream can "
        "run at all: it's the first step for bringing any new or updated mechanism "
        "into the suite."
    ),
    "csv2xml": (
        "Much combustion validation data exists only as points in a printed "
        "figure -- a plot digitized by clicking along a curve, or a small "
        "table typed in by hand. That raw (x, y) data isn't yet usable "
        "anywhere else in the suite: the target-file pipeline expects a "
        "ReSpecTh-schema XML with the specific structure the existing "
        "xml2csv.py converter enforces, not a bare CSV. This tool builds "
        "that XML -- experiment type, apparatus, composition, and every "
        "data point -- from a CSV plus a short description of the "
        "experiment (fuel, pressure, whether temperature or equivalence "
        "ratio varies), and immediately runs the result back through the "
        "real xml2csv.py converter so you see the actual .target row count "
        "it produces, not just a well-formed file that might still be "
        "rejected downstream. This is what turns a digitized figure into "
        "an experimental target the optimizer can actually tune against."
    ),
    "mechanism_check": (
        "Mechanisms are large, hand-edited files, and small mistakes hide easily: a "
        "rate constant faster than physically possible (exceeding the collision "
        "limit -- molecules literally cannot react more often than they collide), a "
        "species' thermodynamic properties that jump discontinuously at the "
        "temperature where two curve-fits are supposed to meet smoothly, a reaction "
        "typed in twice. Any one of these can make a simulation silently produce "
        "wrong answers, or crash outright. This tool runs a battery of physical and "
        "structural sanity checks over a mechanism and, for the common faults, can "
        "repair them automatically -- catching problems before hours of simulation "
        "time are spent on a mechanism that was broken from the start."
    ),
    "mechanism_fusion": (
        "Different research groups publish different rate constants for the same "
        "reaction, often disagreeing by a factor of 2 or more. Rather than trusting "
        "one source, this tool merges several mechanisms into a single consensus: "
        "for every reaction two or more sources agree exists, it computes a "
        "geometric-mean rate constant and an honest uncertainty band spanning the "
        "underlying disagreement, then checks that every merged species still "
        "balances chemically. The result is a mechanism whose stated rate constants "
        "reflect the real spread of expert opinion in the literature, not one "
        "author's choice -- and a starting point the suite's optimization tools can "
        "then tune against real experimental data."
    ),
    "mechanism_reduction": (
        "A detailed mechanism can have hundreds of species and thousands of "
        "reactions, most of which barely matter for a given fuel/condition -- "
        "carrying them all costs simulation time for no accuracy benefit, and can "
        "even slow convergence. Skeletal reduction methods (DRG, DRGEP, PFA, "
        "DRGPSA, DRGSA) trace which species and reactions are actually coupled to "
        "your target species (usually fuel, O2, and the main products) at a given "
        "error tolerance ε, and drop everything below that coupling threshold. "
        "Multi-condition union-reduce runs this across several operating "
        "conditions at once and keeps the union of what each condition needs, so "
        "the reduced mechanism stays valid across the whole range you actually "
        "care about, not just one point."
    ),
    "brute_force_sa": (
        "Not every reaction in a mechanism actually matters for a given prediction "
        "-- most barely move the answer, and a handful dominate it. This tool finds "
        "out which is which for ignition timing: it perturbs each reaction's rate "
        "parameters (A, n, Ea -- the pre-exponential factor, temperature exponent, "
        "and activation energy of the Arrhenius rate law) one at a time and measures "
        "how much the simulated result shifts, using real finite-difference "
        "simulations rather than an approximation. Knowing which handful of "
        "reactions actually drive the prediction tells you exactly where "
        "uncertainty in the literature matters and where it's noise -- and which "
        "rate constants are worth spending optimization effort tuning."
    ),
    "idt_sensitivity": (
        "Ignition delay time -- how long a fuel/air mixture sits before it "
        "spontaneously ignites -- is one of the most commonly measured combustion "
        "quantities, and one of the most sensitive to mechanism accuracy. This tool "
        "computes exactly which reactions control that delay at a given temperature, "
        "pressure, and equivalence ratio, ranking them by d ln(τ)/d ln(k) (how much "
        "a small change in a reaction's rate shifts the ignition delay, on a "
        "logarithmic scale) and presenting them as a tornado chart -- the reactions "
        "with the longest bars are the ones actually controlling when the mixture "
        "ignites. That ranking is the map for where to focus mechanism refinement or "
        "optimization."
    ),
    "nominal_sim": (
        "Before comparing mechanisms or running a full optimization campaign, it's "
        "often useful to just run one simulation and see the answer -- a single "
        "shock-tube ignition delay, a rapid-compression-machine point, a specific "
        "named condition from a target file. This tool runs exactly that one case "
        "with a chosen mechanism and shows the result, without the overhead of "
        "setting up a comparison or optimization run. It's the quickest way to sanity"
        "-check a mechanism against a single real data point, or to confirm a "
        "specific condition behaves as expected before committing it to a larger "
        "campaign."
    ),
    "mso_compare": (
        "Judging whether one mechanism is actually better than another means "
        "running both against the same real experimental targets and looking at the "
        "errors side by side. This tool runs that full four-phase pipeline: simulate "
        "every target condition with each mechanism, merge the results into a common "
        "structure, compute error metrics against the observed data, and generate "
        "comparison plots -- all in one pass across as many as three mechanism "
        "variants (e.g. before and after an optimization stage) at once. The output "
        "is a direct, apples-to-apples answer to 'did this change actually improve "
        "agreement with experiment.'"
    ),
    "forward_uq": (
        "A mechanism's rate constants are never known exactly -- every one carries "
        "real experimental or theoretical uncertainty. This tool asks: if you take "
        "that uncertainty seriously and propagate it through the simulation (via "
        "Monte-Carlo sampling over a fast polynomial-response-surface stand-in for "
        "the full solver, plus Sobol sensitivity indices), how uncertain is the "
        "final prediction, and which reactions' uncertainty contributes the most to "
        "that spread? The output is a confidence band around the simulated result, "
        "not just a single number -- and a ranked list of which reactions' "
        "uncertainty is actually worth tightening if you want a more confident "
        "prediction."
    ),
    "uncertainty_generator": (
        "Optimization and uncertainty-propagation tools need to know, for every "
        "reaction, how far its rate constant is allowed to move -- an uncertainty "
        "bound, usually expressed as a factor on the pre-exponential term or a range "
        "on activation energy. This tool builds those bounds from sensitivity "
        "analysis results and a classification of each reaction's uncertainty class, "
        "producing the XML/CSV files the rest of the suite consumes, along with "
        "screening plots and LaTeX-ready tables. Getting these bounds right matters "
        "directly: too tight and a real improvement gets ruled out before "
        "optimization can find it; too loose and the search wastes time exploring "
        "physically implausible rate constants."
    ),
    "rmip_suite": (
        "A full combustion mechanism can have hundreds of species and thousands of "
        "reactions, but for a specific application (a specific fuel, temperature "
        "range, engine condition) most of them are irrelevant. This tool treats the "
        "mechanism as a graph -- species and reactions as nodes, connected by who "
        "reacts with whom -- and uses that structure to compare two mechanisms "
        "directly, systematically reduce a large mechanism down to the smallest "
        "reaction set that still reproduces the target behavior, run global "
        "sensitivity analysis across the whole reaction network at once, and select "
        "which parameters are worth including in an optimization. It's the tool for "
        "\"can I use a smaller, faster mechanism here without losing accuracy.\""
    ),
    "target_add_opt": (
        "This is the toolkit for building the input files an actual optimization "
        "campaign runs on: `.target` files (the real experimental data points a "
        "mechanism is being tuned against -- ignition delays, flame speeds, species "
        "profiles, each with its own uncertainty), `.add` files (which reactions are "
        "allowed to move and by how much), and `.opt` files (the campaign "
        "configuration itself), plus the native-extension build step some solver "
        "backends need. It's the pre-split original all-in-one tool -- every "
        "individual capability here also exists as one of the more focused apps "
        "above, kept as a single workspace for building an optimization campaign "
        "end to end."
    ),
    "mso_studio": (
        "Every capability above -- conversion, health-checking, fusion, sensitivity, "
        "uncertainty, comparison, reduction, optimization -- also exists here, in "
        "one unified window with a shared results/report/log system, instead of as "
        "13 separate programs. It also carries the tools that don't fit anywhere "
        "else in the catalog: curve/symbolic fitting, an interactive plot digitizer "
        "for pulling numeric data out of a published figure, and PDF question-"
        "answering/table-extraction for mining literature papers directly. If "
        "you're going to be doing sustained work across several of these "
        "capabilities in one sitting, this is the more efficient place to do it."
    ),
    "optimize_workbench": (
        "The four pages that make up the actual tuning workflow -- Sensitivity "
        "Analysis, Target Workbench, Optimization, Forward UQ -- bundled into one "
        "dedicated window instead of requiring the full MSO Studio shell. Same "
        "PySide6 pages either way; this is scoped to just the optimization "
        "workflow so nothing else is in the way while you work the tuning loop."
    ),
    "optimization": (
        "Once you know which reactions matter (sensitivity analysis) and how far "
        "their rate constants are allowed to move (uncertainty bounds), "
        "optimization searches within those bounds for the combination that best "
        "matches a set of real experimental targets -- ignition delays, flame "
        "speeds, species profiles -- simultaneously, not one at a time. The result "
        "is a mechanism variant with the same physical structure but rate "
        "constants nudged, within their honest uncertainty, toward better "
        "agreement with measured reality."
    ),
    "target_workbench": (
        "Experimental target files are hand-assembled and prone to the same small "
        "mistakes as mechanisms: a composition that doesn't sum to 1, a deviation "
        "of exactly 0 that would dominate the objective, duplicate conditions "
        "double-counted. This page loads a .target/.add pair, runs a battery of "
        "sanity checks, generates NTC-aware dummy points to densify sparse "
        "ignition-delay datasets, suggests which rows can be dropped without "
        "losing T/P/phi coverage, and writes a new versioned pair -- the source "
        "file is never touched, and every write is gated by re-parsing the result "
        "with the real combustion_target parser before calling it done."
    ),
    "tools": (
        "The utilities that don't fit anywhere else: fitting a curve or symbolic "
        "expression to data, an interactive digitizer for pulling numeric values "
        "back out of a published figure, and PDF question-answering/table "
        "extraction for mining a literature paper's reported data directly instead "
        "of transcribing it by hand."
    ),
    "project_manager": (
        "A campaign's real record of what actually happened -- which pipeline "
        "ran, when, with what inputs, and every file it produced -- indexed under "
        "one project directory instead of scattered across whatever working "
        "directory each tool happened to be run from. Create or open that "
        "project, browse its full run history and registered artifacts, add "
        "freeform notes, resume a prior run's artifacts into a fresh directory, "
        "and build the cross-pipeline LaTeX report from everything registered "
        "so far."
    ),
    "doctor": (
        "Most of this suite's heavier capabilities depend on an external "
        "environment being set up correctly -- a specific Python virtual "
        "environment, an external solver binary, a particular library version -- and "
        "the failure mode when one is missing is usually a confusing crash deep "
        "inside a run, hours after starting it. This tool checks, up front, whether "
        "everything a given task needs is actually present and working, and tells "
        "you exactly what to install and how if it isn't -- plus a plain-language "
        "\"how do I...\" search that routes a task description straight to the right "
        "command or app. It's the tool for finding out a problem BEFORE it costs you "
        "a wasted run, not after."
    ),
}


@dataclass(frozen=True)
class WelcomePage:
    title: str
    body: str


WELCOME_PAGES: List[WelcomePage] = [
    WelcomePage(
        title="Welcome to the Mechanism Optimization Suite",
        body=(
            "This suite exists to answer one question, over and over, as reliably "
            "as possible: given a proposed chemical-kinetics mechanism for how a "
            "fuel burns, how well does it actually predict reality -- and how can "
            "it be made to predict reality better? Combustion mechanisms are the "
            "chemistry engines inside engine simulations, safety analyses, and "
            "emissions predictions; getting them right, with honestly quantified "
            "uncertainty, is the whole point of the tools you're about to see."
        ),
    ),
    WelcomePage(
        title="Five stages, left to right",
        body=(
            "The sidebar groups every tool by where it sits in a typical workflow. "
            "PREPARE gets a mechanism into a usable, verified state (convert "
            "formats, check for faults, merge multiple sources into one "
            "consensus). ANALYSE finds out which reactions actually matter for a "
            "given prediction. QUANTIFY & OPTIMISE turns that understanding into "
            "honest uncertainty bounds and, ultimately, a better-tuned mechanism. "
            "INTEGRATED is the single unified window covering all of the above "
            "plus a few extras. UTILITIES is the toolchain doctor and command "
            "finder. You don't have to follow this order -- but if you're not sure "
            "where to start, it's a reasonable one."
        ),
    ),
    WelcomePage(
        title="Finding your way around",
        body=(
            "Click a tile to preview it in the panel above -- title, description, "
            "and a real result from that tool's own domain, not a generic icon. "
            "Hovering just lifts the tile visually; it won't change what LAUNCH "
            "acts on, so you can browse freely. Search filters by name and "
            "description as you type. Star a tile to pin it under Favourites; "
            "anything you've launched shows up under Recent. Each tile's LAUNCH "
            "button starts that tool as its own independent process, so a crash "
            "in one can never take down the launcher or anything else you have "
            "open."
        ),
    ),
    WelcomePage(
        title="If something won't launch",
        body=(
            "Open Doctor / Recipes from the sidebar or the status strip at the "
            "bottom of this window. Doctor checks whether everything a given task "
            "needs -- a Python environment, an external solver, a library version "
            "-- is actually present, and tells you exactly what's missing and how "
            "to fix it, before you spend time on a run that was doomed to fail. "
            "Recipes answers plain-language questions like \"how do I convert a "
            "mechanism\" by pointing you straight at the right tool or command."
        ),
    ),
]


# ===========================================================================
# MSO Studio module-page hero metadata (Phase 5 rollout).
# ===========================================================================
# mso_studio.gui.modules.PAGE_FACTORIES module ids don't all match the
# launcher catalog ids 1:1 (e.g. "uncertainty" vs "uncertainty_generator",
# and "sensitivity" covers what the launcher splits into two separate tiles
# -- brute_force_sa and idt_sensitivity) -- this table is the single,
# explicit mapping from a Studio module id to its hero's icon, one-line
# tagline, the generated-art asset key to reuse (see art.ASSET_DIR;
# "" means no dedicated asset exists yet, so ModuleHero falls back to the
# plain gradient+glyph rendering, same fallback tile_art() already has),
# and its learn-more paragraph -- reusing LEARN_MORE's text where the
# module covers the same ground as a launcher tile, with fresh text for
# the three Studio-only pages (optimization/automation/tools) that have no
# launcher-catalog equivalent.
@dataclass(frozen=True)
class ModuleMeta:
    icon: str
    tagline: str
    art_key: str
    learn_more: str
    # Key into launcher.catalog.STAGES, so each Studio module hero gets a
    # gradient matching its workflow stage instead of one fixed gradient
    # for all 12 (the original design deliberately avoided threading
    # launcher-stage grouping through Studio's nav, since module ids don't
    # all map 1:1 to catalog ids -- this is the same mapping made explicit
    # rather than left as a single hardcoded stage).
    stage: str = "prepare"


MODULE_META: dict[str, ModuleMeta] = {
    "mechanism_converter": ModuleMeta(
        icon="fa5s.exchange-alt", art_key="mechanism_converter",
        tagline="CHEMKIN .inp/thermo/trans → Cantera YAML",
        learn_more=LEARN_MORE["mechanism_converter"], stage="prepare"),
    "csv2xml": ModuleMeta(
        icon="fa5s.file-export", art_key="csv2xml",
        tagline="Turn a digitized/tabulated CSV into a real .target",
        learn_more=LEARN_MORE["csv2xml"], stage="prepare"),
    "mechanism_check": ModuleMeta(
        icon="fa5s.stethoscope", art_key="mechanism_check",
        tagline="Diagnose and auto-repair a mechanism",
        learn_more=LEARN_MORE["mechanism_check"], stage="prepare"),
    "mechanism_fusion": ModuleMeta(
        icon="fa5s.layer-group", art_key="mechanism_fusion",
        tagline="Merge N mechanisms into a consensus set",
        learn_more=LEARN_MORE["mechanism_fusion"], stage="prepare"),
    "mechanism_reduction": ModuleMeta(
        icon="fa5s.compress-arrows-alt", art_key="mechanism_reduction",
        tagline="DRG/DRGEP/PFA/DRGPSA/CSP graph reduction",
        learn_more=LEARN_MORE["mechanism_reduction"], stage="prepare"),
    "uncertainty": ModuleMeta(
        icon="fa5s.flask", art_key="uncertainty_generator",
        tagline="Build uncertainty bounds for optimization/UQ",
        learn_more=LEARN_MORE["uncertainty_generator"], stage="quantify"),
    "forward_uq": ModuleMeta(
        icon="fa5s.wave-square", art_key="forward_uq",
        tagline="PRS-surrogate Monte-Carlo/Sobol UQ",
        learn_more=LEARN_MORE["forward_uq"], stage="quantify"),
    "nominal_sim": ModuleMeta(
        icon="fa5s.thermometer-half", art_key="nominal_sim",
        tagline="Run one named nominal simulation",
        learn_more=LEARN_MORE["nominal_sim"], stage="explore"),
    "mso_compare": ModuleMeta(
        icon="fa5s.chart-line", art_key="mso_compare",
        tagline="Full 4-phase comparison pipeline",
        learn_more=LEARN_MORE["mso_compare"], stage="compare"),
    "sensitivity": ModuleMeta(
        icon="fa5s.sliders-h", art_key="idt_sensitivity",
        tagline="Which reactions actually control the result",
        stage="explore",
        learn_more=(
            "Two ways to answer the same question -- which reactions actually "
            "control a prediction. Brute-force perturbs each reaction's rate "
            "parameters and re-runs real simulations to measure the effect "
            "directly; ignition-delay sensitivity computes d ln(τ)/d ln(k) "
            "specifically for ignition timing and ranks the results as a tornado "
            "chart. Both point at the same thing: where mechanism uncertainty "
            "actually matters for your prediction, and where it's noise you can "
            "safely ignore."
        )),
    "optimization": ModuleMeta(
        icon="fa5s.bullseye", art_key="target_add_opt",
        tagline="Tune rate constants against real experimental data",
        stage="optimise",
        learn_more=(
            "Once you know which reactions matter (sensitivity analysis) and how "
            "far their rate constants are allowed to move (uncertainty bounds), "
            "optimization searches within those bounds for the combination that "
            "best matches a set of real experimental targets -- ignition delays, "
            "flame speeds, species profiles -- simultaneously, not one at a time. "
            "The result is a mechanism variant with the same physical structure "
            "but rate constants nudged, within their honest uncertainty, toward "
            "better agreement with measured reality."
        )),
    "target_workbench": ModuleMeta(
        icon="fa5s.bullseye", art_key="",
        tagline="Curate a .target/.add pair before it reaches Addendum",
        stage="optimise",
        learn_more=(
            "Experimental target files are hand-assembled and prone to the same "
            "small mistakes as mechanisms: a composition that doesn't sum to 1, a "
            "deviation of exactly 0 that would dominate the objective, duplicate "
            "conditions double-counted. This page loads a .target/.add pair, runs "
            "a battery of sanity checks, generates NTC-aware dummy points to "
            "densify sparse ignition-delay datasets, suggests which rows can be "
            "dropped without losing T/P/phi coverage, and writes a new versioned "
            "pair -- the source file is never touched, and every write is gated "
            "by re-parsing the result with the real combustion_target parser "
            "before calling it done."
        )),
    "optimize_workbench": ModuleMeta(
        icon="fa5s.bullseye", art_key="target_add_opt",
        tagline="Sensitivity, uncertainty-adjacent optimization, and target curation in one window",
        stage="optimise",
        learn_more=(
            "The four pages that make up the actual tuning workflow -- Sensitivity "
            "Analysis, Optimization, Forward UQ, and Target Workbench -- bundled "
            "into one dedicated window instead of requiring the full MSO Studio "
            "shell. Reachable both as its own launcher tile and as ordinary pages "
            "inside MSO Studio; this is the same PySide6 pages, just scoped to the "
            "optimization workflow so nothing else is in the way."
        )),
    "automation": ModuleMeta(
        icon="fa5s.robot", art_key="",
        tagline="Batch-run comparison/optimization tasks unattended",
        stage="compare",
        learn_more=(
            "Mechanism-comparison and optimization campaigns often mean running "
            "the same pipeline across many mechanism variants or target sets in "
            "sequence -- tedious and error-prone to drive by hand. This page "
            "queues and runs those tasks unattended, plus fits a fast polynomial "
            "response surface (Meta-PRS) as a cheap stand-in for the full solver "
            "when a campaign needs many evaluations. It's the tool for turning a "
            "one-off comparison into a repeatable, walk-away campaign."
        )),
    "rmip_suite": ModuleMeta(
        icon="fa5s.project-diagram", art_key="rmip_suite",
        tagline="Graph comparison, reduction, global SA",
        learn_more=LEARN_MORE["rmip_suite"], stage="explore"),
    "tools": ModuleMeta(
        icon="fa5s.toolbox", art_key="",
        tagline="Curve fitting, plot digitizer, literature mining",
        stage="utilities",
        learn_more=(
            "The utilities that don't fit anywhere else: fitting a curve or "
            "symbolic expression to data, an interactive digitizer for pulling "
            "numeric values back out of a published figure (paste in a plot "
            "image, click along the curve, get real numbers back), and PDF "
            "question-answering / table extraction for mining a literature paper's "
            "reported data directly instead of transcribing it by hand. Each one "
            "solves the same underlying problem -- getting real numbers out of a "
            "source that isn't already a clean CSV."
        )),
    "project_manager": ModuleMeta(
        icon="fa5s.folder-open", art_key="",
        tagline="Run history, artifacts, notes, and reports for a project",
        stage="utilities",
        learn_more=(
            "A campaign's real record of what actually happened -- which "
            "pipeline ran, when, with what inputs, and every file it "
            "produced -- indexed under one project directory instead of "
            "scattered across whatever working directory each tool happened "
            "to be run from. This page creates or opens that project, "
            "browses its full run history and registered artifacts, adds "
            "freeform notes, resumes a prior run's artifacts into a fresh "
            "directory, and builds the cross-pipeline LaTeX report from "
            "everything registered so far."
        )),
}


# ===========================================================================
# Structured module help (Ansys/STAR-CCM+-style per-module documentation).
# ===========================================================================
# LEARN_MORE (above) is a single explanatory paragraph. This is the same
# information broken into the fixed shape Ansys Workbench and STAR-CCM+ use
# for every module's help: what it's for, what it needs, what it produces,
# what has to happen first, a worked scenario, gotchas, and related tools.
# Keyed the same as MODULE_META -- launcher CATALOG ids plus the four
# Studio-only ids (sensitivity/optimization/automation/tools) that have no
# launcher-catalog equivalent. Content is drawn from LEARN_MORE, each
# module's own docstring/description, the mso_project footprints/ registry
# (the ground truth for what a pipeline actually writes), and
# MarkDownFiles/README.md's documented step ordering -- not invented.
@dataclass(frozen=True)
class ModuleHelp:
    purpose: str
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    prerequisites: Tuple[str, ...]
    typical_use: str
    tips: Tuple[str, ...]
    see_also: Tuple[str, ...]


MODULE_HELP: Dict[str, ModuleHelp] = {
    "mechanism_converter": ModuleHelp(
        purpose=(
            "Translates a CHEMKIN-format mechanism (species, thermodynamic "
            "polynomials, transport properties, reaction rate laws) into the "
            "Cantera YAML format every other tool in this suite runs on, "
            "auto-fixing the formatting quirks (Fortran-style exponents, "
            "Windows line endings, case-sensitivity slips) that make raw "
            "CHEMKIN files fail to load otherwise."
        ),
        inputs=("CHEMKIN .inp mechanism file", "CHEMKIN thermo file",
                "CHEMKIN transport file (optional)"),
        outputs=("Cantera mechanism (.yaml)", "FlameMaster .pre file",
                 "RMIP SQLite database entry"),
        prerequisites=("None -- this is the entry point for a new or "
                       "updated mechanism."),
        typical_use=(
            "You've downloaded a mechanism from a supplemental-materials ZIP "
            "or a group's website and it's in CHEMKIN's .inp/.dat format. "
            "Run this first, before anything else touches the mechanism."
        ),
        tips=("If the converter reports a parse failure, check for Fortran "
              "exponents (1.0D+10 instead of 1.0E+10) or CRLF line endings "
              "first -- both are auto-fixed, but a report means something "
              "else is wrong.",
              "Always run Mechanism Health-Check on the output before using "
              "it anywhere else."),
        see_also=("mechanism_check", "mechanism_fusion"),
    ),
    "csv2xml": ModuleHelp(
        purpose=(
            "Builds a ReSpecTh-schema XML from a CSV of (x, y[, y_plus, "
            "y_minus]) points -- including the plot digitizer's own CSV "
            "export -- plus a short description of the experiment, then "
            "immediately round-trip-checks the result through the real "
            "xml2csv.py converter so you know the actual .target row count "
            "it produces, not just whether the XML is well-formed."
        ),
        inputs=("Digitized or hand-tabulated CSV (x, y[, y_plus, y_minus])",
                "Target type (Tig / Fls / Flf_a)", "Fuel composition",
                "Fixed pressure and (temperature or equivalence ratio)"),
        outputs=("ReSpecTh-schema XML", "Round-trip .target preview and "
                 "row count from the real xml2csv.py converter"),
        prerequisites=("A digitized CSV (Tools -> plot digitizer) or a "
                       "hand-prepared one.",),
        typical_use=(
            "You've digitized a curve from a published figure, or typed in "
            "a small table of ignition-delay or laminar-burning-velocity "
            "points by hand, and need it as a real experimental target "
            "xml2csv.py can turn into a .target/.add pair."
        ),
        tips=("If the round-trip check reports 0 rows, the most common "
              "cause is a fuel name the round-trip's fuel_list doesn't "
              "include -- this page always passes your own Fuel "
              "composition field through, so double-check it matches the "
              "species names in your CSV's intended composition.",
              "A 3rd/4th CSV column is read as an uncertainty "
              "(y_plus/y_minus) band, not a second observable."),
        see_also=("tools", "target_workbench"),
    ),
    "mechanism_check": ModuleHelp(
        purpose=(
            "Runs 13 physical and structural sanity checks over a mechanism "
            "(rate NaN / collision-limit / thermo-discontinuity / duplicate "
            "reactions / third-body duplicates / PLOG sanity / element "
            "balance / unit sanity / completeness / CSP timescales / syntax) "
            "and auto-repairs the common, unambiguous faults."
        ),
        inputs=("Cantera mechanism (.yaml)",),
        outputs=("Diagnostic report", "Repaired mechanism (.fixed.yaml) "
                 "when auto-repair applies"),
        prerequisites=("Mechanism Converter, if the mechanism is still in "
                       "CHEMKIN format.",),
        typical_use=(
            "After converting or fusing a mechanism, and before spending any "
            "simulation time on it -- catching a broken mechanism here costs "
            "seconds; catching it after a failed overnight campaign costs a "
            "day."
        ),
        tips=("A collision-limit hit almost always means a typo in the "
              "pre-exponential factor's exponent, not a genuinely fast "
              "reaction.",
              "Thermo discontinuities are easy to miss by eye but make CVODE "
              "integration fail intermittently, far from the actual cause."),
        see_also=("mechanism_converter", "mechanism_fusion"),
    ),
    "mechanism_fusion": ModuleHelp(
        purpose=(
            "Merges two or more mechanisms into one consensus mechanism: "
            "geometric-mean rate constants for reactions where sources agree "
            "but disagree on the number, with an honest uncertainty band "
            "spanning the real spread, plus element-balance validation on "
            "every merged species."
        ),
        inputs=("Two or more Cantera mechanisms (.yaml)",),
        outputs=("Fused mechanism (.yaml and CHEMKIN .inp)",
                 "Fusion uncertainty table", "Fusion provenance record"),
        prerequisites=("Every input mechanism should already have passed "
                       "Mechanism Health-Check.",),
        typical_use=(
            "You have two published mechanisms for the same fuel that "
            "disagree on several key reactions, and want one defensible "
            "starting point -- with uncertainty bounds derived from that "
            "disagreement -- to hand to Sensitivity Analysis and "
            "Optimization."
        ),
        tips=("A reaction only appears with a fused uncertainty band if two "
              "or more sources actually contain it; reactions unique to one "
              "source pass through unchanged.",),
        see_also=("mechanism_check", "uncertainty_generator"),
    ),
    "mechanism_reduction": ModuleHelp(
        purpose=(
            "Skeletal mechanism reduction via graph-based coupling analysis "
            "(DRG/DRGEP/PFA/DRGPSA/DRGSA): traces which species and "
            "reactions are actually coupled to the target species at a "
            "chosen error tolerance ε, and drops everything below that "
            "threshold. Multi-condition union-reduce runs the sweep across "
            "several operating conditions and keeps the union of what each "
            "needs, so the reduced mechanism stays valid across the whole "
            "range of interest."
        ),
        inputs=("Cantera mechanism (.yaml)", "Target species",
                "Operating condition(s): T, P, φ", "ε threshold or sweep range"),
        outputs=("Reduction steps JSON (species/reaction count vs. ε)",
                 "Reduced mechanism (.yaml)"),
        prerequisites=("Mechanism Health-Check should already pass on the "
                       "full mechanism.",),
        typical_use=(
            "You have a detailed, validated mechanism and need a smaller one "
            "for a CFD run or a large sampling campaign where solve time "
            "matters, without losing accuracy for the fuel/conditions you "
            "actually care about."
        ),
        tips=("Run the ε sweep before committing to one threshold -- the "
              "species/reaction count vs. ε curve usually has a knee where "
              "further reduction starts costing real accuracy.",
              "Multi-condition union-reduce costs more species/reactions "
              "than any single condition alone, but the result is honestly "
              "valid across all of them -- a single-condition reduction "
              "silently is not."),
        see_also=("mechanism_check", "rmip_suite"),
    ),
    "brute_force_sa": ModuleHelp(
        purpose=(
            "Perturbs each reaction's Arrhenius parameters (A, n, Ea) one at "
            "a time and re-runs real finite-difference simulations to "
            "measure how much the result shifts -- not an approximation, "
            "genuine forward/central-difference sensitivity."
        ),
        inputs=("Cantera mechanism (.yaml)", "target.opt configuration"),
        outputs=("Sensitivity coefficient table", "Sensitivity run log"),
        prerequisites=("Mechanism must pass Mechanism Health-Check.",
                       "A target.opt file (Target/Add/Opt Generator)."),
        typical_use=(
            "Before committing to an optimization campaign, find out which "
            "handful of reactions actually control the targets you care "
            "about, so the campaign's uncertainty bounds and search effort "
            "focus there instead of on reactions that barely move the "
            "answer."
        ),
        tips=("This re-runs the solver per perturbation, so it scales with "
              "reaction count -- prefer Ignition-Delay Sensitivity for a "
              "quick single-condition look first.",),
        see_also=("idt_sensitivity", "uncertainty_generator", "optimization"),
    ),
    "idt_sensitivity": ModuleHelp(
        purpose=(
            "Computes d ln(tau)/d ln(k) -- how much a small change in each "
            "reaction's rate constant shifts ignition delay, on a "
            "logarithmic scale -- for one temperature/pressure/equivalence-"
            "ratio condition, and ranks the results as a tornado chart."
        ),
        inputs=("Cantera mechanism (.yaml)", "T / P / phi condition"),
        outputs=("Ranked sensitivity result", "Tornado chart figure"),
        prerequisites=("Mechanism must pass Mechanism Health-Check.",),
        typical_use=(
            "A quick, single-condition answer to \"which reactions control "
            "ignition timing here\" -- faster to set up than the full "
            "brute-force sweep when you only care about one condition."
        ),
        tips=("Selectable IDT definition (max dT/dt vs. species-based) can "
              "shift the ranking meaningfully near the ignition onset -- "
              "check both if the top reaction looks surprising.",),
        see_also=("brute_force_sa", "nominal_sim"),
    ),
    "nominal_sim": ModuleHelp(
        purpose=(
            "Runs one named simulation (a single shock-tube ignition delay, "
            "an RCM point, any one named condition from a target file) with "
            "a chosen mechanism, without the overhead of a full comparison "
            "or optimization setup."
        ),
        inputs=("Cantera mechanism (.yaml)", "A named target condition"),
        outputs=("Simulation summary", "Result data (.csv)", "Run log"),
        prerequisites=("Mechanism must pass Mechanism Health-Check.",),
        typical_use=(
            "You just fused or optimized a mechanism and want to sanity-"
            "check one specific real data point before committing it to a "
            "full MSO Compare campaign."
        ),
        tips=("This is explicitly the suite's reference simulation "
              "implementation -- if a result here looks wrong, the same "
              "condition inside MSO Compare or an SA run will look wrong "
              "too.",),
        see_also=("mso_compare", "idt_sensitivity"),
    ),
    "mso_compare": ModuleHelp(
        purpose=(
            "Runs the full four-phase comparison pipeline -- simulate every "
            "target condition, merge results into a common structure, "
            "compute error metrics against observed data, generate "
            "comparison plots -- across up to six mechanism variants at "
            "once (e.g. nominal vs. an optimization stage, HTC vs. LTC)."
        ),
        inputs=("One or more Cantera mechanisms (.yaml)",
                "mso_config.yaml pipeline configuration", "Target data"),
        outputs=("Merged comparison dataset", "Error-metric summary",
                 "Comparison figures"),
        prerequisites=("Each mechanism should pass Mechanism Health-Check.",
                       "A working mso_config.yaml (Target/Add/Opt Generator "
                       "or RMIP Suite's Target Builder)."),
        typical_use=(
            "Answering \"did this optimization stage actually improve "
            "agreement with experiment\" by running the before/after "
            "mechanisms against the same targets side by side."
        ),
        tips=("Up to six run labels (nominal/stage1/stage2 x htc/ltc) can "
              "run in one pass -- set up mso_config.yaml once rather than "
              "invoking this per variant.",),
        see_also=("nominal_sim", "forward_uq"),
    ),
    "forward_uq": ModuleHelp(
        purpose=(
            "Propagates Arrhenius rate-constant uncertainty through a "
            "trained polynomial-response-surface (PRS) surrogate via Monte-"
            "Carlo / Sobol / LHS sampling, producing a confidence band on "
            "predicted observables and Sobol indices ranking which "
            "reactions' uncertainty drives that spread."
        ),
        inputs=("Cached PRS coefficients (from a prior optimization run)",
                "target.opt configuration", "Uncertainty bounds (XML)"),
        outputs=("Observable uncertainty distributions", "Sobol indices",
                 "Forward-UQ figures"),
        prerequisites=("A completed Optimization run with cached PRS "
                       "coefficients under Opt/.",
                       "Uncertainty Quantification and File Generator, for "
                       "the bounds."),
        typical_use=(
            "After optimizing a mechanism, quantify how confident the "
            "resulting predictions actually are given the remaining rate-"
            "constant uncertainty, rather than reporting a single number."
        ),
        tips=("This needs an existing PRS cache -- it does not run new "
              "Cantera simulations itself, so it's fast, but only as "
              "trustworthy as the surrogate it's built on.",),
        see_also=("optimization", "mso_compare"),
    ),
    "uncertainty_generator": ModuleHelp(
        purpose=(
            "Builds the uncertainty bounds (a factor on the pre-exponential "
            "term, or a range on activation energy) every downstream "
            "optimization and forward-UQ tool needs, from sensitivity-"
            "analysis results and a per-reaction uncertainty classification."
        ),
        inputs=("Sensitivity analysis output", "Mechanism (.yaml)",
                "Uncertainty classification CSV"),
        outputs=("Uncertainty bounds (XML/CSV)", "Screening plots",
                 "LaTeX-ready tables", "Filtered mechanism"),
        prerequisites=("Brute-Force SA or Ignition-Delay Sensitivity, for "
                       "the input sensitivity data.",),
        typical_use=(
            "After sensitivity analysis identifies which reactions matter, "
            "turn that into the honest uncertainty bounds an optimization "
            "campaign or Forward UQ run will search within."
        ),
        tips=("Bounds too tight can rule out a real improvement before "
              "optimization can find it; too loose wastes the search "
              "exploring physically implausible rate constants -- this is "
              "worth getting right, not just filling in defaults.",),
        see_also=("brute_force_sa", "optimization", "forward_uq"),
    ),
    "rmip_suite": ModuleHelp(
        purpose=(
            "Treats a mechanism as a graph -- species and reactions as "
            "nodes -- to compare two mechanisms structurally, reduce a "
            "large mechanism to the smallest reaction set that still "
            "reproduces target behavior, run global sensitivity analysis "
            "across the whole network, select active parameters for "
            "optimization, build target files, and import ReSpecTh/RKD "
            "experimental data."
        ),
        inputs=("One or two mechanisms parsed into the RMIP database",
                "ReSpecTh/RKD XML experiment files (for import)"),
        outputs=("Graph comparison report", "Reduced mechanism",
                 "Global-SA table", "Active-parameter selection",
                 "Target files"),
        prerequisites=("Mechanism must already be parsed into an RMIP "
                       "database (rmip_cli.py parse, or via the Converter).",),
        typical_use=(
            "\"Can I use a smaller, faster mechanism here without losing "
            "accuracy\" -- or building the experimental-target files an "
            "optimization campaign needs from a ReSpecTh dataset."
        ),
        tips=("The Graph Comparison tab opens a separate graph-viewer "
              "process on its own port -- close it from the tab, not just "
              "the browser window, to free the port.",),
        see_also=("mechanism_fusion", "target_add_opt"),
    ),
    "target_add_opt": ModuleHelp(
        purpose=(
            "The original all-in-one toolkit for building the input files "
            "an optimization campaign runs on: .target files (real "
            "experimental data points with their own uncertainty), .add "
            "files (which reactions may move and by how much), .opt files "
            "(the campaign configuration), plus a native-extension build "
            "tab and LaTeX compile."
        ),
        inputs=("Experimental data (ignition delay, flame speed, species "
                "profiles)", "Mechanism reaction list"),
        outputs=(".target files", ".add files", ".opt configuration file"),
        prerequisites=("None strictly, but a mechanism that has passed "
                       "Mechanism Health-Check makes reaction indices "
                       "meaningful.",),
        typical_use=(
            "Every capability here also exists as one of the more focused "
            "apps (RMIP Suite's Target Builder, Uncertainty Generator) -- "
            "reach for this when you want the whole campaign-setup "
            "workspace in one window instead of switching between them."
        ),
        tips=("This is the pre-split monolith -- new features generally "
              "land in the focused apps first; check there before assuming "
              "a capability is missing here.",),
        see_also=("rmip_suite", "uncertainty_generator", "optimization"),
    ),
    "mso_studio": ModuleHelp(
        purpose=(
            "The unified PySide6 integrator -- every capability above, plus "
            "Tools (curve fitting, plot digitizer, PDF mining), in one "
            "window with a shared results/report/log system, instead of 13 "
            "separate programs."
        ),
        inputs=("Whatever the selected module page needs -- see that "
                "module's own help.",),
        outputs=("Whatever the selected module page produces, plus a "
                "unified LaTeX report across a session's runs.",),
        prerequisites=("Runs in its own .venv-gui interpreter, the same one "
                       "the launcher itself runs in.",),
        typical_use=(
            "Sustained work across several capabilities in one sitting -- "
            "open once, work through Prepare -> Analyse -> Quantify without "
            "relaunching a new process per step."
        ),
        tips=("The Workbench page in this launcher tracks the same step "
              "sequence across launches, whether you run each step here or "
              "as its own standalone app.",),
        see_also=("automation", "tools"),
    ),
    "doctor": ModuleHelp(
        purpose=(
            "Checks whether the environment a given task needs -- a Python "
            "virtual environment, an external solver binary, a library "
            "version -- is actually present and working, and tells you "
            "exactly what's missing and how to fix it. Its Recipes tab "
            "routes a plain-language task description to the right command "
            "or app."
        ),
        inputs=("A task name, or a plain-language description of what you "
                "want to do.",),
        outputs=("A preflight pass/fail report with fix instructions.",),
        prerequisites=("None -- this has no dependencies of its own; it's "
                       "what you run to find out what everything *else* "
                       "needs.",),
        typical_use=(
            "A module won't launch, or crashes early, and you're not sure "
            "why -- run Doctor for that task before re-running blind."
        ),
        tips=("The status strip at the bottom of the launcher window opens "
              "this directly -- its boot-time LEDs are the same checks.",),
        see_also=(),
    ),
    "sensitivity": ModuleHelp(
        purpose=(
            "Two backends for the same question -- which reactions actually "
            "control a prediction -- as tabs in one page: brute-force "
            "Arrhenius (A, n, Ea) perturbation with real re-simulation, and "
            "ignition-delay d ln(tau)/d ln(k) for a single condition."
        ),
        inputs=("Cantera mechanism (.yaml)", "target.opt or a T/P/phi "
                "condition, depending on the tab"),
        outputs=("Sensitivity tables and tornado charts (per tab)",),
        prerequisites=("Mechanism must pass Mechanism Health-Check.",),
        typical_use=(
            "Deciding where to focus mechanism refinement or optimization "
            "effort before spending real search time."
        ),
        tips=("Pick the IDT tab for a fast single-condition look; pick "
              "brute-force when you need results across the full target "
              "set.",),
        see_also=("optimization", "uncertainty"),
    ),
    "optimization": ModuleHelp(
        purpose=(
            "PRS-surrogate posterior optimization of Arrhenius A-factors "
            "(genetic algorithm + L-BFGS-B/SLSQP gradient refinement) "
            "against real experimental targets simultaneously, within "
            "uncertainty bounds established beforehand."
        ),
        inputs=("Mechanism (.yaml)", "target.opt / .add files",
                "Uncertainty bounds (XML)"),
        outputs=("Optimized mechanism", "Cost/error history",
                 "Cached PRS coefficients (for Forward UQ)"),
        prerequisites=("Sensitivity Analysis, to know which reactions to "
                       "let move.",
                       "Uncertainty Quantification and File Generator, for "
                       "the bounds those reactions may move within."),
        typical_use=(
            "The core tuning step: search within honest uncertainty bounds "
            "for the rate-constant combination that best matches measured "
            "ignition delays / flame speeds / species profiles at once."
        ),
        tips=("This can run for a long time -- the page stays responsive "
              "and Stop works, so it's safe to start and check back on.",),
        see_also=("sensitivity", "uncertainty", "forward_uq"),
    ),
    "target_workbench": ModuleHelp(
        purpose=(
            "Curates a .target/.add pair before it enters the pipeline: sanity "
            "checks, NTC-aware dummy-point generation for sparse datasets, "
            "coverage-based row pruning, and a gated write to a new versioned "
            "pair. The source .target/.add is never modified."
        ),
        inputs=(".target file", ".add file (optional)"),
        outputs=("Versioned <stem>_v<N>.target + .add",
                 "Curation log (.yaml) recording every drop/add/finding"),
        prerequisites=("The .target/.add pair produced by the Experimental "
                       "Data step this suite's pipelines already require.",),
        typical_use=(
            "Before handing a target set to Optimization or Forward UQ: load "
            "it, run checks, fix or drop what the checks flag, optionally "
            "densify a sparse dataset with dummy points, then write a new "
            "version to point the rest of the pipeline at."
        ),
        tips=("Nothing is written until you click Write -- staged drops and "
              "dummy points live only in this page's session state until then.",),
        see_also=("optimization", "forward_uq"),
    ),
    "optimize_workbench": ModuleHelp(
        purpose=(
            "The optimization workflow's four pages (Sensitivity Analysis, "
            "Target Workbench, Optimization, Forward UQ) in one dedicated "
            "window, for working the tuning loop without the rest of MSO "
            "Studio's modules in the sidebar."
        ),
        inputs=("Mechanism (.yaml)", ".target/.add files", "Uncertainty bounds (XML)"),
        outputs=("Optimized mechanism", "Cost/error history",
                 "Curated target files", "Forward-UQ posterior distributions"),
        prerequisites=("A mechanism and a .target/.add pair to optimize against.",),
        typical_use=(
            "Open this instead of full MSO Studio when the task at hand is "
            "specifically tuning a mechanism: curate targets, check "
            "sensitivity, run the optimization, then propagate the result's "
            "uncertainty forward -- all four steps one window-switch apart."
        ),
        tips=(),
        see_also=("sensitivity", "optimization", "forward_uq", "target_workbench"),
    ),
    "automation": ModuleHelp(
        purpose=(
            "Drives unattended, multi-day comparison/optimization campaigns "
            "across many mechanism variants or target sets, plus fits a "
            "fast Meta-PRS response surface as a cheap solver stand-in when "
            "a campaign needs many evaluations."
        ),
        inputs=("A campaign YAML (task list)", "Already-validated factorial "
                ".tex reports (for Meta-PRS fitting)"),
        outputs=("Task run status", "Meta-PRS fit + trust-gated "
                 "optimization result"),
        prerequisites=("Each individual task should already work as a "
                       "standalone run -- Automation orchestrates, it "
                       "doesn't debug a broken single run.",),
        typical_use=(
            "Turning a one-off comparison you've already validated by hand "
            "into a repeatable, walk-away campaign across many variants."
        ),
        tips=("Run the thread-budget preflight before launching a daemon -- "
              "it catches oversubscription that would otherwise show up as "
              "a slow, silent campaign.",),
        see_also=("mso_compare", "optimization"),
    ),
    "tools": ModuleHelp(
        purpose=(
            "Utilities for getting real numbers out of a source that isn't "
            "already a clean CSV: curve/symbolic fitting, an interactive "
            "plot digitizer, PDF question-answering and table extraction, "
            "Jupyter notebook generation, and native-extension (.so) builds."
        ),
        inputs=("A plot image, a PDF, tabular data, or existing .csv/.pkl "
                "results to notebook-ify.",),
        outputs=("Fitted curve/expression", "Digitized data points",
                 "Extracted tables", "Generated notebook", "Built .so"),
        prerequisites=(),
        typical_use=(
            "A published paper reports a result only as a figure or a "
            "scanned table -- digitize or extract it here instead of "
            "transcribing by hand."
        ),
        tips=("The native-extension build tab is also where you rebuild "
              "parallel_yaml_writer.so / shuffle.so after a toolchain "
              "change -- the same step Doctor's preflight checks for.",),
        see_also=("doctor",),
    ),
    "project_manager": ModuleHelp(
        purpose=(
            "Creates or opens an mso_project.Project (project.yaml + "
            "artifacts.db), browses its full run history and every "
            "registered artifact, adds notes, resumes a prior run's "
            "artifacts into a fresh directory, and builds the "
            "cross-pipeline LaTeX report."
        ),
        inputs=("A project directory (new or existing)",),
        outputs=("Run/artifact index (queryable)", "Notes",
                 "Materialized run artifacts (resume)", "LaTeX report + notebook"),
        prerequisites=("Runs/artifacts only appear here once something has "
                       "registered them into the project -- e.g. via "
                       "mso_project.Project.capture() around a pipeline "
                       "invocation.",),
        typical_use=(
            "Working across a multi-day campaign: open the project once, "
            "check what's run so far and what it produced, add a note "
            "about a decision, and build the report when it's time to "
            "write up results."
        ),
        tips=("The launcher's own Workbench panel shows the same project "
              "read-only for quick browsing without opening MSO Studio -- "
              "this page is where the mutating actions (notes/resume/"
              "report) actually live.",),
        see_also=(),
    ),
}

