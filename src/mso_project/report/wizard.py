"""
mso_project.report.wizard
===========================
Interactive CLI prompts that review/edit a project's ``ReportRules`` and
write the result to ``report_rules.yaml`` -- the "asked from the user" half
of this project's report-rules requirement (``rules.py``'s ``DEFAULT_RULES``
is the "coded" half). Styled after ``run_report.py``'s existing wizard
(plain ``input()`` prompts, Enter-to-accept-default), not a new UI paradigm.

Usage
-----
    from mso_project.report.wizard import run_wizard
    rules = run_wizard(project)   # prompts on stdin/stdout, returns + saves ReportRules
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .rules import ReportRules

if TYPE_CHECKING:
    from ..project import Project


def _prompt(question: str, default: str) -> str:
    suffix = f" [{default}]" if default else ""
    ans = input(f"{question}{suffix}: ").strip()
    return ans or default


def _prompt_yes_no(question: str, default: bool) -> bool:
    d = "Y/n" if default else "y/N"
    ans = input(f"{question} ({d}): ").strip().lower()
    if not ans:
        return default
    return ans in ("y", "yes")


def _prompt_list(question: str, default: list) -> list:
    default_str = ",".join(default)
    ans = input(f"{question} [comma-separated, default: {default_str}]: ").strip()
    if not ans:
        return list(default)
    return [x.strip() for x in ans.split(",") if x.strip()]


def run_wizard(project: "Project", rules_path: str = None) -> ReportRules:
    """Run the interactive rules wizard against ``project``. Returns the
    resulting ``ReportRules`` and writes it to
    ``<project.root>/report_rules.yaml`` (or ``rules_path`` if given),
    keeping a ``.bak`` of whatever was there before (see
    ``ReportRules.to_yaml``)."""
    rules_path = rules_path or str(project.root / "report_rules.yaml")
    current = ReportRules.load_or_default(rules_path)

    print("\n== mso_project report rules ==")
    print("Press Enter at any prompt to keep the current/default value.\n")

    current.title = _prompt("Report title", current.title or project.info.get("name", "Project Report"))
    current.author = _prompt("Author", current.author)
    current.sections_main = _prompt_list("Main-report sections, in order", current.sections_main)
    current.sections_supplementary = _prompt_list(
        "Supplementary sections, in order", current.sections_supplementary)

    print("\n-- Role routing overrides --")
    print("Known roles from this project's tracked artifacts:")
    roles = sorted({a["role"] for a in project.list_artifacts()})
    if not roles:
        print("  (none tracked yet -- routing overrides can be added later by re-running "
              "this wizard or hand-editing report_rules.yaml)")
    for role in roles:
        current_route = current.role_routing.get(role, "(kind-based default)")
        ans = _prompt(
            f"  '{role}' -> main/supplementary/exclude/auto",
            current.role_routing.get(role, "auto"))
        if ans in ("main", "supplementary", "exclude"):
            current.role_routing[role] = ans
        elif ans == "auto" and role in current.role_routing:
            del current.role_routing[role]

    current.max_figures_main = int(_prompt(
        "Max figures embedded directly in the main report",
        str(current.max_figures_main)))
    current.holdout_error_flag_pct = float(_prompt(
        "Flag cases whose hold-out error exceeds this percent",
        str(current.holdout_error_flag_pct)))
    current.include_notes = _prompt_yes_no("Include user notes in the report", current.include_notes)
    bib = _prompt("BibTeX file path (blank for none)", current.bib_path or "")
    current.bib_path = bib or None

    current.to_yaml(rules_path)
    print(f"\nSaved: {rules_path}")
    return current
