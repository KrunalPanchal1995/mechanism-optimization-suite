"""
mso_project.cli
=================
Command-line entry point: ``python -m mso_project <subcommand> ...``.

Subcommands
-----------
    init <dir> --name NAME [--description DESC]
    register <dir> <path> [--role ROLE] [--run-id ID]
    scan <dir> <path> [--pipeline NAME]
    capture <dir> <pipeline> <label> <working_dir> -- <command...>
    list-runs <dir> [--pipeline NAME]
    list-artifacts <dir> [--run-id ID] [--role ROLE]
    resume <dir> <run_id> <target_dir> [--link]
    note <dir> <title> [--body-file PATH]
    report <dir> [--configure] [--out DIR]

``capture`` wraps an arbitrary subprocess -- everything after a literal
``--`` is the command to run, so ANY tool (instrumented for mso_project or
not) gets captured with zero code changes to that tool::

    python -m mso_project capture /data/PROJECTS/AMMONIA_2026 run_script case1 \\
        /data/runs/case1 -- python3.9 src/MUQ-SAC/run_script.py target.opt
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from typing import List, Optional

from .project import Project
from .report.wizard import run_wizard


def _split_on_dashdash(argv: List[str]):
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mso_project", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd")

    s_init = sub.add_parser("init", help="Create a new project")
    s_init.add_argument("dir")
    s_init.add_argument("--name", required=True)
    s_init.add_argument("--description", default="")

    s_reg = sub.add_parser("register", help="Register one existing file")
    s_reg.add_argument("dir")
    s_reg.add_argument("path")
    s_reg.add_argument("--role", default=None)
    s_reg.add_argument("--run-id", default=None)

    s_scan = sub.add_parser("scan", help="Retro-import an existing directory")
    s_scan.add_argument("dir")
    s_scan.add_argument("path")
    s_scan.add_argument("--pipeline", default=None)

    s_cap = sub.add_parser(
        "capture",
        help="Run a command inside a captured directory-diff (pass the command after --)")
    s_cap.add_argument("dir")
    s_cap.add_argument("pipeline")
    s_cap.add_argument("label")
    s_cap.add_argument("working_dir")

    s_runs = sub.add_parser("list-runs", help="List tracked runs")
    s_runs.add_argument("dir")
    s_runs.add_argument("--pipeline", default=None)

    s_arts = sub.add_parser("list-artifacts", help="List tracked artifacts")
    s_arts.add_argument("dir")
    s_arts.add_argument("--run-id", default=None)
    s_arts.add_argument("--role", default=None)

    s_res = sub.add_parser("resume", help="Materialize a run's artifacts into a directory")
    s_res.add_argument("dir")
    s_res.add_argument("run_id")
    s_res.add_argument("target_dir")
    s_res.add_argument("--link", action="store_true",
                        help="Symlink instead of copy (unsafe for roles rewritten in place)")

    s_note = sub.add_parser("note", help="Add a freeform note to the project")
    s_note.add_argument("dir")
    s_note.add_argument("title")
    s_note.add_argument("--body-file", default=None,
                         help="File containing the note body (else read from stdin)")

    s_rep = sub.add_parser("report", help="Build the LaTeX report + supplementary + notebook")
    s_rep.add_argument("dir")
    s_rep.add_argument("--configure", action="store_true",
                        help="Run the interactive rules wizard before building")
    s_rep.add_argument("--out", default=None)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    own_argv, command = _split_on_dashdash(argv)
    args = _build_parser().parse_args(own_argv)

    if args.cmd is None:
        _build_parser().print_help()
        return 0

    if args.cmd == "init":
        proj = Project.create(args.dir, name=args.name, description=args.description)
        print(f"Created project: {proj.root}")
        proj.close()
        return 0

    if args.cmd == "register":
        proj = Project.open(args.dir)
        aid = proj.register(args.path, role=args.role, run_id=args.run_id)
        print(f"Registered artifact_id={aid}: {args.path}")
        proj.close()
        return 0

    if args.cmd == "scan":
        proj = Project.open(args.dir)
        ids = proj.scan(args.path, pipeline=args.pipeline)
        print(f"Registered {len(ids)} artifact(s) from {args.path}")
        proj.close()
        return 0

    if args.cmd == "capture":
        if not command:
            print("capture requires a command after '--', e.g.:\n"
                  "  mso_project capture <dir> run_script case1 <working_dir> -- "
                  "python3.9 src/MUQ-SAC/run_script.py target.opt", file=sys.stderr)
            return 2
        proj = Project.open(args.dir)
        with proj.capture(args.pipeline, args.label, args.working_dir,
                           backend_command=command) as run_id:
            print(f"Capturing run_id={run_id}")
            print(f"$ {' '.join(command)}")
            proc = subprocess.run(command, cwd=args.working_dir)
        proj.close()
        return proc.returncode

    if args.cmd == "list-runs":
        proj = Project.open(args.dir)
        for run in proj.list_runs(pipeline=args.pipeline):
            print(f"{run['run_id']}\t{run['pipeline']}\t{run['label']}\t"
                  f"{run['status']}\t{run['started_at']}")
        proj.close()
        return 0

    if args.cmd == "list-artifacts":
        proj = Project.open(args.dir)
        for a in proj.list_artifacts(run_id=args.run_id, role=args.role):
            print(f"{a['artifact_id']}\t{a['role']}\t{a['kind']}\t{a['path']}")
        proj.close()
        return 0

    if args.cmd == "resume":
        proj = Project.open(args.dir)
        mode = "link" if args.link else "copy"
        materialized = proj.resume(args.run_id, args.target_dir, mode=mode)
        print(f"Materialized {len(materialized)} artifact(s) into {args.target_dir} (mode={mode})")
        proj.close()
        return 0

    if args.cmd == "note":
        proj = Project.open(args.dir)
        body = open(args.body_file).read() if args.body_file else sys.stdin.read()
        aid = proj.add_note(args.title, body)
        print(f"Added note (artifact_id={aid})")
        proj.close()
        return 0

    if args.cmd == "report":
        proj = Project.open(args.dir)
        rules = run_wizard(proj) if args.configure else None
        from .report import build_report
        result = build_report(proj, rules=rules, out_dir=args.out)
        print(f"Main report:    {result.main_tex}"
              + (f" (compiled: {result.main_pdf})" if result.main_pdf else " (not compiled)"))
        print(f"Supplementary:  {result.supplementary_tex}"
              + (f" (compiled: {result.supplementary_pdf})" if result.supplementary_pdf else " (not compiled)"))
        print(f"Notebook:       {result.notebook_path}")
        print(f"Manifest:       {result.manifest_path}")
        proj.close()
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
