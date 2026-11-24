#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_graph_sa.sh
#
# One-command runner for the sensitivity graph analysis on the Methyl
# Butanoate (MB) 2026 dataset.  Reuses the same target.opt config and the
# sens_parameters.pkl that run_script.py produced.
#
# Usage:
#   ./run_graph_sa.sh                       # defaults to the MB dataset
#   ./run_graph_sa.sh path/to/target.opt    # any other target.opt
#   ./run_graph_sa.sh target.opt --top 30 --no-full-graph   # extra args pass through
#
# Outputs land in outputs/graph_sa/ :
#   sensitive_subgraph.png, full_graph_sensitive.png,
#   sensitive_reactions.csv, sensitive_species.csv
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

DEFAULT_TARGET="$HERE/DATASET/METHYL_BUTANOATE_2026/target.opt"

if [[ $# -ge 1 && "$1" != --* ]]; then
    TARGET="$1"; shift
else
    TARGET="$DEFAULT_TARGET"
fi

echo "Running graph SA analysis on: $TARGET"
"$PYTHON" "$HERE/graph_sa_run.py" "$TARGET" --out "$HERE/outputs/graph_sa" "$@"
