#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_forward_uq.sh
#
# Runs the forward UQ pipeline (forward_uq.py) and, if it fails, retries
# up to MAX_RETRIES times with an exponential back-off pause.
#
# Usage:
#   chmod +x run_forward_uq.sh
#   ./run_forward_uq.sh                        # default options
#   ./run_forward_uq.sh --n_samples 100000     # pass extra args to forward_uq.py
#
# Logs are written to forward_uq_run.log
# ---------------------------------------------------------------------------

set -euo pipefail

CONFIG="target.opt"
LOGFILE="forward_uq_run.log"
MAX_RETRIES=5
INITIAL_WAIT=30   # seconds before first retry
PYTHON="python3"

# Any extra arguments (e.g. --n_samples 100000 --method lhs) are forwarded
EXTRA_ARGS=("$@")

# ---------------------------------------------------------------------------
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"; }

run_pipeline() {
    log "Starting forward_uq.py ..."
    "$PYTHON" forward_uq.py "$CONFIG" "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$LOGFILE"
    return "${PIPESTATUS[0]}"
}

# ---------------------------------------------------------------------------
log "======================================================="
log "  Forward UQ scheduler"
log "  Config     : $CONFIG"
log "  Max retries: $MAX_RETRIES"
log "  Log        : $LOGFILE"
log "======================================================="

attempt=0
wait_sec=$INITIAL_WAIT

while true; do
    attempt=$(( attempt + 1 ))
    log "--- Attempt $attempt / $MAX_RETRIES ---"

    if run_pipeline; then
        log "Pipeline finished successfully on attempt $attempt."
        exit 0
    fi

    exit_code=$?
    log "Pipeline exited with code $exit_code."

    if (( attempt >= MAX_RETRIES )); then
        log "ERROR: Reached maximum retries ($MAX_RETRIES). Giving up."
        exit 1
    fi

    log "Waiting ${wait_sec}s before retry ..."
    sleep "$wait_sec"
    wait_sec=$(( wait_sec * 2 ))   # exponential back-off: 30s, 60s, 120s, ...
done
