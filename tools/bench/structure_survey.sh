#!/bin/bash
#
# structure_survey.sh - measure every source video and say which folder
# it belongs in. Single click:
#
#   ./tools/bench/structure_survey.sh
#   LOG_LEVEL=debug ./tools/bench/structure_survey.sh      # chattier terminal
#   SURVEY_GEOMETRY_SLICES=5 ./tools/bench/structure_survey.sh
#
# Covers resources/uncensored_vids/ AND the staging tree resources/source/.
# Moves nothing. Writes a reviewable mv -n script you read, then run.
#
# PHASES
#   1. structure-scan    whole-file shot-cut scans, CPU only, files in
#                        parallel. Written to the cache betatv's full run
#                        reads, so the real run reuses them.
#   2. geometry slices   detection (no render) on a few short slices
#                        spread through each file. Needs the GPU. This is
#                        the only signal that identifies split-screen.
#                        Asks first; unattended, it runs.
#   3. structure         the per-file table, recommendations, and the move
#                        script, holding back any move that rests on a
#                        slice, would lower the sample rate, or would
#                        override a manual folder on timing alone.
#
# OUTPUT
#   ../output/benchmarks/structure_survey_<timestamp>/
#       structure_survey.log    everything, every level
#       scan/ geometry/ structure/   each phase's own betabench log + results.json
#       moves.sh                the script, when any move passed
#
# Terminal output is filtered by LOG_LEVEL (trace|debug|info|warn|error,
# default info). The log file keeps everything regardless.

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

STAMP="$(date '+%Y%m%d_%H%M%S')"
SURVEY_DIR="$(cd "$REPO_ROOT/.." && pwd)/output/benchmarks/structure_survey_$STAMP"
mkdir -p "$SURVEY_DIR"
export HARNESS_LOG_FILE="$SURVEY_DIR/structure_survey.log"

# shellcheck source=/dev/null
source "$REPO_ROOT/tools/tuning/harness_lib.sh"

: "${SURVEY_GEOMETRY_SLICES:=3}"
BENCH="$REPO_ROOT/tools/bench/betabench.py"

log_info "structure survey: output in $SURVEY_DIR"
log_info "python: $PYTHON_BIN ($PYTHON_BIN_SOURCE)"
if ! python_env_require cv2 2>>"$HARNESS_LOG_FILE"; then
    log_error "the project Python cannot import cv2; see $HARNESS_LOG_FILE"
    exit 1
fi

# --- phase 1 ----------------------------------------------------------------
if ! run_logged "phase 1: whole-file shot-cut scans (CPU)" \
        "$PYTHON_BIN" "$BENCH" structure-scan --yes \
        --out-dir "$SURVEY_DIR/scan" --console-level info; then
    log_error "phase 1 failed; the structure table would rest on whatever scans already exist. Stopping."
    exit 1
fi

# --- phase 2 ----------------------------------------------------------------
run_geometry=1
if [ -t 0 ]; then
    if ! wait_for_confirmation "Phase 2 runs detection (no render) on $SURVEY_GEOMETRY_SLICES slice(s) per file, on the GPU. Without it split-screen cannot be recognised. Run it?"; then
        run_geometry=0
        log_warn "phase 2 skipped at your request; split-screen recommendations will rest on whatever detection caches already exist"
    fi
else
    log_warn "no terminal to ask on, so phase 2 runs (the default)"
fi
if [ "$run_geometry" -eq 1 ] && [ "$SURVEY_GEOMETRY_SLICES" -gt 0 ]; then
    run_logged "phase 2: geometry slices (GPU, detection only)" \
        "$PYTHON_BIN" "$BENCH" structure-scan --yes \
        --geometry-slices "$SURVEY_GEOMETRY_SLICES" \
        --out-dir "$SURVEY_DIR/geometry" --console-level info \
        || log_warn "phase 2 failed; continuing with the geometry that exists"
fi

# --- phase 3 ----------------------------------------------------------------
if ! run_logged "phase 3: structure table and move script" \
        "$PYTHON_BIN" "$BENCH" structure --yes \
        --write-move-script "$SURVEY_DIR/moves.sh" \
        --out-dir "$SURVEY_DIR/structure" --console-level info; then
    log_error "phase 3 failed; see $HARNESS_LOG_FILE"
    exit 1
fi

# Replay the recommendation section to the terminal, at info.
BENCH_LOG="$SURVEY_DIR/structure/betabench.log"
if [ -f "$BENCH_LOG" ]; then
    sed -n '/=== source folder/,/^.*quick-cut boundary/p' "$BENCH_LOG" \
        | sed -E 's/^[0-9-]+ [0-9:,]+ \[[A-Z ]+\] //' \
        | while IFS= read -r line; do log_info "$line"; done
fi

if [ -f "$SURVEY_DIR/moves.sh" ]; then
    log_info "move script: $SURVEY_DIR/moves.sh"
    log_info "read the reason above each move, then run it: bash '$SURVEY_DIR/moves.sh'"
else
    log_info "no move passed the safety rules, so there is no script to run"
fi
log_info "full per-file table: $BENCH_LOG"
