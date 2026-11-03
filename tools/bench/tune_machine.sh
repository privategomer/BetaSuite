#!/bin/bash
#
# tune_machine.sh - measure this machine's hardware settings and offer to
# write them to its machine config (../resources/betaconfig_machine.py).
# Single click:
#
#   ./tools/bench/tune_machine.sh
#   LOG_LEVEL=debug ./tools/bench/tune_machine.sh          # chattier terminal
#   TUNE_VIDEO=../resources/uncensored_vids/x.mp4 ./tools/bench/tune_machine.sh
#   TUNE_SKIP="encode scan" ./tools/bench/tune_machine.sh  # leave stages out
#   TUNE_YES=1 ./tools/bench/tune_machine.sh               # write the proposal without asking
#
# Takes roughly 15 to 30 minutes. Close anything heavy first: a betatv run,
# a game or a video call skews every number. It asks before starting if the
# machine looks busy, and asks before writing anything.
#
# WHAT IT MEASURES (tools/bench/tune_machine.py, which has the detail)
#   inventory  CPU, memory, GPU, driver, encoders, decoders, providers
#   detect     detection batch size, with a same-detections check
#   prefetch   frames decoded ahead of the model during detection
#   render     render workers x decoder threads, with memory per worker
#   encode     the current encoder against faster ones, quality checked
#   scan       structure-scan worker processes
#
# Reads only: no cache is read or written, encodes go to a scratch folder
# that is removed. The only file it changes is the machine config, and
# only after you say yes.
#
# OUTPUT
#   ../output/benchmarks/tune_machine_<timestamp>/
#       tune_machine_run.log   everything, every level
#       tune_machine.log       the measurement's own log, at trace
#       results.json           every number
#       proposal.json          proposed settings, with the reason for each
#       proposal.args          the same as NAME=VALUE lines
#
# Terminal output is filtered by LOG_LEVEL (trace|debug|info|warn|error,
# default info). The log file keeps everything regardless.

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

STAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="$(cd "$REPO_ROOT/.." && pwd)/output/benchmarks/tune_machine_$STAMP"
mkdir -p "$RUN_DIR"
export HARNESS_LOG_FILE="$RUN_DIR/tune_machine_run.log"

# shellcheck source=/dev/null
source "$REPO_ROOT/tools/tuning/harness_lib.sh"

: "${TUNE_VIDEO:=}"
: "${TUNE_SKIP:=}"
: "${TUNE_YES:=0}"
TUNE="$REPO_ROOT/tools/bench/tune_machine.py"
MACHINE="$REPO_ROOT/tools/setup/machine_config.py"

log_info "tune_machine: output in $RUN_DIR"
log_info "python: $PYTHON_BIN ($PYTHON_BIN_SOURCE)"
if ! python_env_require cv2 2>>"$HARNESS_LOG_FILE"; then
    log_error "the project Python cannot import cv2; see $HARNESS_LOG_FILE"
    exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
    log_error "ffmpeg is not on PATH; the render and encode stages encode through it. See SETUP.md."
    exit 1
fi

# ---------------------------------------------------------------- is the machine quiet?
BUSY="$("$PYTHON_BIN" "$TUNE" preflight 2>>"$HARNESS_LOG_FILE")"
if [ -n "$BUSY" ]; then
    while IFS= read -r line; do log_warn "$line"; done <<< "$BUSY"
    if ! wait_for_confirmation "The machine looks busy, which skews the numbers. Measure anyway?"; then
        log_info "stopped before measuring; nothing was changed"
        exit 0
    fi
fi

# ---------------------------------------------------------------- measure
TUNE_ARGS=( measure --out-dir "$RUN_DIR" --console-level info )
[ -n "$TUNE_VIDEO" ] && TUNE_ARGS+=( --video "$TUNE_VIDEO" )
# shellcheck disable=SC2206
[ -n "$TUNE_SKIP" ] && TUNE_ARGS+=( --skip $TUNE_SKIP )

log_info "this takes 15 to 30 minutes; to watch it: tail -F '$RUN_DIR/tune_machine.log'"
if ! run_logged "tune_machine measure" "$PYTHON_BIN" "$TUNE" "${TUNE_ARGS[@]}"; then
    log_error "the measurement failed; the reason is near the end of $RUN_DIR/tune_machine.log"
    exit 1
fi

# Replay the summary to the terminal, at info. The markers are pinned by
# tests/test_tune_machine.py so renaming one cannot silently empty this.
TUNE_LOG="$RUN_DIR/tune_machine.log"
if [ -f "$TUNE_LOG" ]; then
    sed -n '/=== tune_machine: summary ===/,/=== end of tune_machine summary ===/p' "$TUNE_LOG" \
        | sed -E 's/^[0-9-]+ [0-9:,]+ \[[A-Z ]+\] //' \
        | while IFS= read -r line; do log_info "$line"; done
fi

# ---------------------------------------------------------------- write?
mapfile -t PROPOSAL < "$RUN_DIR/proposal.args"
if [ "${#PROPOSAL[@]}" -eq 0 ]; then
    log_info "nothing to write; every number: $RUN_DIR/results.json"
    exit 0
fi
if [ "$TUNE_YES" != 1 ] && ! wait_for_confirmation "Write these ${#PROPOSAL[@]} setting(s) to the machine config?"; then
    log_info "not written. To write them later:"
    log_info "  $PYTHON_BIN $MACHINE set ${PROPOSAL[*]}"
    exit 0
fi
if run_logged "write the machine config" "$PYTHON_BIN" "$MACHINE" --log-file "$HARNESS_LOG_FILE" \
        set "${PROPOSAL[@]}" --comment "tune_machine $STAMP"; then
    log_info "written; to see every setting and where it comes from: $PYTHON_BIN $MACHINE show"
else
    log_error "not written: the change failed validation; see $HARNESS_LOG_FILE"
    exit 1
fi
log_info "every number: $RUN_DIR/results.json"
