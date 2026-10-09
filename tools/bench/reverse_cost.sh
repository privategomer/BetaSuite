#!/bin/bash
#
# reverse_cost.sh - what reverse mode would cost to render on this machine,
# before it is built (phase 0 of the reverse-mode plan). Single click:
#
#   ./tools/bench/reverse_cost.sh
#   LOG_LEVEL=debug ./tools/bench/reverse_cost.sh         # chattier terminal
#   REVERSE_SECONDS=30 ./tools/bench/reverse_cost.sh      # longer, steadier pipeline pass
#   REVERSE_VIDEO=../resources/uncensored_vids/x.mp4 ./tools/bench/reverse_cost.sh
#
# Reads only. Encodes go to a scratch folder that is removed afterwards.
# Needs no GPU and asks nothing, so it is safe to background with nohup.
#
# WHAT IT MEASURES (betabench reverse)
#   1. each candidate frame style, the window blends and the fade, on
#      real frames from the source with the most pixels per frame
#   2. the real render pipeline (decode, composite, the real chunk encode,
#      the real worker count) with and without a triggered frame layer
#   3. how much of each file's runtime a trigger would hold on, from the
#      whole-file detection caches a real run already wrote
#
# OUTPUT
#   ../output/benchmarks/reverse_cost_<timestamp>/
#       reverse_cost.log    everything, every level, including the bench's output
#       betabench.log       the bench's own log, at trace
#       results.json        every number, machine-readable
#
# Terminal output is filtered by LOG_LEVEL (trace|debug|info|warn|error,
# default info). The log file keeps everything regardless.

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

STAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="$(cd "$REPO_ROOT/.." && pwd)/output/benchmarks/reverse_cost_$STAMP"
mkdir -p "$RUN_DIR"
export HARNESS_LOG_FILE="$RUN_DIR/reverse_cost.log"

# shellcheck source=/dev/null
source "$REPO_ROOT/tools/tuning/harness_lib.sh"

: "${REVERSE_SECONDS:=15}"
: "${REVERSE_VIDEO:=}"
BENCH="$REPO_ROOT/tools/bench/betabench.py"

log_info "reverse mode cost: output in $RUN_DIR"
log_info "python: $PYTHON_BIN ($PYTHON_BIN_SOURCE)"
if ! python_env_require cv2 2>>"$HARNESS_LOG_FILE"; then
    log_error "the project Python cannot import cv2; see $HARNESS_LOG_FILE"
    exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
    log_error "ffmpeg is not on PATH; the pipeline pass encodes through it. See SETUP.md."
    exit 1
fi

BENCH_ARGS=( reverse --yes --seconds "$REVERSE_SECONDS" --out-dir "$RUN_DIR" --console-level info )
if [ -n "$REVERSE_VIDEO" ]; then
    BENCH_ARGS+=( --video "$REVERSE_VIDEO" )
fi

log_info "this takes a few minutes; to watch it: tail -F '$HARNESS_LOG_FILE'"
if ! run_logged "betabench reverse (${REVERSE_SECONDS}s per render worker)" \
        "$PYTHON_BIN" "$BENCH" "${BENCH_ARGS[@]}"; then
    log_error "the bench failed; the reason is near the end of $HARNESS_LOG_FILE"
    exit 1
fi

# Replay the summary to the terminal, at info. The markers are pinned by
# tests/test_reverse_bench.py so renaming one cannot silently empty this.
BENCH_LOG="$RUN_DIR/betabench.log"
if [ -f "$BENCH_LOG" ]; then
    sed -n '/=== reverse mode cost: summary ===/,/=== end of reverse mode summary ===/p' "$BENCH_LOG" \
        | sed -E 's/^[0-9-]+ [0-9:,]+ \[[A-Z ]+\] //' \
        | while IFS= read -r line; do log_info "$line"; done
fi
log_info "every number: $RUN_DIR/results.json"
log_info "full detail:  $BENCH_LOG"
