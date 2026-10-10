#!/bin/bash
#
# suppression_ab.sh - the current class_suppression rules against a
# proposed set, on the videos where they differ most. Single click:
#
#   ./tools/tuning/suppression_ab.sh
#   AB_VIDEOS=3 AB_CLIPS=8 ./tools/tuning/suppression_ab.sh
#   AB_PROPOSED=my_rules.json ./tools/tuning/suppression_ab.sh
#
# The proposed set defaults to what `betabench suppression` suggests from
# the same caches. Reads caches only; never touches betaconfig.py.
#
# PHASES
#   1. rank      replays every cached video under both rule sets and
#                ranks them by box-seconds that differ. CPU, minutes.
#   2. clips     for the top AB_VIDEOS videos, AB_CLIPS windows of
#                AB_CLIP_SECONDS each where the sets differ most:
#                  *_diff.mp4  outlines, grey both / orange current only /
#                              magenta proposed only
#                  *_ab.mp4    real censor, current LEFT, proposed RIGHT
#                Asks first; unattended, it runs.
#   3. full      both rule sets over each chosen video whole (hours).
#                Asks first; unattended, it runs only with AB_FULL=1.
#
# OUTPUT
#   ../output/benchmarks/suppression_ab_<timestamp>/
#       suppression_ab.log   everything, every level
#       betabench.log        the Python side's own log
#       ranking.json         both rule sets, every video's numbers, windows
#       *.mp4                the renders
#
# Env vars: AB_VIDEOS (2), AB_CLIPS (6), AB_CLIP_SECONDS (15),
# AB_PROPOSED (unset: bench suggestions), AB_BACKEND, AB_VARIANT,
# AB_FULL (0), LOG_LEVEL (info; terminal only, the log keeps everything).

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

STAMP="$(date '+%Y%m%d_%H%M%S')"
AB_DIR="$(cd "$REPO_ROOT/.." && pwd)/output/benchmarks/suppression_ab_$STAMP"
mkdir -p "$AB_DIR"
export HARNESS_LOG_FILE="$AB_DIR/suppression_ab.log"

# shellcheck source=/dev/null
source "$REPO_ROOT/tools/tuning/harness_lib.sh"

: "${AB_VIDEOS:=2}"
: "${AB_CLIPS:=6}"
: "${AB_CLIP_SECONDS:=15}"
: "${AB_PROPOSED:=}"
: "${AB_BACKEND:=}"
: "${AB_VARIANT:=}"
: "${AB_FULL:=0}"
TOOL="$REPO_ROOT/tools/tuning/suppression_ab.py"

COMMON=( --out-dir "$AB_DIR" --videos "$AB_VIDEOS" --clips "$AB_CLIPS"
         --clip-seconds "$AB_CLIP_SECONDS" --console-level info )
[ -n "$AB_BACKEND" ] && COMMON+=( --backend "$AB_BACKEND" )
[ -n "$AB_VARIANT" ] && COMMON+=( --variant "$AB_VARIANT" )

log_info "suppression A/B: output in $AB_DIR"
log_info "python: $PYTHON_BIN ($PYTHON_BIN_SOURCE)"
if ! python_env_require cv2 2>>"$HARNESS_LOG_FILE"; then
    log_error "the project Python cannot import cv2; see $HARNESS_LOG_FILE"
    exit 1
fi

# --- phase 1 ----------------------------------------------------------------
RANK=( "$PYTHON_BIN" "$TOOL" --phase rank "${COMMON[@]}" )
[ -n "$AB_PROPOSED" ] && RANK+=( --proposed "$AB_PROPOSED" )
if ! run_logged "phase 1: rank every video by how much the two rule sets differ (CPU)" "${RANK[@]}"; then
    log_error "phase 1 failed; see $HARNESS_LOG_FILE"
    exit 1
fi

# Replay the rule sets, the ranking table and the chosen windows, at info.
PY_LOG="$AB_DIR/betabench.log"
strip_stamp() { sed -E 's/^[0-9-]+ [0-9:,]+ \[[A-Z ]+\] //'; }
if [ -f "$PY_LOG" ]; then
    {
        grep -E '\] (current rules:|proposed rules|  rule )' "$PY_LOG"
        sed -n '/\]   video  /,/\]   differ:/p' "$PY_LOG"
        grep -E '\] (chosen:|no video renders)' "$PY_LOG"
    } | strip_stamp | while IFS= read -r line; do log_info "$line"; done
fi
if [ ! -f "$AB_DIR/ranking.json" ]; then
    log_error "phase 1 wrote no ranking.json; see $HARNESS_LOG_FILE"
    exit 1
fi
if grep -q '"chosen": \[\]' "$AB_DIR/ranking.json"; then
    log_warn "no video renders differently under the two rule sets; nothing to render"
    exit 0
fi

# --- phase 2 ----------------------------------------------------------------
run_clips=1
if [ -t 0 ]; then
    if ! wait_for_confirmation "Phase 2 renders $AB_CLIPS clip(s) of ${AB_CLIP_SECONDS}s per chosen video, three renders each (outlines, current, proposed). Run it?"; then
        run_clips=0
        log_warn "phase 2 skipped at your request; ranking.json is in $AB_DIR"
    fi
else
    log_warn "no terminal to ask on, so phase 2 runs (the default)"
fi
if [ "$run_clips" -eq 1 ]; then
    run_logged "phase 2: clip renders" \
        "$PYTHON_BIN" "$TOOL" --phase render "${COMMON[@]}" \
        || log_warn "phase 2 failed; see $HARNESS_LOG_FILE"
fi

# --- phase 3 ----------------------------------------------------------------
run_full="$AB_FULL"
if [ -t 0 ] && [ "$run_full" != "1" ]; then
    if wait_for_confirmation "Phase 3 renders each chosen video whole under both rule sets (hours). Run it?"; then
        run_full=1
    fi
elif [ "$run_full" != "1" ]; then
    log_info "no terminal to ask on, so phase 3 is skipped (set AB_FULL=1 to run it unattended)"
fi
if [ "$run_full" = "1" ]; then
    run_logged "phase 3: whole-video renders under both rule sets" \
        "$PYTHON_BIN" "$TOOL" --phase render --no-overlay --no-censored --full "${COMMON[@]}" \
        || log_warn "phase 3 failed; see $HARNESS_LOG_FILE"
fi

ls "$AB_DIR"/*.mp4 >/dev/null 2>&1 && for clip in "$AB_DIR"/*.mp4; do log_info "  $(basename "$clip")"; done
log_info "done: $AB_DIR"
