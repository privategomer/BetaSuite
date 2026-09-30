#!/usr/bin/env bash
# tune_nudenet_suppression.sh - single-click harness for the nudenet_v3
# class_suppression tuning workflow described in CONFIG_REFERENCE.md's
# "class_suppression is per-backend" section:
#
#   1. analyze_suppression_pairs.py   - real per-pair overlap stats
#      against whatever nudenet_v3 detection cache already exists
#   2. derive_suppression_rules.py    - turns step 1's real output into
#      candidate min_iou rules with reasoning (proposes nothing for pairs
#      with too little/no data, rather than guessing)
#   3. replay_tune.py, once per video found under uncensored_vids, against
#      replay_variants_nudenet_v3.json (or REPLAY_VARIANTS_FILE if set) -
#      compares each variant's surviving_after_suppression/label_counts
#      against baseline, fast (no GPU/detection re-run, reads the cache
#      replay_tune.py needs from step 1's underlying data)
#   4. analyze_jitter.py / analyze_track_breaks.py - best-effort re-check,
#      since suppression changes what survives to tracking (per
#      CONFIG_REFERENCE.md's step 5)
#
# All output (including every wrapped tool's raw stdout/stderr) goes to
# the log file this harness's shared harness_lib.sh creates next to this
# script; only leveled summary lines print live, filtered by LOG_LEVEL
# (default info) - see harness_lib.sh's own header comment for the full
# logging contract.
#
# This harness only READS existing detection caches (steps 1-2, 4) or
# replays against them without re-running the neural net (step 3, via
# replay_tune.py) - it never runs betatv.py/betastare.py itself and
# never clears or writes to a detection cache. If nudenet_v3 has no
# cache yet, run betatv.py/betastare.py with detector_backend['selected']
# set to nudenet_v3 first (see README.md), then run this harness.
#
# Usage (single click - all required input is prompted for up front,
# nothing hangs mid-run waiting on a terminal that might not be there if
# this gets backgrounded with nohup):
#     ./tune_nudenet_suppression.sh
#
# Non-interactively (e.g. under nohup/cron - skips the prompt, requires
# the env var to already be set or the harness exits rather than hanging):
#     PREVIEW_START_SECONDS=450 nohup ./tune_nudenet_suppression.sh > /dev/null 2>&1 &
#     tail -F tune_nudenet_suppression.log
#
# Env vars (all optional except PREVIEW_START_SECONDS, which is prompted
# for interactively if unset and stdin is a terminal):
#   PREVIEW_START_SECONDS   the preview-slice offset your nudenet_v3
#                           detection cache was built with (must match a
#                           real prior betatv.py --preview run exactly,
#                           same requirement replay_tune.py itself has -
#                           see its docstring). No sane default exists,
#                           so this is the one genuinely-required input.
#   PREVIEW_SECONDS         preview_max_seconds the cache was built with
#                           (default: betaconfig.py's own preview_max_seconds)
#   REPLAY_VARIANTS_FILE    variants file for step 3 (default:
#                           replay_variants_nudenet_v3.json, next to this
#                           script)
#   BACKEND                 which registered backend step 3's replay_tune.py
#                           runs against (default: nudenet_v3 - this harness's
#                           whole purpose - regardless of whatever
#                           betaconfig.detector_backend['selected'] currently
#                           is in betaconfig.py on disk). Passed to
#                           replay_tune.py as --backend, which sets
#                           BETASUITE_DETECTOR_BACKEND_OVERRIDE for that one
#                           process only - betaconfig.py itself is never
#                           touched. Set BACKEND=retinanet_v2 to replay
#                           against that backend's caches instead, e.g. to
#                           compare against retinanet_v2's own variants file.
#                           analyze_suppression_pairs.py/analyze_jitter.py/
#                           analyze_track_breaks.py (steps 1 and 4) are
#                           already backend-aware on their own - they report
#                           every backend with real cached data as its own
#                           section regardless of this setting, so BACKEND
#                           only affects step 3.
#   LOG_LEVEL               trace|debug|info|warn|error (default: info) -
#                           see harness_lib.sh
#   HARNESS_LOG_FILE        log file path (default: tune_nudenet_suppression.log
#                           next to this script)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$SCRIPT_DIR/harness_lib.sh"

cd "$CORE_DIR"

if [ -f ../bin/activate ]; then
    # shellcheck disable=SC1091
    source ../bin/activate
    log_debug "activated venv at ../bin/activate"
else
    log_warn "no venv found at ../bin/activate - continuing with whatever python3 is on PATH"
fi

REPLAY_VARIANTS_FILE="${REPLAY_VARIANTS_FILE:-$SCRIPT_DIR/replay_variants_nudenet_v3.json}"
PREVIEW_SECONDS="${PREVIEW_SECONDS:-}"
BACKEND="${BACKEND:-nudenet_v3}"

log_info "=== tune_nudenet_suppression.sh starting ==="
log_info "core dir: $CORE_DIR"
log_info "variants file: $REPLAY_VARIANTS_FILE"
log_info "backend for step 3 (replay_tune.py): $BACKEND (set BACKEND=<name> to override; does not touch betaconfig.py - see this script's header)"

# --- required input: PREVIEW_START_SECONDS, prompted up front (not mid-run) ---
# this is the one genuinely-required, can't-default value replay_tune.py
# needs (see its own --preview-start-seconds docs) - asked for here,
# before any real work starts, so a backgrounded/nohup'd run never hits
# an unattended prompt partway through (see harness_lib.sh's own note on
# this).
if [ -z "${PREVIEW_START_SECONDS:-}" ]; then
    if [ -t 0 ]; then
        log_info "AWAITING INPUT: PREVIEW_START_SECONDS not set in the environment."
        read -r -p "Enter the preview-start-seconds offset your nudenet_v3 detection cache was built with: " PREVIEW_START_SECONDS
        log_info "input received: PREVIEW_START_SECONDS=$PREVIEW_START_SECONDS"
    else
        log_error "PREVIEW_START_SECONDS is not set and stdin isn't a terminal (non-interactive run) - there is no safe default for this value (it must match your real nudenet_v3 cache exactly). Set PREVIEW_START_SECONDS and re-run. Aborting."
        exit 1
    fi
fi
if ! [[ "$PREVIEW_START_SECONDS" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    log_error "PREVIEW_START_SECONDS='$PREVIEW_START_SECONDS' doesn't look like a number. Aborting."
    exit 1
fi
export PREVIEW_START_SECONDS
[ -n "$PREVIEW_SECONDS" ] && export PREVIEW_SECONDS

if [ ! -f "$REPLAY_VARIANTS_FILE" ]; then
    log_error "variants file not found: $REPLAY_VARIANTS_FILE - set REPLAY_VARIANTS_FILE or create it (see this script's header). Aborting."
    exit 1
fi

declare -a STEP_RESULTS=()

# --- 1. analyze_suppression_pairs.py: real overlap stats, all cached backends ---
log_info "=== [1/4] analyze_suppression_pairs.py ==="
SUPPRESSION_STATS_TMP="$(mktemp)"
if run_logged "analyze_suppression_pairs.py" python3 "$SCRIPT_DIR/../analysis/analyze_suppression_pairs.py"; then
    STEP_RESULTS+=("OK   [1/4] analyze_suppression_pairs.py")
else
    STEP_RESULTS+=("FAIL [1/4] analyze_suppression_pairs.py (see log)")
fi
# re-run once more, capturing stdout directly to a temp file for step 2 to
# parse (run_logged sends the real leveled copy to the harness log above;
# this second invocation is cheap - it's a read-only cache scan, no
# detection re-run - and keeps derive_suppression_rules.py's input exactly
# matching what step 1 above already showed, rather than trying to
# re-extract it from the interleaved harness log).
python3 "$SCRIPT_DIR/../analysis/analyze_suppression_pairs.py" > "$SUPPRESSION_STATS_TMP" 2>>"$HARNESS_LOG_FILE"

# --- 2. derive_suppression_rules.py: candidate rules from step 1's real output ---
log_info "=== [2/4] derive_suppression_rules.py ==="
DERIVED_RULES_OUT="$SCRIPT_DIR/derived_suppression_rules_$(date +%Y%m%d_%H%M%S).txt"
if python3 "$SCRIPT_DIR/derive_suppression_rules.py" "$SUPPRESSION_STATS_TMP" > "$DERIVED_RULES_OUT" 2>>"$HARNESS_LOG_FILE"; then
    log_info "derive_suppression_rules.py finished - candidates written to $DERIVED_RULES_OUT"
    STEP_RESULTS+=("OK   [2/4] derive_suppression_rules.py -> $DERIVED_RULES_OUT")
else
    log_error "derive_suppression_rules.py failed - see $HARNESS_LOG_FILE"
    STEP_RESULTS+=("FAIL [2/4] derive_suppression_rules.py (see log)")
fi
rm -f "$SUPPRESSION_STATS_TMP"

# --- 3. replay_tune.py, once per video, against the nudenet_v3 variants file ---
log_info "=== [3/4] replay_tune.py (per video under uncensored_vids, backend: $BACKEND, variants: $(basename "$REPLAY_VARIANTS_FILE")) ==="
VIDEO_LIST=$(python3 -c "
import betaconst, os
for root, d_names, f_names in os.walk(betaconst.video_path_uncensored):
    for fname in f_names:
        print(fname)
")
if [ -z "$VIDEO_LIST" ]; then
    log_warn "no video files found under video_path_uncensored - skipping replay_tune.py (step 3)"
    STEP_RESULTS+=("SKIP [3/4] replay_tune.py (no video files found)")
else
    while IFS= read -r fname; do
        [ -z "$fname" ] && continue
        if run_logged "replay_tune.py --video $fname" python3 "$SCRIPT_DIR/replay_tune.py" --video "$fname" --preview-start-seconds "$PREVIEW_START_SECONDS" --variants "$REPLAY_VARIANTS_FILE" --backend "$BACKEND"; then
            STEP_RESULTS+=("OK   [3/4] replay_tune.py --video $fname")
        else
            STEP_RESULTS+=("FAIL [3/4] replay_tune.py --video $fname (missing cache for this video/offset? see log)")
        fi
    done <<< "$VIDEO_LIST"
fi

# --- 4. re-check jitter/track-breaks now that suppression may be about to change ---
# best-effort, same as tune_harness.sh/run_all_analysis.sh - a tool
# erroring on one moved/renamed source file shouldn't block the rest.
log_info "=== [4/4] analyze_jitter.py / analyze_track_breaks.py (best-effort) ==="
if run_logged "analyze_jitter.py" python3 "$SCRIPT_DIR/../analysis/analyze_jitter.py"; then
    STEP_RESULTS+=("OK   [4/4] analyze_jitter.py")
else
    STEP_RESULTS+=("FAIL [4/4] analyze_jitter.py (see log)")
fi
if run_logged "analyze_track_breaks.py" python3 "$SCRIPT_DIR/../analysis/analyze_track_breaks.py"; then
    STEP_RESULTS+=("OK   [4/4] analyze_track_breaks.py")
else
    STEP_RESULTS+=("FAIL [4/4] analyze_track_breaks.py (see log)")
fi

log_info "=== tune_nudenet_suppression.sh finished ==="
log_info "--- summary ---"
for r in "${STEP_RESULTS[@]}"; do
    log_info "  $r"
done
log_info "derived rule candidates: $DERIVED_RULES_OUT"
log_info "full log: $HARNESS_LOG_FILE"
log_info "next step (manual, per CONFIG_REFERENCE.md): review $DERIVED_RULES_OUT and the replay_tune.py deltas in the log above, then hand-edit $REPLAY_VARIANTS_FILE (or betaconfig.py's detector_backend['nudenet_v3']['class_suppression'] directly) with whatever rules you decide to keep."

for r in "${STEP_RESULTS[@]}"; do
    if [[ "$r" == FAIL* ]]; then
        exit 1
    fi
done
exit 0
