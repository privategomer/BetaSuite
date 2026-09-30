#!/usr/bin/env bash
#
# run_all_analysis.sh - everything worth knowing about the run that just
# finished, in one command.
#
# This is the single-click harness. By default it covers every model
# configuration that has written detections - each backend, each variant
# - because it asks the cache directory what exists rather than asking
# betaconfig what is selected. A run you did last night cannot go
# unanalysed because config has since moved on.
#
#   ./tools/analysis/run_all_analysis.sh
#   ./tools/analysis/run_all_analysis.sh --labels exposed_vulva exposed_breast
#   ./tools/analysis/run_all_analysis.sh --include-preview
#   ./tools/analysis/run_all_analysis.sh --quick        # skip the slow steps
#   ./tools/analysis/run_all_analysis.sh --log-level trace --console-level warn
#
# NARROWING TO ONE VARIANT
#
#   ./tools/analysis/run_all_analysis.sh --variants 640m
#
# Once a variant is the one you are tuning, re-analysing the others
# costs real time for output you read past. --variants narrows every
# step that reads caches. Backend alone cannot express it: 320n and 640m
# are both nudenet_v3. It only changes what is READ - nothing on disk is
# touched, so widening it again later needs no re-run, and the caches
# for the other variants stay available for a side-by-side whenever you
# want one.
#
# HOW LONG IT TAKES
#
#   ./tools/analysis/run_all_analysis.sh --jobs 6       # more at once
#   ./tools/analysis/run_all_analysis.sh --no-parallel  # one at a time
#
# The middle steps are independent read-only consumers of the same
# caches, so they run several at a time (3 by default, or
# BETASUITE_ANALYSIS_JOBS). Their output is buffered to per-step logs
# and replayed in order at the end, so the console transcript reads
# exactly as it did when this was strictly sequential. summarize_run
# stays first and auto_tune stays last, because those two are ordered
# with respect to everything else. Use --no-parallel when debugging a
# single step and you want its output live.
#
# WHAT IT RUNS, AND WHY IN THIS ORDER
#
#   1. summarize_run.py         Where the wall clock actually went, per
#                               configuration, and what was detected.
#                               FIRST because it sets the ceiling on
#                               everything after it: tuning the stage
#                               that took 16% of a run caps the saving
#                               at 16%, whatever the tuning output says.
#
#   2. betabench.py all         Derives suppression / hysteresis / structure /
#                               geometry / dedup values from the same
#                               caches, and times decode, detect and
#                               render on this hardware. Every variant
#                               of every backend, at its native size.
#
#   3. analyze_suppression_pairs.py   Cross-label confusion signatures.
#   3b. analyze_geometry_impact.py    What a candidate geometry bound would
#                                     do to real detections, per action.
#                                     Its absence is why a bad bound once
#                                     shipped unnoticed.
#   4. analyze_score_distribution.py  Per-label confidence distributions.
#   5. analyze_jitter.py $VARIANT_FLAG              Position-smoothing alpha SWEEP, not
#                                     just the current value - a single
#                                     alpha tells you where you are and
#                                     nothing about where to go.
#   6. analyze_track_breaks.py $VARIANT_FLAG        match_distance / track_max_gap sweeps.
#   7. analyze_style_flicker.py $VARIANT_FLAG       paired_style gate diagnostics.
#
#   8. auto_tune.py             A dry run of the staged tuner: what it
#                               would change and why, writing nothing.
#                               Re-run it with --apply when you agree.
#
# EVERY STEP'S output is teed live AND saved under
# ../output/logs/analysis_runs/<timestamp>/. A step that fails does not
# stop the suite; the summary at the end says which ones did, and the
# exit status is non-zero if any did, so this is safe to schedule.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$CORE_DIR"

if [ -f ../bin/activate ]; then
    # shellcheck disable=SC1091
    source ../bin/activate
fi

LOG_LEVEL="debug"
CONSOLE_LEVEL="info"
QUICK=0
ALPHA_SWEEP="0.15 0.25 0.35 0.50 0.70"
declare -a PASSTHROUGH=()
VARIANTS=""
# How many of the independent analyze_* steps run at once. They are
# read-only consumers of the same detection caches and write to separate
# log files, so they do not contend for anything but CPU. 3 is a floor
# that helps on any machine without oversubscribing a small one; raise
# it with --jobs N, or set --no-parallel to get the old strictly
# sequential behaviour back when debugging a single step.
JOBS="${BETASUITE_ANALYSIS_JOBS:-3}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --log-level)     LOG_LEVEL="$2"; shift 2 ;;
        --console-level) CONSOLE_LEVEL="$2"; shift 2 ;;
        --quick)         QUICK=1; shift ;;
        --variants)      shift; VARIANTS=""
                         while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do
                             VARIANTS="$VARIANTS $1"; shift
                         done ;;
        --jobs)          JOBS="$2"; shift 2 ;;
        --no-parallel)   JOBS=1; shift ;;
        --sweep-alpha)   shift; ALPHA_SWEEP=""
                         while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do
                             ALPHA_SWEEP="$ALPHA_SWEEP $1"; shift
                         done ;;
        --help|-h)       sed -n '2,52p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)               PASSTHROUGH+=("$1"); shift ;;
    esac
done

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$CORE_DIR/../output/logs/analysis_runs/$TS"
mkdir -p "$LOG_DIR"

echo "=== run_all_analysis.sh starting at $(date) ==="
echo "logs for this run: $LOG_DIR"
echo "console level: $CONSOLE_LEVEL   log-file level: $LOG_LEVEL"
if [ "${#PASSTHROUGH[@]}" -gt 0 ]; then
    echo "extra args (passed to every tool that accepts them): ${PASSTHROUGH[*]}"
fi
if [ "$QUICK" -eq 1 ]; then
    echo "--quick: skipping the steps that load models or time rendering"
fi
echo
echo "Configurations this run will cover (read from the cache, not from config):"
python3 - <<'PYEOF'
import sys
sys.path.insert( 0, '.' )
import betautils_cache_paths as bu_cache
configurations = bu_cache.configurations_to_analyse( None )
for config in configurations:
    print( "  %-32s %d video(s) cached"%( config.label, len( config.file_hashes ) ) )
if not configurations:
    print( "  (none - run betatv.py on some footage first)" )
PYEOF
echo

# Built once so every step gets the same filter, or nothing at all when
# no --variants was given.
VARIANT_FLAG=""
if [ -n "$VARIANTS" ]; then
    VARIANT_FLAG="--variants$VARIANTS"
    echo "variant filter: analysing only$VARIANTS"
fi

# Each entry: <label>|<command>. The label is what the summary prints.
declare -a STEPS=()

STEPS+=("summarize_run|python3 $SCRIPT_DIR/summarize_run.py")

if [ "$QUICK" -eq 0 ]; then
    STEPS+=("betabench_all|python3 $CORE_DIR/tools/bench/betabench.py all --yes $VARIANT_FLAG --log-level $LOG_LEVEL --console-level $CONSOLE_LEVEL")
    STEPS+=("betabench_render_noapprox|python3 $CORE_DIR/tools/bench/betabench.py render --yes --no-blur-approximation --log-level $LOG_LEVEL --console-level $CONSOLE_LEVEL")
else
    # structure is cache-only and the profile question depends on it,
    # so --quick keeps it alongside geometry.
    STEPS+=("betabench_cache_only|python3 $CORE_DIR/tools/bench/betabench.py geometry --yes $VARIANT_FLAG --log-level $LOG_LEVEL --console-level $CONSOLE_LEVEL")
    STEPS+=("betabench_structure|python3 $CORE_DIR/tools/bench/betabench.py structure --yes $VARIANT_FLAG --log-level $LOG_LEVEL --console-level $CONSOLE_LEVEL")
fi

STEPS+=("analyze_suppression_pairs|python3 $SCRIPT_DIR/analyze_suppression_pairs.py")
STEPS+=("analyze_geometry_impact|python3 $SCRIPT_DIR/analyze_geometry_impact.py ${PASSTHROUGH[*]:-}")
STEPS+=("analyze_score_distribution|python3 $SCRIPT_DIR/analyze_score_distribution.py ${PASSTHROUGH[*]:-}")
STEPS+=("analyze_jitter|python3 $SCRIPT_DIR/analyze_jitter.py $VARIANT_FLAG --sweep-alpha $ALPHA_SWEEP ${PASSTHROUGH[*]:-}")
STEPS+=("analyze_track_breaks|python3 $SCRIPT_DIR/analyze_track_breaks.py $VARIANT_FLAG ${PASSTHROUGH[*]:-}")
STEPS+=("analyze_style_flicker|python3 $SCRIPT_DIR/analyze_style_flicker.py $VARIANT_FLAG ${PASSTHROUGH[*]:-}")
STEPS+=("auto_tune_dry_run|python3 $CORE_DIR/tools/tuning/auto_tune.py $VARIANT_FLAG --out-dir $LOG_DIR/auto_tune --log-level $LOG_LEVEL --console-level $CONSOLE_LEVEL")

declare -a RESULTS=()
TOTAL=${#STEPS[@]}

# WHY SOME STEPS RUN IN PARALLEL AND SOME DO NOT
#
# summarize_run has to go first - it is the header everything else is
# read against. auto_tune_dry_run has to go last, because it reports the
# config the other steps just described.
#
# Everything between is an independent, READ-ONLY consumer of the same
# detection caches, writing to its own log file. Nothing there mutates
# config or cache, so nothing contends except CPU. Run sequentially they
# were the bulk of the suite's wall clock; the two betabench steps in
# particular sit idle waiting on single-threaded numpy while four other
# scripts have work queued.
#
# Parallel output would interleave unreadably, so each step's output
# goes ONLY to its log file while it runs, and the log is echoed back in
# STEPS order once the batch finishes. The console transcript therefore
# reads exactly as it did sequentially - it just arrives later and all
# at once.
#
# --no-parallel restores strict sequencing for debugging one step.

run_one_step() {
    # $1 label, $2 command. Output to the step's log only.
    local label="$1" command="$2"
    local log_file="$LOG_DIR/${label}.log"
    # shellcheck disable=SC2086
    $command > "$log_file" 2>&1
    echo "$?" > "$LOG_DIR/.${label}.status"
}

replay_step_log() {
    local label="$1" index="$2"
    local log_file="$LOG_DIR/${label}.log"
    local status=1
    [ -f "$LOG_DIR/.${label}.status" ] && status="$(cat "$LOG_DIR/.${label}.status")"
    rm -f "$LOG_DIR/.${label}.status"

    echo "=== [$index/$TOTAL] $label ==="
    [ -f "$log_file" ] && cat "$log_file"

    if [ "$status" -eq 0 ]; then
        RESULTS+=("OK   $label")
    else
        RESULTS+=("FAIL $label (exit $status) - see $log_file")
        echo "WARNING: $label exited with status $status - continuing with the rest of the suite, check $log_file"
    fi
    echo
}

# Split the steps: first and last stay pinned, the middle can batch.
FIRST_LABEL="${STEPS[0]%%|*}"
LAST_INDEX=$(( TOTAL - 1 ))
LAST_LABEL="${STEPS[$LAST_INDEX]%%|*}"

STEP=1
run_one_step "$FIRST_LABEL" "${STEPS[0]#*|}"
replay_step_log "$FIRST_LABEL" "$STEP"

if [ "$JOBS" -gt 1 ] && [ "$TOTAL" -gt 2 ]; then
    echo "running ${JOBS} analysis step(s) at a time; output is replayed in order below"
    echo
fi

running=0
declare -a BATCH=()
for (( i=1; i<LAST_INDEX; i++ )); do
    entry="${STEPS[$i]}"
    label="${entry%%|*}"
    command="${entry#*|}"
    BATCH+=("$label")
    if [ "$JOBS" -gt 1 ]; then
        run_one_step "$label" "$command" &
        running=$(( running + 1 ))
        if [ "$running" -ge "$JOBS" ]; then
            wait -n 2>/dev/null || wait
            running=$(( running - 1 ))
        fi
    else
        run_one_step "$label" "$command"
    fi
done
wait

for label in "${BATCH[@]}"; do
    STEP=$(( STEP + 1 ))
    replay_step_log "$label" "$STEP"
done

STEP=$(( STEP + 1 ))
run_one_step "$LAST_LABEL" "${STEPS[$LAST_INDEX]#*|}"
replay_step_log "$LAST_LABEL" "$STEP"

echo "=== run_all_analysis.sh finished at $(date) ==="
echo "--- summary ---"
for r in "${RESULTS[@]}"; do
    echo "  $r"
done
echo
echo "full logs: $LOG_DIR"
echo
echo "Next: the tuner above ran as a DRY RUN and wrote nothing. When you agree with what"
echo "it proposed, apply it with:"
echo "    python3 tools/tuning/auto_tune.py --apply"
echo "It backs betaconfig.py up before every write and verifies each value took effect."

for r in "${RESULTS[@]}"; do
    if [[ "$r" == FAIL* ]]; then
        exit 1
    fi
done
exit 0
