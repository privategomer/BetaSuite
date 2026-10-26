#!/bin/bash
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
#   ./tools/analysis/run_all_analysis.sh --log-level trace --console-level warn
#
# TIMING STEPS ARE OPT-IN
#
#   ./tools/analysis/run_all_analysis.sh --with-encode     # encoder speed/quality (~1 h)
#   ./tools/analysis/run_all_analysis.sh --with-reverse    # reverse-mode render cost (~25 min)
#   ./tools/analysis/run_all_analysis.sh --with-hardware   # decode/detect/render timing
#   ./tools/analysis/run_all_analysis.sh --with-timing     # all three
#
# These measure the machine, not the run: their answer changes when the
# hardware, drivers or encoder settings do, not after every batch of
# footage. A default run reads caches only. --quick is accepted and does
# nothing, since that is now the default.
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
#   2. betabench.py cache       Derives suppression / hysteresis / structure /
#                               geometry / dedup values from the same
#                               caches. Every variant of every backend, at
#                               its native size. Writes moves.sh, the
#                               reviewable source-folder moves the
#                               structure survey recommends.
#   2b. analyze_label_content.py      Which videos show which censored label,
#                                     and which mix them: where to judge a
#                                     per-label setting (AB_FOCUS).
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
#   7b. analyze_cut_bursts.py $VARIANT_FLAG         Whether merging bursts of shot cuts
#                                     (fades, flashes) would recover censoring,
#                                     replayed per profile. Nothing is wired in
#                                     until its numbers say so.
#
#   T. (opt-in, see above) TIMING steps, one at a time after the
#      parallel batch, so no other step's CPU use is measured with them:
#      betabench.py hardware    decode / detect / render timing, plus
#                               render without the blur approximation
#                               (--with-hardware)
#      betabench.py encode      Render encode speed and quality per
#                               encoder on this machine, through the
#                               real chunk command and worker count.
#                               Informs encode_video_codec / preset / crf
#                               (--with-encode).
#      betabench.py reverse     What reverse mode would cost to render,
#                               from this run's caches (--with-reverse).
#
#   8. auto_tune.py             A dry run of the staged tuner: what it
#                               would change and why, writing nothing.
#                               Re-run it with --apply when you agree.
#
# EVERY STEP'S output is teed live AND saved under
# ../output/logs/analysis_runs/<timestamp>/. A step that fails does not
# stop the suite; the summary at the end says which ones did, and the
# exit status is non-zero if any did, so this is safe to schedule.

# Resolve the project venv explicitly rather than trusting PATH.
# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/../lib/python_env.sh"

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
WITH_HARDWARE=0
WITH_ENCODE=0
WITH_REVERSE=0
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
        --with-hardware) WITH_HARDWARE=1; shift ;;
        --with-encode)   WITH_ENCODE=1; shift ;;
        --with-reverse)  WITH_REVERSE=1; shift ;;
        --with-timing)   WITH_HARDWARE=1; WITH_ENCODE=1; WITH_REVERSE=1; shift ;;
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
        --help|-h)       awk 'NR == 1 { next } /^#/ { print; next } { exit }' "${BASH_SOURCE[0]}" \
                             | sed 's/^# \{0,1\}//'; exit 0 ;;
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
    echo "--quick: nothing to skip, timing steps are already opt-in (--with-timing to run them)"
fi
TIMING_ON=""
[ "$WITH_HARDWARE" -eq 1 ] && TIMING_ON="$TIMING_ON hardware"
[ "$WITH_ENCODE" -eq 1 ] && TIMING_ON="$TIMING_ON encode"
[ "$WITH_REVERSE" -eq 1 ] && TIMING_ON="$TIMING_ON reverse"
echo "timing steps:${TIMING_ON:- none (--with-encode, --with-reverse, --with-hardware or --with-timing)}"
echo
echo "Configurations this run will cover (read from the cache, not from config):"
"$PYTHON_BIN" - <<'PYEOF'
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

STEPS+=("summarize_run|${PYTHON_BIN} $SCRIPT_DIR/summarize_run.py")

# Every betabench step gets its own --out-dir under this run's folder.
# Without one, betabench names its folder after the current second, and
# two steps launched together in the parallel batch landed in the SAME
# folder: one results.json overwrote the other and both wrote one
# interleaved betabench.log.
STEPS+=("betabench_cache|${PYTHON_BIN} $CORE_DIR/tools/bench/betabench.py cache --yes $VARIANT_FLAG --write-move-script $LOG_DIR/moves.sh --out-dir $LOG_DIR/betabench_cache --log-level $LOG_LEVEL --console-level $CONSOLE_LEVEL")
STEPS+=("analyze_label_content|${PYTHON_BIN} $SCRIPT_DIR/analyze_label_content.py $VARIANT_FLAG --json $LOG_DIR/label_content.json")

STEPS+=("analyze_suppression_pairs|${PYTHON_BIN} $SCRIPT_DIR/analyze_suppression_pairs.py")
STEPS+=("analyze_geometry_impact|${PYTHON_BIN} $SCRIPT_DIR/analyze_geometry_impact.py ${PASSTHROUGH[*]:-}")
STEPS+=("analyze_score_distribution|${PYTHON_BIN} $SCRIPT_DIR/analyze_score_distribution.py ${PASSTHROUGH[*]:-}")
STEPS+=("analyze_jitter|${PYTHON_BIN} $SCRIPT_DIR/analyze_jitter.py $VARIANT_FLAG --sweep-alpha $ALPHA_SWEEP ${PASSTHROUGH[*]:-}")
STEPS+=("analyze_track_breaks|${PYTHON_BIN} $SCRIPT_DIR/analyze_track_breaks.py $VARIANT_FLAG ${PASSTHROUGH[*]:-}")
STEPS+=("analyze_style_flicker|${PYTHON_BIN} $SCRIPT_DIR/analyze_style_flicker.py $VARIANT_FLAG ${PASSTHROUGH[*]:-}")
STEPS+=("analyze_cut_bursts|${PYTHON_BIN} $SCRIPT_DIR/analyze_cut_bursts.py $VARIANT_FLAG --json $LOG_DIR/analyze_cut_bursts.json")
STEPS+=("auto_tune_dry_run|${PYTHON_BIN} $CORE_DIR/tools/tuning/auto_tune.py $VARIANT_FLAG --out-dir $LOG_DIR/auto_tune --log-level $LOG_LEVEL --console-level $CONSOLE_LEVEL")

# Timing steps: run one at a time, after the parallel batch and before
# auto_tune. They measure wall clock and throughput, so anything running
# beside them would be measured with them.
declare -a TIMING_STEPS=()
if [ "$WITH_HARDWARE" -eq 1 ]; then
    TIMING_STEPS+=("betabench_hardware|${PYTHON_BIN} $CORE_DIR/tools/bench/betabench.py hardware --yes $VARIANT_FLAG --out-dir $LOG_DIR/betabench_hardware --log-level $LOG_LEVEL --console-level $CONSOLE_LEVEL")
    TIMING_STEPS+=("betabench_render_noapprox|${PYTHON_BIN} $CORE_DIR/tools/bench/betabench.py render --yes --no-blur-approximation --out-dir $LOG_DIR/betabench_render_noapprox --log-level $LOG_LEVEL --console-level $CONSOLE_LEVEL")
fi
if [ "$WITH_ENCODE" -eq 1 ]; then
    TIMING_STEPS+=("betabench_encode|${PYTHON_BIN} $CORE_DIR/tools/bench/betabench.py encode --yes --out-dir $LOG_DIR/betabench_encode --log-level $LOG_LEVEL --console-level $CONSOLE_LEVEL")
fi
if [ "$WITH_REVERSE" -eq 1 ]; then
    TIMING_STEPS+=("betabench_reverse|${PYTHON_BIN} $CORE_DIR/tools/bench/betabench.py reverse --yes --out-dir $LOG_DIR/betabench_reverse --log-level $LOG_LEVEL --console-level $CONSOLE_LEVEL")
fi

declare -a RESULTS=()
TOTAL=$(( ${#STEPS[@]} + ${#TIMING_STEPS[@]} ))

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
# LAST_INDEX indexes STEPS, so it is counted from STEPS alone: TOTAL also
# counts the timing steps, which are a separate list.
FIRST_LABEL="${STEPS[0]%%|*}"
LAST_INDEX=$(( ${#STEPS[@]} - 1 ))
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

for entry in ${TIMING_STEPS[@]+"${TIMING_STEPS[@]}"}; do
    label="${entry%%|*}"
    run_one_step "$label" "${entry#*|}"
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
echo "--- where the recommendations are ---"
where() {   # where LABEL PATH: print the line only when the file was written
    [ -e "$2" ] && printf "  %-34s %s\n" "$1" "$2"
    return 0
}
where "tuner proposals (auto_tune):" "$LOG_DIR/auto_tune/decisions.json"
where "suppression/hysteresis/geometry:" "$LOG_DIR/betabench_cache/results.json"
where "structure: source-folder moves" "$LOG_DIR/moves.sh"
where "structure: per-file table" "$LOG_DIR/betabench_cache.log"
where "which videos show which label:" "$LOG_DIR/label_content.json"
where "near-duplicate videos:" "$LOG_DIR/analyze_label_content.log"
where "cut-burst merge, best variant:" "$LOG_DIR/analyze_cut_bursts.json"
where "encoder speed and quality:" "$LOG_DIR/betabench_encode/results.json"
where "reverse-mode cost:" "$LOG_DIR/betabench_reverse/results.json"
where "decode/detect/render timing:" "$LOG_DIR/betabench_hardware/results.json"
echo "  every step's full output:           $LOG_DIR/<step>.log"
echo
echo "Next: the tuner above ran as a DRY RUN and wrote nothing. Values it marked 'worth"
echo "judging by eye' go through the A/B first: bash tools/tuning/ab_compare.sh"
echo "When you agree with the rest, apply it with:"
echo "    ${PYTHON_BIN} tools/tuning/auto_tune.py --apply"
echo "It backs betaconfig.py up before every write and verifies each value took effect."

for r in "${RESULTS[@]}"; do
    if [[ "$r" == FAIL* ]]; then
        exit 1
    fi
done
exit 0
