#!/bin/bash
#
# ab_compare.sh - the config as it is against one change, on your own
# footage, judged by eye. One command runs the whole round:
#
#   bash tools/tuning/ab_compare.sh                  # asks what to compare, then runs it
#   AB_EXPERIMENT=min_track_hits bash tools/tuning/ab_compare.sh
#   AB_EXPERIMENT=suppression AB_FOCUS=exposed_vulva bash tools/tuning/ab_compare.sh
#   AB_EXPERIMENT=tracking AB_SET="exposed_breast.match_distance_multiplier=0.75" bash tools/tuning/ab_compare.sh
#
# Pieces of a round, if you want one on its own (DIR: newest run by default):
#   bash tools/tuning/ab_compare.sh explain [DIR]    # rewrite clips.md + feedback.csv
#   bash tools/tuning/ab_compare.sh decide [DIR]     # verdicts -> chosen.json + config lines
#   bash tools/tuning/ab_compare.sh confirm [DIR]    # DIR's chosen side, on videos not yet judged
#
# WHAT TO COMPARE (AB_EXPERIMENT; the menu lists them with a line each)
#   suppression      class_suppression rules: the bench's retune/drop proposal
#   min_track_hits   auto_tune's value against the config, per label
#   match_distance   auto_tune's tightening candidate against the config
#   cut_bursts       merging bursts of shot cuts (fades, flashes) before tracking
#   tracking         any per-label setting, named in AB_SET
#
# A ROUND (at a terminal)
#   1  rank      replays every cached video both ways, ranks them by the
#                seconds of censoring that differ (CPU, minutes)
#   1b explain   clips.md: every difference in each clip, when, which
#                side censors it, and what caused it. feedback.csv: one
#                row per difference with a verdict column to fill in
#   2  clips     the top AB_VIDEOS videos, AB_CLIPS windows each:
#                  *_ab.mp4    the real censor, config LEFT, change RIGHT
#                  *_diff.mp4  outlines: grey both, orange LEFT only,
#                              magenta RIGHT only, cause under the label
#                asks first
#   3  full      both sides over each chosen video whole (hours), asks
#   4  judge     waits while you watch the clips and put right / left /
#                same in feedback.csv, then decides: chosen.json and the
#                config lines to paste
#   5  confirm   asks whether to check the chosen side on videos you
#                have not judged yet, then runs that round the same way
# Unattended, it runs phases 1 to 2 (3 with AB_FULL=1) and stops; run
# decide and confirm when you are back.
#
# OUTPUT
#   ../output/benchmarks/ab_<experiment>_<timestamp>/
#       ab_compare.log   everything, every level
#       betabench.log    the Python side's own log
#       ranking.json     both sides, every video's numbers, the windows
#       clips.md         what differs in each clip, and why
#       feedback.csv     your verdicts; decide reads them
#       chosen.json      written by decide (chosen_rules.json too, for suppression)
#       *.mp4            the renders
#
# Env vars: AB_EXPERIMENT (menu, or suppression unattended), AB_SET
# (space-separated label.key=value, or name=value for cut_bursts),
# AB_FOCUS (labels to rank by), AB_VIDEOS (2), AB_CLIPS (6),
# AB_CLIP_SECONDS (15), AB_PROPOSED (a side as JSON), AB_BACKEND,
# AB_VARIANT, AB_FULL (0), LOG_LEVEL (info; terminal only, the log keeps
# everything). Reads caches only; never touches betaconfig.py.

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

MODE="${1:-run}"
BENCH_ROOT="$(cd "$REPO_ROOT/.." && pwd)/output/benchmarks"
mkdir -p "$BENCH_ROOT"
TOOL="$REPO_ROOT/tools/tuning/ab_compare.py"

# The run with the latest timestamp in its name (ab_* or the older
# suppression_ab_*) that has a ranking.json. The name, not the folder's
# mtime: writing feedback.csv into an older run must not make it newest.
newest_run() {
    local dir stamp
    for dir in "$BENCH_ROOT"/ab_* "$BENCH_ROOT"/suppression_ab_*; do
        [ -f "$dir/ranking.json" ] || continue
        stamp="$(basename "$dir" | grep -oE '[0-9]{8}_[0-9]{6}' | tail -n 1)"
        echo "${stamp:-0} $dir"
    done | sort | tail -n 1 | cut -d' ' -f2-
}

case "$MODE" in
    run)
        START_DIR="$BENCH_ROOT/.ab_start_$(date '+%Y%m%d_%H%M%S')"
        ;;
    explain|decide|confirm)
        START_DIR="${2:-$(newest_run)}"
        if [ -z "$START_DIR" ] || [ ! -f "$START_DIR/ranking.json" ]; then
            echo "no A/B run with a ranking.json found (${START_DIR:-none}); run a round first: bash $0" >&2
            exit 1
        fi
        ;;
    *)
        echo "usage: $0 [explain|decide|confirm [DIR]]" >&2
        exit 2
        ;;
esac
mkdir -p "$START_DIR"
export HARNESS_LOG_FILE="$START_DIR/ab_compare.log"

# shellcheck source=/dev/null
source "$REPO_ROOT/tools/tuning/harness_lib.sh"

: "${AB_EXPERIMENT:=}"
: "${AB_SET:=}"
: "${AB_FOCUS:=}"
: "${AB_VIDEOS:=2}"
: "${AB_CLIPS:=6}"
: "${AB_CLIP_SECONDS:=15}"
: "${AB_PROPOSED:=}"
: "${AB_BACKEND:=}"
: "${AB_VARIANT:=}"
: "${AB_FULL:=0}"

log_info "python: $PYTHON_BIN ($PYTHON_BIN_SOURCE)"
if ! python_env_require cv2 2>>"$HARNESS_LOG_FILE"; then
    log_error "the project Python cannot import cv2; see $HARNESS_LOG_FILE"
    exit 1
fi

strip_stamp() { sed -E 's/^[0-9-]+ [0-9:,]+ \[[A-Z ]+\] //'; }

# Every Python call for run directory $1: its own log, its own folder.
common_args() {
    local dir="$1"
    COMMON=( --out-dir "$dir" --videos "$AB_VIDEOS" --clips "$AB_CLIPS"
             --clip-seconds "$AB_CLIP_SECONDS" --console-level info )
    [ -n "$AB_BACKEND" ] && COMMON+=( --backend "$AB_BACKEND" )
    [ -n "$AB_VARIANT" ] && COMMON+=( --variant "$AB_VARIANT" )
    return 0
}

# Replay the Python log of run $1 from line $2 onwards, lines matching $3, at info.
replay_since() {
    tail -n +"$2" "$1/betabench.log" 2>/dev/null | grep -E "$3" | strip_stamp \
        | while IFS= read -r line; do log_info "$line"; done
}
log_lines() { wc -l < "$1/betabench.log" 2>/dev/null || echo 0; }

# The reason Python gave for stopping, from this harness log's last block.
report_python_stop() {
    tail -n 40 "$HARNESS_LOG_FILE" | grep -E 'SystemExit|nothing to compare|flagged nothing|needs --set|wants|no experiment|no full-run' \
        | tail -n 3 | while IFS= read -r line; do log_error "  $line"; done
}

run_explain() {
    local dir="$1"
    common_args "$dir"
    local from=$(( $(log_lines "$dir") + 1 ))
    if run_logged "explain: what differs in each clip, and what caused it" \
            "$PYTHON_BIN" "$TOOL" --phase explain "${COMMON[@]}"; then
        replay_since "$dir" "$from" '(difference\(s\) to judge|verdicts go in|already exists)'
        return 0
    fi
    log_warn "explain failed; see $HARNESS_LOG_FILE"
    return 1
}

run_decide() {
    local dir="$1"
    common_args "$dir"
    local from=$(( $(log_lines "$dir") + 1 ))
    if ! run_logged "decide: verdicts to the chosen side" "$PYTHON_BIN" "$TOOL" --phase decide "${COMMON[@]}"; then
        log_error "decide failed (is feedback.csv in $dir?); see $HARNESS_LOG_FILE"
        return 1
    fi
    tail -n +"$from" "$dir/betabench.log" | sed -n '/verdicts from/,$p' | strip_stamp \
        | grep -v '^next round' | while IFS= read -r line; do log_info "$line"; done
    return 0
}

# chosen.json holds something other than {}: a change won.
has_adopted() {
    [ -f "$1/chosen.json" ] && [ "$(tr -d ' \n\t' < "$1/chosen.json")" != "{}" ]
}

# run_round DIR EXPERIMENT [extra rank args...]: phases 1 to 5.
run_round() {
    local dir="$1" experiment="$2"; shift 2
    mkdir -p "$dir"
    export HARNESS_LOG_FILE="$dir/ab_compare.log"
    touch "$HARNESS_LOG_FILE"
    log_info "A/B round ($experiment): $dir"
    common_args "$dir"

    # --- phase 1 ------------------------------------------------------------
    local rank=( "$PYTHON_BIN" "$TOOL" --phase rank --experiment "$experiment" "${COMMON[@]}" "$@" )
    if ! run_logged "phase 1: rank every video by how much the two sides differ (CPU)" "${rank[@]}"; then
        log_error "phase 1 stopped:"
        report_python_stop
        return 1
    fi
    if [ -f "$dir/betabench.log" ]; then
        {
            grep -E '\] (experiment:|config as it is:|proposed \(|  change |leaving out)' "$dir/betabench.log"
            sed -n '/\]   video  /,/\]   differ:/p' "$dir/betabench.log"
            grep -E '\] (chosen:|no video renders)' "$dir/betabench.log"
        } | strip_stamp | while IFS= read -r line; do log_info "$line"; done
    fi
    if [ ! -f "$dir/ranking.json" ]; then
        log_error "phase 1 wrote no ranking.json; see $HARNESS_LOG_FILE"
        return 1
    fi
    if grep -q '"chosen": \[\]' "$dir/ranking.json"; then
        log_warn "no video renders differently on the two sides; nothing to judge"
        return 0
    fi

    # --- phase 1b ------------------------------------------------------------
    run_explain "$dir"

    # --- phase 2 -------------------------------------------------------------
    local run_clips=1
    if [ -t 0 ]; then
        if ! wait_for_confirmation "Phase 2 renders $AB_CLIPS clip(s) of ${AB_CLIP_SECONDS}s per chosen video (outlines plus side by side). Run it?"; then
            run_clips=0
            log_warn "phase 2 skipped at your request; ranking.json and clips.md are in $dir"
        fi
    else
        log_warn "no terminal to ask on, so phase 2 runs (the default)"
    fi
    local clips_failed=0
    if [ "$run_clips" -eq 1 ]; then
        if ! run_logged "phase 2: clip renders" "$PYTHON_BIN" "$TOOL" --phase render "${COMMON[@]}"; then
            clips_failed=1
            log_warn "phase 2 failed; see $HARNESS_LOG_FILE"
        fi
    fi

    # --- phase 3 -------------------------------------------------------------
    # Same render path as phase 2, so a phase 2 failure would only repeat
    # itself for hours. AB_FULL=1 still forces it.
    local run_full="$AB_FULL"
    if [ "$clips_failed" -eq 1 ] && [ "$run_full" != "1" ]; then
        log_error "phase 3 skipped: phase 2's renders failed, and phase 3 renders the same way (AB_FULL=1 forces it)"
        run_full=0
    elif [ -t 0 ] && [ "$run_full" != "1" ]; then
        if wait_for_confirmation "Phase 3 renders each chosen video whole on both sides (hours). Run it?"; then
            run_full=1
        fi
    elif [ "$run_full" != "1" ]; then
        log_info "no terminal to ask on, so phase 3 is skipped (AB_FULL=1 runs it unattended)"
    fi
    if [ "$run_full" = "1" ]; then
        run_logged "phase 3: whole-video renders on both sides" \
            "$PYTHON_BIN" "$TOOL" --phase render --no-overlay --no-censored --full "${COMMON[@]}" \
            || log_warn "phase 3 failed; see $HARNESS_LOG_FILE"
    fi
    if ls "$dir"/*.mp4 >/dev/null 2>&1; then
        for clip in "$dir"/*.mp4; do log_info "  $(basename "$clip")"; done
    fi

    judge_and_continue "$dir"
}

# Phases 4 and 5: wait for the verdicts, decide, offer the confirm round.
judge_and_continue() {
    local dir="$1"
    if ! wait_for_enter "Watch the clips in $dir with clips.md open beside them, and put right / left / same in each row of feedback.csv (blank if you cannot tell)."; then
        log_info "stopped before deciding. When the verdicts are in: bash tools/tuning/ab_compare.sh decide $dir"
        return 0
    fi
    run_decide "$dir" || return 1
    if ! has_adopted "$dir"; then
        log_info "nothing won, so there is nothing to confirm; the config stays as it is"
        return 0
    fi
    if wait_for_confirmation "Check the chosen side on videos you have not judged yet (a confirm round, same steps)?"; then
        run_confirm "$dir"
    else
        log_info "later: bash tools/tuning/ab_compare.sh confirm $dir"
        log_info "or paste the config lines above into betaconfig.py if you are already sure"
    fi
}

# run_confirm PREVIOUS_DIR: the chosen side against the config, on fresh videos.
run_confirm() {
    local previous="$1"
    if ! has_adopted "$previous"; then
        log_warn "$previous has no chosen change to confirm (run decide there first, or every change kept the config)"
        return 1
    fi
    local experiment
    experiment="$("$PYTHON_BIN" "$TOOL" --run-experiment "$previous" 2>>"$HARNESS_LOG_FILE")"
    if [ -z "$experiment" ]; then
        log_error "could not read which experiment $previous compared; see $HARNESS_LOG_FILE"
        return 1
    fi
    # The Python side takes the experiment, proposal, focus and the videos
    # to leave out from the previous round itself.
    run_round "$BENCH_ROOT/ab_${experiment}_$(date '+%Y%m%d_%H%M%S')_confirm" "$experiment" \
        --confirm-from "$previous"
}

case "$MODE" in
    explain)
        run_explain "$START_DIR"
        exit $?
        ;;
    decide)
        run_decide "$START_DIR"
        status=$?
        [ "$status" -eq 0 ] && has_adopted "$START_DIR" \
            && log_info "check it on videos you have not judged: bash tools/tuning/ab_compare.sh confirm $START_DIR"
        exit $status
        ;;
    confirm)
        run_confirm "$START_DIR"
        exit $?
        ;;
esac

# --- run: choose the experiment, then one round ---------------------------------
if [ -z "$AB_EXPERIMENT" ]; then
    mapfile -t MENU < <("$PYTHON_BIN" "$TOOL" --list-experiments 2>>"$HARNESS_LOG_FILE")
    if [ "${#MENU[@]}" -eq 0 ]; then
        log_error "could not list the experiments; see $HARNESS_LOG_FILE"
        exit 1
    fi
    choose_option "What should the A/B compare?" 1 "${MENU[@]}"
    AB_EXPERIMENT="${CHOSEN_OPTION%% - *}"
fi
EXTRA=()
[ -n "$AB_PROPOSED" ] && EXTRA+=( --proposed "$AB_PROPOSED" )
for item in $AB_SET; do EXTRA+=( --set "$item" ); done
# shellcheck disable=SC2206
[ -n "$AB_FOCUS" ] && EXTRA+=( --focus $AB_FOCUS )

FIRST_DIR="$BENCH_ROOT/ab_${AB_EXPERIMENT}_$(date '+%Y%m%d_%H%M%S')"
# The start log covered the menu; the round keeps its own in its folder.
mkdir -p "$FIRST_DIR"
cat "$HARNESS_LOG_FILE" >> "$FIRST_DIR/ab_compare.log" 2>/dev/null
rm -rf "$START_DIR"
run_round "$FIRST_DIR" "$AB_EXPERIMENT" ${EXTRA[@]+"${EXTRA[@]}"}
status=$?
log_info "done: $FIRST_DIR"
exit $status
