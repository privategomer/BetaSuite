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
# Pick up a round that stopped (closed terminal, q, unattended run), from
# the step it had not finished; DIR is the newest run by default:
#   bash tools/tuning/ab_compare.sh resume [DIR]
# DIR is a folder path, or just its name as output/benchmarks lists it.
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
#   3  next      asks: judge the clips now, render the whole videos on
#                both sides first (hours), or stop and resume later.
#                Judging comes before the hours of whole-video renders
#   4  judge     waits while you watch the clips and put right / left /
#                same in feedback.csv (and waits again while it has no
#                verdicts), then decides: chosen.json and the config
#                lines to paste
#   5  confirm   asks whether to check the chosen side on videos you
#                have not judged yet, then runs that round the same way
# Unattended, it runs phases 1 to 2 (whole videos too with AB_FULL=1)
# and stops; run resume when you are back. Every finished file is kept
# on resume, and a half-rendered whole video resumes from its chunks.
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
case "$MODE" in
    -h|--help|help)
        awk 'NR == 1 { next } /^#/ { print; next } { exit }' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        exit 0
        ;;
esac
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
    explain|decide|confirm|resume)
        START_DIR="${2:-$(newest_run)}"
        START_DIR="${START_DIR%/}"
        # A bare folder name, as `ls output/benchmarks` shows it, means
        # that folder under output/benchmarks.
        if [ -n "$START_DIR" ] && [ ! -d "$START_DIR" ] && [ -d "$BENCH_ROOT/$START_DIR" ]; then
            START_DIR="$BENCH_ROOT/$START_DIR"
        fi
        if [ -z "$START_DIR" ] || [ ! -f "$START_DIR/ranking.json" ]; then
            echo "no A/B run with a ranking.json found (${START_DIR:-none}); run a round first: bash $0" >&2
            exit 1
        fi
        ;;
    *)
        echo "usage: $0 [resume|explain|decide|confirm [DIR]]   ($0 --help for more)" >&2
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

# read_status DIR: how far DIR's round got, into ST_<key> (see ab_compare.py --status).
# Fails, and says so, when the status cannot be read: a status that
# silently reads as "nothing done" sent a finished round back to "start
# a new one" when ab_compare.py was an older copy without --status.
read_status() {
    ST_ranked=0; ST_experiment=""; ST_videos=0; ST_explained=0; ST_clips=0; ST_full=0
    ST_rows=0; ST_verdicts=0; ST_decided=0; ST_adopted=0
    local output key value
    if ! output="$("$PYTHON_BIN" "$TOOL" --status "$1" 2>>"$HARNESS_LOG_FILE")" \
            || ! grep -q '^ranked=' <<< "$output"; then
        log_error "could not read how far the round in $1 got: ab_compare.py --status failed (see $HARNESS_LOG_FILE). If ab_compare.sh and ab_compare.py are from different versions, update both"
        return 1
    fi
    while IFS='=' read -r key value; do
        case "$key" in
            ranked|experiment|videos|explained|clips|full|rows|verdicts|decided|adopted)
                printf -v "ST_$key" '%s' "$value" ;;
        esac
    done <<< "$output"
}

list_renders() {
    if ls "$1"/*.mp4 >/dev/null 2>&1; then
        for clip in "$1"/*.mp4; do log_info "  $(basename "$clip")"; done
    fi
}

resume_hint() {
    log_info "pick up where this stopped: bash tools/tuning/ab_compare.sh resume $1"
}

# run_round DIR EXPERIMENT [extra rank args...]: rank, then the rest of the round.
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
    continue_round "$dir"
}

# continue_round DIR: every step DIR's round has not finished yet, in
# order. A new round arrives here straight after ranking; `resume` comes
# here for a round that stopped (a closed terminal, a q, an unattended run).
continue_round() {
    local dir="$1" rc
    common_args "$dir"
    read_status "$dir" || return 1
    if [ "$ST_ranked" != "1" ]; then
        log_error "$dir has no ranking.json: phase 1 never finished, so there is nothing to pick up. Start a new round: bash tools/tuning/ab_compare.sh"
        return 1
    fi
    if [ "$ST_videos" = "0" ]; then
        log_warn "no video renders differently on the two sides; nothing to judge"
        return 0
    fi
    if [ "$ST_decided" = "1" ]; then
        log_info "already decided: $dir/chosen.json"
        after_decide "$dir"
        return $?
    fi
    if [ "$ST_explained" != "1" ]; then
        run_explain "$dir" || return 1
    fi
    if [ "$ST_clips" != "1" ]; then
        render_clips "$dir"
        rc=$?
        if [ "$rc" -eq 2 ]; then return 0; fi
        if [ "$rc" -ne 0 ]; then return 1; fi
    fi
    judge_menu "$dir"
}

# render_clips DIR: phase 2, asking first. 0 rendered, 1 failed, 2 declined.
render_clips() {
    local dir="$1"
    if [ -t 0 ]; then
        if ! wait_for_confirmation "Phase 2 renders each chosen video's clip windows (outlines plus side by side; minutes). Run it?"; then
            log_warn "phase 2 skipped at your request; ranking.json and clips.md are in $dir"
            resume_hint "$dir"
            return 2
        fi
    else
        log_warn "no terminal to ask on, so phase 2 runs (the default)"
    fi
    if ! run_logged "phase 2: clip renders" "$PYTHON_BIN" "$TOOL" --phase render --skip-existing "${COMMON[@]}"; then
        log_warn "phase 2 failed; see $HARNESS_LOG_FILE"
        return 1
    fi
    list_renders "$dir"
    return 0
}

# render_full DIR: phase 3, the whole videos on both sides (hours). A
# whole video an earlier attempt finished is kept, and a half-finished
# one resumes from its finished chunks.
render_full() {
    local dir="$1"
    if ! run_logged "phase 3: whole-video renders on both sides (hours)" \
            "$PYTHON_BIN" "$TOOL" --phase render --no-overlay --no-censored --full --skip-existing "${COMMON[@]}"; then
        log_warn "phase 3 failed; see $HARNESS_LOG_FILE"
        return 1
    fi
    list_renders "$dir"
    return 0
}

# judge_menu DIR: the clips are ready. Judge now, render the whole videos
# first, or stop; judging comes before the hours of whole-video renders,
# so a closed terminal during them costs no verdicts.
judge_menu() {
    local dir="$1"
    read_status "$dir" || return 1
    if [ ! -t 0 ]; then
        if [ "$AB_FULL" = "1" ] && [ "$ST_full" != "1" ]; then
            render_full "$dir"
        fi
        log_info "the clips are ready; put verdicts in $dir/feedback.csv"
        resume_hint "$dir"
        return 0
    fi
    local options=( "judge the clips now" )
    if [ "$ST_full" != "1" ]; then
        options+=( "render the whole videos first (hours), then judge" )
    fi
    options+=( "stop here and pick up later" )
    choose_option "The clips are ready. What next?" 1 "${options[@]}"
    case "$CHOSEN_OPTION" in
        render*)
            render_full "$dir" || return 1
            ;;
        stop*)
            resume_hint "$dir"
            return 0
            ;;
    esac
    judge_and_decide "$dir"
}

# judge_and_decide DIR: wait for verdicts (again while there are none), decide.
judge_and_decide() {
    local dir="$1"
    while true; do
        if ! wait_for_enter "Watch the clips in $dir with clips.md open beside them, and put right / left / same in each row of feedback.csv (blank if you cannot tell)."; then
            log_info "stopped before deciding"
            resume_hint "$dir"
            return 0
        fi
        read_status "$dir" || return 1
        if [ "$ST_verdicts" != "0" ]; then
            break
        fi
        log_warn "feedback.csv has no verdicts yet (0 of $ST_rows rows); save it, then press Enter again"
    done
    log_info "$ST_verdicts of $ST_rows row(s) have a verdict"
    run_decide "$dir" || return 1
    after_decide "$dir"
}

# after_decide DIR: offer the confirm round when a change won.
after_decide() {
    local dir="$1"
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
    resume)
        log_info "resuming the round in $START_DIR"
        continue_round "$START_DIR"
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
