#!/bin/bash
# cleanup_output.sh - free disk under ../output, keeping everything a later decision needs.
#
#   bash tools/maintenance/cleanup_output.sh
#
# Shows what each category would free, then asks before each one:
#   decided_renders   video files of A/B rounds that have a decision
#                     (chosen.json); their record stays
#   stale_renders     video files of undecided rounds untouched for
#                     CLEANUP_STALE_DAYS (14), asked one round at a time;
#                     resume renders whatever is missing
#   aborted_rounds    round folders that never finished ranking (logs only)
#   log_rotation      betasuite.log past CLEANUP_LOG_MAX_MB (50),
#                     compressed and emptied, CLEANUP_KEEP_LOGS (5) kept
#
# Never touched: stats, caches (detection caches are about 25 h of GPU
# work), analysis runs, betabench results, and every round's ranking.json,
# clips.md, feedback.csv, chosen.json and logs.
#
# Unattended (no terminal), it shows the plan and removes nothing.
# Log: ../output/logs/cleanup_output.log, every level; LOG_LEVEL (info)
# filters the terminal only.

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1
case "${1:-}" in
    -h|--help|help)
        awk 'NR == 1 { next } /^#/ { print; next } { exit }' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        exit 0
        ;;
esac

OUTPUT_ROOT="$(cd "$REPO_ROOT/.." && pwd)/output"
mkdir -p "$OUTPUT_ROOT/logs"
export HARNESS_LOG_FILE="${HARNESS_LOG_FILE:-$OUTPUT_ROOT/logs/cleanup_output.log}"
# shellcheck source=/dev/null
source "$REPO_ROOT/tools/tuning/harness_lib.sh"

: "${CLEANUP_STALE_DAYS:=14}"
: "${CLEANUP_LOG_MAX_MB:=50}"
: "${CLEANUP_KEEP_LOGS:=5}"
TOOL="$REPO_ROOT/tools/maintenance/cleanup_output.py"
TOOL_ARGS=( --stale-days "$CLEANUP_STALE_DAYS" --log-max-mb "$CLEANUP_LOG_MAX_MB"
            --keep-logs "$CLEANUP_KEEP_LOGS" )

# py ARGS...: run the tool, each output line to the log function its prefix names.
py() {
    local line level
    "$PYTHON_BIN" "$TOOL" "${TOOL_ARGS[@]}" "$@" 2>>"$HARNESS_LOG_FILE" | while IFS= read -r line; do
        level="${line%% *}"
        case "$level" in
            DEBUG) log_debug "${line#* }" ;;
            WARN)  log_warn "${line#* }" ;;
            INFO)  log_info "${line#* }" ;;
            *)     log_info "$line" ;;
        esac
    done
    return "${PIPESTATUS[0]}"
}

summary() { "$PYTHON_BIN" "$TOOL" "${TOOL_ARGS[@]}" --summary "$1" 2>>"$HARNESS_LOG_FILE"; }

human() { numfmt --to=iec --suffix=B "$1" 2>/dev/null || echo "$1 bytes"; }

log_info "python: $PYTHON_BIN ($PYTHON_BIN_SOURCE)"
log_info "output folder: $OUTPUT_ROOT"
py || { log_error "could not read the output folder; see $HARNESS_LOG_FILE"; exit 1; }

if [ ! -t 0 ]; then
    log_warn "no terminal to ask on, so nothing is removed; run it at a terminal to choose"
    exit 0
fi

ask_category() {
    local category="$1" question="$2" count size
    read -r count size <<< "$(summary "$category")"
    [ "${count:-0}" -gt 0 ] || return 0
    if wait_for_confirmation "$question ($count item(s), $(human "$size"))?"; then
        py --apply "$category"
    else
        log_info "kept: $category"
    fi
}

ask_category decided_renders "Remove the renders of decided A/B rounds, keeping each round's record"
ask_category aborted_rounds "Remove round folders that never finished ranking (logs only)"

# Read into an array first: a loop reading the list on stdin would hand
# each prompt the next list line instead of the terminal.
mapfile -t STALE < <("$PYTHON_BIN" "$TOOL" "${TOOL_ARGS[@]}" --list stale_renders 2>>"$HARNESS_LOG_FILE")
for entry in ${STALE[@]+"${STALE[@]}"}; do
    IFS=$'\t' read -r name size why <<< "$entry"
    [ -n "$name" ] || continue
    if wait_for_confirmation "Remove the renders of $name ($size, $why)? Its record stays and resume re-renders"; then
        py --apply stale_renders --only "$name"
    else
        log_info "kept: $name"
    fi
done

ask_category log_rotation "Compress and empty betasuite.log"
log_info "done; the full record of what was removed is in $HARNESS_LOG_FILE"
