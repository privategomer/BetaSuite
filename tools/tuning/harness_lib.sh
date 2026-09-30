#!/usr/bin/env bash
# harness_lib.sh - shared logging + control-flow helpers for single-click
# harness scripts. Source this near the top of any harness:
#
#   source "$(dirname "$0")/harness_lib.sh"
#
# Standing rule this implements (applies to every harness built from here
# on, any project): single-click to run, with any phase that genuinely
# needs human input built in as an explicit wait/prompt rather than
# assumed; ALL output (including a wrapped command's stdout/stderr) goes
# to a log file; the log supports trace/debug/info/warn/error levels,
# defaulting to info; stdout is filtered by that same level so a long
# unattended run doesn't spam the terminal (or nohup file) while the log
# file itself always keeps everything, at every level, unfiltered.
#
# --- configuration (env vars, all optional) ---
#   HARNESS_LOG_FILE   path to the log file (default: <script-name>.log
#                       next to the harness that sourced this)
#   LOG_LEVEL           trace|debug|info|warn|error (default: info) -
#                       controls stdout only; the log file always gets
#                       everything regardless of this setting
#
# --- logging ---
#   log_trace/log_debug/log_info/log_warn/log_error "message"
#       always appended to HARNESS_LOG_FILE; printed to stdout too only if
#       the level meets LOG_LEVEL's threshold
#
# --- running a subprocess ---
#   run_logged "description" cmd arg1 arg2 ...
#       runs the command, sends ALL of its stdout+stderr into the log file
#       (never the terminal directly, regardless of LOG_LEVEL - a chatty
#       subprocess's raw output isn't leveled, so it always goes to the
#       file only), logs one info-level start line and one info/error-level
#       finish line with elapsed time, and returns the command's real exit
#       status. Use this to wrap every real step of a harness.
#
# --- interactive phases ---
#   wait_for_confirmation "question" [timeout_seconds]
#       blocks on a y/n prompt at the terminal (skips waiting and returns
#       failure if stdin isn't a terminal, e.g. running under nohup - see
#       below). Logs the prompt and the answer either way. Returns 0 for
#       yes, 1 for no/timeout/non-interactive.
#
#       IMPORTANT: a harness that might run via `nohup ... &` (the normal
#       way to background a long harness) has no terminal to prompt on if
#       it reaches a wait_for_confirmation call. Design harnesses so any
#       required input happens either (a) up front, before backgrounding,
#       or (b) is genuinely optional - the harness should have a sane
#       default it proceeds with (logged at warn level) rather than hang
#       forever on a prompt nobody can see.

set -uo pipefail

: "${LOG_LEVEL:=info}"
_HARNESS_LIB_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${HARNESS_LOG_FILE:=$_HARNESS_LIB_SCRIPT_DIR/$(basename "${0%.sh}").log}"
# create the log file the instant this library is sourced, before any real
# work happens - closes the race where `nohup ./harness.sh & ; tail -f
# that.log` runs the tail before the backgrounded script has gotten far
# enough to write its first line, which makes plain `tail -f` exit
# immediately with "No such file or directory" instead of waiting. Still
# use `tail -F` (capital) rather than `-f` in any run instructions - it
# retries if the file isn't there yet instead of giving up once, so this
# is defense in depth, not the only fix.
touch "$HARNESS_LOG_FILE" 2>/dev/null || true

declare -A _LOG_LEVEL_RANK=( [trace]=0 [debug]=1 [info]=2 [warn]=3 [error]=4 )

_log_rank_of() {
    local lvl="${1,,}"
    if [ -n "${_LOG_LEVEL_RANK[$lvl]+x}" ]; then
        echo "${_LOG_LEVEL_RANK[$lvl]}"
    else
        echo "2"  # unknown level name - treat as info rather than fail
    fi
}

_log() {
    local level="$1"; shift
    local msg="$*"
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    local line="[$ts] [${level^^}] $msg"
    echo "$line" >> "$HARNESS_LOG_FILE"
    local msg_rank floor_rank
    msg_rank=$(_log_rank_of "$level")
    floor_rank=$(_log_rank_of "$LOG_LEVEL")
    if [ "$msg_rank" -ge "$floor_rank" ]; then
        echo "$line"
    fi
}

log_trace() { _log trace "$@"; }
log_debug() { _log debug "$@"; }
log_info()  { _log info  "$@"; }
log_warn()  { _log warn  "$@"; }
log_error() { _log error "$@"; }

run_logged() {
    local desc="$1"; shift
    log_info "=== starting: $desc ==="
    log_debug "command: $*"
    {
        echo "----- begin output: $desc -----"
    } >> "$HARNESS_LOG_FILE"
    local start_ts
    start_ts=$(date +%s)
    "$@" >> "$HARNESS_LOG_FILE" 2>&1
    local status=$?
    local elapsed=$(( $(date +%s) - start_ts ))
    echo "----- end output: $desc (exit $status) -----" >> "$HARNESS_LOG_FILE"
    if [ $status -eq 0 ]; then
        log_info "=== finished: $desc (${elapsed}s) ==="
    else
        log_error "=== failed: $desc (exit $status, ${elapsed}s) - see $HARNESS_LOG_FILE for full output ==="
    fi
    return $status
}

wait_for_confirmation() {
    local prompt="$1"
    local timeout_seconds="${2:-}"
    if [ ! -t 0 ]; then
        log_warn "AWAITING INPUT skipped: '$prompt' - stdin isn't a terminal (running unattended/backgrounded), proceeding with the no-confirmation default"
        return 1
    fi
    log_info "AWAITING INPUT: $prompt (y/n)"
    local reply
    if [ -n "$timeout_seconds" ]; then
        if ! read -r -t "$timeout_seconds" -p "$prompt (y/n) " reply; then
            log_warn "input timed out after ${timeout_seconds}s, proceeding with the no-confirmation default"
            return 1
        fi
    else
        read -r -p "$prompt (y/n) " reply
    fi
    log_info "input received: $reply"
    case "$reply" in
        [yY]|[yY][eE][sS]) return 0 ;;
        *) return 1 ;;
    esac
}
