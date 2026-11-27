#!/bin/bash
# upstream_check.sh - find code that still comes from the upstream project.
#
#   bash tools/release/upstream_check.sh [inventory|verify|gate] [TREE]
#
# Asks for a scratch folder outside the repository (or reads
# UPSTREAM_SCRATCH), puts the upstream tree at its pinned commit in
# <scratch>/upstream (a shallow network fetch, or the same commit from this
# repository's history when offline), then runs the chosen check over TREE
# (default: this repository) with tools/release/upstream_check.py:
#   inventory  blame, verbatim lines of 12+ characters and per-unit shingle
#              shares (the default)
#   verify     fails on any upstream line of 40+ characters or run of 3
#              consecutive upstream lines outside
#              tools/release/upstream_allowlist.json
#   gate       inventory without blame, failing on any verbatim line or on
#              a unit whose shingle share reaches UPSTREAM_GATE_THRESHOLD
#              (0.3); for a tree with no history
# Reports go to <scratch>/reports/<check>.md and .json. They hold paths,
# line numbers, unit names, scores and line hashes, never upstream text.
# Log: <state folder>/logs/upstream_check.log, every level; LOG_LEVEL
# (info) filters the terminal only.

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1
case "${1:-}" in
    -h|--help|help)
        awk 'NR == 1 { next } /^#/ { print; next } { exit }' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        exit 0
        ;;
esac
# shellcheck source=/dev/null
source "$REPO_ROOT/tools/lib/python_env.sh"
# shellcheck source=/dev/null
source "$REPO_ROOT/tools/lib/data_root.sh"
LOG_DIR="$(betasuite_path "$REPO_ROOT" output/logs)" || exit 1
mkdir -p "$LOG_DIR"
export HARNESS_LOG_FILE="${HARNESS_LOG_FILE:-$LOG_DIR/upstream_check.log}"
# shellcheck source=/dev/null
source "$REPO_ROOT/tools/tuning/harness_lib.sh"

CHECK="${1:-inventory}"
TREE="${2:-$REPO_ROOT}"
: "${UPSTREAM_GATE_THRESHOLD:=0.3}"
case "$CHECK" in
    inventory|verify|gate) ;;
    *) log_error "unknown check '$CHECK' (inventory, verify or gate)"; exit 2 ;;
esac
if [ ! -d "$TREE" ]; then
    log_error "no such folder: $TREE"
    exit 2
fi
TREE="$(cd "$TREE" && pwd)"

# py ARGS...: run the script, each output line to the log function its prefix names.
py() {
    local line level
    log_trace "run: $PYTHON_BIN tools/release/upstream_check.py $*"
    "$PYTHON_BIN" tools/release/upstream_check.py "$@" 2>>"$HARNESS_LOG_FILE" | while IFS= read -r line; do
        level="${line%% *}"
        case "$level" in
            TRACE) log_trace "${line#* }" ;;
            DEBUG) log_debug "${line#* }" ;;
            WARN)  log_warn "${line#* }" ;;
            ERROR) log_error "${line#* }" ;;
            INFO)  log_info "${line#* }" ;;
            *)     log_info "$line" ;;
        esac
    done
    return "${PIPESTATUS[0]}"
}

SCRATCH="${UPSTREAM_SCRATCH:-}"
if [ -z "$SCRATCH" ]; then
    if [ ! -t 0 ]; then
        log_error "no terminal to ask for the scratch folder; set UPSTREAM_SCRATCH"
        exit 2
    fi
    log_info "AWAITING INPUT: scratch folder for the upstream copy and reports (outside the repository)"
    read -r -p "Scratch folder: " SCRATCH
    log_info "input received: $SCRATCH"
fi
if [ -z "$SCRATCH" ]; then
    log_error "no scratch folder given"
    exit 2
fi
mkdir -p "$SCRATCH" || exit 1
SCRATCH="$(cd "$SCRATCH" && pwd)"
case "$SCRATCH/" in
    "$REPO_ROOT"/*|"$TREE"/*)
        log_error "the scratch folder must be outside the repository and the scanned tree: $SCRATCH"
        exit 2
        ;;
esac

log_info "engine python: $PYTHON_BIN ($PYTHON_BIN_SOURCE)"
log_info "check: $CHECK; tree: $TREE; scratch: $SCRATCH"
py fetch --dest "$SCRATCH/upstream" --local-repo "$REPO_ROOT" || exit 1

ARGS=( "$CHECK" --upstream "$SCRATCH/upstream" --repo "$TREE" --out "$SCRATCH/reports" )
case "$CHECK" in
    verify) ARGS+=( --allowlist "$TREE/tools/release/upstream_allowlist.json" ) ;;
    gate)   ARGS+=( --threshold "$UPSTREAM_GATE_THRESHOLD" ) ;;
esac
py "${ARGS[@]}"
status=$?
if [ "$status" -eq 0 ]; then
    log_info "done: $CHECK passed or finished; reports in $SCRATCH/reports"
else
    log_error "$CHECK finished with exit $status; reports in $SCRATCH/reports"
fi
exit "$status"
