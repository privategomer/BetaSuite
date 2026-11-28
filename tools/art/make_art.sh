#!/bin/bash
# make_art.sh - redraw the figure art: SVGs, PNGs, manifest, contact sheets.
#
#   bash tools/art/make_art.sh
#
# 1. make_art.py writes assets/art/<style>/figures/*.svg and style.json
#    (stdlib only).
# 2. rasterize.py renders every figure and tile PNG and manifest.json with
#    the pinned rasterizer in tools/art/.venv. If that venv is missing, it
#    asks before creating it from tools/art/requirements-art.txt. It needs
#    the system cairo library.
# 3. contact_sheet.py draws one sheet per style into the log folder's
#    art_sheets/ (engine Python, for the censor looks).
# Commit the changed assets afterwards; the test suite checks that the SVGs
# match the generator and that every PNG matches its SVG.
# Log: <state folder>/logs/make_art.log, every level; LOG_LEVEL (info)
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
# shellcheck source=/dev/null
source "$REPO_ROOT/tools/lib/python_env.sh"
# shellcheck source=/dev/null
source "$REPO_ROOT/tools/lib/data_root.sh"
LOG_DIR="$(betasuite_path "$REPO_ROOT" output/logs)" || exit 1
mkdir -p "$LOG_DIR"
export HARNESS_LOG_FILE="${HARNESS_LOG_FILE:-$LOG_DIR/make_art.log}"
# shellcheck source=/dev/null
source "$REPO_ROOT/tools/tuning/harness_lib.sh"

ART_VENV="$REPO_ROOT/tools/art/.venv"
SHEETS="$LOG_DIR/art_sheets"

# py PYTHON SCRIPT ARGS...: run a tool, each output line to the log function its prefix names.
py() {
    local line level
    "$@" 2>>"$HARNESS_LOG_FILE" | while IFS= read -r line; do
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

log_info "engine python: $PYTHON_BIN ($PYTHON_BIN_SOURCE)"
py python3 tools/art/make_art.py || { log_error "make_art.py failed; see $HARNESS_LOG_FILE"; exit 1; }

if [ ! -x "$ART_VENV/bin/python" ]; then
    if wait_for_confirmation "Create the rasterizer venv in tools/art/.venv from requirements-art.txt"; then
        run_logged "create venv" python3 -m venv "$ART_VENV" || exit 1
        run_logged "install rasterizer" "$ART_VENV/bin/python" -m pip install -r tools/art/requirements-art.txt || exit 1
    else
        log_warn "no rasterizer venv, so the PNGs and manifest were not redrawn"
        exit 1
    fi
fi
py "$ART_VENV/bin/python" tools/art/rasterize.py || { log_error "rasterize.py failed; see $HARNESS_LOG_FILE"; exit 1; }
py "$PYTHON_BIN" tools/art/contact_sheet.py --out "$SHEETS" || { log_error "contact_sheet.py failed; see $HARNESS_LOG_FILE"; exit 1; }
log_info "done; contact sheets in $SHEETS"
