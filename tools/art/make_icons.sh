#!/bin/bash
# make_icons.sh - redraw the app icon set: SVGs, PNGs, icon.ico, the shell's bundle icons, a legibility sheet.
#
#   bash tools/art/make_icons.sh
#
# 1. icons.py writes assets/icons/<icon>/*.svg and icon.json (stdlib only).
# 2. rasterize_icons.py draws every PNG and icon.ico, writes
#    assets/icons/manifest.json, copies the default icon into
#    ui/src-tauri/icons, and draws the legibility sheet (every icon at every
#    size on light, dark and gray) into the log folder's icon_sheets/. It
#    runs in tools/art/.venv; if that venv is missing, it asks before
#    creating it from tools/art/requirements-art.txt. It needs the system
#    cairo library.
# Commit the changed assets afterwards; the test suite checks that the SVGs
# match the generator and that every PNG and icon file matches the manifest.
# Log: <state folder>/logs/make_icons.log, every level; LOG_LEVEL (info)
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
export HARNESS_LOG_FILE="${HARNESS_LOG_FILE:-$LOG_DIR/make_icons.log}"
# shellcheck source=/dev/null
source "$REPO_ROOT/tools/tuning/harness_lib.sh"

ART_VENV="$REPO_ROOT/tools/art/.venv"
SHEETS="$LOG_DIR/icon_sheets"

# py PYTHON SCRIPT ARGS...: run a tool, each output line to the log function its prefix names.
py() {
    local line level
    log_trace "run: $*"
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

py python3 tools/art/icons.py || { log_error "icons.py failed; see $HARNESS_LOG_FILE"; exit 1; }

if [ ! -x "$ART_VENV/bin/python" ]; then
    if wait_for_confirmation "Create the rasterizer venv in tools/art/.venv from requirements-art.txt"; then
        run_logged "create venv" python3 -m venv "$ART_VENV" || exit 1
        run_logged "install rasterizer" "$ART_VENV/bin/python" -m pip install -r tools/art/requirements-art.txt || exit 1
    else
        log_warn "no rasterizer venv, so the PNGs, icon files and manifest were not redrawn"
        exit 1
    fi
fi
py "$ART_VENV/bin/python" tools/art/rasterize_icons.py --sheet "$SHEETS" || { log_error "rasterize_icons.py failed; see $HARNESS_LOG_FILE"; exit 1; }
log_info "done; legibility sheet in $SHEETS"
