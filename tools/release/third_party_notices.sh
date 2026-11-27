#!/bin/bash
# third_party_notices.sh - regenerate THIRD_PARTY_NOTICES.md and audit every dependency's license.
#
#   bash tools/release/third_party_notices.sh
#
# Runs tools/release/third_party_notices.py with the engine's Python
# (PYTHON_BIN). It needs cargo, ui/node_modules (it asks before running
# npm ci when they are missing) and the engine's packages installed. A
# requirement not installed in this Python (a Windows-only or CPU-only
# package, say) can be looked up on PyPI's public JSON API: it asks first.
# A license outside the policy in the script, and not in
# tools/release/license_review.json, fails the run; commit the two
# written files afterwards. The test suite checks they match the lockfiles.
# Log: <state folder>/logs/third_party_notices.log, every level; LOG_LEVEL
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
export HARNESS_LOG_FILE="${HARNESS_LOG_FILE:-$LOG_DIR/third_party_notices.log}"
# shellcheck source=/dev/null
source "$REPO_ROOT/tools/tuning/harness_lib.sh"

# py ARGS...: run the script, each output line to the log function its prefix names.
py() {
    local line level
    log_trace "run: python3 tools/release/third_party_notices.py $*"
    python3 tools/release/third_party_notices.py --python "$PYTHON_BIN" "$@" 2>>"$HARNESS_LOG_FILE" | while IFS= read -r line; do
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
if ! command -v cargo >/dev/null 2>&1; then
    log_error "cargo is needed to read the shell's crates"
    exit 1
fi
if [ ! -d ui/node_modules ]; then
    if wait_for_confirmation "ui/node_modules is missing. Run npm ci in ui/ to install the locked packages"; then
        run_logged "npm ci" bash -c "cd ui && npm ci" || exit 1
    else
        log_error "the JavaScript packages' licenses are read from ui/node_modules"
        exit 1
    fi
fi
missing=$("$PYTHON_BIN" - <<'EOF'
import importlib.metadata as md, re, sys
sys.path.insert(0, 'tools/release')
import third_party_notices as t
for name in sorted(t.requirement_names()):
    try:
        md.distribution(name)
    except md.PackageNotFoundError:
        print(name)
EOF
)
args=()
if [ -n "$missing" ]; then
    log_info "not installed in this Python: $(echo "$missing" | tr '\n' ' ')"
    if wait_for_confirmation "Look up these packages' licenses on PyPI (pypi.org, public package data only)"; then
        args+=(--pypi)
    else
        log_error "without them the notices are incomplete; nothing was written"
        exit 1
    fi
fi
if py "${args[@]}"; then
    log_info "done; review and commit THIRD_PARTY_NOTICES.md and assets/notices/third_party.json"
else
    log_error "a license needs review or a source could not be read; see $HARNESS_LOG_FILE"
    exit 1
fi
