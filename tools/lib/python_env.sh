#!/bin/bash
# python_env.sh - resolve the Python this repo should run, once.
#
# Source it near the top of any shell script that calls Python:
#
#     source "$(dirname "$0")/../lib/python_env.sh"
#     "$PYTHON_BIN" tools/analysis/summarize_run.py
#
# WHY THIS EXISTS
# ---------------
# Every script here used to call bare `python3` and rely on the venv
# being activated. That breaks in three ways, and all three are quiet:
#
#   - under `sudo`, PATH is reset, so `python3` is the SYSTEM python. The
#     first import of cv2 then fails with ModuleNotFoundError, which
#     reads like a broken install rather than a lost venv.
#   - from a shell that never activated the venv (a cron job, a fresh
#     terminal, an IDE's run button), same thing.
#   - with a DIFFERENT project's venv active, the script runs against
#     that one's packages. No error at all, just wrong versions.
#
# Resolution order, most specific first:
#   1. $PYTHON_BIN if the caller already set it (explicit wins)
#   2. this repo's own .venv/bin/python
#   3. $VIRTUAL_ENV/bin/python, if some venv is active
#   4. python3 on PATH
#
# This file has NO side effects beyond setting PYTHON_BIN: no logging, no
# traps, no cd, no file creation. It is safe to source from anything,
# including scripts that do not use harness_lib.sh.

# Repo root is two levels up from tools/lib/.
_PYTHON_ENV_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "$_PYTHON_ENV_LIB_DIR/../.." && pwd)"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "$_REPO_ROOT/.venv/bin/python" ]]; then
        PYTHON_BIN="$_REPO_ROOT/.venv/bin/python"
    elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
        PYTHON_BIN="$VIRTUAL_ENV/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi
export PYTHON_BIN

# Where it came from, for a script that wants to log the choice. Set
# alongside PYTHON_BIN rather than recomputed, so the two cannot disagree.
if [[ -z "${PYTHON_BIN_SOURCE:-}" ]]; then
    case "$PYTHON_BIN" in
        "$_REPO_ROOT/.venv/bin/python") PYTHON_BIN_SOURCE="this repo's .venv" ;;
        python3)                        PYTHON_BIN_SOURCE="python3 on PATH (no venv found)" ;;
        *)                              PYTHON_BIN_SOURCE="PYTHON_BIN or VIRTUAL_ENV" ;;
    esac
fi
export PYTHON_BIN_SOURCE

# Confirm the interpreter can actually import what this app needs, and say
# what to do about it rather than letting a scan fail partway in on a bare
# traceback. Call explicitly; sourcing this file does not run it, because
# some callers (the test runner) do not need cv2.
python_env_require() {
    local module="${1:-cv2}"
    if "$PYTHON_BIN" -c "import $module" >/dev/null 2>&1; then
        return 0
    fi
    echo "ERROR: $PYTHON_BIN cannot import $module." >&2
    echo "       Using: $PYTHON_BIN_SOURCE" >&2
    echo "       If you ran this with sudo, do not: sudo resets PATH and" >&2
    echo "       drops the project venv. Run it as yourself." >&2
    echo "       Otherwise the venv is incomplete; see SETUP.md." >&2
    return 1
}
