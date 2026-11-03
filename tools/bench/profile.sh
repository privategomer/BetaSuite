#!/bin/bash
#
# profile.sh - where a tool's time and memory go, as flame graphs. Single
# click, any Python tool in this repo:
#
#   ./tools/bench/profile.sh tools/analysis/analyze_cut_bursts.py
#   ./tools/bench/profile.sh tools/bench/betabench.py cache --yes
#   PROFILE=memory ./tools/bench/profile.sh tools/tuning/auto_tune.py
#   PROFILE=cpu ./tools/bench/profile.sh betatv.py --preview
#   PROFILE_PID=12345 ./tools/bench/profile.sh       # a step already running
#
# PROFILE picks what to measure: both (default), memory or cpu. "both"
# runs the tool twice, once under each profiler, because each one slows
# the other's numbers. PROFILE_PID attaches to a running Python process
# for PROFILE_SECONDS (default 30) instead, CPU only: the way to look
# inside one step of run_all_analysis.sh while it runs (ps shows the
# pid).
#
# THE TOOLS (requirements-dev.txt; the script offers to install them)
#   memray   every allocation, with the stack that made it. Its flame
#            graph opens at the moment of peak memory: the widest boxes
#            are what was holding memory when the process was largest,
#            which is what the kernel's OOM killer weighs. Linux and
#            macOS only.
#   py-spy   samples the stack 100 times a second from outside the
#            process, so the tool runs at nearly full speed and needs no
#            code change. Wide boxes are where the time went.
# The install goes into the project's Python, with uv when that venv has
# no pip (setup makes it with uv, which leaves pip out; a bare `pip` then
# reaches the system Python, which refuses). Unattended, nothing is
# installed. Without them it falls back to the standard library: tracemalloc
# (tools/lib/trace_memory.py: top lines at the peak) and cProfile (top
# functions by cumulative time). Coarser, but they run everywhere.
#
# Analogy: tracing (OpenTelemetry) is a delivery service's tracking
# numbers, built to follow a request across machines. A profiler is an
# X-ray of one process. For "why does this script use 6 GB" you want the
# X-ray.
#
# OUTPUT
#   ../output/benchmarks/profile_<timestamp>_<tool>/
#       profile.log               everything, every level, including the tool's output
#       memory_flamegraph.html    memray, open in a browser
#       memory_stats.txt          memray: peak, top allocators
#       memory_top_lines.txt      tracemalloc fallback
#       cpu_flamegraph.svg        py-spy, open in a browser
#       cpu_top.txt               cProfile fallback
#       meter.json                an unprofiled run's wall time and peak memory
#
# Terminal output is filtered by LOG_LEVEL (trace|debug|info|warn|error,
# default info). The log file keeps everything regardless.

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

: "${PROFILE:=both}"
: "${PROFILE_PID:=}"
: "${PROFILE_SECONDS:=30}"

if [ -n "$PROFILE_PID" ]; then
    TOOL_NAME="pid${PROFILE_PID}"
elif [ "$#" -gt 0 ]; then
    TOOL_NAME="$(basename "${1%.py}")"
else
    awk 'NR == 1 { next } /^#/ { print; next } { exit }' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 2
fi

STAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="$(cd "$REPO_ROOT/.." && pwd)/output/benchmarks/profile_${STAMP}_${TOOL_NAME}"
mkdir -p "$RUN_DIR"
export HARNESS_LOG_FILE="$RUN_DIR/profile.log"

# shellcheck source=/dev/null
source "$REPO_ROOT/tools/tuning/harness_lib.sh"

has_module() { "$PYTHON_BIN" -c "import $1" >/dev/null 2>&1; }
pyspy_bin() {
    local candidate
    candidate="$(dirname "$PYTHON_BIN")/py-spy"
    if [ -x "$candidate" ]; then echo "$candidate"; return 0; fi
    command -v py-spy 2>/dev/null
}

log_info "profiling into $RUN_DIR"
log_debug "python: $PYTHON_BIN ($PYTHON_BIN_SOURCE)"

# Sets INSTALL_ARGV to the command that installs requirements-dev.txt
# into $PYTHON_BIN's environment (its own pip when it has one, else uv's)
# and INSTALL_COMMAND to it as text; both empty when neither exists.
INSTALL_ARGV=()
INSTALL_COMMAND=""
find_dev_install() {
    local uv
    if "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
        INSTALL_ARGV=( "$PYTHON_BIN" -m pip install -r requirements-dev.txt )
    else
        uv="$(command -v uv 2>/dev/null || true)"
        [ -z "$uv" ] && [ -x "$HOME/.local/bin/uv" ] && uv="$HOME/.local/bin/uv"
        [ -n "$uv" ] && INSTALL_ARGV=( "$uv" pip install --python "$PYTHON_BIN" -r requirements-dev.txt )
    fi
    INSTALL_COMMAND="${INSTALL_ARGV[*]}"
}

memray_supported() {
    case "$(uname -s)" in Linux|Darwin) return 0 ;; *) return 1 ;; esac
}

PY_SPY="$(pyspy_bin || true)"
missing=""
[ -z "$PY_SPY" ] && missing="$missing py-spy"
memray_supported && ! has_module memray && missing="$missing memray"
if [ -n "$missing" ]; then
    find_dev_install
    if [ -z "$INSTALL_COMMAND" ]; then
        log_warn "not installed:$missing, and this Python has neither pip nor uv to install them with"
    elif wait_for_confirmation "not installed:$missing. Install requirements-dev.txt now ($INSTALL_COMMAND)?"; then
        run_logged "install profilers" "${INSTALL_ARGV[@]}" || log_warn "the install failed; falling back to the standard library profilers"
        PY_SPY="$(pyspy_bin || true)"
    else
        log_info "not installing; to get flame graphs later: $INSTALL_COMMAND"
    fi
fi

if [ -n "$PROFILE_PID" ]; then
    if [ -z "$PY_SPY" ]; then
        log_error "attaching to a running process needs py-spy: ${INSTALL_COMMAND:-install requirements-dev.txt}"
        exit 1
    fi
    log_info "sampling pid $PROFILE_PID for ${PROFILE_SECONDS}s"
    if ! run_logged "py-spy record (pid $PROFILE_PID)" \
            "$PY_SPY" record --pid "$PROFILE_PID" --duration "$PROFILE_SECONDS" --rate 100 \
            --output "$RUN_DIR/cpu_flamegraph.svg"; then
        log_error "py-spy could not attach. On Linux that is usually ptrace being restricted:"
        log_error "  run this as the same user as the step, or once: sudo sysctl kernel.yama.ptrace_scope=0"
        exit 1
    fi
    log_info "CPU flame graph: $RUN_DIR/cpu_flamegraph.svg"
    exit 0
fi

SCRIPT="$1"
shift
if [ ! -f "$SCRIPT" ]; then
    log_error "no such script: $SCRIPT (give a path relative to $REPO_ROOT)"
    exit 1
fi
case "$PROFILE" in
    both|memory|cpu) ;;
    *) log_error "PROFILE must be both, memory or cpu, not '$PROFILE'"; exit 1 ;;
esac
log_info "tool: $SCRIPT $*"

status=0

# 1. An unprofiled run first: the honest wall time and peak memory, so
# the profiled runs' overhead is visible beside them.
log_info "1. plain run (wall time and peak memory)"
if run_logged "plain run" "$PYTHON_BIN" tools/lib/step_meter.py --record "$RUN_DIR/meter.json" -- \
        "$PYTHON_BIN" "$SCRIPT" "$@"; then
    log_info "   $(cat "$RUN_DIR/meter.json")"
else
    status=1
    log_warn "   the tool failed on its own ($(cat "$RUN_DIR/meter.json" 2>/dev/null)); profiling it anyway"
fi

if [ "$PROFILE" = "both" ] || [ "$PROFILE" = "memory" ]; then
    if has_module memray; then
        log_info "2. memory, under memray (expect 1.5 to 3 times slower)"
        if run_logged "memray run" "$PYTHON_BIN" -m memray run --force -o "$RUN_DIR/memory.bin" "$SCRIPT" "$@"; then
            run_logged "memray flamegraph" "$PYTHON_BIN" -m memray flamegraph --force \
                -o "$RUN_DIR/memory_flamegraph.html" "$RUN_DIR/memory.bin" || status=1
            "$PYTHON_BIN" -m memray stats "$RUN_DIR/memory.bin" > "$RUN_DIR/memory_stats.txt" 2>&1 || status=1
            # The capture is large and the two files above hold what it says.
            rm -f "$RUN_DIR/memory.bin"
            log_info "   flame graph at the peak: $RUN_DIR/memory_flamegraph.html"
            log_info "   peak and top allocators: $RUN_DIR/memory_stats.txt"
        else
            status=1
        fi
    else
        log_warn "memray is not installed (or this system cannot run it); using tracemalloc."
        log_warn "  For allocation flame graphs: ${INSTALL_COMMAND:-install requirements-dev.txt}"
        log_info "2. memory, under tracemalloc (slow: an allocation-heavy tool runs tens of times slower)"
        run_logged "tracemalloc run" "$PYTHON_BIN" tools/lib/trace_memory.py \
            --out "$RUN_DIR/memory_top_lines.txt" -- "$SCRIPT" "$@" || status=1
        log_info "   top lines at the peak: $RUN_DIR/memory_top_lines.txt"
    fi
fi

if [ "$PROFILE" = "both" ] || [ "$PROFILE" = "cpu" ]; then
    if [ -n "$PY_SPY" ]; then
        log_info "3. CPU, sampled by py-spy"
        if run_logged "py-spy record" "$PY_SPY" record --rate 100 --output "$RUN_DIR/cpu_flamegraph.svg" -- \
                "$PYTHON_BIN" "$SCRIPT" "$@"; then
            log_info "   flame graph: $RUN_DIR/cpu_flamegraph.svg"
        else
            status=1
            log_warn "   py-spy failed; on Linux that is usually ptrace being restricted, see $HARNESS_LOG_FILE"
        fi
    else
        log_warn "py-spy is not installed; using cProfile."
        log_warn "  For a CPU flame graph: ${INSTALL_COMMAND:-install requirements-dev.txt}"
        log_info "3. CPU, under cProfile"
        if run_logged "cProfile run" "$PYTHON_BIN" -m cProfile -o "$RUN_DIR/cpu.prof" "$SCRIPT" "$@"; then
            "$PYTHON_BIN" -c "import pstats, sys; pstats.Stats( sys.argv[1], stream=sys.stdout ).sort_stats( 'cumulative' ).print_stats( 40 )" \
                "$RUN_DIR/cpu.prof" > "$RUN_DIR/cpu_top.txt" 2>&1 || status=1
            log_info "   top functions by cumulative time: $RUN_DIR/cpu_top.txt"
        else
            status=1
        fi
    fi
fi

log_info "done; everything is in $RUN_DIR"
exit "$status"
