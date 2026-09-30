#!/usr/bin/env bash
# track_break_investigation.sh - single-click investigation of the track
# continuity breaks (resets) analyze_jitter.py flagged: why they happen,
# and what loosening match-distance/max_gap would actually do about them.
#
# Read-only against your existing detection caches - no GPU work, nothing
# here reruns betatv.py or touches betaconfig.py. Runs straight through
# with no input needed (root-causing and replaying against a cache is
# purely computational), safe to background with nohup.
#
#   1. analyze_jitter.py (current alpha) - baseline jitter/reset numbers,
#      for a before/after reference point against the deeper dig below
#   2. analyze_track_breaks.py - root-causes every reset event (distance-
#      blocked / gap-blocked / both / genuinely fresh) and sweeps match-
#      distance and max_gap multipliers independently against your real
#      cached footage to show how many resets each would actually prevent
#
# All output (including analyze_jitter.py's/analyze_track_breaks.py's own
# stdout) goes to the log file below; only this script's own log_* lines
# print to the terminal, filtered by LOG_LEVEL (default info).
#
# Run it:
#   chmod +x track_break_investigation.sh harness_lib.sh
#   nohup ./track_break_investigation.sh > /dev/null 2>&1 &
#   tail -F ../../../output/logs/track_break_investigation.log
#
# (use tail -F, capital F, not -f - it retries if the log file doesn't
# exist yet instead of exiting immediately, in case you run the tail
# before the backgrounded script has written its first line)
#
# Override defaults: LOG_LEVEL=debug ./track_break_investigation.sh (see
# harness_lib.sh for what each level shows)

# lives in tools/tuning/ - pin CWD to the core dir (two levels up) so the
# log path and analyze_*.py invocations below behave exactly as they did
# when this script sat directly in the core dir.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
ANALYSIS_DIR="$CORE_DIR/tools/analysis"
cd "$CORE_DIR"

mkdir -p ../output/logs
export HARNESS_LOG_FILE="../output/logs/track_break_investigation.log"
source "$SCRIPT_DIR/harness_lib.sh"

if [ -f ../bin/activate ]; then
    source ../bin/activate
fi

log_info "=== track_break_investigation.sh starting ==="
log_info "log level: $LOG_LEVEL - full detail always in $HARNESS_LOG_FILE regardless of this setting"

run_logged "analyze_jitter.py (baseline reference)" python3 "$ANALYSIS_DIR/analyze_jitter.py"
JITTER_STATUS=$?
if [ $JITTER_STATUS -ne 0 ]; then
    log_warn "analyze_jitter.py exited with status $JITTER_STATUS - continuing anyway, analyze_track_breaks.py doesn't depend on it"
fi

run_logged "analyze_track_breaks.py (root-cause + multiplier sweep)" python3 "$ANALYSIS_DIR/analyze_track_breaks.py"
BREAKS_STATUS=$?
if [ $BREAKS_STATUS -ne 0 ]; then
    log_error "analyze_track_breaks.py exited with status $BREAKS_STATUS - check $HARNESS_LOG_FILE for the real error before trusting any earlier steps' output"
fi

log_info "=== track_break_investigation.sh finished ==="
log_info "full results (including analyze_track_breaks.py's per-multiplier tables) are in $HARNESS_LOG_FILE"
