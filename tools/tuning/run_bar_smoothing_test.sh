#!/usr/bin/env bash
# run_bar_smoothing_test.sh - single-click entry point for
# run_bar_smoothing_test.py (see that file's docstring for what it does
# and why it needs to exist separately from replay_tune.py).
#
# Run it with:
#     chmod +x run_bar_smoothing_test.sh && ./run_bar_smoothing_test.sh
#
# Every line of output (including everything from betatv.py itself) goes
# to ../../../output/logs/run_bar_smoothing_test.log regardless of what you see on screen.
# What prints to THIS terminal is filtered by BAR_TEST_LOG_LEVEL
# (trace|debug|info|warn|error, default info) - trace level also echoes
# betatv.py's own console output live, useful the first time to confirm
# real detection is actually progressing rather than stuck:
#
#     BAR_TEST_LOG_LEVEL=trace ./run_bar_smoothing_test.sh
#
# Override the preview window (default: start at 10s, 20s long) if you
# know a better point in this specific video to sample from:
#
#     PREVIEW_START_SECONDS=25 PREVIEW_SECONDS=20 ./run_bar_smoothing_test.sh
#
# This never runs anything concurrently with another betatv.py process -
# the GPU on this box is single-consumer (see betasuite.log's OOM
# history) - so if a real production run is in progress, wait for it to
# finish first (same wait-loop tune_harness.sh uses).

set -uo pipefail

cd "$(dirname "$0")"

if [ -f ../../../bin/activate ]; then
    source ../../../bin/activate
fi

echo "=== checking for an already-running betatv.py process ==="
while pgrep -f "betatv\.py" > /dev/null 2>&1; do
    echo "$(date): betatv.py still running elsewhere - waiting 120s before checking again"
    sleep 120
done
echo "GPU clear, no other betatv.py process detected. proceeding."
echo

python3 run_bar_smoothing_test.py
STATUS=$?

echo
if [ $STATUS -ne 0 ]; then
    echo "run_bar_smoothing_test.py exited with status $STATUS - check ../../../output/logs/run_bar_smoothing_test.log for details."
    echo "IMPORTANT: verify betaconfig.py looks like your real config before doing anything else - the script"
    echo "restores it in a finally-block even on error, but this is worth a manual double-check regardless."
else
    echo "done. Results are in ../../../bar_smoothing_test_results/ - full log in ../../../output/logs/run_bar_smoothing_test.log"
fi
exit $STATUS
