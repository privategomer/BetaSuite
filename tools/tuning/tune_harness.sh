#!/usr/bin/env bash
# tune_harness.sh - sequences the full post-run tuning workflow, unattended:
#
#   0. waits for any already-running betatv.py process to finish before
#      starting anything - the GPU on this box is single-consumer and has
#      OOM'd repeatedly even at nn_batch_size=1 on large files (see
#      betasuite.log), so nothing here runs concurrently with a real
#      production pass.
#   1. batch_ab_test.py --batch-sizes 1 2 --video-censor-fps 9 - now that
#      nn_batch_size=2 is the real betaconfig.py value, this just reconfirms
#      batch=2 still gives identical detections to the untouched batch=1
#      baseline before trusting it in production - batch_ab_test.py always
#      needs at least 2 --batch-sizes to compare against each other (it's a
#      comparator, not a single-size runner), so 1 stays in as the trusted
#      baseline. Also (re)populates the preview-slice cache steps 2-5 below
#      read from - batch_ab_test.py's whole method (clear cache, rerun,
#      compare numerically) only works apples-to-apples within its own
#      preview-slice cache - see its docstring for why a naive sweep across
#      batch sizes measures nothing real otherwise.
#   2. summarize_tune_sweep.py --preview-offset (scoped to just this run's
#      slice, per tune_sweep.sh's own note: without the offset filter,
#      older rows from a different preview slice mix in and the comparison
#      stops being apples-to-apples)
#   3. analyze_suppression_pairs.py - re-checks suppression stats against
#      whatever's in the cache now (cumulative across everything ever run,
#      not just this session)
#   4. replay_tune.py, once per video found under uncensored_vids, against
#      the fps=9 preview-slice cache step 1 just wrote (nn_batch_size isn't
#      part of the cache key, so it doesn't matter which batch size wrote
#      it), using replay_variants_example.json as a starting point - swap
#      in your own variants file if you want something specific tested
#      instead
#   5. analyze_jitter.py - best-effort; any video whose source file has
#      been moved/renamed since it was censored gets skipped with a
#      warning rather than failing the run, same as last time
#
# Everything after step 1 only reads caches/stats - no further GPU
# contention risk once batch_ab_test.py finishes.
#
# Run this with nohup so it survives closing the terminal (from tools/tuning/,
# create the output/logs dir first if it doesn't exist yet - mkdir -p ../../../output/logs):
#     nohup ./tune_harness.sh > ../../../output/logs/tune_harness_run.log 2>&1 &
# then check progress any time with:
#     tail -F ../../../output/logs/tune_harness_run.log
# (tail -F, capital F, retries if the file isn't there yet instead of
# exiting immediately - harmless here since the shell redirect creates the
# file before the script even starts, but it's the safer habit generally)

set -uo pipefail

# this script lives in tools/tuning/ - CORE_DIR is the actual BetaSuite-0.2.4
# directory (two levels up), and everything below is written to behave
# exactly as it did when this file sat directly in the core dir: CWD is
# pinned to CORE_DIR (so every CWD-relative path below - ../bin/activate,
# betaconst.video_path_uncensored, etc - resolves the same as before), while
# sibling scripts that moved along with this one are invoked via their new,
# explicit locations. (tune_harness_run.log itself is only ever written by
# whatever nohup redirect the user starts this with - see the usage comment
# above - not by this script directly.)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
ANALYSIS_DIR="$CORE_DIR/tools/analysis"
cd "$CORE_DIR"

# --- venv ---
if [ -f ../bin/activate ]; then
    source ../bin/activate
fi

PREVIEW_START_SECONDS="${PREVIEW_START_SECONDS:-450}"
PREVIEW_SECONDS="${PREVIEW_SECONDS:-33}"
VIDEO_CENSOR_FPS="${VIDEO_CENSOR_FPS:-9}"
export PREVIEW_START_SECONDS PREVIEW_SECONDS

echo "=== tune_harness.sh starting at $(date) ==="
echo "preview_start_seconds=$PREVIEW_START_SECONDS preview_seconds=$PREVIEW_SECONDS video_censor_fps=$VIDEO_CENSOR_FPS"
echo "override any of these by setting the env var before running, e.g. VIDEO_CENSOR_FPS=8 ./tune_harness.sh"

# --- 0. wait for any in-progress betatv.py run to clear the GPU ---
echo
echo "=== [0/5] checking for an already-running betatv.py process ==="
while pgrep -f "betatv\.py" > /dev/null 2>&1; do
    echo "$(date): betatv.py still running elsewhere - waiting 120s before checking again (GPU is single-consumer on this box, see betasuite.log's OOM history)"
    sleep 120
done
echo "GPU clear, no other betatv.py process detected. proceeding."

# --- 1. batch_ab_test.py: baseline (1) vs the real nn_batch_size (2) at fps=9 ---
echo
echo "=== [1/5] batch_ab_test.py --batch-sizes 1 2 --video-censor-fps $VIDEO_CENSOR_FPS ==="
python3 "$SCRIPT_DIR/batch_ab_test.py" --batch-sizes 1 2 --video-censor-fps "$VIDEO_CENSOR_FPS"
BATCH_STATUS=$?
if [ $BATCH_STATUS -ne 0 ]; then
    echo "WARNING: batch_ab_test.py exited with status $BATCH_STATUS, check the output above before trusting the steps below"
fi

# --- 2. summarize_tune_sweep.py, scoped to this run's slice ---
echo
echo "=== [2/5] summarize_tune_sweep.py --preview-offset $PREVIEW_START_SECONDS ==="
python3 "$ANALYSIS_DIR/summarize_tune_sweep.py" --preview-offset "$PREVIEW_START_SECONDS"

# --- 3. analyze_suppression_pairs.py ---
echo
echo "=== [3/5] analyze_suppression_pairs.py ==="
python3 "$ANALYSIS_DIR/analyze_suppression_pairs.py"

# --- 4. replay_tune.py, once per video, against the fresh fps=9 preview cache from step 1 ---
# (nn_batch_size isn't part of the detection cache key - batching shouldn't
# change detections - so this reads whichever preview-slice cache step 1
# just populated, regardless of which batch size wrote it)
echo
echo "=== [4/5] replay_tune.py (per video, fps=9 preview-slice cache from step 1, variants from replay_variants_example.json) ==="
VIDEO_LIST=$(python3 -c "
import betaconst, os
for root, d_names, f_names in os.walk(betaconst.video_path_uncensored):
    for fname in f_names:
        print(fname)
")
if [ -z "$VIDEO_LIST" ]; then
    echo "no video files found under $(python3 -c 'import betaconst; print(betaconst.video_path_uncensored)') - skipping replay_tune.py"
else
    while IFS= read -r fname; do
        echo "--- replay_tune.py --video \"$fname\" ---"
        python3 "$SCRIPT_DIR/replay_tune.py" --video "$fname" --preview-start-seconds "$PREVIEW_START_SECONDS" --variants "$SCRIPT_DIR/replay_variants_example.json"
    done <<< "$VIDEO_LIST"
fi

# --- 5. analyze_jitter.py ---
echo
echo "=== [5/5] analyze_jitter.py ==="
python3 "$ANALYSIS_DIR/analyze_jitter.py"

echo
echo "=== tune_harness.sh finished at $(date) ==="
