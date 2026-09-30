#!/usr/bin/env bash
# run_test_plan.sh - runs the full picture_size/fps sweep + nn_batch_size
# test sequence in order, pausing at the two points that actually need a
# human: watching the rendered previews to pick a picture_sizes candidate,
# and reviewing the final batch_ab_test.py results.
#
# What this does NOT try to automate: picking which picture_sizes combo
# looks best (that's a "go watch the video" judgment call, not a stat), and
# any follow-up digging into a specific lapse/tie-break (bring that back to
# Claude, it needs the actual footage/timestamps in front of it).
#
# Usage: chmod +x run_test_plan.sh && ./run_test_plan.sh
#
# Override the pinned preview slice if you want:
#     PREVIEW_START_SECONDS=450 PREVIEW_SECONDS=33 ./run_test_plan.sh

set -euo pipefail

# lives in tools/tuning/ - pin CWD to the core dir (two levels up), same
# reasoning as tune_sweep.sh/tune_harness.sh in this folder.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
ANALYSIS_DIR="$CORE_DIR/tools/analysis"
cd "$CORE_DIR"

PREVIEW_START_SECONDS="${PREVIEW_START_SECONDS:-450}"
PREVIEW_SECONDS="${PREVIEW_SECONDS:-33}"

pause() {
    echo
    echo ">>> $1"
    read -r -p ">>> press enter (or type anything) to continue: " _
    echo
}

echo "=== step 1/4: clearing cached detections for preview_start_seconds=$PREVIEW_START_SECONDS ==="
# same '%.1f' formatting box_hash_path_for/betatv.py use for the cache
# filename suffix, so this actually matches what's on disk regardless of
# whether PREVIEW_START_SECONDS was passed as e.g. "450" or "450.5"
PREVIEW_SUFFIX=$(printf '%.1f' "$PREVIEW_START_SECONDS")
CLEARED=0
for f in output/cache/vid_hashes/*-preview@${PREVIEW_SUFFIX}s.gz; do
    [ -e "$f" ] || continue
    rm -f "$f"
    CLEARED=$((CLEARED + 1))
done
echo "cleared $CLEARED cached detection file(s) at this preview slice - every combo below will run fresh."

echo
echo "=== step 2/4: running tune_sweep.sh (picture_sizes x fps sweep) ==="
PREVIEW_START_SECONDS="$PREVIEW_START_SECONDS" PREVIEW_SECONDS="$PREVIEW_SECONDS" "$SCRIPT_DIR/tune_sweep.sh"

echo
echo "=== step 3/4: summarizing sweep ==="
python3 "$ANALYSIS_DIR/summarize_tune_sweep.py" --preview-offset "$PREVIEW_START_SECONDS"

pause "Sweep done. Go watch the preview outputs for each picture_sizes combo now and decide which one (or two, for a normal/strict split) you want nn_batch_size tested against."

echo "Which picture_sizes do you want the batch test run against?"
read -r -p "  picture_sizes (space separated, e.g. '1280 2000'): " CHOSEN_SIZES
if [[ "$CHOSEN_SIZES" == *,* ]]; then
    echo "ERROR: that looks comma-separated - picture_sizes here are space-separated (e.g. '1280 2000', not '1280,2000'). Re-run and try again." >&2
    exit 1
fi
read -r -p "  video_censor_fps [default 8]: " CHOSEN_FPS
CHOSEN_FPS="${CHOSEN_FPS:-8}"

echo
echo "=== step 4/4: nn_batch_size test on picture_sizes=[$CHOSEN_SIZES] fps=$CHOSEN_FPS ==="
python3 "$SCRIPT_DIR/batch_ab_test.py" --batch-sizes 1 4 8 \
    --picture-sizes $CHOSEN_SIZES \
    --video-censor-fps "$CHOSEN_FPS" \
    --preview-start-seconds "$PREVIEW_START_SECONDS" \
    --preview-seconds "$PREVIEW_SECONDS"

pause "All done. Review the batch_ab_test.py verdicts above (IDENTICAL / MATCHES WITHIN TOLERANCE / REAL DIFFERENCES FOUND) and the rendered previews. If you're down to two finalists and want a stat tie-break, or spot a lapse once you're running for real, bring it back to Claude."

echo "test plan complete."
