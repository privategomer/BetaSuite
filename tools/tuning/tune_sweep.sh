#!/usr/bin/env bash
# tune_sweep.sh - runs betatv.py in preview mode once per combination of
# picture_sizes / video_censor_fps listed below, all against the SAME
# pinned preview slice (so speed/quality differences you see afterward are
# actually caused by the config change, not by different videos or a
# different random slice each time).
#
# nn_batch_size is deliberately NOT swept here - see batch_ab_test.py in
# this folder instead. Reason: nn_batch_size doesn't affect the detection
# cache key (box_hash_path in betatv.py), which is correct - it's not
# supposed to change detections, just how many frames get batched per
# neural-net call - but that also means sweeping it here would silently
# replay a cached result for every batch size after the first, making its
# timing meaningless (confirmed: that's exactly what happened the one time
# this script did sweep it). picture_sizes and video_censor_fps genuinely
# DO change detections and are correctly cache-keyed per-combo already, so
# sweeping them here is safe as-is and always produces a real, freshly
# detected, inspectable output file per combination.
#
# You run this yourself - it just repeatedly invokes betatv.py with CLI
# overrides (see betautils_cli.py). It never edits betaconfig.py; nothing
# here is a permanent change. After it finishes, run:
#     python3 tools/analysis/summarize_tune_sweep.py --preview-offset $PREVIEW_START_SECONDS
# to get a table comparing detection_seconds/encode_seconds/label_counts
# across every combination - passing --preview-offset matters: without it,
# older rows from a different preview slice (different footage) get mixed
# into the same comparison and the numbers stop being apples-to-apples.
#
# Edit the two arrays below to whatever you want to test, then:
#     chmod +x tune_sweep.sh && ./tune_sweep.sh
#
# NOTE: this reprocesses every file currently in betaconst.video_path_uncensored
# once per combination. The defaults below are 5 picture_sizes x 3 fps = 15
# combinations, so with several test videos in that folder this is 15 full
# preview passes over every one of them - trim the folder down to 1-2
# representative videos first if you just want a quick read on the effect
# of one knob, or trim these arrays down to fewer values.

set -euo pipefail

# lives in tools/tuning/ - pin CWD to the core dir (two levels up) so
# betatv.py and its CWD-relative constants (betaconst.video_path_uncensored,
# etc) resolve exactly as they did when this script sat in the core dir.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$CORE_DIR"

# --- knobs to sweep - edit these ---
PICTURE_SIZES_OPTIONS=(
    "1280"
    "2000"
    "1280 2000"
)
FPS_OPTIONS=(
    "9"
)

# --- fixed for every run, so every combination is compared on the same slice ---
PREVIEW_START_SECONDS="${PREVIEW_START_SECONDS:-30.0}"   # override: PREVIEW_START_SECONDS=45 ./tune_sweep.sh
PREVIEW_SECONDS="${PREVIEW_SECONDS:-20}"

# Sanity-check the arrays above before running anything: bash arrays are
# newline/space-separated, not comma-separated, so a trailing comma glued
# onto a quoted entry (e.g. "10", instead of "10") silently becomes part
# of that array element's string value instead of a separator - it won't
# fail until argparse rejects it as an invalid number, potentially several
# full preview passes into the sweep. Caught here instead, before the
# first run.
for arr_name in PICTURE_SIZES_OPTIONS FPS_OPTIONS; do
    declare -n arr_ref="$arr_name"
    for entry in "${arr_ref[@]}"; do
        if [[ "$entry" == *,* ]]; then
            echo "ERROR: $arr_name contains a comma in entry '$entry' - these are bash arrays (space/newline-separated), not comma-separated. Remove the comma(s) and try again." >&2
            exit 1
        fi
    done
done

run_count=0
total_runs=$(( ${#PICTURE_SIZES_OPTIONS[@]} * ${#FPS_OPTIONS[@]} ))

for sizes in "${PICTURE_SIZES_OPTIONS[@]}"; do
    for fps in "${FPS_OPTIONS[@]}"; do
        run_count=$((run_count + 1))
        echo "=== sweep run $run_count/$total_runs: picture_sizes=[$sizes] video_censor_fps=$fps ==="
        python3 betatv.py \
            --preview on \
            --preview-start-seconds "$PREVIEW_START_SECONDS" \
            --preview-seconds "$PREVIEW_SECONDS" \
            --picture-sizes $sizes \
            --video-censor-fps "$fps"
    done
done

echo
echo "sweep done ($total_runs runs). Now run: python3 tools/analysis/summarize_tune_sweep.py --preview-offset $PREVIEW_START_SECONDS"
