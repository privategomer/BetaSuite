#!/bin/bash
#
# suppression_ab.sh - the suppression rules A/B, under its old name.
#
# The A/B now covers more than suppression (min_track_hits, match
# distance, shot-cut bursts, any per-label setting): see
# tools/tuning/ab_compare.sh. This runs it with the suppression
# experiment, so every mode and env var it took still works:
#
#   bash tools/tuning/suppression_ab.sh                  # a whole round
#   bash tools/tuning/suppression_ab.sh explain [DIR]
#   bash tools/tuning/suppression_ab.sh decide [DIR]
#   bash tools/tuning/suppression_ab.sh confirm [DIR]
#   AB_FOCUS=exposed_vulva bash tools/tuning/suppression_ab.sh

AB_EXPERIMENT="${AB_EXPERIMENT:-suppression}" exec bash "$(dirname "${BASH_SOURCE[0]}")/ab_compare.sh" "$@"
