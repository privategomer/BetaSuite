#!/usr/bin/env python3
"""
suppression_ab.py - the suppression experiment of ab_compare.py, under its old name.

Everything lives in ab_compare.py (the A/B engine) and ab_experiments.py
(what each experiment changes). This name runs the suppression
experiment and keeps the functions earlier scripts and tests import
from here. New work: tools/tuning/ab_compare.py / ab_compare.sh.
"""

import os
import sys

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )

import ab_compare as ab
import ab_experiments

from ab_compare import (                                             # noqa: F401
    COLOUR_BOTH, COLOUR_CURRENT_ONLY, COLOUR_PROPOSED_ONLY, MATCH_IOU, FEEDBACK_COLUMNS, VERDICTS,
    _load_betabench, box_seconds, cause_seconds, compare_video, compare_with_causes, diff_boxes,
    explain_video, motion_interpolation_off, noted, overlay_boxes, phase_decide, phase_explain,
    phase_rank, phase_render, pick_windows, rank_videos, render_video_comparison, render_window,
    replay_video, short_cause, side_by_side, tally_verdicts, window_events, bu_tuning )
from ab_experiments import (                                         # noqa: F401
    TRACKING_CAUSE, SuppressionLog, attribute, changes_by_rule, choose_rules, proposed_rules,
    rule_changes )


def main( argv=None ):
    return ab.main( argv, default_experiment='suppression' )


if __name__ == '__main__':
    raise SystemExit( main() )
