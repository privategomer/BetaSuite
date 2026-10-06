#!/usr/bin/env python3
"""
analyze_style_flicker.py - originally built to check a specific hypothesis
for the "a black bar appears briefly on an already-censored breast"
report: that it came from smooth_boxes's own paired_style mechanism
(betatv.py) rolling a style INDEPENDENTLY for a box, rather than sharing
it with an already-live box of the same label.

That hypothesis was confirmed against real cached footage: paired_style
used to only share a style between two boxes of a label detected in the
exact SAME frame timestamp (`live_this_frame == 2`) - by design, since 2+
OTHER simultaneous boxes was meant to read as multiple different people,
who should resolve independently. In practice that same-instant
requirement was almost never actually met: 796 of 799 (99.6%) independent
style resolves had another live box of the label already present, and 789
of those (99%) visually mismatched it. Distance data (normalized to box
size, same units as match_distance_multiplier) showed these were
overwhelmingly the SAME real pair just missing exact-same-instant
co-detection - median 0.09x box size, 99% under 1.0x, max 1.47x across
the real sample - not two different people, so proximity was the right
test, not frame-exactness.

smooth_boxes has since been changed accordingly: a newly-resolving box now
shares style with the nearest other currently-live track of the label
if exactly one is within `paired_style_max_distance` (default 2.0x box
size) - see the `paired_style` block in betatv.py, and
CONFIG_REFERENCE.md's item_overrides['exposed_breast'] section for the
full writeup. This tool now serves as the ongoing validation/regression
check for that fix: it does NOT patch or change smooth_boxes's behavior,
it's read-only, replaying your real cached detections through the actual,
unmodified smooth_boxes/apply_class_suppression (same extract-and-
instrument technique as analyze_jitter.py/analyze_track_breaks.py/
replay_tune.py) to measure, on YOUR real footage:

  - how often an independent style roll happens with ZERO other live
    boxes of that label (normal "first exposure" variety, not a bug)
  - how often it happens with 1+ OTHER already-live boxes of that label
    present post-fix (should be sharply lower than pre-fix, since most of
    those should now be sharing instead of independently resolving),
    broken out by whether the newly-rolled style DIFFERS from what's
    already showing, and specifically how often that mismatch lands on
    'bar'
  - for any REMAINING mismatch events, how close the newly-independent
    box is to the nearest other live box, in both raw px and normalized
    to box size - if a genuinely-far cluster shows up (different people
    in multi-person footage, correctly still resolving independently)
    it should now be visible as separated from a near-zero remainder,
    rather than blended into one near-universal population like before
    the fix

Prerequisite: same as analyze_jitter.py - real (non-preview) detection
caches must already exist for each backend's resolved picture_sizes (per
backend, not the shared betaconfig.picture_sizes) at your video_censor_fps.

As of 2026-09-16 this reports every registered backend with real (or, with
--include-preview, preview-slice) cache for this config as its own
section, same as analyze_jitter.py/analyze_track_breaks.py - previously
single-backend, and previously had a real bug (see box_hash_path_for's
FIXED comment below) where discover_full_run_videos's glob didn't pin the
backend name into what it searched for, which silently miscomputed
file_hash (backend name stuck to the end of it) and made every source
video look "not found" regardless of which directory was actually
searched - not a resources/source/ lookup problem, a cache-parsing one.

--include-preview (added 2026-09-16): same opt-in/CAVEAT as
analyze_jitter.py's flag of the same name.

Usage:
    python3 analyze_style_flicker.py
    python3 analyze_style_flicker.py --labels exposed_breast
    python3 analyze_style_flicker.py --include-preview
"""

import argparse
import copy
import os
import re
import sys

import cv2

import sys as _sys, os as _os
_sys.path.insert( 0, _os.path.dirname( _os.path.dirname( _os.path.dirname( _os.path.abspath( __file__ ) ) ) ) )  # so 'import betaconst'/'betaconfig'/etc still resolve after this script moved into tools/analysis or tools/tuning

import betaconst
import betautils_detector as bu_detector
import betaconfig
import betautils_hash as bu_hash
import betautils_censor as bu_censor
import betautils_cache_paths as bu_cache
import betautils_config as bu_config
import betautils_track as bu_track
from betautils_cache_paths import (
    box_hash_path_for, discover_full_run_videos, build_hash_to_video_path )

VID_HASH_DIR = bu_cache.VID_HASH_DIR

_events = []


def _reset_events():
    _events.clear()


_last_nearby_count = None
_last_exclusions = None
_current_video = None


def _record_exclusion( reason, x, y, w, h, prev, ref_start=None ):
    # fires inside _nearest_unambiguous_live_track's candidate loop, once
    # per OTHER track excluded BEFORE distance is even checked - 'gap'
    # for the max_gap/liveness check, 'cut' for _cut_between (a shot cut
    # separating this track from the reference frame). Since exclusion
    # happens upstream of the real loop's own distance computation, this
    # computes distance itself (same box-size normalization as
    # _record_independent_style_event) so a close-but-excluded track can
    # be told apart from a legitimately-far one that just happened to
    # also fail this check - the report's whole point. Reset per new-box
    # resolve via _reset_exclusions.
    #
    # ref_start (only meaningful for reason='gap') is the box currently
    # being resolved's own start time - ref_start - prev['end'] is the
    # actual gap duration that tripped max_gap, letting the report show
    # real gap-size distribution instead of just a close/far distance -
    # needed to size a track_max_gap increase off data instead of a guess.
    global _last_exclusions
    cx, cy = x + w/2, y + h/2
    pcx, pcy = prev['x'] + prev['w']/2, prev['y'] + prev['h']/2
    dist = ( (cx-pcx)**2 + (cy-pcy)**2 ) ** 0.5
    scale = max( prev['w'], prev['h'], w, h )
    norm_dist = ( dist / scale ) if scale else None
    gap_duration = ( ref_start - prev['end'] ) if ( reason == 'gap' and ref_start is not None ) else None
    _last_exclusions.append( { 'reason': reason, 'dist': dist, 'norm_dist': norm_dist, 'gap_duration': gap_duration } )


def _reset_exclusions():
    # called right where _nearest_unambiguous_live_track starts building
    # its candidate list (candidates = []) - anchors the per-call
    # exclusion tally to exactly this call, discarding whatever the PREVIOUS
    # call to this function left behind.
    global _last_exclusions
    _last_exclusions = []


def _record_nearby_count( nearby_count ):
    # fires right where smooth_boxes's _nearest_unambiguous_live_track
    # computes its full distance/liveness/max_gap/shot-cut-filtered
    # candidate list (len(candidates), captured right after it's sorted -
    # this is the true "how many OTHER tracks were even in range" count,
    # independent of whether the function's own tiebreak margin then
    # decided the nearest one was unambiguous enough to actually share
    # with). Stashed here (rather than passed as an argument) because this
    # hook fires BEFORE _record_independent_style_event, only when the
    # independent-resolve branch is about to run - so by the time that
    # second hook fires, this is exactly the candidate count that led to
    # it (whether because candidates was empty, or because it had 2+ and
    # the nearest two were too close together to call unambiguously).
    global _last_nearby_count
    _last_nearby_count = nearby_count


def _record_independent_style_event( label, box, tracks, paired_style ):
    # fires right after a box's style was resolved INDEPENDENTLY (the
    # else-branch of smooth_boxes's "shared_style is not None" check) -
    # `tracks` at this point holds every OTHER box of this same label
    # already live this frame (continuing tracks already updated, plus
    # any earlier new-this-frame box in this same loop) - see this
    # script's docstring for why that's exactly the right thing to check.
    #
    # NOTE: `tracks` here is NOT distance/liveness-filtered - it's every
    # currently-tracked box of the label, which is why num_other_live can
    # be high even in a scene with one real close pair (a second person
    # elsewhere in frame counts too). _last_nearby_count (see
    # _record_nearby_count above) is the ACTUAL len(nearby) from
    # _nearby_live_track_indices - the real gate paired_style used to
    # decide not to share - and is what actually explains why this event
    # happened: 0 means nothing survived the distance/liveness filter,
    # 2+ means more than one candidate did (the "multiple people" case
    # the len(nearby)==1 restriction exists to protect), either way not
    # a bug - only worth investigating if nearby_count is consistently
    # low (0) at small nearest_dist, which would mean something OTHER
    # than distance (liveness/max_gap, most likely) is excluding a track
    # that should visually have counted as "the same nearby pair".
    resolved_type = box['censor_style'].get( 'type' )
    cx = box['x'] + box['w']/2
    cy = box['y'] + box['h']/2
    other_types = []
    nearest_dist = None
    nearest_norm_dist = None
    for prev in tracks:
        other_types.append( prev['censor_style'].get( 'type' ) )
        pcx = prev['x'] + prev['w']/2
        pcy = prev['y'] + prev['h']/2
        dist = ( (cx-pcx)**2 + (cy-pcy)**2 ) ** 0.5
        # same normalization betatv.py's own match_distance_multiplier uses
        # (dist >= multiplier * max(prev.w, prev.h, box.w, box.h)) - expressing
        # distance in these units, not raw px, is what makes a future
        # paired_style_max_distance default directly comparable to/
        # consistent with the existing match-distance knob, and comparable
        # across videos of different resolutions/box sizes.
        scale = max( prev['w'], prev['h'], box['w'], box['h'] )
        norm_dist = ( dist / scale ) if scale else None
        if nearest_dist is None or dist < nearest_dist:
            nearest_dist = dist
            nearest_norm_dist = norm_dist

    mismatch = bool( other_types ) and any( t != resolved_type for t in other_types )
    exclusions = _last_exclusions or []
    # "close" here uses the SAME normalized-distance definition as the
    # report's existing close-by-raw-distance check (nearest_norm_dist <
    # 1.0x box size) - an exclusion this close is the specifically
    # suspicious case: excluded for gap/cut BEFORE distance was even
    # checked, yet would have easily passed the distance test.
    close_gap_exclusions = sum( 1 for e in exclusions if e['reason'] == 'gap' and e['norm_dist'] is not None and e['norm_dist'] < 1.0 )
    close_cut_exclusions = sum( 1 for e in exclusions if e['reason'] == 'cut' and e['norm_dist'] is not None and e['norm_dist'] < 1.0 )
    close_gap_durations = [
        e['gap_duration'] for e in exclusions
        if e['reason'] == 'gap' and e['norm_dist'] is not None and e['norm_dist'] < 1.0 and e['gap_duration'] is not None
    ]
    _events.append( {
        'label': label,
        't': box.get( 't', box.get( 'start' ) ),
        'resolved_type': resolved_type,
        'num_other_live': len( other_types ),
        'other_types': other_types,
        'mismatch': mismatch,
        'nearest_dist': nearest_dist,
        'nearest_norm_dist': nearest_norm_dist,
        'paired_style_on': paired_style,
        'nearby_gate_count': _last_nearby_count,
        'gap_exclusions': sum( 1 for e in exclusions if e['reason'] == 'gap' ),
        'cut_exclusions': sum( 1 for e in exclusions if e['reason'] == 'cut' ),
        'close_gap_exclusions': close_gap_exclusions,
        'close_cut_exclusions': close_cut_exclusions,
        'close_gap_durations': close_gap_durations,
        'video': _current_video,
    } )


class _StyleFlickerObserver( bu_track.TrackingObserver ):
    """Records every independent style resolve, with what was live nearby."""

    def __init__( self, settings_by_label=None ):
        self.settings_by_label = settings_by_label or {}

    def on_independent_style_resolve( self, label, box, tracks, settings ):
        self.settings_by_label[label] = settings
        _record_independent_style_event( label, box, tracks, settings.paired_style )


def _load_instrumented_pipeline():
    """
    The real tracking and suppression functions, with this tool's
    instrumentation attached through betautils_track's observer hook.

    Until 2.1 this read betatv.py's SOURCE TEXT, regex-matched the
    smooth_boxes body out of it, string-replaced instrumentation calls
    into specific lines, and exec'd the result. It broke the moment any
    of those lines moved, which is what happened in the 2.1 refactor.

    betautils_track.TrackingObserver is the supported extension point
    now: observers cannot change a tracking decision, so the real
    pipeline still runs exactly as it does in production.

    Returns:
        A (smooth_boxes, apply_class_suppression) pair. Install the
        observer around the call with bu_track.tracking_observer(...).
    """
    return { 'smooth_boxes': bu_track.smooth_boxes,
             'apply_class_suppression': bu_track.apply_class_suppression,
             'observer': _StyleFlickerObserver() }



def expected_bar_fraction( label, backend_name ):
    # the true baseline odds of an independent resolve landing on 'bar' -
    # computed from the label's actual configured censor_style list rather
    # than assumed, since the list isn't evenly split across type FAMILIES
    # (e.g. exposed_breast has 6 blur / 6 pixel / 3 bar, not 5/5/5) and
    # weights need not be equal either. Returns None if the label has no
    # list-form censor_style to compute this from (a single-dict style has
    # no variety to weight in the first place).
    style = bu_detector.get_item_overrides( label, backend_name ).get( 'censor_style' )
    if not isinstance( style, list ) or not style:
        return None
    total_weight = sum( entry.get( 'weight', 1 ) for entry in style )
    if not total_weight:
        return None
    bar_weight = sum( entry.get( 'weight', 1 ) for entry in style if entry.get( 'type' ) == 'bar' )
    return bar_weight / total_weight


# box_hash_path_for/discover_full_run_videos/_walk_and_hash_for/
# build_hash_to_video_path moved to betautils_cache_paths.py (shared
# with analyze_jitter.py/analyze_track_breaks.py/etc - see that
# module's docstring for why) - imported as bu_cache below.


def main():
    global _current_video
    parser = argparse.ArgumentParser( description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter )
    parser.add_argument( '--labels', nargs='+', default=None,
        help="restrict to these labels (default: everything currently in items_to_censor that has paired_style set)" )
    parser.add_argument( '--picture-sizes', type=int, nargs='+', default=None,
        help="analyse exactly these sizes instead of discovering what is on disk. By default every tool covers EVERY model configuration that has written detections." )
    parser.add_argument( '--video-censor-fps', type=float, default=float( betaconfig.video_censor_fps ) )
    parser.add_argument( '--include-preview', action='store_true',
        help="also use preview-slice caches for any video that has no real (non-preview) cache yet - see "
             "analyze_jitter.py's docstring CAVEAT (same tradeoffs apply here)." )
    parser.add_argument( '--variants', nargs='+', default=None,
        help="restrict analysis to these model variants, e.g. --variants 640m. Backend "
             "alone cannot express this - 320n and 640m are both nudenet_v3 - so this is "
             "the knob for focusing on one variant once it is the one you are tuning. It "
             "only narrows which cached configurations are READ; nothing on disk changes, "
             "so widening it again later needs no re-run" )
    args = parser.parse_args()

    # paired_style is a backend-tunable item override, so
    # which labels are worth checking is a per-backend question - the
    # labels are resolved inside the backend loop below, not once here
    # from whichever backend happens to be selected.
    global_min_prob = getattr( betaconfig, 'global_min_prob', betaconst.global_min_prob )

    if args.include_preview:
        print( "!!! --include-preview is on: videos with no real (non-preview) cache will use a preview-slice "
               "cache instead. Gate-count/gap-duration stats below reflect that short window, not real full-video "
               "behavior - not comparable to a real-run number. !!!" )
        print()

    ns = _load_instrumented_pipeline()
    smooth_boxes_fn = ns['smooth_boxes']
    apply_suppression_fn = ns['apply_class_suppression']

    configurations = bu_cache.configurations_to_analyse(
        args.picture_sizes, fps=args.video_censor_fps, min_prob=global_min_prob,
        include_preview=args.include_preview, variant_names=args.variants )
    print( "analysing %d configuration(s): %s"%(
        len( configurations ), bu_cache.describe_configurations( configurations ) ) )
    print()

    any_backend_had_videos = False
    for config in configurations:
        backend_name = config.backend_name
        sizes = config.picture_sizes
        videos, preview_used_for = discover_full_run_videos(
            sizes, args.video_censor_fps, global_min_prob, backend_name, include_preview=args.include_preview )
        print( "#" * 70 )
        print( "configuration: %s"%(config.label) )
        print( "#" * 70 )

        labels = args.labels or [
            label for label in betaconfig.items_to_censor
            if bu_detector.get_item_overrides( label, backend_name ).get( 'paired_style' )
        ]
        if not labels:
            print( "  no labels with paired_style set for this backend (or none matched --labels) - nothing to check here." )
            print()
            continue
        if not videos:
            print( "  no real (non-preview) detection caches found for this backend at picture_sizes=%s fps=%s "
                   "min_prob=%.3f - skipping (run betatv.py with detector_backend['selected']=%r on real footage "
                   "first to get data here%s)."%(
                       sizes, args.video_censor_fps, global_min_prob, backend_name,
                       ", or check the preview-cache warnings above if --include-preview was used" if args.include_preview else " (or pass --include-preview to also try preview-slice caches)" ) )
            print()
            continue
        any_backend_had_videos = True

        num_preview = len( preview_used_for )
        if num_preview:
            print( "found %d video(s) with usable caches for this config (%d real, %d preview-slice)"%(
                len(videos), len(videos)-num_preview, num_preview) )
            print( "  preview-slice video(s): %s"%(
                ", ".join( "%s (%s)"%(h, preview_used_for[h]) for h in sorted(preview_used_for) ) ) )
        else:
            print( "found %d video(s) with complete real detection caches for this config"%(len(videos)) )
        hash_to_path = build_hash_to_video_path( videos.keys() )
        missing = set(videos.keys()) - set(hash_to_path.keys())
        if missing:
            print( "  (%d of those source video file(s) could not be found under %s or %s - probably moved/deleted "
                   "since being censored; they'll be skipped)"%(
                       len(missing), betaconst.video_path_uncensored, betaconst.video_path_source_backup ) )
        print()

        _reset_events()
        dims_missing = []
        for file_hash, cache_paths in videos.items():
            video_path = hash_to_path.get( file_hash )
            if not video_path:
                dims_missing.append( file_hash )
                continue

            flat_raw = []
            for path in cache_paths:
                if os.path.exists( path ):
                    flat_raw.extend( bu_hash.read_json( path ) )
            if not flat_raw:
                continue

            cap = cv2.VideoCapture( video_path )
            vid_w = int( cap.get( cv2.CAP_PROP_FRAME_WIDTH ) )
            vid_h = int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT ) )
            cap.release()
            if not vid_w or not vid_h:
                dims_missing.append( file_hash )
                continue

            raw_copy = copy.deepcopy( flat_raw )
            surviving, _ = apply_suppression_fn(
                raw_copy,
                bu_detector.get_class_suppression( backend_name ),
                bu_config.get_parts_to_blur( backend_name ) )

            boxes = []
            for r in surviving:
                res = bu_censor.process_raw_box( r, vid_w, vid_h )
                if res and res['label'] in labels:
                    boxes.append( res )
            if not boxes:
                continue

            _current_video = os.path.basename( video_path )
            with bu_track.tracking_observer( ns['observer'] ):
                smooth_boxes_fn( boxes, backend_name=backend_name )

        if dims_missing:
            print( "  (skipped %d video(s) - source file no longer found)"%(len(dims_missing)) )

        by_label = {}
        for ev in _events:
            by_label.setdefault( ev['label'], [] ).append( ev )

        if not by_label:
            print( "no independent style-resolve events recorded for %s - either paired_style isn't set for these labels, "
                   "or every resolve in this footage happened to share a style (unlikely at n>0 detections, worth "
                   "double-checking --labels)."%(labels) )
            print()
            continue

        for label, events in by_label.items():
            total = len( events )
            solo = [ e for e in events if e['num_other_live'] == 0 ]
            with_others = [ e for e in events if e['num_other_live'] > 0 ]
            mismatched = [ e for e in with_others if e['mismatch'] ]
            mismatched_to_bar = [ e for e in mismatched if e['resolved_type'] == 'bar' ]
            bar_overall = [ e for e in events if e['resolved_type'] == 'bar' ]

            expected_bar = expected_bar_fraction( label, backend_name )
            if expected_bar is not None:
                baseline_note = "expected ~%.0f%% from this label's own configured censor_style weights, if styles are picked independent of this"%(100*expected_bar)
            else:
                baseline_note = "no list-form censor_style found for this label to compute an expected baseline from"

            print( "=== %s ==="%(label) )
            print( "  independent style-resolve events: %d total"%(total) )
            print( "    resolved to 'bar': %d (%.0f%% of all independent resolves - %s)"%(
                len(bar_overall), 100*len(bar_overall)/total if total else 0, baseline_note ) )
            print( "    solo (no other live box of this label at that moment - normal 'first exposure' variety, not a mismatch): %d (%.0f%%)"%(
                len(solo), 100*len(solo)/total if total else 0 ) )
            print( "    with other live box(es) already present: %d (%.0f%%)"%(
                len(with_others), 100*len(with_others)/total if total else 0 ) )

            # the real paired_style gate (len(nearby)==1) is distance AND
            # liveness/max_gap/shot-cut filtered - 'with other live box(es)'
            # above is NOT (it just checks whether ANY track of the label is
            # currently tracked at all, anywhere in frame). This breakdown is
            # what actually explains why paired_style didn't fire for each
            # event: 0 means nothing survived the real gate's distance/
            # liveness filter even though `tracks` had something in it
            # (worth investigating if this is common AND nearest_dist is
            # small - would mean max_gap/liveness, not distance, is the
            # real blocker); 2+ means more than one candidate did survive,
            # which is the "probably multiple people" case the ==1
            # restriction exists to protect and is not a bug.
            gate_counts = [ e['nearby_gate_count'] for e in with_others if e.get( 'nearby_gate_count' ) is not None ]
            if gate_counts:
                from collections import Counter
                gate_hist = Counter( gate_counts )
                print( "      of those, the actual paired_style gate (len(nearby), distance+liveness filtered) saw: " + ", ".join(
                    "%d nearby=%d (%.0f%%)"%(n, gate_hist[n], 100*gate_hist[n]/len(gate_counts)) for n in sorted( gate_hist ) ) )
                zero_nearby = [ e for e in with_others if e.get( 'nearby_gate_count' ) == 0 and e['nearest_dist'] is not None ]
                if zero_nearby:
                    close_zero_nearby = [ e for e in zero_nearby if e['nearest_norm_dist'] is not None and e['nearest_norm_dist'] < 1.0 ]
                    if close_zero_nearby:
                        print( "      of the 'gate saw 0 nearby' events, %d (%.0f%% of those) were still within 1.0x box-size of the "
                               "nearest live box by raw distance - the real gate is filtering these out for a reason OTHER than distance"%(
                                   len(close_zero_nearby), 100*len(close_zero_nearby)/len(zero_nearby) ) )
                # direct answer to WHICH non-distance reason is doing the
                # filtering - max_gap/liveness ('gap') vs a shot cut
                # ('cut') actually separating a genuinely-close pair - so this
                # doesn't have to be inferred from nearest_dist/gate_count
                # alone. Only counts exclusions that were themselves close by
                # distance (same <1.0x norm_dist bar as above) - a track
                # excluded for being far away is expected, not a signal.
                total_close_gap = sum( e.get( 'close_gap_exclusions', 0 ) for e in with_others )
                total_close_cut = sum( e.get( 'close_cut_exclusions', 0 ) for e in with_others )
                if total_close_gap or total_close_cut:
                    print( "      of ALL exclusions (any nearby_gate_count, not just 0) that were still close (<1.0x box-size) when "
                           "excluded: %d were excluded by track_max_gap/liveness ('gap'), %d were excluded by a detected shot cut "
                           "('cut') - %s"%(
                               total_close_gap, total_close_cut,
                               "gap/max_gap is the dominant cause, worth raising track_max_gap or investigating why liveness lapsed" if total_close_gap > total_close_cut
                               else "shot-cut detection is the dominant cause - check whether cuts are being flagged mid-pair (a real cut wrongly splitting one continuous shot, or shot_cut_threshold too sensitive)" if total_close_cut > total_close_gap
                               else "roughly even split between the two causes" ) )
                    # actual gap-DURATION distribution for the close-and-gap-excluded
                    # cases - answers "how much would track_max_gap need to go up
                    # by" with real numbers instead of guessing at a new value.
                    close_gap_durations = sorted(
                        d for e in with_others for d in e.get( 'close_gap_durations', [] )
                    )
                    if close_gap_durations:
                        n = len( close_gap_durations )
                        def _pct( p ):
                            idx = min( n - 1, int( p * n ) )
                            return close_gap_durations[idx]
                        print( "      gap DURATION (seconds) for those close-and-gap-excluded cases: "
                               "n=%d median=%.2fs p75=%.2fs p90=%.2fs max=%.2fs - current track_max_gap "
                               "would need to exceed the relevant percentile to stop excluding most of these"%(
                                   n, _pct(0.5), _pct(0.75), _pct(0.90), close_gap_durations[-1] ) )
                        # PER-VIDEO breakdown - a large aggregate gap duration is
                        # ambiguous: it could mean a few files with huge gaps (most
                        # likely coincidental same-screen-position reuse between
                        # unrelated scenes in compilation/multi-clip footage, where
                        # "close by pixel position" is meaningless since it's a
                        # different clip entirely - NOT a bug, NOT something
                        # raising track_max_gap should try to bridge) or a lot of
                        # files each with a handful of genuinely-short lapses
                        # (which WOULD be worth bridging with a bigger max_gap).
                        # This tells them apart per file instead of guessing from
                        # one blended number.
                        by_video = {}
                        for e in with_others:
                            for d in e.get( 'close_gap_durations', [] ):
                                by_video.setdefault( e.get('video') or '(unknown)', [] ).append( d )
                        if len( by_video ) > 1:
                            print( "      same, broken out per video (n=count of close-and-gap-excluded cases in that file):" )
                            for vid, durs in sorted( by_video.items(), key=lambda kv: -len(kv[1]) ):
                                durs.sort()
                                vn = len( durs )
                                print( "        %s: n=%d median=%.2fs p90=%.2fs max=%.2fs"%(
                                    vid, vn, durs[vn//2], durs[min(vn-1,int(0.9*vn))], durs[-1] ) )
            if with_others:
                print( "      of those, resolved to a DIFFERENT style than what's already showing (the real 'flicker' case): %d (%.0f%%)"%(
                    len(mismatched), 100*len(mismatched)/len(with_others) ) )
                if mismatched:
                    print( "        of those mismatches, specifically landed on 'bar': %d (%.0f%%)"%(
                        len(mismatched_to_bar), 100*len(mismatched_to_bar)/len(mismatched) ) )

                    # distance breakdown across ALL mismatches, not just the
                    # bar ones - bar is the most visually noticeable outcome
                    # but it's one style family among several, and picking a
                    # distance threshold for a fix off bar-only evidence would
                    # be tuning against a biased subset. If mismatches at large
                    # distance look the same as mismatches at small distance
                    # (all styles, not just bar, and no distance separation),
                    # that's real signal it's genuinely a mixed population of
                    # "same real pair, different frame" (close) and "different
                    # people" (far) cases, independent of which style got rolled.
                    def _dist_stats( evs ):
                        dists = sorted( e['nearest_dist'] for e in evs if e['nearest_dist'] is not None )
                        if not dists:
                            return None
                        return ( len(dists), dists[len(dists)//2],
                                 dists[min(len(dists)-1, int(len(dists)*0.9))], dists[-1] )

                    all_stats = _dist_stats( mismatched )
                    if all_stats:
                        n, med, p90, mx = all_stats
                        print( "        distance (px) to the nearest already-live box of this label, ALL mismatches (any style): "
                               "n=%d median=%.1f p90=%.1f max=%.1f"%(n, med, p90, mx) )
                    bar_stats = _dist_stats( mismatched_to_bar )
                    if bar_stats:
                        n, med, p90, mx = bar_stats
                        print( "        distance (px) to the nearest already-live box of this label, bar-mismatches only: "
                               "n=%d median=%.1f p90=%.1f max=%.1f"%(n, med, p90, mx) )
                    non_bar = [ e for e in mismatched if e['resolved_type'] != 'bar' ]
                    non_bar_stats = _dist_stats( non_bar )
                    if non_bar_stats:
                        n, med, p90, mx = non_bar_stats
                        print( "        distance (px) to the nearest already-live box of this label, non-bar mismatches: "
                               "n=%d median=%.1f p90=%.1f max=%.1f"%(n, med, p90, mx) )
                    if all_stats and non_bar_stats and bar_stats:
                        print( "        (similar medians across bar vs. non-bar suggests the distance pattern is driven by the "
                               "mismatch mechanism itself, not by which style happened to get rolled)" )

                    # same distances, but normalized to box size (dist / max(w,h)
                    # of the two boxes involved) - the same units betatv.py's own
                    # match_distance_multiplier uses. Raw px medians above aren't
                    # directly usable as a paired_style_max_distance default
                    # since they're tied to this footage's specific resolution/
                    # box scale; this histogram is, and can be compared directly
                    # against match_distance_multiplier's own default of 1.0.
                    norm_dists = sorted( e['nearest_norm_dist'] for e in mismatched if e['nearest_norm_dist'] is not None )
                    if norm_dists:
                        n = len( norm_dists )
                        buckets = [ 1.0, 2.0, 3.0, 5.0, 10.0 ]
                        counts = []
                        prev_edge = 0.0
                        for edge in buckets:
                            c = sum( 1 for d in norm_dists if prev_edge <= d < edge )
                            counts.append( ( "[%.0fx-%.0fx)"%(prev_edge, edge), c ) )
                            prev_edge = edge
                        counts.append( ( ">=%.0fx"%(buckets[-1]), sum( 1 for d in norm_dists if d >= buckets[-1] ) ) )
                        print( "        distance normalized to box size (dist / max(w,h) of the two boxes - same units as "
                               "match_distance_multiplier), ALL mismatches: n=%d median=%.2fx p90=%.2fx max=%.2fx"%(
                                   n, norm_dists[n//2], norm_dists[min(n-1,int(n*0.9))], norm_dists[-1] ) )
                        print( "          histogram: " + ", ".join( "%s=%d (%.0f%%)"%(label_, c, 100*c/n) for label_, c in counts ) )
            print()

        print( "reading this: check the 'resolved to bar' rate against the printed expected-baseline percentage FIRST - if they're " )
        print( "close, bar isn't over-represented, it's just the most visually jarring of the roughly-equally-likely outcomes, so it " )
        print( "gets noticed more than its real frequency. The more important signal is 'with other live box(es)' - if that's near- " )
        print( "universal (not just a rare edge case), the exactly-2 paired_style gate is essentially never firing when it visually " )
        print( "should. Then use the distance breakdown: if ALL/non-bar/bar-only medians all land close together and small, that's " )
        print( "consistent with the SAME real pair just missing exact-same-frame co-detection (not two different people) - the " )
        print( "concrete case for extending paired_style's sharing to any nearby already-live box of the label, not only when " )
        print( "exactly 2 are co-detected in the identical frame. A long tail out to large distances is more consistent with " )
        print( "genuinely different people (multi-scene/compilation footage) and should stay independent." )

    if not any_backend_had_videos:
        print( "no real (non-preview) detection caches found for any of (%s) at fps=%s min_prob=%.3f - run "
               "betatv.py on real footage first (not just --preview on), or pass --include-preview to also "
               "try preview-slice caches."%(
                   bu_cache.describe_configurations( configurations ),
                   args.video_censor_fps, global_min_prob ) )
        sys.exit(1)


if __name__ == '__main__':
    main()
