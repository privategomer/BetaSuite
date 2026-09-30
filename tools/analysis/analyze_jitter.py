#!/usr/bin/env python3
"""
analyze_jitter.py - measures how much smooth_boxes's tracking/smoothing
actually settles real per-frame detection noise on footage you've already
run through betatv.py, instead of guessing at position_smoothing (alpha)
by eye.

Why this exists: lowering position_smoothing reduces frame-to-frame jitter
by design - it's an exponential moving average (see smooth_boxes in
betatv.py): new_position = alpha*raw_detection + (1-alpha)*track's_last_
smoothed_position, so a lower alpha weights each new raw detection less
and keeps more of the track's existing position, meaning less snap - but
jitter can ALSO come from two things alpha can't fix at all:

  - track continuity breaking: smooth_boxes only blends a new detection
    into an existing track if it lands close enough (within
    max(prev_w,prev_h,new_w,new_h) pixels) and soon enough (within
    2/video_censor_fps seconds). Miss that and it's not blended at all -
    a brand new track starts at the raw, UNSMOOTHED position, a hard snap
    rather than gradual noise. No alpha value changes this, since the
    reset bypasses the blend entirely.
  - curved shapes (circle/ellipse) making the exact same positional noise
    look far more obviously "jittery" than a box shape would (there's
    already a comment about this in smooth_boxes itself).

This tool measures the real thing on your actual cached footage instead
of guessing which of those is actually happening:

  - per-track frame-to-frame position jump AFTER smoothing, at your
    CURRENT position_smoothing plus whatever --sweep-alpha values you
    pass, so you can see in real numbers whether a given alpha helps and
    by how much
  - how often a detection continues an existing track vs starts a brand
    new one, and specifically flags new tracks that had another track
    nearby-but-just-outside-threshold (a likely "reset", not a fresh
    appearance) - exactly the case alpha tuning can't touch

This uses the REAL smooth_boxes/apply_class_suppression functions,
extracted from betatv.py's actual source at runtime (same technique
replay_tune.py uses - see _load_real_pipeline_functions), with two small
instrumentation hooks inserted into the EXTRACTED COPY of smooth_boxes's
source (never the real file on disk) at its two existing decision points -
"this detection matched an existing track" and "this detection started a
new track" - purely to record which track each real detection ended up
on. No matching or smoothing logic is reimplemented anywhere in this
tool; the hooks only record what the real code already decided.

Prerequisite: real (non-preview) detection caches must already exist for
each backend's CURRENT resolved picture_sizes (per backend, not the
shared betaconfig.picture_sizes) at your video_censor_fps/global_min_prob -
i.e. you've actually run betatv.py on real footage, not just preview
slices. This only reads that cache and opens the matching source video
(to get its real width/height, same as process_raw_box needs); it never
runs the neural net itself.

--include-preview (added 2026-09-16): also reads preview-slice caches
(the '-preview'/'-preview@<offset>s' suffixed cache files a
'betatv.py --preview on' run writes) when no real run exists yet, or
you specifically want to compare backends against identical short
slices rather than wait for full real runs on every video. Read the
CAVEAT this flag prints before trusting numbers from it: a preview
slice is a short (tens of seconds), boundary-truncated window starting
mid-video, so absolute counts here (total track resets, gap durations
"over the video") reflect that narrow window, not real full-video
behavior, and aren't comparable to a real-run number or across videos
of different real lengths. What IS safe to read from preview data: a
same-slice, backend-vs-backend comparison - both backends' preview
caches cover the exact same seconds of the exact same video, so a
difference between them here is a real difference in how that backend
handled that footage, not an artifact of slice length.

Usage:
    python3 analyze_jitter.py
    python3 analyze_jitter.py --sweep-alpha 0.05 0.10 0.15 0.20 0.30
    python3 analyze_jitter.py --labels exposed_breast exposed_vulva
    python3 analyze_jitter.py --include-preview
"""

import argparse
import contextlib
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
    box_hash_path_for, discover_full_run_videos, build_hash_to_video_path, fmt_stats )

VID_HASH_DIR = bu_cache.VID_HASH_DIR


# ---- module-level event buffers the instrumentation hooks write to ----
_match_events = []
_new_track_events = []

def _reset_events():
    _match_events.clear()
    _new_track_events.clear()

def _record_match_event( label, ti, dist, b ):
    # Fires from inside the REAL smooth_boxes (extracted+instrumented, see
    # below) right after it decides this detection continues track `ti`,
    # but BEFORE smoothing mutates b's x/y/w/h - so `dist` here is exactly
    # the raw jump from the track's last (already-smoothed) position to
    # this frame's new raw detection: the noise alpha is being asked to
    # absorb.
    b['_diag_track_id'] = (label, ti)
    _match_events.append( { 'label': label, 'dist': dist } )

def _record_new_track_event( label, tracks, ti, b ):
    # Fires right after the real code appends a brand new track. `ti` is
    # that track's stable index for the rest of this label's processing
    # (tracks is append-only within one label, indices are never reused).
    b['_diag_track_id'] = (label, ti)
    # Heuristic, for reporting only: was there ANOTHER currently-live
    # track for this label within a looser 1.5x version of the real
    # match distance? If so this new track is more likely a missed
    # match/reset than a genuinely fresh appearance. This does NOT change
    # or re-run the real matching decision, which already happened before
    # this hook fires - it only flags the new track for the report.
    cx = b['x'] + b['w']/2
    cy = b['y'] + b['h']/2
    nearest = None
    for other_ti, prev in enumerate( tracks ):
        if other_ti == ti:
            continue
        pcx = prev['x'] + prev['w']/2
        pcy = prev['y'] + prev['h']/2
        d = ( (cx-pcx)**2 + (cy-pcy)**2 ) ** 0.5
        loose_threshold = 1.5 * max( prev['w'], prev['h'], b['w'], b['h'] )
        if d < loose_threshold and ( nearest is None or d < nearest ):
            nearest = d
    _new_track_events.append( { 'label': label, 'near_miss_dist': nearest } )


class _JitterObserver( bu_track.TrackingObserver ):
    """Records every match and every new track, for the jitter report."""

    def on_match( self, label, track_index, distance, box, tracks, settings ):
        _record_match_event( label, track_index, distance, box )

    def on_new_track( self, label, tracks, track_index, box, settings ):
        _record_new_track_event( label, tracks, track_index, box )


def _load_real_pipeline_functions():
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
    return bu_track.smooth_boxes, bu_track.apply_class_suppression



# box_hash_path_for/discover_full_run_videos/_walk_and_hash_for/
# build_hash_to_video_path/fmt_stats moved to betautils_cache_paths.py
# (shared with analyze_track_breaks.py/analyze_style_flicker.py/etc -
# see that module's docstring for why) - imported as bu_cache below.


@contextlib.contextmanager
def _swept_alpha( backend_name, labels, alpha ):
    """
    Temporarily force one position_smoothing value for this sweep point.

    Setting betaconfig.default_position_smoothing alone is not enough,
    and was the reason an alpha sweep could come back suspiciously
    flat. default_position_smoothing is only the FALLBACK: any label
    with its own position_smoothing override keeps it, and since v2.1
    that override can live in the backend's own
    detector_backend[<name>]['item_overrides'] block, which
    get_item_overrides merges on top of the shared one.

    Analogy: the default is the thermostat setting for the house; a
    per-label override is a space heater in one room. Turning the
    thermostat down does nothing to the room with the heater running.

    So this writes the swept value into both tiers - the global default
    and the backend block for every label being reported on - and
    restores both afterwards, including on Ctrl-C.
    """
    backend_setting = betaconfig.detector_backend.setdefault( backend_name, {} )
    saved_overrides = copy.deepcopy( backend_setting.get( 'item_overrides', {} ) )
    saved_default = betaconfig.default_position_smoothing

    swept = copy.deepcopy( saved_overrides )
    for label in labels:
        label_block = dict( swept.get( label, {} ) )
        label_block['position_smoothing'] = alpha
        swept[label] = label_block
    backend_setting['item_overrides'] = swept
    betaconfig.default_position_smoothing = alpha
    bu_config.invalidate_config_caches()
    try:
        yield
    finally:
        backend_setting['item_overrides'] = saved_overrides
        betaconfig.default_position_smoothing = saved_default
        bu_config.invalidate_config_caches()


def run_one_alpha( alpha, videos, hash_to_path, smooth_boxes_fn, apply_suppression_fn,
                   labels_filter, backend_name ):
    """
    Replay every cached video once at one position_smoothing value.

    backend_name is pinned rather than defaulted: this tool loops over
    every registered backend, and letting the tracking settings resolve
    to the *selected* backend would measure one backend's detections
    under another backend's smoothing config.
    """
    labels_to_sweep = labels_filter or set( betaconfig.items_to_censor )
    suppression_rules = bu_detector.get_class_suppression( backend_name )
    parts_to_blur = bu_config.get_parts_to_blur( backend_name )
    _reset_events()
    frame_deltas_by_label = {}
    dims_missing = []
    with _swept_alpha( backend_name, labels_to_sweep, alpha ):
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
            surviving, _ = apply_suppression_fn( raw_copy, suppression_rules, parts_to_blur )

            boxes = []
            for r in surviving:
                res = bu_censor.process_raw_box( r, vid_w, vid_h )
                if res and ( not labels_filter or res['label'] in labels_filter ):
                    boxes.append( res )
            if not boxes:
                continue

            with bu_track.tracking_observer( _JitterObserver() ):
                tracked_boxes, _smoothing_stats = smooth_boxes_fn( boxes, backend_name=backend_name )

            # group the ORIGINAL (non-interpolated) boxes by the track id
            # the instrumentation hooks tagged them with, then measure
            # consecutive-frame center-to-center pixel movement on their
            # FINAL SMOOTHED positions (tracked_boxes/boxes are the same
            # mutated objects) - this is the actual "how far does the
            # visible censor box jump frame to frame" number.
            by_track = {}
            for b in boxes:
                tid = b.get( '_diag_track_id' )
                if tid is None:
                    continue  # shouldn't happen, but don't crash the sweep over one odd box
                by_track.setdefault( tid, [] ).append( b )

            for (label, ti), track_boxes in by_track.items():
                track_boxes.sort( key=lambda b: b['t'] )
                for prev_b, cur_b in zip( track_boxes, track_boxes[1:] ):
                    pcx = prev_b['x'] + prev_b['w']/2
                    pcy = prev_b['y'] + prev_b['h']/2
                    ccx = cur_b['x'] + cur_b['w']/2
                    ccy = cur_b['y'] + cur_b['h']/2
                    delta = ( (ccx-pcx)**2 + (ccy-pcy)**2 ) ** 0.5
                    frame_deltas_by_label.setdefault( label, [] ).append( delta )

    match_dists_by_label = {}
    for ev in _match_events:
        match_dists_by_label.setdefault( ev['label'], [] ).append( ev['dist'] )

    new_track_by_label = {}
    near_miss_by_label = {}
    for ev in _new_track_events:
        new_track_by_label[ ev['label'] ] = new_track_by_label.get( ev['label'], 0 ) + 1
        if ev['near_miss_dist'] is not None:
            near_miss_by_label[ ev['label'] ] = near_miss_by_label.get( ev['label'], 0 ) + 1

    return {
        'frame_deltas_by_label': frame_deltas_by_label,
        'match_dists_by_label': match_dists_by_label,
        'new_track_by_label': new_track_by_label,
        'near_miss_by_label': near_miss_by_label,
        'dims_missing': dims_missing,
    }


def print_alpha_result( alpha, result, labels ):
    print( "=== position_smoothing (alpha) = %.2f ==="%(alpha) )
    if result['dims_missing']:
        print( "  (skipped %d video(s) - source file no longer found under %s or %s to read real width/height)"%(
            len(result['dims_missing']), betaconst.video_path_uncensored, betaconst.video_path_source_backup ) )
    for label in labels:
        deltas = result['frame_deltas_by_label'].get( label, [] )
        match_dists = result['match_dists_by_label'].get( label, [] )
        new_tracks = result['new_track_by_label'].get( label, 0 )
        near_misses = result['near_miss_by_label'].get( label, 0 )
        matches = len( match_dists )
        total_decisions = matches + new_tracks
        print( "  %s"%(label) )
        print( "    smoothed frame-to-frame jump (px):  %s"%(fmt_stats(deltas)) )
        print( "    raw jump smoothing had to absorb (px): %s"%(fmt_stats(match_dists)) )
        if total_decisions:
            print( "    continued existing track: %d/%d (%.0f%%)   started new track: %d/%d (%.0f%%)"%(
                matches, total_decisions, 100*matches/total_decisions,
                new_tracks, total_decisions, 100*new_tracks/total_decisions ) )
        if new_tracks:
            print( "    of new tracks, likely resets (another track was nearby but missed threshold): %d/%d (%.0f%%)"%(
                near_misses, new_tracks, 100*near_misses/new_tracks ) )
    print()


def main():
    parser = argparse.ArgumentParser( description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter )
    parser.add_argument( '--sweep-alpha', type=float, nargs='+', default=None,
        help="position_smoothing values to compare (default: just your current betaconfig.py value)" )
    parser.add_argument( '--labels', nargs='+', default=None,
        help="restrict to these labels (default: everything currently in items_to_censor)" )
    parser.add_argument( '--picture-sizes', type=int, nargs='+', default=None,
        help="analyse exactly these sizes instead of discovering what is on disk. By default every tool covers EVERY model configuration that has written detections (betautils_cache_paths.configurations_to_analyse), so a variant you ran last night cannot go unanalysed because config has since moved on. Use this only to force a shape that discovery would not produce - for instance a backend deliberately run at several sizes at once." )
    parser.add_argument( '--video-censor-fps', type=float, default=float( betaconfig.video_censor_fps ),
        help="must match the video_censor_fps your real caches were built with (default: betaconfig.py's current value)" )
    parser.add_argument( '--include-preview', action='store_true',
        help="also use preview-slice caches for any video that has no real (non-preview) cache yet - see this "
             "script's docstring CAVEAT before trusting numbers this produces: safe for backend-vs-backend "
             "comparison on the same slice, not safe as an absolute/full-video number." )
    parser.add_argument( '--variants', nargs='+', default=None,
        help="restrict analysis to these model variants, e.g. --variants 640m. Backend "
             "alone cannot express this - 320n and 640m are both nudenet_v3 - so this is "
             "the knob for focusing on one variant once it is the one you are tuning. It "
             "only narrows which cached configurations are READ; nothing on disk changes, "
             "so widening it again later needs no re-run" )
    args = parser.parse_args()

    labels = args.labels or list( betaconfig.items_to_censor )
    global_min_prob = getattr( betaconfig, 'global_min_prob', betaconst.global_min_prob )

    smooth_boxes_fn, apply_suppression_fn = _load_real_pipeline_functions()

    alphas = args.sweep_alpha or [ betaconfig.default_position_smoothing ]

    if args.include_preview:
        print( "!!! --include-preview is on: videos with no real (non-preview) cache will use a preview-slice "
               "cache instead. CAVEAT: a preview slice is a short, boundary-truncated window starting mid-video - "
               "absolute counts below (total resets, gap durations 'over the video') reflect that narrow window, "
               "NOT real full-video behavior, and are NOT comparable to a real-run number or across videos of "
               "different real lengths. What IS safe to read: a same-slice, backend-vs-backend comparison, since "
               "both backends' preview caches (when both exist) cover the exact same seconds of the exact same "
               "video. Videos marked '[PREVIEW SLICE]' below used preview data; unmarked videos used a real cache. !!!" )
        print()

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
        if not videos:
            print( "  no real (non-preview) detection caches found for this backend at "
                   "picture_sizes=%s fps=%s min_prob=%.3f - skipping (run betatv.py with "
                   "detector_backend['selected']=%r on real footage first to get data here%s)."%(
                       sizes, args.video_censor_fps, global_min_prob, backend_name,
                       ", or check the preview-cache warnings above if --include-preview was used" if args.include_preview else " (or pass --include-preview to also try preview-slice caches)" ) )
            print()
            continue
        any_backend_had_videos = True

        num_preview = len( preview_used_for )
        if num_preview:
            print( "found %d video(s) with usable caches for this config (%d real, %d preview-slice)"%(
                len(videos), len(videos)-num_preview, num_preview) )
            print( "  preview-slice video(s) (see CAVEAT above - stats below MIX real and preview data across "
                   "videos, so treat totals as approximate; only a backend-vs-backend diff on the SAME video is "
                   "solid): %s"%( ", ".join( "%s (%s)"%(h, preview_used_for[h]) for h in sorted(preview_used_for) ) ) )
        else:
            print( "found %d video(s) with complete real detection caches for this config"%(len(videos)) )
        hash_to_path = build_hash_to_video_path( videos.keys() )
        missing = set(videos.keys()) - set(hash_to_path.keys())
        if missing:
            print( "  (%d of those source video file(s) could not be found under %s or %s - probably moved/deleted "
                   "since being censored; they'll be skipped)"%(
                       len(missing), betaconst.video_path_uncensored, betaconst.video_path_source_backup ) )
        print()

        for alpha in alphas:
            result = run_one_alpha( alpha, videos, hash_to_path, smooth_boxes_fn,
                                    apply_suppression_fn, set(labels), backend_name )
            print_alpha_result( alpha, result, labels )

    if not any_backend_had_videos:
        print( "no %sdetection caches found for any of (%s) at fps=%s min_prob=%.3f - "
               "run betatv.py on real footage first%s."%(
                   "" if args.include_preview else "real (non-preview) ",
                   bu_cache.describe_configurations( configurations ),
                   args.video_censor_fps, global_min_prob,
                   "" if args.include_preview else " (not just --preview on), or pass --include-preview to also try preview-slice caches" ) )
        sys.exit(1)

    print( "reading this: 'smoothed frame-to-frame jump' is the actual visible motion of the censor box between "
           "consecutive real detections - that's the real jitter number, and it's the one to compare across alpha "
           "values. 'started new track' happening often, especially with a high 'likely resets' fraction, means "
           "some of what you're seeing is track continuity breaking, not EMA noise - lowering alpha further won't "
           "fix those; loosening the match distance/max_gap, raising video_censor_fps, or checking why the raw "
           "detections are jumping that far in the first place would be the next things to look at. Note each "
           "backend's numbers above come from that backend's OWN detections - a difference between backends here "
           "may reflect a real detection-quality difference (see analyze_suppression_pairs.py), not just alpha/"
           "smoothing behavior." )


if __name__ == '__main__':
    main()
