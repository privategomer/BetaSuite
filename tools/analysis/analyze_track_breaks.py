#!/usr/bin/env python3
"""
analyze_track_breaks.py - root-causes the "likely reset" events
analyze_jitter.py flags (a new track started when another track was nearby
but missed the match threshold), tests what loosening track_max_gap and/or
match_distance_multiplier would actually do to your real cached footage,
and checks whether loosening match_distance_multiplier risks a different
failure mode: two DIFFERENT simultaneous same-label instances (e.g. two
people) getting merged into one track by the nearest-neighbor greedy
assignment ("contention").

track_max_gap and match_distance_multiplier are now real item_overrides
knobs in betaconfig.py (see smooth_boxes in betatv.py) - this tool no
longer patches synthetic multipliers into a source-level copy of the
matching formulas the way an earlier version did. It sets the REAL
per-label overrides via betaconfig.item_overrides before each sweep value
and re-runs the real, unmodified smooth_boxes/apply_class_suppression
against your cached detections - betatv.py itself is still never touched,
and betaconfig.py on disk is never touched either (only the in-memory
betaconfig module is mutated for the duration of one combo, then
restored) - so this remains a read-only replay against the cache, just
using the real config surface instead of a parallel testing-only one.

This tool answers three questions on your actual cached detections:

  1. For every reset event, WHY did it reset - was the nearest live track
     too far away (distance-blocked), did too much time pass since that
     track's last real hit (gap-blocked), both, or was there genuinely no
     nearby track at all (a real fresh appearance, not fixable by loosening
     either threshold)?
  2. If you loosened track_max_gap and/or match_distance_multiplier, how
     many of those resets would the REAL matching code actually have
     avoided?
  3. NEW: if you loosened match_distance_multiplier, how often would two
     DIFFERENT simultaneous same-label boxes both become viable candidates
     for the SAME existing track in the same frame ("contention")? The
     greedy nearest-first assignment only ever awards the track to one of
     them, but a contention event where the two candidates' distances are
     close together is exactly the shape of a real mis-merge risk (a
     confident, close-call assignment is more likely to have picked the
     wrong one of two genuinely different instances) - whereas a contention
     event where one candidate is much closer than the other is low-risk
     (there's little ambiguity about which one is the real match). This is
     the concrete evidence the match-distance-loosening risk hypothesis
     needed before raising its default was safe to consider.

Prerequisite: same as analyze_jitter.py - real (non-preview) detection
caches must already exist for each backend's resolved picture_sizes (per
backend, not the shared betaconfig.picture_sizes) at your video_censor_fps.

--include-preview (added 2026-09-16): same opt-in/same CAVEAT as
analyze_jitter.py's flag of the same name - falls back to a preview-slice
cache for any video with no real cache yet. Reset counts from a preview
slice are NOT comparable to a real-run count (short, boundary-truncated
window); a same-slice backend-vs-backend comparison is the safe read.

Usage:
    python3 analyze_track_breaks.py
    python3 analyze_track_breaks.py --match-distance-multipliers 1.0 1.5 2.0 3.0
    python3 analyze_track_breaks.py --track-max-gap-multipliers 1.0 2.0 3.0 5.0
    python3 analyze_track_breaks.py --labels exposed_breast exposed_vulva
    python3 analyze_track_breaks.py --include-preview
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
_contention_events = []


def _reset_events():
    _match_events.clear()
    _new_track_events.clear()
    _contention_events.clear()


def _record_match_event( label, ti, dist, b ):
    # fires right after the real code decides this detection continues
    # track `ti`
    _match_events.append( { 'label': label, 'dist': dist } )


def _record_new_track_event( label, tracks, ti, b, max_gap, match_distance_multiplier ):
    # fires right after the real code appends a brand new track (this
    # detection did NOT match any existing track under the CURRENT
    # per-label overrides). For every OTHER currently-live track, work out
    # exactly why it didn't match: too far (distance-blocked), too much
    # time passed (gap-blocked), both, or it wasn't a candidate at all
    # because it's simply not the nearest. This mirrors the real
    # candidate-finding logic in smooth_boxes exactly (same distance/gap
    # values this combo is actually running with) so the classification is
    # trustworthy, not a separate guess.
    cx = b['x'] + b['w']/2
    cy = b['y'] + b['h']/2
    best = None  # (dist, gap, threshold, max_gap) for the nearest-in-time-and-space live track
    for prev in tracks:
        if prev is tracks[ti]:
            continue
        gap = b['start'] - prev['end']
        pcx = prev['x'] + prev['w']/2
        pcy = prev['y'] + prev['h']/2
        dist = ( (cx-pcx)**2 + (cy-pcy)**2 ) ** 0.5
        threshold = match_distance_multiplier * max( prev['w'], prev['h'], b['w'], b['h'] )
        if best is None or dist < best[0]:
            best = ( dist, gap, threshold, max_gap )

    if best is None:
        reason = 'no-nearby-track'  # genuinely nothing else live for this label
    else:
        dist, gap, threshold, mg = best
        dist_blocked = dist >= threshold
        gap_blocked = gap > mg
        if dist_blocked and gap_blocked:
            reason = 'both-blocked'
        elif dist_blocked:
            reason = 'distance-blocked'
        elif gap_blocked:
            reason = 'gap-blocked'
        else:
            reason = 'unexplained'  # shouldn't happen - would mean the real code should have matched this
            # (known, disclosed gap: the real algorithm does global greedy
            # assignment across every box/track in the frame - a detection
            # can also fail to match because a DIFFERENT box claimed the
            # nearest track first (contention, see _record_frame_candidates
            # below), which this single-nearest-track check doesn't model)

    _new_track_events.append( {
        'label': label,
        'reason': reason,
        'nearest_dist': best[0] if best else None,
        'nearest_gap': best[1] if best else None,
        'score': b.get('score'),
    } )


def _record_frame_candidates( label, candidates ):
    # fires once per frame, right after the real candidate list is built
    # (every (box, track) pairing close enough in time+space to plausibly
    # match) but BEFORE the greedy nearest-first assignment picks winners.
    # A track (ti) that shows up as a candidate for 2+ DIFFERENT boxes in
    # the same frame is "contested" - the real code will award it to
    # whichever candidate is closest, but if a second, different real
    # instance was also close enough to be a candidate, that's the
    # concrete shape of the cross-instance mis-merge risk: loosening
    # match_distance_multiplier only ever makes MORE boxes qualify as
    # candidates, so it can only ever create contention, never remove it.
    by_ti = {}
    for dist, b, ti in candidates:
        by_ti.setdefault( ti, [] ).append( ( dist, id(b) ) )
    for ti, entries in by_ti.items():
        distinct_boxes = { bid for _, bid in entries }
        if len( distinct_boxes ) < 2:
            continue
        dists = sorted( d for d, _ in entries )
        closest, second_closest = dists[0], dists[1]
        # "risky" = the two closest competing candidates are within 25% of
        # each other - a close call the greedy assignment could plausibly
        # get wrong if they're actually two different real instances,
        # versus one candidate being clearly, unambiguously nearest
        risky = ( second_closest > 0 and (second_closest - closest) / second_closest < 0.25 )
        _contention_events.append( {
            'label': label,
            'num_candidates': len( distinct_boxes ),
            'closest_dist': closest,
            'second_closest_dist': second_closest,
            'risky': risky,
        } )


class _TrackBreakObserver( bu_track.TrackingObserver ):
    """
    Records matches, new tracks, and the full per-frame candidate pool.

    The candidate pool is the one this tool cannot get any other way:
    contention (one track being a plausible match for two different
    boxes in the same frame) is only visible BEFORE the greedy
    nearest-first assignment resolves it.
    """

    def on_match( self, label, track_index, distance, box, tracks, settings ):
        _record_match_event( label, track_index, distance, box )

    def on_new_track( self, label, tracks, track_index, box, settings ):
        _record_new_track_event( label, tracks, track_index, box,
                                 settings.max_gap, settings.match_distance_multiplier )

    def on_frame_candidates( self, label, candidates, tracks, settings ):
        _record_frame_candidates( label, candidates )


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
             'observer': _TrackBreakObserver() }



# box_hash_path_for/discover_full_run_videos/_walk_and_hash_for/
# build_hash_to_video_path/fmt_stats moved to betautils_cache_paths.py
# (shared with analyze_jitter.py/analyze_style_flicker.py/etc - see
# that module's docstring for why) - imported as bu_cache below.


def _real_baseline_track_max_gap( label, backend_name ):
    """
    The effective track_max_gap smooth_boxes would use for this label.

    Mirrors smooth_boxes's own auto-derivation exactly, so a
    gap-multiplier sweep scales the REAL effective baseline for this
    label (which varies per label - vulva's confirmed 1.2s
    interpolation_max_gap already exceeds the 2.0/fps formula, breast's
    does not) rather than a single global number that does not match
    what smooth_boxes actually uses.

    Resolved per backend: every tracking key moved into each backend's
    own detector_backend[<name>]['item_overrides'] block, so reading the
    shared betaconfig.item_overrides directly would silently find
    nothing and fall back to the global default.
    """
    overrides = bu_detector.get_item_overrides( label, backend_name )
    if 'track_max_gap' in overrides:
        return overrides['track_max_gap']
    default_track_max_gap = 2.0 / betaconfig.video_censor_fps
    interpolation_max_gap = overrides.get( 'interpolation_max_gap', betaconfig.default_interpolation_max_gap )
    return max( default_track_max_gap, interpolation_max_gap )


@contextlib.contextmanager
def _swept_overrides( backend_name, labels, match_mult, gap_mult ):
    """
    Temporarily apply one sweep point to the BACKEND's override block.

    Why not betaconfig.item_overrides, which is what this did before
    v2.1: get_item_overrides() merges a backend's own block on top of
    the shared one for every backend-tunable key, and both
    match_distance_multiplier and track_max_gap are backend-tunable. So
    a sweep that wrote to the shared dict was silently overwritten by
    the backend block for exactly the two keys it was sweeping - every
    row of the sweep ran with identical settings and the table looked
    flat for a reason that had nothing to do with the footage.

    Analogy: the shared dict is a note left on the fridge; the backend
    block is the person standing at the stove. Whatever the note says,
    the cook does what the cook does.

    Both multipliers scale each label's REAL current value rather than
    replacing it, so multiplier 1.0 reproduces current behaviour
    exactly - which is what makes it usable as the sweep's baseline.

    The original block is restored on the way out whatever happens,
    including on Ctrl-C.
    """
    backend_setting = betaconfig.detector_backend.setdefault( backend_name, {} )
    saved = copy.deepcopy( backend_setting.get( 'item_overrides', {} ) )
    swept = copy.deepcopy( saved )
    for label in labels:
        effective = bu_detector.get_item_overrides( label, backend_name )
        label_block = dict( swept.get( label, {} ) )
        label_block['match_distance_multiplier'] = (
            match_mult * effective.get( 'match_distance_multiplier', 1.0 ) )
        label_block['track_max_gap'] = _real_baseline_track_max_gap( label, backend_name ) * gap_mult
        swept[label] = label_block
    backend_setting['item_overrides'] = swept
    bu_config.invalidate_config_caches()
    try:
        yield
    finally:
        backend_setting['item_overrides'] = saved
        bu_config.invalidate_config_caches()


def run_one_combo( match_mult, gap_mult, videos, hash_to_path, ns, labels_filter,
                   backend_name ):
    """
    Replay every cached video once at one (match, gap) multiplier pair.

    Args:
        match_mult: Scales each label's real match_distance_multiplier.
        gap_mult: Scales each label's real effective track_max_gap.
        videos: {file_hash: [cache_path, ...]} from discover_full_run_videos.
        hash_to_path: {file_hash: source video path}.
        ns: The instrumented pipeline namespace from
            _load_instrumented_pipeline().
        labels_filter: Labels to sweep and report on, or a falsy value
            for everything in items_to_censor.
        backend_name: The backend whose caches these are. Pinned, never
            defaulted: the tool loops over every registered backend, and
            resolving to the *selected* one would analyse one backend's
            detections with another backend's tracking settings.
    """
    smooth_boxes_fn = ns['smooth_boxes']
    apply_suppression_fn = ns['apply_class_suppression']

    labels_to_sweep = labels_filter or set( betaconfig.items_to_censor )
    with _swept_overrides( backend_name, labels_to_sweep, match_mult, gap_mult ):
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
                if res and ( not labels_filter or res['label'] in labels_filter ):
                    boxes.append( res )
            if not boxes:
                continue

            with bu_track.tracking_observer( ns['observer'] ):
                smooth_boxes_fn( boxes, backend_name=backend_name )

    match_dists_by_label = {}
    for ev in _match_events:
        match_dists_by_label.setdefault( ev['label'], [] ).append( ev['dist'] )

    new_track_by_label = {}
    reason_counts_by_label = {}
    for ev in _new_track_events:
        new_track_by_label[ ev['label'] ] = new_track_by_label.get( ev['label'], 0 ) + 1
        reason_counts_by_label.setdefault( ev['label'], {} )
        reason_counts_by_label[ ev['label'] ][ ev['reason'] ] = reason_counts_by_label[ ev['label'] ].get( ev['reason'], 0 ) + 1

    contention_by_label = {}
    for ev in _contention_events:
        stats = contention_by_label.setdefault( ev['label'], { 'total': 0, 'risky': 0 } )
        stats['total'] += 1
        if ev['risky']:
            stats['risky'] += 1

    return {
        'match_dists_by_label': match_dists_by_label,
        'new_track_by_label': new_track_by_label,
        'reason_counts_by_label': reason_counts_by_label,
        'contention_by_label': contention_by_label,
        'dims_missing': dims_missing,
    }


def print_combo_result( match_mult, gap_mult, result, labels ):
    print( "=== match_distance_multiplier=%.2f  track_max_gap_multiplier=%.2f ==="%(match_mult, gap_mult) )
    if result['dims_missing']:
        print( "  (skipped %d video(s) - source file no longer found)"%(len(result['dims_missing'])) )
    for label in labels:
        matches = len( result['match_dists_by_label'].get( label, [] ) )
        new_tracks = result['new_track_by_label'].get( label, 0 )
        total = matches + new_tracks
        contention = result['contention_by_label'].get( label )
        if not total and not contention:
            continue
        if total:
            print( "  %s: continued=%d/%d (%.0f%%)  new_track=%d/%d (%.0f%%)"%(
                label, matches, total, 100*matches/total, new_tracks, total, 100*new_tracks/total ) )
            reasons = result['reason_counts_by_label'].get( label, {} )
            if reasons:
                parts = ", ".join( "%s=%d"%(r,c) for r,c in sorted(reasons.items(), key=lambda x:-x[1]) )
                print( "    new-track reasons: %s"%(parts) )
        if contention:
            print( "    contention (2+ different boxes competing for the same track in one frame): "
                   "%d event(s), %d (%.0f%%) of them 'risky' (closest two candidates within 25%% of each other)"%(
                       contention['total'], contention['risky'],
                       100*contention['risky']/contention['total'] if contention['total'] else 0 ) )
    print()


def main():
    parser = argparse.ArgumentParser( description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter )
    parser.add_argument( '--match-distance-multipliers', type=float, nargs='+', default=[1.0, 1.5, 2.0, 3.0],
        help="match_distance_multiplier values to test (multiplied onto each label's real current value, default 1.0), track_max_gap held at its real multiplier=1.0 value for each of these (default: 1.0 1.5 2.0 3.0)" )
    parser.add_argument( '--track-max-gap-multipliers', type=float, nargs='+', default=[1.0, 2.0, 3.0, 5.0],
        help="track_max_gap multipliers to test (scales each label's own real auto-derived or overridden baseline), match_distance_multiplier held at its real multiplier=1.0 value for each of these (default: 1.0 2.0 3.0 5.0)" )
    parser.add_argument( '--labels', nargs='+', default=None,
        help="restrict to these labels (default: everything currently in items_to_censor)" )
    parser.add_argument( '--picture-sizes', type=int, nargs='+', default=None,
        help="analyse exactly these sizes instead of discovering what is on disk. By default every tool covers EVERY model configuration that has written detections (betautils_cache_paths.configurations_to_analyse), so a variant you ran last night cannot go unanalysed because config has since moved on." )
    parser.add_argument( '--video-censor-fps', type=float, default=float( betaconfig.video_censor_fps ) )
    parser.add_argument( '--include-preview', action='store_true',
        help="also use preview-slice caches for any video that has no real (non-preview) cache yet - see "
             "analyze_jitter.py's docstring CAVEAT (same tradeoffs apply here): safe for backend-vs-backend "
             "comparison on the same slice, not safe as an absolute/full-video reset count." )
    parser.add_argument( '--variants', nargs='+', default=None,
        help="restrict analysis to these model variants, e.g. --variants 640m. Backend "
             "alone cannot express this - 320n and 640m are both nudenet_v3 - so this is "
             "the knob for focusing on one variant once it is the one you are tuning. It "
             "only narrows which cached configurations are READ; nothing on disk changes, "
             "so widening it again later needs no re-run" )
    args = parser.parse_args()

    labels = args.labels or list( betaconfig.items_to_censor )
    global_min_prob = getattr( betaconfig, 'global_min_prob', betaconst.global_min_prob )

    ns = _load_instrumented_pipeline()

    if args.include_preview:
        print( "!!! --include-preview is on: videos with no real (non-preview) cache will use a preview-slice "
               "cache instead. A preview slice is short and boundary-truncated - reset COUNTS below reflect that "
               "narrow window, not real full-video behavior, and aren't comparable to a real-run count or across "
               "videos of different real lengths. Safe to read: a same-slice, backend-vs-backend comparison. !!!" )
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
        print( "=" * 70 )
        print( "configuration: %s"%(config.label) )
        print( "=" * 70 )
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
            print( "  preview-slice video(s): %s"%(
                ", ".join( "%s (%s)"%(h, preview_used_for[h]) for h in sorted(preview_used_for) ) ) )
        else:
            print( "found %d video(s) with complete real detection caches for this config"%(len(videos)) )
        hash_to_path = build_hash_to_video_path( videos.keys() )
        missing = set(videos.keys()) - set(hash_to_path.keys())
        if missing:
            print( "  (%d of those source video file(s) could not be found under %s or %s - skipped)"%(
                len(missing), betaconst.video_path_uncensored, betaconst.video_path_source_backup) )
        print()

        print( "############################################################" )
        print( "# baseline (multiplier=1.0 for both - this IS your real,    #" )
        print( "# current smooth_boxes behavior) with reset root-cause      #" )
        print( "# classification and contention check                       #" )
        print( "############################################################" )
        baseline = run_one_combo( 1.0, 1.0, videos, hash_to_path, ns, set(labels), backend_name )
        print_combo_result( 1.0, 1.0, baseline, labels )
        print( "reading the reason breakdown: 'distance-blocked' means the nearest live track was close enough in time but too far away - " )
        print( "loosening match_distance_multiplier alone would fix these. 'gap-blocked' is the reverse - close enough in space but too much time " )
        print( "passed - raising track_max_gap (or video_censor_fps) would fix these. 'both-blocked' needs both loosened together. " )
        print( "'no-nearby-track' means there was nothing else live for this label at all - a genuinely fresh appearance (or the model missed " )
        print( "several consecutive frames badly), not something either threshold can fix." )
        print()
        print( "reading contention: this is the real evidence for whether loosening match_distance_multiplier's default is safe. A contention " )
        print( "event means 2+ different boxes in one frame both qualified as candidates for the same existing track - the greedy assignment picks " )
        print( "the closest one, but if a 'risky' one (the two closest candidates were within 25% of each other), there's real ambiguity about " )
        print( "which one was actually the correct match. If risky contention stays near zero across the match-distance sweep below, loosening it " )
        print( "is probably safe for your footage. If it climbs meaningfully, that's the concrete downside the hypothesis predicted." )
        print()

        print( "############################################################" )
        print( "# match-distance sweep (track_max_gap held at its real value) #" )
        print( "############################################################" )
        for mult in args.match_distance_multipliers:
            if mult == 1.0:
                continue  # already shown above as the baseline
            result = run_one_combo( mult, 1.0, videos, hash_to_path, ns, set(labels), backend_name )
            print_combo_result( mult, 1.0, result, labels )

        print( "############################################################" )
        print( "# track_max_gap sweep (match-distance held at its real value) #" )
        print( "############################################################" )
        for mult in args.track_max_gap_multipliers:
            if mult == 1.0:
                continue  # already shown above as the baseline
            result = run_one_combo( 1.0, mult, videos, hash_to_path, ns, set(labels), backend_name )
            print_combo_result( 1.0, mult, result, labels )

        print( "how to read the sweeps: the deciding number is risky contention ADDED per track reset AVOIDED, against the baseline. A " )
        print( "change that trades a few risky assignments for each reset it fixes is worth making; one that trades dozens is not, however " )
        print( "good the reset count looks on its own. Compute it per label - the two labels here have repeatedly wanted opposite answers." )
        print()
        print( "what 'no-nearby-track' should do, and it differs per sweep. Across the MATCH-DISTANCE sweep it must stay flat: match " )
        print( "distance cannot change whether anything was live for this label, so movement there means something is wrong. Across the " )
        print( "TRACK_MAX_GAP sweep it should FALL, and that fall is the mechanism working, not a fault: a larger gap keeps tracks " )
        print( "eligible for longer, so fewer moments have nothing live to match against. Before 2.1.1 this tool told you a fall meant " )
        print( "'something's off' in both sweeps, which is wrong for the gap sweep and sent at least one reader looking for a bug." )
        print()

    if not any_backend_had_videos:
        print( "no %sdetection caches found for any of (%s) at fps=%s min_prob=%.3f - "
               "run betatv.py on real footage first%s."%(
                   "" if args.include_preview else "real (non-preview) ",
                   bu_cache.describe_configurations( configurations ),
                   args.video_censor_fps, global_min_prob,
                   "" if args.include_preview else " (not just --preview on), or pass --include-preview to also try preview-slice caches" ) )
        sys.exit(1)

    print( "note: each backend's numbers above come from that backend's OWN detections and tracking replay - a " )
    print( "difference in reset/contention behavior between backends may reflect a real detection-quality or " )
    print( "box-stability difference (see analyze_suppression_pairs.py / analyze_jitter.py), not just these " )
    print( "multiplier sweeps." )


if __name__ == '__main__':
    main()
