#!/usr/bin/env python3
"""
replay_tune.py - fast, no-recompute testing for the config knobs that sit
AFTER detection: interpolation_max_gap, class_suppression (margin/min_iou),
min_prob, time_safety, width/height_area_safety - anything under
item_overrides or class_suppression in betaconfig.py.

Why this is worth having as its own tool: all of the above operate purely
on the raw detections a video already produced - they don't change what
the neural net sees, only how those cached results get filtered
(class_suppression, min_prob) and turned into tracked, time-padded boxes
(smooth_boxes: interpolation, time_safety). picture_sizes and
video_censor_fps are different - they change what the net actually looks
at, so testing them means re-running real detection (tune_sweep.sh) and
then watching the result (no way around that). But interpolation_max_gap/
class_suppression/min_prob/time_safety/area_safety can be replayed against
a detection cache that's ALREADY been computed once, near-instantly, for
as many candidate values as you want - no GPU time, no waiting.

This uses the REAL smooth_boxes/apply_class_suppression functions from
betatv.py (extracted from its actual source at runtime, not
reimplemented - see _load_real_pipeline_functions below) and the real
process_raw_box from betautils_censor.py, so results match what betatv.py
would actually do, not an approximation of it.

What it can and can't tell you: it gives you exact counts (how many raw
detections survived suppression, how many tracked boxes came out, how
many were interpolated, which suppression rules fired how often) for the
CURRENT config vs one or more variants you define - a fast way to narrow
down which candidate values are worth actually watching, and to catch
obviously-wrong changes (e.g. a suppression tweak that suddenly kills 90%
of your vulva detections) without opening a video player. It can NOT tell
you whether a tracked box's smoothed/interpolated POSITION visually looks
right - that's still a "watch the actual preview output" question, same
as picture_sizes/video_censor_fps. Use this to narrow down 1-2 promising
variants, then set them for real in betaconfig.py and run a normal
preview to look at them.

Prerequisite: a detection cache must already exist for the video/
picture_sizes/video_censor_fps/preview-start-seconds combo you want to
test (i.e. you've already run betatv.py --preview on with those settings
at least once, e.g. via tune_sweep.sh or a plain CLI run). This script
only READS that cache - it never runs the neural net itself.

Usage:
    # just show current-config counts for one video (no variants)
    python3 replay_tune.py --video "myfile.mp4" --preview-start-seconds 450

    # compare current config against variants defined in a JSON file
    python3 replay_tune.py --video "myfile.mp4" --preview-start-seconds 450 \\
        --variants replay_variants_example.json

Variants file format (see replay_variants_example.json for a real starting
point covering both interpolation_max_gap and class_suppression):
[
  {
    "name": "vulva gap=1.6",
    "item_overrides": { "exposed_vulva": { "interpolation_max_gap": 1.6 } }
  },
  {
    "name": "breast/face_femme suppression loosened",
    "class_suppression": { "exposed_breast": [
        { "suppressed_by": "covered_breast", "margin": 0.13, "min_iou": 0.55 },
        { "suppressed_by": "face_femme", "margin": 0.10, "min_iou": 0.20 },
        { "suppressed_by": "face_masc",  "margin": 0.10, "min_iou": 0.20 }
    ] }
  }
]
"item_overrides" in a variant patches ONLY the keys you list, on top of
the item's current real config (other keys - censor_style, min_prob,
area_safety, etc. - stay whatever betaconfig.py already has, unless you
list them too). "class_suppression" in a variant REPLACES that label's
entire rule list wholesale (list every rule you want in effect for that
label, not just the one you're changing) - suppression rules are
positional/order-independent per label, so partial-merge doesn't make
sense the way it does for item_overrides.
"""

import argparse
import copy
import json
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
import betautils_video as bu_video
import betautils_cache_paths as bu_cache
from betautils_cache_paths import load_cached_raw_boxes


def _load_real_pipeline_functions():
    """
    The real tracking and suppression functions, imported directly.

    Until 2.1 these lived inside betatv.py, which this tool could not
    import (betatv had side effects at import time), so it read betatv's
    SOURCE TEXT, regex-matched the two function bodies plus the import
    block they depended on, and exec'd them. That worked, and it was
    fragile in exactly the way this project keeps getting bitten by: the
    extraction silently depended on those two functions staying adjacent
    and on the regex anchor below them never moving.

    They are a module now, so this is an import.

    Returns:
        A (smooth_boxes, apply_class_suppression) pair - the same
        objects betatv.py itself calls.
    """
    import betautils_track as bu_track
    return bu_track.smooth_boxes, bu_track.apply_class_suppression


# box_hash_path_for/load_cached_raw_boxes moved to betautils_cache_paths.py
# (shared with analyze_jitter.py/batch_ab_test.py/etc - see that module's
# docstring for why) - imported as bu_cache above. backend_name is now an
# explicit arg (resolved once via bu_detector.selected_backend_name() at
# each call site below) rather than derived inside the path function
# itself - the same fix analyze_style_flicker.py needed after a real bug
# where deriving it internally let a stale/wrong backend silently leak in.
def box_hash_path_for( file_hash, size, fps, min_prob, cache_suffix ):
    # thin wrapper kept for this file's own test/back-compat call shape -
    # resolves backend_name here so callers below don't have to change.
    return bu_cache.box_hash_path_for(
        file_hash, size, fps, min_prob, bu_detector.selected_backend_name(), cache_suffix )


def deep_merge_item_overrides( base_overrides, patch ):
    merged = copy.deepcopy( base_overrides )
    for label, keys in patch.items():
        merged.setdefault( label, {} )
        merged[label] = dict( merged[label] )
        merged[label].update( keys )
    return( merged )


def run_variant( flat_raw, vid_w, vid_h, smooth_boxes_fn, apply_suppression_fn,
                  item_overrides_patch=None, class_suppression_patch=None ):
    # monkeypatches betaconfig IN MEMORY for the duration of this one
    # variant, then always restores it in the finally block - never writes
    # to betaconfig.py, and a crash mid-variant can't leave the real
    # config mutated for whatever runs next.
    #
    # class_suppression is patched into the CURRENTLY SELECTED backend's
    # own detector_backend[<name>]['class_suppression'] block, not the
    # flat top-level betaconfig.class_suppression - apply_class_suppression
    # resolves per-backend now (see betautils_detector.get_class_suppression),
    # checking that block FIRST, so patching only the top-level attribute
    # would be silently ignored for any backend that has its own block set
    # (both registered backends do, even nudenet_v3's deliberately-empty
    # one) - same fix shape as betautils_cli.py's --nn-batch-size override.
    orig_item_overrides = betaconfig.item_overrides
    backend_name = bu_detector.selected_backend_name()
    backend_setting = getattr( betaconfig, 'detector_backend', None )
    orig_backend_suppression = None
    had_backend_suppression_key = False
    if backend_setting is not None and backend_name in backend_setting:
        had_backend_suppression_key = 'class_suppression' in backend_setting[ backend_name ]
        orig_backend_suppression = backend_setting[ backend_name ].get( 'class_suppression' )
    else:
        # No block for this backend. A class_suppression patch cannot be
        # applied at all in that state (see the raise below); these are
        # kept only so the finally block has defined names.
        orig_flat_suppression = None
        had_flat_suppression_attr = False
    try:
        if item_overrides_patch:
            betaconfig.item_overrides = deep_merge_item_overrides( orig_item_overrides, item_overrides_patch )
        if class_suppression_patch:
            base_suppression = bu_detector.get_class_suppression( backend_name )
            merged_suppression = copy.deepcopy( base_suppression )
            merged_suppression.update( class_suppression_patch )
            if backend_setting is not None and backend_name in backend_setting:
                backend_setting[ backend_name ][ 'class_suppression' ] = merged_suppression
            else:
                # The flat betaconfig.class_suppression fallback was removed
                # in 2.5, so writing it here would patch nothing and every
                # variant would silently replay identical results. Fail
                # loudly instead of reporting a meaningless comparison.
                raise RuntimeError(
                    "cannot apply a class_suppression patch: betaconfig.detector_backend has no "
                    "block for backend %r. Suppression rules are per-backend as of 2.5, so there "
                    "is nowhere to put the patch."%( backend_name, ) )

        # Memoised views of betaconfig are stale the moment this
        # function patches it, and process_raw_box reads one of them.
        import betautils_config as bu_config
        bu_config.invalidate_config_caches()

        raw_copy = copy.deepcopy( flat_raw )
        surviving, suppression_counts = apply_suppression_fn( raw_copy )
        # apply_class_suppression reports {'total', 'renderable'} per rule
        # now; this report compares totals.
        suppression_counts = { rule: entry['total'] if isinstance( entry, dict ) else entry
                               for rule, entry in suppression_counts.items() }

        label_counts = {}
        for r in surviving:
            lbl = r['class_id']
            label_counts[lbl] = label_counts.get( lbl, 0 ) + 1

        boxes = []
        for r in surviving:
            res = bu_censor.process_raw_box( r, vid_w, vid_h )
            if res:
                boxes.append( res )

        tracked_boxes, smoothing_stats = smooth_boxes_fn( boxes )

        return( {
            'raw_detections': len( flat_raw ),
            'surviving_after_suppression': len( surviving ),
            'suppression_counts': suppression_counts,
            'label_counts': label_counts,
            'tracked_boxes': len( tracked_boxes ),
            'interpolated_boxes': smoothing_stats['interpolated'],
        } )
    finally:
        betaconfig.item_overrides = orig_item_overrides
        import betautils_config as bu_config
        bu_config.invalidate_config_caches()
        if backend_setting is not None and backend_name in backend_setting:
            if had_backend_suppression_key:
                backend_setting[ backend_name ][ 'class_suppression' ] = orig_backend_suppression
            else:
                backend_setting[ backend_name ].pop( 'class_suppression', None )
        # No flat-attribute branch to unwind: a patch without a backend
        # block raises above rather than writing betaconfig directly.


def print_result( label, result ):
    print( "  %s"%(label) )
    print( "    raw detections in this slice: %d"%(result['raw_detections']) )
    print( "    survived class_suppression:   %d  (suppression fired: %s)"%(
        result['surviving_after_suppression'], result['suppression_counts'] or '{} (nothing fired)' ) )
    print( "    label counts:                 %s"%(result['label_counts']) )
    print( "    tracked boxes after smoothing: %d  (%d interpolated)"%(
        result['tracked_boxes'], result['interpolated_boxes'] ) )


def print_delta( baseline, variant ):
    def dict_delta( base_d, var_d ):
        keys = sorted( set(base_d.keys()) | set(var_d.keys()) )
        parts = []
        for k in keys:
            b, v = base_d.get(k,0), var_d.get(k,0)
            if b != v:
                parts.append( "%s: %d->%d (%+d)"%(k, b, v, v-b) )
        return( "; ".join(parts) if parts else "no change" )

    print( "    vs baseline: survived %+d, tracked_boxes %+d, interpolated %+d"%(
        variant['surviving_after_suppression'] - baseline['surviving_after_suppression'],
        variant['tracked_boxes'] - baseline['tracked_boxes'],
        variant['interpolated_boxes'] - baseline['interpolated_boxes'] ) )
    print( "    label_counts delta: %s"%(dict_delta(baseline['label_counts'], variant['label_counts'])) )
    print( "    suppression_counts delta: %s"%(dict_delta(baseline['suppression_counts'], variant['suppression_counts'])) )


def main():
    parser = argparse.ArgumentParser( description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter )
    parser.add_argument( '--video', required=True, help="video filename (as it appears under betaconst.video_path_uncensored) to replay" )
    parser.add_argument( '--picture-sizes', type=int, nargs='+', default=None,
        help="picture_sizes the detection caches were built with. Default: resolved for the SELECTED backend (betautils_detector.get_picture_sizes), which is what a real run actually used - not the shared betaconfig.picture_sizes, which a backend with its own or native sizes never uses." )
    parser.add_argument( '--video-censor-fps', type=float, default=float( betaconfig.video_censor_fps ),
        help="must match the video_censor_fps the detection cache was built with (default: betaconfig.py's current value)" )
    parser.add_argument( '--preview-start-seconds', type=float, required=True,
        help="the preview slice offset the detection cache was built with - required, since there's no sensible default that would reliably hit an existing cache" )
    parser.add_argument( '--preview-seconds', type=float, default=float( os.environ.get( 'PREVIEW_SECONDS', getattr( betaconfig, 'preview_max_seconds', 20.0 ) ) ),
        help="preview_max_seconds the detection cache was built with (default: betaconfig.py's preview_max_seconds, or $PREVIEW_SECONDS if set) - needed to correctly predict the cache suffix for a video shorter than --preview-start-seconds, which falls back to a whole-file '-preview' suffix instead of '-preview@<offset>s' (see betautils_video.resolve_preview_slice)" )
    parser.add_argument( '--variants', help="path to a JSON file of variants to compare against the current config (see this script's docstring for the format). If omitted, just reports the current config's counts." )
    parser.add_argument( '--backend', choices=sorted( bu_detector._BACKENDS.keys() ), default=None,
        help="which registered backend to replay against, overriding betaconfig.detector_backend['selected'] "
             "for this process only - betaconfig.py itself is never touched. Equivalent to setting "
             "BETASUITE_DETECTOR_BACKEND_OVERRIDE before running this script (same mechanism "
             "compare_backends_orchestrator.py uses); this flag is just a more convenient way to set that "
             "same override for a single one-off replay_tune.py invocation, without needing a separate env "
             "var export. If both this flag and the env var are set, this flag wins for this process (it sets "
             "the env var itself, before any backend-dependent code runs)." )
    args = parser.parse_args()

    # Resolve per backend rather than from the shared betaconfig list:
    # since v2.1 a backend can declare its own picture_sizes, or inherit
    # its model's native sizes, so the shared list is frequently not the
    # size anything was actually detected at.
    args.picture_sizes = bu_cache.resolve_picture_sizes(
        args.picture_sizes, bu_detector.selected_backend_name() )

    if args.backend:
        os.environ[ 'BETASUITE_DETECTOR_BACKEND_OVERRIDE' ] = args.backend

    # printed unconditionally (not just when --backend is passed) since
    # this was the actual source of confusion that motivated adding
    # --backend in the first place: replay_tune.py silently replaying
    # against whatever betaconfig.detector_backend['selected'] happened
    # to be, with no output saying so, while variants/cache expectations
    # were being reasoned about as if it were a different backend.
    print( "active backend for this replay: %s%s"%(
        bu_detector.selected_backend_name(),
        " (from --backend)" if args.backend else (
            " (from $BETASUITE_DETECTOR_BACKEND_OVERRIDE)" if os.environ.get( 'BETASUITE_DETECTOR_BACKEND_OVERRIDE' )
            else " (from betaconfig.py's detector_backend['selected'])" ) ) )

    video_path = os.path.join( betaconst.video_path_uncensored, args.video )
    if not os.path.exists( video_path ):
        print( "no such file under %s: %s"%(betaconst.video_path_uncensored, args.video) )
        sys.exit(1)

    cap = cv2.VideoCapture( video_path )
    vid_w = int( cap.get( cv2.CAP_PROP_FRAME_WIDTH ) )
    vid_h = int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT ) )
    vid_fps = cap.get( cv2.CAP_PROP_FPS )
    num_frames = cap.get( cv2.CAP_PROP_FRAME_COUNT )
    cap.release()
    if not vid_w or not vid_h:
        print( "couldn't read frame dimensions from %s"%(video_path) )
        sys.exit(1)

    global_min_prob = getattr( betaconfig, 'global_min_prob', betaconst.global_min_prob )
    # a video shorter than --preview-start-seconds falls back (in betatv.py)
    # to processing the whole file with a plain '-preview' suffix instead of
    # '-preview@<offset>s' - resolve_preview_slice replicates that exactly
    # so this always looks for the cache file betatv.py actually wrote,
    # not the one a naive '-preview@<start>s' guess would expect
    preview_offset_seconds, preview_window_unlimited = bu_video.resolve_preview_slice(
        vid_fps, num_frames, True, args.preview_seconds, args.preview_start_seconds )
    cache_suffix = bu_video.preview_cache_suffix( True, preview_offset_seconds )
    if preview_window_unlimited:
        print( "note: --preview-start-seconds %.1fs is past the end of %s - this video's cache was built from the "
               "WHOLE file (suffix '-preview', not '-preview@%.1fs')"%(
            args.preview_start_seconds, args.video, args.preview_start_seconds ) )
    file_hash = bu_hash.md5_for_file( video_path, 16 )

    flat_raw, missing = load_cached_raw_boxes(
        file_hash, args.picture_sizes, args.video_censor_fps, global_min_prob, bu_detector.selected_backend_name(), cache_suffix )
    if missing:
        print( "missing detection cache for size(s) %s - run a preview pass with these exact settings first:"%(
            [m[0] for m in missing] ) )
        print( "    python3 betatv.py --preview on --preview-start-seconds %s --picture-sizes %s --video-censor-fps %s"%(
            args.preview_start_seconds, " ".join(map(str,args.picture_sizes)), args.video_censor_fps ) )
        for size, path in missing:
            print( "    (expected: %s)"%(path) )
        sys.exit(1)

    print( "loaded %d cached raw detections for %s (picture_sizes=%s fps=%s preview_start=%s)"%(
        len(flat_raw), args.video, args.picture_sizes, args.video_censor_fps, args.preview_start_seconds ) )
    print()

    smooth_boxes_fn, apply_suppression_fn = _load_real_pipeline_functions()

    baseline = run_variant( flat_raw, vid_w, vid_h, smooth_boxes_fn, apply_suppression_fn )
    print( "=== baseline (current betaconfig.py, unmodified) ===" )
    print_result( "baseline", baseline )

    if not args.variants:
        return

    with open( args.variants, 'r' ) as f:
        variants = json.load( f )

    print()
    print( "=== variants ===" )
    for v in variants:
        name = v.get( 'name', '(unnamed variant)' )
        result = run_variant(
            flat_raw, vid_w, vid_h, smooth_boxes_fn, apply_suppression_fn,
            item_overrides_patch=v.get( 'item_overrides' ),
            class_suppression_patch=v.get( 'class_suppression' ),
        )
        print()
        print_result( name, result )
        print_delta( baseline, result )


if __name__ == '__main__':
    main()
