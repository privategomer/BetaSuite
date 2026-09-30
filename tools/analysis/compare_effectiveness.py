#!/usr/bin/env python3
"""
compare_effectiveness.py - answers "does a bigger picture_size / higher
video_censor_fps actually catch real exposures the other setting missed,
or does it just cost more time for basically the same coverage?" using
your own cached detections, instead of eyeballing raw label counts.

Why raw 'labels' counts (as printed by tune_sweep.sh/summarize_tune_sweep.py)
can't answer this: video_censor_fps controls how many times per second the
net samples the video. A single continuous exposure that lasts 3 real
seconds produces roughly 3x as many raw per-frame detections at fps=24 as
it does at fps=8 - NOT because fps=24 caught 3x more real content, just
because it sampled the same ongoing event 3x as often. So "more labels"
at a higher fps or bigger picture_size is expected even when detection
quality is IDENTICAL, and tells you nothing about whether the setting is
worth its extra processing time.

What this does instead: runs the real apply_class_suppression/
process_raw_box/smooth_boxes pipeline (same functions replay_tune.py
extracts from betatv.py - never reimplemented) against two ALREADY-
COMPUTED detection caches for the same video/preview slice (config A and
config B - typically the same video at two different picture_sizes, or
two different video_censor_fps), collapses each into discrete per-label
EVENTS (a continuous span of time a label was on-screen, not one row per
sampled frame), and compares event-for-event: does config B find any
continuous exposure that config A missed ENTIRELY (no temporally-
overlapping event of that label in A at all), and vice versa? That's a
real "did this setting buy me anything" signal a raw label count can't
give you - a bigger picture_size that's just resampling the exact same
handful of real events more precisely will show 100% event overlap even
if its raw label count is much higher.

What this still can't tell you: whether an "A-only" or "B-only" event is
a real catch or a false positive unique to that setting - that's still a
"go look at the actual preview output at that timestamp" question. Use
this to find WHERE to look (the timestamps of the unique events), not as
a substitute for looking.

Prerequisite: detection caches for both configs, same video, same
preview-start-seconds, must already exist (run tune_sweep.sh, or plain
betatv.py --preview on calls, with both configs first - this script only
reads the cache, it never runs the neural net).

Usage:
    # compare picture_size 1280 vs 2560, both at fps=8
    python3 compare_effectiveness.py --video "myfile.mp4" --preview-start-seconds 450 \\
        --sizes-a 1280 --fps-a 8 --sizes-b 2560 --fps-b 8

    # compare fps=8 vs fps=14, same picture_size
    python3 compare_effectiveness.py --video "myfile.mp4" --preview-start-seconds 450 \\
        --sizes-a 1280 --fps-a 8 --sizes-b 1280 --fps-b 14

    # multi-size combos work too (space-separated)
    python3 compare_effectiveness.py --video "myfile.mp4" --preview-start-seconds 450 \\
        --sizes-a 1280 --fps-a 8 --sizes-b 1280 2560 --fps-b 8
"""

import argparse
import copy
import os
import sys

import cv2

import sys as _sys, os as _os
_sys.path.insert( 0, _os.path.dirname( _os.path.dirname( _os.path.dirname( _os.path.abspath( __file__ ) ) ) ) )  # so 'import betaconst'/'betaconfig'/etc still resolve after this script moved into tools/analysis or tools/tuning

import betaconst
import betaconfig
import betautils_hash as bu_hash
import betautils_censor as bu_censor

import replay_tune as rt


def run_pipeline_for_config( flat_raw, vid_w, vid_h, fps, smooth_boxes_fn, apply_suppression_fn ):
    # smooth_boxes uses betaconfig.video_censor_fps directly (for
    # max_gap/frame_step - see its own comments), so that has to reflect
    # whichever config's fps we're currently processing, same monkeypatch-
    # then-restore pattern replay_tune.run_variant uses for item_overrides/
    # class_suppression.
    orig_fps = betaconfig.video_censor_fps
    try:
        betaconfig.video_censor_fps = fps
        raw_copy = copy.deepcopy( flat_raw )
        surviving, _ = apply_suppression_fn( raw_copy )
        boxes = []
        for r in surviving:
            res = bu_censor.process_raw_box( r, vid_w, vid_h )
            if res:
                boxes.append( res )
        tracked_boxes, _ = smooth_boxes_fn( boxes )
        return( tracked_boxes )
    finally:
        betaconfig.video_censor_fps = orig_fps


def boxes_to_events( tracked_boxes ):
    # Collapses a flat list of per-frame (+interpolated) censor boxes into
    # discrete per-label EVENTS: a continuous span of [start,end] time one
    # instance of a label was on-screen. Simplification: this merges ANY
    # two same-label boxes whose time spans touch, so two simultaneous
    # separate instances of the same label (e.g. two breasts at once)
    # collapse into one event rather than two - fine for "did this
    # setting catch this general moment in time", not precise for exact
    # instance counting. Good enough for the A-vs-B comparison this script
    # does; don't use event counts here as a substitute for label_counts
    # if you need per-instance precision.
    by_label = {}
    for b in tracked_boxes:
        by_label.setdefault( b['label'], [] ).append( b )

    events_by_label = {}
    for label, boxes in by_label.items():
        boxes = sorted( boxes, key=lambda b: b['start'] )
        events = []
        for b in boxes:
            if events and b['start'] <= events[-1][1]:
                events[-1] = ( events[-1][0], max( events[-1][1], b['end'] ) )
            else:
                events.append( ( b['start'], b['end'] ) )
        events_by_label[label] = events
    return( events_by_label )


def intervals_overlap( a, b ):
    return( a[0] < b[1] and b[0] < a[1] )


def compare_events( events_a, events_b ):
    labels = sorted( set( events_a.keys() ) | set( events_b.keys() ) )
    for label in labels:
        ea = events_a.get( label, [] )
        eb = events_b.get( label, [] )
        a_only = [ e for e in ea if not any( intervals_overlap(e, o) for o in eb ) ]
        b_only = [ e for e in eb if not any( intervals_overlap(e, o) for o in ea ) ]
        shared_a = [ e for e in ea if e not in a_only ]

        dur = lambda evs: sum( e[1]-e[0] for e in evs )

        print( "\n  %s:"%(label) )
        print( "    config A: %d event(s), %.1fs total on-screen"%(len(ea), dur(ea)) )
        print( "    config B: %d event(s), %.1fs total on-screen"%(len(eb), dur(eb)) )
        print( "    shared (both configs caught it, in the same time window): %d event(s)"%(len(shared_a)) )
        if a_only:
            print( "    A-only (B completely missed this): %d event(s), timestamps: %s"%(
                len(a_only), ", ".join( "%.1f-%.1fs"%(s,e) for s,e in a_only ) ) )
        if b_only:
            print( "    B-only (A completely missed this): %d event(s), timestamps: %s"%(
                len(b_only), ", ".join( "%.1f-%.1fs"%(s,e) for s,e in b_only ) ) )
        if not a_only and not b_only and (ea or eb):
            print( "    no unique events either way - same real-world coverage from this data, whichever config is cheaper wins" )


def main():
    parser = argparse.ArgumentParser( description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter )
    parser.add_argument( '--video', required=True )
    parser.add_argument( '--preview-start-seconds', type=float, required=True,
        help="must match the preview-start-seconds both caches were built with" )
    parser.add_argument( '--sizes-a', type=int, nargs='+', required=True )
    parser.add_argument( '--fps-a', type=float, required=True )
    parser.add_argument( '--sizes-b', type=int, nargs='+', required=True )
    parser.add_argument( '--fps-b', type=float, required=True )
    args = parser.parse_args()

    video_path = os.path.join( betaconst.video_path_uncensored, args.video )
    if not os.path.exists( video_path ):
        print( "no such file under %s: %s"%(betaconst.video_path_uncensored, args.video) )
        sys.exit(1)

    cap = cv2.VideoCapture( video_path )
    vid_w = int( cap.get( cv2.CAP_PROP_FRAME_WIDTH ) )
    vid_h = int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT ) )
    cap.release()
    if not vid_w or not vid_h:
        print( "couldn't read frame dimensions from %s"%(video_path) )
        sys.exit(1)

    global_min_prob = getattr( betaconfig, 'global_min_prob', betaconst.global_min_prob )
    file_hash = bu_hash.md5_for_file( video_path, 16 )
    cache_suffix = '-preview@%.1fs'%(args.preview_start_seconds)

    smooth_boxes_fn, apply_suppression_fn = rt._load_real_pipeline_functions()

    def load_or_die( sizes, fps, label ):
        flat_raw, missing = rt.load_cached_raw_boxes( file_hash, sizes, fps, global_min_prob, cache_suffix )
        if missing:
            print( "missing detection cache for config %s (size(s) %s @ fps=%s) - run this first:"%(
                label, [m[0] for m in missing], fps ) )
            print( "    python3 betatv.py --preview on --preview-start-seconds %s --picture-sizes %s --video-censor-fps %s"%(
                args.preview_start_seconds, " ".join(map(str,sizes)), fps ) )
            sys.exit(1)
        return( flat_raw )

    raw_a = load_or_die( args.sizes_a, args.fps_a, "A" )
    raw_b = load_or_die( args.sizes_b, args.fps_b, "B" )

    print( "config A: picture_sizes=%s video_censor_fps=%s (%d raw detections)"%(args.sizes_a, args.fps_a, len(raw_a)) )
    print( "config B: picture_sizes=%s video_censor_fps=%s (%d raw detections)"%(args.sizes_b, args.fps_b, len(raw_b)) )

    tracked_a = run_pipeline_for_config( raw_a, vid_w, vid_h, args.fps_a, smooth_boxes_fn, apply_suppression_fn )
    tracked_b = run_pipeline_for_config( raw_b, vid_w, vid_h, args.fps_b, smooth_boxes_fn, apply_suppression_fn )

    events_a = boxes_to_events( tracked_a )
    events_b = boxes_to_events( tracked_b )

    print( "\n=== per-label event comparison ===" )
    compare_events( events_a, events_b )
    print()
    print( "Reminder: an 'A-only'/'B-only' event means one config found NOTHING overlapping it in the other -" )
    print( "not a difference in box position/size. Go look at those specific timestamps in each config's" )
    print( "preview output before concluding a setting is 'better' - a unique event could be a real catch" )
    print( "the other setting missed, OR a false positive unique to that setting." )


if __name__ == '__main__':
    main()
