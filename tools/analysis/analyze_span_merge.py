#!/usr/bin/env python3
"""
analyze_span_merge.py - checks why a paired_style item's 'span' bar
(betautils_censor.py's collapse_boxes_for_style) might not visibly bridge
two breasts even when both are genuinely on screen at once.

Why this needed its own tool rather than guessing from the code: span
only fires when a render frame's live boxes for that label+resolved-style
group into a "piece" of EXACTLY 2 (see collapse_boxes_for_style's
`if len(piece) != 2: return piece`) - by design, so 3+ simultaneous boxes
(read as multiple people) don't get wrongly bridged. If real per-frame
detection/tracking noise ever puts a 3rd box of the same label+style
briefly live at the same instant (a duplicate/near-duplicate detection,
or a contention/reset event spinning up a short-lived extra track - the
same kind of thing analyze_track_breaks.py and analyze_style_flicker.py
were built to check elsewhere in this pipeline), that ONE frame's piece
size becomes 3 and span silently falls back to unmerged for it - even
though the two real breasts ARE both on screen. This tool replays your
real cached detections through the actual smooth_boxes/
apply_class_suppression AND the exact same per-native-frame live_boxes
windowing loop betatv.py's real renderer uses (see the render loop this
was extracted from), so it can report the REAL piece-size history for a
preview window without needing to actually decode/render any video.

What it reports, for each label with a 'span'-configured style in its
censor_style list:
  - how many native video frames in the tested window had exactly
    0 / 1 / 2 / 3+ live boxes of that label sharing the exact same
    resolved style (the "would span" / "nothing to span" / "span should
    fire" / "span skipped, blocked by piece size" buckets)
  - for any 3+ frames (the interesting case - proof span was blocked),
    how many DISTINCT underlying boxes (by object identity, since the
    same box object stays "live" across many consecutive native frames)
    were involved across the whole window, and their approximate
    positions/time windows, so you can tell a genuine 3rd body part
    apart from a short-lived duplicate/contention artifact
  - the longest continuous run of "should span" (piece size == 2) native
    frames, and the longest continuous run of "span blocked" (piece size
    3+) native frames, as a rough sense of whether any blocking is
    transient (a flicker) or sustained (something structurally off)

This tool does NOT patch or change smooth_boxes/collapse_boxes_for_style
- it's read-only, same extract-and-instrument technique as
analyze_jitter.py/analyze_track_breaks.py/analyze_style_flicker.py/
replay_tune.py.

Prerequisite: same as replay_tune.py - a detection cache must already
exist for the video/picture_sizes/video_censor_fps/preview-start-seconds
combo you want to check (i.e. you've already run betatv.py --preview on,
or run_bar_smoothing_test.py, with those settings at least once).

Usage:
    python3 analyze_span_merge.py --video "myfile.mp4" --preview-start-seconds 10
    python3 analyze_span_merge.py --video "myfile.mp4" --preview-start-seconds 10 --labels exposed_breast
"""

import argparse
import copy
import glob
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
import betautils_config as bu_config
import betautils_track as bu_track
from betautils_cache_paths import load_cached_raw_boxes


def _load_real_pipeline_functions():
    """
    The real tracking and suppression functions, imported.

    This used to regex the functions out of betatv.py's source text and
    exec them. That coupled the tool to another file's line-by-line
    layout, so the v2.1.0 refactor - which only moved those functions
    into betautils_track.py - broke it with a RuntimeError at startup.
    Importing them means the tool replays exactly the code the real run
    used, and a future move is a rename, not a breakage.

    Returns:
        (smooth_boxes, apply_class_suppression).
    """
    return bu_track.smooth_boxes, bu_track.apply_class_suppression


# box_hash_path_for/load_cached_raw_boxes moved to betautils_cache_paths.py
# (shared with analyze_jitter.py/replay_tune.py/etc - see that module's
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


def labels_with_span( labels_filter ):
    found = []
    for label in betaconfig.items_to_censor:
        if labels_filter and label not in labels_filter:
            continue
        style = betaconfig.item_overrides.get( label, {} ).get( 'censor_style' )
        entries = style if isinstance( style, list ) else ( [style] if isinstance( style, dict ) else [] )
        for entry in entries:
            strategy = entry.get( 'merge', betaconfig.censor_overlap_strategy.get( entry.get('type'), 'none' ) )
            if strategy == 'span':
                found.append( label )
                break
    return found


def replay_live_box_history( boxes, vid_fps, start_frame, num_frames ):
    # exact replica of betatv.py's render-loop live_boxes bookkeeping
    # (see the chunk render loop in smooth_boxes's caller) - boxes must
    # already be sorted by 'start' for the pending_boxes.pop(0) logic to
    # behave the same way the real loop relies on
    boxes_sorted = sorted( boxes, key=lambda b: b['start'] )
    start_time = start_frame / vid_fps
    live_boxes = [ b for b in boxes_sorted if b['start'] <= start_time and b['end'] > start_time ]
    pending_boxes = [ b for b in boxes_sorted if b['start'] > start_time ]

    history = []  # one entry per native frame: list of live box dicts (by reference)
    for j in range( start_frame, start_frame + num_frames ):
        curr_time = j / vid_fps
        live_boxes = [ b for b in live_boxes if b['end'] >= curr_time ]
        while pending_boxes and pending_boxes[0]['start'] <= curr_time:
            live_boxes.append( pending_boxes.pop(0) )
        history.append( list( live_boxes ) )
    return history


def main():
    parser = argparse.ArgumentParser( description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter )
    parser.add_argument( '--video', required=True )
    parser.add_argument( '--picture-sizes', type=int, nargs='+', default=None,
        help="picture_sizes the detection caches were built with. Default: resolved for the SELECTED backend (betautils_detector.get_picture_sizes), which is what a real run actually used - not the shared betaconfig.picture_sizes, which a backend with its own or native sizes never uses." )
    parser.add_argument( '--video-censor-fps', type=float, default=float( betaconfig.video_censor_fps ) )
    parser.add_argument( '--preview-start-seconds', type=float, required=True )
    parser.add_argument( '--preview-seconds', type=float, default=float( os.environ.get( 'PREVIEW_SECONDS', getattr( betaconfig, 'preview_max_seconds', 20.0 ) ) ) )
    parser.add_argument( '--labels', nargs='+', default=None, help="restrict to these labels (default: every label with a 'span'-configured style)" )
    args = parser.parse_args()

    # Resolve per backend rather than from the shared betaconfig list:
    # since v2.1 a backend can declare its own picture_sizes, or inherit
    # its model's native sizes, so the shared list is frequently not the
    # size anything was actually detected at.
    args.picture_sizes = bu_cache.resolve_picture_sizes(
        args.picture_sizes, bu_detector.selected_backend_name() )

    labels = labels_with_span( args.labels )
    if not labels:
        print( "no labels currently have a 'span'-configured censor_style entry (or none matched --labels) - nothing to check" )
        sys.exit(1)
    print( "checking span-configured label(s): %s"%(labels) )

    video_path = os.path.join( betaconst.video_path_uncensored, args.video )
    if not os.path.exists( video_path ):
        print( "no such file under %s: %s"%(betaconst.video_path_uncensored, args.video) )
        sys.exit(1)

    cap = cv2.VideoCapture( video_path )
    vid_w = int( cap.get( cv2.CAP_PROP_FRAME_WIDTH ) )
    vid_h = int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT ) )
    vid_fps = cap.get( cv2.CAP_PROP_FPS )
    num_frames_total = cap.get( cv2.CAP_PROP_FRAME_COUNT )
    cap.release()
    if not vid_w or not vid_h or not vid_fps:
        print( "couldn't read frame dimensions/fps from %s"%(video_path) )
        sys.exit(1)

    global_min_prob = getattr( betaconfig, 'global_min_prob', betaconst.global_min_prob )
    preview_offset_seconds, preview_window_unlimited = bu_video.resolve_preview_slice(
        vid_fps, num_frames_total, True, args.preview_seconds, args.preview_start_seconds )
    cache_suffix = bu_video.preview_cache_suffix( True, preview_offset_seconds )
    if preview_window_unlimited:
        print( "note: --preview-start-seconds %.1fs is past the end of this video - its cache was built from "
               "the WHOLE file (suffix '-preview')"%(args.preview_start_seconds) )
    file_hash = bu_hash.md5_for_file( video_path, 16 )

    flat_raw, missing = load_cached_raw_boxes(
        file_hash, args.picture_sizes, args.video_censor_fps, global_min_prob, bu_detector.selected_backend_name(), cache_suffix )
    if missing:
        print( "missing detection cache for size(s) %s - run a preview pass with these exact settings first"%([m[0] for m in missing]) )
        sys.exit(1)
    print( "loaded %d cached raw detections (picture_sizes=%s fps=%s preview_start=%s)"%(
        len(flat_raw), args.picture_sizes, args.video_censor_fps, args.preview_start_seconds) )

    smooth_boxes_fn, apply_suppression_fn = _load_real_pipeline_functions()

    backend_name = bu_detector.selected_backend_name()
    raw_copy = copy.deepcopy( flat_raw )
    surviving, _ = apply_suppression_fn(
        raw_copy,
        bu_detector.get_class_suppression( backend_name ),
        bu_config.get_parts_to_blur( backend_name ) )
    boxes = []
    for r in surviving:
        res = bu_censor.process_raw_box( r, vid_w, vid_h )
        if res:
            boxes.append( res )
    # smooth_boxes returns a NEW list: it adds interpolated boxes and
    # drops unconfirmed tracks, so the input list is not the result.
    boxes, _smoothing_stats = smooth_boxes_fn( boxes, backend_name=backend_name )
    print( "%d tracked/interpolated boxes total across all labels"%(len(boxes)) )
    print()

    if preview_window_unlimited:
        start_frame = 0
        num_native_frames = int( num_frames_total )
    else:
        start_frame = round( preview_offset_seconds * vid_fps )
        num_native_frames = min( int(num_frames_total) - start_frame, int( round( args.preview_seconds * vid_fps ) ) )
    print( "replaying %d native frames (vid_fps=%.3f) starting at frame %d (t=%.2fs)"%(
        num_native_frames, vid_fps, start_frame, start_frame/vid_fps) )
    print()

    for label in labels:
        label_boxes = [ b for b in boxes if b['label'] == label ]
        history = replay_live_box_history( label_boxes, vid_fps, start_frame, num_native_frames )

        # group each frame's live boxes into "pieces" exactly like
        # censor_img_for_boxes does (sort by censor_style_sort, then
        # adjacency-group by exact censor_style_key equality) - restricted
        # to pieces whose style actually resolves to a 'span' strategy, so
        # a label mixing span and non-span style variants only counts the
        # span-relevant pieces here
        size_counts = { 0: 0, 1: 0, 2: 0 }
        size_3plus = 0
        seen_box_ids_in_3plus = {}
        run_lengths = { 'span_ready': [], 'span_blocked': [] }
        cur_ready_run = 0
        cur_blocked_run = 0

        for frame_idx, live in enumerate( history ):
            live_sorted = sorted( live, key=lambda b: bu_censor.censor_style_sort( b['censor_style'] ) )
            pieces = []
            for b in live_sorted:
                if pieces and bu_censor.censor_style_key( pieces[-1][0]['censor_style'] ) == bu_censor.censor_style_key( b['censor_style'] ):
                    pieces[-1].append( b )
                else:
                    pieces.append( [b] )

            # only pieces whose resolved style is actually 'span' matter here
            span_piece_size = None
            for piece in pieces:
                style = piece[0]['censor_style']
                strategy = style.get( 'merge', betaconfig.censor_overlap_strategy.get( style.get('type'), 'none' ) )
                if strategy == 'span':
                    span_piece_size = len( piece )
                    if span_piece_size >= 3:
                        for b in piece:
                            seen_box_ids_in_3plus[ id(b) ] = b
                    break  # a label with only one span-configured style entry will have at most one such piece

            if span_piece_size is None:
                size_counts[0] = size_counts.get(0,0) + 1
                cur_ready_run, cur_blocked_run = 0, 0
                continue

            if span_piece_size == 2:
                size_counts[2] += 1
                cur_ready_run += 1
                if cur_blocked_run:
                    run_lengths['span_blocked'].append( cur_blocked_run )
                cur_blocked_run = 0
            elif span_piece_size == 1:
                size_counts[1] += 1
                cur_ready_run, cur_blocked_run = 0, 0
            else:
                size_3plus += 1
                cur_blocked_run += 1
                if cur_ready_run:
                    run_lengths['span_ready'].append( cur_ready_run )
                cur_ready_run = 0

        if cur_ready_run:
            run_lengths['span_ready'].append( cur_ready_run )
        if cur_blocked_run:
            run_lengths['span_blocked'].append( cur_blocked_run )

        total = len( history )
        print( "=== %s ==="%(label) )
        print( "  native frames with 0 live span-style box:  %d (%.0f%%) - nothing to span, expected"%(size_counts.get(0,0), 100*size_counts.get(0,0)/total if total else 0) )
        print( "  native frames with 1 live span-style box:  %d (%.0f%%) - only one instance on screen, nothing to span"%(size_counts.get(1,0), 100*size_counts.get(1,0)/total if total else 0) )
        print( "  native frames with 2 live span-style boxes: %d (%.0f%%) - SHOULD be spanning right now"%(size_counts.get(2,0), 100*size_counts.get(2,0)/total if total else 0) )
        print( "  native frames with 3+ live span-style boxes: %d (%.0f%%) - span BLOCKED (piece size != 2) even though 2+ are visible"%(size_3plus, 100*size_3plus/total if total else 0) )
        if size_3plus:
            print( "    distinct underlying boxes involved across all 3+ frames: %d"%(len(seen_box_ids_in_3plus)) )
            for b in list(seen_box_ids_in_3plus.values())[:10]:
                print( "      box: t=%.2f-%.2fs pos=(%d,%d) size=(%d,%d)"%(b['start'], b['end'], b['x'], b['y'], b['w'], b['h']) )
            if len(seen_box_ids_in_3plus) > 10:
                print( "      ... and %d more"%(len(seen_box_ids_in_3plus)-10) )
        if run_lengths['span_ready']:
            print( "  longest continuous 'should be spanning' run: %d frames (%.2fs)"%(max(run_lengths['span_ready']), max(run_lengths['span_ready'])/vid_fps) )
        if run_lengths['span_blocked']:
            print( "  longest continuous 'span blocked by piece size' run: %d frames (%.2fs)"%(max(run_lengths['span_blocked']), max(run_lengths['span_blocked'])/vid_fps) )
        print()

    print( "reading this: if 'SHOULD be spanning' is 0 across the whole window, span never had a real chance to fire at " )
    print( "all in this clip (nothing to fix - there just wasn't a genuine 2-simultaneous moment for this label/style). " )
    print( "If 'SHOULD be spanning' is a meaningful chunk of frames but you didn't see a spanning bar when watching, that's " )
    print( "the discrepancy to chase - check the actual rendered output at one of those specific timestamps. If '3+ blocked' " )
    print( "frames exist and their box list positions look like the same real body part at nearly the same position (not " )
    print( "two clearly different locations), that's a duplicate/contention artifact stealing span's exactly-2 slot, same " )
    print( "root-cause family as what analyze_track_breaks.py/analyze_style_flicker.py check elsewhere in this pipeline." )


if __name__ == '__main__':
    main()
