#!/usr/bin/env python3
"""
batch_ab_test.py - answers ONE question rigorously: does raising
nn_batch_size actually speed up detection, and does it change what gets
detected?

Why this needs its own tool instead of tune_sweep.sh: nn_batch_size is
deliberately NOT part of the detection cache key (box_hash_path in
betatv.py) - it's not supposed to change results, only how many frames get
batched into one neural-net call, so reusing cached detections across
different nn_batch_size values is normally correct and saves real time.
But that same fact means a naive sweep across batch sizes measures nothing
real: the SECOND (and every later) batch size you test for a given
picture_sizes/video_censor_fps combo just replays the cache the FIRST one
wrote, so its "detection_seconds" collapses to nearly zero - not because
batching is fast, but because no detection happened at all. That's exactly
what showed up in the last tune_sweep.sh run (nn_batch_size=4 rows
reporting ~0.2-0.3s average detection time, next to 100-260s for
nn_batch_size=1 on the same combo).

What this script does differently:
  1. Before EACH batch size is tested, it deletes the exact detection
     cache file(s) that combo would hit (and any stale .checkpoint next to
     them) - so every batch size genuinely re-runs detection from scratch
     against the SAME video/picture_sizes/video_censor_fps/preview slice,
     and betasuite_stats.jsonl's detection_seconds for that run is real.
  2. It then reads the freshly-written detection cache for each batch size
     directly (it's just gzipped JSON - see betautils_hash.py) and
     compares the raw detections between batch sizes NUMERICALLY: same
     count per sampled frame, and for frames that do line up, how far
     apart are the box positions/scores. This is a stronger and more
     honest quality check than eyeballing the rendered video for this
     SPECIFIC axis, because nn_batch_size isn't supposed to change
     detections at all (unlike picture_sizes/video_censor_fps, which
     deliberately do change what's detected and genuinely need visual
     inspection instead - see tune_sweep.sh for those).

This still produces normal preview .mp4 output for every batch size tested
(betatv.py runs for real, nothing here fakes that) - so if the numeric
comparison flags a real difference, the video is right there to look at
too. It does NOT touch betaconfig.py.

Usage:
    python3 batch_ab_test.py --batch-sizes 1 4 8
    python3 batch_ab_test.py --batch-sizes 1 4 --picture-sizes 1500 2000 --video-censor-fps 10
    python3 batch_ab_test.py --batch-sizes 1 4 --preview-start-seconds 450 --preview-seconds 20

    # or, same as tune_sweep.sh, via env vars instead of flags (flags win if both are given):
    PREVIEW_START_SECONDS=450 PREVIEW_SECONDS=33 python3 batch_ab_test.py --batch-sizes 1 4 8

The FIRST value in --batch-sizes is treated as the trusted baseline
(normally 1 - BetaSuite's original, known-working, one-frame-per-call
behavior) that every other batch size is compared against.

Backend-aware through the multi-adapter architecture: by default this
runs the FULL comparison once per registered detector backend (see
betautils_detector.py's _BACKENDS), each entirely real (cache cleared,
real betatv.py subprocess run), and closes with a cross-backend summary
table of which batch size was actually fastest FOR EACH backend - since
nn_batch_size is currently one shared betaconfig.py setting, not
per-backend, and the two models' batching behavior is not guaranteed to
match (this is the direct follow-up to a real compare_models_perf.py run
that showed retinanet_v2 NOT speeding up under batching the way
nudenet_v3 did - see print_cross_backend_summary's own docstring). Pass
--backends to restrict to one or more specific backends, e.g.:
    python3 batch_ab_test.py --batch-sizes 1 2 4 8 --backends retinanet_v2
    python3 batch_ab_test.py --batch-sizes 1 2 4 8 --backends retinanet_v2 nudenet_v3
"""

import argparse
import os
import subprocess
import sys
import time

import cv2

import sys as _sys, os as _os
_sys.path.insert( 0, _os.path.dirname( _os.path.dirname( _os.path.dirname( _os.path.abspath( __file__ ) ) ) ) )  # so 'import betaconst'/'betaconfig'/etc still resolve after this script moved into tools/analysis or tools/tuning

import betaconst
import betautils_detector as bu_detector
import betaconfig
import betautils_hash as bu_hash
import betautils_video as bu_video
import betautils_cache_paths as bu_cache


# box_hash_path_for's actual formula now lives in betautils_cache_paths.py
# (shared with analyze_jitter.py/replay_tune.py/etc - see that module's
# docstring for why) - this stays a thin wrapper (not a plain import)
# because this file's own (cache_suffix, backend_name) argument ORDER is
# reversed from the shared module's (backend_name, preview_suffix='') and
# every call site below already relies on that order; adapting here is
# lower-risk than reordering every call site of a function clear_cache
# (a destructive operation) also depends on getting right.
def box_hash_path_for( file_hash, size, fps, min_prob, cache_suffix, backend_name ):
    return bu_cache.box_hash_path_for( file_hash, size, fps, min_prob, backend_name, cache_suffix )


def clear_cache( path ):
    cleared = False
    for p in ( path, path + '.checkpoint' ):
        if os.path.exists( p ):
            os.remove( p )
            cleared = True
    return( cleared )


def find_video_files():
    # same os.walk betatv.py itself uses, so file_hash/root pairing here
    # matches exactly what betatv.py will process
    files = []
    for root, d_names, f_names in os.walk( betaconst.video_path_uncensored ):
        for fname in f_names:
            files.append( ( root, fname ) )
    return( files )


def load_stats_rows( stats_path ):
    import json
    rows = []
    try:
        with open( stats_path, 'r' ) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append( json.loads( line ) )
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return( rows )


def round_t( t ):
    return( round( t, 6 ) )


def diff_boxes( baseline_boxes, other_boxes, score_tol, coord_tol ):
    """Groups both box lists by their 't' (sample timestamp) and, frame by
    frame, matches each baseline box to its nearest same-class candidate in
    the other run (nearest-neighbor by position, not just list position -
    robust even if per-frame ordering ever differs between batch sizes).
    Returns a summary dict, not a pass/fail verdict - the caller decides
    what tolerance means "fine" for their purposes."""
    from collections import defaultdict

    base_by_t = defaultdict( list )
    for b in baseline_boxes:
        base_by_t[ round_t( b['t'] ) ].append( b )
    other_by_t = defaultdict( list )
    for b in other_boxes:
        other_by_t[ round_t( b['t'] ) ].append( b )

    all_ts = sorted( set( base_by_t.keys() ) | set( other_by_t.keys() ) )

    frames_compared = 0
    frames_count_mismatch = []
    matched_pairs = 0
    unmatched_baseline = 0
    unmatched_other = 0
    max_score_diff = 0.0
    max_coord_diff = 0.0
    exact_matches = 0
    tolerant_matches = 0
    real_mismatches = []  # (t, baseline_box, nearest_other_box, reason)

    for t in all_ts:
        frames_compared += 1
        base_frame = list( base_by_t.get( t, [] ) )
        other_frame = list( other_by_t.get( t, [] ) )
        if len( base_frame ) != len( other_frame ):
            frames_count_mismatch.append( ( t, len(base_frame), len(other_frame) ) )

        remaining_other = list( other_frame )
        for bb in base_frame:
            best = None
            best_dist = None
            for ob in remaining_other:
                if ob['class_id'] != bb['class_id']:
                    continue
                dist = ( (bb['x']-ob['x'])**2 + (bb['y']-ob['y'])**2 ) ** 0.5
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best = ob
            if best is None:
                unmatched_baseline += 1
                real_mismatches.append( ( t, bb, None, "no matching class_id in other run's frame" ) )
                continue
            remaining_other.remove( best )
            matched_pairs += 1
            score_diff = abs( bb['score'] - best['score'] )
            coord_diff = max( abs(bb['x']-best['x']), abs(bb['y']-best['y']), abs(bb['w']-best['w']), abs(bb['h']-best['h']) )
            max_score_diff = max( max_score_diff, score_diff )
            max_coord_diff = max( max_coord_diff, coord_diff )
            if score_diff == 0 and coord_diff == 0:
                exact_matches += 1
            elif score_diff <= score_tol and coord_diff <= coord_tol:
                tolerant_matches += 1
            else:
                real_mismatches.append( ( t, bb, best, "score_diff=%.6f coord_diff=%.2f exceeds tolerance"%(score_diff, coord_diff) ) )
        unmatched_other += len( remaining_other )
        for ob in remaining_other:
            real_mismatches.append( ( t, None, ob, "extra detection with no baseline counterpart" ) )

    return( {
        'frames_compared': frames_compared,
        'frames_count_mismatch': frames_count_mismatch,
        'matched_pairs': matched_pairs,
        'unmatched_baseline': unmatched_baseline,
        'unmatched_other': unmatched_other,
        'exact_matches': exact_matches,
        'tolerant_matches': tolerant_matches,
        'real_mismatches': real_mismatches,
        'max_score_diff': max_score_diff,
        'max_coord_diff': max_coord_diff,
    } )


def run_one_backend( backend_name, args, video_files, global_min_prob ):
    """
    Runs this tool's full batch-size A/B comparison (clear cache -> real
    betatv.py subprocess run -> compare raw detections -> report timing)
    once, entirely under one specific detector backend - via
    BETASUITE_DETECTOR_BACKEND_OVERRIDE (see betautils_detector.py),
    never by touching betaconfig.py. Split out from main() so main() can
    call this once per --backends entry and get a real, independently-
    measured verdict for each model, instead of only ever testing
    whichever backend betaconfig.detector_backend['selected'] happens to
    be set to.

    Args:
        backend_name: Which registered backend (bu_detector._BACKENDS
            key) this comparison should run entirely under.
        args: The parsed argparse.Namespace from main().
        video_files: Pre-discovered (root, fname) pairs from
            find_video_files() - discovered once in main() and passed
            in, not re-walked per backend, since the set of files on
            disk doesn't depend on which backend is being tested.
        global_min_prob: betaconfig.global_min_prob (or its
            betaconst.py fallback), resolved once in main().
    """
    env = dict( os.environ )
    env[ 'BETASUITE_DETECTOR_BACKEND_OVERRIDE' ] = backend_name

    # cache_suffix depends on each video's own length, not just
    # --preview-start-seconds: a video shorter than that offset falls back
    # (in betatv.py) to processing the WHOLE file with a plain '-preview'
    # suffix instead of '-preview@<start>s' - resolve_preview_slice
    # replicates that exactly so this always clears/reads the cache file
    # betatv.py actually wrote, not a naive '-preview@<start>s' guess
    # (this was silently wrong before: any video shorter than
    # --preview-start-seconds always "produced no cache" from this script's
    # point of view, even though betatv.py ran it to completion every time)
    file_hashes = {}
    cache_suffix_by_video = {}
    offset_by_fname = {}  # fname -> the preview_offset_seconds betasuite_stats.jsonl rows for this video will carry
    for root, fname in video_files:
        path = os.path.join( root, fname )
        file_hashes[(root, fname)] = bu_hash.md5_for_file( path, 16 )
        cap = cv2.VideoCapture( path )
        vid_fps = cap.get( cv2.CAP_PROP_FPS )
        num_frames = cap.get( cv2.CAP_PROP_FRAME_COUNT )
        cap.release()
        preview_offset_seconds, preview_window_unlimited = bu_video.resolve_preview_slice(
            vid_fps, num_frames, True, args.preview_seconds, args.preview_start_seconds )
        cache_suffix_by_video[(root, fname)] = bu_video.preview_cache_suffix( True, preview_offset_seconds )
        offset_by_fname[fname] = preview_offset_seconds
        if preview_window_unlimited:
            print( "note: preview_start_seconds=%.1fs is past the end of %s - this video will be processed and "
                   "cached as a WHOLE-file preview run (suffix '-preview'), not a %.1fs slice - its "
                   "betasuite_stats.jsonl row will carry preview_offset_seconds=0.0, not %.1f"%(
                args.preview_start_seconds, fname, args.preview_seconds, args.preview_start_seconds ) )

    # boxes_by_batch[batch][(root,fname)][size] = raw_boxes list (or None
    # if that combo's cache never got written - a failed/skipped file)
    boxes_by_batch = {}
    run_start_by_batch = {}

    for batch in args.batch_sizes:
        print( "=== [backend=%s] clearing caches and running detection at nn_batch_size=%d ==="%(backend_name, batch) )
        cleared_count = 0
        for (root, fname) in video_files:
            file_hash = file_hashes[(root, fname)]
            cache_suffix = cache_suffix_by_video[(root, fname)]
            for size in args.picture_sizes:
                cache_path = box_hash_path_for( file_hash, size, args.video_censor_fps, global_min_prob, cache_suffix, backend_name )
                if clear_cache( cache_path ):
                    cleared_count += 1
        print( "cleared %d existing cache file(s) for this combo before running"%(cleared_count) )

        run_start_by_batch[batch] = time.time()
        cmd = [
            sys.executable, 'betatv.py',
            '--preview', 'on',
            '--preview-start-seconds', str( args.preview_start_seconds ),
            '--preview-seconds', str( args.preview_seconds ),
            '--picture-sizes', *[ str(s) for s in args.picture_sizes ],
            '--video-censor-fps', str( args.video_censor_fps ),
            '--nn-batch-size', str( batch ),
        ]
        print( "running (backend=%s via BETASUITE_DETECTOR_BACKEND_OVERRIDE): %s"%(backend_name, " ".join(cmd)) )
        subprocess.run( cmd, check=False, env=env )

        per_video_sizes = {}
        for (root, fname) in video_files:
            file_hash = file_hashes[(root, fname)]
            cache_suffix = cache_suffix_by_video[(root, fname)]
            sizes_boxes = {}
            for size in args.picture_sizes:
                cache_path = box_hash_path_for( file_hash, size, args.video_censor_fps, global_min_prob, cache_suffix, backend_name )
                if os.path.exists( cache_path ):
                    sizes_boxes[size] = bu_hash.read_json( cache_path )
                else:
                    sizes_boxes[size] = None
                    print( "WARNING: no detection cache written for %s at size %d, batch %d, backend %s - that run likely failed, see its output above"%(fname, size, batch, backend_name) )
            per_video_sizes[(root, fname)] = sizes_boxes
        boxes_by_batch[batch] = per_video_sizes
        print()

    # timing: pull real detection_seconds from betasuite_stats.jsonl,
    # scoped to rows written during each batch's run window and matching
    # this comparison's config, rather than timing the whole subprocess
    # (which would also include model-load and encode overhead)
    stats_path = getattr( betaconfig, 'stats_path', '../output/stats/betasuite_stats.jsonl' )
    stats_rows = load_stats_rows( stats_path )
    timing_by_batch = {}
    for batch in args.batch_sizes:
        window_start = run_start_by_batch[batch]
        per_file_timing = {}
        for row in stats_rows:
            if row.get( 'timestamp', 0 ) < window_start:
                continue
            if row.get( 'nn_batch_size' ) != batch:
                continue
            # belt-and-suspenders: the time-window scoping above already
            # guarantees this row came from this backend's own subprocess
            # run (each run_one_backend call launches betatv.py with
            # BETASUITE_DETECTOR_BACKEND_OVERRIDE=backend_name), but
            # betasuite_stats.jsonl rows have recorded which backend
            # produced them since 2026-09-15 (see betatv.py's stats-row
            # write) - checking it directly costs nothing and guards
            # against a row landing just outside its true window (clock
            # skew, a slow-starting subprocess) being misattributed.
            # Rows written before that field existed have no
            # 'detector_backend' key at all - row.get(...) returns None,
            # which never equals a real backend_name, so such a row is
            # simply excluded here rather than crashing; that's fine,
            # since a pre-field row can't have come from this comparison
            # run anyway (the window-start filter above already excludes
            # anything old enough to predate this field).
            if row.get( 'detector_backend' ) != backend_name:
                continue
            if row.get( 'video_censor_fps' ) != args.video_censor_fps:
                continue
            if row.get( 'picture_sizes' ) != args.picture_sizes:
                continue
            # expected offset is per-video, not a single global value - a
            # video shorter than --preview-start-seconds writes its stats
            # row with preview_offset_seconds=0.0 (whole-file fallback),
            # not args.preview_start_seconds - see offset_by_fname above
            expected_offset = offset_by_fname.get( row.get('file') )
            if expected_offset is None or row.get( 'preview_offset_seconds' ) != expected_offset:
                continue
            per_file_timing[ row.get('file') ] = row.get('detection_seconds')
        timing_by_batch[batch] = per_file_timing

    # --- report ---
    baseline_batch = args.batch_sizes[0]
    print( "=" * 70 )
    print( "RESULTS for backend=%s (baseline = nn_batch_size=%d)"%(backend_name, baseline_batch) )
    print( "=" * 70 )

    size_label = "+".join( str(s) for s in args.picture_sizes )
    for (root, fname) in video_files:
        print( "\n%s"%(fname) )

        # detection_seconds comes from ONE stats row betatv.py writes per
        # FILE, after every requested picture_size finishes detection AND
        # the final encode succeeds (see betatv.py's bu_log.write_stats
        # call) - it is the combined time across ALL of args.picture_sizes
        # together, never a per-size breakdown, and it's only written at
        # all if the whole file made it to a successful encode. Printing it
        # once here (instead of once per size, identically, which used to
        # make picture_size=1280 and picture_size=2000 show the exact same
        # number and look like two independent measurements) avoids
        # misreading it as a per-size time. A partial failure (e.g. one
        # size in the combo OOMs) means NO stats row was ever written for
        # that file/batch at all, so there's no way to recover "how long
        # did just the size that succeeded take" from this data - that's
        # flagged explicitly below rather than silently printed as '?'.
        print( "  detection_seconds (combined across picture_sizes=[%s] - betatv.py logs one combined time per file, not split per size):"%(size_label) )
        for batch in args.batch_sizes:
            det_s = timing_by_batch[batch].get( fname )
            tag = "  (baseline)" if batch == baseline_batch else ""
            if det_s is not None:
                print( "    batch=%-3d: %s%s"%(batch, det_s, tag) )
            else:
                any_size_failed = any( boxes_by_batch[batch][(root, fname)].get(size) is None for size in args.picture_sizes )
                if any_size_failed:
                    print( "    batch=%-3d: no timing available - at least one picture_size failed for this file/batch (run failed before the final combined stats row could be written), see run output above"%(batch) )
                else:
                    print( "    batch=%-3d: no timing available - all sizes produced a cache but no matching stats row was found (unexpected, check betasuite_stats.jsonl)"%(batch) )

        for size in args.picture_sizes:
            baseline_boxes = boxes_by_batch[baseline_batch][(root, fname)].get( size )
            print( "  picture_size=%d raw-detection comparison:"%(size) )
            if baseline_boxes is None:
                print( "    baseline run produced no cache - can't compare, check the run output above" )
                continue
            print( "    batch=%-3d (baseline): %4d raw boxes"%(baseline_batch, len(baseline_boxes)) )
            for batch in args.batch_sizes[1:]:
                other_boxes = boxes_by_batch[batch][(root, fname)].get( size )
                if other_boxes is None:
                    print( "    batch=%-3d: no cache produced - run failed, see output above"%(batch) )
                    continue
                summary = diff_boxes( baseline_boxes, other_boxes, args.score_tol, args.coord_tol )
                verdict = "IDENTICAL" if not summary['real_mismatches'] and summary['exact_matches']==summary['matched_pairs']==len(baseline_boxes)==len(other_boxes) else (
                    "MATCHES WITHIN TOLERANCE (float noise only)" if not summary['real_mismatches'] else
                    "REAL DIFFERENCES FOUND - worth a visual check" )
                print( "    batch=%-3d: %4d raw boxes -- %s"%(batch, len(other_boxes), verdict) )
                print( "        exact matches=%d  tolerant matches=%d  unmatched(baseline)=%d  unmatched(other)=%d  max_score_diff=%.6f  max_coord_diff=%.2fpx"%(
                    summary['exact_matches'], summary['tolerant_matches'], summary['unmatched_baseline'], summary['unmatched_other'],
                    summary['max_score_diff'], summary['max_coord_diff'] ) )
                if summary['frames_count_mismatch']:
                    print( "        %d frame(s) had a different NUMBER of detections between batch sizes, e.g. t=%.2fs: baseline=%d other=%d"%(
                        len(summary['frames_count_mismatch']), summary['frames_count_mismatch'][0][0], summary['frames_count_mismatch'][0][1], summary['frames_count_mismatch'][0][2] ) )
                if summary['real_mismatches']:
                    t0, bb0, ob0, reason0 = summary['real_mismatches'][0]
                    print( "        first real mismatch at t=%.2fs: %s"%(t0, reason0) )
                    print( "        -> worth looking at this exact timestamp in both preview outputs for size=%d, batch=%d vs batch=%d"%(size, baseline_batch, batch) )

    # per-batch, per-file detection_seconds for this backend, so main() can
    # build the cross-backend "best batch size per model" summary at the end
    return {
        batch: { fname: timing_by_batch[batch].get(fname) for (root, fname) in video_files }
        for batch in args.batch_sizes
    }


def print_cross_backend_summary( backends, batch_sizes, timing_by_backend, video_files ):
    """
    Prints one closing table: for each backend, the mean detection_seconds
    (across every video that produced timing for that batch) at each
    batch size tested, plus which batch size was actually fastest for
    that backend - a real, apples-to-apples "does raising nn_batch_size
    even help THIS model" verdict per backend, in one place, rather than
    something you have to reconstruct by scrolling back through each
    backend's own per-file section above.

    This is exactly the question that prompted building this: a real
    compare_models_perf.py run showed retinanet_v2's batched-mode
    per-image cost HIGHER than its single-image cost (the opposite of
    what batching should do), while nudenet_v3 showed no such
    regression - but that was raw-inference-only timing on one frame
    repeated, not a real, cache-cleared, full detection pass across real
    footage. This summary is the same question asked with this tool's
    stronger methodology: real detection, real cache invalidation, timed
    via betatv.py's own betasuite_stats.jsonl rows.
    """
    print( "=" * 70 )
    print( "CROSS-BACKEND SUMMARY: mean detection_seconds per batch size, and the fastest batch size per backend" )
    print( "=" * 70 )
    print( "(mean is across every video in this run that produced a timed stats row for that batch/backend combo -" )
    print( " see each backend's own per-file section above for the individual numbers this is averaging)" )
    print()

    header = "%-16s"%("backend",)
    for batch in batch_sizes:
        header += "  %-18s"%("batch=%d"%(batch),)
    header += "  fastest"
    print( header )

    for backend_name in backends:
        timing_by_batch = timing_by_backend[ backend_name ]
        means = {}
        for batch in batch_sizes:
            per_file = timing_by_batch.get( batch, {} )
            values = [ v for v in per_file.values() if v is not None ]
            means[ batch ] = ( sum(values) / len(values) ) if values else None

        row = "%-16s"%(backend_name,)
        for batch in batch_sizes:
            m = means[ batch ]
            row += "  %-18s"%("%.2fs"%(m) if m is not None else "no data",)

        timed_batches = { b: m for b, m in means.items() if m is not None }
        if timed_batches:
            fastest_batch = min( timed_batches, key=timed_batches.get )
            row += "  batch=%d"%(fastest_batch)
        else:
            row += "  (no timed data)"
        print( row )

    print()
    print( "If a backend's fastest batch size here isn't nn_batch_size=2 (betaconfig.py's current, shared default)," )
    print( "that's real evidence THIS backend wants a different value - nn_batch_size is currently one global setting" )
    print( "shared by every backend (see betaconfig.py), not per-backend, so if the two models genuinely disagree on" )
    print( "what batch size is fastest, the fix is either picking whichever value is 'good enough' for both, or adding" )
    print( "a per-backend nn_batch_size override the same way detector_backend[<name>] already holds each backend's" )
    print( "other tunable settings - worth deciding once this data is in, not before." )


def main():
    parser = argparse.ArgumentParser( description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter )
    parser.add_argument( '--batch-sizes', type=int, nargs='+', default=[1, 4, 8],
        help="batch sizes to test, in order - the FIRST is the trusted baseline everything else is compared against (default: 1 4 8)" )
    parser.add_argument( '--backends', nargs='+', default=None,
        help="which detector backends to test (default: every registered backend - see betautils_detector.py's _BACKENDS). Each backend gets its own full, independent A/B comparison, run entirely via BETASUITE_DETECTOR_BACKEND_OVERRIDE - betaconfig.py's detector_backend['selected'] is never read or written by this tool once this flag path is used." )
    parser.add_argument( '--picture-sizes', type=int, nargs='+', default=None,
        help="picture_sizes the detection caches were built with. Default: resolved for the SELECTED backend (betautils_detector.get_picture_sizes), which is what a real run actually used - not the shared betaconfig.picture_sizes, which a backend with its own or native sizes never uses." )
    parser.add_argument( '--video-censor-fps', type=float, default=float( betaconfig.video_censor_fps ),
        help="video_censor_fps to use for every run (default: betaconfig.py's current value)" )
    parser.add_argument( '--preview-start-seconds', type=float, default=float( os.environ.get( 'PREVIEW_START_SECONDS', 30.0 ) ),
        help="pinned preview slice start, same for every run (default: 30.0, or $PREVIEW_START_SECONDS if set - same env var tune_sweep.sh reads, e.g. PREVIEW_START_SECONDS=450 ./batch_ab_test.py)" )
    parser.add_argument( '--preview-seconds', type=float, default=float( os.environ.get( 'PREVIEW_SECONDS', 20.0 ) ),
        help="preview slice length (default: 20.0, or $PREVIEW_SECONDS if set)" )
    parser.add_argument( '--score-tol', type=float, default=1e-4,
        help="max score difference still counted as 'the same detection' (accounts for ordinary floating-point noise between batched/unbatched GPU inference, not a real difference) - default 1e-4" )
    parser.add_argument( '--coord-tol', type=float, default=1.0,
        help="max x/y/w/h pixel difference still counted as 'the same detection' - default 1.0px" )
    args = parser.parse_args()

    # Resolve per backend rather than from the shared betaconfig list:
    # a backend can declare its own picture_sizes, or inherit
    # its model's native sizes, so the shared list is frequently not the
    # size anything was actually detected at.
    args.picture_sizes = bu_cache.resolve_picture_sizes(
        args.picture_sizes, bu_detector.selected_backend_name() )

    if len( args.batch_sizes ) < 2:
        print( "need at least 2 --batch-sizes to compare (a baseline plus at least one to test against it)" )
        sys.exit(1)

    backends = args.backends or sorted( bu_detector._BACKENDS.keys() )

    global_min_prob = getattr( betaconfig, 'global_min_prob', betaconst.global_min_prob )

    video_files = find_video_files()
    if not video_files:
        print( "no video files found under %s"%(betaconst.video_path_uncensored) )
        sys.exit(1)

    print( "videos found: %d"%(len(video_files)) )
    print( "backends to test: %s"%(", ".join(backends)) )
    print( "batch sizes to test (baseline first): %s"%(args.batch_sizes) )
    print( "picture_sizes=%s video_censor_fps=%s preview_start_seconds=%s preview_seconds=%s"%(
        args.picture_sizes, args.video_censor_fps, args.preview_start_seconds, args.preview_seconds ) )
    print()

    timing_by_backend = {}
    for backend_name in backends:
        print( "#" * 70 )
        print( "# backend: %s"%(backend_name) )
        print( "#" * 70 )
        timing_by_backend[ backend_name ] = run_one_backend( backend_name, args, video_files, global_min_prob )
        print()

    if len( backends ) > 1:
        print_cross_backend_summary( backends, args.batch_sizes, timing_by_backend, video_files )


if __name__ == '__main__':
    main()
