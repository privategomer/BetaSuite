#!/usr/bin/env python3
"""
compare_models_perf.py - times both registered detector backends'
actual inference (get_session + raw_boxes_for_img/raw_boxes_for_imgs -
see betautils_detector.py), at each of your configured picture_sizes,
and reports latency/throughput side by side.

Why this exists: switching betaconfig.detector_backend['selected'] is
a real trade-off - a different model's accuracy characteristics almost
certainly come with a different speed, and "which one is faster on MY
actual hardware, at MY actual picture_sizes" is a real machine
measurement, not something that can be read out of either model's own
published benchmarks (those are near-certainly on different hardware,
possibly different batch sizes/providers than what you actually run).
This runs each backend for real, on this machine, right now.

What this does NOT do: measure detection quality/accuracy at all - see
analyze_suppression_pairs.py (now backend-aware: it reports its
box-size/confusion-pair-IoU stats once per detector backend found in
your cache, so you can compare quality side by side using real cached
detections from both models) for that half of the comparison. This
tool is purely "how fast", using either synthetic noise images (no
source footage required) or one real frame pulled from an actual video
you point it at with --video (see betaconst.video_path_uncensored /
video_path_source_backup for where it looks).

Timing methodology: a few warmup calls per (backend, size) combo
first (to absorb one-time costs - CUDA context/kernel init, ONNX
Runtime's own lazy graph optimizations - that a real long-running
betatv.py process would also only pay once, not per frame), discarded
from the reported numbers; then the real timed iterations. Reports
both the single-image path (raw_boxes_for_img, what betastare.py always
uses and what betatv.py falls back to) and the batched path
(raw_boxes_for_imgs at --batch-size images, what betatv.py actually
uses for its 'per chunk of nn_batch_size frames' sampling loop) since
they can have meaningfully different per-image throughput.

Usage:
    python3 compare_models_perf.py
    python3 compare_models_perf.py --sizes 1280 640
    python3 compare_models_perf.py --video "myfile.mp4"          # real frame instead of synthetic noise
    python3 compare_models_perf.py --iterations 30 --warmup 5
    python3 compare_models_perf.py --backends nudenet_v3          # only time one backend
"""

import argparse
import os
import statistics
import sys
import time

import cv2
import numpy as np

import sys as _sys, os as _os
_sys.path.insert( 0, _os.path.dirname( _os.path.dirname( _os.path.dirname( _os.path.abspath( __file__ ) ) ) ) )  # so 'import betaconst'/'betaconfig'/etc still resolve after this script moved into tools/analysis or tools/tuning

import betaconst
import betaconfig
import betautils_detector as bu_detector
import betautils_cache_paths as bu_cache


def find_real_frame( video_name ):
    """
    Locate video_name under betaconst.video_path_uncensored (falling
    back to betaconst.video_path_source_backup - see betaconst.py's own
    comment on that fallback), open it, and return its first real frame
    as a raw BGR numpy array.

    Args:
        video_name: A filename (not a full path) to look for under
            either directory.

    Returns:
        An HxWxC BGR uint8 numpy array.

    Raises:
        SystemExit: video_name isn't found under either directory, or
            couldn't be opened/read - printed reason, exit code 1
            (matches this tool suite's existing convention, e.g.
            compare_effectiveness.py's load_or_die).
    """
    for search_dir in (betaconst.video_path_uncensored, betaconst.video_path_source_backup):
        candidate = os.path.join( search_dir, video_name )
        if os.path.exists( candidate ):
            cap = cv2.VideoCapture( candidate )
            ok, frame = cap.read()
            cap.release()
            if not ok or frame is None:
                print( "found %s but couldn't read a frame from it"%(candidate) )
                sys.exit(1)
            print( "using a real frame from %s (%dx%d)"%(candidate, frame.shape[1], frame.shape[0]) )
            return frame
    print( "no such file under %s or %s: %s"%(betaconst.video_path_uncensored, betaconst.video_path_source_backup, video_name) )
    sys.exit(1)


def synthetic_frame( width=1920, height=1080 ):
    """
    A random-noise HxWxC BGR uint8 image at the given resolution - pixel
    content doesn't matter for timing (only tensor shape/size does, and
    how busy the CPU/GPU actually is doing real convolutions on real-
    looking data volume), so this needs no source footage at all. 1920x
    1080 is a common real-world source resolution, close to what this
    tool's timings will actually resemble in practice.
    """
    return( ( np.random.rand( height, width, 3 ) * 255 ).astype( np.uint8 ) )


# fmt_ms_stats's body now lives in betautils_cache_paths.py (shared with
# analyze_jitter.py's/analyze_track_breaks.py's near-identical fmt_stats -
# see that module's docstring) - thin wrapper kept so call sites below
# don't need to change.
def fmt_ms_stats( values_ms ):
    return bu_cache.fmt_ms_stats( values_ms )


def time_calls( fn, n_warmup, n_timed ):
    """
    Runs fn() n_warmup times (discarded), then n_timed times, returning
    the list of per-call wall-clock durations in milliseconds for the
    timed calls only.
    """
    for _ in range( n_warmup ):
        fn()
    durations_ms = []
    for _ in range( n_timed ):
        t0 = time.perf_counter()
        fn()
        durations_ms.append( ( time.perf_counter() - t0 ) * 1000.0 )
    return( durations_ms )


def time_one_backend_size( backend_name, size, frame, batch_size, n_warmup, n_timed ):
    """
    Times both the single-image and batched inference paths for one
    (backend, size) combination.

    Returns:
        A dict: {'provider', 'single_ms', 'batch_ms', 'batch_size'} -
        'single_ms'/'batch_ms' are lists of per-call durations (ms);
        'provider' is the ONNX Runtime execution provider actually in
        use (e.g. 'CUDAExecutionProvider' or 'CPUExecutionProvider') -
        confirms whether GPU is really engaged, not just configured to
        be, since ONNX Runtime silently falls back to CPU if a
        requested provider isn't actually available in this
        environment. None if this backend/size combo raised (reported,
        not fatal to the rest of the sweep - see main()).
    """
    detector = bu_detector.get_detector( backend_name )
    session = detector.get_session()
    provider = session.get_providers()[0] if session.get_providers() else 'unknown'

    single_ms = time_calls(
        lambda: detector.raw_boxes_for_img( frame, size, session, t=0.0 ),
        n_warmup, n_timed )

    batch_frames = [ frame ] * batch_size
    batch_ts = [ float(i) for i in range( batch_size ) ]
    batch_ms = time_calls(
        lambda: detector.raw_boxes_for_imgs( batch_frames, size, session, batch_ts ),
        n_warmup, n_timed )

    return( { 'provider': provider, 'single_ms': single_ms, 'batch_ms': batch_ms, 'batch_size': batch_size } )


def main():
    parser = argparse.ArgumentParser( description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter )
    parser.add_argument( '--sizes', type=int, nargs='+', default=None,
        help="picture sizes to time (default: the union of every timed backend's own resolved "
             "picture_sizes, so each backend is timed at least at the size it actually runs at)" )
    parser.add_argument( '--backends', nargs='+', default=None, choices=sorted( bu_detector._BACKENDS.keys() ),
        help="which backends to time (default: every registered backend)" )
    parser.add_argument( '--video', default=None,
        help="use one real frame from this video (searched under uncensored_vids/ then source/) instead of synthetic noise" )
    parser.add_argument( '--batch-size', type=int, default=None,
        help="batch size for the batched (raw_boxes_for_imgs) timing (default: betaconfig.nn_batch_size)" )
    parser.add_argument( '--warmup', type=int, default=3,
        help="warmup calls per (backend, size) combo, discarded from timing (default: 3)" )
    parser.add_argument( '--iterations', type=int, default=15,
        help="timed calls per (backend, size) combo (default: 15)" )
    args = parser.parse_args()

    backends = args.backends or sorted( bu_detector._BACKENDS.keys() )

    # Every backend is timed at every size, so the output stays a
    # rectangular size-by-backend table and a column really is
    # comparable. What changed is the DEFAULT set of sizes: the
    # shared betaconfig.picture_sizes is no longer what any given
    # backend runs at, so timing nudenet_v3 at 1280 (its old shared
    # size) answers a question nobody is asking. The union of each
    # backend's own resolved sizes keeps the table rectangular while
    # guaranteeing every backend appears at its real operating size.
    sizes = args.sizes or sorted( {
        size
        for backend_name in backends
        for size in bu_detector.get_picture_sizes( backend_name )
    } )

    frame = find_real_frame( args.video ) if args.video else synthetic_frame()
    if not args.video:
        print( "using a synthetic %dx%d random-noise frame (pass --video to time against a real frame instead)"%(
            frame.shape[1], frame.shape[0] ) )

    print( "backends: %s"%(", ".join(backends)) )
    print( "sizes: %s"%(sizes) )
    if args.batch_size is None:
        # per-backend resolution (see betautils_detector.get_nn_batch_size) -
        # each backend times at its OWN configured batch size, since the adapter
        # architecture retired the one-shared-value assumption; printed per-backend below
        # instead of once up front.
        print( "batch size: per-backend (betaconfig.detector_backend[<name>]['nn_batch_size'], via betautils_detector.get_nn_batch_size)" )
    else:
        print( "batch size: %d (--batch-size, forced for every backend)"%(args.batch_size) )
    print( "warmup=%d  iterations=%d"%(args.warmup, args.iterations) )
    print( "betaconfig.gpu_enabled=%s  betaconfig.cuda_device_id=%s\n"%(
        getattr(betaconfig,'gpu_enabled',None), getattr(betaconfig,'cuda_device_id',None) ) )

    results = {}  # (backend, size) -> result dict or None
    for backend_name in backends:
        batch_size = args.batch_size or bu_detector.get_nn_batch_size( backend_name )
        for size in sizes:
            print( "=== timing %s @ size=%d (batch_size=%d) ==="%(backend_name, size, batch_size) )
            try:
                result = time_one_backend_size( backend_name, size, frame, batch_size, args.warmup, args.iterations )
            except Exception as err:
                print( "  FAILED: %s"%(err) )
                results[ (backend_name, size) ] = None
                continue
            results[ (backend_name, size) ] = result
            print( "  execution provider: %s"%(result['provider']) )
            print( "  single-image (raw_boxes_for_img):  %s"%(fmt_ms_stats(result['single_ms'])) )
            single_median = statistics.median( result['single_ms'] )
            print( "    -> %.1f images/sec (1/median)"%(1000.0/single_median if single_median > 0 else float('inf')) )
            print( "  batched, batch_size=%d (raw_boxes_for_imgs): %s"%(batch_size, fmt_ms_stats(result['batch_ms'])) )
            batch_median = statistics.median( result['batch_ms'] )
            effective_per_image_ms = batch_median / batch_size if batch_size else batch_median
            print( "    -> %.1fms/image effective, %.1f images/sec effective"%(
                effective_per_image_ms, 1000.0/effective_per_image_ms if effective_per_image_ms > 0 else float('inf') ) )
            print()

    print( "=== summary (median single-image ms, median effective batched ms/image) ===" )
    header = "%-16s"%("size",)
    for backend_name in backends:
        header += "  %-34s"%(backend_name,)
    print( header )
    for size in sizes:
        row = "%-16d"%(size,)
        for backend_name in backends:
            result = results.get( (backend_name, size) )
            if result is None:
                row += "  %-34s"%("FAILED",)
                continue
            single_median = statistics.median( result['single_ms'] )
            batch_median = statistics.median( result['batch_ms'] ) / result['batch_size']
            row += "  %-34s"%("%.1fms single / %.1fms batched"%(single_median, batch_median),)
        print( row )

    print()
    print( "Reminder: this measures raw model inference only (get_session + raw_boxes_for_img/raw_boxes_for_imgs) -" )
    print( "not a full betatv.py/betastare.py run, which also spends time on video decode, suppression, tracking," )
    print( "and rendering the actual censored output. Use this to compare the two MODELS' own cost, not to predict" )
    print( "a full run's wall-clock time." )


if __name__ == '__main__':
    main()
