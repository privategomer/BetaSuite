"""
betautils_estimate.py - how long a run will take, from how long earlier
runs on this machine took.

Every processed file's run record carries a timing (made here by
timing_for, kept by the catalog's history beside the run): the seconds
spent detecting and rendering, how much work each was, and the identity
of what did the work. An estimate takes the recent runs whose identity
matches what the next run would use and turns their median rates into
seconds for the files asked about:

  detect   sampled frames per second, over the detection sizes, from
           runs that detected from scratch (a cached or resumed
           detection says nothing about the model's speed)
  render   pixel-frames per second, from every processed run: decode,
           censor and encode all scale with the frame's area
  overhead the rest of a run's wall time (probing, hashing, shot cuts,
           muxing), as a median per file

A file whose detections are all cached costs no detection time. With no
comparable run, the answer says so (basis none) rather than guess.

Everything here but comparable_rows is pure, so the arithmetic is tested
without a catalog or a video.
"""

import json


RECENT_RUNS = 10

# What makes detection speed comparable between two runs, and render speed.
DETECT_IDENTITY = ( 'backend', 'variant', 'sizes', 'gpu', 'nn_batch_size', 'reverse' )
RENDER_IDENTITY = ( 'preset', 'gpu' )

FRESH = 'fresh'
CACHED = 'cached'
PARTIAL = 'partial'


def identity():
    """What the next run would do its work with, from the live config (as betatv resolves it)."""
    import betaconfig
    import betautils_detector as bu_detector
    import betautils_reverse as bu_reverse
    backend = bu_detector.selected_backend_name()
    if getattr( betaconfig, 'preview_mode_enabled', False ):
        preset = getattr( betaconfig, 'preview_encode_preset', 'ultrafast' )
    else:
        preset = getattr( betaconfig, 'encode_preset', 'fast' )
    return {
        'backend': backend,
        'variant': bu_detector.selected_variant_name( backend ),
        'sizes': list( bu_detector.get_picture_sizes( backend ) ),
        'gpu': bool( getattr( betaconfig, 'gpu_enabled', 0 ) ),
        'nn_batch_size': bu_detector.get_nn_batch_size( backend ),
        'reverse': bool( bu_reverse.person_windows_enabled() ),
        'preset': preset,
    }


def timing_for( detect_seconds, render_seconds, total_seconds, detect_cache, scanned_seconds, sample_fps,
                sizes, width, height, vid_fps, run_identity ):
    """The timing a processed file's run record carries."""
    return {
        'detect_seconds': round( float( detect_seconds ), 3 ),
        'render_seconds': round( float( render_seconds ), 3 ),
        'total_seconds': round( float( total_seconds ), 3 ),
        'detect_cache': detect_cache,
        'sampled_frames': int( round( float( scanned_seconds or 0 ) * float( sample_fps or 0 ) ) ) * len( sizes ),
        'rendered_frames': int( round( float( scanned_seconds or 0 ) * float( vid_fps or 0 ) ) ),
        'width': int( width or 0 ),
        'height': int( height or 0 ),
        'identity': dict( run_identity ),
    }


def _matches( timing, wanted, keys ):
    got = timing.get( 'identity' ) or {}
    return all( got.get( key ) == wanted.get( key ) for key in keys )


def rates( timings, wanted ):
    """
    { detect, render, overhead, detect_runs, render_runs } from timings,
    newest first, that match wanted. A rate is None when no run gives it.
    """
    import statistics
    detect, render, overhead = [], [], []
    for timing in timings:
        if not isinstance( timing, dict ):
            continue
        detect_seconds = timing.get( 'detect_seconds' ) or 0
        render_seconds = timing.get( 'render_seconds' ) or 0
        if ( len( detect ) < RECENT_RUNS and timing.get( 'detect_cache' ) == FRESH and detect_seconds > 0
                and timing.get( 'sampled_frames' ) and _matches( timing, wanted, DETECT_IDENTITY ) ):
            detect.append( timing['sampled_frames'] / detect_seconds )
        pixels = ( timing.get( 'width' ) or 0 ) * ( timing.get( 'height' ) or 0 )
        if ( len( render ) < RECENT_RUNS and render_seconds > 0 and pixels and timing.get( 'rendered_frames' )
                and _matches( timing, wanted, RENDER_IDENTITY ) ):
            render.append( timing['rendered_frames'] * pixels / render_seconds )
            overhead.append( max( 0.0, ( timing.get( 'total_seconds' ) or 0 ) - detect_seconds - render_seconds ) )
    return {
        'detect': statistics.median( detect ) if detect else None,
        'render': statistics.median( render ) if render else None,
        'overhead': statistics.median( overhead ) if overhead else 0.0,
        'detect_runs': len( detect ),
        'render_runs': len( render ),
    }


def window_seconds( duration, preview_mode_enabled, preview_max_seconds ):
    """The seconds of a file a run works on: all of it, or the preview slice."""
    duration = max( 0.0, float( duration or 0 ) )
    if preview_mode_enabled and preview_max_seconds:
        return min( duration, float( preview_max_seconds ) )
    return duration


def file_seconds( work, measured ):
    """
    Seconds for one file, or None when a rate it needs is unknown.

    work: { seconds, fps, width, height, sample_fps, sizes, detect_cache }
    measured: rates()'s answer.
    """
    seconds = work['seconds']
    detect = 0.0
    if work.get( 'detect_cache' ) != CACHED:
        if not measured['detect']:
            return None
        detect = seconds * work['sample_fps'] * work['sizes'] / measured['detect']
    if not measured['render']:
        return None
    render = seconds * work['fps'] * work['width'] * work['height'] / measured['render']
    return detect + render + measured['overhead']


def comparable_rows( store, limit=200 ):
    """The timings of the latest processed runs, newest first, from the catalog's runs table."""
    rows = store.read( "SELECT timing FROM runs WHERE outcome = 'processed' AND timing IS NOT NULL"
                       " ORDER BY ended_at DESC, id DESC LIMIT ?", ( limit, ) )
    out = []
    for row in rows:
        try:
            out.append( json.loads( row['timing'] ) )
        except ( TypeError, ValueError ):
            continue
    return out
