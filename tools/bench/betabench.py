#!/usr/bin/env python3
"""
betabench.py - BetaSuite's measurement harness.

One script for every "we should measure that before deciding" question.
Each subcommand answers a specific configuration question with numbers
from YOUR hardware and YOUR footage, and prints the setting it suggests.

    decode       Is per-sample seeking really slower than sequential
                 grab/retrieve on this box, for these files? Decides how
                 much the detection-stage work is worth.
    detect       What does one sampled frame actually cost, per backend,
                 per variant, per picture size, split into decode /
                 preprocess / inference / postprocess? Decides
                 picture_sizes, model_variant and nn_batch_size.
    render       What does one censored box cost per frame, per style?
                 Decides censor_style weights and
                 blur_fast_approximation.
    geometry     What box areas and aspect ratios does each label
                 actually produce? Suggests min/max_area_fraction and
                 min/max_aspect_ratio for the geometry sanity filter.
    suppression  For each overlapping cross-label pair, what are the IoU
                 and score-margin distributions? Suggests class_suppression
                 min_iou and margin.
    dedup        When several picture sizes are configured, how much do
                 their detections overlap? Suggests
                 cross_size_dedup['iou_threshold'].
    hysteresis   What does each label's score distribution look like?
                 Suggests min_prob and min_prob_continue.
    structure    Per source file, not pooled: how fast does it cut, how
                 many subjects are on screen, where do they sit in frame?
                 Recommends a source folder per file, in uncensored_vids/
                 and in the staging tree (resources/source/), and with
                 --write-move-script writes the safe moves as a script.
    structure-scan
                 Whole-file shot-cut scans for every source file not yet
                 scanned, in parallel on the CPU, written to the cache the
                 next real run reads. --geometry-slices N also runs
                 detection (no render) on N slices spread through each
                 file, which is what split-screen is recognised from.
    all          Every read-only subcommand, in order.

RUNNING IT
    python3 tools/bench/betabench.py all
    python3 tools/bench/betabench.py decode --video ../resources/uncensored_vids/clip.mp4
    python3 tools/bench/betabench.py detect --backends nudenet_v3 --sizes 320 640
    python3 tools/bench/betabench.py suppression --backend nudenet_v3

    Single command, no setup. Any step that needs a decision from you
    stops and asks (pass --yes to accept the defaults and never prompt,
    which is what an unattended run wants).

OUTPUT
    Everything - including anything a subprocess writes - goes to
        ../output/benchmarks/<timestamp>/betabench.log
    at --log-level (default debug), and to your terminal at
    --console-level (default info), so the file keeps the detail while
    the console stays readable. Machine-readable results are written
    alongside it as results.json.

    geometry / suppression / dedup / hysteresis read the detection
    caches a real run already produced, and structure reads those plus
    the shot-cut caches. They do not run the model, so
    they are fast and they describe exactly what your last real run saw.
    Run betatv.py first if the cache is empty.
"""

import argparse
import json
import logging
import math
import glob
import os
import statistics
import sys
import time

# tools/bench/../.. so 'import betaconfig' resolves however this is invoked.
sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.dirname(
    os.path.abspath( __file__ ) ) ) ) )

import betaconfig
import betaconst
import betautils_cache_paths as bu_cache
import betautils_config as bu_config
import betautils_detector as bu_detector
import betautils_hash as bu_hash
import betautils_track as bu_track
import betautils_tuning as bu_tuning
import betautils_video as bu_video


TRACE = 5
LEVEL_NAMES = {
    'trace': TRACE,
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warn': logging.WARNING,
    'error': logging.ERROR,
}

CORE_DIR = os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def build_logger( run_dir, log_level_name, console_level_name ):
    """
    A logger writing everything to run_dir/betabench.log and a filtered
    view to the terminal.

    Args:
        run_dir: This run's own output directory.
        log_level_name: Level for the file.
        console_level_name: Level for the terminal.

    Returns:
        A configured logging.Logger.
    """
    logging.addLevelName( TRACE, 'TRACE' )
    logger = logging.getLogger( 'betabench' )
    logger.propagate = False
    logger.setLevel( TRACE )
    for handler in list( logger.handlers ):
        logger.removeHandler( handler )

    os.makedirs( run_dir, exist_ok=True )
    file_handler = logging.FileHandler( os.path.join( run_dir, 'betabench.log' ), encoding='UTF-8' )
    file_handler.setLevel( LEVEL_NAMES[ log_level_name ] )
    file_handler.setFormatter( logging.Formatter( '%(asctime)s [%(levelname)-5s] %(message)s' ) )
    logger.addHandler( file_handler )

    console_handler = logging.StreamHandler( sys.stdout )
    console_handler.setLevel( LEVEL_NAMES[ console_level_name ] )
    console_handler.setFormatter( logging.Formatter( '%(message)s' ) )
    logger.addHandler( console_handler )
    return logger


def confirm( logger, question, assume_yes ):
    """
    Ask the operator a yes/no question, unless --yes was passed.

    Every phase that needs a decision routes through here rather than
    assuming one, so an interactive run never surprises you and an
    unattended run never blocks.
    """
    if assume_yes:
        logger.info( "%s [auto-yes]"%(question) )
        return True
    logger.info( question )
    try:
        answer = input( "  [y/N] " ).strip().lower()
    except EOFError:
        return False
    return answer in ( 'y', 'yes' )


# ---------------------------------------------------------------------------
# Small stats helpers
# ---------------------------------------------------------------------------

def describe( values ):
    """
    Summarise a list of numbers.

    Returns:
        A dict with n, min, p1, p25, median, p75, p99, max and mean, or
        {'n': 0} for an empty list.
    """
    if not values:
        return { 'n': 0 }
    ordered = sorted( values )
    return {
        'n': len( ordered ),
        'min': ordered[0],
        'p1': bu_cache.percentile( ordered, 0.01 ),
        'p25': bu_cache.percentile( ordered, 0.25 ),
        'median': statistics.median( ordered ),
        'p75': bu_cache.percentile( ordered, 0.75 ),
        'p99': bu_cache.percentile( ordered, 0.99 ),
        'max': ordered[-1],
        'mean': statistics.fmean( ordered ),
    }


def format_describe( summary, unit='', fmt='%.4f' ):
    """One-line rendering of describe()'s output."""
    if not summary.get( 'n' ):
        return 'no samples'
    parts = [ 'n=%d'%(summary['n']) ]
    for key in ( 'p1', 'p25', 'median', 'p75', 'p99', 'max' ):
        parts.append( '%s=%s%s'%(key, fmt%(summary[key]), unit) )
    return ' '.join( parts )


def find_sample_videos( limit ):
    """
    Videos to measure against, newest-looking first.

    Returns:
        A list of paths under betaconst.video_path_uncensored.
    """
    found = []
    for root, _dirs, file_names in os.walk( betaconst.video_path_uncensored ):
        for fname in sorted( file_names ):
            if fname.lower().endswith( VIDEO_EXTENSIONS ):
                found.append( os.path.join( root, fname ) )
    return found[:limit]


def load_cached_detections( backend_name, logger, preview_suffix=None, picture_sizes=None ):
    """
    Every cached raw detection for one backend, across every video.

    Reads the detection caches a real run already wrote, so this is fast
    and describes exactly what that run saw. Falls back to preview
    caches when no full-run cache exists.

    Args:
        backend_name: Which backend's caches to read.
        logger: The bench logger.
        preview_suffix: Force a specific preview suffix, or None to try
            real caches first.
        picture_sizes: Sizes to read, or None to resolve this backend's
            own. Pass explicitly to analyse caches written at a size the
            backend is no longer configured for - e.g. reading a
            nudenet_v3 640m run's caches while the config has been
            switched back to 320n.

    Returns:
        A (raw_boxes, file_count) pair.
    """
    picture_sizes = bu_cache.resolve_picture_sizes( picture_sizes, backend_name )
    fps = betaconfig.video_censor_fps
    min_prob = getattr( betaconfig, 'global_min_prob', betaconst.global_min_prob )

    # discover_full_run_videos returns a (result, preview_used_for)
    # pair. Unpacking it is not optional: a bare tuple is always truthy
    # and has no .items(), so treating it as the dict alone turned "no
    # caches yet" into an AttributeError and never produced the
    # intended warning.
    discovered, _preview_used_for = bu_cache.discover_full_run_videos(
        picture_sizes, fps, min_prob, backend_name, include_preview=True )
    if not discovered:
        logger.warning( "no detection caches found for backend=%s at picture_sizes=%s fps=%s "
                        "min_prob=%.3f - run betatv.py first (a --preview run is enough to get "
                        "started)"%(backend_name, picture_sizes, fps, min_prob) )
        return [], 0

    raw_boxes = []
    for file_hash, cache_paths in discovered.items():
        for cache_path in cache_paths:
            try:
                loaded = bu_hash.read_json( cache_path )
            except Exception as err:
                logger.warning( "unreadable cache %s: %s"%(cache_path, err) )
                continue
            # Tag each box with its source video. Box geometry is in
            # that video's native pixels, so anything expressed as a
            # fraction of the frame needs to know WHICH frame - see
            # _frame_areas_for.
            for raw in loaded:
                raw['_file_hash'] = file_hash
            raw_boxes.extend( loaded )
    logger.debug( "loaded %d cached detection(s) across %d video(s) for backend=%s"%(
        len( raw_boxes ), len( discovered ), backend_name ) )
    return raw_boxes, len( discovered )


# ---------------------------------------------------------------------------
# decode: seek vs sequential
# ---------------------------------------------------------------------------

def bench_decode( args, logger ):
    """
    Measure per-sample seeking against sequential grab/retrieve.

    This is the measurement behind BetaSuite's single largest
    performance change. The detection loop used to seek to each sampled
    frame; it now decodes sequentially and skips unwanted frames with
    grab(). On a long-GOP stream a seek makes the decoder restart from
    the previous keyframe, so asking for every third frame can cost more
    than decoding all of them - but how much more depends entirely on
    the build of OpenCV, the codec, and the GOP length, which is why
    this measures rather than assumes.

    Reports the per-sample cost of each strategy and the speedup.
    """
    import cv2

    videos = ( [ args.video ] if args.video else find_sample_videos( args.max_videos ) )
    if not videos:
        logger.error( "no videos found under %s - pass --video PATH"%(betaconst.video_path_uncensored) )
        return {}

    sample_fps = args.sample_fps or betaconfig.video_censor_fps
    results = []

    for path in videos:
        info = bu_video.probe_stream_info( path )
        logger.info( "decode: %s (%s %sx%s)"%(
            os.path.basename( path ), info.get( 'codec_name', '?' ),
            info.get( 'width', '?' ), info.get( 'height', '?' ) ) )

        capture = cv2.VideoCapture( path )
        vid_fps = capture.get( cv2.CAP_PROP_FPS )
        num_frames = capture.get( cv2.CAP_PROP_FRAME_COUNT )
        capture.release()
        if not vid_fps:
            logger.warning( "  unreadable frame rate, skipping" )
            continue

        limit_seconds = args.seconds
        limit_frames = min( num_frames, limit_seconds * vid_fps )

        # Strategy A: one seek per sample, the earlier behaviour.
        capture = cv2.VideoCapture( path )
        started = time.perf_counter()
        seek_samples = 0
        try:
            sample_index = 0
            while True:
                target = bu_video.sample_frame_index( sample_index, 0.0, vid_fps, sample_fps )
                if target >= limit_frames:
                    break
                capture.set( cv2.CAP_PROP_POS_FRAMES, target )
                retrieved, _frame = capture.read()
                if not retrieved:
                    break
                seek_samples += 1
                sample_index += 1
        finally:
            capture.release()
        seek_seconds = time.perf_counter() - started

        # Strategy B: sequential grab, retrieve only on samples.
        capture = cv2.VideoCapture( path )
        started = time.perf_counter()
        sequential_samples = 0
        try:
            for _index, _t, _frame in bu_video.iter_sampled_frames(
                    capture, vid_fps, sample_fps, 0.0, limit_frames, limit_seconds ):
                sequential_samples += 1
        finally:
            capture.release()
        sequential_seconds = time.perf_counter() - started

        seek_ms = 1000 * seek_seconds / max( seek_samples, 1 )
        sequential_ms = 1000 * sequential_seconds / max( sequential_samples, 1 )
        speedup = seek_ms / sequential_ms if sequential_ms else float( 'inf' )

        logger.info( "  seek-per-sample   %7.2fs  %4d samples  %7.2f ms/sample"%(
            seek_seconds, seek_samples, seek_ms ) )
        logger.info( "  grab+retrieve     %7.2fs  %4d samples  %7.2f ms/sample"%(
            sequential_seconds, sequential_samples, sequential_ms ) )
        logger.info( "  sequential is %.1fx faster here"%(speedup) )

        results.append( {
            'video': os.path.basename( path ),
            'codec': info.get( 'codec_name' ),
            'vid_fps': vid_fps,
            'sample_fps': sample_fps,
            'seek_ms_per_sample': round( seek_ms, 3 ),
            'sequential_ms_per_sample': round( sequential_ms, 3 ),
            'speedup': round( speedup, 2 ),
        } )

    if results:
        speedups = [ r['speedup'] for r in results ]
        logger.info( "" )
        logger.info( "VERDICT: sequential decoding is %.1fx faster on median across %d file(s). "
                     "BetaSuite already uses it everywhere; this confirms it on your hardware."%(
            statistics.median( speedups ), len( results ) ) )
    return { 'decode': results }


# ---------------------------------------------------------------------------
# detect: per-stage detection cost
# ---------------------------------------------------------------------------

def bench_detect( args, logger ):
    """
    Measure what one sampled frame costs, split by stage.

    Runs the real adapter on real frames, so the numbers include this
    box's actual execution provider. The postprocess column is what the
    vectorised anchor decode replaced - at a 1280 blob the Python loop it
    replaced was roughly 97ms per frame on its own.

    Reports, per backend / variant / size: decode, preprocess+inference
    (one batched call), postprocess, and the implied throughput. Use it
    to choose picture_sizes, model_variant and nn_batch_size.
    """
    import cv2

    videos = ( [ args.video ] if args.video else find_sample_videos( 1 ) )
    if not videos:
        logger.error( "no videos found under %s - pass --video PATH"%(betaconst.video_path_uncensored) )
        return {}
    path = videos[0]

    backends = args.backends or bu_detector.registered_backend_names()
    results = []

    for backend_name in backends:
        module = bu_detector.get_detector( backend_name )
        # Default to EVERY variant this backend declares, not just the
        # one config has selected. Measuring only the selected variant
        # is how a 640m export goes unmeasured beside a 320n run: the
        # question "which variant should I use" cannot be answered by a
        # sweep that only ran one of them.
        variants = args.variants or bu_detector.variant_names( backend_name ) or [ None ]
        for variant in variants:
            if variant:
                os.environ[ bu_detector._VARIANT_OVERRIDE_ENV_VAR ] = variant
            elif bu_detector._VARIANT_OVERRIDE_ENV_VAR in os.environ:
                del os.environ[ bu_detector._VARIANT_OVERRIDE_ENV_VAR ]

            try:
                session = module.get_session()
            except Exception as err:
                logger.warning( "cannot load %s%s: %s"%(
                    backend_name, ' (%s)'%(variant) if variant else '', err ) )
                continue

            # Each variant is timed at ITS OWN native size by default.
            # Pairing every variant with every size mostly measures
            # configurations nobody should run - a 320n export fed a
            # 640 blob is slower AND less accurate - so the useful
            # default sweep is one size per variant.
            native_size = bu_detector.native_size_for_variant( backend_name, variant ) if variant else None
            if args.sizes:
                sizes = args.sizes
            elif native_size:
                sizes = [ native_size ]
            else:
                sizes = bu_detector.get_picture_sizes( backend_name )
            batch_sizes = args.batch_sizes or [ bu_detector.get_nn_batch_size( backend_name ) ]

            for size in sizes:
                for batch_size in batch_sizes:
                    label = "%s%s size=%d batch=%d"%(
                        backend_name, '/%s'%(variant) if variant else '', size, batch_size )
                    logger.info( "detect: %s" % label )

                    capture = cv2.VideoCapture( path )
                    vid_fps = capture.get( cv2.CAP_PROP_FPS ) or 30.0
                    frames = []
                    timestamps = []
                    decode_started = time.perf_counter()
                    try:
                        for index, timestamp, frame in bu_video.iter_sampled_frames(
                                capture, vid_fps, betaconfig.video_censor_fps,
                                max_seconds=args.seconds ):
                            frames.append( frame )
                            timestamps.append( timestamp )
                            if len( frames ) >= args.max_frames:
                                break
                    finally:
                        capture.release()
                    decode_seconds = time.perf_counter() - decode_started

                    if not frames:
                        logger.warning( "  no frames decoded, skipping" )
                        continue

                    # Warm-up: the first inference pays graph/kernel init.
                    module.raw_boxes_for_imgs( frames[:1], size, session, timestamps[:1] )

                    detect_started = time.perf_counter()
                    detections = 0
                    for start in range( 0, len( frames ), batch_size ):
                        batch = frames[ start : start+batch_size ]
                        batch_times = timestamps[ start : start+batch_size ]
                        detections += len( module.raw_boxes_for_imgs(
                            batch, size, session, batch_times ) )
                    detect_seconds = time.perf_counter() - detect_started

                    frame_count = len( frames )
                    decode_ms = 1000 * decode_seconds / frame_count
                    detect_ms = 1000 * detect_seconds / frame_count
                    total_ms = decode_ms + detect_ms

                    logger.info( "  decode %6.2f ms/frame   detect %7.2f ms/frame   "
                                 "total %7.2f ms/frame   %5.1f frames/s   %d detection(s)"%(
                        decode_ms, detect_ms, total_ms, 1000/total_ms if total_ms else 0, detections ) )
                    logger.info( "  providers: %s"%(bu_detector.session_provider_summary( session )) )

                    results.append( {
                        'backend': backend_name,
                        'variant': variant,
                        'size': size,
                        'batch_size': batch_size,
                        'frames': frame_count,
                        'decode_ms_per_frame': round( decode_ms, 3 ),
                        'detect_ms_per_frame': round( detect_ms, 3 ),
                        'total_ms_per_frame': round( total_ms, 3 ),
                        'detections': detections,
                        'providers': bu_detector.session_provider_summary( session ),
                    } )

    os.environ.pop( bu_detector._VARIANT_OVERRIDE_ENV_VAR, None )

    if results:
        logger.info( "" )
        best = min( results, key=lambda row: row['total_ms_per_frame'] )
        logger.info( "FASTEST: %s%s at size %d, batch %d - %.2f ms/frame"%(
            best['backend'], '/%s'%(best['variant']) if best['variant'] else '',
            best['size'], best['batch_size'], best['total_ms_per_frame'] ) )
        logger.info( "Speed is only half the decision: check detections-per-frame too, and look at "
                     "real output before choosing. A configuration that is fast because it detects "
                     "nothing is not the one you want." )
        if any( 'CUDA' not in ( row['providers'] or '' ) for row in results ) \
                and getattr( betaconfig, 'gpu_enabled', 0 ):
            logger.warning( "gpu_enabled is set but at least one session ran without "
                            "CUDAExecutionProvider - those rows are CPU numbers." )
    return { 'detect': results }


# ---------------------------------------------------------------------------
# render: per-style censoring cost
# ---------------------------------------------------------------------------

def bench_render( args, logger ):
    """
    Measure what one censored box costs per frame, per configured style.

    Every style variant in betaconfig gets timed at a few representative
    box sizes on synthetic image data, so the answer does not depend on
    having particular footage to hand. Use it to see which style
    variants are expensive before weighting them heavily, and to confirm
    what blur_fast_approximation buys on this box.
    """
    import numpy as np
    import betautils_censor as bu_censor

    rng = np.random.default_rng( 0 )

    styles = []
    default_style = getattr( betaconfig, 'default_censor_style', None )
    if default_style:
        entries = default_style if isinstance( default_style, list ) else [ default_style ]
        styles.extend( ( 'default_censor_style', entry ) for entry in entries )
    # BOTH override tiers, because a per-label censor_style can live in
    # either. Reading only the shared betaconfig.item_overrides made this
    # table silently omit every style configured inside
    # detector_backend[<backend>]['item_overrides'] - it happened to be
    # complete only because the styles in use sat in the shared tier.
    # An omitted style is an un-measured render cost, which is the one
    # thing this subcommand exists to report.
    override_tiers = [ ( 'item_overrides', getattr( betaconfig, 'item_overrides', {} ) ) ]
    for backend_name in bu_detector.registered_backend_names():
        backend_overrides = bu_detector.get_backend_config( backend_name ).get( 'item_overrides' )
        if backend_overrides:
            override_tiers.append(
                ( "detector_backend[%r]"%(backend_name,), backend_overrides ) )

    seen_styles = set()
    for tier_name, overrides in override_tiers:
        for label, override in overrides.items():
            configured = override.get( 'censor_style' )
            if not configured:
                continue
            entries = configured if isinstance( configured, list ) else [ configured ]
            for entry in entries:
                # One label can appear in both tiers; the resolved value
                # is what renders, so do not time the same (label, style)
                # twice just because two tiers mention it.
                fingerprint = ( label, repr( sorted( entry.items() ) ) )
                if fingerprint in seen_styles:
                    continue
                seen_styles.add( fingerprint )
                styles.append( ( label, entry ) )

    # Collapse style entries that cost the same to render.
    #
    # Render cost depends on what the pixel work is (type, method/pattern,
    # strength, feather, shape) and how big the box is. It does NOT depend
    # on width_area_safety / height_area_safety / weight: area safety
    # resizes the box, and box size is already swept separately below, so
    # two entries differing only in padding are the same measurement taken
    # twice. exposed_breast alone carries 33 entries whose distinct render
    # costs number about a third of that, and every duplicate is a full
    # trials x repeats burst - the single largest cost in this subcommand.
    #
    # The collapsed-away entries are counted and reported, so a reader can
    # see the table covers fewer rows than betaconfig has styles, and why.
    def _cost_key( owner, style ):
        return (
            style.get( 'type' ),
            style.get( 'pattern' ) or style.get( 'method' ) or '',
            style.get( 'strength' ),
            round( float( style.get( 'feather', 0 ) ), 4 ),
            bu_censor.resolve_censor_shape(
                style, getattr( betaconfig, 'default_censor_shape', 'box' ) ),
            # A sticker's cost is dominated by which directory it loads
            # from, not by its padding, so keep that in the key.
            style.get( 'dir' ),
            # Bars draw a filled span whose cost scales with thickness.
            style.get( 'thickness' ),
            owner,
        )

    deduped, seen, collapsed = [], set(), 0
    for owner, style in styles:
        key = _cost_key( owner, style )
        if key in seen:
            collapsed += 1
            continue
        seen.add( key )
        deduped.append( ( owner, style ) )
    if collapsed:
        logger.info( "%d of %d configured style entries collapse onto %d distinct render "
                     "costs (entries differing only in area safety or weight render "
                     "identically); timing the distinct ones."%(
                         collapsed, len( styles ), len( deduped ) ) )
    styles = deduped

    box_sizes = args.box_sizes or [ 120, 240, 400 ]
    repeats = args.repeats
    results = []

    # Frames are sized to the box rather than to 1920x1080.
    #
    # Every renderer in betautils_censor takes explicit x/y/w/h and only
    # touches the box plus its feather ring, so the surrounding pixels
    # contribute nothing to the measurement - but frame.copy() before each
    # timed burst copies all of them. At 1080p that is 6.2MB per burst,
    # and across the whole subcommand it was over 100GB of memcpy timing
    # nothing. The frame here is the box plus a generous margin, which
    # measures the same per-box cost with ~20x less copying.
    #
    # The margin must clear the feather ring and any style that draws
    # outside the box (a bar spans wider than its box). MARGIN_FACTOR is
    # deliberately loose: an undersized frame would clip the render and
    # silently understate the cost, which is a worse failure than copying
    # a few more pixels than strictly needed.
    MARGIN_FACTOR = 1.5
    MIN_MARGIN_PX = 64
    frame_cache = {}

    def _frame_for( box_size, feather ):
        margin = max( MIN_MARGIN_PX,
                      int( box_size * ( MARGIN_FACTOR - 1.0 + float( feather or 0 ) ) ) )
        side = box_size + 2 * margin
        if side not in frame_cache:
            frame_cache[side] = rng.integers( 0, 255, ( side, side, 3 ), dtype=np.uint8 )
        return frame_cache[side], margin

    for owner, style in styles:
        shape = bu_censor.resolve_censor_shape(
            style, getattr( betaconfig, 'default_censor_shape', 'box' ) )
        feather = style.get( 'feather', 0 )
        for box_size in box_sizes:
            frame, margin = _frame_for( box_size, feather )
            box = {
                'x': margin, 'y': margin, 'w': box_size, 'h': box_size,
                'censor_style': style, 'censor_shape': shape,
                'censor_sticker_seed': 0.5, 'label': owner,
                'score': 0.9, 'start': 0.0, 'end': 1.0,
            }
            bu_censor.clear_mask_cache()
            # One untimed call so the shape-mask and sticker caches are
            # warm; a per-frame cost with a cold cache is not the cost a
            # real render pays after the first frame.
            bu_censor.censor_image( frame.copy(), box )

            # Several independent trials, reported as their MEDIAN.
            # One timed burst is not enough on a machine doing anything
            # else: the first version of this took a single mean of 20
            # repeats and reported 0.27 and 1.12 ms for two identical
            # configurations in the same run, a 4x spread that made
            # every small difference in the table unreadable. The median
            # of several trials is stable; the spread is printed so a
            # difference smaller than the noise can be recognised as
            # one rather than acted on.
            trials = []
            for _trial in range( max( 1, args.trials ) ):
                working = frame.copy()
                started = time.perf_counter()
                for _ in range( repeats ):
                    bu_censor.censor_image( working, box )
                trials.append( 1000 * ( time.perf_counter() - started ) / repeats )
            trials.sort()
            elapsed_ms = statistics.median( trials )
            spread = ( trials[-1] / trials[0] ) if trials[0] > 0 else 0.0

            descriptor = "%s/%s"%(style['type'], style.get( 'pattern' ) or style.get( 'method' ) or '')
            logger.debug( "render: %-18s %-14s %3dpx feather=%.2f -> %6.2f ms/box/frame "
                          "(median of %d, %.2f-%.2f)"%(
                              owner, descriptor, box_size, feather, elapsed_ms,
                              len( trials ), trials[0], trials[-1] ) )
            results.append( {
                'owner': owner,
                'type': style['type'],
                'variant': descriptor,
                'strength': style.get( 'strength' ),
                'box_size': box_size,
                'feather': feather,
                'ms_per_box_per_frame': round( elapsed_ms, 3 ),
                'ms_min': round( trials[0], 3 ),
                'ms_max': round( trials[-1], 3 ),
                'trials': len( trials ),
                'spread': round( spread, 2 ),
            } )

    if results:
        logger.info( "" )
        logger.info( "Per-box cost at each size, worst first (ms per box per rendered frame):" )
        for row in sorted( results, key=lambda r: -r['ms_per_box_per_frame'] )[:12]:
            logger.info( "  %-18s %-14s %3dpx  %6.2f ms   (%.2f-%.2f over %d trial(s))"%(
                row['owner'], row['variant'], row['box_size'], row['ms_per_box_per_frame'],
                row['ms_min'], row['ms_max'], row['trials'] ) )

        worst_spread = max( row['spread'] for row in results )
        if worst_spread > 1.5:
            logger.info( "" )
            logger.info( "measurement noise: the widest per-configuration spread was %.1fx. Treat any "
                         "difference smaller than that as noise, and raise --trials or --repeats "
                         "before acting on a close call."%(worst_spread) )
        worst = max( results, key=lambda r: r['ms_per_box_per_frame'] )
        fps = betaconfig.video_censor_fps
        logger.info( "" )
        logger.info( "At a 30fps source, a single live box of the worst style above adds roughly "
                     "%.1fs of render time per second of video."%(worst['ms_per_box_per_frame']*30/1000) )
        logger.info( "blur_fast_approximation is currently %s."%(
            'ON' if getattr( betaconfig, 'blur_fast_approximation', True ) else 'OFF' ) )
        logger.info( "Re-run with --no-blur-approximation to see what it is saving. (fps=%s)"%(fps) )
    return { 'render': results }


# ---------------------------------------------------------------------------
# geometry: suggest the sanity filter's bounds
# ---------------------------------------------------------------------------

def _frame_areas_for( raw_boxes, args, logger ):
    """
    The real frame area behind each cached detection.

    Box coordinates are in their source video's native pixels, so an
    area expressed as a fraction of the frame is only meaningful against
    THAT video's frame. This used to assume 1920x1080 for everything and
    warn about it, which quietly mis-scales every number from a video
    that is not 1920x1080 - and a portrait phone clip at 1080x1920 has
    the same pixel count, so even the warning would not have caught it:
    the area fractions come out right and the aspect bounds come out
    inverted.

    Analogy: the detections are measurements in inches; the frame size
    is the ruler. Using one ruler for footage shot on several is how a
    label ends up with an area bound that rejects real detections.

    Resolves each cached file_hash back to its source video and reads
    the real dimensions once per video. Falls back to --frame-width /
    --frame-height, then to 1920x1080, for a video that can no longer be
    found, and says how many boxes that affected.

    Returns:
        A ( {file_hash: frame_area}, fallback_area ) pair.
    """
    import cv2

    fallback_width = args.frame_width or 1920
    fallback_height = args.frame_height or 1080
    fallback_area = float( fallback_width * fallback_height )

    file_hashes = { raw.get( '_file_hash' ) for raw in raw_boxes }
    file_hashes.discard( None )
    if not file_hashes:
        logger.warning( "cached detections carry no source video tag; assuming %dx%d for every "
                        "area fraction"%(fallback_width, fallback_height) )
        return {}, fallback_area

    hash_to_path = bu_cache.build_hash_to_video_path( file_hashes )
    areas = {}
    for file_hash in sorted( file_hashes ):
        path = hash_to_path.get( file_hash )
        if not path:
            continue
        capture = cv2.VideoCapture( path )
        width = int( capture.get( cv2.CAP_PROP_FRAME_WIDTH ) )
        height = int( capture.get( cv2.CAP_PROP_FRAME_HEIGHT ) )
        capture.release()
        if width and height:
            areas[file_hash] = float( width * height )
            logger.debug( "frame size for %s: %dx%d"%(file_hash, width, height) )

    missing = len( file_hashes ) - len( areas )
    if missing:
        logger.warning( "%d of %d source video(s) could not be measured (moved or deleted); their "
                        "detections fall back to %dx%d"%(
                            missing, len( file_hashes ), fallback_width, fallback_height ) )
    distinct = len( set( areas.values() ) )
    if distinct > 1:
        logger.info( "source videos span %d distinct frame sizes; area fractions are computed "
                     "against each video's own frame, not a single assumed one"%(distinct) )
    return areas, fallback_area


def _bench_geometry_one( args, logger, config, raw_boxes ):
    """
    Per-label box area and aspect-ratio distributions from cached
    detections, with suggested geometry-filter bounds.

    WHAT THE GEOMETRY FILTER IS FOR
        A detector occasionally latches onto the wrong thing at the
        wrong scale, confidently. An 'exposed_vulva' box covering 40% of
        a 1080p frame is the model having found a torso; a six-pixel box
        is noise. Confidence cannot separate those from real detections,
        because these misfires are often confident. Shape can.

        Think of it as a coin sorter. Confidence asks "does this look
        like a coin?". The geometry filter asks "is it coin-sized?" -
        a different question, and a cheap one, that catches things
        the first question waves through.

    WHAT THE SUGGESTION MEANS
        Bounds are proposed at the 1st and 99th percentile of what your
        own footage produced, widened by a safety factor. That keeps
        roughly the middle 98% of real detections and clips only the
        extreme tails. Nothing is written for you: read the numbers,
        look at what sits outside them, then set the values by hand.

        These are suggestions from ONE sample of footage. Framing
        changes the distribution - close-up material and wide-shot
        material legitimately produce different box areas for the same
        label - so a bound derived from one library is not a universal
        constant.
    """
    backend_name = config.backend_name

    frame_areas, fallback_area = _frame_areas_for( raw_boxes, args, logger )

    by_label = {}
    for raw in raw_boxes:
        frame_area = frame_areas.get( raw.get( '_file_hash' ), fallback_area )
        entry = by_label.setdefault( raw['class_id'], { 'area': [], 'aspect': [] } )
        entry['area'].append( ( raw['w'] * raw['h'] ) / frame_area )
        if raw['h']:
            entry['aspect'].append( raw['w'] / raw['h'] )

    censored_labels = set( getattr( betaconfig, 'items_to_censor', [] ) )
    results = []
    logger.info( "" )
    logger.info( "Geometry distributions (area as a fraction of each detection's own source frame):" )

    for label in sorted( by_label ):
        area_summary = describe( by_label[label]['area'] )
        aspect_summary = describe( by_label[label]['aspect'] )
        marker = '*' if label in censored_labels else ' '
        logger.info( "%s %-18s area   %s"%(marker, label, format_describe( area_summary, fmt='%.5f' )) )
        logger.info( "  %-18s aspect %s"%('', format_describe( aspect_summary, fmt='%.2f' )) )

        suggestion = {}
        if area_summary['n'] >= args.min_samples:
            suggestion['min_area_fraction'] = round( area_summary['p1'] / args.widen, 6 )
            suggestion['max_area_fraction'] = round( min( 1.0, area_summary['p99'] * args.widen ), 6 )
            suggestion['min_aspect_ratio'] = round( aspect_summary['p1'] / args.widen, 3 )
            suggestion['max_aspect_ratio'] = round( aspect_summary['p99'] * args.widen, 3 )
        results.append( {
            'label': label, 'censored': label in censored_labels,
            'area': area_summary, 'aspect': aspect_summary, 'suggested': suggestion,
        } )

    logger.info( "" )
    logger.info( "Suggested geometry-filter bounds (p1/p99 widened by %.2fx), for the labels you "
                 "actually censor. Paste into detector_backend[%r]['item_overrides'][<label>]:"%(
        args.widen, backend_name ) )
    for row in results:
        if row['censored'] and row['suggested']:
            logger.info( "    '%s': {"%(row['label']) )
            for key, value in row['suggested'].items():
                logger.info( "        '%s': %s,"%(key, value) )
            logger.info( "    }," )
    logger.info( "" )
    logger.info( "Sanity-check before using these: a bound that rejects a detection you WANT is "
                 "worse than no bound at all. Start with max_area_fraction only - the "
                 "'model found a torso' case is the one this reliably catches." )
    return { 'geometry': results }


# ---------------------------------------------------------------------------
# suppression: suggest class_suppression thresholds
# ---------------------------------------------------------------------------

def _bench_suppression_one( args, logger, config, raw_boxes ):
    """
    Cross-label overlap statistics, with suggested class_suppression
    min_iou and margin per pair.

    HOW A SUPPRESSION RULE WORKS
        "Label L is suppressed by label S when an S detection at the
        same instant overlaps it by at least min_iou and outscores it by
        at least margin." It exists for one specific failure: the model
        reporting the same patch of pixels as two different things, and
        being more confident about the wrong one.

        The analogy is two witnesses describing one object. If they are
        looking at clearly different places (low IoU) both can be right.
        If they are looking at the same place, the one who is markedly
        more certain wins - and 'markedly' is what margin sets. A margin
        of 0 means a bare tie is enough, which is usually too generous.

    WHAT THIS MEASURES
        For every ordered label pair seen overlapping at the same
        instant, the IoU distribution and the score-difference
        distribution. min_iou is proposed at the 25th percentile of
        observed overlaps (so the rule fires on typical overlaps, not
        only extreme ones) and margin at the median score difference
        where the suppressor wins.

    WHEN TO RE-RUN THIS
        Any time something upstream of the numbers changes: a different
        model variant, a different picture size, or a change to
        nms_mode. All three change the box geometry the thresholds are
        fitted to. The shipped nudenet_v3 rules were fitted at size 1280
        with class-agnostic NMS and have NOT yet been re-derived for the
        current 320/per-class configuration.
    """
    backend_name = config.backend_name

    censored_labels = set( getattr( betaconfig, 'items_to_censor', [] ) )
    pair_data = {}

    by_instant = {}
    for index, raw in enumerate( raw_boxes ):
        by_instant.setdefault( raw['t'], [] ).append( index )

    for indices in by_instant.values():
        if len( indices ) < 2:
            continue
        for i in indices:
            subject = raw_boxes[i]
            if subject['class_id'] not in censored_labels:
                continue
            for j in indices:
                if i == j:
                    continue
                other = raw_boxes[j]
                if other['class_id'] == subject['class_id']:
                    continue
                iou = bu_track.intersection_over_union( subject, other )
                if iou <= 0:
                    continue
                key = '%s<-%s'%(subject['class_id'], other['class_id'])
                entry = pair_data.setdefault( key, { 'iou': [], 'margin': [] } )
                entry['iou'].append( iou )
                entry['margin'].append( other['score'] - subject['score'] )

    results = []
    logger.info( "" )
    logger.info( "Cross-label overlaps at the same instant, for censored labels only:" )
    for key in sorted( pair_data, key=lambda k: -len( pair_data[k]['iou'] ) ):
        iou_summary = describe( pair_data[key]['iou'] )
        margin_summary = describe( pair_data[key]['margin'] )
        if iou_summary['n'] < args.min_samples:
            continue
        positive_margins = [ m for m in pair_data[key]['margin'] if m > 0 ]
        win_rate = len( positive_margins ) / iou_summary['n']

        suggested_min_iou = round( max( 0.01, iou_summary['p25'] ), 3 )
        suggested_margin = ( round( statistics.median( positive_margins ), 3 )
                             if positive_margins else 0.0 )

        logger.info( "  %-40s n=%-6d iou p25=%.3f median=%.3f p75=%.3f   "
                     "suppressor wins %.0f%% of the time"%(
            key, iou_summary['n'], iou_summary['p25'], iou_summary['median'],
            iou_summary['p75'], 100*win_rate ) )
        logger.info( "      suggested: { 'suppressed_by': '%s', 'min_iou': %s, 'margin': %s }"%(
            key.split( '<-' )[1], suggested_min_iou, suggested_margin ) )
        if win_rate < 0.2:
            logger.info( "      NOTE: the suppressor rarely outscores this label, so a positive "
                         "margin would make the rule almost never fire. Consider leaving this pair "
                         "out rather than setting margin to 0 to force it." )

        results.append( {
            'pair': key, 'iou': iou_summary, 'margin': margin_summary,
            'suppressor_win_rate': round( win_rate, 4 ),
            'suggested_min_iou': suggested_min_iou,
            'suggested_margin': suggested_margin,
        } )

    logger.info( "" )
    logger.info( "These are starting points, not answers. A suppression rule removes real "
                 "detections when it is wrong, so add one pair at a time and look at the output." )
    return { 'suppression': results }


# ---------------------------------------------------------------------------
# dedup: suggest cross_size_dedup['iou_threshold']
# ---------------------------------------------------------------------------

def _bench_dedup_one( args, logger, config, raw_boxes ):
    """
    How much detections from different picture sizes overlap.

    Only meaningful with more than one entry in picture_sizes. With one
    size, cross-size dedup is a no-op by construction and there is
    nothing to tune.

    Reports, per label, the IoU distribution between same-label
    detections at the same instant that came from DIFFERENT sizes. A
    clean bimodal split - a cluster near 1.0 (the same object seen
    twice) and a cluster near 0 (genuinely different objects) - means
    any threshold in the valley works. Overlapping clusters mean the
    sizes disagree about geometry and the threshold matters.
    """
    backend_name = config.backend_name
    picture_sizes = config.picture_sizes
    if len( picture_sizes ) < 2:
        logger.info( "picture_sizes for %s is %s - only one size, so cross-size dedup is a no-op "
                     "and there is nothing to tune here."%(config.label, picture_sizes) )
        return {}

    by_instant = {}
    for index, raw in enumerate( raw_boxes ):
        by_instant.setdefault( raw['t'], [] ).append( index )

    by_label = {}
    for indices in by_instant.values():
        for position, i in enumerate( indices ):
            for j in indices[ position+1 : ]:
                first, second = raw_boxes[i], raw_boxes[j]
                if first['class_id'] != second['class_id']:
                    continue
                if first.get( 'size', 0 ) == second.get( 'size', 0 ):
                    continue
                by_label.setdefault( first['class_id'], [] ).append(
                    bu_track.intersection_over_union( first, second ) )

    results = []
    logger.info( "" )
    logger.info( "Cross-size same-label IoU, per label:" )
    for label in sorted( by_label ):
        summary = describe( by_label[label] )
        strong = sum( 1 for value in by_label[label] if value >= 0.6 )
        logger.info( "  %-18s %s   %.0f%% at IoU>=0.60"%(
            label, format_describe( summary, fmt='%.3f' ), 100*strong/max( summary['n'], 1 ) ) )
        results.append( { 'label': label, 'iou': summary, 'fraction_above_0_6': strong/max( summary['n'], 1 ) } )

    logger.info( "" )
    logger.info( "Current cross_size_dedup['iou_threshold'] is %s. Raise it if real, distinct "
                 "detections are being merged; lower it if duplicates survive."%(
        getattr( betaconfig, 'cross_size_dedup', {} ).get( 'iou_threshold', 0.60 ) ) )
    return { 'dedup': results }


# ---------------------------------------------------------------------------
# hysteresis: suggest min_prob / min_prob_continue
# ---------------------------------------------------------------------------

def _bench_hysteresis_one( args, logger, config, raw_boxes ):
    """
    Per-label score distributions, with suggested min_prob and
    min_prob_continue.

    WHAT SCORE HYSTERESIS IS
        One threshold makes a flickering decision. A detection sitting
        near min_prob crosses it back and forth frame to frame, so the
        censor blinks even though the thing being censored never went
        anywhere.

        A thermostat has the same problem and the same fix: it starts
        heating at one temperature and stops at a slightly different
        one, so it does not chatter on and off around a single set
        point. min_prob is the temperature that starts the track;
        min_prob_continue is the lower one that keeps it going. A
        detection below min_prob can extend a track that already
        exists, but can never begin one.

    THE SUGGESTION
        min_prob at the 25th percentile of observed scores, and
        min_prob_continue midway between that and global_min_prob. Both
        must stay above global_min_prob, which is a hard floor applied
        before either of them is ever consulted.
    """
    backend_name = config.backend_name

    global_min_prob = getattr( betaconfig, 'global_min_prob', betaconst.global_min_prob )
    censored_labels = set( getattr( betaconfig, 'items_to_censor', [] ) )

    by_label = {}
    for raw in raw_boxes:
        by_label.setdefault( raw['class_id'], [] ).append( raw['score'] )

    results = []
    logger.info( "" )
    logger.info( "Score distributions (global_min_prob = %.3f is already applied):"%(global_min_prob) )
    for label in sorted( by_label ):
        summary = describe( by_label[label] )
        marker = '*' if label in censored_labels else ' '
        logger.info( "%s %-18s %s"%(marker, label, format_describe( summary, fmt='%.3f' )) )

        suggestion = {}
        if label in censored_labels and summary['n'] >= args.min_samples:
            suggested_min_prob = round( max( global_min_prob + 0.01, summary['p25'] ), 3 )
            suggested_continue = round(
                max( global_min_prob + 0.005,
                     global_min_prob + ( suggested_min_prob - global_min_prob ) / 2 ), 3 )
            suggestion = { 'min_prob': suggested_min_prob,
                           'min_prob_continue': suggested_continue }
        results.append( { 'label': label, 'scores': summary, 'suggested': suggestion } )

    logger.info( "" )
    logger.info( "Suggested, for detector_backend[%r]['item_overrides']:"%(backend_name) )
    for row in results:
        if row['suggested']:
            logger.info( "    '%s': { 'min_prob': %s, 'min_prob_continue': %s },"%(
                row['label'], row['suggested']['min_prob'], row['suggested']['min_prob_continue'] ) )
    logger.info( "" )
    logger.info( "A min_prob at the 25th percentile discards the least confident quarter of what "
                 "the model reported for that label. Whether that is right depends on whether that "
                 "quarter was noise or was real - look at output at two or three candidate values "
                 "before committing." )
    return { 'hysteresis': results }



# ---------------------------------------------------------------------------
# Per-configuration dispatch
# ---------------------------------------------------------------------------

def _per_configuration( kind, body ):
    """
    Turn a single-configuration measurement into one that covers them all.

    Every cache-reading subcommand used to measure exactly one backend:
    whichever betaconfig had selected, at whichever size that backend
    was configured for. That is one configuration out of however many
    have actually been run, and the others went unmeasured in silence -
    which is how a full 640m run sat on disk for a day with nobody
    looking at it.

    This wraps a body that measures ONE configuration so the subcommand
    measures every configuration found on disk, each under its own
    heading, with its results tagged by configuration label so a results
    file can never be read as if two variants were one.

    Args:
        kind: The results-dict key this subcommand writes.
        body: f( args, logger, config, raw_boxes ) -> results dict.

    Returns:
        A f( args, logger ) suitable for the SUBCOMMANDS table.
    """
    def run( args, logger ):
        # --variants is forwarded here, not just declared. Without it
        # the flag parsed cleanly and was then ignored, so a run narrowed
        # to one variant still reported on every variant on disk while
        # the header claimed the filter had been applied. A silently
        # wrong report is worse than a crash, because nothing prompts
        # anyone to re-read it.
        configurations = bu_cache.configurations_to_analyse(
            args.sizes,
            backend_names=[ args.backend ] if args.backend else None,
            variant_names=getattr( args, 'variants', None ),
            fps=args.sample_fps or betaconfig.video_censor_fps,
            include_preview=True,
            logger=logger )
        merged = []
        for config in configurations:
            logger.info( "" )
            logger.info( "-" * 72 )
            logger.info( "  configuration: %s"%(config.label) )
            logger.info( "-" * 72 )
            raw_boxes, file_count = load_cached_detections(
                config.backend_name, logger, picture_sizes=config.picture_sizes )
            if not raw_boxes:
                continue
            logger.info( "loaded %d cached detection(s) across %d video(s) for %s"%(
                len( raw_boxes ), file_count, config.label ) )
            produced = body( args, logger, config, raw_boxes ) or {}
            for row in produced.get( kind, [] ):
                if isinstance( row, dict ):
                    row = dict( row )
                    row['configuration'] = config.label
                    row['backend'] = config.backend_name
                    row['variant'] = config.variant
                    row['size'] = config.size
                merged.append( row )
        if not merged:
            logger.warning( "no cached detections found for any configuration - run betatv.py first "
                            "(a --preview run is enough to get started)" )
        return { kind: merged }
    run.__name__ = 'bench_' + kind
    run.__doc__ = body.__doc__
    return run


# --- Folder recommendation -------------------------------------------------
# The structure bench already measures everything needed to say which
# source folder a file belongs in (see profile_by_source_dir in
# CONFIG_REFERENCE.md). These are the boundaries it applies.
#
# The split-screen signature. Panes coexist rather than cut, so a
# split-screen file is invisible to every timing measure and shows up
# spatially instead: several boxes live at once, split evenly between the
# screen halves and staying there. Read off a 9-file sample, where split
# files ran p90 simultaneous 5-9 with left-half near 0.50 and the one
# non-split file with high simultaneity sat at left-half 0.13. A sample
# that small makes these a starting point, not a tuned setting.
SPLIT_MIN_P90_SIMULTANEOUS = 5
SPLIT_LEFT_HALF_RANGE = ( 0.40, 0.60 )

# The compilation boundary: footage that spends most of its runtime in
# shots of a few seconds. Measured by DURATION, never by counting shots.
# A median shot length gets this backwards on real footage: a scene with
# one flurry of twenty 0.1s false cuts in it has a 0.1s median while
# 93% of its runtime is a single long take, and a genuinely fast-cutting
# file measured a LONGER 0.50s median. Ranked by median those two come
# out in the wrong order; ranked by duration share they separate by 5x.
COMPILATION_SHOT_SECONDS = 10.0
COMPILATION_MIN_RUNTIME_FRACTION = 0.5


def _quick_cut_threshold( backend_name=None ):
    """
    The configured short-shot share at which quick_cut takes over.

    Read from betaconfig rather than restated here, so this bench cannot
    recommend a folder the app would then disagree with. Returns
    (threshold, profile_name), or (None, None) when no auto-selectable
    variant sets one.
    """
    profiles = bu_detector.get_profiles( backend_name )
    if profiles.get( 'match_on' ) != 'short_shot_fraction':
        return ( None, None )
    best = ( None, None )
    for name, variant in ( profiles.get( 'variants' ) or {} ).items():
        if variant.get( 'manual_only' ):
            continue
        threshold = variant.get( 'min_short_shot_fraction' )
        if threshold is None:
            continue
        if best[0] is None or threshold < best[0]:
            best = ( threshold, name )
    return best


def _folder_for_profile( profile_name ):
    """
    The folder that maps to a profile, or None.

    Inverts betaconfig.profile_by_source_dir so this reports the folder
    the user actually configured rather than a name invented here. With
    several folders mapping to one profile the shallowest is reported,
    because that is where a new file of that kind would normally go.
    """
    mapping = getattr( betaconfig, 'profile_by_source_dir', None ) or {}
    candidates = [ key for key, name in mapping.items() if name == profile_name ]
    if not candidates:
        return None
    return sorted( candidates, key=lambda key: (
        len( bu_detector._dir_key_parts( key ) ), key ) )[0]


def _write_move_script( path, moves, logger ):
    """
    Write a reviewable shell script that sorts the source tree.

    A script rather than doing the moves here, for two reasons: this
    bench's job is to measure, and the recommendations include a
    compilation boundary read off a small sample. Someone should read
    the reasons before 70 files move.

    Every move is `mv -n`, which never overwrites. The script checks
    each source still exists, counts what it did, and is safe to re-run:
    a second run finds the files already moved and does nothing.
    """
    lines = [
        '#!/bin/bash',
        '# Generated by betabench structure. Sorts source videos into the',
        '# folders betaconfig.profile_by_source_dir maps to profiles.',
        '#',
        '# READ THE REASONS BELOW FIRST. The compilation/scene boundary is a',
        '# heuristic read off a small sample, and split-screen is detected',
        '# from box geometry, not from anything infallible.',
        '#',
        '# Every move is `mv -n`: it will not overwrite an existing file.',
        '# Safe to re-run; already-moved files are skipped.',
        '',
        'set -uo pipefail',
        'moved=0; skipped=0',
        '',
        'move_one() {',
        '    local src="$1" dest_dir="$2"',
        '    if [[ ! -f "$src" ]]; then',
        '        echo "skip (not found): $src"; skipped=$((skipped+1)); return',
        '    fi',
        '    mkdir -p "$dest_dir" || return',
        '    if mv -n "$src" "$dest_dir"/; then',
        '        if [[ -f "$src" ]]; then',
        '            echo "skip (target exists): $src"; skipped=$((skipped+1))',
        '        else',
        '            echo "moved: $src -> $dest_dir/"; moved=$((moved+1))',
        '        fi',
        '    else',
        '        echo "FAILED: $src"; skipped=$((skipped+1))',
        '    fi',
        '}',
        '',
    ]
    for move in moves:
        lines.append( '# %s'%( move['why'], ) )
        lines.append( 'move_one %s %s'%(
            _shell_quote( move['source_path'] ),
            _shell_quote( move['dest_dir'] ) ) )
        lines.append( '' )
    lines += [
        'echo',
        'echo "moved $moved file(s), skipped $skipped"',
        'echo "Re-run betabench structure afterwards to confirm each file"',
        'echo "now sits where its measured structure puts it."',
        '',
    ]
    directory = os.path.dirname( os.path.abspath( path ) )
    if directory:
        os.makedirs( directory, exist_ok=True )
    with open( path, 'w', newline='\n' ) as handle:
        handle.write( '\n'.join( lines ) )
    os.chmod( path, 0o755 )
    logger.info( "move script written to %s (%d move(s)). Read it, then run it."%(
        path, len( moves ) ) )


def _shell_quote( text ):
    """Single-quote a path for the generated script, safely."""
    return "'" + str( text ).replace( "'", "'\\''" ) + "'"


# --- Source tree survey ----------------------------------------------------
# structure works per SOURCE FILE, enumerated from disk, not per detection
# cache. It used to iterate the caches of one detection configuration at
# the global sample rate, so every file a profile had sampled at another
# rate was invisible: 8 of 56 files in one survey, exactly the 8 the app
# had measured as quick_cut. Starting from the files also lets it say
# which ones were never measured instead of silently omitting them.

VIDEO_EXTENSIONS = ( '.mp4', '.mkv', '.avi', '.mov', '.webm', '.m4v' )

# Below this many sampled frames holding a censored box, the spatial
# signals are noise: the file that started this had boxes in ONE frame.
# A judgment call, not a tuned value: one 20s slice at 9 fps is 180
# samples, and 30 is a sixth of that.
SPLIT_MIN_FRAMES_WITH_BOXES = 30


def _source_trees():
    """
    (label, root) for every tree a source video can sit in.

    The working tree betatv processes, then the staging tree
    (video_path_source_backup). The same folder names mean the same
    profiles in both, so a file can be sorted while it waits in staging
    and keep its profile when it is moved across.
    """
    trees = [ ( 'uncensored_vids', betaconst.video_path_uncensored ) ]
    staging = getattr( betaconst, 'video_path_source_backup', None )
    if staging and ( os.path.realpath( staging )
                     != os.path.realpath( betaconst.video_path_uncensored ) ):
        trees.append( ( 'source', staging ) )
    return trees


def _source_videos():
    """Every video file in every source tree, with where it sits."""
    found = []
    for tree_label, root in _source_trees():
        if not os.path.isdir( root ):
            continue
        for dirpath, dirnames, filenames in os.walk( root ):
            dirnames.sort()
            for name in sorted( filenames ):
                if not name.lower().endswith( VIDEO_EXTENSIONS ):
                    continue
                path = os.path.join( dirpath, name )
                relative = os.path.relpath( dirpath, root )
                found.append( {
                    'path': path,
                    'name': name,
                    'tree': tree_label,
                    'tree_root': root,
                    # Original case, for display and for building moves;
                    # the folder RULE is matched case-insensitively.
                    'current_folder': ( '(root)' if relative in ( '.', '' )
                                        else relative.replace( os.sep, '/' ) ),
                } )
    return found


def _censored_boxes_by_hash( file_hashes, labels, backend_name=None, variant_names=None ):
    """
    Censored-label boxes per file, from ONE detection configuration each.

    Any configuration will do for geometry, at any sample rate, preview
    slices included: where boxes sit does not depend on how often they
    were sampled. Mixing configurations would not do: two picture sizes
    or two backends see the same frame, and their boxes would count as
    two simultaneous subjects. So per file this keeps the configuration
    that saw the most distinct sampled frames, with every slice of it.

    backend_name and variant_names narrow which configurations count,
    for --backend and --variants. They used to be honoured through the
    per-configuration loop this replaced, and a flag that parses and is
    then ignored is worse than no flag.

    Returns:
        {file_hash: (boxes, configuration_label)}
    """
    groups = {}
    for path in glob.glob( os.path.join( bu_cache.VID_HASH_DIR, '*.gz' ) ):
        parsed = bu_cache.parse_any_cache_name( path )
        if not parsed or parsed['file_hash'] not in file_hashes:
            continue
        if backend_name and parsed['backend'] != backend_name:
            continue
        if variant_names and bu_cache.configuration_variant_of_cache_filename(
                path ) not in variant_names:
            continue
        key = ( parsed['file_hash'], parsed['backend'], parsed['size'],
                parsed['fps'], parsed['min_prob'], parsed['detection_key'] )
        groups.setdefault( key, [] ).append( path )

    best = {}
    for key, paths in groups.items():
        boxes = []
        for path in paths:
            try:
                boxes.extend( b for b in bu_hash.read_json( path )
                              if b.get( 'class_id' ) in labels )
            except Exception:                              # noqa: BLE001
                continue
        frames = len( { round( float( b.get( 't', 0 ) ), 3 ) for b in boxes } )
        file_hash = key[0]
        if file_hash not in best or frames > best[file_hash][0]:
            label = '%s @%d %gfps'%( key[1], key[2], key[3] )
            best[file_hash] = ( frames, boxes, label )
    return { file_hash: ( boxes, label ) for file_hash, ( _f, boxes, label ) in best.items() }


def _geometry_of( boxes, frame_width, frame_height ):
    """Spatial structure of one file's censored boxes. See bench_structure."""
    import statistics

    per_frame = {}
    for box in boxes:
        key = round( float( box.get( 't', 0 ) ), 3 )
        per_frame[key] = per_frame.get( key, 0 ) + 1
    counts = sorted( per_frame.values() )
    xs = [ float( b.get( 'x', 0 ) ) + float( b.get( 'w', 0 ) ) / 2 for b in boxes ]
    left_half = None
    if frame_width and xs:
        left_half = sum( 1 for x in xs if x < frame_width / 2 ) / len( xs )
    area_fraction = None
    if frame_width and frame_height and boxes:
        area_fraction = statistics.median(
            float( b.get( 'w', 0 ) ) * float( b.get( 'h', 0 ) ) for b in boxes
        ) / float( frame_width * frame_height )
    return {
        'detections': len( boxes ),
        'frames_with_boxes': len( counts ),
        'median_simultaneous': statistics.median( counts ) if counts else 0,
        'p90_simultaneous': counts[int( 0.9 * ( len( counts ) - 1 ) )] if counts else 0,
        'left_half_fraction': round( left_half, 3 ) if left_half is not None else None,
        'median_area_fraction': round( area_fraction, 5 ) if area_fraction else None,
    }


def _measure_source( source, file_hash, boxes, geometry_label, fps, threshold ):
    """One row: timing from the best cached cut scan, geometry from boxes."""
    duration = bu_video.probe_duration_seconds( source['path'] )
    cached = bu_video.cached_shot_cuts( file_hash, fps, threshold, duration )
    if cached is None:
        signals = bu_detector.structure_signals( None, None )
        mid_fraction = None
        evidence = None
    else:
        signals = bu_detector.structure_signals( cached.cuts, cached.scanned_seconds )
        mid_fraction = ( bu_detector.short_shot_fraction_for(
                             cached.cuts, cached.scanned_seconds, COMPILATION_SHOT_SECONDS )
                         if signals.is_measured else None )
        evidence = ( 'whole file' if cached.is_whole_file
                     else 'slice' ) if signals.is_measured else None
    width, height = bu_tuning.frame_size_for_video( source['path'] )
    row = {
        'file_hash': file_hash,
        'source': source['name'],
        'source_path': source['path'],
        'tree': source['tree'],
        'tree_root': source['tree_root'],
        'current_folder': source['current_folder'],
        'duration_seconds': round( duration, 1 ) if duration else None,
        'scanned_seconds': ( round( signals.scanned_seconds, 1 )
                             if signals.scanned_seconds is not None else None ),
        'timing_evidence': evidence,
        'cuts': signals.cut_count,
        'cuts_per_min': ( round( signals.cuts_per_min, 1 )
                          if signals.cuts_per_min is not None else None ),
        'median_shot_seconds': ( round( signals.median_shot_seconds, 2 )
                                 if signals.median_shot_seconds is not None else None ),
        'short_shot_fraction': ( round( signals.short_shot_fraction, 4 )
                                 if signals.short_shot_fraction is not None else None ),
        'mid_shot_fraction': round( mid_fraction, 4 ) if mid_fraction is not None else None,
        'geometry_configuration': geometry_label,
    }
    row.update( _geometry_of( boxes, width, height ) )
    return row


def bench_structure( args, logger ):
    """
    Describe each source video's STRUCTURE, per file, and say which
    source folder it belongs in.

    What this reports, per file:
      scanned, evidence              how much footage the timing numbers
                                     rest on, and whether it was the
                                     whole file or a slice
      cuts/min, <1s, <10s            how fast the footage cuts, as shares
                                     of RUNTIME
      frames, p90, L-half            how many sampled frames held a
                                     censored box, how many boxes share a
                                     frame, which half of the screen
                                     they sit in
      belongs in                     the recommended folder

    Timing comes from shot-cut caches, preferring a whole-file scan
    (betabench structure-scan writes them, and the next real run reuses
    them). Geometry comes from whatever detection caches exist.

    Nothing here changes config or moves a file. --write-move-script
    writes a reviewable script of the moves that pass every safety rule
    in _move_verdict; the rest are listed for review.
    """
    fps = betaconfig.video_censor_fps
    threshold = getattr( betaconfig, 'shot_cut_threshold', 0.5 )
    labels = set( getattr( betaconfig, 'items_to_censor', [] ) )

    sources = _source_videos()
    if not sources:
        logger.warning( "no source videos under %s"%(
            ' or '.join( root for _l, root in _source_trees() ), ) )
        return {}
    logger.info( "found %d source video(s) across %s"%(
        len( sources ), ', '.join( label for label, _r in _source_trees() ) ) )

    started = time.perf_counter()
    for source in sources:
        try:
            source['file_hash'] = bu_hash.md5_for_file( source['path'], 16 )
        except OSError as err:
            logger.warning( "cannot read %s: %s"%( source['path'], err ) )
            source['file_hash'] = None
    bu_hash.flush_file_hash_memo()
    logger.debug( "hashed %d file(s) in %.1fs"%( len( sources ), time.perf_counter() - started ) )

    hashes = { s['file_hash'] for s in sources if s['file_hash'] }
    boxes_by_hash = _censored_boxes_by_hash(
        hashes, labels, getattr( args, 'backend', None ), getattr( args, 'variants', None ) )

    rows = []
    for source in sources:
        if not source['file_hash']:
            continue
        boxes, geometry_label = boxes_by_hash.get( source['file_hash'], ( [], None ) )
        rows.append( _measure_source( source, source['file_hash'], boxes,
                                      geometry_label, fps, threshold ) )

    quick_threshold, quick_profile = _quick_cut_threshold( getattr( args, 'backend', None ) )
    for row in rows:
        profile, why = _recommend_profile( row, quick_threshold, quick_profile )
        row['recommended_profile'] = profile
        row['recommended_reason'] = why
        row['recommended_folder'] = _folder_for_profile( profile ) if profile else None

    _log_structure_table( rows, logger )
    _report_folder_moves( args, rows, logger, quick_threshold, quick_profile )
    return { 'structure': rows }


def _log_structure_table( rows, logger ):
    def pct( value ):
        return '%.0f%%'%( 100 * value ) if value is not None else 'n/a'

    logger.info( "" )
    logger.info( "Per-file structure (censored labels only):" )
    logger.info( "  %-32s %-15s %7s %-6s %7s %5s %5s %6s %5s %6s  %s"%(
        'source', 'folder', 'scanned', 'from', 'cuts/m', '<1s', '<10s',
        'frames', 'p90', 'L-half', 'belongs in' ) )
    for row in rows:
        logger.info( "  %-32s %-15s %7s %-6s %7s %5s %5s %6d %5s %6s  %s"%(
            row['source'][:32],
            ( row['current_folder'] if row['tree'] == 'uncensored_vids'
              else 'src:' + row['current_folder'] )[:15],
            ( '%.0fs'%row['scanned_seconds'] if row['scanned_seconds'] is not None else 'n/a' ),
            { 'whole file': 'whole', 'slice': 'slice', None: '-' }[row['timing_evidence']],
            '%.1f'%row['cuts_per_min'] if row['cuts_per_min'] is not None else 'n/a',
            pct( row['short_shot_fraction'] ), pct( row['mid_shot_fraction'] ),
            row['frames_with_boxes'], row['p90_simultaneous'],
            '%.2f'%row['left_half_fraction'] if row['left_half_fraction'] is not None else 'n/a',
            row['recommended_profile'] or '?' ) )

    unscanned = [ r for r in rows if r['timing_evidence'] is None ]
    sliced = [ r for r in rows if r['timing_evidence'] == 'slice' ]
    logger.info( "" )
    if unscanned:
        logger.info( "%d file(s) have no shot-cut scan at all, so nothing can be said about "
                     "their timing: run 'betabench structure-scan'"%( len( unscanned ), ) )
    if sliced:
        logger.info( "%d file(s) are measured from a preview slice only. A slice from the "
                     "start of a file mostly measures its intro; 'betabench structure-scan' "
                     "measures the whole file, and the next real run reuses that scan"%(
                         len( sliced ), ) )
    logger.info( "reading this: 'scanned' is the footage the timing numbers rest on. '<1s' "
                 "and '<10s' are shares of RUNTIME in shots shorter than that, never shares "
                 "of shots. 'frames' is how many sampled frames held a censored box; 'p90' "
                 "is how many boxes share a frame at the 90th percentile; 'L-half' near 0.50 "
                 "with a high p90 is the split-screen signature." )


def _recommend_profile( row, quick_threshold, quick_profile ):
    """
    Which profile a measured file's structure puts it in, and why.

    Order matters. Split-screen is tested first because its failure mode
    is spatial, not temporal: a generous match radius merges two panes'
    subjects into one track and drags a censor across the divider, and
    that happens whatever the cut rate. A split-screen compilation is
    still a split-screen problem first.

    Returns (profile_name, reason). profile_name is None when nothing
    was measured: "never scanned" is not "no cuts", and recommending
    'scene' for it used to send unmeasured files out of their folders.
    """
    p90 = row.get( 'p90_simultaneous' ) or 0
    left = row.get( 'left_half_fraction' )
    frames = row.get( 'frames_with_boxes' )
    low, high = SPLIT_LEFT_HALF_RANGE
    if ( p90 >= SPLIT_MIN_P90_SIMULTANEOUS
         and left is not None and low <= left <= high
         and ( frames is None or frames >= SPLIT_MIN_FRAMES_WITH_BOXES ) ):
        return ( 'split_screen',
                 '%d boxes live at once, %.0f%% of them in the left half, across %s '
                 'sampled frames: subjects sit on both sides and stay there'%(
                     p90, 100 * left, frames if frames is not None else 'an unknown number of' ) )

    short = row.get( 'short_shot_fraction' )
    mid = row.get( 'mid_shot_fraction' )
    scanned = row.get( 'scanned_seconds' )
    over = ' over %.0fs scanned'%( scanned, ) if scanned else ''

    if short is None:
        return ( None, 'shot cuts never scanned' )

    if ( quick_threshold is not None and quick_profile
         and short >= quick_threshold ):
        return ( quick_profile,
                 '%.0f%% of runtime in sub-%gs shots%s, at or over the '
                 'configured %.0f%%'%( 100 * short, bu_detector.SHORT_SHOT_SECONDS, over,
                                       100 * quick_threshold ) )

    if mid is not None and mid >= COMPILATION_MIN_RUNTIME_FRACTION:
        return ( 'compilation',
                 '%.0f%% of runtime in sub-%gs shots%s, at or over this '
                 'bench\'s %.0f%%'%( 100 * mid, COMPILATION_SHOT_SECONDS, over,
                                     100 * COMPILATION_MIN_RUNTIME_FRACTION ) )

    return ( 'scene', 'only %.0f%% of runtime in sub-%gs shots%s'%(
        100 * ( mid or 0.0 ), COMPILATION_SHOT_SECONDS, over ) )


def _profile_of_folder( row ):
    """The profile the file's CURRENT folder declares, or None."""
    return bu_detector.source_dir_rule( row['source_path'],
                                        uncensored_root=row['tree_root'] )[0]


def _move_verdict( row ):
    """
    Whether a recommended move goes in the script, or is listed for review.

    Returns ('script' | 'review', why). Every rule protects either the
    user's own judgment or the detections:

      1. Timing evidence from a slice is review-only. The survey that
         prompted this ran on 20-second slices from the start of each
         file and moved files on what their intros looked like.
      2. Never script a move that lowers the sample rate. Fewer samples
         means fewer detections on the shortest shots, which is the one
         direction this tool must never push automatically.
      3. A file in a manual_only profile's folder (compilation,
         split-screen) is there because someone judged it. Timing cannot
         see either distinction, so timing alone does not override it,
         unless the move raises the sample rate (rule 2's mirror).
      4. Only the tree root and mapped folders are scripted. An unmapped
         folder (titty_drops/, completed/) is the user's own
         organisation; the move is listed, not performed.
    """
    destination = row['recommended_profile']
    current = _profile_of_folder( row )
    destination_fps = bu_detector.get_profile_sample_fps( destination )
    current_fps = bu_detector.get_profile_sample_fps( current )
    variants = bu_detector.get_profiles().get( 'variants' ) or {}
    spatial = destination == 'split_screen'

    if not spatial and row.get( 'timing_evidence' ) != 'whole file':
        return ( 'review', 'timing from a preview slice; run structure-scan for the whole file' )
    if current and destination_fps < current_fps:
        return ( 'review', 'would lower sampling from %g to %g fps'%( current_fps, destination_fps ) )
    if ( current and ( variants.get( current ) or {} ).get( 'manual_only' )
         and not spatial and destination_fps <= current_fps ):
        return ( 'review', '%s/ is a judgment timing cannot see; not overridden'%(
            row['current_folder'], ) )
    if current is None and row['current_folder'] != '(root)':
        return ( 'review', '%s/ is not a mapped folder; move by hand or map it'%(
            row['current_folder'], ) )
    return ( 'script', '' )


def _report_folder_moves( args, rows, logger, quick_threshold, quick_profile ):
    """
    What sorting the source trees would change, and optionally a script.

    A move stays inside the tree the file is in: a file waiting in the
    staging tree is sorted there, not pulled into the working tree. Only
    moves that pass _move_verdict are scripted.
    """
    logger.info( "" )
    if not getattr( betaconfig, 'profile_by_source_dir', None ):
        logger.info( "profile_by_source_dir is empty, so no folder maps to a profile and "
                     "sorting these files would change nothing. See CONFIG_REFERENCE.md -> "
                     "'The source folder convention'." )
        return

    scripted, review = [], []
    for row in rows:
        folder = row.get( 'recommended_folder' )
        if not folder or row['current_folder'].lower() == folder.lower():
            continue
        # Already in a folder that declares the recommended profile, even
        # if it is not the folder that would be suggested for a new file
        # (a nested 'compilation/split-screen' key, say).
        if _profile_of_folder( row ) == row['recommended_profile']:
            continue
        verdict, why_not = _move_verdict( row )
        move = {
            'source': row['source'],
            'source_path': row['source_path'],
            'dest_dir': os.path.join( row['tree_root'], folder ),
            'from': row['current_folder'],
            'to': folder,
            'tree': row['tree'],
            'why': '%s: %s'%( row['source'], row['recommended_reason'] ),
            'held_back': why_not,
        }
        ( scripted if verdict == 'script' else review ).append( move )

    def where( move ):
        prefix = '' if move['tree'] == 'uncensored_vids' else 'source:'
        return '%s%s/ -> %s%s/'%( prefix, move['from'], prefix, move['to'] )

    logger.info( "=== source folder: %d move(s) to script, %d to review ==="%(
        len( scripted ), len( review ) ) )
    for move in scripted:
        logger.info( "  %-40s %s"%( where( move ), move['source'] ) )
        logger.debug( "      %s"%( move['why'], ) )
    if review:
        logger.info( "" )
        logger.info( "for review, not scripted:" )
        for move in review:
            logger.info( "  %-40s %s"%( where( move ), move['source'] ) )
            logger.info( "      %s; held back: %s"%( move['why'], move['held_back'] ) )

    if quick_threshold is not None:
        logger.info( "" )
        logger.info( "quick-cut boundary: %.0f%% of runtime in sub-%gs shots, read from "
                     "betaconfig (profile %r), so this cannot contradict what the app "
                     "would select"%( 100 * quick_threshold,
                                      bu_detector.SHORT_SHOT_SECONDS, quick_profile ) )
    logger.info( "compilation boundary: %.0f%% of runtime in sub-%gs shots. THIS BENCH'S "
                 "heuristic, not a tuned app setting, because 'compilation' is manual_only "
                 "and has no configured threshold"%(
                     100 * COMPILATION_MIN_RUNTIME_FRACTION, COMPILATION_SHOT_SECONDS ) )
    logger.info( "split-screen: p90 simultaneous >= %d with left-half share in %.2f-%.2f "
                 "over at least %d sampled frames. Read off a 9-file sample, so verify a "
                 "file before trusting it"%(
                     SPLIT_MIN_P90_SIMULTANEOUS, SPLIT_LEFT_HALF_RANGE[0],
                     SPLIT_LEFT_HALF_RANGE[1], SPLIT_MIN_FRAMES_WITH_BOXES ) )

    target = getattr( args, 'write_move_script', None )
    if target and scripted:
        _write_move_script( target, scripted, logger )
    elif target:
        logger.info( "no moves passed the safety rules, so no script was written" )
    elif scripted:
        logger.info( "" )
        logger.info( "re-run with --write-move-script moves.sh to get a reviewable script "
                     "that performs these moves with mv -n" )


# --- structure-scan --------------------------------------------------------

def _default_scan_workers():
    """
    Parallel whole-file scans to run: one per core, at most 4.

    Measured on a 2-core box with 1080p H.264: 2 workers finished four
    40-second clips 1.36x faster than 1, so OpenCV's own decode threading
    leaves cores idle and parallel files win. The cap of 4 is a judgment
    call, not a measurement: it keeps a spinning disk from seeking itself
    to death on several multi-GB files at once. --workers overrides it.
    """
    return max( 1, min( 4, os.cpu_count() or 1 ) )


def bench_structure_scan( args, logger ):
    """
    Whole-file shot-cut scans for every source video not yet scanned.

    CPU only, files in parallel, no detection. Writes the cache betatv's
    full run reads, so nothing here is wasted: the real run finds its scan
    already done. Then, with --geometry-slices N, runs detection (no
    render) on N short slices spread through each file, which is what
    gives split-screen something to be recognised from.
    """
    import concurrent.futures

    fps = betaconfig.video_censor_fps
    threshold = getattr( betaconfig, 'shot_cut_threshold', 0.5 )
    sources = _source_videos()
    pending = []
    for source in sources:
        try:
            file_hash = bu_hash.md5_for_file( source['path'], 16 )
        except OSError as err:
            logger.warning( "cannot read %s: %s"%( source['path'], err ) )
            continue
        source['file_hash'] = file_hash
        if os.path.exists( bu_cache.shot_cut_path_for( file_hash, fps, threshold ) ):
            logger.debug( "already scanned: %s"%( source['name'], ) )
            continue
        pending.append( source )
    bu_hash.flush_file_hash_memo()

    workers = getattr( args, 'workers', None ) or _default_scan_workers()
    logger.info( "%d of %d source video(s) need a whole-file shot-cut scan; %d worker(s)"%(
        len( pending ), len( sources ), workers ) )
    results = []
    if pending:
        started = time.perf_counter()
        with concurrent.futures.ProcessPoolExecutor( max_workers=workers ) as pool:
            const_paths = { name: getattr( betaconst, name )
                            for name in bu_video.SCAN_WORKER_PATHS }
            futures = { pool.submit( bu_video.scan_whole_file_in_worker, s['path'],
                                     s['file_hash'], fps, threshold, const_paths ): s
                        for s in pending }
            for done_count, future in enumerate( concurrent.futures.as_completed( futures ), 1 ):
                path, cut_count, seconds, error = future.result()
                name = os.path.basename( path )
                if error:
                    logger.warning( "[%d/%d] scan failed for %s: %s"%(
                        done_count, len( pending ), name, error ) )
                else:
                    logger.info( "[%d/%d] %s: %d cut(s) in %.0fs"%(
                        done_count, len( pending ), name, cut_count, seconds ) )
                results.append( { 'path': path, 'cuts': cut_count,
                                  'seconds': round( seconds, 1 ), 'error': error } )
        logger.info( "whole-file scans finished in %.0fs"%( time.perf_counter() - started, ) )

    slices = getattr( args, 'geometry_slices', 0 ) or 0
    if slices > 0:
        _detect_geometry_slices( args, logger, [ s for s in sources if s.get( 'file_hash' ) ],
                                 slices )
    return { 'structure_scan': results }


def geometry_slice_offsets( duration_seconds, slice_seconds, count ):
    """
    Where N geometry slices start: evenly spread, never at the very start.

    Fractions (i + 1) / (N + 1) of the room left for a slice, so 3 slices
    sit at 25%, 50% and 75%. The start is avoided on purpose: intros and
    title cards are the least representative footage in a file, and a
    survey made only of first-20-second slices was what this replaced.

    Returns:
        A sorted list of distinct offsets in seconds, one decimal place
        (the precision the preview cache suffix records), or [0.0] for a
        file too short to slice.
    """
    if not duration_seconds or duration_seconds <= slice_seconds or count <= 0:
        return [ 0.0 ]
    room = duration_seconds - slice_seconds
    return sorted( { round( room * ( i + 1 ) / ( count + 1 ), 1 ) for i in range( count ) } )


def _detect_geometry_slices( args, logger, sources, count ):
    """
    Detection, no render, on short slices spread through each file.

    Writes the same detection caches a --preview run at that offset
    would, through betatv's own detection function, so a later preview
    of the same slice reuses them. Rendering is skipped entirely: it is
    most of a preview run's cost and contributes nothing to geometry.
    """
    import betatv

    slice_seconds = getattr( args, 'slice_seconds', None ) or 20.0
    detector = bu_detector.get_detector()
    session = detector.get_session()
    picture_sizes = bu_detector.get_picture_sizes()
    fps = betaconfig.video_censor_fps
    started = time.perf_counter()
    logger.info( "geometry: %d slice(s) of %gs per file, detection only"%( count, slice_seconds ) )
    for index, source in enumerate( sources, 1 ):
        try:
            video = bu_video.open_video_source( source['path'], source['file_hash'], logger )
        except bu_video.VideoOpenError as err:
            logger.warning( "cannot open %s: %s"%( source['name'], err ) )
            continue
        try:
            duration = ( video.num_frames / video.vid_fps ) if video.vid_fps else None
            offsets = geometry_slice_offsets( duration, slice_seconds, count )
            for offset in offsets:
                bounded = slice_seconds if duration and duration > slice_seconds else None
                betatv.detect_boxes_for_video(
                    video, source['name'], session, source['file_hash'], picture_sizes,
                    bu_cache.preview_cache_suffix( True, offset ), offset, bounded,
                    logger, fps )
            logger.info( "[%d/%d] %s: slices at %s"%( index, len( sources ), source['name'],
                                                     ', '.join( '%.0fs'%o for o in offsets ) ) )
        finally:
            video.cap.release()
    logger.info( "geometry slices finished in %.0fs"%( time.perf_counter() - started, ) )


bench_geometry    = _per_configuration( 'geometry',    _bench_geometry_one )
bench_suppression = _per_configuration( 'suppression', _bench_suppression_one )
bench_dedup       = _per_configuration( 'dedup',       _bench_dedup_one )
bench_hysteresis  = _per_configuration( 'hysteresis',  _bench_hysteresis_one )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

SUBCOMMANDS = {
    'decode': bench_decode,
    'detect': bench_detect,
    'render': bench_render,
    'geometry': bench_geometry,
    'suppression': bench_suppression,
    'dedup': bench_dedup,
    'hysteresis': bench_hysteresis,
    'structure': bench_structure,
    'structure-scan': bench_structure_scan,
}

# Everything 'all' runs, in a sensible order: cheap cache reads first,
# then the measurements that actually load a model.
ALL_ORDER = [ 'structure', 'geometry', 'hysteresis', 'suppression', 'dedup',
              'render', 'decode', 'detect' ]


def build_parser():
    """The full CLI. One parser, flags shared across subcommands."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter )
    parser.add_argument( 'command', choices=sorted( SUBCOMMANDS.keys() ) + [ 'all' ],
        help="which measurement to run" )

    parser.add_argument( '--yes', action='store_true',
        help="accept every default and never prompt - what an unattended run wants" )
    parser.add_argument( '--log-level', choices=sorted( LEVEL_NAMES ), default='trace',
        help="level for the LOG FILE (default: trace, so the file keeps everything)" )
    parser.add_argument( '--console-level', choices=sorted( LEVEL_NAMES ), default='info',
        help="level for the TERMINAL (default: info)" )
    parser.add_argument( '--out-dir', default=None,
        help="where to write this run's log and results (default: ../output/benchmarks/<timestamp>)" )

    parser.add_argument( '--backend', default=None,
        help="backend to analyse (default: the selected one)" )
    parser.add_argument( '--backends', nargs='+', default=None,
        help="backends to measure, for 'detect' (default: every registered one)" )
    parser.add_argument( '--variants', nargs='+', default=None,
        help="model variants to measure, for 'detect'. Defaults to EVERY variant each backend "
             "declares, each at its own native size, so no variant goes unmeasured just because "
             "config has another one selected (e.g. --variants 320n 640m to narrow it)" )
    parser.add_argument( '--sizes', type=int, nargs='+', default=None,
        help="picture sizes to measure, for 'detect'" )
    parser.add_argument( '--batch-sizes', type=int, nargs='+', default=None,
        help="nn_batch_size values to measure, for 'detect'" )

    parser.add_argument( '--video', default=None,
        help="a specific video to measure against (default: the first found)" )
    parser.add_argument( '--max-videos', type=int, default=3,
        help="how many videos 'decode' measures (default: 3)" )
    parser.add_argument( '--seconds', type=float, default=30.0,
        help="seconds of video to measure (default: 30)" )
    parser.add_argument( '--max-frames', type=int, default=120,
        help="cap on sampled frames per measurement, for 'detect' (default: 120)" )
    parser.add_argument( '--write-move-script', default=None, metavar='PATH',
        help="structure only: write a reviewable shell script that sorts the source "
             "videos into the folders profile_by_source_dir maps to profiles. Every "
             "move is 'mv -n', which never overwrites, and the script is safe to "
             "re-run. Nothing is moved by betabench itself" )
    parser.add_argument( '--workers', type=int, default=None,
        help="structure-scan only: whole-file scans to run in parallel (default: one per "
             "CPU core, at most 4)" )
    parser.add_argument( '--geometry-slices', type=int, default=0,
        help="structure-scan only: after the timing scans, run detection (no render) on this "
             "many short slices spread through each file, for split-screen geometry. Needs "
             "the model; 3 puts slices at 25%%, 50%% and 75%% (default: 0, off)" )
    parser.add_argument( '--slice-seconds', type=float, default=20.0,
        help="structure-scan only: length of each geometry slice (default: 20)" )
    parser.add_argument( '--sample-fps', type=float, default=None,
        help="sampling rate to measure at (default: betaconfig.video_censor_fps)" )

    parser.add_argument( '--box-sizes', type=int, nargs='+', default=None,
        help="box sizes to measure, for 'render' (default: 120 240 400)" )
    parser.add_argument( '--repeats', type=int, default=20,
        help="timed repetitions per trial, for 'render' (default: 20)" )
    parser.add_argument( '--trials', type=int, default=3,
        help="independent trials per style, for 'render'; the reported number is their median "
             "and the spread across them is printed, because a single burst on a busy machine "
             "varies by several times. 3 is enough to reject a one-off outlier, which is all "
             "this needs for the usual 'which styles are expensive' question; raise it to 7+ "
             "when two styles are close enough that the printed spread hides the difference "
             "(default: 3)" )
    parser.add_argument( '--no-blur-approximation', action='store_true',
        help="force blur_fast_approximation off for this measurement" )

    parser.add_argument( '--frame-width', type=int, default=None,
        help="source frame width, for 'geometry' area fractions" )
    parser.add_argument( '--frame-height', type=int, default=None,
        help="source frame height, for 'geometry' area fractions" )
    parser.add_argument( '--min-samples', type=int, default=30,
        help="minimum observations before a suggestion is offered (default: 30)" )
    parser.add_argument( '--widen', type=float, default=1.25,
        help="safety factor applied to suggested geometry bounds (default: 1.25)" )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    run_dir = args.out_dir or os.path.join(
        betaconst.benchmark_dir, time.strftime( '%Y%m%d_%H%M%S' ) )
    logger = build_logger( run_dir, args.log_level, args.console_level )

    if args.no_blur_approximation:
        betaconfig.blur_fast_approximation = False
        bu_config.invalidate_config_caches()

    logger.info( "betabench: command=%s" % args.command )
    logger.info( "output directory: %s"%(os.path.abspath( run_dir )) )
    logger.debug( "backend=%s picture_sizes=%s video_censor_fps=%s global_min_prob=%s"%(
        bu_detector.selected_backend_name(), bu_detector.get_picture_sizes(),
        betaconfig.video_censor_fps,
        getattr( betaconfig, 'global_min_prob', betaconst.global_min_prob ) ) )

    commands = ALL_ORDER if args.command == 'all' else [ args.command ]
    results = { 'run': { 'command': args.command, 'started': time.time() } }
    failed = []

    for index, name in enumerate( commands ):
        if args.command == 'all' and name in ( 'detect', ) and index > 0:
            if not confirm( logger, "Next: '%s'. It loads every model and runs real inference, "
                                    "which takes a few minutes. Continue?"%(name), args.yes ):
                logger.info( "skipped '%s' at your request"%(name) )
                continue
        logger.info( "" )
        logger.info( "=" * 72 )
        logger.info( "  %s"%(name.upper()) )
        logger.info( "=" * 72 )
        try:
            results.update( SUBCOMMANDS[name]( args, logger ) or {} )
        except KeyboardInterrupt:
            logger.warning( "interrupted during '%s'"%(name) )
            failed.append( name )
            break
        except Exception as err:
            logger.error( "'%s' failed: %r"%(name, err) )
            logger.debug( "failure detail", exc_info=True )
            failed.append( name )

    results['run']['finished'] = time.time()
    results['run']['failed'] = failed
    results_path = os.path.join( run_dir, 'results.json' )
    bu_hash.write_json_plain( results, results_path )
    logger.info( "" )
    logger.info( "results written to %s"%(os.path.abspath( results_path )) )
    logger.info( "full log at       %s"%(os.path.abspath( os.path.join( run_dir, 'betabench.log' ) )) )
    # Non-zero when anything failed. It always returned 0, so a harness
    # wrapping it reported a failed subcommand as finished: the survey's
    # geometry phase "succeeded" in 0 seconds with no model on disk.
    if failed:
        logger.error( "failed: %s"%( ', '.join( failed ), ) )
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit( main() )
