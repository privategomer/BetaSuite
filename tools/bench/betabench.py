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
    structure    Per video, not pooled: how fast does it cut, how many
                 subjects are on screen, where do they sit in frame?
                 Decides whether footage needs per-profile settings, and
                 which profile a file belongs to.
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

    structure / geometry / suppression / dedup / hysteresis read the
    detection caches a real run already produced. They do not run the model, so
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
            if fname.lower().endswith( ( '.mp4', '.mkv', '.avi', '.mov', '.webm', '.m4v' ) ):
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


def _bench_structure_one( args, logger, config, raw_boxes ):
    """
    Describe each source video's STRUCTURE, per video rather than pooled.

    Every other subcommand pools all footage into one distribution,
    which is the right thing when the question is "what does this model
    do". It is the wrong thing when the question is "does this file need
    different settings from that one", because a compilation and a long
    single-scene clip average into a description of neither.

    What this reports, per video:
      cuts/min, median shot length   how fast the footage cuts
      simultaneous boxes             how many subjects are on screen
      spatial spread                 whether boxes cluster in separate
                                     screen regions (split-screen) or
                                     move through one
      box area                       how large the subject is in frame

    These are STRUCTURAL properties, readable from caches that already
    exist. They are deliberately not content labels: "solo" and "couple"
    cannot be recovered from detections, but "cuts four times a second"
    and "boxes live in two fixed vertical halves" can, and those are
    what actually want different tracking and timing settings.

    Nothing here changes config. It is a classifier input: read it to
    decide which files deserve a profile, and to check whether a profile
    you already have matches the footage you are about to run.
    """
    import statistics

    fps = args.sample_fps or betaconfig.video_censor_fps
    threshold = getattr( betaconfig, 'shot_cut_threshold', 0.5 )
    labels = set( getattr( betaconfig, 'items_to_censor', [] ) )

    by_hash = {}
    for raw in raw_boxes:
        file_hash = raw.get( '_file_hash' )
        if file_hash:
            by_hash.setdefault( file_hash, [] ).append( raw )
    if not by_hash:
        logger.warning( "cached detections carry no source video tag; structure needs one "
                        "per video, so there is nothing to report" )
        return {}

    hash_to_path = bu_cache.build_hash_to_video_path( set( by_hash ) )
    rows = []
    for file_hash in sorted( by_hash ):
        boxes = [ b for b in by_hash[file_hash] if b.get( 'class_id' ) in labels ]
        if not boxes:
            continue
        times = sorted( float( b.get( 't', 0 ) ) for b in boxes )
        span = max( times[-1] - times[0], 1e-6 )

        # Shot cuts come from the cache a real run already wrote. A file
        # scanned with shot_cut_detection_enabled off simply has none,
        # and is reported as unknown rather than as zero - "no cuts
        # detected" and "cuts never looked for" are different facts.
        # shot_cut_path_for's key carries the preview suffix and bound,
        # which this bench does not know: it is reading caches a run
        # wrote, and that run may have been a --preview one. Asking for
        # the bare (hash, fps, threshold) path therefore MISSED every
        # preview cache and reported "no shot-cut caches found" while
        # the run that produced these very detections had logged 50
        # cuts. Fall back to a glob on the hash+fps+threshold prefix,
        # newest first, so a preview cache is found too.
        cut_path = bu_cache.shot_cut_path_for( file_hash, fps, threshold )
        if not os.path.exists( cut_path ):
            prefix = os.path.join( betaconst.shot_cut_dir,
                                   '%s-%g-%.3f'%( file_hash, fps, threshold ) )
            candidates = sorted( glob.glob( prefix + '*.gz' ),
                                 key=os.path.getmtime, reverse=True )
            cut_path = candidates[0] if candidates else cut_path
        cuts = None
        if os.path.exists( cut_path ):
            try:
                cuts = list( bu_hash.read_json( cut_path ) )
            except Exception:
                cuts = None

        cuts_per_min, median_shot = None, None
        if cuts is not None:
            cuts_per_min = 60.0 * len( cuts ) / span
            if len( cuts ) > 1:
                shots = [ b - a for a, b in zip( sorted( cuts ), sorted( cuts )[1:] ) ]
                median_shot = statistics.median( shots ) if shots else None

        # Simultaneous boxes: how many of this label are live in one
        # sampled frame. High values mean several subjects, which is
        # what makes match_distance risky.
        per_frame = {}
        for b in boxes:
            per_frame.setdefault( round( float( b.get( 't', 0 ) ), 3 ), 0 )
            per_frame[round( float( b.get( 't', 0 ) ), 3 )] += 1
        counts = sorted( per_frame.values() )
        median_simul = statistics.median( counts ) if counts else 0
        p90_simul = counts[int( 0.9 * ( len( counts ) - 1 ) )] if counts else 0

        # Spatial spread: a split-screen has boxes in fixed, separated
        # regions, so the horizontal centres are strongly bimodal. One
        # subject moving through frame gives a broad unimodal spread.
        # Reported as the fraction of boxes in each vertical third and
        # each horizontal half, which is enough to see the pattern
        # without committing to a clustering algorithm.
        xs = [ float( b.get( 'x', 0 ) ) + float( b.get( 'w', 0 ) ) / 2 for b in boxes ]
        ys = [ float( b.get( 'y', 0 ) ) + float( b.get( 'h', 0 ) ) / 2 for b in boxes ]
        path = hash_to_path.get( file_hash )
        vid_w, vid_h = ( 0, 0 )
        if path:
            vid_w, vid_h = bu_tuning.frame_size_for_video( path )
        left_half = right_half = None
        if vid_w:
            left_half = sum( 1 for x in xs if x < vid_w / 2 ) / len( xs )
            right_half = 1.0 - left_half

        areas = [ float( b.get( 'w', 0 ) ) * float( b.get( 'h', 0 ) ) for b in boxes ]
        area_fraction = None
        if vid_w and vid_h:
            area_fraction = statistics.median( areas ) / float( vid_w * vid_h )

        rows.append( {
            'file_hash': file_hash,
            'source': os.path.basename( path ) if path else file_hash,
            'span_seconds': round( span, 1 ),
            'detections': len( boxes ),
            'cuts': len( cuts ) if cuts is not None else None,
            'cuts_per_min': round( cuts_per_min, 1 ) if cuts_per_min is not None else None,
            'median_shot_seconds': round( median_shot, 2 ) if median_shot else None,
            'median_simultaneous': median_simul,
            'p90_simultaneous': p90_simul,
            'left_half_fraction': round( left_half, 3 ) if left_half is not None else None,
            'median_area_fraction': round( area_fraction, 5 ) if area_fraction else None,
        } )

    if not rows:
        return {}

    logger.info( "" )
    logger.info( "Per-video structure (censored labels only):" )
    logger.info( "  %-34s %7s %8s %8s %7s %7s %7s"%(
        'source', 'mins', 'cuts/min', 'med shot', 'simul', 'p90', 'L-half' ) )
    for row in rows:
        logger.info( "  %-34s %7.1f %8s %8s %7s %7s %7s"%(
            row['source'][:34],
            row['span_seconds'] / 60.0,
            '%.1f'%row['cuts_per_min'] if row['cuts_per_min'] is not None else '  n/a',
            '%.2fs'%row['median_shot_seconds'] if row['median_shot_seconds'] else '   n/a',
            row['median_simultaneous'],
            row['p90_simultaneous'],
            '%.2f'%row['left_half_fraction'] if row['left_half_fraction'] is not None else ' n/a' ) )

    known = [ r for r in rows if r['cuts_per_min'] is not None ]
    if known:
        rates = sorted( r['cuts_per_min'] for r in known )
        spread = ( rates[-1] / rates[0] ) if rates[0] > 0 else float( 'inf' )
        logger.info( "" )
        logger.info( "cut rate across these files: %.1f to %.1f per minute (%.1fx spread)"%(
            rates[0], rates[-1], spread ) )
        if spread >= 3.0:
            logger.info( "  That is a wide spread, which is the case FOR per-profile settings: "
                         "time_safety and track_max_gap that suit the slowest-cutting file here "
                         "are too loose for the fastest, and vice versa." )
        else:
            logger.info( "  These files cut at similar rates, so one set of timing values "
                         "probably suits all of them. A profile split would be answering a "
                         "question this footage is not asking - add faster-cutting footage "
                         "before deciding." )
    else:
        logger.info( "" )
        logger.info( "no shot-cut caches found for these files. Set "
                     "shot_cut_detection_enabled = True and re-run betatv.py; the cut rate is "
                     "the single most useful structural number here." )

    logger.info( "" )
    logger.info( "reading this: 'simul' is how many boxes of a censored label are live in one "
                 "sampled frame - consistently 2+ means several subjects, which is what makes "
                 "a loose match_distance risky. 'L-half' near 0.50 with high 'simul' is the "
                 "split-screen signature: boxes split evenly between screen halves and stay "
                 "there. Near 0 or 1 means the action sits on one side." )

    return { 'structure': rows }


bench_geometry    = _per_configuration( 'geometry',    _bench_geometry_one )
bench_suppression = _per_configuration( 'suppression', _bench_suppression_one )
bench_dedup       = _per_configuration( 'dedup',       _bench_dedup_one )
bench_hysteresis  = _per_configuration( 'hysteresis',  _bench_hysteresis_one )
bench_structure   = _per_configuration( 'structure',   _bench_structure_one )


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
    parser.add_argument( '--log-level', choices=sorted( LEVEL_NAMES ), default='debug',
        help="level for the LOG FILE (default: debug)" )
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
            break
        except Exception as err:
            logger.error( "'%s' failed: %r"%(name, err) )
            logger.debug( "failure detail", exc_info=True )

    results['run']['finished'] = time.time()
    results_path = os.path.join( run_dir, 'results.json' )
    bu_hash.write_json_plain( results, results_path )
    logger.info( "" )
    logger.info( "results written to %s"%(os.path.abspath( results_path )) )
    logger.info( "full log at       %s"%(os.path.abspath( os.path.join( run_dir, 'betabench.log' ) )) )
    return 0


if __name__ == '__main__':
    raise SystemExit( main() )
