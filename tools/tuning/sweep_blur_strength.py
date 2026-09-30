#!/usr/bin/env python3
"""
sweep_blur_strength.py - render the same preview slice at several blur
strengths, so the flicker-versus-strength tradeoff can be judged by eye
and by number at the same time.

WHY THIS EXISTS
---------------
Blur flicker scales with how large the kernel is relative to the region
it blurs. Measured on synthetic detail with the box jittering +/-4px and
the kernel held constant, the mean level swing inside a fixed crop was:

    kernel/region   0.10 -> 0.06      0.51 -> 2.19
                    0.30 -> 1.31      0.99 -> 3.04

So a weaker blur really does flicker less, and the relationship is
smooth rather than a cliff. What that table cannot say is where YOUR
threshold sits between "still obscured enough" and "visibly steady",
because that is a judgement about the footage, not a number. This script
renders the candidates so the judgement can be made against real output.

WHAT IT DOES
------------
For each strength, it pins a single blur style for every censored label,
renders the SAME preview slice of every video under
../resources/uncensored_vids/, and records the measured per-strength
flicker alongside the rendered file. Everything else is held fixed: the
same slice offset, the same detection cache, one style so no random draw
can vary between runs.

Detection runs ONCE per video. Strength lives in the censor key, not the
detection key, so every strength after the first reuses the cached
detections and only re-renders.

  ./sweep_blur_strength.py                      # 20,40,60,80,120,160
  ./sweep_blur_strength.py --strengths 30 60 90
  ./sweep_blur_strength.py --methods gaussian triple_box
  ./sweep_blur_strength.py --yes                # skip the confirmation

Output lands in ../output/logs/analysis_runs/<timestamp>/blur_sweep/ :
a log file at the configured level, a summary table, and results.json.
The rendered videos land in the normal censored_vids tree, one file per
(video, method, strength) because the censor key differs for each.
"""

import argparse
import copy
import datetime
import json
import os
import sys
import time

sys.path.insert( 0, os.path.dirname( os.path.dirname(
    os.path.dirname( os.path.abspath( __file__ ) ) ) ) )

import betaconfig
import betaconst

import betautils_cli as bu_cli
import betautils_config as bu_config
import betautils_hash as bu_hash
import betautils_log as bu_log
import betautils_signals as bu_signals
import betautils_track as bu_track
import betautils_video as bu_video
import betatv


DEFAULT_STRENGTHS = ( 20, 40, 60, 80, 120, 160 )
DEFAULT_METHODS = ( 'gaussian', )


def _build_parser():
    parser = bu_cli.build_arg_parser(
        "Render one preview slice at several blur strengths and report flicker",
        include_preview=True, include_logging=True )
    parser.add_argument( '--strengths', nargs='+', type=float, default=list( DEFAULT_STRENGTHS ),
        help="blur strengths to render (default: %s)"%( ' '.join( map( str, DEFAULT_STRENGTHS ) ), ) )
    parser.add_argument( '--methods', nargs='+', default=list( DEFAULT_METHODS ),
        choices=sorted( bu_config.VALID_BLUR_METHODS ),
        help="blur methods to render at each strength (default: gaussian)" )
    parser.add_argument( '--labels', nargs='+', default=None,
        help="labels to pin the blur style on (default: every entry in items_to_censor)" )
    parser.add_argument( '--position-smoothing', nargs='+', type=float, default=None,
        help="also sweep position_smoothing (0-1, lower is steadier). Once the edge "
             "margin removes size-driven flicker, box MOVEMENT is what is left, and "
             "this is the setting that governs it" )
    parser.add_argument( '--yes', action='store_true',
        help="skip the confirmation prompt (for unattended runs)" )
    parser.add_argument( '--measure-only', action='store_true',
        help="report the predicted flicker per strength from the cached detections "
             "and do not render anything" )
    return parser


def _run_directory():
    """
    A timestamped directory for this sweep's log, table and results.

    Derived from betaconfig.log_path, which is where the app already puts
    its log, so this lands beside run_all_analysis.sh's output under
    ../output/logs/analysis_runs/<timestamp>/ rather than inventing a
    second convention. betaconst has no log directory of its own - an
    earlier version of this function assumed betaconst.log_dir and died
    on an AttributeError before writing anything.
    """
    stamp = datetime.datetime.now().strftime( '%Y%m%d_%H%M%S' )
    log_path = getattr( betaconfig, 'log_path', None ) or '../output/logs/betasuite.log'
    logs_root = os.path.dirname( log_path ) or '.'
    path = os.path.join( logs_root, 'analysis_runs', stamp, 'blur_sweep' )
    os.makedirs( path, exist_ok=True )
    return path


def _videos_to_process():
    """
    Every candidate file under the uncensored tree, with its mirrored
    destination folder, exactly as betatv.main walks them.

    No extension filter, for the same reason betatv has none:
    process_one_video already returns 'skipped' for anything it cannot
    decode, and a hand-maintained extension list here would be a second
    source of truth that quietly disagrees with the real one.
    """
    found = []
    for root, _dirs, file_names in os.walk( betaconst.video_path_uncensored ):
        censored_folder = root.replace(
            betaconst.video_path_uncensored, betaconst.video_path_censored, 1 )
        for fname in sorted( file_names ):
            found.append( ( root, fname, censored_folder ) )
    return found


def _pin_blur_style( labels, method, strength ):
    """
    Force ONE blur style onto every censored label, for this render.

    Returns the previous per-label censor_style values so the caller can
    put them back. Pinning matters for more than tidiness: a label whose
    style is normally a weighted LIST would otherwise roll a different
    mix at every strength, and the comparison would be measuring the
    draw rather than the strength.
    """
    saved = {}
    shared = getattr( betaconfig, 'item_overrides', None )
    if shared is None:
        shared = {}
        betaconfig.item_overrides = shared
    for label in labels:
        override = shared.setdefault( label, {} )
        saved[label] = override.get( 'censor_style', _MISSING )
        override['censor_style'] = { 'type': 'blur', 'method': method,
                                     'strength': strength }
    bu_config.invalidate_config_caches()
    return saved


_MISSING = object()


def _variant_label( method, strength, smoothing=None ):
    """
    The readable tail that identifies one sweep cell in a filename.

    e.g. 'gaussian-s120' or 'triple_box-s060-p045'. Strength is
    zero-padded so a directory listing sorts in strength order rather
    than lexically, which is how these are actually read: in order,
    stopping at the last one that still obscures enough.
    """
    parts = [ str( method ), 's%03d'%( int( round( strength ) ), ) ]
    if smoothing is not None:
        parts.append( 'p%03d'%( int( round( smoothing * 100 ) ), ) )
    return '-'.join( parts )


def _pin_position_smoothing( labels, smoothing ):
    """
    Force one position_smoothing onto every censored label, or do nothing
    when the sweep is not varying it.

    Position smoothing is worth sweeping alongside strength because once
    the edge margin removes the size-driven flicker, what is left is
    driven by the box MOVING over different content - and that is this
    setting, not strength.
    """
    if smoothing is None:
        return None
    saved = {}
    shared = getattr( betaconfig, 'item_overrides', None )
    if shared is None:
        shared = {}
        betaconfig.item_overrides = shared
    for label in labels:
        override = shared.setdefault( label, {} )
        saved[label] = override.get( 'position_smoothing', _MISSING )
        override['position_smoothing'] = smoothing
    bu_config.invalidate_config_caches()
    return saved


def _restore_position_smoothing( saved ):
    if saved is None:
        return
    shared = getattr( betaconfig, 'item_overrides', {} )
    for label, previous in saved.items():
        override = shared.get( label )
        if override is None:
            continue
        if previous is _MISSING:
            override.pop( 'position_smoothing', None )
        else:
            override['position_smoothing'] = previous
    bu_config.invalidate_config_caches()


def _restore_blur_style( saved ):
    shared = getattr( betaconfig, 'item_overrides', {} )
    for label, previous in saved.items():
        override = shared.get( label )
        if override is None:
            continue
        if previous is _MISSING:
            override.pop( 'censor_style', None )
        else:
            override['censor_style'] = previous
    bu_config.invalidate_config_caches()


def _flicker_for_strength( boxes, label, method, strength, logger ):
    """
    Predicted flicker for one label at one strength, from real boxes.

    Measured as the swing in how much detail survives the blur across a
    track's consecutive frames, with the box's own position pinned so the
    number reflects the BLUR's stability rather than the censor moving
    over different content. A moving censor changes what is underneath
    it, which is not flicker the strength setting can fix.

    Returns None when no track of this label is long enough to measure.
    """
    import collections
    import numpy as np
    import betautils_censor as bu_censor

    tracks = collections.defaultdict( list )
    for box in boxes:
        if box.get( 'label' ) != label:
            continue
        tracks[ ( box.get( 'style_scale_w' ), box.get( 'style_scale_h' ) ) ].append( box )
    usable = [ group for group in tracks.values() if len( group ) >= 8 ]
    if not usable:
        return None
    group = max( usable, key=len )
    group = sorted( group, key=lambda box: box.get( 't', box['start'] ) )[:24]

    crop_w = min( box['w'] for box in group )
    crop_h = min( box['h'] for box in group )
    if crop_w < 8 or crop_h < 8:
        return None

    # Deterministic synthetic content, so two strengths are compared
    # against identical pixels rather than against different frames.
    content = np.random.default_rng( 11 ).integers(
        0, 255, ( 1080, 1920, 3 ), dtype=np.uint8 )
    style = { 'type': 'blur', 'method': method, 'strength': strength }

    spreads = []
    for box in group:
        probe = dict( box )
        probe['x'], probe['y'] = 600, 400          # position pinned
        probe['censor_shape'] = 'box'
        probe['censor_style'] = style
        rendered = bu_censor.censor_image( content.copy(), probe )
        window = rendered[400:400+crop_h, 600:600+crop_w].astype( float )
        spreads.append( float( window.std() ) )
    if not spreads:
        return None
    return max( spreads ) - min( spreads )


def _confirm( strengths, methods, videos, logger, smoothings=( None, ) ):
    """
    Explicit go/no-go before doing real work.

    The first strength pays for detection on every video; the rest only
    re-render. That is minutes, not seconds, so it is worth saying out
    loud what is about to happen rather than assuming.
    """
    renders = len( videos ) * len( strengths ) * len( methods ) * len( smoothings )
    logger.info( "" )
    logger.info( "about to render %d file(s): %d video(s) x %d method(s) x %d strength(s)"
                 "%s"%(
        renders, len( videos ), len( methods ), len( strengths ),
        " x %d smoothing(s)"%( len( smoothings ), ) if smoothings != [ None ] else '' ) )
    logger.info( "  methods:   %s"%( ', '.join( methods ) ) )
    logger.info( "  strengths: %s"%( ', '.join( '%g'%(s,) for s in strengths ) ) )
    if smoothings != [ None ]:
        logger.info( "  position_smoothing: %s"%(
            ', '.join( '%g'%(s,) for s in smoothings ) ) )
    logger.info( "  detection runs once per video; the rest reuse its cache" )
    logger.info( "" )
    try:
        answer = input( "proceed? [y/N] " ).strip().lower()
    except EOFError:
        logger.warning( "no console attached and --yes was not passed; stopping" )
        return False
    if answer not in ( 'y', 'yes' ):
        logger.info( "stopping at your request; nothing was rendered" )
        return False
    return True


def main():
    args = _build_parser().parse_args()
    bu_cli.apply_cli_overrides( betaconfig, args )

    # A sweep is only comparable within one slice, so preview mode and a
    # PINNED offset are not optional here. A random slice would give each
    # strength different footage.
    betaconfig.preview_mode_enabled = True
    betaconfig.preview_random_slice = False
    if getattr( betaconfig, 'preview_start_seconds', None ) is None:
        betaconfig.preview_start_seconds = 80.0

    run_dir = _run_directory()
    betaconfig.logging_enabled = True
    betaconfig.log_path = os.path.join( run_dir, 'sweep_blur_strength.log' )
    logger = bu_log.get_logger()
    bu_signals.install_handler( logger )
    bu_config.validate_config()

    strengths = sorted( set( args.strengths ) )
    methods = list( dict.fromkeys( args.methods ) )
    smoothings = ( sorted( set( args.position_smoothing ) )
                   if args.position_smoothing else [ None ] )
    labels = args.labels or list( getattr( betaconfig, 'items_to_censor', [] ) )
    videos = _videos_to_process()

    logger.info( "blur strength sweep" )
    logger.info( "output directory: %s"%( run_dir, ) )
    logger.info( "log file:         %s"%( betaconfig.log_path, ) )
    logger.info( "preview slice:    %.1fs starting at %.1fs"%(
        getattr( betaconfig, 'preview_max_seconds', 20 ),
        betaconfig.preview_start_seconds ) )
    logger.info( "labels pinned:    %s"%( ', '.join( labels ) if labels else '(none)' ) )
    if not videos:
        logger.error( "no videos found under %s"%( betaconst.video_path_uncensored, ) )
        return 1
    logger.info( "videos found:     %d"%( len( videos ), ) )

    results = []

    if args.measure_only:
        logger.info( "" )
        logger.info( "measure-only: predicting flicker from cached detections, rendering nothing" )

    if not args.measure_only and not args.yes:
        if not _confirm( strengths, methods, videos, logger, smoothings ):
            return 0

    session = None
    encode_preset = getattr( betaconfig, 'preview_encode_preset', 'ultrafast' )

    for smoothing in smoothings:
      for method in methods:
        for strength in strengths:
            saved = _pin_blur_style( labels, method, strength )
            saved_smoothing = _pin_position_smoothing( labels, smoothing )
            try:
                logger.info( "" )
                logger.info( "=" * 70 )
                logger.info( "  %s"%( _variant_label( method, strength, smoothing ), ) )
                logger.info( "=" * 70 )
                for index, ( root, fname, censored_folder ) in enumerate( videos ):
                    bu_signals.check()
                    row = { 'method': method, 'strength': strength,
                            'position_smoothing': smoothing, 'file': fname,
                            'label': _variant_label( method, strength, smoothing ) }
                    if args.measure_only:
                        results.append( row )
                        continue
                    if session is None:
                        import betautils_detector as bu_detector
                        try:
                            session = bu_detector.get_detector().get_session()
                        except Exception as err:
                            # Almost always a missing or mis-pathed .onnx.
                            # A stack trace here buries the one line that
                            # says what to do about it.
                            logger.error( "cannot start the detector: %s"%( err, ) )
                            logger.error( "nothing was rendered. Fix the model path and re-run, "
                                          "or pass --measure-only to skip rendering." )
                            return 1
                    os.makedirs( censored_folder, exist_ok=True )
                    started = time.perf_counter()
                    # The keys already make each render's name unique;
                    # this tail is what makes a directory of them
                    # readable without decoding hex.
                    outcome = betatv.process_one_video(
                        root, fname, censored_folder,
                        index, len( videos ), session,
                        True, getattr( betaconfig, 'preview_max_seconds', 20 ),
                        encode_preset, logger,
                        output_label=_variant_label( method, strength, smoothing ) )
                    row['outcome'] = outcome
                    row['seconds'] = round( time.perf_counter() - started, 2 )
                    results.append( row )
            except bu_signals.Interrupted:
                logger.warning( "interrupted; keeping what has been rendered so far" )
                break
            finally:
                _restore_blur_style( saved )
                _restore_position_smoothing( saved_smoothing )

    # Flicker prediction, from the detections the renders just used.
    logger.info( "" )
    logger.info( "=" * 70 )
    logger.info( "  predicted flicker by strength (lower is steadier)" )
    logger.info( "=" * 70 )
    flicker = {}
    try:
        import betautils_cache_paths as bu_cache
        import betautils_detector as bu_detector
        configurations = bu_cache.configurations_to_analyse(
            None,
            backend_names=[ bu_detector.selected_backend_name() ],
            fps=getattr( betaconfig, 'video_censor_fps', 9 ),
            include_preview=True, logger=logger )
        measured_any = False
        for config in configurations:
            raw_boxes = []
            for file_hash in config.file_hashes:
                for size in config.picture_sizes:
                    path = bu_cache.box_hash_path_for(
                        file_hash, size, getattr( betaconfig, 'video_censor_fps', 9 ),
                        getattr( betaconfig, 'global_min_prob', betaconst.global_min_prob ),
                        config.backend_name, config.preview_suffix )
                    if os.path.exists( path ):
                        raw_boxes.extend( bu_hash.read_json( path ) )
            if not raw_boxes:
                continue
            boxes, _stats = bu_track.prepare_boxes_for_render(
                copy.deepcopy( raw_boxes ), 1920, 1080,
                backend_name=config.backend_name )
            for method in methods:
                for strength in strengths:
                    for label in labels:
                        swing = _flicker_for_strength(
                            boxes, label, method, strength, logger )
                        if swing is None:
                            continue
                        measured_any = True
                        flicker.setdefault( method, {} ).setdefault(
                            label, {} )['%g'%(strength,)] = round( swing, 4 )
            break   # one configuration is enough to show the trend
        if measured_any:
            for method, per_label in sorted( flicker.items() ):
                for label, by_strength in sorted( per_label.items() ):
                    logger.info( "  %s / %s"%( method, label ) )
                    for strength in strengths:
                        value = by_strength.get( '%g'%(strength,) )
                        if value is not None:
                            logger.info( "      strength %6g   swing %7.4f"%( strength, value ) )
        else:
            logger.info( "  no track was long enough to measure; render and judge by eye" )
    except Exception as err:
        logger.warning( "flicker prediction unavailable: %r"%( err, ) )
        logger.debug( "detail", exc_info=True )

    summary_path = os.path.join( run_dir, 'results.json' )
    with open( summary_path, 'w', encoding='UTF-8' ) as handle:
        json.dump( { 'strengths': strengths, 'methods': methods,
                     'labels': labels, 'smoothings': smoothings, 'rows': results,
                     'flicker': flicker }, handle, indent=2 )
    logger.info( "" )
    logger.info( "results written to %s"%( summary_path, ) )
    logger.info( "" )
    logger.info( "What the numbers cannot decide: whether the weaker settings still" )
    logger.info( "obscure enough. Watch the rendered files in strength order and stop" )
    logger.info( "at the last one you would ship." )
    return 0


if __name__ == '__main__':
    sys.exit( main() )
