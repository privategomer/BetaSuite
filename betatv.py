"""
betatv.py - Entry point for censoring video files.

Walks betaconst.video_path_uncensored recursively and, for every video
found, detects and tracks the configured body parts, renders a censored
copy in resumable chunks, muxes the original audio back in, and writes
the result into the mirrored directory structure under
betaconst.video_path_censored.

    main()
      -> _startup()                       CLI args, config validation,
                                          signal handler, logging
      -> for each file found by os.walk:
          -> process_one_video()
              -> bu_video.open_video_source()        decode + fallbacks
              -> _resolve_preview_slice()            where to start/stop
              -> _build_paths()                      cache and output names
                                                     (an existing, valid
                                                     output here ends the
                                                     file before any
                                                     scanning work)
              -> bu_video.shot_cuts_for_source()     cached scene cuts
              -> detect_boxes_for_video()            per size, cached,
                                                     checkpointed
              -> bu_track.prepare_boxes_for_render() dedup -> geometry
                                                     filter -> suppression
                                                     -> tracking
              -> bu_render.render_video()            parallel chunks,
                                                     concat, audio mux

This module is deliberately thin. The work lives in modules that the
tuning and analysis tools can import directly:

    betautils_video     decoding, frame sampling, shot cuts
    betautils_track     dedup, filtering, suppression, tracking
    betautils_censor    turning a box into pixels
    betautils_render    chunked, parallel rendering and muxing
    betautils_cache_paths   every cache path, output name and cache key

Before 2.1, tracking and suppression lived here, which is why
tools/tuning/replay_tune.py had to read this file's source text and exec
the two function bodies out of it to be sure it was testing the real
code.
"""

import os
import time

import betaconst
import betaconfig

import betautils_cache_paths as bu_cache
import betautils_cli as bu_cli
import betautils_config as bu_config
import betautils_detector as bu_detector
import betautils_hash as bu_hash
import betautils_log as bu_log
import betautils_render as bu_render
import betautils_signals as bu_signals
import betautils_track as bu_track
import betautils_video as bu_video


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def _startup():
    """
    Parse CLI flags, apply them over betaconfig, validate, and prepare
    the process.

    Fail-fast by design: validate_config() exits before any real work if
    the configuration is wrong, rather than surfacing a KeyError an hour
    into a render.

    Returns:
        The shared logger.

    Side effects:
        Mutates betaconfig with CLI overrides, may SystemExit(1) on a
        bad config, may prompt on stdin when input_delete_probability is
        non-zero, and installs the cooperative SIGINT handler.
    """
    parser = bu_cli.build_arg_parser( "BetaTV: censor videos",
                                      include_preview=True, include_logging=True )
    args = parser.parse_args()
    bu_cli.apply_cli_overrides( betaconfig, args )

    logger = bu_log.get_logger()
    bu_signals.install_handler( logger )

    bu_config.validate_config()
    bu_config.verify_input_delete_probability()
    return logger


def _resolve_active_encode_preset():
    """The x264 preset this run should use: preview's, or the real one."""
    if getattr( betaconfig, 'preview_mode_enabled', False ):
        return getattr( betaconfig, 'preview_encode_preset', 'ultrafast' )
    return getattr( betaconfig, 'encode_preset', 'fast' )


def _log_run_banner( logger ):
    """
    Log what this run is actually configured to do, once, up front.

    Everything here is something that has caused real confusion after
    the fact: which backend and variant ran, at which detection sizes,
    and whether the run was a preview slice rather than the real thing.
    """
    backend = bu_detector.selected_backend_name()
    variant = bu_detector.selected_variant_name( backend )
    logger.info( "BetaTV starting: backend=%s%s picture_sizes=%s video_censor_fps=%s "
                 "nn_batch_size=%s global_min_prob=%s"%(
        backend,
        ' variant=%s'%(variant) if variant else '',
        bu_detector.get_picture_sizes( backend ),
        betaconfig.video_censor_fps,
        bu_detector.get_nn_batch_size( backend ),
        getattr( betaconfig, 'global_min_prob', betaconst.global_min_prob ) ) )

    if getattr( betaconfig, 'preview_mode_enabled', False ):
        preview_max_seconds = getattr( betaconfig, 'preview_max_seconds', 20 )
        explicit_start = getattr( betaconfig, 'preview_start_seconds', None )
        random_slice = getattr( betaconfig, 'preview_random_slice', False )
        window = ( "a %.0fs slice"%(preview_max_seconds)
                   if ( explicit_start is not None or random_slice )
                   else "the first %.0fs"%(preview_max_seconds) )
        logger.warning( "PREVIEW MODE: only %s of each video will be processed, at the '%s' "
                        "encode preset. Turn preview_mode_enabled off for real runs."%(
            window, getattr( betaconfig, 'preview_encode_preset', 'ultrafast' ) ) )


# ---------------------------------------------------------------------------
# Per-file setup
# ---------------------------------------------------------------------------

def _resolve_preview_slice( fname, source, preview_mode_enabled, preview_max_seconds, logger ):
    """
    Where in this specific video a preview render starts and stops.

    The chosen offset is applied as a direct seek, never an ffmpeg trim,
    so nothing before the start point is decoded.

    Decided per file rather than per run, because preview_start_seconds
    is one fixed setting applied across a directory that can hold videos
    of very different lengths.

    Args:
        fname: Filename, for log messages.
        source: A betautils_video.VideoSource.
        preview_mode_enabled: Whether preview mode is on.
        preview_max_seconds: Preview window length.
        logger: The shared logger.

    Returns:
        A (preview_offset_seconds, preview_window_unlimited) pair.
        preview_window_unlimited is True when preview_start_seconds is
        past the end of THIS video, in which case this one file is
        processed in full rather than clamped to a slice the user never
        asked for. Every other file in the run still gets its slice.
    """
    explicit_start = getattr( betaconfig, 'preview_start_seconds', None )
    random_slice = getattr( betaconfig, 'preview_random_slice', False )
    offset, unlimited = bu_video.resolve_preview_slice(
        source.vid_fps, source.num_frames, preview_mode_enabled,
        preview_max_seconds, explicit_start, random_slice )

    if not preview_mode_enabled:
        return offset, unlimited

    duration = source.num_frames / source.vid_fps if source.vid_fps else 0.0
    if unlimited:
        logger.warning( "preview_start_seconds=%.1fs is past the end of %s (%.1fs long) - "
                        "processing the whole file instead of a slice."%(
            explicit_start, fname, duration ) )
    elif explicit_start is not None and offset != explicit_start:
        logger.warning( "preview_start_seconds=%.1fs would leave less than the requested %.1fs "
                        "on %s (%.1fs long) - using %.1fs so the full sample fits."%(
            explicit_start, preview_max_seconds, fname, duration, offset ) )
    elif explicit_start is not None or random_slice:
        pin = ( "" if explicit_start is not None
                else " [pin this slice with preview_start_seconds = %.2f]"%(offset) )
        logger.info( "preview slice for %s: %.1fs starting at %.1fs of %.1fs%s"%(
            fname, preview_max_seconds, offset, duration, pin ) )
    return offset, unlimited


def _build_paths( censored_folder, stem, file_hash, picture_sizes,
                  preview_mode_enabled, preview_offset_seconds, sample_fps=None,
                  output_label='' ):
    """
    Every path this file needs, from the one module that owns naming.

    Returns:
        A dict with 'final', 'intermediate', 'preview_suffix',
        'detection_key', 'censor_key' and 'encode_key'.

    Note:
        All three keys are in the output filename, so two runs that
        differ in ANY setting that changes the bytes on disk get
        different names. Before 2.1 only a narrow "censor hash" was
        embedded, deliberately excluding every tracking setting, so
        re-running with a changed track_max_gap silently overwrote the
        previous output and left nothing to compare.
    """
    sample_fps = sample_fps or betaconfig.video_censor_fps
    preview_suffix = bu_cache.preview_cache_suffix( preview_mode_enabled, preview_offset_seconds )
    detection_key = bu_cache.detection_key( fps=sample_fps )
    censor_key = bu_cache.censor_key()
    encode_key = bu_cache.encode_key( preview_mode_enabled )
    container = getattr( betaconfig, 'render_chunk_container', 'mkv' )

    final_path, intermediate_path = bu_cache.video_output_paths(
        censored_folder, stem, file_hash, picture_sizes, sample_fps,
        detection_key, censor_key, encode_key, preview_suffix, container,
        output_label )

    return {
        'final': final_path,
        'intermediate': intermediate_path,
        'preview_suffix': preview_suffix,
        'detection_key': detection_key,
        'censor_key': censor_key,
        'encode_key': encode_key,
        'sample_fps': sample_fps,
    }


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_boxes_for_size( source, fname, size, session, file_hash, preview_suffix,
                           preview_offset_seconds, preview_max_seconds, logger,
                           sample_fps=None ):
    """
    Raw detections for one video at one detection size.

    Served from cache when a completed one exists, resumed from a
    checkpoint when a previous run was interrupted partway, and
    otherwise produced by sampling the video at
    betaconfig.video_censor_fps and running the model.

    Frames are fetched by betautils_video.iter_sampled_frames, which
    decodes sequentially and skips unwanted frames with grab(). The
    pre-2.1 loop seeked once per sample, which on a long-GOP stream cost
    roughly 40x more per sampled frame.

    Args:
        source: A betautils_video.VideoSource.
        fname: Filename, for log messages.
        size: The detection size.
        session: The detector's opaque session handle.
        file_hash: Content hash of the source video.
        preview_suffix: Cache-key suffix for this preview slice.
        preview_offset_seconds: Where sampling starts.
        preview_max_seconds: Preview window length, or None for the
            whole file.
        logger: The shared logger.

    Returns:
        A list of raw box dicts.
    """
    backend_name = bu_detector.selected_backend_name()
    sample_fps = sample_fps or betaconfig.video_censor_fps
    global_min_prob = getattr( betaconfig, 'global_min_prob', betaconst.global_min_prob )
    cache_path = bu_cache.box_hash_path_for(
        file_hash, size, sample_fps, global_min_prob,
        backend_name, preview_suffix )

    if os.path.exists( cache_path ):
        try:
            cached = bu_hash.read_json( cache_path )
            logger.info( "size %d: %d cached detection(s) for %s"%(size, len( cached ), fname) )
            return cached
        except Exception as err:
            logger.warning( "corrupt detection cache %s (%s), regenerating"%(cache_path, err) )
            try:
                os.remove( cache_path )
            except OSError:
                pass

    checkpoint = bu_hash.read_checkpoint( cache_path )
    if checkpoint:
        raw_boxes = list( checkpoint['boxes'] )
        start_sample_index = checkpoint['next_frame']
        logger.info( "size %d: resuming from checkpoint at sample %d (%d box(es) so far)"%(
            size, start_sample_index, len( raw_boxes ) ) )
    else:
        raw_boxes = []
        start_sample_index = 0

    detector = bu_detector.get_detector( backend_name )
    batch_size = max( 1, bu_detector.get_nn_batch_size( backend_name ) )
    checkpoint_interval = getattr( betaconfig, 'detection_checkpoint_frames', 500 )

    frame_buffer = []
    time_buffer = []
    last_checkpoint_sample = start_sample_index

    total_samples = None
    if source.vid_fps:
        span_seconds = ( preview_max_seconds if preview_max_seconds is not None
                         else source.num_frames / source.vid_fps - preview_offset_seconds )
        total_samples = max( 1, int( span_seconds * sample_fps ) )

    def flush():
        """Run the model over whatever is buffered, and clear the buffer."""
        if not frame_buffer:
            return []
        detections = detector.raw_boxes_for_imgs( frame_buffer, size, session, time_buffer )
        frame_buffer.clear()
        time_buffer.clear()
        return detections

    capture = source.open_capture()
    sample_index = start_sample_index
    next_t = bu_video.sample_time( start_sample_index, preview_offset_seconds,
                                   sample_fps )
    try:
        with bu_log.ProgressReporter( "detect %d"%(size), total=total_samples ) as progress:
            for sample_index, timestamp, frame in bu_video.iter_sampled_frames(
                    capture, source.vid_fps, sample_fps,
                    offset_seconds=preview_offset_seconds,
                    num_frames=source.num_frames,
                    max_seconds=preview_max_seconds,
                    start_sample_index=start_sample_index ):
                bu_signals.check()

                frame_buffer.append( frame )
                time_buffer.append( timestamp )
                if len( frame_buffer ) >= batch_size:
                    raw_boxes.extend( flush() )

                progress.update( sample_index + 1 )
                next_t = bu_video.sample_time( sample_index + 1, preview_offset_seconds,
                                               sample_fps )

                # Only checkpoint with an empty buffer, so raw_boxes is
                # always consistent with "everything up to this sample is
                # accounted for".
                if ( not frame_buffer and checkpoint_interval > 0
                        and sample_index + 1 - last_checkpoint_sample >= checkpoint_interval ):
                    bu_hash.write_checkpoint( raw_boxes, sample_index + 1, next_t, cache_path )
                    last_checkpoint_sample = sample_index + 1

            raw_boxes.extend( flush() )  # any partial batch at the end
    except bu_signals.Interrupted:
        raw_boxes.extend( flush() )
        bu_hash.write_checkpoint( raw_boxes, sample_index + 1, next_t, cache_path )
        logger.info( "size %d: checkpointed %d detection(s) at sample %d before stopping"%(
            size, len( raw_boxes ), sample_index + 1 ) )
        raise
    finally:
        capture.release()

    bu_hash.write_json( raw_boxes, cache_path )
    bu_hash.clear_checkpoint( cache_path )
    logger.info( "size %d: %d detection(s) for %s"%(size, len( raw_boxes ), fname) )
    return raw_boxes


def detect_boxes_for_video( source, fname, session, file_hash, picture_sizes,
                            preview_suffix, preview_offset_seconds, preview_max_seconds,
                            logger, sample_fps=None ):
    """
    Raw detections across every configured detection size, flattened.

    Args:
        source: A betautils_video.VideoSource.
        fname: Filename, for log messages.
        session: The detector's opaque session handle.
        file_hash: Content hash of the source video.
        picture_sizes: Sizes to run, resolved for the active backend.
        preview_suffix: Cache-key suffix for this preview slice.
        preview_offset_seconds: Where sampling starts.
        preview_max_seconds: Preview window length, or None.
        logger: The shared logger.

    Returns:
        One flat list of raw box dicts across every size.
    """
    all_raw_boxes = []
    for size in picture_sizes:
        all_raw_boxes.extend( detect_boxes_for_size(
            source, fname, size, session, file_hash, preview_suffix,
            preview_offset_seconds, preview_max_seconds, logger, sample_fps ) )
    return all_raw_boxes


# ---------------------------------------------------------------------------
# Per-file orchestration
# ---------------------------------------------------------------------------

def process_one_video( root, fname, censored_folder, file_index, total_files, session,
                       preview_mode_enabled, preview_max_seconds, active_encode_preset,
                       logger, output_label='' ):
    """
    Detect, track, render and mux one video.

    Args:
        root: Directory fname was found in.
        fname: Filename being processed.
        censored_folder: Destination directory, mirroring root's
            position under betaconst.video_path_censored.
        file_index: 0-based index within this directory, for progress.
        total_files: File count in this directory, for progress.
        session: The detector's opaque session handle.
        preview_mode_enabled: Whether preview mode is on.
        preview_max_seconds: Preview window length.
        active_encode_preset: x264 preset for this run.
        logger: The shared logger.
        output_label: Cosmetic tail for the OUTPUT filename only, so a
            sweep that renders one source under many settings produces
            names a person can read. Never reaches a cache path.

    Returns:
        'processed', 'skipped' or 'failed'.

    Raises:
        betautils_signals.Interrupted: the user asked to stop. Deliberately
            NOT caught here - `except Exception` below gives the run its
            one-bad-file-is-not-fatal behaviour, and an interrupt must
            pass straight through it. Before 2.1 this caught
            BaseException, so Ctrl-C was reported as a failed file and
            the run simply moved on to the next one.
    """
    uncensored_path = os.path.join( root, fname )
    stem, _suffix = os.path.splitext( fname )
    label = "%d/%d"%(file_index+1, total_files)

    try:
        file_hash = bu_hash.md5_for_file( uncensored_path, 16 )
    except OSError as err:
        logger.debug( "skipping %s (%s): %s"%(label, fname, err) )
        return 'skipped'

    try:
        source = bu_video.open_video_source( uncensored_path, file_hash, logger )
    except bu_video.VideoOpenError as err:
        logger.info( "skipping %s (not a decodable video): %s"%(label, fname) )
        logger.debug( "  %s"%(err) )
        return 'skipped'

    started_at = time.perf_counter()
    try:
        preview_offset_seconds, preview_window_unlimited = _resolve_preview_slice(
            fname, source, preview_mode_enabled, preview_max_seconds, logger )
        effective_preview_seconds = ( preview_max_seconds
                                      if preview_mode_enabled and not preview_window_unlimited
                                      else None )

        picture_sizes = bu_detector.get_picture_sizes()

        # A randomised preview slice is a request for a NEW sample. Its
        # offset is written into the filename at one-decimal precision,
        # so a fresh draw can land on an existing file's name; skipping
        # it would report a slice that was never rendered.
        random_preview_slice = (
            preview_mode_enabled
            and getattr( betaconfig, 'preview_start_seconds', None ) is None
            and getattr( betaconfig, 'preview_random_slice', False ) )

        def already_done( candidate_paths ):
            """
            An existing, decodable output at this path IS what this run
            would produce: the name carries all three cache keys, the
            sample rate and the preview slice. Probing rather than
            stat-ing means a truncated output from an interrupted run is
            rebuilt instead of accepted as finished.
            """
            return ( not random_preview_slice
                     and os.path.exists( candidate_paths['final'] )
                     and bu_video.probe_video_ok( candidate_paths['final'] ) )

        # When no profile changes the sample rate, the output name is
        # known before anything is scanned, so a repeat run skips at zero
        # cost. When one does, the name depends on which profile the
        # footage selects, and that needs the shot-cut scan first - which
        # is cached, so a repeat run still skips without detecting.
        paths = None
        if len( bu_detector.profile_sample_rates() ) == 1:
            paths = _build_paths( censored_folder, stem, file_hash, picture_sizes,
                                  preview_mode_enabled, preview_offset_seconds,
                                  output_label=output_label )
            if already_done( paths ):
                logger.info( "skipping %s (output already present and valid): %s"%(label, fname) )
                return 'skipped'

        logger.info( "processing %s: %s"%(label, fname) )
        if source.transcoded:
            logger.info( "  decoding via transcoded copy: %s"%(source.decode_path) )

        # The scan always runs at the GLOBAL rate, never a profile's: the
        # profile is chosen FROM this scan, so letting the profile set the
        # scan rate would be circular. Cut timestamps are times, not
        # sample indices, so they stay valid at whatever rate detection
        # then runs.
        preview_suffix = bu_cache.preview_cache_suffix( preview_mode_enabled, preview_offset_seconds )
        shot_cut_times = bu_video.shot_cuts_for_source(
            source, file_hash, betaconfig.video_censor_fps,
            getattr( betaconfig, 'shot_cut_threshold', 0.5 ),
            preview_offset_seconds, effective_preview_seconds, preview_suffix,
            enabled=getattr( betaconfig, 'shot_cut_detection_enabled', True ),
            logger=logger )

        # Which structure profile this video's settings come from, chosen
        # by its measured shot-cut rate. The rate is cuts divided by the
        # length of the window SCANNED, not by the spread between the
        # first and last cut, which reads a quiet clip with one brief
        # flurry as a compilation. A file with no scan reports None and
        # takes the configured default rather than a guessed one.
        cuts_per_min = None
        scanned_seconds = None
        if shot_cut_times:
            scanned_seconds = effective_preview_seconds
            if scanned_seconds is None and source.vid_fps:
                scanned_seconds = source.num_frames / source.vid_fps
            if scanned_seconds and scanned_seconds > 0:
                cuts_per_min = 60.0 * len( shot_cut_times ) / scanned_seconds
        median_shot_seconds = bu_detector.median_shot_seconds_for(
            shot_cut_times, scanned_seconds )
        short_shot_fraction = bu_detector.short_shot_fraction_for(
            shot_cut_times, scanned_seconds )
        profile_name = bu_detector.select_profile_name(
            cuts_per_min, source_path=os.path.join( root, fname ),
            short_shot_fraction=short_shot_fraction )
        sample_fps = bu_detector.get_profile_sample_fps( profile_name )
        if profile_name:
            # Report the signal the profiles actually match on, not
            # whichever one is handy: printing a cut rate beside a
            # profile chosen by median shot length reads as the reason
            # for the choice when it had nothing to do with it.
            match_on = bu_detector.get_profiles().get( 'match_on', 'cuts_per_min' )
            if match_on == 'short_shot_fraction':
                measured = ( "%.1f%% of runtime in short shots"%(
                                 100.0 * short_shot_fraction, )
                             if short_shot_fraction is not None else None )
            else:
                measured = ( "%.1f cuts/min"%(cuts_per_min,)
                             if cuts_per_min is not None else None )
            logger.info( "structure profile: %s (%s), sampling at %s fps"%(
                profile_name,
                measured or "no shot-cut data, using the default",
                sample_fps ) )

        if paths is None or paths['sample_fps'] != sample_fps:
            paths = _build_paths( censored_folder, stem, file_hash, picture_sizes,
                                  preview_mode_enabled, preview_offset_seconds, sample_fps,
                                  output_label=output_label )
            if already_done( paths ):
                logger.info( "skipping %s (output already present and valid): %s"%(label, fname) )
                return 'skipped'

        raw_boxes = detect_boxes_for_video(
            source, fname, session, file_hash, picture_sizes, paths['preview_suffix'],
            preview_offset_seconds, effective_preview_seconds, logger, sample_fps )

        boxes, track_stats = bu_track.prepare_boxes_for_render(
            raw_boxes, source.width, source.height, shot_cut_times,
            logger=logger, profile_name=profile_name, sample_fps=sample_fps )
        track_stats['structure_profile'] = profile_name
        track_stats['cuts_per_min'] = ( round( cuts_per_min, 2 )
                                        if cuts_per_min is not None else None )
        track_stats['median_shot_seconds'] = ( round( median_shot_seconds, 3 )
                                               if median_shot_seconds is not None else None )
        track_stats['short_shot_fraction'] = ( round( short_shot_fraction, 4 )
                                              if short_shot_fraction is not None else None )

        detection_finished_at = time.perf_counter()

        render_stats = bu_render.render_video(
            source, boxes, paths['intermediate'], paths['final'], active_encode_preset,
            preview_mode_enabled, preview_window_unlimited,
            preview_offset_seconds, preview_max_seconds, logger )

        finished_at = time.perf_counter()

        delete_note = bu_config.delete_file_with_probability( uncensored_path, paths['final'] )
        if delete_note:
            logger.info( "%s%s"%(fname, delete_note) )

        detection_seconds = detection_finished_at - started_at
        encode_seconds = finished_at - detection_finished_at
        logger.info( "done %s: %s  detect=%.1fs render=%.1fs total=%.1fs  "
                     "raw=%d rendered=%d interpolated=%d"%(
            label, fname, detection_seconds, encode_seconds, finished_at - started_at,
            track_stats.get( 'raw_detections', 0 ), track_stats.get( 'rendered_boxes', 0 ),
            track_stats.get( 'interpolated', 0 ) ) )

        bu_log.write_stats( _stats_record(
            fname, paths, preview_mode_enabled, preview_offset_seconds,
            detection_seconds, encode_seconds, finished_at - started_at,
            picture_sizes, session, source, track_stats, render_stats ) )
        return 'processed'

    except bu_signals.Interrupted:
        raise
    except Exception as err:
        logger.warning( "failed %s: %s (%r)"%(label, fname, err) )
        logger.debug( "failure detail", exc_info=True )
        return 'failed'
    finally:
        source.release()


def _stats_record( fname, paths, preview_mode_enabled, preview_offset_seconds,
                   detection_seconds, encode_seconds, total_seconds,
                   picture_sizes, session, source, track_stats, render_stats ):
    """
    Build the JSON-Lines stats row for one processed file.

    Everything recorded here exists because it has, at some point, been
    the thing needed to explain a result after the fact: which backend
    and variant ran, at which sizes and batch size, on which execution
    provider (so a slow run can be attributed to a silent CPU fallback),
    and which cache keys the output filename carries.
    """
    backend_name = bu_detector.selected_backend_name()
    record = {
        'file': fname,
        'output': os.path.basename( paths['final'] ),
        'preview_mode': preview_mode_enabled,
        'preview_offset_seconds': round( preview_offset_seconds, 3 ) if preview_mode_enabled else None,
        'detection_seconds': round( detection_seconds, 3 ),
        'encode_seconds': round( encode_seconds, 3 ),
        'total_seconds': round( total_seconds, 3 ),
        # Run identity - what produced these numbers.
        'detector_backend': backend_name,
        'detector_variant': bu_detector.selected_variant_name( backend_name ),
        'execution_providers': bu_detector.session_provider_summary( session ),
        'picture_sizes': list( picture_sizes ),
        'video_censor_fps': paths.get( 'sample_fps',
                                       getattr( betaconfig, 'video_censor_fps', None ) ),
        'nn_batch_size': bu_detector.get_nn_batch_size( backend_name ),
        'global_min_prob': getattr( betaconfig, 'global_min_prob', betaconst.global_min_prob ),
        'detection_key': paths['detection_key'],
        'censor_key': paths['censor_key'],
        'encode_key': paths['encode_key'],
        'transcoded_source': source.transcoded,
        'video_width': source.width,
        'video_height': source.height,
        'video_fps': round( source.vid_fps, 4 ) if source.vid_fps else None,
        'video_frames': int( source.num_frames ),
    }
    record.update( { 'track_%s'%(key): value for key, value in track_stats.items() } )
    record.update( { 'render_%s'%(key): value for key, value in render_stats.items() } )

    # Flat aliases for the field names the analysis tools have always
    # read (summarize_tune_sweep.py, batch_ab_test.py). Cheap to keep,
    # and renaming them would break every historical stats file those
    # tools can still be pointed at.
    record['label_counts'] = track_stats.get( 'label_counts', {} )
    record['interpolated_boxes'] = track_stats.get( 'interpolated', 0 )
    record['suppression_counts'] = {
        rule: entry['total'] if isinstance( entry, dict ) else entry
        for rule, entry in track_stats.get( 'suppression_counts', {} ).items() }
    return record


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """
    Walk betaconst.video_path_uncensored and censor everything in it.

    Returns:
        A process exit code: 0 for a clean finish, 130 for a user
        interrupt (the conventional 128 + SIGINT).
    """
    logger = _startup()
    _log_run_banner( logger )

    preview_mode_enabled = getattr( betaconfig, 'preview_mode_enabled', False )
    preview_max_seconds = getattr( betaconfig, 'preview_max_seconds', 20 )
    active_encode_preset = _resolve_active_encode_preset()

    session = bu_detector.get_detector().get_session()

    counts = { 'processed': 0, 'skipped': 0, 'failed': 0 }
    exit_code = 0
    run_started_at = time.perf_counter()

    try:
        for root, _dirs, file_names in os.walk( betaconst.video_path_uncensored ):
            censored_folder = root.replace(
                betaconst.video_path_uncensored, betaconst.video_path_censored, 1 )
            os.makedirs( censored_folder, exist_ok=True )
            logger.info( "scanning %s (%d file(s))"%(root, len( file_names )) )

            for file_index, fname in enumerate( sorted( file_names ) ):
                bu_signals.check()
                outcome = process_one_video(
                    root, fname, censored_folder, file_index, len( file_names ), session,
                    preview_mode_enabled, preview_max_seconds, active_encode_preset, logger )
                counts[outcome] += 1
    except bu_signals.Interrupted:
        logger.warning( "run stopped by user after %s"%(
            bu_log.format_duration( time.perf_counter() - run_started_at ) ) )
        exit_code = 130
    finally:
        bu_hash.flush_file_hash_memo()

    logger.info( "run finished in %s: %d processed, %d skipped, %d failed"%(
        bu_log.format_duration( time.perf_counter() - run_started_at ),
        counts['processed'], counts['skipped'], counts['failed'] ) )
    return exit_code


if __name__ == '__main__':
    raise SystemExit( main() )
