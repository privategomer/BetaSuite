"""
betastare.py - Entry point for censoring still images.

Walks betaconst.picture_path_uncensored recursively, and for every image
file found, detects and censors configured body parts, writing the
result under the mirrored directory structure in
betaconst.picture_path_censored. Already-censored output (same source
image content + same censoring config, per the filename's embedded
hashes) is skipped rather than reprocessed.

Orchestration:
    main()
        -> _parse_cli_args_and_validate_config()
        -> _count_total_files()
        -> for each (root, fname) found by os.walk:
            -> _process_one_file()
                -> _censored_output_path()
                -> _raw_boxes_for_all_sizes()      (only if output doesn't already exist)
                    -> _raw_boxes_for_size()        (cached per image+size+min_prob)
                -> _censorable_boxes_from_raw()
                -> _resolve_box_styles()
                -> betautils_censor.censor_img_for_boxes()
"""

import cv2
import os
import time
import hashlib

import betaconst
import betaconfig

import betautils_config as bu_config
import betautils_hash   as bu_hash
import betautils_detector as bu_detector
import betautils_censor as bu_censor
import betautils_cli    as bu_cli
import betautils_cache_paths as bu_cache
import betautils_log    as bu_log
import betautils_signals as bu_signals
import betautils_track  as bu_track


def _parse_cli_args_and_validate_config():
    """
    Parse betastare.py's command-line flags, apply any overrides onto
    betaconfig, then validate the effective config and (if configured)
    prompt for the input-delete-probability safety confirmation.

    Side effects:
        Mutates the betaconfig module via CLI overrides. May print and
        raise SystemExit(1) if the config is invalid (validate_config),
        or print/prompt/quit() if input_delete_probability is non-zero
        and the user doesn't confirm (verify_input_delete_probability).
    """
    cli_parser = bu_cli.build_arg_parser( "BetaStare: censor photos", include_preview=False, include_logging=False )
    cli_args = cli_parser.parse_args()
    bu_cli.apply_cli_overrides( betaconfig, cli_args )

    bu_config.validate_config()
    bu_config.verify_input_delete_probability()


def _count_total_files():
    """
    Count every file under betaconst.picture_path_uncensored, across all
    subdirectories, so progress can be reported as "N/total" while
    processing.

    Returns:
        Total file count (int) - includes non-image files, since the
        actual walk below also encounters and skips those, keeping the
        "N/total" counter consistent with what the main loop will
        actually iterate over.
    """
    total_files = 0
    for _root, _dir_names, file_names in os.walk( betaconst.picture_path_uncensored ):
        total_files += len( file_names )
    return total_files


def _censored_output_path( censored_folder, stem, suffix, image_hash ):
    """
    Output path for one censored image.

    Delegates to betautils_cache_paths, the single module that owns
    every name BetaSuite writes. The filename carries the source image
    hash plus the detection and censor keys, so an output produced under
    different settings can never be mistaken for this one.

    Args:
        censored_folder: Destination directory, already created.
        stem: Source filename without its extension.
        suffix: Source filename's extension, including the dot.
        image_hash: Content hash of the source image.

    Returns:
        The full destination path.
    """
    return bu_cache.picture_output_path(
        censored_folder, stem, suffix, image_hash,
        bu_detector.get_picture_sizes(),
        bu_cache.detection_key(), bu_cache.censor_key() )


def _raw_boxes_for_size( image, size, session, image_hash ):
    """
    Get raw detections for one image at one detection size, from cache
    if available, otherwise by running the neural net (and caching the
    result for next time).

    Args:
        image: The full uncensored image (BGR numpy array).
        size: The detection size to run the model at (an entry from
            betaconfig.picture_sizes).
        session: An onnxruntime.InferenceSession from
            the active detector adapter's get_session() (see
            betautils_detector.py).
        image_hash: Content hash of the source image, used as part of
            the cache filename.

    Returns:
        A (raw_boxes, used_neural_net) pair: raw_boxes is the list of
        raw box dicts for this image at this size (see the active
        detector adapter's raw_boxes_for_img - betautils_detector.py);
        used_neural_net is True if this required an actual model
        inference (cache miss), False if served entirely from cache.
    """
    # backend name baked into the cache path (part of the adapter
    # architecture) so switching betaconfig.detector_backend['selected']
    # can never silently serve back a different model's stale
    # detections under the same cache key - see betautils_cache_paths.
    # pic_hash_path_for's docstring, which documents this in full (same
    # reasoning applies here verbatim). This used to be its own
    # independent copy of the formula (a real cache-path duplication
    # bug source, see betautils_cache_paths.py's module docstring for
    # the video-side incident that formula drift already caused once) -
    # now calls the one shared implementation instead.
    global_min_prob = getattr( betaconfig, 'global_min_prob', betaconst.global_min_prob )
    backend_name = bu_detector.selected_backend_name()
    box_hash_path = bu_cache.pic_hash_path_for( image_hash, size, global_min_prob, backend_name )

    if os.path.exists( box_hash_path ):
        return bu_hash.read_json( box_hash_path ), False

    raw_boxes = bu_detector.get_detector().raw_boxes_for_img( image, size, session, 0 )
    bu_hash.write_json( raw_boxes, box_hash_path )
    return raw_boxes, True


def _raw_boxes_for_all_sizes( image, session, image_hash ):
    """
    Run (or load from cache) detection at every configured picture size.

    Args:
        image: The full uncensored image (BGR numpy array).
        session: An onnxruntime.InferenceSession from
            the active detector adapter's get_session() (see
            betautils_detector.py).
        image_hash: Content hash of the source image, used for caching.

    Returns:
        A (all_raw_boxes, used_neural_net) pair: all_raw_boxes is a list
        with one raw-box list per configured picture size; used_
        neural_net is True if ANY size required an actual model
        inference (cache miss), matching the original per-run "nn"
        column in this script's progress output (which reflects only
        the LAST size checked, not an OR across all sizes - preserved
        here exactly, since it's cosmetic and this script's own printed
        record of past behavior may reference it).
    """
    used_neural_net = False
    all_raw_boxes = []
    for size in bu_detector.get_picture_sizes():
        raw_boxes, used_neural_net = _raw_boxes_for_size( image, size, session, image_hash )
        all_raw_boxes.append( raw_boxes )

    if betaconfig.debug_mode & 1:
        bu_log.get_logger().debug( "raw detections: %r"%(all_raw_boxes,) )

    return all_raw_boxes, used_neural_net


def _censorable_boxes_from_raw( all_raw_boxes, img_w, img_h ):
    """
    Turn every raw detection (across every configured picture size) into
    a censorable box, leaving each box's censor_style UNRESOLVED (still
    whatever's configured - a single dict or a list to randomly choose
    from).

    Args:
        all_raw_boxes: A list of per-size raw-box lists, as returned by
            _raw_boxes_for_all_sizes.
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        A flat list of censorable box dicts (see
        betautils_censor.process_raw_box) across every size, with
        unresolved censor_style.
    """
    parts_to_blur = bu_config.get_parts_to_blur()
    boxes = []
    for raw_boxes in all_raw_boxes:
        for raw_box in raw_boxes:
            censorable_box = bu_censor.process_raw_box( raw_box, img_w, img_h, parts_to_blur )
            if censorable_box:
                boxes.append( censorable_box )
    return boxes


def _resolve_box_styles( boxes ):
    """
    Resolve each box's still-unresolved censor_style/censor_shape to one
    concrete choice, using the same resolution functions betatv.py's
    smooth_boxes uses for tracked video detections.

    betastare has no tracking concept (one photo detection IS the whole
    instance, unlike a video track that persists across frames), so this
    resolves once per box here rather than once per track. Skipping this
    step is exactly what broke on any item configured with a LIST
    censor_style (currently just exposed_breast) -
    collapse_boxes_for_style/censor_img_for_boxes expect a resolved dict
    and do resolved_style['type'], which raises "list indices must be
    integers or slices, not str" the moment resolved_style is still the
    unresolved list.

    No pairing logic here (unlike betatv.py's paired_style, which makes
    two simultaneous video instances share one rolled style) - two
    exposed_breast detections in the same photo can each roll
    independently; harmless today since exposed_breast's list currently
    has only one entry anyway, and only a cosmetic mismatch (not a
    crash) if that ever changes.

    Args:
        boxes: A list of censorable box dicts with unresolved
            censor_style (mutated in place).
    """
    for box in boxes:
        box['censor_style'] = bu_censor.resolve_censor_style( box['censor_style'] )
        box['censor_shape'] = bu_censor.resolve_censor_shape( box['censor_style'], box.get('censor_shape', 'box') )


def _process_one_file( root, fname, censored_folder, file_index, total_files, session, logger ):
    """
    Process exactly one file found under betaconst.picture_path_uncensored:
    skip it if it's not a readable image or its censored counterpart
    already exists, otherwise detect, censor, write, and print one
    progress line - matching this script's original per-file behavior
    (including timing breakdown and any failure recovery) exactly.

    Args:
        root: Directory fname was found in (an os.walk root).
        fname: The filename being processed.
        censored_folder: Destination directory for this root, mirroring
            root's position under betaconst.picture_path_censored
            (already created by the caller).
        file_index: 1-based index of this file across the whole run, for
            the printed "N/total" progress counter.
        total_files: Total file count across the whole run, from
            _count_total_files().
        session: An onnxruntime.InferenceSession from
            the active detector adapter's get_session() (see
            betautils_detector.py).
        logger: The shared betasuite logger.

    Side effects:
        Logs one line per file. A failure is logged and swallowed so one
        bad file never aborts the run; a deliberate interrupt is
        re-raised so Ctrl-C actually stops.
    """
    start_time = time.perf_counter()
    try:
        (stem, suffix) = os.path.splitext( fname )

        uncensored_path = os.path.join( root, fname )
        image = cv2.imread( uncensored_path )

        if image is None:
            logger.debug( "skipping %d/%d (not an image): %s"%(file_index, total_files, fname) )
            return

        (img_h, img_w, _channels) = image.shape
        image_hash = hashlib.md5(image).hexdigest()[16:]

        censored_path = _censored_output_path( censored_folder, stem, suffix, image_hash )

        after_hash_time = time.perf_counter()
        if os.path.exists( censored_path ):
            logger.info( "skipping %d/%d (output exists): %s"%(file_index, total_files, fname) )
            return

        # Timing checkpoints below intentionally mirror the original
        # script's four-way breakdown exactly: t1 (after_hash_time) is
        # after image read + hash + path construction; t2
        # (after_detect_time) is after raw detection ONLY (before
        # converting raw boxes to censorable ones or resolving styles);
        # t3 (after_render_time) is after that conversion + style
        # resolution + actually rendering the censored image (but before
        # writing it to disk); t4 (after_write_time) is after writing
        # the file and the probabilistic-delete step.
        all_raw_boxes, used_neural_net = _raw_boxes_for_all_sizes( image, session, image_hash )
        after_detect_time = time.perf_counter()

        boxes = _censorable_boxes_from_raw( all_raw_boxes, img_w, img_h )
        _resolve_box_styles( boxes )
        image = bu_censor.censor_img_for_boxes( image, boxes )
        after_render_time = time.perf_counter()

        cv2.imwrite( censored_path, image )
        delete_result = bu_config.delete_file_with_probability( uncensored_path, censored_path )
        after_write_time = time.perf_counter()

        logger.info( "processed %d/%d%s (nn=%d) hash=%.3fs detect=%.3fs render=%.3fs write=%.3fs "
                     "total=%.3fs: %s"%(
            file_index, total_files, delete_result, used_neural_net,
            after_hash_time-start_time, after_detect_time-after_hash_time,
            after_render_time-after_detect_time, after_write_time-after_render_time,
            after_write_time-start_time, fname ) )
    except bu_signals.Interrupted:
        # A deliberate stop must never be mistaken for a bad file.
        raise
    except Exception as err:
        logger.warning( "failed %d/%d: %s (%r)"%(file_index, total_files, fname, err) )
        logger.debug( "failure detail", exc_info=True )


def main():
    """
    Entry point: validate config, then censor every image under
    betaconst.picture_path_uncensored, mirroring the directory structure
    into betaconst.picture_path_censored.

    Returns:
        A process exit code: 0 clean, 130 on a user interrupt.
    """
    _parse_cli_args_and_validate_config()

    logger = bu_log.get_logger()
    bu_signals.install_handler( logger )

    session = bu_detector.get_detector().get_session()
    logger.info( "BetaStare starting: backend=%s picture_sizes=%s"%(
        bu_detector.selected_backend_name(), bu_detector.get_picture_sizes() ) )

    total_files = _count_total_files()
    file_index = 0
    exit_code = 0

    try:
        for root, _dir_names, file_names in os.walk( betaconst.picture_path_uncensored ):
            censored_folder = root.replace(
                betaconst.picture_path_uncensored, betaconst.picture_path_censored, 1 )
            os.makedirs( censored_folder, exist_ok=True )
            logger.info( "scanning %s (%d file(s))"%(root, len( file_names )) )

            for fname in sorted( file_names ):
                bu_signals.check()
                file_index += 1
                _process_one_file( root, fname, censored_folder, file_index,
                                   total_files, session, logger )
    except bu_signals.Interrupted:
        logger.warning( "run stopped by user" )
        exit_code = 130
    finally:
        bu_hash.flush_file_hash_memo()

    return exit_code


if __name__ == '__main__':
    raise SystemExit( main() )
