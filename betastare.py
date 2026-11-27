"""
betastare.py - Entry point for censoring still images.

Walks betaconst.picture_path_uncensored recursively and, for every image
found, detects and censors the configured body parts and writes the
result into the mirrored folder under betaconst.picture_path_censored.
An image whose output already exists is skipped: the output name carries
the image's content hash and the detection and censor keys, so an
existing output is exactly the one this run would write.

    main()
      -> _parse_cli_args_and_validate_config()   CLI args, config checks
      -> _count_total_files()                    for N/total progress
      -> for each file found by os.walk:
          -> _process_one_file()
              -> _censored_output_path()          output name
              -> _raw_boxes_for_all_sizes()       detect at every
                                                  picture size, each
                                                  size cached per image
              -> bu_still.still_boxes()           looks resolved as a
                                                  video's first frame
                                                  would resolve them
              -> bu_censor.censor_img_for_boxes() render (or reverse
                                                  mode's compositor)
"""

import hashlib
import os
import time

import cv2

import betaconfig
import betaconst
import betautils_cache_paths as bu_cache
import betautils_censor as bu_censor
import betautils_cli as bu_cli
import betautils_config as bu_config
import betautils_detector as bu_detector
import betautils_hash as bu_hash
import betautils_log as bu_log
import betautils_person as bu_person
import betautils_reverse as bu_reverse
import betautils_runlock as bu_runlock
import betautils_signals as bu_signals
import betautils_still as bu_still


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def _parse_cli_args_and_validate_config():
    """
    Parse CLI flags, apply them over betaconfig, and validate.

    Fail-fast: a wrong configuration, or one with nothing to censor,
    exits here before any model is loaded.

    Side effects:
        Mutates betaconfig with the CLI overrides; may SystemExit on a
        bad config or when nothing is chosen to render.
    """
    parser = bu_cli.build_arg_parser( "BetaStare: censor photos",
                                      include_preview=False, include_logging=False )
    args = parser.parse_args()
    bu_cli.apply_cli_overrides( betaconfig, args )

    bu_log.log_data_layout( bu_log.get_logger() )
    bu_log.get_logger().info( "pictures: put them in %s; results go to %s"%(
        betaconst.picture_path_uncensored, betaconst.picture_path_censored ) )

    bu_config.validate_config()
    bu_config.exit_if_nothing_to_render( bu_log.get_logger() )


def _count_total_files():
    """
    Every file under the uncensored picture tree, images or not.

    The main walk meets and skips non-images too, so counting all files
    keeps the N/total progress consistent with it.
    """
    total = 0
    for _root, _dirs, names in os.walk( betaconst.picture_path_uncensored ):
        total += len( names )
    return total


# ---------------------------------------------------------------------------
# Per file
# ---------------------------------------------------------------------------

def _censored_output_path( censored_folder, stem, suffix, image_hash ):
    """
    Where one image's censored copy goes.

    betautils_cache_paths owns every name BetaSuite writes; this only
    gathers its inputs.
    """
    return bu_cache.picture_output_path( censored_folder, stem, suffix, image_hash,
                                         bu_detector.get_picture_sizes(),
                                         bu_cache.detection_key(), bu_cache.censor_key(),
                                         bu_cache.output_mode_marker() )


def _raw_boxes_for_size( image, size, session, image_hash ):
    """
    Raw detections for one image at one picture size, cached.

    Returns:
        ( raw_boxes, used_neural_net ): used_neural_net is False when the
        detections came from the cache.
    """
    global_min_prob = getattr( betaconfig, 'global_min_prob', betaconst.global_min_prob )
    backend_name = bu_detector.selected_backend_name()
    # The backend is part of the path so a switched backend is never
    # served another model's detections. This is the one shared path
    # builder; never a local copy of its formula.
    path = bu_cache.pic_hash_path_for( image_hash, size, global_min_prob, backend_name )
    if os.path.exists( path ):
        return bu_hash.read_json( path ), False

    raw_boxes = bu_detector.get_detector().raw_boxes_for_img( image, size, session, 0 )
    bu_hash.write_json( raw_boxes, path )
    return raw_boxes, True


def _raw_boxes_for_all_sizes( image, session, image_hash ):
    """
    Raw detections for one image at every picture size, in order.

    Returns:
        ( all_raw_boxes, used_neural_net ): one list of boxes per size.
        used_neural_net is the LAST size's answer, not an OR over sizes;
        it only feeds the nn= field of the progress line.
    """
    all_raw_boxes = []
    used_neural_net = False
    for size in bu_detector.get_picture_sizes():
        raw_boxes, used_neural_net = _raw_boxes_for_size( image, size, session, image_hash )
        all_raw_boxes.append( raw_boxes )

    if betaconfig.debug_mode & 1:
        bu_log.get_logger().debug( "raw detections: %r"%( all_raw_boxes, ) )
    return all_raw_boxes, used_neural_net


def _render( image, all_raw_boxes, boxes, img_w, img_h ):
    """Censor one image: reverse mode's compositor, or the normal render."""
    if not bu_reverse.enabled():
        return bu_censor.censor_img_for_boxes( image, boxes )

    flat = [ raw_box for raw_boxes in all_raw_boxes for raw_box in raw_boxes ]
    frame_alpha = bu_reverse.frame_alpha_for_still( flat, boxes )
    window_mask = None
    if frame_alpha > 0 and bu_reverse.person_windows_enabled():
        people = bu_person.persons_for_imgs( [ image ], [ 0.0 ] )
        window_mask = bu_person.window_mask_for_still( people, flat, img_w, img_h )
    return bu_censor.censor_img_reverse( image, boxes, frame_alpha,
                                         bu_reverse.settings()['frame_style'], window_mask )


def _process_one_file( root, fname, censored_folder, file_index, total_files, session, logger ):
    """
    Censor one file into censored_folder.

    One bad file never stops the run: any failure is logged and the walk
    moves on. A deliberate stop (bu_signals.Interrupted) still propagates.

    Args:
        root: The folder holding the file.
        fname: The file's name.
        censored_folder: Where its output goes.
        file_index, total_files: For the N/total progress lines.
        session: The detector session.
        logger: Where progress goes.
    """
    try:
        started_at = time.perf_counter()
        stem, suffix = os.path.splitext( fname )
        image = cv2.imread( os.path.join( root, fname ) )
        if image is None:
            logger.debug( "skipping %d/%d (not an image): %s"%( file_index, total_files, fname ) )
            return

        img_h, img_w = image.shape[:2]
        # The last 16 hex characters of the pixels' MD5. Output names and
        # caches depend on this exact value.
        image_hash = hashlib.md5( image.tobytes() ).hexdigest()[16:32]
        output_path = _censored_output_path( censored_folder, stem, suffix, image_hash )
        hashed_at = time.perf_counter()

        if os.path.exists( output_path ):
            logger.info( "skipping %d/%d (output exists): %s"%( file_index, total_files, fname ) )
            return

        all_raw_boxes, used_neural_net = _raw_boxes_for_all_sizes( image, session, image_hash )
        detected_at = time.perf_counter()

        boxes = bu_still.still_boxes( all_raw_boxes, img_w, img_h )
        image = _render( image, all_raw_boxes, boxes, img_w, img_h )
        rendered_at = time.perf_counter()

        cv2.imwrite( output_path, image )
        written_at = time.perf_counter()

        logger.info( "processed %d/%d (nn=%d) hash=%.3fs detect=%.3fs render=%.3fs write=%.3fs "
                     "total=%.3fs: %s"%(
            file_index, total_files, used_neural_net,
            hashed_at - started_at, detected_at - hashed_at, rendered_at - detected_at,
            written_at - rendered_at, written_at - started_at, fname ) )
    except bu_signals.Interrupted:
        raise
    except Exception as err:
        logger.warning( "failed %d/%d: %s (%r)"%( file_index, total_files, fname, err ) )
        logger.debug( "failure detail", exc_info=True )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """
    Censor everything in betaconst.picture_path_uncensored.

    Returns:
        A process exit code: 0 for a clean finish, bu_runlock.EXIT_LOCKED
        when another run holds the library's run lock, 130 for a user
        stop (the conventional 128 + SIGINT).
    """
    _parse_cli_args_and_validate_config()
    logger = bu_log.get_logger()

    try:
        run_lock = bu_runlock.acquire_library( 'betastare', logger=logger )
    except bu_runlock.RunLocked as locked:
        logger.error( locked.message() )
        return bu_runlock.EXIT_LOCKED

    bu_signals.install_handler( logger )

    try:
        session = bu_detector.get_detector().get_session()
    except BaseException:
        # A model that fails to load, or a Ctrl-C while it loads, must
        # not leave the lock behind for the next run.
        run_lock.release()
        raise

    logger.info( "BetaStare starting: backend=%s picture_sizes=%s"%(
        bu_detector.selected_backend_name(), bu_detector.get_picture_sizes() ) )
    if bu_reverse.enabled():
        logger.info( "REVERSE MODE: windows (uncensored) %s, censored %s, trigger labels %s"%(
            bu_reverse.window_labels(), bu_reverse.censor_labels(),
            sorted( bu_reverse.trigger_labels() ) ) )

    exit_code = 0
    total = _count_total_files()
    file_index = 0
    try:
        for root, _dirs, names in os.walk( betaconst.picture_path_uncensored ):
            output_folder = root.replace( betaconst.picture_path_uncensored,
                                          betaconst.picture_path_censored, 1 )
            os.makedirs( output_folder, exist_ok=True )
            logger.info( "scanning %s (%d file(s))"%( root, len( names ) ) )
            for name in sorted( names ):
                bu_signals.check()
                file_index += 1
                _process_one_file( root, name, output_folder, file_index, total, session, logger )
    except bu_signals.Interrupted:
        logger.warning( "run stopped by user" )
        exit_code = 130
    finally:
        # However the walk ends, the hash memo is saved and the lock goes
        # back, so the next run neither re-hashes nor reports it left
        # behind.
        bu_hash.flush_file_hash_memo()
        run_lock.release()
    return exit_code


if __name__ == '__main__':
    raise SystemExit( main() )
