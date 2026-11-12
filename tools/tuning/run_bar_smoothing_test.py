#!/usr/bin/env python3
"""
run_bar_smoothing_test.py - runs a real preview render (visual output you
watch, not just stats) once per variant defined in bar_smoothing_variants.py
against the single video currently under ../uncensored_vids, so you can
compare bar width/pairing/aspect-ratio options and default_position_smoothing
values by eye.

Why this needs its own harness instead of replay_tune.py: replay_tune.py
deliberately only reports counts (raw detections, suppression, tracked/
interpolated box counts) - it says so in its own docstring, because it's
built for the class_suppression/interpolation_max_gap kind of tuning where
a number IS the answer. censor_style thickness/merge/width_area_safety and
default_position_smoothing don't move any of those counts at all (they
only change how an already-tracked box gets DRAWN or SMOOTHED) - so
replay_tune.py would show a flat "no change" for every variant here even
though the rendered video looks completely different. There's no way
around actually re-rendering and watching the output for this kind of
question.

What this script does, once per variant:
  1. Writes a local config for the variant: your betaconfig_local.py
     with item_overrides['exposed_breast']['censor_style'] and/or
     default_position_smoothing set to the variant's values, into
     ../bar_smoothing_test_results/<variant_name>.betaconfig_local.py.
     betaconfig.py and your own betaconfig_local.py are never written:
     betatv.py reads the variant's file through BETASUITE_LOCAL_CONFIG.
     This used to rewrite betaconfig.py in place and restore it after,
     so a crash or a killed run left the shipped file on a test variant.
  2. Validates the variant (betautils_config.validate_config() with that
     file) before ever touching betatv.py - a variant that fails
     validation is skipped (logged, not silently ignored) rather than
     aborting the whole run.
  3. Quarantines (moves, never deletes) any existing preview output for
     this video already sitting in the censored folder. This matters even
     for variants that only change censor_style, and is CRITICAL for the
     default_position_smoothing variants specifically: betatv.py's output
     filename embeds a censor_hash, but that hash is computed only from
     parts_to_blur/picture_sizes/overlap+scale strategy/watermark flag
     (see betautils_hash.get_censor_hash) - default_position_smoothing
     is NOT part of it. That means two runs that only differ in
     default_position_smoothing would ask betatv.py to write to the
     EXACT SAME output path, and betatv.py's own resumability logic
     ("chunk/avi already exists and passes ffprobe -> skip re-rendering
     it") would silently reuse the FIRST run's stale render instead of
     actually re-rendering with the new smoothing value - you'd end up
     comparing one real render against 4 identical copies of it. Moving
     anything already there out of the way before each run forces a
     genuine fresh render every time, for every variant.
  4. Runs betatv.py --preview on for the configured preview window,
     streaming its output into this run's log file (and to your terminal,
     filtered by BAR_TEST_LOG_LEVEL).
  5. Copies whatever fresh preview .mp4 that run produced into
     ../bar_smoothing_test_results/<variant_name>.mp4 - the original stays
     in the normal censored folder too, this just also gives you one
     clearly-labeled copy per variant, side by side, for easy comparison.

Nothing to restore afterwards, whether a variant succeeded, failed, was
skipped or the run was interrupted: no config file was edited.

Prerequisite: exactly one video under ../uncensored_vids (this script
tests against a single file on purpose - see the conversation this was
built from). picture_sizes/video_censor_fps/min_prob come from whatever
your config currently has live for exposed_breast/exposed_vulva - the
first variant run will do real neural-net detection once (if no matching
cache exists yet for this exact preview window) and every variant after
that reuses the same cache (censor_style/position_smoothing changes never
affect detection, only rendering), so only the first run should be slow.

Usage (single-click - see run_bar_smoothing_test.sh):
    ./run_bar_smoothing_test.sh
    # or directly:
    python3 run_bar_smoothing_test.py
    PREVIEW_START_SECONDS=15 PREVIEW_SECONDS=20 BAR_TEST_LOG_LEVEL=debug python3 run_bar_smoothing_test.py

Edit bar_smoothing_variants.py to change which variants get tested - this
script always re-imports it fresh, nothing about the variant list is
hardcoded here.
"""

import glob
import logging
import os
import shutil
import subprocess
import sys
import time

TRACE = 5
logging.addLevelName( TRACE, 'TRACE' )

SCRIPT_DIR = os.path.dirname( os.path.abspath( __file__ ) )
# BASE_DIR is BetaSuite-0.2.4 (the app root) regardless of where this
# script itself lives - it moved into tools/tuning/, but betatv.py,
# betaconfig.py, and every betaconst.*_path constant this script reads
# still need to be resolved/run relative to the app root, not to
# tools/tuning/.
BASE_DIR = os.path.abspath( os.path.join( SCRIPT_DIR, '..', '..' ) )
RESULTS_DIR = os.path.join( BASE_DIR, '..', 'bar_smoothing_test_results' )
QUARANTINE_DIR = os.path.join( BASE_DIR, '..', 'bar_smoothing_test_quarantine' )

_LEVEL_MAP = { 'trace': TRACE, 'debug': logging.DEBUG, 'info': logging.INFO, 'warn': logging.WARNING, 'error': logging.ERROR }


def _setup_logging():
    log_dir = os.path.join( BASE_DIR, '..', 'output', 'logs' )
    os.makedirs( log_dir, exist_ok=True )
    log_path = os.path.join( log_dir, 'run_bar_smoothing_test.log' )
    logger = logging.getLogger( 'bar_smoothing_test' )
    logger.setLevel( TRACE )
    logger.propagate = False

    file_handler = logging.FileHandler( log_path, encoding='UTF-8' )
    file_handler.setLevel( TRACE )
    file_handler.setFormatter( logging.Formatter( '%(asctime)s [%(levelname)s] %(message)s' ) )
    logger.addHandler( file_handler )

    stdout_level_name = os.environ.get( 'BAR_TEST_LOG_LEVEL', 'info' ).lower()
    stdout_level = _LEVEL_MAP.get( stdout_level_name, logging.INFO )
    stream_handler = logging.StreamHandler( sys.stdout )
    stream_handler.setLevel( stdout_level )
    stream_handler.setFormatter( logging.Formatter( '[%(levelname)s] %(message)s' ) )
    logger.addHandler( stream_handler )

    logger.info( "logging to %s (stdout filtered at '%s' - set BAR_TEST_LOG_LEVEL=trace|debug|info|warn|error to change)"%(
        os.path.abspath( log_path ), stdout_level_name ) )
    return logger


# Handlers are attached in main(): importing this module (a test does)
# must not write into the real output/logs.
log = logging.getLogger( 'bar_smoothing_test' )


def _local_config_text():
    """Your local config's text (betaconfig_local.py, or BETASUITE_LOCAL_CONFIG's file), or ''."""
    sys.path.insert( 0, BASE_DIR )
    import betautils_local as bu_local
    path = bu_local.configured_path()
    if path and os.path.isfile( path ):
        with open( path, 'r', encoding='UTF-8' ) as handle:
            return handle.read()
    return ''


def build_variant_text( base_text, variant ):
    """The local config for one variant: base_text with its values set, as local_config.py set would."""
    sys.path.insert( 0, BASE_DIR )
    import betautils_local as bu_local
    updates = {}
    if variant.get( 'censor_style' ) is not None:
        updates[ ( 'item_overrides', 'exposed_breast', 'censor_style' ) ] = list( variant['censor_style'] )
    if variant.get( 'position_smoothing' ) is not None:
        updates[ ( 'default_position_smoothing', ) ] = variant['position_smoothing']
    try:
        return bu_local.edit_text( base_text or bu_local.template_text(), updates,
                                   comment='bar smoothing variant %s'%( variant['name'], ) )
    except ValueError as err:
        raise RuntimeError( "your local config cannot be edited: %s"%( err, ) )


def _variant_environment( local_path ):
    return dict( os.environ, BETASUITE_LOCAL_CONFIG=local_path )


def _validate_variant( local_path ):
    config_check = subprocess.run(
        [ sys.executable, '-c', 'import betautils_config as c; c.validate_config(); print("OK")' ],
        cwd=BASE_DIR, capture_output=True, text=True, env=_variant_environment( local_path ) )
    if config_check.returncode != 0 or 'OK' not in config_check.stdout:
        return( False, "the variant failed validate_config():\n%s%s"%( config_check.stdout, config_check.stderr ) )
    return( True, None )


def _find_single_uncensored_video():
    # import here (not at module top) so a missing/misconfigured betaconst
    # fails inside main()'s error handling, not at import time before
    # logging is even set up
    sys.path.insert( 0, BASE_DIR )
    import betaconst
    video_dir = betaconst.video_path_uncensored
    if not os.path.isabs( video_dir ):
        video_dir = os.path.join( BASE_DIR, video_dir )
    entries = [ f for f in os.listdir( video_dir ) if os.path.isfile( os.path.join( video_dir, f ) ) ]
    if len( entries ) == 0:
        raise RuntimeError( "no files found under %s - this script expects exactly one test video there"%(video_dir) )
    if len( entries ) > 1:
        raise RuntimeError(
            "found %d files under %s, expected exactly 1 (this script is built to test against a single "
            "video) - move the others out temporarily: %s"%(len(entries), video_dir, entries) )
    return( os.path.join( video_dir, entries[0] ), entries[0] )


def _quarantine_existing_previews( file_hash, run_label ):
    import betaconst
    censored_dir = betaconst.video_path_censored
    if not os.path.isabs( censored_dir ):
        censored_dir = os.path.join( BASE_DIR, censored_dir )
    pattern = os.path.join( censored_dir, '*%s*-preview*'%(file_hash) )
    matches = glob.glob( pattern )
    if not matches:
        log.debug( "no pre-existing preview output for this video/hash - nothing to quarantine" )
        return
    dest_dir = os.path.join( QUARANTINE_DIR, run_label )
    os.makedirs( dest_dir, exist_ok=True )
    for path in matches:
        dest = os.path.join( dest_dir, os.path.basename( path ) )
        shutil.move( path, dest )
        log.warning( "moved pre-existing preview output out of the way (never deleted): %s -> %s"%(path, dest) )


def _collect_fresh_output( file_hash, variant_name ):
    import betaconst
    censored_dir = betaconst.video_path_censored
    if not os.path.isabs( censored_dir ):
        censored_dir = os.path.join( BASE_DIR, censored_dir )
    pattern = os.path.join( censored_dir, '*%s*-preview*.mp4'%(file_hash) )
    matches = glob.glob( pattern )
    if not matches:
        log.error( "betatv.py finished but no matching preview .mp4 was found at %s - check the run's own "
                   "output above/in the log for the real failure"%(pattern) )
        return( None )
    if len( matches ) > 1:
        matches.sort( key=os.path.getmtime, reverse=True )
        log.warning( "found %d matching preview outputs, using the newest by mtime: %s (others: %s)"%(
            len(matches), matches[0], matches[1:]) )
    src = matches[0]
    os.makedirs( RESULTS_DIR, exist_ok=True )
    dest = os.path.join( RESULTS_DIR, '%s.mp4'%(variant_name) )
    shutil.copy2( src, dest )
    log.info( "saved result: %s"%(os.path.abspath( dest )) )
    return( dest )


def _run_betatv_preview( preview_start_seconds, preview_seconds, local_path ):
    cmd = [ sys.executable, 'betatv.py', '--preview', 'on',
            '--preview-start-seconds', str(preview_start_seconds),
            '--preview-seconds', str(preview_seconds) ]
    log.info( "running: %s"%(" ".join(cmd)) )
    proc = subprocess.Popen( cmd, cwd=BASE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                             env=_variant_environment( local_path ) )
    for line in proc.stdout:
        log.log( TRACE, "betatv.py: %s"%(line.rstrip()) )
    returncode = proc.wait()
    if returncode != 0:
        log.error( "betatv.py exited with status %d - see the trace-level lines above (or the log file) for its output"%(returncode) )
    return( returncode == 0 )


def main():
    _setup_logging()
    try:
        import bar_smoothing_variants
    except ImportError as err:
        log.error( "could not import bar_smoothing_variants.py: %s"%(err) )
        sys.exit(1)

    preview_start_seconds = float( os.environ.get( 'PREVIEW_START_SECONDS', 10.0 ) )
    preview_seconds = float( os.environ.get( 'PREVIEW_SECONDS', 20.0 ) )
    log.info( "preview_start_seconds=%s preview_seconds=%s (override with env vars PREVIEW_START_SECONDS/PREVIEW_SECONDS)"%(
        preview_start_seconds, preview_seconds) )
    log.warning( "preview_start_seconds defaults to 10s - this is a guess since I don't know this specific "
                "video's real duration; if 10s isn't a representative point in the clip, Ctrl-C now and "
                "re-run with PREVIEW_START_SECONDS=<seconds> set to something better" )

    base_text = _local_config_text()

    try:
        video_path, video_name = _find_single_uncensored_video()
    except Exception as err:
        log.error( str(err) )
        sys.exit(1)
    log.info( "testing against: %s"%(video_name) )

    sys.path.insert( 0, BASE_DIR )
    import betautils_hash as bu_hash
    file_hash = bu_hash.md5_for_file( video_path, 16 )
    log.debug( "file_hash=%s"%(file_hash) )

    results = []
    for idx, variant in enumerate( bar_smoothing_variants.VARIANTS ):
        name = variant['name']
        log.info( "" )
        log.info( "=== variant %d/%d: %s ==="%(idx+1, len(bar_smoothing_variants.VARIANTS), name) )
        log.info( variant.get( 'description', '(no description)' ) )
        t0 = time.time()

        try:
            variant_text = build_variant_text( base_text, variant )
        except RuntimeError as err:
            log.error( "skipping %s: %s"%(name, err) )
            results.append( ( name, 'skipped', str(err) ) )
            continue

        os.makedirs( RESULTS_DIR, exist_ok=True )
        local_path = os.path.abspath( os.path.join( RESULTS_DIR, '%s.betaconfig_local.py'%( name, ) ) )
        with open( local_path, 'w', encoding='UTF-8' ) as f:
            f.write( variant_text )
        log.info( "variant config: %s"%( local_path, ) )

        ok, reason = _validate_variant( local_path )
        if not ok:
            log.error( "skipping %s - %s"%(name, reason) )
            results.append( ( name, 'skipped', reason ) )
            continue

        _quarantine_existing_previews( file_hash, name )

        success = _run_betatv_preview( preview_start_seconds, preview_seconds, local_path )
        if not success:
            results.append( ( name, 'betatv.py failed', None ) )
            continue

        dest = _collect_fresh_output( file_hash, name )
        elapsed = time.time() - t0
        if dest:
            log.info( "%s done in %.1fs"%(name, elapsed) )
            results.append( ( name, 'ok', dest ) )
        else:
            results.append( ( name, 'no output found', None ) )

    log.info( "" )
    log.info( "=== summary ===" )
    ok_count = sum( 1 for _, status, _ in results if status == 'ok' )
    log.info( "%d/%d variants produced a result"%(ok_count, len(results)) )
    for name, status, extra in results:
        if status == 'ok':
            log.info( "  OK      %s -> %s"%(name, extra) )
        else:
            log.warning( "  %-8s%s%s"%(status.upper(), name, "  (%s)"%(extra) if extra else "") )
    log.info( "results folder: %s"%(os.path.abspath( RESULTS_DIR )) )


if __name__ == '__main__':
    main()
