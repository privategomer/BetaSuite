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
  1. Rewrites betaconfig.py's item_overrides['exposed_breast']['censor_style']
     and/or default_position_smoothing to the variant's values (always
     computed fresh from the REAL original file content, never cumulative
     from the previous variant's edit).
  2. Validates the rewritten file (ast.parse, then betautils_config.
     validate_config()) before ever touching betatv.py - a variant that
     fails validation is skipped (logged, not silently ignored) rather
     than aborting the whole run.
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
  6. Restores betaconfig.py to its real original content, every time,
     whether the variant succeeded, failed, or was skipped.

At the very end (or if you Ctrl-C partway through), betaconfig.py is
guaranteed back to byte-for-byte what it was before this script ran -
review the final log line confirming that rather than just assuming it.

Prerequisite: exactly one video under ../uncensored_vids (this script
tests against a single file on purpose - see the conversation this was
built from). picture_sizes/video_censor_fps/min_prob come from whatever
betaconfig.py currently has live for exposed_breast/exposed_vulva - the
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
import re
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
BETACONFIG_PATH = os.path.join( BASE_DIR, 'betaconfig.py' )
BACKUP_PATH = BETACONFIG_PATH + '.bar_test_backup'
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


log = _setup_logging()


def _read_original_betaconfig():
    if os.path.exists( BACKUP_PATH ):
        log.error( "%s already exists - this means an earlier run of this script did not finish cleanly "
                   "(crashed, was killed, or the machine restarted mid-run) and betaconfig.py may currently "
                   "be sitting on a TEST VARIANT, not your real config. Refusing to proceed automatically: "
                   "compare betaconfig.py against %s yourself, restore whichever is actually correct, delete "
                   "%s once you've confirmed betaconfig.py is right, then re-run this script."%(
                       BACKUP_PATH, BACKUP_PATH, BACKUP_PATH ) )
        sys.exit(1)
    with open( BETACONFIG_PATH, 'r', encoding='UTF-8' ) as f:
        text = f.read()
    with open( BACKUP_PATH, 'w', encoding='UTF-8' ) as f:
        f.write( text )
    log.debug( "backed up current betaconfig.py to %s before making any changes"%(BACKUP_PATH) )
    return text


def _restore_betaconfig( original_text ):
    with open( BETACONFIG_PATH, 'w', encoding='UTF-8' ) as f:
        f.write( original_text )
    with open( BETACONFIG_PATH, 'r', encoding='UTF-8' ) as f:
        written_back = f.read()
    if written_back != original_text:
        log.error( "betaconfig.py restore did not verify byte-for-byte after writing - "
                   "DO NOT trust betaconfig.py right now. The untouched original is preserved at %s - "
                   "copy it over betaconfig.py by hand."%(BACKUP_PATH) )
        sys.exit(1)
    log.debug( "betaconfig.py restored and verified against the backup" )


CENSOR_STYLE_PATTERN = re.compile(
    r"(    'exposed_breast': \{\n        'censor_shape': 'circle',\n        'censor_style': \[\n)(.*?)(\n        \],\n)",
    re.S )
SMOOTHING_PATTERN = re.compile( r'^default_position_smoothing = [0-9.]+(\s*#.*)?$', re.M )


def _format_style_entry( entry ):
    parts = []
    for key in ( 'type', 'shape', 'color', 'merge', 'span_extend', 'thickness', 'feather', 'weight', 'width_area_safety', 'height_area_safety' ):
        if key in entry:
            parts.append( "%r: %r"%(key, entry[key]) )
    return( "            { %s },"%(", ".join( parts )) )


def build_variant_text( original_text, variant ):
    text = original_text

    if variant.get( 'censor_style' ) is not None:
        m = CENSOR_STYLE_PATTERN.search( text )
        if not m:
            raise RuntimeError(
                "could not find item_overrides['exposed_breast']['censor_style'] in betaconfig.py at its "
                "expected anchor - betaconfig.py's structure has changed since this script was written "
                "against it. Ask Claude to regenerate run_bar_smoothing_test.py against the current betaconfig.py." )
        new_body = "\n".join( _format_style_entry( e ) for e in variant['censor_style'] )
        text = text[:m.start()] + m.group(1) + new_body + m.group(3) + text[m.end():]

    if variant.get( 'position_smoothing' ) is not None:
        m = SMOOTHING_PATTERN.search( text )
        if not m:
            raise RuntimeError(
                "could not find default_position_smoothing in betaconfig.py at its expected anchor - "
                "betaconfig.py's structure has changed since this script was written against it. "
                "Ask Claude to regenerate run_bar_smoothing_test.py against the current betaconfig.py." )
        comment = m.group(1) or "  # 0-1: lower = smoother, higher = snappier/more jitter"
        replacement = "default_position_smoothing = %s%s"%(variant['position_smoothing'], comment)
        text = text[:m.start()] + replacement + text[m.end():]

    return text


def _validate_betaconfig():
    # two separate subprocess checks, cheapest/most-diagnostic first: a
    # syntax error should never even try to import betaconfig (clearer
    # error, no chance of a partial/confusing traceback from deeper in
    # validate_config)
    syntax_check = subprocess.run(
        [ sys.executable, '-c', 'import ast; ast.parse(open("betaconfig.py").read())' ],
        cwd=BASE_DIR, capture_output=True, text=True )
    if syntax_check.returncode != 0:
        return( False, "betaconfig.py has a syntax error after patching:\n%s"%(syntax_check.stderr) )

    config_check = subprocess.run(
        [ sys.executable, '-c', 'import betautils_config as c; c.validate_config(); print("OK")' ],
        cwd=BASE_DIR, capture_output=True, text=True )
    if config_check.returncode != 0 or 'OK' not in config_check.stdout:
        return( False, "betaconfig.py failed validate_config() after patching:\n%s%s"%(config_check.stdout, config_check.stderr) )

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


def _run_betatv_preview( preview_start_seconds, preview_seconds ):
    cmd = [ sys.executable, 'betatv.py', '--preview', 'on',
            '--preview-start-seconds', str(preview_start_seconds),
            '--preview-seconds', str(preview_seconds) ]
    log.info( "running: %s"%(" ".join(cmd)) )
    proc = subprocess.Popen( cmd, cwd=BASE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1 )
    for line in proc.stdout:
        log.log( TRACE, "betatv.py: %s"%(line.rstrip()) )
    returncode = proc.wait()
    if returncode != 0:
        log.error( "betatv.py exited with status %d - see the trace-level lines above (or the log file) for its output"%(returncode) )
    return( returncode == 0 )


def main():
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

    original_text = _read_original_betaconfig()

    try:
        video_path, video_name = _find_single_uncensored_video()
    except Exception as err:
        log.error( str(err) )
        _restore_betaconfig( original_text )
        os.remove( BACKUP_PATH )
        sys.exit(1)
    log.info( "testing against: %s"%(video_name) )

    sys.path.insert( 0, BASE_DIR )
    import betautils_hash as bu_hash
    file_hash = bu_hash.md5_for_file( video_path, 16 )
    log.debug( "file_hash=%s"%(file_hash) )

    results = []
    try:
        for idx, variant in enumerate( bar_smoothing_variants.VARIANTS ):
            name = variant['name']
            log.info( "" )
            log.info( "=== variant %d/%d: %s ==="%(idx+1, len(bar_smoothing_variants.VARIANTS), name) )
            log.info( variant.get( 'description', '(no description)' ) )
            t0 = time.time()

            try:
                patched_text = build_variant_text( original_text, variant )
            except RuntimeError as err:
                log.error( "skipping %s: %s"%(name, err) )
                results.append( ( name, 'skipped', str(err) ) )
                continue

            with open( BETACONFIG_PATH, 'w', encoding='UTF-8' ) as f:
                f.write( patched_text )

            ok, reason = _validate_betaconfig()
            if not ok:
                log.error( "skipping %s - patched betaconfig.py failed validation: %s"%(name, reason) )
                results.append( ( name, 'skipped', reason ) )
                _restore_betaconfig( original_text )
                continue

            _quarantine_existing_previews( file_hash, name )

            success = _run_betatv_preview( preview_start_seconds, preview_seconds )
            if not success:
                results.append( ( name, 'betatv.py failed', None ) )
                _restore_betaconfig( original_text )
                continue

            dest = _collect_fresh_output( file_hash, name )
            _restore_betaconfig( original_text )

            elapsed = time.time() - t0
            if dest:
                log.info( "%s done in %.1fs"%(name, elapsed) )
                results.append( ( name, 'ok', dest ) )
            else:
                results.append( ( name, 'no output found', None ) )
    finally:
        _restore_betaconfig( original_text )
        if os.path.exists( BACKUP_PATH ):
            os.remove( BACKUP_PATH )
        log.info( "" )
        log.info( "betaconfig.py confirmed restored to its original content - backup file removed" )

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
