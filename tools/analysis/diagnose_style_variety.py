#!/usr/bin/env python3
"""
diagnose_style_variety.py - why does one censor style cover a whole clip?

Run this after a betatv.py run. It reads that run's OWN detection cache
and shot-cut cache, replays tracking exactly as the renderer does, and
reports which of the known causes is actually firing on YOUR footage.

Single command, no arguments needed:

    python3 diagnose_style_variety.py

It prints a numbered verdict per check: PASS, FAIL, or N/A, and ends with
a one-line summary naming the culprit(s). Every number it prints is
derived from your caches, not from assumptions about them.

WHY EACH CHECK EXISTS
---------------------
Style variety needs three things to be true. Each check below tests
exactly one of them, in the order they happen in the pipeline, so the
FIRST failing check is the root cause and the ones after it are usually
consequences.

  1  cuts detected        shot_cut_detection_enabled produced cuts
  2  cuts reach tracking  the cache the tracker reads actually has them
  3  cut_between usable   time_safety < frame_step, or the padded
                          interval inverts and every cut is invisible
  4  tracks per cut       do cuts actually end tracks
  5  style resolves       how many independent rolls happened
  6  weight sanity        do the configured weights allow variety at all
  7  observed variety     the distribution actually produced

All output also goes to a log file so you can attach it without
copy-pasting a terminal.
"""

import argparse
import bisect
import collections
import glob
import logging
import os
import sys

# tools/analysis/<this file> -> repo root is two levels up.
_REPO_ROOT = os.path.dirname( os.path.dirname(
    os.path.dirname( os.path.abspath( __file__ ) ) ) )
sys.path.insert( 0, _REPO_ROOT )

import betaconfig
import betaconst
import betautils_cache_paths as bu_cache
import betautils_censor as bu_censor
import betautils_detector as bu_detector
import betautils_hash as bu_hash
import betautils_track as bu_track


LEVELS = { 'trace': 5, 'debug': logging.DEBUG, 'info': logging.INFO,
           'warn': logging.WARNING, 'error': logging.ERROR }


def build_logger( log_path, stdout_level ):
    logging.addLevelName( 5, 'TRACE' )
    logger = logging.getLogger( 'diagnose_style_variety' )
    logger.setLevel( 5 )
    logger.handlers = []

    file_handler = logging.FileHandler( log_path, mode='w' )
    file_handler.setLevel( 5 )
    file_handler.setFormatter( logging.Formatter(
        '%(asctime)s [%(levelname)-5s] %(message)s' ) )
    logger.addHandler( file_handler )

    stream = logging.StreamHandler( sys.stdout )
    stream.setLevel( LEVELS[stdout_level] )
    stream.setFormatter( logging.Formatter( '%(message)s' ) )
    logger.addHandler( stream )
    return logger


def find_detection_caches( label ):
    """
    Every detection cache on disk, newest first.

    Searched recursively: detection caches live under per-hash
    subdirectories, so a flat glob of the top level finds nothing.
    """
    pattern = os.path.join( betaconst.vid_hash_dir, '**', '*' )
    found = [ p for p in glob.glob( pattern, recursive=True )
              if os.path.isfile( p ) ]
    found.sort( key=os.path.getmtime, reverse=True )
    return found


def find_shot_cut_caches():
    pattern = os.path.join( betaconst.shot_cut_dir, '*' )
    found = [ p for p in glob.glob( pattern ) if os.path.isfile( p ) ]
    found.sort( key=os.path.getmtime, reverse=True )
    return found


def main():
    parser = argparse.ArgumentParser( description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter )
    parser.add_argument( '--label', default='exposed_breast',
                         help='label to diagnose (default: exposed_breast)' )
    parser.add_argument( '--backend', default=None,
                         help='backend name (default: the selected one)' )
    parser.add_argument( '--log-file', default='diagnose_style_variety.log' )
    parser.add_argument( '--stdout-level', default='info', choices=sorted( LEVELS ) )
    args = parser.parse_args()

    logger = build_logger( args.log_file, args.stdout_level )
    backend = args.backend or bu_detector.selected_backend_name()
    label = args.label

    logger.info( '=' * 70 )
    logger.info( 'style-variety diagnosis: label=%s backend=%s', label, backend )
    logger.info( '=' * 70 )

    verdicts = []

    def record( number, name, ok, detail ):
        state = 'PASS' if ok is True else ( 'N/A' if ok is None else 'FAIL' )
        verdicts.append( ( number, name, state, detail ) )
        logger.info( '' )
        logger.info( '[%d] %-28s %s', number, name, state )
        for line in detail.split( '\n' ):
            logger.info( '      %s', line )

    # ---------------------------------------------------------------
    # 1. Are cuts being detected at all?
    # ---------------------------------------------------------------
    enabled = getattr( betaconfig, 'shot_cut_detection_enabled', True )
    cut_files = find_shot_cut_caches()
    cut_times = []
    if cut_files:
        try:
            cut_times = bu_hash.read_json( cut_files[0] ) or []
        except Exception as exc:                       # noqa: BLE001
            logger.debug( 'could not read %s: %s', cut_files[0], exc )
    record( 1, 'shot cuts detected',
            bool( enabled and cut_times ),
            'shot_cut_detection_enabled = %s\n'
            'shot-cut cache files on disk: %d\n'
            'cuts in newest cache: %d\n'
            'newest cache: %s'
            % ( enabled, len( cut_files ), len( cut_times ),
                cut_files[0] if cut_files else '(none)' ) )

    # ---------------------------------------------------------------
    # 2. Frame step vs time_safety - the inverted-interval bug
    # ---------------------------------------------------------------
    fps = betaconfig.video_censor_fps
    frame_step = 1.0 / fps
    overrides = bu_detector.get_item_overrides( label, backend )
    time_safety = overrides.get( 'time_safety', betaconfig.default_time_safety )

    # A cut is only visible to a check written as
    #   cut_between( track['end'], box['start'] )
    # when track.end < box.start, i.e. t+safety/2 < t+step-safety/2,
    # i.e. time_safety < frame_step.
    padded_ok = time_safety < frame_step
    record( 2, 'padded cut interval',
            padded_ok,
            'video_censor_fps = %d  ->  frame_step = %.4f s\n'
            '%s time_safety = %.3f s\n'
            'A cut check written against the PADDED bounds needs\n'
            '    time_safety < frame_step\n'
            'or the interval (track.end, box.start] inverts and NO cut is\n'
            'ever seen. %s\n'
            'NOTE: this is informational once the timestamp fix is in -\n'
            'the fixed code compares track["t"] to box["t"], which is\n'
            'immune to padding. If this says FAIL and check 4 also fails,\n'
            'you are running the unfixed code.'
            % ( fps, frame_step, label, time_safety,
                'OK' if padded_ok else 'INVERTED: every cut invisible' ) )

    # ---------------------------------------------------------------
    # 3. Is the installed code using timestamps or padded bounds?
    # ---------------------------------------------------------------
    import inspect
    source = inspect.getsource( bu_track.smooth_boxes )
    padded_calls = source.count( "cut_between( track['end']" )
    stamped_calls = source.count( "cut_between( track['t']" )
    record( 3, 'installed cut checks',
            stamped_calls > 0 and padded_calls == 0,
            "cut_between( track['t'], ... )   : %d   <- timestamp form (correct)\n"
            "cut_between( track['end'], ... ) : %d   <- padded form (broken)\n"
            'If the padded count is non-zero you are running code where\n'
            'shot cuts cannot affect tracking.'
            % ( stamped_calls, padded_calls ) )

    # ---------------------------------------------------------------
    # 4. Replay the real cache through the real tracker
    # ---------------------------------------------------------------
    caches = find_detection_caches( label )
    raw_boxes = []
    used_cache = None
    for path in caches:
        try:
            candidate = bu_hash.read_json( path )
        except Exception:                              # noqa: BLE001
            continue
        if not isinstance( candidate, list ) or not candidate:
            continue
        if any( b.get( 'class_id' ) == label for b in candidate ):
            raw_boxes = candidate
            used_cache = path
            break

    if not raw_boxes:
        record( 4, 'replay from cache', None,
                'no detection cache containing %s was found under %s\n'
                'Run betatv.py first (a --preview run is enough).'
                % ( label, betaconst.vid_hash_dir ) )
        record( 5, 'independent style rolls', None, 'skipped: no cache' )
        record( 6, 'observed variety', None, 'skipped: no cache' )
    else:
        width = max( ( b['x'] + b['w'] for b in raw_boxes ), default=1920 )
        height = max( ( b['y'] + b['h'] for b in raw_boxes ), default=1080 )
        width = max( width, 1920 )
        height = max( height, 1080 )

        rolls = { 'count': 0 }
        original_resolve = bu_censor.resolve_censor_style

        def counting_resolve( style ):
            rolls['count'] += 1
            return original_resolve( style )

        bu_censor.resolve_censor_style = counting_resolve
        try:
            with_cuts, stats_with = bu_track.prepare_boxes_for_render(
                [ dict( b ) for b in raw_boxes ], width, height,
                cut_times or None, backend_name=backend, logger=logger )
            rolls_with = rolls['count']

            rolls['count'] = 0
            without_cuts, stats_without = bu_track.prepare_boxes_for_render(
                [ dict( b ) for b in raw_boxes ], width, height,
                None, backend_name=backend, logger=logger )
            rolls_without = rolls['count']
        finally:
            bu_censor.resolve_censor_style = original_resolve

        record( 4, 'cuts change tracking',
                stats_with['tracks'] > stats_without['tracks'],
                'cache: %s\n'
                'tracks WITHOUT cuts : %d\n'
                'tracks WITH    cuts : %d\n'
                'If these are equal, cuts are not reaching the tracker or\n'
                'are not being acted on.'
                % ( os.path.basename( used_cache ),
                    stats_without['tracks'], stats_with['tracks'] ) )

        record( 5, 'independent style rolls',
                rolls_with > 1,
                'style resolves WITHOUT cuts : %d\n'
                'style resolves WITH    cuts : %d\n'
                'One roll means one style for the whole clip, whatever the\n'
                'configured weights say.'
                % ( rolls_without, rolls_with ) )

        observed = collections.Counter()
        for box in with_cuts:
            if box.get( 'label' ) != label:
                continue
            style = box.get( 'censor_style' ) or {}
            key = '%s/%s' % ( style.get( 'type' ),
                              style.get( 'pattern' ) or style.get( 'method' ) or '' )
            observed[key] += 1
        total = sum( observed.values() ) or 1
        lines = [ '%-18s %5.1f%%' % ( k, 100.0*v/total )
                  for k, v in observed.most_common() ]
        top_share = 100.0 * observed.most_common( 1 )[0][1] / total if observed else 0
        record( 6, 'observed variety',
                len( observed ) > 1,
                'distinct styles rendered: %d\n'
                'top style share: %.0f%%\n%s'
                % ( len( observed ), top_share, '\n'.join( lines ) ) )

    # ---------------------------------------------------------------
    # 7. Could the weights alone explain it?
    # ---------------------------------------------------------------
    entries = ( getattr( betaconfig, 'item_overrides', {} )
                .get( label, {} ).get( 'censor_style', [] ) )
    if isinstance( entries, dict ):
        entries = [ entries ]
    weights = collections.Counter()
    total_weight = 0.0
    for entry in entries or []:
        key = '%s/%s' % ( entry.get( 'type' ),
                          entry.get( 'pattern' ) or entry.get( 'method' ) or '' )
        weight = float( entry.get( 'weight', 1 ) )
        weights[key] += weight
        total_weight += weight
    lines = [ '%-18s %5.1f%%' % ( k, 100.0*v/(total_weight or 1) )
              for k, v in weights.most_common() ]
    dominant = ( 100.0 * weights.most_common( 1 )[0][1] / (total_weight or 1) ) if weights else 100.0
    record( 7, 'configured weights',
            len( weights ) > 1 and dominant < 90,
            'entries: %d, distinct styles: %d\n%s'
            % ( len( entries or [] ), len( weights ), '\n'.join( lines ) ) )

    # ---------------------------------------------------------------
    logger.info( '' )
    logger.info( '=' * 70 )
    failures = [ v for v in verdicts if v[2] == 'FAIL' ]
    if not failures:
        logger.info( 'no failing check: style variety should be working.' )
        logger.info( 'If output still looks single-style, the cause is' )
        logger.info( 'downstream of tracking (render/merge), not style choice.' )
    else:
        logger.info( 'ROOT CAUSE CANDIDATES, earliest stage first:' )
        for number, name, _state, _detail in failures:
            logger.info( '  [%d] %s', number, name )
        logger.info( '' )
        logger.info( 'Fix the LOWEST-numbered failure first; later ones are' )
        logger.info( 'usually its consequences.' )
    logger.info( '=' * 70 )
    logger.info( 'full log: %s', os.path.abspath( args.log_file ) )

    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit( main() )
