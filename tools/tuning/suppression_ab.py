#!/usr/bin/env python3
"""
suppression_ab.py - A/B two class_suppression rule sets on real footage.

The current rules (betaconfig) against a proposed set: by default what
`betabench suppression` suggests from the same caches, i.e. each
configured pair retuned where the data shows a same-spot cluster and
dropped where it does not. --proposed FILE.json supplies any other set
({label: [rule, ...]}, the class_suppression shape).

PHASES (--phase rank | render | all)
  rank    Replays every cached video through the production pipeline
          under both rule sets (same profile, cuts and held censor style)
          and ranks the videos by how many seconds of censor box differ.
          Writes ranking.json: the two rule sets, every video's numbers,
          the top --videos, and for each the --clips windows where the
          two sets differ most.
  render  For the videos in ranking.json, renders each window:
            <video>_clipNN_diff.mp4       one render with outlines, no
                                          censoring: grey where both sets
                                          censor, orange where only the
                                          current rules do, magenta where
                                          only the proposed rules do
            <video>_clipNN_ab.mp4         the real censor under each set,
                                          side by side: current on the
                                          LEFT, proposed on the RIGHT
          --full also renders both sets over each whole video (hours).

Nothing touches betaconfig.py or any cache. Output goes to
../output/benchmarks/suppression_ab_<timestamp>/ unless --out-dir.
tools/tuning/suppression_ab.sh runs it with prompts and a log.
"""

import argparse
import bisect
import contextlib
import copy
import importlib.util
import json
import os
import subprocess
import sys
import time

CORE_DIR = os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )
sys.path.insert( 0, CORE_DIR )

import betaconfig
import betaconst
import betautils_cache_paths as bu_cache
import betautils_detector as bu_detector
import betautils_render as bu_render
import betautils_track as bu_track
import betautils_tuning as bu_tuning
import betautils_video as bu_video


COLOUR_BOTH = ( 160, 160, 160 )
COLOUR_CURRENT_ONLY = ( 255, 140, 0 )
COLOUR_PROPOSED_ONLY = ( 255, 0, 255 )
MATCH_IOU = 0.3


def _load_betabench():
    path = os.path.join( CORE_DIR, 'tools', 'bench', 'betabench.py' )
    spec = importlib.util.spec_from_file_location( '_betabench_for_ab', path )
    module = importlib.util.module_from_spec( spec )
    spec.loader.exec_module( module )
    return module


# --- rule sets ----------------------------------------------------------------

def _rule_list( rules ):
    return [ rules ] if isinstance( rules, dict ) else list( rules or [] )


def proposed_rules( configured, suggestions ):
    """
    The configured rules with each pair retuned or dropped per the bench.

    suggestions is betabench suppression's rows ({'pair': 'L<-S',
    'suggested_min_iou', 'suggested_margin'}). A configured pair with a
    suggestion takes its min_iou and margin; one the bench measured and
    found no same-spot cluster for is dropped; one with too few overlaps
    to measure (no row) stays as configured, since there is no evidence
    either way. Pairs that are not configured are never added: adding
    one is a separate decision with its own evidence.
    """
    by_pair = { row['pair']: row for row in suggestions }
    proposed = {}
    for label, rules in configured.items():
        for rule in _rule_list( rules ):
            row = by_pair.get( '%s<-%s'%( label, rule['suppressed_by'] ) )
            if row is None:
                proposed.setdefault( label, [] ).append( dict( rule ) )
                continue
            if row.get( 'suggested_min_iou' ) is None:
                continue
            proposed.setdefault( label, [] ).append( {
                'suppressed_by': rule['suppressed_by'],
                'margin': row['suggested_margin'],
                'min_iou': row['suggested_min_iou'],
            } )
    return proposed


def rule_changes( current, proposed ):
    """One line per pair: what the proposed set does to it."""
    def pairs( rules ):
        return { ( label, rule['suppressed_by'] ): rule
                 for label, entries in rules.items() for rule in _rule_list( entries ) }
    before, after = pairs( current ), pairs( proposed )
    lines = []
    for key in sorted( set( before ) | set( after ) ):
        name = '%s<-%s'%key
        old, new = before.get( key ), after.get( key )
        if new is None:
            lines.append( "%s: dropped"%( name, ) )
        elif old is None:
            lines.append( "%s: added (iou %s, margin %s)"%( name, new.get( 'min_iou' ), new.get( 'margin' ) ) )
        elif ( old.get( 'min_iou' ), old.get( 'margin' ) ) == ( new.get( 'min_iou' ), new.get( 'margin' ) ):
            lines.append( "%s: unchanged"%( name, ) )
        else:
            lines.append( "%s: retuned iou %s -> %s, margin %s -> %s"%(
                name, old.get( 'min_iou' ), new.get( 'min_iou' ), old.get( 'margin' ), new.get( 'margin' ) ) )
    return lines


# --- comparing two renders --------------------------------------------------------

def _iou( a, b ):
    return bu_track.intersection_over_union( a, b )


def diff_boxes( current, proposed, min_iou=MATCH_IOU ):
    """
    (both, current_only, proposed_only) between two rendered box lists.

    A proposed box matches a current box of the same label that is live
    at overlapping times and overlaps it by at least min_iou. Each
    current box matches at most once. both holds the proposed side of
    each match.
    """
    by_label = {}
    for index, box in enumerate( current ):
        by_label.setdefault( box.get( 'label' ), [] ).append( ( box['start'], index ) )
    for entries in by_label.values():
        entries.sort()
    longest = max( ( box['end'] - box['start'] for box in current ), default=0.0 )
    used = set()
    both, proposed_only = [], []
    for box in proposed:
        entries = by_label.get( box.get( 'label' ), [] )
        starts = [ start for start, _index in entries ] if entries else []
        position = bisect.bisect_right( starts, box['end'] ) - 1
        matched = None
        while position >= 0 and starts[position] >= box['start'] - longest:
            index = entries[position][1]
            other = current[index]
            if ( index not in used and other['end'] >= box['start']
                 and _iou( box, other ) >= min_iou ):
                matched = index
                break
            position -= 1
        if matched is None:
            proposed_only.append( box )
        else:
            used.add( matched )
            both.append( box )
    current_only = [ box for index, box in enumerate( current ) if index not in used ]
    return both, current_only, proposed_only


def box_seconds( boxes ):
    return sum( max( 0.0, box['end'] - box['start'] ) for box in boxes )


def pick_windows( events, duration, length, count ):
    """
    Up to `count` non-overlapping [start, end) windows of `length` seconds
    holding the most differing box-seconds. events are boxes (start/end).
    """
    if not events or duration <= 0:
        return []
    length = min( length, duration )
    points = sorted( ( ( box['start'] + box['end'] ) / 2.0,
                       max( 0.0, box['end'] - box['start'] ) ) for box in events )
    times = [ point[0] for point in points ]
    weights = [ point[1] for point in points ]
    prefix = [ 0.0 ]
    for weight in weights:
        prefix.append( prefix[-1] + weight )
    chosen = []
    candidates = []
    for start_index, centre in enumerate( times ):
        start = max( 0.0, min( centre - length / 2.0, duration - length ) )
        lo = bisect.bisect_left( times, start )
        hi = bisect.bisect_left( times, start + length )
        candidates.append( ( prefix[hi] - prefix[lo], start ) )
    candidates.sort( key=lambda item: ( -item[0], item[1] ) )
    for weight, start in candidates:
        if len( chosen ) >= count:
            break
        if weight <= 0:
            break
        if any( start < end and other_start < start + length for other_start, end in chosen ):
            continue
        chosen.append( ( start, start + length ) )
    return sorted( chosen )


# --- replaying one video under a rule set ----------------------------------------------

def replay_video( video_path, cache_paths, backend_name, rules, tracking ):
    """Rendered boxes for one video under `rules`, production pipeline, style held fixed."""
    raw = bu_tuning._read_raw_boxes( cache_paths )
    vid_w, vid_h = bu_tuning.frame_size_for_video( video_path )
    if not raw or not vid_w or not vid_h:
        return []
    with bu_tuning.class_suppression_in_effect( backend_name, rules ), \
            bu_tuning._style_choice_held_fixed():
        rendered, _stats = bu_track.prepare_boxes_for_render(
            raw, vid_w, vid_h, shot_cut_times=tracking['shot_cut_times'],
            backend_name=backend_name, logger=bu_tuning._QUIET_LOGGER,
            profile_name=tracking['profile_name'], sample_fps=tracking['sample_fps'] )
    return rendered


def _seconds_by_label( boxes ):
    totals = {}
    for box in boxes:
        label = box.get( 'label' )
        totals[label] = totals.get( label, 0.0 ) + max( 0.0, box['end'] - box['start'] )
    return { label: round( seconds, 2 ) for label, seconds in sorted( totals.items() ) }


def compare_video( file_hash, video_path, cache_paths, backend_name, current, proposed, labels,
                   clip_seconds, clips, replay=None, duration_of=None ):
    """
    One video's difference between the two rule sets.

    Returns (row, both, current_only, proposed_only, current_boxes,
    proposed_boxes). row carries the numbers the ranking sorts on and
    the clip windows where the sets differ most.
    """
    replay = replay or replay_video
    duration_of = duration_of or bu_video.probe_duration_seconds
    tracking = bu_tuning.replay_tracking_context( file_hash, video_path, cache_paths, backend_name )
    current_boxes = replay( video_path, cache_paths, backend_name, current, tracking )
    proposed_boxes = replay( video_path, cache_paths, backend_name, proposed, tracking )
    both, current_only, proposed_only = diff_boxes( current_boxes, proposed_boxes )
    confident = bu_tuning.confident_detections(
        bu_tuning._read_raw_boxes( cache_paths ), labels, bu_tuning.CONFIDENT_SCORE )
    missed_now = sum( bu_tuning.uncovered_detections( confident, current_boxes ).values() )
    missed_after = sum( bu_tuning.uncovered_detections( confident, proposed_boxes ).values() )
    censored = box_seconds( current_boxes )
    row = {
        'file_hash': file_hash, 'video': video_path, 'name': os.path.basename( video_path ),
        'profile': tracking['profile_name'], 'cache_paths': list( cache_paths ),
        'current_only_boxes': len( current_only ), 'proposed_only_boxes': len( proposed_only ),
        'current_only_seconds': round( box_seconds( current_only ), 2 ),
        'proposed_only_seconds': round( box_seconds( proposed_only ), 2 ),
        'current_only_by_label': _seconds_by_label( current_only ),
        'proposed_only_by_label': _seconds_by_label( proposed_only ),
        'censored_seconds_now': round( censored, 2 ),
        'confident_detections': sum( len( found ) for found in confident.values() ),
        'confident_restored': missed_now - missed_after,
    }
    row['differing_seconds'] = round( row['current_only_seconds'] + row['proposed_only_seconds'], 2 )
    row['differing_share'] = round( row['differing_seconds'] / censored, 4 ) if censored else 0.0
    row['windows'] = pick_windows( current_only + proposed_only, duration_of( video_path ) or 0.0,
                                   clip_seconds, clips )
    return row, both, current_only, proposed_only, current_boxes, proposed_boxes


def rank_videos( videos, hash_to_path, backend_name, current, proposed, labels, logger,
                 clip_seconds=15.0, clips=6, replay=None, duration_of=None ):
    """Every video's difference between the two rule sets, most-changed first."""
    rows = []
    for number, ( file_hash, cache_paths ) in enumerate( sorted( videos.items() ), 1 ):
        video_path = hash_to_path.get( file_hash )
        if not video_path:
            logger.debug( "rank: no source video for %s, skipped"%( file_hash, ) )
            continue
        row = compare_video( file_hash, video_path, cache_paths, backend_name, current, proposed,
                             labels, clip_seconds, clips, replay=replay,
                             duration_of=duration_of )[0]
        # Each video is replayed exactly once here, so its parsed cache is
        # dead weight for the rest of the loop (gigabytes across a library).
        bu_tuning.forget_raw_boxes( cache_paths )
        rows.append( row )
        logger.info( "rank %d/%d: %s differs by %.1f box-second(s)"%(
            number, len( videos ), row['name'], row['differing_seconds'] ) )
    rows.sort( key=lambda row: ( -row['differing_seconds'], row['name'] ) )
    return rows


# --- rendering -----------------------------------------------------------------------

def overlay_boxes( both, current_only, proposed_only ):
    """The three groups as outline-only debug boxes, one colour per group."""
    styled = []
    for colour, group in ( ( COLOUR_BOTH, both ), ( COLOUR_CURRENT_ONLY, current_only ),
                           ( COLOUR_PROPOSED_ONLY, proposed_only ) ):
        for box in group:
            styled.append( dict( box, censor_style={ 'type': 'debug', 'color': colour },
                                 censor_shape='box' ) )
    styled.sort( key=lambda box: box['start'] )
    return styled


@contextlib.contextmanager
def motion_interpolation_off():
    """
    The overlay mixes boxes from two renders whose track ids overlap, and
    motion interpolation links boxes by track id, so it would slide one
    set's box toward the other's. Outlines held per sample are enough to
    review.
    """
    saved = getattr( betaconfig, 'render_motion_interpolation', True )
    betaconfig.render_motion_interpolation = False
    try:
        yield
    finally:
        betaconfig.render_motion_interpolation = saved


def render_window( video_path, file_hash, boxes, out_path, start, length, logger ):
    """
    Render [start, start + length) of a video with `boxes` to out_path.
    start None renders the whole video.
    """
    source = bu_video.open_video_source( video_path, file_hash, logger )
    try:
        whole = start is None
        bu_render.render_video( source, boxes, out_path + '.video.mkv', out_path,
                                getattr( betaconfig, 'encode_preset', 'fast' ),
                                not whole, False, 0.0 if whole else start,
                                0.0 if whole else length, logger )
    finally:
        source.release()
    return out_path


def side_by_side( left, right, out_path, logger ):
    """left | right in one file, left's audio. False when ffmpeg refuses."""
    command = [ 'ffmpeg', '-y', '-loglevel', 'error', '-i', left, '-i', right,
                '-filter_complex', '[0:v][1:v]hstack=inputs=2[v]', '-map', '[v]', '-map', '0:a?',
                '-c:v', getattr( betaconfig, 'encode_video_codec', 'libx264' ),
                '-crf', str( getattr( betaconfig, 'encode_crf', 17 ) ),
                '-preset', getattr( betaconfig, 'encode_preset', 'fast' ), '-c:a', 'copy', out_path ]
    result = subprocess.run( command, capture_output=True, text=True )
    if result.returncode != 0:
        logger.warning( "side-by-side failed (%s); keeping the two clips separately"%(
            result.stderr.strip().splitlines()[-1] if result.stderr.strip() else result.returncode, ) )
        return False
    return True


def render_video_comparison( row, ranking, logger, out_dir, overlay=True, censored=True, full=False,
                             render=None, stack=None ):
    """Every requested render for one ranked video."""
    render = render or render_window
    stack = stack or side_by_side
    backend_name = ranking['backend']
    _row, both, current_only, proposed_only, current_boxes, proposed_boxes = compare_video(
        row['file_hash'], row['video'], row['cache_paths'], backend_name,
        ranking['current_rules'], ranking['proposed_rules'], set( ranking['labels'] ), 0.0, 0,
        duration_of=lambda _path: 0.0 )
    stem = os.path.splitext( row['name'] )[0][ :60 ].replace( ' ', '_' )
    produced = []
    for number, ( start, end ) in enumerate( row.get( 'windows', [] ), 1 ):
        length = end - start
        tag = '%s_clip%02d_%ds'%( stem, number, int( start ) )
        if overlay:
            path = os.path.join( out_dir, tag + '_diff.mp4' )
            logger.info( "rendering %s (outlines: grey both, orange current only, magenta "
                         "proposed only)"%( os.path.basename( path ), ) )
            with motion_interpolation_off():
                render( row['video'], row['file_hash'],
                        overlay_boxes( both, current_only, proposed_only ),
                        path, start, length, logger )
            produced.append( path )
        if censored:
            left = os.path.join( out_dir, tag + '_current.mp4' )
            right = os.path.join( out_dir, tag + '_proposed.mp4' )
            render( row['video'], row['file_hash'], current_boxes, left, start, length, logger )
            render( row['video'], row['file_hash'], proposed_boxes, right, start, length, logger )
            combined = os.path.join( out_dir, tag + '_ab.mp4' )
            if stack( left, right, combined, logger ):
                os.remove( left )
                os.remove( right )
                produced.append( combined )
                logger.info( "rendered %s (current LEFT, proposed RIGHT)"%(
                    os.path.basename( combined ), ) )
            else:
                produced += [ left, right ]
    if full:
        for name, boxes in ( ( 'current', current_boxes ), ( 'proposed', proposed_boxes ) ):
            path = os.path.join( out_dir, '%s_full_%s.mp4'%( stem, name ) )
            logger.info( "rendering the whole video under the %s rules: %s"%( name, path ) )
            render( row['video'], row['file_hash'], boxes, path, None, None, logger )
            produced.append( path )
    return produced


# --- phases ---------------------------------------------------------------------------

def _configuration( args, logger ):
    min_prob = getattr( betaconfig, 'global_min_prob', betaconst.global_min_prob )
    backend_name = args.backend or bu_detector.selected_backend_name()
    configurations = list( bu_cache.configurations_to_analyse(
        None, backend_names=[ backend_name ], min_prob=min_prob, include_preview=False,
        variant_names=[ args.variant ] if args.variant else None ) )
    if not configurations:
        raise SystemExit( "no full-run detection caches for %s"%( backend_name, ) )
    selected_variant = bu_detector.selected_variant_name( backend_name )
    for config in configurations:
        if config.variant and config.variant == selected_variant:
            return config, min_prob
    if len( configurations ) > 1:
        logger.warning( "several configurations on disk; using %s (pass --variant to choose)"%(
            configurations[0].label, ) )
    return configurations[0], min_prob


def _log_ranking( logger, rows, shown ):
    logger.info( "" )
    logger.info( "  %-50s %-12s %10s %7s %10s %10s %9s"%(
        'video', 'profile', 'differ (s)', 'share', 'only now', 'only new', 'restored' ) )
    for row in rows[ :shown ]:
        logger.info( "  %-50s %-12s %10.1f %6.1f%% %10.1f %10.1f %9d"%(
            row['name'][ :50 ], row['profile'] or '', row['differing_seconds'],
            100.0 * row['differing_share'], row['current_only_seconds'],
            row['proposed_only_seconds'], row['confident_restored'] ) )
    logger.info( "  differ: box-seconds censored under one set and not the other (the ranking). "
                 "share: differ over the seconds censored now. only now / only new: censored "
                 "under the current / proposed rules only. restored: confident detections "
                 "(score >= %.2f) the proposed rules censor that the current ones leave bare"%(
                     bu_tuning.CONFIDENT_SCORE, ) )


def phase_rank( args, logger, betabench ):
    config, min_prob = _configuration( args, logger )
    backend_name = config.backend_name
    # Every sample rate, so quick-cut videos (detected at their profile's
    # rate) are ranked too. The proposal below still comes from the
    # global rate's caches, the same data `betabench suppression` reads.
    videos, rates = bu_cache.videos_across_sample_rates( config, min_prob )
    hash_to_path = bu_cache.build_hash_to_video_path( videos.keys() )
    current = copy.deepcopy( bu_detector.get_class_suppression( backend_name ) or {} )
    if args.proposed:
        with open( args.proposed ) as handle:
            proposed = json.load( handle )
        source = args.proposed
    else:
        logger.info( "deriving the proposed rules from %s's caches (betabench suppression)"%(
            config.label, ) )
        raw_boxes, _count = betabench.load_cached_detections(
            backend_name, logger, picture_sizes=config.picture_sizes, configuration=config )
        bench_args = argparse.Namespace( min_samples=args.min_samples )
        rows = betabench._bench_suppression_one( bench_args, logger, config, raw_boxes )['suppression']
        del raw_boxes
        proposed = proposed_rules( current, rows )
        source = 'betabench suppression suggestions'
    logger.info( "current rules:  %s"%( json.dumps( current, sort_keys=True ), ) )
    logger.info( "proposed rules (%s): %s"%( source, json.dumps( proposed, sort_keys=True ) ) )
    changes = rule_changes( current, proposed )
    for line in changes:
        logger.info( "  rule %s"%( line, ) )
    if all( line.endswith( ': unchanged' ) for line in changes ):
        raise SystemExit( "the proposed rules are the current rules; nothing to compare" )

    labels = sorted( betaconfig.items_to_censor )
    logger.info( "replaying %d video(s) of %s (sample rates %s) under both rule sets"%(
        len( videos ), config.label, ", ".join( '%g'%( rate, ) for rate in rates ) or 'none' ) )
    rows = rank_videos( videos, hash_to_path, backend_name, current, proposed, set( labels ), logger,
                        clip_seconds=args.clip_seconds, clips=args.clips )
    _log_ranking( logger, rows, max( 10, args.videos ) )
    chosen = [ row for row in rows[ :args.videos ] if row['differing_seconds'] > 0 ]
    for row in chosen:
        logger.info( "chosen: %s, %d clip window(s): %s"%(
            row['name'], len( row['windows'] ),
            ", ".join( "%ds-%ds"%( start, end ) for start, end in row['windows'] ) ) )
    if not chosen:
        logger.warning( "no video renders differently under the two rule sets" )
    ranking = { 'backend': backend_name, 'configuration': config.label, 'labels': labels,
                'sample_rates': rates,
                'current_rules': current, 'proposed_rules': proposed, 'proposed_source': source,
                'changes': changes,
                'videos': rows, 'chosen': [ row['file_hash'] for row in chosen ] }
    path = os.path.join( args.out_dir, 'ranking.json' )
    with open( path, 'w' ) as handle:
        json.dump( ranking, handle, indent=2, default=str )
    logger.info( "ranking written to %s"%( path, ) )
    return ranking


def phase_render( args, logger, ranking ):
    by_hash = { row['file_hash']: row for row in ranking['videos'] }
    produced = []
    for file_hash in ranking['chosen']:
        row = by_hash[file_hash]
        logger.info( "" )
        logger.info( "=== %s ==="%( row['name'], ) )
        produced += render_video_comparison( row, ranking, logger, args.out_dir,
                                             overlay=not args.no_overlay,
                                             censored=not args.no_censored, full=args.full )
    logger.info( "" )
    logger.info( "%d file(s) in %s"%( len( produced ), args.out_dir ) )
    for path in produced:
        logger.info( "  %s"%( os.path.basename( path ), ) )
    return produced


def build_parser():
    parser = argparse.ArgumentParser( description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter )
    parser.add_argument( '--phase', choices=( 'rank', 'render', 'all' ), default='all' )
    parser.add_argument( '--videos', type=int, default=2,
        help="how many of the most-changed videos to render (default: 2)" )
    parser.add_argument( '--clips', type=int, default=6,
        help="clip windows per video, where the sets differ most (default: 6)" )
    parser.add_argument( '--clip-seconds', type=float, default=15.0,
        help="length of each clip (default: 15)" )
    parser.add_argument( '--no-overlay', action='store_true', help="skip the outline renders" )
    parser.add_argument( '--no-censored', action='store_true', help="skip the side-by-side renders" )
    parser.add_argument( '--full', action='store_true',
        help="also render each chosen video whole under both rule sets (hours)" )
    parser.add_argument( '--proposed', default=None, metavar='FILE.json',
        help="a proposed rule set instead of the bench's suggestions" )
    parser.add_argument( '--ranking', default=None, metavar='ranking.json',
        help="render phase: the rank phase's output (default: --out-dir/ranking.json)" )
    parser.add_argument( '--min-samples', type=int, default=30,
        help="overlaps a pair needs before the bench suggests a rule for it (default: 30)" )
    parser.add_argument( '--backend', default=None )
    parser.add_argument( '--variant', default=None )
    parser.add_argument( '--out-dir', default=None )
    parser.add_argument( '--log-level', default='debug',
                         choices=( 'trace', 'debug', 'info', 'warn', 'error' ) )
    parser.add_argument( '--console-level', default='info',
                         choices=( 'trace', 'debug', 'info', 'warn', 'error' ) )
    return parser


def main( argv=None ):
    args = build_parser().parse_args( argv )
    args.out_dir = args.out_dir or os.path.join(
        os.path.dirname( CORE_DIR ), 'output', 'benchmarks',
        'suppression_ab_%s'%( time.strftime( '%Y%m%d_%H%M%S' ), ) )
    os.makedirs( args.out_dir, exist_ok=True )
    betabench = _load_betabench()
    logger = betabench.build_logger( args.out_dir, args.log_level, args.console_level )
    logger.info( "suppression A/B: output in %s"%( args.out_dir, ) )
    ranking = None
    if args.phase in ( 'rank', 'all' ):
        ranking = phase_rank( args, logger, betabench )
    if args.phase in ( 'render', 'all' ):
        if ranking is None:
            with open( args.ranking or os.path.join( args.out_dir, 'ranking.json' ) ) as handle:
                ranking = json.load( handle )
        phase_render( args, logger, ranking )
    return 0


if __name__ == '__main__':
    raise SystemExit( main() )
