#!/usr/bin/env python3
"""
ab_compare.py - the config as it is against one proposed change, on real footage, judged by eye.

What changes is an experiment (tools/tuning/ab_experiments.py):
suppression rules, min_track_hits, match distance, the shot-cut burst
merge, or any per-label setting. The rest is the same for all of them:

PHASES (--phase rank | explain | render | decide | all)
  rank    Replays every cached video through the production pipeline
          twice, as configured and with the change (same profile, cuts
          and held censor style), and ranks the videos by how many
          seconds of censor box differ. --focus LABEL ranks by that
          label's differences only. --exclude-from RANKING skips videos
          an earlier round already showed you. Writes ranking.json.
  explain clips.md: every difference in each clip window, its time in
          the clip, which side censors it, and the change that caused
          it. feedback.csv: one row per difference, with a verdict
          column (right / left / same) for you to fill in.
  render  For each chosen video, each window:
            <video>_clipNN_<s>s_diff.mp4  outlines only: grey where both
                                          censor, orange where only the
                                          config does, magenta where only
                                          the change does
            <video>_clipNN_<s>s_ab.mp4    the real censor, config LEFT,
                                          change RIGHT
          --full also renders both sides over each whole video (hours).
  decide  Reads the verdicts and writes chosen.json and the config lines
          to paste: each change takes the side its verdicts favour; a
          tie or no verdict keeps the config.

Nothing touches betaconfig.py or any cache. Output goes to
../output/benchmarks/ab_<experiment>_<timestamp>/ unless --out-dir.
tools/tuning/ab_compare.sh runs the whole round with prompts and a log.
"""

import argparse
import bisect
import contextlib
import csv
import functools
import importlib.util
import json
import math
import os
import subprocess
import sys
import time

CORE_DIR = os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )
sys.path.insert( 0, CORE_DIR )
sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )

import betaconfig
import betaconst
import betautils_cache_paths as bu_cache
import betautils_detector as bu_detector
import betautils_render as bu_render
import betautils_track as bu_track
import betautils_tuning as bu_tuning
import betautils_video as bu_video

import ab_experiments
from ab_experiments import TRACKING_CAUSE


COLOUR_BOTH = ( 160, 160, 160 )
COLOUR_CURRENT_ONLY = ( 255, 140, 0 )
COLOUR_PROPOSED_ONLY = ( 255, 0, 255 )
MATCH_IOU = 0.3
_BUCKET_SECONDS = ab_experiments.BUCKET_SECONDS


def _load_betabench():
    path = os.path.join( CORE_DIR, 'tools', 'bench', 'betabench.py' )
    spec = importlib.util.spec_from_file_location( '_betabench_for_ab', path )
    module = importlib.util.module_from_spec( spec )
    spec.loader.exec_module( module )
    return module


def normalised( ranking ):
    """
    ranking.json with 'experiment', 'current' and 'proposed' filled in.
    Runs written before the experiments existed are suppression runs
    and named the sides current_rules / proposed_rules.
    """
    ranking = dict( ranking )
    ranking.setdefault( 'experiment', 'suppression' )
    if 'current' not in ranking:
        ranking['current'] = ranking.get( 'current_rules', {} )
    if 'proposed' not in ranking:
        ranking['proposed'] = ranking.get( 'proposed_rules', {} )
    return ranking


# --- comparing two renders --------------------------------------------------------

def _iou( a, b ):
    return bu_track.intersection_over_union( a, b )


def _time_index( boxes ):
    """{(label, bucket): [box, ...]}, each box under every bucket its live time touches."""
    index = {}
    for box in boxes:
        label = box.get( 'label' )
        for bucket in range( int( math.floor( box['start'] / _BUCKET_SECONDS ) ),
                             int( math.floor( box['end'] / _BUCKET_SECONDS ) ) + 1 ):
            index.setdefault( ( label, bucket ), [] ).append( box )
    return index


def _covered( boxes, index, min_iou ):
    """[True where some indexed box of the same label is live at an overlapping time and overlaps by min_iou]."""
    covered = []
    for box in boxes:
        label = box.get( 'label' )
        hit = False
        for bucket in range( int( math.floor( box['start'] / _BUCKET_SECONDS ) ),
                             int( math.floor( box['end'] / _BUCKET_SECONDS ) ) + 1 ):
            for other in index.get( ( label, bucket ), () ):
                if ( other['start'] <= box['end'] and other['end'] >= box['start']
                     and _iou( box, other ) >= min_iou ):
                    hit = True
                    break
            if hit:
                break
        covered.append( hit )
    return covered


def diff_boxes( current, proposed, min_iou=MATCH_IOU ):
    """
    (both, current_only, proposed_only) between two rendered box lists.

    A box is censored under the other side too when that side has a box
    of the same label live at an overlapping time that overlaps it by at
    least min_iou. Coverage, not one-to-one matching: each sample's box
    overlaps its neighbours' in time, and pairing them off greedily left
    one spurious box per track on each side (9% of identical renders).
    both holds the proposed boxes the current side also censors.

    Time-bucketed, so the cost is linear in boxes rather than quadratic;
    the quadratic version spent ten minutes on one split-screen file.
    """
    current_index = _time_index( current )
    proposed_index = _time_index( proposed )
    proposed_covered = _covered( proposed, current_index, min_iou )
    current_covered = _covered( current, proposed_index, min_iou )
    both = [ box for box, hit in zip( proposed, proposed_covered ) if hit ]
    proposed_only = [ box for box, hit in zip( proposed, proposed_covered ) if not hit ]
    current_only = [ box for box, hit in zip( current, current_covered ) if not hit ]
    return both, current_only, proposed_only


def box_seconds( boxes ):
    return sum( max( 0.0, box['end'] - box['start'] ) for box in boxes )


def focused( boxes, focus ):
    """The boxes of the focus labels, or all of them when there is no focus."""
    if not focus:
        return list( boxes )
    return [ box for box in boxes if box.get( 'label' ) in focus ]


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
    for centre in times:
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


# --- replaying one video under one side ------------------------------------------------

def replay_video( video_path, cache_paths, backend_name, side, tracking, observer=None,
                  experiment=None ):
    """
    Rendered boxes for one video with `side` in effect, production
    pipeline, style held fixed. observer (a bu_track.TrackingObserver)
    sees the replay, e.g. every detection suppression drops.
    """
    experiment = experiment or ab_experiments.experiment_named( 'suppression' )
    raw = bu_tuning._read_raw_boxes( cache_paths )
    vid_w, vid_h = bu_tuning.frame_size_for_video( video_path )
    if not raw or not vid_w or not vid_h:
        return []
    with experiment.applied( side, backend_name, tracking ), \
            bu_tuning._style_choice_held_fixed(), \
            bu_track.tracking_observer( observer ):
        rendered, _stats = bu_track.prepare_boxes_for_render(
            raw, vid_w, vid_h, shot_cut_times=tracking['shot_cut_times'],
            backend_name=backend_name, logger=bu_tuning._QUIET_LOGGER,
            profile_name=tracking['profile_name'], sample_fps=tracking['sample_fps'] )
    return rendered


def _replayer( replay, experiment ):
    return replay or functools.partial( replay_video, experiment=experiment )


def _seconds_by_label( boxes ):
    totals = {}
    for box in boxes:
        label = box.get( 'label' )
        totals[label] = totals.get( label, 0.0 ) + max( 0.0, box['end'] - box['start'] )
    return { label: round( seconds, 2 ) for label, seconds in sorted( totals.items() ) }


def compare_video( file_hash, video_path, cache_paths, backend_name, current, proposed, labels,
                   clip_seconds, clips, replay=None, duration_of=None, experiment=None, focus=None ):
    """
    One video's difference between the two sides.

    Returns (row, both, current_only, proposed_only, current_boxes,
    proposed_boxes). row carries the numbers the ranking sorts on and
    the clip windows where the sides differ most; with a focus, both are
    over the focus labels' differences only.
    """
    experiment = experiment or ab_experiments.experiment_named( 'suppression' )
    replay = _replayer( replay, experiment )
    duration_of = duration_of or bu_video.probe_duration_seconds
    tracking = bu_tuning.replay_tracking_context( file_hash, video_path, cache_paths, backend_name )
    current_boxes = replay( video_path, cache_paths, backend_name, current, tracking )
    proposed_boxes = replay( video_path, cache_paths, backend_name, proposed, tracking )
    both, current_only, proposed_only = diff_boxes( current_boxes, proposed_boxes )
    raw_boxes = bu_tuning.raw_boxes_for_reading( cache_paths )
    confident = bu_tuning.confident_detections( raw_boxes, labels, bu_tuning.CONFIDENT_SCORE )
    duration = duration_of( video_path ) or 0.0
    missed_now = sum( bu_tuning.uncovered_detections( confident, current_boxes ).values() )
    missed_after = sum( bu_tuning.uncovered_detections( confident, proposed_boxes ).values() )
    censored = box_seconds( focused( current_boxes, focus ) )
    ranked_current, ranked_proposed = focused( current_only, focus ), focused( proposed_only, focus )
    row = {
        'file_hash': file_hash, 'video': video_path, 'name': os.path.basename( video_path ),
        'profile': tracking['profile_name'], 'cache_paths': list( cache_paths ),
        'current_only_boxes': len( ranked_current ), 'proposed_only_boxes': len( ranked_proposed ),
        'current_only_seconds': round( box_seconds( ranked_current ), 2 ),
        'proposed_only_seconds': round( box_seconds( ranked_proposed ), 2 ),
        'current_only_by_label': _seconds_by_label( current_only ),
        'proposed_only_by_label': _seconds_by_label( proposed_only ),
        'censored_seconds_now': round( censored, 2 ),
        'confident_detections': sum( len( found ) for found in confident.values() ),
        'confident_restored': missed_now - missed_after,
    }
    row['differing_seconds'] = round( row['current_only_seconds'] + row['proposed_only_seconds'], 2 )
    row['differing_share'] = round( row['differing_seconds'] / censored, 4 ) if censored else 0.0
    row['windows'] = pick_windows( ranked_current + ranked_proposed, duration, clip_seconds, clips )
    # What choose_videos compares to tell a copy of this video apart.
    row['signature'] = bu_tuning.content_signature( duration, bu_tuning.label_content(
        raw_boxes, labels, bu_tuning.cache_sample_rate( cache_paths ) ) )
    return row, both, current_only, proposed_only, current_boxes, proposed_boxes


def compare_with_causes( file_hash, video_path, cache_paths, backend_name, current, proposed,
                         replay=None, experiment=None ):
    """
    Both renders, their difference, and the cause of every differing box.

    proposed_only boxes are explained by what the CURRENT side does (for
    suppression, a current rule drops their detection); current_only
    boxes by what the proposed side does.
    """
    experiment = experiment or ab_experiments.experiment_named( 'suppression' )
    replay = _replayer( replay, experiment )
    changes = experiment.changes_by_cause( current, proposed, backend_name )
    tracking = bu_tuning.replay_tracking_context( file_hash, video_path, cache_paths, backend_name )
    current_log, proposed_log = experiment.make_observer(), experiment.make_observer()
    current_boxes = replay( video_path, cache_paths, backend_name, current, tracking,
                            observer=current_log )
    proposed_boxes = replay( video_path, cache_paths, backend_name, proposed, tracking,
                             observer=proposed_log )
    both, current_only, proposed_only = diff_boxes( current_boxes, proposed_boxes )
    return {
        'current_boxes': current_boxes, 'proposed_boxes': proposed_boxes, 'both': both,
        'current_only': current_only, 'proposed_only': proposed_only,
        'current_only_causes': experiment.attribute( current_only, proposed_log, changes ),
        'proposed_only_causes': experiment.attribute( proposed_only, current_log, changes ),
    }


def window_events( boxes, causes, start, end, side, join_gap=0.5 ):
    """
    The differing boxes live in [start, end), as events: one per label and
    cause, spans merged when less than join_gap apart. Times are clip
    relative.
    """
    grouped = {}
    for box, cause in zip( boxes, causes ):
        if box['end'] <= start or box['start'] >= end:
            continue
        grouped.setdefault( ( box.get( 'label' ), cause ), [] ).append(
            ( max( box['start'], start ) - start, min( box['end'], end ) - start ) )
    events = []
    for ( label, cause ), spans in sorted( grouped.items() ):
        spans.sort()
        merged = [ list( spans[0] ) ]
        for span_start, span_end in spans[1:]:
            if span_start - merged[-1][1] < join_gap:
                merged[-1][1] = max( merged[-1][1], span_end )
            else:
                merged.append( [ span_start, span_end ] )
        events.append( { 'side': side, 'label': label, 'cause': cause, 'boxes': len( spans ),
                         'spans': [ ( round( a, 2 ), round( b, 2 ) ) for a, b in merged ] } )
    return events


SIDE_RIGHT = 'RIGHT only'
SIDE_LEFT = 'LEFT only'
SIDE_BOTH = 'both differ'
WHAT_DIFFERS = {
    SIDE_RIGHT: 'RIGHT censors, LEFT bare',
    SIDE_LEFT: 'LEFT censors, RIGHT bare',
    SIDE_BOTH: 'both censor, boxes differ (place or size)',
}


def _intersects( a, b ):
    return ( a['x'] < b['x'] + b['w'] and b['x'] < a['x'] + a['w']
             and a['y'] < b['y'] + b['h'] and b['y'] < a['y'] + a['h'] )


def _censored_nearby( box, index ):
    """Whether the other side has a box of this label live at the same time that touches this one."""
    label = box.get( 'label' )
    for bucket in range( int( math.floor( box['start'] / _BUCKET_SECONDS ) ),
                         int( math.floor( box['end'] / _BUCKET_SECONDS ) ) + 1 ):
        for other in index.get( ( label, bucket ), () ):
            if other['start'] <= box['end'] and other['end'] >= box['start'] and _intersects( box, other ):
                return True
    return False


def window_moments( result, start, end, join_gap=0.5 ):
    """
    The differences live in [start, end), as moments: one per stretch of
    time, label, cause and kind, which is one feedback row each.

    A box counted on one side only (no box of its label overlaps it by
    MATCH_IOU on the other) is often still censored there, just by a box
    in another place or of another size. When the other side has a box
    of the same label live at that time and touching it, the moment is
    SIDE_BOTH, whichever side it came from: both censor, the boxes
    differ, and the question is which covers better. Otherwise it is
    SIDE_RIGHT or SIDE_LEFT: one side censors, the other leaves it bare.
    Spans less than join_gap apart join; times are clip relative.

    result is compare_with_causes' dict.
    """
    other_side = { SIDE_RIGHT: _time_index( result['current_boxes'] ),
                   SIDE_LEFT: _time_index( result['proposed_boxes'] ) }
    grouped = {}
    for side, boxes, causes in ( ( SIDE_RIGHT, result['proposed_only'], result['proposed_only_causes'] ),
                                 ( SIDE_LEFT, result['current_only'], result['current_only_causes'] ) ):
        for box, cause in zip( boxes, causes ):
            if box['end'] <= start or box['start'] >= end:
                continue
            kind = SIDE_BOTH if _censored_nearby( box, other_side[side] ) else side
            grouped.setdefault( ( box.get( 'label' ), cause, kind ), [] ).append(
                ( max( box['start'], start ) - start, min( box['end'], end ) - start ) )
    moments = []
    for ( label, cause, kind ), spans in grouped.items():
        spans.sort()
        merged = [ list( spans[0] ) ]
        for span_start, span_end in spans[1:]:
            if span_start - merged[-1][1] < join_gap:
                merged[-1][1] = max( merged[-1][1], span_end )
            else:
                merged.append( [ span_start, span_end ] )
        for span_start, span_end in merged:
            moments.append( { 'side': kind, 'label': label, 'cause': cause,
                              'span': ( round( span_start, 2 ), round( span_end, 2 ) ) } )
    moments.sort( key=lambda moment: ( moment['span'][0], moment['label'] or '', moment['side'] ) )
    return moments


def cause_seconds( boxes, causes ):
    """{cause: box-seconds} over a whole video, largest first."""
    totals = {}
    for box, cause in zip( boxes, causes ):
        totals[cause] = totals.get( cause, 0.0 ) + max( 0.0, box['end'] - box['start'] )
    return dict( sorted( ( ( cause, round( seconds, 1 ) ) for cause, seconds in totals.items() ),
                         key=lambda item: -item[1] ) )


def rank_videos( videos, hash_to_path, backend_name, current, proposed, labels, logger,
                 clip_seconds=15.0, clips=6, replay=None, duration_of=None, experiment=None,
                 focus=None ):
    """Every video's difference between the two sides, most-changed first."""
    rows = []
    for number, ( file_hash, cache_paths ) in enumerate( sorted( videos.items() ), 1 ):
        video_path = hash_to_path.get( file_hash )
        if not video_path:
            logger.debug( "rank: no source video for %s, skipped"%( file_hash, ) )
            continue
        row = compare_video( file_hash, video_path, cache_paths, backend_name, current, proposed,
                             labels, clip_seconds, clips, replay=replay,
                             duration_of=duration_of, experiment=experiment, focus=focus )[0]
        # Each video is replayed exactly once here, so its parsed cache is
        # dead weight for the rest of the loop (gigabytes across a library).
        bu_tuning.forget_raw_boxes( cache_paths )
        rows.append( row )
        logger.info( "rank %d/%d: %s differs by %.1f box-second(s)"%(
            number, len( videos ), row['name'], row['differing_seconds'] ) )
    rows.sort( key=lambda row: ( -row['differing_seconds'], row['name'] ) )
    return rows


# --- rendering -----------------------------------------------------------------------

def short_cause( cause ):
    """'exposed_breast<-face_femme' -> '<-face_femme': the label is drawn on the line above."""
    return ab_experiments.SuppressionExperiment().note( cause, {} )


def noted( boxes, causes, note=None ):
    """Copies of boxes carrying their cause's note (note(cause), suppression's by default) for the outline."""
    note = note or short_cause
    return [ dict( box, debug_note=note( cause ) ) for box, cause in zip( boxes, causes ) ]


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
    side's box toward the other's. Outlines held per sample are enough to
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
    """
    left | right in one file, left's audio. False when ffmpeg refuses.
    Written beside out_path and moved into place when complete, so a run
    killed mid-stack never leaves a half file that looks finished.
    """
    tmp_path = bu_cache.temp_sibling( out_path )
    # The render's own encoder arguments, so NVENC gets its quality as -cq
    # (handed -crf it ignores it and encodes at its default bitrate).
    # Threads 0: x264 picks its own, as it did when none were passed.
    encode = bu_render.encoder_options( getattr( betaconfig, 'encode_video_codec', 'libx264' ),
                                        getattr( betaconfig, 'encode_crf', 17 ),
                                        getattr( betaconfig, 'encode_preset', 'fast' ), 0 )
    command = ( [ 'ffmpeg', '-y', '-loglevel', 'error', '-i', left, '-i', right,
                  '-filter_complex', '[0:v][1:v]hstack=inputs=2[v]', '-map', '[v]', '-map', '0:a?' ]
                + encode + [ '-c:a', 'copy', tmp_path ] )
    result = subprocess.run( command, capture_output=True, text=True )
    if result.returncode != 0:
        if os.path.exists( tmp_path ):
            os.remove( tmp_path )
        logger.warning( "side-by-side failed (%s); keeping the two clips separately"%(
            result.stderr.strip().splitlines()[-1] if result.stderr.strip() else result.returncode, ) )
        return False
    os.replace( tmp_path, out_path )
    return True


def _stem( row ):
    return os.path.splitext( row['name'] )[0][ :60 ].replace( ' ', '_' )


def clip_tags( row ):
    """One tag per clip window: '<stem>_clipNN_<start>s', the prefix of its files."""
    stem = _stem( row )
    return [ '%s_clip%02d_%ds'%( stem, number, int( start ) )
             for number, ( start, _end ) in enumerate( row.get( 'windows', [] ), 1 ) ]


def clip_done( out_dir, tag ):
    """A window's side-by-side exists: the stacked file, or both halves when stacking failed."""
    return ( os.path.exists( os.path.join( out_dir, tag + '_ab.mp4' ) )
             or ( os.path.exists( os.path.join( out_dir, tag + '_current.mp4' ) )
                  and os.path.exists( os.path.join( out_dir, tag + '_proposed.mp4' ) ) ) )


def full_paths( out_dir, row ):
    return [ os.path.join( out_dir, '%s_full_%s.mp4'%( _stem( row ), name ) )
             for name in ( 'current', 'proposed' ) ]


def render_video_comparison( row, ranking, logger, out_dir, overlay=True, censored=True, full=False,
                             render=None, stack=None, compare=None, skip_existing=False ):
    """
    Every requested render for one ranked video. skip_existing leaves
    alone every file a finished earlier attempt wrote (all of them are
    written whole or not at all), and skips the replay too when nothing
    is left to render: how a resumed round picks up where it stopped.
    """
    ranking = normalised( ranking )
    render = render or render_window
    stack = stack or side_by_side
    if skip_existing:
        wanted = []
        for tag in clip_tags( row ):
            if overlay:
                wanted.append( os.path.exists( os.path.join( out_dir, tag + '_diff.mp4' ) ) )
            if censored:
                wanted.append( clip_done( out_dir, tag ) )
        if full:
            wanted += [ os.path.exists( path ) for path in full_paths( out_dir, row ) ]
        if all( wanted ):
            logger.info( "already rendered, nothing to do: %s"%( row['name'], ) )
            return []
    experiment = ab_experiments.experiment_named( ranking['experiment'] )
    compare = compare or functools.partial( compare_with_causes, experiment=experiment )
    result = compare( row['file_hash'], row['video'], row['cache_paths'], ranking['backend'],
                      ranking['current'], ranking['proposed'] )
    both, current_boxes, proposed_boxes = result['both'], result['current_boxes'], result['proposed_boxes']
    def note( cause ):
        return experiment.note( cause, ranking['proposed'] )
    current_only = noted( result['current_only'], result['current_only_causes'], note )
    proposed_only = noted( result['proposed_only'], result['proposed_only_causes'], note )
    produced = []
    for tag, ( start, end ) in zip( clip_tags( row ), row.get( 'windows', [] ) ):
        length = end - start
        if overlay:
            path = os.path.join( out_dir, tag + '_diff.mp4' )
            if skip_existing and os.path.exists( path ):
                logger.info( "already rendered: %s"%( os.path.basename( path ), ) )
            else:
                logger.info( "rendering %s (outlines: grey both, orange config only, magenta "
                             "change only)"%( os.path.basename( path ), ) )
                with motion_interpolation_off():
                    render( row['video'], row['file_hash'],
                            overlay_boxes( both, current_only, proposed_only ),
                            path, start, length, logger )
                produced.append( path )
        if censored and skip_existing and clip_done( out_dir, tag ):
            logger.info( "already rendered: %s side by side"%( tag, ) )
        elif censored:
            left = os.path.join( out_dir, tag + '_current.mp4' )
            right = os.path.join( out_dir, tag + '_proposed.mp4' )
            render( row['video'], row['file_hash'], current_boxes, left, start, length, logger )
            render( row['video'], row['file_hash'], proposed_boxes, right, start, length, logger )
            combined = os.path.join( out_dir, tag + '_ab.mp4' )
            if stack( left, right, combined, logger ):
                os.remove( left )
                os.remove( right )
                produced.append( combined )
                logger.info( "rendered %s (config LEFT, change RIGHT)"%(
                    os.path.basename( combined ), ) )
            else:
                produced += [ left, right ]
    if full:
        for path, name, boxes in zip( full_paths( out_dir, row ), ( 'current', 'proposed' ),
                                      ( current_boxes, proposed_boxes ) ):
            if skip_existing and os.path.exists( path ):
                logger.info( "already rendered: %s"%( os.path.basename( path ), ) )
                continue
            logger.info( "rendering the whole video, %s side: %s"%( name, path ) )
            render( row['video'], row['file_hash'], boxes, path, None, None, logger )
            produced.append( path )
    return produced


def round_status( out_dir ):
    """
    How far the round in out_dir got, for resuming it: ranked, explained,
    clips (every window's side by side), full (both whole videos of every
    chosen one), verdicts filled in of rows, decided, adopted (a change won).
    """
    status = { 'ranked': 0, 'experiment': '', 'videos': 0, 'explained': 0, 'clips': 0,
               'full': 0, 'rows': 0, 'verdicts': 0, 'decided': 0, 'adopted': 0 }
    path = os.path.join( out_dir, 'ranking.json' )
    if not os.path.exists( path ):
        return status
    with open( path ) as handle:
        ranking = normalised( json.load( handle ) )
    by_hash = { row['file_hash']: row for row in ranking.get( 'videos', [] ) }
    chosen = [ by_hash[file_hash] for file_hash in ranking.get( 'chosen', [] ) if file_hash in by_hash ]
    status.update( ranked=1, experiment=ranking['experiment'], videos=len( chosen ) )
    feedback = os.path.join( out_dir, 'feedback.csv' )
    status['explained'] = int( os.path.exists( os.path.join( out_dir, 'clips.md' ) )
                               and os.path.exists( feedback ) )
    status['clips'] = int( bool( chosen ) and all( clip_done( out_dir, tag )
                                                   for row in chosen for tag in clip_tags( row ) ) )
    status['full'] = int( bool( chosen ) and all( os.path.exists( full )
                                                  for row in chosen for full in full_paths( out_dir, row ) ) )
    if os.path.exists( feedback ):
        with open( feedback, newline='', encoding='UTF-8' ) as handle:
            rows = list( csv.DictReader( handle ) )
        status['rows'] = len( rows )
        status['verdicts'] = verdict_count( feedback )
    chosen_path = os.path.join( out_dir, 'chosen.json' )
    if os.path.exists( chosen_path ):
        status['decided'] = 1
        with open( chosen_path ) as handle:
            status['adopted'] = int( bool( json.load( handle ) ) )
    return status


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


def _log_ranking( logger, rows, shown, focus ):
    logger.info( "" )
    logger.info( "  %-50s %-12s %10s %7s %10s %10s %9s"%(
        'video', 'profile', 'differ (s)', 'share', 'only now', 'only new', 'restored' ) )
    for row in rows[ :shown ]:
        logger.info( "  %-50s %-12s %10.1f %6.1f%% %10.1f %10.1f %9d"%(
            row['name'][ :50 ], row['profile'] or '', row['differing_seconds'],
            100.0 * row['differing_share'], row['current_only_seconds'],
            row['proposed_only_seconds'], row['confident_restored'] ) )
    logger.info( "  differ: box-seconds censored on one side and not the other%s (the ranking). "
                 "share: differ over the seconds censored now. only now / only new: censored by "
                 "the config / the change only. restored: confident detections (score >= %.2f) "
                 "the change censors that the config leaves bare"%(
                     ( ", %s only"%( ", ".join( focus ), ) ) if focus else "",
                     bu_tuning.CONFIDENT_SCORE ) )


def choose_videos( rows, count, earlier=None, logger=None ):
    """
    The `count` most-changed rows that differ at all, skipping a copy.

    A copy is a row whose signature bu_tuning.same_content matches a video
    already chosen here or shown by an earlier round (earlier: {name:
    signature}). The first cut-burst round spent both its videos on one
    compilation saved twice under two names.
    """
    chosen = []
    seen = list( ( earlier or {} ).items() )
    for row in rows:
        if len( chosen ) >= count or row['differing_seconds'] <= 0:
            break
        copy_of = next( ( name for name, signature in seen
                          if bu_tuning.same_content( row.get( 'signature' ), signature ) ), None )
        if copy_of:
            if logger:
                logger.info( "skipping %s: the same footage as %s"%( row['name'], copy_of ) )
            continue
        chosen.append( row )
        seen.append( ( row['name'], row.get( 'signature' ) ) )
    return chosen


def earlier_signatures( paths ):
    """{name: signature} of the videos earlier rounds showed, where their ranking recorded one."""
    found = {}
    for path in paths or []:
        if os.path.isdir( path ):
            path = os.path.join( path, 'ranking.json' )
        with open( path ) as handle:
            earlier = json.load( handle )
        shown = set( earlier.get( 'chosen', [] ) )
        for row in earlier.get( 'videos', [] ):
            if row.get( 'file_hash' ) in shown and row.get( 'signature' ):
                found[row['name']] = row['signature']
    return found


def excluded_hashes( paths ):
    """Videos earlier rounds already showed: each ranking's chosen and its own exclusions."""
    found = set()
    for path in paths or []:
        if os.path.isdir( path ):
            path = os.path.join( path, 'ranking.json' )
        with open( path ) as handle:
            earlier = json.load( handle )
        found.update( earlier.get( 'chosen', [] ) )
        found.update( earlier.get( 'excluded', [] ) )
    return found


def phase_rank( args, logger, betabench ):
    experiment = ab_experiments.experiment_named( args.experiment )
    config, min_prob = _configuration( args, logger )
    backend_name = config.backend_name
    # Every sample rate, so quick-cut videos (detected at their
    # profile's rate) are ranked too.
    videos, rates = bu_cache.videos_across_sample_rates( config, min_prob )
    excluded = excluded_hashes( args.exclude_from )
    if excluded:
        videos = { file_hash: paths for file_hash, paths in videos.items() if file_hash not in excluded }
        logger.info( "leaving out %d video(s) earlier rounds already showed"%( len( excluded ), ) )
    hash_to_path = bu_cache.build_hash_to_video_path( videos.keys() )
    current = experiment.current( backend_name )
    if args.proposed:
        with open( args.proposed ) as handle:
            proposed = json.load( handle )
        source = args.proposed
    else:
        proposed, source = experiment.propose( args, logger, config, betabench )
    logger.info( "experiment: %s (%s)"%( experiment.name, experiment.summary ) )
    logger.info( "config as it is: %s"%( json.dumps( current, sort_keys=True ), ) )
    logger.info( "proposed (%s): %s"%( source, json.dumps( proposed, sort_keys=True ) ) )
    changes = experiment.change_lines( current, proposed, backend_name )
    for line in changes:
        logger.info( "  change %s"%( line, ) )
    if experiment.is_noop( current, proposed, backend_name ):
        raise SystemExit( "the proposed side is the config as it is; nothing to compare" )

    labels = sorted( betaconfig.items_to_censor )
    focus = sorted( args.focus ) if args.focus else []
    logger.info( "replaying %d video(s) of %s (sample rates %s) on both sides%s"%(
        len( videos ), config.label, ", ".join( '%g'%( rate, ) for rate in rates ) or 'none',
        ( "; ranking by %s only"%( ", ".join( focus ), ) ) if focus else "" ) )
    rows = rank_videos( videos, hash_to_path, backend_name, current, proposed, set( labels ), logger,
                        clip_seconds=args.clip_seconds, clips=args.clips, experiment=experiment,
                        focus=focus )
    _log_ranking( logger, rows, max( 10, args.videos ), focus )
    chosen = choose_videos( rows, args.videos, earlier_signatures( args.exclude_from ), logger )
    for row in chosen:
        logger.info( "chosen: %s, %d clip window(s): %s"%(
            row['name'], len( row['windows'] ),
            ", ".join( "%ds-%ds"%( start, end ) for start, end in row['windows'] ) ) )
    if not chosen:
        logger.warning( "no video renders differently on the two sides" )
    ranking = { 'experiment': experiment.name, 'backend': backend_name, 'configuration': config.label,
                'labels': labels, 'sample_rates': rates, 'focus': focus,
                'current': current, 'proposed': proposed, 'proposed_source': source,
                'changes': changes, 'excluded': sorted( excluded ),
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
                                             censored=not args.no_censored, full=args.full,
                                             skip_existing=args.skip_existing )
    logger.info( "" )
    logger.info( "%d file(s) in %s"%( len( produced ), args.out_dir ) )
    for path in produced:
        logger.info( "  %s"%( os.path.basename( path ), ) )
    return produced


# --- the eye exam: explain the clips, collect verdicts, decide ----------------------------

FEEDBACK_COLUMNS = ( 'id', 'clip', 'when_in_clip', 'source_seconds', 'label', 'censored_on',
                     'cause', 'change', 'verdict', 'note' )
VERDICTS = { 'r': 'proposed', 'right': 'proposed', 'proposed': 'proposed', 'p': 'proposed',
             'l': 'current', 'left': 'current', 'current': 'current', 'c': 'current',
             '=': 'same', 'same': 'same', 's': 'same' }


def _clock( seconds ):
    return '%d:%04.1f'%( int( seconds // 60 ), seconds % 60 )


def _change_text( changes, cause ):
    if cause == TRACKING_CAUSE:
        return "no single change: a knock-on effect on tracking nearby"
    return changes.get( cause, "unchanged" )


def explain_video( row, ranking, video_number, compare=None ):
    """(markdown lines, feedback rows) for one chosen video's clip windows."""
    ranking = normalised( ranking )
    experiment = ab_experiments.experiment_named( ranking['experiment'] )
    compare = compare or functools.partial( compare_with_causes, experiment=experiment )
    result = compare( row['file_hash'], row['video'], row['cache_paths'], ranking['backend'],
                      ranking['current'], ranking['proposed'] )
    changes = experiment.changes_by_cause( ranking['current'], ranking['proposed'], ranking['backend'] )
    stem = _stem( row )
    lines = [ "## Video %d: %s"%( video_number, row['name'] ), "",
              "Whole video, box-seconds by cause:", "" ]
    for side, key in ( ( 'RIGHT only (the change censors, the config leaves bare)', 'proposed_only' ),
                       ( 'LEFT only (the config censors, the change leaves bare)', 'current_only' ) ):
        totals = cause_seconds( result[key], result[key + '_causes'] )
        lines.append( "- %s: %s"%( side, ", ".join( "%s %.1fs"%( cause, seconds )
                                                    for cause, seconds in totals.items() ) or "none" ) )
    lines.append( "" )
    rows = []
    for clip_number, ( start, end ) in enumerate( row.get( 'windows', [] ), 1 ):
        tag = '%s_clip%02d_%ds'%( stem, clip_number, int( start ) )
        lines += [ "### Clip %d: `%s_ab.mp4` (and `_diff.mp4`)"%( clip_number, tag ),
                   "", "Source %s to %s. LEFT is the config as it is, RIGHT the change."%(
                       _clock( start ), _clock( end ) ), "" ]
        moments = window_moments( result, start, end )
        if not moments:
            lines += [ "No differences inside this clip.", "" ]
            continue
        lines += [ "| id | when in clip | label | what differs | cause | change |",
                   "|---|---|---|---|---|---|" ]
        for item, moment in enumerate( moments, 1 ):
            row_id = 'v%dc%d.%d'%( video_number, clip_number, item )
            span_start, span_end = moment['span']
            when = "%s-%s"%( _clock( span_start ), _clock( span_end ) )
            source = "%.1f-%.1f"%( start + span_start, start + span_end )
            change = _change_text( changes, moment['cause'] )
            lines.append( "| %s | %s | %s | %s | %s | %s |"%(
                row_id, when, moment['label'], WHAT_DIFFERS[moment['side']], moment['cause'], change ) )
            rows.append( { 'id': row_id, 'clip': tag + '_ab.mp4', 'when_in_clip': when,
                           'source_seconds': source, 'label': moment['label'],
                           'censored_on': moment['side'], 'cause': moment['cause'], 'change': change,
                           'verdict': '', 'note': '' } )
        lines.append( "" )
    return lines, rows


def phase_explain( args, logger, ranking, compare=None ):
    """Write clips.md (what differs where, and why) and feedback.csv (one verdict per row)."""
    ranking = normalised( ranking )
    experiment = ab_experiments.experiment_named( ranking['experiment'] )
    by_hash = { row['file_hash']: row for row in ranking['videos'] }
    lines = [ "# A/B, %s: what differs in each clip"%( experiment.title, ), "",
              experiment.explain_intro(), "",
              "Each row is one moment: one label, at one stretch of time in the clip. `what "
              "differs` (`censored_on` in feedback.csv) is one of three things:", "",
              "- RIGHT censors, LEFT bare (`RIGHT only`): RIGHT is right if something real is "
              "exposed there, LEFT if the RIGHT box sits on nothing", "",
              "- LEFT censors, RIGHT bare (`LEFT only`): the same question the other way round", "",
              "- both censor, boxes differ (`both differ`): both sides have a box there, in "
              "another place or of another size; pick the side whose box covers the body part", "",
              "`cause` is the setting responsible. `change` is what the proposal does to that "
              "setting; for suppression it describes the rule, so `dropped` means the proposal "
              "deletes that rule from the config.", "",
              "Verdict per row, in feedback.csv or in chat by id: `right` (the RIGHT side looks "
              "right there), `left` (the LEFT side does), `same`, or blank if you cannot tell. "
              "Then `bash tools/tuning/ab_compare.sh decide` (or `resume`).", "",
              "Changes under test:", "" ]
    lines += [ "- %s"%( line, ) for line in experiment.change_lines(
        ranking['current'], ranking['proposed'], ranking['backend'] ) ]
    if ranking.get( 'focus' ):
        lines += [ "", "Videos and windows were picked by %s differences only."%(
            ", ".join( ranking['focus'] ), ) ]
    lines.append( "" )
    rows = []
    for number, file_hash in enumerate( ranking['chosen'], 1 ):
        logger.info( "explaining %s"%( by_hash[file_hash]['name'], ) )
        video_lines, video_rows = explain_video( by_hash[file_hash], ranking, number, compare=compare )
        lines += video_lines
        rows += video_rows
    markdown = os.path.join( args.out_dir, 'clips.md' )
    with open( markdown, 'w', encoding='UTF-8' ) as handle:
        handle.write( "\n".join( lines ) + "\n" )
    feedback = os.path.join( args.out_dir, 'feedback.csv' )
    if os.path.exists( feedback ) and verdict_count( feedback ):
        feedback = os.path.join( args.out_dir, 'feedback_template.csv' )
        logger.warning( "feedback.csv already has verdicts and is kept; the fresh template is %s"%(
            feedback, ) )
    elif os.path.exists( feedback ):
        logger.info( "feedback.csv had no verdicts yet, so it is replaced" )
    with open( feedback, 'w', newline='', encoding='UTF-8' ) as handle:
        writer = csv.DictWriter( handle, fieldnames=FEEDBACK_COLUMNS )
        writer.writeheader()
        writer.writerows( rows )
    logger.info( "%d difference(s) to judge: %s"%( len( rows ), markdown ) )
    logger.info( "verdicts go in %s"%( feedback, ) )
    return rows


def verdict_count( path ):
    """Rows of a feedback.csv that carry a verdict (0 for an unreadable file)."""
    try:
        with open( path, newline='', encoding='UTF-8' ) as handle:
            return sum( 1 for row in csv.DictReader( handle )
                        if VERDICTS.get( ( row.get( 'verdict' ) or '' ).strip().lower() ) )
    except ( OSError, csv.Error, UnicodeDecodeError ):
        return 0


def tally_verdicts( rows ):
    """{cause: {'proposed': n, 'current': n, 'same': n, 'unjudged': n}}, knock-on rows under TRACKING_CAUSE."""
    tally = {}
    for row in rows:
        counts = tally.setdefault( row['cause'], { 'proposed': 0, 'current': 0, 'same': 0,
                                                   'unjudged': 0 } )
        verdict = VERDICTS.get( ( row.get( 'verdict' ) or '' ).strip().lower() )
        counts[ verdict or 'unjudged' ] += 1
    return tally


def phase_decide( args, logger, ranking ):
    """Read the verdicts, write chosen.json, and print the config lines."""
    ranking = normalised( ranking )
    experiment = ab_experiments.experiment_named( ranking['experiment'] )
    path = args.feedback or os.path.join( args.out_dir, 'feedback.csv' )
    with open( path, newline='', encoding='UTF-8' ) as handle:
        rows = list( csv.DictReader( handle ) )
    tally = tally_verdicts( rows )
    chosen, decisions = experiment.choose( ranking['current'], ranking['proposed'], tally,
                                           ranking['backend'] )
    logger.info( "verdicts from %s"%( path, ) )
    logger.info( "  %-36s %8s %8s %6s %9s   decision"%( 'change', 'right', 'left', 'same',
                                                         'unjudged' ) )
    for cause in sorted( tally ):
        counts = tally[cause]
        logger.info( "  %-36s %8d %8d %6d %9d   %s"%(
            cause, counts['proposed'], counts['current'], counts['same'], counts['unjudged'],
            decisions.get( cause, 'informational (no single change)' if cause == TRACKING_CAUSE
                           else 'unchanged' ) ) )
    for cause, decision in sorted( decisions.items() ):
        if cause not in tally:
            logger.info( "  %-36s %8s %8s %6s %9s   %s"%( cause, '-', '-', '-', '-', decision ) )
    written = []
    for name in sorted( { 'chosen.json', experiment.chosen_file } ):
        out = os.path.join( args.out_dir, name )
        with open( out, 'w' ) as handle:
            json.dump( chosen, handle, indent=2 )
        written.append( out )
    logger.info( "" )
    logger.info( "chosen side written to %s"%( ", ".join( written ), ) )
    adopted = sorted( cause for cause, decision in decisions.items() if decision == 'proposed' )
    if not adopted:
        logger.info( "every change kept the config as it is; nothing to paste" )
    else:
        logger.info( "to adopt %s, put this in betaconfig.py:"%( ", ".join( adopted ), ) )
        for line in experiment.chosen_lines( chosen, ranking['current'], ranking['backend'] ):
            logger.info( "    %s"%( line, ) )
        logger.info( "next round: check the chosen side on videos you have not judged yet, before "
                     "pasting: bash tools/tuning/ab_compare.sh confirm" )
    follow_up = experiment.follow_up_lines( ranking['current'], ranking['proposed'], decisions,
                                            ranking['backend'] )
    if follow_up:
        logger.info( "" )
        logger.info( "also worth pasting, %s"%( follow_up[0], ) )
        for line in follow_up[1:]:
            logger.info( "    %s"%( line, ) )
    return chosen, decisions


def confirming( args ):
    """
    Point args at a confirm round of args.confirm_from: its experiment,
    its chosen side as the proposal, its focus unless one was given, and
    every video it or its own earlier rounds showed left out.
    """
    previous = args.confirm_from
    with open( os.path.join( previous, 'ranking.json' ) ) as handle:
        ranking = normalised( json.load( handle ) )
    chosen = os.path.join( previous, 'chosen.json' )
    if not os.path.exists( chosen ):
        raise SystemExit( "%s has no chosen.json; run decide there first"%( previous, ) )
    with open( chosen ) as handle:
        if not json.load( handle ):
            raise SystemExit( "%s chose nothing (every change kept the config); nothing to "
                              "confirm"%( previous, ) )
    args.experiment = ranking['experiment']
    args.proposed = chosen
    args.exclude_from = list( args.exclude_from or [] ) + [ previous ]
    args.focus = args.focus or ranking.get( 'focus' ) or None
    return args


def build_parser():
    parser = argparse.ArgumentParser( description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter )
    parser.add_argument( '--experiment', default='suppression',
                         choices=[ experiment.name for experiment in ab_experiments.experiments() ],
        help="what to compare (default: suppression)" )
    parser.add_argument( '--phase', choices=( 'rank', 'explain', 'render', 'decide', 'all' ),
                         default='all',
        help="all = rank, explain, render. decide reads the verdicts in feedback.csv" )
    parser.add_argument( '--feedback', default=None, metavar='feedback.csv',
        help="decide phase: the verdicts (default: --out-dir/feedback.csv)" )
    parser.add_argument( '--videos', type=int, default=2,
        help="how many of the most-changed videos to render (default: 2)" )
    parser.add_argument( '--clips', type=int, default=6,
        help="clip windows per video, where the sides differ most (default: 6)" )
    parser.add_argument( '--clip-seconds', type=float, default=15.0,
        help="length of each clip (default: 15)" )
    parser.add_argument( '--focus', nargs='+', default=None, metavar='LABEL',
        help="rank videos and pick windows by these labels' differences only" )
    parser.add_argument( '--exclude-from', nargs='+', default=None, metavar='RANKING',
        help="leave out the videos these earlier rounds showed (ranking.json or its folder)" )
    parser.add_argument( '--no-overlay', action='store_true', help="skip the outline renders" )
    parser.add_argument( '--no-censored', action='store_true', help="skip the side-by-side renders" )
    parser.add_argument( '--full', action='store_true',
        help="also render each chosen video whole on both sides (hours)" )
    parser.add_argument( '--skip-existing', action='store_true',
        help="render phase: leave every file an earlier attempt finished (resuming a round)" )
    parser.add_argument( '--status', default=None, metavar='DIR',
        help="print how far DIR's round got, one key=value per line, then stop" )
    parser.add_argument( '--proposed', default=None, metavar='FILE.json',
        help="the proposed side as JSON (e.g. an earlier round's chosen.json)" )
    parser.add_argument( '--set', action='append', default=None, metavar='label.key=value',
        help="build the proposed side by hand: label.key=value for per-label settings, "
             "name=value for cut_bursts; repeat for more" )
    parser.add_argument( '--auto-tune', default=None, metavar='decisions.json',
        help="auto_tune decisions to take review values from (default: the newest analysis run's)" )
    parser.add_argument( '--ranking', default=None, metavar='ranking.json',
        help="explain/render/decide: the rank phase's output (default: --out-dir/ranking.json)" )
    parser.add_argument( '--min-samples', type=int, default=30,
        help="suppression: overlaps a pair needs before the bench suggests a rule (default: 30)" )
    parser.add_argument( '--no-additions', dest='additions', action='store_false',
        help="suppression: only retune or drop configured rules, never propose adding a pair" )
    parser.add_argument( '--confirm-from', default=None, metavar='DIR',
        help="rank phase: DIR's chosen side against the config, same experiment and focus, "
             "leaving out every video DIR's rounds already showed" )
    parser.add_argument( '--run-experiment', default=None, metavar='DIR',
        help="print the experiment DIR's round compared, then stop" )
    parser.add_argument( '--list-experiments', action='store_true',
        help="print each experiment's name and what it compares, then stop" )
    parser.add_argument( '--backend', default=None )
    parser.add_argument( '--variant', default=None )
    parser.add_argument( '--out-dir', default=None )
    parser.add_argument( '--log-level', default='debug',
                         choices=( 'trace', 'debug', 'info', 'warn', 'error' ) )
    parser.add_argument( '--console-level', default='info',
                         choices=( 'trace', 'debug', 'info', 'warn', 'error' ) )
    return parser


def main( argv=None, default_experiment=None ):
    parser = build_parser()
    if default_experiment:
        parser.set_defaults( experiment=default_experiment )
    args = parser.parse_args( argv )
    if args.list_experiments:
        for experiment in ab_experiments.experiments():
            print( "%s - %s"%( experiment.name, experiment.summary ) )
        return 0
    if args.status:
        for key, value in round_status( args.status ).items():
            print( "%s=%s"%( key, value ) )
        return 0
    if args.run_experiment:
        with open( os.path.join( args.run_experiment, 'ranking.json' ) ) as handle:
            print( normalised( json.load( handle ) )['experiment'] )
        return 0
    if args.confirm_from:
        confirming( args )
    args.out_dir = args.out_dir or os.path.join(
        os.path.dirname( CORE_DIR ), 'output', 'benchmarks',
        'ab_%s_%s'%( args.experiment, time.strftime( '%Y%m%d_%H%M%S' ) ) )
    os.makedirs( args.out_dir, exist_ok=True )
    betabench = _load_betabench()
    logger = betabench.build_logger( args.out_dir, args.log_level, args.console_level )
    logger.info( "A/B (%s): output in %s"%( args.experiment, args.out_dir ) )
    ranking = None
    if args.phase in ( 'rank', 'all' ):
        ranking = phase_rank( args, logger, betabench )
    if ranking is None:
        with open( args.ranking or os.path.join( args.out_dir, 'ranking.json' ) ) as handle:
            ranking = normalised( json.load( handle ) )
    if args.phase in ( 'explain', 'all' ):
        phase_explain( args, logger, ranking )
    if args.phase in ( 'render', 'all' ):
        phase_render( args, logger, ranking )
    if args.phase == 'decide':
        phase_decide( args, logger, ranking )
    return 0


if __name__ == '__main__':
    raise SystemExit( main() )
