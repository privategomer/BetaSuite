#!/usr/bin/env python3
"""
summarize_run.py - what a finished betatv.py run actually cost and found.

This is the first thing to run after a real run completes. Everything
else in tools/analysis and tools/bench answers "how should I tune X";
this one answers the question that comes before it: which stage is
actually expensive, and did the detections change.

It reads betasuite_stats.jsonl, the one JSON-Lines row per processed
file that every run appends, and groups those rows by the configuration
that produced them - backend, model variant, picture sizes, sample rate,
batch size. Rows from different configurations are never pooled: the
whole point of a two-backend run is the comparison between the groups,
and averaging them together destroys exactly the number you ran it for.

WHAT THE STAGES MEAN

  detection  Sampling frames and running the model. Scales with
             video_censor_fps, picture size, and model size.
  render     Drawing censor boxes onto every native frame and encoding
             them. Scales with the NATIVE frame count and the number of
             boxes per frame - not with the sample rate. A video can
             therefore be cheap to detect and expensive to render.
  other      Whatever total_seconds has left after those two: hashing,
             transcode, audio mux, shot-cut scan.

The ratio between them is the whole point. Tuning detection settings on
a run where render is 80% of the wall clock buys you 20% of a
percentage - the arithmetic sets the ceiling on what any detection-side
change can possibly save, before you spend an evening on it.

Analogy: this is the itemised bill, not the menu. Read it before
deciding what to order differently.

Usage:
    python3 tools/analysis/summarize_run.py
    python3 tools/analysis/summarize_run.py --last 7      # most recent 7 rows only
    python3 tools/analysis/summarize_run.py --since 2026-09-17
    python3 tools/analysis/summarize_run.py --include-preview
    python3 tools/analysis/summarize_run.py --stats-path /path/to/other.jsonl
"""

import argparse
import datetime
import json
import os
import sys

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) ) )

import betaconfig
import betautils_cache_paths as bu_cache
import betautils_log as bu_log


DEFAULT_STATS_PATH = '../output/stats/betasuite_stats.jsonl'


def load_rows( stats_path ):
    """
    Every readable stats row, oldest first.

    A malformed line is skipped rather than fatal: stats are appended
    from a running pipeline, so a row truncated by an interrupted run is
    an expected thing to find, not a reason to refuse to report.

    Returns:
        A (rows, skipped_count) pair.
    """
    rows = []
    skipped = 0
    if not os.path.exists( stats_path ):
        return rows, skipped
    with open( stats_path, 'r', encoding='UTF-8' ) as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append( json.loads( line ) )
            except Exception:
                skipped += 1
    return rows, skipped


def group_key( row ):
    """
    The configuration a row was produced under.

    Rows are only ever compared within a key. Two rows that differ in
    any of these fields describe different runs of different software,
    and a mean across them means nothing.
    """
    return (
        row.get( 'detector_backend', 'unknown' ),
        row.get( 'detector_variant' ) or '-',
        tuple( row.get( 'picture_sizes' ) or () ),
        row.get( 'video_censor_fps' ),
        row.get( 'nn_batch_size' ),
        bool( row.get( 'preview_mode' ) ),
    )


def describe_key( key ):
    backend, variant, sizes, fps, batch, preview = key
    return "%s/%s  sizes=%s  fps=%s  batch=%s%s"%(
        backend, variant, list( sizes ), fps, batch,
        "  [PREVIEW SLICE]" if preview else "" )


def video_seconds( row ):
    """Source duration in seconds, or 0 when the row cannot say."""
    frames = row.get( 'video_frames' ) or 0
    fps = row.get( 'video_fps' ) or 0
    if frames and fps:
        return frames / fps
    return 0.0


def summarise_group( rows ):
    """Aggregate one configuration's rows into the numbers worth printing."""
    totals = {
        'files': len( rows ),
        'video_seconds': 0.0,
        'detection_seconds': 0.0,
        'encode_seconds': 0.0,
        'total_seconds': 0.0,
        'frames_written': 0,
        'chunks': 0,
        'chunks_reused': 0,
        'tracks': 0,
        'interpolated': 0,
        'raw_detections': 0,
        'cross_size_deduped': 0,
        'dropped_provisional': 0,
        'dropped_unconfirmed': 0,
    }
    worker_counts = {}
    label_counts = {}
    providers = set()

    for row in rows:
        totals['video_seconds'] += video_seconds( row )
        totals['detection_seconds'] += row.get( 'detection_seconds' ) or 0.0
        totals['encode_seconds'] += row.get( 'encode_seconds' ) or 0.0
        totals['total_seconds'] += row.get( 'total_seconds' ) or 0.0
        totals['frames_written'] += row.get( 'render_frames_written' ) or 0
        totals['chunks'] += row.get( 'render_chunks' ) or 0
        totals['chunks_reused'] += row.get( 'render_chunks_reused' ) or 0
        totals['tracks'] += row.get( 'track_tracks' ) or 0
        totals['interpolated'] += row.get( 'track_interpolated' ) or row.get( 'interpolated_boxes' ) or 0
        totals['raw_detections'] += row.get( 'track_raw_detections' ) or 0
        totals['cross_size_deduped'] += row.get( 'track_cross_size_deduped' ) or 0
        totals['dropped_provisional'] += row.get( 'track_dropped_provisional' ) or 0
        totals['dropped_unconfirmed'] += row.get( 'track_dropped_unconfirmed' ) or 0

        workers = row.get( 'render_render_workers' )
        if workers is not None:
            worker_counts[workers] = worker_counts.get( workers, 0 ) + 1

        for label, count in ( row.get( 'label_counts' ) or
                              row.get( 'track_label_counts' ) or {} ).items():
            label_counts[label] = label_counts.get( label, 0 ) + count

        provider = row.get( 'execution_providers' )
        if provider:
            providers.add( provider if isinstance( provider, str ) else ",".join( provider ) )

    totals['worker_counts'] = worker_counts
    totals['label_counts'] = label_counts
    totals['providers'] = providers
    return totals


def print_group( key, rows ):
    totals = summarise_group( rows )
    fmt = bu_log.format_duration

    detect = totals['detection_seconds']
    encode = totals['encode_seconds']
    total = totals['total_seconds']
    other = max( 0.0, total - detect - encode )

    print( "=" * 78 )
    print( describe_key( key ) )
    print( "=" * 78 )
    if totals['providers']:
        print( "  execution providers: %s"%(", ".join( sorted( totals['providers'] )) ) )
        if not any( 'CUDA' in p or 'Tensorrt' in p or 'ROCm' in p
                    for p in totals['providers'] ):
            print( "  !! no GPU provider in that list - this ran on CPU. Everything below is a "
                   "CPU number and is not comparable to a GPU run." )
    print( "  files: %d   source video: %s"%(totals['files'], fmt( totals['video_seconds'] )) )
    print()

    def stage( name, seconds ):
        share = 100.0 * seconds / total if total else 0.0
        realtime = seconds / totals['video_seconds'] if totals['video_seconds'] else 0.0
        print( "    %-10s %10s  %5.1f%% of wall clock   %5.2fx source duration"%(
            name, fmt( seconds ), share, realtime ) )

    print( "  where the wall clock went:" )
    stage( 'detection', detect )
    stage( 'render', encode )
    stage( 'other', other )
    stage( 'TOTAL', total )
    print()

    # Name the dominant stage explicitly. The arithmetic below is the
    # only thing standing between "I tuned detection for an evening" and
    # "I tuned the stage that was 84% of the bill".
    if total:
        dominant, dominant_seconds = max(
            ( ('detection', detect), ('render', encode), ('other', other) ),
            key=lambda pair: pair[1] )
        share = 100.0 * dominant_seconds / total
        print( "  %s is the dominant cost at %.0f%%. Making everything ELSE instantaneous would "
               "save at most %.0f%% of this run's wall clock."%(
                   dominant, share, 100.0 - share ) )
        print()

    if totals['chunks']:
        reuse = 100.0 * totals['chunks_reused'] / totals['chunks']
        print( "  render: %d chunk(s), %d reused from cache (%.0f%%), %d frame(s) written"%(
            totals['chunks'], totals['chunks_reused'], reuse, totals['frames_written'] ) )
        if totals['worker_counts']:
            histogram = ", ".join( "%d worker(s) x%d file(s)"%(workers, count)
                                   for workers, count in sorted( totals['worker_counts'].items() ) )
            print( "    parallelism actually used: %s"%(histogram) )
            max_workers = max( totals['worker_counts'] )
            configured = getattr( betaconfig, 'render_workers', 0 )
            chunk_seconds = getattr( betaconfig, 'render_chunk_seconds', None )
            if max_workers <= 2 and ( configured == 0 or configured > 2 ):
                print( "    note: a file is split into ceil(duration / render_chunk_seconds) chunks and "
                       "one worker renders one chunk, so a file shorter than 2x render_chunk_seconds "
                       "(currently %s) can never use more than 2 workers no matter what render_workers "
                       "says. Lowering render_chunk_seconds is what turns idle cores into parallelism."%(
                           chunk_seconds ) )
        print()

    if totals['label_counts']:
        print( "  detections kept, by label:" )
        for label, count in sorted( totals['label_counts'].items(),
                                    key=lambda pair: -pair[1] ):
            print( "    %-24s %8d"%(label, count) )
        print()

    print( "  tracking: %d track(s), %d interpolated box(es)"%(
        totals['tracks'], totals['interpolated'] ) )
    if totals['raw_detections']:
        print( "    raw detections before filtering: %d"%(totals['raw_detections']) )
    for name, key_name in ( ( 'cross-size duplicates removed', 'cross_size_deduped' ),
                            ( 'dropped, never reached min_prob', 'dropped_provisional' ),
                            ( 'dropped, unconfirmed tracks', 'dropped_unconfirmed' ) ):
        if totals[key_name]:
            print( "    %-34s %d"%(name + ':', totals[key_name]) )
    print()


def print_comparison( groups ):
    """
    The cross-configuration table: the reason two backends were run.

    Deliberately narrow. Detections per source minute and wall clock per
    source minute are the two numbers that survive a difference in which
    videos each configuration happened to process; raw totals do not.
    """
    print( "=" * 78 )
    print( "SIDE BY SIDE (per minute of source video, so groups that processed" )
    print( "different files are still comparable)" )
    print( "=" * 78 )
    print( "  %-42s %10s %10s %9s"%("configuration", "detect/min", "render/min", "dets/min") )
    for key in sorted( groups, key=describe_key ):
        totals = summarise_group( groups[key] )
        minutes = totals['video_seconds'] / 60.0
        if not minutes:
            continue
        detections = sum( totals['label_counts'].values() )
        print( "  %-42s %9.1fs %9.1fs %9.0f"%(
            describe_key( key )[:42],
            totals['detection_seconds'] / minutes,
            totals['encode_seconds'] / minutes,
            detections / minutes ) )
    print()
    print( "  More detections per minute is not automatically better - it is only better if the "
           "extra ones are real. Confirm with tools/bench/betabench.py hysteresis (are the extra "
           "detections low-confidence?) and by watching the output." )
    print()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter )
    parser.add_argument( '--stats-path', default=None,
        help="stats file to read (default: betaconfig.stats_path)" )
    parser.add_argument( '--last', type=int, default=None,
        help="only the most recent N rows - the quickest way to look at just the run that "
             "finished, without the history behind it" )
    parser.add_argument( '--since', default=None,
        help="only rows at or after this date/time (YYYY-MM-DD or YYYY-MM-DDTHH:MM)" )
    parser.add_argument( '--include-preview', action='store_true',
        help="also include preview-slice runs. Off by default: a preview row's totals describe a "
             "short slice, so pooling them with real runs makes both numbers wrong." )
    args = parser.parse_args()

    stats_path = args.stats_path or getattr( betaconfig, 'stats_path', None ) or DEFAULT_STATS_PATH
    rows, skipped = load_rows( stats_path )

    if not rows:
        # Not an error. "No run has written stats here yet" is a normal
        # state on a fresh checkout, and a suite that reports FAIL for
        # it teaches you to ignore its failures. A wrong FILTER is a
        # user error and does exit non-zero, further down.
        print( "no stats rows found at %s - nothing to summarise yet."%(stats_path) )
        print( "stats_enabled is %r in betaconfig.py; a run only writes rows when it is true."%(
            getattr( betaconfig, 'stats_enabled', None ) ) )
        sys.exit( 0 )
    if skipped:
        print( "(%d unparseable line(s) skipped - usually a row truncated by an interrupted run)"%(skipped) )

    if args.since:
        try:
            cutoff = datetime.datetime.fromisoformat( args.since ).timestamp()
        except ValueError:
            print( "could not parse --since %r - expected YYYY-MM-DD or YYYY-MM-DDTHH:MM"%(args.since) )
            sys.exit( 2 )
        rows = [ row for row in rows if ( row.get( 'timestamp' ) or 0 ) >= cutoff ]

    if not args.include_preview:
        rows = [ row for row in rows if not row.get( 'preview_mode' ) ]

    if args.last:
        rows = rows[-args.last:]

    if not rows:
        print( "no rows left after filtering. Drop --since/--last, or pass --include-preview if "
               "the run you are looking for was a preview run." )
        sys.exit( 1 )

    groups = {}
    for row in rows:
        groups.setdefault( group_key( row ), [] ).append( row )

    newest = max( ( row.get( 'timestamp' ) or 0 ) for row in rows )
    oldest = min( ( row.get( 'timestamp' ) or 0 ) for row in rows )
    print( "%d row(s) from %s across %d configuration(s), %s to %s"%(
        len( rows ), stats_path, len( groups ),
        datetime.datetime.fromtimestamp( oldest ).strftime( '%Y-%m-%d %H:%M' ),
        datetime.datetime.fromtimestamp( newest ).strftime( '%Y-%m-%d %H:%M' ) ) )
    print()

    for key in sorted( groups, key=describe_key ):
        print_group( key, groups[key] )

    if len( groups ) > 1:
        print_comparison( groups )

    print( "next steps, in the order that usually pays off:" )
    print( "  1. tools/bench/betabench.py all --yes      derive suppression/hysteresis/geometry/dedup" )
    print( "     values from these same caches (add --sizes N to read a size this backend is no" )
    print( "     longer configured for)" )
    print( "  2. tools/analysis/run_all_analysis.sh      tracking and style diagnostics per backend" )
    print( "  3. change ONE thing, re-run, and compare the groups above" )


if __name__ == '__main__':
    main()
