#!/usr/bin/env python3
"""
cleanup_output.py - free disk under ../output without losing anything a later decision needs.

Nearly all of the growth is A/B video: on 2026-10-23, 29 GB of a 29.5 GB
output folder was clip and whole-video renders, and 27.3 GB of that was
whole-video renders of two rounds already decided. Every one of them can
be made again from the detection caches. What cannot be made again
cheaply, or at all, is never touched:

  stats/                 every run's timings and settings, for comparing
                         runs across settings and versions
  cache/                 detection caches (about 25 h of GPU work for the
                         library), shot cuts, run keys, file hashes
  logs/analysis_runs/    every analysis hub run, the tuning evidence
  each round's record    ranking.json, clips.md, feedback.csv, chosen.json,
                         chosen_rules.json and its logs
  benchmark results      betabench's timestamped folders

What it removes, by category:

  decided_renders   the video files of A/B rounds that have a chosen.json:
                    the decision is written down
  stale_renders     the video files of A/B rounds with a ranking.json and
                    no decision, untouched for --stale-days (14). Offered
                    one round at a time; resume renders whatever is missing
  aborted_rounds    round folders with no ranking.json (a start that
                    never finished ranking), untouched for an hour and
                    holding nothing but logs
  log_rotation      logs/betasuite.log past --log-max-mb (50): compressed
                    to betasuite.log.1.gz (older copies shift up, the
                    newest --keep-logs (5) kept) and emptied in place.
                    Skipped while it changed in the last 10 minutes, since
                    a run is likely writing to it

Nothing is removed without --apply. tools/maintenance/cleanup_output.sh
shows the plan and asks before each category.

    python3 tools/maintenance/cleanup_output.py
    python3 tools/maintenance/cleanup_output.py --apply decided_renders
    python3 tools/maintenance/cleanup_output.py --apply stale_renders --only ab_suppression_20261019_092944
"""

import argparse
import gzip
import os
import shutil
import sys
import time

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.dirname(
    os.path.abspath( __file__ ) ) ) ) )

import betaconfig
import betaconst


CATEGORIES = ( 'decided_renders', 'stale_renders', 'aborted_rounds', 'log_rotation' )
ROUND_PREFIXES = ( 'ab_', 'suppression_ab_', '.ab_start_' )
VIDEO_EXTENSIONS = ( '.mp4', '.mkv', '.mov', '.avi', '.webm' )
# What a half-finished render leaves beside its output: the chunk
# concat list. Chunk and .tmp files are video files and match above.
RENDER_LEFTOVER_SUFFIXES = ( '.concat.txt', )
DECISION_FILES = ( 'chosen.json', 'chosen_rules.json' )
ROUND_RECORD_FILES = ( 'ranking.json', 'clips.md', 'feedback.csv' ) + DECISION_FILES
ABORTED_MIN_AGE_SECONDS = 3600
LOG_BUSY_SECONDS = 600


class Item:
    """One thing a category would remove: a round's renders, a round folder, or a log."""

    def __init__( self, category, name, path, files, size, why ):
        self.category = category
        self.name = name
        self.path = path
        self.files = files
        self.size = size
        self.why = why


def output_root():
    """The output folder, from betaconst's own benchmark path."""
    return os.path.dirname( os.path.abspath( betaconst.benchmark_dir ) )


def human( size ):
    for unit in ( 'B', 'KB', 'MB', 'GB', 'TB' ):
        if size < 1024 or unit == 'TB':
            return ( '%d %s'%( size, unit ) if unit == 'B' else '%.1f %s'%( size, unit ) )
        size /= 1024.0


def _inside( path, root ):
    real, base = os.path.realpath( path ), os.path.realpath( root )
    return real == base or real.startswith( base + os.sep )


def _files( folder ):
    """Every regular file under folder, never through a symlink."""
    found = []
    for directory, subdirectories, names in os.walk( folder, followlinks=False ):
        subdirectories[:] = [ name for name in subdirectories
                              if not os.path.islink( os.path.join( directory, name ) ) ]
        for name in names:
            path = os.path.join( directory, name )
            if not os.path.islink( path ) and os.path.isfile( path ):
                found.append( path )
    return sorted( found )


def is_render( path ):
    lowered = path.lower()
    return lowered.endswith( VIDEO_EXTENSIONS ) or lowered.endswith( RENDER_LEFTOVER_SUFFIXES )


def _newest_change( paths, folder ):
    times = [ os.path.getmtime( path ) for path in paths ] or [ os.path.getmtime( folder ) ]
    return max( times )


def round_items( benchmarks, now, stale_days ):
    """The decided_renders, stale_renders and aborted_rounds items under benchmarks/."""
    items = []
    if not os.path.isdir( benchmarks ):
        return items
    for name in sorted( os.listdir( benchmarks ) ):
        folder = os.path.join( benchmarks, name )
        if not name.startswith( ROUND_PREFIXES ) or os.path.islink( folder ) or not os.path.isdir( folder ):
            continue
        files = _files( folder )
        renders = [ path for path in files if is_render( path ) ]
        size = sum( os.path.getsize( path ) for path in renders )
        age = now - _newest_change( files, folder )
        has = lambda record: os.path.exists( os.path.join( folder, record ) )
        if any( has( record ) for record in DECISION_FILES ):
            if renders:
                items.append( Item( 'decided_renders', name, folder, renders, size,
                                    "decided; %d render(s)"%( len( renders ), ) ) )
        elif has( 'ranking.json' ):
            if renders and age >= stale_days * 86400:
                items.append( Item( 'stale_renders', name, folder, renders, size,
                                    "no decision, untouched %d day(s); %d render(s)"%(
                                        age // 86400, len( renders ) ) ) )
        elif not renders and not any( has( record ) for record in ROUND_RECORD_FILES ):
            if age >= ABORTED_MIN_AGE_SECONDS:
                items.append( Item( 'aborted_rounds', name, folder, files,
                                    sum( os.path.getsize( path ) for path in files ),
                                    "never finished ranking; %d log file(s)"%( len( files ), ) ) )
    return items


def log_item( log_path, now, max_mb ):
    """The log_rotation item for log_path, or None."""
    if not log_path or not os.path.isfile( log_path ) or os.path.islink( log_path ):
        return None
    size = os.path.getsize( log_path )
    if size < max_mb * 1024 * 1024:
        return None
    if now - os.path.getmtime( log_path ) < LOG_BUSY_SECONDS:
        return Item( 'log_rotation', os.path.basename( log_path ), log_path, [], 0,
                     "past %d MB but written in the last %d minutes; left for a quieter moment"%(
                         max_mb, LOG_BUSY_SECONDS // 60 ) )
    return Item( 'log_rotation', os.path.basename( log_path ), log_path, [ log_path ], size,
                 "past %d MB; compressed and emptied"%( max_mb, ) )


def plan( root, now=None, stale_days=14, log_path=None, log_max_mb=50 ):
    """Every item each category would remove, nothing removed."""
    now = time.time() if now is None else now
    items = round_items( os.path.join( root, 'benchmarks' ), now, stale_days )
    item = log_item( log_path, now, log_max_mb )
    if item:
        items.append( item )
    return items


def rotate_log( log_path, keep ):
    """
    Compress log_path to .1.gz (shifting older copies up, keeping `keep`)
    and empty it in place. Emptied rather than replaced: a run holding it
    open in append mode carries on writing to the same file.
    """
    for number in range( keep, 0, -1 ):
        older = '%s.%d.gz'%( log_path, number )
        if not os.path.exists( older ):
            continue
        if number == keep:
            os.remove( older )
        else:
            os.replace( older, '%s.%d.gz'%( log_path, number + 1 ) )
    with open( log_path, 'rb' ) as source, gzip.open( '%s.1.gz'%( log_path, ), 'wb' ) as target:
        shutil.copyfileobj( source, target )
    with open( log_path, 'r+b' ) as handle:
        handle.truncate( 0 )


def apply( items, root, category, only=None, keep_logs=5, out=print ):
    """Remove one category's items (only: one item by name). Returns bytes freed."""
    freed = 0
    for item in items:
        if item.category != category or ( only and item.name != only ) or not item.files:
            continue
        if not _inside( item.path, root ):
            out( "WARN refusing %s: outside %s"%( item.path, root ) )
            continue
        if category == 'log_rotation':
            rotate_log( item.path, keep_logs )
            out( "INFO rotated %s (%s) into %s.1.gz"%( item.path, human( item.size ), item.path ) )
            freed += item.size
            continue
        if category == 'aborted_rounds':
            shutil.rmtree( item.path )
            out( "INFO removed %s (%s)"%( item.path, human( item.size ) ) )
            freed += item.size
            continue
        for path in item.files:
            if not _inside( path, item.path ) or os.path.islink( path ):
                out( "WARN refusing %s: not a file inside %s"%( path, item.path ) )
                continue
            size = os.path.getsize( path )
            os.remove( path )
            freed += size
            out( "DEBUG removed %s (%s)"%( path, human( size ) ) )
        out( "INFO %s: removed %d render(s), %s; its record stays"%(
            item.name, len( item.files ), human( item.size ) ) )
    return freed


def print_plan( items, out=print ):
    out( "INFO %-16s %-46s %10s  %s"%( 'category', 'item', 'size', 'why' ) )
    for category in CATEGORIES:
        for item in items:
            if item.category == category:
                out( "INFO %-16s %-46s %10s  %s"%( category, item.name[ :46 ], human( item.size ), item.why ) )
    total = sum( item.size for item in items )
    out( "INFO total that could be freed: %s. Never touched: stats, caches, analysis runs, every "
         "round's ranking/clips/feedback/decision files and logs"%( human( total ), ) )


def build_parser():
    parser = argparse.ArgumentParser( description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter )
    parser.add_argument( '--apply', choices=CATEGORIES, default=None,
        help="remove this category's items (default: print the plan only)" )
    parser.add_argument( '--only', default=None, metavar='NAME',
        help="with --apply: just the item of this name (a round folder's name)" )
    parser.add_argument( '--summary', choices=CATEGORIES, default=None,
        help="print '<count> <bytes>' for one category and stop" )
    parser.add_argument( '--list', choices=CATEGORIES, default=None, dest='list_category',
        help="print '<name>\\t<size>\\t<why>' per item of one category and stop" )
    parser.add_argument( '--stale-days', type=float, default=14,
        help="an undecided round counts as stale after this long untouched (default: %(default)s)" )
    parser.add_argument( '--log-max-mb', type=float, default=50,
        help="rotate betasuite.log past this size (default: %(default)s)" )
    parser.add_argument( '--keep-logs', type=int, default=5,
        help="compressed betasuite.log copies kept (default: %(default)s)" )
    parser.add_argument( '--root', default=None, help="the output folder (default: ../output)" )
    return parser


def main( argv=None, now=None, out=print ):
    args = build_parser().parse_args( argv )
    root = os.path.abspath( args.root or output_root() )
    if args.root:
        log_path = os.path.join( root, 'logs', 'betasuite.log' )
    else:
        log_path = os.path.abspath( getattr( betaconfig, 'log_path', None )
                                    or os.path.join( root, 'logs', 'betasuite.log' ) )
    items = plan( root, now=now, stale_days=args.stale_days, log_path=log_path,
                  log_max_mb=args.log_max_mb )
    if args.summary:
        chosen = [ item for item in items if item.category == args.summary and item.files ]
        if args.only:
            chosen = [ item for item in chosen if item.name == args.only ]
        out( "%d %d"%( len( chosen ), sum( item.size for item in chosen ) ) )
        return 0
    if args.list_category:
        for item in items:
            if item.category == args.list_category and item.files:
                out( "%s\t%s\t%s"%( item.name, human( item.size ), item.why ) )
        return 0
    if not args.apply:
        print_plan( items, out )
        return 0
    freed = apply( items, root, args.apply, only=args.only, keep_logs=args.keep_logs, out=out )
    out( "INFO freed %s (%s%s)"%( human( freed ), args.apply,
                                  ", %s"%( args.only, ) if args.only else "" ) )
    return 0


if __name__ == '__main__':
    raise SystemExit( main() )
