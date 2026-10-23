#!/usr/bin/env python3
"""
analyze_label_content.py - which videos show which censored label, and which mix them.

Tuning on a library that is mostly one label tunes for that label: the
2026-10 suppression A/B picked its videos by how much the rule sets
differ, and breast-heavy compilations dominate that number. This lists,
from the detection caches alone (no replay, no video decode):

  - per censored label, the videos with the most seconds of it
  - the videos with the most MIXED seconds, where two censored labels
    are on screen at once: where one label's suppression rule can take
    another's censor away, so a cross-label rule is judged there
  - per mixed video, which pairs make up the mix
  - near-duplicate videos: the same footage twice (a re-encode, another
    container, a trim), by runtime and labelled seconds, which every
    bench, auto_tune and A/B ranking otherwise counts twice

A second counts for a label when a detection of it scores at least
--confident-score (0.5) in that sample. Seconds are samples over the
cache's own rate, so quick-cut videos at 18 fps compare with the rest.

Use the names here with the A/B tool's focus option, e.g.
AB_FOCUS=exposed_vulva bash tools/tuning/ab_compare.sh, which ranks
videos by how much that label's censoring differs.

    python3 tools/analysis/analyze_label_content.py
    python3 tools/analysis/analyze_label_content.py --top 15 --json content.json
"""

import argparse
import json
import os
import sys

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.dirname(
    os.path.abspath( __file__ ) ) ) ) )

import betaconfig
import betaconst
import betautils_cache_paths as bu_cache
import betautils_tuning as bu_tuning
import betautils_video as bu_video


def content_rows( content, names, durations=None ):
    """One row per video: name, per-label seconds, mixed seconds, pairs, and share of runtime."""
    rows = []
    for file_hash, found in content.items():
        duration = ( durations or {} ).get( file_hash )
        rows.append( {
            'file_hash': file_hash,
            'name': names.get( file_hash, file_hash ),
            'seconds': found['seconds'],
            'mixed_seconds': found['mixed_seconds'],
            'pair_seconds': found['pair_seconds'],
            'duration_seconds': duration,
        } )
    return rows


def mark_duplicates( rows ):
    """
    [(row, row)] for each near-duplicate pair (bu_tuning.near_duplicates),
    and every row's 'near_duplicate_of': the names of its copies.
    """
    by_hash = { row['file_hash']: row for row in rows }
    signatures = { row['file_hash']: bu_tuning.content_signature( row['duration_seconds'], row )
                   for row in rows }
    pairs = [ ( by_hash[first], by_hash[second] )
              for first, second in bu_tuning.near_duplicates( signatures ) ]
    for row in rows:
        row['near_duplicate_of'] = []
    for first, second in pairs:
        first['near_duplicate_of'].append( second['name'] )
        second['near_duplicate_of'].append( first['name'] )
    return pairs


def print_duplicates( pairs, out=print ):
    out( "" )
    out( "Near-duplicate videos (runtime and every labelled-seconds figure agree: the same "
         "footage twice):" )
    if not pairs:
        out( "  none" )
        return
    for first, second in pairs:
        out( "  %s"%( first['name'], ) )
        out( "    = %s  (%.0fs and %.0fs)"%( second['name'], first['duration_seconds'],
                                             second['duration_seconds'] ) )
    hours = sum( min( first['duration_seconds'], second['duration_seconds'] )
                 for first, second in pairs ) / 3600.0
    out( "  %d pair(s), %.1f h of footage counted twice by every bench, auto_tune and A/B "
         "ranking, and processed twice by every full run. The A/B ranking already skips a "
         "copy of a video it picked; setting one of each aside is your call."%( len( pairs ), hours ) )


def top_by( rows, key, count ):
    """The `count` rows with the most of `key` ('mixed' or a label), ties by name; zeros left out."""
    def amount( row ):
        return row['mixed_seconds'] if key == 'mixed' else row['seconds'].get( key, 0.0 )
    ranked = sorted( ( row for row in rows if amount( row ) > 0 ),
                     key=lambda row: ( -amount( row ), row['name'] ) )
    return ranked[ :count ]


def leaning( row, label ):
    """label's share of the row's labelled seconds, 0.0 when it has none."""
    total = sum( row['seconds'].values() )
    return row['seconds'].get( label, 0.0 ) / total if total else 0.0


def top_leaning( rows, label, count, min_seconds=60.0 ):
    """
    The rows where `label` makes up the largest share of the labelled
    seconds, among those with at least min_seconds of it. The seconds
    ranking favours long videos; this one finds the short videos that
    are mostly one label.
    """
    ranked = sorted( ( row for row in rows if row['seconds'].get( label, 0.0 ) >= min_seconds ),
                     key=lambda row: ( -leaning( row, label ), row['name'] ) )
    return ranked[ :count ]


def _share( seconds, duration ):
    return ( "%5.1f%%"%( 100.0 * seconds / duration, ) ) if duration else "     -"


def print_report( rows, labels, count, out=print ):
    short = { label: label.replace( 'exposed_', '' ) for label in labels }
    header = ( "  %-46s "%( 'video', ) + " ".join( "%9s"%( short[label][ :9 ], ) for label in labels )
               + " %9s %7s"%( 'mixed', 'share' ) )
    for key in list( labels ) + [ 'mixed' ]:
        out( "" )
        if key == 'mixed':
            out( "Most MIXED seconds (two or more censored labels on screen at once):" )
        else:
            out( "Most %s seconds:"%( key, ) )
        ranked = top_by( rows, key, count )
        if not ranked:
            out( "  none" )
            continue
        out( header )
        for row in ranked:
            amount = row['mixed_seconds'] if key == 'mixed' else row['seconds'].get( key, 0.0 )
            out( "  %-46s "%( row['name'][ :46 ], )
                 + " ".join( "%8.0fs"%( row['seconds'].get( label, 0.0 ), ) for label in labels )
                 + " %8.0fs %7s"%( row['mixed_seconds'], _share( amount, row['duration_seconds'] ) ) )
            if key == 'mixed':
                pairs = sorted( ( ( seconds, pair ) for pair, seconds in row['pair_seconds'].items()
                                  if seconds > 0 ), reverse=True )
                out( "      " + ", ".join( "%s %.0fs"%( pair.replace( 'exposed_', '' ), seconds )
                                           for seconds, pair in pairs ) )
    if len( labels ) > 1:
        for label in labels:
            out( "" )
            out( "Most %s-leaning (its share of the labelled seconds, at least 60s of it):"%(
                label, ) )
            ranked = top_leaning( rows, label, count )
            if not ranked:
                out( "  none" )
                continue
            for row in ranked:
                out( "  %-46s %5.1f%%  (%s)"%( row['name'][ :46 ], 100.0 * leaning( row, label ),
                                             ", ".join( "%s %.0fs"%( short[other], row['seconds'].get( other, 0.0 ) )
                                                        for other in labels ) ) )
    out( "" )
    out( "seconds: samples with a detection scoring at least the confident score, over the "
         "cache's sample rate. share: the ranked column over the video's runtime. Mixed counts "
         "two labels anywhere in the frame, so split-screen panels count as mixed." )


def build_parser():
    parser = argparse.ArgumentParser( description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter )
    parser.add_argument( '--labels', nargs='+', default=None,
        help="labels to measure (default: items_to_censor)" )
    parser.add_argument( '--top', type=int, default=10,
        help="videos listed per ranking (default: %(default)s)" )
    parser.add_argument( '--confident-score', type=float, default=bu_tuning.CONFIDENT_SCORE,
        help="a detection at or above this score counts (default: %(default)s)" )
    parser.add_argument( '--backends', nargs='+', default=None,
        help="backends to analyse (default: every one with caches)" )
    parser.add_argument( '--variants', nargs='+', default=None, dest='model_variants',
        help="model variants to analyse (default: every one with caches)" )
    parser.add_argument( '--json', default=None, metavar='FILE',
        help="also write every video's numbers here" )
    return parser


def main( argv=None ):
    args = build_parser().parse_args( argv )
    labels = sorted( args.labels or betaconfig.items_to_censor )
    min_prob = getattr( betaconfig, 'global_min_prob', betaconst.global_min_prob )
    configurations = bu_cache.configurations_to_analyse(
        None, backend_names=args.backends, variant_names=args.model_variants,
        min_prob=min_prob, include_preview=False )
    if not configurations:
        print( "no detection caches found: run betatv.py on real footage first" )
        return 1
    report = {}
    for config in configurations:
        print( "" )
        print( "#" * 72 )
        print( "# %s"%( config.label, ) )
        print( "#" * 72 )
        videos, rates = bu_cache.videos_across_sample_rates( config, min_prob )
        print( "# %d video(s), sample rates %s, labels %s"%(
            len( videos ), ", ".join( '%g'%( rate, ) for rate in rates ) or 'none',
            ", ".join( labels ) ) )
        hash_to_path = bu_cache.build_hash_to_video_path( videos.keys() )
        names = { file_hash: os.path.basename( path ) for file_hash, path in hash_to_path.items() }
        durations = { file_hash: bu_video.probe_duration_seconds( path )
                      for file_hash, path in hash_to_path.items() }
        content = bu_tuning.content_by_video( videos, labels, args.confident_score )
        rows = content_rows( content, names, durations )
        print_report( rows, labels, args.top )
        print_duplicates( mark_duplicates( rows ) )
        report[config.label] = rows
    if args.json:
        with open( args.json, 'w', encoding='UTF-8' ) as handle:
            json.dump( report, handle, indent=2 )
        print( "" )
        print( "every video's numbers: %s"%( args.json, ) )
    return 0


if __name__ == '__main__':
    raise SystemExit( main() )
