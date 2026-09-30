#!/usr/bin/env python3
"""
summarize_tune_sweep.py - reads betaconfig.stats_path (default
../output/stats/betasuite_stats.jsonl, one JSON object per processed file - see
betautils_log.write_stats) and prints a table comparing runs grouped by
the tuning config that produced them: (picture_sizes, video_censor_fps,
nn_batch_size). Meant to be run after tune_sweep.sh, but works on any
stats file with these fields present (older stats lines from before this
was added won't have picture_sizes/video_censor_fps/nn_batch_size and are
skipped, with a note).

Usage:
    python3 summarize_tune_sweep.py
    python3 summarize_tune_sweep.py --preview-offset 30.0   # only rows from that pinned slice
    python3 summarize_tune_sweep.py --since 1700000000       # only rows with timestamp >= this
    python3 summarize_tune_sweep.py --stats-path /path/to/betasuite_stats.jsonl
"""

import sys as _sys, os as _os
_sys.path.insert( 0, _os.path.dirname( _os.path.dirname( _os.path.dirname( _os.path.abspath( __file__ ) ) ) ) )  # so 'import betaconst'/'betaconfig'/etc still resolve after this script moved into tools/analysis or tools/tuning

import argparse
import json
import sys
from collections import defaultdict

def load_betaconfig_stats_path():
    try:
        import betaconfig
        return getattr( betaconfig, 'stats_path', '../output/stats/betasuite_stats.jsonl' )
    except Exception:
        return '../output/stats/betasuite_stats.jsonl'

def main():
    parser = argparse.ArgumentParser( description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter )
    parser.add_argument( '--stats-path', default=None, help="override the stats jsonl path (default: betaconfig.stats_path)" )
    parser.add_argument( '--preview-offset', type=float, default=None, help="only include rows whose preview_offset_seconds matches this (use the same value you passed tune_sweep.sh, so you're not accidentally comparing different slices)" )
    parser.add_argument( '--since', type=float, default=None, help="only include rows with timestamp >= this unix time (use to exclude runs from before your current sweep)" )
    args = parser.parse_args()

    stats_path = args.stats_path or load_betaconfig_stats_path()

    try:
        with open( stats_path, 'r' ) as f:
            lines = f.readlines()
    except FileNotFoundError:
        print( "no stats file found at %r - run tune_sweep.sh (or betatv.py with stats_enabled=True) first"%(stats_path) )
        sys.exit(1)

    skipped_old = 0
    skipped_filtered = 0
    groups = defaultdict( list )

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads( line )
        except json.JSONDecodeError:
            continue

        if 'picture_sizes' not in rec or 'video_censor_fps' not in rec or 'nn_batch_size' not in rec:
            skipped_old += 1
            continue

        if args.preview_offset is not None and rec.get( 'preview_offset_seconds' ) != args.preview_offset:
            skipped_filtered += 1
            continue
        if args.since is not None and rec.get( 'timestamp', 0 ) < args.since:
            skipped_filtered += 1
            continue

        key = ( tuple( rec['picture_sizes'] ), rec['video_censor_fps'], rec['nn_batch_size'] )
        groups[key].append( rec )

    if not groups:
        print( "no matching stats rows found in %r"%(stats_path) )
        if skipped_old:
            print( "(%d rows skipped - written before picture_sizes/video_censor_fps/nn_batch_size were recorded)"%(skipped_old) )
        if skipped_filtered:
            print( "(%d rows skipped - didn't match --preview-offset/--since filters)"%(skipped_filtered) )
        sys.exit(0)

    header = "%-22s %8s %10s | %8s %8s %8s | %10s %6s %8s"%(
        "picture_sizes", "fps", "batch", "n_files", "avg_det_s", "avg_enc_s", "total_s", "labels", "interp" )
    print( header )
    print( "-" * len(header) )

    for key in sorted( groups.keys() ):
        sizes, fps, batch = key
        recs = groups[key]
        n = len( recs )
        avg_det = sum( r['detection_seconds'] for r in recs ) / n
        avg_enc = sum( r['encode_seconds'] for r in recs ) / n
        total_s = sum( r['total_seconds'] for r in recs )
        total_labels = sum( sum( r.get('label_counts', {}).values() ) for r in recs )
        total_interp = sum( r.get('interpolated_boxes', 0) for r in recs )
        sizes_str = "+".join( str(s) for s in sizes )
        print( "%-22s %8s %10s | %8d %8.2f %8.2f | %10.2f %6d %8d"%(
            sizes_str, fps, batch, n, avg_det, avg_enc, total_s, total_labels, total_interp ) )

    print()
    print( "avg_det_s/avg_enc_s are per-file averages within each group; total_s is summed across all files in the group." )
    print( "labels = total detections kept across all files/frames in the group (higher isn't automatically better - check it's catching real things, not noise, by spot-checking the actual censored preview output)." )
    if skipped_old:
        print( "(%d older stats rows skipped - missing picture_sizes/video_censor_fps/nn_batch_size)"%(skipped_old) )
    if skipped_filtered:
        print( "(%d rows skipped by --preview-offset/--since filters)"%(skipped_filtered) )

if __name__ == '__main__':
    main()
