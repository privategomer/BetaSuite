#!/usr/bin/env python3
"""
analyze_score_distribution.py - reports the raw confidence-score
distribution of cached detections, per label, per detector backend -
built specifically to ground a per-backend min_prob decision in real
data rather than a guess, the same way analyze_suppression_pairs.py
grounds class_suppression's min_iou/margin.

Why this didn't already exist: analyze_suppression_pairs.py's box-size
stats section reports w/h/area per label, but nothing in this repo
reports the SCORE distribution of accepted detections per label per
backend - which is exactly what you need to reason about "is this
backend's min_prob for this label too permissive/strict" once you
suspect two backends need genuinely different thresholds (see
betaconfig.detector_backend's own module comment: nudenet_v3 and
retinanet_v2 produce structurally different output, so a value tuned
against one model's data isn't assumed correct for the other's).

This tool does NOT tell you the "right" min_prob - there's no ground-
truth false-positive labeling anywhere in this codebase yet (a
detection's score alone doesn't say whether it was a real detection or
a false positive), so nothing here can compute an actual false-positive
rate at a given threshold. What it DOES give you: the full score
histogram and percentile breakdown for a label under each backend that
has real cache for it, side by side, so you can reason about relative
permissiveness - e.g. "backend A's median score for this label is X,
backend B's is Y" and "N% of backend A's accepted detections for this
label would fall below backend B's current min_prob". That comparison
is the actual grounding for "set nudenet_v3's threshold to roughly
match retinanet_v2's effective acceptance behavior" - it's a relative,
data-driven starting point for you to review, not a fully automated
answer.

Run from inside BetaSuite-env/ (same place you run betatv.py from):

    python3 tools/analysis/analyze_score_distribution.py
    python3 tools/analysis/analyze_score_distribution.py --labels exposed_vulva exposed_breast
    python3 tools/analysis/analyze_score_distribution.py --include-preview

Reads only the cache (vid_hashes/pic_hashes); never touches
betaconfig.py or runs the neural net.
"""

import argparse
import glob
import gzip
import json
import os
import statistics
import sys

import sys as _sys, os as _os
_sys.path.insert( 0, _os.path.dirname( _os.path.dirname( _os.path.dirname( _os.path.abspath( __file__ ) ) ) ) )  # so 'import betaconst'/'betaconfig'/etc still resolve after this script moved into tools/analysis or tools/tuning

import betaconst
import betaconfig
import betautils_detector as bu_detector
import betautils_cache_paths as bu_cache

# One source of truth for every cache path and filename - see
# betautils_cache_paths.py's docstring for why these are not literals.
VID_HASH_DIR = bu_cache.VID_HASH_DIR
PIC_HASH_DIR = bu_cache.PIC_HASH_DIR

PERCENTILES = ( 5, 10, 25, 50, 75, 90, 95 )


# Parsed in exactly one place - see betautils_cache_paths.py.
backend_of_cache_filename = bu_cache.backend_of_cache_filename
is_preview_cache_filename = bu_cache.is_preview_cache_filename


def load_cache_dir( path, include_preview ):
    """
    Yield (filename, raw_boxes) for every cache file in path.

    include_preview matters more here than elsewhere: a score
    distribution is exactly the kind of statistic a short,
    boundary-truncated preview slice skews, because its frames are both
    fewer and more alike than a whole video's.
    """
    return bu_cache.iter_cache_files( path, include_preview=include_preview )


def percentiles_of( values, pcts ):
    """
    Returns {pct: value} for each pct in pcts, using
    statistics.quantiles-equivalent linear interpolation (no numpy
    dependency assumed elsewhere in this repo's analysis tools).
    """
    if not values:
        return { p: None for p in pcts }
    ordered = sorted( values )
    n = len( ordered )
    result = {}
    for p in pcts:
        if n == 1:
            result[p] = ordered[0]
            continue
        rank = ( p / 100.0 ) * ( n - 1 )
        lo = int( rank )
        hi = min( lo + 1, n - 1 )
        frac = rank - lo
        result[p] = ordered[lo] + ( ordered[hi] - ordered[lo] ) * frac
    return result


def histogram_buckets( values, bucket_width=0.05, lo=0.0, hi=1.0 ):
    """
    Fixed-width histogram from lo to hi (default: the full valid score
    range in 0.05-wide buckets) - returns an ordered list of
    (bucket_lo, bucket_hi, count) tuples covering every bucket in range,
    including empty ones, so the printed histogram's shape is never
    misleading by omission.
    """
    n_buckets = int( round( (hi - lo) / bucket_width ) )
    counts = [0] * n_buckets
    for v in values:
        idx = int( (v - lo) / bucket_width )
        if idx >= n_buckets:
            idx = n_buckets - 1  # v == hi exactly
        if idx < 0:
            idx = 0
        counts[idx] += 1
    return [ ( lo + i*bucket_width, lo + (i+1)*bucket_width, counts[i] ) for i in range(n_buckets) ]


def print_histogram( values, bucket_width=0.05, bar_width=50 ):
    buckets = histogram_buckets( values, bucket_width=bucket_width )
    max_count = max( (c for _, _, c in buckets), default=0 )
    if max_count == 0:
        print( "    (no values)" )
        return
    for blo, bhi, count in buckets:
        bar_len = int( round( bar_width * count / max_count ) ) if max_count else 0
        bar = '#' * bar_len
        print( "    %.2f-%.2f: %5d %s"%(blo, bhi, count, bar) )


def fraction_below( values, threshold ):
    if not values:
        return None
    return sum( 1 for v in values if v < threshold ) / len( values )


def print_report_for_backend( backend_name, scores_by_label, labels_filter, thresholds_by_backend_label, bucket_width=0.05 ):
    """
    Args:
        backend_name: Which backend this section is reporting on.
        scores_by_label: {label: [score, ...]} for THIS backend's cache.
        labels_filter: Optional set of labels to restrict to, or None
            for every label with cache.
        thresholds_by_backend_label: {backend_name: {label: min_prob}}
            across EVERY backend found (not just this one) - built once
            in main() - so this can show, per label, what fraction of
            this backend's own cached scores would fall below every
            OTHER backend's current min_prob for that same label.
    """
    labels = sorted( scores_by_label.keys() )
    if labels_filter:
        labels = [ l for l in labels if l in labels_filter ]

    for label in labels:
        scores = scores_by_label[label]
        if not scores:
            continue
        print( "  %s  (n=%d)"%(label, len(scores)) )
        pcts = percentiles_of( scores, PERCENTILES )
        pct_str = "  ".join( "p%d=%.3f"%(p, pcts[p]) for p in PERCENTILES )
        print( "    %s  min=%.3f  max=%.3f  mean=%.3f"%(
            pct_str, min(scores), max(scores), statistics.mean(scores) ) )
        print_histogram( scores, bucket_width=bucket_width )

        current_min_prob = thresholds_by_backend_label.get( backend_name, {} ).get( label )
        if current_min_prob is not None:
            frac = fraction_below( scores, current_min_prob )
            print( "    fraction of cached detections that would be rejected by this backend's CURRENT min_prob "
                   "(%.3f): %.1f%% (informational only - most of this cache was already filtered at detection "
                   "time by whatever min_prob was active when it was produced, so this undercounts true rejection "
                   "rate; it's a floor, not the real fraction)"%(current_min_prob, frac*100 if frac is not None else 0.0) )

        for other_backend, other_label_thresholds in sorted( thresholds_by_backend_label.items() ):
            if other_backend == backend_name:
                continue
            other_threshold = other_label_thresholds.get( label )
            if other_threshold is None:
                continue
            frac = fraction_below( scores, other_threshold )
            if frac is None:
                continue
            print( "    if this backend used %s's current min_prob for this label (%.3f): %.1f%% of this "
                   "backend's cached detections would fall below it"%(other_backend, other_threshold, frac*100) )
        print()


def main():
    parser = argparse.ArgumentParser( description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter )
    parser.add_argument( '--labels', nargs='+', default=None,
        help="restrict the report to these labels (default: every label with cached detections)" )
    parser.add_argument( '--include-preview', action='store_true',
        help="also include preview-slice cache (default: real/full-run cache only - see this tool's own docstring "
             "for why score distribution is sensitive to preview truncation)" )
    parser.add_argument( '--bucket-width', type=float, default=0.05,
        help="histogram bucket width (default 0.05)" )
    args = parser.parse_args()

    labels_filter = set( args.labels ) if args.labels else None

    sources = []
    for name, boxes in load_cache_dir( VID_HASH_DIR, args.include_preview ):
        sources.append( (name, boxes) )
    for name, boxes in load_cache_dir( PIC_HASH_DIR, args.include_preview ):
        sources.append( (name, boxes) )

    if not sources:
        print( "No%s cache files found in %s or %s - run betatv.py/betastare.py on some footage first."%(
            "" if args.include_preview else " real (non-preview)", VID_HASH_DIR, PIC_HASH_DIR ) )
        if not args.include_preview:
            print( "(pass --include-preview to also check preview-slice cache)" )
        return

    # Grouped by CONFIGURATION (backend + size + detection key) rather
    # than by backend name. A score distribution is a property of one
    # set of model weights; pooling a 320n run and a 640m run under the
    # heading "nudenet_v3" produces percentiles that belong to neither,
    # and those percentiles are exactly what min_prob gets derived from.
    scores_by_configuration_label = {}
    backend_of_configuration = {}
    for name, boxes in sources:
        configuration = bu_cache.configuration_label_of_cache_filename( name )
        backend_of_configuration[configuration] = backend_of_cache_filename( name )
        scores_by_configuration_label.setdefault( configuration, {} )
        for b in boxes:
            label = b['class_id']
            scores_by_configuration_label[configuration].setdefault( label, [] ).append( b['score'] )

    configurations_found = sorted( scores_by_configuration_label.keys() )
    print( "Loaded %d cache file(s) across %d configuration(s): %s\n"%(
        len(sources), len(configurations_found), ", ".join(configurations_found) ) )

    # The current min_prob for every (configuration, label) pair with
    # cache, so each section can show "what would happen at another
    # configuration's current threshold" as a concrete comparison point
    # rather than an abstract percentile. Resolved against the
    # configuration's BACKEND, since min_prob is a per-backend override.
    compare_thresholds = {}
    for configuration in configurations_found:
        backend = backend_of_configuration[configuration]
        compare_thresholds[configuration] = {}
        for label in scores_by_configuration_label[configuration].keys():
            mp = bu_detector.get_item_overrides( label, backend ).get( 'min_prob' )
            if mp is not None:
                compare_thresholds[configuration][label] = mp

    for configuration in configurations_found:
        header = "############################################################\n# configuration: %s\n############################################################"%(configuration)
        print( header )
        print_report_for_backend(
            configuration, scores_by_configuration_label[configuration], labels_filter,
            thresholds_by_backend_label=compare_thresholds, bucket_width=args.bucket_width )
        print()

    if len( configurations_found ) > 1:
        print( "Reminder: a lower median score for a label under one configuration isn't automatically 'worse' - different" )
        print( "model architectures, and different variants of one model, produce genuinely different confidence" )
        print( "calibration, not just noisier versions of the same signal. So a min_prob derived from one variant's" )
        print( "percentiles does not transfer to another variant. What's actionable: if you believe a label's min_prob" )
        print( "is letting through false positives you don't see under the other, compare where each backend's REAL" )
        print( "detections (that you know are correct, e.g. from watching output) sit in their own histogram above -" )
        print( "this tool can't tell you which detections were false positives, only where the whole distribution sits." )


if __name__ == '__main__':
    main()
