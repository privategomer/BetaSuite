#!/usr/bin/env python3
"""
analyze_geometry_impact.py - what a candidate geometry bound would
actually DO to your real cached detections, per label, per candidate
value, before you put it in betaconfig.

WHY THIS EXISTS
---------------
Every other tunable in this repo has an analysis tool that answers "what
would this value do to real data": analyze_suppression_pairs for
min_iou/margin, analyze_score_distribution for min_prob,
analyze_track_breaks for match_distance and track_max_gap. Geometry had
none, and that gap has a specific consequence in this project's history.

betabench's geometry section prints a suggested max_area_fraction from
p99 x 1.25 and calls them "starting points". Nothing measured what
happened next. What happened next, on real footage, was that the
suggested cap for exposed_breast (0.123238) sat BELOW the largest real
detections of that label by a factor of 3.9 - and because a violated
bound used to mean "drop the detection", those close-ups rendered with no
censor at all. Geometry was switched off and stayed off for months.

The lesson is not that the suggestion was wrong; p99 x1.25 is a
reasonable place to start. The lesson is that "starting point" is only
safe when something can tell you the cost of starting there.

WHAT IT REPORTS
---------------
For each censored label, and for each candidate bound, how many real
cached detections violate it - and critically, WHERE in the size
distribution those violations sit, because that is what distinguishes
"this bound catches the misfires" from "this bound also catches every
close-up".

It also reports what the violations COST under each geometry_action:

    drop    that many detections lose their censor entirely
    clamp   that many censors shrink, by this much on average
    flag    nothing changes; the count is the measurement

WHAT IT CANNOT TELL YOU
-----------------------
Which of the violating detections were misfires and which were correct
close-ups. There is no ground-truth labelling anywhere in this codebase,
so no tool here can compute a real false-positive rate. What this gives
you is the SHAPE of the tradeoff - how many detections a bound touches
and how far they sit from it - which is what you need to decide whether
a bound is discriminating or just blunt.

The honest reading: a bound whose violations cluster far above it is
probably catching misfires. A bound whose violations run continuously
from just-above to far-above is cutting through the middle of your real
distribution, and no action setting makes that a good bound.
"""

import argparse
import os
import sys

sys.path.insert( 0, os.path.dirname( os.path.dirname(
    os.path.dirname( os.path.abspath( __file__ ) ) ) ) )

import betaconfig

import betautils_cache_paths as bu_cache
import betautils_config as bu_config
import betautils_detector as bu_detector
import betautils_log as bu_log
import betautils_track as bu_track


VID_HASH_DIR = bu_cache.VID_HASH_DIR
PIC_HASH_DIR = bu_cache.PIC_HASH_DIR


def percentile( ordered, fraction ):
    """Linear-interpolation-free percentile; ordered must be sorted."""
    if not ordered:
        return 0.0
    index = min( len( ordered ) - 1,
                 max( 0, int( round( fraction * ( len( ordered ) - 1 ) ) ) ) )
    return ordered[index]


def frame_size_for( boxes ):
    """
    The frame the area fraction is measured against.

    Cached boxes are already in SOURCE pixel space (a 640-size cache of an
    854x480 video carries boxes up to 854x480), so the frame is the
    largest extent any box reaches. Guessing a fixed 1920x1080 here would
    silently rescale every fraction on any other resolution, which is the
    same class of error this tool exists to catch.
    """
    width = max( ( box['x'] + box['w'] for box in boxes ), default=0 )
    height = max( ( box['y'] + box['h'] for box in boxes ), default=0 )
    return max( 1.0, float( width ) ), max( 1.0, float( height ) )


def collect( include_preview, labels_filter ):
    """{configuration: {label: [ (area_fraction, aspect, w, h) ]}}"""
    per_configuration = {}
    sources = list( bu_cache.iter_cache_files( VID_HASH_DIR, include_preview ) )
    sources += list( bu_cache.iter_cache_files( PIC_HASH_DIR, include_preview ) )
    for name, boxes in sources:
        if not boxes:
            continue
        configuration = bu_cache.configuration_label_of_cache_filename( name )
        frame_w, frame_h = frame_size_for( boxes )
        frame_area = frame_w * frame_h
        bucket = per_configuration.setdefault( configuration, {} )
        for box in boxes:
            label = box['class_id']
            if labels_filter and label not in labels_filter:
                continue
            if not box['h']:
                continue
            bucket.setdefault( label, [] ).append( (
                ( box['w'] * box['h'] ) / frame_area,
                box['w'] / box['h'],
                box['w'], box['h'] ) )
    return per_configuration, len( sources )


def report_bound( logger, label, rows, bound_name, values, candidates ):
    """One bound, several candidate values, against the real distribution."""
    ordered = sorted( values )
    n = len( ordered )
    logger.info( "" )
    logger.info( "  %s / %s  (n=%d)"%( label, bound_name, n ) )
    logger.info( "    observed: p50=%.5f p90=%.5f p99=%.5f max=%.5f"%(
        percentile( ordered, 0.50 ), percentile( ordered, 0.90 ),
        percentile( ordered, 0.99 ), ordered[-1] ) )

    is_max = bound_name.startswith( 'max' )
    logger.info( "    %-12s %9s %8s  %-38s"%(
        'candidate', 'violates', 'of all', 'where the violations sit' ) )
    for candidate in candidates:
        if is_max:
            bad = [ value for value in ordered if value > candidate ]
        else:
            bad = [ value for value in ordered if value < candidate ]
        if not bad:
            logger.info( "    %-12.5f %9d %7.2f%%  -"%( candidate, 0, 0.0 ) )
            continue
        bad.sort()
        # How far past the bound the violations reach. A tight cluster
        # just past the bound means the bound is cutting the middle of
        # the real distribution; a distant cluster means it is catching
        # outliers.
        ratios = [ ( value / candidate ) if candidate else 0.0 for value in bad ]
        ratios.sort()
        logger.info( "    %-12.5f %9d %7.2f%%  %.2fx-%.2fx past it (median %.2fx)"%(
            candidate, len( bad ), 100.0*len( bad )/n,
            ratios[0], ratios[-1], percentile( ratios, 0.50 ) ) )

    if is_max and n:
        logger.info( "    cost per action, at the tightest candidate above:" )
        tightest = min( candidates )
        bad = [ value for value in ordered if value > tightest ]
        if bad:
            # Under clamp, a box is scaled so its area lands on the bound.
            # The linear shrink is sqrt of the area ratio.
            shrinks = sorted( ( tightest / value ) ** 0.5 for value in bad )
            logger.info( "      drop  : %d detection(s) lose their censor entirely"%len( bad ) )
            logger.info( "      clamp : %d censor(s) shrink to %.0f%%-%.0f%% of their "
                         "linear size (median %.0f%%)"%(
                             len( bad ), 100*shrinks[0], 100*shrinks[-1],
                             100*percentile( shrinks, 0.50 ) ) )
            logger.info( "      flag  : nothing changes; %d counted"%len( bad ) )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter )
    parser.add_argument( '--labels', nargs='+', default=None,
        help="restrict to these labels (default: the labels you actually censor)" )
    parser.add_argument( '--all-labels', action='store_true',
        help="report every label with cached detections, not just censored ones" )
    parser.add_argument( '--include-preview', action='store_true',
        help="also read preview-slice caches. A preview slice is short and "
             "boundary-truncated, so its size distribution is both smaller and "
             "more uniform than a real run's - which is exactly the statistic "
             "this tool reports. Real cache only by default." )
    parser.add_argument( '--percentiles', type=float, nargs='+',
        default=[ 0.99, 0.995, 1.0 ],
        help="candidate bounds are drawn from these percentiles of the observed "
             "distribution, each also widened by --widen (default: 0.99 0.995 1.0)" )
    parser.add_argument( '--widen', type=float, nargs='+', default=[ 1.0, 1.25, 1.5 ],
        help="multipliers applied to each percentile (default: 1.0 1.25 1.5). "
             "betabench's own suggestion is p99 x 1.25" )
    args = parser.parse_args()

    logger = bu_log.get_logger()

    if args.all_labels:
        labels_filter = None
    elif args.labels:
        labels_filter = set( args.labels )
    else:
        labels_filter = set( bu_config.get_parts_to_blur(
            bu_detector.selected_backend_name() ).keys() )

    per_configuration, source_count = collect( args.include_preview, labels_filter )
    if not per_configuration:
        logger.error( "no%s cache files found - run betatv.py on real footage first"%(
            "" if args.include_preview else " real (non-preview)" ) )
        if not args.include_preview:
            logger.error( "(pass --include-preview to also read preview caches)" )
        return 1

    logger.info( "loaded %d cache file(s) across %d configuration(s)"%(
        source_count, len( per_configuration ) ) )
    if labels_filter is not None:
        logger.info( "labels: %s"%( ", ".join( sorted( labels_filter ) ), ) )

    for configuration in sorted( per_configuration ):
        logger.info( "" )
        logger.info( "=" * 74 )
        logger.info( "  %s"%configuration )
        logger.info( "=" * 74 )
        for label in sorted( per_configuration[configuration] ):
            rows = per_configuration[configuration][label]
            areas = [ row[0] for row in rows ]
            aspects = [ row[1] for row in rows ]
            for bound_name, values in ( ( 'max_area_fraction', areas ),
                                        ( 'max_aspect_ratio', aspects ) ):
                ordered = sorted( values )
                candidates = sorted( { round( percentile( ordered, pct ) * widen, 6 )
                                       for pct in args.percentiles
                                       for widen in args.widen } )
                report_bound( logger, label, rows, bound_name, values, candidates )

        # What is configured RIGHT NOW, so the report is actionable rather
        # than hypothetical.
        logger.info( "" )
        logger.info( "  currently configured, for comparison:" )
        backend = bu_detector.selected_backend_name()
        for label in sorted( per_configuration[configuration] ):
            settings = bu_config.get_label_settings( label, backend )
            configured = { key: settings.get( key ) for key in
                           ( 'min_area_fraction', 'max_area_fraction',
                             'min_aspect_ratio', 'max_aspect_ratio' )
                           if settings.get( key ) is not None }
            action = settings.get( 'geometry_action' )
            if configured:
                logger.info( "    %-16s %s  action=%r"%( label, configured, action ) )
            else:
                logger.info( "    %-16s no geometry bounds set  action=%r"%( label, action ) )

    logger.info( "" )
    logger.info( "reading this: a bound whose violations sit FAR past it (median 2x+) is" )
    logger.info( "catching outliers, which is what a sanity filter is for. A bound whose" )
    logger.info( "violations start just past it and run continuously upward is cutting" )
    logger.info( "through the middle of your real distribution - no action setting makes" )
    logger.info( "that a good bound, it is the wrong bound." )
    logger.info( "" )
    logger.info( "and remember what this cannot see: which violations were misfires and" )
    logger.info( "which were correct close-ups. There is no ground truth here. Under" )
    logger.info( "geometry_action 'clamp' that matters less than it used to, because a" )
    logger.info( "wrong guess shrinks a censor instead of deleting it." )
    return 0


if __name__ == '__main__':
    sys.exit( main() )
