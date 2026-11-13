#!/usr/bin/env python3
"""
Pulls real detection box sizes out of BetaSuite's raw-detection cache
(../output/cache/vid_hashes/*.gz and ../output/cache/pic_hashes/*.gz, same format betatv.py/betastare.py
write) and reports, per class_suppression pair, what IoU values AND what
confidence-score deltas are actually achievable and actually occurring -
instead of guessing at min_iou/margin values the way the current
betaconfig.py numbers were picked. min_iou and margin are separate,
AND'd gates in apply_class_suppression (betatv.py) - IoU says "these two
boxes are probably the same spot"; margin says "and the suppressing
label was clearly more confident than the suppressed one" - so this
reports both, from the same overlapping-pair population.

Run this from inside BetaSuite-0.2.4/ (same place you run betatv.py from),
with the venv active, e.g.:

    python3 analyze_suppression_pairs.py

It only reads the cache; it doesn't touch betaconfig.py or run the neural
net. The more footage you've run through betatv.py/betastare.py (which is
what populates the cache), the more this will actually tell you - right now
it's whatever happens to be cached already, which may be a small, biased
sample.
"""

import glob
import gzip
import json
from array import array
import os
import sys

import sys as _sys, os as _os
_sys.path.insert( 0, _os.path.dirname( _os.path.dirname( _os.path.dirname( _os.path.abspath( __file__ ) ) ) ) )  # so 'import betaconst'/'betaconfig'/etc still resolve after this script moved into tools/analysis or tools/tuning

import betaconst
import betautils_detector as bu_detector
import betautils_cache_paths as bu_cache

# One source of truth for every cache path and filename - see
# betautils_cache_paths.py's docstring for why these are not literals.
VID_HASH_DIR = bu_cache.VID_HASH_DIR
PIC_HASH_DIR = bu_cache.PIC_HASH_DIR

# The pairs worth checking overlap for, regardless of which backend(s)
# currently have a class_suppression rule for them. pairs_to_check() adds
# every configured rule this list does not name. Not read from betaconfig.py
# alone, since class_suppression is per-backend now
# (see betautils_detector.get_class_suppression) and this list is meant
# to surface candidates for a backend that has no rule yet just as much
# as confirm ones that do. Edit this list by hand if you want to check a
# new pair.
PAIRS_OF_INTEREST = [
    ('exposed_breast', 'covered_breast'),
    ('exposed_breast', 'face_femme'),
    ('exposed_breast', 'face_masc'),
    ('exposed_breast', 'exposed_belly'),
    ('exposed_breast', 'covered_belly'),   # too sparse to tune yet (see betaconfig.py comment) - kept here to re-check as footage grows
    ('exposed_breast', 'exposed_buttocks'),
    ('exposed_breast', 'exposed_chest'),
    # exposed_breast vs exposed_vulva and exposed_breast vs exposed_penis were
    # both checked (see betaconfig.py's class_suppression comments) - real
    # overlap exists but deliberately NOT turned into suppression rules:
    # both labels are actively censored, and unlike the other pairs above,
    # suppressing one for the other risks leaving genuinely exposed,
    # adjacent-but-real content completely uncensored rather than just
    # removing a wrong censor. Kept here so they get re-checked as more
    # footage comes in, not because a rule is planned.
    ('exposed_breast', 'exposed_vulva'),
    ('exposed_breast', 'exposed_penis'),
    ('exposed_vulva',  'exposed_armpits'), # flagged as a real bimodal signal (like exposed_penis/vulva below) but no rule added yet - pending a decision on whether exposed_armpits winning should leave that spot uncensored
    ('exposed_vulva',  'exposed_anus'),
    ('exposed_vulva',  'covered_vulva'),
    ('exposed_vulva',  'face_femme'),
    ('exposed_vulva',  'face_masc'),
    ('exposed_vulva',  'exposed_penis'),
]

# fine-grained histogram just for pairs where we need to see whether the IoU
# distribution is bimodal (a "same spot, model confused" cluster near 1.0 vs
# a "two different real things touching" cluster lower down) rather than a
# single blob - a single min_iou threshold can only separate those two
# populations if they're actually separated in the data.
HISTOGRAM_PAIRS = [
    ('exposed_vulva', 'exposed_penis'),
]


def pairs_to_check():
    """
    PAIRS_OF_INTEREST, then every pair a backend's class_suppression
    configures that the list does not already name.

    The list alone missed rules that exist: the nudenet vulva rules
    against exposed_belly, exposed_breast and covered_breast were never
    reported, so their overlaps could not be checked here at all.
    """
    pairs = list( PAIRS_OF_INTEREST )
    for backend_name in bu_detector.registered_backend_names():
        for label, rules in sorted( ( bu_detector.get_class_suppression( backend_name ) or {} ).items() ):
            for rule in ( [ rules ] if isinstance( rules, dict ) else rules or [] ):
                pair = ( label, rule.get( 'suppressed_by' ) )
                if pair[1] and pair not in pairs:
                    pairs.append( pair )
    return pairs

def label_of(raw, classes):
    # raw['class_id'] is already BetaSuite's canonical string label
    # (see betautils_detector.py) - 'classes' arg kept for
    # call-site compatibility but no longer used for a lookup here.
    return raw['class_id']

def iou(a, b):
    ax2, ay2 = a['x']+a['w'], a['y']+a['h']
    bx2, by2 = b['x']+b['w'], b['y']+b['h']
    ix1, iy1 = max(a['x'], b['x']), max(a['y'], b['y'])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    if inter <= 0:
        return 0.0
    union = a['w']*a['h'] + b['w']*b['h'] - inter
    return inter/union if union > 0 else 0.0

def load_cache_dir( path ):
    """
    Yield (source_filename, raw_boxes) for every cache file in path.

    A thin alias for betautils_cache_paths.iter_newest_cache_files (each
    video's newest run per configuration), kept under
    the historical name this tool's body uses. A video cache holds every
    sampled frame's boxes for one video concatenated together (matched
    by 't'); a picture cache holds one photo's boxes, all at the same t.
    """
    return bu_cache.iter_newest_cache_files( path, include_preview=True )


# fmt_stats's body now lives in betautils_cache_paths.py (shared with
# analyze_jitter.py's/compare_models_perf.py's near-identical copies -
# see that module's docstring) - this file always wanted 'min' included
# (include_min=True), unlike analyze_jitter.py's/analyze_track_breaks.py's
# copy, which is why this stayed a thin wrapper instead of a plain
# import of the shared function under this name.
def fmt_stats( values, fmt='%.0f' ):
    return bu_cache.fmt_stats( values, fmt=fmt, include_min=True )


# Which backend wrote a cache file is parsed in exactly one place -
# betautils_cache_paths.backend_of_cache_filename - because every tool
# that reimplemented it drifted, which is the class of bug this module
# exists to end.
backend_of_cache_filename = bu_cache.backend_of_cache_filename


class ConfigurationReport:
    """
    One configuration's box-size and pair-overlap numbers, built one
    cache file at a time and printed once at the end.

    Only the numbers the report prints are kept, as array('d') (8 bytes
    a value). The tool used to hold every file's parsed detections at
    once, about 500 bytes a detection: gigabytes on a library of
    millions, beside the other tools the analysis hub runs in parallel.
    The output is identical: the same values in the same order.
    """

    def __init__( self, classes, label_names ):
        self.classes = classes
        self.label_names = label_names
        self.sizes_by_label = {}
        self.pairs = {}
        for label_a, label_b in pairs_to_check():
            if label_a in label_names and label_b in label_names:
                self.pairs[ (label_a, label_b) ] = {
                    'ious': array( 'd' ),
                    # smaller box area / larger box area, for ALL cross pairs regardless of overlap
                    'area_ratios': array( 'd' ),
                    # score(label_b) - score(label_a), overlapping pairs only - what margin gates on
                    'score_deltas': array( 'd' ),
                    'pair_count': 0 }

    def add( self, boxes ):
        """Fold one cache file's detections in."""
        classes = self.classes
        sizes_by_label = self.sizes_by_label
        for b in boxes:
            label = label_of(b, classes)
            sizes = sizes_by_label.get( label )
            if sizes is None:
                sizes = sizes_by_label[label] = { 'w': array( 'd' ), 'h': array( 'd' ), 'area': array( 'd' ) }
            sizes['w'].append( b['w'] )
            sizes['h'].append( b['h'] )
            sizes['area'].append( b['w']*b['h'] )

        for ( label_a, label_b ), pair in self.pairs.items():
            ious = pair['ious']
            area_ratios = pair['area_ratios']
            score_deltas = pair['score_deltas']
            # Group each label's boxes by timestamp FIRST, then only compare
            # within matching timestamp buckets - this was previously a
            # brute-force cross product (every label_a box against every
            # label_b box in the whole video, filtered by timestamp
            # afterward), which is fine for small counts but becomes
            # billions of comparisons once a label has 30-50k detections in
            # one video (exposed_breast/exposed_belly/face_femme etc all
            # get there fast at 8fps over a real-length video) - the same
            # thing apply_class_suppression itself does per-frame, never a
            # full cross product, so grouping first here just matches that
            # and turns "would take hours" into "finishes in seconds"
            # without changing which pairs get compared at all.
            a_by_t = {}
            b_by_t = {}
            for b in boxes:
                label = label_of(b, classes)
                if label == label_a:
                    a_by_t.setdefault( b['t'], [] ).append( b )
                elif label == label_b:
                    b_by_t.setdefault( b['t'], [] ).append( b )
            if not a_by_t or not b_by_t:
                continue
            for t, a_boxes in a_by_t.items():
                b_boxes = b_by_t.get( t )
                if not b_boxes:
                    continue
                for a in a_boxes:
                    for b in b_boxes:
                        pair['pair_count'] += 1
                        area_a = a['w']*a['h']
                        area_b = b['w']*b['h']
                        area_ratios.append( min(area_a,area_b) / max(area_a,area_b) if max(area_a,area_b) > 0 else 0 )
                        this_iou = iou(a,b)
                        if this_iou > 0:
                            ious.append( this_iou )
                            # margin gates on score(suppressed_by) - score(label) - see
                            # apply_class_suppression in betatv.py: 'if other['score'] - raw['score'] >= margin'.
                            # Only meaningful for pairs that actually overlap (same reason IoU stats above
                            # are also overlap-only) - a same-instant pair with zero overlap was never a
                            # suppression candidate in the first place, so its score delta doesn't bear on
                            # what margin should be.
                            score_deltas.append( b['score'] - a['score'] )

    def print( self ):
        _print_report( self )


def print_report_for( sources, classes, label_names ):
    """
    Runs the full box-size-stats + class_suppression-pair-overlap report
    (this tool's original single-backend behavior) against one group of
    (kind, name, boxes) sources. Split out from main() so it can be
    called once per detector backend when cache from more than one
    backend is present (see main()), instead of pooling every backend's
    detections together into one misleading combined report.
    """
    report = ConfigurationReport( classes, label_names )
    for _kind, _name, boxes in sources:
        report.add( boxes )
    report.print()


def _print_report( report ):
    sizes_by_label = report.sizes_by_label
    label_names = report.label_names

    print( "=== Box size stats per label (all cached detections, any score) ===" )
    for label in sorted( sizes_by_label.keys() ):
        s = sizes_by_label[label]
        print( "%-16s  w: %s"%(label, fmt_stats(s['w'])) )
        print( "%-16s  h: %s"%('', fmt_stats(s['h'])) )
        print( "%-16s  area: %s"%('', fmt_stats(s['area'])) )
    print()

    # ---- for each pair of interest, find actually-overlapping instances ----
    print( "=== Overlap stats for your class_suppression pairs ===" )
    for label_a, label_b in pairs_to_check():
        if label_a not in label_names or label_b not in label_names:
            print( "%s vs %s: unknown class name, skipping"%(label_a, label_b) )
            continue

        pair = report.pairs[ (label_a, label_b) ]
        ious = pair['ious']
        area_ratios = pair['area_ratios']
        score_deltas = pair['score_deltas']
        pair_count = pair['pair_count']

        print( "\n%s suppressed_by %s:"%(label_a, label_b) )
        if pair_count == 0:
            print( "  no same-instant (%s, %s) pairs found in cache at all - either this combination never"%(label_a, label_b) )
            print( "  co-occurred in your test footage, or one/both classes never got detected. Can't say anything" )
            print( "  about achievable IoU from this data; run more footage through and re-check." )
            continue

        print( "  same-instant pairs found: %d  (of which actually overlapping: %d)"%(pair_count, len(ious)) )
        if area_ratios:
            print( "  area ratio (smaller/larger box), all same-instant pairs: %s"%(fmt_stats(area_ratios, '%.3f')) )
        if ious:
            print( "  IoU, overlapping pairs only:                            %s"%(fmt_stats(ious, '%.3f')) )
            # what fraction of the *overlapping* pairs would survive each candidate min_iou
            for threshold in (0.05, 0.10, 0.20, 0.30, 0.50, 0.60, 0.70, 0.85):
                survive = sum( 1 for v in ious if v >= threshold )
                print( "    fraction of overlapping pairs with IoU >= %.2f: %d/%d (%.0f%%)"%(threshold, survive, len(ious), 100*survive/len(ious)) )

            print( "  score(%s) - score(%s), overlapping pairs only:  %s"%(label_b, label_a, fmt_stats(score_deltas, '%.3f')) )
            # what fraction of overlapping pairs would survive each candidate margin, i.e. what
            # fraction has score(suppressed_by) - score(label) >= margin - directly answers "if I
            # set margin to X, how often does this rule actually get a chance to suppress" (assuming
            # min_iou is also cleared - the two gates are AND'd in apply_class_suppression, this is
            # margin's contribution in isolation, same framing as the min_iou threshold table above).
            for threshold in (0.00, 0.05, 0.10, 0.15, 0.20, 0.30):
                survive = sum( 1 for v in score_deltas if v >= threshold )
                print( "    fraction of overlapping pairs with score delta >= %.2f: %d/%d (%.0f%%)"%(threshold, survive, len(score_deltas), 100*survive/len(score_deltas)) )

            if (label_a, label_b) in HISTOGRAM_PAIRS:
                print( "  IoU histogram (looking for bimodal split = 'same spot' cluster vs 'adjacent real objects' cluster):" )
                buckets = [0]*10
                for v in ious:
                    idx = min(9, int(v*10))
                    buckets[idx] += 1
                # scale bars to a fixed max width instead of printing one
                # '#' per raw count - with real footage a bucket can hold
                # tens of thousands of detections, which used to print a
                # single line tens of thousands of characters long.
                MAX_BAR_WIDTH = 60
                biggest = max( buckets ) if buckets else 0
                for i, count in enumerate(buckets):
                    lo, hi = i/10, (i+1)/10
                    bar_len = round( MAX_BAR_WIDTH * count / biggest ) if biggest > 0 else 0
                    bar = '#' * bar_len
                    print( "    [%.1f-%.1f): %6d %s"%(lo, hi, count, bar) )
        else:
            print( "  none of the same-instant pairs actually overlapped (IoU=0 for all)." )


def main():
    classes = betaconst.classes
    label_names = set( classes.keys() )

    # One pass over the files, folding each into its configuration's
    # report and letting its detections go before the next is read.
    reports = {}
    files_by_configuration = {}
    file_count = { 'video': 0, 'photo': 0 }
    for kind, directory in ( ('video', VID_HASH_DIR), ('photo', PIC_HASH_DIR) ):
        for name, boxes in load_cache_dir( directory ):
            configuration = bu_cache.configuration_label_of_cache_filename( name )
            report = reports.get( configuration )
            if report is None:
                report = reports[configuration] = ConfigurationReport( classes, label_names )
            report.add( boxes )
            del boxes
            files_by_configuration[configuration] = files_by_configuration.get( configuration, 0 ) + 1
            file_count[kind] += 1

    if not reports:
        print( "No cache files found in %s or %s - run betatv.py/betastare.py on some footage first."%(VID_HASH_DIR, PIC_HASH_DIR) )
        return

    # Group by which detector backend actually produced each cache file
    # (parsed from its own filename - see backend_of_cache_filename).
    # Pooling different backends' detections together would be
    # misleading (their box size distributions and confusion patterns
    # are genuinely different, not just noisier versions of each other),
    # so this reports each backend found in cache as its own fully
    # separate section - if you've run betatv.py/betastare.py under
    # both retinanet_v2 and nudenet_v3 (see betaconfig.py's
    # detector_backend['selected']), this is a direct side-by-side
    # comparison of their real detection-quality signatures on your own
    # footage: same box-size-stats/confusion-pair-IoU methodology,
    # applied per model instead of blended.
    # Grouped by CONFIGURATION (backend + size + detection key), not by
    # backend name. Two variants of one backend are two sets of model
    # weights; pooling them produces a section whose numbers describe
    # neither. See betautils_cache_paths.configuration_label_of_cache_filename.
    print( "Loaded %d cache file(s) (%d video, %d photo) across %d configuration(s): %s\n"%(
        file_count['video'] + file_count['photo'],
        file_count['video'],
        file_count['photo'],
        len(reports),
        ", ".join( sorted(reports.keys()) ) ) )

    for configuration in sorted( reports.keys() ):
        header = "############################################################\n# configuration: %s  (%d cache file(s))\n############################################################"%(
            configuration, files_by_configuration[configuration] )
        print( header )
        reports[configuration].print()
        print()

    if len(reports) > 1:
        print( "Reminder: box-size and confusion-pair-IoU distributions above are each configuration's OWN signature - a" )
        print( "smaller/larger typical box size, or a different fraction of overlapping pairs at a given IoU threshold," )
        print( "isn't automatically 'better' or 'worse' on its own. What IS directly comparable across sections: whether" )
        print( "a confusion pattern you were trying to fix (e.g. penis/vulva/breast same-spot overlap, or faces getting" )
        print( "picked up as vulva/breast) shrinks, stays the same, or gets worse from one section to the next. Note that" )
        print( "two sections of the same backend at different sizes are two different sets of model weights, so that" )
        print( "comparison is as meaningful between them as it is between backends." )


if __name__ == '__main__':
    main()
