#!/usr/bin/env python3
"""
Turns analyze_suppression_pairs.py's real stdout output into candidate
class_suppression rules for a backend, instead of eyeballing the printed
stats by hand the way retinanet_v2's original values were derived.

Why this exists: nudenet_v3's class_suppression block is deliberately
empty ({}) as of the 2026-09-15 per-backend migration (see
CONFIG_REFERENCE.md's "class_suppression is per-backend" section) -
retinanet_v2's tuned min_iou values don't transfer, because nudenet_v3's
boxes run structurally smaller/tighter and its IoU distributions for the
same label pairs sit far lower on the same footage (confirmed: 0.036
median IoU for exposed_breast<-covered_breast on nudenet_v3 vs 0.475 for
retinanet_v2). nudenet_v3 needs its own rules derived from its own real
overlap data, the same way retinanet_v2's were - this script mechanizes
that derivation so it's reproducible as more footage gets cached, rather
than a one-off manual read of the printed stats.

Usage:
    # pipe analyze_suppression_pairs.py's real output in directly
    python3 analyze_suppression_pairs.py > /tmp/suppression_stats.txt
    python3 derive_suppression_rules.py /tmp/suppression_stats.txt

    # or read from stdin
    python3 analyze_suppression_pairs.py | python3 derive_suppression_rules.py

Output: a class_suppression dict (Python literal, ready to paste into
betaconfig.py's detector_backend[<name>]['class_suppression'], or into a
replay_tune.py variants file's "class_suppression" key) plus, for every
pair, the reasoning it used - so this is a starting point to sanity-check
against the same judgment calls documented in CONFIG_REFERENCE.md
(protect real detections over catching every false positive; treat a
clean bimodal IoU split as "adjacent real objects" vs "same-spot
misclassification"; don't rule on pairs with too little data), not a
value to trust blindly and paste in unread.

Limitations, stated plainly:
- This can only derive rules for pairs that HAVE overlap data in
  whatever footage is currently cached. A small/biased sample (see the
  caution both analyze_suppression_pairs.py's own docstring and the user
  gave about a single breast-heavy test video) produces a small/biased
  ruleset - re-run as more, more varied footage gets cached, don't treat
  a first pass as final.
- The heuristics below (min-data thresholds, the bimodal-split
  detection, the "clear X% at this IoU"/"clear X% at this margin"
  percentile picks) are a reasonable STARTING methodology modeled on how
  retinanet_v2's values were reasoned about in CONFIG_REFERENCE.md, not
  a formula guaranteed to produce the same quality of judgment a human
  tuning pass gives.
- margin is proposed alongside min_iou (requires analyze_suppression_pairs.py's
  score-delta reporting in its input - added alongside this script's margin
  support; older captured output without a "score(X) - score(Y)..." line per
  pair won't have anything to propose a margin from). margin is treated as
  REQUIRED, not optional: apply_class_suppression's own default when margin is
  absent is 0.0, which is the loosest possible gate (fires whenever the
  suppressing label's score is even just >= the suppressed label's) rather
  than a neutral "no margin gate" - so a rule proposed here always carries an
  explicit margin value (possibly 0.00, deliberately chosen, if the data shows
  that's genuinely the right gate - see propose_margin's own docstring), never
  an omitted key relying on that default.
"""

import re
import sys


MIN_OVERLAPPING_PAIRS = 20  # below this, there's not enough signal to propose anything - matches the spirit of CONFIG_REFERENCE.md's "covered_belly has no rule" call (~2-4 pairs judged not enough)


def parse_stats_output( text ):
    """
    Parses the '=== Overlap stats for your class_suppression pairs ===',
    per real analyze_suppression_pairs.py output, into a list of dicts:
    { 'label_a', 'label_b', 'pair_count', 'overlap_count',
      'iou_median', 'thresholds': {0.05: (survive,total), ...},
      'bimodal': bool }
    Deliberately regex-based against the real printed format rather than
    importing analyze_suppression_pairs.py and recomputing - this way it
    works against output the user pastes/redirects from a real run
    without needing that run to happen in the same process, and stays
    honest about only ever seeing what was actually printed.
    """
    results = []
    blocks = re.split( r'\n(?=\S.*? suppressed_by \S.*?:)', text )
    for block in blocks:
        m = re.match( r'(\S+) suppressed_by (\S+):', block )
        if not m:
            continue
        label_a, label_b = m.group(1), m.group(2)

        if 'no same-instant' in block:
            results.append( { 'label_a': label_a, 'label_b': label_b, 'pair_count': 0, 'overlap_count': 0 } )
            continue

        pc = re.search( r'same-instant pairs found:\s*(\d+)\s*\(of which actually overlapping:\s*(\d+)\)', block )
        if not pc:
            continue
        pair_count, overlap_count = int(pc.group(1)), int(pc.group(2))

        entry = { 'label_a': label_a, 'label_b': label_b, 'pair_count': pair_count, 'overlap_count': overlap_count }

        med = re.search( r'IoU, overlapping pairs only:\s*n=\d+\s+min=-?[\d.]+\s+median=(-?[\d.]+)', block )
        if med:
            entry['iou_median'] = float( med.group(1) )

        thresholds = {}
        for tm in re.finditer( r'fraction of overlapping pairs with IoU >= ([\d.]+): (\d+)/(\d+) \((\d+)%\)', block ):
            thresholds[ float(tm.group(1)) ] = ( int(tm.group(2)), int(tm.group(3)), int(tm.group(4)) )
        entry['thresholds'] = thresholds

        # score delta = score(suppressed_by) - score(label) on overlapping pairs only - what
        # margin gates on (see analyze_suppression_pairs.py's own score-delta section, added
        # alongside this parsing - apply_class_suppression's margin check is
        # 'other[score] - raw[score] >= margin').
        score_med = re.search( r'score\(\S+\) - score\(\S+\), overlapping pairs only:\s*n=\d+\s+min=-?[\d.]+\s+median=(-?[\d.]+)', block )
        if score_med:
            entry['score_delta_median'] = float( score_med.group(1) )

        score_thresholds = {}
        for tm in re.finditer( r'fraction of overlapping pairs with score delta >= (-?[\d.]+): (\d+)/(\d+) \((\d+)%\)', block ):
            score_thresholds[ float(tm.group(1)) ] = ( int(tm.group(2)), int(tm.group(3)), int(tm.group(4)) )
        entry['score_thresholds'] = score_thresholds

        # crude bimodal check: a real histogram line dump follows for
        # HISTOGRAM_PAIRS - if present, flag pairs with a clear low AND
        # high cluster (both buckets holding a meaningful share) as
        # "adjacent real objects" candidates, same judgment call
        # CONFIG_REFERENCE.md documents for exposed_penis/exposed_vulva.
        hist = re.findall( r'\[([\d.]+)-([\d.]+)\):\s*(\d+)', block )
        if hist:
            counts = [ int(c) for _, _, c in hist ]
            total = sum(counts) or 1
            low_cluster = sum( counts[0:2] ) / total   # [0.0-0.2)
            high_cluster = sum( counts[8:10] ) / total  # [0.8-1.0)
            entry['bimodal'] = ( low_cluster >= 0.15 and high_cluster >= 0.15 )
        else:
            entry['bimodal'] = False

        results.append( entry )
    return results


def propose_rule( entry ):
    """
    Returns (min_iou, reasoning_str) or (None, reasoning_str) if there's
    not enough signal to propose anything for this pair.
    """
    label_a, label_b = entry['label_a'], entry['label_b']
    overlap_count = entry.get( 'overlap_count', 0 )

    if entry.get( 'pair_count', 0 ) == 0:
        return( None, "no same-instant pairs at all in cached data yet - can't propose anything, need more footage" )

    if overlap_count < MIN_OVERLAPPING_PAIRS:
        return( None, "only %d overlapping pairs (< %d minimum) - not enough signal, same call CONFIG_REFERENCE.md made for covered_belly"%(
            overlap_count, MIN_OVERLAPPING_PAIRS ) )

    thresholds = entry.get( 'thresholds', {} )
    if not thresholds:
        return( None, "no threshold breakdown available in this output - can't propose a min_iou" )

    if entry.get( 'bimodal' ):
        # "adjacent real objects" pattern (exposed_penis/exposed_vulva
        # shape in CONFIG_REFERENCE.md): pick the IoU just past where the
        # high cluster starts, sacrificing the ambiguous middle/low
        # cluster deliberately, same tradeoff documented there.
        for t in sorted( thresholds.keys() ):
            survive, total, pct = thresholds[t]
            if pct <= 25:  # roughly "start of the high cluster" - only ~1/4 of overlaps still clear this bar
                return( t, "bimodal IoU split detected (low + high clusters both present) - treating as "
                    "'genuinely adjacent real objects' vs 'same-spot misclassification', same as the "
                    "exposed_penis/exposed_vulva rule in CONFIG_REFERENCE.md. Picked min_iou=%.2f "
                    "(~%d%% of overlaps clear it), deliberately sacrificing the ambiguous middle/low "
                    "cluster to avoid suppressing real adjacent content - VERIFY this against the "
                    "actual histogram before trusting it, this heuristic is approximate."%(t, pct) )
        # fell through: even the loosest threshold clears >25%, use it anyway with a flag
        loosest = min( thresholds.keys() )
        return( loosest, "bimodal split flagged but no clean 'start of high cluster' point found automatically - "
            "defaulting to loosest tested threshold %.2f, MANUAL REVIEW NEEDED"%(loosest) )

    # non-bimodal / "same spot, model confused" pattern: pick near the
    # median, matching how e.g. covered_breast (median 0.62, chosen 0.55)
    # and exposed_buttocks (median 0.175, chosen 0.30 - tighter than
    # median since buttocks/breast are similarly confident) were picked
    # in CONFIG_REFERENCE.md - median is a reasonable starting anchor,
    # not a hard rule.
    median = entry.get( 'iou_median' )
    if median is None:
        return( None, "no median IoU reported - can't anchor a proposal" )
    # snap to the nearest of the tested threshold values so the proposal
    # is always something analyze_suppression_pairs.py actually measured
    # survival-rate for, not an interpolated number nobody checked.
    candidates = sorted( thresholds.keys() )
    nearest = min( candidates, key=lambda t: abs(t - median) )
    survive, total, pct = thresholds[nearest]
    return( nearest, "not bimodal - 'same spot, model confused' pattern, anchored near median IoU %.3f. "
        "Picked min_iou=%.2f (%d%% of overlaps clear it) as a starting point - tighten below median if "
        "%s is the label actually being censored (protect it over catching more false positives, same "
        "bias CONFIG_REFERENCE.md documents for covered_breast/face rules), loosen if not."%(
        median, nearest, pct, label_a ) )


def propose_margin( entry, label_a ):
    """
    Returns (margin, reasoning_str) or (None, reasoning_str) if there's
    not enough signal to propose one. Called only when propose_rule
    already found enough overlap signal to propose a min_iou - margin is
    required alongside min_iou (not an optional extra gate), since
    apply_class_suppression's margin default of 0.0 is the loosest
    possible gate (fires whenever the suppressing label's score is even
    just equal to the suppressed label's), not a neutral "off" value -
    see CONFIG_REFERENCE.md and this script's own docstring history for
    why that silent default isn't safe to leave unset.
    """
    score_thresholds = entry.get( 'score_thresholds', {} )
    if not score_thresholds:
        return( None, "no score-delta breakdown available in this output - re-run analyze_suppression_pairs.py "
            "(needs the score-delta reporting added alongside this margin support) and re-derive" )

    median = entry.get( 'score_delta_median' )
    if median is None:
        return( None, "no median score delta reported - can't anchor a margin proposal" )

    label_b = entry['label_b']

    # If even margin=0.00 (the loosest gate - suppressing label merely has
    # to be >= as confident) only clears on a small minority of overlapping
    # pairs, the suppressing label is *usually* LESS confident than the
    # label it's supposedly correcting - i.e. real signal that these are
    # two independently-detected real things, not "same spot, model
    # picked the wrong label". A positive margin here wouldn't refine the
    # rule, it would gut it: propose 0.0 explicitly (still a real,
    # deliberate value, not the unset/never-checked default) and flag for
    # manual review rather than silently accepting a value that stops the
    # rule from firing in practice.
    at_zero = score_thresholds.get( 0.00 )
    if at_zero is not None:
        survive0, total0, pct0 = at_zero
        if pct0 < 30:
            return( 0.00, "median score delta %.3f, only %d%% of overlapping pairs even clear margin=0.00 "
                "(suppressing label %s is usually LESS confident than %s here, not more) - a positive margin "
                "would gut this rule rather than refine it. Proposing the loosest real value (0.00, gates on "
                "IoU alone) rather than leaving margin unset - MANUAL REVIEW: consider whether this pair "
                "should have a rule at all if the suppressing label so rarely wins on confidence."%(
                median, pct0, label_b, label_a) )

    # Normal case: pick the margin near where roughly a third of
    # overlapping pairs still clear it - tighter than the median (so the
    # gate does real work, not just "usually true anyway"), looser than
    # requiring the suppressing label to dominate outright - same
    # "protect the censored label, don't demand overwhelming evidence"
    # bias CONFIG_REFERENCE.md documents for min_iou picks. Snapped to a
    # threshold this script's own data source actually measured, same
    # discipline as propose_rule's min_iou snapping.
    candidates = sorted( score_thresholds.keys() )
    target_pct = 33
    best = min( candidates, key=lambda t: abs( score_thresholds[t][2] - target_pct ) )
    survive, total, pct = score_thresholds[best]
    return( best, "median score delta %.3f. Picked margin=%.2f (%d%% of overlapping pairs still clear it) as a "
        "starting point, targeting roughly a third surviving so the gate does real filtering work without "
        "demanding %s dominate %s outright - tighten if %s is the label actually being censored and false "
        "suppressions are still showing up, loosen toward 0.00 if the rule is barely firing."%(
        median, best, pct, label_b, label_a, label_a) )


def main():
    if len(sys.argv) > 1:
        with open( sys.argv[1], 'r' ) as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    entries = parse_stats_output( text )
    if not entries:
        print( "no '<label> suppressed_by <label>:' blocks found in the input - make sure you're piping in "
            "analyze_suppression_pairs.py's real stdout (or a file containing it), not something else." )
        sys.exit(1)

    print( "=== Derived class_suppression candidates ===" )
    print( "(read the reasoning for each - this is a starting point for a real tuning pass, not a value to paste in unread. margin is required alongside min_iou - see this script's docstring for why a missing margin isn't a safe default.)" )
    print()

    proposed = {}
    for entry in entries:
        min_iou, iou_reasoning = propose_rule( entry )
        label_a, label_b = entry['label_a'], entry['label_b']
        print( "%s suppressed_by %s:"%(label_a, label_b) )
        print( "  min_iou: %s"%(iou_reasoning) )
        if min_iou is None:
            print()
            continue

        margin, margin_reasoning = propose_margin( entry, label_a )
        print( "  margin:  %s"%(margin_reasoning) )
        if margin is None:
            print( "  -> NOT PROPOSED: min_iou has signal but margin doesn't - margin is required, so no rule is proposed for this pair until score-delta data is available (re-run analyze_suppression_pairs.py)." )
            print()
            continue

        print( "  -> PROPOSED: { 'suppressed_by': '%s', 'margin': %.2f, 'min_iou': %.2f }"%(label_b, margin, min_iou) )
        proposed.setdefault( label_a, [] ).append( { 'suppressed_by': label_b, 'margin': round(margin, 2), 'min_iou': round(min_iou, 2) } )
        print()

    print( "=== As a Python dict (paste into betaconfig.py's detector_backend['nudenet_v3']['class_suppression'], "
        "or a replay_tune.py variants file's 'class_suppression' key, AFTER reviewing every reasoning line above) ===" )
    print( "{" )
    for label, rules in proposed.items():
        print( "    %r: %s,"%(label, rules) )
    print( "}" )


if __name__ == '__main__':
    main()
