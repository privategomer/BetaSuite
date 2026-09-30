"""
test_derive_suppression_rules.py - regression coverage for
tools/tuning/derive_suppression_rules.py's parsing and proposal logic,
added 2026-09-16 alongside margin derivation.

Why this needed its own test: this tool's output is what gets pasted
directly into betaconfig.py's class_suppression blocks (see
CONFIG_REFERENCE.md's nudenet_v3 tuning workflow) - a regression here
silently ships a wrong min_iou/margin into a real censoring config, not
just a wrong analysis report. Margin support in particular has a sharp
edge worth pinning down: apply_class_suppression (betatv.py) treats a
missing margin as 0.0, the LOOSEST possible gate, not a neutral "off"
value - so this tool must never propose a min_iou without also
proposing a margin (even an explicit 0.00), and must correctly detect
the "suppressing label rarely wins on confidence" case (median score
delta negative, low survival even at margin=0.00) rather than
proposing a margin that would silently gut the rule.

These tests build synthetic analyze_suppression_pairs.py-shaped text
blocks by hand (not against real cache data - that's what the manual
2026-09-16 nudenet_v3 margin derivation used, see CONFIG_REFERENCE.md)
so the parsing regexes and proposal heuristics are pinned down
independently of any one real run's numbers.
"""

import importlib.util
import os
import unittest

_HERE = os.path.dirname( os.path.abspath( __file__ ) )
_REPO_ROOT = os.path.dirname( _HERE )  # BetaSuite-0.2.4/


def _load_module( relative_path, module_name ):
    full_path = os.path.join( _REPO_ROOT, relative_path )
    spec = importlib.util.spec_from_file_location( module_name, full_path )
    module = importlib.util.module_from_spec( spec )
    spec.loader.exec_module( module )
    return module


dsr = _load_module( 'tools/tuning/derive_suppression_rules.py', 'derive_suppression_rules' )


def _make_block( label_a, label_b, pair_count, overlap_count, iou_median,
                  iou_thresholds, score_median, score_thresholds, bimodal_hist=None ):
    """
    Builds one '<label_a> suppressed_by <label_b>:' text block in
    analyze_suppression_pairs.py's real printed shape, so parse_stats_output
    can be tested against realistic input without needing a real cache run.
    iou_thresholds / score_thresholds: {threshold: (survive, total, pct)}.
    """
    lines = [ "%s suppressed_by %s:"%(label_a, label_b) ]
    lines.append( "  same-instant pairs found: %d  (of which actually overlapping: %d)"%(pair_count, overlap_count) )
    lines.append( "  area ratio (smaller/larger box), all same-instant pairs: n=%d  min=0.010  median=0.500  mean=0.500  p90=0.900  max=1.000"%pair_count )
    lines.append( "  IoU, overlapping pairs only:                            n=%d  min=0.000  median=%.3f  mean=%.3f  p90=%.3f  max=%.3f"%(
        overlap_count, iou_median, iou_median, iou_median, iou_median) )
    for t in sorted( iou_thresholds.keys() ):
        survive, total, pct = iou_thresholds[t]
        lines.append( "    fraction of overlapping pairs with IoU >= %.2f: %d/%d (%d%%)"%(t, survive, total, pct) )
    lines.append( "  score(%s) - score(%s), overlapping pairs only:  n=%d  min=-0.500  median=%.3f  mean=%.3f  p90=0.500  max=0.500"%(
        label_b, label_a, overlap_count, score_median, score_median) )
    for t in sorted( score_thresholds.keys() ):
        survive, total, pct = score_thresholds[t]
        lines.append( "    fraction of overlapping pairs with score delta >= %.2f: %d/%d (%d%%)"%(t, survive, total, pct) )
    if bimodal_hist:
        lines.append( "  IoU histogram (looking for bimodal split = 'same spot' cluster vs 'adjacent real objects' cluster):" )
        for lo, hi, count in bimodal_hist:
            lines.append( "    [%.1f-%.1f): %6d"%(lo, hi, count) )
    return "\n".join( lines )


class TestParseStatsOutput( unittest.TestCase ):

    def test_parses_pair_and_overlap_counts( self ):
        block = _make_block( 'exposed_breast', 'covered_breast', 55444, 775, 0.036,
            { 0.05: (225,775,29) }, -0.112, { 0.00: (203,775,26) } )
        entries = dsr.parse_stats_output( block )
        self.assertEqual( len(entries), 1 )
        self.assertEqual( entries[0]['label_a'], 'exposed_breast' )
        self.assertEqual( entries[0]['label_b'], 'covered_breast' )
        self.assertEqual( entries[0]['pair_count'], 55444 )
        self.assertEqual( entries[0]['overlap_count'], 775 )

    def test_parses_iou_and_score_delta_medians( self ):
        block = _make_block( 'exposed_vulva', 'face_femme', 2949, 60, 0.079,
            { 0.10: (21,60,35) }, 0.056, { 0.10: (19,60,32) } )
        entries = dsr.parse_stats_output( block )
        self.assertAlmostEqual( entries[0]['iou_median'], 0.079 )
        self.assertAlmostEqual( entries[0]['score_delta_median'], 0.056 )

    def test_parses_negative_score_delta_median( self ):
        # covered_belly-shaped case: score deltas can be negative (the
        # suppressing label is LESS confident) - regex must handle the
        # leading '-' the IoU median never has.
        block = _make_block( 'exposed_breast', 'covered_belly', 3931, 30, 0.011,
            { 0.05: (1,30,3) }, -0.251, { 0.00: (0,30,0) } )
        entries = dsr.parse_stats_output( block )
        self.assertAlmostEqual( entries[0]['score_delta_median'], -0.251 )

    def test_no_same_instant_pairs( self ):
        text = "exposed_vulva suppressed_by face_masc:\n  no same-instant (exposed_vulva, face_masc) pairs found in cache at all\n"
        entries = dsr.parse_stats_output( text )
        self.assertEqual( entries[0]['pair_count'], 0 )
        self.assertEqual( entries[0]['overlap_count'], 0 )

    def test_multiple_blocks_in_one_input( self ):
        block1 = _make_block( 'exposed_breast', 'covered_breast', 100, 50, 0.5, { 0.30: (25,50,50) }, 0.0, { 0.00: (25,50,50) } )
        block2 = _make_block( 'exposed_vulva', 'exposed_penis', 200, 100, 0.2, { 0.10: (60,100,60) }, 0.03, { 0.10: (35,100,35) } )
        entries = dsr.parse_stats_output( block1 + "\n\n" + block2 )
        self.assertEqual( len(entries), 2 )
        self.assertEqual( { e['label_a'] for e in entries }, {'exposed_breast', 'exposed_vulva'} )


class TestProposeRule( unittest.TestCase ):

    def test_below_min_overlap_threshold_proposes_nothing( self ):
        entry = { 'label_a': 'a', 'label_b': 'b', 'pair_count': 100, 'overlap_count': 6,
                   'thresholds': {0.05: (3,6,50)}, 'iou_median': 0.1, 'bimodal': False }
        min_iou, reasoning = dsr.propose_rule( entry )
        self.assertIsNone( min_iou )
        self.assertIn( 'not enough signal', reasoning )

    def test_zero_pairs_proposes_nothing( self ):
        entry = { 'label_a': 'a', 'label_b': 'b', 'pair_count': 0, 'overlap_count': 0 }
        min_iou, reasoning = dsr.propose_rule( entry )
        self.assertIsNone( min_iou )

    def test_non_bimodal_anchors_near_median( self ):
        entry = { 'label_a': 'exposed_breast', 'label_b': 'exposed_buttocks', 'pair_count': 7059, 'overlap_count': 662,
                   'thresholds': {0.05:(573,662,87), 0.10:(554,662,84), 0.20:(518,662,78), 0.30:(476,662,72), 0.50:(0,662,0)},
                   'iou_median': 0.389, 'bimodal': False }
        min_iou, reasoning = dsr.propose_rule( entry )
        self.assertEqual( min_iou, 0.30 )  # nearest tested threshold to median 0.389

    def test_bimodal_picks_start_of_high_cluster( self ):
        entry = { 'label_a': 'exposed_vulva', 'label_b': 'exposed_penis', 'pair_count': 264, 'overlap_count': 157,
                   'thresholds': {0.05:(141,157,90), 0.10:(95,157,61), 0.20:(44,157,28), 0.30:(11,157,7)},
                   'iou_median': 0.147, 'bimodal': True }
        min_iou, reasoning = dsr.propose_rule( entry )
        self.assertEqual( min_iou, 0.30 )  # first threshold with pct <= 25
        self.assertIn( 'bimodal', reasoning )


class TestProposeMargin( unittest.TestCase ):

    def test_low_survival_at_zero_proposes_zero_and_flags_review( self ):
        # covered_belly-shaped: suppressing label almost never more
        # confident - a positive margin would gut the rule.
        entry = { 'label_b': 'covered_belly', 'score_thresholds': { 0.00: (0,30,0), 0.05: (0,30,0) }, 'score_delta_median': -0.251 }
        margin, reasoning = dsr.propose_margin( entry, 'exposed_breast' )
        self.assertEqual( margin, 0.00 )
        self.assertIn( 'MANUAL REVIEW', reasoning )
        self.assertIn( 'gut', reasoning )

    def test_normal_case_targets_roughly_third_surviving( self ):
        entry = { 'label_b': 'face_femme', 'score_thresholds': { 0.00: (58,274,58), 0.05: (139,274,51), 0.10: (116,274,42),
                                          0.15: (99,274,36), 0.20: (83,274,30), 0.30: (53,274,19) },
                   'score_delta_median': 0.058 }
        margin, reasoning = dsr.propose_margin( entry, 'exposed_breast' )
        # 0.15 (36%) is nearer to the 33% target than 0.20 (30%) or 0.10 (42%)
        self.assertEqual( margin, 0.15 )

    def test_no_score_thresholds_proposes_nothing( self ):
        entry = { 'score_thresholds': {}, 'score_delta_median': None }
        margin, reasoning = dsr.propose_margin( entry, 'exposed_breast' )
        self.assertIsNone( margin )
        self.assertIn( 're-run analyze_suppression_pairs.py', reasoning )

    def test_missing_median_proposes_nothing( self ):
        entry = { 'score_thresholds': { 0.00: (5,10,50) } }
        margin, reasoning = dsr.propose_margin( entry, 'exposed_breast' )
        self.assertIsNone( margin )


class TestMarginRequiredAlongsideMinIou( unittest.TestCase ):
    """
    The user's explicit instruction: margin should be required per
    suppression rule, since no default convention (apply_class_suppression's
    0.0 fallback) is a safe stand-in. main()'s proposed dict must never
    contain a rule with min_iou but no margin key.
    """

    def test_full_pipeline_never_emits_min_iou_without_margin( self ):
        block = _make_block( 'exposed_breast', 'covered_breast', 55444, 775, 0.036,
            { 0.05: (225,775,29), 0.10: (26,775,3), 0.20: (6,775,1), 0.30: (4,775,1) },
            -0.112, { 0.00: (203,775,26), 0.05: (139,775,18), 0.10: (93,775,12) } )
        entries = dsr.parse_stats_output( block )
        min_iou, _ = dsr.propose_rule( entries[0] )
        self.assertIsNotNone( min_iou )
        margin, _ = dsr.propose_margin( entries[0], entries[0]['label_a'] )
        self.assertIsNotNone( margin )  # even the "gut the rule" case still returns an explicit 0.00, never None+min_iou


if __name__ == '__main__':
    unittest.main()
