"""
test_class_promotion.py - relabelling a detection when evidence says it
is something more specific.

WHY PROMOTION EXISTS
--------------------
NudeNet reports penetration as covered_vulva, because the penis is what
is covering it. Measured on the 640m caches, covered_vulva outnumbered
exposed_vulva 1.29:1, and a quarter of covered_vulva detections score
below the label's own min_prob. Lowering covered_vulva's min_prob would
catch them and also every clothed crotch. Promotion instead asks for
corroboration: a covered_vulva with an exposed_penis on top of it is
treated as exposed_vulva, and one on its own is left alone.

WHAT THESE TESTS PIN
--------------------
  - the rule's gates: source min_prob, evidence label, score and overlap
  - the two overlap measures, and why IoU alone misses the real case
  - 'any' vs 'all' evidence
  - no double box when the model already found the target
  - additive only: nothing is ever removed, and a relabelled box keeps
    its geometry
  - stage order in the full pipeline (target's own gates apply after)
  - validation fails closed on typos, because every key is a gate
"""

import copy
import os
import sys
import unittest

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )

import betaconfig

import betautils_config as bu_config
import betautils_track as bu_track


def _box( label, x, y, w, h, score=0.6, t=1.0 ):
    return { 'class_id': label, 'score': score, 't': t,
             'x': x, 'y': y, 'w': w, 'h': h, 'size': 640 }


PENETRATION_RULE = {
    'exposed_vulva': [
        { 'from': 'covered_vulva', 'min_prob': 0.30,
          'requires': [ { 'label': 'exposed_penis', 'min_prob': 0.30,
                          'min_source_overlap': 0.10 } ] },
    ],
}


def _labels( boxes ):
    return sorted( box['class_id'] for box in boxes )


class TestRuleGates( unittest.TestCase ):

    def test_evidence_promotes_the_source( self ):
        boxes = [ _box( 'covered_vulva', 100, 100, 80, 80, 0.40 ),
                  _box( 'exposed_penis', 110, 60, 50, 160, 0.70 ) ]
        result, counts = bu_track.apply_class_promotion( boxes, PENETRATION_RULE )
        self.assertEqual( _labels( result ), [ 'exposed_penis', 'exposed_vulva' ] )
        self.assertEqual( counts['exposed_vulva<-covered_vulva']['promoted'], 1 )

    def test_no_evidence_leaves_the_source_alone( self ):
        # The clothed-crotch case. This is the whole point of requiring
        # corroboration rather than lowering covered_vulva's min_prob.
        boxes = [ _box( 'covered_vulva', 100, 100, 80, 80, 0.40 ) ]
        result, counts = bu_track.apply_class_promotion( boxes, PENETRATION_RULE )
        self.assertEqual( _labels( result ), [ 'covered_vulva' ] )
        self.assertEqual( counts['exposed_vulva<-covered_vulva']['no_evidence'], 1 )

    def test_a_source_below_min_prob_is_not_even_counted( self ):
        boxes = [ _box( 'covered_vulva', 100, 100, 80, 80, 0.20 ),
                  _box( 'exposed_penis', 110, 60, 50, 160, 0.70 ) ]
        result, counts = bu_track.apply_class_promotion( boxes, PENETRATION_RULE )
        self.assertEqual( _labels( result ), [ 'covered_vulva', 'exposed_penis' ] )
        self.assertEqual( counts, {} )

    def test_weak_evidence_does_not_count( self ):
        boxes = [ _box( 'covered_vulva', 100, 100, 80, 80, 0.40 ),
                  _box( 'exposed_penis', 110, 60, 50, 160, 0.20 ) ]
        result, _counts = bu_track.apply_class_promotion( boxes, PENETRATION_RULE )
        self.assertEqual( _labels( result ), [ 'covered_vulva', 'exposed_penis' ] )

    def test_non_overlapping_evidence_does_not_count( self ):
        boxes = [ _box( 'covered_vulva', 100, 100, 80, 80, 0.40 ),
                  _box( 'exposed_penis', 900, 500, 50, 160, 0.90 ) ]
        result, _counts = bu_track.apply_class_promotion( boxes, PENETRATION_RULE )
        self.assertIn( 'covered_vulva', _labels( result ) )

    def test_evidence_from_another_instant_does_not_count( self ):
        # Same rule as suppression: only boxes from the same sampled
        # frame are ever compared.
        boxes = [ _box( 'covered_vulva', 100, 100, 80, 80, 0.40, t=1.0 ),
                  _box( 'exposed_penis', 110, 60, 50, 160, 0.90, t=1.2 ) ]
        result, _counts = bu_track.apply_class_promotion( boxes, PENETRATION_RULE )
        self.assertIn( 'covered_vulva', _labels( result ) )


class TestOverlapMeasures( unittest.TestCase ):
    """
    min_source_overlap exists because IoU gets the motivating case wrong.
    A long penis box laid across a small vulva box covers the vulva
    completely and still scores a low IoU, because IoU divides by the
    union and the penis box is mostly elsewhere.
    """

    SOURCE = _box( 'covered_vulva', 100, 100, 60, 60, 0.40 )
    LONG_EVIDENCE = _box( 'exposed_penis', 100, 0, 60, 400, 0.80 )

    def _rule( self, **requirement ):
        requirement.setdefault( 'label', 'exposed_penis' )
        return { 'exposed_vulva': [ { 'from': 'covered_vulva',
                                      'requires': [ requirement ] } ] }

    def test_the_fixture_really_has_low_iou_and_full_coverage( self ):
        self.assertLess( bu_track.intersection_over_union(
            self.SOURCE, self.LONG_EVIDENCE ), 0.2 )
        self.assertAlmostEqual( bu_track._intersection_over_source(
            self.SOURCE, self.LONG_EVIDENCE ), 1.0 )

    def test_iou_misses_full_coverage_by_a_long_box( self ):
        result, _ = bu_track.apply_class_promotion(
            [ dict( self.SOURCE ), dict( self.LONG_EVIDENCE ) ], self._rule( min_iou=0.3 ) )
        self.assertIn( 'covered_vulva', _labels( result ) )

    def test_source_overlap_catches_it( self ):
        result, _ = bu_track.apply_class_promotion(
            [ dict( self.SOURCE ), dict( self.LONG_EVIDENCE ) ],
            self._rule( min_source_overlap=0.5 ) )
        self.assertIn( 'exposed_vulva', _labels( result ) )

    def test_both_measures_must_hold_when_both_are_given( self ):
        result, _ = bu_track.apply_class_promotion(
            [ dict( self.SOURCE ), dict( self.LONG_EVIDENCE ) ],
            self._rule( min_source_overlap=0.5, min_iou=0.3 ) )
        self.assertIn( 'covered_vulva', _labels( result ) )

    def test_no_measure_means_any_overlap( self ):
        touching = _box( 'exposed_penis', 155, 155, 40, 40, 0.8 )
        result, _ = bu_track.apply_class_promotion(
            [ dict( self.SOURCE ), touching ], self._rule() )
        self.assertIn( 'exposed_vulva', _labels( result ) )


class TestEvidenceModes( unittest.TestCase ):

    def _rule( self, mode ):
        return { 'exposed_vulva': [ {
            'from': 'covered_vulva', 'requires_mode': mode,
            'requires': [ { 'label': 'exposed_penis' },
                          { 'label': 'exposed_buttocks' } ] } ] }

    def _boxes( self, *evidence ):
        return [ _box( 'covered_vulva', 100, 100, 80, 80, 0.5 ) ] + [
            _box( label, 110, 110, 60, 60, 0.8 ) for label in evidence ]

    def test_any_needs_one( self ):
        result, _ = bu_track.apply_class_promotion(
            self._boxes( 'exposed_penis' ), self._rule( 'any' ) )
        self.assertIn( 'exposed_vulva', _labels( result ) )

    def test_all_needs_every_one( self ):
        result, _ = bu_track.apply_class_promotion(
            self._boxes( 'exposed_penis' ), self._rule( 'all' ) )
        self.assertIn( 'covered_vulva', _labels( result ) )
        result, _ = bu_track.apply_class_promotion(
            self._boxes( 'exposed_penis', 'exposed_buttocks' ), self._rule( 'all' ) )
        self.assertIn( 'exposed_vulva', _labels( result ) )

    def test_an_empty_requires_promotes_unconditionally( self ):
        # Legitimate but blunt: a source-label min_prob and nothing else.
        rule = { 'exposed_vulva': [ { 'from': 'covered_vulva', 'min_prob': 0.45 } ] }
        result, _ = bu_track.apply_class_promotion( self._boxes(), rule )
        self.assertIn( 'exposed_vulva', _labels( result ) )


class TestNothingIsLostOrDoubled( unittest.TestCase ):

    def test_promotion_never_changes_the_box_count( self ):
        boxes = [ _box( 'covered_vulva', 100, 100, 80, 80, 0.40 ),
                  _box( 'exposed_penis', 110, 60, 50, 160, 0.70 ),
                  _box( 'exposed_breast', 600, 100, 90, 90, 0.80 ) ]
        result, _ = bu_track.apply_class_promotion( boxes, PENETRATION_RULE )
        self.assertEqual( len( result ), len( boxes ) )

    def test_a_promoted_box_keeps_its_geometry_and_score( self ):
        source = _box( 'covered_vulva', 101, 102, 83, 84, 0.41 )
        boxes = [ source, _box( 'exposed_penis', 110, 60, 50, 160, 0.70 ) ]
        result, _ = bu_track.apply_class_promotion( boxes, PENETRATION_RULE )
        promoted = [ box for box in result if box['class_id'] == 'exposed_vulva' ][0]
        for key in ( 'x', 'y', 'w', 'h', 'score', 't' ):
            self.assertEqual( promoted[key], source[key] )
        self.assertEqual( promoted['promoted_from'], 'covered_vulva' )

    def test_the_input_is_not_mutated( self ):
        boxes = [ _box( 'covered_vulva', 100, 100, 80, 80, 0.40 ),
                  _box( 'exposed_penis', 110, 60, 50, 160, 0.70 ) ]
        before = copy.deepcopy( boxes )
        bu_track.apply_class_promotion( boxes, PENETRATION_RULE )
        self.assertEqual( boxes, before )

    def test_an_already_found_target_is_not_duplicated( self ):
        # Two boxes for one object means two tracks and two styles.
        boxes = [ _box( 'covered_vulva', 100, 100, 80, 80, 0.40 ),
                  _box( 'exposed_vulva', 102, 101, 80, 80, 0.35 ),
                  _box( 'exposed_penis', 110, 60, 50, 160, 0.70 ) ]
        result, counts = bu_track.apply_class_promotion( boxes, PENETRATION_RULE )
        self.assertEqual( _labels( result ).count( 'exposed_vulva' ), 1 )
        self.assertEqual( counts['exposed_vulva<-covered_vulva']['already_covered'], 1 )

    def test_a_promoted_box_cannot_be_evidence_in_the_same_pass( self ):
        # Rule order must not change the outcome, so evidence is always
        # read from ORIGINAL labels.
        rules = {
            'exposed_vulva': [ { 'from': 'covered_vulva',
                                 'requires': [ { 'label': 'exposed_penis' } ] } ],
            'exposed_anus': [ { 'from': 'exposed_buttocks',
                                'requires': [ { 'label': 'exposed_vulva' } ] } ],
        }
        boxes = [ _box( 'covered_vulva', 100, 100, 80, 80, 0.5 ),
                  _box( 'exposed_penis', 110, 60, 50, 160, 0.8 ),
                  _box( 'exposed_buttocks', 90, 90, 120, 120, 0.8 ) ]
        result, _ = bu_track.apply_class_promotion( boxes, rules )
        self.assertIn( 'exposed_buttocks', _labels( result ) )


class TestPipelineOrder( unittest.TestCase ):
    """Promotion runs first, so the promoted box meets the TARGET's gates."""

    def setUp( self ):
        self._saved = copy.deepcopy( betaconfig.detector_backend )
        self._saved_items = list( betaconfig.items_to_censor )
        self.backend = betaconfig.detector_backend['selected']
        betaconfig.detector_backend = copy.deepcopy( betaconfig.detector_backend )
        block = betaconfig.detector_backend[self.backend]
        block['class_promotion'] = copy.deepcopy( PENETRATION_RULE )
        block['class_suppression'] = {}
        block.pop( 'profiles', None )
        overrides = block.setdefault( 'item_overrides', {} )
        overrides.setdefault( 'exposed_vulva', {} ).update(
            min_prob=0.20, min_track_hits=1, width_area_safety=0.0, height_area_safety=0.0 )
        betaconfig.items_to_censor = [ 'exposed_vulva' ]
        bu_config.invalidate_config_caches()

    def tearDown( self ):
        betaconfig.detector_backend = self._saved
        betaconfig.items_to_censor = self._saved_items
        bu_config.invalidate_config_caches()

    def _raw( self, covered_score, frames=6 ):
        raw = []
        for index in range( frames ):
            t = index / 9.0
            raw.append( _box( 'covered_vulva', 800, 500, 80, 80, covered_score, t ) )
            raw.append( _box( 'exposed_penis', 810, 450, 50, 180, 0.8, t ) )
        return raw

    def test_an_uncensored_source_gets_censored_as_the_target( self ):
        # covered_vulva is NOT in items_to_censor here, so without
        # promotion nothing renders.
        boxes, stats = bu_track.prepare_boxes_for_render(
            self._raw( 0.40 ), 1920, 1080, backend_name=self.backend )
        self.assertTrue( boxes )
        self.assertTrue( all( box['label'] == 'exposed_vulva' for box in boxes ) )
        self.assertEqual( stats['promoted'], 6 )

    def test_the_targets_min_prob_still_applies( self ):
        # A promoted box keeps the source's score, so a rule min_prob set
        # below the target's own min_prob promotes boxes that are then
        # dropped. Worth knowing when a rule reports promotions and the
        # render shows nothing.
        overrides = betaconfig.detector_backend[self.backend]['item_overrides']
        overrides['exposed_vulva']['min_prob'] = 0.50
        bu_config.invalidate_config_caches()
        boxes, stats = bu_track.prepare_boxes_for_render(
            self._raw( 0.40 ), 1920, 1080, backend_name=self.backend )
        self.assertEqual( stats['promoted'], 6 )
        self.assertEqual( boxes, [] )

    def test_the_targets_suppression_still_applies( self ):
        block = betaconfig.detector_backend[self.backend]
        block['class_suppression'] = { 'exposed_vulva': [
            { 'suppressed_by': 'exposed_penis', 'margin': 0.1, 'min_iou': 0.0 } ] }
        bu_config.invalidate_config_caches()
        boxes, stats = bu_track.prepare_boxes_for_render(
            self._raw( 0.40 ), 1920, 1080, backend_name=self.backend )
        self.assertEqual( stats['promoted'], 6 )
        self.assertEqual( boxes, [] )


class TestValidation( unittest.TestCase ):
    """Every key is a gate, so a typo must fail rather than fail open."""

    def setUp( self ):
        self._saved = copy.deepcopy( betaconfig.detector_backend )
        self.backend = betaconfig.detector_backend['selected']
        betaconfig.detector_backend = copy.deepcopy( betaconfig.detector_backend )
        bu_config.invalidate_config_caches()

    def tearDown( self ):
        betaconfig.detector_backend = self._saved
        bu_config.invalidate_config_caches()

    def _errors_for( self, promotion ):
        betaconfig.detector_backend[self.backend]['class_promotion'] = promotion
        bu_config.invalidate_config_caches()
        return [ error for error in bu_config._collect_validation_errors()
                 if 'class_promotion' in error ]

    def test_the_shipped_shape_is_valid( self ):
        self.assertEqual( self._errors_for( copy.deepcopy( PENETRATION_RULE ) ), [] )

    def test_a_misspelled_rule_key_is_an_error( self ):
        rule = { 'exposed_vulva': [ { 'from': 'covered_vulva', 'min_prb': 0.3 } ] }
        self.assertTrue( self._errors_for( rule ) )

    def test_a_misspelled_requires_is_an_error( self ):
        # 'require' would otherwise mean "no evidence needed": promote
        # every covered_vulva. Fails open in the worst direction.
        rule = { 'exposed_vulva': [ { 'from': 'covered_vulva',
                                      'require': [ { 'label': 'exposed_penis' } ] } ] }
        self.assertTrue( self._errors_for( rule ) )

    def test_a_misspelled_evidence_key_is_an_error( self ):
        rule = { 'exposed_vulva': [ { 'from': 'covered_vulva', 'requires': [
            { 'label': 'exposed_penis', 'min_source_overlp': 0.1 } ] } ] }
        self.assertTrue( self._errors_for( rule ) )

    def test_an_unknown_label_is_an_error( self ):
        rule = { 'exposed_vulva': [ { 'from': 'covred_vulva' } ] }
        self.assertTrue( self._errors_for( rule ) )

    def test_an_uncensored_target_is_an_error( self ):
        # Relabelling into a label nothing renders deletes the source's
        # censor if the source was censored.
        rule = { 'exposed_anus': [ { 'from': 'covered_vulva' } ] }
        self.assertTrue( self._errors_for( rule ) )

    def test_an_out_of_range_threshold_is_an_error( self ):
        rule = { 'exposed_vulva': [ { 'from': 'covered_vulva', 'min_prob': 30 } ] }
        self.assertTrue( self._errors_for( rule ) )

    def test_evidence_that_also_suppresses_the_target_warns( self ):
        block = betaconfig.detector_backend[self.backend]
        block['class_promotion'] = { 'exposed_vulva': [ { 'from': 'covered_vulva',
            'requires': [ { 'label': 'exposed_anus' } ] } ] }
        block['class_suppression'] = { 'exposed_vulva': [
            { 'suppressed_by': 'exposed_anus', 'margin': 0.1, 'min_iou': 0.05 } ] }
        bu_config.invalidate_config_caches()
        warnings = bu_config._collect_validation_warnings()
        self.assertTrue( any( 'exposed_anus' in warning and 'promotion' in warning
                              for warning in warnings ) )


class TestOverlapAnywhere( unittest.TestCase ):
    """
    overlap: 'anywhere' - co-presence in the frame IS the evidence.

    WHY THIS EXISTS
    ---------------
    "Is this clothed crotch in a penetration scene" is a fact about the
    SCENE, not about which pixels touch which. The penis may be elsewhere
    in frame; the covered_vulva box may sit beside rather than under it.

    Measured on 9 real caches (488k detections, 9,302 covered_vulva above
    0.3): only 18.9% had any of exposed_penis / exposed_vulva /
    exposed_anus live at the same instant, and of those, 91% had ZERO
    pixel overlap with it. The overlap-based rule therefore promoted 1
    detection across the whole set. Requiring only co-presence promotes
    1,452.

    The spatial tests are not wrong - min_iou and min_source_overlap
    answer "are these the same object", which is the right question for a
    duplicate. They are the wrong question for scene context.
    """

    ANYWHERE_RULE = {
        'exposed_vulva': [
            { 'from': 'covered_vulva', 'min_prob': 0.30,
              'requires': [ { 'label': 'exposed_penis', 'min_prob': 0.30,
                              'overlap': 'anywhere' } ] },
        ],
    }

    def test_evidence_on_the_far_side_of_the_frame_still_promotes( self ):
        # The case the overlap rule could not express. These two boxes do
        # not touch at all.
        boxes = [ _box( 'covered_vulva', 100, 100, 40, 40 ),
                  _box( 'exposed_penis', 1500, 800, 60, 120 ) ]
        result, counts = bu_track.apply_class_promotion( boxes, self.ANYWHERE_RULE )
        labels = sorted( box['class_id'] for box in result )
        self.assertEqual( labels, [ 'exposed_penis', 'exposed_vulva' ] )
        self.assertEqual( counts['exposed_vulva<-covered_vulva']['promoted'], 1 )

    def test_the_same_pair_does_NOT_promote_under_an_overlap_rule( self ):
        # Pins the contrast, so the new mode is provably doing something
        # the old one could not.
        boxes = [ _box( 'covered_vulva', 100, 100, 40, 40 ),
                  _box( 'exposed_penis', 1500, 800, 60, 120 ) ]
        result, counts = bu_track.apply_class_promotion( boxes, PENETRATION_RULE )
        self.assertEqual( sorted( box['class_id'] for box in result ),
                          [ 'covered_vulva', 'exposed_penis' ] )
        self.assertEqual( counts['exposed_vulva<-covered_vulva']['no_evidence'], 1 )

    def test_no_evidence_in_frame_still_does_not_promote( self ):
        # 'anywhere' must not become 'always'. A clothed crotch alone is
        # the 82.8% case that should end up uncensored.
        boxes = [ _box( 'covered_vulva', 100, 100, 40, 40 ),
                  _box( 'exposed_belly', 200, 300, 200, 150 ) ]
        result, counts = bu_track.apply_class_promotion( boxes, self.ANYWHERE_RULE )
        self.assertEqual( sorted( box['class_id'] for box in result ),
                          [ 'covered_vulva', 'exposed_belly' ] )
        self.assertEqual( counts['exposed_vulva<-covered_vulva']['no_evidence'], 1 )

    def test_evidence_in_a_DIFFERENT_frame_does_not_promote( self ):
        # 'anywhere' is anywhere in SPACE, not anywhere in time. Evidence
        # is still same-instant only.
        boxes = [ _box( 'covered_vulva', 100, 100, 40, 40, t=1.0 ),
                  _box( 'exposed_penis', 1500, 800, 60, 120, t=9.0 ) ]
        result, counts = bu_track.apply_class_promotion( boxes, self.ANYWHERE_RULE )
        self.assertEqual( sorted( box['class_id'] for box in result ),
                          [ 'covered_vulva', 'exposed_penis' ] )
        self.assertEqual( counts['exposed_vulva<-covered_vulva']['no_evidence'], 1 )

    def test_evidence_min_prob_is_still_enforced( self ):
        boxes = [ _box( 'covered_vulva', 100, 100, 40, 40 ),
                  _box( 'exposed_penis', 1500, 800, 60, 120, score=0.10 ) ]
        _result, counts = bu_track.apply_class_promotion( boxes, self.ANYWHERE_RULE )
        self.assertEqual( counts['exposed_vulva<-covered_vulva']['no_evidence'], 1 )

    def test_duplicate_suppression_still_applies( self ):
        # A real exposed_vulva already covering this box must still block
        # the promotion, or tracking gets two boxes for one object.
        boxes = [ _box( 'covered_vulva', 100, 100, 40, 40 ),
                  _box( 'exposed_vulva', 100, 100, 40, 40 ),
                  _box( 'exposed_penis', 1500, 800, 60, 120 ) ]
        _result, counts = bu_track.apply_class_promotion( boxes, self.ANYWHERE_RULE )
        self.assertEqual( counts['exposed_vulva<-covered_vulva']['already_covered'], 1 )

    def test_any_of_several_evidence_labels_is_enough( self ):
        rule = { 'exposed_vulva': [
            { 'from': 'covered_vulva', 'min_prob': 0.30, 'requires': [
                { 'label': 'exposed_penis', 'min_prob': 0.30, 'overlap': 'anywhere' },
                { 'label': 'exposed_anus',  'min_prob': 0.30, 'overlap': 'anywhere' } ] } ] }
        boxes = [ _box( 'covered_vulva', 100, 100, 40, 40 ),
                  _box( 'exposed_anus', 900, 700, 50, 50 ) ]
        _result, counts = bu_track.apply_class_promotion( boxes, rule )
        self.assertEqual( counts['exposed_vulva<-covered_vulva']['promoted'], 1 )

    def test_requires_mode_all_still_means_all( self ):
        rule = { 'exposed_vulva': [
            { 'from': 'covered_vulva', 'min_prob': 0.30, 'requires_mode': 'all',
              'requires': [
                { 'label': 'exposed_penis', 'min_prob': 0.30, 'overlap': 'anywhere' },
                { 'label': 'exposed_anus',  'min_prob': 0.30, 'overlap': 'anywhere' } ] } ] }
        boxes = [ _box( 'covered_vulva', 100, 100, 40, 40 ),
                  _box( 'exposed_anus', 900, 700, 50, 50 ) ]
        _result, counts = bu_track.apply_class_promotion( boxes, rule )
        self.assertEqual( counts['exposed_vulva<-covered_vulva']['no_evidence'], 1 )


class TestOverlapAnywhereValidation( unittest.TestCase ):

    def setUp( self ):
        self._saved = copy.deepcopy( betaconfig.detector_backend )
        self.backend = betaconfig.detector_backend['selected']

    def tearDown( self ):
        betaconfig.detector_backend = self._saved
        bu_config.invalidate_config_caches()

    def _install( self, requirement ):
        betaconfig.detector_backend = copy.deepcopy( betaconfig.detector_backend )
        betaconfig.detector_backend[self.backend]['class_promotion'] = {
            'exposed_vulva': [ { 'from': 'covered_vulva', 'min_prob': 0.30,
                                 'requires': [ requirement ] } ] }
        bu_config.invalidate_config_caches()

    def test_anywhere_is_accepted( self ):
        self._install( { 'label': 'exposed_penis', 'min_prob': 0.3,
                         'overlap': 'anywhere' } )
        self.assertEqual( [ e for e in bu_config._collect_validation_errors()
                            if 'overlap' in e ], [] )

    def test_an_unknown_overlap_mode_is_rejected( self ):
        self._install( { 'label': 'exposed_penis', 'min_prob': 0.3,
                         'overlap': 'somewhere' } )
        self.assertTrue( any( 'overlap' in e
                              for e in bu_config._collect_validation_errors() ) )

    def test_anywhere_combined_with_a_spatial_test_is_rejected( self ):
        # Silently ignoring one of two contradictory settings is how a
        # rule ends up not doing what its config says.
        self._install( { 'label': 'exposed_penis', 'min_prob': 0.3,
                         'overlap': 'anywhere', 'min_source_overlap': 0.1 } )
        self.assertTrue( any( 'overlap' in e
                              for e in bu_config._collect_validation_errors() ) )


if __name__ == '__main__':
    unittest.main()

