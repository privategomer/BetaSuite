"""
test_class_suppression_margin_required.py - regression coverage for
betautils_config._validate_class_suppression's margin-required check,
added 2026-09-16 per explicit user direction: apply_class_suppression's
own fallback when margin is absent (betatv.py: rule.get('margin', 0.0))
is 0.0, the LOOSEST possible gate (fires whenever the suppressing
label's score is even just >= the suppressed label's) - not a neutral
"off" default. A rule author omitting margin is silently getting the
loosest setting, not skipping an optional refinement, so validate_config
must catch a rule missing it the same way it already catches one missing
'suppressed_by'.

Also covers the negative-margin hard error, added the same day in the
same conversation: margin < 0 would let the suppressing label be LESS
confident than the label it's overriding and still win, inverting the
whole point of the gate - no legitimate use case, confirmed with the
user, so this is always an error (unlike margin == 0.0 exactly, which
is a real value with real use cases - see
test_class_suppression_zero_margin.py for that one's warning-not-error
treatment; the user's own framing was explicit: "<0 is error; ==0 is
warning").
"""

import unittest

import betaconfig
import betautils_config as bu_config


class TestClassSuppressionMarginRequired( unittest.TestCase ):

    def setUp( self ):
        self._had_detector_backend = hasattr( betaconfig, 'detector_backend' )
        if self._had_detector_backend:
            self._orig_detector_backend = betaconfig.detector_backend

    def tearDown( self ):
        if self._had_detector_backend:
            betaconfig.detector_backend = self._orig_detector_backend
        elif hasattr( betaconfig, 'detector_backend' ):
            del betaconfig.detector_backend

    def test_rule_missing_margin_is_flagged( self ):
        betaconfig.detector_backend = {
            'selected': 'nudenet_v3',
            'nudenet_v3': { 'class_suppression': {
                'exposed_breast': [ { 'suppressed_by': 'covered_breast', 'min_iou': 0.05 } ],
            } },
            'retinanet_v2': {},
        }
        errors = []
        bu_config._validate_class_suppression( errors, { 'exposed_breast', 'covered_breast' } )
        self.assertTrue( any( "missing required key 'margin'" in e for e in errors ), errors )

    def test_rule_missing_min_iou_is_flagged( self ):
        betaconfig.detector_backend = {
            'selected': 'nudenet_v3',
            'nudenet_v3': { 'class_suppression': {
                'exposed_breast': [ { 'suppressed_by': 'covered_breast', 'margin': 0.0 } ],
            } },
            'retinanet_v2': {},
        }
        errors = []
        bu_config._validate_class_suppression( errors, { 'exposed_breast', 'covered_breast' } )
        self.assertTrue( any( "missing required key 'min_iou'" in e for e in errors ), errors )

    def test_rule_with_both_margin_and_min_iou_produces_no_margin_error( self ):
        betaconfig.detector_backend = {
            'selected': 'nudenet_v3',
            'nudenet_v3': { 'class_suppression': {
                'exposed_breast': [ { 'suppressed_by': 'covered_breast', 'margin': 0.00, 'min_iou': 0.05 } ],
            } },
            'retinanet_v2': {},
        }
        errors = []
        bu_config._validate_class_suppression( errors, { 'exposed_breast', 'covered_breast' } )
        self.assertEqual( errors, [] )

    def test_explicit_zero_margin_is_not_treated_as_missing( self ):
        # margin: 0.00 is a real, deliberately-chosen value (see
        # derive_suppression_rules.py's propose_margin "gut the rule"
        # case) - must not be confused with the key being absent.
        betaconfig.detector_backend = {
            'selected': 'nudenet_v3',
            'nudenet_v3': { 'class_suppression': {
                'exposed_vulva': [ { 'suppressed_by': 'exposed_anus', 'margin': 0.0, 'min_iou': 0.05 } ],
            } },
            'retinanet_v2': {},
        }
        errors = []
        bu_config._validate_class_suppression( errors, { 'exposed_vulva', 'exposed_anus' } )
        self.assertFalse( any( 'margin' in e for e in errors ), errors )

    def test_non_numeric_margin_is_flagged( self ):
        betaconfig.detector_backend = {
            'selected': 'nudenet_v3',
            'nudenet_v3': { 'class_suppression': {
                'exposed_breast': [ { 'suppressed_by': 'covered_breast', 'margin': 'loose', 'min_iou': 0.05 } ],
            } },
            'retinanet_v2': {},
        }
        errors = []
        bu_config._validate_class_suppression( errors, { 'exposed_breast', 'covered_breast' } )
        self.assertTrue( any( "['margin'] must be a number" in e for e in errors ), errors )

    def test_negative_margin_is_flagged( self ):
        betaconfig.detector_backend = {
            'selected': 'nudenet_v3',
            'nudenet_v3': { 'class_suppression': {
                'exposed_breast': [ { 'suppressed_by': 'covered_breast', 'margin': -0.05, 'min_iou': 0.05 } ],
            } },
            'retinanet_v2': {},
        }
        errors = []
        bu_config._validate_class_suppression( errors, { 'exposed_breast', 'covered_breast' } )
        self.assertTrue( any( 'is negative' in e for e in errors ), errors )

    def test_zero_margin_is_not_flagged_as_negative( self ):
        # 0.0 is a real, valid value (just a low one) - must not trip
        # the "< 0" check, only genuinely negative numbers should.
        betaconfig.detector_backend = {
            'selected': 'nudenet_v3',
            'nudenet_v3': { 'class_suppression': {
                'exposed_breast': [ { 'suppressed_by': 'covered_breast', 'margin': 0.0, 'min_iou': 0.05 } ],
            } },
            'retinanet_v2': {},
        }
        errors = []
        bu_config._validate_class_suppression( errors, { 'exposed_breast', 'covered_breast' } )
        self.assertFalse( any( 'is negative' in e for e in errors ), errors )

    def test_real_shipped_nudenet_v3_and_retinanet_v2_rules_pass( self ):
        # full-config smoke test: whatever's actually in betaconfig.py
        # right now must validate clean, both backends - this is the
        # real regression guard for "someone edits betaconfig.py and
        # forgets margin on a new rule."
        errors = []
        bu_config._validate_class_suppression( errors, set( __import__('betaconst').classes.keys() ) )
        self.assertEqual( errors, [] )


if __name__ == '__main__':
    unittest.main()
