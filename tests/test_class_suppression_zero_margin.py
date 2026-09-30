"""
test_class_suppression_zero_margin.py - regression coverage for
betautils_config._validate_class_suppression_zero_margin, added
2026-09-16 alongside the real config edit that raised every nudenet_v3
class_suppression rule that had been exactly margin=0.00 up to 0.10 by
hand (one pair went to 0.15).

Why this is a WARNING and not an error (see
_validate_class_suppression_zero_margin's own docstring in
betautils_config.py for the fuller reasoning): margin=0.0 is a real,
sometimes-correct value that derive_suppression_rules.py legitimately
proposes when the suppressing label rarely beats the suppressed label's
confidence at all (a positive margin there would gut the rule, not
refine it - see CONFIG_REFERENCE.md's nudenet_v3 margin section). The
check is deliberately margin == 0.0 exactly, not "below some floor" -
the user's own correction (2026-09-16) was explicit: "<0 is error; ==0
is warning" - a rule someone deliberately sets to something in between,
like 0.05, is a reviewed, in-between choice this warning has no opinion
on, not a case it should also flag.
"""

import unittest

import betaconfig
import betautils_config as bu_config


class TestClassSuppressionZeroMargin( unittest.TestCase ):

    def setUp( self ):
        self._had_detector_backend = hasattr( betaconfig, 'detector_backend' )
        if self._had_detector_backend:
            self._orig_detector_backend = betaconfig.detector_backend

    def tearDown( self ):
        if self._had_detector_backend:
            betaconfig.detector_backend = self._orig_detector_backend
        elif hasattr( betaconfig, 'detector_backend' ):
            del betaconfig.detector_backend

    def test_margin_zero_is_warned( self ):
        betaconfig.detector_backend = {
            'selected': 'nudenet_v3',
            'nudenet_v3': { 'class_suppression': {
                'exposed_breast': [ { 'suppressed_by': 'covered_breast', 'margin': 0.00, 'min_iou': 0.05 } ],
            } },
            'retinanet_v2': {},
        }
        warnings = []
        bu_config._validate_class_suppression_zero_margin( warnings, { 'exposed_breast', 'covered_breast' } )
        self.assertEqual( len(warnings), 1 )
        self.assertIn( 'margin', warnings[0] )

    def test_margin_slightly_above_zero_is_not_warned( self ):
        # the corrected semantics: this check is exact-zero only, NOT a
        # "below some floor" check - 0.05 (an in-between, reviewed value)
        # must not trip it, even though it's still low.
        betaconfig.detector_backend = {
            'selected': 'nudenet_v3',
            'nudenet_v3': { 'class_suppression': {
                'exposed_vulva': [ { 'suppressed_by': 'exposed_anus', 'margin': 0.05, 'min_iou': 0.05 } ],
            } },
            'retinanet_v2': {},
        }
        warnings = []
        bu_config._validate_class_suppression_zero_margin( warnings, { 'exposed_vulva', 'exposed_anus' } )
        self.assertEqual( warnings, [] )

    def test_margin_at_ten_is_not_warned( self ):
        betaconfig.detector_backend = {
            'selected': 'nudenet_v3',
            'nudenet_v3': { 'class_suppression': {
                'exposed_breast': [ { 'suppressed_by': 'covered_breast', 'margin': 0.10, 'min_iou': 0.05 } ],
            } },
            'retinanet_v2': {},
        }
        warnings = []
        bu_config._validate_class_suppression_zero_margin( warnings, { 'exposed_breast', 'covered_breast' } )
        self.assertEqual( warnings, [] )

    def test_negative_margin_is_not_double_warned_here( self ):
        # negative margin is a hard ERROR from _validate_class_suppression,
        # not this warning check's job - this function should skip it
        # silently rather than producing a redundant/confusing warning
        # on top of that error.
        betaconfig.detector_backend = {
            'selected': 'nudenet_v3',
            'nudenet_v3': { 'class_suppression': {
                'exposed_breast': [ { 'suppressed_by': 'covered_breast', 'margin': -0.05, 'min_iou': 0.05 } ],
            } },
            'retinanet_v2': {},
        }
        warnings = []
        bu_config._validate_class_suppression_zero_margin( warnings, { 'exposed_breast', 'covered_breast' } )
        self.assertEqual( warnings, [] )

    def test_missing_margin_is_not_double_warned_here( self ):
        # missing margin is _validate_class_suppression's error to raise,
        # not this function's - must be skipped here too, not crash or
        # duplicate-flag.
        betaconfig.detector_backend = {
            'selected': 'nudenet_v3',
            'nudenet_v3': { 'class_suppression': {
                'exposed_breast': [ { 'suppressed_by': 'covered_breast', 'min_iou': 0.05 } ],
            } },
            'retinanet_v2': {},
        }
        warnings = []
        bu_config._validate_class_suppression_zero_margin( warnings, { 'exposed_breast', 'covered_breast' } )
        self.assertEqual( warnings, [] )

    def test_warnings_never_block_validate_config( self ):
        # the real point of this being a warning channel: validate_config
        # must still succeed (no SystemExit) even with a margin=0.0 rule
        # present, as long as nothing else is a hard error.
        #
        # retinanet_v2 needs an explicit picture_sizes here: it declares
        # no native size, and since 2.5 there is no shared
        # betaconfig.picture_sizes to fall back on, so an empty block
        # would be a genuine hard error and mask what this test checks.
        betaconfig.detector_backend = {
            'selected': 'nudenet_v3',
            'nudenet_v3': { 'class_suppression': {
                'exposed_breast': [ { 'suppressed_by': 'covered_breast', 'margin': 0.00, 'min_iou': 0.05 } ],
            } },
            'retinanet_v2': { 'picture_sizes': [ 1280 ] },
        }
        try:
            bu_config.validate_config()
        except SystemExit:
            self.fail( "validate_config() raised SystemExit on a warning-only condition (margin == 0.0)" )

    def test_real_shipped_config_has_no_zero_margin_rules( self ):
        # regression pin: the user's 2026-09-16 hand edit raised every
        # nudenet_v3 rule that had been exactly 0.00 up to 0.10 - the
        # live config should currently produce zero warnings from this
        # check. If this starts failing because a new rule was added at
        # 0.00, that's this warning correctly doing its job, not a test
        # bug - update the config or explicitly accept the warning,
        # don't just delete this test.
        warnings = []
        bu_config._validate_class_suppression_zero_margin( warnings, set( __import__('betaconst').classes.keys() ) )
        self.assertEqual( warnings, [], warnings )


if __name__ == '__main__':
    unittest.main()
