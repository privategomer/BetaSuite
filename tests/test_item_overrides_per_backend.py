"""
test_item_overrides_per_backend.py - regression coverage for
bu_detector.get_item_overrides and its validation
(betautils_config._validate_backend_item_overrides), added 2026-09-16
alongside making item_overrides backend-tunable.

Why this needed its own dedicated test: unlike get_class_suppression
(a wholesale per-backend replacement, already covered by existing
tests), get_item_overrides does a per-KEY merge - the shared block's
censor_style/censor_shape must survive untouched even when a backend
sets its own track_max_gap for the same label, and a key set in a
backend's block that ISN'T backend-tunable must be ignored by the
resolver and flagged by validation, not silently applied. None of the
existing tests exercised any of that.
"""

import unittest

import betaconfig
import betautils_config as bu_config
import betautils_detector as bu_detector


class TestGetItemOverrides( unittest.TestCase ):

    def setUp( self ):
        self._had_item_overrides = hasattr( betaconfig, 'item_overrides' )
        if self._had_item_overrides:
            self._orig_item_overrides = betaconfig.item_overrides
        self._had_detector_backend = hasattr( betaconfig, 'detector_backend' )
        if self._had_detector_backend:
            self._orig_detector_backend = betaconfig.detector_backend

    def tearDown( self ):
        if self._had_item_overrides:
            betaconfig.item_overrides = self._orig_item_overrides
        elif hasattr( betaconfig, 'item_overrides' ):
            del betaconfig.item_overrides
        if self._had_detector_backend:
            betaconfig.detector_backend = self._orig_detector_backend
        elif hasattr( betaconfig, 'detector_backend' ):
            del betaconfig.detector_backend

    def test_no_backend_override_falls_back_to_shared( self ):
        betaconfig.item_overrides = { 'exposed_breast': { 'track_max_gap': 4.5, 'censor_shape': 'circle' } }
        betaconfig.detector_backend = { 'selected': 'nudenet_v3', 'nudenet_v3': {}, 'retinanet_v2': {} }
        resolved = bu_detector.get_item_overrides( 'exposed_breast', 'nudenet_v3' )
        self.assertEqual( resolved['track_max_gap'], 4.5 )
        self.assertEqual( resolved['censor_shape'], 'circle' )

    def test_backend_override_wins_for_tunable_key( self ):
        betaconfig.item_overrides = { 'exposed_breast': { 'track_max_gap': 4.5, 'censor_shape': 'circle' } }
        betaconfig.detector_backend = {
            'selected': 'nudenet_v3',
            'nudenet_v3': { 'item_overrides': { 'exposed_breast': { 'track_max_gap': 2.0 } } },
            'retinanet_v2': {},
        }
        nudenet_resolved = bu_detector.get_item_overrides( 'exposed_breast', 'nudenet_v3' )
        retinanet_resolved = bu_detector.get_item_overrides( 'exposed_breast', 'retinanet_v2' )
        self.assertEqual( nudenet_resolved['track_max_gap'], 2.0 )
        self.assertEqual( retinanet_resolved['track_max_gap'], 4.5 )

    def test_censor_style_and_shape_never_overridden_per_backend( self ):
        # even if get_item_overrides is asked to resolve for a backend
        # whose item_overrides block (invalidly) sets censor_shape, the
        # resolver itself must ignore it - censor_shape/censor_style stay
        # shared no matter what. Validation (tested separately below) is
        # what surfaces this as an error to fix, but the resolver must
        # never silently apply it either way.
        betaconfig.item_overrides = { 'exposed_breast': { 'censor_shape': 'circle' } }
        betaconfig.detector_backend = {
            'selected': 'nudenet_v3',
            'nudenet_v3': { 'item_overrides': { 'exposed_breast': { 'censor_shape': 'box' } } },
            'retinanet_v2': {},
        }
        resolved = bu_detector.get_item_overrides( 'exposed_breast', 'nudenet_v3' )
        self.assertEqual( resolved['censor_shape'], 'circle' )  # shared value, NOT the backend's 'box'

    def test_backend_override_for_label_not_in_shared_block( self ):
        # a label with no shared override at all can still get a
        # backend-specific one - resolved result is just that backend's
        # tunable keys, no crash on a missing shared entry.
        betaconfig.item_overrides = {}
        betaconfig.detector_backend = {
            'selected': 'nudenet_v3',
            'nudenet_v3': { 'item_overrides': { 'exposed_vulva': { 'min_prob': 0.3 } } },
            'retinanet_v2': {},
        }
        resolved = bu_detector.get_item_overrides( 'exposed_vulva', 'nudenet_v3' )
        self.assertEqual( resolved['min_prob'], 0.3 )

    def test_defaults_to_selected_backend( self ):
        betaconfig.item_overrides = { 'exposed_breast': {} }
        betaconfig.detector_backend = {
            'selected': 'nudenet_v3',
            'nudenet_v3': { 'item_overrides': { 'exposed_breast': { 'track_max_gap': 2.0 } } },
            'retinanet_v2': {},
        }
        resolved = bu_detector.get_item_overrides( 'exposed_breast' )  # no explicit backend_name
        self.assertEqual( resolved['track_max_gap'], 2.0 )


class TestBackendItemOverridesValidation( unittest.TestCase ):

    def setUp( self ):
        self._had_item_overrides = hasattr( betaconfig, 'item_overrides' )
        if self._had_item_overrides:
            self._orig_item_overrides = betaconfig.item_overrides
        self._had_detector_backend = hasattr( betaconfig, 'detector_backend' )
        if self._had_detector_backend:
            self._orig_detector_backend = betaconfig.detector_backend
        self._had_global_min_prob = hasattr( betaconfig, 'global_min_prob' )
        if self._had_global_min_prob:
            self._orig_global_min_prob = betaconfig.global_min_prob

    def tearDown( self ):
        if self._had_item_overrides:
            betaconfig.item_overrides = self._orig_item_overrides
        elif hasattr( betaconfig, 'item_overrides' ):
            del betaconfig.item_overrides
        if self._had_detector_backend:
            betaconfig.detector_backend = self._orig_detector_backend
        elif hasattr( betaconfig, 'detector_backend' ):
            del betaconfig.detector_backend
        if self._had_global_min_prob:
            betaconfig.global_min_prob = self._orig_global_min_prob
        elif hasattr( betaconfig, 'global_min_prob' ):
            del betaconfig.global_min_prob

    def test_valid_backend_override_produces_no_errors( self ):
        betaconfig.item_overrides = { 'exposed_breast': { 'censor_shape': 'circle', 'track_max_gap': 4.5, 'interpolation_max_gap': 0.5 } }
        betaconfig.detector_backend = {
            'selected': 'nudenet_v3',
            'nudenet_v3': { 'item_overrides': { 'exposed_breast': { 'track_max_gap': 2.0 } } },
            'retinanet_v2': {},
        }
        errors = []
        bu_config._validate_backend_item_overrides( errors, { 'exposed_breast' } )
        self.assertEqual( errors, [] )

    def test_censor_style_in_backend_block_is_flagged( self ):
        betaconfig.item_overrides = { 'exposed_breast': {} }
        betaconfig.detector_backend = {
            'selected': 'nudenet_v3',
            'nudenet_v3': { 'item_overrides': { 'exposed_breast': { 'censor_shape': 'circle' } } },
            'retinanet_v2': {},
        }
        errors = []
        bu_config._validate_backend_item_overrides( errors, { 'exposed_breast' } )
        self.assertTrue( any( 'censor_shape' in e and "aren't backend-tunable" in e for e in errors ), errors )

    def test_unknown_class_in_backend_block_is_flagged( self ):
        betaconfig.item_overrides = {}
        betaconfig.detector_backend = {
            'selected': 'nudenet_v3',
            'nudenet_v3': { 'item_overrides': { 'not_a_real_label': { 'min_prob': 0.5 } } },
            'retinanet_v2': {},
        }
        errors = []
        bu_config._validate_backend_item_overrides( errors, { 'exposed_breast' } )
        self.assertTrue( any( 'not_a_real_label' in e for e in errors ), errors )

    def test_resolved_track_max_gap_below_shared_interpolation_max_gap_is_flagged( self ):
        # the combination-only bug this validation exists to catch: the
        # SHARED block's interpolation_max_gap is fine on its own, and a
        # backend's track_max_gap override is fine on its own, but
        # together the backend's track_max_gap ends up below the shared
        # interpolation_max_gap - only visible once resolved.
        betaconfig.item_overrides = { 'exposed_breast': { 'interpolation_max_gap': 3.0 } }
        betaconfig.detector_backend = {
            'selected': 'nudenet_v3',
            'nudenet_v3': { 'item_overrides': { 'exposed_breast': { 'track_max_gap': 1.0 } } },
            'retinanet_v2': {},
        }
        errors = []
        bu_config._validate_backend_item_overrides( errors, { 'exposed_breast' } )
        self.assertTrue( any( 'track_max_gap' in e and 'interpolation_max_gap' in e for e in errors ), errors )

    def test_resolved_min_prob_below_global_floor_is_flagged( self ):
        betaconfig.global_min_prob = 0.20
        betaconfig.item_overrides = { 'exposed_breast': {} }
        betaconfig.detector_backend = {
            'selected': 'nudenet_v3',
            'nudenet_v3': { 'item_overrides': { 'exposed_breast': { 'min_prob': 0.10 } } },
            'retinanet_v2': {},
        }
        errors = []
        bu_config._validate_min_prob_floor_consistency( errors )
        self.assertTrue( any( 'nudenet_v3' in e and 'min_prob' in e for e in errors ), errors )


class TestSharedItemOverridesRejectsBackendTunableKeys( unittest.TestCase ):
    """
    2026-09-17: every backend-tunable key (min_prob, width_area_safety,
    height_area_safety, time_safety, track_max_gap, interpolation_max_gap,
    paired_style, etc) moved OUT of the shared betaconfig.item_overrides
    block and into each backend's own detector_backend[<name>]
    ['item_overrides'][label] block - see betaconfig.py's own comment on
    item_overrides for why (per-user decision: no default convention
    should be assumed for a genuinely per-model value). This class
    covers the other half of that change: _validate_item_overrides now
    actively FLAGS a backend-tunable key still sitting in the shared
    block, rather than silently accepting it as a dead leftover that
    bu_detector.get_item_overrides never actually reads.
    """

    def setUp( self ):
        self._had_item_overrides = hasattr( betaconfig, 'item_overrides' )
        if self._had_item_overrides:
            self._orig_item_overrides = betaconfig.item_overrides

    def tearDown( self ):
        if self._had_item_overrides:
            betaconfig.item_overrides = self._orig_item_overrides
        elif hasattr( betaconfig, 'item_overrides' ):
            del betaconfig.item_overrides

    def test_min_prob_in_shared_block_is_flagged( self ):
        betaconfig.item_overrides = { 'exposed_breast': { 'min_prob': 0.37, 'censor_shape': 'circle' } }
        errors = []
        bu_config._validate_item_overrides( errors, { 'exposed_breast' } )
        self.assertTrue( any( 'min_prob' in e and 'backend-tunable' in e for e in errors ), errors )

    def test_track_max_gap_in_shared_block_is_flagged( self ):
        betaconfig.item_overrides = { 'exposed_vulva': { 'track_max_gap': 3.6 } }
        errors = []
        bu_config._validate_item_overrides( errors, { 'exposed_vulva' } )
        self.assertTrue( any( 'track_max_gap' in e and 'backend-tunable' in e for e in errors ), errors )

    def test_censor_style_and_censor_shape_alone_produce_no_errors( self ):
        betaconfig.item_overrides = {
            'exposed_breast': { 'censor_shape': 'circle', 'censor_style': { 'type': 'blur', 'method': 'gaussian', 'strength': 20 } },
        }
        errors = []
        bu_config._validate_item_overrides( errors, { 'exposed_breast' } )
        self.assertEqual( errors, [] )

    def test_real_shipped_config_has_no_backend_tunable_keys_in_shared_block( self ):
        # regression pin against the live config - if this ever starts
        # failing, someone added a backend-tunable key back into the
        # shared item_overrides block instead of a backend's own block.
        errors = []
        bu_config._validate_item_overrides( errors, set( betaconfig.item_overrides.keys() ) )
        backend_tunable_errors = [ e for e in errors if 'backend-tunable' in e ]
        self.assertEqual( backend_tunable_errors, [], backend_tunable_errors )


if __name__ == '__main__':
    unittest.main()
