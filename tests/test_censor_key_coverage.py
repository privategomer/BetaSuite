"""
test_censor_key_coverage.py - every setting that changes rendered output
must change the censor key.

WHY THIS FILE EXISTS
--------------------
The output filename carries the detection, censor and encode keys, and
betatv.py skips a file outright when a valid output already sits at that
path. That skip is only correct while the keys are complete: a setting
that changes rendered bytes but not the key means a tuning run silently
reuses the previous render and reports it as the new result.

This is not hypothetical. Structure profiles, style dwell and the
shot-cut settings all shipped missing from censor_identity(). Profiles
were the worst of them - they override the per-label timing values AFTER
those values are resolved, so the 'labels' entry in the identity looks
complete while describing settings the render did not actually use.
Shot cuts hid behind their own separate scan cache: lowering the
threshold correctly rescanned and found more cuts, then reused a file
rendered from the old ones.

The tests below drive each setting the way a person tuning would, and
assert the key moves. They deliberately do NOT assert a specific key
value - that would just be a change-detector test that has to be updated
every time anything is added.
"""

import copy
import unittest

import betaconfig

import betautils_cache_paths as bu_cache
import betautils_config as bu_config


class CensorKeyHarness( unittest.TestCase ):
    """Restores betaconfig wholesale, since these tests mutate it deeply."""

    # Attributes each test may touch. Deep-copied so a nested edit to
    # detector_backend cannot survive into another test.
    _WATCHED = (
        'detector_backend', 'default_style_min_dwell_seconds',
        'default_profile_enabled', 'default_position_smoothing',
        'shot_cut_threshold', 'shot_cut_detection_enabled',
        'default_censor_shape', 'default_censor_style',
        'censor_scale_strategy', 'items_to_censor',
    )

    def setUp( self ):
        self._saved = { name: copy.deepcopy( getattr( betaconfig, name ) )
                        for name in self._WATCHED if hasattr( betaconfig, name ) }
        self._absent = [ name for name in self._WATCHED
                         if not hasattr( betaconfig, name ) ]
        bu_config.invalidate_config_caches()

    def tearDown( self ):
        for name, value in self._saved.items():
            setattr( betaconfig, name, value )
        for name in self._absent:
            if hasattr( betaconfig, name ):
                delattr( betaconfig, name )
        bu_config.invalidate_config_caches()

    def key( self ):
        bu_config.invalidate_config_caches()
        return bu_cache.censor_key()

    def assert_key_changes( self, mutate, what ):
        """Apply mutate() to betaconfig and require the censor key to move."""
        before = self.key()
        mutate( betaconfig )
        after = self.key()
        self.assertNotEqual(
            before, after,
            "%s changes rendered output but not the censor key, so a run "
            "with the new value would reuse the old render and report it "
            "as the new result"%( what, ) )


class TestStructureProfilesAreKeyed( CensorKeyHarness ):

    def _profiles( self, config ):
        return config.detector_backend['nudenet_v3'].setdefault(
            'profiles', { 'default': 'scene', 'match_on': 'cuts_per_min',
                          'variants': {} } )

    def test_a_profile_timing_value_changes_the_key( self ):
        def mutate( config ):
            variants = self._profiles( config ).setdefault( 'variants', {} )
            scene = variants.setdefault( 'scene', { 'min_cuts_per_min': 0,
                                                    'item_overrides': {} } )
            overrides = scene.setdefault( 'item_overrides', {} )
            overrides.setdefault( 'exposed_breast', {} )['track_max_gap'] = 99.0
        self.assert_key_changes( mutate, "a profile's track_max_gap" )

    def test_a_profile_threshold_changes_the_key( self ):
        # Which profile a given video selects is part of the configuration
        # even though the selection itself depends on the video.
        def mutate( config ):
            variants = self._profiles( config ).setdefault( 'variants', {} )
            scene = variants.setdefault( 'scene', { 'min_cuts_per_min': 0,
                                                    'item_overrides': {} } )
            scene['min_cuts_per_min'] = 12.5
        self.assert_key_changes( mutate, "a profile's min_cuts_per_min" )

    def test_the_default_profile_changes_the_key( self ):
        def mutate( config ):
            self._profiles( config )['default'] = 'compilation'
        self.assert_key_changes( mutate, "the default profile name" )

    def test_adding_a_profile_changes_the_key( self ):
        def mutate( config ):
            variants = self._profiles( config ).setdefault( 'variants', {} )
            variants['frenetic'] = {
                'min_cuts_per_min': 200,
                'item_overrides': { 'exposed_breast': { 'track_max_gap': 0.05 } },
            }
        self.assert_key_changes( mutate, "adding a profile variant" )

    def test_disabling_profiles_changes_the_key( self ):
        self.assert_key_changes(
            lambda config: setattr( config, 'default_profile_enabled', False ),
            "default_profile_enabled" )


class TestStyleDwellIsKeyed( CensorKeyHarness ):

    def test_the_dwell_default_changes_the_key( self ):
        current = getattr( betaconfig, 'default_style_min_dwell_seconds', 0.0 )
        self.assert_key_changes(
            lambda config: setattr( config, 'default_style_min_dwell_seconds',
                                    current + 7.5 ),
            "default_style_min_dwell_seconds" )


class TestShotCutSettingsAreKeyed( CensorKeyHarness ):
    """
    Shot cuts feed tracking, so they change rendered output for a fixed
    set of detections. Their own scan cache is keyed separately, which is
    exactly why their absence from the censor key went unnoticed.
    """

    def test_the_shot_cut_threshold_changes_the_key( self ):
        self.assert_key_changes(
            lambda config: setattr( config, 'shot_cut_threshold', 0.99 ),
            "shot_cut_threshold" )

    def test_disabling_shot_cut_detection_changes_the_key( self ):
        self.assert_key_changes(
            lambda config: setattr( config, 'shot_cut_detection_enabled', False ),
            "shot_cut_detection_enabled" )


class TestPreviouslyCoveredSettingsStayCovered( CensorKeyHarness ):
    """
    Guards the settings that were already keyed. A refactor of
    censor_identity() that drops one of these would be just as damaging
    as the gaps above, and far less obvious.
    """

    def test_smoothing_changes_the_key( self ):
        self.assert_key_changes(
            lambda config: setattr( config, 'default_position_smoothing', 0.11 ),
            "default_position_smoothing" )

    def test_the_default_shape_changes_the_key( self ):
        self.assert_key_changes(
            lambda config: setattr( config, 'default_censor_shape', 'ellipse' ),
            "default_censor_shape" )

    def test_the_scale_strategy_changes_the_key( self ):
        current = getattr( betaconfig, 'censor_scale_strategy', 'feature' )
        self.assert_key_changes(
            lambda config: setattr( config, 'censor_scale_strategy',
                                    'frame' if current != 'frame' else 'feature' ),
            "censor_scale_strategy" )

    def test_the_censored_item_list_changes_the_key( self ):
        def mutate( config ):
            config.items_to_censor = list( config.items_to_censor ) + [ 'exposed_anus' ]
        self.assert_key_changes( mutate, "items_to_censor" )


class TestTheKeyIsStable( CensorKeyHarness ):

    def test_reading_the_key_twice_gives_the_same_answer( self ):
        # Dict ordering, float formatting and any set() that sneaks into
        # the identity would all show up here.
        self.assertEqual( self.key(), self.key() )

    def test_an_unrelated_setting_does_not_change_the_key( self ):
        # The inverse guard: a key that changes on everything is as
        # useless as one that changes on nothing, because it would
        # invalidate every cached render on an unrelated edit.
        before = self.key()
        betaconfig.__dict__['_test_only_unused_setting'] = 'x'
        try:
            self.assertEqual(
                before, self.key(),
                "an unrelated setting moved the censor key; that discards "
                "valid cached renders on every unrelated config edit" )
        finally:
            betaconfig.__dict__.pop( '_test_only_unused_setting', None )


if __name__ == '__main__':
    unittest.main()
