"""
test_profile_sample_rate.py - a structure profile can set its own
detection sample rate.

WHY
---
Fast-cut footage has short shots. On a real preview slice at 172 cuts/min
the median shot was 0.2s: at 9 fps that is fewer than two samples, and
min_track_hits is 2, so most shots could never confirm a track and their
exposure was never censored. Sampling every file faster fixes that and
costs detection time on every file; sampling only the files that need it
costs it where it buys something.

THE ORDERING CONSTRAINT
-----------------------
The profile is chosen FROM the shot-cut scan, so the scan must run at the
global rate - a profile setting its own scan rate would be circular. Cut
timestamps are times, not sample indices, so they stay valid at whatever
rate detection then runs. Tracking must then use the rate detection
actually ran at: every frame-count rule (interpolation steps, the
two-frame default track_max_gap) is expressed in that step.
"""

import copy
import os
import sys
import unittest

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )

import betaconfig

import betautils_config as bu_config
import betautils_detector as bu_detector
import betautils_track as bu_track


class ProfileRateHarness( unittest.TestCase ):

    def setUp( self ):
        self._saved_backend = copy.deepcopy( betaconfig.detector_backend )
        self._saved_fps = betaconfig.video_censor_fps
        self._saved_enabled = getattr( betaconfig, 'default_profile_enabled', True )
        self.backend = betaconfig.detector_backend['selected']
        betaconfig.detector_backend = copy.deepcopy( betaconfig.detector_backend )
        betaconfig.detector_backend[self.backend]['profiles'] = {
            'default': 'scene', 'match_on': 'cuts_per_min',
            'variants': {
                'compilation': { 'min_cuts_per_min': 40, 'video_censor_fps': 15 },
                'scene': { 'min_cuts_per_min': 0 },
            },
        }
        betaconfig.video_censor_fps = 9
        betaconfig.default_profile_enabled = True
        bu_config.invalidate_config_caches()

    def tearDown( self ):
        betaconfig.detector_backend = self._saved_backend
        betaconfig.video_censor_fps = self._saved_fps
        betaconfig.default_profile_enabled = self._saved_enabled
        bu_config.invalidate_config_caches()


class TestRateResolution( ProfileRateHarness ):

    def test_a_profile_with_a_rate_uses_it( self ):
        self.assertEqual( bu_detector.get_profile_sample_fps( 'compilation' ), 15 )

    def test_a_profile_without_a_rate_uses_the_global_one( self ):
        self.assertEqual( bu_detector.get_profile_sample_fps( 'scene' ), 9 )

    def test_no_profile_uses_the_global_rate( self ):
        self.assertEqual( bu_detector.get_profile_sample_fps( None ), 9 )

    def test_an_unknown_profile_uses_the_global_rate( self ):
        self.assertEqual( bu_detector.get_profile_sample_fps( 'nope' ), 9 )

    def test_the_candidate_rates_cover_every_profile( self ):
        self.assertEqual( bu_detector.profile_sample_rates(), { 9, 15 } )

    def test_disabled_profiles_leave_only_the_global_rate( self ):
        # Which in turn lets betatv resolve the output name before the
        # shot-cut scan and skip a finished file at zero cost.
        betaconfig.default_profile_enabled = False
        self.assertEqual( bu_detector.profile_sample_rates(), { 9 } )

    def test_the_rate_is_in_the_censor_key( self ):
        import betautils_cache_paths as bu_cache
        before = bu_cache.censor_key()
        betaconfig.detector_backend[self.backend]['profiles']['variants'][
            'compilation']['video_censor_fps'] = 12
        self.assertNotEqual( before, bu_cache.censor_key() )


class TestValidation( ProfileRateHarness ):

    def _profile_errors( self ):
        return [ error for error in bu_config._collect_validation_errors()
                 if "['profiles']" in error ]

    def test_the_shipped_shape_is_valid( self ):
        self.assertEqual( self._profile_errors(), [] )

    def test_a_non_positive_rate_is_an_error( self ):
        betaconfig.detector_backend[self.backend]['profiles']['variants'][
            'compilation']['video_censor_fps'] = 0
        self.assertTrue( self._profile_errors() )

    def test_a_misspelled_variant_key_is_an_error( self ):
        # Profiles shipped unvalidated; a typo here silently made the
        # profile behave like its neighbour.
        betaconfig.detector_backend[self.backend]['profiles']['variants'][
            'compilation']['video_censor_fsp'] = 15
        self.assertTrue( self._profile_errors() )

    def test_a_default_naming_no_variant_is_an_error( self ):
        betaconfig.detector_backend[self.backend]['profiles']['default'] = 'scen'
        self.assertTrue( self._profile_errors() )


class TestTrackingUsesTheSampledRate( ProfileRateHarness ):
    """
    Two detections 0.2s apart. Sampled at 10 fps that is a one-frame hole
    worth interpolating; sampled at 5 fps it is two adjacent samples with
    nothing missing between them. Reading the global rate instead of the
    sampled one gets one of those two wrong.
    """

    def setUp( self ):
        super().setUp()
        block = betaconfig.detector_backend[self.backend]
        block.pop( 'profiles', None )
        block.setdefault( 'item_overrides', {} ).setdefault( 'exposed_breast', {} ).update(
            interpolation_enabled=True, interpolation_max_gap=1.0,
            track_max_gap=1.0, min_track_hits=1 )
        bu_config.invalidate_config_caches()

    def _raw( self ):
        return [ { 'class_id': 'exposed_breast', 'score': 0.9, 't': t,
                   'x': 600, 'y': 400, 'w': 200, 'h': 200 } for t in ( 1.0, 1.2 ) ]

    def test_a_missing_sample_is_filled_at_the_denser_rate( self ):
        _boxes, stats = bu_track.prepare_boxes_for_render(
            self._raw(), 1920, 1080, backend_name=self.backend, sample_fps=10 )
        self.assertEqual( stats['interpolated'], 1 )

    def test_adjacent_samples_are_not_filled_at_the_sparser_rate( self ):
        _boxes, stats = bu_track.prepare_boxes_for_render(
            self._raw(), 1920, 1080, backend_name=self.backend, sample_fps=5 )
        self.assertEqual( stats['interpolated'], 0 )


if __name__ == '__main__':
    unittest.main()
