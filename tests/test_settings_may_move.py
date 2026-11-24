"""
test_settings_may_move.py - the keys a setting can move for some files
only are real.

A file's detection key uses the sample rate of the structure profile it
gets, and a preview run's caches and output carry the slice it renders.
A setting that changes which profile a file gets, or which slice, can
re-detect and re-render that file without moving any key of the config
as a whole. The registry says so in may_move, and config.set reports it,
so the app can say "may re-detect some files". This file shows each
such claim on concrete files.
"""

import os
import random
import unittest

import betaconfig
import betaconst
import betautils_cache_paths as bu_cache
import betautils_config as bu_config
import betautils_config_store as bu_store
import betautils_detector as bu_detector
import betautils_settings as bu_settings
import betautils_video as bu_video

from tests import config_harness as harness

# Files in the shipped folder convention's folders and outside it, each
# not measured (shot cuts off), measured as calm, or as fast-cut.
FOLDERS = ( 'quick-cut', 'scene', 'compilation', 'elsewhere' )
SIGNALS = ( ( None, None ), ( 0.0, 0.0 ), ( 25.0, 0.15 ), ( 40.0, 0.15 ), ( 40.0, 0.6 ) )


def sample_rates():
    """{ ( folder, signals ): the sample rate that file's detection runs at }."""
    bu_config.invalidate_config_caches()
    rates = {}
    for folder in FOLDERS:
        source = os.path.join( betaconst.video_path_uncensored, folder, 'clip.mp4' )
        for cuts_per_min, short_share in SIGNALS:
            name, _reason = bu_detector.profile_selection( cuts_per_min, source_path=source,
                                                           short_shot_fraction=short_share )
            rates[( folder, cuts_per_min, short_share )] = bu_detector.get_profile_sample_fps( name )
    return rates


# path -> the value it is changed to, chosen to reach a file whose profile it moves.
PROFILE_CASES = {
    ( 'detector_backend', 'nudenet_v3', 'profiles', 'default' ): 'quick_cut',
    ( 'detector_backend', 'nudenet_v3', 'profiles', 'match_on' ): 'cuts_per_min',
    ( 'detector_backend', 'nudenet_v3', 'profiles', 'variants', 'quick_cut', 'min_short_shot_fraction' ): 0.2,
    ( 'detector_backend', 'nudenet_v3', 'profiles', 'variants', 'quick_cut', 'manual_only' ): True,
    ( 'profile_by_source_dir', 'quick-cut' ): 'compilation',
    ( 'profile_by_path_pattern', ): { '*elsewhere*': 'quick_cut' },
    ( 'force_structure_profile', ): 'compilation',
}

# min_cuts_per_min is read only while match_on is cuts_per_min, so it is
# shown under that setting.
UNDER_CUTS_PER_MIN = {
    ( 'detector_backend', 'nudenet_v3', 'profiles', 'variants', 'quick_cut', 'min_cuts_per_min' ): 30,
}

# The shot cuts a file has decide which profile it measures as. Shown on
# real footage only (the scan reads frames), so argued here: a threshold
# or burst merge that changes the cuts changes the measured share.
SHOT_CUT_ARGUED = { 'shot_cut_detection_enabled', 'shot_cut_threshold', 'shot_cut_burst_window',
                    'shot_cut_burst_max_span', 'shot_cut_burst_keep' }

PREVIEW_CASES = {
    'preview_max_seconds': 30,
    'preview_start_seconds': 45.0,
    'preview_random_slice': True,
}


def preview_suffix():
    """The cache and output suffix of a preview of a 10-minute, 30 fps file."""
    random.seed( 7 )
    seconds = getattr( betaconfig, 'preview_max_seconds', 20 )
    enabled = getattr( betaconfig, 'preview_mode_enabled', False )
    offset, unlimited = bu_video.resolve_preview_slice(
        30.0, 30*600, enabled, seconds, getattr( betaconfig, 'preview_start_seconds', None ),
        getattr( betaconfig, 'preview_random_slice', False ) )
    return bu_cache.preview_cache_suffix( enabled, offset, None if unlimited else seconds, 600.0 )


class TestProfileChoiceMovesSomeFilesSampleRate( unittest.TestCase ):

    def assert_some_file_moves( self, path, value, setup=None ):
        with harness.shipped():
            betaconfig.detector_backend['selected'] = 'nudenet_v3'
            if setup:
                setup()
            before = sample_rates()
            keys_before = bu_store.current_keys()
            harness.set_path( path, value )
            after = sample_rates()
            keys_after = bu_store.current_keys()
        moved = [ case for case in before if before[case] != after[case] ]
        self.assertTrue( moved, "%s changes no file's sample rate"%( '.'.join( path ), ) )
        # ...which the config's own detection keys do not show, hence may_move.
        self.assertEqual( keys_before['detection'], keys_after['detection'], path )

    def test_each_profile_choice_setting_moves_some_files_rate( self ):
        for path, value in PROFILE_CASES.items():
            with self.subTest( path='.'.join( path ) ):
                self.assert_some_file_moves( path, value )

    def test_a_cut_rate_threshold_moves_a_rate_under_that_signal( self ):
        def by_cuts():
            profiles = betaconfig.detector_backend['nudenet_v3']['profiles']
            profiles['match_on'] = 'cuts_per_min'
            quick = profiles['variants']['quick_cut']
            quick.pop( 'min_short_shot_fraction', None )
            quick['min_cuts_per_min'] = 20
        for path, value in UNDER_CUTS_PER_MIN.items():
            self.assert_some_file_moves( path, value, setup=by_cuts )

    def test_every_profile_choice_entry_is_shown_or_argued( self ):
        shown = { bu_settings.find( path ).dotted for path in list( PROFILE_CASES ) + list( UNDER_CUTS_PER_MIN ) }
        claimed = { entry.dotted for entry in bu_settings.REGISTRY
                    if entry.may_move == frozenset( { bu_settings.DETECTION } ) }
        self.assertEqual( claimed, shown | SHOT_CUT_ARGUED )


class TestPreviewSettingsMoveTheSlice( unittest.TestCase ):

    def test_each_preview_setting_changes_the_slice_a_preview_renders( self ):
        for name, value in PREVIEW_CASES.items():
            with self.subTest( setting=name ), harness.shipped():
                betaconfig.preview_mode_enabled = True
                before = preview_suffix()
                setattr( betaconfig, name, value )
                self.assertNotEqual( before, preview_suffix() )

    def test_turning_preview_on_changes_the_slice( self ):
        with harness.shipped():
            before = preview_suffix()
            betaconfig.preview_mode_enabled = True
            self.assertNotEqual( before, preview_suffix() )

    def test_every_preview_entry_is_shown( self ):
        claimed = { entry.dotted for entry in bu_settings.REGISTRY
                    if entry.may_move == frozenset( { bu_settings.DETECTION, bu_settings.CENSOR } ) }
        self.assertEqual( claimed, set( PREVIEW_CASES ) | { 'preview_mode_enabled' } )


class TestRunLevelDetectionKeys( unittest.TestCase ):
    """A run detects at every picture size and every rate a profile can pick; each is a key."""

    def test_an_added_picture_size_moves_the_detection_keys( self ):
        with harness.shipped():
            betaconfig.detector_backend['selected'] = 'retinanet_v2'
            before = bu_store.current_keys()
            betaconfig.detector_backend['retinanet_v2']['picture_sizes'] = [ 1280, 640 ]
            after = bu_store.current_keys()
        self.assertEqual( [ 'detection' ], bu_store.moved( before, after ) )

    def test_a_profiles_own_rate_moves_the_detection_keys( self ):
        with harness.shipped():
            before = bu_store.current_keys()
            variants = betaconfig.detector_backend['nudenet_v3']['profiles']['variants']
            variants['quick_cut']['video_censor_fps'] = 24
            after = bu_store.current_keys()
        self.assertIn( 'detection', bu_store.moved( before, after ) )


if __name__ == '__main__':
    unittest.main()
