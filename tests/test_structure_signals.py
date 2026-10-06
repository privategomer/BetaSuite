"""
test_structure_signals.py - the shared structure measurement, end to end
from cut list to selected profile.

THE BUG THESE PIN
    "Scanned, found no cuts" and "never scanned" were the same value
    everywhere downstream of the scanner. shot_spans_for returned [] for
    both, so short_shot_fraction_for returned None for both, so betatv
    logged "configured default (no shot-cut data)" for a file it had just
    scanned (13 of 56 files in one preview run) and betabench printed
    n/a. The selector itself handled 0.0 correctly; it was never handed
    one. The outcome only matched by luck: the default was 'scene', the
    profile zero cuts measure as. These tests set a different default so
    the luck cannot hide the bug.

    Also pinned: the window. scanned_window_seconds is the one
    denominator, and two tools used to divide by the spread of their
    DETECTIONS instead, which scored a file whose censored boxes fell in
    one sampled frame at 600,000,000 cuts a minute.
"""

import copy
import unittest
from unittest import mock

import betaconfig

import betautils_config as bu_config
import betautils_detector as bu_detector


PROFILES = {
    # Deliberately NOT 'scene': with scene as the default, a zero-cut file
    # lands on scene whether it is measured or not, and the conflation is
    # invisible.
    'default': 'quick_cut',
    'match_on': 'short_shot_fraction',
    'variants': {
        'quick_cut': { 'min_short_shot_fraction': 0.15, 'video_censor_fps': 18 },
        'compilation': { 'manual_only': True },
        'scene': {},
    },
}


class SignalsHarness( unittest.TestCase ):

    def setUp( self ):
        self._saved_backend = copy.deepcopy( betaconfig.detector_backend )
        self._saved = { name: getattr( betaconfig, name, None ) for name in (
            'default_profile_enabled', 'force_structure_profile',
            'profile_by_path_pattern', 'profile_by_source_dir' ) }
        backend = betaconfig.detector_backend['selected']
        betaconfig.detector_backend = copy.deepcopy( betaconfig.detector_backend )
        betaconfig.detector_backend[backend]['profiles'] = copy.deepcopy( PROFILES )
        betaconfig.default_profile_enabled = True
        betaconfig.force_structure_profile = None
        betaconfig.profile_by_path_pattern = {}
        betaconfig.profile_by_source_dir = {}
        bu_config.invalidate_config_caches()

    def tearDown( self ):
        betaconfig.detector_backend = self._saved_backend
        for name, value in self._saved.items():
            setattr( betaconfig, name, value )
        bu_config.invalidate_config_caches()

    def select( self, cuts, scanned ):
        signals = bu_detector.structure_signals( cuts, scanned )
        return bu_detector.profile_selection(
            signals.cuts_per_min, short_shot_fraction=signals.short_shot_fraction )


class TestZeroCutsAreAMeasurement( SignalsHarness ):

    def test_precondition_unmeasured_takes_the_non_scene_default( self ):
        # If this fails, the tests below cannot tell measured from not.
        self.assertEqual( self.select( None, 600.0 ), ( 'quick_cut', 'the configured default' ) )

    def test_a_scanned_file_with_no_cuts_is_measured_as_scene( self ):
        self.assertEqual( self.select( [], 600.0 ),
                          ( 'scene', 'measured short_shot_fraction' ) )

    def test_zero_cuts_without_a_known_window_is_not_measured( self ):
        # No window, no denominator: refuse to measure rather than guess.
        self.assertEqual( self.select( [], None )[1], 'the configured default' )

    def test_one_shot_spans_the_whole_window( self ):
        self.assertEqual( bu_detector.shot_spans_for( [], 42.0 ), [ 42.0 ] )

    def test_never_scanned_has_no_spans_even_with_a_known_window( self ):
        # Knowing the file's length is not the same as having scanned it.
        self.assertEqual( bu_detector.shot_spans_for( None, 42.0 ), [] )

    def test_the_signals_say_zero_not_none( self ):
        signals = bu_detector.structure_signals( [], 600.0 )
        self.assertTrue( signals.is_measured )
        self.assertEqual( signals.cut_count, 0 )
        self.assertEqual( signals.cuts_per_min, 0.0 )
        self.assertEqual( signals.short_shot_fraction, 0.0 )
        self.assertEqual( signals.median_shot_seconds, 600.0 )

    def test_never_scanned_says_none_everywhere( self ):
        signals = bu_detector.structure_signals( None, 600.0 )
        self.assertFalse( signals.is_measured )
        for name in ( 'cut_count', 'scanned_seconds', 'cuts_per_min',
                      'short_shot_fraction', 'median_shot_seconds' ):
            self.assertIsNone( getattr( signals, name ), name )

    def test_fast_cutting_still_selects_quick_cut( self ):
        # 30 cuts half a second apart in a 60s window: 15s of sub-1s
        # shots is 25% of runtime, over the 15% threshold.
        cuts = [ 10.0 + 0.5 * n for n in range( 30 ) ]
        self.assertEqual( self.select( cuts, 60.0 )[0], 'quick_cut' )


class TestDisabledScanIsNotAZeroCutScan( unittest.TestCase ):

    def test_a_disabled_scan_reports_none( self ):
        import betautils_video as bu_video
        # The source is never touched when disabled, so a sentinel suffices.
        self.assertIsNone( bu_video.shot_cuts_for_source(
            object(), 'hash', 9, 0.5, enabled=False ) )

    def test_a_disabled_scan_is_not_measured_downstream( self ):
        import betautils_video as bu_video
        cuts = bu_video.shot_cuts_for_source( object(), 'hash', 9, 0.5, enabled=False )
        self.assertFalse( bu_detector.structure_signals( cuts, 600.0 ).is_measured )


class TestScannedWindow( unittest.TestCase ):

    def test_a_full_file_scan_covers_the_whole_duration( self ):
        self.assertEqual( bu_detector.scanned_window_seconds( 2096.0 ), 2096.0 )

    def test_a_preview_covers_its_bound( self ):
        self.assertEqual( bu_detector.scanned_window_seconds( 2096.0, 0.0, 20.0 ), 20.0 )

    def test_a_bound_past_the_end_is_clamped_to_what_was_there( self ):
        # A 20s preview of a 15s clip scanned 15s, not 20: counting five
        # seconds that do not exist dilutes every share.
        self.assertEqual( bu_detector.scanned_window_seconds( 15.0, 0.0, 20.0 ), 15.0 )

    def test_an_offset_slice_near_the_end_is_clamped( self ):
        self.assertEqual( bu_detector.scanned_window_seconds( 100.0, 90.0, 20.0 ), 10.0 )

    def test_an_offset_without_a_bound_runs_to_the_end( self ):
        self.assertEqual( bu_detector.scanned_window_seconds( 100.0, 30.0 ), 70.0 )

    def test_an_unknown_duration_is_unknown( self ):
        self.assertIsNone( bu_detector.scanned_window_seconds( None, 0.0, 20.0 ) )
        self.assertIsNone( bu_detector.scanned_window_seconds( 0.0, 0.0, 20.0 ) )

    def test_an_offset_at_or_past_the_end_scanned_nothing( self ):
        self.assertIsNone( bu_detector.scanned_window_seconds( 100.0, 100.0, 20.0 ) )


class TestDeclaredSelection( SignalsHarness ):

    def test_nothing_declared_means_measurement_decides( self ):
        self.assertIsNone( bu_detector.declared_profile_selection( '/x/y.mp4' ) )

    def test_a_forced_profile_is_a_declaration( self ):
        betaconfig.force_structure_profile = 'compilation'
        self.assertEqual( bu_detector.declared_profile_selection( '/x/y.mp4' ),
                          ( 'compilation', 'forced by --profile' ) )

    def test_a_forced_name_that_is_not_a_variant_is_still_decided( self ):
        # Decided as "nothing", with a reason; never silently measured.
        betaconfig.force_structure_profile = 'nope'
        name, reason = bu_detector.declared_profile_selection( '/x/y.mp4' )
        self.assertIsNone( name )
        self.assertIn( 'not a variant', reason )

    def test_a_source_folder_is_a_declaration( self ):
        betaconfig.profile_by_source_dir = { 'comps': 'compilation' }
        root = '/videos/'
        with mock.patch.object( bu_detector.betaconst, 'video_path_uncensored', root ):
            self.assertEqual(
                bu_detector.declared_profile_selection( '/videos/comps/a.mp4' ),
                ( 'compilation', 'source folder comps/' ) )

    def test_measured_selection_ignores_every_declaration( self ):
        betaconfig.force_structure_profile = 'compilation'
        self.assertEqual(
            bu_detector.measured_profile_selection( 0.0, short_shot_fraction=0.0 ),
            ( 'scene', 'measured short_shot_fraction' ) )

    def test_profile_selection_still_puts_declarations_first( self ):
        betaconfig.force_structure_profile = 'compilation'
        self.assertEqual(
            bu_detector.profile_selection( 0.0, short_shot_fraction=0.9 ),
            ( 'compilation', 'forced by --profile' ) )



class TestShotCutCacheNames( unittest.TestCase ):
    """The window a cached scan covered, read back from its filename."""

    def test_round_trip_through_the_writer( self ):
        import betautils_cache_paths as bu_cache
        path = bu_cache.shot_cut_path_for( 'feedface', 9, 0.5, '-preview@30.0s', 20 )
        parsed = bu_cache.parse_shot_cut_path( path )
        self.assertEqual( parsed['file_hash'], 'feedface' )
        self.assertEqual( parsed['max_seconds'], 20.0 )
        self.assertEqual( parsed['offset_seconds'], 30.0 )
        self.assertEqual( parsed['preview_suffix'], '-preview@30.0s' )

    def test_a_whole_file_scan_has_no_bound_or_offset( self ):
        import betautils_cache_paths as bu_cache
        parsed = bu_cache.parse_shot_cut_path( bu_cache.shot_cut_path_for( 'feedface', 9, 0.5 ) )
        self.assertIsNone( parsed['max_seconds'] )
        self.assertEqual( parsed['offset_seconds'], 0.0 )

    def test_foreign_names_are_rejected( self ):
        import betautils_cache_paths as bu_cache
        for name in ( 'notes.txt', 'abc-9-0.5.gz', 'abc-9-0.500-max20s-thumbnail.gz' ):
            self.assertIsNone( bu_cache.parse_shot_cut_path( name ), name )


class TestCachedShotCuts( unittest.TestCase ):

    def setUp( self ):
        import tempfile
        import betaconst
        self.tmp = tempfile.mkdtemp( prefix='betasuite-cuts-' )
        self._saved = betaconst.shot_cut_dir
        betaconst.shot_cut_dir = self.tmp

    def tearDown( self ):
        import shutil
        import betaconst
        betaconst.shot_cut_dir = self._saved
        shutil.rmtree( self.tmp, ignore_errors=True )

    def write( self, cuts, suffix='', bound=None ):
        import betautils_cache_paths as bu_cache
        import betautils_hash as bu_hash
        bu_hash.write_json( cuts, bu_cache.shot_cut_path_for( 'feedface', 9, 0.5, suffix, bound ) )

    def load( self, duration, prefer=None ):
        import betautils_video as bu_video
        return bu_video.cached_shot_cuts( 'feedface', 9, 0.5, duration, prefer_preview_suffix=prefer )

    def test_an_offset_slice_near_the_end_covers_only_what_was_left( self ):
        self.write( [ 2.0 ], '-preview@30.0s', 20 )
        cached = self.load( 40.0 )
        self.assertEqual( cached.scanned_seconds, 10.0 )
        self.assertEqual( cached.offset_seconds, 30.0 )
        self.assertFalse( cached.is_whole_file )

    def test_a_whole_file_scan_is_preferred_by_default( self ):
        self.write( [ 1.0 ], '-preview', 20 )
        self.write( [ 5.0, 6.0 ] )
        cached = self.load( 60.0 )
        self.assertTrue( cached.is_whole_file )
        self.assertEqual( cached.cuts, [ 5.0, 6.0 ] )

    def test_a_requested_slice_wins_over_the_whole_file( self ):
        # A replay of a slice's detections must measure the slice's window.
        self.write( [ 1.0 ], '-preview', 20 )
        self.write( [ 5.0, 6.0 ] )
        cached = self.load( 60.0, prefer='-preview' )
        self.assertEqual( cached.scanned_seconds, 20.0 )
        self.assertEqual( cached.cuts, [ 1.0 ] )

    def test_nothing_cached_is_none_not_empty( self ):
        self.assertIsNone( self.load( 60.0 ) )

    def test_an_unknown_duration_leaves_the_window_unknown( self ):
        self.write( [ 5.0 ] )
        self.assertIsNone( self.load( None ).scanned_seconds )


if __name__ == '__main__':
    unittest.main()
