"""
test_cut_burst_production.py - merging bursts of shot cuts in a real run.

A fade arrives from the shot-cut scan as a burst of cuts a sample apart,
and each cut ends every track, so the slivers between them never confirm
and the censor blinks. shot_cut_burst_window (off at 0) collapses each
burst to one cut before tracking. These tests pin: the config resolves
to the merge and back off; tracking actually confirms a track inside a
merged burst that it could not before; the censor key moves when the
merge is on and stays put while it is off (so adding the setting kept
every finished render valid); bad values fail validation; and replays
follow the config unless a caller asks for the scanned cuts.
"""

import copy
import os
import sys
import unittest
from unittest import mock

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )

import betaconfig
import betautils_cache_paths as bu_cache
import betautils_config as bu_config
import betautils_detector as bu_detector
import betautils_track as bu_track
import betautils_tuning as bu_tuning


_KEYS = ( 'shot_cut_burst_window', 'shot_cut_burst_max_span', 'shot_cut_burst_keep' )


class _BurstConfig( unittest.TestCase ):

    def setUp( self ):
        self._saved = { key: getattr( betaconfig, key ) for key in _KEYS if hasattr( betaconfig, key ) }
        for key in _KEYS:
            if hasattr( betaconfig, key ):
                delattr( betaconfig, key )
        bu_config.invalidate_config_caches()

    def tearDown( self ):
        for key in _KEYS:
            if hasattr( betaconfig, key ):
                delattr( betaconfig, key )
        for key, value in self._saved.items():
            setattr( betaconfig, key, value )
        bu_config.invalidate_config_caches()

    def turn_on( self, window=0.35, max_span=None, keep='first' ):
        betaconfig.shot_cut_burst_window = window
        betaconfig.shot_cut_burst_max_span = max_span
        betaconfig.shot_cut_burst_keep = keep
        bu_config.invalidate_config_caches()


class TestSettings( _BurstConfig ):

    def test_off_unless_the_window_is_above_zero( self ):
        self.assertIsNone( bu_track.cut_burst_settings() )
        betaconfig.shot_cut_burst_window = 0.0
        self.assertIsNone( bu_track.cut_burst_settings() )

    def test_on_reads_all_three( self ):
        self.turn_on( 0.35, 0.6, 'last' )
        self.assertEqual( bu_track.cut_burst_settings(),
                          { 'window': 0.35, 'max_span': 0.6, 'keep': 'last' } )

    def test_keep_defaults_to_first( self ):
        betaconfig.shot_cut_burst_window = 0.35
        self.assertEqual( bu_track.cut_burst_settings()['keep'], 'first' )
        self.assertIsNone( bu_track.cut_burst_settings()['max_span'] )


class TestTrackingCutTimes( _BurstConfig ):

    CUTS = [ 5.0, 10.0, 10.1, 10.2, 20.0 ]

    def test_off_leaves_the_cuts_alone( self ):
        self.assertEqual( bu_track.tracking_cut_times( self.CUTS ), self.CUTS )

    def test_on_follows_the_config( self ):
        self.turn_on( 0.35, None, 'first' )
        self.assertEqual( bu_track.tracking_cut_times( self.CUTS ), [ 5.0, 10.0, 20.0 ] )
        self.turn_on( 0.35, None, 'last' )
        self.assertEqual( bu_track.tracking_cut_times( self.CUTS ), [ 5.0, 10.2, 20.0 ] )

    def test_false_means_as_scanned_whatever_the_config( self ):
        self.turn_on()
        self.assertEqual( bu_track.tracking_cut_times( self.CUTS, False ), self.CUTS )

    def test_an_explicit_merge_overrides_the_config( self ):
        self.assertEqual( bu_track.tracking_cut_times(
            self.CUTS, { 'window': 0.35, 'max_span': None, 'keep': 'last' } ), [ 5.0, 10.2, 20.0 ] )

    def test_no_cuts_is_an_empty_list( self ):
        self.turn_on()
        self.assertEqual( bu_track.tracking_cut_times( None ), [] )
        self.assertEqual( bu_track.tracking_cut_times( [] ), [] )


def _box( t, x=500 ):
    return {
        'label': 'exposed_breast', 'start': t, 'end': t + 0.1, 't': t,
        'x': x, 'y': 300, 'w': 120, 'h': 120, 'score': 0.9,
        'censor_style': { 'type': 'blur', 'method': 'gaussian', 'strength': 16 },
        'censor_shape': 'box', '_raw_x': x, '_raw_y': 300, '_raw_w': 120, '_raw_h': 120,
        '_vid_w': 1920, '_vid_h': 1080,
    }


class TestTrackingThroughABurst( _BurstConfig ):
    """
    Three detections inside a fade's burst of cuts, a sample apart. As
    scanned, each sits alone between two cuts and never reaches two
    hits, so nothing renders. Merged to the burst's first cut, they are
    one shot and one confirmed track.
    """

    CUTS = [ 10.1, 10.2, 10.3 ]

    def setUp( self ):
        super().setUp()
        self._orig_default = betaconfig.default_min_track_hits
        self._orig_backend = copy.deepcopy( betaconfig.detector_backend )
        betaconfig.default_min_track_hits = 2
        for name in bu_detector.registered_backend_names():
            block = betaconfig.detector_backend.get( name )
            if isinstance( block, dict ):
                for overrides in block.get( 'item_overrides', {} ).values():
                    overrides.pop( 'min_track_hits', None )
        bu_config.invalidate_config_caches()

    def tearDown( self ):
        betaconfig.default_min_track_hits = self._orig_default
        betaconfig.detector_backend = self._orig_backend
        super().tearDown()

    def smooth( self, **kwargs ):
        boxes = [ _box( 10.15 ), _box( 10.25 ), _box( 10.35 ) ]
        return bu_track.smooth_boxes( boxes, list( self.CUTS ), sample_fps=10, **kwargs )

    def test_as_scanned_nothing_confirms( self ):
        rendered, stats = self.smooth()
        self.assertEqual( [ box for box in rendered if not box.get( 'interpolated' ) ], [] )
        self.assertEqual( stats['cuts_merged'], 0 )

    def test_merged_the_track_confirms( self ):
        self.turn_on( 0.35, None, 'first' )
        rendered, stats = self.smooth()
        self.assertGreaterEqual( len( rendered ), 3 )
        self.assertEqual( stats['cuts_merged'], 2 )

    def test_a_caller_can_ask_for_the_scanned_cuts( self ):
        self.turn_on( 0.35, None, 'first' )
        rendered, stats = self.smooth( cut_bursts=False )
        self.assertEqual( stats['cuts_merged'], 0 )
        self.assertEqual( [ box for box in rendered if not box.get( 'interpolated' ) ], [] )

    def test_prepare_boxes_for_render_passes_it_down( self ):
        seen = {}

        def fake_smooth( boxes, cuts, backend, profile, fps, logger=None, cut_bursts=None ):
            seen['cut_bursts'] = cut_bursts
            return [], {}

        with mock.patch.object( bu_track, 'prepare_tracking_inputs',
                                return_value=( [], { 'raw_detections': 0 } ) ), \
                mock.patch.object( bu_track, 'smooth_boxes', fake_smooth ):
            bu_track.prepare_boxes_for_render( [], 1920, 1080, [ 1.0 ], cut_bursts=False )
        self.assertIs( seen['cut_bursts'], False )


class TestCensorKey( _BurstConfig ):

    def key( self ):
        bu_config.invalidate_config_caches()
        return bu_cache.censor_key()

    def test_off_leaves_the_key_as_it_was_before_the_setting_existed( self ):
        absent = self.key()
        betaconfig.shot_cut_burst_window = 0.0
        betaconfig.shot_cut_burst_keep = 'last'
        self.assertEqual( self.key(), absent )
        self.assertNotIn( 'shot_cut_bursts', bu_cache.censor_identity() )

    def test_on_moves_the_key_and_every_change_moves_it_again( self ):
        absent = self.key()
        self.turn_on( 0.35, None, 'first' )
        first = self.key()
        self.assertNotEqual( first, absent )
        self.turn_on( 0.35, None, 'last' )
        self.assertNotEqual( self.key(), first )
        self.turn_on( 0.25, None, 'first' )
        self.assertNotEqual( self.key(), first )
        self.turn_on( 0.35, 0.6, 'first' )
        self.assertNotEqual( self.key(), first )


class TestValidation( _BurstConfig ):

    def errors( self ):
        return [ error for error in bu_config._collect_validation_errors() if 'shot_cut_burst' in error ]

    def test_defaults_and_good_values_pass( self ):
        self.assertEqual( self.errors(), [] )
        self.turn_on( 0.35, 0.6, 'last' )
        self.assertEqual( self.errors(), [] )

    def test_bad_values_fail( self ):
        for key, value in ( ( 'shot_cut_burst_window', -0.1 ), ( 'shot_cut_burst_window', True ),
                            ( 'shot_cut_burst_window', '0.35' ), ( 'shot_cut_burst_max_span', 0 ),
                            ( 'shot_cut_burst_max_span', False ), ( 'shot_cut_burst_keep', 'middle' ) ):
            self.setUp()
            setattr( betaconfig, key, value )
            self.assertEqual( len( self.errors() ), 1, "%s = %r should fail"%( key, value ) )
            self.tearDown()


class TestReplaysFollowTheConfig( _BurstConfig ):

    def replay( self, **kwargs ):
        seen = {}

        def fake_prepare( raw, w, h, **options ):
            seen.update( options )
            return [], {}

        context = { 'shot_cut_times': [ 10.0, 10.1 ], 'profile_name': None, 'sample_fps': 9,
                    'signals': None }
        with mock.patch.object( bu_tuning, '_read_raw_boxes', return_value=[ { 'class_id': 'x' } ] ), \
                mock.patch.object( bu_tuning, 'frame_size_for_video', return_value=( 640, 360 ) ), \
                mock.patch.object( bu_tuning, 'replay_tracking_context', return_value=context ), \
                mock.patch.object( bu_tuning.bu_track, 'prepare_boxes_for_render', fake_prepare ):
            bu_tuning.measure_replay( { 'h': [ 'c.gz' ] }, { 'h': '/v.mp4' }, 'nudenet_v3',
                                      [ 'exposed_breast' ], **kwargs )
        return seen

    def test_default_follows_the_config_like_a_run( self ):
        self.assertIsNone( self.replay()['cut_bursts'] )

    def test_scanned_cuts_on_request( self ):
        self.assertIs( self.replay( cut_bursts=False )['cut_bursts'], False )

    def test_a_cut_transform_replaces_the_configured_merge( self ):
        seen = self.replay( cut_transform=lambda cuts: cuts[ -1: ] )
        self.assertIs( seen['cut_bursts'], False )
        self.assertEqual( seen['shot_cut_times'], [ 10.1 ] )


if __name__ == '__main__':
    unittest.main()
