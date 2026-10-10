"""
test_replay_fidelity.py - the post-run analysers replay what a real run does.

analyze_track_breaks, analyze_jitter and analyze_style_flicker used to
call smooth_boxes with no structure profile and no shot cuts, after a
hand-built suppression step with no promotion and no geometry filter. A
real run picks a profile per file (which overrides track_max_gap, match
distance and min_track_hits per label) and stops tracks at cuts, so the
analysers' sweeps described a configuration no run used: vulva tracks
replayed with a 43.2 s gap that every real run overrides to 4 or 8.
"""

import copy
import importlib.util
import os
import types
import unittest
from unittest import mock

import betaconfig
import betautils_config as bu_config
import betautils_detector as bu_detector
import betautils_tuning as bu_tuning


def _repo_path( *parts ):
    return os.path.join( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ), *parts )


def _load_tool( relative, name ):
    spec = importlib.util.spec_from_file_location( name, _repo_path( relative ) )
    module = importlib.util.module_from_spec( spec )
    spec.loader.exec_module( module )
    return module


TRACKING = { 'shot_cut_times': [ 1.5, 4.0 ], 'profile_name': 'compilation', 'sample_fps': 9,
             'signals': None }


class _FakeCapture( object ):
    def __init__( self, *_args ):
        pass

    def get( self, prop ):
        import cv2
        return { cv2.CAP_PROP_FRAME_WIDTH: 640, cv2.CAP_PROP_FRAME_HEIGHT: 360 }.get( prop, 0 )

    def release( self ):
        pass


class TestAnalysersReplayUnderTheRealProfileAndCuts( unittest.TestCase ):

    def replay( self, tool, call ):
        """Run one tool's replay with the pipeline faked; return smooth_boxes' kwargs."""
        seen = []

        def fake_smooth( boxes, **kwargs ):
            seen.append( kwargs )
            return boxes, {}

        box = { 'label': 'exposed_breast', 'x': 0, 'y': 0, 'w': 10, 'h': 10 }
        with mock.patch.object( tool.bu_tuning, 'replay_tracking_inputs',
                                return_value=( [ box ], TRACKING ) ), \
             mock.patch.object( tool.bu_hash, 'read_json', return_value=[ { 'class_id': 'x' } ] ), \
             mock.patch.object( tool.os.path, 'exists', return_value=True ), \
             mock.patch.object( tool.cv2, 'VideoCapture', _FakeCapture ):
            call( fake_smooth )
        return seen

    def assert_real_context( self, seen ):
        self.assertTrue( seen, "the replay never reached smooth_boxes" )
        for kwargs in seen:
            self.assertEqual( kwargs.get( 'profile_name' ), 'compilation' )
            self.assertEqual( kwargs.get( 'shot_cut_times' ), [ 1.5, 4.0 ] )
            self.assertEqual( kwargs.get( 'sample_fps' ), 9 )
            self.assertEqual( kwargs.get( 'backend_name' ), 'nudenet_v3' )

    def test_track_breaks( self ):
        tool = _load_tool( 'tools/analysis/analyze_track_breaks.py', '_tb_fidelity' )
        ns = { 'smooth_boxes': None, 'apply_class_suppression': None, 'observer': None }

        def call( fake_smooth ):
            ns['smooth_boxes'] = fake_smooth
            ns['observer'] = tool.bu_track.TrackingObserver()
            tool.run_one_combo( 1.0, 1.0, { 'h1': [ 'c.gz' ] }, { 'h1': '/v.mp4' }, ns,
                                { 'exposed_breast' }, 'nudenet_v3' )
        self.assert_real_context( self.replay( tool, call ) )

    def test_jitter( self ):
        tool = _load_tool( 'tools/analysis/analyze_jitter.py', '_jit_fidelity' )

        def call( fake_smooth ):
            tool.run_one_alpha( 0.5, { 'h1': [ 'c.gz' ] }, { 'h1': '/v.mp4' }, fake_smooth, None,
                                { 'exposed_breast' }, 'nudenet_v3' )
        self.assert_real_context( self.replay( tool, call ) )


class TestEveryHubAnalyserPassesTheContext( unittest.TestCase ):
    """A source-level backstop for the analysers run_all_analysis.sh runs."""

    def test_each_smooth_boxes_call_carries_profile_and_cuts( self ):
        import re
        for relative in ( 'tools/analysis/analyze_track_breaks.py',
                          'tools/analysis/analyze_jitter.py',
                          'tools/analysis/analyze_style_flicker.py' ):
            with open( _repo_path( relative ) ) as handle:
                source = handle.read()
            calls = re.findall( r'smooth_boxes_fn\((.*?)\)\s*$', source, re.S | re.M )
            self.assertTrue( calls, relative )
            for call in re.findall( r'smooth_boxes_fn\(([^)]*\))', source ):
                self.assertIn( "profile_name=tracking['profile_name']", call, relative )
                self.assertIn( "shot_cut_times=tracking['shot_cut_times']", call, relative )


class TestReplayTrackingInputs( unittest.TestCase ):

    def test_boxes_come_from_the_production_pre_tracking_stages( self ):
        produced = [ { 'label': 'exposed_breast' }, { 'label': 'exposed_vulva' } ]
        with mock.patch.object( bu_tuning, 'replay_tracking_context', return_value=TRACKING ), \
             mock.patch.object( bu_tuning.bu_track, 'prepare_tracking_inputs',
                                return_value=( produced, {} ) ) as prepare:
            boxes, tracking = bu_tuning.replay_tracking_inputs(
                'h1', '/v.mp4', [ 'c.gz' ], [ { 'raw': 1 } ], 640, 360, 'nudenet_v3',
                labels={ 'exposed_vulva' } )
        prepare.assert_called_once()
        self.assertEqual( boxes, [ { 'label': 'exposed_vulva' } ] )
        self.assertIs( tracking, TRACKING )

    def test_no_label_filter_keeps_everything( self ):
        produced = [ { 'label': 'exposed_breast' }, { 'label': 'exposed_vulva' } ]
        with mock.patch.object( bu_tuning, 'replay_tracking_context', return_value=TRACKING ), \
             mock.patch.object( bu_tuning.bu_track, 'prepare_tracking_inputs',
                                return_value=( produced, {} ) ):
            boxes, _tracking = bu_tuning.replay_tracking_inputs(
                'h1', '/v.mp4', [ 'c.gz' ], [], 640, 360, 'nudenet_v3' )
        self.assertEqual( len( boxes ), 2 )


class TestTrackBreakSweepReachesProfileValues( unittest.TestCase ):

    def setUp( self ):
        self.saved = copy.deepcopy( betaconfig.detector_backend )

    def tearDown( self ):
        betaconfig.detector_backend = self.saved
        bu_config.invalidate_config_caches()

    def test_a_profile_value_is_scaled_and_restored( self ):
        backend, label = 'nudenet_v3', 'exposed_breast'
        block = betaconfig.detector_backend.setdefault( backend, {} )
        block['profiles'] = { 'variants': { 'scene': { 'item_overrides': {
            label: { 'track_max_gap': 3.0, 'match_distance_multiplier': 0.8 } } } } }
        bu_config.invalidate_config_caches()
        tool = _load_tool( 'tools/analysis/analyze_track_breaks.py', '_tb_profile_sweep' )
        with tool._swept_overrides( backend, { label }, match_mult=3.0, gap_mult=4.0 ):
            swept = bu_detector.get_profile_item_overrides( 'scene', label, backend )
            self.assertAlmostEqual( swept['track_max_gap'], 12.0 )
            self.assertAlmostEqual( swept['match_distance_multiplier'], 2.4 )
        restored = bu_detector.get_profile_item_overrides( 'scene', label, backend )
        self.assertAlmostEqual( restored['track_max_gap'], 3.0 )
        self.assertAlmostEqual( restored['match_distance_multiplier'], 0.8 )

    def test_a_profile_that_does_not_set_a_key_is_left_alone( self ):
        backend, label = 'nudenet_v3', 'exposed_breast'
        block = betaconfig.detector_backend.setdefault( backend, {} )
        block['profiles'] = { 'variants': { 'scene': { 'item_overrides': {
            label: { 'style_min_dwell_seconds': 3.0 } } } } }
        bu_config.invalidate_config_caches()
        tool = _load_tool( 'tools/analysis/analyze_track_breaks.py', '_tb_profile_untouched' )
        with tool._swept_overrides( backend, { label }, match_mult=3.0, gap_mult=4.0 ):
            swept = bu_detector.get_profile_item_overrides( 'scene', label, backend )
            self.assertNotIn( 'track_max_gap', swept )


if __name__ == '__main__':
    unittest.main()
