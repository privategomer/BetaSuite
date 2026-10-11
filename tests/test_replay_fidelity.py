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
            self.assertIs( kwargs.get( 'logger' ), bu_tuning._QUIET_LOGGER,
                           "a replay must not write per-video lines into the run log" )

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


class TestReplayRendered( unittest.TestCase ):

    def test_the_production_pipeline_gets_profile_cuts_rate_and_a_quiet_logger( self ):
        seen = {}

        def fake_prepare( raw, w, h, **kwargs ):
            seen.update( kwargs, raw=raw, size=( w, h ) )
            return [ 'box' ], { 'interpolated': 3 }

        with mock.patch.object( bu_tuning, 'replay_tracking_context', return_value=TRACKING ), \
             mock.patch.object( bu_tuning.bu_track, 'prepare_boxes_for_render', fake_prepare ):
            rendered, stats, tracking = bu_tuning.replay_rendered(
                'h1', '/v.mp4', [ 'c.gz' ], [ { 'class_id': 'x' } ], 640, 360, 'nudenet_v3' )
        self.assertEqual( rendered, [ 'box' ] )
        self.assertEqual( stats, { 'interpolated': 3 } )
        self.assertIs( tracking, TRACKING )
        self.assertEqual( seen['profile_name'], 'compilation' )
        self.assertEqual( seen['shot_cut_times'], [ 1.5, 4.0 ] )
        self.assertEqual( seen['sample_fps'], 9 )
        self.assertEqual( seen['backend_name'], 'nudenet_v3' )
        self.assertIs( seen['logger'], bu_tuning._QUIET_LOGGER )
        self.assertEqual( seen['size'], ( 640, 360 ) )

    def test_a_given_context_is_used_as_is( self ):
        given = dict( TRACKING, profile_name='quick_cut' )
        with mock.patch.object( bu_tuning, 'replay_tracking_context' ) as context, \
             mock.patch.object( bu_tuning.bu_track, 'prepare_boxes_for_render',
                                return_value=( [], {} ) ) as prepare:
            bu_tuning.replay_rendered( 'h1', '/v.mp4', [], [], 640, 360, 'nudenet_v3', tracking=given )
        context.assert_not_called()
        self.assertEqual( prepare.call_args.kwargs['profile_name'], 'quick_cut' )


class TestItemOverridesInEffect( unittest.TestCase ):
    """A replay_tune variant must be what the pipeline reads, whichever tier sets the key."""

    BACKEND = '_overrides_test_backend'

    def setUp( self ):
        self.saved_backends = copy.deepcopy( betaconfig.detector_backend )
        self.saved_shared = copy.deepcopy( betaconfig.item_overrides )
        betaconfig.detector_backend[self.BACKEND] = {
            'item_overrides': { 'exposed_breast': { 'track_max_gap': 0.5 } },
            'model_variant': 'big',
            'variants': { 'big': { 'item_overrides': { 'exposed_breast': { 'track_max_gap': 0.7 } } },
                          'small': { 'item_overrides': { 'exposed_breast': { 'track_max_gap': 0.9 } } } },
            'profiles': { 'variants': {
                'scene': { 'item_overrides': { 'exposed_breast': { 'track_max_gap': 3.0 } } },
                'quick_cut': { 'item_overrides': { 'exposed_breast': { 'time_safety': 0.06 } } } } },
        }
        bu_config.invalidate_config_caches()

    def tearDown( self ):
        betaconfig.detector_backend = self.saved_backends
        betaconfig.item_overrides = self.saved_shared
        bu_config.invalidate_config_caches()

    def block( self ):
        return betaconfig.detector_backend[self.BACKEND]

    def test_a_shadowed_key_is_patched_on_every_tier_that_sets_it( self ):
        before = copy.deepcopy( self.block() )
        with bu_tuning.item_overrides_in_effect(
                self.BACKEND, { 'exposed_breast': { 'track_max_gap': 1.25 } }, 'scene' ):
            self.assertEqual( bu_detector.get_item_overrides( 'exposed_breast', self.BACKEND )['track_max_gap'],
                              1.25 )
            self.assertEqual( bu_detector.get_profile_item_overrides(
                'scene', 'exposed_breast', self.BACKEND )['track_max_gap'], 1.25 )
            # the inactive variant and a profile that never set the key stay as they were
            self.assertEqual( self.block()['variants']['small']['item_overrides']['exposed_breast'],
                              { 'track_max_gap': 0.9 } )
            self.assertNotIn( 'track_max_gap', self.block()['profiles']['variants']['quick_cut'][
                'item_overrides']['exposed_breast'] )
        self.assertEqual( self.block(), before )

    def test_another_profile_is_not_touched( self ):
        with bu_tuning.item_overrides_in_effect(
                self.BACKEND, { 'exposed_breast': { 'track_max_gap': 1.25 } }, 'quick_cut' ):
            self.assertEqual( bu_detector.get_profile_item_overrides(
                'scene', 'exposed_breast', self.BACKEND )['track_max_gap'], 3.0 )

    def test_a_render_key_goes_to_the_shared_tier( self ):
        with bu_tuning.item_overrides_in_effect(
                self.BACKEND, { 'exposed_breast': { 'censor_shape': 'box' } } ):
            self.assertEqual( betaconfig.item_overrides['exposed_breast']['censor_shape'], 'box' )
            self.assertNotIn( 'censor_shape', self.block()['item_overrides']['exposed_breast'] )
            self.assertEqual( bu_detector.get_item_overrides( 'exposed_breast', self.BACKEND )['censor_shape'],
                              'box' )
        self.assertEqual( betaconfig.item_overrides, self.saved_shared )

    def test_restored_after_an_exception_and_for_a_new_backend( self ):
        with self.assertRaises( RuntimeError ):
            with bu_tuning.item_overrides_in_effect( '_brand_new', { 'exposed_breast': { 'track_max_gap': 1.0 } } ):
                self.assertIn( '_brand_new', betaconfig.detector_backend )
                raise RuntimeError( 'boom' )
        self.assertNotIn( '_brand_new', betaconfig.detector_backend )
        self.assertEqual( betaconfig.item_overrides, self.saved_shared )


class TestReplayTuneVariants( unittest.TestCase ):
    """replay_tune: the variant is what the production pipeline reads."""

    def setUp( self ):
        self.tool = _load_tool( 'tools/tuning/replay_tune.py', '_replay_tune_fidelity' )
        self.saved = copy.deepcopy( betaconfig.detector_backend )

    def tearDown( self ):
        betaconfig.detector_backend = self.saved
        bu_config.invalidate_config_caches()

    def test_overrides_and_suppression_reach_the_pipeline_then_restore( self ):
        backend = bu_detector.selected_backend_name()
        seen = {}

        def fake_replay( file_hash, video_path, cache_paths, raw, w, h, backend_name, tracking=None ):
            seen['gap'] = bu_detector.get_profile_item_overrides(
                tracking['profile_name'], 'exposed_breast', backend_name ).get( 'track_max_gap' )
            seen['backend_gap'] = bu_detector.get_item_overrides( 'exposed_breast', backend_name ).get(
                'track_max_gap' )
            seen['rules'] = bu_detector.get_class_suppression( backend_name )
            seen['tracking'] = tracking
            return [ 'a', 'b' ], { 'after_suppression': 7, 'label_counts': { 'exposed_breast': 7 },
                                   'interpolated': 1,
                                   'suppression_counts': { 'r': { 'total': 4, 'renderable': 2 } } }, tracking

        rules = { 'exposed_breast': [ { 'suppressed_by': 'belly', 'margin': 0.5, 'min_iou': 0.9 } ] }
        before_rules = copy.deepcopy( bu_detector.get_class_suppression( backend ) )
        with mock.patch.object( self.tool.bu_tuning, 'replay_tracking_context', return_value=TRACKING ):
            result = self.tool.run_variant(
                [ { 'class_id': 'x' } ] * 9, 640, 360, 'h1', '/v.mp4', [ 'c.gz' ],
                item_overrides_patch={ 'exposed_breast': { 'track_max_gap': 0.42 } },
                class_suppression_patch=rules, replay=fake_replay )
        self.assertEqual( seen['backend_gap'], 0.42 )
        if 'track_max_gap' in bu_detector.get_profile_item_overrides( 'compilation', 'exposed_breast', backend ):
            self.assertEqual( seen['gap'], 0.42, "the profile value would otherwise win" )
        self.assertEqual( seen['rules']['exposed_breast'], rules['exposed_breast'] )
        for label, label_rules in before_rules.items():
            if label != 'exposed_breast':
                self.assertEqual( seen['rules'][label], label_rules,
                                  "a patch replaces only the labels it lists" )
        self.assertIs( seen['tracking'], TRACKING )
        self.assertEqual( bu_detector.get_class_suppression( backend ), before_rules )
        self.assertEqual( result, {
            'raw_detections': 9, 'surviving_after_suppression': 7, 'suppression_counts': { 'r': 4 },
            'label_counts': { 'exposed_breast': 7 }, 'tracked_boxes': 2, 'interpolated_boxes': 1,
            'profile': 'compilation' } )


class TestRepairedTools( unittest.TestCase ):

    def test_compare_effectiveness_imports_and_replays_each_config_at_its_own_caches( self ):
        tool = _load_tool( 'tools/analysis/compare_effectiveness.py', '_compare_eff' )
        calls = []

        def fake_replay( file_hash, video_path, cache_paths, raw, w, h, backend_name ):
            calls.append( ( file_hash, cache_paths, backend_name ) )
            raw.clear()
            return [ 'box' ], {}, TRACKING

        raw = [ { 'class_id': 'x' } ]
        rendered = tool.run_pipeline_for_config( 'h1', '/v.mp4', [ 'a.gz' ], raw, 640, 360,
                                                 'nudenet_v3', replay=fake_replay )
        self.assertEqual( rendered, [ 'box' ] )
        self.assertEqual( calls, [ ( 'h1', [ 'a.gz' ], 'nudenet_v3' ) ] )
        self.assertEqual( raw, [ { 'class_id': 'x' } ], "the caller's detections are not consumed" )

    def test_span_merge_and_replay_tune_use_the_production_replay( self ):
        for relative in ( 'tools/analysis/analyze_span_merge.py', 'tools/tuning/replay_tune.py' ):
            with open( _repo_path( relative ) ) as handle:
                source = handle.read()
            self.assertIn( 'bu_tuning.replay_rendered', source, relative )
            self.assertNotIn( 'process_raw_box(', source, relative )
