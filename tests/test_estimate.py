"""
test_estimate.py - time estimates from earlier runs' timings.

Covers betautils_estimate's arithmetic (which runs count, the rates, a
file's seconds, the preview window) and betautils_serve_estimate's
answer for job.estimate (cache state, unknown rates, look_seen), with
synthetic timings and a fake stream probe: no video, no catalog, no
clock.
"""

import os
import shutil
import tempfile
import unittest
from unittest import mock

import betautils_estimate as bu_estimate
import betautils_serve_estimate as bu_serve_estimate

IDENTITY = { 'backend': 'nudenet_v3', 'variant': '320n', 'sizes': [ 320 ], 'gpu': True, 'nn_batch_size': 4,
             'reverse': False, 'preset': 'fast' }


def timing( detect=10.0, render=20.0, total=32.0, cache='fresh', sampled=500, rendered=3000, width=1920,
            height=1080, **identity ):
    ident = dict( IDENTITY )
    ident.update( identity )
    return { 'detect_seconds': detect, 'render_seconds': render, 'total_seconds': total, 'detect_cache': cache,
             'sampled_frames': sampled, 'rendered_frames': rendered, 'width': width, 'height': height,
             'identity': ident }


class TestRates( unittest.TestCase ):

    def test_rates_from_one_matching_run( self ):
        rates = bu_estimate.rates( [ timing() ], IDENTITY )
        self.assertEqual( rates['detect'], 50.0 )
        self.assertEqual( rates['render'], 3000 * 1920 * 1080 / 20.0 )
        self.assertEqual( rates['overhead'], 2.0 )
        self.assertEqual( ( rates['detect_runs'], rates['render_runs'] ), ( 1, 1 ) )

    def test_no_runs_gives_no_rates( self ):
        rates = bu_estimate.rates( [], IDENTITY )
        self.assertEqual( ( rates['detect'], rates['render'], rates['overhead'] ), ( None, None, 0.0 ) )

    def test_the_median_of_the_matching_runs( self ):
        rates = bu_estimate.rates( [ timing( detect=10.0 ), timing( detect=5.0 ), timing( detect=50.0 ) ], IDENTITY )
        self.assertEqual( rates['detect'], 50.0 )

    def test_cached_or_resumed_detection_says_nothing_about_detect_speed( self ):
        rates = bu_estimate.rates( [ timing( cache='cached', detect=0.1 ), timing( cache='partial', detect=1.0 ) ],
                                   IDENTITY )
        self.assertIsNone( rates['detect'] )
        self.assertEqual( rates['render_runs'], 2, "their render still counts" )

    def test_another_model_or_size_is_not_compared_for_detection( self ):
        for field, value in ( ( 'variant', '640m' ), ( 'sizes', [ 640 ] ), ( 'gpu', False ), ( 'backend', 'x' ),
                              ( 'nn_batch_size', 2 ), ( 'reverse', True ) ):
            rates = bu_estimate.rates( [ timing( **{ field: value } ) ], IDENTITY )
            self.assertIsNone( rates['detect'], field )

    def test_another_preset_or_gpu_is_not_compared_for_render( self ):
        for field, value in ( ( 'preset', 'ultrafast' ), ( 'gpu', False ) ):
            self.assertIsNone( bu_estimate.rates( [ timing( **{ field: value } ) ], IDENTITY )['render'], field )
        self.assertIsNotNone( bu_estimate.rates( [ timing( variant='640m' ) ], IDENTITY )['render'],
                              "the model does not change render speed" )

    def test_only_the_most_recent_runs_count( self ):
        recent = [ timing( detect=10.0 ) ] * bu_estimate.RECENT_RUNS
        rates = bu_estimate.rates( recent + [ timing( detect=1.0 ) ] * 50, IDENTITY )
        self.assertEqual( rates['detect'], 50.0 )
        self.assertEqual( rates['detect_runs'], bu_estimate.RECENT_RUNS )

    def test_unusable_timings_are_skipped( self ):
        rates = bu_estimate.rates( [ None, 'x', timing( detect=0, render=0 ), timing( width=0 ),
                                     timing( sampled=0 ) ], IDENTITY )
        self.assertEqual( rates['detect_runs'], 1 )
        self.assertEqual( rates['render_runs'], 1, "only the one with no samples still has a render" )

    def test_overhead_is_never_negative( self ):
        self.assertEqual( bu_estimate.rates( [ timing( total=1.0 ) ], IDENTITY )['overhead'], 0.0 )


class TestFileSeconds( unittest.TestCase ):

    MEASURED = { 'detect': 50.0, 'render': 1920 * 1080 * 150.0, 'overhead': 2.0 }

    def work( self, **fields ):
        out = { 'seconds': 100.0, 'fps': 30.0, 'width': 1920, 'height': 1080, 'sample_fps': 5.0, 'sizes': 1,
                'detect_cache': 'fresh' }
        out.update( fields )
        return out

    def test_detect_render_and_overhead_add_up( self ):
        # 500 samples at 50/s, 3000 frames at 150/s, 2 s overhead
        self.assertAlmostEqual( bu_estimate.file_seconds( self.work(), self.MEASURED ), 10.0 + 20.0 + 2.0 )

    def test_every_size_is_detected( self ):
        self.assertAlmostEqual( bu_estimate.file_seconds( self.work( sizes=2 ), self.MEASURED ), 20.0 + 20.0 + 2.0 )

    def test_a_cached_detection_costs_nothing_and_needs_no_detect_rate( self ):
        measured = dict( self.MEASURED, detect=None )
        self.assertAlmostEqual( bu_estimate.file_seconds( self.work( detect_cache='cached' ), measured ), 22.0 )

    def test_a_partial_detection_is_charged_in_full( self ):
        self.assertAlmostEqual( bu_estimate.file_seconds( self.work( detect_cache='partial' ), self.MEASURED ), 32.0 )

    def test_render_scales_with_the_frame_area( self ):
        seconds = bu_estimate.file_seconds( self.work( width=3840, height=2160 ), self.MEASURED )
        self.assertAlmostEqual( seconds, 10.0 + 80.0 + 2.0 )

    def test_a_missing_rate_gives_no_estimate( self ):
        self.assertIsNone( bu_estimate.file_seconds( self.work(), dict( self.MEASURED, detect=None ) ) )
        self.assertIsNone( bu_estimate.file_seconds( self.work(), dict( self.MEASURED, render=None ) ) )


class TestWindow( unittest.TestCase ):

    def test_a_full_run_is_the_whole_file( self ):
        self.assertEqual( bu_estimate.window_seconds( 600, False, 20 ), 600.0 )

    def test_a_preview_is_its_slice_or_the_file_when_shorter( self ):
        self.assertEqual( bu_estimate.window_seconds( 600, True, 20 ), 20.0 )
        self.assertEqual( bu_estimate.window_seconds( 12, True, 20 ), 12.0 )

    def test_a_preview_with_no_length_is_the_whole_file( self ):
        self.assertEqual( bu_estimate.window_seconds( 600, True, 0 ), 600.0 )


class TestTimingFor( unittest.TestCase ):

    def test_counts_the_work_of_every_size( self ):
        out = bu_estimate.timing_for( 1.23456, 2.0, 3.5, 'fresh', 10.0, 5, [ 320, 640 ], 1920, 1080, 30.0, IDENTITY )
        self.assertEqual( out['sampled_frames'], 100 )
        self.assertEqual( out['rendered_frames'], 300 )
        self.assertEqual( out['detect_seconds'], 1.235 )
        self.assertEqual( out['identity'], IDENTITY )
        self.assertIsNot( out['identity'], IDENTITY )


def probe( width='1920', height='1080', rate='30/1', duration='100.0' ):
    return lambda path: { 'width': width, 'height': height, 'avg_frame_rate': rate, 'duration': duration }


class TestStreamFacts( unittest.TestCase ):

    def test_reads_ffprobe_fields( self ):
        self.assertEqual( bu_serve_estimate.stream_facts( 'x', probe( rate='30000/1001' ) ),
                          { 'duration': 100.0, 'fps': 30000 / 1001, 'width': 1920.0, 'height': 1080.0 } )

    def test_duration_from_the_frame_count_when_the_stream_has_none( self ):
        facts = bu_serve_estimate.stream_facts(
            'x', lambda path: { 'width': '10', 'height': '10', 'avg_frame_rate': '25/1', 'nb_frames': '250' } )
        self.assertEqual( facts['duration'], 10.0 )

    def test_an_unreadable_file_has_no_facts( self ):
        with mock.patch( 'betautils_video.probe_duration_seconds', return_value=None ):
            self.assertIsNone( bu_serve_estimate.stream_facts( 'x', lambda path: {} ) )
            self.assertIsNone( bu_serve_estimate.stream_facts( 'x', probe( rate='0/0' ) ) )
            self.assertIsNone( bu_serve_estimate.stream_facts( 'x', probe( width='' ) ) )
            self.assertIsNone( bu_serve_estimate.stream_facts( 'x', probe( height='0' ) ) )


class TestAnswer( unittest.TestCase ):

    CTX = { 'identity': IDENTITY, 'preview': False, 'preview_max': 20, 'sample_fps': 5.0, 'sizes': [ 320 ],
            'backend': 'nudenet_v3', 'min_prob': 0.2, 'censor_key': 'abc' }

    def answer( self, files=( '/a.mp4', ), ctx=None, timings=None, hash_of=None, seen=None, facts=None ):
        return bu_serve_estimate.answer(
            list( files ), ctx or self.CTX, [ timing() ] if timings is None else timings,
            hash_of or ( lambda path: None ), seen or ( lambda key: False ), facts or probe() )

    def test_a_file_from_history( self ):
        out = self.answer()
        # 500 samples / 50 per s, 3000 frames / 150 per s, 2 s overhead
        self.assertEqual( out['files'], [ { 'seconds': 32.0, 'detect_cache': 'fresh' } ] )
        self.assertEqual( out['seconds'], 32.0 )
        self.assertEqual( out['basis'], 'history' )
        self.assertEqual( out['runs'], { 'detect': 1, 'render': 1 } )

    def test_files_add_up( self ):
        self.assertEqual( self.answer( files=( '/a.mp4', '/b.mp4' ) )['seconds'], 64.0 )

    def test_no_history_says_so( self ):
        out = self.answer( timings=[] )
        self.assertEqual( ( out['seconds'], out['basis'] ), ( None, 'none' ) )
        self.assertEqual( out['files'][0]['seconds'], None )

    def test_render_history_alone_is_history_and_estimates_a_cached_file( self ):
        out = self.answer( timings=[ timing( cache='cached', detect=0.1 ) ], hash_of=lambda path: 'h',
                           ctx=dict( self.CTX, sample_fps=None ) )
        self.assertEqual( out['basis'], 'history' )
        self.assertEqual( out['runs'], { 'detect': 0, 'render': 1 } )
        self.assertIsNone( out['seconds'], "a fresh detection needs a rate no run gave" )

    def test_one_unreadable_file_leaves_the_total_unknown( self ):
        calls = iter( [ probe()( 'a' ), {} ] )
        with mock.patch( 'betautils_video.probe_duration_seconds', return_value=None ):
            out = self.answer( files=( '/a.mp4', '/b.mp4' ), facts=lambda path: next( calls ) )
        self.assertEqual( out['files'][0]['seconds'], 32.0 )
        self.assertEqual( out['files'][1], { 'seconds': None, 'detect_cache': None } )
        self.assertIsNone( out['seconds'] )

    def test_a_preview_is_its_slice( self ):
        out = self.answer( ctx=dict( self.CTX, preview=True ) )
        # 100 samples / 50, 600 frames / 150, 2 s overhead
        self.assertEqual( out['seconds'], 8.0 )

    def test_look_seen_asks_for_the_jobs_censor_key( self ):
        asked = []
        self.assertTrue( self.answer( seen=lambda key: asked.append( key ) or True )['look_seen'] )
        self.assertEqual( asked, [ 'abc' ] )
        self.assertFalse( self.answer()['look_seen'] )


class TestCacheState( unittest.TestCase ):

    CTX = dict( TestAnswer.CTX, sizes=[ 320, 640 ] )
    FACTS = { 'duration': 100.0, 'fps': 30.0, 'width': 1920.0, 'height': 1080.0 }

    def setUp( self ):
        self.dir = tempfile.mkdtemp( prefix='estimate-cache-' )
        self.addCleanup( shutil.rmtree, self.dir, True )
        patcher = mock.patch( 'betautils_cache_paths.box_hash_path_for',
                              side_effect=lambda digest, size, *rest: os.path.join( self.dir, '%s-%d.json'%( digest, size ) ) )
        patcher.start()
        self.addCleanup( patcher.stop )

    def touch( self, name ):
        with open( os.path.join( self.dir, name ), 'w' ) as handle:
            handle.write( '[]' )

    def state( self, digest='h', ctx=None ):
        return bu_serve_estimate.cache_state( digest, self.FACTS, ctx or self.CTX )

    def test_fresh_partial_and_cached( self ):
        self.assertEqual( self.state(), 'fresh' )
        self.touch( 'h-320.json' )
        self.assertEqual( self.state(), 'partial' )
        self.touch( 'h-640.json' )
        self.assertEqual( self.state(), 'cached' )

    def test_a_checkpoint_is_partial( self ):
        with mock.patch( 'betautils_hash.has_checkpoint', side_effect=lambda path: path.endswith( '640.json' ) ):
            self.assertEqual( self.state(), 'partial' )

    def test_an_unknown_file_or_a_preview_is_fresh( self ):
        self.touch( 'h-320.json' )
        self.touch( 'h-640.json' )
        self.assertEqual( self.state( digest=None ), 'fresh' )
        self.assertEqual( self.state( ctx=dict( self.CTX, preview=True ) ), 'fresh' )


class TestPreviewOverridesWhileBusy( unittest.TestCase ):

    BASE = dict( TestAnswer.CTX, identity=dict( IDENTITY, preset='fast' ) )

    def test_preview_overrides_apply_over_the_busy_settings( self ):
        with mock.patch( 'betaconfig.preview_encode_preset', 'ultrafast', create=True ):
            out = bu_serve_estimate._with_preview_overrides(
                self.BASE, { ( 'preview_mode_enabled', ): True, ( 'preview_max_seconds', ): 30 } )
        self.assertEqual( ( out['preview'], out['preview_max'], out['identity']['preset'] ), ( True, 30, 'ultrafast' ) )
        self.assertEqual( self.BASE['identity']['preset'], 'fast', "the busy settings are not changed" )

    def test_other_overrides_are_not_applied( self ):
        out = bu_serve_estimate._with_preview_overrides( self.BASE, { ( 'encode_preset', ): 'slow' } )
        self.assertEqual( out, self.BASE )


if __name__ == '__main__':
    unittest.main()
