"""
test_tune_machine.py - the hardware tuning harness's choices: what it
measures, what it proposes, and that what it proposes is something the
machine config editor accepts.
"""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import betaconfig

REPO = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )


def _load( relative, name ):
    spec = importlib.util.spec_from_file_location( name, os.path.join( REPO, relative ) )
    module = importlib.util.module_from_spec( spec )
    spec.loader.exec_module( module )
    return module


tune = _load( 'tools/bench/tune_machine.py', 'tune_machine_under_test' )


class TestChoosing( unittest.TestCase ):

    def test_a_near_tie_goes_to_the_cheaper_setting( self ):
        rows = [ { 'n': 1, 'speed': 10.0 }, { 'n': 2, 'speed': 19.6 }, { 'n': 4, 'speed': 20.0 } ]
        choice = tune.fastest_within_noise( rows, lambda r: r['speed'], lambda r: r['n'] )
        self.assertEqual( choice['n'], 2 )
        rows[1]['speed'] = 19.0
        self.assertEqual( tune.fastest_within_noise( rows, lambda r: r['speed'], lambda r: r['n'] )['n'], 4 )
        self.assertIsNone( tune.fastest_within_noise( [ { 'n': 1, 'speed': 0 } ],
                                                      lambda r: r['speed'], lambda r: r['n'] ) )

    def test_decoder_threads_shared_out_only_when_it_pays( self ):
        def rows( default_fps, default_mb, shared_fps, shared_mb ):
            return [ { 'decode_threads_setting': None, 'fps': default_fps, 'peak_growth_mb': default_mb },
                     { 'decode_threads_setting': 0, 'fps': shared_fps, 'peak_growth_mb': shared_mb } ]
        pick = lambda *a: tune.decoder_mode_choice( rows( *a ) )['decode_threads_setting']
        self.assertEqual( pick( 50, 4000, 56, 4000 ), 0, "clearly faster" )
        self.assertEqual( pick( 50, 4000, 50, 2500 ), 0, "as fast, clearly lighter" )
        self.assertIsNone( pick( 50, 4000, 50.5, 3990 ), "a tie keeps what every render did" )
        self.assertIsNone( pick( 50, 4000, 45, 2000 ), "lighter but slower is not chosen" )

    @staticmethod
    def _rows( *pairs ):
        return [ { 'workers': workers, 'fps': fps } for workers, fps in pairs ]

    def test_fewer_workers_at_the_hardest_file_are_never_proposed( self ):
        # The 2026-11-03 run on a 16-core laptop: a 4096x2160 file, 4 and
        # more workers skipped for memory. Proposing 2 would also have
        # held 1080p renders, which ran fine at 8, to 2.
        rows = self._rows( ( 1, 31.76 ), ( 1, 31.22 ), ( 2, 37.97 ), ( 2, 37.93 ) )
        value, why = tune.render_workers_proposal( rows, [ 4, 6, 8, 10, 12 ], 16, 0, '4096x2160' )
        self.assertIsNone( value )
        self.assertIn( 'not measured', why )
        self.assertIn( 'memory cap', why )
        value, why = tune.render_workers_proposal( self._rows( ( 4, 50 ), ( 8, 50.5 ) ), [], 16, 0, '1920x1080' )
        self.assertIsNone( value, "fewer was as fast with nothing skipped: still not proposed" )

    def test_more_workers_are_proposed_when_measured_faster( self ):
        rows = self._rows( ( 4, 40 ), ( 8, 60 ), ( 10, 66 ), ( 12, 66.5 ) )
        value, why = tune.render_workers_proposal( rows, [], 16, 0, '1920x1080' )
        self.assertEqual( value, 10 )
        self.assertIn( 'against 60.0 at 8', why )
        self.assertIsNone( tune.render_workers_proposal( self._rows( ( 8, 60 ), ( 10, 61 ) ), [], 16, 0, 'x' )[0],
                           "within noise of the current count" )
        self.assertIsNone( tune.render_workers_proposal( self._rows( ( 10, 61 ) ), [], 16, 0, 'x' )[0],
                           "the current count was never measured" )
        self.assertEqual( tune.render_workers_proposal( self._rows( ( 4, 30 ), ( 8, 60 ) ), [], 16, 4, 'x' )[0], 0,
                          "the auto count is written as 0" )

    def test_the_bar_is_the_measured_spread_when_that_is_wider( self ):
        self.assertAlmostEqual( tune.measured_noise( 22.01, 23.18 ), 0.0532, places=3 )
        self.assertEqual( tune.measured_noise( 10.0, 10.1 ), tune.NOISE )
        self.assertEqual( tune.measured_noise( 0, 10 ), tune.NOISE )

    def test_memory_proposal_takes_the_worst_case_with_margin( self ):
        rows = [ { 'mb_per_megapixel': 90.0 }, { 'mb_per_megapixel': 121.0 }, { 'mb_per_megapixel': None } ]
        self.assertEqual( tune.mb_per_megapixel_proposal( rows ), 160 )   # 151.25 rounded up
        self.assertEqual( tune.mb_per_megapixel_proposal( [ { 'mb_per_megapixel': 10.0 } ] ), 50 )
        self.assertIsNone( tune.mb_per_megapixel_proposal( [] ) )

    def test_worker_counts_stay_within_cores_and_nvenc_sessions( self ):
        self.assertEqual( tune.render_worker_counts( 16 ), [ 1, 2, 4, 6, 8, 10, 12, 16 ] )
        self.assertEqual( tune.render_worker_counts( 16, nvenc_sessions=5 ), [ 1, 2, 4 ] )
        self.assertEqual( tune.render_worker_counts( 2 ), [ 1, 2 ] )
        self.assertEqual( tune.render_worker_counts( 6 ), [ 1, 2, 3, 4, 6 ] )
        self.assertEqual( tune.auto_render_workers( 16 ), 8 )
        self.assertEqual( tune.auto_render_workers( 1 ), 1 )

    def test_batch_and_prefetch_candidates_include_the_current_value( self ):
        self.assertEqual( tune.batch_sizes_to_try( 4 ), [ 1, 2, 4, 8, 16 ] )
        self.assertEqual( tune.batch_sizes_to_try( 6 ), [ 1, 2, 4, 6, 8, 16 ] )
        self.assertEqual( tune.prefetch_depths_to_try( 4, 8 ), [ 0, 4, 8, 16 ] )
        self.assertEqual( tune.prefetch_depths_to_try( 4, 6 ), [ 0, 4, 6, 8, 16 ] )

    def test_busy_reasons( self ):
        self.assertEqual( tune.busy_reasons( 0.5, 16, { 'utilization.gpu': '3' }, None ), [] )
        reasons = tune.busy_reasons( 9.0, 16, { 'utilization.gpu': '87' }, 4242 )
        self.assertEqual( len( reasons ), 3 )
        self.assertIn( 'pid 4242', reasons[0] )
        self.assertEqual( tune.busy_reasons( None, None, {}, None ), [] )


class TestProposalIsWhatTheEditorTakes( unittest.TestCase ):

    def test_every_proposed_line_parses( self ):
        tool = _load( 'tools/setup/machine_config.py', 'machine_config_for_tune' )
        proposal = { 'render_workers': { 'value': 6, 'reason': 'x' },
                     'render_decode_threads': { 'value': None, 'reason': 'x' },
                     'encode_video_codec': { 'value': 'h264_nvenc', 'reason': 'x' },
                     'detector_backend.nudenet_v3.nn_batch_size': { 'value': 8, 'reason': 'x' } }
        proposal['preview_encode_preset'] = { 'value': '7', 'reason': 'a string that looks like a number' }
        parsed = dict( tool.parse_assignment( line ) for line in tune.proposal_args( proposal ) )
        self.assertEqual( parsed, {
            ( 'render_workers', ): 6, ( 'render_decode_threads', ): None,
            ( 'encode_video_codec', ): 'h264_nvenc', ( 'preview_encode_preset', ): '7',
            ( 'detector_backend', 'nudenet_v3', 'nn_batch_size' ): 8 } )


class TestMemorySampling( unittest.TestCase ):

    def test_children_and_rss_from_proc( self ):
        root = tempfile.mkdtemp( prefix='betasuite-proc-' )
        self.addCleanup( shutil.rmtree, root, True )
        def process( pid, ppid, rss_kb, comm='python3' ):
            os.makedirs( os.path.join( root, str( pid ) ) )
            with open( os.path.join( root, str( pid ), 'stat' ), 'w' ) as handle:
                handle.write( "%d (%s) S %d 1 1\n"%( pid, comm, ppid ) )
            with open( os.path.join( root, str( pid ), 'status' ), 'w' ) as handle:
                handle.write( "Name:\t%s\nVmRSS:\t%d kB\n"%( comm, rss_kb ) )
        process( 100, 1, 1024 * 300 )
        process( 101, 100, 1024 * 50, comm='ff mpeg (x)' )
        process( 102, 100, 1024 * 25 )
        process( 103, 7, 1024 * 999 )
        os.makedirs( os.path.join( root, 'self' ) )
        self.assertEqual( sorted( tune.child_pids( 100, root ) ), [ 101, 102 ] )
        self.assertAlmostEqual( tune.tree_rss_mb( 100, root ), 375.0 )

    def test_the_sampler_splits_python_from_the_encoders( self ):
        reads = iter( [ ( 100.0, 0.0 ), ( 400.0, 900.0 ), ( 300.0, 1200.0 ) ] + [ ( 150.0, 10.0 ) ] * 100 )
        with tune.MemorySampler( interval=0.01, rss=lambda: next( reads ), available=lambda: None ) as sampler:
            import time
            time.sleep( 0.1 )
        self.assertEqual( sampler.peak_mb, 1500.0 )
        self.assertEqual( sampler.own_growth_mb(), 300.0 )
        self.assertEqual( sampler.peak_children_mb, 1200.0 )

    def test_the_sampler_keeps_the_peak_and_the_low( self ):
        rss = iter( [ 100.0, 300.0, 250.0, 260.0, 260.0, 260.0 ] + [ 200.0 ] * 100 )
        available = iter( [ 9000.0, 4000.0, 6000.0 ] + [ 7000.0 ] * 100 )
        with tune.MemorySampler( interval=0.01, rss=lambda: next( rss ),
                                 available=lambda: next( available ) ) as sampler:
            import time
            time.sleep( 0.1 )
        self.assertEqual( sampler.baseline_mb, 100.0 )
        self.assertEqual( sampler.peak_mb, 300.0 )
        self.assertEqual( sampler.growth_mb(), 200.0 )
        self.assertEqual( sampler.min_available_mb, 4000.0 )


@unittest.skipUnless( shutil.which( 'ffmpeg' ), 'needs ffmpeg' )
class TestStagesWithAStandInModel( unittest.TestCase ):

    def setUp( self ):
        self.dir = tempfile.mkdtemp( prefix='betasuite-tune-' )
        self.addCleanup( shutil.rmtree, self.dir, True )
        self.video = os.path.join( self.dir, 'clip.mp4' )
        subprocess.run( [ 'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i',
                          'testsrc2=size=96x64:rate=10:duration=12', '-c:v', 'libx264',
                          '-preset', 'ultrafast', self.video ], check=True )
        class Model( object ):
            @staticmethod
            def raw_boxes_for_imgs( frames, size, session, times ):
                return [ { 'class_id': 'exposed_breast', 'score': 0.5, 't': t } for t in times ]
        self.context = {
            'args': argparse.Namespace( detect_frames=16, prefetch_seconds=3.0 ),
            'betaconfig': betaconfig, 'proposal': {}, 'results': {}, 'batch_now': 4, 'prefetch_now': 8,
            'video': tune._video_info( self.video ),
            'detector': ( 'nudenet_v3', Model, None, 320 ),
        }
        self.logger = mock.Mock()

    def test_detect_measures_every_batch_and_checks_detections( self ):
        with mock.patch.object( tune, 'nvidia_smi', return_value={} ):
            result = tune.stage_detect( self.context, self.logger )
        self.assertEqual( [ row['batch'] for row in result['rows'] ], [ 1, 2, 4, 8, 16 ] )
        self.assertTrue( all( row['same_as_batch_1'] for row in result['rows'] ) )
        self.assertTrue( all( row['detections'] == 16 for row in result['rows'] ) )
        self.assertIn( self.context['batch_best'], ( 1, 2, 4, 8, 16 ) )

    def test_a_gain_inside_the_current_settings_own_spread_is_not_proposed( self ):
        # 2026-11-03, 1080p: batch 2 timed 10% faster than batch 4 while
        # batch 1 and 16 sat beside 4. Here batch 2 is 8% faster, and
        # batch 4 timed again comes out 15% slower than the first time.
        import time
        seen = []
        class Jittery( object ):
            @staticmethod
            def raw_boxes_for_imgs( frames, size, session, times ):
                seen.append( len( frames ) )
                repeat = 16 in seen[ : -1 ]
                per_frame = 0.0020 * ( 0.92 if len( frames ) == 2 else 1.0 ) * ( 1.15 if repeat else 1.0 )
                time.sleep( per_frame * len( frames ) )
                return [ { 'class_id': 'exposed_breast', 'score': 0.5, 't': t } for t in times ]
        self.context['detector'] = ( 'nudenet_v3', Jittery, None, 320 )
        with mock.patch.object( tune, 'nvidia_smi', return_value={} ):
            result = tune.stage_detect( self.context, self.logger )
        self.assertGreater( result['noise'], 0.08 )
        self.assertEqual( self.context['proposal'], {} )
        current = next( r for r in result['rows'] if r['batch'] == 4 )
        self.assertIn( 'ms_per_frame_repeat', current )

    def test_detections_that_change_with_the_batch_are_flagged_and_not_proposed( self ):
        class Rounding( object ):
            @staticmethod
            def raw_boxes_for_imgs( frames, size, session, times ):
                label = 'exposed_breast' if len( frames ) < 8 else 'covered_breast'
                return [ { 'class_id': label, 'score': 0.5, 't': t } for t in times ]
        self.context['detector'] = ( 'nudenet_v3', Rounding, None, 320 )
        with mock.patch.object( tune, 'nvidia_smi', return_value={} ):
            result = tune.stage_detect( self.context, self.logger )
        flags = { row['batch']: row['same_as_batch_1'] for row in result['rows'] }
        self.assertEqual( flags, { 1: True, 2: True, 4: True, 8: False, 16: False } )
        self.assertIn( result['chosen'], ( 1, 2, 4 ) )
        self.assertTrue( self.logger.warning.called )

    def test_prefetch_runs_the_detection_loop_at_each_depth( self ):
        import betautils_video as bu_video
        self.context['batch_best'] = 2
        depths = []
        real = bu_video.prefetched
        def spy( iterable, depth ):
            depths.append( depth )
            return real( iterable, depth )
        with mock.patch.object( bu_video, 'prefetched', side_effect=spy ):
            result = tune.stage_prefetch( self.context, self.logger )
        self.assertEqual( depths, [ 0, 0, 2, 4, 8, 0 ], "a warm-up, then each depth, then none again" )
        self.assertEqual( [ row['depth'] for row in result['rows'] ], [ 0, 2, 4, 8, 0 ] )
        self.assertTrue( all( row['samples'] == result['rows'][0]['samples'] > 0 for row in result['rows'] ) )


class TestScripts( unittest.TestCase ):

    def test_the_shell_wrapper_parses_and_replays_the_summary( self ):
        script = os.path.join( REPO, 'tools', 'bench', 'tune_machine.sh' )
        if shutil.which( 'bash' ):
            done = subprocess.run( [ 'bash', '-n', script ], capture_output=True, text=True )
            self.assertEqual( done.returncode, 0, done.stderr )
        with open( script, encoding='utf-8' ) as handle:
            text = handle.read()
        self.assertIn( tune.SUMMARY_START, text )
        self.assertIn( tune.SUMMARY_END, text )
        self.assertIn( 'wait_for_confirmation', text, "asks before writing" )
        self.assertIn( "trap '", text, "Ctrl+C at the prompt says how to write later" )
        self.assertNotIn( '—', text )


if __name__ == '__main__':
    unittest.main()
