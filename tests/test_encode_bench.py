"""
test_encode_bench.py - betabench encode, and the encoder arguments the
real render shares with it.

The bench is only worth trusting if its baseline row is exactly what the
real render runs, so the production chunk command is pinned here first.
Then the pieces that decide the recommendation: candidates, quality
measures, the choice rule, the real-render estimate, and an end-to-end
pass on a small synthetic clip (NVENC is absent in CI, which exercises
the unavailable path).
"""

import json
import os
import shutil
import subprocess
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

import betaconfig
import betautils_render as bu_render

from tests.test_betatv_end_to_end import _ffmpeg_available
from tests.test_structure_bench import betabench, _quiet_logger


class TestEncoderOptions( unittest.TestCase ):

    def test_the_production_x264_command_is_unchanged( self ):
        saved = ( betaconfig.encode_video_codec, betaconfig.encode_crf )
        betaconfig.encode_video_codec, betaconfig.encode_crf = 'libx264', 17
        try:
            command = bu_render._chunk_encode_command( 'out.mkv', 1920, 1080, 30.0, 'fast', 2 )
        finally:
            betaconfig.encode_video_codec, betaconfig.encode_crf = saved
        self.assertEqual( command[ command.index( '-an' ) + 1 : ], [
            '-c:v', 'libx264', '-crf', '17', '-preset', 'fast', '-threads', '2',
            '-pix_fmt', 'yuv420p', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', 'out.mkv' ] )

    def test_nvenc_gets_constant_quality_not_crf( self ):
        options = bu_render.encoder_options( 'h264_nvenc', 19, 'p5', 2 )
        self.assertEqual( options, [ '-c:v', 'h264_nvenc', '-preset', 'p5', '-rc', 'vbr',
                                     '-cq', '19', '-b:v', '0' ] )
        self.assertNotIn( '-crf', options, "NVENC ignores -crf without an error" )

    def test_x264_preset_names_are_translated_for_nvenc( self ):
        for x264, nvenc in ( ( 'ultrafast', 'p1' ), ( 'fast', 'p3' ), ( 'medium', 'p4' ),
                             ( 'veryslow', 'p7' ) ):
            options = bu_render.encoder_options( 'h264_nvenc', 19, x264, 1 )
            self.assertEqual( options[ options.index( '-preset' ) + 1 ], nvenc )

    def test_native_nvenc_presets_pass_through( self ):
        options = bu_render.encoder_options( 'hevc_nvenc', 21, 'p6', 1 )
        self.assertEqual( options[ options.index( '-preset' ) + 1 ], 'p6' )

    def test_the_command_takes_an_explicit_encoder( self ):
        command = bu_render._chunk_encode_command( 'o.mkv', 64, 48, 30.0, 'p5', 2,
                                                   codec='h264_nvenc', quality=21 )
        self.assertIn( 'h264_nvenc', command )
        self.assertEqual( command[ command.index( '-cq' ) + 1 ], '21' )


class TestCandidates( unittest.TestCase ):

    def setUp( self ):
        self.saved = ( betaconfig.encode_video_codec, betaconfig.encode_preset,
                       betaconfig.encode_crf )
        betaconfig.encode_video_codec = 'libx264'
        betaconfig.encode_preset = 'fast'
        betaconfig.encode_crf = 17

    def tearDown( self ):
        ( betaconfig.encode_video_codec, betaconfig.encode_preset,
          betaconfig.encode_crf ) = self.saved

    def test_spec_parsing( self ):
        self.assertEqual( betabench.parse_encoder_spec( 'h264_nvenc:p5:19' ),
                          { 'codec': 'h264_nvenc', 'preset': 'p5', 'quality': 19 } )
        self.assertEqual( betabench.parse_encoder_spec( 'libx264:slow:17.5' )['quality'], 17.5 )
        for bad in ( 'libx264', 'libx264:fast', 'libx264::17', 'a:b:c:d' ):
            with self.assertRaises( ValueError ):
                betabench.parse_encoder_spec( bad )

    def test_the_current_encoder_always_leads( self ):
        rows = betabench.encoder_candidates( [ 'libx264:veryfast:17' ], nvenc_usable=False )
        self.assertEqual( rows[0], { 'codec': 'libx264', 'preset': 'fast', 'quality': 17 } )
        self.assertEqual( len( rows ), 2 )

    def test_the_current_encoder_is_not_timed_twice( self ):
        rows = betabench.encoder_candidates( [ 'libx264:fast:17' ], nvenc_usable=False )
        self.assertEqual( len( rows ), 1 )

    def test_defaults_without_nvenc( self ):
        rows = betabench.encoder_candidates( None, nvenc_usable=False )
        self.assertEqual( [ ( r['codec'], r['preset'] ) for r in rows ],
                          [ ( 'libx264', 'fast' ), ( 'libx264', 'faster' ), ( 'libx264', 'veryfast' ) ] )

    def test_defaults_with_nvenc_sweep_cq( self ):
        rows = betabench.encoder_candidates( None, nvenc_usable=True )
        nvenc = [ r for r in rows if r['codec'] == 'h264_nvenc' ]
        self.assertEqual( [ r['quality'] for r in nvenc if r['preset'] == 'p5' ], [ 17, 19, 21, 23 ] )
        self.assertIn( { 'codec': 'h264_nvenc', 'preset': 'p7', 'quality': 19 }, nvenc )

    def test_nvenc_is_dropped_when_unusable_even_if_asked_for( self ):
        rows = betabench.encoder_candidates( [ 'h264_nvenc:p5:19' ], nvenc_usable=False )
        self.assertEqual( [ r['codec'] for r in rows ], [ 'libx264' ] )


class TestQualityMeasures( unittest.TestCase ):

    def setUp( self ):
        rng = np.random.default_rng( 7 )
        self.frame = rng.integers( 0, 256, ( 72, 96, 3 ), dtype=np.uint8 )
        self.rng = rng

    def _noisy( self, sigma ):
        noise = self.rng.normal( 0, sigma, self.frame.shape )
        return np.clip( self.frame.astype( float ) + noise, 0, 255 ).astype( np.uint8 )

    def test_identical_frames_are_perfect( self ):
        mse, ssim = betabench.frame_quality( self.frame, self.frame.copy() )
        self.assertEqual( mse, 0.0 )
        self.assertAlmostEqual( ssim, 1.0, places=5 )
        self.assertEqual( betabench.psnr_db( mse ), 100.0 )
        self.assertEqual( betabench.ssim_db( 1.0 ), 100.0 )

    def test_more_noise_scores_lower_on_both( self ):
        light = betabench.frame_quality( self.frame, self._noisy( 3 ) )
        heavy = betabench.frame_quality( self.frame, self._noisy( 20 ) )
        self.assertGreater( heavy[0], light[0] )
        self.assertLess( heavy[1], light[1] )

    def test_db_conversions( self ):
        self.assertAlmostEqual( betabench.ssim_db( 0.99 ), 20.0, places=6 )
        self.assertAlmostEqual( betabench.psnr_db( 255.0 ** 2 / 1000.0 ), 30.0, places=6 )


class TestChoice( unittest.TestCase ):

    def row( self, name, fps, ssim, psnr, size=100, error=None ):
        return { 'codec': name, 'preset': 'x', 'quality': 0, 'fps': fps, 'ssim_db': ssim,
                 'psnr_db': psnr, 'bytes': size, 'error': error }

    def test_fastest_within_tolerance_wins( self ):
        rows = [ self.row( 'now', 18, 20.0, 40.0 ), self.row( 'a', 30, 19.9, 39.9 ),
                 self.row( 'b', 50, 19.8, 39.8 ) ]
        self.assertIs( betabench.choose_encoder( rows ), rows[2] )

    def test_faster_but_worse_is_rejected( self ):
        rows = [ self.row( 'now', 18, 20.0, 40.0 ), self.row( 'worse ssim', 90, 19.7, 40.0 ),
                 self.row( 'worse psnr', 80, 20.0, 39.7 ), self.row( 'ok', 25, 20.0, 40.0 ) ]
        self.assertIs( betabench.choose_encoder( rows ), rows[3] )

    def test_exactly_at_tolerance_counts( self ):
        rows = [ self.row( 'now', 18, 20.0, 40.0 ), self.row( 'edge', 30, 19.75, 39.75 ) ]
        self.assertIs( betabench.choose_encoder( rows ), rows[1] )

    def test_ties_go_to_the_smaller_file( self ):
        rows = [ self.row( 'now', 18, 20.0, 40.0 ), self.row( 'big', 30, 20.0, 40.0, size=200 ),
                 self.row( 'small', 30, 20.0, 40.0, size=150 ) ]
        self.assertIs( betabench.choose_encoder( rows ), rows[2] )

    def test_failed_rows_never_win( self ):
        rows = [ self.row( 'now', 18, 20.0, 40.0 ),
                 self.row( 'broken', 99, 30.0, 50.0, error='Cannot load libcuda.so.1' ) ]
        self.assertIs( betabench.choose_encoder( rows ), rows[0] )

    def test_nothing_is_judged_when_the_baseline_failed( self ):
        rows = [ self.row( 'now', 0, None, None, error='x' ), self.row( 'a', 30, 20.0, 40.0 ) ]
        self.assertIsNone( betabench.choose_encoder( rows ) )


class TestSegmentsAndEstimates( unittest.TestCase ):

    def test_windows_run_back_to_back_from_a_quarter_in( self ):
        self.assertEqual( betabench.encode_segments( 10000, 30.0, 10, 3 ),
                          [ ( 2500, 300 ), ( 2800, 300 ), ( 3100, 300 ) ] )

    def test_a_late_start_is_pulled_back_to_fit( self ):
        segments = betabench.encode_segments( 1000, 30.0, 10, 3 )
        self.assertEqual( segments[0][0] + 3 * 300, 1000 )

    def test_a_short_file_shortens_every_window( self ):
        segments = betabench.encode_segments( 100, 30.0, 10, 4 )
        self.assertEqual( segments, [ ( 0, 25 ), ( 25, 25 ), ( 50, 25 ), ( 75, 25 ) ] )

    def test_prediction_keeps_the_unmeasured_cost( self ):
        # Real 18 fr/s; bench baseline 36: half of each real frame is
        # compositing. A candidate twice as fast as the baseline only
        # removes half of the other half.
        self.assertAlmostEqual( betabench.predicted_render_fps( 18.0, 36.0, 72.0 ), 24.0 )

    def test_prediction_never_charges_a_negative_remainder( self ):
        self.assertAlmostEqual( betabench.predicted_render_fps( 40.0, 30.0, 60.0 ), 60.0 )

    def test_prediction_needs_all_three( self ):
        self.assertIsNone( betabench.predicted_render_fps( None, 30.0, 60.0 ) )

    def test_real_render_rate_from_the_stats_log( self ):
        tmp = tempfile.mkdtemp( prefix='betasuite-stats-' )
        self.addCleanup( shutil.rmtree, tmp, True )
        path = os.path.join( tmp, 'stats.jsonl' )
        base = { 'video_width': 1920, 'video_height': 1080, 'preview_mode': False,
                 'render_chunks_reused': 0 }
        records = [
            dict( base, encode_seconds=100.0, render_frames_written=2000 ),   # 20
            dict( base, encode_seconds=100.0, render_frames_written=1600 ),   # 16
            dict( base, encode_seconds=100.0, render_frames_written=1800 ),   # 18
            dict( base, encode_seconds=10.0, render_frames_written=900, preview_mode=True ),
            dict( base, encode_seconds=10.0, render_frames_written=900, render_chunks_reused=3 ),
            dict( base, encode_seconds=10.0, render_frames_written=900, video_width=1280 ),
        ]
        with open( path, 'w' ) as handle:
            for record in records:
                handle.write( json.dumps( record ) + '\n' )
            handle.write( 'not json\n' )
        saved = betaconfig.stats_path
        betaconfig.stats_path = path
        try:
            self.assertEqual( betabench.real_render_fps( 1920, 1080 ), ( 18.0, 3 ) )
        finally:
            betaconfig.stats_path = saved

    def test_error_line_names_the_cause_not_the_consequence( self ):
        text = ( "[h264_nvenc @ 0x1] Cannot load libcuda.so.1\n"
                 "Error while filtering: Operation not permitted\n"
                 "[out#0/null @ 0x2] Nothing was written into output file\n" )
        self.assertEqual( betabench.ffmpeg_error_line( text, 'h264_nvenc' ),
                          "[h264_nvenc @ 0x1] Cannot load libcuda.so.1" )
        self.assertEqual( betabench.ffmpeg_error_line(
            "Input #0, lavfi\n[h264_nvenc @ 0x1] OpenEncodeSessionEx failed\nlast\n", 'h264_nvenc' ),
            "[h264_nvenc @ 0x1] OpenEncodeSessionEx failed" )
        self.assertEqual( betabench.ffmpeg_error_line( "first\nsecond\n", 'libx264' ), "first" )
        self.assertIsNone( betabench.ffmpeg_error_line( "  \n", 'libx264' ) )


class TestNvencProbe( unittest.TestCase ):

    def test_an_unlisted_encoder_opens_nothing( self ):
        with mock.patch.object( betabench, '_ffmpeg_lists_encoder', return_value=False ):
            sessions, reason = betabench.probe_nvenc_sessions( 6 )
        self.assertEqual( sessions, 0 )
        self.assertIn( 'without h264_nvenc', reason )

    def test_counts_the_sessions_that_opened_at_once( self ):
        outcomes = iter( [ 0, 0, 0, 1, 1, 1 ] )

        class FakeProcess( object ):
            def __init__( self, *args, **kwargs ):
                self.returncode = next( outcomes )

            def communicate( self ):
                err = b"" if self.returncode == 0 else \
                    b"[h264_nvenc @ 0x1] OpenEncodeSessionEx failed: out of memory (10)\nmore\n"
                return b"", err

        with mock.patch.object( betabench, '_ffmpeg_lists_encoder', return_value=True ), \
             mock.patch( 'subprocess.Popen', FakeProcess ):
            sessions, reason = betabench.probe_nvenc_sessions( 6 )
        self.assertEqual( sessions, 3 )
        self.assertIn( 'OpenEncodeSessionEx', reason )


@unittest.skipUnless( _ffmpeg_available(), "ffmpeg is needed to build the clip" )
class TestEncodeBenchEndToEnd( unittest.TestCase ):

    @classmethod
    def setUpClass( cls ):
        cls.tmp = tempfile.mkdtemp( prefix='betasuite-encode-' )
        cls.clip = os.path.join( cls.tmp, 'clip.mp4' )
        subprocess.run(
            [ 'ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi',
              '-i', 'testsrc2=size=160x120:rate=30', '-t', '4', '-c:v', 'libx264',
              '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', cls.clip ], check=True )

    @classmethod
    def tearDownClass( cls ):
        shutil.rmtree( cls.tmp, ignore_errors=True )

    def run_bench( self, encoders, nvenc=( 0, 'Cannot load libcuda.so.1' ) ):
        args = types.SimpleNamespace( video=self.clip, seconds=1.0, workers=2,
                                      encoders=encoders, quality_stride=5 )
        with mock.patch.object( betabench, 'probe_nvenc_sessions', return_value=nvenc ):
            return betabench.bench_encode( args, _quiet_logger() )['encode']

    def test_nvenc_rows_run_only_the_sessions_that_open( self ):
        # One session of two: the NVENC row must not start a second encode
        # the driver would refuse. (Without a GPU here the row fails; on a
        # machine with NVENC it succeeds. Either way it runs one worker.)
        result = self.run_bench( [ 'h264_nvenc:p5:19' ], nvenc=( 1, 'session limit' ) )
        nvenc_row = result['rows'][1]
        self.assertEqual( nvenc_row['codec'], 'h264_nvenc' )
        self.assertEqual( nvenc_row['workers'], 1 )
        self.assertEqual( result['rows'][0]['workers'], 2 )

    def test_quality_is_scored_on_every_stride_th_frame( self ):
        calls = []
        real = betabench.frame_quality

        def counting( reference, distorted ):
            calls.append( 1 )
            return real( reference, distorted )

        with mock.patch.object( betabench, 'frame_quality', counting ):
            self.run_bench( [ 'libx264:ultrafast:17' ] )
        # Two encoders, two workers, 30 frames each, every 5th frame.
        self.assertEqual( len( calls ), 2 * 2 * 6 )

    def test_an_encode_that_drops_frames_is_a_failure( self ):
        with mock.patch.object( betabench, '_measure_encode_quality',
                                return_value=( 40.0, 20.0, 59 ) ):
            result = self.run_bench( [ 'libx264:ultrafast:17' ] )
        self.assertEqual( result['rows'][1]['error'], 'encoded 59 frames from 60' )

    def test_every_encoder_is_timed_and_scored_against_the_source( self ):
        result = self.run_bench( [ 'libx264:ultrafast:17' ] )
        rows = result['rows']
        self.assertEqual( [ r['preset'] for r in rows ], [ betaconfig.encode_preset, 'ultrafast' ] )
        for row in rows:
            self.assertIsNone( row['error'] )
            self.assertEqual( row['frames'], 2 * 30 )
            self.assertGreater( row['fps'], 0 )
            self.assertGreater( row['bytes'], 0 )
            self.assertGreater( row['ssim_db'], 10.0, "a CRF 17 encode is close to its source" )
        self.assertGreater( result['decode_fps'], 0 )
        self.assertEqual( result['nvenc_sessions'], 0 )
        self.assertEqual( result['nvenc_unavailable_reason'], 'Cannot load libcuda.so.1' )
        self.assertIsNotNone( result['recommended'] )

    def test_a_failing_encoder_is_reported_not_fatal( self ):
        result = self.run_bench( [ 'no_such_encoder:fast:17' ] )
        broken = result['rows'][1]
        self.assertIsNotNone( broken['error'] )
        self.assertIs( result['recommended'], result['rows'][0] )

    def test_nothing_is_left_behind( self ):
        import betaconst
        os.makedirs( betaconst.benchmark_dir, exist_ok=True )
        before = set( os.listdir( betaconst.benchmark_dir ) )
        self.run_bench( [ 'libx264:ultrafast:17' ] )
        self.assertEqual( set( os.listdir( betaconst.benchmark_dir ) ) - before, set(),
                          "every encoded segment is scratch and must be removed" )


if __name__ == '__main__':
    unittest.main()
