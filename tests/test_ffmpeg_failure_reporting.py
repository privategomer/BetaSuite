"""
tests/test_ffmpeg_failure_reporting.py - what the render says when ffmpeg fails.

On 2026-10-23/24 two files each lost one render chunk to an encoder that
quit within seconds, three attempts running, and the log said only
BrokenPipeError: ffmpeg's own message went to the terminal and its exit
status was never read. These tests pin that the encoder's reason now
reaches the error and the log file, that capturing stderr cannot
deadlock or grow without bound, and that a failed file leaves a record
beside the stats instead of vanishing from them.
"""

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )

import betaconfig
import betautils_log as bu_log
import betautils_render as bu_render


def _captured( data ):
    """A finished FfmpegStderr over the given bytes."""
    return bu_render.FfmpegStderr( io.BufferedReader( io.BytesIO( data ) ) ).finish()


class TestFfmpegStderr( unittest.TestCase ):

    def test_carriage_returns_and_newlines_both_end_a_line( self ):
        captured = _captured( b'banner\nframe=1\rframe=2\r\n\nlast line' )
        self.assertEqual( captured.lines(), [ 'banner', 'frame=1', 'frame=2', 'last line' ] )
        self.assertEqual( captured.line_count, 4 )

    def test_keeps_the_first_and_last_lines_and_says_how_many_it_dropped( self ):
        data = b''.join( b'line %d\n'%( index, ) for index in range( 100 ) )
        captured = _captured( data )
        lines = captured.lines()
        head = bu_render.FfmpegStderr.HEAD_LINES
        tail = bu_render.FfmpegStderr.TAIL_LINES
        self.assertEqual( captured.line_count, 100 )
        self.assertEqual( lines[:head], [ 'line %d'%( index, ) for index in range( head ) ] )
        self.assertEqual( lines[head], '... %d line(s) not kept ...'%( 100 - head - tail, ) )
        self.assertEqual( lines[head + 1:], [ 'line %d'%( index, ) for index in range( 100 - tail, 100 ) ] )

    def test_no_marker_when_nothing_was_dropped( self ):
        lines = _captured( b'a\nb\nc\n' ).lines()
        self.assertEqual( lines, [ 'a', 'b', 'c' ] )

    def test_an_endless_unbroken_line_is_cut_rather_than_held_whole( self ):
        captured = _captured( b'x' * ( bu_render.FfmpegStderr.MAX_LINE_BYTES * 5 ) )
        self.assertEqual( [ len( line ) for line in captured.lines() ],
                          [ bu_render.FfmpegStderr.MAX_LINE_BYTES ] )

    def test_the_line_naming_the_encoder_wins_over_the_first( self ):
        captured = _captured( b'x264 [error]: invalid profile\n'
                              b'[libx264 @ 0x1] Error setting profile no_such_profile.\n'
                              b'Nothing was written into output file\n' )
        self.assertEqual( captured.error_line( 'libx264' ),
                          '[libx264 @ 0x1] Error setting profile no_such_profile.' )
        self.assertEqual( captured.error_line(), 'x264 [error]: invalid profile' )
        self.assertIsNone( _captured( b'' ).error_line( 'libx264' ) )

    def test_a_process_that_floods_stderr_still_finishes( self ):
        # The reason this drains on a thread: an unread pipe fills at
        # 64 KB and the child blocks forever. 2 MB is well past that.
        child = [ sys.executable, '-c',
                  "import sys\nfor _ in range(20000): sys.stderr.write('e' * 100 + '\\n')" ]
        process = subprocess.Popen( child, stderr=subprocess.PIPE )
        captured = bu_render.FfmpegStderr( process.stderr )
        self.assertEqual( process.wait( timeout=30 ), 0 )
        captured.finish()
        self.assertEqual( captured.line_count, 20000 )

    def test_betabench_reads_the_same_function( self ):
        sys.path.insert( 0, os.path.join( os.path.dirname( os.path.dirname(
            os.path.abspath( __file__ ) ) ), 'tools', 'bench' ) )
        import betabench
        self.assertIs( betabench.ffmpeg_error_line, bu_render.ffmpeg_error_line )


class TestChunkFailureReason( unittest.TestCase ):

    NVENC_STDERR = ( b'[h264_nvenc @ 0x1] OpenEncodeSessionEx failed: out of memory (10)\n'
                     b'Error while opening encoder\n' )

    def test_a_broken_pipe_names_the_encoder_and_its_reason_instead( self ):
        reason = bu_render.chunk_failure_reason(
            1, 7, BrokenPipeError( 32, 'Broken pipe' ), _captured( self.NVENC_STDERR ), 'h264_nvenc' )
        self.assertEqual( reason, "ffmpeg exited 1 after 7 frame(s) written: "
                                  "[h264_nvenc @ 0x1] OpenEncodeSessionEx failed: out of memory (10)" )
        self.assertNotIn( 'BrokenPipe', reason )

    def test_a_separate_frame_loop_error_is_kept_beside_the_encoder_reason( self ):
        reason = bu_render.chunk_failure_reason(
            1, 3, RuntimeError( 'censor blew up' ), _captured( self.NVENC_STDERR ), 'h264_nvenc' )
        self.assertTrue( reason.startswith( "ffmpeg exited 1 after 3 frame(s) written: [h264_nvenc" ) )
        self.assertTrue( reason.endswith( "; the frame loop also raised RuntimeError('censor blew up')" ) )

    def test_an_encoder_that_said_nothing_still_reports_its_exit( self ):
        self.assertEqual( bu_render.chunk_failure_reason( 234, 0, None, _captured( b'' ), 'libx264' ),
                          "ffmpeg exited 234 after 0 frame(s) written" )
        self.assertEqual( bu_render.chunk_failure_reason( 1, 0, None, None, None ),
                          "ffmpeg exited 1 after 0 frame(s) written" )

    def test_killed_by_a_signal_counts_as_a_failure( self ):
        self.assertTrue( bu_render.chunk_failure_reason( -9, 5, None, None, 'libx264' )
                         .startswith( "ffmpeg exited -9" ) )

    def test_a_frame_loop_error_with_a_healthy_encoder_is_reported_as_itself( self ):
        error = RuntimeError( 'decoder fell over' )
        self.assertEqual( bu_render.chunk_failure_reason( 0, 9, error, _captured( b'' ), 'libx264' ),
                          repr( error ) )

    def test_an_encoder_that_never_started_reports_why( self ):
        error = FileNotFoundError( 2, 'No such file or directory' )
        self.assertEqual( bu_render.chunk_failure_reason( None, 0, error, None, None ), repr( error ) )

    def test_success_is_no_failure( self ):
        self.assertIsNone( bu_render.chunk_failure_reason( 0, 40, None, _captured( b'' ), 'libx264' ) )


class TestCommandCodec( unittest.TestCase ):

    def test_reads_the_encoder_the_command_actually_asks_for( self ):
        command = bu_render._chunk_encode_command( 'o.mkv', 64, 48, 30.0, 'p5', 2,
                                                   codec='h264_nvenc', quality=21 )
        self.assertEqual( bu_render._command_codec( command ), 'h264_nvenc' )
        self.assertIsNone( bu_render._command_codec( [ 'ffmpeg', '-i', 'x', '-c', 'copy', 'y' ] ) )
        self.assertIsNone( bu_render._command_codec( [ 'ffmpeg', '-c:v' ] ) )


class TestLogFfmpegStderr( unittest.TestCase ):

    def _levels( self, returncode, data, debug_mode=0 ):
        logger = bu_log.get_logger()
        with mock.patch.object( betaconfig, 'debug_mode', debug_mode, create=True ):
            with self.assertLogs( logger, level='DEBUG' ) as captured:
                logger.debug( 'marker' )
                bu_render._log_ffmpeg_stderr( logger, 'render chunk 0 attempt 1', returncode,
                                              _captured( data ) )
        return [ ( record.levelname, record.getMessage() ) for record in captured.records[1:] ]

    def test_everything_reaches_the_file_at_debug( self ):
        records = self._levels( 1, b'first\nsecond\n' )
        self.assertEqual( records[0][0], 'DEBUG' )
        self.assertIn( 'first', records[0][1] )
        self.assertIn( 'second', records[0][1] )

    def test_a_failure_gets_no_second_warning( self ):
        self.assertEqual( [ level for level, _message in self._levels( 1, b'oops\n' ) ], [ 'DEBUG' ] )

    def test_errors_printed_on_a_successful_exit_are_a_warning( self ):
        records = self._levels( 0, b'[aac @ 0x1] decode error\n' )
        self.assertEqual( [ level for level, _message in records ], [ 'DEBUG', 'WARNING' ] )
        self.assertIn( 'decode error', records[1][1] )

    def test_debug_mode_chatter_on_success_stays_at_debug( self ):
        records = self._levels( 0, b'ffmpeg version 8\nframe=10\n', debug_mode=1 )
        self.assertEqual( [ level for level, _message in records ], [ 'DEBUG' ] )

    def test_silence_logs_nothing( self ):
        self.assertEqual( self._levels( 0, b'' ), [] )


@unittest.skipUnless( shutil.which( 'ffmpeg' ), "needs ffmpeg" )
class TestRealEncoderFailure( unittest.TestCase ):
    """A real ffmpeg that dies while frames are being piped to it."""

    @classmethod
    def setUpClass( cls ):
        import betautils_video as bu_video
        cls.tmpdir = tempfile.mkdtemp( prefix='betasuite-ffmpeg-fail-' )
        cls.source_path = os.path.join( cls.tmpdir, 'source.mp4' )
        subprocess.run(
            [ 'ffmpeg', '-y', '-loglevel', 'error',
              '-f', 'lavfi', '-i', 'testsrc2=size=64x48:rate=10:duration=4',
              '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p',
              cls.source_path ], check=True )
        cls.source = bu_video.open_video_source( cls.source_path, 'failhash' )

    @classmethod
    def tearDownClass( cls ):
        cls.source.release()
        shutil.rmtree( cls.tmpdir, ignore_errors=True )

    def test_the_encoders_own_reason_replaces_broken_pipe( self ):
        plan = bu_render.ChunkPlan( 0, 0, 40, os.path.join( self.tmpdir, 'v.part0000.mkv' ),
                                    is_last=True )
        logger = bu_log.get_logger()
        failing_options = lambda *args, **kwargs: [ '-c:v', 'libx264', '-profile:v', 'no_such_profile' ]
        with mock.patch.object( bu_render, 'encoder_options', failing_options ):
            with self.assertLogs( logger, level='DEBUG' ) as logged:
                with self.assertRaises( bu_render.RenderError ) as raised:
                    bu_render.render_one_chunk( plan, self.source, [], 'ultrafast', 1, 0, 0,
                                                None, logger )
        message = str( raised.exception )
        self.assertIn( "ffmpeg exited", message )
        self.assertIn( "[libx264 @", message )
        self.assertIn( "no_such_profile", message )
        self.assertNotIn( "BrokenPipe", message )
        # Precondition: the pipe really did break. Fewer frames reached
        # the encoder than the 40 the source holds, so the old code
        # would have reported only BrokenPipeError here.
        match = re.search( r"ffmpeg exited -?\d+ after (\d+) frame\(s\) written", message )
        self.assertIsNotNone( match, message )
        self.assertLess( int( match.group( 1 ) ), 40 )
        debug_text = '\n'.join( record.getMessage() for record in logged.records
                                if record.levelname == 'DEBUG' )
        self.assertIn( 'Error while opening encoder', debug_text )
        self.assertFalse( os.path.exists( plan.path ), "a failed chunk must never be promoted" )

    def test_a_failed_concat_says_why( self ):
        logger = bu_log.get_logger()
        plans = [ bu_render.ChunkPlan( index, index * 10, index * 10 + 10,
                                       os.path.join( self.tmpdir, 'gone.part%04d.mkv'%( index, ) ),
                                       is_last=( index == 1 ) )
                  for index in range( 2 ) ]
        with self.assertLogs( logger, level='DEBUG' ):
            with self.assertRaises( bu_render.RenderError ) as raised:
                bu_render.concat_chunks( plans, os.path.join( self.tmpdir, 'joined.mkv' ), 0, 0,
                                         logger )
        message = str( raised.exception )
        self.assertRegex( message, r"ffmpeg exited -?\d+: \S" )

    def test_a_failed_mux_says_why( self ):
        logger = bu_log.get_logger()
        missing = os.path.join( self.tmpdir, 'no-such-intermediate.mkv' )
        with self.assertLogs( logger, level='DEBUG' ):
            with self.assertRaises( bu_render.RenderError ) as raised:
                bu_render.mux_audio( missing, self.source_path,
                                     os.path.join( self.tmpdir, 'final.mp4' ), 0, 0, logger )
        self.assertIn( "ffmpeg exited", str( raised.exception ) )
        self.assertIn( "No such file or directory", str( raised.exception ) )


class TestFailureRecords( unittest.TestCase ):

    def setUp( self ):
        self.tmpdir = tempfile.mkdtemp( prefix='betasuite-failures-' )
        self.stats_path = os.path.join( self.tmpdir, 'stats', 'betasuite_stats.jsonl' )

    def tearDown( self ):
        shutil.rmtree( self.tmpdir, ignore_errors=True )

    def test_the_failures_file_sits_beside_the_stats_file( self ):
        self.assertEqual( bu_log.failures_path_for( '../output/stats/betasuite_stats.jsonl' ),
                          '../output/stats/betasuite_stats.failures.jsonl' )
        self.assertEqual( bu_log.failures_path_for( 'runs/stats' ), 'runs/stats.failures.jsonl' )

    def test_a_failure_goes_to_its_own_file_and_never_into_the_stats( self ):
        with mock.patch.object( betaconfig, 'stats_enabled', True ), \
             mock.patch.object( betaconfig, 'stats_path', self.stats_path ):
            bu_log.write_failure( { 'file': 'a.mkv', 'error': 'RenderError(...)' } )
        self.assertFalse( os.path.exists( self.stats_path ) )
        with open( bu_log.failures_path_for( self.stats_path ), encoding='UTF-8' ) as handle:
            records = [ json.loads( line ) for line in handle ]
        self.assertEqual( len( records ), 1 )
        self.assertEqual( records[0]['file'], 'a.mkv' )
        self.assertIn( 'timestamp', records[0] )

    def test_stats_switched_off_writes_no_failures_either( self ):
        with mock.patch.object( betaconfig, 'stats_enabled', False ), \
             mock.patch.object( betaconfig, 'stats_path', self.stats_path ):
            bu_log.write_failure( { 'file': 'a.mkv' } )
        self.assertFalse( os.path.exists( bu_log.failures_path_for( self.stats_path ) ) )


if __name__ == '__main__':
    unittest.main()
