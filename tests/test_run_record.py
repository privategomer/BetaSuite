"""
One file's RunRecord and the RunRecorder that hands it to every sink:
the stats and failure logs as before, and any other sink (the catalog's
history) without a failure there ever failing the run.
"""

import json
import os
import unittest
from unittest import mock

import betaconfig
import betatv
import betautils_hash as bu_hash
import betautils_job as bu_job
import betautils_run_record as bu_run_record

from tests.test_betatv_end_to_end import BetaTvHarness, _build_clip


class Keep:
    """A sink that keeps what it is given."""

    def __init__( self, required=False, fail=False ):
        self.required = required
        self.fail = fail
        self.records = []
        self.moves = []

    def record( self, run ):
        if self.fail:
            raise RuntimeError( 'sink broke' )
        self.records.append( run )

    def moved( self, run ):
        self.moves.append( ( run, run.moved_to ) )


class Logger:
    def __init__( self ):
        self.warnings = []

    def warning( self, message, *args, **kwargs ):
        self.warnings.append( message )


class TestRecorder( unittest.TestCase ):
    def test_every_sink_gets_the_record_in_order( self ):
        order = []
        first, second = Keep(), Keep()
        first.record = lambda run: order.append( 'first' )
        second.record = lambda run: order.append( 'second' )
        recorder = bu_run_record.RunRecorder( [ first, second ] )
        recorder.record( recorder.start( '/x.mp4' ).end( 'processed' ) )
        self.assertEqual( order, [ 'first', 'second' ] )

    def test_a_best_effort_sink_that_fails_is_logged_and_the_rest_still_get_it( self ):
        logger = Logger()
        broken, after = Keep( fail=True ), Keep()
        recorder = bu_run_record.RunRecorder( [ broken, after ], logger=logger )
        recorder.record( recorder.start( '/x.mp4' ).end( 'processed' ) )
        self.assertEqual( len( after.records ), 1 )
        self.assertEqual( len( logger.warnings ), 1 )

    def test_a_required_sink_that_fails_fails_the_caller( self ):
        recorder = bu_run_record.RunRecorder( [ Keep( required=True, fail=True ), Keep() ] )
        with self.assertRaises( RuntimeError ):
            recorder.record( recorder.start( '/x.mp4' ).end( 'processed' ) )

    def test_a_move_reaches_the_sinks_that_take_moves( self ):
        keep = Keep()
        plain = mock.Mock( spec=[ 'record', 'required' ] )
        plain.required = False
        recorder = bu_run_record.RunRecorder( [ plain, keep ] )
        run = recorder.start( '/a/x.mp4' ).end( 'processed' )
        recorder.moved( run, '/b/x.mp4' )
        self.assertEqual( keep.moves, [ ( run, '/b/x.mp4' ) ] )

    def test_the_log_sinks_are_required( self ):
        self.assertTrue( bu_run_record.StatsLogSink.required )
        self.assertTrue( bu_run_record.FailureLogSink.required )

    def test_the_record_carries_the_program_and_job( self ):
        recorder = bu_run_record.RunRecorder( [], program='engine' )
        run = recorder.start( '/x.mp4', job='j1' )
        self.assertEqual( ( run.program, run.job, run.outcome ), ( 'engine', 'j1', None ) )


class TestThroughThePipeline( BetaTvHarness ):
    """process_one_video fills one record per file and hands it on."""

    def setUp( self ):
        super().setUp()
        self.keep = Keep()
        self.recorder = bu_run_record.RunRecorder(
            [ bu_run_record.StatsLogSink(), bu_run_record.FailureLogSink(), self.keep ],
            program='betatv', logger=self.logger )
        self.reporter = bu_job.FileReporter( None, 'job-1', 0, self.recorder )

    def run_file( self, fname='clip.mp4', folder=None ):
        return betatv.process_one_video(
            folder or self.uncensored, fname, self.censored, 0, 1, self.detector.get_session(),
            getattr( betaconfig, 'preview_mode_enabled', False ), 20, 'ultrafast', self.logger,
            reporter=self.reporter )

    def stats_rows( self ):
        path = betaconfig.stats_path
        if not os.path.exists( path ):
            return []
        with open( path ) as handle:
            return [ json.loads( line ) for line in handle if line.strip() ]

    def test_a_processed_file( self ):
        clip = os.path.join( self.uncensored, 'clip.mp4' )
        _build_clip( clip )
        self.assertEqual( self.run_file(), 'processed' )
        [ run ] = self.keep.records
        self.assertEqual( ( run.outcome, run.reason, run.program, run.job, run.path ),
                          ( 'processed', None, 'betatv', 'job-1', clip ) )
        self.assertEqual( run.hash, bu_hash.md5_for_file( clip, 16, use_memo=False ) )
        self.assertEqual( run.output, os.path.join( self.censored, self.outputs()[0] ) )
        [ row ] = self.stats_rows()
        self.assertEqual( run.keys, { 'detection': row['detection_key'], 'censor': row['censor_key'],
                                      'encode': row['encode_key'] } )
        self.assertEqual( run.profile, row['track_structure_profile'] )
        self.assertIsNotNone( run.rule )
        self.assertEqual( run.stats['output'], row['output'] )
        self.assertLessEqual( run.started_at, run.ended_at )

    def test_a_file_whose_output_is_there_is_skipped_as_output_present( self ):
        _build_clip( os.path.join( self.uncensored, 'clip.mp4' ) )
        self.run_file()
        self.keep.records.clear()
        self.assertEqual( self.run_file(), 'skipped' )
        [ run ] = self.keep.records
        self.assertEqual( ( run.outcome, run.reason ), ( 'skipped', 'output.present' ) )
        self.assertEqual( run.output, os.path.join( self.censored, self.outputs()[0] ) )
        self.assertEqual( len( self.stats_rows() ), 1, 'a skip writes no stats row' )

    def test_a_file_that_is_not_a_video( self ):
        with open( os.path.join( self.uncensored, 'notes.mp4' ), 'wb' ) as handle:
            handle.write( b'not a video at all' * 50 )
        self.assertEqual( self.run_file( 'notes.mp4' ), 'skipped' )
        [ run ] = self.keep.records
        self.assertEqual( ( run.outcome, run.reason ), ( 'skipped', 'source.not_video' ) )
        self.assertIsNotNone( run.hash )

    def test_a_file_that_cannot_be_read_has_no_hash( self ):
        self.assertEqual( self.run_file( 'gone.mp4' ), 'skipped' )
        [ run ] = self.keep.records
        self.assertEqual( ( run.outcome, run.reason, run.hash ), ( 'skipped', 'source.unreadable', None ) )

    def test_a_failed_file_carries_its_error_and_writes_the_failure_row( self ):
        _build_clip( os.path.join( self.uncensored, 'clip.mp4' ) )
        with mock.patch.object( betatv, 'detect_boxes_for_video', side_effect=RuntimeError( 'bad' ) ):
            self.assertEqual( self.run_file(), 'failed' )
        [ run ] = self.keep.records
        self.assertEqual( ( run.outcome, run.reason, run.error_type ), ( 'failed', 'file.failed', 'RuntimeError' ) )
        self.assertEqual( run.failure['error'], "RuntimeError('bad')" )
        self.assertIsNone( run.stats )

    def test_a_full_disk_while_writing_the_stats_row_is_still_a_disk_full_stop( self ):
        _build_clip( os.path.join( self.uncensored, 'clip.mp4' ) )
        written = []
        with mock.patch( 'betautils_log.write_stats', side_effect=OSError( 28, 'No space left on device' ) ), \
                mock.patch( 'betautils_log.write_failure', side_effect=written.append ):
            self.assertEqual( self.run_file(), 'out_of_space' )
        [ run ] = self.keep.records
        self.assertEqual( ( run.outcome, run.reason ), ( 'out_of_space', 'disk.full' ) )
        self.assertEqual( len( written ), 1, 'the failure row, and no second try at the stats row' )

    def test_a_broken_history_sink_never_fails_the_file( self ):
        _build_clip( os.path.join( self.uncensored, 'clip.mp4' ) )
        self.recorder.sinks.append( Keep( fail=True ) )
        self.assertEqual( self.run_file(), 'processed' )
        self.assertEqual( len( self.stats_rows() ), 1 )

    def test_a_moved_source_is_reported_after_the_record( self ):
        betaconfig.move_completed_sources = True
        folder = os.path.join( self.uncensored, 'scene' )
        os.makedirs( folder )
        _build_clip( os.path.join( folder, 'clip.mp4' ) )
        self.assertEqual( self.run_file( folder=folder ), 'processed' )
        [ ( run, moved_to ) ] = self.keep.moves
        self.assertEqual( moved_to, os.path.join( self.completed, 'scene', 'clip.mp4' ) )
        self.assertIs( run, self.keep.records[0] )

    def test_without_a_recorder_the_logs_are_written_as_before( self ):
        _build_clip( os.path.join( self.uncensored, 'clip.mp4' ) )
        self.assertEqual( self.run_one(), 'processed' )
        self.assertEqual( len( self.stats_rows() ), 1 )


if __name__ == '__main__':
    unittest.main()
