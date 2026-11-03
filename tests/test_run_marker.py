"""
tests/test_run_marker.py - noticing a run that died without a word.

On 2026-10-26 the out-of-memory killer SIGKILLed betatv twice in a 4K
render; the only trace anywhere was the shell printing "Killed". A
marker that exists exactly while a run is running turns that into a
warning and a failure record at the next start. These tests drive the
marker with an injected clock, pid and process check.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )

import betaconfig
import betautils_log as bu_log


class _Clock:
    def __init__( self, now=1_700_000_000.0 ):
        self.now = now

    def __call__( self ):
        return self.now


class MarkerHarness( unittest.TestCase ):

    def setUp( self ):
        self.tmpdir = tempfile.mkdtemp( prefix='betasuite-marker-' )
        self.stats_path = os.path.join( self.tmpdir, 'stats', 'betasuite_stats.jsonl' )
        self.path = bu_log.run_marker_path_for( self.stats_path )
        self.logger = bu_log.get_logger()
        self.clock = _Clock()
        self._patches = [ mock.patch.object( betaconfig, 'stats_enabled', True ),
                          mock.patch.object( betaconfig, 'stats_path', self.stats_path ) ]
        for patch in self._patches:
            patch.start()

    def tearDown( self ):
        for patch in self._patches:
            patch.stop()
        shutil.rmtree( self.tmpdir, ignore_errors=True )

    def marker( self, pid=4242, boot_id='boot-now', running=False ):
        return bu_log.RunMarker( self.path, self.logger, clock=self.clock, pid=pid, boot_id=boot_id,
                                 is_running=lambda _pid: running )

    def leave_marker( self, **fields ):
        record = { 'pid': 999, 'boot_id': 'boot-now', 'started_at': 1_600_000_000.0,
                   'file': 'big4k.mp4', 'file_started_at': 1_600_000_500.0, 'preview_mode': False }
        record.update( fields )
        os.makedirs( os.path.dirname( self.path ), exist_ok=True )
        with open( self.path, 'w', encoding='UTF-8' ) as handle:
            json.dump( record, handle )

    def failures( self ):
        path = bu_log.failures_path_for( self.stats_path )
        if not os.path.exists( path ):
            return []
        with open( path, encoding='UTF-8' ) as handle:
            return [ json.loads( line ) for line in handle ]


class TestMarkerLifecycle( MarkerHarness ):

    def test_the_marker_is_the_running_paths_neighbour( self ):
        self.assertEqual( bu_log.run_marker_path_for( '../output/stats/betasuite_stats.jsonl' ),
                          '../output/stats/betasuite_stats.running.json' )

    def test_start_names_the_run_and_processing_names_the_file( self ):
        marker = self.marker()
        marker.start()
        self.clock.now += 30
        marker.now_processing( 'a.mkv', True )
        with open( self.path, encoding='UTF-8' ) as handle:
            record = json.load( handle )
        self.assertEqual( record, { 'pid': 4242, 'boot_id': 'boot-now',
                                    'started_at': 1_700_000_000.0, 'file': 'a.mkv',
                                    'file_started_at': 1_700_000_030.0, 'preview_mode': True } )
        self.assertFalse( os.path.exists( self.path + '.tmp' ), "written whole and renamed in" )

    def test_finish_removes_only_its_own_marker( self ):
        marker = self.marker()
        marker.start()
        marker.finish()
        self.assertFalse( os.path.exists( self.path ) )

        marker = self.marker( pid=4242 )
        marker.start()
        self.leave_marker( pid=777 )     # another run replaced it meanwhile
        marker.finish()
        self.assertTrue( os.path.exists( self.path ), "another run's marker must survive" )

    def test_finish_before_start_does_nothing( self ):
        self.leave_marker( pid=777 )
        self.marker().finish()
        self.assertTrue( os.path.exists( self.path ) )

    def test_processing_before_start_writes_nothing( self ):
        self.marker().now_processing( 'a.mkv', False )
        self.assertFalse( os.path.exists( self.path ) )

    def test_a_write_cut_short_leaves_the_last_good_marker( self ):
        marker = self.marker()
        marker.start()

        def dies_mid_write( record, handle ):
            handle.write( '{"pid": ' )
            raise OSError( 'killed mid-write' )

        with mock.patch.object( bu_log.json, 'dump', dies_mid_write ):
            with self.assertLogs( self.logger, level='WARNING' ):
                marker.now_processing( 'a.mkv', False )
        self.assertEqual( marker.read()['pid'], 4242, "the marker on disk is still whole" )


class TestReportPrevious( MarkerHarness ):

    def _report( self, **marker_args ):
        with self.assertLogs( self.logger, level='DEBUG' ) as logged:
            self.logger.debug( 'marker' )
            state = self.marker( **marker_args ).report_previous()
        return state, [ ( record.levelname, record.getMessage() ) for record in logged.records[1:] ]

    def test_no_marker_no_report( self ):
        state, records = self._report()
        self.assertEqual( ( state, records ), ( 'none', [] ) )
        self.assertEqual( self.failures(), [] )

    def test_a_dead_run_is_reported_and_its_file_recorded_as_failed( self ):
        self.leave_marker()
        state, records = self._report( running=False )
        self.assertEqual( state, 'ended' )
        self.assertEqual( records[0][0], 'WARNING' )
        self.assertIn( "ended without finishing while processing big4k.mp4", records[0][1] )
        self.assertIn( "journalctl -k", records[0][1] )
        failures = self.failures()
        self.assertEqual( len( failures ), 1 )
        self.assertEqual( failures[0]['file'], 'big4k.mp4' )
        self.assertEqual( failures[0]['timestamp'], 1_600_000_500.0,
            "dated when the file started, so summarize_run's windows place it right" )
        self.assertEqual( failures[0]['unfinished_run_pid'], 999 )
        self.assertFalse( failures[0]['preview_mode'] )
        self.assertFalse( os.path.exists( self.stats_path ), "never a stats row" )

    def test_a_marker_from_before_a_reboot_is_dead_whatever_the_pid_says( self ):
        self.leave_marker( boot_id='boot-before' )
        state, _records = self._report( running=True )
        self.assertEqual( state, 'ended' )
        self.assertEqual( len( self.failures() ), 1 )

    def test_a_run_still_going_is_warned_about_not_recorded( self ):
        self.leave_marker()
        state, records = self._report( running=True )
        self.assertEqual( state, 'running' )
        self.assertIn( "another betatv run (pid 999", records[0][1] )
        self.assertEqual( self.failures(), [] )

    def test_unknown_liveness_reports_with_a_caveat( self ):
        self.leave_marker()
        state, records = self._report( running=None )
        self.assertEqual( state, 'unknown' )
        self.assertIn( "cannot check whether it is still running", records[0][1] )
        self.assertEqual( len( self.failures() ), 1 )

    def test_a_run_that_died_between_files_records_no_file( self ):
        self.leave_marker( file=None, file_started_at=None )
        state, records = self._report( running=False )
        self.assertEqual( state, 'ended' )
        self.assertIn( "between files", records[0][1] )
        self.assertEqual( self.failures(), [] )

    def test_a_garbled_marker_is_ignored( self ):
        os.makedirs( os.path.dirname( self.path ), exist_ok=True )
        with open( self.path, 'w', encoding='UTF-8' ) as handle:
            handle.write( '{"pid": 9' )
        self.assertEqual( self.marker().report_previous(), 'none' )


@unittest.skipUnless( os.name == 'posix', "pid checks are POSIX-only" )
class TestBetatvProcessRunning( unittest.TestCase ):

    def setUp( self ):
        self.proc_root = tempfile.mkdtemp( prefix='betasuite-proc-' )

    def tearDown( self ):
        shutil.rmtree( self.proc_root, ignore_errors=True )

    def _cmdline( self, pid, text ):
        os.makedirs( os.path.join( self.proc_root, str( pid ) ) )
        with open( os.path.join( self.proc_root, str( pid ), 'cmdline' ), 'wb' ) as handle:
            handle.write( text )

    def test_a_live_betatv_is_running( self ):
        self._cmdline( os.getpid(), b'python\x00betatv.py\x00' )
        self.assertTrue( bu_log.betatv_process_running( os.getpid(), self.proc_root ) )

    def test_a_reused_pid_running_something_else_is_not( self ):
        self._cmdline( os.getpid(), b'/usr/bin/firefox\x00' )
        self.assertFalse( bu_log.betatv_process_running( os.getpid(), self.proc_root ) )

    def test_a_live_pid_with_no_readable_command_line_counts_as_running( self ):
        self.assertTrue( bu_log.betatv_process_running( os.getpid(), self.proc_root ) )

    def test_a_pid_owned_by_someone_else_still_counts( self ):
        self._cmdline( 4321, b'python\x00betatv.py\x00' )
        with mock.patch.object( bu_log.os, 'kill', side_effect=PermissionError( 1, 'not yours' ) ):
            self.assertTrue( bu_log.betatv_process_running( 4321, self.proc_root ) )

    def test_a_pid_that_is_gone_is_not_running( self ):
        child = subprocess.Popen( [ sys.executable, '-c', 'pass' ] )
        child.wait()
        self.assertFalse( bu_log.betatv_process_running( child.pid, self.proc_root ) )

    def test_nonsense_pids_are_unknown( self ):
        self.assertIsNone( bu_log.betatv_process_running( None ) )
        self.assertIsNone( bu_log.betatv_process_running( 0 ) )

    def test_never_signals_a_process_off_posix( self ):
        # On Windows os.kill( pid, 0 ) TERMINATES the process.
        with mock.patch.object( bu_log.os, 'name', 'nt' ), \
             mock.patch.object( bu_log.os, 'kill' ) as kill:
            self.assertIsNone( bu_log.betatv_process_running( os.getpid() ) )
        kill.assert_not_called()


class TestBootId( unittest.TestCase ):

    def test_reads_the_id_or_none( self ):
        with tempfile.NamedTemporaryFile( 'w', delete=False ) as handle:
            handle.write( 'abc-123\n' )
        try:
            self.assertEqual( bu_log.current_boot_id( handle.name ), 'abc-123' )
        finally:
            os.remove( handle.name )
        self.assertIsNone( bu_log.current_boot_id( '/no/such/boot_id' ) )


if __name__ == '__main__':
    unittest.main()
