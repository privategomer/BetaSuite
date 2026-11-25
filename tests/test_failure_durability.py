"""
test_failure_durability.py - a full disk never tears or loses a failure
record, and stops the job with disk.full.

The disk is filled by replacing betautils_log._write, the one write the
stats and failures appends make, with one that writes part of a line and
then fails with ENOSPC, the way a filling disk does.
"""

import errno
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import betaconfig
import betautils_job as bu_job
import betautils_log as bu_log
import betatv

from tests.test_betatv_end_to_end import BetaTvHarness, _build_clip


class FillingDisk:
    """Writes `room` bytes in all, then fails each write with ENOSPC until `freed`."""

    def __init__( self, room ):
        self.room = room
        self.freed = False
        self.real = os.write

    def free( self ):
        self.freed = True

    def __call__( self, fd, data ):
        if self.freed:
            return self.real( fd, data )
        if self.room <= 0:
            raise OSError( errno.ENOSPC, os.strerror( errno.ENOSPC ) )
        written = self.real( fd, bytes( data[:self.room] ) )
        self.room -= written
        return written


class _Folder( unittest.TestCase ):

    def setUp( self ):
        self.tmp = tempfile.mkdtemp( prefix='betasuite-durable-' )
        self.addCleanup( shutil.rmtree, self.tmp, True )
        self.path = os.path.join( self.tmp, 'stats.failures.jsonl' )
        patcher = mock.patch.object( bu_log, 'configured_stats_path',
                                     return_value=os.path.join( self.tmp, 'stats.jsonl' ) )
        patcher.start()
        self.addCleanup( patcher.stop )

    def lines( self ):
        with open( self.path, 'rb' ) as handle:
            return handle.read().split( b'\n' )


class TestAppend( _Folder ):

    def test_a_full_disk_mid_line_leaves_no_torn_line_and_the_reserve_saves_the_record( self ):
        bu_log._append_record( self.path, { 'file': 'first' }, 'failure record' )
        self.assertTrue( bu_log.ensure_reserve( size=4096 ) )
        disk = FillingDisk( room=10 )
        real_release = bu_log.release_reserve

        def release():
            disk.free()
            return real_release()

        with mock.patch.object( bu_log, '_write', disk ), \
                mock.patch.object( bu_log, 'release_reserve', release ):
            bu_log._append_record( self.path, { 'file': 'second', 'error': 'x' * 200 }, 'failure record' )
        lines = self.lines()
        self.assertEqual( lines[-1], b'', "the file ends on a whole line" )
        self.assertEqual( [ json.loads( line )['file'] for line in lines[:-1] ], [ 'first', 'second' ] )
        self.assertFalse( os.path.exists( bu_log.reserve_path() ), "the reserve was given up" )

    def test_without_a_reserve_the_file_is_left_as_it_was( self ):
        bu_log._append_record( self.path, { 'file': 'first' }, 'failure record' )
        before = open( self.path, 'rb' ).read()
        bu_log.release_reserve()
        with mock.patch.object( bu_log, '_write', FillingDisk( room=7 ) ), \
                mock.patch.object( bu_log, 'get_logger' ) as logger:
            bu_log._append_record( self.path, { 'file': 'second' }, 'failure record' )
        self.assertEqual( open( self.path, 'rb' ).read(), before )
        self.assertIn( 'failed to write failure record', logger.return_value.warning.call_args[0][0] )

    def test_another_error_is_not_retried( self ):
        bu_log.ensure_reserve( size=4096 )

        def denied( fd, data ):
            raise OSError( errno.EACCES, 'denied' )

        with mock.patch.object( bu_log, '_write', denied ), mock.patch.object( bu_log, 'get_logger' ):
            bu_log._append_record( self.path, { 'file': 'x' }, 'failure record' )
        self.assertTrue( os.path.exists( bu_log.reserve_path() ), "the reserve is only for a full disk" )

    def test_the_reserve_is_made_once_and_given_up( self ):
        self.assertTrue( bu_log.ensure_reserve( size=8192 ) )
        self.assertEqual( os.path.getsize( bu_log.reserve_path() ), 8192 )
        self.assertTrue( bu_log.ensure_reserve( size=8192 ) )
        self.assertTrue( bu_log.release_reserve() )
        self.assertFalse( bu_log.release_reserve() )


class FullThenFreed:
    """A log stream whose first flush fails with ENOSPC; it keeps what was written, as a buffer does."""

    def __init__( self, path ):
        self.path = path
        self.pending = ''
        self.failed = False

    def write( self, text ):
        self.pending += text

    def flush( self ):
        if not self.failed:
            self.failed = True
            raise OSError( errno.ENOSPC, os.strerror( errno.ENOSPC ) )
        with open( self.path, 'a', encoding='utf-8' ) as handle:
            handle.write( self.pending )
        self.pending = ''

    def close( self ):
        pass


class TestTheLog( _Folder ):

    def test_a_full_disk_gives_up_the_reserve_and_the_line_is_written( self ):
        import logging
        path = os.path.join( self.tmp, 'betasuite.log' )
        handler = bu_log.LogFileHandler( path, encoding='UTF-8', delay=True )
        handler.setFormatter( logging.Formatter( '%(message)s' ) )
        handler.stream = FullThenFreed( path )
        self.addCleanup( handler.close )
        self.assertTrue( bu_log.ensure_reserve( size=4096 ) )
        logger = logging.getLogger( 'test_failure_durability.log' )
        logger.propagate = False
        logger.addHandler( handler )
        self.addCleanup( logger.removeHandler, handler )
        with mock.patch( 'logging.Handler.handleError' ) as reported:
            logger.error( "the disk filled here" )
        reported.assert_not_called()
        self.assertEqual( open( path, encoding='utf-8' ).read(), "the disk filled here\n" )
        self.assertFalse( os.path.exists( bu_log.reserve_path() ) )

    def test_without_a_reserve_the_error_is_reported_as_before( self ):
        import logging
        path = os.path.join( self.tmp, 'betasuite.log' )
        handler = bu_log.LogFileHandler( path, encoding='UTF-8', delay=True )
        handler.stream = FullThenFreed( path )
        self.addCleanup( handler.close )
        bu_log.release_reserve()
        record = logging.LogRecord( 'x', logging.ERROR, __file__, 1, "line", None, None )
        with mock.patch( 'logging.Handler.handleError' ) as reported:
            handler.emit( record )
        reported.assert_called_once()

    def test_the_loggers_use_it( self ):
        logger = bu_log.tool_logger( 'test_failure_durability.tool', log_path=os.path.join( self.tmp, 't.log' ),
                                     console_level=None )
        self.addCleanup( lambda: [ handler.close() for handler in logger.handlers ] )
        self.assertTrue( any( isinstance( handler, bu_log.LogFileHandler ) for handler in logger.handlers ) )


class TestTheJobStops( BetaTvHarness ):

    def test_a_full_disk_stops_the_queue_with_its_code_and_the_record_is_whole( self ):
        for name in ( 'a.mp4', 'b.mp4' ):
            _build_clip( os.path.join( self.uncensored, name ) )
        events = []
        runner = bu_job.Runner( self.logger, emit=lambda name, data: events.append( ( name, data ) ) )
        job = bu_job.Job( 'j', [ bu_job.JobFile( os.path.join( self.uncensored, name ), self.censored )
                                 for name in ( 'a.mp4', 'b.mp4' ) ] )
        disk = FillingDisk( room=25 )
        real_release = bu_log.release_reserve

        def release():
            disk.free()
            return real_release()

        def full( *args, **kwargs ):
            raise OSError( errno.ENOSPC, 'No space left on device' )

        with mock.patch.object( betatv, 'detect_boxes_for_video', full ), \
                mock.patch.object( bu_log, '_write', disk ), \
                mock.patch.object( bu_log, 'release_reserve', release ):
            self.assertEqual( runner.run( job ), bu_job.STOPPED )
        self.assertEqual( job.code, 'disk.full' )
        errors = [ data for name, data in events if name == 'job.error' ]
        self.assertEqual( [ error['code'] for error in errors ], [ 'disk.full' ] )
        with open( bu_log.failures_path_for( betaconfig.stats_path ), 'rb' ) as handle:
            raw = handle.read()
        self.assertTrue( raw.endswith( b'\n' ) )
        records = [ json.loads( line ) for line in raw.splitlines() ]
        self.assertEqual( [ record['file'] for record in records ], [ 'a.mp4' ] )
        self.assertIn( 'No space left on device', records[0]['error'] )


if __name__ == '__main__':
    unittest.main()
