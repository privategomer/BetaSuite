"""
The catalog never stops a run, whatever goes wrong opening it; the
rebuild keeps the whole old file; a background hash lets go of the disk
once a run starts; one server opens one catalog however many threads
ask at once. (From the M5 independent review.)
"""

import io
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from unittest import mock

import betaconst
import betautils_catalog as bu_catalog
import betautils_catalog_history as bu_history
import betautils_catalog_index as bu_index
import betautils_catalog_rebuild as bu_rebuild
import betautils_catalog_scan as bu_scan
import betautils_catalog_store as bu_store
import betautils_hash as bu_hash
import betautils_serve as bu_serve
import betautils_serve_catalog as bu_serve_catalog

REPO = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )


class _Tmp( unittest.TestCase ):
    def setUp( self ):
        self.root = tempfile.mkdtemp( prefix='catalog-robust-' )
        self.addCleanup( shutil.rmtree, self.root, ignore_errors=True )


class Logger:
    handlers = ()

    def __init__( self ):
        self.warnings = []

    def warning( self, message, *args, **kwargs ):
        self.warnings.append( message%args if args else message )

    def info( self, *args, **kwargs ):
        pass

    debug = trace = error = info

    def exception( self, *args, **kwargs ):
        pass


@unittest.skipIf( os.name == 'nt' or ( hasattr( os, 'geteuid' ) and os.geteuid() == 0 ),
                  "needs a folder this user cannot write" )
class TestAnUnwritableCatalog( _Tmp ):
    def setUp( self ):
        super().setUp()
        self.locked = os.path.join( self.root, 'locked' )
        os.makedirs( self.locked )
        os.chmod( self.locked, 0o555 )
        self.addCleanup( os.chmod, self.locked, 0o755 )
        self.path = os.path.join( self.locked, 'data', 'catalog.sqlite' )

    def test_the_store_says_unavailable_not_an_os_error( self ):
        with self.assertRaises( bu_store.CatalogError ) as caught:
            bu_store.Store( self.path )
        self.assertEqual( ( caught.exception.code, caught.exception.params ),
                          ( 'catalog.unavailable', { 'reason': 'not_writable' } ) )

    def test_the_command_line_goes_on_without_it( self ):
        logger = Logger()
        self.assertIsNone( bu_history.CliHistory.open( logger, path=self.path ) )
        self.assertEqual( len( logger.warnings ), 1 )

    def test_an_engine_job_goes_on_without_it( self ):
        server = bu_serve.Server( io.BytesIO(), io.BytesIO(), Logger() )
        server.catalog_factory = lambda **kwargs: bu_catalog.Catalog( path=self.path, seed_folders=[], **kwargs )
        types = bu_serve_catalog.job_types_for( server )
        under = os.path.join( betaconst.video_path_uncensored, 'scene', 'x.mp4' )
        self.assertEqual( types.declared( under, 'a' * 16 )[0], 'scene', 'the rules still decide' )
        sink = bu_serve_catalog.job_recorder_for( server ).sinks[-1]
        import betautils_run_record as bu_run_record
        run = bu_run_record.RunRecord( under )
        run.hash = 'a' * 16
        sink.record( run.end( 'processed' ) )   # nothing raised
        self.assertIsNone( server.catalog )


class TestTheCommandLineSurvivesAnyOpeningError( _Tmp ):
    def test_an_unexpected_error_opening_is_a_warning( self ):
        logger = Logger()
        with mock.patch.object( bu_store, 'Store', side_effect=RuntimeError( 'unexpected' ) ):
            self.assertIsNone( bu_history.CliHistory.open( logger, path=os.path.join( self.root, 'c.sqlite' ) ) )
        self.assertEqual( len( logger.warnings ), 1 )


class TestAFactoryThatRaisesAnythingNeverFailsAJob( _Tmp ):
    def test_any_error_opening_is_logged_once_and_the_rules_decide( self ):
        logger = Logger()
        server = bu_serve.Server( io.BytesIO(), io.BytesIO(), logger )

        def broken( **kwargs ):
            raise PermissionError( 13, 'denied' )

        server.catalog_factory = broken
        types = bu_serve_catalog.job_types_for( server )
        for _ in range( 3 ):
            self.assertIsNone( types.declared( '/elsewhere/x.mp4', 'b' * 16 ) )
        self.assertEqual( len( logger.warnings ), 1 )


class TestOneCatalogPerServer( _Tmp ):
    def test_two_threads_asking_at_once_open_it_once( self ):
        server = bu_serve.Server( io.BytesIO(), io.BytesIO(), Logger() )
        made = []
        gate = threading.Event()

        def slow( **kwargs ):
            made.append( 1 )
            gate.wait( 5 )
            return bu_catalog.Catalog( path=os.path.join( self.root, 'c.sqlite' ), seed_folders=[],
                                       **dict( kwargs, interval=0 ) )

        server.catalog_factory = slow
        self.addCleanup( server.stop_catalog )
        results = []
        threads = [ threading.Thread( target=lambda: results.append( bu_serve_catalog.catalog_for( server ) ) )
                    for _ in range( 2 ) ]
        for thread in threads:
            thread.start()
        time.sleep( 0.2 )
        gate.set()
        for thread in threads:
            thread.join( 10 )
        self.assertEqual( len( made ), 1 )
        self.assertIs( results[0], results[1] )


_WRITE_AND_DIE = textwrap.dedent( """
    import os, sys
    sys.path.insert( 0, sys.argv[1] )
    import betautils_catalog_index as bu_index
    import betautils_catalog_store as bu_store
    store = bu_store.Store( sys.argv[2] )
    store._db.execute( 'PRAGMA wal_autocheckpoint = 0' )
    index = bu_index.Index( store )
    path = sys.argv[3]
    index.record_hash( path, bu_index.Stat.of( os.stat( path ) ), 'd' * 16 )
    index.set_pin( 'd' * 16, 'compilation' )
    os._exit( 0 )   # killed: no close, no checkpoint
""" )


class TestTheRebuildKeepsTheWholeOldFile( _Tmp ):
    def test_what_was_still_in_the_log_is_in_the_kept_copy( self ):
        db_path = os.path.join( self.root, 'data', 'catalog.sqlite' )
        video = os.path.join( self.root, 'v.mp4' )
        with open( video, 'wb' ) as handle:
            handle.write( os.urandom( 1000 ) )
        subprocess.run( [ sys.executable, '-c', _WRITE_AND_DIE, REPO, db_path, video ], check=True,
                        stdin=subprocess.DEVNULL )
        self.assertTrue( os.path.exists( db_path + '-wal' ) and os.path.getsize( db_path + '-wal' ) > 0,
                         'the pin is only in the write-ahead log' )
        summary = bu_rebuild.rebuild( db_path, seed_folders=[], hasher=lambda path: 'x' * 16 )
        self.assertEqual( summary['carried_pins'], 1 )
        kept = sqlite3.connect( db_path + bu_rebuild.OLD_SUFFIX )
        self.addCleanup( kept.close )
        self.assertEqual( kept.execute( 'SELECT hash, type_pin FROM items' ).fetchall(), [ ( 'd' * 16, 'compilation' ) ] )


class TestABackgroundHashLetsGo( _Tmp ):
    def test_md5_stops_when_asked_and_memoises_nothing( self ):
        path = os.path.join( self.root, 'big.mp4' )
        with open( path, 'wb' ) as handle:
            handle.write( os.urandom( 9 * 1024 * 1024 ) )
        saved = ( betaconst.file_hash_cache_path, bu_hash._file_hash_memo, bu_hash._file_hash_memo_dirty )

        def restore():
            betaconst.file_hash_cache_path, bu_hash._file_hash_memo, bu_hash._file_hash_memo_dirty = saved

        self.addCleanup( restore )
        betaconst.file_hash_cache_path = os.path.join( self.root, 'memo.json' )
        bu_hash._file_hash_memo, bu_hash._file_hash_memo_dirty = None, False
        asked = []

        def stop():
            asked.append( 1 )
            return len( asked ) > 1   # stop after the first chunk

        with self.assertRaises( bu_hash.HashStopped ):
            bu_hash.md5_for_file( path, 16, should_stop=stop )
        self.assertFalse( any( bu_hash._file_hash_memo.values() ) if bu_hash._file_hash_memo else False )
        self.assertEqual( bu_hash.md5_for_file( path, 16, use_memo=False ),
                          bu_hash.md5_for_file( path, 16, should_stop=lambda: False ) )

    def test_a_run_starting_mid_hash_leaves_the_file_pending( self ):
        watched = os.path.join( self.root, 'videos' )
        os.makedirs( watched )
        path = os.path.join( watched, 'v.mp4' )
        with open( path, 'wb' ) as handle:
            handle.write( os.urandom( 9 * 1024 * 1024 ) )
        state = { 'checks': 0, 'busy_after': None }

        def busy():
            state['checks'] += 1
            return state['busy_after'] is not None and state['checks'] > state['busy_after']

        catalog = bu_catalog.Catalog( path=os.path.join( self.root, 'c.sqlite' ), seed_folders=[], settle=0.0,
                                      busy=busy, thumbnails=False )
        self.addCleanup( catalog.store.close )
        catalog.add_folder( watched )
        catalog.scanner.pass_now()
        saved = betaconst.file_hash_cache_path
        self.addCleanup( setattr, betaconst, 'file_hash_cache_path', saved )
        betaconst.file_hash_cache_path = os.path.join( self.root, 'memo.json' )
        with mock.patch.object( bu_hash, '_file_hash_memo', {} ):
            state['busy_after'] = state['checks'] + 1   # idle at the start, busy once reading
            catalog.scanner.hash_ready()
            self.assertEqual( catalog.list()['items'], [] )
            self.assertEqual( catalog.list()['pending'], 1 )
            state['busy_after'] = None
            catalog.scanner.hash_ready()
        self.assertEqual( len( catalog.list()['items'] ), 1 )


if __name__ == '__main__':
    unittest.main()
