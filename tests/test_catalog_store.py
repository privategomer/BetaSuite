"""
The catalog's database file: schema steps, backups, refusal of a newer
file, the shared and exclusive lock.
"""

import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import unittest

import betautils_catalog_store as bu_store

# The checksums of the shipped steps. A shipped step is never edited:
# a change to the schema is a new step. Update this only by appending.
SHIPPED_CHECKSUMS = {
    1: '2d41624e4d6d14ef75d5c8f4f1aff89e5c2533e676768b08cabd6c95b5005de2',
}

V2_FOR_TESTS = ( 2, 'test_step', "CREATE TABLE extra ( id INTEGER PRIMARY KEY, note TEXT );\n"
                                 "ALTER TABLE items ADD COLUMN extra_note TEXT;\n" )


class _Tmp( unittest.TestCase ):
    def setUp( self ):
        self.dir = tempfile.mkdtemp( prefix='catalog-store-' )
        self.addCleanup( shutil.rmtree, self.dir, ignore_errors=True )
        self.path = os.path.join( self.dir, 'catalog.sqlite' )
        self.now = 1000.0

    def clock( self ):
        return self.now

    def open( self, **kwargs ):
        store = bu_store.Store( self.path, clock=self.clock, **kwargs )
        self.addCleanup( store.close )
        return store


class TestFromEmpty( _Tmp ):
    def test_a_new_file_is_at_the_latest_version_with_every_step_recorded( self ):
        store = self.open()
        self.assertEqual( store.read( 'PRAGMA user_version' )[0][0], bu_store.LATEST_VERSION )
        rows = store.read( 'SELECT version, name, applied_at, checksum FROM schema_migrations ORDER BY version' )
        self.assertEqual( [ row['version'] for row in rows ],
                          [ step[0] for step in bu_store.MIGRATIONS ] )
        for row, ( _version, name, sql ) in zip( rows, bu_store.MIGRATIONS ):
            self.assertEqual( row['name'], name )
            self.assertEqual( row['applied_at'], 1000.0 )
            self.assertEqual( row['checksum'], bu_store.step_checksum( sql ) )

    def test_the_tables_exist( self ):
        store = self.open()
        names = { row['name'] for row in store.read( "SELECT name FROM sqlite_master WHERE type='table'" ) }
        self.assertTrue( { 'folders', 'items', 'locations', 'outputs', 'schema_migrations' } <= names )

    def test_the_file_uses_wal( self ):
        store = self.open()
        self.assertEqual( store.read( 'PRAGMA journal_mode' )[0][0].lower(), 'wal' )

    def test_a_new_file_takes_no_backup( self ):
        self.open()
        self.assertEqual( [ name for name in os.listdir( self.dir ) if name.endswith( '.bak' ) ], [] )

    def test_opening_again_applies_nothing( self ):
        self.open().close()
        db = sqlite3.connect( self.path )
        self.addCleanup( db.close )
        self.assertEqual( bu_store.migrate( db, self.path ), [] )

    def test_the_default_path_is_in_the_data_folder( self ):
        import betautils_paths as bu_paths
        self.assertEqual( bu_store.default_path(),
                          os.path.join( bu_paths.kind_dir( 'data' ), 'catalog.sqlite' ) )


class TestShippedSteps( unittest.TestCase ):
    def test_versions_count_up_from_one( self ):
        self.assertEqual( [ step[0] for step in bu_store.MIGRATIONS ],
                          list( range( 1, len( bu_store.MIGRATIONS ) + 1 ) ) )
        self.assertEqual( bu_store.LATEST_VERSION, len( bu_store.MIGRATIONS ) )

    def test_a_shipped_step_is_never_edited( self ):
        for version, _name, sql in bu_store.MIGRATIONS:
            if version in SHIPPED_CHECKSUMS:
                self.assertEqual( bu_store.step_checksum( sql ), SHIPPED_CHECKSUMS[version],
                                  'step %d changed; add a new step instead'%( version ) )

    def test_every_step_but_the_newest_is_pinned( self ):
        # The newest step may still change on its milestone branch until
        # it ships; everything before it is pinned above.
        for version, _name, _sql in bu_store.MIGRATIONS[:-1]:
            self.assertIn( version, SHIPPED_CHECKSUMS )


class TestFromPreviousVersion( _Tmp ):
    def migrations( self ):
        return bu_store.MIGRATIONS[:1] + ( V2_FOR_TESTS, )

    def make_v1( self ):
        db = sqlite3.connect( self.path, isolation_level=None )
        bu_store.migrate( db, self.path, migrations=bu_store.MIGRATIONS[:1], clock=self.clock )
        db.execute( "INSERT INTO items ( hash, size, name, added_at ) VALUES ( 'a'||'0123456789abcde', 5, 'x.mp4', 1 )" )
        db.close()

    def test_a_previous_version_moves_forward_and_keeps_its_rows( self ):
        self.make_v1()
        db = sqlite3.connect( self.path, isolation_level=None )
        self.addCleanup( db.close )
        self.now = 2000.0
        self.assertEqual( bu_store.migrate( db, self.path, migrations=self.migrations(), clock=self.clock ), [ 2 ] )
        self.assertEqual( bu_store.schema_version( db ), 2 )
        self.assertEqual( db.execute( 'SELECT name, extra_note FROM items' ).fetchall(), [ ( 'x.mp4', None ) ] )
        self.assertEqual( db.execute( 'SELECT version, applied_at FROM schema_migrations ORDER BY version' ).fetchall(),
                          [ ( 1, 1000.0 ), ( 2, 2000.0 ) ] )

    def test_a_copy_of_the_previous_version_is_kept( self ):
        self.make_v1()
        db = sqlite3.connect( self.path, isolation_level=None )
        self.addCleanup( db.close )
        bu_store.migrate( db, self.path, migrations=self.migrations(), clock=self.clock )
        backup = self.path + '.v1.bak'
        self.assertTrue( os.path.exists( backup ) )
        copy = sqlite3.connect( backup )
        self.addCleanup( copy.close )
        self.assertEqual( bu_store.schema_version( copy ), 1 )
        self.assertEqual( copy.execute( 'SELECT name FROM items' ).fetchall(), [ ( 'x.mp4', ) ] )

    def test_only_the_newest_backups_are_kept( self ):
        for version in range( 1, 6 ):
            path = '%s.v%d.bak'%( self.path, version )
            open( path, 'wb' ).close()
            os.utime( path, ( version, version ) )
        self.make_v1()
        db = sqlite3.connect( self.path, isolation_level=None )
        self.addCleanup( db.close )
        bu_store.migrate( db, self.path, migrations=self.migrations(), clock=self.clock )
        kept = sorted( name for name in os.listdir( self.dir ) if name.endswith( '.bak' ) )
        self.assertEqual( len( kept ), bu_store.BACKUPS_KEPT )
        self.assertIn( 'catalog.sqlite.v1.bak', kept )

    def test_a_failing_step_changes_nothing( self ):
        self.make_v1()
        broken = ( 2, 'broken', "CREATE TABLE extra ( id INTEGER );\nALTER TABLE no_such_table ADD COLUMN x;\n" )
        db = sqlite3.connect( self.path, isolation_level=None )
        self.addCleanup( db.close )
        with self.assertRaises( sqlite3.OperationalError ):
            bu_store.migrate( db, self.path, migrations=bu_store.MIGRATIONS[:1] + ( broken, ), clock=self.clock )
        self.assertEqual( bu_store.schema_version( db ), 1 )
        tables = { row[0] for row in db.execute( "SELECT name FROM sqlite_master WHERE type='table'" ) }
        self.assertNotIn( 'extra', tables )
        self.assertEqual( db.execute( 'SELECT COUNT(*) FROM schema_migrations' ).fetchone()[0], 1 )

    def test_an_edited_step_is_logged_not_refused( self ):
        self.make_v1()
        db = sqlite3.connect( self.path, isolation_level=None )
        self.addCleanup( db.close )
        db.execute( "UPDATE schema_migrations SET checksum = 'x' WHERE version = 1" )
        logger = _Logger()
        bu_store.migrate( db, self.path, clock=self.clock, logger=logger )
        self.assertTrue( any( 'step 1 differs' in line for line in logger.warnings ) )


class TestNewerFile( _Tmp ):
    def test_a_file_newer_than_the_code_is_refused_and_left_alone( self ):
        db = sqlite3.connect( self.path, isolation_level=None )
        db.execute( 'PRAGMA user_version = %d'%( bu_store.LATEST_VERSION + 1 ) )
        db.close()
        before = _read( self.path )
        with self.assertRaises( bu_store.CatalogError ) as caught:
            bu_store.Store( self.path, clock=self.clock )
        self.assertEqual( caught.exception.code, 'catalog.too_new' )
        self.assertEqual( caught.exception.params,
                          { 'version': bu_store.LATEST_VERSION + 1, 'supported': bu_store.LATEST_VERSION } )
        self.assertEqual( _read( self.path ), before )

    def test_a_refused_open_gives_the_lock_back( self ):
        db = sqlite3.connect( self.path, isolation_level=None )
        db.execute( 'PRAGMA user_version = 99' )
        db.close()
        with self.assertRaises( bu_store.CatalogError ):
            bu_store.Store( self.path, clock=self.clock )
        os.remove( self.path )
        bu_store.Store( self.path, clock=self.clock, exclusive=True ).close()

    def test_a_file_that_is_not_a_database_is_unavailable_as_corrupt( self ):
        with open( self.path, 'wb' ) as handle:
            handle.write( b'not a database at all, just some bytes' * 100 )
        with self.assertRaises( bu_store.CatalogError ) as caught:
            bu_store.Store( self.path, clock=self.clock )
        self.assertEqual( caught.exception.code, 'catalog.unavailable' )
        self.assertEqual( caught.exception.params, { 'reason': 'corrupt' } )


class TestTransactions( _Tmp ):
    def test_a_write_that_raises_is_rolled_back( self ):
        store = self.open()
        with self.assertRaises( RuntimeError ):
            with store.write() as db:
                db.execute( "INSERT INTO folders ( path, added_at ) VALUES ( '/x', 1 )" )
                raise RuntimeError( 'boom' )
        self.assertEqual( store.read( 'SELECT COUNT(*) FROM folders' )[0][0], 0 )

    def test_another_connections_commit_moves_data_version( self ):
        store = self.open()
        before = store.data_version()
        other = sqlite3.connect( self.path, isolation_level=None )
        self.addCleanup( other.close )
        other.execute( "INSERT INTO folders ( path, added_at ) VALUES ( '/y', 1 )" )
        self.assertNotEqual( store.data_version(), before )

    def test_foreign_keys_cascade( self ):
        store = self.open()
        with store.write() as db:
            db.execute( "INSERT INTO items ( hash, size, name, added_at ) VALUES ( 'h', 1, 'n', 1 )" )
            db.execute( "INSERT INTO locations ( hash, path, size, mtime_ns, dev, ino, added_at, seen_at )"
                        " VALUES ( 'h', '/p', 1, 1, 1, 1, 1, 1 )" )
        with store.write() as db:
            db.execute( "DELETE FROM items WHERE hash = 'h'" )
        self.assertEqual( store.read( 'SELECT COUNT(*) FROM locations' )[0][0], 0 )


_HOLD = textwrap.dedent( """
    import sys, time
    sys.path.insert( 0, sys.argv[1] )
    import betautils_catalog_store as bu_store
    store = bu_store.Store( sys.argv[2], exclusive=( sys.argv[3] == 'x' ) )
    print( 'held', flush=True )
    sys.stdin.readline()
""" )


class TestLock( _Tmp ):
    def hold( self, mode ):
        repo = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
        child = subprocess.Popen( [ sys.executable, '-c', _HOLD, repo, self.path, mode ],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True )
        self.addCleanup( child.wait, 30 )
        self.addCleanup( child.stdin.close )
        self.assertEqual( child.stdout.readline().strip(), 'held' )
        return child

    def test_two_shared_openers_work_together( self ):
        self.hold( 's' )
        self.open().close()

    def test_exclusive_is_refused_while_another_process_has_it_open( self ):
        self.hold( 's' )
        with self.assertRaises( bu_store.CatalogError ) as caught:
            bu_store.Store( self.path, exclusive=True )
        self.assertEqual( caught.exception.code, 'catalog.unavailable' )
        self.assertEqual( caught.exception.params, { 'reason': 'in_use' } )

    def test_shared_is_refused_while_another_process_holds_it_exclusive( self ):
        self.hold( 'x' )
        with self.assertRaises( bu_store.CatalogError ) as caught:
            bu_store.Store( self.path )
        self.assertEqual( caught.exception.params, { 'reason': 'in_use' } )

    def test_the_lock_is_free_once_the_holder_ends( self ):
        child = self.hold( 's' )
        child.stdin.close()
        child.wait( 30 )
        bu_store.Store( self.path, exclusive=True ).close()


def _read( path ):
    with open( path, 'rb' ) as handle:
        return handle.read()


class _Logger:
    def __init__( self ):
        self.warnings = []

    def warning( self, message, *args ):
        self.warnings.append( message%args if args else message )

    def info( self, *_args ):
        pass


if __name__ == '__main__':
    unittest.main()
