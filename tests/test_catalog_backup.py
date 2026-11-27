"""
A backup of the catalog and putting it back: feedback (with pins and
history) comes back exactly as it was backed up, on this machine or a new
one; a backup from an older version takes the schema steps since; a
damaged, foreign or newer file is refused and the catalog is untouched;
restore refuses while the catalog is open; the tool asks, logs and exits
with its codes.
"""

import io
import os
import shutil
import sqlite3
import tempfile
import unittest

import betautils_catalog as bu_catalog
import betautils_catalog_backup as bu_backup
import betautils_catalog_feedback as bu_feedback
import betautils_catalog_store as bu_store

from tests.test_catalog_feedback import KEYS, _Library, hasher


def feedback_rows( path ):
    db = sqlite3.connect( path )
    try:
        return db.execute( 'SELECT hash, at_seconds, label, verdict, detection_key, censor_key, encode_key, preview,'
                           ' created_at, app_version FROM feedback ORDER BY id' ).fetchall()
    finally:
        db.close()


class _Backed( _Library ):
    def setUp( self ):
        super().setUp()
        self.feedback.add( self.output, 12.5, 'missed', 'exposed_breast' )
        self.feedback.add( self.preview, 3, 'too_much' )
        self.catalog.pin_type( self.digest, 'compilation' )
        self.kept = os.path.join( self.root, 'kept' )


class TestBackup( _Backed ):
    def test_a_backup_beside_an_open_catalog_holds_everything_so_far( self ):
        summary = bu_backup.backup( self.kept, path=self.db_path )   # the catalog is open here
        self.assertTrue( summary['path'].startswith( self.kept + os.sep ) )
        self.assertEqual( ( summary['version'], summary['feedback'], summary['items'] ),
                          ( bu_store.LATEST_VERSION, 2, 1 ) )
        self.assertEqual( feedback_rows( summary['path'] ), feedback_rows( self.db_path ) )
        self.assertEqual( [ name for name in os.listdir( self.kept ) ], [ os.path.basename( summary['path'] ) ] )

    def test_a_backup_is_one_file_with_nothing_beside_it( self ):
        target = os.path.join( self.root, 'one.sqlite' )
        bu_backup.backup( target, path=self.db_path )
        db = sqlite3.connect( target )
        try:
            self.assertEqual( db.execute( 'PRAGMA journal_mode' ).fetchone()[0].lower(), 'delete' )
        finally:
            db.close()
        self.assertFalse( os.path.exists( target + bu_backup.PARTIAL_SUFFIX ) )

    def test_an_existing_target_is_kept_unless_asked( self ):
        target = os.path.join( self.root, 'one.sqlite' )
        with open( target, 'w' ) as handle:
            handle.write( 'mine' )
        with self.assertRaises( bu_backup.BackupError ) as caught:
            bu_backup.backup( target, path=self.db_path )
        self.assertEqual( caught.exception.code, 'backup.exists' )
        with open( target ) as handle:
            self.assertEqual( handle.read(), 'mine' )
        bu_backup.backup( target, path=self.db_path, overwrite=True )
        self.assertEqual( len( feedback_rows( target ) ), 2 )

    def test_a_folder_named_like_a_file_gets_a_backup_inside( self ):
        folder = os.path.join( self.root, 'odd.sqlite' )
        os.makedirs( folder )
        summary = bu_backup.backup( folder, path=self.db_path )
        self.assertEqual( os.path.dirname( summary['path'] ), folder )

    def test_the_catalog_itself_is_never_the_target( self ):
        with self.assertRaises( bu_backup.BackupError ) as caught:
            bu_backup.backup( self.db_path, path=self.db_path, overwrite=True )
        self.assertEqual( caught.exception.code, 'backup.same_file' )

    def test_no_catalog_no_backup( self ):
        with self.assertRaises( bu_backup.BackupError ) as caught:
            bu_backup.backup( self.kept, path=os.path.join( self.root, 'none.sqlite' ) )
        self.assertEqual( caught.exception.code, 'backup.no_catalog' )


class TestRestore( _Backed ):
    def test_feedback_comes_back_exactly_as_backed_up( self ):
        saved = bu_backup.backup( self.kept, path=self.db_path )['path']
        before = feedback_rows( self.db_path )
        later = self.feedback.add( self.output, 40, 'missed' )
        self.feedback.remove( self.feedback.of_item( self.digest )[0]['id'] )
        self.catalog.close()
        result = bu_backup.restore( saved, path=self.db_path )
        self.assertEqual( feedback_rows( self.db_path ), before )
        self.assertEqual( ( result['restored']['feedback'], result['version'] ), ( 2, bu_store.LATEST_VERSION ) )
        self.assertEqual( result['kept'], self.db_path + bu_backup.OLD_SUFFIX )
        # The replaced file is kept whole, the mark made after the backup with it.
        self.assertIn( 40.0, [ row[1] for row in feedback_rows( result['kept'] ) ] )
        self.assertNotIn( later['at_seconds'], [ row[1] for row in feedback_rows( self.db_path ) ] )

    def test_a_new_machine_gets_feedback_pins_and_history_back_once_the_file_is_found( self ):
        saved = bu_backup.backup( self.kept, path=self.db_path )['path']
        self.catalog.close()
        elsewhere = os.path.join( self.root, 'new-machine' )
        videos = os.path.join( elsewhere, 'videos' )
        os.makedirs( videos )
        shutil.copyfile( self.source, os.path.join( videos, 'renamed.mp4' ) )
        new_db = os.path.join( elsewhere, 'data', 'catalog.sqlite' )
        result = bu_backup.restore( saved, path=new_db )
        self.assertIsNone( result['kept'] )
        catalog = bu_catalog.Catalog( path=new_db, seed_folders=[], settle=0.0, hasher=hasher, thumbnails=False )
        self.addCleanup( catalog.close )
        catalog.add_folder( videos )
        catalog.scanner.pass_now()
        catalog.scanner.hash_ready()
        item = catalog.index.item_row( self.digest )
        self.assertEqual( item['type_pin'], 'compilation' )
        self.assertIn( os.path.join( videos, 'renamed.mp4' ),
                       [ row['path'] for row in catalog.index.locations_of( self.digest ) if row['state'] == 'present' ] )
        marks = bu_feedback.Feedback( catalog.store ).of_item( self.digest )
        self.assertEqual( [ ( row['at_seconds'], row['verdict'], row['label'] ) for row in marks ],
                          [ ( 12.5, 'missed', 'exposed_breast' ), ( 603.0, 'too_much', None ) ] )
        self.assertEqual( marks[0]['keys'], KEYS )
        self.assertEqual( len( catalog.history.runs_of( self.digest ) ), 2 )

    def test_a_backup_from_an_older_version_takes_the_steps_since( self ):
        old = os.path.join( self.root, 'old.sqlite' )
        db = sqlite3.connect( old, isolation_level=None )
        bu_store.migrate( db, old, migrations=bu_store.MIGRATIONS[:3] )
        db.execute( "INSERT INTO items ( hash, size, name, added_at ) VALUES ( '0123456789abcdef', 5, 'x.mp4', 1 )" )
        db.close()
        self.catalog.close()
        result = bu_backup.restore( old, path=self.db_path )
        self.assertEqual( ( result['restored']['version'], result['version'] ), ( 3, bu_store.LATEST_VERSION ) )
        self.assertEqual( feedback_rows( self.db_path ), [] )
        self.assertTrue( os.path.exists( '%s.v3%s'%( self.db_path, bu_store.BACKUP_SUFFIX ) ) )

    def test_restore_refuses_while_the_catalog_is_open( self ):
        saved = bu_backup.backup( self.kept, path=self.db_path )['path']
        before = feedback_rows( self.db_path )
        with self.assertRaises( bu_store.CatalogError ) as caught:
            bu_backup.restore( saved, path=self.db_path )
        self.assertEqual( ( caught.exception.code, caught.exception.params ), ( 'catalog.unavailable', { 'reason': 'in_use' } ) )
        self.assertFalse( os.path.exists( self.db_path + bu_backup.OLD_SUFFIX ) )
        self.assertEqual( feedback_rows( self.db_path ), before )


class TestRefused( unittest.TestCase ):
    def setUp( self ):
        self.root = tempfile.mkdtemp( prefix='catalog-backup-' )
        self.addCleanup( shutil.rmtree, self.root, ignore_errors=True )
        self.db_path = os.path.join( self.root, 'catalog.sqlite' )
        store = bu_store.Store( self.db_path )
        with store.write() as db:
            db.execute( "INSERT INTO feedback ( hash, at_seconds, verdict, detection_key, censor_key, encode_key,"
                        " created_at ) VALUES ( 'aaaaaaaaaaaaaaaa', 1, 'missed', 'd', 'c', 'e', 1 )" )
        store.close()
        self.before = feedback_rows( self.db_path )

    def refused( self, source, code, reason=None ):
        with self.assertRaises( ( bu_backup.BackupError, bu_store.CatalogError ) ) as caught:
            bu_backup.restore( source, path=self.db_path )
        self.assertEqual( caught.exception.code, code )
        if reason is not None:
            self.assertEqual( caught.exception.params['reason'], reason )
        self.assertEqual( feedback_rows( self.db_path ), self.before )
        self.assertFalse( os.path.exists( self.db_path + bu_backup.OLD_SUFFIX ) )
        self.assertFalse( os.path.exists( self.db_path + bu_backup.RESTORE_SUFFIX ) )

    def test_a_file_that_is_not_sqlite( self ):
        junk = os.path.join( self.root, 'junk.sqlite' )
        with open( junk, 'wb' ) as handle:
            handle.write( b'not a database at all' * 100 )
        self.refused( junk, 'backup.not_a_catalog', 'damaged' )

    def test_a_damaged_catalog( self ):
        damaged = os.path.join( self.root, 'damaged.sqlite' )
        shutil.copyfile( self.db_path, damaged )
        size = os.path.getsize( damaged )
        with open( damaged, 'r+b' ) as handle:   # past the header: the pages the tables live in
            handle.seek( 4096 )
            handle.write( b'\xff' * ( size - 4096 ) )
        self.refused( damaged, 'backup.not_a_catalog', 'damaged' )

    def test_damage_in_a_table_only_a_full_read_finds( self ):
        damaged = os.path.join( self.root, 'damaged.sqlite' )
        shutil.copyfile( self.db_path, damaged )
        db = sqlite3.connect( damaged )
        root = db.execute( "SELECT rootpage FROM sqlite_master WHERE name = 'locations'" ).fetchone()[0]
        size = db.execute( 'PRAGMA page_size' ).fetchone()[0]
        db.close()
        with open( damaged, 'r+b' ) as handle:   # the page type byte of the locations table
            handle.seek( ( root - 1 )*size )
            handle.write( b'\x01' )
        self.refused( damaged, 'backup.not_a_catalog', 'damaged' )

    def test_a_catalog_whose_schema_version_was_never_set( self ):
        unset = os.path.join( self.root, 'unset.sqlite' )
        shutil.copyfile( self.db_path, unset )
        db = sqlite3.connect( unset )
        db.execute( 'PRAGMA user_version = 0' )
        db.close()
        self.refused( unset, 'backup.not_a_catalog', 'not_a_catalog' )

    def test_a_stale_log_beside_the_replaced_catalog_is_not_applied_to_the_restored_one( self ):
        saved = os.path.join( self.root, 'saved.sqlite' )
        bu_backup.backup( saved, path=self.db_path )
        # A write left in the old file's log, as an engine killed mid-run leaves it.
        db = sqlite3.connect( self.db_path, isolation_level=None )
        db.execute( 'PRAGMA wal_autocheckpoint = 0' )
        db.execute( "INSERT INTO feedback ( hash, at_seconds, verdict, detection_key, censor_key, encode_key,"
                    " created_at ) VALUES ( 'bbbbbbbbbbbbbbbb', 2, 'missed', 'd', 'c', 'e', 1 )" )
        with open( self.db_path + '-wal', 'rb' ) as handle:
            stale = handle.read()
        db.close()
        with open( self.db_path + '-wal', 'wb' ) as handle:
            handle.write( stale )
        bu_backup.restore( saved, path=self.db_path )
        self.assertEqual( feedback_rows( self.db_path ), self.before )
        self.assertEqual( len( feedback_rows( self.db_path + bu_backup.OLD_SUFFIX ) ), 2 )

    def test_another_sqlite_file( self ):
        other = os.path.join( self.root, 'other.sqlite' )
        db = sqlite3.connect( other )
        db.execute( 'CREATE TABLE notes ( id INTEGER )' )
        db.execute( 'PRAGMA user_version = 1' )
        db.close()
        self.refused( other, 'backup.not_a_catalog', 'not_a_catalog' )

    def test_an_empty_sqlite_file( self ):
        empty = os.path.join( self.root, 'empty.sqlite' )
        sqlite3.connect( empty ).close()
        self.refused( empty, 'backup.not_a_catalog', 'not_a_catalog' )

    def test_a_catalog_from_a_newer_version( self ):
        newer = os.path.join( self.root, 'newer.sqlite' )
        shutil.copyfile( self.db_path, newer )
        db = sqlite3.connect( newer )
        db.execute( 'PRAGMA user_version = %d'%( bu_store.LATEST_VERSION + 1 ) )
        db.close()
        self.refused( newer, 'catalog.too_new' )

    def test_a_missing_file( self ):
        self.refused( os.path.join( self.root, 'nowhere.sqlite' ), 'backup.no_such_file' )

    def test_the_catalog_itself( self ):
        self.refused( self.db_path, 'backup.same_file' )


class TestTool( _Backed ):
    """tools/setup/catalog_backup.py: asks, logs, exit codes."""

    def tool( self, *argv, stdin='' ):
        import importlib.util
        repo = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
        spec = importlib.util.spec_from_file_location( 'catalog_backup_tool',
                                                       os.path.join( repo, 'tools', 'setup', 'catalog_backup.py' ) )
        module = importlib.util.module_from_spec( spec )
        spec.loader.exec_module( module )
        self.log = os.path.join( self.root, 'backup.log' )
        return module.main( [ '--catalog', self.db_path, '--log-file', self.log, '--console-level', 'error' ]
                            + list( argv ), stdin=io.StringIO( stdin ), stdout=io.StringIO() )

    def logged( self ):
        with open( self.log ) as handle:
            return handle.read()

    def test_backup_then_restore_round_trips_feedback( self ):
        target = os.path.join( self.root, 'saved.sqlite' )
        before = feedback_rows( self.db_path )
        self.assertEqual( self.tool( 'backup', '--to', target ), 0 )
        self.assertIn( 'catalog backed up', self.logged() )
        self.catalog.close()
        os.remove( self.db_path )
        self.assertEqual( self.tool( 'restore', target, '--yes' ), 0 )
        self.assertIn( 'catalog restored', self.logged() )
        self.assertEqual( feedback_rows( self.db_path ), before )

    def test_restore_without_a_terminal_or_yes_changes_nothing( self ):
        target = os.path.join( self.root, 'saved.sqlite' )
        self.tool( 'backup', '--to', target )
        self.catalog.close()
        self.assertEqual( self.tool( 'restore', target ), 0 )
        self.assertFalse( os.path.exists( self.db_path + bu_backup.OLD_SUFFIX ) )

    def test_refusals_exit_1_and_bad_arguments_2( self ):
        target = os.path.join( self.root, 'saved.sqlite' )
        self.tool( 'backup', '--to', target )
        self.assertEqual( self.tool( 'restore', target, '--yes' ), 1 )   # the catalog is open
        self.assertIn( 'close it', self.logged() )
        self.assertEqual( self.tool( 'restore' ), 2 )
        with self.assertRaises( SystemExit ) as caught:
            self.tool( 'copy' )
        self.assertEqual( caught.exception.code, 2 )

    def test_no_action_and_no_terminal_does_nothing( self ):
        self.assertEqual( self.tool(), 0 )
        self.assertFalse( os.path.exists( bu_backup.default_folder() ) and os.listdir( bu_backup.default_folder() ) )


if __name__ == '__main__':
    unittest.main()
