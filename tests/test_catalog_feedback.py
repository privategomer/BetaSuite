"""
Feedback in the catalog: a mark keeps the source hash, the time in the
source, the label, the verdict and all three keys of the result it was
made on; the keys come from the catalog, never the caller; a preview's
time is moved onto the source's timeline; a mark survives schema steps
and a rebuild, item on disk or not; forgetting the item removes it.
"""

import os
import shutil
import sqlite3
import tempfile
import unittest

import betaconst
import betautils_cache_paths as bu_cache
import betautils_catalog as bu_catalog
import betautils_catalog_feedback as bu_feedback
import betautils_catalog_rebuild as bu_rebuild
import betautils_catalog_store as bu_store
import betautils_hash as bu_hash
import betautils_run_record as bu_run_record

FIXTURE_V1 = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), 'golden', 'catalog_v1.sql' )
KEYS = { 'detection': 'd1a', 'censor': 'c2b', 'encode': 'e3c' }


def hasher( path ):
    return bu_hash.md5_for_file( path, 16, use_memo=False )


class _Library( unittest.TestCase ):
    """One source with a finished output and a preview run, in a temporary catalog."""

    def setUp( self ):
        self.root = tempfile.mkdtemp( prefix='catalog-feedback-' )
        self.addCleanup( shutil.rmtree, self.root, ignore_errors=True )
        self.db_path = os.path.join( self.root, 'data', 'catalog.sqlite' )
        self.videos = os.path.join( self.root, 'videos' )
        os.makedirs( self.videos )
        os.makedirs( betaconst.video_path_censored, exist_ok=True )
        self.source = os.path.join( self.videos, 'clip.mp4' )
        with open( self.source, 'wb' ) as handle:
            handle.write( os.urandom( 4000 ) )
        self.digest = hasher( self.source )
        self.now = 5000.0
        self.output = self._file( betaconst.video_path_censored, bu_cache.video_output_basename(
            'clip', self.digest, [ 640 ], 30, KEYS['detection'], KEYS['censor'], KEYS['encode'] ) + '.mp4' )
        self.preview = self._file( betaconst.video_path_censored, bu_cache.video_output_basename(
            'clip', self.digest, [ 640 ], 30, 'aa1', 'bb2', 'ff3',
            preview_suffix=bu_cache.preview_cache_suffix( True, 600.0, 20, None ) ) + '.mp4' )
        self.catalog = self.open_catalog()
        self.catalog.add_folder( self.videos )
        self.catalog.scanner.pass_now()
        self.catalog.scanner.hash_ready()
        history = self.catalog.history
        history.record( self._run( 'processed', output=self.output, keys=KEYS ) )
        history.record( self._run( 'processed', output=self.preview, preview=True,
                                   keys={ 'detection': 'aa1', 'censor': 'bb2', 'encode': 'ff3' } ) )
        self.feedback = bu_feedback.Feedback( self.catalog.store, clock=lambda: self.now )

    def open_catalog( self ):
        catalog = bu_catalog.Catalog( path=self.db_path, seed_folders=[], settle=0.0, hasher=hasher, thumbnails=False )
        self.addCleanup( catalog.close )
        return catalog

    def _file( self, folder, name ):
        path = os.path.join( folder, name )
        with open( path, 'wb' ) as handle:
            handle.write( b'censored' )
        self.addCleanup( lambda: os.path.exists( path ) and os.remove( path ) )
        return path

    def _run( self, outcome, **fields ):
        record = bu_run_record.RunRecord( self.source, program='engine', job='j' )
        record.hash = self.digest
        for key, value in fields.items():
            setattr( record, key, value )
        return record.end( outcome, None )

    def rows( self, path=None ):
        db = sqlite3.connect( path or self.db_path )
        try:
            return db.execute( 'SELECT hash, at_seconds, label, verdict, detection_key, censor_key, encode_key,'
                               ' preview, created_at FROM feedback ORDER BY id' ).fetchall()
        finally:
            db.close()


class TestRecord( _Library ):
    def test_a_mark_on_an_output_keeps_every_field_of_section_8( self ):
        record = self.feedback.add( self.output, 12.5, 'missed', 'exposed_breast' )
        self.assertEqual( ( record['hash'], record['at_seconds'], record['label'], record['verdict'], record['keys'],
                            record['preview'], record['created_at'] ),
                          ( self.digest, 12.5, 'exposed_breast', 'missed', KEYS, False, 5000.0 ) )
        self.assertEqual( self.rows(), [ ( self.digest, 12.5, 'exposed_breast', 'missed', 'd1a', 'c2b', 'e3c', 0,
                                           5000.0 ) ] )

    def test_not_sure_is_kept_without_a_label( self ):
        record = self.feedback.add( self.output, 3, 'too_much' )
        self.assertIsNone( record['label'] )
        self.assertEqual( self.rows()[0][3], 'too_much' )

    def test_a_preview_mark_is_on_the_source_timeline_with_the_preview_keys( self ):
        record = self.feedback.add( self.preview, 4.25, 'too_much', 'exposed_buttocks' )
        self.assertEqual( ( record['at_seconds'], record['preview'], record['keys'] ),
                          ( 604.25, True, { 'detection': 'aa1', 'censor': 'bb2', 'encode': 'ff3' } ) )

    def test_a_result_lists_its_own_marks_on_its_own_timeline( self ):
        self.feedback.add( self.output, 30, 'missed', 'exposed_breast' )
        self.feedback.add( self.preview, 2, 'too_much' )
        self.feedback.add( self.preview, 1, 'missed' )
        result, records = self.feedback.of_result( self.preview )
        self.assertEqual( ( result['hash'], result['preview'], result['offset'] ), ( self.digest, True, 600.0 ) )
        self.assertEqual( [ ( row['time'], row['at_seconds'], row['verdict'] ) for row in records ],
                          [ ( 1.0, 601.0, 'missed' ), ( 2.0, 602.0, 'too_much' ) ] )
        self.assertEqual( [ row['time'] for row in self.feedback.of_result( self.output )[1] ], [ 30.0 ] )
        self.assertEqual( [ row['at_seconds'] for row in self.feedback.of_item( self.digest ) ], [ 30.0, 601.0, 602.0 ] )

    def test_a_source_is_not_a_result( self ):
        with self.assertRaises( bu_store.CatalogError ) as caught:
            self.feedback.add( self.source, 1, 'missed' )
        self.assertEqual( caught.exception.code, 'feedback.not_a_result' )
        self.assertEqual( self.feedback.of_result( self.source ), ( None, [] ) )
        self.assertEqual( self.rows(), [] )

    def test_an_output_name_of_an_unknown_item_is_not_a_result( self ):
        stray = self._file( betaconst.video_path_censored, bu_cache.video_output_basename(
            'other', 'f' * 16, [ 640 ], 30, 'd', 'c', 'e' ) + '.mp4' )
        with self.assertRaises( bu_store.CatalogError ):
            self.feedback.add( stray, 1, 'missed' )

    def test_a_renamed_result_the_catalog_never_saw_is_not_a_result( self ):
        renamed = self._file( betaconst.video_path_censored, 'my clip.mp4' )
        with self.assertRaises( bu_store.CatalogError ):
            self.feedback.add( renamed, 1, 'missed' )

    def test_a_result_in_a_folder_the_catalog_does_not_scan_is_found_by_its_name( self ):
        elsewhere = os.path.join( self.root, 'elsewhere' )
        os.makedirs( elsewhere )
        copy = self._file( elsewhere, os.path.basename( self.output ) )
        self.assertEqual( self.feedback.add( copy, 1, 'missed' )['keys'], KEYS )

    def test_a_result_no_longer_on_disk_is_not_a_result( self ):
        os.remove( self.output )
        os.remove( self.preview )
        self.catalog.scanner.pass_now()   # the output row is kept, marked not present
        for path in ( self.output, self.preview ):
            with self.assertRaises( bu_store.CatalogError ):
                self.feedback.add( path, 1, 'missed' )

    def test_bad_values_are_refused_before_anything_is_written( self ):
        for args in ( ( 1, 'wrong' ), ( -1, 'missed' ), ( float( 'nan' ), 'missed' ), ( float( 'inf' ), 'missed' ),
                      ( True, 'missed' ), ( '3', 'missed' ) ):
            with self.assertRaises( ValueError ):
                self.feedback.add( self.output, *args )
        with self.assertRaises( ValueError ):
            self.feedback.add( self.output, 1, 'missed', 'exposed_elbow' )
        self.assertEqual( self.rows(), [] )

    def test_the_table_itself_refuses_another_verdict( self ):
        with self.assertRaises( sqlite3.IntegrityError ):
            with self.catalog.store.write() as db:
                db.execute( "INSERT INTO feedback ( hash, at_seconds, verdict, detection_key, censor_key, encode_key,"
                            " created_at ) VALUES ( 'h', 1, 'fine', 'd', 'c', 'e', 1 )" )

    def test_remove_drops_one_mark( self ):
        first = self.feedback.add( self.output, 1, 'missed' )
        second = self.feedback.add( self.output, 2, 'missed' )
        self.assertEqual( self.feedback.remove( first['id'] ), self.digest )
        self.assertEqual( [ row['id'] for row in self.feedback.of_item( self.digest ) ], [ second['id'] ] )
        with self.assertRaises( bu_store.CatalogError ) as caught:
            self.feedback.remove( first['id'] )
        self.assertEqual( caught.exception.code, 'feedback.no_such_record' )

    def test_forgetting_the_item_removes_its_feedback_only( self ):
        self.feedback.add( self.output, 1, 'missed' )
        with self.catalog.store.write() as db:
            db.execute( "INSERT INTO feedback ( hash, at_seconds, verdict, detection_key, censor_key, encode_key,"
                        " created_at ) VALUES ( 'ffffffffffffffff', 1, 'missed', 'd', 'c', 'e', 1 )" )
        self.catalog.forget( self.digest )
        self.assertEqual( [ row[0] for row in self.rows() ], [ 'ffffffffffffffff' ] )


class TestPreviewOffset( unittest.TestCase ):
    def name( self, offset, seconds ):
        return bu_cache.video_output_basename( 'clip', 'a' * 16, [ 640 ], 30, 'd', 'c', 'e',
                                               preview_suffix=bu_cache.preview_cache_suffix(
                                                   True, offset, seconds, None ) ) + '.mp4'

    def test_offsets_read_from_preview_names( self ):
        self.assertEqual( bu_feedback.preview_offset( self.name( 600.0, 20 ) ), 600.0 )
        self.assertEqual( bu_feedback.preview_offset( self.name( 12.5, None ) ), 12.5 )
        self.assertEqual( bu_feedback.preview_offset( self.name( 0.0, 20 ) ), 0.0 )

    def test_a_full_output_or_another_file_has_no_offset( self ):
        self.assertEqual( bu_feedback.preview_offset( bu_cache.video_output_basename(
            'clip', 'a' * 16, [ 640 ], 30, 'd', 'c', 'e' ) + '.mp4' ), 0.0 )
        self.assertEqual( bu_feedback.preview_offset( 'clip-preview@600.0s.mp4' ), 0.0 )


class TestSurvives( _Library ):
    def test_feedback_survives_a_later_schema_step( self ):
        self.feedback.add( self.output, 7, 'missed', 'exposed_vulva' )
        before = self.rows()
        self.catalog.close()
        later = bu_store.MIGRATIONS + ( ( bu_store.LATEST_VERSION + 1, 'test_later_step',
                                          "ALTER TABLE feedback ADD COLUMN note TEXT;\n"
                                          "CREATE TABLE later ( id INTEGER PRIMARY KEY );\n" ), )
        db = sqlite3.connect( self.db_path, isolation_level=None )
        try:
            self.assertEqual( bu_store.migrate( db, self.db_path, migrations=later ), [ bu_store.LATEST_VERSION + 1 ] )
        finally:
            db.close()
        self.assertEqual( self.rows(), before )
        backup = '%s.v%d.bak'%( self.db_path, bu_store.LATEST_VERSION )
        self.assertEqual( self.rows( backup ), before )

    def test_feedback_survives_a_rebuild_with_its_item_on_disk_or_not( self ):
        self.feedback.add( self.output, 7, 'missed', 'exposed_vulva' )
        self.feedback.add( self.preview, 1, 'too_much' )
        with self.catalog.store.write() as db:   # an item the catalog lost long ago
            db.execute( "INSERT INTO feedback ( hash, at_seconds, verdict, detection_key, censor_key, encode_key,"
                        " created_at ) VALUES ( 'eeeeeeeeeeeeeeee', 1, 'missed', 'd', 'c', 'e', 1 )" )
        before = self.rows()
        self.catalog.close()
        os.remove( self.source )   # the item is kept for its feedback, its path missing
        summary = bu_rebuild.rebuild( self.db_path, seed_folders=[], hasher=hasher )
        self.assertEqual( summary['carried_feedback'], 3 )
        self.assertEqual( self.rows(), before )
        db = sqlite3.connect( self.db_path )
        try:
            self.assertEqual( db.execute( 'SELECT COUNT(*) FROM items WHERE hash = ?', ( self.digest, ) ).fetchone()[0], 1 )
        finally:
            db.close()


    def test_an_item_known_only_by_its_feedback_is_kept_by_a_rebuild( self ):
        with self.catalog.store.write() as db:
            db.execute( "INSERT INTO items ( hash, size, name, added_at ) VALUES ( 'dddddddddddddddd', 9, 'gone.mp4', 1 )" )
            db.execute( "INSERT INTO feedback ( hash, at_seconds, verdict, detection_key, censor_key, encode_key,"
                        " created_at ) VALUES ( 'dddddddddddddddd', 1, 'missed', 'd', 'c', 'e', 1 )" )
        self.catalog.close()
        bu_rebuild.rebuild( self.db_path, seed_folders=[], hasher=hasher )
        db = sqlite3.connect( self.db_path )
        try:
            self.assertEqual( db.execute( "SELECT name FROM items WHERE hash = 'dddddddddddddddd'" ).fetchall(),
                              [ ( 'gone.mp4', ) ] )
        finally:
            db.close()


class TestFromOlderFiles( unittest.TestCase ):
    def setUp( self ):
        self.root = tempfile.mkdtemp( prefix='catalog-feedback-old-' )
        self.addCleanup( shutil.rmtree, self.root, ignore_errors=True )
        self.db_path = os.path.join( self.root, 'catalog.sqlite' )

    def test_a_version_three_file_gains_an_empty_feedback_table_and_keeps_its_runs( self ):
        db = sqlite3.connect( self.db_path, isolation_level=None )
        bu_store.migrate( db, self.db_path, migrations=bu_store.MIGRATIONS[:3] )
        db.execute( "INSERT INTO items ( hash, size, name, added_at ) VALUES ( '0123456789abcdef', 5, 'x.mp4', 1 )" )
        db.execute( "INSERT INTO runs ( hash, path, started_at, ended_at, outcome, timing ) VALUES"
                    " ( '0123456789abcdef', '/x.mp4', 1, 2, 'processed', '{}' )" )
        db.close()
        store = bu_store.Store( self.db_path )
        self.addCleanup( store.close )
        self.assertEqual( store.read( 'PRAGMA user_version' )[0][0], bu_store.LATEST_VERSION )
        self.assertEqual( store.read( 'SELECT COUNT(*) FROM runs' )[0][0], 1 )
        self.assertEqual( store.read( 'SELECT COUNT(*) FROM feedback' )[0][0], 0 )
        self.assertTrue( os.path.exists( self.db_path + '.v3.bak' ) )

    def test_a_version_one_file_moves_up_through_the_feedback_step( self ):
        db = sqlite3.connect( self.db_path )
        with open( FIXTURE_V1 ) as handle:
            db.executescript( handle.read() )
        db.close()
        store = bu_store.Store( self.db_path )
        self.addCleanup( store.close )
        self.assertEqual( store.read( 'SELECT COUNT(*) FROM feedback' )[0][0], 0 )
        self.assertEqual( store.read( 'SELECT COUNT(*) FROM items' )[0][0], 2 )

    def test_a_rebuild_of_a_version_three_file_carries_no_feedback_and_works( self ):
        db = sqlite3.connect( self.db_path, isolation_level=None )
        bu_store.migrate( db, self.db_path, migrations=bu_store.MIGRATIONS[:3] )
        db.close()
        carried = bu_rebuild.read_old( self.db_path )
        self.assertTrue( carried.readable )
        self.assertEqual( carried.feedback, [] )


if __name__ == '__main__':
    unittest.main()
