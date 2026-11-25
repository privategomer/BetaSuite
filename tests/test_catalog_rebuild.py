"""
A catalog rebuilt from disk matches the one it replaces: items, paths
and their availability, outputs and done, from the disk; folders, pins,
history (and a failed status) and items not on disk now, carried from
the old file. Without a readable old file only what the disk holds comes
back, and the tests say exactly what is lost.
"""

import os
import shutil
import sqlite3
import tempfile
import unittest

import betaconst
import betautils_cache_paths as bu_cache
import betautils_catalog as bu_catalog
import betautils_catalog_history as bu_history
import betautils_catalog_index as bu_index
import betautils_catalog_rebuild as bu_rebuild
import betautils_catalog_store as bu_store
import betautils_hash as bu_hash
import betautils_run_record as bu_run_record


def hasher( path ):
    return bu_hash.md5_for_file( path, 16, use_memo=False )


def projection( path ):
    """Everything a rebuild should reproduce, with ids and clock values left out."""
    db = sqlite3.connect( path )
    try:
        folders = { row[0]: ( row[1], row[2] ) for row in db.execute( 'SELECT id, path, type_from_folder FROM folders' ) }
        return {
            'folders': sorted( folders.values() ),
            'items': sorted( db.execute( 'SELECT hash, kind, size, status, type_pin, last_profile, last_rule,'
                                         ' last_rule_params FROM items' ).fetchall() ),
            'locations': sorted( ( row[0], row[1], folders.get( row[2], ( None, ) )[0], row[3], row[4], row[5] )
                                 for row in db.execute( 'SELECT hash, path, folder_id, size, mtime_ns, state'
                                                        ' FROM locations' ) ),
            'outputs': sorted( db.execute( 'SELECT hash, path, detection_key, censor_key, encode_key, present'
                                           ' FROM outputs' ).fetchall() ),
            'runs': sorted( db.execute( 'SELECT %s FROM runs'%( ', '.join( bu_rebuild._RUN_COLUMNS ) ) ).fetchall(),
                            key=repr ),
        }
    finally:
        db.close()


class _Library( unittest.TestCase ):
    def setUp( self ):
        self.root = tempfile.mkdtemp( prefix='catalog-rebuild-' )
        self.addCleanup( shutil.rmtree, self.root, ignore_errors=True )
        self.db_path = os.path.join( self.root, 'data', 'catalog.sqlite' )
        self.videos = os.path.join( self.root, 'videos' )
        self.share = os.path.join( self.root, 'share' )
        os.makedirs( betaconst.video_path_censored, exist_ok=True )
        self.outputs = []
        self.addCleanup( self._remove_outputs )
        self.digests = {}
        self.build_library()

    def _remove_outputs( self ):
        for path in self.outputs:
            if os.path.exists( path ):
                os.remove( path )

    def write( self, relative, size, folder=None ):
        path = os.path.join( folder or self.videos, relative )
        os.makedirs( os.path.dirname( path ), exist_ok=True )
        with open( path, 'wb' ) as handle:
            handle.write( os.urandom( size ) )
        return path

    def record_run( self, history, name, outcome, reason=None, **fields ):
        path = self.paths[name]
        record = bu_run_record.RunRecord( path, program='engine', job='j' )
        record.hash = self.digests[name]
        for key, value in fields.items():
            setattr( record, key, value )
        history.record( record.end( outcome, reason ) )

    def build_library( self ):
        self.paths = {
            'pinned': self.write( 'scene/pinned.mp4', 3000 ),
            'failed': self.write( 'failed.mkv', 3100 ),
            'done': self.write( 'done.mp4', 3200 ),
            'away': self.write( 'away.mp4', 3300, folder=self.share ),
        }
        shutil.copyfile( self.paths['pinned'], os.path.join( self.videos, 'copy-of-pinned.mp4' ) )
        for name, path in self.paths.items():
            self.digests[name] = hasher( path )
        catalog = bu_catalog.Catalog( path=self.db_path, seed_folders=[], settle=0.0, hasher=hasher, thumbnails=False )
        try:
            catalog.add_folder( self.videos, type_from_folder=True )
            catalog.add_folder( self.share )
            catalog.scanner.pass_now()
            catalog.scanner.hash_ready()
            output = os.path.join( betaconst.video_path_censored, bu_cache.video_output_basename(
                'done', self.digests['done'], [ 640 ], 9, 'd1', 'c2', 'e3' ) + '.mp4' )
            with open( output, 'wb' ) as handle:
                handle.write( b'censored' )
            self.outputs.append( output )
            history = catalog.history
            self.record_run( history, 'failed', 'failed', 'file.failed', error_type='RenderError', stage='render',
                      profile='scene', rule='measured', rule_params={ 'signal': 'short_shot_fraction' } )
            self.record_run( history, 'done', 'processed', None, output=output, profile='quick_cut', rule='measured',
                      keys={ 'detection': 'd1', 'censor': 'c2', 'encode': 'e3' } )
            self.record_run( history, 'away', 'processed', None, preview=True )
            catalog.pin_type( self.digests['pinned'], 'compilation' )
            catalog.scanner.pass_now()
        finally:
            catalog.close()


class TestRebuiltMatches( _Library ):
    def test_a_rebuilt_catalog_equals_the_one_it_replaces( self ):
        before = projection( self.db_path )
        self.assertEqual( len( before['items'] ), 4 )
        self.assertEqual( len( before['runs'] ), 3 )
        summary = bu_rebuild.rebuild( self.db_path, hasher=hasher )
        self.assertEqual( projection( self.db_path ), before )
        self.assertEqual( ( summary['items'], summary['carried_pins'], summary['carried_runs'], summary['old_readable'] ),
                          ( 4, 1, 3, True ) )

    def test_items_on_an_unreachable_share_come_back_offline_with_their_history( self ):
        os.rename( self.share, self.share + '-unmounted' )
        before = projection( self.db_path )
        summary = bu_rebuild.rebuild( self.db_path, hasher=hasher )
        after = projection( self.db_path )
        self.assertEqual( summary['carried_missing'], 1 )
        away = self.digests['away']
        self.assertEqual( [ location[5] for location in after['locations'] if location[0] == away ], [ 'offline' ] )
        self.assertEqual( [ run for run in after['runs'] if run[0] == away ],
                          [ run for run in before['runs'] if run[0] == away ] )

    def test_the_old_file_is_kept_beside_it( self ):
        before = projection( self.db_path )
        bu_rebuild.rebuild( self.db_path, hasher=hasher )
        self.assertEqual( projection( self.db_path + bu_rebuild.OLD_SUFFIX ), before )

    def test_a_rebuilt_catalog_opens_and_lists( self ):
        bu_rebuild.rebuild( self.db_path, hasher=hasher )
        catalog = bu_catalog.Catalog( path=self.db_path, seed_folders=[], thumbnails=False )
        self.addCleanup( catalog.close )
        listed = { item['item']: item for item in catalog.list()['items'] }
        self.assertEqual( listed[self.digests['pinned']]['copies'], 2 )
        self.assertEqual( listed[self.digests['pinned']]['pin'], 'compilation' )
        self.assertEqual( listed[self.digests['failed']]['status'], 'failed' )
        self.assertEqual( listed[self.digests['done']]['outputs'], 1 )


class TestWithoutTheOldFile( _Library ):
    def test_only_what_the_disk_holds_comes_back( self ):
        before = projection( self.db_path )
        with open( self.db_path, 'wb' ) as handle:
            handle.write( b'not a database' * 200 )
        for side in ( '-wal', '-shm' ):
            if os.path.exists( self.db_path + side ):
                os.remove( self.db_path + side )
        summary = bu_rebuild.rebuild( self.db_path, seed_folders=[ self.videos, self.share ], hasher=hasher )
        after = projection( self.db_path )
        self.assertFalse( summary['old_readable'] )
        # From disk: the same items, paths and outputs; done from the output.
        self.assertEqual( { row[0] for row in after['items'] }, { row[0] for row in before['items'] } )
        self.assertEqual( [ ( row[0], row[1] ) for row in after['locations'] ],
                          [ ( row[0], row[1] ) for row in before['locations'] ] )
        self.assertEqual( after['outputs'], before['outputs'] )
        status = { row[0]: row[3] for row in after['items'] }
        self.assertEqual( status[self.digests['done']], 'done' )
        # Lost with the file, as documented: pins, history, a failed status.
        self.assertEqual( after['runs'], [] )
        self.assertEqual( [ row[4] for row in after['items'] ], [ None ] * 4 )
        self.assertEqual( status[self.digests['failed']], 'new' )

    def test_a_missing_catalog_is_built_from_the_seed_folders( self ):
        os.remove( self.db_path )
        for side in ( '-wal', '-shm' ):
            if os.path.exists( self.db_path + side ):
                os.remove( self.db_path + side )
        summary = bu_rebuild.rebuild( self.db_path, seed_folders=[ self.videos ], hasher=hasher )
        self.assertEqual( summary['items'], 3 )
        self.assertEqual( projection( self.db_path )['folders'], [ ( bu_index.norm( self.videos ), 1 ) ] )


class TestRefusals( _Library ):
    def test_refused_while_the_catalog_is_open( self ):
        store = bu_store.Store( self.db_path )
        self.addCleanup( store.close )
        before = projection( self.db_path )
        with self.assertRaises( bu_store.CatalogError ) as caught:
            bu_rebuild.rebuild( self.db_path, hasher=hasher )
        self.assertEqual( ( caught.exception.code, caught.exception.params ), ( 'catalog.unavailable', { 'reason': 'in_use' } ) )
        self.assertEqual( projection( self.db_path ), before )

    def test_a_file_newer_than_the_code_is_refused_and_left_alone( self ):
        db = sqlite3.connect( self.db_path, isolation_level=None )
        db.execute( 'PRAGMA user_version = %d'%( bu_store.LATEST_VERSION + 1 ) )
        db.close()
        with self.assertRaises( bu_store.CatalogError ) as caught:
            bu_rebuild.rebuild( self.db_path, hasher=hasher )
        self.assertEqual( caught.exception.code, 'catalog.too_new' )
        self.assertFalse( os.path.exists( self.db_path + bu_rebuild.NEW_SUFFIX ) )


class TestTool( _Library ):
    """tools/setup/catalog_rebuild.py: asks, logs, refuses while the catalog is open."""

    def tool( self, *argv ):
        import importlib.util
        import io
        repo = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
        spec = importlib.util.spec_from_file_location( 'catalog_rebuild_tool',
                                                       os.path.join( repo, 'tools', 'setup', 'catalog_rebuild.py' ) )
        module = importlib.util.module_from_spec( spec )
        spec.loader.exec_module( module )
        out = io.StringIO()
        code = module.main( [ '--catalog', self.db_path, '--log-file', os.path.join( self.root, 'rebuild.log' ),
                              '--console-level', 'error' ] + list( argv ), stdin=io.StringIO(), stdout=out )
        return code

    def test_yes_rebuilds_and_logs( self ):
        before = projection( self.db_path )
        self.assertEqual( self.tool( '--yes' ), 0 )
        self.assertEqual( projection( self.db_path ), before )
        self.assertTrue( os.path.exists( self.db_path + bu_rebuild.OLD_SUFFIX ) )
        with open( os.path.join( self.root, 'rebuild.log' ) ) as handle:
            self.assertIn( 'catalog rebuilt', handle.read() )

    def test_without_a_terminal_or_yes_nothing_changes( self ):
        self.assertEqual( self.tool(), 0 )
        self.assertFalse( os.path.exists( self.db_path + bu_rebuild.OLD_SUFFIX ) )

    def test_an_open_catalog_is_refused_with_exit_1( self ):
        store = bu_store.Store( self.db_path )
        self.addCleanup( store.close )
        self.assertEqual( self.tool( '--yes' ), 1 )


if __name__ == '__main__':
    unittest.main()
