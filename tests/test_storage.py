"""
test_storage.py - the Storage view's sizes and its clean-up, on a
scratch tree.

What each test prevents:
  - a size that counts the wrong files (a finished output as an
    unfinished render, an installed model as a leftover), or follows a
    link out of its folder
  - clean-up that removes something kept (outputs, models in use, the
    catalog, logs, stats), or a shared download cache
  - hours of detections, or resume points, cleared without confirm
  - clean-up of what a running job writes
  - a request that is refused partway, after removing some of it
"""

import os
import shutil
import tempfile
import unittest

import betautils_storage as bu_storage

from tests import protocol_schemas as schemas


def _write( path, size=10 ):
    os.makedirs( os.path.dirname( path ), exist_ok=True )
    with open( path, 'wb' ) as handle:
        handle.write( b'x' * size )


class _Tree( unittest.TestCase ):

    def setUp( self ):
        self.root = tempfile.mkdtemp( prefix='storage-' )
        self.addCleanup( shutil.rmtree, self.root, True )
        r = self.root
        self.dirs = {
            'detections': os.path.join( r, 'cache', 'vid_hashes' ),
            'transcodes': os.path.join( r, 'cache', 'transcode' ),
            'outputs': os.path.join( r, 'library', 'censored_vids' ),
            'models': os.path.join( r, 'data', 'model' ),
            'logs': os.path.join( r, 'state', 'logs' ),
            'shared': os.path.join( r, 'home', '.cache', 'uv' ),
        }
        _write( os.path.join( self.dirs['detections'], 'a', 'one.json' ), 100 )
        _write( os.path.join( self.dirs['detections'], 'b', 'two.json' ), 50 )
        _write( os.path.join( self.dirs['transcodes'], 't.mkv' ), 30 )
        _write( os.path.join( self.dirs['outputs'], 'clip-d1-c2-e3.mp4' ), 400 )
        _write( os.path.join( self.dirs['outputs'], 'clip-d1-c2-e3.video.part0001.mkv' ), 70 )
        _write( os.path.join( self.dirs['outputs'], 'clip-d1-c2-e3.tmp.mp4' ), 5 )
        _write( os.path.join( self.dirs['models'], '640m.onnx' ), 900 )
        _write( os.path.join( self.dirs['models'], '320n.onnx.part' ), 20 )
        _write( os.path.join( self.dirs['models'], 'old.onnx.bad' ), 25 )
        _write( os.path.join( self.dirs['logs'], 'betasuite.log' ), 60 )
        _write( os.path.join( self.dirs['shared'], 'wheel.whl' ), 800 )
        C = bu_storage.Category
        self.cats = [
            C( 'detections', 'cache', bu_storage.COSTLY, [ self.dirs['detections'] ] ),
            C( 'transcodes', 'cache', bu_storage.SAFE, [ self.dirs['transcodes'] ] ),
            C( 'unfinished_renders', 'outputs', bu_storage.RESUME, [ self.dirs['outputs'] ],
               match=bu_storage._is_partial_output ),
            C( 'outputs', 'outputs', bu_storage.KEPT, [ self.dirs['outputs'] ],
               match=lambda n: not bu_storage._is_partial_output( n ) ),
            C( 'model_leftovers', 'data', bu_storage.SAFE, [ self.dirs['models'] ],
               match=bu_storage._is_model_leftover ),
            C( 'models', 'data', bu_storage.KEPT, [ self.dirs['models'] ],
               match=lambda n: not bu_storage._is_model_leftover( n ) ),
            C( 'logs', 'state', bu_storage.KEPT, [ self.dirs['logs'] ] ),
            C( 'shared_download_cache', 'cache', bu_storage.SHARED, [ self.dirs['shared'] ] ),
            C( 'missing', 'cache', bu_storage.SAFE, [ os.path.join( r, 'nowhere' ) ] ),
        ]

    def sizes( self ):
        return { row['id']: ( row['bytes'], row['files'] ) for row in bu_storage.usage( self.cats ) }


class TestUsage( _Tree ):

    def test_each_category_counts_its_own_files( self ):
        sizes = self.sizes()
        self.assertEqual( sizes['detections'], ( 150, 2 ) )
        self.assertEqual( sizes['unfinished_renders'], ( 75, 2 ) )
        self.assertEqual( sizes['outputs'], ( 400, 1 ) )
        self.assertEqual( sizes['model_leftovers'], ( 45, 2 ) )
        self.assertEqual( sizes['models'], ( 900, 1 ) )
        self.assertEqual( sizes['missing'], ( 0, 0 ) )

    def test_a_single_file_counts_only_when_it_matches( self ):
        path = os.path.join( self.dirs['models'], '640m.onnx' )
        self.assertEqual( bu_storage.measure( path, bu_storage._is_model_leftover ), ( 0, 0 ) )
        self.assertEqual( bu_storage.measure( path ), ( 900, 1 ) )

    def test_rows_carry_kind_and_policy_and_no_paths( self ):
        for row in bu_storage.usage( self.cats ):
            self.assertEqual( set( row ), { 'id', 'kind', 'policy', 'bytes', 'files' } )

    @unittest.skipUnless( hasattr( os, 'symlink' ), 'needs symlinks' )
    def test_a_link_out_of_the_folder_is_not_followed( self ):
        outside = os.path.join( self.root, 'outside' )
        _write( os.path.join( outside, 'big.bin' ), 10000 )
        try:
            os.symlink( outside, os.path.join( self.dirs['transcodes'], 'link' ) )
        except OSError:
            self.skipTest( 'symlinks not allowed here' )
        self.assertEqual( self.sizes()['transcodes'], ( 30, 1 ) )
        bu_storage.clean( self.cats, [ 'transcodes' ] )
        self.assertTrue( os.path.exists( os.path.join( outside, 'big.bin' ) ) )

    def test_real_categories_list( self ):
        ids = [ cat.id for cat in bu_storage.categories( previous_cache='/x', shared_download_cache='/y' ) ]
        for expected in ( 'detections', 'shot_cuts', 'transcodes', 'look_previews', 'thumbnails',
                          'download_cache', 'unfinished_renders', 'model_leftovers', 'models', 'outputs',
                          'catalog', 'logs', 'stats', 'benchmarks', 'run_keys', 'previous_cache',
                          'shared_download_cache' ):
            self.assertIn( expected, ids )
        policies = { cat.id: cat.policy for cat in bu_storage.categories( shared_download_cache='/y' ) }
        for kept in ( 'outputs', 'models', 'catalog', 'logs', 'stats', 'run_keys' ):
            self.assertEqual( policies[kept], bu_storage.KEPT, kept )
        self.assertEqual( policies['detections'], bu_storage.COSTLY )
        # Benchmark results are tuning evidence nothing rebuilds: never cleared without a confirm.
        self.assertEqual( policies['benchmarks'], bu_storage.COSTLY )
        self.assertEqual( policies['shared_download_cache'], bu_storage.SHARED )


class TestClean( _Tree ):

    def test_safe_category_is_removed( self ):
        freed = bu_storage.clean( self.cats, [ 'transcodes', 'model_leftovers' ] )
        self.assertEqual( freed, [ { 'id': 'transcodes', 'bytes': 30, 'files': 1 },
                                   { 'id': 'model_leftovers', 'bytes': 45, 'files': 2 } ] )
        self.assertFalse( os.path.exists( self.dirs['transcodes'] ) )
        self.assertTrue( os.path.exists( os.path.join( self.dirs['models'], '640m.onnx' ) ) )

    def test_unfinished_renders_leave_finished_outputs( self ):
        bu_storage.clean( self.cats, [ 'unfinished_renders' ], confirm=True )
        self.assertEqual( os.listdir( self.dirs['outputs'] ), [ 'clip-d1-c2-e3.mp4' ] )

    def test_costly_and_resume_need_confirm( self ):
        for category in ( 'detections', 'unfinished_renders' ):
            with self.assertRaises( bu_storage.StorageError ) as caught:
                bu_storage.clean( self.cats, [ category ] )
            self.assertEqual( caught.exception.code, 'storage.needs_confirm' )
        self.assertEqual( self.sizes()['detections'], ( 150, 2 ) )
        bu_storage.clean( self.cats, [ 'detections' ], confirm=True )
        self.assertEqual( self.sizes()['detections'], ( 0, 0 ) )

    def test_kept_and_shared_are_never_removed( self ):
        for category in ( 'outputs', 'models', 'logs', 'shared_download_cache' ):
            with self.assertRaises( bu_storage.StorageError ) as caught:
                bu_storage.clean( self.cats, [ category ], confirm=True )
            self.assertEqual( caught.exception.code, 'storage.not_cleanable' )
        self.assertEqual( self.sizes()['shared_download_cache'], ( 800, 1 ) )

    def test_refused_while_a_job_writes_it( self ):
        with self.assertRaises( bu_storage.StorageError ) as caught:
            bu_storage.clean( self.cats, [ 'detections' ], confirm=True, busy=True )
        self.assertEqual( caught.exception.code, 'storage.busy' )
        bu_storage.clean( self.cats, [ 'model_leftovers' ], busy=True )   # not written by runs

    def test_nothing_removed_when_any_is_refused( self ):
        with self.assertRaises( bu_storage.StorageError ):
            bu_storage.clean( self.cats, [ 'transcodes', 'outputs' ] )
        self.assertEqual( self.sizes()['transcodes'], ( 30, 1 ) )

    def test_unknown_category( self ):
        with self.assertRaises( bu_storage.StorageError ) as caught:
            bu_storage.clean( self.cats, [ 'everything' ] )
        self.assertEqual( caught.exception.code, 'storage.unknown_category' )

    def test_missing_folder_frees_nothing( self ):
        self.assertEqual( bu_storage.clean( self.cats, [ 'missing' ] ), [ { 'id': 'missing', 'bytes': 0, 'files': 0 } ] )


class _Server:

    def __init__( self, busy=False ):
        import types
        self.jobs = types.SimpleNamespace( busy_job=lambda: 'job-1' if busy else None )
        self._logger = types.SimpleNamespace( info=lambda *a, **k: None )


class TestServe( _Tree ):

    def setUp( self ):
        super().setUp()
        from unittest import mock
        import betautils_serve_storage as bu_serve_storage
        self.serve = bu_serve_storage
        self.state = os.path.join( self.root, 'state' )
        patches = [
            mock.patch.object( bu_serve_storage, '_categories', lambda: self.cats ),
            mock.patch.object( bu_serve_storage, '_state_file', lambda name: os.path.join( self.state, name ) ),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup( patch.stop )

    def test_usage( self ):
        from unittest import mock
        with mock.patch.object( self.serve, 'note_cache_folder', lambda: None ):
            answer = self.serve.storage_usage( _Server(), {} )
        self.assertEqual( { row['id'] for row in answer['categories'] }, { cat.id for cat in self.cats } )
        self.assertIn( answer['cache_source'], ( 'variable', 'setting', 'default' ) )
        self.assertFalse( answer['busy'] )

    def test_clean_and_its_errors( self ):
        import betautils_protocol as bu_proto
        self.assertEqual( self.serve.storage_clean( _Server(), { 'categories': [ 'transcodes' ] } ),
                          { 'freed': [ { 'id': 'transcodes', 'bytes': 30, 'files': 1 } ] } )
        cases = ( ( { 'categories': [ 'detections' ] }, False, 'storage.needs_confirm' ),
                  ( { 'categories': [ 'outputs' ], 'confirm': True }, False, 'storage.not_cleanable' ),
                  ( { 'categories': [ 'detections' ], 'confirm': True }, True, 'storage.busy' ),
                  ( { 'categories': [ 'nope' ] }, False, 'storage.unknown_category' ),
                  ( { 'categories': [] }, False, 'command.bad_params' ),
                  ( { 'categories': 'transcodes' }, False, 'command.bad_params' ),
                  ( { 'categories': [ 'transcodes' ], 'confirm': 'yes' }, False, 'command.bad_params' ) )
        for data, busy, code in cases:
            with self.subTest( data=data ):
                with self.assertRaises( bu_proto.ProtocolError ) as caught:
                    self.serve.storage_clean( _Server( busy ), data )
                self.assertEqual( caught.exception.error['code'], code )

    def test_a_moved_cache_leaves_its_old_folder_listed( self ):
        old = os.path.join( self.root, 'old-cache' )
        new = os.path.join( self.root, 'new-cache' )
        os.makedirs( old )
        self.assertIsNone( self.serve.note_cache_folder( old ) )
        self.assertIsNone( self.serve.note_cache_folder( old ) )
        self.assertEqual( self.serve.note_cache_folder( new ), old )
        self.assertEqual( self.serve.previous_cache(), old )
        shutil.rmtree( old )
        self.assertIsNone( self.serve.previous_cache(), 'a folder deleted by hand is no longer listed' )

    def test_a_cache_moved_inside_the_old_one_is_not_listed( self ):
        old = os.path.join( self.root, 'cache' )
        os.makedirs( os.path.join( old, 'inner' ) )
        self.serve.note_cache_folder( old )
        self.assertIsNone( self.serve.note_cache_folder( os.path.join( old, 'inner' ) ) )

    def test_shared_download_cache_only_when_present( self ):
        home = os.path.join( self.root, 'home' )
        self.assertEqual( self.serve.shared_download_cache( { 'HOME': home } ), self.dirs['shared'] )
        self.assertIsNone( self.serve.shared_download_cache( { 'HOME': os.path.join( self.root, 'nobody' ) } ) )


@schemas.needs_jsonschema
class TestSchemas( _Tree ):

    def test_answers( self ):
        import betautils_hardware as bu_hardware
        schemas.validate_command_result( 'storage.usage', {
            'categories': bu_storage.usage( self.cats ), 'disks': bu_hardware.disks( { 'cache': self.root } ),
            'cache_source': 'setting', 'busy': True } )
        schemas.validate_command_result( 'storage.clean',
                                         { 'freed': bu_storage.clean( self.cats, [ 'transcodes' ] ) } )
        schemas.validate_command_data( 'storage.clean', { 'categories': [ 'detections' ], 'confirm': True } )


if __name__ == '__main__':
    unittest.main()
