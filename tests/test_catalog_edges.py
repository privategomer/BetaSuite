"""
Edges of the catalog that the first round of the mutation check showed
no test covered: a same-size rewrite, a filesystem without inodes, a
touch without new bytes, the widest file ids, the deepest folder for a
chosen file, a file the walk skipped, the last run's type, a run on a
file that is gone, the blur itself, and the memo's own temporary file.
"""

import os
import shutil
import tempfile
import time
import unittest
from unittest import mock

import betaconst
import betautils_catalog as bu_catalog
import betautils_catalog_history as bu_history
import betautils_catalog_index as bu_index
import betautils_catalog_store as bu_store
import betautils_catalog_types as bu_types
import betautils_hash as bu_hash
import betautils_run_record as bu_run_record


def hasher( path ):
    return bu_hash.md5_for_file( path, 16, use_memo=False )


class _Catalog( unittest.TestCase ):
    def setUp( self ):
        self.root = tempfile.mkdtemp( prefix='catalog-edges-' )
        self.addCleanup( shutil.rmtree, self.root, ignore_errors=True )
        self.watched = os.path.join( self.root, 'videos' )
        os.makedirs( self.watched )
        self.now = [ 1000.0 ]
        self.catalog = bu_catalog.Catalog( path=os.path.join( self.root, 'c.sqlite' ), seed_folders=[], settle=0.0,
                                           hasher=hasher, thumbnails=False, clock=lambda: self.now[0] )
        self.addCleanup( self.catalog.store.close )

    def write( self, relative, content ):
        path = os.path.join( self.watched, relative )
        os.makedirs( os.path.dirname( path ), exist_ok=True )
        with open( path, 'wb' ) as handle:
            handle.write( content )
        return path

    def index_all( self ):
        self.catalog.scanner.pass_now()
        self.catalog.scanner.hash_ready()

    def hashes( self ):
        return sorted( item['item'] for item in self.catalog.list()['items'] )


class TestSameSizeRewrite( _Catalog ):
    def test_new_bytes_of_the_same_size_at_a_known_path_are_hashed_again( self ):
        self.catalog.add_folder( self.watched )
        path = self.write( 'a.mp4', b'a' * 4096 )
        self.index_all()
        before = self.hashes()
        stat = os.stat( path )
        with open( path, 'wb' ) as handle:
            handle.write( b'b' * 4096 )
        os.utime( path, ns=( stat.st_atime_ns, stat.st_mtime_ns + 10**9 ) )
        self.index_all()
        self.assertNotEqual( self.hashes(), before )
        self.assertEqual( self.hashes(), [ hasher( path ) ] )


class TestNoInodes( unittest.TestCase ):
    def test_a_zero_inode_never_makes_two_files_one( self ):
        one = bu_index.Stat( 10, 20, 1, 0 )
        other = bu_index.Stat( 10, 20, 1, 0 )
        self.assertFalse( one.same_inode( other ) )
        self.assertTrue( bu_index.Stat( 10, 20, 1, 7 ).same_inode( bu_index.Stat( 10, 20, 1, 7 ) ) )


class TestTouched( _Catalog ):
    def test_a_touch_without_new_bytes_is_unchanged( self ):
        path = self.write( 't.mp4', b't' * 2048 )
        digest = hasher( path )
        index = self.catalog.index
        index.record_hash( path, bu_index.Stat.of( os.stat( path ) ), digest )
        os.utime( path, ns=( time.time_ns(), time.time_ns() + 5 * 10**9 ) )
        outcome, _touched = index.record_hash( path, bu_index.Stat.of( os.stat( path ) ), digest )
        self.assertEqual( outcome, 'unchanged' )


class TestWidestIds( unittest.TestCase ):
    def test_ids_just_under_two_to_the_65_still_fit( self ):
        for value in ( ( 1 << 65 ) - 1, ( 1 << 64 ) + ( 1 << 63 ) ):
            self.assertTrue( -( 1 << 63 ) <= bu_index.signed64( value ) < 1 << 63, value )


class TestDeepestFolder( _Catalog ):
    def test_a_chosen_file_belongs_to_the_deepest_watched_folder( self ):
        self.catalog.add_folder( self.watched )
        inner_path = os.path.join( self.watched, 'inner' )
        os.makedirs( inner_path )
        inner, _ = self.catalog.add_folder( inner_path )
        path = self.write( 'inner/x.mp4', b'x' * 1500 )
        self.catalog.add_files( [ path ] )
        self.catalog.scanner.hash_ready()
        [ location ] = self.catalog.get( hasher( path ) )['locations']
        self.assertEqual( location['folder'], inner['folder'] )


class TestSkippedIsNotMissing( _Catalog ):
    def test_a_file_the_walk_skips_but_that_is_still_there_is_not_missing( self ):
        self.catalog.add_folder( self.watched )
        path = self.write( 'z.mp4', b'z' * 3000 )
        self.index_all()
        with open( path, 'wb' ):
            pass   # now empty: the walk skips it, but it has not gone anywhere
        self.catalog.scanner.pass_now()
        self.assertEqual( self.catalog.list()['items'][0]['availability'], 'present' )


class TestLastRunType( _Catalog ):
    def test_with_nothing_declared_the_library_shows_the_last_runs_type( self ):
        path = self.write( 'm.mp4', b'm' * 2500 )
        digest = hasher( path )
        run = bu_run_record.RunRecord( path )
        run.hash = digest
        run.profile, run.rule, run.rule_params = 'quick_cut', 'measured', { 'signal': 'short_shot_fraction' }
        self.catalog.history.record( run.end( 'processed' ) )
        self.assertEqual( self.catalog.list()['items'][0]['type'],
                          { 'profile': 'quick_cut', 'rule': 'measured', 'params': { 'signal': 'short_shot_fraction' },
                            'when': 'last_run' } )
        self.assertEqual( bu_types.shown_type( '/x.mp4', last=None )['when'], 'at_run' )


class TestRunOnAFileThatIsGone( _Catalog ):
    def test_a_run_on_an_unknown_file_that_is_gone_records_nothing( self ):
        run = bu_run_record.RunRecord( os.path.join( self.watched, 'gone.mp4' ) )
        run.hash = 'c' * 16
        self.assertEqual( self.catalog.history.record( run.end( 'processed' ) ), set() )
        self.assertEqual( self.catalog.store.read( 'SELECT COUNT(*) FROM runs' )[0][0], 0 )


class TestTheBlur( unittest.TestCase ):
    def test_the_blur_takes_out_what_shrinking_leaves( self ):
        import cv2
        import numpy
        import betautils_catalog_thumbs as bu_thumbs
        rng = numpy.random.default_rng( 3 )
        frame = ( rng.integers( 0, 2, size=( 360, 640, 1 ) ) * 255 ).astype( numpy.uint8 ).repeat( 3, axis=2 )
        frame[:, ::2] = 255 - frame[:, ::2]   # fine detail everywhere
        blurred = bu_thumbs.blur( frame )
        shrunk = cv2.resize( frame, ( blurred.shape[1], blurred.shape[0] ), interpolation=cv2.INTER_AREA )
        sharpness = lambda image: cv2.Laplacian( cv2.cvtColor( image, cv2.COLOR_BGR2GRAY ), cv2.CV_64F ).var()
        self.assertLess( sharpness( blurred ), sharpness( shrunk ) * 0.8 )


class TestMemoTemporaryFile( unittest.TestCase ):
    def test_each_process_writes_the_memo_through_its_own_temporary_file( self ):
        root = tempfile.mkdtemp( prefix='memo-tmp-' )
        self.addCleanup( shutil.rmtree, root, ignore_errors=True )
        saved = ( betaconst.file_hash_cache_path, bu_hash._file_hash_memo, bu_hash._file_hash_memo_dirty )
        self.addCleanup( lambda: setattr_all( saved ) )
        betaconst.file_hash_cache_path = os.path.join( root, 'file_hashes.json' )
        bu_hash._file_hash_memo, bu_hash._file_hash_memo_dirty = { 'k': 'v' }, True
        with mock.patch.object( bu_hash, '_write_json_plain_via' ) as write:
            bu_hash.flush_file_hash_memo()
        temporary = write.call_args[0][2]
        self.assertIn( '.%d.'%( os.getpid() ), os.path.basename( temporary ) )
        self.assertNotEqual( temporary, betaconst.file_hash_cache_path + '.tmp' )


def setattr_all( saved ):
    betaconst.file_hash_cache_path, bu_hash._file_hash_memo, bu_hash._file_hash_memo_dirty = saved


if __name__ == '__main__':
    unittest.main()
