"""
test_file_hash_memo.py - a moved file is not re-read, a changed one is.

WHY
    The memo was keyed on the file's path, so every move cost a full
    re-read of a multi-GB file: sorting ~70 files into source folders
    made the next betabench run spend 4m39s hashing. The workflow moves
    each file at least twice more (staged into uncensored_vids/, then
    archived to completed/). And only betatv and betastare persisted the
    memo, so analysis tools re-hashed archived files on every run.
"""

import os
import shutil
import tempfile
import unittest
from unittest import mock

import betaconst
import betautils_cache_paths as bu_cache
import betautils_hash as bu_hash


class MemoHarness( unittest.TestCase ):

    def setUp( self ):
        self.tmp = tempfile.mkdtemp( prefix='betasuite-memo-' )
        self._saved = ( betaconst.file_hash_cache_path, bu_hash._file_hash_memo,
                        bu_hash._file_hash_memo_dirty )
        betaconst.file_hash_cache_path = os.path.join( self.tmp, 'file_hashes.json' )
        bu_hash._file_hash_memo = None
        bu_hash._file_hash_memo_dirty = False
        self.path = os.path.join( self.tmp, 'a', 'video.mkv' )
        os.makedirs( os.path.dirname( self.path ) )
        with open( self.path, 'wb' ) as handle:
            handle.write( b'frame data ' * 1000 )

    def tearDown( self ):
        ( betaconst.file_hash_cache_path, bu_hash._file_hash_memo,
          bu_hash._file_hash_memo_dirty ) = self._saved
        shutil.rmtree( self.tmp, ignore_errors=True )

    def hash_without_reading( self, path ):
        # Fail only on opening the VIDEO; the memo file itself must still
        # load, or a "next process" test could never hit.
        real_open = open

        def guarded_open( name, *args, **kwargs ):
            if os.path.realpath( name ) == os.path.realpath( path ):
                raise AssertionError( 're-read %s'%( path, ) )
            return real_open( name, *args, **kwargs )

        with mock.patch( 'betautils_hash.open', create=True, side_effect=guarded_open ):
            return bu_hash.md5_for_file( path, 16 )

    def fresh_process( self ):
        bu_hash.flush_file_hash_memo()
        bu_hash._file_hash_memo = None


class TestMovedFiles( MemoHarness ):

    def test_precondition_an_unseen_file_is_read( self ):
        with self.assertRaises( AssertionError ):
            self.hash_without_reading( self.path )

    def test_a_moved_file_is_not_re_read( self ):
        original = bu_hash.md5_for_file( self.path, 16 )
        moved = os.path.join( self.tmp, 'b', 'video.mkv' )
        os.makedirs( os.path.dirname( moved ) )
        os.rename( self.path, moved )
        self.assertEqual( self.hash_without_reading( moved ), original )

    def test_a_moved_file_is_not_re_read_by_the_next_process_either( self ):
        original = bu_hash.md5_for_file( self.path, 16 )
        self.fresh_process()
        moved = os.path.join( self.tmp, 'moved.mkv' )
        os.rename( self.path, moved )
        self.assertEqual( self.hash_without_reading( moved ), original )

    def test_a_copy_is_re_read( self ):
        bu_hash.md5_for_file( self.path, 16 )
        copied = os.path.join( self.tmp, 'copy.mkv' )
        shutil.copy( self.path, copied )
        with self.assertRaises( AssertionError ):
            self.hash_without_reading( copied )

    def test_changed_content_gets_a_new_hash( self ):
        original = bu_hash.md5_for_file( self.path, 16 )
        stat = os.stat( self.path )
        with open( self.path, 'r+b' ) as handle:
            handle.write( b'DIFFERENT' )
        os.utime( self.path, ns=( stat.st_atime_ns, stat.st_mtime_ns + 1000 ) )
        self.assertNotEqual( bu_hash.md5_for_file( self.path, 16 ), original )

    def test_the_hash_length_is_part_of_the_identity( self ):
        short = bu_hash.md5_for_file( self.path, 8 )
        self.assertEqual( len( bu_hash.md5_for_file( self.path, 16 ) ), 16 )
        self.assertEqual( len( short ), 8 )


class TestAnalysisToolsPersistTheMemo( MemoHarness ):

    def test_build_hash_to_video_path_flushes( self ):
        target = bu_hash.md5_for_file( self.path, 16, use_memo=False )
        saved = ( betaconst.video_path_uncensored, betaconst.video_path_source_backup )
        try:
            betaconst.video_path_uncensored = os.path.join( self.tmp, 'a' ) + os.sep
            betaconst.video_path_source_backup = os.path.join( self.tmp, 'none' ) + os.sep
            self.assertEqual( bu_cache.build_hash_to_video_path( [ target ] ),
                              { target: self.path } )
        finally:
            betaconst.video_path_uncensored, betaconst.video_path_source_backup = saved
        bu_hash._file_hash_memo = None
        self.assertEqual( self.hash_without_reading( self.path ), target )


class TestTwoProcessesShareTheMemo( MemoHarness ):
    """
    The engine's catalog scanner and a command-line run can each hold
    the memo at once. Neither flush may drop what the other wrote.
    """

    def other_process_flushes( self, path ):
        mine = ( bu_hash._file_hash_memo, bu_hash._file_hash_memo_dirty )
        bu_hash._file_hash_memo, bu_hash._file_hash_memo_dirty = None, False
        digest = bu_hash.md5_for_file( path, 16 )
        bu_hash.flush_file_hash_memo()
        bu_hash._file_hash_memo, bu_hash._file_hash_memo_dirty = mine
        return digest

    def test_a_flush_keeps_entries_another_process_flushed_meanwhile( self ):
        other = os.path.join( self.tmp, 'b', 'other.mkv' )
        os.makedirs( os.path.dirname( other ) )
        with open( other, 'wb' ) as handle:
            handle.write( b'other frames ' * 1000 )
        bu_hash._load_file_hash_memo()                 # this process loads first
        other_digest = self.other_process_flushes( other )
        mine = bu_hash.md5_for_file( self.path, 16 )
        bu_hash.flush_file_hash_memo()
        bu_hash._file_hash_memo = None
        self.assertEqual( self.hash_without_reading( other ), other_digest )
        self.assertEqual( self.hash_without_reading( self.path ), mine )

    def test_a_flush_leaves_no_temporary_file( self ):
        bu_hash.md5_for_file( self.path, 16 )
        bu_hash.flush_file_hash_memo()
        self.assertEqual( sorted( os.listdir( self.tmp ) ), [ 'a', 'file_hashes.json' ] )

    def test_a_corrupt_memo_on_disk_is_replaced_not_merged( self ):
        with open( betaconst.file_hash_cache_path, 'w' ) as handle:
            handle.write( '{ not json' )
        digest = bu_hash.md5_for_file( self.path, 16 )
        bu_hash.flush_file_hash_memo()
        bu_hash._file_hash_memo = None
        self.assertEqual( self.hash_without_reading( self.path ), digest )


if __name__ == '__main__':
    unittest.main()
