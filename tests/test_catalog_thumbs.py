"""
Thumbnails under the privacy rule: only a source's blurred thumbnail is
ever written; reveal decodes the clear frame in memory and writes
nothing; a censored output's thumbnail is stored like any other.

The proof that matters: after a real run (stub detector) and every kind
of thumbnail, every file under the test roots that is an image is read
back, and no image of a source is larger than the blurred size.
"""

import base64
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest

import betautils_catalog as bu_catalog
import betautils_catalog_index as bu_index
import betautils_catalog_store as bu_store
import betautils_catalog_thumbs as bu_thumbs
import betautils_hash as bu_hash
import betautils_paths as bu_paths

import tests
from tests.test_betatv_end_to_end import BetaTvHarness, _build_clip, _ffmpeg_available

IMAGE_MAGIC = ( b'\x89PNG', b'\xff\xd8\xff', b'GIF8', b'BM', b'II*\x00', b'MM\x00*' )


def _is_image( path ):
    try:
        with open( path, 'rb' ) as handle:
            head = handle.read( 12 )
    except OSError:
        return False
    return head.startswith( IMAGE_MAGIC ) or ( head[:4] == b'RIFF' and head[8:12] == b'WEBP' )


def _size_of( path ):
    import cv2
    image = cv2.imread( path, cv2.IMREAD_UNCHANGED )
    return None if image is None else image.shape[:2]


class _Writes:
    """Every file this process opens for writing while active (an audit hook)."""

    _lock = threading.Lock()
    _active = None
    _installed = False

    @classmethod
    def install( cls ):
        if not cls._installed:
            sys.addaudithook( cls._hook )
            cls._installed = True

    @classmethod
    def _hook( cls, event, args ):
        active = cls._active
        if active is None:
            return
        if event == 'open' and len( args ) >= 2:
            path, mode = args[0], args[1]
            flags = args[2] if len( args ) > 2 else 0
            writing = ( isinstance( mode, str ) and any( letter in mode for letter in 'wax+' ) ) or (
                isinstance( flags, int ) and flags & ( os.O_WRONLY | os.O_RDWR | os.O_CREAT ) )
            # The null device (a child's stdin=DEVNULL) is not a file on disk.
            if writing and isinstance( path, ( str, bytes, os.PathLike ) ) and os.fsdecode( path ) != os.devnull:
                active['opened'].append( os.fsdecode( path ) )
        elif event in ( 'os.rename', 'os.replace' ):
            active['opened'].append( os.fsdecode( args[1] ) )
        elif event == 'subprocess.Popen':
            active['commands'].append( list( args[1] ) )

    def __enter__( self ):
        self.install()
        self.record = { 'opened': [], 'commands': [] }
        type( self )._active = self.record
        return self.record

    def __exit__( self, *_exc ):
        type( self )._active = None
        return False


@unittest.skipUnless( _ffmpeg_available(), "ffmpeg is needed to build the test clips" )
class TestThumbnails( unittest.TestCase ):
    def setUp( self ):
        self.root = tempfile.mkdtemp( prefix='catalog-thumbs-' )
        self.addCleanup( shutil.rmtree, self.root, ignore_errors=True )
        self.catalog = bu_catalog.Catalog( path=os.path.join( self.root, 'c.sqlite' ), seed_folders=[],
                                           thumbnails=False )
        self.addCleanup( self.catalog.store.close )
        self.clip = os.path.join( self.root, 'videos', 'clip.mp4' )
        _build_clip( self.clip, width=640, height=360 )
        self.digest = bu_hash.md5_for_file( self.clip, 16, use_memo=False )
        self.catalog.index.record_hash( self.clip, bu_index.Stat.of( os.stat( self.clip ) ), self.digest )
        self.addCleanup( bu_thumbs.forget, self.digest )

    def decode( self, answer ):
        import cv2
        import numpy
        data = base64.b64decode( answer['data'] )
        return cv2.imdecode( numpy.frombuffer( data, dtype=numpy.uint8 ), cv2.IMREAD_COLOR )

    def test_the_blurred_thumbnail_is_tiny_and_cached_in_the_cache_folder( self ):
        first = self.catalog.thumbnail( self.digest )
        self.assertEqual( ( first['mime'], first['cached'] ), ( 'image/png', False ) )
        self.assertLessEqual( max( first['width'], first['height'] ), bu_thumbs.BLUR_SIZE )
        path = bu_thumbs.blurred_path( self.digest )
        self.assertTrue( bu_index.is_within( path, bu_paths.kind_dir( 'cache' ) ) )
        self.assertLessEqual( max( _size_of( path ) ), bu_thumbs.BLUR_SIZE )
        again = self.catalog.thumbnail( self.digest )
        self.assertTrue( again['cached'] )
        self.assertEqual( again['data'], first['data'] )

    def test_the_blurred_thumbnail_keeps_no_detail( self ):
        import cv2
        image = self.decode( self.catalog.thumbnail( self.digest ) )
        clear = self.decode( self.catalog.thumbnail( self.digest, reveal=True ) )
        sharpness = lambda picture: cv2.Laplacian( cv2.cvtColor( picture, cv2.COLOR_BGR2GRAY ), cv2.CV_64F ).var()
        upscaled = cv2.resize( image, ( clear.shape[1], clear.shape[0] ), interpolation=cv2.INTER_LINEAR )
        self.assertLess( sharpness( upscaled ), sharpness( clear ) / 20 )

    def test_reveal_writes_nothing_and_ffmpeg_writes_to_a_pipe( self ):
        with _Writes() as record:
            answer = self.catalog.thumbnail( self.digest, reveal=True )
        self.assertEqual( record['opened'], [] )
        self.assertTrue( record['commands'] )
        for command in record['commands']:
            self.assertEqual( command[-1], 'pipe:1', command )
        self.assertEqual( answer['mime'], 'image/jpeg' )
        self.assertEqual( answer['width'], bu_thumbs.REVEAL_SIZE )
        self.assertFalse( answer['cached'] )
        self.assertFalse( os.path.exists( bu_thumbs.blurred_path( self.digest ) ), 'reveal does not even cache the blur' )

    def test_making_the_blurred_one_writes_only_the_blurred_file( self ):
        with _Writes() as record:
            self.catalog.thumbnail( self.digest )
        written = { os.path.normcase( os.path.abspath( path ) ) for path in record['opened'] }
        target = os.path.normcase( os.path.abspath( bu_thumbs.blurred_path( self.digest ) ) )
        self.assertEqual( { path for path in written if not path.startswith( target ) }, set() )

    def test_an_item_with_no_readable_file_is_missing( self ):
        os.remove( self.clip )
        self.catalog.index.mark( [ self.clip ], bu_index.MISSING )
        with self.assertRaises( bu_store.CatalogError ) as caught:
            self.catalog.thumbnail( self.digest )
        self.assertEqual( ( caught.exception.code, caught.exception.params ), ( 'catalog.missing', { 'item': self.digest } ) )

    def test_a_file_that_is_not_a_video_fails_as_not_video( self ):
        path = os.path.join( self.root, 'videos', 'notes.mp4' )
        with open( path, 'wb' ) as handle:
            handle.write( b'no frames here' * 100 )
        digest = bu_hash.md5_for_file( path, 16, use_memo=False )
        self.catalog.index.record_hash( path, bu_index.Stat.of( os.stat( path ) ), digest )
        with self.assertRaises( bu_store.CatalogError ) as caught:
            self.catalog.thumbnail( digest )
        self.assertEqual( caught.exception.params, { 'item': digest, 'reason': 'not_video' } )
        self.assertFalse( os.path.exists( bu_thumbs.blurred_path( digest ) ) )

    def test_forget_removes_the_blurred_thumbnail( self ):
        self.catalog.thumbnail( self.digest )
        self.catalog.forget( self.digest )
        self.assertFalse( os.path.exists( bu_thumbs.blurred_path( self.digest ) ) )


@unittest.skipUnless( _ffmpeg_available(), "ffmpeg is needed to build the test clips" )
class TestNewItemsGetTheirBlurredThumbnail( unittest.TestCase ):
    def test_the_scanner_makes_it_after_indexing_and_not_while_busy( self ):
        root = tempfile.mkdtemp( prefix='catalog-thumbs-scan-' )
        self.addCleanup( shutil.rmtree, root, ignore_errors=True )
        busy = [ True ]
        catalog = bu_catalog.Catalog( path=os.path.join( root, 'c.sqlite' ), seed_folders=[], settle=0.0,
                                      busy=lambda: busy[0],
                                      hasher=lambda path: bu_hash.md5_for_file( path, 16, use_memo=False ) )
        self.addCleanup( catalog.store.close )
        clip = os.path.join( root, 'videos', 'c.mp4' )
        _build_clip( clip )
        digest = bu_hash.md5_for_file( clip, 16, use_memo=False )
        self.addCleanup( bu_thumbs.forget, digest )
        catalog.add_folder( os.path.join( root, 'videos' ) )
        catalog.scanner.pass_now()
        catalog.scanner.hash_ready()
        self.assertEqual( catalog.list()['items'], [], 'nothing is hashed while busy' )
        busy[0] = False
        catalog.scanner.hash_ready()
        self.assertTrue( os.path.exists( bu_thumbs.blurred_path( digest ) ) )


class TestNoClearImageOfASourceIsWritten( BetaTvHarness ):
    """
    A real run, then every kind of thumbnail; then every file under the
    test roots (the OS folders, the run's tree) that is an image must be
    no larger than the blurred size, or a censored output's thumbnail.
    """

    def test_scan_every_file_after_a_run_and_thumbnails( self ):
        clip = os.path.join( self.uncensored, 'clip.mp4' )
        _build_clip( clip, width=640, height=360 )
        self.assertEqual( self.run_one(), 'processed' )
        catalog = bu_catalog.Catalog( path=os.path.join( self.tmpdir, 'catalog.sqlite' ), seed_folders=[],
                                      thumbnails=False )
        self.addCleanup( catalog.store.close )
        digest = bu_hash.md5_for_file( clip, 16, use_memo=False )
        self.addCleanup( bu_thumbs.forget, digest )
        catalog.index.record_hash( clip, bu_index.Stat.of( os.stat( clip ) ), digest )
        output = os.path.join( self.censored, self.outputs()[0] )
        catalog.index.add_output( digest, output, {
            'preview': False, 'detection_key': 'd', 'censor_key': 'c', 'encode_key': 'e' } )
        catalog.thumbnail( digest )
        catalog.thumbnail( digest, reveal=True )
        catalog.thumbnail( digest, of='output' )
        self.addCleanup( lambda: os.path.exists( bu_thumbs.output_path( output ) )
                         and os.remove( bu_thumbs.output_path( output ) ) )

        images, checked = [], 0
        for top in ( tests.TEST_OUTPUT_ROOT, self.tmpdir ):
            for folder, _dirs, names in os.walk( top ):
                for name in names:
                    path = os.path.join( folder, name )
                    checked += 1
                    if _is_image( path ):
                        images.append( path )
        self.assertGreater( checked, 5, 'the scan looked at the files of the run' )
        self.assertIn( os.path.normcase( bu_thumbs.blurred_path( digest ) ),
                       [ os.path.normcase( path ) for path in images ], 'the scan found the blurred thumbnail' )
        for path in images:
            if os.path.normcase( path ) == os.path.normcase( bu_thumbs.output_path( output ) ):
                continue   # a censored output's thumbnail, stored clear by design
            size = _size_of( path )
            self.assertIsNotNone( size, path )
            self.assertLessEqual( max( size ), bu_thumbs.BLUR_SIZE, 'a clear image was written: %s'%( path, ) )


if __name__ == '__main__':
    unittest.main()
