"""
test_pic_hash_path_unified.py - regression coverage for the still-photo
side of the cache-path unification (2026-09-17): betastare.py's
_raw_boxes_for_size used to build its own independent copy of the
picture-cache path formula inline, rather than calling a shared
implementation (unlike betatv.py's video-cache path, which already had
betautils_cache_paths.box_hash_path_for as its one shared implementation
alongside a betatv.py-internal wrapper, kept in sync only by
test_cache_path_duplicates_stay_in_sync.py). That's exactly the same
"independently duplicated formula" shape that module's own docstring
says already caused a real bug once on the video side (see
betautils_cache_paths.py's module docstring) - so the photo side got
folded in too: betautils_cache_paths.pic_hash_path_for is now the one
implementation, and betastare._raw_boxes_for_size calls it directly.

This test proves the unification actually happened (betastare really
calls the shared function, not a second copy of the formula) rather
than just proving the two formulas currently produce the same string
by coincidence - a real code-path assertion, not a string comparison.
"""

import os
import tempfile
import shutil
import unittest
from unittest import mock

import betaconfig
import betaconst

import betastare
import betautils_cache_paths as bu_cache


class TestBetastareCallsSharedPicHashPathFor( unittest.TestCase ):
    """
    Confirms betastare._raw_boxes_for_size resolves its cache path via
    betautils_cache_paths.pic_hash_path_for (the shared implementation)
    rather than building the path itself inline - patches
    pic_hash_path_for and checks it was actually called with the
    expected args, so this fails loudly if betastare.py ever regresses
    back to an inline duplicate formula.
    """

    def setUp( self ):
        self._tmpdir = tempfile.mkdtemp()
        self._orig_cwd = os.getcwd()
        self._work_subdir = os.path.join( self._tmpdir, 'BetaSuite-env' )
        os.makedirs( self._work_subdir, exist_ok=True )
        os.chdir( self._work_subdir )

        self._had_attr = hasattr( betaconfig, 'detector_backend' )
        if self._had_attr:
            self._original = betaconfig.detector_backend
        betaconfig.detector_backend = { 'selected': 'nudenet_v3', 'retinanet_v2': {}, 'nudenet_v3': {} }
        betaconfig.global_min_prob = 0.2

    def tearDown( self ):
        os.chdir( self._orig_cwd )
        shutil.rmtree( self._tmpdir, ignore_errors=True )
        if self._had_attr:
            betaconfig.detector_backend = self._original
        elif hasattr( betaconfig, 'detector_backend' ):
            del betaconfig.detector_backend

    def test_raw_boxes_for_size_calls_shared_pic_hash_path_for( self ):
        import numpy as np

        fake_image = np.zeros( (10, 10, 3), dtype='uint8' )
        expected_path = bu_cache.pic_hash_path_for( 'fakehash', 640, 0.2, 'nudenet_v3' )

        with mock.patch.object( bu_cache, 'pic_hash_path_for', wraps=bu_cache.pic_hash_path_for ) as spy:
            # patched onto the bu_cache module object betastare imported,
            # so betastare.bu_cache.pic_hash_path_for(...) hits the spy
            betastare.bu_cache.pic_hash_path_for = spy
            try:
                with mock.patch.object( betastare.bu_detector, 'get_detector' ) as mock_get_detector:
                    mock_get_detector.return_value.raw_boxes_for_img.return_value = []
                    betastare._raw_boxes_for_size( fake_image, 640, session=None, image_hash='fakehash' )
            finally:
                betastare.bu_cache.pic_hash_path_for = bu_cache.pic_hash_path_for

        spy.assert_called_once_with( 'fakehash', 640, 0.2, 'nudenet_v3' )
        self.assertIn( 'fakehash-nudenet_v3-%s-640-0.200-d'%(betaconst.picture_saved_box_version),
                       expected_path )
        self.assertTrue( expected_path.endswith( '.gz' ) )


class TestPicHashPathForFormula( unittest.TestCase ):
    """
    Direct coverage of betautils_cache_paths.pic_hash_path_for's own
    formula/shape - independent of betastare.py, since standalone tools
    may want to call this directly too (mirroring box_hash_path_for's
    own use by tools/analysis, tools/tuning).
    """

    def test_backend_name_is_baked_into_the_path( self ):
        retinanet_path = bu_cache.pic_hash_path_for( 'abc123', 640, 0.1, 'retinanet_v2' )
        nudenet_path = bu_cache.pic_hash_path_for( 'abc123', 640, 0.1, 'nudenet_v3' )
        self.assertNotEqual( retinanet_path, nudenet_path )
        self.assertIn( 'retinanet_v2', retinanet_path )
        self.assertIn( 'nudenet_v3', nudenet_path )

    def test_no_fps_or_preview_suffix_params( self ):
        # sanity: still images have no frame rate/preview-slice concept,
        # unlike box_hash_path_for's video formula - this just confirms
        # the signature stayed narrow (4 args) rather than picking up
        # video-only params by copy-paste from box_hash_path_for
        import inspect
        sig = inspect.signature( bu_cache.pic_hash_path_for )
        self.assertEqual( list( sig.parameters.keys() ),
            [ 'image_hash', 'size', 'min_prob', 'backend_name', 'det_key' ] )
        # det_key is optional: callers inside a run let it resolve from
        # the current config, tools pin it explicitly.
        self.assertIs( sig.parameters['det_key'].default, None )

    def test_lands_under_pic_hash_dir( self ):
        path = bu_cache.pic_hash_path_for( 'abc123', 640, 0.1, 'retinanet_v2' )
        self.assertTrue( path.startswith( bu_cache.PIC_HASH_DIR ) )


if __name__ == '__main__':
    unittest.main()
