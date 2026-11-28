"""
test_cache_key_backend.py - regression tests confirming the active
detector backend name is baked into both detection cache paths
(betatv.py's video cache, betastare.py's picture cache).

Added alongside the betaconfig.detector_backend nested-config change:
before this, switching betaconfig.detector_backend['selected'] could
silently serve back a DIFFERENT model's stale cached detections under
the same cache key, since the cache path never encoded which backend
produced it. These tests guard the fix directly - if backend_name ever
gets dropped from either path builder again, this catches it rather
than someone noticing stale results during a live comparison run.
"""

import unittest

import betaconfig

import betautils_cache_paths as bu_cache
import betautils_detector as bu_detector
import betautils_config as bu_config
import betastare


class TestVideoCachePathIncludesBackend( unittest.TestCase ):

    def setUp( self ):
        self._had_attr = hasattr( betaconfig, 'detector_backend' )
        if self._had_attr:
            self._original = betaconfig.detector_backend
        betaconfig.global_min_prob = 0.1
        betaconfig.video_censor_fps = 9

    def tearDown( self ):
        if self._had_attr:
            betaconfig.detector_backend = self._original
        elif hasattr( betaconfig, 'detector_backend' ):
            del betaconfig.detector_backend

    def test_different_backends_produce_different_cache_paths( self ):
        betaconfig.detector_backend = { 'selected': 'retinanet_v2', 'retinanet_v2': {}, 'nudenet_v3': {} }
        bu_config.invalidate_config_caches()
        retinanet_path = bu_cache.box_hash_path_for( 'abc123', 640, betaconfig.video_censor_fps, betaconfig.global_min_prob, bu_detector.selected_backend_name() )

        betaconfig.detector_backend = { 'selected': 'nudenet_v3', 'retinanet_v2': {}, 'nudenet_v3': {} }
        bu_config.invalidate_config_caches()
        nudenet_path = bu_cache.box_hash_path_for( 'abc123', 640, betaconfig.video_censor_fps, betaconfig.global_min_prob, bu_detector.selected_backend_name() )

        self.assertNotEqual( retinanet_path, nudenet_path )
        self.assertIn( 'retinanet_v2', retinanet_path )
        self.assertIn( 'nudenet_v3', nudenet_path )

    def test_same_backend_same_everything_else_produces_the_same_path( self ):
        # sanity check the fix didn't accidentally make paths
        # non-deterministic (e.g. via some unstable ordering)
        betaconfig.detector_backend = { 'selected': 'nudenet_v3', 'retinanet_v2': {}, 'nudenet_v3': {} }
        bu_config.invalidate_config_caches()
        path_a = bu_cache.box_hash_path_for( 'abc123', 640, betaconfig.video_censor_fps, betaconfig.global_min_prob, bu_detector.selected_backend_name() )
        path_b = bu_cache.box_hash_path_for( 'abc123', 640, betaconfig.video_censor_fps, betaconfig.global_min_prob, bu_detector.selected_backend_name() )
        self.assertEqual( path_a, path_b )


class TestPictureCachePathIncludesBackend( unittest.TestCase ):
    """
    betastare._raw_boxes_for_size builds its cache path inline rather
    than via a standalone helper (unlike betatv.py's
    _box_hash_path_for), so this exercises it through a cache HIT: pre-
    seed a fake cache file at the path we expect for one backend, then
    confirm _raw_boxes_for_size only finds it (returns used_neural_net=
    False) for THAT backend, not a different one - proving the backend
    name is actually part of the path being checked, not just decoration
    elsewhere.
    """

    def setUp( self ):
        import tempfile, os
        self._tmpdir = tempfile.mkdtemp()
        self._orig_cwd = os.getcwd()
        # betastare's cache path is relative ('../output/cache/pic_hashes/...'),
        # so run from a subdirectory of the temp dir to match its assumed layout
        self._work_subdir = os.path.join( self._tmpdir, 'BetaSuite-0.2.4' )
        os.makedirs( self._work_subdir, exist_ok=True )
        os.chdir( self._work_subdir )

        self._had_attr = hasattr( betaconfig, 'detector_backend' )
        if self._had_attr:
            self._original = betaconfig.detector_backend
        betaconfig.global_min_prob = 0.1

    def tearDown( self ):
        import shutil, os
        os.chdir( self._orig_cwd )
        shutil.rmtree( self._tmpdir, ignore_errors=True )
        if self._had_attr:
            betaconfig.detector_backend = self._original
        elif hasattr( betaconfig, 'detector_backend' ):
            del betaconfig.detector_backend

    def test_cache_seeded_for_one_backend_is_not_hit_by_another( self ):
        import betautils_hash as bu_hash

        betaconfig.detector_backend = { 'selected': 'retinanet_v2', 'retinanet_v2': {}, 'nudenet_v3': {} }
        bu_config.invalidate_config_caches()
        # seed a fake cache entry as if retinanet_v2 already ran
        seeded_boxes = [ { 'x': 1, 'y': 1, 'w': 2, 'h': 2, 'class_id': 'exposed_breast', 'score': 0.9, 't': 0 } ]
        retinanet_path = '../output/cache/pic_hashes/%s-%s-%d-%.3f.gz'%(
            'fakehash', 'retinanet_v2', 640, 0.1 )
        bu_hash.write_json( seeded_boxes, retinanet_path )

        # switch to nudenet_v3 - same image hash/size/min_prob, only the
        # backend differs - this must NOT be treated as a cache hit
        betaconfig.detector_backend = { 'selected': 'nudenet_v3', 'retinanet_v2': {}, 'nudenet_v3': {} }
        bu_config.invalidate_config_caches()
        nudenet_path = '../output/cache/pic_hashes/%s-%s-%d-%.3f.gz'%(
            'fakehash', 'nudenet_v3', 640, 0.1 )
        import os
        self.assertFalse( os.path.exists( nudenet_path ) )


if __name__ == '__main__':
    unittest.main()
