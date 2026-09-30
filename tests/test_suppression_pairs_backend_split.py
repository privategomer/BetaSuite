"""
test_suppression_pairs_backend_split.py - regression test for
tools/analysis/analyze_suppression_pairs.py's backend_of_cache_filename,
added when that tool was made backend-aware (splits its box-size/
confusion-pair-IoU report into one section per detector backend, instead
of pooling every backend's cached detections into one misleading
combined report - see that tool's module-level comments in main()).

Only covers the parsing function itself (pure, no cache I/O needed) -
the full per-backend report split was verified manually against real
on-device cache filenames and a synthetic two-backend cache directory
during development; not re-covered here since main() would need a real
on-disk cache fixture to exercise end-to-end.
"""

import importlib.util
import os
import unittest

_HERE = os.path.dirname( os.path.abspath( __file__ ) )
_REPO_ROOT = os.path.dirname( _HERE )


def _load_module():
    full_path = os.path.join( _REPO_ROOT, 'tools/analysis/analyze_suppression_pairs.py' )
    spec = importlib.util.spec_from_file_location( 'analyze_suppression_pairs_under_test', full_path )
    module = importlib.util.module_from_spec( spec )
    spec.loader.exec_module( module )
    return module


class TestBackendOfCacheFilename( unittest.TestCase ):

    @classmethod
    def setUpClass( cls ):
        cls.mod = _load_module()

    def test_recognizes_nudenet_v3_video_cache_filename( self ):
        self.assertEqual(
            self.mod.backend_of_cache_filename( '533681d290694825-nudenet_v3-2-1280-9-0.200.gz' ),
            'nudenet_v3' )

    def test_recognizes_retinanet_v2_video_cache_filename( self ):
        self.assertEqual(
            self.mod.backend_of_cache_filename( 'abc123-retinanet_v2-2-640-9-0.100.gz' ),
            'retinanet_v2' )

    def test_recognizes_backend_in_picture_cache_filename( self ):
        # picture cache has one fewer field (no fps) but backend is still
        # the second '-'-delimited segment
        self.assertEqual(
            self.mod.backend_of_cache_filename( 'abc123-nudenet_v3-2-640-0.200.gz' ),
            'nudenet_v3' )

    def test_recognizes_backend_with_preview_suffix( self ):
        self.assertEqual(
            self.mod.backend_of_cache_filename( 'abc123-retinanet_v2-2-1280-9-0.200-preview@10.0s.gz' ),
            'retinanet_v2' )

    def test_pre_backend_tagging_legacy_filename_is_not_misattributed( self ):
        # a cache file written before backend-tagging existed - must NOT
        # be silently treated as belonging to whichever backend happens
        # to be selected right now
        self.assertEqual(
            self.mod.backend_of_cache_filename( 'abc123-2-1280-9-0.200.gz' ),
            'unknown/legacy' )

    def test_unrecognized_second_field_falls_back_to_unknown( self ):
        self.assertEqual(
            self.mod.backend_of_cache_filename( 'abc123-some_future_thing-2-1280-9-0.200.gz' ),
            'unknown/legacy' )


if __name__ == '__main__':
    unittest.main()
