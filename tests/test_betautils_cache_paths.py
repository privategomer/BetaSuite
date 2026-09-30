"""
test_betautils_cache_paths.py - direct coverage for the cache-discovery
logic in betautils_cache_paths.py (discover_full_run_videos/
build_hash_to_video_path), which previously only had indirect coverage
via test_cache_path_duplicates_stay_in_sync.py's box_hash_path_for
string checks. This is the logic that actually caused the real
2026-09-16 analyze_style_flicker.py bug (backend name not pinned into
the discovery glob corrupted the extracted file_hash, making every
video appear "not found") - box_hash_path_for producing the right
STRING was never the part that broke; discover_full_run_videos parsing
filenames back apart was. These tests build real .gz cache files on
disk under a temp dir (monkeypatching VID_HASH_DIR) rather than mocking
glob, so a future change to the real filename shape is caught the same
way a real cache directory would catch it.
"""

import gzip
import json
import os
import shutil
import tempfile
import unittest

import betautils_cache_paths as bu_cache


class TestDiscoverFullRunVideos( unittest.TestCase ):

    def setUp( self ):
        self._tmpdir = tempfile.mkdtemp()
        self._orig_vid_hash_dir = bu_cache.VID_HASH_DIR
        bu_cache.VID_HASH_DIR = self._tmpdir

    def tearDown( self ):
        bu_cache.VID_HASH_DIR = self._orig_vid_hash_dir
        shutil.rmtree( self._tmpdir, ignore_errors=True )

    def _write_cache( self, file_hash, size, backend_name, preview_suffix='', fps=9, min_prob=0.2, content=None ):
        path = os.path.join( self._tmpdir, os.path.basename(
            bu_cache.box_hash_path_for( file_hash, size, fps, min_prob, backend_name, preview_suffix ) ) )
        with gzip.open( path, 'wt' ) as f:
            json.dump( content if content is not None else [], f )
        return path

    def test_real_cache_found_for_matching_backend( self ):
        self._write_cache( 'abc123', 800, 'nudenet_v3' )
        self._write_cache( 'abc123', 1280, 'nudenet_v3' )
        result, preview_used_for = bu_cache.discover_full_run_videos( [800, 1280], 9, 0.2, 'nudenet_v3' )
        self.assertIn( 'abc123', result )
        self.assertEqual( len( result['abc123'] ), 2 )
        self.assertEqual( preview_used_for, {} )

    def test_backend_isolation_a_different_backends_cache_is_not_picked_up( self ):
        # this is the exact shape of the real bug: a cache written under
        # one backend must never be discovered when asking for another.
        self._write_cache( 'abc123', 800, 'retinanet_v2' )
        self._write_cache( 'abc123', 1280, 'retinanet_v2' )
        result, _ = bu_cache.discover_full_run_videos( [800, 1280], 9, 0.2, 'nudenet_v3' )
        self.assertEqual( result, {} )

    def test_backend_name_does_not_pollute_extracted_file_hash( self ):
        # regression check for the exact real bug: a naive glob suffix
        # without the backend name baked in would extract
        # 'abc123-nudenet_v3' as the "file_hash" instead of clean
        # 'abc123'. Confirm the key really is the clean hash.
        self._write_cache( 'abc123', 800, 'nudenet_v3' )
        result, _ = bu_cache.discover_full_run_videos( [800], 9, 0.2, 'nudenet_v3' )
        self.assertIn( 'abc123', result )
        self.assertNotIn( 'abc123-nudenet_v3', result )

    def test_incomplete_size_set_is_excluded( self ):
        # only 800 exists, 1280 doesn't - not a "complete run" for either size.
        self._write_cache( 'abc123', 800, 'nudenet_v3' )
        result, _ = bu_cache.discover_full_run_videos( [800, 1280], 9, 0.2, 'nudenet_v3' )
        self.assertEqual( result, {} )

    def test_preview_not_used_when_include_preview_false( self ):
        self._write_cache( 'ghi789', 800, 'nudenet_v3', preview_suffix='-preview@30.0s' )
        result, preview_used_for = bu_cache.discover_full_run_videos( [800], 9, 0.2, 'nudenet_v3', include_preview=False )
        self.assertEqual( result, {} )
        self.assertEqual( preview_used_for, {} )

    def test_preview_used_as_fallback_when_no_real_cache( self ):
        self._write_cache( 'ghi789', 800, 'nudenet_v3', preview_suffix='-preview@30.0s' )
        result, preview_used_for = bu_cache.discover_full_run_videos( [800], 9, 0.2, 'nudenet_v3', include_preview=True )
        self.assertIn( 'ghi789', result )
        self.assertEqual( preview_used_for.get( 'ghi789' ), '-preview@30.0s' )

    def test_real_cache_preferred_over_preview_for_same_hash( self ):
        self._write_cache( 'abc123', 800, 'nudenet_v3' )
        self._write_cache( 'abc123', 800, 'nudenet_v3', preview_suffix='-preview@30.0s' )
        result, preview_used_for = bu_cache.discover_full_run_videos( [800], 9, 0.2, 'nudenet_v3', include_preview=True )
        self.assertIn( 'abc123', result )
        self.assertNotIn( 'abc123', preview_used_for )  # real was used, not preview
        real_path = bu_cache.box_hash_path_for( 'abc123', 800, 9, 0.2, 'nudenet_v3' )
        self.assertTrue( result['abc123'][0].endswith( os.path.basename( real_path ) ) )

    def test_ambiguous_multi_offset_preview_is_skipped( self ):
        # two DIFFERENT preview offsets for the same file_hash/size -
        # not safe to guess which one the caller wants, so this hash
        # must be skipped entirely rather than silently picking one.
        self._write_cache( 'jkl000', 800, 'nudenet_v3', preview_suffix='-preview@10.0s' )
        self._write_cache( 'jkl000', 800, 'nudenet_v3', preview_suffix='-preview@30.0s' )
        result, preview_used_for = bu_cache.discover_full_run_videos( [800], 9, 0.2, 'nudenet_v3', include_preview=True )
        self.assertNotIn( 'jkl000', result )

    def test_no_caches_at_all_returns_empty( self ):
        result, preview_used_for = bu_cache.discover_full_run_videos( [800], 9, 0.2, 'nudenet_v3' )
        self.assertEqual( result, {} )
        self.assertEqual( preview_used_for, {} )


class TestFmtStats( unittest.TestCase ):

    def test_empty( self ):
        self.assertEqual( bu_cache.fmt_stats( [] ), "n=0" )

    def test_basic_stats( self ):
        result = bu_cache.fmt_stats( [ 1.0, 2.0, 3.0 ] )
        self.assertIn( "n=3", result )
        self.assertIn( "median=2.0", result )

    def test_fmt_ms_stats_includes_min_and_unit( self ):
        result = bu_cache.fmt_ms_stats( [ 5.0, 10.0, 20.0 ] )
        self.assertIn( "min=5.0ms", result )
        self.assertIn( "max=20.0ms", result )


if __name__ == '__main__':
    unittest.main()


class TestCosmeticOutputLabel( unittest.TestCase ):
    """
    A sweep renders one source many times. The keys already keep those
    outputs from colliding, but a directory of them is unreadable without
    decoding hex, so an optional label is appended to the OUTPUT name.

    The two properties that matter: it never reaches a cache path (or two
    labelled runs would stop sharing cached detections), and it cannot
    inject anything into a filename.
    """

    def test_no_label_is_the_old_name_exactly( self ):
        without = bu_cache.video_output_basename(
            'clip', 'abc123', [ 640 ], 9, 'd1', 'c1', 'e1', '-preview@80.0s' )
        self.assertTrue( without.endswith( '-preview@80.0s' ) )

    def test_a_label_is_appended_at_the_end( self ):
        labelled = bu_cache.video_output_basename(
            'clip', 'abc123', [ 640 ], 9, 'd1', 'c1', 'e1', '-preview@80.0s',
            'gaussian-s120' )
        self.assertTrue( labelled.endswith( '-preview@80.0s-gaussian-s120' ) )

    def test_a_label_does_not_change_any_cache_path( self ):
        # The whole point: a labelled render still reuses the detections.
        detection = bu_cache.box_hash_path_for( 'abc123', 640, 9, 0.12,
                                                'nudenet_v3', '-preview@80.0s' )
        shot_cuts = bu_cache.shot_cut_path_for( 'abc123', 9, 0.4, '-preview@80.0s' )
        for path in ( detection, shot_cuts ):
            self.assertNotIn( 'gaussian', path )
            self.assertNotIn( 's120', path )

    def test_unsafe_characters_are_replaced( self ):
        self.assertEqual( bu_cache.sanitise_output_label( 'a/b c:d' ), '-a-b-c-d' )
        self.assertNotIn( '/', bu_cache.sanitise_output_label( '../../etc/passwd' ) )

    def test_an_empty_label_adds_nothing( self ):
        for empty in ( '', None, '---' ):
            self.assertEqual( bu_cache.sanitise_output_label( empty ), '' )

    def test_the_label_survives_into_both_output_paths( self ):
        final_path, intermediate = bu_cache.video_output_paths(
            '/tmp/out', 'clip', 'abc123', [ 640 ], 9, 'd1', 'c1', 'e1',
            '-preview@80.0s', 'mkv', 'triple_box-s060' )
        self.assertIn( 'triple_box-s060', final_path )
        self.assertIn( 'triple_box-s060', intermediate )
