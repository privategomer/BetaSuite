"""
test_cache_path_duplicates_stay_in_sync.py - guards the single-source-of-
truth rule for filenames.

THE HISTORY
    Cache path formulas were once re-implemented independently in up to
    six standalone tools. They drifted, and the drift caused real bugs:
    one copy's glob omitted the detector backend name, which corrupted
    the file_hash it parsed back out and made every video look "not
    found"; another copy did not model the whole-file preview fallback,
    so every video shorter than preview_start_seconds silently missed
    the cache its tool expected.

    Those duplicates are gone. betautils_cache_paths is the only module
    that may build a cache path, an output filename, or a cache key.

WHAT THIS FILE ASSERTS
    1. No other module in the tree contains a path-format string that
       looks like one of these names. A second copy reappearing is the
       failure mode; a static scan is the only thing that catches it
       before it drifts.
    2. Every path builder produces something that round-trips: the
       discovery code can recover the file_hash from a name the builder
       produced.
    3. Changing a setting that changes detections changes the detection
       cache path, and changing one that does not, does not.
"""

import os
import re
import unittest

import betaconfig
import betautils_cache_paths as bu_cache
import betautils_config as bu_config
import betautils_detector as bu_detector


REPO_ROOT = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )

# The module allowed to know what these files are called.
NAMING_MODULE = 'betautils_cache_paths.py'

# A format string mentioning one of the cache directories is how a
# duplicate formula looks in source.
SUSPECT_PATTERNS = [
    re.compile( r"['\"][^'\"]*vid_hashes/%s" ),
    re.compile( r"['\"][^'\"]*pic_hashes/%s" ),
    re.compile( r"['\"][^'\"]*shot_cuts/%s" ),
    re.compile( r"['\"][^'\"]*transcode_cache/%s" ),
]


def _python_sources():
    """Every .py file in the tree except the naming module and tests."""
    for dirpath, dirnames, filenames in os.walk( REPO_ROOT ):
        dirnames[:] = [ d for d in dirnames
                        if d not in ( '__pycache__', 'tests', '.git' ) ]
        for filename in sorted( filenames ):
            if not filename.endswith( '.py' ) or filename == NAMING_MODULE:
                continue
            yield os.path.join( dirpath, filename )


class TestNoDuplicatePathFormulas( unittest.TestCase ):

    def test_no_module_other_than_the_naming_module_builds_a_cache_path( self ):
        offenders = []
        for path in _python_sources():
            with open( path, 'r', encoding='UTF-8' ) as source_file:
                source = source_file.read()
            for pattern in SUSPECT_PATTERNS:
                for match in pattern.finditer( source ):
                    offenders.append( '%s: %s'%(
                        os.path.relpath( path, REPO_ROOT ), match.group( 0 ) ) )
        self.assertEqual( offenders, [],
            "a cache-path format string appeared outside %s. Every cache path, output "
            "filename and cache key must come from that module - a second copy is how the "
            "backend-name-missing-from-the-glob bug happened. Offenders:\n  %s"%(
                NAMING_MODULE, '\n  '.join( offenders ) ) )


class TestPathsRoundTrip( unittest.TestCase ):

    FILE_HASH = 'deadbeefcafebabe'
    SIZE = 1280
    FPS = 9
    MIN_PROB = 0.2
    BACKEND = 'nudenet_v3'

    def test_file_hash_survives_a_round_trip_through_a_real_path( self ):
        path = bu_cache.box_hash_path_for(
            self.FILE_HASH, self.SIZE, self.FPS, self.MIN_PROB, self.BACKEND )
        self.assertEqual( bu_cache._file_hash_from_cache_path( path ), self.FILE_HASH )

    def test_preview_suffix_survives_a_round_trip( self ):
        suffix = bu_cache.preview_cache_suffix( True, 10.0 )
        self.assertEqual( suffix, '-preview@10.0s' )
        path = bu_cache.box_hash_path_for(
            self.FILE_HASH, self.SIZE, self.FPS, self.MIN_PROB, self.BACKEND, suffix )
        self.assertTrue( path.endswith( '-preview@10.0s.gz' ), path )
        self.assertEqual( bu_cache._file_hash_from_cache_path( path ), self.FILE_HASH )

    def test_whole_file_preview_gets_the_plain_suffix( self ):
        # A preview run of a file shorter than preview_start_seconds
        # processes the whole file, so it is not a slice of anything and
        # must not get an '@<offset>s' implying one. This is the exact
        # case two hand-copied formulas failed to model.
        self.assertEqual( bu_cache.preview_cache_suffix( True, 0.0 ), '-preview' )
        self.assertEqual( bu_cache.preview_cache_suffix( False, 12.5 ), '' )

    def test_backend_name_is_in_the_path( self ):
        for backend_name in bu_detector.registered_backend_names():
            path = bu_cache.box_hash_path_for(
                self.FILE_HASH, self.SIZE, self.FPS, self.MIN_PROB, backend_name )
            self.assertIn( backend_name, path )


class TestDetectionKeySensitivity( unittest.TestCase ):
    """
    The detection key must change exactly when raw detections would.

    Before 2.1 the cache path carried only size/fps/min_prob, so
    changing nudenet_v3's nms_iou or candidate_floor silently served
    back detections computed under the old value.
    """

    def setUp( self ):
        self.original = betaconfig.detector_backend
        betaconfig.detector_backend = {
            'selected': 'nudenet_v3',
            'nudenet_v3': {
                'model_variant': '320n',
                'candidate_floor': 0.2,
                'nms_iou': 0.45,
                'nms_mode': 'per_class',
                'nn_batch_size': 4,
            },
            'retinanet_v2': { 'picture_sizes': [ 1280 ], 'nn_batch_size': 2 },
        }
        bu_config.invalidate_config_caches()

    def tearDown( self ):
        betaconfig.detector_backend = self.original
        bu_config.invalidate_config_caches()

    def _key( self ):
        return bu_cache.detection_key( 'nudenet_v3', 320, 9, 0.2 )

    def test_nms_iou_change_invalidates_the_detection_cache( self ):
        before = self._key()
        betaconfig.detector_backend['nudenet_v3']['nms_iou'] = 0.60
        self.assertNotEqual( before, self._key() )

    def test_candidate_floor_change_invalidates_the_detection_cache( self ):
        before = self._key()
        betaconfig.detector_backend['nudenet_v3']['candidate_floor'] = 0.35
        self.assertNotEqual( before, self._key() )

    def test_nms_mode_change_invalidates_the_detection_cache( self ):
        before = self._key()
        betaconfig.detector_backend['nudenet_v3']['nms_mode'] = 'agnostic'
        self.assertNotEqual( before, self._key() )

    def test_model_variant_change_invalidates_the_detection_cache( self ):
        before = self._key()
        betaconfig.detector_backend['nudenet_v3']['model_variant'] = '640m'
        self.assertNotEqual( before, self._key() )

    def test_batch_size_does_not_invalidate_the_detection_cache( self ):
        # Batching changes how many frames go into one inference call,
        # never which detections come out, so reusing detections across
        # batch sizes is correct and saves real time.
        # tests/test_batch_size_invariance.py enforces the premise.
        before = self._key()
        betaconfig.detector_backend['nudenet_v3']['nn_batch_size'] = 1
        self.assertEqual( before, self._key() )

    def test_retinanet_has_no_detection_affecting_tunables( self ):
        # Score filtering and NMS live inside that export's graph, so
        # there is nothing outside size/fps/min_prob to key on.
        self.assertEqual( bu_detector.get_detection_identity( 'retinanet_v2' ), {} )


class TestCensorKeySensitivity( unittest.TestCase ):
    """
    Every setting that changes rendered output must change the censor
    key. Before 2.1 the output filename deliberately excluded tracking
    settings, so re-running with a changed track_max_gap overwrote the
    previous output and left nothing to compare.
    """

    def setUp( self ):
        self.original_overrides = betaconfig.item_overrides
        self.original_backend = betaconfig.detector_backend
        bu_config.invalidate_config_caches()

    def tearDown( self ):
        betaconfig.item_overrides = self.original_overrides
        betaconfig.detector_backend = self.original_backend
        bu_config.invalidate_config_caches()

    def test_track_max_gap_change_changes_the_censor_key( self ):
        before = bu_cache.censor_key()
        import copy
        mutated = copy.deepcopy( betaconfig.detector_backend )
        backend = mutated['selected']
        mutated[backend].setdefault( 'item_overrides', {} ).setdefault(
            'exposed_breast', {} )['track_max_gap'] = 99.0
        betaconfig.detector_backend = mutated
        bu_config.invalidate_config_caches()
        self.assertNotEqual( before, bu_cache.censor_key() )

    def test_watermark_change_changes_the_censor_key( self ):
        before = bu_cache.censor_key()
        original = betaconfig.enable_betasuite_watermark
        try:
            betaconfig.enable_betasuite_watermark = not original
            bu_config.invalidate_config_caches()
            self.assertNotEqual( before, bu_cache.censor_key() )
        finally:
            betaconfig.enable_betasuite_watermark = original
            bu_config.invalidate_config_caches()

    def test_blur_approximation_change_changes_the_censor_key( self ):
        before = bu_cache.censor_key()
        original = getattr( betaconfig, 'blur_fast_approximation', True )
        try:
            betaconfig.blur_fast_approximation = not original
            bu_config.invalidate_config_caches()
            self.assertNotEqual( before, bu_cache.censor_key() )
        finally:
            betaconfig.blur_fast_approximation = original
            bu_config.invalidate_config_caches()


class TestEncodeKey( unittest.TestCase ):

    def test_crf_change_changes_the_encode_key( self ):
        before = bu_cache.encode_key( False )
        original = getattr( betaconfig, 'encode_crf', 17 )
        try:
            betaconfig.encode_crf = original + 3
            self.assertNotEqual( before, bu_cache.encode_key( False ) )
        finally:
            betaconfig.encode_crf = original

    def test_preview_and_real_runs_get_different_encode_keys( self ):
        # They use different presets by design, so a preview output can
        # never be mistaken for a real one on filename alone.
        self.assertNotEqual( bu_cache.encode_key( True ), bu_cache.encode_key( False ) )


class TestOutputNaming( unittest.TestCase ):

    def test_output_name_carries_all_three_keys( self ):
        final_path, intermediate_path = bu_cache.video_output_paths(
            '/out', 'clip', 'abc123', [ 320 ], 9, 'aaaaaa', 'bbbbbb', 'cccc' )
        self.assertIn( '-daaaaaa-', final_path )
        self.assertIn( '-cbbbbbb-', final_path )
        self.assertIn( '-ecccc', final_path )
        self.assertTrue( final_path.endswith( '.mp4' ) )
        self.assertTrue( intermediate_path.endswith( '.mkv' ) )

    def test_temp_sibling_keeps_the_extension( self ):
        # ffmpeg picks its muxer from the extension, so a name ending
        # '.mkv.tmp' makes it fail to pick one at all.
        self.assertEqual( bu_cache.temp_sibling( '/a/b.mkv' ), '/a/b.tmp.mkv' )

    def test_chunk_paths_are_ordered_and_distinct( self ):
        paths = [ bu_cache.chunk_path_for( '/a/b.mkv', index ) for index in range( 3 ) ]
        self.assertEqual( len( set( paths ) ), 3 )
        self.assertEqual( paths, sorted( paths ) )


if __name__ == '__main__':
    unittest.main()
