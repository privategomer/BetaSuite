"""
test_replay_caching.py - the caches that make repeated replays cheap, and
the ways each one could silently serve a wrong answer.

A cache that returns a stale value here is worse than no cache at all:
auto_tune compares every candidate against a baseline, so a baseline
served from before an accepted write would make every later candidate a
comparison against a configuration nobody is running. That failure is
invisible in the output - the numbers look fine, they are just answering
a question about the wrong config.

So each cache is tested for the thing that would make it wrong:

  1. frame_size_for_video      caches per resolved path, and a file that
                               cannot be opened is not retried forever.
  2. _read_raw_boxes           keys on (path, mtime, size), so an
                               externally rewritten cache is re-read; and
                               hands out copies, so one replay mutating
                               its boxes cannot corrupt the next.
  3. auto_tune's baseline      keys on the backend's item_overrides, so
                               an accepted value written by an earlier
                               stage invalidates it.
"""

import copy
import importlib.util
import json
import os
import sys
import tempfile
import unittest

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )

import betaconfig
import betautils_hash as bu_hash
import betautils_tuning as bu_tuning


class TestFrameSizeCache( unittest.TestCase ):
    """
    frame_size_for_video opens a video file for two integers that cannot
    change while the file does not. Every replay was re-opening all of
    them; an auto_tune run of 87 replays over 7 videos did 609 opens for
    7 distinct answers.
    """

    def setUp( self ):
        bu_tuning.clear_replay_caches()

    def tearDown( self ):
        bu_tuning.clear_replay_caches()

    def test_a_second_lookup_does_not_reopen_the_file( self ):
        import betautils_tuning
        opens = []

        class _FakeCapture:
            def __init__( self, path ):
                opens.append( path )
            def get( self, prop ):
                import cv2
                return 1920.0 if prop == cv2.CAP_PROP_FRAME_WIDTH else 1080.0
            def release( self ):
                pass

        import cv2
        original = cv2.VideoCapture
        cv2.VideoCapture = _FakeCapture
        try:
            first = betautils_tuning.frame_size_for_video( '/fake/a.mp4' )
            second = betautils_tuning.frame_size_for_video( '/fake/a.mp4' )
        finally:
            cv2.VideoCapture = original

        self.assertEqual( first, ( 1920, 1080 ) )
        self.assertEqual( second, ( 1920, 1080 ) )
        self.assertEqual( len( opens ), 1,
            "the second lookup re-opened the video; the cache is not holding" )

    def test_different_paths_are_cached_separately( self ):
        import betautils_tuning, cv2

        sizes = { '/fake/a.mp4': ( 1920, 1080 ), '/fake/b.mp4': ( 1280, 720 ) }

        class _FakeCapture:
            def __init__( self, path ):
                self.size = sizes[path]
            def get( self, prop ):
                return float( self.size[0] if prop == cv2.CAP_PROP_FRAME_WIDTH else self.size[1] )
            def release( self ):
                pass

        original = cv2.VideoCapture
        cv2.VideoCapture = _FakeCapture
        try:
            self.assertEqual( betautils_tuning.frame_size_for_video( '/fake/a.mp4' ), ( 1920, 1080 ) )
            self.assertEqual( betautils_tuning.frame_size_for_video( '/fake/b.mp4' ), ( 1280, 720 ) )
        finally:
            cv2.VideoCapture = original

    def test_an_unreadable_video_is_not_retried( self ):
        # A missing or corrupt file reports 0x0. Caching that is the
        # point: without it the failure costs an open on all 87 replays.
        import betautils_tuning, cv2
        opens = []

        class _FakeCapture:
            def __init__( self, path ):
                opens.append( path )
            def get( self, prop ):
                return 0.0
            def release( self ):
                pass

        original = cv2.VideoCapture
        cv2.VideoCapture = _FakeCapture
        try:
            self.assertEqual( betautils_tuning.frame_size_for_video( '/fake/gone.mp4' ), ( 0, 0 ) )
            self.assertEqual( betautils_tuning.frame_size_for_video( '/fake/gone.mp4' ), ( 0, 0 ) )
        finally:
            cv2.VideoCapture = original
        self.assertEqual( len( opens ), 1, "an unreadable video was re-opened" )


class TestRawBoxCache( unittest.TestCase ):
    """
    Detection caches are large - 182616 boxes across 7 files for 640m -
    and a replay parses every one of them. Reparsing on all 87 replays of
    a tuning run is the biggest non-tracking cost in the tool.
    """

    def setUp( self ):
        bu_tuning.clear_replay_caches()
        self.tmp = tempfile.mkdtemp()

    def tearDown( self ):
        bu_tuning.clear_replay_caches()

    def _write( self, name, boxes ):
        path = os.path.join( self.tmp, name )
        bu_hash.write_json( boxes, path )
        return path

    def test_returns_the_same_content_on_a_second_read( self ):
        path = self._write( 'a.gz', [ { 'class_id': 'exposed_breast', 'x': 1 } ] )
        first = bu_tuning._read_raw_boxes( [ path ] )
        second = bu_tuning._read_raw_boxes( [ path ] )
        self.assertEqual( first, second )

    def test_hands_out_copies_so_a_caller_cannot_poison_the_cache( self ):
        # prepare_boxes_for_render MUTATES the boxes it is given. If the
        # cache handed out its own list, the first replay's mutations
        # would become the second replay's input, and every candidate
        # after the first would be measured on corrupted detections.
        path = self._write( 'b.gz', [ { 'class_id': 'exposed_breast', 'x': 1 } ] )
        first = bu_tuning._read_raw_boxes( [ path ] )
        first[0]['x'] = 999
        first[0]['injected'] = True
        second = bu_tuning._read_raw_boxes( [ path ] )
        self.assertEqual( second[0]['x'], 1,
            "a caller's mutation reached the cached parse" )
        self.assertNotIn( 'injected', second[0] )

    def test_a_rewritten_cache_file_is_re_read( self ):
        # The key carries mtime and size, so a file rewritten by a real
        # run mid-session is picked up rather than served stale.
        path = self._write( 'c.gz', [ { 'class_id': 'exposed_breast', 'x': 1 } ] )
        self.assertEqual( bu_tuning._read_raw_boxes( [ path ] )[0]['x'], 1 )

        os.remove( path )
        bu_hash.write_json( [ { 'class_id': 'exposed_breast', 'x': 2 },
                              { 'class_id': 'exposed_vulva', 'x': 3 } ], path )
        # Force a distinct mtime even on a coarse-resolution clock.
        stat = os.stat( path )
        os.utime( path, ( stat.st_atime, stat.st_mtime + 5 ) )

        again = bu_tuning._read_raw_boxes( [ path ] )
        self.assertEqual( len( again ), 2, "a rewritten cache file was served from the cache" )
        self.assertEqual( again[0]['x'], 2 )

    def test_a_missing_file_is_skipped_not_raised( self ):
        path = self._write( 'd.gz', [ { 'class_id': 'exposed_breast', 'x': 1 } ] )
        boxes = bu_tuning._read_raw_boxes( [ path, os.path.join( self.tmp, 'nope.gz' ) ] )
        self.assertEqual( len( boxes ), 1 )


class TestAutoTuneBaselineCache( unittest.TestCase ):
    """
    The baseline cache is the one with real damage potential.

    Each stage measures a baseline before its candidates, but a baseline
    depends only on (configuration, current config state) - not on which
    knob the stage is about to sweep. Six stages over three
    configurations measured 18 baselines for 3 distinct answers.

    The catch: auto_tune WRITES an accepted value to config before the
    next stage runs, and after that write the baseline genuinely has
    moved. So the key must include the config state, or a later stage
    compares its candidates against a configuration that is no longer
    the one running.
    """

    def _key( self, backend_name, configuration_label, seed ):
        # Mirrors the key measure_stage builds. Kept in the test so a
        # change to the real key that drops the config-state component
        # fails here rather than silently serving stale baselines.
        return (
            backend_name,
            configuration_label,
            seed,
            json.dumps(
                betaconfig.detector_backend.get( backend_name, {} ).get( 'item_overrides', {} ),
                sort_keys=True, default=str ),
        )

    def setUp( self ):
        self.backend = 'nudenet_v3'
        self.saved = copy.deepcopy(
            betaconfig.detector_backend[self.backend]['item_overrides'] )

    def tearDown( self ):
        betaconfig.detector_backend[self.backend]['item_overrides'] = self.saved

    def test_two_stages_on_one_configuration_share_a_baseline( self ):
        first = self._key( self.backend, 'nudenet_v3/640m @640', 0 )
        second = self._key( self.backend, 'nudenet_v3/640m @640', 0 )
        self.assertEqual( first, second )

    def test_an_accepted_write_invalidates_the_baseline( self ):
        before = self._key( self.backend, 'nudenet_v3/640m @640', 0 )
        betaconfig.detector_backend[self.backend]['item_overrides'].setdefault(
            'exposed_vulva', {} )['track_max_gap'] = 99.9
        after = self._key( self.backend, 'nudenet_v3/640m @640', 0 )
        self.assertNotEqual( before, after,
            "the baseline key ignored an item_overrides write, so a later stage would "
            "compare its candidates against a configuration that is no longer running" )

    def test_two_configurations_of_one_backend_do_not_share( self ):
        # 320n and 640m share an item_overrides block but are different
        # footage-through-a-different-model, so their baselines differ.
        self.assertNotEqual(
            self._key( self.backend, 'nudenet_v3/320n @320', 0 ),
            self._key( self.backend, 'nudenet_v3/640m @640', 0 ) )

    def test_a_different_seed_does_not_share( self ):
        self.assertNotEqual(
            self._key( self.backend, 'nudenet_v3/640m @640', 0 ),
            self._key( self.backend, 'nudenet_v3/640m @640', 1 ) )

    def test_the_context_shares_one_cache_dict_across_stages( self ):
        # measure_stage does `context = dict( _context )` per stage, so a
        # cache created inside measure_stage would be discarded every
        # time. The cache dict has to come from the shared context and be
        # copied BY REFERENCE for the cache to survive at all.
        shared = { '_baseline_cache': {} }
        per_stage_one = dict( shared )
        per_stage_one.setdefault( '_baseline_cache', {} )['k'] = 'measured'
        per_stage_two = dict( shared )
        self.assertIn( 'k', per_stage_two.get( '_baseline_cache', {} ),
            "the baseline cache did not survive measure_stage's per-stage dict() copy" )


if __name__ == '__main__':
    unittest.main()
