"""
test_reverse_bench.py - betabench reverse, phase 0 of reverse mode.

The bench decides the frame-style default and whether reverse mode is
affordable, so its arithmetic is pinned here: the trigger's runtime share
(a union of intervals, weighted by duration, never a count), the
whole-render estimate (which must be able to report a SPEEDUP, not floor
it at zero), the prototype window blend (alpha applied once, so no clear
original survives at a window's corners), the frame hook on the shared
pipeline runner, and which detection caches the share is allowed to read.
"""

import logging
import os
import shutil
import subprocess
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

import betaconfig
import betaconst
import betautils_cache_paths as bu_cache
import betautils_detector as bu_detector
import betautils_hash as bu_hash

from tests.test_betatv_end_to_end import _ffmpeg_available
from tests.test_structure_bench import betabench, _quiet_logger


def _textured( height, width, seed=0 ):
    """Noise, so a blur or a mosaic is guaranteed to change nearly every pixel."""
    return np.random.default_rng( seed ).integers( 0, 255, ( height, width, 3 ), dtype=np.uint8 )


class TestTriggerSeconds( unittest.TestCase ):

    def test_no_firings_is_no_time( self ):
        self.assertEqual( betabench.trigger_seconds( [], 10.0, 0.5, 60.0 ), 0.0 )

    def test_one_firing_covers_its_sample_period_plus_the_hold( self ):
        self.assertAlmostEqual( betabench.trigger_seconds( [ 1.0 ], 10.0, 0.5, 60.0 ), 0.6 )

    def test_overlapping_holds_are_not_double_counted( self ):
        # Three firings 0.1s apart with a 0.5s hold cover 1.0 to 1.8, not
        # three times 0.6. Counting firings instead of time is the mistake.
        self.assertAlmostEqual(
            betabench.trigger_seconds( [ 1.0, 1.1, 1.2 ], 10.0, 0.5, 60.0 ), 0.8 )

    def test_disjoint_firings_add( self ):
        self.assertAlmostEqual( betabench.trigger_seconds( [ 1.0, 5.0 ], 10.0, 0.0, 60.0 ), 0.2 )

    def test_order_does_not_matter( self ):
        self.assertAlmostEqual( betabench.trigger_seconds( [ 5.0, 1.0, 1.1 ], 10.0, 0.0, 60.0 ),
                                betabench.trigger_seconds( [ 1.0, 1.1, 5.0 ], 10.0, 0.0, 60.0 ) )

    def test_a_hold_past_the_end_of_the_file_is_clipped( self ):
        self.assertAlmostEqual( betabench.trigger_seconds( [ 9.9 ], 10.0, 0.5, 10.0 ), 0.1 )


class TestPredictedSlowdown( unittest.TestCase ):

    def test_against_the_bench_baseline( self ):
        # A triggered frame costs 0.2s instead of 0.1s; half the file is
        # triggered; so half the frames take twice as long: +50%.
        self.assertAlmostEqual(
            betabench.predicted_reverse_slowdown( 10.0, 5.0, None, 0.5 ), 0.5 )

    def test_against_a_real_render_rate( self ):
        # The same 0.1s extra per triggered frame, against a real render's
        # 0.25s per frame (it also composites today's censors).
        self.assertAlmostEqual(
            betabench.predicted_reverse_slowdown( 10.0, 5.0, 4.0, 0.5 ), 0.2 )

    def test_a_speedup_is_reported_not_floored( self ):
        # Blurred frames can be cheaper to encode than they cost to blur.
        # Flooring at zero would print "+0%" over a real 50% saving.
        self.assertAlmostEqual(
            betabench.predicted_reverse_slowdown( 5.0, 10.0, None, 1.0 ), -0.5 )

    def test_no_share_means_no_estimate( self ):
        self.assertIsNone( betabench.predicted_reverse_slowdown( 10.0, 5.0, None, None ) )

    def test_nothing_triggered_costs_nothing( self ):
        self.assertEqual( betabench.predicted_reverse_slowdown( 10.0, 5.0, None, 0.0 ), 0.0 )


class TestWindowRects( unittest.TestCase ):

    def test_windows_fit_the_frame_and_cover_the_requested_area( self ):
        for width, height in ( ( 1920, 1080 ), ( 640, 360 ), ( 1080, 1920 ) ):
            # 0.30 x 3 windows is wider than the frame's thirds, so the
            # clamp to the frame edge has to engage.
            for fraction, count in ( ( 0.0105, 2 ), ( 0.1031, 2 ), ( 0.30, 3 ) ):
                rects = betabench.reverse_window_rects( width, height, fraction, count )
                self.assertEqual( len( rects ), count )
                for x, y, w, h in rects:
                    self.assertGreaterEqual( x, 0 )
                    self.assertGreaterEqual( y, 0 )
                    self.assertLessEqual( x + w, width )
                    self.assertLessEqual( y + h, height )
                    self.assertAlmostEqual( w * h / float( width * height ), fraction, delta=0.002 )

    def test_typical_windows_do_not_overlap( self ):
        ( ax, _ay, aw, _ah ), ( bx, _by, _bw, _bh ) = betabench.reverse_window_rects(
            1920, 1080, 0.0105, 2 )
        self.assertLessEqual( ax + aw, bx )


class TestWindowBlend( unittest.TestCase ):

    def test_a_hard_edged_window_shows_exactly_what_it_was_given( self ):
        out = np.zeros( ( 60, 80, 3 ), np.uint8 )
        shown = _textured( 60, 80 )
        betabench.reverse_blend_window( out, shown, ( 10, 5, 30, 20 ), 'box', 0 )
        np.testing.assert_array_equal( out[ 5:25, 10:40 ], shown[ 5:25, 10:40 ] )
        self.assertFalse( out[ :5 ].any(), "nothing outside the window changes" )
        self.assertFalse( out[ :, 40: ].any() )

    def test_no_clear_original_survives_in_a_censored_windows_corners( self ):
        # The plan's leak test. A censored window is the style rendered on
        # the ORIGINAL, blended into the obscured frame through the
        # window's alpha once. Outside the ellipse the alpha is 0, so the
        # corners must show the obscured frame. If the alpha were applied
        # twice, or the original blended in anywhere, original pixels
        # would appear inside the window's bounding box.
        original = _textured( 120, 160, seed=1 )
        obscured = _textured( 120, 160, seed=2 )
        censored = _textured( 120, 160, seed=3 )
        out = obscured.copy()
        rect = ( 40, 30, 80, 60 )
        betabench.reverse_blend_window( out, censored, rect, 'ellipse', 0.25 )
        x, y, w, h = rect
        window = out[ y:y+h, x:x+w ]
        same_as_original = np.all( window == original[ y:y+h, x:x+w ], axis=2 )
        self.assertLess( same_as_original.mean(), 0.01,
                         "%.1f%% of the window's pixels are the clear original"%(
                             100.0 * same_as_original.mean() ) )
        np.testing.assert_array_equal( window[ 0, 0 ], obscured[ y, x ],
                                       "an ellipse's corner is outside it: the frame shows" )


class TestComposite( unittest.TestCase ):

    def test_hard_windows_show_the_original_and_everything_else_is_obscured( self ):
        frame = _textured( 90, 160, seed=4 )
        before = frame.copy()
        rects = [ ( 10, 20, 30, 30 ), ( 100, 20, 30, 30 ) ]
        style = { 'type': 'pixel', 'pattern': 'mosaic', 'strength': 10 }
        with mock.patch.object( betabench, 'REVERSE_WINDOW_SHAPE', 'box' ), \
             mock.patch.object( betabench, 'REVERSE_WINDOW_FEATHER', 0 ):
            out = betabench.reverse_composite( frame, style, rects )
        np.testing.assert_array_equal( frame, before, "the caller's frame is never modified" )
        for x, y, w, h in rects:
            np.testing.assert_array_equal( out[ y:y+h, x:x+w ], frame[ y:y+h, x:x+w ] )
        outside = np.ones( frame.shape[:2], bool )
        for x, y, w, h in rects:
            outside[ y:y+h, x:x+w ] = False
        unchanged = np.all( out == frame, axis=2 ) & outside
        self.assertLess( unchanged.sum() / float( outside.sum() ), 0.05,
                         "outside every window the frame style has been applied" )


@unittest.skipUnless( _ffmpeg_available(), "ffmpeg is needed to build the clip" )
class TestRunSegmentsFrameTransform( unittest.TestCase ):

    @classmethod
    def setUpClass( cls ):
        cls.tmp = tempfile.mkdtemp( prefix='betasuite-reverse-' )
        cls.clip = os.path.join( cls.tmp, 'clip.mp4' )
        subprocess.run(
            [ 'ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi',
              '-i', 'testsrc2=size=160x120:rate=30', '-t', '2', '-c:v', 'libx264',
              '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', cls.clip ], check=True )

    @classmethod
    def tearDownClass( cls ):
        shutil.rmtree( cls.tmp, ignore_errors=True )

    def test_every_frame_passes_through_the_transform_before_encoding( self ):
        seen = []

        def black( frame ):
            seen.append( frame.shape )
            return np.zeros_like( frame )

        encoder = { 'codec': 'libx264', 'preset': 'ultrafast', 'quality': 17 }
        out_dir = tempfile.mkdtemp( dir=self.tmp )
        _wall, results = betabench._run_segments( self.clip, [ ( 0, 20 ), ( 20, 20 ) ], 160, 120,
                                                  30.0, encoder, 1, out_dir,
                                                  frame_transform=black )
        self.assertEqual( len( seen ), 40 )
        self.assertTrue( all( r['error'] is None for r in results ) )
        import cv2
        capture = cv2.VideoCapture( results[0]['path'] )
        ok, frame = capture.read()
        capture.release()
        self.assertTrue( ok )
        self.assertLess( frame.mean(), 2.0, "what was encoded is the transformed frame" )

    def test_no_transform_is_the_old_behaviour( self ):
        encoder = { 'codec': 'libx264', 'preset': 'ultrafast', 'quality': 17 }
        out_dir = tempfile.mkdtemp( dir=self.tmp )
        _wall, results = betabench._run_segments( self.clip, [ ( 0, 20 ) ], 160, 120,
                                                  30.0, encoder, 1, out_dir )
        import cv2
        capture = cv2.VideoCapture( results[0]['path'] )
        ok, frame = capture.read()
        capture.release()
        self.assertTrue( ok )
        self.assertGreater( frame.mean(), 20.0, "the source frame, untouched" )


@unittest.skipUnless( _ffmpeg_available(), "ffmpeg is needed to build the clip" )
class TestTriggerShare( unittest.TestCase ):
    """Which caches the share reads, and that it is duration-weighted."""

    def setUp( self ):
        self.root = tempfile.mkdtemp( prefix='betasuite-share-' )
        self.saved = {
            'video_path_uncensored': betaconst.video_path_uncensored,
            'video_path_source_backup': betaconst.video_path_source_backup,
            'VID_HASH_DIR': bu_cache.VID_HASH_DIR,
            'items_to_censor': list( betaconfig.items_to_censor ),
        }
        self.videos = os.path.join( self.root, 'vids' ) + os.sep
        self.caches = os.path.join( self.root, 'cache' )
        os.makedirs( self.videos )
        os.makedirs( self.caches )
        betaconst.video_path_uncensored = self.videos
        betaconst.video_path_source_backup = self.videos
        bu_cache.VID_HASH_DIR = self.caches
        betaconfig.items_to_censor = [ 'exposed_breast' ]
        self.backend = bu_detector.selected_backend_name()

        self.clip = os.path.join( self.videos, 'clip.mp4' )
        subprocess.run(
            [ 'ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi',
              '-i', 'testsrc2=size=64x48:rate=10', '-t', '10', '-c:v', 'libx264',
              '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', self.clip ], check=True )
        self.file_hash = bu_hash.md5_for_file( self.clip, 16 )

    def tearDown( self ):
        betaconst.video_path_uncensored = self.saved['video_path_uncensored']
        betaconst.video_path_source_backup = self.saved['video_path_source_backup']
        bu_cache.VID_HASH_DIR = self.saved['VID_HASH_DIR']
        betaconfig.items_to_censor = self.saved['items_to_censor']
        shutil.rmtree( self.root, ignore_errors=True )

    def _write_cache( self, boxes, backend=None, suffix='' ):
        name = '%s-%s-3-640-10-0.12-dabc%s.gz'%(
            self.file_hash, backend or self.backend, suffix )
        bu_hash.write_json( boxes, os.path.join( self.caches, name ) )

    @staticmethod
    def _box( t, score=0.9, label='exposed_breast' ):
        return { 'x': 1, 'y': 1, 'w': 5, 'h': 5, 'class_id': label, 'score': score, 't': t }

    def _share( self, min_prob=0.30, hold=0.0 ):
        args = types.SimpleNamespace( backend=None, trigger_min_prob=min_prob, trigger_hold=hold )
        return betabench._reverse_trigger_share( args, _quiet_logger() )

    def test_share_is_triggered_time_over_runtime( self ):
        # Twenty firings at 10 fps, 2.0s to 3.9s: 2.0s of a 10s file.
        self._write_cache( [ self._box( 2.0 + i / 10.0 ) for i in range( 20 ) ] )
        share = self._share()
        self.assertEqual( share['files_with_whole_file_cache'], 1 )
        self.assertAlmostEqual( share['share_by_duration'], 0.2, delta=0.01 )

    def test_labels_outside_the_window_set_and_weak_scores_do_not_trigger( self ):
        self._write_cache( [ self._box( 1.0, score=0.2 ),
                             self._box( 5.0, label='covered_vulva' ) ] )
        self.assertEqual( self._share()['triggered_seconds'], 0.0 )

    def test_preview_slices_and_other_backends_are_not_read( self ):
        # A preview slice cannot say what share of the FILE is triggered,
        # and a second backend's boxes for the same file would double count.
        self._write_cache( [ self._box( 1.0 ) ], suffix='-preview0.0' )
        self._write_cache( [ self._box( 1.0 ) ], backend='some_other_backend' )
        self.assertEqual( self._share()['files_with_whole_file_cache'], 0 )


@unittest.skipUnless( _ffmpeg_available(), "ffmpeg is needed to build the clip" )
class TestReverseBenchEndToEnd( unittest.TestCase ):

    @classmethod
    def setUpClass( cls ):
        cls.tmp = tempfile.mkdtemp( prefix='betasuite-reverse-e2e-' )
        cls.clip = os.path.join( cls.tmp, 'clip.mp4' )
        subprocess.run(
            [ 'ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi',
              '-i', 'testsrc2=size=160x120:rate=30', '-t', '3', '-c:v', 'libx264',
              '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', cls.clip ], check=True )

    @classmethod
    def tearDownClass( cls ):
        shutil.rmtree( cls.tmp, ignore_errors=True )

    def run_bench( self ):
        args = types.SimpleNamespace( video=self.clip, seconds=0.5, workers=2, trials=1,
                                      repeats=1, backend=None, trigger_min_prob=0.30,
                                      trigger_hold=0.5 )
        return betabench.bench_reverse( args, _quiet_logger() )['reverse']

    def test_every_frame_style_and_both_pipeline_passes_are_measured( self ):
        result = self.run_bench()
        measured = { row['op'] for row in result['ops'] }
        for style in betabench.REVERSE_FRAME_STYLES:
            self.assertIn( 'frame layer: %s'%( betabench._style_label( style ), ), measured )
        passes = result['pipeline']['passes']
        self.assertEqual( len( passes ), 2 )
        for row in passes:
            self.assertIsNone( row['error'] )
            self.assertEqual( row['frames'], 2 * 15 )
            self.assertGreater( row['bytes'], 0 )

    def test_the_summary_is_bracketed_for_the_launcher( self ):
        # tools/bench/reverse_cost.sh replays the lines between these two
        # markers to the terminal; renaming either silently empties it.
        lines = []

        class Capture( logging.Handler ):
            def emit( self, record ):
                lines.append( record.getMessage() )

        logger = logging.getLogger( 'reverse_summary_test' )
        logger.handlers = [ Capture() ]
        logger.setLevel( logging.DEBUG )
        logger.propagate = False
        args = types.SimpleNamespace( video=self.clip, seconds=0.5, workers=1, trials=1,
                                      repeats=1, backend=None, trigger_min_prob=0.30,
                                      trigger_hold=0.5 )
        betabench.bench_reverse( args, logger )
        self.assertIn( betabench.REVERSE_SUMMARY_BEGIN, lines )
        self.assertIn( betabench.REVERSE_SUMMARY_END, lines )
        self.assertLess( lines.index( betabench.REVERSE_SUMMARY_BEGIN ),
                         lines.index( betabench.REVERSE_SUMMARY_END ) )

    def test_nothing_is_left_behind( self ):
        os.makedirs( betaconst.benchmark_dir, exist_ok=True )
        before = set( os.listdir( betaconst.benchmark_dir ) )
        self.run_bench()
        self.assertEqual( set( os.listdir( betaconst.benchmark_dir ) ) - before, set(),
                          "every encoded segment is scratch and must be removed" )


if __name__ == '__main__':
    unittest.main()
