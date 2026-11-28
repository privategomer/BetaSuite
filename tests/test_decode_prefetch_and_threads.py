"""
test_decode_prefetch_and_threads.py - frames decoded ahead of the model
(detection_prefetch_frames) and decoder threads per render worker
(render_decode_threads).

Both change WHEN and HOW frames are decoded, never WHICH: the tests here
hold the frames, their order, the detections and the render output to
exactly what the plain path produces.
"""

import os
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np

import betaconfig
import betautils_render as bu_render
import betautils_video as bu_video


def _ffmpeg():
    return shutil.which( 'ffmpeg' ) is not None


def _clip( directory, seconds=4, rate=10, size='96x64' ):
    path = os.path.join( directory, 'clip.mp4' )
    subprocess.run( [ 'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                      '-f', 'lavfi', '-i', 'testsrc2=size=%s:rate=%d:duration=%d'%( size, rate, seconds ),
                      '-c:v', 'libx264', '-preset', 'ultrafast', '-g', '25', path ], check=True )
    return path


class TestPrefetched( unittest.TestCase ):

    def test_same_items_same_order( self ):
        items = list( range( 50 ) )
        for depth in ( 1, 3, 64 ):
            with bu_video.prefetched( iter( items ), depth ) as stream:
                self.assertEqual( list( stream ), items, depth )

    def test_depth_zero_is_the_iterable_itself( self ):
        stream = bu_video.prefetched( iter( [ 1, 2 ] ), 0 )
        self.assertNotIsInstance( stream, bu_video.Prefetched )
        self.assertEqual( list( stream ), [ 1, 2 ] )
        stream.close()

    def test_an_error_arrives_where_it_happened( self ):
        def source():
            yield 1
            yield 2
            raise ValueError( 'decoder broke' )
        seen = []
        with self.assertRaisesRegex( ValueError, 'decoder broke' ):
            with bu_video.prefetched( source(), 4 ) as stream:
                for item in stream:
                    seen.append( item )
        self.assertEqual( seen, [ 1, 2 ] )

    def test_it_reads_ahead_only_as_far_as_the_depth( self ):
        produced = []
        def source():
            for index in range( 100 ):
                produced.append( index )
                yield index
        stream = bu_video.prefetched( source(), 3 )
        first = next( stream )
        time.sleep( 0.2 )
        # One handed over, three queued, one more waiting to be queued.
        self.assertEqual( first, 0 )
        self.assertLessEqual( len( produced ), 5 )
        stream.close()

    def test_close_stops_the_producer_before_returning( self ):
        # The detection loop releases its VideoCapture right after close();
        # a producer still decoding into it would be a native crash.
        inside = threading.Event()
        def source():
            for index in range( 1000 ):
                inside.set()
                yield index
        stream = bu_video.prefetched( source(), 2 )
        next( stream )
        inside.wait( 1 )
        stream.close()
        self.assertFalse( stream._thread.is_alive() )
        stream.close()   # twice is fine
        with self.assertRaises( StopIteration ):
            next( stream )

    def test_leaving_the_loop_early_closes_it( self ):
        with bu_video.prefetched( iter( range( 1000 ) ), 4 ) as stream:
            for item in stream:
                if item == 3:
                    break
        self.assertFalse( stream._thread.is_alive() )


@unittest.skipUnless( _ffmpeg(), 'needs ffmpeg' )
class TestRealFrames( unittest.TestCase ):

    def setUp( self ):
        self.dir = tempfile.mkdtemp( prefix='betasuite-prefetch-' )
        self.addCleanup( shutil.rmtree, self.dir, True )
        self.path = _clip( self.dir )

    def sampled( self, depth, threads=None, **kwargs ):
        capture = bu_video.open_capture_with_threads( self.path, threads )
        stream = bu_video.prefetched( bu_video.iter_sampled_frames( capture, 10.0, 3, **kwargs ), depth )
        try:
            return [ ( index, t, frame.copy() ) for index, t, frame in stream ]
        finally:
            stream.close()
            capture.release()

    def assertSameFrames( self, a, b ):
        self.assertEqual( [ ( i, t ) for i, t, _f in a ], [ ( i, t ) for i, t, _f in b ] )
        for ( _i, _t, x ), ( _j, _u, y ) in zip( a, b ):
            self.assertTrue( np.array_equal( x, y ) )

    def test_prefetched_sampling_matches_plain_sampling( self ):
        plain = self.sampled( 0 )
        self.assertGreater( len( plain ), 5 )
        self.assertSameFrames( self.sampled( 4 ), plain )
        self.assertSameFrames( self.sampled( 4, offset_seconds=1.0, max_seconds=2.0 ),
                               self.sampled( 0, offset_seconds=1.0, max_seconds=2.0 ) )

    def test_decoder_threads_do_not_change_the_frames( self ):
        if not bu_video.capture_threads_supported():
            self.skipTest( 'this OpenCV cannot set decoder threads' )
        plain = self.sampled( 0 )
        self.assertSameFrames( self.sampled( 0, threads=1 ), plain )
        self.assertSameFrames( self.sampled( 0, threads=3 ), plain )

    def test_a_capture_the_parameter_cannot_open_falls_back( self ):
        import cv2
        real = cv2.VideoCapture
        calls = []
        def fake( *args ):
            calls.append( args )
            capture = real( *args ) if len( args ) == 1 else mock.Mock( isOpened=lambda: False )
            return capture
        with mock.patch.object( cv2, 'VideoCapture', side_effect=fake ):
            capture = bu_video.open_capture_with_threads( self.path, 2 )
        try:
            self.assertTrue( capture.isOpened() )
            self.assertEqual( len( calls ), 2 if bu_video.capture_threads_supported() else 1 )
        finally:
            capture.release()

    def test_detection_finds_the_same_boxes_either_way( self ):
        # The real detection loop, with a stand-in model that "detects"
        # each frame's mean colour, so a frame out of order or missing
        # changes the result.
        import betatv
        import betautils_detector as bu_detector
        class Model( object ):
            calls = []
            @staticmethod
            def raw_boxes_for_imgs( frames, size, session, times ):
                Model.calls.append( len( frames ) )
                return [ { 'x': 0, 'y': 0, 'w': 1, 'h': 1, 'class_id': 'exposed_breast',
                           'score': float( frame.mean() ), 't': t, 'size': size }
                         for frame, t in zip( frames, times ) ]
        source = bu_video.VideoSource( self.path, self.path, None, 10.0, 40, 96, 64, False )
        results = []
        saved = getattr( betaconfig, 'detection_prefetch_frames', 0 )
        self.addCleanup( setattr, betaconfig, 'detection_prefetch_frames', saved )
        for depth in ( 0, 8 ):
            betaconfig.detection_prefetch_frames = depth
            cache = os.path.join( self.dir, 'boxes_%d.json'%( depth, ) )
            with mock.patch.object( bu_detector, 'get_detector', return_value=Model ), \
                 mock.patch.object( bu_detector, 'get_nn_batch_size', return_value=3 ), \
                 mock.patch( 'betautils_cache_paths.box_hash_path_for', return_value=cache ):
                results.append( betatv.detect_boxes_for_size(
                    source, 'clip.mp4', 320, None, 'abc', '', 0.0, None, mock.Mock(), sample_fps=5 ) )
        self.assertEqual( len( results[0] ), 20 )
        self.assertEqual( results[0], results[1] )


class TestRenderDecodeThreads( unittest.TestCase ):

    def setUp( self ):
        saved = getattr( betaconfig, 'render_decode_threads', None )
        self.addCleanup( setattr, betaconfig, 'render_decode_threads', saved )

    def test_the_setting( self ):
        with mock.patch.object( bu_render.os, 'cpu_count', return_value=16 ):
            betaconfig.render_decode_threads = None
            self.assertIsNone( bu_render.decode_threads( 8 ) )
            betaconfig.render_decode_threads = 0
            self.assertEqual( bu_render.decode_threads( 8 ), 2 )
            self.assertEqual( bu_render.decode_threads( 3 ), 5 )
            self.assertEqual( bu_render.decode_threads( 32 ), 1 )
            betaconfig.render_decode_threads = 6
            self.assertEqual( bu_render.decode_threads( 8 ), 6 )

    @unittest.skipUnless( _ffmpeg(), 'needs ffmpeg' )
    def test_each_chunk_opens_its_capture_with_them_and_renders_the_same( self ):
        directory = tempfile.mkdtemp( prefix='betasuite-decode-threads-' )
        self.addCleanup( shutil.rmtree, directory, True )
        path = _clip( directory, seconds=2 )
        outputs = []
        for threads in ( None, 2 ):
            source = bu_video.VideoSource( path, path, None, 10.0, 20, 96, 64, False )
            opened = []
            real_open = source.open_capture
            def spy( threads_arg=None, real_open=real_open, opened=opened ):
                opened.append( threads_arg )
                return real_open( threads_arg )
            source.open_capture = spy
            plan = bu_render.ChunkPlan( 0, 0, 20, os.path.join( directory, 'chunk_%s.mkv'%( threads, ) ),
                                        is_last=True )
            written = bu_render.render_one_chunk( plan, source, [], 'ultrafast', 1, 0, 0, None,
                                                  mock.Mock(), decoder_threads=threads )
            self.assertEqual( written, 20 )
            self.assertEqual( opened, [ threads ] )
            outputs.append( plan.path )
        frames = []
        for output in outputs:
            capture = bu_video.open_capture_with_threads( output )
            chunk = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                chunk.append( frame )
            capture.release()
            frames.append( chunk )
        self.assertEqual( len( frames[0] ), 20 )
        self.assertTrue( all( np.array_equal( a, b ) for a, b in zip( *frames ) ) )


class TestWiring( unittest.TestCase ):
    """The settings reach the code that decodes."""

    def test_a_source_hands_its_threads_to_the_capture( self ):
        source = bu_video.VideoSource( 'a.mp4', 'decoded.mkv', None, 10.0, 20, 8, 8, False )
        with mock.patch.object( bu_video, 'open_capture_with_threads', return_value='cap' ) as opened:
            self.assertEqual( source.open_capture( 3 ), 'cap' )
            source.open_capture()
        self.assertEqual( opened.call_args_list, [ mock.call( 'decoded.mkv', 3 ), mock.call( 'decoded.mkv', None ) ] )

    def test_detection_prefetches_by_the_setting_and_closes_before_releasing( self ):
        import betatv
        import betautils_detector as bu_detector
        order = []
        class Stream( object ):
            def __init__( self, items ):
                self.items = iter( items )
            def __iter__( self ):
                return self.items
            def close( self ):
                order.append( 'close' )
        capture = mock.Mock()
        capture.release.side_effect = lambda: order.append( 'release' )
        source = mock.Mock( vid_fps=10.0, num_frames=20 )
        source.open_capture.return_value = capture
        frames = [ ( i, i / 5.0, np.zeros( ( 4, 4, 3 ), np.uint8 ) ) for i in range( 4 ) ]
        depths = []
        def fake_prefetched( iterable, depth ):
            depths.append( depth )
            return Stream( list( iterable ) )
        model = mock.Mock()
        model.raw_boxes_for_imgs.side_effect = lambda imgs, *a: []
        saved = getattr( betaconfig, 'detection_prefetch_frames', 0 )
        self.addCleanup( setattr, betaconfig, 'detection_prefetch_frames', saved )
        betaconfig.detection_prefetch_frames = 5
        directory = tempfile.mkdtemp( prefix='betasuite-wiring-' )
        self.addCleanup( shutil.rmtree, directory, True )
        with mock.patch.object( bu_video, 'prefetched', side_effect=fake_prefetched ), \
             mock.patch.object( bu_video, 'iter_sampled_frames', return_value=iter( frames ) ), \
             mock.patch.object( bu_detector, 'get_detector', return_value=model ), \
             mock.patch( 'betautils_cache_paths.box_hash_path_for',
                         return_value=os.path.join( directory, 'b.json' ) ):
            betatv.detect_boxes_for_size( source, 'x.mp4', 320, None, 'abc', '', 0.0, None,
                                          mock.Mock(), sample_fps=5 )
        self.assertEqual( depths, [ 5 ] )
        self.assertEqual( order, [ 'close', 'release' ] )

    @unittest.skipUnless( _ffmpeg(), 'needs ffmpeg' )
    def test_a_render_gives_every_chunk_the_shared_out_threads( self ):
        import betautils_log as bu_log
        directory = tempfile.mkdtemp( prefix='betasuite-render-threads-' )
        self.addCleanup( shutil.rmtree, directory, True )
        path = os.path.join( directory, 'source.mp4' )
        subprocess.run( [ 'ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi',
                          '-i', 'testsrc2=size=64x48:rate=10:duration=40', '-c:v', 'libx264',
                          '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', path ], check=True )
        source = bu_video.open_video_source( path, 'threadshash' )
        self.addCleanup( source.release )
        names = ( 'render_workers', 'render_chunk_seconds', 'render_decode_threads',
                  'render_worker_mb_per_megapixel' )
        saved = { name: getattr( betaconfig, name, None ) for name in names }
        self.addCleanup( lambda: [ setattr( betaconfig, n, v ) for n, v in saved.items() ] )
        betaconfig.render_workers, betaconfig.render_chunk_seconds = 2, 15
        betaconfig.render_decode_threads, betaconfig.render_worker_mb_per_megapixel = 0, None
        seen = []
        real = bu_render.render_one_chunk
        def spy( *args, **kwargs ):
            seen.append( kwargs.get( 'decoder_threads' ) )
            return real( *args, **kwargs )
        with mock.patch.object( bu_render, 'render_one_chunk', side_effect=spy ), \
             mock.patch.object( bu_render.os, 'cpu_count', return_value=6 ):
            bu_render.render_video( source, [], os.path.join( directory, 'v.mkv' ),
                                    os.path.join( directory, 'v.mp4' ), 'ultrafast', False, False,
                                    0.0, 20, bu_log.get_logger() )
        self.assertGreaterEqual( len( seen ), 2 )
        self.assertEqual( set( seen ), { 3 } )


class TestStructureScanWorkers( unittest.TestCase ):

    def test_the_setting_replaces_the_guess( self ):
        import sys
        sys.path.insert( 0, os.path.join( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ),
                                          'tools', 'bench' ) )
        import betabench
        saved = getattr( betaconfig, 'structure_scan_workers', 0 )
        self.addCleanup( setattr, betaconfig, 'structure_scan_workers', saved )
        with mock.patch.object( betabench.os, 'cpu_count', return_value=16 ):
            betaconfig.structure_scan_workers = 0
            self.assertEqual( betabench._default_scan_workers(), 4 )
            betaconfig.structure_scan_workers = 9
            self.assertEqual( betabench._default_scan_workers(), 9 )


if __name__ == '__main__':
    unittest.main()
