"""
test_render_pipeline.py - chunk planning, parallelism and the
truncation guard.

The render is the part of BetaSuite most likely to fail halfway and
most expensive to redo, so the properties asserted here are about
trusting what is already on disk:

  A chunk is trusted only when ffmpeg succeeded, the container is
  readable, AND the frame count matches what was planned. Before 2.1 a
  chunk whose decode died mid-file was a perfectly valid video that was
  simply too short - it passed the container check, got promoted, and a
  later resume skipped it, silently truncating the output.

  Chunks are independent: each reconstructs its own live/pending box
  state from the shared, read-only box list. That is what makes running
  them concurrently safe, and it is asserted directly rather than
  assumed.
"""

import os
import shutil
import subprocess
import tempfile
import unittest

import betaconfig
import betautils_render as bu_render


def _ffmpeg_available():
    return shutil.which( 'ffmpeg' ) is not None


def _box( start, end, x=0 ):
    return {
        'start': start, 'end': end, 't': ( start + end ) / 2,
        'x': x, 'y': 0, 'w': 10, 'h': 10,
        'censor_style': { 'type': 'blur', 'method': 'gaussian', 'strength': 10 },
        'censor_shape': 'box', 'censor_sticker_seed': 0.5,
        'label': 'exposed_breast', 'score': 0.9,
    }


class TestChunkPlanning( unittest.TestCase ):

    def setUp( self ):
        self.original_chunk_seconds = betaconfig.render_chunk_seconds
        self.original_workers = getattr( betaconfig, 'render_workers', 0 )

    def tearDown( self ):
        betaconfig.render_chunk_seconds = self.original_chunk_seconds
        betaconfig.render_workers = self.original_workers

    def test_a_short_video_still_splits_enough_to_fill_the_workers( self ):
        # THE BUG: chunk length was render_chunk_seconds flat, so chunk
        # COUNT came from duration alone - and workers are capped at the
        # chunk count. On a real 10-file run, 4 files rendered at 1-4
        # workers instead of 6 for no reason but arithmetic: 22.9 minutes,
        # 10.3% of render time.
        betaconfig.render_chunk_seconds = 180
        betaconfig.render_workers = 6
        # 74s at 50fps - the real 4K file that rendered on ONE worker.
        plans = bu_render.plan_chunks( '/out/v.mkv', 50.0, int( 74*50 ),
                                       False, False, 0.0, 20 )
        self.assertGreaterEqual( len( plans ), 5,
            "a 74s video on a 6-worker box got 1 chunk, so 1 worker" )
        # 5 not 6: the 15s floor caps a 74s video at 5 chunks, which is the
        # floor working, not the fix falling short. 5 workers against 1 is
        # the win being claimed.
        self.assertGreaterEqual( bu_render.resolve_worker_count( len( plans ) ), 5 )

    def test_a_long_video_keeps_the_configured_chunk_length( self ):
        # The fix must only ever SHORTEN chunks, and only when a fixed
        # length would starve workers. A long file already has plenty.
        betaconfig.render_chunk_seconds = 180
        betaconfig.render_workers = 6
        plans = bu_render.plan_chunks( '/out/v.mkv', 30.0, int( 2798*30 ),
                                       False, False, 0.0, 20 )
        lengths = [ plan.end_frame - plan.start_frame for plan in plans ]
        self.assertEqual( max( lengths ), int( round( 180*30 ) ),
                          "configured chunk length must be untouched here" )

    def test_a_very_short_clip_is_not_split_into_slivers( self ):
        # Below the floor, per-chunk ffmpeg startup and the concat would
        # cost more than the parallelism wins.
        betaconfig.render_chunk_seconds = 180
        betaconfig.render_workers = 6
        plans = bu_render.plan_chunks( '/out/v.mkv', 30.0, int( 10*30 ),
                                       False, False, 0.0, 20 )
        self.assertEqual( len( plans ), 1, "a 10s clip needs no splitting" )

    def test_the_split_never_loses_or_duplicates_a_frame( self ):
        # The fix changes chunk boundaries, which is exactly where a
        # render silently truncates or double-encodes.
        betaconfig.render_chunk_seconds = 180
        betaconfig.render_workers = 6
        for fps, secs in ( ( 50.0, 74 ), ( 60.0, 210 ), ( 24.0, 606 ),
                           ( 30.0, 2798 ), ( 25.0, 240 ) ):
            frames = int( secs*fps )
            plans = bu_render.plan_chunks( '/out/v.mkv', fps, frames,
                                           False, False, 0.0, 20 )
            self.assertEqual( plans[0].start_frame, 0, ( fps, secs ) )
            self.assertEqual( plans[-1].end_frame, frames, ( fps, secs ) )
            for earlier, later in zip( plans, plans[1:] ):
                self.assertEqual( earlier.end_frame, later.start_frame,
                                  ( fps, secs ) )
            self.assertTrue( plans[-1].is_last )
            self.assertEqual( sum( p.end_frame - p.start_frame for p in plans ),
                              frames, ( fps, secs ) )

    def test_a_single_worker_box_is_left_alone( self ):
        betaconfig.render_chunk_seconds = 180
        betaconfig.render_workers = 1
        plans = bu_render.plan_chunks( '/out/v.mkv', 30.0, int( 200*30 ),
                                       False, False, 0.0, 20 )
        self.assertEqual( len( plans ), 2, "200s / 180s, unchanged" )

    def test_chunks_cover_the_whole_range_without_gaps_or_overlaps( self ):
        betaconfig.render_chunk_seconds = 10
        plans = bu_render.plan_chunks( '/out/v.mkv', 25, 1000, False, False, 0.0, 20 )
        self.assertEqual( plans[0].start_frame, 0 )
        self.assertEqual( plans[-1].end_frame, 1000 )
        for previous, following in zip( plans, plans[1:] ):
            self.assertEqual( previous.end_frame, following.start_frame )

    def test_only_the_last_chunk_is_marked_last( self ):
        betaconfig.render_chunk_seconds = 10
        plans = bu_render.plan_chunks( '/out/v.mkv', 25, 1000, False, False, 0.0, 20 )
        self.assertTrue( plans[-1].is_last )
        self.assertFalse( any( plan.is_last for plan in plans[:-1] ) )

    def test_chunking_disabled_gives_exactly_one_chunk( self ):
        betaconfig.render_chunk_seconds = 0
        plans = bu_render.plan_chunks( '/out/v.mkv', 25, 1000, False, False, 0.0, 20 )
        self.assertEqual( len( plans ), 1 )
        self.assertEqual( plans[0].expected_frames, 1000 )

    def test_preview_mode_always_uses_one_chunk( self ):
        betaconfig.render_chunk_seconds = 10
        plans = bu_render.plan_chunks( '/out/v.mkv', 25, 10000, True, False, 4.0, 20 )
        self.assertEqual( len( plans ), 1 )
        self.assertEqual( plans[0].start_frame, 100 )
        self.assertEqual( plans[0].end_frame, 100 + 500 )

    def test_a_preview_window_is_clamped_to_the_end_of_the_file( self ):
        betaconfig.render_chunk_seconds = 0
        plans = bu_render.plan_chunks( '/out/v.mkv', 25, 200, True, False, 4.0, 60 )
        self.assertEqual( plans[0].end_frame, 200 )

    def test_a_whole_file_preview_covers_the_whole_file( self ):
        betaconfig.render_chunk_seconds = 0
        plans = bu_render.plan_chunks( '/out/v.mkv', 25, 500, True, True, 0.0, 20 )
        self.assertEqual( plans[0].start_frame, 0 )
        self.assertEqual( plans[0].end_frame, 500 )

    def test_a_zero_length_range_still_produces_one_plan( self ):
        betaconfig.render_chunk_seconds = 10
        plans = bu_render.plan_chunks( '/out/v.mkv', 25, 0, False, False, 0.0, 20 )
        self.assertEqual( len( plans ), 1 )

    def test_chunk_paths_are_distinct_and_ordered( self ):
        betaconfig.render_chunk_seconds = 5
        plans = bu_render.plan_chunks( '/out/v.mkv', 25, 1000, False, False, 0.0, 20 )
        paths = [ plan.path for plan in plans ]
        self.assertEqual( len( set( paths ) ), len( paths ) )
        self.assertEqual( paths, sorted( paths ) )


class TestWorkerResolution( unittest.TestCase ):

    def setUp( self ):
        self.original = getattr( betaconfig, 'render_workers', 0 )

    def tearDown( self ):
        betaconfig.render_workers = self.original

    def test_auto_never_exceeds_the_chunk_count( self ):
        betaconfig.render_workers = 0
        self.assertEqual( bu_render.resolve_worker_count( 1 ), 1 )
        self.assertLessEqual( bu_render.resolve_worker_count( 3 ), 3 )

    def test_auto_is_always_at_least_one( self ):
        betaconfig.render_workers = 0
        self.assertGreaterEqual( bu_render.resolve_worker_count( 8 ), 1 )

    def test_an_explicit_setting_is_honoured_but_capped_by_chunk_count( self ):
        betaconfig.render_workers = 6
        self.assertEqual( bu_render.resolve_worker_count( 12 ), 6 )
        self.assertEqual( bu_render.resolve_worker_count( 2 ), 2 )

    def test_encoder_threads_divide_the_box_between_workers( self ):
        # Each worker runs both a Python compositing loop and an x264
        # encoder; oversubscribing makes every chunk slower without
        # finishing any sooner.
        self.assertGreaterEqual( bu_render._encoder_threads( 1 ), 1 )
        self.assertLessEqual( bu_render._encoder_threads( 8 ),
                              bu_render._encoder_threads( 1 ) )


class TestBoxWindowing( unittest.TestCase ):
    """
    Each chunk must reconstruct exactly the box state a single
    continuous pass would have had at its start frame.
    """

    def setUp( self ):
        self.boxes = [
            _box( 0.0, 1.0, x=0 ),     # already over by t=2
            _box( 1.5, 3.0, x=10 ),    # straddles t=2
            _box( 2.0, 4.0, x=20 ),    # starts exactly at t=2
            _box( 5.0, 6.0, x=30 ),    # starts later
        ]

    def test_live_boxes_are_the_ones_spanning_the_boundary( self ):
        live, _pending = bu_render.boxes_for_chunk( self.boxes, 2.0 )
        self.assertEqual( [ box['x'] for box in live ], [ 10, 20 ] )

    def test_pending_boxes_are_the_ones_that_start_later( self ):
        _live, pending = bu_render.boxes_for_chunk( self.boxes, 2.0 )
        self.assertEqual( [ box['x'] for box in pending ], [ 30 ] )

    def test_every_box_lands_in_exactly_one_bucket_or_neither( self ):
        live, pending = bu_render.boxes_for_chunk( self.boxes, 2.0 )
        self.assertEqual( len( set( id( b ) for b in live )
                              & set( id( b ) for b in pending ) ), 0 )

    def test_chunk_zero_has_nothing_live_and_everything_pending_or_current( self ):
        live, pending = bu_render.boxes_for_chunk( self.boxes, 0.0 )
        self.assertEqual( len( live ) + len( pending ), len( self.boxes ) )

    def test_the_caller_s_boxes_are_not_copied_or_mutated( self ):
        # Views, not copies: the box dicts are shared read-only across
        # every worker, which is only safe because nothing on the render
        # path writes to them.
        live, pending = bu_render.boxes_for_chunk( self.boxes, 2.0 )
        for box in live + pending:
            self.assertIn( id( box ), [ id( original ) for original in self.boxes ] )


class TestChunkCompletionChecks( unittest.TestCase ):
    """
    The truncation guard: a valid-but-short chunk must not be trusted.
    """

    def setUp( self ):
        self.tmpdir = tempfile.mkdtemp( prefix='betasuite-render-' )
        self.plan = bu_render.ChunkPlan(
            0, 0, 100, os.path.join( self.tmpdir, 'v.part0000.mkv' ), is_last=False )

    def tearDown( self ):
        shutil.rmtree( self.tmpdir, ignore_errors=True )

    def test_a_missing_chunk_is_not_complete( self ):
        self.assertFalse( bu_render.chunk_is_complete( self.plan ) )

    @unittest.skipUnless( _ffmpeg_available(), "ffmpeg is needed to build the chunk" )
    def test_a_chunk_with_no_metadata_is_not_trusted( self ):
        _build_tiny_video( self.plan.path, frames=10 )
        self.assertFalse( bu_render.chunk_is_complete( self.plan ),
            "a chunk whose frame count was never recorded must be re-rendered, not trusted" )

    @unittest.skipUnless( _ffmpeg_available(), "ffmpeg is needed to build the chunk" )
    def test_a_short_chunk_is_not_trusted( self ):
        _build_tiny_video( self.plan.path, frames=10 )
        bu_render._write_chunk_metadata( self.plan, frames_written=42 )
        self.assertFalse( bu_render.chunk_is_complete( self.plan ),
            "a chunk that is valid but short is exactly the silent-truncation case" )

    @unittest.skipUnless( _ffmpeg_available(), "ffmpeg is needed to build the chunk" )
    def test_a_complete_chunk_is_trusted( self ):
        _build_tiny_video( self.plan.path, frames=10 )
        bu_render._write_chunk_metadata( self.plan, frames_written=self.plan.expected_frames )
        self.assertTrue( bu_render.chunk_is_complete( self.plan ) )

    @unittest.skipUnless( _ffmpeg_available(), "ffmpeg is needed to build the chunk" )
    def test_frame_count_verification_can_be_disabled( self ):
        _build_tiny_video( self.plan.path, frames=10 )
        self.assertTrue( bu_render.chunk_is_complete( self.plan, verify_frame_counts=False ) )


def _build_tiny_video( path, frames=10, width=32, height=32 ):
    """A tiny valid H.264 file, for the container-level checks."""
    os.makedirs( os.path.dirname( path ), exist_ok=True )
    subprocess.run(
        [ 'ffmpeg', '-y', '-loglevel', 'error',
          '-f', 'lavfi', '-i', 'color=c=black:s=%dx%d:d=1'%(width, height),
          '-frames:v', str( frames ), '-c:v', 'libx264', '-preset', 'ultrafast',
          '-pix_fmt', 'yuv420p', path ],
        check=True )


@unittest.skipUnless( _ffmpeg_available(), "ffmpeg is needed for a real render" )
class TestEndToEndRender( unittest.TestCase ):
    """
    A real, if tiny, render: decode a synthetic clip, censor it, encode
    chunks, concatenate them and mux. Exercised both serially and in
    parallel, because the whole point of the parallel path is that it
    produces the same thing.
    """

    @classmethod
    def setUpClass( cls ):
        import betautils_video as bu_video
        cls.tmpdir = tempfile.mkdtemp( prefix='betasuite-render-e2e-' )
        cls.source_path = os.path.join( cls.tmpdir, 'source.mp4' )
        subprocess.run(
            [ 'ffmpeg', '-y', '-loglevel', 'error',
              '-f', 'lavfi', '-i', 'testsrc2=size=128x96:rate=10:duration=4',
              '-f', 'lavfi', '-i', 'sine=frequency=440:duration=4',
              '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p',
              '-c:a', 'aac', '-shortest', cls.source_path ],
            check=True )
        cls.bu_video = bu_video

    @classmethod
    def tearDownClass( cls ):
        shutil.rmtree( cls.tmpdir, ignore_errors=True )

    def setUp( self ):
        self.original_workers = getattr( betaconfig, 'render_workers', 0 )
        self.original_chunk_seconds = betaconfig.render_chunk_seconds

    def tearDown( self ):
        betaconfig.render_workers = self.original_workers
        betaconfig.render_chunk_seconds = self.original_chunk_seconds

    def _render( self, tag, workers, chunk_seconds ):
        betaconfig.render_workers = workers
        betaconfig.render_chunk_seconds = chunk_seconds
        source = self.bu_video.open_video_source( self.source_path, 'testhash' )
        try:
            boxes = [ _box( 0.0, 10.0, x=20 ) ]
            intermediate = os.path.join( self.tmpdir, '%s.video.mkv'%(tag) )
            final = os.path.join( self.tmpdir, '%s.mp4'%(tag) )
            stats = bu_render.render_video(
                source, boxes, intermediate, final, 'ultrafast',
                False, False, 0.0, 20 )
            return final, stats
        finally:
            source.release()

    def test_serial_render_produces_a_valid_output_with_audio( self ):
        final, stats = self._render( 'serial', workers=1, chunk_seconds=0 )
        self.assertTrue( os.path.exists( final ) )
        self.assertTrue( self.bu_video.probe_video_ok( final ) )
        self.assertTrue( self.bu_video.video_file_has_audio( final ),
            "the source audio must survive the mux" )
        self.assertEqual( stats['chunks'], 1 )

    def test_parallel_chunked_render_produces_a_valid_output( self ):
        final, stats = self._render( 'parallel', workers=3, chunk_seconds=1 )
        self.assertTrue( os.path.exists( final ) )
        self.assertTrue( self.bu_video.probe_video_ok( final ) )
        self.assertGreater( stats['chunks'], 1 )

    def _render_boxes( self, tag, boxes, workers, chunk_seconds ):
        betaconfig.render_workers = workers
        betaconfig.render_chunk_seconds = chunk_seconds
        source = self.bu_video.open_video_source( self.source_path, 'testhash' )
        try:
            intermediate = os.path.join( self.tmpdir, '%s.video.mkv'%(tag) )
            final = os.path.join( self.tmpdir, '%s.mp4'%(tag) )
            stats = bu_render.render_video(
                source, boxes, intermediate, final, 'ultrafast',
                False, False, 0.0, 20 )
            return final, stats
        finally:
            source.release()

    def _moving_track( self ):
        # A real multi-sample track: 9fps sampling on a 10fps source, each
        # sample live for longer than the interval, so consecutive samples
        # overlap exactly as they do on real footage. The single static box
        # the other tests use never engages the interpolation path at all.
        boxes = []
        for index in range( 12 ):
            time = index / 9.0
            box = _box( time - 0.085, time + 0.085, x=10 + index*4 )
            box['t'] = time
            box['_track_id'] = 0
            box['_vid_w'] = 128
            box['_vid_h'] = 96
            boxes.append( box )
        return sorted( boxes, key=lambda entry: entry['start'] )

    def _read_frames( self, path ):
        import cv2
        capture = cv2.VideoCapture( path )
        frames = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None: break
                frames.append( frame )
        finally:
            capture.release()
        return frames

    def test_interpolation_changes_the_rendered_pixels( self ):
        # THE test for this feature. The other end-to-end tests only check
        # that ffmpeg produced a playable file, which stays true even when
        # interpolation is bypassed entirely: both of them SURVIVED a
        # mutation that fed the frame loop the un-interpolated box list.
        # A test that cannot fail when the feature is disconnected is worse
        # than no test, so this one compares rendered output directly.
        #
        # Measured on this fixture: 37 of 40 frames differ, total absolute
        # difference ~65k. The first three frames match because the track's
        # first sample has no earlier sample to slide from.
        import numpy

        original = getattr( betaconfig, 'render_motion_interpolation', True )
        boxes = self._moving_track()
        try:
            betaconfig.render_motion_interpolation = True
            on_path, stats = self._render_boxes(
                'motion-on', boxes, workers=1, chunk_seconds=0 )
            betaconfig.render_motion_interpolation = False
            off_path, _ = self._render_boxes(
                'motion-held', boxes, workers=1, chunk_seconds=0 )
        finally:
            betaconfig.render_motion_interpolation = original

        self.assertGreater( stats['frames_written'], 0 )
        on_frames = self._read_frames( on_path )
        off_frames = self._read_frames( off_path )
        self.assertGreater( len( on_frames ), 20, "too few frames to judge" )
        self.assertEqual( len( on_frames ), len( off_frames ) )

        differing = sum(
            1 for left, right in zip( on_frames, off_frames )
            if numpy.abs( left.astype( int ) - right.astype( int ) ).sum() > 0 )
        self.assertGreater(
            differing, len( on_frames ) // 2,
            "interpolated and held renders should differ on most frames; only "
            "%d of %d differed. If this is zero the frame loop is not using "
            "the interpolated boxes at all."%( differing, len( on_frames ) ) )

    def test_a_moving_track_renders_with_interpolation_on( self ):
        original = getattr( betaconfig, 'render_motion_interpolation', True )
        betaconfig.render_motion_interpolation = True
        try:
            final, stats = self._render_boxes(
                'motion-valid', self._moving_track(), workers=1, chunk_seconds=0 )
            self.assertTrue( self.bu_video.probe_video_ok( final ) )
            self.assertGreater( stats['frames_written'], 0 )
        finally:
            betaconfig.render_motion_interpolation = original

    def test_interpolation_survives_a_parallel_chunked_render( self ):
        # The index is built once and shared across workers, so a chunked
        # parallel render is where a mutation or a per-chunk state bug would
        # show up as chunk-order-dependent output.
        original = getattr( betaconfig, 'render_motion_interpolation', True )
        betaconfig.render_motion_interpolation = True
        try:
            boxes = self._moving_track()
            serial, _ = self._render_boxes(
                'motion-serial', boxes, workers=1, chunk_seconds=0 )
            parallel, stats = self._render_boxes(
                'motion-parallel', boxes, workers=3, chunk_seconds=1 )
            self.assertGreater( stats['chunks'], 1 )
            self.assertTrue( self.bu_video.probe_video_ok( parallel ) )
            serial_info = self.bu_video.probe_stream_info( serial )
            parallel_info = self.bu_video.probe_stream_info( parallel )
            self.assertAlmostEqual( float( serial_info.get( 'duration', 0 ) ),
                                    float( parallel_info.get( 'duration', 0 ) ),
                                    delta=0.35 )
        finally:
            betaconfig.render_motion_interpolation = original

    def test_disabling_interpolation_still_renders( self ):
        original = getattr( betaconfig, 'render_motion_interpolation', True )
        betaconfig.render_motion_interpolation = False
        try:
            final, _ = self._render_boxes(
                'motion-off', self._moving_track(), workers=1, chunk_seconds=0 )
            self.assertTrue( self.bu_video.probe_video_ok( final ) )
        finally:
            betaconfig.render_motion_interpolation = original

    def test_parallel_and_serial_renders_agree_on_duration( self ):
        serial_final, _ = self._render( 'agree-serial', workers=1, chunk_seconds=0 )
        parallel_final, _ = self._render( 'agree-parallel', workers=3, chunk_seconds=1 )
        serial_info = self.bu_video.probe_stream_info( serial_final )
        parallel_info = self.bu_video.probe_stream_info( parallel_final )
        self.assertEqual( serial_info.get( 'width' ), parallel_info.get( 'width' ) )
        self.assertEqual( serial_info.get( 'height' ), parallel_info.get( 'height' ) )
        self.assertAlmostEqual( float( serial_info.get( 'duration', 0 ) ),
                                float( parallel_info.get( 'duration', 0 ) ), delta=0.35 )

    def test_chunk_files_are_cleaned_up_after_a_successful_concat( self ):
        final, _stats = self._render( 'cleanup', workers=2, chunk_seconds=1 )
        self.assertTrue( os.path.exists( final ) )
        leftovers = [ name for name in os.listdir( self.tmpdir )
                      if '.part' in name or name.endswith( '.concat.txt' ) ]
        self.assertEqual( leftovers, [] )

    def test_the_intermediate_is_removed_once_the_mux_succeeds( self ):
        self._render( 'intermediate', workers=1, chunk_seconds=0 )
        self.assertFalse(
            os.path.exists( os.path.join( self.tmpdir, 'intermediate.video.mkv' ) ) )

    def test_a_second_render_reuses_the_finished_output( self ):
        final, _first = self._render( 'reuse', workers=1, chunk_seconds=0 )
        _final, second = self._render( 'reuse', workers=1, chunk_seconds=0 )
        self.assertEqual( second['chunks_rendered'], 0 )
        self.assertTrue( os.path.exists( final ) )

    def test_the_box_list_is_unchanged_after_a_parallel_render( self ):
        import copy
        betaconfig.render_workers = 3
        betaconfig.render_chunk_seconds = 1
        source = self.bu_video.open_video_source( self.source_path, 'testhash' )
        try:
            boxes = [ _box( 0.0, 1.5, x=10 ), _box( 1.0, 3.0, x=40 ) ]
            before = copy.deepcopy( boxes )
            bu_render.render_video(
                source, boxes,
                os.path.join( self.tmpdir, 'nomutate.video.mkv' ),
                os.path.join( self.tmpdir, 'nomutate.mp4' ),
                'ultrafast', False, False, 0.0, 20 )
            self.assertEqual( boxes, before,
                "a parallel render mutated the shared box list, which makes its output "
                "depend on chunk scheduling order" )
        finally:
            source.release()


if __name__ == '__main__':
    unittest.main()
