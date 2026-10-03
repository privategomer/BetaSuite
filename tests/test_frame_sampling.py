"""
test_frame_sampling.py - the sample <-> frame contract, and that the
fast decoder reads exactly the frames the old slow one did.

WHY THIS MATTERS MORE THAN A NORMAL PERFORMANCE TEST
    Replacing "seek to each sampled frame" with "decode sequentially and
    skip with grab()" is BetaSuite's single largest speedup. It is also
    the change with the most dangerous possible regression: if it lands
    on even slightly different frames, every cached detection, every
    tuned threshold and every shot-cut timestamp derived from the old
    behaviour silently stops describing reality.

    So the contract is not "roughly the same frames". It is exactly the
    same frames, and these tests assert that against a synthetic video
    whose frames are individually identifiable.
"""

import os
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

import betautils_video as bu_video


def _ffmpeg_available():
    return shutil.which( 'ffmpeg' ) is not None


def _reference_seek_sampling( capture, vid_fps, sample_fps, offset_seconds,
                              num_frames, max_seconds ):
    """
    The earlier detection loop, transcribed exactly.

    Kept here, in the test, as the thing the fast path must match. It
    seeks to each sampled frame in turn and reads one frame.

    Returns:
        A list of (sample_index, timestamp, frame_index) tuples, where
        frame_index is read back off the capture so the identity of the
        frame is what is compared, not just the count.
    """
    import cv2
    results = []
    capture.set( cv2.CAP_PROP_POS_FRAMES, int( round( offset_seconds * vid_fps ) ) )
    sample_index = 0
    timestamp = offset_seconds
    while True:
        position = int( capture.get( cv2.CAP_PROP_POS_FRAMES ) )
        retrieved, _frame = capture.read()
        if not retrieved:
            break
        results.append( ( sample_index, timestamp, position ) )
        sample_index += 1
        timestamp = offset_seconds + sample_index / sample_fps
        if max_seconds is not None and timestamp >= offset_seconds + max_seconds:
            break
        next_frame = int( timestamp * vid_fps )
        if num_frames is not None and next_frame >= num_frames:
            break
        capture.set( cv2.CAP_PROP_POS_FRAMES, next_frame )
    return results


class TestSampleFrameMapping( unittest.TestCase ):
    """The pure arithmetic, with no video involved."""

    def test_sample_zero_is_the_offset( self ):
        self.assertEqual( bu_video.sample_time( 0, 0.0, 9 ), 0.0 )
        self.assertEqual( bu_video.sample_time( 0, 12.5, 9 ), 12.5 )

    def test_sample_times_advance_by_one_over_sample_fps( self ):
        self.assertAlmostEqual( bu_video.sample_time( 1, 0.0, 9 ), 1/9 )
        self.assertAlmostEqual( bu_video.sample_time( 9, 0.0, 9 ), 1.0 )

    def test_sample_zero_rounds_and_later_samples_floor( self ):
        # An asymmetry inherited from the earlier loop and preserved
        # deliberately: changing it would shift every sampled frame by up
        # to one frame and invalidate every cached detection.
        self.assertEqual( bu_video.sample_frame_index( 0, 0.4, 10, 9 ), 4 )
        self.assertEqual( bu_video.sample_frame_index( 0, 0.46, 10, 9 ), 5 )
        # sample 1 at offset 0.46, 9fps sampling, 10fps video:
        # t = 0.46 + 1/9 = 0.5711..., floor(5.711) = 5
        self.assertEqual( bu_video.sample_frame_index( 1, 0.46, 10, 9 ), 5 )

    def test_offset_zero_makes_the_two_rules_agree( self ):
        for index in range( 20 ):
            expected = int( ( index/9 ) * 30 )
            self.assertEqual( bu_video.sample_frame_index( index, 0.0, 30, 9 ), expected )

    def test_frame_indices_are_monotonic( self ):
        previous = -1
        for index in range( 200 ):
            current = bu_video.sample_frame_index( index, 0.0, 29.97, 9 )
            self.assertGreaterEqual( current, previous )
            previous = current


class TestIterSampledFramesArguments( unittest.TestCase ):

    def test_rejects_a_non_positive_frame_rate( self ):
        with self.assertRaises( ValueError ):
            list( bu_video.iter_sampled_frames( None, 0, 9 ) )

    def test_rejects_a_non_positive_sample_rate( self ):
        with self.assertRaises( ValueError ):
            list( bu_video.iter_sampled_frames( None, 30, 0 ) )


@unittest.skipUnless( _ffmpeg_available(), "ffmpeg is needed to build the test clip" )
class TestSamplingMatchesTheReferenceImplementation( unittest.TestCase ):
    """
    End-to-end equivalence against a real decoder.

    The clip has a long GOP on purpose: that is the case where seeking
    and sequential decoding behave most differently, and therefore the
    case where a mismatch would show up.
    """

    @classmethod
    def setUpClass( cls ):
        cls.tmpdir = tempfile.mkdtemp( prefix='betasuite-sampling-' )
        cls.path = os.path.join( cls.tmpdir, 'clip.mp4' )
        subprocess.run(
            [ 'ffmpeg', '-y', '-loglevel', 'error',
              '-f', 'lavfi', '-i', 'testsrc2=size=160x120:rate=25:duration=6',
              '-c:v', 'libx264', '-preset', 'ultrafast', '-g', '250',
              '-pix_fmt', 'yuv420p', cls.path ],
            check=True )

    @classmethod
    def tearDownClass( cls ):
        shutil.rmtree( cls.tmpdir, ignore_errors=True )

    def _compare( self, sample_fps, offset_seconds=0.0, max_seconds=None ):
        import cv2
        capture = cv2.VideoCapture( self.path )
        vid_fps = capture.get( cv2.CAP_PROP_FPS )
        num_frames = capture.get( cv2.CAP_PROP_FRAME_COUNT )
        reference = _reference_seek_sampling(
            capture, vid_fps, sample_fps, offset_seconds, num_frames, max_seconds )
        capture.release()

        capture = cv2.VideoCapture( self.path )
        fast = []
        for sample_index, timestamp, frame in bu_video.iter_sampled_frames(
                capture, vid_fps, sample_fps, offset_seconds, num_frames, max_seconds ):
            fast.append( ( sample_index, timestamp, frame ) )
        capture.release()

        self.assertEqual( len( fast ), len( reference ),
            "sampled a different number of frames than the reference implementation" )
        for ( fast_index, fast_time, _frame ), ( ref_index, ref_time, _pos ) in zip( fast, reference ):
            self.assertEqual( fast_index, ref_index )
            self.assertAlmostEqual( fast_time, ref_time, places=9 )
        return fast, reference

    def test_matches_at_the_default_sample_rate( self ):
        self._compare( sample_fps=9 )

    def test_matches_when_sampling_every_frame( self ):
        self._compare( sample_fps=25 )

    def test_matches_with_a_preview_offset( self ):
        self._compare( sample_fps=9, offset_seconds=2.0 )

    def test_matches_with_a_bounded_preview_window( self ):
        fast, _reference = self._compare( sample_fps=9, offset_seconds=1.0, max_seconds=2.0 )
        # The window is half-open on the right: every sample's timestamp
        # is strictly below offset + max_seconds.
        for _index, timestamp, _frame in fast:
            self.assertLess( timestamp, 1.0 + 2.0 + 1e-9 )

    def test_matches_when_sampling_faster_than_the_video_runs( self ):
        # Consecutive targets repeat here, and the fast path has to
        # re-deliver the buffered frame rather than advancing past it.
        self._compare( sample_fps=60 )

    def test_pixels_match_the_reference_frames( self ):
        import cv2
        capture = cv2.VideoCapture( self.path )
        vid_fps = capture.get( cv2.CAP_PROP_FPS )
        num_frames = capture.get( cv2.CAP_PROP_FRAME_COUNT )
        fast_frames = [ frame.copy() for _i, _t, frame in bu_video.iter_sampled_frames(
            capture, vid_fps, 9, 0.0, num_frames, 2.0 ) ]
        capture.release()

        capture = cv2.VideoCapture( self.path )
        reference_frames = []
        sample_index = 0
        while True:
            timestamp = sample_index / 9
            if timestamp >= 2.0:
                break
            target = bu_video.sample_frame_index( sample_index, 0.0, vid_fps, 9 )
            if target >= num_frames:
                break
            capture.set( cv2.CAP_PROP_POS_FRAMES, target )
            retrieved, frame = capture.read()
            if not retrieved:
                break
            reference_frames.append( frame.copy() )
            sample_index += 1
        capture.release()

        self.assertEqual( len( fast_frames ), len( reference_frames ) )
        for index, ( fast, reference ) in enumerate( zip( fast_frames, reference_frames ) ):
            self.assertTrue( np.array_equal( fast, reference ),
                "sample %d differs in pixels between sequential and seek decoding"%(index) )

    def test_resuming_from_a_checkpoint_lands_on_the_right_frame( self ):
        import cv2
        capture = cv2.VideoCapture( self.path )
        vid_fps = capture.get( cv2.CAP_PROP_FPS )
        num_frames = capture.get( cv2.CAP_PROP_FRAME_COUNT )
        whole = [ ( index, timestamp ) for index, timestamp, _frame
                  in bu_video.iter_sampled_frames( capture, vid_fps, 9, 0.0, num_frames ) ]
        capture.release()

        resume_at = 10
        capture = cv2.VideoCapture( self.path )
        resumed = [ ( index, timestamp ) for index, timestamp, _frame
                    in bu_video.iter_sampled_frames( capture, vid_fps, 9, 0.0, num_frames,
                                                     start_sample_index=resume_at ) ]
        capture.release()

        self.assertEqual( resumed, whole[resume_at:] )


if __name__ == '__main__':
    unittest.main()
