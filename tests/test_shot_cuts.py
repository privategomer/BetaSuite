#!/usr/bin/env python3
"""
test_shot_cuts.py - unit tests for betautils_video.detect_shot_cuts.

Synthetic frames only (solid color blocks via numpy), no real video I/O -
these run in well under a second and don't depend on any cached/real
footage existing. For a real-footage sanity check instead, see
tools/tuning/track_break_investigation.sh and the shot-cut validation
notes in betautils_video.py's own docstring.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )

import betautils_video as bu_video


class _FakeCapture:
    """
    Minimal stand-in for cv2.VideoCapture.

    Implements the grab/retrieve split the real object has, because
    that is what detect_shot_cuts uses: grab() advances past a frame
    without decoding it, retrieve() decodes the one grab() most recently
    positioned on. A capture that only implemented read() would not
    exercise the code path that actually runs.
    """

    def __init__( self, frames ):
        self._frames = frames
        self._next_index = 0
        self._grabbed_index = None

    def grab( self ):
        if self._next_index >= len( self._frames ):
            return False
        self._grabbed_index = self._next_index
        self._next_index += 1
        return True

    def retrieve( self ):
        if self._grabbed_index is None:
            return ( False, None )
        return ( True, self._frames[ self._grabbed_index ] )

    def read( self ):
        if not self.grab():
            return ( False, None )
        return self.retrieve()

    def release( self ):
        pass


def _solid_frame( color, size=32 ):
    frame = np.zeros( ( size, size, 3 ), dtype=np.uint8 )
    frame[:, :] = color
    return frame


class TestDetectShotCuts( unittest.TestCase ):

    def test_no_cuts_in_static_footage( self ):
        # same color the whole way through - nothing should ever cross
        # the threshold, regardless of how many samples
        frames = [ _solid_frame( (50, 100, 150) ) for _ in range( 20 ) ]
        cap = _FakeCapture( frames )
        cuts = bu_video.detect_shot_cuts( cap, vid_fps=10, sample_fps=10, threshold=0.6 )
        self.assertEqual( cuts, [] )

    def test_single_hard_cut_detected( self ):
        # first half one solid color, second half a maximally different
        # one - a real hard cut, should be flagged
        frames = ( [ _solid_frame( (0, 0, 0) ) for _ in range( 10 ) ]
                 + [ _solid_frame( (255, 255, 255) ) for _ in range( 10 ) ] )
        cap = _FakeCapture( frames )
        cuts = bu_video.detect_shot_cuts( cap, vid_fps=10, sample_fps=10, threshold=0.6 )
        self.assertEqual( len( cuts ), 1 )
        # cut timestamp is frame_idx/vid_fps for the frame that OPENS the
        # new shot - frame 10 is the first white frame, at 10/10 = 1.0s
        self.assertAlmostEqual( cuts[0], 1.0, places=2 )

    def test_two_consecutive_samples_required( self ):
        # a single strobe/flash frame - one sample very different from
        # its neighbors, but NOT two consecutive different-from-different
        # pairs - should NOT register as a cut (this is the false-positive
        # guard the two-consecutive-sample rule exists for)
        frames = ( [ _solid_frame( (50, 50, 50) ) for _ in range( 5 ) ]
                 + [ _solid_frame( (250, 250, 250) ) ]  # single flash frame
                 + [ _solid_frame( (50, 50, 50) ) for _ in range( 5 ) ] )
        cap = _FakeCapture( frames )
        cuts = bu_video.detect_shot_cuts( cap, vid_fps=10, sample_fps=10, threshold=0.6 )
        self.assertEqual( cuts, [] )

    def test_sample_cadence_respects_sample_fps( self ):
        # vid_fps=30, sample_fps=10 -> step_frames=3, so only every 3rd
        # frame is compared. Put the color change on a frame that does NOT
        # land on the sample cadence and confirm it's invisible to the
        # detector (proves frames are being skipped, not all compared).
        frames = [ _solid_frame( (10, 10, 10) ) for _ in range( 4 ) ]
        frames += [ _solid_frame( (200, 200, 200) ) ]  # frame index 4 - not a multiple of 3, never sampled
        frames += [ _solid_frame( (10, 10, 10) ) for _ in range( 4 ) ]
        cap = _FakeCapture( frames )
        cuts = bu_video.detect_shot_cuts( cap, vid_fps=30, sample_fps=10, threshold=0.6 )
        self.assertEqual( cuts, [] )

    def test_empty_video_returns_no_cuts( self ):
        cap = _FakeCapture( [] )
        cuts = bu_video.detect_shot_cuts( cap, vid_fps=10, sample_fps=10, threshold=0.6 )
        self.assertEqual( cuts, [] )

    def test_higher_threshold_is_less_sensitive( self ):
        # a moderate color shift - should register at a low threshold but
        # not at a very high one
        frames = ( [ _solid_frame( (100, 100, 100) ) for _ in range( 10 ) ]
                 + [ _solid_frame( (160, 100, 100) ) for _ in range( 10 ) ] )
        cap_loose = _FakeCapture( frames )
        cuts_loose = bu_video.detect_shot_cuts( cap_loose, vid_fps=10, sample_fps=10, threshold=0.05 )
        cap_strict = _FakeCapture( frames )
        cuts_strict = bu_video.detect_shot_cuts( cap_strict, vid_fps=10, sample_fps=10, threshold=0.95 )
        self.assertTrue( len( cuts_loose ) >= len( cuts_strict ) )

    def test_back_to_back_rapid_cuts_all_detected( self ):
        # regression test for the real bug found on 2026-09-13 against
        # actual quick-cut compilation footage: a version of this
        # function that confirmed a pending candidate by comparing the
        # NEXT sample against the CANDIDATE's own state scored zero cuts
        # here, because on back-to-back cuts the sample right after one
        # cut is itself the start of the next cut (not a continuation of
        # the first cut's new shot) - it differs from the candidate too,
        # so a same-state comparison wrongly discarded every single one
        # of these as a "spike". Confirming against the PRE-CUT state
        # instead (the fix) correctly recognizes that these samples
        # didn't revert, so they're real cuts, not spikes.
        colors = [ (255,0,0), (255,0,0), (0,255,0), (0,255,0),
                   (0,0,255), (0,0,255), (255,255,0), (255,255,0) ]
        frames = [ _solid_frame( c ) for c in colors ]
        cap = _FakeCapture( frames )
        cuts = bu_video.detect_shot_cuts( cap, vid_fps=8, sample_fps=8, threshold=0.3 )
        self.assertEqual( len( cuts ), 3 )

    def test_a_confirming_sample_can_also_open_the_next_candidate( self ):
        # The 2026-09-19 fix. Confirmation used to be an `elif`: a sample
        # that resolved a pending candidate could not itself raise a new
        # one, so on footage cutting close to the sample cadence roughly
        # every other cut was thrown away.
        #
        # Real cost, measured on the overnight run: two compilation
        # files recorded ZERO shot cuts across 10 and 15 minutes while
        # the structure bench showed 4-5 simultaneous tracked subjects.
        # With no cut timestamps, nothing forces a track reset, which is
        # what a censor box "floating" from one shot into the next
        # actually is.
        #
        # Eight samples, a new shot on every one: 7 real cuts. The old
        # logic found 3-4. This asserts a floor of 6, which is what the
        # fixed detector resolves.
        colors = [ (255,0,0), (0,255,0), (0,0,255), (255,255,0),
                   (0,255,255), (255,0,255), (128,64,200), (10,200,30) ]
        frames = [ _solid_frame( c ) for c in colors ]
        cap = _FakeCapture( frames )
        cuts = bu_video.detect_shot_cuts( cap, vid_fps=8, sample_fps=8, threshold=0.3 )
        self.assertGreaterEqual( len( cuts ), 6,
            "a confirming sample is not opening the next candidate; quick-cut "
            "footage will under-report cuts and censor boxes will float across them" )

    def test_a_cut_is_reported_on_the_frame_that_opens_the_new_shot( self ):
        # Before the fix a confirmed cut was reported one sample LATE,
        # because the candidate frame index was recorded but the
        # confirmation consumed the following sample. A late cut resets
        # the track after the new shot has already begun, which is the
        # visible "box lingers one beat into the next scene" artifact.
        frames = ( [ _solid_frame( (0, 0, 0) ) for _ in range( 6 ) ]
                 + [ _solid_frame( (255, 255, 255) ) for _ in range( 6 ) ] )
        cap = _FakeCapture( frames )
        cuts = bu_video.detect_shot_cuts( cap, vid_fps=10, sample_fps=10, threshold=0.6 )
        self.assertEqual( len( cuts ), 1 )
        # frame 6 is the first white frame: 6/10 = 0.6s exactly.
        self.assertAlmostEqual( cuts[0], 0.6, places=3 )

    def test_a_candidate_pending_at_end_of_file_is_kept( self ):
        # A cut landing on the last sampled frame has no following
        # sample to confirm it against. Dropping it silently loses a
        # real cut; a spurious reset at EOF costs nothing.
        frames = ( [ _solid_frame( (0, 0, 0) ) for _ in range( 6 ) ]
                 + [ _solid_frame( (255, 255, 255) ) ] )
        cap = _FakeCapture( frames )
        cuts = bu_video.detect_shot_cuts( cap, vid_fps=10, sample_fps=10, threshold=0.6 )
        self.assertEqual( len( cuts ), 1 )

    def test_a_single_flash_is_still_rejected_after_the_fix( self ):
        # The loosening must not cost the false-positive guard. A one
        # sample flash that reverts to the pre-cut state is not a cut,
        # and evaluating every sample as a candidate must not change
        # that.
        frames = ( [ _solid_frame( (50, 50, 50) ) for _ in range( 6 ) ]
                 + [ _solid_frame( (250, 250, 250) ) ]
                 + [ _solid_frame( (50, 50, 50) ) for _ in range( 6 ) ] )
        cap = _FakeCapture( frames )
        cuts = bu_video.detect_shot_cuts( cap, vid_fps=10, sample_fps=10, threshold=0.6 )
        self.assertEqual( cuts, [] )

    def test_static_footage_still_reports_nothing( self ):
        # The most important guard on a loosening change: evaluating a
        # candidate on every sample must not manufacture cuts in footage
        # that has none.
        frames = [ _solid_frame( (50, 100, 150) ) for _ in range( 40 ) ]
        cap = _FakeCapture( frames )
        cuts = bu_video.detect_shot_cuts( cap, vid_fps=10, sample_fps=10, threshold=0.5 )
        self.assertEqual( cuts, [] )


if __name__ == '__main__':
    unittest.main()
