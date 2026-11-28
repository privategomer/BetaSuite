"""
test_blur_kernel_stability.py - a continuous track's blur kernel must not
change from frame to frame.

THE SYMPTOM
-----------
Blur visibly pulsing between strong and weak on a subject that is barely
moving, so the censored area becomes readable on the weak frames. Unlike
earlier flicker theories this one is about the BLUR ITSELF changing
strength, not about the box edge moving.

THE CHAIN
---------
    tracked box jitters a few px
      -> kernel re-clamped against the LIVE box (not the stable one)
        -> kernel size changes frame to frame
          -> sigma changes
            -> downscale = int(sigma // 4) STEPS, e.g. 10 -> 9
              -> _approximate_gaussian_blur resamples on a different grid
                -> visibly different amount of blur

The last step is why this reads as a step change rather than a gentle
drift: the downscale factor is an integer, so it does not ease between
values.

WHY IT ONLY SHOWS AT HIGH STRENGTH
----------------------------------
_clamp_kernel_to_region only binds when the kernel approaches the size of
the region it blurs. A small kernel never reaches the limit, so the live
box it was clamped against never mattered. Measured on a real 640m cache
(exposed_breast, 21 tracks of 5+ boxes):

    strength  60 : median kernel swing  0px,  0/21 tracks unstable
    strength 100 : median kernel swing  4px,  8/21 tracks unstable
    strength 140 : median kernel swing 18px, 16/21 tracks unstable

That is exactly the direction someone goes when they want a stronger
censor, so the bug punished the setting most likely to be raised.

WHY THE OBVIOUS FIX IS WRONG
----------------------------
min(clamp_to_reference, clamp_to_live) looks safer and does not work: on
the same cache the live box is smaller than the track reference on 52%
of boxes (median 0.89x), so the live term keeps winning and keeps
jittering - it left 7 of 21 tracks unstable. Clamping only ever REDUCES
a kernel, so declining to clamp against the live box can only blur more,
never less, and cannot under-censor.
"""

import math
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )

import betautils_censor as bu_censor


FRAME = np.zeros( ( 1080, 1920, 3 ), np.uint8 )

# One track's stable reference size, and the live boxes a real tracked
# box takes across consecutive frames at that reference.
REFERENCE = ( 300, 280 )
LIVE_BOXES = ( ( 300, 280 ), ( 292, 268 ), ( 305, 286 ), ( 288, 262 ),
               ( 300, 280 ), ( 280, 250 ), ( 310, 290 ), ( 295, 272 ) )


def kernel_for( strength, live_w, live_h, reference=REFERENCE ):
    """
    The kernel the REAL blur_image uses for one frame of one track.

    Observed by intercepting the blur call rather than recomputed here.
    An earlier version of this file reimplemented the strength-and-clamp
    arithmetic, which meant the tests agreed with themselves and passed
    happily with the original bug put back. A test that does not execute
    the code under test cannot fail for the right reason.
    """
    scale_w, scale_h = reference
    style = { 'type': 'blur', 'method': 'gaussian', 'strength': strength }
    seen = []

    real_approximate = bu_censor._approximate_gaussian_blur
    real_gaussian = cv2.GaussianBlur

    def watch_approximate( region, kernel_size ):
        seen.append( kernel_size )
        return real_approximate( region, kernel_size )

    def watch_gaussian( src, ksize, sigma, **kwargs ):
        seen.append( ksize[0] )
        return real_gaussian( src, ksize, sigma, **kwargs )

    bu_censor._approximate_gaussian_blur = watch_approximate
    cv2.GaussianBlur = watch_gaussian
    try:
        bu_censor.blur_image( FRAME.copy(), 600, 400, live_w, live_h, style,
                              'box', 0, scale_w=scale_w, scale_h=scale_h )
    finally:
        bu_censor._approximate_gaussian_blur = real_approximate
        cv2.GaussianBlur = real_gaussian

    assert seen, "blur_image did not blur anything"
    return seen[0]


def downscale_for( kernel_size ):
    """The integer downscale _approximate_gaussian_blur would pick."""
    return max( 1, int( bu_censor._gaussian_sigma_for_kernel( kernel_size ) // 4 ) )


class TestKernelIsStableAcrossATrack( unittest.TestCase ):

    def test_the_kernel_never_changes_while_the_box_jitters( self ):
        for strength in ( 60, 100, 140, 200, 300 ):
            kernels = { kernel_for( strength, w, h ) for w, h in LIVE_BOXES }
            self.assertEqual(
                len( kernels ), 1,
                "at strength %d the blur kernel took %d different values "
                "across one track's frames (%s) while only the box jittered; "
                "the kernel is being clamped against the live box again"
                %( strength, len( kernels ), sorted( kernels ) ) )

    def test_the_downscale_factor_never_steps( self ):
        # The step that makes it visible rather than subtle.
        for strength in ( 100, 140, 200, 300 ):
            factors = { downscale_for( kernel_for( strength, w, h ) )
                        for w, h in LIVE_BOXES }
            self.assertEqual(
                len( factors ), 1,
                "at strength %d the approximation's downscale factor stepped "
                "between %s across one track; each step resamples the region "
                "on a different grid and reads as the blur changing strength"
                %( strength, sorted( factors ) ) )

    def test_high_strength_is_not_worse_than_low_strength( self ):
        # The regression had a strength threshold: stable at 60, unstable
        # at 140. Stability must not depend on how hard the censor is
        # being pushed.
        low = { kernel_for( 60, w, h ) for w, h in LIVE_BOXES }
        high = { kernel_for( 200, w, h ) for w, h in LIVE_BOXES }
        self.assertEqual( len( low ), len( high ), 1 )
        self.assertEqual(
            len( high ), 1,
            "raising strength reintroduced kernel instability; this is the "
            "exact regression, since raising strength is what someone does "
            "when they want a stronger censor" )


class TestRenderedOutputIsStable( unittest.TestCase ):
    """
    The end-to-end version: render the same content through one track's
    jittering boxes and require the visible result to hold steady.
    """

    @staticmethod
    def _content():
        rng = np.random.default_rng( 11 )
        return rng.integers( 0, 255, ( 1080, 1920, 3 ), dtype=np.uint8 )

    def test_blur_output_is_identical_when_only_position_moves( self ):
        # Box size held CONSTANT, position moving. This isolates what the
        # kernel fix controls: with a stable reference the kernel, sigma
        # and downscale are all fixed, so the same content blurred at a
        # different offset must come out identically.
        #
        # Size is held fixed on purpose. A differently SIZED region
        # resamples on a different internal grid inside
        # _approximate_gaussian_blur (small_w = w // downscale), which
        # moves the output slightly no matter how stable the kernel is.
        # That residue is second-order and is not what this test is for;
        # size smoothing is what limits it, and
        # test_size_smoothing_and_clamp.py covers that.
        content = self._content()
        style = { 'type': 'blur', 'method': 'gaussian', 'strength': 160 }
        width, height = REFERENCE

        spreads = []
        for offset in ( 0, 3, 7, 2, 11, 5 ):
            box = { 'x': 600 + offset, 'y': 400, 'w': width, 'h': height,
                    'label': 'exposed_breast', 'censor_style': style,
                    'censor_shape': 'box',
                    'style_scale_w': REFERENCE[0], 'style_scale_h': REFERENCE[1] }
            rendered = bu_censor.censor_image( content.copy(), box )
            region = rendered[400:400+height, 600+offset:600+offset+width]
            spreads.append( float( region.astype( float ).std() ) )

        swing = max( spreads ) - min( spreads )
        self.assertLess(
            swing, 0.30,
            "the amount of detail surviving the blur swung %.3f levels "
            "while only the box POSITION changed; the kernel is not stable"
            %( swing, ) )

    def test_a_jittering_box_does_not_step_the_blur_amount( self ):
        # The end-to-end form of the downscale-stepping bug, expressed as
        # the quantity that actually stepped. Before the fix this set had
        # two distinct downscale factors at strength 160; it must have one.
        factors = { downscale_for( kernel_for( 160, w, h ) )
                    for w, h in LIVE_BOXES }
        self.assertEqual(
            len( factors ), 1,
            "one track's frames produced downscale factors %s; each change "
            "resamples the blur on a different grid, which is the visible "
            "step between strong and weak blur"%( sorted( factors ), ) )


class TestClampStillGuardsTheRegion( unittest.TestCase ):
    """
    The clamp exists for a real reason and must keep working. Removing
    the live-box clamp must not turn into removing the clamp.
    """

    def test_a_kernel_is_still_clamped_to_its_reference_region( self ):
        # A huge strength on a small reference must still be trimmed.
        kernel = kernel_for( 5000, 120, 120, reference=( 120, 120 ) )
        self.assertLessEqual(
            kernel, 120,
            "an enormous strength produced a kernel larger than the region "
            "it blurs; past that point the result is BORDER reflection "
            "rather than the region's own pixels" )

    def test_the_clamp_only_ever_reduces( self ):
        # The property that makes dropping the live-box clamp safe: a
        # clamp can never make a kernel bigger, so not applying one can
        # never blur less than applying it.
        for kernel in ( 3, 21, 101, 301, 999 ):
            for region in ( 40, 120, 400 ):
                clamped = bu_censor._clamp_kernel_to_region(
                    kernel, region, region, 'gaussian' )
                self.assertLessEqual(
                    clamped, max( kernel, 3 ),
                    "clamping increased a kernel; the safety argument for "
                    "skipping the live-box clamp depends on it only ever "
                    "reducing" )

    def test_a_box_style_is_clamped_tighter_than_gaussian( self ):
        # A flat box window reaches further than a Gaussian of the same
        # nominal size, so it saturates sooner.
        gaussian = bu_censor._clamp_kernel_to_region( 999, 200, 200, 'gaussian' )
        box = bu_censor._clamp_kernel_to_region( 999, 200, 200, 'box' )
        self.assertLess( box, gaussian )


class TestStillsAndToolsKeepWorking( unittest.TestCase ):
    """
    style_scale_w/h only exists on tracked video boxes. Stills, live
    capture and hand-built bench boxes have none, and must fall back to
    the live box rather than crashing or blurring nothing.
    """

    def test_a_box_without_a_track_reference_still_blurs( self ):
        content = np.full( ( 400, 400, 3 ), 200, np.uint8 )
        content[150:250, 150:250] = 20
        box = { 'x': 100, 'y': 100, 'w': 200, 'h': 200,
                'label': 'exposed_breast', 'censor_shape': 'box',
                'censor_style': { 'type': 'blur', 'method': 'gaussian',
                                  'strength': 60 } }
        before = content[100:300, 100:300].astype( float ).std()
        rendered = bu_censor.censor_image( content.copy(), box )
        after = rendered[100:300, 100:300].astype( float ).std()
        self.assertLess(
            after, before,
            "a box with no style_scale_w/h (a still, or a bench box) was "
            "not blurred at all" )


if __name__ == '__main__':
    unittest.main()
