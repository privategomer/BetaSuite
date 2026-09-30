"""
test_blur_methods_and_margin.py - the triple_box blur method, and the
edge margin that removes the last size-driven blur flicker.

WHAT triple_box IS
------------------
Three passes of a box blur. Repeated box convolution converges to a
Gaussian (central limit theorem), and by the third pass the shape is
within a few percent of one. cv2.blur is a running-sum filter, so its
cost per pixel does not grow with the window, which is why three box
passes beat one true Gaussian at the kernel sizes censoring uses.

Measured at matched obscuring power on a 400x400 region of synthetic
detail: exact Gaussian k=101 cost 12.53ms, triple_box k=87 cost 1.02ms
for the same residual detail, mean absolute difference 0.38/255.

WHY NOT cv2.stackBlur
---------------------
Tried, measured, rejected. Its blur strength is not monotonic in kernel
size at these sizes - residual detail on the same fixture was 11.3 at
k=31, 15.0 at k=101 and 72.8 at k=401, so past roughly k=31 a LARGER
kernel obscures LESS. A censor whose strength silently reverses above a
threshold is the wrong kind of risk. A test below pins that finding so
nobody re-adds it on the strength of its speed alone.

WHAT THE MARGIN IS
------------------
A blur whose kernel is a large fraction of its region reads mostly
BORDER_DEFAULT edge reflection rather than real content, and that
reflection depends on where the box's edge sits. So the result changed
whenever the box resized by a pixel, even with the kernel constant -
the flicker that survived stabilising the kernel.

Reading a margin of real neighbouring pixels instead removes the
dependence, because those neighbours do not move when the box resizes.
Measured with the box jittering +/-4px and the kernel held constant,
mean level swing inside a fixed crop:

    kernel/region   no margin   with margin
        0.30          1.31         0.00
        0.51          2.19         0.00
        0.99          3.04         0.00

The margin is read-only. What is composited back is still exactly the
box, so it cannot change what is censored - a test below pins that too.
"""

import math
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )

import betaconfig

import betautils_censor as bu_censor
import betautils_config as bu_config


def _detailed_frame( height=900, width=900, seed=3 ):
    """Structured, high-contrast content - what a censor has to destroy."""
    rng = np.random.default_rng( seed )
    frame = np.zeros( ( height, width, 3 ), np.uint8 )
    for i in range( 0, width, 16 ):
        frame[:, i:i+8] = 210
    for j in range( 0, height, 16 ):
        frame[j:j+8, :] = np.minimum( frame[j:j+8, :].astype( int ) + 45, 255 )
    frame = cv2.GaussianBlur( frame, ( 5, 5 ), 0 )
    return np.clip( frame.astype( int ) + rng.integers( -18, 18, frame.shape ),
                    0, 255 ).astype( np.uint8 )


class MarginHarness( unittest.TestCase ):

    def setUp( self ):
        self._saved = { name: getattr( betaconfig, name, None )
                        for name in ( 'blur_edge_margin', 'blur_fast_approximation' ) }
        self.frame = _detailed_frame()

    def tearDown( self ):
        for name, value in self._saved.items():
            if value is None:
                if hasattr( betaconfig, name ):
                    delattr( betaconfig, name )
            else:
                setattr( betaconfig, name, value )
        bu_config.invalidate_config_caches()

    def render( self, x, y, w, h, method='gaussian', strength=160, margin=True,
                reference=None ):
        betaconfig.blur_edge_margin = margin
        box = { 'x': x, 'y': y, 'w': w, 'h': h,
                'label': 'exposed_breast', 'censor_shape': 'box',
                'censor_style': { 'type': 'blur', 'method': method,
                                  'strength': strength } }
        reference = reference or ( w, h )
        box['style_scale_w'], box['style_scale_h'] = reference
        return bu_censor.blur_image(
            self.frame.copy(), x, y, w, h, box['censor_style'], 'box', 0,
            scale_w=reference[0], scale_h=reference[1] )


class TestTripleBoxApproximatesAGaussian( MarginHarness ):

    def test_it_obscures_about_as_much_as_a_gaussian( self ):
        region = self.frame[300:700, 300:700]
        exact = cv2.GaussianBlur( region, ( 101, 101 ), 0,
                                  borderType=cv2.BORDER_DEFAULT ).astype( float )
        approx = bu_censor._triple_box_blur( region, 101 ).astype( float )
        self.assertLess(
            abs( approx.std() - exact.std() ), 0.35 * exact.std(),
            "triple_box left a very different amount of detail than a true "
            "Gaussian of the same kernel; it is meant to approximate one" )

    def test_it_is_close_to_the_gaussian_pixel_for_pixel( self ):
        region = self.frame[300:700, 300:700]
        exact = cv2.GaussianBlur( region, ( 101, 101 ), 0,
                                  borderType=cv2.BORDER_DEFAULT ).astype( float )
        approx = bu_censor._triple_box_blur( region, 101 ).astype( float )
        self.assertLess( np.abs( approx - exact ).mean(), 3.0 )

    def test_stronger_kernels_never_obscure_materially_less( self ):
        # The property stackBlur fails badly.
        #
        # Not strict monotonicity: past the point where the box window
        # exceeds the region, every blur saturates and reads mostly border
        # reflection, so the curve flattens and can tick up by a hair.
        # Measured here: 15.43, 1.22, 0.70, 0.45, 0.46, 0.47 - a floor,
        # reached and held. stackBlur on the same fixture went 11.3 -> 15.0
        # -> 72.8, a 6.5x REVERSAL. The bar below separates the two.
        region = self.frame[300:700, 300:700]
        details = [ bu_censor._triple_box_blur( region, k ).astype( float ).std()
                    for k in ( 21, 61, 101, 201, 301, 401 ) ]
        for earlier, later in zip( details, details[1:] ):
            self.assertLessEqual(
                later, max( earlier * 1.10, earlier + 0.05 ),
                "a larger kernel obscured materially LESS: %s. That is the "
                "failure that disqualified cv2.stackBlur."
                %( [ round(d,2) for d in details ], ) )
        self.assertLess( details[-1], details[0] * 0.2 )

    def test_stackblur_would_have_failed_that_property( self ):
        # Documents the rejection with a live measurement rather than a
        # comment, so the claim can be re-checked on a new OpenCV.
        if not hasattr( cv2, 'stackBlur' ):
            self.skipTest( 'cv2.stackBlur not available in this build' )
        region = self.frame[300:700, 300:700]
        small = cv2.stackBlur( region, ( 31, 31 ) ).astype( float ).std()
        large = cv2.stackBlur( region, ( 401, 401 ) ).astype( float ).std()
        self.assertGreater(
            large, small,
            "cv2.stackBlur has become monotonic in this OpenCV build, so the "
            "reason triple_box was preferred over it no longer holds - "
            "re-read the rejection note in betautils_censor before acting" )

    def test_the_box_width_carries_the_right_variance( self ):
        # Three passes of width w have variance 3*w^2/12; that must match
        # the Gaussian's sigma^2, or the two do not blur alike.
        for kernel in ( 21, 61, 101, 201 ):
            sigma = bu_censor._gaussian_sigma_for_kernel( kernel )
            width = bu_censor._box_width_for_sigma( sigma )
            combined = math.sqrt( bu_censor.TRIPLE_BOX_PASSES * ( width**2 ) / 12.0 )
            self.assertLess( abs( combined - sigma ), 0.12 * sigma )

    def test_the_box_width_is_always_odd( self ):
        # An even window shifts the image half a pixel per pass, which
        # over three passes drifts the censored area visibly.
        for kernel in range( 5, 400, 2 ):
            width = bu_censor._box_width_for_sigma(
                bu_censor._gaussian_sigma_for_kernel( kernel ) )
            self.assertEqual( width % 2, 1 )
            self.assertGreaterEqual( width, 3 )

    def test_it_is_a_registered_method( self ):
        self.assertIn( 'triple_box', bu_config.VALID_BLUR_METHODS )

    def test_every_registered_method_renders( self ):
        for method in sorted( bu_config.VALID_BLUR_METHODS ):
            out = self.render( 300, 300, 200, 200, method=method )
            before = self.frame[300:500, 300:500].astype( float ).std()
            after = out[300:500, 300:500].astype( float ).std()
            self.assertLess( after, before,
                             "%s did not obscure anything"%( method, ) )


class TestMarginRemovesSizeDrivenFlicker( MarginHarness ):

    SIZES = ( 196, 198, 200, 202, 204, 202, 198 )

    def _swing( self, method, margin, crop=190, strength=160 ):
        means = []
        for size in self.SIZES:
            out = self.render( 300, 300, size, size, method=method,
                               margin=margin, strength=strength,
                               reference=( 200, 200 ) )
            means.append( out[300:300+crop, 300:300+crop].astype( float ).mean() )
        return max( means ) - min( means )

    def test_the_margin_removes_the_swing( self ):
        for method in ( 'gaussian', 'triple_box', 'box' ):
            without = self._swing( method, margin=False )
            with_margin = self._swing( method, margin=True )
            self.assertLess(
                with_margin, 0.01,
                "%s still swung %.4f levels with the margin on; the blur is "
                "still a function of where the box edge sits"%( method, with_margin ) )
            self.assertGreater(
                without, with_margin,
                "%s showed no size dependence even without the margin, so "
                "this fixture is not exercising the thing being fixed"%( method, ) )

    def _swing_exact( self, strength ):
        """Swing with the downscale approximation deliberately disabled."""
        saved = getattr( betaconfig, 'blur_fast_approximation', True )
        betaconfig.blur_fast_approximation = False
        try:
            return self._swing( 'gaussian', margin=False, strength=strength )
        finally:
            betaconfig.blur_fast_approximation = saved

    def test_a_weaker_blur_flickers_less_on_the_exact_path( self ):
        # The answer to "would lowering strength reduce flicker": on a true
        # Gaussian, yes, and monotonically. Measured 0.64 / 2.46 / 3.04 /
        # 3.04 at strength 20 / 60 / 120 / 200.
        swings = [ self._swing_exact( s ) for s in ( 20, 60, 120 ) ]
        for weaker, stronger in zip( swings, swings[1:] ):
            self.assertLessEqual( weaker, stronger + 1e-9,
                "swing did not grow with strength: %s"%( [ round(x,3) for x in swings ], ) )
        self.assertLess( swings[0], swings[-1] * 0.5 )

    def test_the_approximation_path_is_NOT_monotonic_in_strength( self ):
        # The counterintuitive half, and the reason the advice needs a
        # measurement rather than a rule of thumb.
        #
        # _approximate_gaussian_blur downscales by int(sigma // 4). A
        # bigger kernel means a coarser downscale, and a coarser downscale
        # averages away the very border sensitivity that causes the swing.
        # So on the approximation path a STRONGER blur can be STEADIER:
        # measured 0.64 at strength 20 against 0.35 at strength 120.
        #
        # Anyone lowering strength to chase flicker on this path would
        # make it worse. Pinned here so that stays visible.
        saved = getattr( betaconfig, 'blur_fast_approximation', True )
        betaconfig.blur_fast_approximation = True
        try:
            weak = self._swing( 'gaussian', margin=False, strength=20 )
            strong = self._swing( 'gaussian', margin=False, strength=120 )
        finally:
            betaconfig.blur_fast_approximation = saved
        self.assertLess(
            strong, weak,
            "the approximation path has become monotonic in strength "
            "(weak %.3f, strong %.3f); if that is a deliberate change, the "
            "tuning advice in CONFIG_REFERENCE needs revisiting"
            %( weak, strong ) )

    def test_the_margin_is_flat_across_every_strength( self ):
        # Which is what makes the whole strength-versus-flicker tradeoff
        # moot while the margin is on.
        for strength in ( 20, 60, 120, 200 ):
            self.assertLess(
                self._swing( 'gaussian', margin=True, strength=strength ), 0.01 )


class TestMarginCannotChangeWhatIsCensored( MarginHarness ):

    def test_the_output_region_is_exactly_the_box( self ):
        # The margin is read-only. Everything outside the box must be
        # byte-identical to the input.
        x, y, w, h = 300, 300, 200, 200
        out = self.render( x, y, w, h, margin=True )
        mask = np.ones( out.shape[:2], bool )
        mask[y:y+h, x:x+w] = False
        self.assertTrue(
            np.array_equal( out[mask], self.frame[mask] ),
            "pixels outside the censor box changed; the margin is supposed to "
            "be read-only" )

    def test_the_whole_box_is_still_obscured( self ):
        x, y, w, h = 300, 300, 200, 200
        out = self.render( x, y, w, h, margin=True )
        before = self.frame[y:y+h, x:x+w].astype( float ).std()
        after = out[y:y+h, x:x+w].astype( float ).std()
        self.assertLess( after, before * 0.5 )

    def test_a_box_at_the_frame_edge_still_works( self ):
        # The margin is clipped at the frame boundary, so the region comes
        # back smaller than requested and the crop offsets have to absorb
        # that. Getting this wrong is a shape mismatch, not a soft error.
        height, width = self.frame.shape[:2]
        for x, y, w, h in ( ( 0, 0, 150, 150 ),
                            ( width-150, 0, 150, 150 ),
                            ( 0, height-150, 150, 150 ),
                            ( width-150, height-150, 150, 150 ) ):
            out = self.render( x, y, w, h, margin=True )
            self.assertEqual( out.shape, self.frame.shape )
            before = self.frame[y:y+h, x:x+w].astype( float ).std()
            after = out[y:y+h, x:x+w].astype( float ).std()
            self.assertLess( after, before,
                             "corner box at (%d,%d) was not obscured"%( x, y ) )

    def test_a_box_covering_the_whole_frame_still_works( self ):
        small = _detailed_frame( 200, 200 )
        betaconfig.blur_edge_margin = True
        out = bu_censor.blur_image(
            small.copy(), 0, 0, 200, 200,
            { 'type': 'blur', 'method': 'triple_box', 'strength': 160 },
            'box', 0, scale_w=200, scale_h=200 )
        self.assertEqual( out.shape, small.shape )
        self.assertLess( out.astype( float ).std(), small.astype( float ).std() )


class TestSettingsAreKeyedAndValidated( unittest.TestCase ):

    def setUp( self ):
        self._saved = getattr( betaconfig, 'blur_edge_margin', None )

    def tearDown( self ):
        if self._saved is None:
            if hasattr( betaconfig, 'blur_edge_margin' ):
                del betaconfig.blur_edge_margin
        else:
            betaconfig.blur_edge_margin = self._saved
        bu_config.invalidate_config_caches()

    def test_blur_edge_margin_is_in_the_censor_key( self ):
        # It changes rendered bytes, so a run with it flipped must not
        # reuse the other run's output.
        import betautils_cache_paths as bu_cache
        betaconfig.blur_edge_margin = True
        bu_config.invalidate_config_caches()
        before = bu_cache.censor_key()
        betaconfig.blur_edge_margin = False
        bu_config.invalidate_config_caches()
        self.assertNotEqual( before, bu_cache.censor_key() )

    def test_a_non_boolean_margin_is_rejected( self ):
        betaconfig.blur_edge_margin = 'yes'
        bu_config.invalidate_config_caches()
        self.assertTrue( any( 'blur_edge_margin' in error
                              for error in bu_config._collect_validation_errors() ) )

    def test_an_unknown_blur_method_is_rejected( self ):
        errors = []
        bu_config._check_blur_style_fields(
            { 'type': 'blur', 'method': 'stack' }, 'test', errors )
        self.assertTrue( errors )


if __name__ == '__main__':
    unittest.main()
