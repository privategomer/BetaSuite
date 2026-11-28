"""
A feathered circle or ellipse must read as a circle or ellipse.

The feather is a Gaussian blur of the shape's mask. With the default
reflecting border it mirrored the mask at the box edge, so alpha stayed near
1 right up to all four edges and dropped to 0 in one pixel: a square with
rounded corners, worse the heavier the feather. These tests pin the shape
itself (alpha at the box edge, at the corners, past the box) and the one
user-visible consequence, a blur that is continuous across the box edge.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )

import betautils_censor as bu_censor          # noqa: E402
import betautils_cache_paths as bu_cache      # noqa: E402


BOXES = ( ( 306, 299 ), ( 334, 255 ), ( 219, 259 ) )
FEATHERS = ( 0.25, 0.45, 0.55 )


def _edge_midpoints( alpha ):
    h, w = alpha.shape[:2]
    return [ alpha[ h//2, 0, 0 ], alpha[ h//2, w-1, 0 ], alpha[ 0, w//2, 0 ], alpha[ h-1, w//2, 0 ] ]


class TestAlphaShape( unittest.TestCase ):

    def test_the_box_edge_is_no_longer_nearly_opaque( self ):
        for w, h in BOXES:
            for feather in FEATHERS:
                with self.subTest( w=w, h=h, feather=feather ):
                    alpha = bu_censor._alpha_for( 'ellipse', w, h, feather )
                    # 0.95 to 0.98 before the fix; the ramp's middle is 0.5
                    for value in _edge_midpoints( alpha ):
                        self.assertLess( value, 0.65 )

    def test_the_corners_and_the_centre_still_read_as_an_ellipse( self ):
        alpha = bu_censor._alpha_for( 'ellipse', 306, 299, 0.55 )
        self.assertLess( alpha[ 0, 0, 0 ], 0.02 )
        self.assertLess( alpha[ -1, -1, 0 ], 0.02 )
        self.assertGreater( alpha[ 299//2, 306//2, 0 ], 0.95 )

    def test_a_padded_ramp_finishes_outside_the_box( self ):
        w, h, feather = 306, 299, 0.55
        pad = bu_censor.feather_pad( 'ellipse', w, h, feather )
        self.assertGreater( pad, 0 )
        alpha = bu_censor._alpha_for( 'ellipse', w, h, feather, pad )
        self.assertEqual( alpha.shape, ( h + 2*pad, w + 2*pad, 1 ) )
        # Nothing left at the padded rectangle's own edge: no cut anywhere
        for value in _edge_midpoints( alpha ):
            self.assertLess( value, 0.03 )
        # And the box edge sits mid-ramp, where an ellipse's edge should be
        inside = alpha[ pad + h//2, pad, 0 ]
        self.assertTrue( 0.4 < inside < 0.6 )

    def test_circle_takes_the_same_path( self ):
        alpha = bu_censor._alpha_for( 'circle', 300, 300, 0.45 )
        for value in _edge_midpoints( alpha ):
            self.assertLess( value, 0.65 )

    def test_a_box_and_an_unfeathered_shape_are_untouched( self ):
        self.assertEqual( bu_censor.feather_pad( 'box', 300, 300, 0.5 ), 0 )
        self.assertEqual( bu_censor.feather_pad( 'ellipse', 300, 300, 0 ), 0 )
        # A box with a feather has always been a plain rectangle
        self.assertTrue( ( bu_censor._alpha_for( 'box', 120, 90, 0.4 ) == 1.0 ).all() )
        # An unfeathered ellipse is still the hard binary mask
        hard = bu_censor._alpha_for( 'ellipse', 120, 90, 0 )
        self.assertEqual( set( np.unique( hard ).tolist() ), { 0.0, 1.0 } )


class TestBlurIsContinuousAcrossTheBoxEdge( unittest.TestCase ):

    def _render( self, x, y, w, h, feather=0.55, shape='ellipse' ):
        rng = np.random.default_rng( 3 )
        base = np.clip( rng.normal( 128, 45, ( 1080, 1920, 3 ) ), 0, 255 ).astype( np.uint8 )
        style = { 'type': 'blur', 'method': 'triple_box', 'shape': shape,
                  'strength': 70, 'feather': feather }
        out = bu_censor.censor_image( base.copy(), {
            'x': x, 'y': y, 'w': w, 'h': h, 'censor_style': style, 'censor_shape': shape,
            'style_scale_w': w, 'style_scale_h': h } )
        diff = np.abs( out.astype( int ) - base.astype( int ) ).sum( 2 ).astype( float )
        return base, out, diff

    def test_no_step_at_the_edge_of_a_heavily_feathered_ellipse( self ):
        x, y, w, h = 842, 528, 306, 299
        _base, _out, diff = self._render( x, y, w, h )
        band = slice( y + h//2 - 15, y + h//2 + 15 )
        outside = diff[ band, x - 1 ].mean()
        inside = diff[ band, x + 1 ].mean()
        # Before the fix: 0 just outside, a plateau just inside
        self.assertGreater( outside, 0.7*inside )

    def test_the_ramp_ends_within_its_padding( self ):
        x, y, w, h = 842, 528, 306, 299
        pad = bu_censor.feather_pad( 'ellipse', w, h, 0.55 )
        _base, _out, diff = self._render( x, y, w, h )
        self.assertEqual( diff[ y - pad - 3 : y - pad, : ].sum(), 0 )
        self.assertEqual( diff[ :, x - pad - 3 : x - pad ].sum(), 0 )
        self.assertEqual( diff[ y + h + pad : y + h + pad + 3, : ].sum(), 0 )

    def test_the_box_corners_stay_uncensored_like_an_ellipse( self ):
        x, y, w, h = 842, 528, 306, 299
        _base, _out, diff = self._render( x, y, w, h )
        self.assertEqual( diff[ y + 2, x + 2 ], 0 )

    def test_a_box_at_the_frame_edge_is_clipped_not_crashed( self ):
        for box in ( ( 0, 0, 200, 150 ), ( 1720, 930, 200, 150 ), ( 0, 500, 300, 300 ) ):
            with self.subTest( box=box ):
                _base, out, _diff = self._render( *box )
                self.assertEqual( out.shape, ( 1080, 1920, 3 ) )

    def test_a_feathered_box_still_changes_exactly_its_rectangle( self ):
        x, y, w, h = 842, 528, 306, 299
        _base, _out, diff = self._render( x, y, w, h, feather=0.4, shape='box' )
        rows, cols = np.nonzero( diff )
        self.assertEqual( ( rows.min(), rows.max(), cols.min(), cols.max() ),
                          ( y, y + h - 1, x, x + w - 1 ) )

    def test_an_unfeathered_ellipse_stays_inside_its_box( self ):
        x, y, w, h = 842, 528, 306, 299
        _base, _out, diff = self._render( x, y, w, h, feather=0 )
        rows, cols = np.nonzero( diff )
        self.assertGreaterEqual( rows.min(), y )
        self.assertLess( rows.max(), y + h )
        self.assertGreaterEqual( cols.min(), x )
        self.assertLess( cols.max(), x + w )


class TestCacheKey( unittest.TestCase ):

    def _label( self, style, shape='box' ):
        return { 'exposed_breast': { 'censor_style': style, 'censor_shape': shape } }

    def test_a_feathered_ellipse_blur_moves_the_key( self ):
        blur = { 'type': 'blur', 'shape': 'ellipse', 'feather': 0.3 }
        self.assertTrue( bu_cache.censor_uses_feathered_shape( self._label( [ blur ] ) ) )

    def test_the_item_shape_counts_when_the_style_sets_none( self ):
        blur = { 'type': 'pixel', 'feather': 0.3 }
        self.assertTrue( bu_cache.censor_uses_feathered_shape( self._label( blur, 'circle' ) ) )

    def test_configs_the_fix_did_not_touch_keep_their_key( self ):
        for style in (
            { 'type': 'blur', 'shape': 'ellipse' },                        # no feather
            { 'type': 'blur', 'shape': 'box', 'feather': 0.3 },            # a box
            { 'type': 'bar', 'shape': 'box', 'feather': 0.3 },
            { 'type': 'sticker', 'feather': 0.3 },                         # own silhouette
        ):
            with self.subTest( style=style ):
                self.assertFalse( bu_cache.censor_uses_feathered_shape(
                    self._label( [ style ], 'ellipse' ) if style['type'] == 'sticker'
                    else self._label( [ style ] ) ) )


if __name__ == '__main__':
    unittest.main()
