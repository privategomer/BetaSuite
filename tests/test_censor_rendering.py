"""
test_censor_rendering.py - turning a box into pixels.

Three classes of assertion here:

  CORRECTNESS   The single-pass overlap merge must not mutate the
                caller's box dicts. It used to, and because merges only
                ever expand, a censored region ratcheted outward frame
                after frame and never shrank back. That bug was latent
                only because the shipped labels happen to use elliptical
                shapes; adding any label without a shape override wakes
                it up.

  THREAD SAFETY Parallel render workers share one box list, so nothing
                on the render path may mutate it. A mutation here turns
                a parallel render's output into a function of chunk
                scheduling order.

  EQUIVALENCE   The fast hex pixelation and the approximate blur must
                produce output of the right shape and character - the
                same region, fully covered, no leakage outside the mask.
"""

import copy
import unittest

import numpy as np

import betaconfig
import betautils_censor as bu_censor


def _noise_image( height=200, width=300 ):
    return np.random.default_rng( 7 ).integers(
        0, 255, ( height, width, 3 ), dtype=np.uint8 )


def _box( x=10, y=10, w=60, h=60, style=None, shape='box', label='exposed_breast' ):
    return {
        'x': x, 'y': y, 'w': w, 'h': h,
        'censor_style': style or { 'type': 'blur', 'method': 'gaussian', 'strength': 20 },
        'censor_shape': shape,
        'censor_sticker_seed': 0.5,
        'label': label, 'score': 0.9, 'start': 0.0, 'end': 1.0,
    }


class TestSinglePassMergeDoesNotMutateItsInput( unittest.TestCase ):
    """
    The regression guard for the ratcheting-box bug.

    Reproduction: two overlapping same-style boxes are merged, and the
    box that seeded the merged segment comes back permanently widened.
    """

    def setUp( self ):
        self.style = { 'type': 'blur', 'method': 'gaussian',
                       'strength': 20, 'merge': 'single-pass' }

    def test_the_originals_are_untouched( self ):
        first = _box( x=100, y=100, w=50, h=50, style=self.style )
        second = _box( x=130, y=100, w=50, h=50, style=self.style )
        before = copy.deepcopy( [ first, second ] )

        merged = bu_censor.collapse_boxes_for_style( [ first, second ], 1920, 1080 )

        self.assertEqual( [ first, second ], before,
            "single-pass merge widened the caller's own box dicts; because merges only "
            "expand, that region grows on every frame and never shrinks back" )
        self.assertEqual( len( merged ), 1 )
        self.assertEqual( merged[0]['x'], 100 )
        self.assertEqual( merged[0]['w'], 80 )

    def test_repeated_merges_are_stable( self ):
        first = _box( x=100, y=100, w=50, h=50, style=self.style )
        second = _box( x=130, y=100, w=50, h=50, style=self.style )
        widths = []
        for _frame in range( 5 ):
            merged = bu_censor.collapse_boxes_for_style( [ first, second ], 1920, 1080 )
            widths.append( merged[0]['w'] )
        self.assertEqual( len( set( widths ) ), 1,
            "the merged width grew across frames: %s"%(widths) )

    def test_span_merge_also_leaves_the_originals_alone( self ):
        style = { 'type': 'bar', 'color': ( 0, 0, 0 ), 'merge': 'span' }
        first = _box( x=100, y=100, w=50, h=50, style=style )
        second = _box( x=400, y=100, w=50, h=50, style=style )
        before = copy.deepcopy( [ first, second ] )
        merged = bu_censor.collapse_boxes_for_style( [ first, second ], 1920, 1080 )
        self.assertEqual( [ first, second ], before )
        self.assertEqual( merged[0]['w'], 350 )

    def test_span_refuses_to_bridge_more_than_two_boxes( self ):
        # Three same-label boxes in one frame are far more likely two
        # people than one person with an unusual body-part count, and a
        # bar spanning two different people reads as one region.
        style = { 'type': 'bar', 'color': ( 0, 0, 0 ), 'merge': 'span' }
        boxes = [ _box( x=x, style=style ) for x in ( 100, 400, 900 ) ]
        merged = bu_censor.collapse_boxes_for_style( boxes, 1920, 1080 )
        self.assertEqual( len( merged ), 3 )

    def test_non_box_shapes_are_never_merged( self ):
        style = { 'type': 'blur', 'strength': 20, 'merge': 'single-pass' }
        boxes = [ _box( x=100, style=style, shape='ellipse' ),
                  _box( x=110, style=style, shape='ellipse' ) ]
        merged = bu_censor.collapse_boxes_for_style( boxes, 1920, 1080 )
        self.assertEqual( len( merged ), 2 )

    def test_an_unknown_strategy_falls_back_to_no_merge( self ):
        style = { 'type': 'blur', 'strength': 20, 'merge': 'some-future-strategy' }
        boxes = [ _box( x=100, style=style ), _box( x=110, style=style ) ]
        merged = bu_censor.collapse_boxes_for_style( boxes, 1920, 1080 )
        self.assertEqual( len( merged ), 2 )


class TestRenderDoesNotMutateItsBoxes( unittest.TestCase ):
    """
    censor_img_for_boxes runs concurrently across render workers that
    share one box list, so it must treat that list as read-only.
    """

    def test_box_list_order_and_contents_survive_a_render( self ):
        image = _noise_image()
        boxes = [ _box( x=10, y=10, label='exposed_breast' ),
                  _box( x=80, y=20, label='exposed_vulva' ),
                  _box( x=40, y=40, label='exposed_breast' ) ]
        before = copy.deepcopy( boxes )
        identity_before = [ id( box ) for box in boxes ]

        bu_censor.censor_img_for_boxes( image, boxes )

        self.assertEqual( boxes, before, "rendering changed the box dicts" )
        self.assertEqual( [ id( box ) for box in boxes ], identity_before,
            "rendering reordered the caller's list" )

    def test_rendering_the_same_boxes_twice_gives_the_same_result( self ):
        boxes = [ _box( x=10, y=10 ), _box( x=40, y=30 ) ]
        base = _noise_image()
        first = bu_censor.censor_img_for_boxes( base.copy(), boxes )
        second = bu_censor.censor_img_for_boxes( base.copy(), boxes )
        self.assertTrue( np.array_equal( first, second ) )

    def test_an_empty_box_list_leaves_the_frame_alone( self ):
        image = _noise_image()
        before = image.copy()
        result = bu_censor.censor_img_for_boxes( image, [] )
        self.assertTrue( np.array_equal( result, before ) )


class TestHexPixelation( unittest.TestCase ):
    """
    The hex grid was rewritten from a per-cell mask-and-scan loop into
    one analytic labelling pass - 20-175ms per box per frame down to
    about a millisecond. The output is a proper tessellation now, so it
    is compared on properties rather than against the old pixels.
    """

    def test_output_shape_and_dtype_are_preserved( self ):
        region = _noise_image( 80, 90 )
        result = bu_censor.pixelate_hex_grid( region, 90, 80, 20 )
        self.assertEqual( result.shape, region.shape )
        self.assertEqual( result.dtype, region.dtype )

    def test_every_pixel_belongs_to_exactly_one_cell( self ):
        labels, count = bu_censor._hex_cell_labels( 60, 40, 15000 )
        self.assertEqual( labels.shape, ( 40, 60 ) )
        self.assertEqual( len( np.unique( labels ) ), count )
        self.assertEqual( labels.min(), 0 )
        self.assertEqual( labels.max(), count - 1 )

    def test_cells_are_contiguous_blocks_of_one_colour( self ):
        region = _noise_image( 60, 60 )
        result = bu_censor.pixelate_hex_grid( region, 60, 60, 20 )
        labels, count = bu_censor._hex_cell_labels( 60, 60, 10000 )
        for label in range( min( count, 12 ) ):
            mask = labels == label
            if not mask.any():
                continue
            colours = np.unique( result[mask].reshape( -1, 3 ), axis=0 )
            self.assertEqual( len( colours ), 1,
                "hex cell %d is not one flat colour"%(label) )

    def test_a_flat_region_stays_flat( self ):
        region = np.full( ( 60, 60, 3 ), 123, dtype=np.uint8 )
        result = bu_censor.pixelate_hex_grid( region, 60, 60, 20 )
        self.assertTrue( np.all( result == 123 ) )

    def test_larger_factor_makes_fewer_cells( self ):
        _small_labels, small_count = bu_censor._hex_cell_labels( 120, 120, 8000 )
        _large_labels, large_count = bu_censor._hex_cell_labels( 120, 120, 30000 )
        self.assertGreater( small_count, large_count )

    def test_the_grid_is_cached_across_calls( self ):
        bu_censor._hex_cell_labels.cache_clear()
        bu_censor._hex_cell_labels( 50, 50, 10000 )
        misses_before = bu_censor._hex_cell_labels.cache_info().misses
        bu_censor._hex_cell_labels( 50, 50, 10000 )
        info = bu_censor._hex_cell_labels.cache_info()
        self.assertEqual( info.misses, misses_before )
        self.assertGreaterEqual( info.hits, 1 )


class TestBlurApproximation( unittest.TestCase ):

    def setUp( self ):
        self.original = getattr( betaconfig, 'blur_fast_approximation', True )

    def tearDown( self ):
        betaconfig.blur_fast_approximation = self.original

    def test_approximate_and_exact_blurs_are_visually_close( self ):
        # Not identical - that is the point of an approximation - but at
        # obscuring strengths the detail the approximation loses is
        # exactly the detail the blur exists to destroy, so the mean
        # difference must stay small.
        region = _noise_image( 200, 200 )
        style = { 'type': 'blur', 'method': 'gaussian', 'strength': 50 }

        betaconfig.blur_fast_approximation = False
        exact = bu_censor.blur_image( region.copy(), 0, 0, 200, 200, style )

        betaconfig.blur_fast_approximation = True
        approximate = bu_censor.blur_image( region.copy(), 0, 0, 200, 200, style )

        difference = np.abs( exact.astype( np.int16 ) - approximate.astype( np.int16 ) )
        self.assertLess( difference.mean(), 8.0,
            "the approximate blur diverged too far from the exact one" )

    def test_small_kernels_are_never_approximated( self ):
        # Below blur_approximation_min_kernel the exact blur is already
        # cheap, and approximating a small kernel is visible.
        region = _noise_image( 100, 100 )
        style = { 'type': 'blur', 'method': 'gaussian', 'strength': 2 }

        betaconfig.blur_fast_approximation = False
        exact = bu_censor.blur_image( region.copy(), 0, 0, 100, 100, style )
        betaconfig.blur_fast_approximation = True
        approximate = bu_censor.blur_image( region.copy(), 0, 0, 100, 100, style )
        self.assertTrue( np.array_equal( exact, approximate ) )

    def test_the_blurred_region_actually_changed( self ):
        region = _noise_image( 120, 120 )
        style = { 'type': 'blur', 'method': 'gaussian', 'strength': 40 }
        result = bu_censor.blur_image( region.copy(), 0, 0, 120, 120, style )
        self.assertFalse( np.array_equal( result, region ) )
        self.assertLess( result.std(), region.std(),
            "a blur must reduce local variation" )


class TestMaskCache( unittest.TestCase ):

    def test_repeated_masks_are_served_from_cache( self ):
        bu_censor.clear_mask_cache()
        bu_censor._alpha_for( 'ellipse', 80, 60, 0.3 )
        misses_before = bu_censor.mask_cache_info().misses
        for _ in range( 5 ):
            bu_censor._alpha_for( 'ellipse', 80, 60, 0.3 )
        info = bu_censor.mask_cache_info()
        self.assertEqual( info.misses, misses_before )
        self.assertGreaterEqual( info.hits, 5 )

    def test_cached_alphas_are_read_only( self ):
        # They are shared across boxes, frames and threads; a caller that
        # wrote to one would corrupt every later use.
        alpha = bu_censor._alpha_for( 'circle', 40, 40, 0.2 )
        self.assertFalse( alpha.flags.writeable )

    def test_alpha_is_in_range_and_shaped_for_broadcasting( self ):
        alpha = bu_censor._alpha_for( 'ellipse', 50, 30, 0.25 )
        self.assertEqual( alpha.shape, ( 30, 50, 1 ) )
        self.assertGreaterEqual( alpha.min(), 0.0 )
        self.assertLessEqual( alpha.max(), 1.0 )

    def test_a_box_mask_with_no_feather_is_fully_opaque( self ):
        alpha = bu_censor._alpha_for( 'box', 20, 20, 0 )
        self.assertTrue( np.all( alpha == 1.0 ) )

    def test_different_feathers_are_cached_separately( self ):
        soft = bu_censor._alpha_for( 'ellipse', 50, 50, 0.5 )
        hard = bu_censor._alpha_for( 'ellipse', 50, 50, 0.0 )
        self.assertFalse( np.array_equal( soft, hard ) )


class TestMaskedCompositing( unittest.TestCase ):

    def test_a_circle_mask_leaves_the_corners_untouched( self ):
        image = np.zeros( ( 60, 60, 3 ), dtype=np.uint8 )
        censored = np.full( ( 40, 40, 3 ), 255, dtype=np.uint8 )
        result = bu_censor.apply_masked_region(
            image, 10, 10, 40, 40, censored, 'circle', feather=0 )
        # centre is inside the circle, the box's own corner is not
        self.assertTrue( np.all( result[30, 30] == 255 ) )
        self.assertTrue( np.all( result[11, 11] == 0 ) )

    def test_a_box_mask_with_no_feather_overwrites_exactly_the_rectangle( self ):
        image = np.zeros( ( 60, 60, 3 ), dtype=np.uint8 )
        censored = np.full( ( 20, 20, 3 ), 200, dtype=np.uint8 )
        result = bu_censor.apply_masked_region(
            image, 5, 5, 20, 20, censored, 'box', feather=0 )
        self.assertTrue( np.all( result[5:25, 5:25] == 200 ) )
        self.assertTrue( np.all( result[0:5, :] == 0 ) )

    def test_feathering_produces_intermediate_values_at_the_edge( self ):
        image = np.zeros( ( 80, 80, 3 ), dtype=np.uint8 )
        censored = np.full( ( 60, 60, 3 ), 255, dtype=np.uint8 )
        result = bu_censor.apply_masked_region(
            image, 10, 10, 60, 60, censored, 'ellipse', feather=0.4 )
        values = set( np.unique( result ) )
        self.assertTrue( values - { 0, 255 },
            "a feathered edge must produce values between the two extremes" )


class TestSafeGeometry( unittest.TestCase ):

    def test_a_box_past_the_frame_edge_is_clamped_inside( self ):
        # A raw box can land past the real frame edge after a large
        # picture size scales the model's padding back up. Without the
        # start-coordinate clamp, image[y:y+h, x:x+w] returns an EMPTY
        # array rather than raising, which is what crashed
        # cv2.GaussianBlur deep inside a render chunk.
        x, y, w, h = bu_censor.compute_safe_geometry(
            5000, 5000, 100, 100, 1920, 1080, 0, 0 )
        self.assertLess( x, 1920 )
        self.assertLess( y, 1080 )
        self.assertGreaterEqual( w, 1 )
        self.assertGreaterEqual( h, 1 )

    def test_positive_area_safety_grows_the_box( self ):
        _x, _y, w, h = bu_censor.compute_safe_geometry(
            500, 500, 100, 100, 1920, 1080, 0.5, 0.5 )
        self.assertGreater( w, 100 )
        self.assertGreater( h, 100 )

    def test_negative_area_safety_shrinks_the_box( self ):
        _x, _y, w, h = bu_censor.compute_safe_geometry(
            500, 500, 100, 100, 1920, 1080, -0.5, -0.5 )
        self.assertLess( w, 100 )
        self.assertLess( h, 100 )

    def test_geometry_never_leaves_the_frame( self ):
        for raw_x, raw_y, raw_w, raw_h in (
                ( 0, 0, 10, 10 ), ( 1900, 1070, 200, 200 ), ( -50, -50, 100, 100 ) ):
            x, y, w, h = bu_censor.compute_safe_geometry(
                raw_x, raw_y, raw_w, raw_h, 1920, 1080, 0.2, 0.2 )
            self.assertGreaterEqual( x, 0 )
            self.assertGreaterEqual( y, 0 )
            self.assertLessEqual( x + w, 1920 )
            self.assertLessEqual( y + h, 1080 )


class TestCensorScaleStrategy( unittest.TestCase ):

    def setUp( self ):
        self.original = betaconfig.censor_scale_strategy

    def tearDown( self ):
        betaconfig.censor_scale_strategy = self.original

    def test_an_unknown_strategy_returns_one_rather_than_none( self ):
        # The pre-2.1 version fell off the end and returned None, which
        # became `strength * None` and a TypeError deep inside a render.
        betaconfig.censor_scale_strategy = 'not-a-strategy'
        image = np.zeros( ( 100, 100, 3 ), dtype=np.uint8 )
        self.assertEqual( bu_censor.censor_scale_for_image_box( image, 50, 50 ), 1 )

    def test_feature_strategy_scales_with_the_box( self ):
        betaconfig.censor_scale_strategy = 'feature'
        image = np.zeros( ( 1080, 1920, 3 ), dtype=np.uint8 )
        self.assertAlmostEqual( bu_censor.censor_scale_for_image_box( image, 200, 100 ), 1.0 )

    def test_image_strategy_scales_with_the_frame( self ):
        betaconfig.censor_scale_strategy = 'image'
        image = np.zeros( ( 1080, 2000, 3 ), dtype=np.uint8 )
        self.assertAlmostEqual( bu_censor.censor_scale_for_image_box( image, 50, 50 ), 2.0 )


class TestShapeResolution( unittest.TestCase ):

    def test_a_styles_own_shape_wins( self ):
        self.assertEqual( bu_censor.resolve_censor_shape(
            { 'type': 'blur', 'shape': 'ellipse' }, 'box' ), 'ellipse' )

    def test_a_type_default_applies_when_the_style_is_silent( self ):
        # A bar is a rectangle by definition, and the merge strategies
        # only fire for shape == 'box', so a bar that inherited an
        # elliptical item shape would silently never merge.
        self.assertEqual( bu_censor.resolve_censor_shape(
            { 'type': 'bar' }, 'circle' ), 'box' )

    def test_the_item_default_is_the_last_resort( self ):
        self.assertEqual( bu_censor.resolve_censor_shape(
            { 'type': 'blur' }, 'circle' ), 'circle' )


class TestStyleResolution( unittest.TestCase ):

    def test_a_single_dict_is_returned_as_is( self ):
        style = { 'type': 'blur', 'strength': 20 }
        self.assertIs( bu_censor.resolve_censor_style( style ), style )

    def test_a_list_resolves_to_one_of_its_entries( self ):
        options = [ { 'type': 'blur', 'strength': 20 },
                    { 'type': 'pixel', 'strength': 10 } ]
        for _ in range( 20 ):
            self.assertIn( bu_censor.resolve_censor_style( options ), options )

    def test_zero_weight_entries_are_never_chosen( self ):
        options = [ { 'type': 'blur', 'strength': 20, 'weight': 0 },
                    { 'type': 'pixel', 'strength': 10, 'weight': 1 } ]
        for _ in range( 30 ):
            self.assertEqual( bu_censor.resolve_censor_style( options )['type'], 'pixel' )


if __name__ == '__main__':
    unittest.main()
