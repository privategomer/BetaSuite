"""
test_sticker_cache.py - the processed-sticker cache changes speed, never pixels.

sticker_image used to resize, crop and feather the PNG on every frame.
It now caches those planes and blends premultiplied colour, which is the
same float32 arithmetic in the same order. reference_sticker_image below
is the old per-frame code, frozen, and every case must match it byte for
byte, including boxes hanging off each frame edge, every feather, and a
sticker scale.
"""

import unittest

import cv2
import numpy as np

import betautils_censor as bu_censor


def reference_sticker_image( image, x, y, w, h, sticker, scale=1.0, feather=0 ):
    """The pre-cache sticker_image body, for one already-chosen sticker."""
    target_w = max( 1, round( w * scale ) )
    target_h = max( 1, round( h * scale ) )
    resized_sticker = cv2.resize( sticker, ( target_w, target_h ), interpolation=cv2.INTER_AREA )
    origin_x = x + w//2 - target_w//2
    origin_y = y + h//2 - target_h//2
    img_h, img_w = image.shape[:2]
    src_x0, src_y0 = max( 0, -origin_x ), max( 0, -origin_y )
    dst_x0, dst_y0 = max( 0, origin_x ), max( 0, origin_y )
    dst_x1 = min( img_w, origin_x + target_w )
    dst_y1 = min( img_h, origin_y + target_h )
    composite_w, composite_h = dst_x1 - dst_x0, dst_y1 - dst_y0
    if composite_w <= 0 or composite_h <= 0:
        return image
    sticker_rgb = resized_sticker[ src_y0:src_y0+composite_h, src_x0:src_x0+composite_w, :3 ]
    sticker_alpha = resized_sticker[
        src_y0:src_y0+composite_h, src_x0:src_x0+composite_w, 3:4 ].astype( np.float32 ) / 255.0
    if feather:
        feathered = bu_censor.feather_mask( ( sticker_alpha[ :, :, 0 ]*255 ).astype( np.uint8 ),
                                            feather, composite_w, composite_h )
        sticker_alpha = ( feathered.astype( np.float32 )/255.0 )[ :, :, None ]
    destination = image[ dst_y0:dst_y1, dst_x0:dst_x1 ].astype( np.float32 )
    blended = sticker_alpha*sticker_rgb.astype( np.float32 ) + ( 1 - sticker_alpha )*destination
    image[ dst_y0:dst_y1, dst_x0:dst_x1 ] = blended.astype( image.dtype )
    return image


class TestStickerCache( unittest.TestCase ):

    KEY = 'test-sticker-cache'

    def setUp( self ):
        rng = np.random.default_rng( 3 )
        sticker = rng.integers( 0, 256, ( 61, 67, 4 ), dtype=np.uint8 )
        sticker[ :, :, 3 ] = cv2.GaussianBlur( sticker[ :, :, 3 ], ( 9, 9 ), 0 )
        self.sticker = sticker
        bu_censor._sticker_cache[ self.KEY ] = [ sticker ]
        self.frame = rng.integers( 0, 256, ( 120, 160, 3 ), dtype=np.uint8 )
        bu_censor.clear_processed_sticker_cache()

    def tearDown( self ):
        bu_censor._sticker_cache.pop( self.KEY, None )
        bu_censor.clear_processed_sticker_cache()

    def style( self, **extra ):
        return dict( { 'type': 'sticker', 'dir': self.KEY }, **extra )

    def assert_matches_reference( self, x, y, w, h, scale=1.0, feather=0, repeats=2 ):
        expected = reference_sticker_image( self.frame.copy(), x, y, w, h, self.sticker,
                                            scale=scale, feather=feather )
        for _ in range( repeats ):     # the second pass is served from the cache
            actual = bu_censor.sticker_image( self.frame.copy(), x, y, w, h,
                                              self.style( scale=scale ), feather=feather,
                                              sticker_seed=0.0 )
            np.testing.assert_array_equal( actual, expected )

    def test_inside_the_frame_at_every_feather( self ):
        for feather in ( 0, 0.15, 0.25, 0.5 ):
            self.assert_matches_reference( 40, 30, 50, 44, feather=feather )

    def test_hanging_off_every_edge( self ):
        for x, y in ( ( -20, 30 ), ( 130, 30 ), ( 40, -15 ), ( 40, 100 ), ( -10, -10 ),
                      ( 140, 110 ) ):
            self.assert_matches_reference( x, y, 50, 44, feather=0.25 )

    def test_scaled_sticker( self ):
        self.assert_matches_reference( 40, 30, 50, 44, scale=1.3, feather=0.2 )

    def test_entirely_outside_is_a_no_op( self ):
        before = self.frame.copy()
        actual = bu_censor.sticker_image( self.frame.copy(), 500, 500, 20, 20, self.style(),
                                          sticker_seed=0.0 )
        np.testing.assert_array_equal( actual, before )

    def test_a_new_size_or_crop_is_never_served_stale( self ):
        self.assert_matches_reference( 40, 30, 50, 44, feather=0.25 )
        self.assert_matches_reference( 40, 30, 52, 44, feather=0.25 )
        self.assert_matches_reference( 40, 30, 50, 44, feather=0.15 )
        self.assert_matches_reference( -5, 30, 50, 44, feather=0.25 )

    def test_cached_planes_are_read_only( self ):
        bu_censor.sticker_image( self.frame.copy(), 40, 30, 50, 44, self.style(), feather=0.2,
                                 sticker_seed=0.0 )
        for premultiplied, inverse_alpha in bu_censor._processed_stickers.values():
            self.assertFalse( premultiplied.flags.writeable )
            self.assertFalse( inverse_alpha.flags.writeable )

    def test_the_cache_stays_inside_its_byte_budget( self ):
        saved = bu_censor._PROCESSED_STICKER_BUDGET_BYTES
        bu_censor._PROCESSED_STICKER_BUDGET_BYTES = 400 * 1024
        try:
            for w in range( 20, 80 ):
                bu_censor.sticker_image( self.frame.copy(), 40, 30, w, 44, self.style(),
                                         sticker_seed=0.0 )
            used = sum( a.nbytes + b.nbytes for a, b in bu_censor._processed_stickers.values() )
            self.assertLessEqual( used, bu_censor._PROCESSED_STICKER_BUDGET_BYTES )
            self.assertEqual( used, bu_censor._processed_stickers_bytes )
            self.assertGreater( len( bu_censor._processed_stickers ), 0 )
        finally:
            bu_censor._PROCESSED_STICKER_BUDGET_BYTES = saved

    def test_the_most_recent_size_survives_eviction( self ):
        saved = bu_censor._PROCESSED_STICKER_BUDGET_BYTES
        bu_censor._PROCESSED_STICKER_BUDGET_BYTES = 400 * 1024
        try:
            for w in range( 20, 80 ):
                bu_censor.sticker_image( self.frame.copy(), 40, 30, w, 44, self.style(),
                                         sticker_seed=0.0 )
            newest = list( bu_censor._processed_stickers )[-1]
            self.assertEqual( newest[2], 79 )
        finally:
            bu_censor._PROCESSED_STICKER_BUDGET_BYTES = saved

    def test_a_size_in_use_is_not_evicted_for_newer_ones( self ):
        # LRU, not FIFO: a track that keeps its size keeps its planes even
        # while other tracks churn through sizes.
        saved = bu_censor._PROCESSED_STICKER_BUDGET_BYTES
        bu_censor._PROCESSED_STICKER_BUDGET_BYTES = 400 * 1024
        try:
            for w in range( 20, 80 ):
                bu_censor.sticker_image( self.frame.copy(), 40, 30, 30, 30, self.style(),
                                         sticker_seed=0.0 )
                bu_censor.sticker_image( self.frame.copy(), 40, 30, w, 44, self.style(),
                                         sticker_seed=0.0 )
            widths = [ key[2] for key in bu_censor._processed_stickers ]
            self.assertIn( 30, widths )
            self.assertLess( len( widths ), 60, "eviction must have happened for this to mean anything" )
        finally:
            bu_censor._PROCESSED_STICKER_BUDGET_BYTES = saved


if __name__ == '__main__':
    unittest.main()
