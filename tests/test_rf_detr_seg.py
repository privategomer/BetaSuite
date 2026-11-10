"""
test_rf_detr_seg.py - the RF-DETR-Seg person backend (segmenters/rf_detr_seg.py).

What each group prevents:
  - Preprocessing: channels left BGR, the ImageNet normalisation wrong or
    missing, a frame letterboxed instead of stretched (so it lands on
    different input pixels than the model was trained on), a mixed batch
    that cannot stack
  - The decoder: a box or mask landing off the person once the stretch is
    undone, especially on a portrait frame where the two axes scale
    differently; the wrong class column (COCO's sparse ids put person at
    1, not 0); logits taken as probabilities; a weak person let in or a
    person at exactly the floor dropped; one query's mask given to
    another; a box off the frame not clamped. Checked against hand-built
    model outputs, so no model file is needed; one smoke test runs the
    real model when it is on disk
  - persons_for_imgs: outputs read by position instead of by name, the
    picture size taken from config instead of from the model file
  - Settings: bad values accepted, a picture_size that cannot take effect
    accepted silently, the cache key missing a setting that changes masks
  - The registry: the backend still reported as not implemented
"""

import os
import unittest
from unittest import mock

import numpy as np

import betautils_person as bu_person
from segmenters import rf_detr_seg

QUERIES = 6
MASK = 96          # small's mask side: 384 / 4


def _outputs( people, queries=QUERIES, mask_side=MASK ):
    """
    Hand-built (boxes, logits, masks) for one image. Each person is
    (cx, cy, w, h, column, logit, mask_logit) with the box as fractions
    of the input; column is the class column given that logit, and every
    query's mask is mask_logit everywhere. Unused queries score nothing.
    """
    boxes = np.full( ( queries, 4 ), 0.5, dtype=np.float32 )
    logits = np.full( ( queries, 91 ), -20.0, dtype=np.float32 )
    masks = np.full( ( queries, mask_side, mask_side ), -20.0, dtype=np.float32 )
    for index, ( cx, cy, w, h, column, logit, mask_logit ) in enumerate( people ):
        boxes[index] = ( cx, cy, w, h )
        logits[ index, column ] = logit
        masks[index] = mask_logit
    return boxes, logits, masks


def _box( x0, y0, x1, y1, img_w, img_h ):
    """An original-pixel box as the model's centre/size fractions."""
    return ( ( x0 + x1 ) / 2 / img_w, ( y0 + y1 ) / 2 / img_h, ( x1 - x0 ) / img_w, ( y1 - y0 ) / img_h )


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

class TestPrep( unittest.TestCase ):

    def test_a_frame_is_stretched_to_the_whole_square_rgb_and_normalised( self ):
        # 1280x720, left half pure red (BGR), right half black.
        frame = np.zeros( ( 720, 1280, 3 ), dtype=np.uint8 )
        frame[ :, :640, 2 ] = 255
        blob = rf_detr_seg.prep_batch( [ frame ], 384 )
        self.assertEqual( blob.shape, ( 1, 3, 384, 384 ) )
        self.assertEqual( blob.dtype, np.float32 )
        red = ( 1.0 - 0.485 ) / 0.229
        self.assertAlmostEqual( float( blob[0, 0, 0, 0] ), red, places=4, msg="R is channel 0" )
        self.assertAlmostEqual( float( blob[0, 1, 0, 0] ), -0.456 / 0.224, places=4 )
        self.assertAlmostEqual( float( blob[0, 2, 0, 0] ), -0.406 / 0.225, places=4 )
        # Stretched, not letterboxed: the top and bottom rows are image, not
        # padding, and the red half is exactly the left half of the input.
        self.assertAlmostEqual( float( blob[0, 0, 383, 0] ), red, places=4 )
        self.assertAlmostEqual( float( blob[0, 0, 200, 191] ), red, places=4 )
        self.assertAlmostEqual( float( blob[0, 0, 200, 192] ), -0.485 / 0.229, places=4 )

    def test_a_portrait_frame_is_stretched_on_its_own_axes( self ):
        # 720x1280: the top quarter white. It fills the top quarter of the
        # square; a letterbox would put it inside a centred column.
        frame = np.zeros( ( 1280, 720, 3 ), dtype=np.uint8 )
        frame[ :320 ] = 255
        blob = rf_detr_seg.prep_batch( [ frame ], 384 )
        white = ( 1.0 - 0.485 ) / 0.229
        self.assertAlmostEqual( float( blob[0, 0, 0, 0] ), white, places=4 )
        self.assertAlmostEqual( float( blob[0, 0, 95, 383] ), white, places=4 )
        self.assertAlmostEqual( float( blob[0, 0, 96, 0] ), -0.485 / 0.229, places=4 )

    def test_the_blob_is_exactly_roboflows_preprocessing( self ):
        # Bilinear (half-pixel, no antialias, which cv2's INTER_LINEAR is),
        # RGB, /255, ImageNet mean and std, channels first, per channel.
        import cv2
        frame = np.random.RandomState( 3 ).randint( 0, 256, ( 90, 160, 3 ), dtype=np.uint8 )
        blob = rf_detr_seg.prep_batch( [ frame ], 48 )
        rgb = cv2.resize( frame, ( 48, 48 ), interpolation=cv2.INTER_LINEAR )[ ..., ::-1 ] / 255.0
        want = ( ( rgb - [ 0.485, 0.456, 0.406 ] ) / [ 0.229, 0.224, 0.225 ] ).transpose( 2, 0, 1 )
        np.testing.assert_allclose( blob[0], want, atol=1e-5 )

    def test_a_mixed_batch_stacks_at_the_models_size( self ):
        blob = rf_detr_seg.prep_batch( [ np.zeros( ( 720, 1280, 3 ), np.uint8 ),
                                         np.zeros( ( 1280, 720, 3 ), np.uint8 ),
                                         np.zeros( ( 31, 17, 3 ), np.uint8 ) ], 312 )
        self.assertEqual( blob.shape, ( 3, 3, 312, 312 ) )


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class TestDecoder( unittest.TestCase ):

    def decode( self, people, img_w=1280, img_h=720, min_prob=0.35, t=0.0, **kwargs ):
        boxes, logits, masks = _outputs( people, **kwargs )
        return rf_detr_seg.decode_one( boxes, logits, masks, img_w, img_h, min_prob, t )

    def test_a_box_and_its_mask_map_back_through_the_stretch( self ):
        # A person at original (200, 100)-(400, 500) in a 1280x720 frame.
        person, = self.decode( [ _box( 200, 100, 400, 500, 1280, 720 ) + ( 1, 5.0, 10.0 ) ], t=2.5 )
        self.assertEqual( person['t'], 2.5 )
        for key, want in zip( ( 'x', 'y', 'w', 'h' ), ( 200, 100, 200, 400 ) ):
            self.assertAlmostEqual( person[key], want, places=2 )
        # A mask cell is 1280/96 = 13.3 px wide and 720/96 = 7.5 px tall.
        mx, my, mw, mh = person['mask_box']
        self.assertTrue( 200 - 13.34 <= mx <= 200 and 400 <= mx + mw <= 400 + 13.34, person['mask_box'] )
        self.assertTrue( 100 - 7.51 <= my <= 100 and 500 <= my + mh <= 500 + 7.51, person['mask_box'] )
        self.assertEqual( person['mask'].shape, ( 54, 15 ) )   # cells 13..66 by 15..29
        self.assertAlmostEqual( mw / person['mask'].shape[1], 1280 / 96, places=3 )
        self.assertAlmostEqual( mh / person['mask'].shape[0], 720 / 96, places=3 )
        self.assertTrue( np.all( person['mask'] > 250 ) )

    def test_a_portrait_frame_scales_each_axis_on_its_own( self ):
        # 720x1280: a person at (100, 200)-(300, 600).
        person, = self.decode( [ _box( 100, 200, 300, 600, 720, 1280 ) + ( 1, 5.0, 10.0 ) ],
                               img_w=720, img_h=1280 )
        for key, want in zip( ( 'x', 'y', 'w', 'h' ), ( 100, 200, 200, 400 ) ):
            self.assertAlmostEqual( person[key], want, places=2 )
        mx, my, mw, mh = person['mask_box']
        self.assertTrue( 100 - 7.51 <= mx <= 100 and 300 <= mx + mw <= 300 + 7.51, person['mask_box'] )
        self.assertTrue( 200 - 13.34 <= my <= 200 and 600 <= my + mh <= 600 + 13.34, person['mask_box'] )

    def test_the_mask_crop_is_the_boxes_own_cells( self ):
        # Mask logits that say where they came from: the crop has to be the
        # cells under the box, not the top-left corner of the mask.
        boxes, logits, masks = _outputs( [ _box( 640, 360, 960, 720, 1280, 720 ) + ( 1, 5.0, -10.0 ) ] )
        masks[ 0, 48:, 48:72 ] = 10.0        # the bottom-right quarter's left half: the box's cells
        person, = rf_detr_seg.decode_one( boxes, logits, masks, 1280, 720, 0.35, 0.0 )
        self.assertEqual( person['mask'].shape, ( 48, 24 ) )
        self.assertTrue( np.all( person['mask'] > 250 ), "every cell under the box is the person" )
        self.assertEqual( person['mask_box'], ( 640.0, 360.0, 320.0, 360.0 ) )

    def test_person_is_column_1_and_other_classes_are_not_people( self ):
        box = _box( 200, 100, 400, 500, 1280, 720 )
        self.assertEqual( self.decode( [ box + ( 0, 8.0, 10.0 ) ] ), [], "column 0 is COCO's unused id" )
        self.assertEqual( self.decode( [ box + ( 3, 8.0, 10.0 ) ] ), [], "column 3 is a car" )
        self.assertEqual( len( self.decode( [ box + ( 1, 8.0, 10.0 ) ] ) ), 1 )

    def test_scores_are_the_sigmoid_of_the_logits_and_the_floor_is_inclusive( self ):
        box = _box( 200, 100, 400, 500, 1280, 720 )
        person, = self.decode( [ box + ( 1, 0.0, 10.0 ) ], min_prob=0.5 )
        self.assertEqual( person['score'], 0.5, "logit 0 is probability one half, kept at a 0.5 floor" )
        person, = self.decode( [ box + ( 1, 1.0, 10.0 ) ] )
        self.assertAlmostEqual( person['score'], 1 / ( 1 + np.exp( -1.0 ) ), places=5 )
        # Logit -0.5 is probability 0.38: kept at 0.35, dropped at 0.4.
        self.assertEqual( len( self.decode( [ box + ( 1, -0.5, 10.0 ) ], min_prob=0.35 ) ), 1 )
        self.assertEqual( self.decode( [ box + ( 1, -0.5, 10.0 ) ], min_prob=0.4 ), [] )
        # A raw logit of 0.4 would pass a 0.35 floor if read as a probability; its sigmoid is 0.6.
        self.assertEqual( self.decode( [ box + ( 1, 0.4, 10.0 ) ], min_prob=0.65 ), [] )

    def test_the_mask_is_the_querys_own_and_its_probability_is_the_sigmoid( self ):
        people = self.decode( [ _box( 100, 100, 300, 500, 1280, 720 ) + ( 1, 4.0, 10.0 ),
                                _box( 700, 100, 900, 500, 1280, 720 ) + ( 1, 3.0, -10.0 ),
                                _box( 1000, 100, 1200, 500, 1280, 720 ) + ( 1, 2.0, 0.0 ) ] )
        self.assertEqual( len( people ), 3 )
        on, off, half = people
        self.assertTrue( np.all( on['mask'] > 250 ) )
        self.assertTrue( np.all( off['mask'] < 5 ) )
        self.assertTrue( np.all( half['mask'] == 128 ), "0.5 rounds to 128, the INSIDE threshold" )

    def test_people_come_best_first_and_overlaps_are_not_merged( self ):
        # No NMS: two nested boxes are kept, since the inner one may be a
        # second person. Merging could hide a woman in front of a man.
        people = self.decode( [ _box( 200, 100, 400, 500, 1280, 720 ) + ( 1, 1.0, 10.0 ),
                                _box( 210, 110, 400, 500, 1280, 720 ) + ( 1, 3.0, 10.0 ) ] )
        self.assertEqual( len( people ), 2 )
        self.assertGreater( people[0]['score'], people[1]['score'] )
        self.assertAlmostEqual( people[0]['x'], 210, places=2 )

    def test_a_box_off_the_frame_is_clamped_inside_it( self ):
        # Regression is unbounded: (-100, -50)-(300, 800) in a 1280x720 frame.
        person, = self.decode( [ _box( -100, -50, 300, 800, 1280, 720 ) + ( 1, 5.0, 10.0 ) ] )
        self.assertEqual( ( person['x'], person['y'] ), ( 0.0, 0.0 ) )
        self.assertAlmostEqual( person['w'], 300, places=2 )
        self.assertAlmostEqual( person['h'], 720, places=2 )
        mx, my, mw, mh = person['mask_box']
        self.assertTrue( mx >= 0 and my >= 0 and my + mh <= 720.001, person['mask_box'] )
        right, = self.decode( [ _box( 1000, 600, 1500, 900, 1280, 720 ) + ( 1, 5.0, 10.0 ) ] )
        self.assertAlmostEqual( right['x'] + right['w'], 1280, places=2 )
        self.assertAlmostEqual( right['y'] + right['h'], 720, places=2 )
        mx, my, mw, mh = right['mask_box']
        self.assertTrue( mx + mw <= 1280.001 and my + mh <= 720.001, right['mask_box'] )

    def test_a_box_wholly_outside_or_thinner_than_a_pixel_is_dropped( self ):
        self.assertEqual( self.decode( [ _box( 1300, 100, 1400, 500, 1280, 720 ) + ( 1, 5.0, 10.0 ) ] ), [] )
        self.assertEqual( self.decode( [ _box( 200, 100, 200.5, 500, 1280, 720 ) + ( 1, 5.0, 10.0 ) ] ), [] )

    def test_a_non_square_mask_scales_each_axis_by_its_own_side( self ):
        # The shipped exports are square, but rfdetr can export other
        # shapes; the decoder must not assume Hm == Wm.
        boxes, logits, masks = _outputs( [ _box( 640, 360, 960, 720, 1280, 720 ) + ( 1, 5.0, 10.0 ) ],
                                         mask_side=96 )
        masks = masks[ :, :48, : ]         # 48 rows, 96 columns
        person, = rf_detr_seg.decode_one( boxes, logits, masks, 1280, 720, 0.35, 0.0 )
        self.assertEqual( person['mask'].shape, ( 24, 24 ) )
        self.assertEqual( person['mask_box'], ( 640.0, 360.0, 320.0, 360.0 ) )

    def test_a_box_at_the_far_edge_never_claims_a_cell_past_the_mask( self ):
        # At 297 px, 297 * (96 / 297) rounds a hair past 96; unclamped, the
        # mask would be one cell narrower than the rectangle it claims.
        person, = self.decode( [ _box( 150, 150, 297, 297, 297, 297 ) + ( 1, 5.0, 10.0 ) ],
                               img_w=297, img_h=297 )
        mx, my, mw, mh = person['mask_box']
        self.assertLessEqual( mx + mw, 297 + 1e-6 )
        self.assertLessEqual( my + mh, 297 + 1e-6 )
        self.assertAlmostEqual( mw / person['mask'].shape[1], 297 / 96, places=6 )
        self.assertAlmostEqual( mh / person['mask'].shape[0], 297 / 96, places=6 )

    def test_a_tiny_box_still_gets_one_mask_cell( self ):
        # Inside one cell each way (x cells 15.4..15.6, y 13.6..13.9), clear of a cell edge.
        person, = self.decode( [ _box( 205, 102, 208, 104, 1280, 720 ) + ( 1, 5.0, 10.0 ) ] )
        self.assertEqual( person['mask'].shape, ( 1, 1 ) )

    def test_nobody_scoring_means_no_people( self ):
        self.assertEqual( self.decode( [] ), [] )


# ---------------------------------------------------------------------------
# persons_for_imgs, with a stand-in session
# ---------------------------------------------------------------------------

class _Port:
    def __init__( self, name, shape ):
        self.name, self.shape = name, shape


class _Session:
    """Answers with the person in _outputs, in a shuffled output order."""

    def __init__( self, size ):
        self.size = size
        self.fed = None

    def get_inputs( self ):
        return [ _Port( 'input', [ 'batch', 3, self.size, self.size ] ) ]

    def get_outputs( self ):
        return [ _Port( 'masks', None ), _Port( 'dets', None ), _Port( 'labels', None ) ]

    def run( self, names, feed ):
        self.fed = feed['input']
        batch = self.fed.shape[0]
        boxes, logits, masks = _outputs( [ ( 0.5, 0.5, 0.5, 0.5, 1, 5.0, 10.0 ) ], mask_side=self.size // 4 )
        return [ np.stack( [ masks ] * batch ), np.stack( [ boxes ] * batch ), np.stack( [ logits ] * batch ) ]


class TestPersonsForImgs( unittest.TestCase ):

    def test_outputs_are_read_by_name_and_each_image_keeps_its_own_size( self ):
        session = _Session( 312 )
        people = rf_detr_seg.persons_for_imgs( [ np.zeros( ( 720, 1280, 3 ), np.uint8 ),
                                                 np.zeros( ( 1280, 720, 3 ), np.uint8 ) ],
                                               session, [ 1.0, 1.1 ], { 'model_variant': 'small' } )
        self.assertEqual( session.fed.shape, ( 2, 3, 312, 312 ), "the size is the model file's" )
        self.assertEqual( [ p['t'] for p in people ], [ 1.0, 1.1 ] )
        for person, ( w, h ) in zip( people, ( ( 1280, 720 ), ( 720, 1280 ) ) ):
            self.assertAlmostEqual( person['x'], w / 4, places=2 )
            self.assertAlmostEqual( person['h'], h / 2, places=2 )

    def test_the_real_model_runs_when_it_is_on_disk( self ):
        path = rf_detr_seg.VARIANTS['nano']['model_path']
        if not os.path.exists( path ):
            self.skipTest( 'rf-detr-seg-nano.onnx not exported here (setup does it)' )
        import onnxruntime
        session = onnxruntime.InferenceSession( path, providers=[ 'CPUExecutionProvider' ] )
        self.assertEqual( [ o.name for o in session.get_outputs() ], [ 'dets', 'labels', 'masks' ] )
        people = rf_detr_seg.persons_for_imgs( [ np.zeros( ( 360, 640, 3 ), np.uint8 ) ] * 2, session,
                                               [ 0.0, 0.1 ], { 'model_variant': 'nano' } )
        self.assertEqual( people, [], "a black frame has nobody in it" )


# ---------------------------------------------------------------------------
# Settings, identity, registry
# ---------------------------------------------------------------------------

class TestSettings( unittest.TestCase ):

    def setUp( self ):
        self._env = os.environ.pop( bu_person.VARIANT_OVERRIDE_ENV_VAR, None )

    def tearDown( self ):
        os.environ.pop( bu_person.VARIANT_OVERRIDE_ENV_VAR, None )
        if self._env is not None:
            os.environ[ bu_person.VARIANT_OVERRIDE_ENV_VAR ] = self._env

    def errors( self, config ):
        errors = []
        rf_detr_seg.validate_config( config, errors )
        return errors

    def test_the_defaults_and_every_size_validate( self ):
        self.assertEqual( self.errors( {} ), [] )
        for name in rf_detr_seg.VARIANTS:
            self.assertEqual( self.errors( { 'model_variant': name, 'min_prob': 0.5, 'batch_size': 8 } ), [] )

    def test_bad_settings_are_rejected_and_named( self ):
        for settings, word in ( ( { 'model_variant': 's' }, 'model_variant' ),
                                ( { 'model_variant': 'xlarge' }, 'model_variant' ),
                                ( { 'min_prob': 0 }, 'min_prob' ), ( { 'min_prob': 1 }, 'min_prob' ),
                                ( { 'min_prob': True }, 'min_prob' ), ( { 'min_prob': '0.4' }, 'min_prob' ),
                                ( { 'batch_size': 0 }, 'batch_size' ), ( { 'batch_size': 2.0 }, 'batch_size' ),
                                ( { 'batch_size': True }, 'batch_size' ),
                                ( { 'picture_size': 384 }, 'picture_size' ) ):
            errors = self.errors( settings )
            self.assertTrue( errors and all( word in e for e in errors ), ( settings, errors ) )

    def test_the_variant_picks_the_model_and_the_override_wins( self ):
        self.assertTrue( rf_detr_seg.resolve_config( {} )['model_path'].endswith( 'rf-detr-seg-small.onnx' ) )
        os.environ[ bu_person.VARIANT_OVERRIDE_ENV_VAR ] = 'nano'
        self.assertTrue( rf_detr_seg.resolve_config( { 'model_variant': 'medium' } )['model_path']
                         .endswith( 'rf-detr-seg-nano.onnx' ) )

    def test_identity_changes_with_every_setting_that_changes_the_masks( self ):
        base = rf_detr_seg.identity( {} )
        for change in ( { 'model_variant': 'medium' }, { 'model_path': '/x/custom.onnx' }, { 'min_prob': 0.5 } ):
            self.assertNotEqual( rf_detr_seg.identity( change ), base, change )
        self.assertEqual( rf_detr_seg.identity( { 'batch_size': 16 } ), base, "batch size changes speed only" )
        self.assertEqual( rf_detr_seg.identity( { 'min_prob': 0.35 } ), base )

    def test_the_registry_reports_it_finished_with_its_real_sizes( self ):
        self.assertTrue( bu_person.is_implemented( 'rf_detr_seg' ) )
        self.assertEqual( bu_person.variant_names( 'rf_detr_seg' ), [ 'medium', 'nano', 'small' ] )
        for name, variant in rf_detr_seg.VARIANTS.items():
            self.assertEqual( os.path.basename( variant['model_path'] ), 'rf-detr-seg-%s.onnx'%( name, ) )
            self.assertEqual( variant['native_size'] % 12, 0, "a multiple of the patch size" )

    def test_the_session_is_built_the_way_every_backend_builds_one( self ):
        with mock.patch.object( rf_detr_seg.bu_detector, 'build_onnx_session', return_value='s' ) as build:
            self.assertEqual( rf_detr_seg.get_session( { 'model_variant': 'nano' } ), 's' )
        self.assertTrue( build.call_args[0][0].endswith( 'rf-detr-seg-nano.onnx' ) )
        self.assertEqual( build.call_args[1]['backend_name'], 'person:rf_detr_seg' )


if __name__ == '__main__':
    unittest.main()
