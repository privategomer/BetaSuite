"""
test_nudenet_v3_adapter.py - regression tests for the NudeNet 3.4.2
detector adapter (detectors/nudenet_v3.py).

Covers:
    - _NATIVE_LABELS / _CLASS_MAP stay in sync as sets (a drift here
      would silently mislabel or drop every detection of the affected
      class - same invariant class as retinanet_v2's
      test_model_class_order_matches_canonical_vocabulary_exactly)
    - every _CLASS_MAP value is a real betaconst.classes key
    - validate_backend_config catches out-of-range candidate_floor/
      nms_iou values
    - _decode_output correctly thresholds, decodes cxcywh->xywh, scales
      padded-model-space back to original-image pixels, and runs NMS,
      using synthetic [N, 22] tensor rows (no real model/ONNX file
      required - same "exercise the code with synthetic tensor data"
      approach test_detector_adapter.py uses for retinanet_v2)
    - _raw_boxes_from_decoded correctly translates native class index
      to canonical string label and applies the global_min_prob floor

Does NOT test actual ONNX inference (get_session()/a real
InferenceSession) - that requires the real vendored model file; out of
scope for a unit test that should run anywhere. That path was verified
manually against the real v3.4-320n.onnx file instead (session loads,
raw_boxes_for_img/raw_boxes_for_imgs both run end-to-end without error
on synthetic random-noise images).
"""

import unittest

import numpy as np

import betaconst
import detectors.nudenet_v3 as nudenet_v3


class TestClassMapping( unittest.TestCase ):

    def test_native_labels_and_class_map_keys_match_exactly( self ):
        # If _NATIVE_LABELS and _CLASS_MAP.keys() ever drift apart
        # (someone edits one without the other), _raw_boxes_from_decoded
        # silently drops every detection of the orphaned class instead
        # of raising - catch that here, not in production.
        self.assertEqual(
            set( nudenet_v3._NATIVE_LABELS ),
            set( nudenet_v3._CLASS_MAP.keys() ) )

    def test_native_labels_has_no_duplicates( self ):
        self.assertEqual(
            len( nudenet_v3._NATIVE_LABELS ),
            len( set( nudenet_v3._NATIVE_LABELS ) ) )

    def test_native_labels_has_eighteen_entries( self ):
        # positional against the real .onnx file's own class order -
        # verified via onnx.load(...).metadata_props, see module
        # docstring. A count drift here means the model file changed
        # out from under this adapter.
        self.assertEqual( len( nudenet_v3._NATIVE_LABELS ), 18 )

    def test_every_class_map_value_is_a_known_canonical_label( self ):
        known = set( betaconst.classes.keys() )
        for native_label, canonical_label in nudenet_v3._CLASS_MAP.items():
            self.assertIn(
                canonical_label, known,
                "%s maps to %r, not in betaconst.classes"%( native_label, canonical_label ) )

    def test_male_breast_exposed_maps_to_exposed_chest( self ):
        # the one judgment-call mapping, explicitly confirmed with the
        # user rather than assumed - regression-guard it specifically
        self.assertEqual(
            nudenet_v3._CLASS_MAP[ 'MALE_BREAST_EXPOSED' ], 'exposed_chest' )

    def test_new_labels_map_to_the_two_added_canonical_classes( self ):
        self.assertEqual(
            nudenet_v3._CLASS_MAP[ 'ARMPITS_COVERED' ], 'covered_armpits' )
        self.assertEqual(
            nudenet_v3._CLASS_MAP[ 'ANUS_COVERED' ], 'covered_anus' )


class TestValidateBackendConfig( unittest.TestCase ):

    def test_defaults_produce_no_errors( self ):
        errors = []
        nudenet_v3.validate_backend_config( {}, errors )
        self.assertEqual( errors, [] )

    def test_candidate_floor_out_of_range_is_caught( self ):
        errors = []
        nudenet_v3.validate_backend_config( { 'candidate_floor': 1.5 }, errors )
        self.assertEqual( len( errors ), 1 )
        self.assertIn( 'candidate_floor', errors[0] )

    def test_negative_nms_iou_is_caught( self ):
        errors = []
        nudenet_v3.validate_backend_config( { 'nms_iou': -0.1 }, errors )
        self.assertEqual( len( errors ), 1 )
        self.assertIn( 'nms_iou', errors[0] )

    def test_non_numeric_candidate_floor_is_caught( self ):
        errors = []
        nudenet_v3.validate_backend_config( { 'candidate_floor': 'high' }, errors )
        self.assertEqual( len( errors ), 1 )


class TestDecodeOutput( unittest.TestCase ):

    def _row( self, cx, cy, w, h, class_index, score, num_classes=18 ):
        class_scores = [ 0.0 ] * num_classes
        class_scores[ class_index ] = score
        return [ cx, cy, w, h ] + class_scores

    def test_single_detection_above_floor_survives( self ):
        # square image (no padding), size=320: one clean detection dead
        # center, class index 3 (FEMALE_BREAST_EXPOSED)
        rows = np.array( [ self._row( 160, 160, 40, 40, 3, 0.9 ) ], dtype=np.float32 )
        decoded = nudenet_v3.decode_output(
            rows.T, pad_w=0, pad_h=0, max_size=320, size=320,
            candidate_floor=0.2, nms_iou=0.45 )
        self.assertEqual( len( decoded ), 1 )
        x, y, w, h, score, class_index = decoded[0]
        self.assertAlmostEqual( x, 140.0, places=3 )
        self.assertAlmostEqual( y, 140.0, places=3 )
        self.assertAlmostEqual( w, 40.0, places=3 )
        self.assertAlmostEqual( h, 40.0, places=3 )
        self.assertAlmostEqual( score, 0.9, places=3 )
        self.assertEqual( class_index, 3 )

    def test_below_candidate_floor_is_dropped( self ):
        rows = np.array( [ self._row( 160, 160, 40, 40, 3, 0.05 ) ], dtype=np.float32 )
        decoded = nudenet_v3.decode_output(
            rows.T, pad_w=0, pad_h=0, max_size=320, size=320,
            candidate_floor=0.2, nms_iou=0.45 )
        self.assertEqual( decoded, [] )

    def test_no_rows_returns_empty_list( self ):
        rows = np.zeros( (0, 22), dtype=np.float32 )
        decoded = nudenet_v3.decode_output(
            rows.T, pad_w=0, pad_h=0, max_size=320, size=320,
            candidate_floor=0.2, nms_iou=0.45 )
        self.assertEqual( decoded, [] )

    def test_padding_scale_maps_back_to_original_image_space( self ):
        # a 640x320 original image padded to 640x640 (pad_h=320), then
        # resized to size=320 for the model (scale = max_size/size = 2.0).
        # A detection at model-space (80, 40, 20, 20) should land at
        # original-space (160, 80, 40, 40).
        rows = np.array( [ self._row( 90, 50, 20, 20, 0, 0.8 ) ], dtype=np.float32 )
        decoded = nudenet_v3.decode_output(
            rows.T, pad_w=0, pad_h=320, max_size=640, size=320,
            candidate_floor=0.2, nms_iou=0.45 )
        self.assertEqual( len( decoded ), 1 )
        x, y, w, h, score, class_index = decoded[0]
        # cx=90,cy=50,w=20,h=20 (model space) -> top-left (80,40) -> x2 scale -> (160,80), w/h -> (40,40)
        self.assertAlmostEqual( x, 160.0, places=3 )
        self.assertAlmostEqual( y, 80.0, places=3 )
        self.assertAlmostEqual( w, 40.0, places=3 )
        self.assertAlmostEqual( h, 40.0, places=3 )

    def test_overlapping_boxes_are_deduped_by_nms( self ):
        rows = np.array( [
            self._row( 160, 160, 40, 40, 3, 0.9 ),
            self._row( 162, 162, 40, 40, 3, 0.85 ),  # near-duplicate of the above
            self._row( 20, 20, 10, 10, 5, 0.6 ),      # unrelated, far away
        ], dtype=np.float32 )
        decoded = nudenet_v3.decode_output(
            rows.T, pad_w=0, pad_h=0, max_size=320, size=320,
            candidate_floor=0.2, nms_iou=0.45 )
        self.assertEqual( len( decoded ), 2 )
        kept_classes = sorted( d[5] for d in decoded )
        self.assertEqual( kept_classes, [ 3, 5 ] )
        # the higher-scoring of the near-duplicate pair should be the one kept
        kept_scores = [ d[4] for d in decoded if d[5] == 3 ]
        self.assertAlmostEqual( kept_scores[0], 0.9, places=3 )


class TestRawBoxesFromDecoded( unittest.TestCase ):

    def test_translates_native_index_to_canonical_string_label( self ):
        # class_index 3 -> FEMALE_BREAST_EXPOSED -> exposed_breast
        decoded = [ ( 10.0, 10.0, 50.0, 50.0, 0.9, 3 ) ]
        raw = nudenet_v3._raw_boxes_from_decoded( decoded, t=3.14, size=320, min_prob_floor=0.0 )
        self.assertEqual( len( raw ), 1 )
        self.assertEqual( raw[0]['class_id'], 'exposed_breast' )
        self.assertIsInstance( raw[0]['class_id'], str )
        self.assertEqual( raw[0]['t'], 3.14 )
        self.assertEqual( raw[0]['x'], 10.0 )
        self.assertEqual( raw[0]['w'], 50.0 )

    def test_below_global_min_prob_floor_is_dropped( self ):
        decoded = [ ( 10.0, 10.0, 50.0, 50.0, 0.3, 3 ) ]
        raw = nudenet_v3._raw_boxes_from_decoded( decoded, t=0.0, size=320, min_prob_floor=0.5 )
        self.assertEqual( raw, [] )

    def test_out_of_range_class_index_is_skipped_not_raised( self ):
        decoded = [ ( 10.0, 10.0, 50.0, 50.0, 0.9, 999 ) ]
        raw = nudenet_v3._raw_boxes_from_decoded( decoded, t=0.0, size=320, min_prob_floor=0.0 )
        self.assertEqual( raw, [] )

    def test_new_canonical_labels_come_through_correctly( self ):
        armpits_covered_index = nudenet_v3._NATIVE_LABELS.index( 'ARMPITS_COVERED' )
        anus_covered_index = nudenet_v3._NATIVE_LABELS.index( 'ANUS_COVERED' )
        decoded = [
            ( 0.0, 0.0, 10.0, 10.0, 0.9, armpits_covered_index ),
            ( 0.0, 0.0, 10.0, 10.0, 0.9, anus_covered_index ),
        ]
        raw = nudenet_v3._raw_boxes_from_decoded( decoded, t=0.0, size=320, min_prob_floor=0.0 )
        labels = sorted( b['class_id'] for b in raw )
        self.assertEqual( labels, [ 'covered_anus', 'covered_armpits' ] )


if __name__ == '__main__':
    unittest.main()
