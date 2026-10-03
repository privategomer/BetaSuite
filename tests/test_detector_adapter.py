"""
test_detector_adapter.py - regression tests for the detector-
adapter interface (betautils_detector.py, detectors/retinanet_v2.py).

Covers:
    - betaconst.classes is dict-shaped and has the expected label set
    - betautils_detector.get_detector() resolves the configured backend
      and returns something implementing the expected interface
    - detectors/retinanet_v2's _MODEL_CLASS_ORDER exactly matches
      betaconst.classes' canonical vocabulary (a mismatch here would
      silently mislabel every detection - this is the single most
      important invariant this adapter has to hold)
    - raw_boxes_from_model_output correctly translates synthetic model
      tensor output into raw box dicts with canonical STRING class_ids
      (not the old numeric index), including the global_min_prob floor
    - betautils_censor.process_raw_box correctly consumes a raw box
      dict shaped by the new adapter (string class_id, not int)

Does NOT test actual ONNX inference (get_session()/a real
InferenceSession) - that requires the real model file and, per
betaconfig.gpu_enabled, a CUDA device; out of scope for a unit test
that should run anywhere. These tests exercise everything up to that
boundary using synthetic tensor data instead.
"""

import os
import unittest

import numpy as np

import betaconfig
import betaconst
import betautils_detector as bu_detector
import betautils_censor as bu_censor
import detectors.retinanet_v2 as retinanet_v2


class TestCanonicalVocabulary( unittest.TestCase ):

    def test_classes_is_dict_shaped( self ):
        self.assertIsInstance( betaconst.classes, dict )

    def test_classes_has_eighteen_labels( self ):
        # was originally 16; grew to 18 when the nudenet_v3 adapter
        # added covered_armpits/covered_anus (real distinctions the
        # RetinaNet-era vocabulary never had - see betaconst.py and
        # detectors/nudenet_v3.py's module docstrings)
        self.assertEqual( len( betaconst.classes ), 18 )

    def test_every_class_has_an_rgb_debug_color( self ):
        for label, color in betaconst.classes.items():
            self.assertIsInstance( label, str )
            self.assertEqual( len( color ), 3 )
            for channel in color:
                self.assertTrue( 0 <= channel <= 255 )


class TestDetectorRegistry( unittest.TestCase ):

    def test_get_detector_resolves_configured_backend( self ):
        detector = bu_detector.get_detector( 'retinanet_v2' )
        self.assertTrue( hasattr( detector, 'get_session' ) )
        self.assertTrue( hasattr( detector, 'raw_boxes_for_img' ) )
        self.assertTrue( hasattr( detector, 'raw_boxes_for_imgs' ) )

    def test_unknown_backend_raises_with_valid_choices_listed( self ):
        with self.assertRaises( ValueError ) as ctx:
            bu_detector.get_detector( 'nonexistent_backend' )
        self.assertIn( 'retinanet_v2', str( ctx.exception ) )


class TestNestedBackendConfigResolution( unittest.TestCase ):
    """
    betaconfig.detector_backend is a single nested dict -
    {'selected': <name>, <name>: {<that backend's own config>}, ...} -
    not two separate flat settings. These cover selected_backend_name/
    get_detector/get_backend_config actually reading that shape
    correctly, including the case detector_backend is unset entirely
    (old-style config / fresh checkout predating this setting).
    """

    def setUp( self ):
        # save/restore so this doesn't leak into other tests via the
        # shared betaconfig module
        self._had_attr = hasattr( betaconfig, 'detector_backend' )
        if self._had_attr:
            self._original = betaconfig.detector_backend

    def tearDown( self ):
        if self._had_attr:
            betaconfig.detector_backend = self._original
        elif hasattr( betaconfig, 'detector_backend' ):
            del betaconfig.detector_backend

    def test_selected_backend_name_reads_the_selected_key( self ):
        betaconfig.detector_backend = { 'selected': 'nudenet_v3', 'nudenet_v3': {}, 'retinanet_v2': {} }
        self.assertEqual( bu_detector.selected_backend_name(), 'nudenet_v3' )

    def test_get_detector_with_no_arg_resolves_via_selected( self ):
        betaconfig.detector_backend = { 'selected': 'nudenet_v3', 'nudenet_v3': {}, 'retinanet_v2': {} }
        detector = bu_detector.get_detector()
        self.assertEqual( detector.__name__, 'detectors.nudenet_v3' )

    def test_get_backend_config_returns_the_selected_backends_own_block( self ):
        betaconfig.detector_backend = {
            'selected': 'nudenet_v3',
            'nudenet_v3': { 'candidate_floor': 0.3 },
            'retinanet_v2': {},
        }
        self.assertEqual( bu_detector.get_backend_config(), { 'candidate_floor': 0.3 } )

    def test_get_backend_config_accepts_an_explicit_name_other_than_selected( self ):
        betaconfig.detector_backend = {
            'selected': 'nudenet_v3',
            'nudenet_v3': { 'candidate_floor': 0.3 },
            'retinanet_v2': {},
        }
        # explicitly asking for a backend's block that ISN'T the
        # selected one still works - useful for inspection/validation
        # without switching what's active
        self.assertEqual( bu_detector.get_backend_config( 'retinanet_v2' ), {} )

    def test_missing_detector_backend_entirely_falls_back_to_retinanet_v2( self ):
        if hasattr( betaconfig, 'detector_backend' ):
            del betaconfig.detector_backend
        self.assertEqual( bu_detector.selected_backend_name(), 'retinanet_v2' )
        self.assertEqual( bu_detector.get_backend_config(), {} )


class TestGetNnBatchSize( unittest.TestCase ):
    """
    get_nn_batch_size's three-tier resolution order: a backend's own
    detector_backend[<name>]['nn_batch_size'] override first, then the
    old flat top-level betaconfig.nn_batch_size as a backward-compatible
    fallback, then 1 if neither is set at all. Added alongside the
    2026-09-15 change moving nn_batch_size from one shared setting into
    per-backend config (real hardware evidence: retinanet_v2 OOMs at
    batch sizes nudenet_v3 handles fine - see CONFIG_REFERENCE.md).
    """

    def setUp( self ):
        self._had_backend_attr = hasattr( betaconfig, 'detector_backend' )
        if self._had_backend_attr:
            self._original_backend = betaconfig.detector_backend
        self._had_top_level_attr = hasattr( betaconfig, 'nn_batch_size' )
        if self._had_top_level_attr:
            self._original_top_level = betaconfig.nn_batch_size

    def tearDown( self ):
        if self._had_backend_attr:
            betaconfig.detector_backend = self._original_backend
        elif hasattr( betaconfig, 'detector_backend' ):
            del betaconfig.detector_backend
        if self._had_top_level_attr:
            betaconfig.nn_batch_size = self._original_top_level
        elif hasattr( betaconfig, 'nn_batch_size' ):
            del betaconfig.nn_batch_size

    def test_backend_specific_override_wins( self ):
        betaconfig.detector_backend = {
            'selected': 'retinanet_v2',
            'retinanet_v2': { 'nn_batch_size': 2 },
            'nudenet_v3': { 'nn_batch_size': 3 },
        }
        betaconfig.nn_batch_size = 1
        self.assertEqual( bu_detector.get_nn_batch_size(), 2 )
        self.assertEqual( bu_detector.get_nn_batch_size( 'nudenet_v3' ), 3 )

    def test_falls_back_to_the_defaults_tier_when_backend_has_no_override( self ):
        # nn_batch_size keeps its detector_backend['defaults'] tier: it is
        # a VRAM/throughput knob, so one value can legitimately cover a
        # backend that has not stated a preference.
        betaconfig.detector_backend = {
            'selected': 'retinanet_v2',
            'defaults': { 'nn_batch_size': 4 },
            'retinanet_v2': {},
            'nudenet_v3': {},
        }
        self.assertEqual( bu_detector.get_nn_batch_size(), 4 )

    def test_the_removed_top_level_setting_is_ignored( self ):
        # betaconfig.nn_batch_size stopped being read. A stale value
        # there must not quietly win; validate_config reports it instead.
        betaconfig.detector_backend = {
            'selected': 'retinanet_v2',
            'retinanet_v2': {},
            'nudenet_v3': {},
        }
        betaconfig.nn_batch_size = 4
        self.assertEqual( bu_detector.get_nn_batch_size(), 1 )

    def test_falls_back_to_one_when_nothing_set_at_all( self ):
        if hasattr( betaconfig, 'detector_backend' ):
            del betaconfig.detector_backend
        if hasattr( betaconfig, 'nn_batch_size' ):
            del betaconfig.nn_batch_size
        self.assertEqual( bu_detector.get_nn_batch_size(), 1 )

    def test_defaults_to_selected_backend_when_no_name_given( self ):
        betaconfig.detector_backend = {
            'selected': 'nudenet_v3',
            'retinanet_v2': { 'nn_batch_size': 2 },
            'nudenet_v3': { 'nn_batch_size': 3 },
        }
        self.assertEqual( bu_detector.get_nn_batch_size(), 3 )


class TestGetClassSuppression( unittest.TestCase ):
    """
    get_class_suppression's three-tier resolution order: a backend's own
    detector_backend[<name>]['class_suppression'] override first, then
    the old flat top-level betaconfig.class_suppression as a backward-
    compatible fallback, then {} (no suppression) if neither is set.
    Added alongside the 2026-09-15 change moving class_suppression from
    one shared setting into per-backend config (real evidence: nudenet_v3's
    boxes run structurally smaller/tighter than retinanet_v2's for the
    same labels on the same footage, and suppression-pair IoU distributions
    for nudenet_v3 run far lower than retinanet_v2's for the same label
    pairs - a single IoU threshold set tuned against retinanet-era data
    can't be assumed correct for nudenet_v3 - see CONFIG_REFERENCE.md).
    """

    def setUp( self ):
        self._had_backend_attr = hasattr( betaconfig, 'detector_backend' )
        if self._had_backend_attr:
            self._original_backend = betaconfig.detector_backend
        self._had_top_level_attr = hasattr( betaconfig, 'class_suppression' )
        if self._had_top_level_attr:
            self._original_top_level = betaconfig.class_suppression

    def tearDown( self ):
        if self._had_backend_attr:
            betaconfig.detector_backend = self._original_backend
        elif hasattr( betaconfig, 'detector_backend' ):
            del betaconfig.detector_backend
        if self._had_top_level_attr:
            betaconfig.class_suppression = self._original_top_level
        elif hasattr( betaconfig, 'class_suppression' ):
            del betaconfig.class_suppression

    def test_backend_specific_override_wins( self ):
        retinanet_rules = { 'exposed_breast': [ { 'suppressed_by': 'covered_breast', 'margin': 0.13, 'min_iou': 0.55 } ] }
        nudenet_rules = { 'exposed_breast': [ { 'suppressed_by': 'covered_breast', 'margin': 0.05, 'min_iou': 0.1 } ] }
        betaconfig.detector_backend = {
            'selected': 'retinanet_v2',
            'retinanet_v2': { 'class_suppression': retinanet_rules },
            'nudenet_v3': { 'class_suppression': nudenet_rules },
        }
        betaconfig.class_suppression = {}
        self.assertEqual( bu_detector.get_class_suppression(), retinanet_rules )
        self.assertEqual( bu_detector.get_class_suppression( 'nudenet_v3' ), nudenet_rules )

    def test_the_removed_top_level_ruleset_is_ignored( self ):
        # betaconfig.class_suppression stopped being read: label
        # vocabularies differ between models, so a shared ruleset can name
        # classes a backend has never heard of. A stale value there must
        # not quietly apply; validate_config reports it instead.
        old_rules = { 'exposed_vulva': [ { 'suppressed_by': 'covered_genitalia_f', 'margin': 0.1, 'min_iou': 0.4 } ] }
        betaconfig.detector_backend = {
            'selected': 'retinanet_v2',
            'retinanet_v2': {},
            'nudenet_v3': {},
        }
        betaconfig.class_suppression = old_rules
        self.assertEqual( bu_detector.get_class_suppression(), {} )

    def test_falls_back_to_empty_dict_when_nothing_set_at_all( self ):
        if hasattr( betaconfig, 'detector_backend' ):
            del betaconfig.detector_backend
        if hasattr( betaconfig, 'class_suppression' ):
            del betaconfig.class_suppression
        self.assertEqual( bu_detector.get_class_suppression(), {} )

    def test_backend_with_deliberately_empty_override_does_not_fall_back( self ):
        # nudenet_v3's real config sets class_suppression explicitly to {}
        # (pending its own tuning pass) rather than omitting the key - this
        # must NOT fall through to the top-level attribute, since an
        # explicit empty block means "no suppression for this backend",
        # not "unset".
        old_rules = { 'exposed_vulva': [ { 'suppressed_by': 'covered_genitalia_f', 'margin': 0.1, 'min_iou': 0.4 } ] }
        betaconfig.detector_backend = {
            'selected': 'nudenet_v3',
            'retinanet_v2': { 'class_suppression': old_rules },
            'nudenet_v3': { 'class_suppression': {} },
        }
        betaconfig.class_suppression = old_rules
        self.assertEqual( bu_detector.get_class_suppression( 'nudenet_v3' ), {} )

    def test_defaults_to_selected_backend_when_no_name_given( self ):
        retinanet_rules = { 'exposed_breast': [ { 'suppressed_by': 'covered_breast', 'margin': 0.13, 'min_iou': 0.55 } ] }
        nudenet_rules = { 'exposed_breast': [ { 'suppressed_by': 'covered_breast', 'margin': 0.05, 'min_iou': 0.1 } ] }
        betaconfig.detector_backend = {
            'selected': 'nudenet_v3',
            'retinanet_v2': { 'class_suppression': retinanet_rules },
            'nudenet_v3': { 'class_suppression': nudenet_rules },
        }
        self.assertEqual( bu_detector.get_class_suppression(), nudenet_rules )


class TestBackendOverrideEnvVar( unittest.TestCase ):
    """
    BETASUITE_DETECTOR_BACKEND_OVERRIDE lets a caller (the
    compare_backends_orchestrator.py tool) flip which backend a real
    betatv.py/betastare.py subprocess run uses without ever touching
    betaconfig.py on disk. These cover that it wins over
    betaconfig.detector_backend['selected'] when set, and has zero
    effect when unset (the default, and every other test in this
    file's own environment).
    """

    ENV_VAR = 'BETASUITE_DETECTOR_BACKEND_OVERRIDE'

    def setUp( self ):
        self._had_attr = hasattr( betaconfig, 'detector_backend' )
        if self._had_attr:
            self._original = betaconfig.detector_backend
        self._had_env = self.ENV_VAR in os.environ
        if self._had_env:
            self._original_env = os.environ[ self.ENV_VAR ]

    def tearDown( self ):
        if self._had_attr:
            betaconfig.detector_backend = self._original
        elif hasattr( betaconfig, 'detector_backend' ):
            del betaconfig.detector_backend
        if self._had_env:
            os.environ[ self.ENV_VAR ] = self._original_env
        elif self.ENV_VAR in os.environ:
            del os.environ[ self.ENV_VAR ]

    def test_env_override_wins_over_betaconfig_selected( self ):
        betaconfig.detector_backend = { 'selected': 'retinanet_v2', 'nudenet_v3': {}, 'retinanet_v2': {} }
        os.environ[ self.ENV_VAR ] = 'nudenet_v3'
        self.assertEqual( bu_detector.selected_backend_name(), 'nudenet_v3' )

    def test_unset_env_var_leaves_betaconfig_selected_in_effect( self ):
        betaconfig.detector_backend = { 'selected': 'retinanet_v2', 'nudenet_v3': {}, 'retinanet_v2': {} }
        if self.ENV_VAR in os.environ:
            del os.environ[ self.ENV_VAR ]
        self.assertEqual( bu_detector.selected_backend_name(), 'retinanet_v2' )

    def test_env_override_also_flows_through_get_detector_and_get_backend_config( self ):
        betaconfig.detector_backend = {
            'selected': 'retinanet_v2',
            'nudenet_v3': { 'candidate_floor': 0.3 },
            'retinanet_v2': {},
        }
        os.environ[ self.ENV_VAR ] = 'nudenet_v3'
        detector = bu_detector.get_detector()
        self.assertEqual( detector.__name__, 'detectors.nudenet_v3' )
        self.assertEqual( bu_detector.get_backend_config(), { 'candidate_floor': 0.3 } )


class TestRetinanetV2ClassMapping( unittest.TestCase ):

    def test_model_class_order_is_a_subset_of_canonical_vocabulary( self ):
        # Every label retinanet_v2 can produce must be a real canonical
        # label (a drift here would silently mislabel a detection with
        # a string nothing downstream recognizes) - but NOT the reverse
        # any more: since nudenet_v3 was added, betaconst.classes is the
        # UNION across every adapter's native vocabulary, and this
        # model's own 16-class training set will never cover
        # nudenet_v3-only distinctions like covered_armpits/covered_anus
        # (this model was never trained on them - see
        # detectors/nudenet_v3.py's module docstring for that history).
        self.assertTrue(
            set( retinanet_v2._MODEL_CLASS_ORDER ).issubset( set( betaconst.classes.keys() ) ) )

    def test_model_class_order_is_exactly_the_original_sixteen( self ):
        # Regression guard for the specific invariant that matters here:
        # this adapter's own trained class set shouldn't silently grow
        # or shrink. If the underlying .onnx model is ever re-exported
        # with a different class list, update this alongside it
        # deliberately - never let it drift unnoticed.
        self.assertEqual( len( retinanet_v2._MODEL_CLASS_ORDER ), 16 )
        self.assertNotIn( 'covered_armpits', retinanet_v2._MODEL_CLASS_ORDER )
        self.assertNotIn( 'covered_anus', retinanet_v2._MODEL_CLASS_ORDER )

    def test_model_class_order_has_no_duplicates( self ):
        self.assertEqual(
            len( retinanet_v2._MODEL_CLASS_ORDER ),
            len( set( retinanet_v2._MODEL_CLASS_ORDER ) ) )


class TestRawBoxesFromModelOutput( unittest.TestCase ):

    def _synthetic_output( self, detections ):
        """
        Build a synthetic [boxes, scores, classes] model-output tuple
        for one image, with the given detections placed in the first
        len(detections) of 300 fixed slots (the rest left at score 0,
        i.e. below any realistic min_prob floor - simulating the
        model's real fixed-300-detection output shape).

        Args:
            detections: list of (x1, y1, x2, y2, score, label) tuples.

        Returns:
            [boxes, scores, classes] as raw_boxes_from_model_output expects.
        """
        boxes = np.zeros( (1, 300, 4), dtype=np.float32 )
        scores = np.zeros( (1, 300), dtype=np.float32 )
        classes = np.zeros( (1, 300), dtype=np.int32 )
        for i, (x1, y1, x2, y2, score, label) in enumerate( detections ):
            boxes[0, i] = [x1, y1, x2, y2]
            scores[0, i] = score
            classes[0, i] = retinanet_v2._MODEL_CLASS_ORDER.index( label )
        return [ boxes, scores, classes ]

    def test_translates_class_index_to_canonical_string_label( self ):
        model_output = self._synthetic_output( [
            (10, 10, 60, 60, 0.9, 'exposed_breast'),
            (100, 100, 150, 180, 0.75, 'exposed_vulva'),
        ] )
        raw = retinanet_v2.raw_boxes_from_model_output( model_output, scale_array=[1.0], t_array=[3.14] )
        self.assertEqual( len( raw[0] ), 2 )
        self.assertEqual( raw[0][0]['class_id'], 'exposed_breast' )
        self.assertEqual( raw[0][1]['class_id'], 'exposed_vulva' )
        self.assertIsInstance( raw[0][0]['class_id'], str )

    def test_below_global_min_prob_floor_is_dropped( self ):
        # everything past the explicitly-set detections is score=0,
        # which must sit at or below any realistic global_min_prob floor
        model_output = self._synthetic_output( [
            (10, 10, 60, 60, 0.9, 'exposed_breast'),
        ] )
        raw = retinanet_v2.raw_boxes_from_model_output( model_output, scale_array=[1.0], t_array=[0.0] )
        self.assertEqual( len( raw[0] ), 1 )

    def test_timestamp_and_geometry_pass_through_correctly( self ):
        model_output = self._synthetic_output( [
            (10, 10, 60, 60, 0.9, 'exposed_breast'),
        ] )
        raw = retinanet_v2.raw_boxes_from_model_output( model_output, scale_array=[1.0], t_array=[7.5] )
        box = raw[0][0]
        self.assertEqual( box['t'], 7.5 )
        self.assertEqual( box['x'], 10.0 )
        self.assertEqual( box['y'], 10.0 )
        self.assertEqual( box['w'], 50.0 )
        self.assertEqual( box['h'], 50.0 )

    def test_resize_scale_is_applied_to_geometry( self ):
        model_output = self._synthetic_output( [
            (20, 20, 120, 120, 0.9, 'exposed_breast'),
        ] )
        # scale=2.0 means the model saw a 2x-upscaled image - coordinates
        # divide back down by scale to land in original-image pixels
        raw = retinanet_v2.raw_boxes_from_model_output( model_output, scale_array=[2.0], t_array=[0.0] )
        box = raw[0][0]
        self.assertEqual( box['x'], 10.0 )
        self.assertEqual( box['y'], 10.0 )
        self.assertEqual( box['w'], 50.0 )
        self.assertEqual( box['h'], 50.0 )


class TestProcessRawBoxConsumesNewShape( unittest.TestCase ):

    def test_string_class_id_resolves_to_a_censorable_box( self ):
        raw = { 'x': 10.0, 'y': 10.0, 'w': 50.0, 'h': 50.0,
                'class_id': 'exposed_breast', 'score': 0.9, 't': 3.14 }
        result = bu_censor.process_raw_box( raw, vid_w=1920, vid_h=1080 )
        self.assertIsNotNone( result )
        self.assertEqual( result['label'], 'exposed_breast' )

    def test_label_not_in_items_to_censor_returns_none( self ):
        raw = { 'x': 10.0, 'y': 10.0, 'w': 50.0, 'h': 50.0,
                'class_id': 'exposed_anus', 'score': 0.99, 't': 0.0 }
        # exposed_anus exists as a suppression-only signal, not in
        # items_to_censor by default - should never be censored directly
        result = bu_censor.process_raw_box( raw, vid_w=1920, vid_h=1080 )
        self.assertIsNone( result )


if __name__ == '__main__':
    unittest.main()
