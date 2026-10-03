"""
test_detector_invariants.py - the properties the detector adapters must
hold, independent of any model file being present.

Three things are asserted here, each of which was a real defect or a
real unverified assumption in earlier versions:

  1. BATCH SIZE INVARIANCE
     nn_batch_size is deliberately excluded from the detection cache
     key, on the premise that batching changes how many frames go into
     one inference call and never which detections come out. That
     premise was asserted in a comment and never enforced, while the
     retinanet adapter was in fact reading the wrong setting and
     silently running at batch 1 regardless. Now it is a test.

  2. VECTORISED DECODE EQUIVALENCE
     nudenet_v3's anchor decode was rewritten from a per-anchor Python
     loop into array operations, roughly 180x faster at a 1280 blob. The
     rewrite must be arithmetically identical, so a scalar reference
     implementation lives in this file and the two are compared on
     randomised input.

  3. PER-CLASS NMS SEMANTICS
     Suppressing only within a label, rather than across all labels,
     is what hands the cross-label decision to class_suppression. The
     difference is asserted directly.

No model file is needed: the session is a stub that returns prepared
tensors, so these run anywhere.
"""

import unittest

import numpy as np

import betaconfig
import betautils_config as bu_config
import betautils_detector as bu_detector
from detectors import nudenet_v3, retinanet_v2


# ---------------------------------------------------------------------------
# Stub sessions
# ---------------------------------------------------------------------------

class _RetinaNetStubSession:
    """
    Stand-in for the retinanet ONNX session.

    Returns a deterministic detection per input image, derived from a
    per-image seed so batching cannot be accidentally "invariant"
    because every image produced the same answer.
    """

    def __init__( self ):
        self.call_count = 0
        self.batch_sizes_seen = []

    def run( self, _output_names, feed ):
        images = feed[ list( feed.keys() )[0] ]
        batch = len( images )
        self.call_count += 1
        self.batch_sizes_seen.append( batch )

        boxes = np.zeros( ( batch, retinanet_v2.DETECTIONS_PER_IMAGE, 4 ), dtype=np.float32 )
        scores = np.zeros( ( batch, retinanet_v2.DETECTIONS_PER_IMAGE ), dtype=np.float32 )
        classes = np.full( ( batch, retinanet_v2.DETECTIONS_PER_IMAGE ), -1, dtype=np.int32 )

        for index, image in enumerate( images ):
            # Seed from the image's own content so each frame in a batch
            # gets its own answer.
            seed = int( abs( float( np.asarray( image ).sum() ) ) ) % 997
            for detection in range( 3 ):
                boxes[index][detection] = [ 10+seed % 50, 20+detection,
                                            60+seed % 50, 80+detection ]
                scores[index][detection] = 0.9 - 0.1*detection
                classes[index][detection] = ( seed + detection ) % len(
                    retinanet_v2._MODEL_CLASS_ORDER )
        return [ boxes, scores, classes ]


class _NudeNetStubSession:
    """Stand-in for the nudenet ONNX session, returning fixed anchors."""

    def __init__( self, anchors_per_image ):
        self.anchors_per_image = anchors_per_image
        self.call_count = 0
        self.batch_sizes_seen = []

    def get_inputs( self ):
        class _Input:
            name = 'images'
        return [ _Input() ]

    def run( self, _output_names, feed ):
        blob = feed[ list( feed.keys() )[0] ]
        batch = blob.shape[0]
        self.call_count += 1
        self.batch_sizes_seen.append( batch )
        output = np.zeros( ( batch, 22, self.anchors_per_image ), dtype=np.float32 )
        for index in range( batch ):
            # One confident detection per image, with a class derived
            # from that image's own CONTENT rather than its position in
            # the batch - otherwise the test would pass trivially,
            # because every batching arrangement would produce the same
            # answer for the wrong reason.
            seed = int( abs( float( blob[index].sum() ) ) ) % 18
            output[index, 0, 0] = 160
            output[index, 1, 0] = 160
            output[index, 2, 0] = 40
            output[index, 3, 0] = 40
            output[index, 4 + seed, 0] = 0.9
        return [ output ]


# ---------------------------------------------------------------------------
# 1. Batch size invariance
# ---------------------------------------------------------------------------

class TestBatchSizeInvariance( unittest.TestCase ):
    """
    Detections must not depend on nn_batch_size.

    This is the premise that lets the detection cache be shared across
    batch sizes. If it ever stops holding, the cache starts serving the
    wrong answers and this test is what says so.
    """

    def setUp( self ):
        self.original_backend = betaconfig.detector_backend
        self.frames = [ ( np.full( ( 48, 64, 3 ), value, dtype=np.uint8 ) )
                        for value in ( 10, 40, 90, 150, 200 ) ]
        self.timestamps = [ index / 9 for index in range( len( self.frames ) ) ]

    def tearDown( self ):
        betaconfig.detector_backend = self.original_backend
        bu_config.invalidate_config_caches()

    def _with_batch_size( self, backend_name, batch_size ):
        betaconfig.detector_backend = dict( self.original_backend )
        betaconfig.detector_backend[backend_name] = dict(
            betaconfig.detector_backend.get( backend_name, {} ) )
        betaconfig.detector_backend[backend_name]['nn_batch_size'] = batch_size
        betaconfig.detector_backend['selected'] = backend_name
        bu_config.invalidate_config_caches()

    def test_retinanet_detections_are_identical_across_batch_sizes( self ):
        results = {}
        sessions = {}
        for batch_size in ( 1, 2, 3, 5 ):
            self._with_batch_size( 'retinanet_v2', batch_size )
            session = _RetinaNetStubSession()
            sessions[batch_size] = session
            results[batch_size] = retinanet_v2.raw_boxes_for_imgs(
                self.frames, 320, session, self.timestamps )

        baseline = results[1]
        for batch_size, boxes in results.items():
            self.assertEqual( boxes, baseline,
                "retinanet detections changed at nn_batch_size=%d"%(batch_size) )

        # ...and the batching actually happened, rather than the test
        # passing because every value silently ran at 1. This is the
        # exact defect: the adapter read the top-level nn_batch_size
        # instead of the per-backend resolved value.
        self.assertEqual( sessions[1].call_count, len( self.frames ) )
        self.assertEqual( sessions[5].call_count, 1 )
        self.assertEqual( sessions[2].batch_sizes_seen, [ 2, 2, 1 ] )

    def test_retinanet_honours_the_per_backend_batch_size( self ):
        self._with_batch_size( 'retinanet_v2', 3 )
        self.assertEqual( bu_detector.get_nn_batch_size( 'retinanet_v2' ), 3 )
        session = _RetinaNetStubSession()
        retinanet_v2.raw_boxes_for_imgs( self.frames, 320, session, self.timestamps )
        self.assertEqual( session.batch_sizes_seen, [ 3, 2 ] )

    def test_retinanet_ignores_the_removed_top_level_setting( self ):
        # betaconfig.nn_batch_size stopped being read. Setting it
        # must not change what the backend batches at.
        #
        # The restore below deletes the attribute when it did not exist
        # beforehand, rather than writing a default back: leaving one
        # behind makes validate_config report a stale top-level setting
        # in every later test in the run.
        self._with_batch_size( 'retinanet_v2', 4 )
        had_top_level = hasattr( betaconfig, 'nn_batch_size' )
        original_top_level = getattr( betaconfig, 'nn_batch_size', None )
        try:
            betaconfig.nn_batch_size = 1
            session = _RetinaNetStubSession()
            retinanet_v2.raw_boxes_for_imgs( self.frames, 320, session, self.timestamps )
            self.assertEqual( session.batch_sizes_seen, [ 4, 1 ] )
        finally:
            if had_top_level:
                betaconfig.nn_batch_size = original_top_level
            else:
                del betaconfig.nn_batch_size

    def test_nudenet_detections_are_identical_across_batch_sizes( self ):
        results = {}
        sessions = {}
        for batch_size in ( 1, 2, 5 ):
            self._with_batch_size( 'nudenet_v3', batch_size )
            session = _NudeNetStubSession( anchors_per_image=64 )
            sessions[batch_size] = session
            results[batch_size] = nudenet_v3.raw_boxes_for_imgs(
                self.frames, 320, session, self.timestamps )

        baseline = results[1]
        for batch_size, boxes in results.items():
            self.assertEqual( boxes, baseline,
                "nudenet detections changed at nn_batch_size=%d"%(batch_size) )
        self.assertEqual( sessions[1].call_count, len( self.frames ) )
        self.assertEqual( sessions[5].call_count, 1 )


# ---------------------------------------------------------------------------
# 2. Vectorised decode equivalence
# ---------------------------------------------------------------------------

def _scalar_reference_decode( raw_output, pad_w, pad_h, max_size, size, candidate_floor ):
    """
    The earlier per-anchor Python decode, transcribed.

    Kept as the reference the vectorised implementation must match. NMS
    is left out: it is the same OpenCV call in both, and comparing
    pre-NMS candidates isolates the arithmetic under test.

    Returns:
        A list of (x, y, w, h, score, class_index) tuples.
    """
    import math
    row_major = np.transpose( raw_output )
    candidates = []
    for row in row_major:
        class_scores = row[4:]
        max_score = float( np.amax( class_scores ) )
        if max_score < candidate_floor:
            continue
        class_index = int( np.argmax( class_scores ) )
        cx, cy, w, h = row[0:4]
        x = float( cx - w/2 )
        y = float( cy - h/2 )
        scale = max_size / size
        x *= scale
        y *= scale
        w = float( w ) * scale
        h = float( h ) * scale
        orig_w = max_size - pad_w
        orig_h = max_size - pad_h
        x = max( 0.0, min( x, orig_w ) )
        y = max( 0.0, min( y, orig_h ) )
        w = min( w, orig_w - x )
        h = min( h, orig_h - y )
        if w <= 0 or h <= 0:
            continue
        candidates.append( ( x, y, w, h, max_score, class_index ) )
    return candidates


class TestVectorisedDecodeMatchesTheReference( unittest.TestCase ):

    def _vectorised_candidates( self, raw_output, pad_w, pad_h, max_size, size, floor ):
        """decode_output with NMS made a no-op, so only the maths is compared."""
        return nudenet_v3.decode_output(
            raw_output, pad_w, pad_h, max_size, size,
            candidate_floor=floor, nms_iou=1.0, nms_mode='agnostic' )

    def test_matches_on_randomised_input( self ):
        generator = np.random.default_rng( 20260917 )
        for trial in range( 25 ):
            anchors = int( generator.integers( 1, 400 ) )
            raw_output = ( generator.random( ( 22, anchors ) ).astype( np.float32 ) * 0.4 )
            # Give a handful of anchors a real detection.
            for _ in range( max( 1, anchors // 20 ) ):
                anchor = int( generator.integers( 0, anchors ) )
                raw_output[0:4, anchor] = generator.random( 4 ).astype( np.float32 ) * 300
                raw_output[ 4 + int( generator.integers( 0, 18 ) ), anchor ] = 0.5 + 0.4*generator.random()

            size = 320
            max_size = int( generator.integers( 320, 1200 ) )
            pad_w = int( generator.integers( 0, max_size // 4 ) )
            pad_h = int( generator.integers( 0, max_size // 4 ) )
            floor = 0.2

            expected = _scalar_reference_decode( raw_output, pad_w, pad_h, max_size, size, floor )
            actual = self._vectorised_candidates( raw_output, pad_w, pad_h, max_size, size, floor )

            self.assertEqual( len( actual ), len( expected ),
                "trial %d: candidate count differs"%(trial) )
            for got, want in zip( sorted( actual ), sorted( expected ) ):
                for index, ( a, b ) in enumerate( zip( got, want ) ):
                    self.assertAlmostEqual( float( a ), float( b ), places=3,
                        msg="trial %d: field %d differs"%(trial, index) )

    def test_rejects_a_row_major_array( self ):
        # Passing the transposed (anchors, 22) layout by mistake would
        # silently decode nonsense for a real model output, so it is
        # rejected rather than tolerated.
        with self.assertRaises( ValueError ):
            nudenet_v3.decode_output( np.zeros( ( 4, 3 ) ), 0, 0, 320, 320, 0.2, 0.45 )

    def test_empty_anchor_set_returns_empty( self ):
        self.assertEqual(
            nudenet_v3.decode_output( np.zeros( ( 22, 0 ) ), 0, 0, 320, 320, 0.2, 0.45 ), [] )


# ---------------------------------------------------------------------------
# 3. Per-class vs class-agnostic NMS
# ---------------------------------------------------------------------------

class TestNmsMode( unittest.TestCase ):

    def _two_overlapping_detections( self, first_class, second_class ):
        """Two heavily overlapping boxes, optionally of different labels."""
        raw_output = np.zeros( ( 22, 2 ), dtype=np.float32 )
        raw_output[0:4, 0] = [ 160, 160, 60, 60 ]
        raw_output[4 + first_class, 0] = 0.90
        raw_output[0:4, 1] = [ 162, 162, 60, 60 ]
        raw_output[4 + second_class, 1] = 0.80
        return raw_output

    def test_per_class_keeps_overlapping_detections_of_different_labels( self ):
        raw_output = self._two_overlapping_detections( 3, 13 )
        decoded = nudenet_v3.decode_output(
            raw_output, 0, 0, 320, 320, 0.2, 0.45, nms_mode='per_class' )
        self.assertEqual( len( decoded ), 2,
            "per-class NMS must leave a cross-label overlap for class_suppression to arbitrate" )
        self.assertEqual( sorted( entry[5] for entry in decoded ), [ 3, 13 ] )

    def test_agnostic_drops_the_lower_scoring_overlap_whatever_its_label( self ):
        raw_output = self._two_overlapping_detections( 3, 13 )
        decoded = nudenet_v3.decode_output(
            raw_output, 0, 0, 320, 320, 0.2, 0.45, nms_mode='agnostic' )
        self.assertEqual( len( decoded ), 1 )
        self.assertEqual( decoded[0][5], 3 )

    def test_both_modes_dedupe_same_label_overlaps( self ):
        raw_output = self._two_overlapping_detections( 3, 3 )
        for mode in ( 'per_class', 'agnostic' ):
            decoded = nudenet_v3.decode_output(
                raw_output, 0, 0, 320, 320, 0.2, 0.45, nms_mode=mode )
            self.assertEqual( len( decoded ), 1, "mode %s kept a same-label duplicate"%(mode) )
            self.assertAlmostEqual( decoded[0][4], 0.90, places=3 )


# ---------------------------------------------------------------------------
# Variants and native sizes
# ---------------------------------------------------------------------------

class TestVariantsAndNativeSizes( unittest.TestCase ):

    def test_every_shipped_variant_declares_a_native_size( self ):
        for name, variant in nudenet_v3.VARIANTS.items():
            self.assertIn( 'native_size', variant, name )
            self.assertIn( 'model_path', variant, name )
            self.assertGreater( variant['native_size'], 0, name )
            self.assertEqual( variant['native_size'] % 32, 0,
                "%s's native size must be a multiple of the 8/16/32 strides"%(name) )

    def test_native_picture_sizes_follows_the_selected_variant( self ):
        self.assertEqual(
            nudenet_v3.native_picture_sizes( { 'model_variant': '320n' } ), [ 320 ] )
        self.assertEqual(
            nudenet_v3.native_picture_sizes( { 'model_variant': '640m' } ), [ 640 ] )

    def test_an_unknown_variant_fails_fast_with_the_valid_choices( self ):
        with self.assertRaises( ValueError ) as caught:
            nudenet_v3.resolve_config( { 'model_variant': 'nope' } )
        self.assertIn( '320n', str( caught.exception ) )
        self.assertIn( '640m', str( caught.exception ) )

    def test_an_explicit_model_path_overrides_the_variant_default( self ):
        resolved = nudenet_v3.resolve_config(
            { 'model_variant': '320n', 'model_path': '/somewhere/custom.onnx' } )
        self.assertEqual( resolved['model_path'], '/somewhere/custom.onnx' )
        self.assertEqual( resolved['native_size'], 320 )

    def test_retinanet_defers_to_the_shared_picture_sizes( self ):
        # It has no single validated input size, so it must not claim one.
        self.assertEqual( retinanet_v2.native_picture_sizes( {} ), [] )

    def test_variant_environment_override_wins( self ):
        import os
        original = os.environ.get( bu_detector._VARIANT_OVERRIDE_ENV_VAR )
        try:
            os.environ[ bu_detector._VARIANT_OVERRIDE_ENV_VAR ] = '640m'
            self.assertEqual( bu_detector.selected_variant_name( 'nudenet_v3' ), '640m' )
            self.assertEqual( nudenet_v3.resolve_config(
                { 'model_variant': '320n' } )['model_variant'], '640m' )
        finally:
            if original is None:
                os.environ.pop( bu_detector._VARIANT_OVERRIDE_ENV_VAR, None )
            else:
                os.environ[ bu_detector._VARIANT_OVERRIDE_ENV_VAR ] = original


class TestRawBoxesCarryTheirSize( unittest.TestCase ):
    """
    Every raw box records which picture size produced it, so cross-size
    dedup can tell a duplicate from two real instances.
    """

    def test_nudenet_stamps_the_size( self ):
        session = _NudeNetStubSession( anchors_per_image=16 )
        boxes = nudenet_v3.raw_boxes_for_imgs(
            [ np.zeros( ( 48, 64, 3 ), dtype=np.uint8 ) ], 320, session, [ 0.0 ] )
        self.assertTrue( boxes )
        for box in boxes:
            self.assertEqual( box['size'], 320 )

    def test_retinanet_stamps_the_size( self ):
        session = _RetinaNetStubSession()
        boxes = retinanet_v2.raw_boxes_for_imgs(
            [ np.full( ( 48, 64, 3 ), 30, dtype=np.uint8 ) ], 640, session, [ 0.0 ] )
        self.assertTrue( boxes )
        for box in boxes:
            self.assertEqual( box['size'], 640 )


if __name__ == '__main__':
    unittest.main()
