"""
test_vision_ipc.py - regression tests for BetaVision's v2.0.0 shared-
memory IPC redesign (betautils_vision.py's box_record_dtype /
boxes_to_shared_array / shared_array_to_boxes).

Covers:
    - a variable-length list of raw box dicts round-trips correctly
      through the fixed-size structured array used for shared memory
      (the core mechanism that replaced v1.0.0's fixed-300-detection
      raw-tensor IPC, which was hardcoded to one specific model's
      output shape)
    - overflow beyond betaconst.bv_detect_max_boxes is clamped, not a
      crash or silent corruption
    - an empty box list round-trips to an empty list
    - the packed array behaves correctly across an actual
      multiprocessing.shared_memory attach/read, not just as a plain
      in-process numpy array (this is what the real
      betavision-detect.py / betavision-censor.py split relies on)
    - the longest current canonical class label ('covered_buttocks')
      round-trips correctly through the fixed-width S32 class_id field
    - a decoded box is consumable end-to-end by
      betautils_censor.process_raw_box, exactly like betatv.py/
      betastare.py's raw boxes are

Does NOT test the real betavision-*.py processes themselves (screen
capture, live display, multi-process orchestration) - those need a
real display/GPU and aren't practical to unit test; this exercises the
IPC payload logic they both depend on.
"""

import unittest
from multiprocessing import shared_memory

import numpy as np

import betaconst
import betautils_vision as bu_vision
import betautils_censor as bu_censor


def _approx_equal_boxes( a, b, tol=1e-5 ):
    if len( a ) != len( b ):
        return False
    for x, y in zip( a, b ):
        if x['class_id'] != y['class_id']:
            return False
        for k in ( 'x', 'y', 'w', 'h', 'score' ):
            if abs( x[k] - y[k] ) > tol:
                return False
        if abs( x['t'] - y['t'] ) > 1e-9:
            return False
    return True


class TestBoxRecordRoundTrip( unittest.TestCase ):

    def test_basic_round_trip( self ):
        raw_boxes = [
            { 'x': 10.0, 'y': 20.0, 'w': 30.0, 'h': 40.0, 'score': 0.91, 't': 3.14, 'class_id': 'exposed_breast' },
            { 'x': 1.0,  'y': 2.0,  'w': 3.0,  'h': 4.0,  'score': 0.55, 't': 3.14, 'class_id': 'covered_vulva' },
        ]
        buf = np.zeros( betaconst.bv_detect_max_boxes, dtype=bu_vision.box_record_dtype )
        count = bu_vision.boxes_to_shared_array( raw_boxes, buf )
        self.assertEqual( count, 2 )
        result = bu_vision.shared_array_to_boxes( buf, count )
        self.assertTrue( _approx_equal_boxes( result, raw_boxes ) )

    def test_empty_list_round_trips_to_empty_list( self ):
        buf = np.zeros( betaconst.bv_detect_max_boxes, dtype=bu_vision.box_record_dtype )
        count = bu_vision.boxes_to_shared_array( [], buf )
        self.assertEqual( count, 0 )
        self.assertEqual( bu_vision.shared_array_to_boxes( buf, count ), [] )

    def test_overflow_beyond_max_boxes_is_clamped( self ):
        one_box = { 'x': 0.0, 'y': 0.0, 'w': 1.0, 'h': 1.0, 'score': 0.5, 't': 0.0, 'class_id': 'exposed_belly' }
        too_many = [ dict( one_box ) for _ in range( betaconst.bv_detect_max_boxes + 50 ) ]
        buf = np.zeros( betaconst.bv_detect_max_boxes, dtype=bu_vision.box_record_dtype )
        count = bu_vision.boxes_to_shared_array( too_many, buf )
        self.assertEqual( count, betaconst.bv_detect_max_boxes )

    def test_longest_canonical_label_round_trips( self ):
        # 'covered_buttocks' (16 chars) is the longest current label -
        # confirms the S32 class_id field has correct headroom and that
        # encode/decode doesn't truncate or corrupt it.
        box = [ { 'x': 0.0, 'y': 0.0, 'w': 1.0, 'h': 1.0, 'score': 0.5, 't': 0.0, 'class_id': 'covered_buttocks' } ]
        buf = np.zeros( betaconst.bv_detect_max_boxes, dtype=bu_vision.box_record_dtype )
        count = bu_vision.boxes_to_shared_array( box, buf )
        result = bu_vision.shared_array_to_boxes( buf, count )
        self.assertEqual( result[0]['class_id'], 'covered_buttocks' )


class TestBoxRecordAcrossRealSharedMemory( unittest.TestCase ):
    """
    Same round-trip as TestBoxRecordRoundTrip, but through an actual
    multiprocessing.shared_memory segment attached by name (as a second
    process would), rather than a plain in-process numpy array - this
    is the real mechanism betavision-detect.py/betavision-censor.py
    rely on, so it's worth confirming separately from the plain-array
    logic above.
    """

    def test_write_then_attach_and_read( self ):
        raw_boxes = [
            { 'x': 10.0, 'y': 20.0, 'w': 30.0, 'h': 40.0, 'score': 0.91, 't': 3.14, 'class_id': 'exposed_breast' },
            { 'x': 1.0,  'y': 2.0,  'w': 3.0,  'h': 4.0,  'score': 0.55, 't': 3.14, 'class_id': 'covered_vulva' },
        ]
        local_buf = np.zeros( betaconst.bv_detect_max_boxes, dtype=bu_vision.box_record_dtype )
        count = bu_vision.boxes_to_shared_array( raw_boxes, local_buf )

        shm = shared_memory.SharedMemory( create=True, size=local_buf.nbytes )
        try:
            shared_arr = np.ndarray( local_buf.shape, dtype=local_buf.dtype, buffer=shm.buf )
            shared_arr[:] = local_buf[:]

            # simulate a separate reader process attaching by name
            reader_shm = shared_memory.SharedMemory( name=shm.name )
            try:
                reader_arr = np.ndarray( local_buf.shape, dtype=local_buf.dtype, buffer=reader_shm.buf )
                result = bu_vision.shared_array_to_boxes( reader_arr, count )
                self.assertTrue( _approx_equal_boxes( result, raw_boxes ) )
            finally:
                reader_shm.close()
        finally:
            shm.close()
            shm.unlink()


class TestDecodedBoxConsumableByCensor( unittest.TestCase ):

    def test_process_raw_box_accepts_a_decoded_box( self ):
        # Confirms a box that has round-tripped through the shared-memory
        # encoding is shaped exactly like any other adapter-produced raw
        # box dict - betautils_censor.process_raw_box doesn't need to
        # know or care that this one came from BetaVision's IPC rather
        # than betatv.py/betastare.py's direct adapter calls.
        buf = np.zeros( betaconst.bv_detect_max_boxes, dtype=bu_vision.box_record_dtype )
        raw = { 'x': 10.0, 'y': 10.0, 'w': 50.0, 'h': 50.0, 'score': 0.9, 't': 3.14, 'class_id': 'exposed_breast' }
        count = bu_vision.boxes_to_shared_array( [ raw ], buf )
        decoded = bu_vision.shared_array_to_boxes( buf, count )[0]

        result = bu_censor.process_raw_box( decoded, vid_w=1920, vid_h=1080 )
        self.assertIsNotNone( result )
        self.assertEqual( result['label'], 'exposed_breast' )


if __name__ == '__main__':
    unittest.main()
