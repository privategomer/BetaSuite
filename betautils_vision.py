"""
betautils_vision.py - Small helpers shared by BetaVision's live
screen-capture censoring pipeline (betavision-screenshot.py /
betavision-detect.py / betavision-censor.py).

These are pure, stateless helpers - screenshot grabbing, shared-memory
naming, cross-fading between two captured frames, and image-size
resolution - kept together because BetaVision's actual process split
(capture vs. detect vs. render, communicating over shared memory) lives
in the betavision-*.py scripts, not here.
"""

import cv2
import numpy as np
import time

import betaconfig
import betaconst


# Fixed-width structured dtype for one raw detection box, used to pass a
# variable-length list of boxes through a fixed-size shared-memory segment
# between betavision-detect.py and betavision-censor.py (see
# boxes_to_shared_array / shared_array_to_boxes below). This is what lets
# BetaVision's IPC carry any detector adapter's output (see
# betautils_detector.py) - a variable number of boxes, each with a
# canonical string class_id - rather than the original IPC, which
# shared RAW MODEL TENSORS sized for one specific model's fixed
# 300-detection output shape and had to be redesigned, not just resized,
# to support swapping adapters at all.
#
# class_id is a fixed-length byte string ('S32') rather than a variable
# length Python str, since numpy structured arrays (and therefore shared
# memory) require fixed-size fields; 32 bytes comfortably covers every
# current canonical label (longest is 'covered_buttocks' at 16) with
# headroom for longer labels a future adapter might introduce.
box_record_dtype = np.dtype( [
    ( 'x',        np.float32 ),
    ( 'y',        np.float32 ),
    ( 'w',        np.float32 ),
    ( 'h',        np.float32 ),
    ( 'score',    np.float32 ),
    ( 't',        np.float64 ),
    ( 'class_id', 'S32' ),
] )


def get_screenshot( sct ):
    """
    Grab one screenshot of the configured capture region.

    Args:
        sct: An active mss (or compatible) screen-capture context, as
            returned by mss.mss().

    Returns:
        A [timestamp, image] pair: timestamp is a time.monotonic()
        reading taken immediately before the grab (so callers can later
        interpolate between two timestamped frames - see
        interpolate_images below), and image is an HxWx3 BGR numpy array
        (the raw grab's alpha channel is dropped).
    """
    capture_region = {
            'left': betaconfig.vision_cap_left,
            'top': betaconfig.vision_cap_top,
            'width': betaconfig.vision_cap_width,
            'height': betaconfig.vision_cap_height,
            'mon': betaconfig.vision_cap_monitor,
    }

    capture_time = time.monotonic()
    captured_image = np.array( sct.grab( capture_region ) )[:, :, :3]

    return( [ capture_time, captured_image ] )


def shm_name_for_screenshot():
    """
    The fixed shared-memory segment name BetaVision's capture and detect
    processes use to exchange the single raw screenshot.

    There's exactly one such segment (the raw, unprocessed
    capture) rather than one per configured detection size - see
    betavision-screenshot.py's module docstring for why preprocessing
    moved downstream into betavision-detect.py.

    Returns:
        The shared-memory segment name, 'raw_grab'.
    """
    return( 'raw_grab' )


def interpolate_images( img1, ts1, img2, ts2, timestamp ):
    """
    Cross-fade between two timestamped frames to approximate what the
    screen looked like at an in-between timestamp.

    Used by BetaVision to smooth over the gap between its (slower)
    detection cadence and its (faster) display cadence: rather than
    holding the last-detected frame's censoring static until the next
    detection arrives, it blends smoothly toward the newer frame.

    Args:
        img1: The earlier frame's image.
        ts1: img1's timestamp. Must be strictly less than ts2.
        img2: The later frame's image.
        ts2: img2's timestamp.
        timestamp: The timestamp to interpolate to.

    Returns:
        img1 unchanged if timestamp is before ts1, img2 unchanged if
        timestamp is after ts2, otherwise a linear cross-fade between
        the two images weighted by how far timestamp sits between ts1
        and ts2.
    """
    assert( ts1 < ts2 )
    if timestamp < ts1:
        return( img1 )
    if ts2 < timestamp:
        return( img2 )

    fraction_toward_img2 = (timestamp - ts1)/(ts2 - ts1)
    fraction_toward_img1 = 1 - fraction_toward_img2

    return( cv2.addWeighted( img1, fraction_toward_img1, img2, fraction_toward_img2, 0 ) )


def boxes_to_shared_array( raw_boxes, out_array ):
    """
    Pack a list of raw box dicts (as produced by a detector adapter's
    raw_boxes_for_img/raw_boxes_for_imgs - see betautils_detector.py) into
    a preallocated fixed-size structured array (dtype box_record_dtype),
    for publishing into shared memory.

    Args:
        raw_boxes: List of raw box dicts, each with 'x', 'y', 'w', 'h',
            'score', 't', 'class_id' (a canonical string label).
        out_array: A np.ndarray of dtype box_record_dtype and length
            betaconst.bv_detect_max_boxes, written into in place (slots
            beyond len(raw_boxes) are left as whatever they already held -
            callers only ever read back the first `count` slots, see
            shared_array_to_boxes, so stale trailing data is harmless).

    Returns:
        The number of boxes actually written - min(len(raw_boxes),
        len(out_array)). If raw_boxes is longer than out_array, the
        excess is silently dropped (see betaconst.bv_detect_max_boxes's
        comment for when this could happen).
    """
    count = min( len( raw_boxes ), len( out_array ) )
    for i in range( count ):
        raw = raw_boxes[i]
        out_array[i] = (
            raw['x'], raw['y'], raw['w'], raw['h'],
            raw['score'], raw['t'],
            raw['class_id'].encode( 'utf-8' ) )
    return count


def shared_array_to_boxes( shared_array, count ):
    """
    Unpack the first `count` slots of a shared box_record_dtype array
    (as published by boxes_to_shared_array) back into a list of raw box
    dicts, in the same shape betautils_censor.process_raw_box expects.

    Args:
        shared_array: A np.ndarray of dtype box_record_dtype (typically
            a shared-memory view, or a local copy of one).
        count: Number of leading slots in shared_array that hold real
            data - everything at or past this index is ignored.

    Returns:
        A list of `count` raw box dicts: {'x', 'y', 'w', 'h', 'score',
        't', 'class_id'}, class_id decoded back to a plain str.
    """
    raw_boxes = []
    for i in range( count ):
        record = shared_array[i]
        raw_boxes.append( {
            'x':        float( record['x'] ),
            'y':        float( record['y'] ),
            'w':        float( record['w'] ),
            'h':        float( record['h'] ),
            'score':    float( record['score'] ),
            't':        float( record['t'] ),
            'class_id': record['class_id'].decode( 'utf-8' ),
        } )
    return raw_boxes


def vision_adj_img_size( max_length ):
    """
    Resolve the image size BetaVision should resize captures to before
    feeding them into the detection model.

    Args:
        max_length: If non-zero, the model expects a square input of
            this side length. If zero, the model runs on the capture's
            native (unresized) dimensions instead.

    Returns:
        A (height, width) tuple: (max_length, max_length) when
        max_length is non-zero, otherwise the configured native capture
        (betaconfig.vision_cap_height, betaconfig.vision_cap_width).
    """
    if max_length != 0:
        return( ( max_length, max_length ) )
    else:
        return( ( betaconfig.vision_cap_height, betaconfig.vision_cap_width ) )
