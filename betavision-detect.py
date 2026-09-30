"""
betavision-detect.py

BetaVision live-pipeline stage 2 of 3 (screenshot -> detect -> censor,
see betavision-screenshot.py and betavision-censor.py). Reads the
latest raw screenshot betavision-screenshot.py published to shared
memory, runs it through the active detector adapter (see
betautils_detector.py) - once per configured picture size - whenever
the image has actually changed since the last check, and publishes the
resulting raw boxes back to shared memory for betavision-censor.py to
turn into censor boxes. Runs forever; stop with Ctrl+C.

As of v2.0.0, preprocessing (resize/pad/normalize) happens here, per
configured size, immediately before that size's detection call - see
betavision-screenshot.py's module docstring for why that moved out of
the capture stage.

Change detection is a two-tier check purely for speed: summing an
image is much cheaper than hashing it, so a cheap sum comparison is
tried first and only escalates to a full MD5 hash on a sum collision.
This means the same image can get re-detected up to twice in a row
(once on the sum check passing "unchanged" incorrectly is impossible -
a sum collision escalates to hash - so at most once extra, when hash
also happens to collide) but never more than that.
"""
import hashlib
import numpy as np
import time
from multiprocessing import shared_memory

import betaconst
import betaconfig

import betautils_vision as bu_vision
import betautils_detector as bu_detector


def open_input_shared_memory():
    """
    Attach to the screenshot-timestamp handshake segments
    betavision-screenshot.py publishes into (see that file's
    init_shared_memory/publish_screenshot).

    Returns:
        A (timestamp1_shm, timestamp2_shm, timestamp1, timestamp2)
        tuple: the two SharedMemory handles (kept alive for as long as
        the segments are needed) and their np.ndarray views.
    """
    timestamp1_shm = shared_memory.SharedMemory( name=betaconst.bv_ss_timestamp1_name )
    timestamp2_shm = shared_memory.SharedMemory( name=betaconst.bv_ss_timestamp2_name )
    timestamp1 = np.ndarray( (1,), dtype=np.float64, buffer=timestamp1_shm.buf )
    timestamp2 = np.ndarray( (1,), dtype=np.float64, buffer=timestamp2_shm.buf )
    return timestamp1_shm, timestamp2_shm, timestamp1, timestamp2


def open_input_screenshot_buffer():
    """
    Attach to the raw screenshot segment betavision-screenshot.py
    publishes into.

    Returns:
        A (raw_shm, raw_screenshot, local_screenshot) triple: raw_shm is
        the SharedMemory handle (kept alive for as long as the segment
        is needed); raw_screenshot is a np.ndarray view directly onto
        shared memory (read fresh every loop iteration); local_screenshot
        is a same-shaped local (non-shared) array this process copies
        each screenshot into before working with it, so a slow read here
        never blocks or races betavision-screenshot.py's next write.
    """
    raw_shm = shared_memory.SharedMemory( name=bu_vision.shm_name_for_screenshot() )
    ( this_height, this_width ) = bu_vision.vision_adj_img_size( 0 )
    raw_screenshot = np.ndarray( ( this_height, this_width, 3 ), dtype=np.uint8, buffer=raw_shm.buf )
    local_screenshot = np.ndarray( ( this_height, this_width, 3 ), dtype=np.uint8 )
    return raw_shm, raw_screenshot, local_screenshot


def init_output_shared_memory():
    """
    Create the shared-memory segments this process publishes detected
    raw boxes into, for betavision-censor.py to read.

    Unlike v1.0.0's IPC (which shared raw model tensors sized for one
    specific model's fixed-300-detection output shape), this publishes
    already-parsed raw box dicts - packed into a fixed-size structured
    array via betautils_vision.boxes_to_shared_array - so it works
    unchanged no matter which detector adapter is configured or how many
    detections a given pass actually produces (up to
    betaconst.bv_detect_max_boxes; see that constant's comment).

    Returns:
        A (local_boxes, remote_boxes, remote_count, out_timestamp1,
        out_timestamp2) tuple: local_boxes is this process's own working
        array (dtype betautils_vision.box_record_dtype, length
        betaconst.bv_detect_max_boxes) that gets copied into
        remote_boxes - the shared-memory view actually read by
        betavision-censor.py - once a detection pass completes.
        remote_count is a shared 1-element int array holding how many
        of remote_boxes' slots are populated. out_timestamp1/2 are the
        shared handshake timestamps this process writes (see
        publish_detection_output).
    """
    local_boxes = np.zeros( betaconst.bv_detect_max_boxes, dtype=bu_vision.box_record_dtype )

    out_shm_boxes = shared_memory.SharedMemory( name=betaconst.bv_detect_shm_boxes_name, create=True, size=local_boxes.nbytes )
    remote_boxes = np.ndarray( local_boxes.shape, local_boxes.dtype, buffer=out_shm_boxes.buf )

    out_shm_count = shared_memory.SharedMemory( name=betaconst.bv_detect_shm_count_name, create=True, size=8 )
    remote_count = np.ndarray( (1,), dtype=np.int64, buffer=out_shm_count.buf )

    out_shm_timestamp1 = shared_memory.SharedMemory( name=betaconst.bv_detect_timestamp1_name, create=True, size=8 )
    out_shm_timestamp2 = shared_memory.SharedMemory( name=betaconst.bv_detect_timestamp2_name, create=True, size=8 )
    out_timestamp1 = np.ndarray( (1,), dtype=np.float64, buffer=out_shm_timestamp1.buf )
    out_timestamp2 = np.ndarray( (1,), dtype=np.float64, buffer=out_shm_timestamp2.buf )

    return ( local_boxes, remote_boxes, remote_count, out_timestamp1, out_timestamp2 )


def wait_for_new_screenshot( in_timestamp1, in_timestamp2, last_timestamp ):
    """
    Busy-waits until betavision-screenshot.py has published a new,
    fully-written screenshot (both handshake timestamps agree, and the
    agreed value is newer than the last one this process consumed).

    Args:
        in_timestamp1: Shared-memory view of the screenshot writer's
            first handshake timestamp.
        in_timestamp2: Shared-memory view of the screenshot writer's
            second handshake timestamp.
        last_timestamp: The capture timestamp this process last
            consumed.

    Returns:
        The new timestamp value that was waited for.
    """
    while( in_timestamp1[0] != in_timestamp2[0] or in_timestamp1[0] == last_timestamp ):
        True
    return in_timestamp1[0]


def copy_screenshot_locally( raw_screenshot, local_screenshot ):
    """
    Copies the shared-memory screenshot view into this process's own
    local array, so the rest of this iteration works on a stable
    snapshot instead of memory betavision-screenshot.py could
    overwrite mid-read.

    Args:
        raw_screenshot: Shared-memory np.ndarray view.
        local_screenshot: Same-shaped local array to copy into.
    """
    local_screenshot[:] = raw_screenshot[:]


def compute_image_sum( local_screenshot ):
    """
    Sums every pixel in the current screenshot - the cheap first tier
    of the two-tier change check (see image_has_changed_given_sum
    below): summing takes ~10ms, far cheaper than a full hash, so this
    always runs first.

    Args:
        local_screenshot: This iteration's locally-copied screenshot
            (see copy_screenshot_locally).

    Returns:
        The pixel sum of local_screenshot.
    """
    return np.sum( local_screenshot )


def image_has_changed_given_sum( local_screenshot, new_sum, previous_sum, previous_hash ):
    """
    Given this iteration's already-computed pixel sum (see
    compute_image_sum), determines whether the image actually changed
    since the last check - escalating to a full MD5 hash only when the
    sum alone can't prove a change.

    Hashing takes a few ms, which is not nothing, so it's only reached
    on a sum collision. This means the same image can get detected
    twice in a row (once on a sum collision triggering the hash check),
    but never more than that.

    Args:
        local_screenshot: This iteration's locally-copied screenshot.
        new_sum: This iteration's pixel sum (from compute_image_sum).
        previous_sum: The pixel sum from the last check.
        previous_hash: The MD5 digest from the last check (0 if the
            hash step wasn't reached last time).

    Returns:
        A (changed, new_hash) pair. new_hash is 0 (not computed)
        whenever new_sum already differs from previous_sum, since the
        sum difference alone is enough to prove a change.
    """
    if new_sum == previous_sum:
        new_hash = hashlib.md5( local_screenshot.tobytes() ).digest()
    else:
        new_hash = 0
    changed = ( new_sum != previous_sum or new_hash != previous_hash )
    return changed, new_hash


def publish_detection_output( last_timestamp, local_boxes, box_count, remote_boxes, remote_count,
                               out_timestamp1, out_timestamp2 ):
    """
    Copy this iteration's detected raw boxes into shared memory for
    betavision-censor.py to read, using the same
    write-timestamp1-then-data-then-timestamp2 torn-read handshake
    betavision-screenshot.py uses for its own publish step.

    Args:
        last_timestamp: This iteration's screenshot capture time.
        local_boxes: This iteration's packed box_record_dtype array (see
            betautils_vision.boxes_to_shared_array).
        box_count: Number of local_boxes' leading slots that hold real
            data this iteration.
        remote_boxes: Shared-memory view to copy local_boxes into.
        remote_count: Shared-memory view to write box_count into.
        out_timestamp1: Shared-memory view written first.
        out_timestamp2: Shared-memory view written last.
    """
    out_timestamp1[0] = last_timestamp
    remote_boxes[:] = local_boxes[:]
    remote_count[0] = box_count
    out_timestamp2[0] = last_timestamp


def main():
    """
    Entry point: attach to betavision-screenshot.py's shared memory,
    create this stage's own output shared memory, then run detection
    in a tight loop forever - each iteration waits for a new
    screenshot, skips re-detecting an unchanged image where possible,
    and publishes whatever raw model output is current (freshly
    detected, or carried over from the last detection) back to shared
    memory.
    """
    detector = bu_detector.get_detector()
    session = detector.get_session()

    _in_timestamp1_shm, _in_timestamp2_shm, in_timestamp1, in_timestamp2 = open_input_shared_memory()
    raw_shm, raw_screenshot, local_screenshot = open_input_screenshot_buffer()

    ( local_boxes, remote_boxes, remote_count,
      out_timestamp1, out_timestamp2 ) = init_output_shared_memory()

    box_count = 0
    last_timestamp = 0
    image_sum = 0
    image_hash = 0
    while True:
        times = [ time.perf_counter() ]

        last_timestamp = wait_for_new_screenshot( in_timestamp1, in_timestamp2, last_timestamp )
        times.append( time.perf_counter() )

        copy_screenshot_locally( raw_screenshot, local_screenshot )
        times.append( time.perf_counter() )

        new_sum = compute_image_sum( local_screenshot )
        times.append( time.perf_counter() )

        changed, new_hash = image_has_changed_given_sum( local_screenshot, new_sum, image_sum, image_hash )
        times.append( time.perf_counter() )
        image_sum, image_hash = new_sum, new_hash

        if changed:
            # One raw_boxes_for_img call per configured picture size,
            # same as betatv.py/betastare.py's pattern - preprocessing
            # (resize/pad/normalize) happens inside the adapter call
            # itself, on the raw screenshot, per size (see
            # betavision-screenshot.py's module docstring for why this
            # moved out of the capture stage).
            # Sizes resolve per backend (betautils_detector.get_picture_sizes):
            # the shared betaconfig.picture_sizes this used to read was
            # removed in 2.5, and reading it here would raise AttributeError.
            all_raw_boxes = []
            for size in bu_detector.get_picture_sizes():
                all_raw_boxes.extend( detector.raw_boxes_for_img( local_screenshot, size, session, last_timestamp ) )
            box_count = bu_vision.boxes_to_shared_array( all_raw_boxes, local_boxes )
        times.append( time.perf_counter() )

        publish_detection_output(
            last_timestamp, local_boxes, box_count, remote_boxes, remote_count, out_timestamp1, out_timestamp2 )
        times.append( time.perf_counter() )

        print( [ '%.3f'%(x-times[0]) for x in times ] )


if __name__ == '__main__':
    main()
