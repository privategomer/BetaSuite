"""
betavision-screenshot.py

BetaVision live-pipeline stage 1 of 3 (screenshot -> detect -> censor,
see betavision-detect.py and betavision-censor.py). Continuously grabs a
screenshot of the configured screen region (betaconfig.vision_cap_*) and
publishes the raw, unprocessed screenshot into shared memory for
betavision-detect.py to pick up. Runs forever; stop with Ctrl+C.

This stage publishes ONE raw screenshot rather than a
separate pre-resized/pre-processed copy per configured detection size -
preprocessing (resize/pad/normalize) is adapter-private (see
betautils_detector.py's module docstring on why it isn't shared), so it
now happens inside betavision-detect.py, once per configured size, right
before that size's detection call - exactly the same division of labor
betatv.py/betastare.py already use.

The two-timestamp handshake (shared_timestamp1/shared_timestamp2)
lets a reader (betavision-detect.py) detect a torn read: it only
trusts a screenshot when both timestamps match, meaning this writer
wasn't mid-write while it was being read.
"""
import cv2
import mss
import numpy as np
import time
from multiprocessing import shared_memory

import betaconfig
import betaconst

import betautils_vision as bu_vision


def init_shared_memory( raw_screenshot, timestamp ):
    """
    Create every shared-memory segment this process publishes into: one
    for the raw screenshot (sized from raw_screenshot's current
    shape/dtype) plus the two handshake timestamp segments.

    Args:
        raw_screenshot: This capture's raw screenshot - used only for
            its shape/dtype/nbytes, to size the shared segment
            correctly.
        timestamp: A 1-element np.array holding the capture time - used
            only for its shape/dtype/nbytes, to size the timestamp
            segments correctly.

    Returns:
        A (shms, shared_screenshot, shared_timestamp1, shared_timestamp2)
        tuple: shms is the list of SharedMemory handles (keep these
        alive for as long as the segments are needed); shared_screenshot
        is the np.ndarray view onto the screenshot segment;
        shared_timestamp1/2 are np.ndarray views onto the two handshake
        timestamp segments.
    """
    name = bu_vision.shm_name_for_screenshot()
    shm = shared_memory.SharedMemory( name=name, create=True, size=raw_screenshot.nbytes )
    shared_screenshot = np.ndarray( raw_screenshot.shape, dtype=raw_screenshot.dtype, buffer=shm.buf )

    timestamp1_shm = shared_memory.SharedMemory( name=betaconst.bv_ss_timestamp1_name, create=True, size=timestamp.nbytes )
    timestamp2_shm = shared_memory.SharedMemory( name=betaconst.bv_ss_timestamp2_name, create=True, size=timestamp.nbytes )
    shared_timestamp1 = np.ndarray( timestamp.shape, dtype=timestamp.dtype, buffer=timestamp1_shm.buf )
    shared_timestamp2 = np.ndarray( timestamp.shape, dtype=timestamp.dtype, buffer=timestamp2_shm.buf )

    shms = [ shm, timestamp1_shm, timestamp2_shm ]
    return shms, shared_screenshot, shared_timestamp1, shared_timestamp2


def publish_screenshot( timestamp, raw_screenshot, shared_screenshot, shared_timestamp1, shared_timestamp2 ):
    """
    Copy this capture's timestamp and raw screenshot into shared memory
    for betavision-detect.py to read.

    Writes timestamp1 BEFORE the image and timestamp2 AFTER it - this
    ordering is the torn-read handshake: a reader that sees
    timestamp1 == timestamp2 knows the image write in between completed
    before it read it.

    Args:
        timestamp: A 1-element np.array holding this capture's time.
        raw_screenshot: This capture's raw screenshot.
        shared_screenshot: The shared-memory np.ndarray view to copy
            raw_screenshot into (see init_shared_memory).
        shared_timestamp1: Shared-memory view written first.
        shared_timestamp2: Shared-memory view written last.
    """
    shared_timestamp1[:] = timestamp[:]
    shared_screenshot[:] = raw_screenshot[:]
    shared_timestamp2[:] = timestamp[:]


def write_debug_image( raw_screenshot ):
    """
    Writes the raw screenshot to disk as an annotated PNG, when
    betaconfig.debug_mode's bit 2 is set - for visually inspecting
    exactly what's being captured.

    Args:
        raw_screenshot: The unmodified screenshot for this capture.
    """
    cv2.imwrite( 'debug-vision-raw-screenshot.png', bu_censor.annotate_image_shape( raw_screenshot ) )


def main():
    """
    Entry point: capture and publish raw screenshots to shared memory in
    a tight loop, forever. Shared-memory segments are created once, on
    the first iteration, then reused (overwritten in place) on later
    iterations.
    """
    with mss.mss() as sct:
        shared_mem_inited = False

        shms = []
        shared_screenshot = None
        shared_timestamp1 = None
        shared_timestamp2 = None

        while True:
            ( capture_time, raw_screenshot ) = bu_vision.get_screenshot( sct )
            timestamp = np.array( [ capture_time ] )

            if not shared_mem_inited:
                shms, shared_screenshot, shared_timestamp1, shared_timestamp2 = init_shared_memory( raw_screenshot, timestamp )
                shared_mem_inited = True

            publish_screenshot( timestamp, raw_screenshot, shared_screenshot, shared_timestamp1, shared_timestamp2 )

            if betaconfig.debug_mode&2:
                write_debug_image( raw_screenshot )

            print( '%.3f'%(time.monotonic()-capture_time ))


if __name__ == '__main__':
    main()
