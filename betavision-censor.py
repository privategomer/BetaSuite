"""
betavision-censor.py

BetaVision live-pipeline stage 3 of 3 (screenshot -> detect -> censor,
see betavision-screenshot.py and betavision-detect.py). Reads the
latest raw model output betavision-detect.py published to shared
memory, converts it into censor boxes, and continuously renders a
live censored view of the captured screen region into an OpenCV
window - each displayed frame is time-delayed by
betaconfig.betavision_delay so a box has already been detected and
converted before the frame it belongs to is ever shown. Runs forever;
stop by focusing the display window and pressing 'q'.
"""
import mss
from multiprocessing import Process, Queue, shared_memory
import time
import cv2
import numpy as np
import win32gui, win32ui

import betaconfig
import betaconst

import betautils_vision as bu_vision
import betautils_censor as bu_censor


def open_detection_shared_memory():
    """
    Attach to the raw-box shared memory betavision-detect.py publishes
    into (see that file's init_output_shared_memory /
    publish_detection_output).

    This is already-parsed raw box dicts (canonical string
    class_id, coordinates already in full-screen pixel space) packed
    into a fixed-size betautils_vision.box_record_dtype array - not raw
    model tensors - so no per-adapter scale factors need computing here
    at all; that translation happens once, inside the adapter, at
    detect-time (see betautils_detector.py).

    Returns:
        A (remote_boxes, remote_count, remote_timestamp1,
        remote_timestamp2) tuple: remote_boxes is the shared-memory
        np.ndarray view (dtype box_record_dtype) onto
        betavision-detect.py's published boxes; remote_count is the
        shared 1-element view of how many of remote_boxes' leading
        slots are populated; the last two are its handshake timestamps.
    """
    boxes_shm = shared_memory.SharedMemory( name=betaconst.bv_detect_shm_boxes_name )
    remote_boxes = np.ndarray( ( betaconst.bv_detect_max_boxes, ), dtype=bu_vision.box_record_dtype, buffer=boxes_shm.buf )

    count_shm = shared_memory.SharedMemory( name=betaconst.bv_detect_shm_count_name )
    remote_count = np.ndarray( (1,), dtype=np.int64, buffer=count_shm.buf )

    out_timestamp1_shm = shared_memory.SharedMemory( name=betaconst.bv_detect_timestamp1_name )
    out_timestamp2_shm = shared_memory.SharedMemory( name=betaconst.bv_detect_timestamp2_name )
    remote_timestamp1 = np.ndarray( (1,), dtype=np.float64, buffer=out_timestamp1_shm.buf )
    remote_timestamp2 = np.ndarray( (1,), dtype=np.float64, buffer=out_timestamp2_shm.buf )

    return remote_boxes, remote_count, remote_timestamp1, remote_timestamp2


def open_display_window():
    """
    Create and size the OpenCV window the live censored view is shown
    in, and open this process's own screen-capture handle (separate
    from betavision-screenshot.py's - each BetaVision stage captures
    its own screenshots for display timing purposes; only the neural
    net's input images flow through shared memory).

    Returns:
        A (window_name, sct) pair: the window's name (needed by every
        later cv2.imshow call) and an open mss.mss() capture handle.
    """
    window_name = 'BetaVision'
    cv2.startWindowThread()
    cv2.namedWindow( window_name, cv2.WINDOW_NORMAL )
    cv2.resizeWindow( window_name, betaconfig.vision_cap_width, betaconfig.vision_cap_height )
    sct = mss.mss()
    return window_name, sct


def refresh_boxes_if_new_detection( remote_timestamp1, remote_timestamp2, last_detect_timestamp,
                                     remote_boxes, remote_count, boxes ):
    """
    If betavision-detect.py has published a new, fully-written
    detection result since the last check (both handshake timestamps
    agree, and the agreed value is newer than the last one consumed),
    converts its raw boxes into censor boxes and appends them to the
    running list of not-yet-expired boxes.

    Args:
        remote_timestamp1: Shared-memory view of the detector's first
            handshake timestamp.
        remote_timestamp2: Shared-memory view of the detector's second
            handshake timestamp.
        last_detect_timestamp: The detection timestamp last consumed.
        remote_boxes: Shared-memory view (dtype box_record_dtype) of
            the detector's published raw boxes.
        remote_count: Shared-memory view of how many of remote_boxes'
            leading slots are populated.
        boxes: The running list of live censor boxes to append any
            newly-converted boxes to, in place. Re-sorted by 'end'
            (ascending) whenever new boxes are added, so the
            expiration check in the main loop can always assume the
            soonest-to-expire box is at the front.

    Returns:
        The (possibly updated) last_detect_timestamp - unchanged if no
        new detection was found.
    """
    if remote_timestamp1[0] == remote_timestamp2[0] and remote_timestamp1[0] != last_detect_timestamp:
        last_detect_timestamp = remote_timestamp1[0]
        local_count = int( remote_count[0] )
        local_boxes = np.ndarray( remote_boxes.shape, dtype=remote_boxes.dtype )
        local_boxes[:] = remote_boxes[:]

        raw_boxes = bu_vision.shared_array_to_boxes( local_boxes, local_count )

        this_boxes = [ bu_censor.process_raw_box( raw, betaconfig.vision_cap_width, betaconfig.vision_cap_height ) for raw in raw_boxes ]
        this_boxes = [ box for box in this_boxes if box ]
        boxes.extend( this_boxes )
        boxes.sort( key=lambda x: x['end'] )

    return last_detect_timestamp


def expire_old_boxes( boxes ):
    """
    Drops every box from the front of boxes (sorted by 'end' ascending)
    whose censor window has already ended, given the same
    betaconfig.betavision_delay used to pick which captured frame is
    actually displayed.

    Args:
        boxes: The running list of live censor boxes, mutated in place.
    """
    while len( boxes ) and boxes[0]['end'] < time.monotonic() - betaconfig.betavision_delay:
        boxes.pop(0)


def trim_stale_screenshots( img_buffer ):
    """
    Drops leading screenshots from img_buffer once they're old enough
    that they can no longer be needed as the "earlier" side of an
    interpolated frame (see resolve_display_frame) - keeping at least
    one screenshot in the buffer at all times.

    Args:
        img_buffer: List of (capture_time, screenshot) pairs, oldest
            first, mutated in place.
    """
    while len( img_buffer ) > 1 and time.monotonic() - img_buffer[1][0] > betaconfig.betavision_delay:
        img_buffer.pop(0)


def resolve_display_frame( img_buffer, frame_timestamp ):
    """
    Determines the frame to actually display for frame_timestamp - a
    fixed delay behind real time, so live detection/conversion always
    has a chance to catch up before a frame is shown.

    Args:
        img_buffer: List of (capture_time, screenshot) pairs, oldest
            first, with at least one entry old enough to cover
            frame_timestamp (guaranteed by the main loop's own check
            before calling this).
        frame_timestamp: The point in time to render, always
            betaconfig.betavision_delay behind time.monotonic().

    Returns:
        The screenshot to display for frame_timestamp - either
        img_buffer[0]'s raw screenshot directly, or (when
        betaconfig.betavision_interpolate is on and a second buffered
        screenshot exists) a blend interpolated between img_buffer[0]
        and img_buffer[1] at frame_timestamp.
    """
    if betaconfig.betavision_interpolate:
        return bu_vision.interpolate_images( img_buffer[0][1], img_buffer[0][0], img_buffer[1][1], img_buffer[1][0], frame_timestamp )
    return img_buffer[0][1]


def draw_cursor_marker( frame ):
    """
    Draws a small filled square at the current mouse cursor's position
    (converted from screen coordinates into this capture region's own
    coordinate space), so the live preview shows where the mouse is
    even though the underlying screen content is censored - only when
    the cursor is actually within the captured region.

    Args:
        frame: The frame to draw onto, in place (also returned).

    Returns:
        frame, annotated with cursor position text as well when
        betaconfig.debug_mode's bit 1 is set.
    """
    flags, hcursor, (cursor_x, cursor_y) = win32gui.GetCursorInfo()
    cursor_x = cursor_x - betaconfig.vision_cap_left
    cursor_y = cursor_y - betaconfig.vision_cap_top

    if 5 < cursor_x < betaconfig.vision_cap_width and 5 < cursor_y < betaconfig.vision_cap_height:
        color = tuple( reversed( betaconfig.vision_cursor_color ) )
        frame[cursor_y-5:cursor_y+5, cursor_x-5:cursor_x+5] = color
        if betaconfig.debug_mode&1:
            frame = cv2.putText( frame, '(%d,%d)'%(cursor_x,cursor_y), (max(cursor_x-10,0),max(cursor_y-10,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2 )
    return frame


def main():
    """
    Entry point: attach to betavision-detect.py's shared memory, open
    the display window, then run the live censor/display loop forever
    - each iteration captures a screenshot, refreshes censor boxes
    from the latest detection (if any), expires boxes/screenshots
    that have aged out, renders a censored+cursor-annotated frame
    delayed by betaconfig.betavision_delay, and shows it. Exits when
    'q' is pressed with the display window focused.
    """
    ( remote_boxes, remote_count,
      remote_timestamp1, remote_timestamp2 ) = open_detection_shared_memory()

    window_name, sct = open_display_window()

    last_detect_timestamp = 0
    img_buffer = []
    boxes = []

    while True:
        times = [ time.perf_counter() ]
        img_buffer.append( bu_vision.get_screenshot( sct ) )

        if betaconfig.debug_mode&2:
            cv2.imwrite( 'debug-vision-precensor.png', img_buffer[-1][1] )

        times.append( time.perf_counter() )
        last_detect_timestamp = refresh_boxes_if_new_detection(
            remote_timestamp1, remote_timestamp2, last_detect_timestamp,
            remote_boxes, remote_count, boxes )

        times.append( time.perf_counter() )
        expire_old_boxes( boxes )

        times.append( time.perf_counter() )
        trim_stale_screenshots( img_buffer )

        times.append( time.perf_counter() )
        frame_timestamp = time.monotonic() - betaconfig.betavision_delay

        # nothing in the buffer is old enough
        if img_buffer[0][0] > frame_timestamp:
            continue

        times.append( time.perf_counter() )
        frame = resolve_display_frame( img_buffer, frame_timestamp )

        times.append( time.perf_counter() )
        live_boxes = [ box for box in boxes if box['start'] < frame_timestamp < box['end'] ]

        times.append( time.perf_counter() )
        frame = bu_censor.censor_img_for_boxes( frame, live_boxes )

        if betaconfig.debug_mode&1:
            frame = bu_censor.annotate_image_shape( frame )

        frame = draw_cursor_marker( frame )

        if betaconfig.debug_mode&2:
            cv2.imwrite( 'debug-vision-postcensor.png', frame )

        times.append( time.perf_counter() )
        cv2.imshow( window_name, frame )
        times.append( time.perf_counter() )
        times_display = [ '%.3f'%(x-times[0]) for x in times ]
        print( times_display )

        if cv2.waitKey(1) & 0xFF==ord("q"):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
