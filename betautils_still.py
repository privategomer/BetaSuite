"""
betautils_still.py - from one still image's detections to censorable boxes.

The one path betastare.py (photos) and betautils_preview.py (look
previews) share, so a preview is exactly what a photo with those
detections would come out as. It follows a video's first frame: each
detection goes through process_raw_box, then its look is resolved as
betautils_track resolves a new track's (paired_style, per-style area
safety, strength scaled from the box). Rendering is then
betautils_censor.censor_img_for_boxes, as for a video frame.
"""

import betautils_censor as bu_censor
import betautils_config as bu_config
import betautils_track as bu_track


def still_boxes( all_raw_boxes, img_w, img_h, parts_to_blur=None, backend_name=None, rng=None ):
    """
    Censorable boxes, looks resolved, for one image.

    Args:
        all_raw_boxes: Lists of raw detections, one list per detection
            size (a photo is detected at several sizes).
        img_w, img_h: Image size in pixels.
        parts_to_blur: The per-label settings table; default the run's
            (bu_config.get_parts_to_blur).
        backend_name: Backend whose per-label settings apply; default the
            selected one.
        rng: A random.Random for every random pick; default the shared
            generator. A preview passes a seeded one.

    Returns:
        A list of box dicts ready for censor_img_for_boxes.
    """
    if parts_to_blur is None:
        parts_to_blur = bu_config.get_parts_to_blur( backend_name )
    boxes = []
    for raw_boxes in all_raw_boxes:
        for raw_box in raw_boxes:
            box = bu_censor.process_raw_box( raw_box, img_w, img_h, parts_to_blur, rng )
            if box:
                boxes.append( box )
    return bu_track.resolve_still_styles( boxes, backend_name, rng )
