"""
betautils_model.py - Loads the ONNX detection model and runs raw
inference: image preprocessing, batched session.run() calls, and turning
the model's raw tensor output into plain-dict "raw box" records.

Orchestration flow for one frame or a batch of frames:
    raw_boxes_for_img/raw_boxes_for_imgs
        -> get_image_resize_scale + prep_img_for_nn   (preprocessing)
        -> detect_raw_boxes
            -> get_raw_model_output                    (batched session.run)
            -> raw_boxes_from_model_output              (tensor -> dict boxes)

Everything downstream (class_suppression, tracking/smoothing in betatv.py)
consumes the plain dicts raw_boxes_from_model_output produces, not the
raw model tensors - this file is the only place that speaks ONNX.
"""

import cv2
import math
import numpy as np
import onnxruntime

import betaconfig
import betaconst
import betautils_detector as bu_detector


def get_session():
    """
    Load the detection model into an ONNX Runtime inference session.

    Uses CUDA if betaconfig.gpu_enabled is truthy (on the device given by
    betaconfig.cuda_device_id), otherwise falls back to CPU.

    Returns:
        A ready-to-use onnxruntime.InferenceSession for
        '../resources/model/detector_v2_default_checkpoint.onnx'.
    """
    if betaconfig.gpu_enabled:
        bu_detector.preload_cuda_libraries()
        providers = [ ( 'CUDAExecutionProvider', { 'device_id': betaconfig.cuda_device_id } ) ]
    else:
        providers = [ ( 'CPUExecutionProvider', {} ) ]

    session = onnxruntime.InferenceSession( '../resources/model/detector_v2_default_checkpoint.onnx', providers=providers )
    return( session )


def get_resize_scale( img_height, img_width, max_length ):
    """
    Compute the scale factor needed to fit an image's longer side to
    max_length, preserving aspect ratio.

    Args:
        img_height: Image height in pixels.
        img_width: Image width in pixels.
        max_length: Target length for the image's longer side. 0 means
            "don't resize" (the model runs on native resolution).

    Returns:
        1 if max_length is 0, otherwise max_length divided by whichever
        of img_height/img_width is larger.
    """
    if max_length == 0:
        return(1)
    else:
        return( max_length/max(img_height, img_width) )


def get_image_resize_scale( raw_img, max_length ):
    """
    Convenience wrapper around get_resize_scale that reads the image's
    height/width directly from its array shape.

    Args:
        raw_img: An HxWxC numpy image array.
        max_length: See get_resize_scale.

    Returns:
        Same as get_resize_scale.
    """
    (img_height, img_width, _channels) = raw_img.shape
    return( get_resize_scale( img_height, img_width, max_length ) )


def prep_img_for_nn( raw_img, size, scale ):
    """
    Resize, pad, and normalize a raw image into the exact format the
    detection model expects as input.

    Args:
        raw_img: An HxWxC numpy image array (BGR, uint8).
        size: If > 0, the image is right/bottom-padded with black pixels
            up to size x size after resizing (the model expects a fixed
            square input when batching non-uniformly-sized frames
            together). If 0, no padding is applied.
        scale: The resize factor to apply, as returned by
            get_resize_scale/get_image_resize_scale.

    Returns:
        A float32 numpy array, resized by scale, optionally padded to
        size x size, with the model's fixed per-channel BGR mean
        ([103.939, 116.779, 123.68]) subtracted. This mean-subtraction
        must exactly match how the model was trained - do not change it
        independently of retraining/re-exporting the model.
    """
    resized_img = cv2.resize( raw_img, None, fx=scale, fy=scale )

    if size > 0:
        (resized_h, resized_w, _channels) = resized_img.shape
        resized_img = cv2.copyMakeBorder( resized_img, 0, size - resized_h, 0, size - resized_w, cv2.BORDER_CONSTANT, value=0 )

    normalized_img = resized_img.astype(np.float32)
    normalized_img -= [103.939, 116.779, 123.68 ]
    return( normalized_img )


def get_raw_model_output( img_array, session ):
    """
    Run the detection model on a batch of preprocessed images, chunking
    the batch according to betaconfig.nn_batch_size.

    Args:
        img_array: A list/array of preprocessed images (as returned by
            prep_img_for_nn), all the same size.
        session: An onnxruntime.InferenceSession from get_session().

    Returns:
        A [boxes, scores, classes] list of numpy arrays, one row per
        input image: boxes is (n, 300, 4), scores is (n, 300), classes
        is (n, 300) - the model's fixed-size top-300-detections output,
        before any min_prob filtering.

    Note:
        Batches session.run() calls the SELECTED BACKEND's resolved
        nn_batch_size images at a time instead of always calling it once
        per image. This used to read the module-level
        betaconfig.nn_batch_size, which was removed - leaving it
        would have pinned this path to 1 no matter what the backend
        block said. At nn_batch_size=1 this is byte-for-byte the same behavior as the
        original one-image-per-call loop (each batch below has exactly
        one image in it) - raising nn_batch_size is what actually
        changes anything. See the nn_batch_size comment in
        betaconfig.py: this hasn't been verified against
        detector_v2_default_checkpoint.onnx specifically, since some
        exported detector graphs are baked to a fixed batch size of 1
        and will error (or silently misbehave) if handed more than one
        image per call - if that happens, set nn_batch_size back to 1 in
        betaconfig.py.
    """
    num_images = len( img_array )
    output = [
            np.zeros( (num_images, 300, 4 ), dtype = np.float32 ),
            np.zeros( (num_images, 300 ), dtype = np.float32 ),
            np.zeros( (num_images, 300 ), dtype = np.int32 ),
    ]

    batch_size = max( 1, bu_detector.get_nn_batch_size() )
    for batch_start in range( 0, num_images, batch_size ):
        batch_end = min( batch_start + batch_size, num_images )
        batch_imgs = [ img_array[j] for j in range( batch_start, batch_end ) ]
        (batch_boxes, batch_scores, batch_classes) = session.run( betaconst.model_outputs, { betaconst.model_input: batch_imgs } )
        output[0][batch_start:batch_end] = batch_boxes
        output[1][batch_start:batch_end] = batch_scores
        output[2][batch_start:batch_end] = batch_classes

    return( output )


def raw_boxes_from_model_output( model_output, scale_array, t_array ):
    """
    Convert the model's raw per-image tensor output into plain-dict "raw
    box" records, filtering out anything below the global confidence
    floor and rescaling box coordinates back to original-image pixels.

    Args:
        model_output: The [boxes, scores, classes] output from
            get_raw_model_output.
        scale_array: One resize scale per image in the batch (parallel
            to model_output's rows), as used to preprocess that image -
            needed to map the model's coordinates (in resized-image
            space) back to original-image pixel coordinates.
        t_array: One timestamp per image in the batch, parallel to
            scale_array. Not one shared timestamp - each image in a
            batch is a different video frame and needs its own 't' on
            the boxes it produces.

    Returns:
        A list of lists: one inner list of raw box dicts per input
        image, each dict shaped
        {'x', 'y', 'w', 'h', 'class_id', 'score', 't'} in original-image
        pixel coordinates. Detections scoring at or below
        betaconfig.global_min_prob (falling back to
        betaconst.global_min_prob if unset) are dropped entirely here,
        before any per-label min_prob override in betaconfig.py ever
        sees them.
    """
    all_raw_boxes = []
    all_boxes   = model_output[0]
    all_scores  = model_output[1]
    all_classes = model_output[2]
    for boxes, scores, classes, scale, t in zip( all_boxes, all_scores, all_classes, scale_array, t_array ):
        frame_raw_boxes = []
        for box, score, class_id in zip( boxes, scores, classes ):
            if score > getattr( betaconfig, 'global_min_prob', betaconst.global_min_prob ):
                frame_raw_boxes.append( {
                    'x': float(math.floor(box[0]/scale)),
                    'y': float(math.floor(box[1]/scale)),
                    'w': float(math.ceil((box[2]-box[0])/scale)),
                    'h': float(math.ceil((box[3]-box[1])/scale)),
                    'class_id': float(class_id),
                    'score': float(score),
                    't': t,
                } )
        all_raw_boxes.append(frame_raw_boxes)
    return( all_raw_boxes )


def detect_raw_boxes( img_array, session, scale_array, t_array ):
    """
    Run the full raw-detection pipeline on a batch of preprocessed
    images: model inference followed by tensor-to-dict conversion.

    Args:
        img_array: Preprocessed images, as prep_img_for_nn produces.
        session: An onnxruntime.InferenceSession from get_session().
        scale_array: One resize scale per image (see
            raw_boxes_from_model_output).
        t_array: One timestamp per image (see
            raw_boxes_from_model_output).

    Returns:
        Same as raw_boxes_from_model_output: a list of per-image raw box
        lists.
    """
    model_output = get_raw_model_output( img_array, session )
    return( raw_boxes_from_model_output( model_output, scale_array, t_array ) )


def raw_boxes_for_img( img, size, session, t ):
    """
    Run detection on a single raw image, handling preprocessing
    internally.

    Args:
        img: An HxWxC numpy image array (BGR, uint8), unprocessed.
        size: Target square input size - see prep_img_for_nn.
        session: An onnxruntime.InferenceSession from get_session().
        t: Timestamp to attach to every box detected in this image.

    Returns:
        A list of raw box dicts for this one image (see
        raw_boxes_from_model_output for the dict shape).
    """
    scale = get_image_resize_scale( img, size )
    preprocessed_img = prep_img_for_nn( img, size, scale )
    raw_boxes = detect_raw_boxes( np.expand_dims( preprocessed_img, axis=0 ), session, [ scale ], [ t ] )[0]
    return( raw_boxes )


def raw_boxes_for_imgs( imgs, size, session, ts ):
    """
    Batched sibling of raw_boxes_for_img: preprocesses a list of frames
    and runs detection on all of them together (subject to
    betaconfig.nn_batch_size chunking inside get_raw_model_output), then
    returns one flat list of raw boxes across every frame in the batch -
    equivalent to calling raw_boxes_for_img once per (img, t) pair and
    concatenating the results, just with fewer/larger session.run()
    calls.

    Args:
        imgs: A list of raw HxWxC numpy image arrays, unprocessed.
        size: Target square input size - see prep_img_for_nn.
        session: An onnxruntime.InferenceSession from get_session().
        ts: One timestamp per image in imgs, parallel to it.

    Returns:
        A single flat list of raw box dicts across every image in imgs
        (see raw_boxes_from_model_output for the dict shape).
    """
    scales = [ get_image_resize_scale( img, size ) for img in imgs ]
    preprocessed_imgs = [ prep_img_for_nn( img, size, scale ) for img, scale in zip( imgs, scales ) ]
    per_frame_boxes = detect_raw_boxes( preprocessed_imgs, session, scales, ts )
    return( [ box for frame_boxes in per_frame_boxes for box in frame_boxes ] )
