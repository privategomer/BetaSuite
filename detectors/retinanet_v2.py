"""
detectors/retinanet_v2.py - Adapter for BetaSuite's original detection
model, '../resources/model/detector_v2_default_checkpoint.onnx'.

A RetinaNet-style export: fixed 300-detections-per-image output, with
NMS and score filtering already baked into the graph (tensor names under
'filtered_detections/...'). All this adapter has to do is preprocess,
run the session, rescale coordinates back to original-image pixels, and
translate the model's numeric class index into BetaSuite's canonical
string labels via _MODEL_CLASS_ORDER.

Because the graph does its own NMS, there is nothing here equivalent to
nudenet_v3's candidate_floor / nms_iou / nms_mode - this backend has no
settings that change which raw detections come out, which is why
detection_identity() below is empty.
"""

import cv2
import math
import numpy as np

import betaconfig
import betaconst
import betautils_detector as bu_detector


MODEL_PATH = '../resources/model/detector_v2_default_checkpoint.onnx'

# Fixed detections per image in this export's output tensors.
DETECTIONS_PER_IMAGE = 300

# Per-channel BGR mean this model was trained with. Changing it without
# retraining and re-exporting silently degrades every detection.
_BGR_MEAN = np.array( [ 103.939, 116.779, 123.68 ], dtype=np.float32 )

# This model's own class-index order, exactly as it was trained and
# exported. NOT the same thing as betaconst.classes' iteration order:
# betaconst.classes is BetaSuite's canonical vocabulary shared across
# every adapter, this is one model's index table. If this ever needs to
# change it means the model was re-exported, not that betaconst changed.
_MODEL_CLASS_ORDER = [
    'exposed_anus',
    'exposed_armpits',
    'covered_belly',
    'exposed_belly',
    'covered_buttocks',
    'exposed_buttocks',
    'face_femme',
    'face_masc',
    'covered_feet',
    'exposed_feet',
    'covered_breast',
    'exposed_breast',
    'covered_vulva',
    'exposed_vulva',
    'exposed_chest',
    'exposed_penis',
]


# ---------------------------------------------------------------------------
# Adapter hooks
# ---------------------------------------------------------------------------

def native_picture_sizes( config ):
    """
    This model's default detection sizes.

    Unlike nudenet_v3 this export has no single validated input size -
    it was trained on variable-size input and is routinely run at 1280.
    Returning [] defers to betaconfig.picture_sizes, preserving the
    earlier behaviour for this backend exactly.
    """
    return []


def detection_identity( config ):
    """
    Settings that change this backend's raw detections: none.

    Score filtering and NMS live inside the exported graph, so the only
    inputs to a detection are the image, the size, and global_min_prob -
    all of which betautils_cache_paths already keys on directly.
    """
    return {}


def validate_backend_config( config, errors ):
    """Sanity-check detector_backend['retinanet_v2'], appending to errors."""
    for size in bu_detector.get_picture_sizes( 'retinanet_v2' ):
        if not isinstance( size, int ) or size < 0:
            errors.append( "retinanet_v2 picture sizes must be non-negative ints "
                           "(0 means 'native resolution, no resize'), got %r"%(size,) )


def get_session():
    """
    Load the detection model into an ONNX Runtime session.

    Provider selection and the "CUDA requested but not active" warning
    are shared - see betautils_detector.build_onnx_session.
    """
    return bu_detector.build_onnx_session( MODEL_PATH, backend_name='retinanet_v2' )


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def get_resize_scale( img_height, img_width, max_length ):
    """
    Scale factor that fits an image's longer side to max_length.

    Args:
        img_height: Height in pixels.
        img_width: Width in pixels.
        max_length: Target longer-side length. 0 means "do not resize".

    Returns:
        1 when max_length is 0, else max_length / max(height, width).
    """
    if max_length == 0:
        return 1
    return max_length / max( img_height, img_width )


def get_image_resize_scale( raw_img, max_length ):
    """get_resize_scale, reading the dimensions from the array's shape."""
    img_height, img_width = raw_img.shape[:2]
    return get_resize_scale( img_height, img_width, max_length )


def prep_img_for_nn( raw_img, size, scale ):
    """
    Resize, pad, and mean-subtract one image into this model's input format.

    Private to this adapter: a different backend needs entirely different
    preprocessing, which is why preprocessing is not part of the shared
    Detector interface.

    Args:
        raw_img: HxWxC numpy array, BGR uint8.
        size: When > 0, the resized image is padded bottom/right with
            black to size x size, so frames of different shapes can be
            batched together. 0 means no padding.
        scale: Resize factor from get_resize_scale.

    Returns:
        A float32 array, resized, optionally padded, with this model's
        fixed per-channel BGR mean subtracted.
    """
    resized_img = cv2.resize( raw_img, None, fx=scale, fy=scale )

    if size > 0:
        resized_h, resized_w = resized_img.shape[:2]
        resized_img = cv2.copyMakeBorder(
            resized_img, 0, size - resized_h, 0, size - resized_w,
            cv2.BORDER_CONSTANT, value=0 )

    normalized_img = resized_img.astype( np.float32 )
    normalized_img -= _BGR_MEAN
    return normalized_img


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def get_raw_model_output( img_array, session, batch_size=None ):
    """
    Run the model over a list of preprocessed images, in batches.

    Args:
        img_array: Preprocessed images, all the same size.
        session: An onnxruntime.InferenceSession from get_session().
        batch_size: Images per session.run() call. None resolves it from
            betautils_detector.get_nn_batch_size('retinanet_v2').

    Returns:
        A [boxes, scores, classes] list of arrays with one row per input
        image: boxes (n, 300, 4), scores (n, 300), classes (n, 300).

    Note:
        Previously this read the TOP-LEVEL betaconfig.nn_batch_size
        rather than the per-backend resolved value, so
        detector_backend['retinanet_v2']['nn_batch_size'] = 2 was
        silently ignored: betatv buffered two frames, handed them over,
        and this split them straight back into two single-image calls.
        Output filenames still said '-batch2'. The per-backend value is
        now the one actually used; batching invariance is enforced by
        tests/test_batch_size_invariance.py.
    """
    num_images = len( img_array )
    output = [
        np.zeros( ( num_images, DETECTIONS_PER_IMAGE, 4 ), dtype=np.float32 ),
        np.zeros( ( num_images, DETECTIONS_PER_IMAGE ),    dtype=np.float32 ),
        np.zeros( ( num_images, DETECTIONS_PER_IMAGE ),    dtype=np.int32 ),
    ]

    if batch_size is None:
        batch_size = bu_detector.get_nn_batch_size( 'retinanet_v2' )
    batch_size = max( 1, int( batch_size ) )

    for batch_start in range( 0, num_images, batch_size ):
        batch_end = min( batch_start + batch_size, num_images )
        batch_imgs = list( img_array[ batch_start : batch_end ] )
        batch_boxes, batch_scores, batch_classes = session.run(
            betaconst.model_outputs, { betaconst.model_input: batch_imgs } )
        output[0][ batch_start : batch_end ] = batch_boxes
        output[1][ batch_start : batch_end ] = batch_scores
        output[2][ batch_start : batch_end ] = batch_classes

    return output


def raw_boxes_from_model_output( model_output, scale_array, t_array, size_array=None ):
    """
    Convert the model's per-image tensors into raw box dicts.

    Vectorised per image: the score filter and coordinate rescale are
    array operations, and only the surviving detections (typically a
    handful out of 300) are materialised as dicts.

    Args:
        model_output: [boxes, scores, classes] from get_raw_model_output.
        scale_array: One resize scale per image, parallel to the rows.
        t_array: One timestamp per image. Each image in a batch is a
            different video frame and needs its own 't'.
        size_array: One picture size per image, recorded on each box so
            cross-size dedup can tell which pass produced it. Defaults
            to 0 (unknown) when omitted.

    Returns:
        A list of per-image lists of raw box dicts, in original-image
        pixel coordinates, with canonical string class_id. Detections at
        or below global_min_prob are dropped here, before any per-label
        min_prob ever sees them.
    """
    all_boxes, all_scores, all_classes = model_output[0], model_output[1], model_output[2]
    min_prob_floor = getattr( betaconfig, 'global_min_prob', betaconst.global_min_prob )
    if size_array is None:
        size_array = [ 0 ] * len( scale_array )

    all_raw_boxes = []
    for boxes, scores, classes, scale, t, size in zip(
            all_boxes, all_scores, all_classes, scale_array, t_array, size_array ):
        scores = np.asarray( scores )
        keep = np.nonzero( scores > min_prob_floor )[0]
        frame_raw_boxes = []
        for index in keep:
            class_index = int( classes[index] )
            if class_index < 0 or class_index >= len( _MODEL_CLASS_ORDER ):
                # A class index this model was never exported with. Only
                # possible with a corrupt or mismatched model file; skip
                # the detection rather than fail the whole run.
                continue
            box = boxes[index]
            frame_raw_boxes.append( {
                'x': float( math.floor( box[0] / scale ) ),
                'y': float( math.floor( box[1] / scale ) ),
                'w': float( math.ceil( ( box[2] - box[0] ) / scale ) ),
                'h': float( math.ceil( ( box[3] - box[1] ) / scale ) ),
                'class_id': _MODEL_CLASS_ORDER[ class_index ],
                'score': float( scores[index] ),
                't': t,
                'size': size,
            } )
        all_raw_boxes.append( frame_raw_boxes )
    return all_raw_boxes


def detect_raw_boxes( img_array, session, scale_array, t_array, size_array=None,
                      batch_size=None ):
    """Inference followed by tensor-to-dict conversion."""
    model_output = get_raw_model_output( img_array, session, batch_size=batch_size )
    return raw_boxes_from_model_output( model_output, scale_array, t_array, size_array )


# ---------------------------------------------------------------------------
# Detector interface
# ---------------------------------------------------------------------------

def raw_boxes_for_img( img, size, session, t ):
    """Detect on one raw image. See betautils_detector.Detector."""
    return raw_boxes_for_imgs( [ img ], size, session, [ t ] )


def raw_boxes_for_imgs( imgs, size, session, ts ):
    """
    Detect on several raw images, batching per nn_batch_size.

    Args:
        imgs: List of raw HxWxC BGR uint8 arrays.
        size: Target square input size - see prep_img_for_nn.
        session: An onnxruntime.InferenceSession from get_session().
        ts: One timestamp per image, parallel to imgs.

    Returns:
        One flat list of raw box dicts across every image.
    """
    if not imgs:
        return []
    scales = [ get_image_resize_scale( img, size ) for img in imgs ]
    preprocessed = [ prep_img_for_nn( img, size, scale ) for img, scale in zip( imgs, scales ) ]
    per_frame_boxes = detect_raw_boxes(
        preprocessed, session, scales, ts, [ size ] * len( imgs ) )
    return [ box for frame_boxes in per_frame_boxes for box in frame_boxes ]
