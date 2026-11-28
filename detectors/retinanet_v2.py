"""
detectors/retinanet_v2.py - Adapter for the RetinaNet detection model
(detector_v2_default_checkpoint.onnx).

The exported graph does the heavy lifting itself: it returns a fixed 300
detections per image, with score filtering and non-maximum suppression
already applied inside the graph. This adapter therefore only:

    1. preprocesses each image (resize, pad to a square, subtract the
       training mean),
    2. runs the session in batches,
    3. rescales the boxes back to the original image, and
    4. maps the model's class indexes to BetaSuite's label names.

Because the filtering lives in the graph, this backend has no detection
settings of its own (no candidate floor, no NMS threshold). Its detection
identity is therefore just the model file's hash: nothing else can change
which raw detections come out.
"""

import math

import cv2
import numpy as np

import betaconfig
import betaconst
import betautils_detector as bu_detector
import betautils_models as bu_models


def _rule( rule_id, text ):
    """An error with a stable rule id for the engine protocol (betautils_settings.rule)."""
    # Imported here: betautils_settings imports the detector registry,
    # which imports this module.
    import betautils_settings as bu_settings
    return bu_settings.rule( rule_id, text )


MODEL_PATH = '../resources/model/detector_v2_default_checkpoint.onnx'

# Fixed by the export: every image yields exactly this many rows, most
# of them below any useful score.
DETECTIONS_PER_IMAGE = 300

# The per-channel mean the model was trained with, in BGR order.
# Changing it without re-exporting the model degrades every detection.
_BGR_MEAN = np.array( [ 103.939, 116.779, 123.68 ], dtype=np.float32 )

# The model's own class index table (index 0 first). This is NOT
# betaconst.classes' order; it changes only if the model is re-exported.
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

# The labels this backend can report, declared for the app: a label
# outside this set is shown as unavailable while this backend is selected.
SUPPORTED_LABELS = frozenset( _MODEL_CLASS_ORDER )


# ---------------------------------------------------------------------------
# Adapter hooks
# ---------------------------------------------------------------------------

def native_picture_sizes( config ):
    """
    No single native size: the graph takes any input, so this backend
    defers to the configured picture sizes (typically 1280).
    """
    return []


def detection_identity( config ):
    """
    The settings that change which raw detections come out of this model.

    Only the model file: scoring and NMS are baked into the graph, so
    there is nothing else to fold into the detection cache key.
    """
    return { 'model_sha256': bu_models.file_identity( MODEL_PATH ) }


def validate_backend_config( config, errors ):
    """
    Sanity-check detector_backend['retinanet_v2'], appending to `errors`.

    The only rule of this model's own is its picture sizes: non-negative
    ints, where 0 means native resolution with no resize.
    """
    for size in bu_detector.get_picture_sizes( 'retinanet_v2' ):
        if not isinstance( size, int ) or size < 0:
            errors.append( _rule( 'backend.size_negative',
                                  "retinanet_v2 picture sizes must be non-negative ints "
                                  "(0 means native resolution, no resize); got %r"%( size, ) ) )


def get_session():
    """
    Load the model into an ONNX Runtime session.

    Provider selection is shared with every adapter; see
    betautils_detector.build_onnx_session.
    """
    return bu_detector.build_onnx_session( MODEL_PATH, backend_name='retinanet_v2' )


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def get_resize_scale( img_height, img_width, max_length ):
    """
    The factor that fits an image's longer side to max_length.

    Returns:
        1 when max_length is 0 (native resolution, no resize), else
        max_length / the longer side.
    """
    if max_length == 0:
        return 1
    return max_length / max( img_height, img_width )


def get_image_resize_scale( raw_img, max_length ):
    """get_resize_scale for an image array."""
    img_height, img_width = raw_img.shape[:2]
    return get_resize_scale( img_height, img_width, max_length )


def prep_img_for_nn( raw_img, size, scale ):
    """
    Turn a raw image into this model's input.

    Private to this adapter: another backend needs its own preprocessing.

    Args:
        raw_img: HxWxC BGR uint8 array.
        size: The square input size; 0 for native resolution, unpadded.
        scale: From get_image_resize_scale( raw_img, size ).

    Returns:
        A float32 array: the image resized by scale, padded with black on
        the bottom and right up to size x size (when size > 0), minus the
        training mean.
    """
    image = cv2.resize( raw_img, None, fx=scale, fy=scale )
    if size > 0:
        img_height, img_width = image.shape[:2]
        # Bottom and right only, so box coordinates need a scale but no
        # offset to map back.
        image = cv2.copyMakeBorder( image, 0, size - img_height, 0, size - img_width,
                                    cv2.BORDER_CONSTANT, value=0 )
    image = image.astype( np.float32 )
    image -= _BGR_MEAN
    return image


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def get_raw_model_output( img_array, session, batch_size=None ):
    """
    Run the model over preprocessed images, in batches.

    The batch size honoured is this backend's own
    (bu_detector.get_nn_batch_size). An older version read a top-level
    setting instead and silently ran one image per call;
    tests/test_batch_size_invariance.py guards against that coming back.

    Args:
        img_array: List of preprocessed images (prep_img_for_nn).
        session: An onnxruntime.InferenceSession from get_session().
        batch_size: Images per session call; default the backend's.

    Returns:
        [ boxes (n, 300, 4) float32, scores (n, 300) float32,
          classes (n, 300) int32 ], n being the number of images.
    """
    count = len( img_array )
    boxes = np.zeros( ( count, DETECTIONS_PER_IMAGE, 4 ), dtype=np.float32 )
    scores = np.zeros( ( count, DETECTIONS_PER_IMAGE ), dtype=np.float32 )
    classes = np.zeros( ( count, DETECTIONS_PER_IMAGE ), dtype=np.int32 )

    if batch_size is None:
        batch_size = bu_detector.get_nn_batch_size( 'retinanet_v2' )
    batch_size = max( 1, int( batch_size ) )

    for start in range( 0, count, batch_size ):
        end = min( start + batch_size, count )
        batch = list( img_array[ start:end ] )
        batch_boxes, batch_scores, batch_classes = session.run(
            betaconst.model_outputs, { betaconst.model_input: batch } )
        boxes[ start:end ] = batch_boxes
        scores[ start:end ] = batch_scores
        classes[ start:end ] = batch_classes
    return [ boxes, scores, classes ]


def raw_boxes_from_model_output( model_output, scale_array, t_array, size_array=None ):
    """
    Turn the model's output into raw box dicts, one list per image.

    Keeps the rows scoring strictly above betaconfig.global_min_prob and
    maps their corners from resized pixels back to the original image.

    Args:
        model_output: [ boxes, scores, classes ] from get_raw_model_output.
        scale_array: Each image's resize scale.
        t_array: Each image's timestamp.
        size_array: Each image's picture size; default all 0.

    Returns:
        A list with one list of raw box dicts per image.
    """
    min_prob_floor = getattr( betaconfig, 'global_min_prob', betaconst.global_min_prob )
    boxes, scores, classes = model_output
    if size_array is None:
        size_array = [ 0 ] * len( boxes )

    all_raw_boxes = []
    for img_boxes, img_scores, img_classes, scale, t, size in zip(
            boxes, scores, classes, scale_array, t_array, size_array ):
        # 300 rows per image and only a few survive the floor: filter as
        # an array, then build dicts for the survivors.
        kept = np.nonzero( img_scores > min_prob_floor )[0]
        frame_raw_boxes = []
        for index in kept:
            class_index = int( img_classes[ index ] )
            if class_index < 0 or class_index >= len( _MODEL_CLASS_ORDER ):
                # Only a corrupt or mismatched model file does this; skip
                # the row rather than fail the run.
                continue
            x1, y1, x2, y2 = img_boxes[ index ]
            frame_raw_boxes.append( {
                'x': float( math.floor( x1 / scale ) ),
                'y': float( math.floor( y1 / scale ) ),
                'w': float( math.ceil( ( x2 - x1 ) / scale ) ),
                'h': float( math.ceil( ( y2 - y1 ) / scale ) ),
                'class_id': _MODEL_CLASS_ORDER[ class_index ],
                'score': float( img_scores[ index ] ),
                't': t,
                'size': size,
            } )
        all_raw_boxes.append( frame_raw_boxes )
    return all_raw_boxes


def detect_raw_boxes( img_array, session, scale_array, t_array, size_array=None, batch_size=None ):
    """get_raw_model_output, then raw_boxes_from_model_output."""
    model_output = get_raw_model_output( img_array, session, batch_size=batch_size )
    return raw_boxes_from_model_output( model_output, scale_array, t_array, size_array )


# ---------------------------------------------------------------------------
# Detector interface
# ---------------------------------------------------------------------------

def raw_boxes_for_img( img, size, session, t ):
    """
    Detect on one raw image: raw_boxes_for_imgs with a batch of one.

    Returns:
        A flat list of raw box dicts.
    """
    return raw_boxes_for_imgs( [ img ], size, session, [ t ] )


def raw_boxes_for_imgs( imgs, size, session, ts ):
    """
    Detect on several raw images.

    Args:
        imgs: List of raw HxWxC BGR uint8 arrays.
        size: The picture size; 0 for native resolution.
        session: An onnxruntime.InferenceSession from get_session().
        ts: One timestamp per image, parallel to imgs.

    Returns:
        One flat list of raw box dicts across every image, in order.
    """
    if not imgs:
        return []
    scales = [ get_image_resize_scale( img, size ) for img in imgs ]
    prepped = [ prep_img_for_nn( img, size, scale ) for img, scale in zip( imgs, scales ) ]
    per_image = detect_raw_boxes( prepped, session, scales, ts, [ size ] * len( imgs ) )
    return [ raw_box for frame_raw_boxes in per_image for raw_box in frame_raw_boxes ]
