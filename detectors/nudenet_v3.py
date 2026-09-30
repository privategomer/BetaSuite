"""
detectors/nudenet_v3.py - Adapter for NudeNet 3.4's YOLOv8-family ONNX
exports.

Structurally unlike retinanet_v2's RetinaNet graph: there is no NMS
baked in. Verified by introspecting the actual .onnx files rather than
assuming general YOLOv8 knowledge applies:

    input  'images'   NCHW [batch, 3, size, size]
    output 'output0'  [batch, 22, N]
                      rows 0..3   box as cx, cy, w, h in model-input px
                      rows 4..21  18 raw class scores, already sigmoid
                                  activated, NOT thresholded or deduped

This adapter does its own thresholding and NMS, from the backend's
candidate_floor / nms_iou / nms_mode settings.

Preprocessing (pad to square bottom-right, then resize+normalise+BGR->RGB
via cv2.dnn.blobFromImage) and the postprocessing coordinate math were
checked against NudeNet's own reference implementation
(nudenet.nudenet.NudeDetector), read as a reference only - it is not a
runtime dependency. Re-check that source if real-footage results ever
look wrong: a subtle pre/post-processing mismatch here would silently
mislocate or mislabel every detection rather than raise.

MODEL VARIANTS
--------------
Two exports ship with BetaSuite, selected by
detector_backend['nudenet_v3']['model_variant'] (or the
BETASUITE_DETECTOR_VARIANT_OVERRIDE environment variable):

    '320n'  v3.4-320n.onnx   12MB   native 320x320   fast, the default
    '640m'  v3.4-640m.onnx  103MB   native 640x640   slower, more capable

Each variant declares its native size, and native_picture_sizes() hands
that to betautils_detector.get_picture_sizes() as the default. Running a
variant at a size it was not exported at is allowed - the graph's input
shape is dynamic - but it is off-spec and this module logs a warning
once per size, because it is not a neutral knob: object scale relative
to the model's receptive field changes, which changes box geometry,
which changes every IoU-based suppression threshold derived from that
output.

CLASS MAPPING
-------------
NudeNet's 18 native labels map onto betaconst.classes mostly 1:1, with
two canonical labels added for real distinctions the RetinaNet-era
vocabulary lacked (ARMPITS_COVERED, ANUS_COVERED) and one judgment call
(MALE_BREAST_EXPOSED -> exposed_chest). See CONFIG_REFERENCE.md.
"""

import os

import cv2
import numpy as np

import betaconfig
import betaconst
import betautils_detector as bu_detector
import betautils_log as bu_log


# This model family's class-index order, exactly as baked into each
# .onnx file's own metadata_props['names']. Positional against the real
# output tensor - do not reorder.
_NATIVE_LABELS = [
    'FEMALE_GENITALIA_COVERED',
    'FACE_FEMALE',
    'BUTTOCKS_EXPOSED',
    'FEMALE_BREAST_EXPOSED',
    'FEMALE_GENITALIA_EXPOSED',
    'MALE_BREAST_EXPOSED',
    'ANUS_EXPOSED',
    'FEET_EXPOSED',
    'BELLY_COVERED',
    'FEET_COVERED',
    'ARMPITS_COVERED',
    'ARMPITS_EXPOSED',
    'FACE_MALE',
    'BELLY_EXPOSED',
    'MALE_GENITALIA_EXPOSED',
    'ANUS_COVERED',
    'FEMALE_BREAST_COVERED',
    'BUTTOCKS_COVERED',
]

# Native NudeNet label -> BetaSuite canonical label (betaconst.classes).
_CLASS_MAP = {
    'FEMALE_GENITALIA_COVERED': 'covered_vulva',
    'FACE_FEMALE':              'face_femme',
    'BUTTOCKS_EXPOSED':         'exposed_buttocks',
    'FEMALE_BREAST_EXPOSED':    'exposed_breast',
    'FEMALE_GENITALIA_EXPOSED': 'exposed_vulva',
    'MALE_BREAST_EXPOSED':      'exposed_chest',
    'ANUS_EXPOSED':             'exposed_anus',
    'FEET_EXPOSED':             'exposed_feet',
    'BELLY_COVERED':            'covered_belly',
    'FEET_COVERED':             'covered_feet',
    'ARMPITS_COVERED':          'covered_armpits',
    'ARMPITS_EXPOSED':          'exposed_armpits',
    'FACE_MALE':                'face_masc',
    'BELLY_EXPOSED':            'exposed_belly',
    'MALE_GENITALIA_EXPOSED':   'exposed_penis',
    'ANUS_COVERED':             'covered_anus',
    'FEMALE_BREAST_COVERED':    'covered_breast',
    'BUTTOCKS_COVERED':         'covered_buttocks',
}

# Canonical label per native class index, precomputed once.
_LABEL_BY_INDEX = [ _CLASS_MAP.get( native ) for native in _NATIVE_LABELS ]

# Shipped model variants. native_size is what the export was validated
# at, and becomes the default picture size for that variant.
VARIANTS = {
    '320n': { 'model_path': '../resources/model/v3.4-320n.onnx', 'native_size': 320 },
    '640m': { 'model_path': '../resources/model/v3.4-640m.onnx', 'native_size': 640 },
}

DEFAULT_VARIANT = '320n'
DEFAULT_CANDIDATE_FLOOR = 0.2
DEFAULT_NMS_IOU = 0.45
DEFAULT_NMS_MODE = 'per_class'

VALID_NMS_MODES = ( 'per_class', 'agnostic' )

# Sizes already warned about, so an off-spec size logs once per run
# rather than once per frame.
_warned_off_spec_sizes = set()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _resolve_variant( config ):
    """
    The active variant name and its descriptor.

    Precedence, most to least authoritative:
      1. BETASUITE_DETECTOR_VARIANT_OVERRIDE, so a sweep can flip
         variants without ever writing betaconfig.py
      2. the config block PASSED IN, so a caller analysing a
         hypothetical configuration gets that one rather than whatever
         betaconfig currently says
      3. detector_backend['defaults']['model_variant']
      4. this module's DEFAULT_VARIANT

    Args:
        config: This backend's detector_backend block.

    Returns:
        A (variant_name, variant_dict) pair.

    Raises:
        ValueError: the resolved variant is not one this build ships.
    """
    name = ( os.environ.get( bu_detector._VARIANT_OVERRIDE_ENV_VAR )
             or config.get( 'model_variant' )
             or bu_detector.get_backend_defaults().get( 'model_variant' )
             or DEFAULT_VARIANT )
    if name not in VARIANTS:
        raise ValueError(
            "detector_backend['nudenet_v3']['model_variant'] = %r is not a known "
            "variant - valid choices: %s"%(name, ', '.join( sorted( VARIANTS ) )) )
    return name, VARIANTS[name]


def resolve_config( config=None ):
    """
    This adapter's fully-resolved settings, with defaults filled in.

    Args:
        config: The raw detector_backend['nudenet_v3'] block. Read from
            betaconfig when omitted.

    Returns:
        A dict with 'model_variant', 'model_path', 'native_size',
        'candidate_floor', 'nms_iou' and 'nms_mode' always present.
    """
    if config is None:
        config = bu_detector.get_backend_config( 'nudenet_v3' )
    variant_name, variant = _resolve_variant( config )
    return {
        'model_variant':   variant_name,
        # An explicit model_path always wins, so an unshipped export can
        # be pointed at without editing this module.
        'model_path':      config.get( 'model_path', variant['model_path'] ),
        'native_size':     config.get( 'native_size', variant['native_size'] ),
        'candidate_floor': config.get( 'candidate_floor', DEFAULT_CANDIDATE_FLOOR ),
        'nms_iou':         config.get( 'nms_iou', DEFAULT_NMS_IOU ),
        'nms_mode':        config.get( 'nms_mode', DEFAULT_NMS_MODE ),
    }


def native_picture_sizes( config ):
    """
    Default picture_sizes for the active variant - its native size.

    See betautils_detector.get_picture_sizes for why this is the default
    rather than the shared betaconfig.picture_sizes value.
    """
    try:
        return [ resolve_config( config )['native_size'] ]
    except ValueError:
        return []


def detection_identity( config ):
    """
    The settings that change which raw detections come out of this model.

    Folded into the detection cache key. Before 2.1, changing nms_iou or
    candidate_floor silently reused detections computed under the old
    value; this is what fixes that.
    """
    resolved = resolve_config( config )
    return {
        'model_variant':   resolved['model_variant'],
        'model_path':      resolved['model_path'],
        'candidate_floor': round( float( resolved['candidate_floor'] ), 6 ),
        'nms_iou':         round( float( resolved['nms_iou'] ), 6 ),
        'nms_mode':        resolved['nms_mode'],
    }


def validate_backend_config( config, errors ):
    """
    Sanity-check detector_backend['nudenet_v3'], appending to `errors`.

    Called by betautils_config._validate_detector_backend for whichever
    backend is selected.
    """
    try:
        resolved = resolve_config( config )
    except ValueError as err:
        errors.append( str( err ) )
        return

    candidate_floor = resolved['candidate_floor']
    if not isinstance( candidate_floor, (int, float) ) or not ( 0 <= candidate_floor <= 1 ):
        errors.append( "detector_backend['nudenet_v3']['candidate_floor'] must be between "
                       "0 and 1, got %r"%(candidate_floor,) )

    nms_iou = resolved['nms_iou']
    if not isinstance( nms_iou, (int, float) ) or not ( 0 <= nms_iou <= 1 ):
        errors.append( "detector_backend['nudenet_v3']['nms_iou'] must be between 0 and 1, "
                       "got %r"%(nms_iou,) )

    if resolved['nms_mode'] not in VALID_NMS_MODES:
        errors.append( "detector_backend['nudenet_v3']['nms_mode'] must be one of %s, got %r"%(
            list( VALID_NMS_MODES ), resolved['nms_mode'] ) )

    for size in bu_detector.get_picture_sizes( 'nudenet_v3' ):
        if not isinstance( size, int ) or size <= 0:
            errors.append( "nudenet_v3 requires positive picture sizes (got %r) - size=0 "
                           "('native resolution, no resize') is a retinanet_v2-only concept; "
                           "this adapter needs a fixed square blob"%(size,) )
        elif size % 32:
            errors.append( "nudenet_v3 picture size %d is not a multiple of 32 - this model "
                           "family's detection strides are 8/16/32, so a size that is not a "
                           "multiple of 32 silently changes the anchor grid"%(size,) )


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

def get_session():
    """
    Load the configured variant into an ONNX Runtime session.

    Provider selection and the "CUDA requested but not active" warning
    are shared with every adapter - see
    betautils_detector.build_onnx_session.
    """
    resolved = resolve_config()
    return bu_detector.build_onnx_session(
        resolved['model_path'], backend_name='nudenet_v3' )


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _prep_image( raw_img, size ):
    """
    Pad a raw image to a square and convert it into this model's blob.

    Padding is bottom/right only, so the top-left origin is unchanged
    and postprocessing needs a scale factor but no offset.

    Args:
        raw_img: HxWxC numpy array, BGR uint8, unprocessed.
        size: Target square input size. Must be > 0.

    Returns:
        A (blob, pad_w, pad_h, max_size) tuple. blob is
        (1, 3, size, size) float32; max_size is max(original h, w).

    Raises:
        ValueError: size is not positive.
    """
    if not size or size <= 0:
        raise ValueError(
            "nudenet_v3 requires a positive picture size (got %r) - size=0 "
            "('native resolution') is not supported by this adapter"%(size,) )

    orig_h, orig_w = raw_img.shape[:2]
    max_size = max( orig_h, orig_w )
    pad_w = max_size - orig_w
    pad_h = max_size - orig_h

    padded = cv2.copyMakeBorder( raw_img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0 )
    blob = cv2.dnn.blobFromImage( padded, 1/255.0, ( size, size ), ( 0, 0, 0 ),
                                  swapRB=True, crop=False )
    return blob, pad_w, pad_h, max_size


def _warn_if_off_spec( size, native_size ):
    """Log once per size when running a variant away from its native size."""
    if size == native_size or size in _warned_off_spec_sizes:
        return
    _warned_off_spec_sizes.add( size )
    bu_log.get_logger().warning(
        "nudenet_v3 is running at size %d but the selected variant was exported and "
        "validated at %d. The graph accepts it, but object scale relative to the model's "
        "receptive field changes, which changes box geometry - any IoU-based suppression "
        "threshold derived from this output is only valid at this size."%(size, native_size) )


# ---------------------------------------------------------------------------
# Postprocessing
# ---------------------------------------------------------------------------

def _run_nms( boxes, scores, class_indices, candidate_floor, nms_iou, nms_mode ):
    """
    Non-maximum suppression over one image's candidate detections.

    nms_mode decides what "overlapping" means across labels:

      'per_class' (default, and BetaSuite's recommendation)
          Detections only suppress others of the SAME label. Two
          different labels overlapping heavily - an exposed_breast
          inside an exposed_belly, an exposed_vulva under a face_femme
          misfire - both survive, and the decision about which is real
          is left to class_suppression, which is the layer that has
          per-pair min_iou and margin thresholds to make it with.

      'agnostic'
          NudeNet's own reference behaviour: the highest-scoring
          detection suppresses any overlapping box regardless of label.
          Faithful to upstream, but it deletes exactly the high-IoU
          cross-label pairs class_suppression exists to arbitrate, and
          it does so with one global threshold and no margin logic. It
          is also the most likely mechanical explanation for this
          backend's cross-label IoU distributions measuring an order of
          magnitude lower than retinanet_v2's on identical footage: the
          overlapping pairs were removed before anything could measure
          them.

    Args:
        boxes: (K, 4) float array of [x, y, w, h] in original-image px.
        scores: (K,) float array.
        class_indices: (K,) int array of native class indices.
        candidate_floor: Score threshold handed to OpenCV's NMS.
        nms_iou: IoU threshold above which the lower score is dropped.
        nms_mode: 'per_class' or 'agnostic'.

    Returns:
        A list of surviving indices into boxes/scores/class_indices.
    """
    if len( boxes ) == 0:
        return []

    box_list = boxes.tolist()
    score_list = scores.tolist()

    def _nms_over( indices ):
        if not indices:
            return []
        subset_boxes = [ box_list[i] for i in indices ]
        subset_scores = [ score_list[i] for i in indices ]
        kept = cv2.dnn.NMSBoxes( subset_boxes, subset_scores,
                                 float( candidate_floor ), float( nms_iou ) )
        # cv2.dnn.NMSBoxes returns a flat array on recent OpenCV builds
        # and an (n,1) column vector on some older ones. Never assume.
        kept = np.asarray( kept ).reshape( -1 ).astype( int ).tolist()
        return [ indices[k] for k in kept ]

    if nms_mode == 'agnostic':
        return sorted( _nms_over( list( range( len( box_list ) ) ) ) )

    survivors = []
    for class_index in np.unique( class_indices ):
        member_indices = np.nonzero( class_indices == class_index )[0].tolist()
        survivors.extend( _nms_over( member_indices ) )
    return sorted( survivors )


def decode_output( raw_output, pad_w, pad_h, max_size, size,
                   candidate_floor, nms_iou, nms_mode=DEFAULT_NMS_MODE ):
    """
    Turn one image's raw (22, N) model output into surviving detections.

    Fully vectorised. The pre-2.1 implementation iterated every anchor in
    Python, calling np.amax and np.argmax on an 18-element slice each
    time; at a 1280 blob that is 33,600 iterations and roughly 97ms per
    frame, which was the single largest cost in the detection pass. The
    array form below is the same arithmetic, ~180x faster, and is
    covered by a test that asserts it matches the scalar reference
    implementation exactly.

    Args:
        raw_output: (22, N) float array for ONE image, NOT transposed.
            Row-major over anchors is a strided view and slower to read.
        pad_w, pad_h, max_size: Padding metadata from _prep_image.
        size: The square size the blob was resized to.
        candidate_floor: Per-anchor confidence floor applied before NMS.
        nms_iou: NMS IoU threshold.
        nms_mode: See _run_nms.

    Returns:
        A list of (x, y, w, h, score, native_class_index) tuples in
        original-image pixel coordinates, top-left origin.
    """
    raw_output = np.asarray( raw_output )
    if raw_output.ndim != 2 or raw_output.shape[0] < 5:
        raise ValueError( "nudenet_v3 expected a (4+classes, anchors) output, got shape %r"%(
            (raw_output.shape,) ) )

    class_scores = raw_output[4:, :]                       # (C, N)
    best_scores = class_scores.max( axis=0 )               # (N,)
    candidates = np.nonzero( best_scores >= candidate_floor )[0]
    if candidates.size == 0:
        return []

    scores = best_scores[candidates].astype( np.float32 )
    class_indices = class_scores[:, candidates].argmax( axis=0 ).astype( np.int32 )

    # model-input space (0..size) -> padded space (0..max_size) -> original
    # image space. Padding was bottom/right only, so this is a pure scale.
    scale = max_size / float( size )
    centre_x = raw_output[0, candidates].astype( np.float64 ) * scale
    centre_y = raw_output[1, candidates].astype( np.float64 ) * scale
    width    = raw_output[2, candidates].astype( np.float64 ) * scale
    height   = raw_output[3, candidates].astype( np.float64 ) * scale

    orig_w = max_size - pad_w
    orig_h = max_size - pad_h

    x = np.clip( centre_x - width / 2.0,  0.0, orig_w )
    y = np.clip( centre_y - height / 2.0, 0.0, orig_h )
    w = np.minimum( width,  orig_w - x )
    h = np.minimum( height, orig_h - y )

    # A box clipped to zero extent cannot censor anything; dropping it
    # here keeps degenerate entries out of NMS and out of the cache.
    usable = np.nonzero( ( w > 0 ) & ( h > 0 ) )[0]
    if usable.size == 0:
        return []

    boxes = np.stack( [ x[usable], y[usable], w[usable], h[usable] ], axis=1 )
    scores = scores[usable]
    class_indices = class_indices[usable]

    survivors = _run_nms( boxes, scores, class_indices, candidate_floor, nms_iou, nms_mode )
    return [ ( float( boxes[i][0] ), float( boxes[i][1] ),
               float( boxes[i][2] ), float( boxes[i][3] ),
               float( scores[i] ), int( class_indices[i] ) )
             for i in survivors ]


def _raw_boxes_from_decoded( decoded, t, size, min_prob_floor ):
    """
    Turn decode_output's tuples into this adapter's raw box dicts.

    global_min_prob is a SECOND, separate floor from candidate_floor:
    candidate_floor decides what NMS even considers, global_min_prob is
    the shared final gate every adapter applies. A detection can clear
    candidate_floor, survive NMS, and still be dropped here.

    Args:
        decoded: Output of decode_output.
        t: Timestamp for every box.
        size: The picture size that produced them. Carried on the box so
            cross-size dedup can tell which pass a detection came from.
        min_prob_floor: betaconfig.global_min_prob.

    Returns:
        A list of raw box dicts.
    """
    frame_raw_boxes = []
    for x, y, w, h, score, class_index in decoded:
        if score <= min_prob_floor:
            continue
        if class_index < 0 or class_index >= len( _LABEL_BY_INDEX ):
            continue  # corrupt/mismatched model file; skip, do not crash the run
        canonical_label = _LABEL_BY_INDEX[ class_index ]
        if canonical_label is None:
            continue  # _NATIVE_LABELS and _CLASS_MAP drifted apart
        frame_raw_boxes.append( {
            'x': x, 'y': y, 'w': w, 'h': h,
            'class_id': canonical_label,
            'score': score,
            't': t,
            'size': size,
        } )
    return frame_raw_boxes


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
        size: Target square input size.
        session: An onnxruntime.InferenceSession from get_session().
        ts: One timestamp per image, parallel to imgs.

    Returns:
        One flat list of raw box dicts across every image.
    """
    if not imgs:
        return []

    resolved = resolve_config()
    candidate_floor = resolved['candidate_floor']
    nms_iou = resolved['nms_iou']
    nms_mode = resolved['nms_mode']
    min_prob_floor = getattr( betaconfig, 'global_min_prob', betaconst.global_min_prob )

    _warn_if_off_spec( size, resolved['native_size'] )

    prepped = [ _prep_image( img, size ) for img in imgs ]
    input_name = session.get_inputs()[0].name
    batch_size = max( 1, bu_detector.get_nn_batch_size( 'nudenet_v3' ) )

    all_raw_boxes = []
    for batch_start in range( 0, len( prepped ), batch_size ):
        batch = prepped[ batch_start : batch_start + batch_size ]
        batch_blob = np.vstack( [ entry[0] for entry in batch ] )
        output = session.run( None, { input_name: batch_blob } )[0]  # (batch, 22, N)
        for offset, ( _blob, pad_w, pad_h, max_size ) in enumerate( batch ):
            decoded = decode_output( output[offset], pad_w, pad_h, max_size, size,
                                     candidate_floor, nms_iou, nms_mode )
            all_raw_boxes.extend( _raw_boxes_from_decoded(
                decoded, ts[ batch_start + offset ], size, min_prob_floor ) )

    return all_raw_boxes
