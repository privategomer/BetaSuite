"""
segmenters/yolo11_seg.py - Ultralytics YOLO11-seg person masks, run as ONNX.

Reverse mode's whole-body windows need people, not parts, and none of
the nudity detectors has a person class. YOLO11-seg is a COCO instance
segmentation model: class 0 is 'person', and every detection carries a
mask. Ultralytics publishes it under AGPL-3.0, the same license as
BetaSuite.

It runs through onnxruntime like every detector backend, so the runtime
environment gains no PyTorch. tools/setup/fetch_models.py exports the
.onnx files once, in a throwaway environment, from Ultralytics' pinned
.pt releases.

MODEL OUTPUTS (exported with dynamic batch and size)
----------------------------------------------------
  output0: (B, 4 + 80 + 32, N)   per candidate: box centre x, centre y,
           width, height in input pixels; 80 class scores (sigmoid
           already applied in the exported head); 32 mask coefficients
  output1: (B, 32, H/4, W/4)     mask prototypes

The input is NOT padded to a square. The longer side is scaled to
picture_size and the shorter one padded only up to the next multiple of
32 (the model's largest stride), so a 16:9 frame runs at 640x384: about
40% less work than 640x640, for the same people.

A candidate's mask is sigmoid(coefficients . prototypes), cropped to its
box. Only the person class is decoded.

Analogy: the prototypes are 32 stencils shared by the whole frame, and
each person is a recipe saying how much of each stencil to mix.
"""

import os

import cv2
import numpy as np

import betautils_detector as bu_detector

VARIANTS = {
    'n': { 'model_path': '../resources/model/yolo11n-seg.onnx', 'native_size': 640 },
    's': { 'model_path': '../resources/model/yolo11s-seg.onnx', 'native_size': 640 },
    'm': { 'model_path': '../resources/model/yolo11m-seg.onnx', 'native_size': 640 },
}

DEFAULT_VARIANT = 's'
DEFAULT_MIN_PROB = 0.35
DEFAULT_NMS_IOU = 0.5
DEFAULT_BATCH_SIZE = 4

PERSON_CLASS = 0
NUM_CLASSES = 80
NUM_MASK_COEFFICIENTS = 32
# Ultralytics' letterbox fill. Matching it keeps the padding looking like
# what the model was trained on.
PAD_VALUE = 114

IMPLEMENTED = True


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _resolve_variant( config ):
    """(variant name, descriptor): env override, then config, then default."""
    import betautils_person as bu_person
    name = ( os.environ.get( bu_person.VARIANT_OVERRIDE_ENV_VAR )
             or config.get( 'model_variant' )
             or DEFAULT_VARIANT )
    if name not in VARIANTS:
        raise ValueError( "person_backend['yolo11_seg']['model_variant'] = %r is not a known variant "
                          "- valid choices: %s"%( name, ', '.join( sorted( VARIANTS ) ) ) )
    return name, VARIANTS[name]


def resolve_config( config ):
    """This adapter's settings with defaults filled in."""
    variant_name, variant = _resolve_variant( config )
    return {
        'model_variant': variant_name,
        'model_path':    config.get( 'model_path', variant['model_path'] ),
        'picture_size':  config.get( 'picture_size', variant['native_size'] ),
        'min_prob':      config.get( 'min_prob', DEFAULT_MIN_PROB ),
        'nms_iou':       config.get( 'nms_iou', DEFAULT_NMS_IOU ),
        'batch_size':    config.get( 'batch_size', DEFAULT_BATCH_SIZE ),
    }


def validate_config( config, errors ):
    """Append every problem with person_backend['yolo11_seg'] to errors."""
    where = "person_backend['yolo11_seg']"
    try:
        resolved = resolve_config( config )
    except ValueError as err:
        errors.append( str( err ) )
        return
    size = resolved['picture_size']
    if not isinstance( size, int ) or isinstance( size, bool ) or size <= 0 or size % 32:
        errors.append( "%s['picture_size'] must be a positive multiple of 32 (the model's largest "
                       "stride), got %r"%( where, size ) )
    for key in ( 'min_prob', 'nms_iou' ):
        value = resolved[key]
        if not isinstance( value, ( int, float ) ) or isinstance( value, bool ) or not ( 0 < value < 1 ):
            errors.append( "%s['%s'] must be a number between 0 and 1, got %r"%( where, key, value ) )
    batch = resolved['batch_size']
    if not isinstance( batch, int ) or isinstance( batch, bool ) or batch < 1:
        errors.append( "%s['batch_size'] must be a whole number >= 1, got %r"%( where, batch ) )


def identity( config ):
    """Every setting that changes which person masks come out, for cache keys."""
    resolved = resolve_config( config )
    return {
        'model_variant': resolved['model_variant'],
        'model_path':    resolved['model_path'],
        'picture_size':  int( resolved['picture_size'] ),
        'min_prob':      round( float( resolved['min_prob'] ), 6 ),
        'nms_iou':       round( float( resolved['nms_iou'] ), 6 ),
    }


def get_session( config, logger=None ):
    resolved = resolve_config( config )
    return bu_detector.build_onnx_session( resolved['model_path'], logger=logger,
                                           backend_name='person:yolo11_seg' )


# ---------------------------------------------------------------------------
# Pre- and postprocessing
# ---------------------------------------------------------------------------

STRIDE = 32


def _round_up( value, multiple ):
    return int( -( -value // multiple ) * multiple )


def prep_batch( imgs, size ):
    """
    Scale each image so its longer side is size, pad it evenly on both
    sides up to a shared multiple of 32 (Ultralytics' letterbox), stack.

    Images in one batch share an input shape: the largest any of them
    needs.

    Returns:
        (blob, placements): blob is (B, 3, H, W) float32 RGB 0..1;
        placements[i] is (scale, left, top) for image i: input pixel
        = original pixel * scale + (left, top).
    """
    scaled = []
    scales = []
    for img in imgs:
        height, width = img.shape[:2]
        scale = size / float( max( height, width ) )
        new_w = max( 1, int( round( width * scale ) ) )
        new_h = max( 1, int( round( height * scale ) ) )
        scaled.append( cv2.resize( img, ( new_w, new_h ), interpolation=cv2.INTER_LINEAR ) )
        scales.append( scale )
    in_w = _round_up( max( img.shape[1] for img in scaled ), STRIDE )
    in_h = _round_up( max( img.shape[0] for img in scaled ), STRIDE )
    blob = np.empty( ( len( scaled ), 3, in_h, in_w ), dtype=np.float32 )
    placements = []
    for index, img in enumerate( scaled ):
        pad_w, pad_h = in_w - img.shape[1], in_h - img.shape[0]
        # Ultralytics rounds the split this way, so the image lands on the
        # same pixels it would in their own predictor.
        left, top = int( round( pad_w / 2 - 0.1 ) ), int( round( pad_h / 2 - 0.1 ) )
        padded = cv2.copyMakeBorder( img, top, pad_h - top, left, pad_w - left, cv2.BORDER_CONSTANT,
                                     value=( PAD_VALUE, PAD_VALUE, PAD_VALUE ) )
        blob[index] = cv2.dnn.blobFromImage( padded, 1/255.0, swapRB=True, crop=False )[0]
        placements.append( ( scales[index], left, top ) )
    return blob, placements


def _sigmoid( values ):
    return 1.0 / ( 1.0 + np.exp( -values ) )


def decode_one( predictions, prototypes, placement, img_w, img_h, input_w, min_prob, nms_iou, t ):
    """
    People in one image.

    Args:
        predictions: (4 + 80 + 32, N) for this image.
        prototypes: (32, H/4, W/4) for this image.
        placement: (scale, left, top) from prep_batch.
        img_w, img_h: The original image size.
        input_w: The model input width (the blob's last dimension).
        min_prob, nms_iou: Person score floor and NMS overlap.
        t: The sample time, copied onto each person.

    Returns:
        A list of person dicts: 't', 'x', 'y', 'w', 'h' (original image
        pixels, clamped inside it), 'score', 'mask', a uint8 array of
        mask probability (0..255) at the prototype resolution, and
        'mask_box', the (x, y, w, h) in original pixels that mask covers
        (the box rounded out to whole prototype cells).
    """
    scale, left, top = placement
    scores = predictions[ 4 + PERSON_CLASS ]
    keep = np.flatnonzero( scores >= min_prob )
    if not keep.size:
        return []
    boxes_cxcywh = predictions[ :4, keep ].T
    kept_scores = scores[keep]
    coefficients = predictions[ 4 + NUM_CLASSES:4 + NUM_CLASSES + NUM_MASK_COEFFICIENTS, keep ].T

    x0 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    y0 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    rects = np.stack( [ x0, y0, boxes_cxcywh[:, 2], boxes_cxcywh[:, 3] ], axis=1 )
    chosen = cv2.dnn.NMSBoxes( rects.tolist(), kept_scores.tolist(), float( min_prob ), float( nms_iou ) )
    chosen = np.array( chosen ).reshape( -1 )
    if not chosen.size:
        return []

    proto_count, proto_h, proto_w = prototypes.shape
    proto_scale = proto_w / float( input_w )
    flat_prototypes = prototypes.reshape( proto_count, -1 )
    people = []
    for index in chosen:
        bx, by, bw, bh = rects[index]
        # Original-image box, clamped inside the image.
        x = max( 0.0, ( bx - left ) / scale )
        y = max( 0.0, ( by - top ) / scale )
        right = min( float( img_w ), ( bx + bw - left ) / scale )
        bottom = min( float( img_h ), ( by + bh - top ) / scale )
        if right - x < 1 or bottom - y < 1:
            continue
        # The same box in prototype cells, at least one cell.
        px0 = int( np.floor( ( x * scale + left ) * proto_scale ) )
        py0 = int( np.floor( ( y * scale + top ) * proto_scale ) )
        px1 = max( px0 + 1, int( np.ceil( ( right * scale + left ) * proto_scale ) ) )
        py1 = max( py0 + 1, int( np.ceil( ( bottom * scale + top ) * proto_scale ) ) )
        px1, py1 = min( px1, proto_w ), min( py1, proto_h )
        logits = coefficients[index] @ flat_prototypes
        full = _sigmoid( logits.reshape( proto_h, proto_w ) )
        crop = full[ py0:py1, px0:px1 ]
        # Whole cells, so the mask covers a slightly larger rectangle than
        # the box; that rectangle, in image pixels, is where it belongs.
        cell = 1.0 / ( scale * proto_scale )
        people.append( {
            't': t, 'x': x, 'y': y, 'w': right - x, 'h': bottom - y,
            'score': float( kept_scores[index] ),
            'mask': np.clip( crop * 255.0 + 0.5, 0, 255 ).astype( np.uint8 ),
            'mask_box': ( ( px0 / proto_scale - left ) / scale, ( py0 / proto_scale - top ) / scale,
                          ( px1 - px0 ) * cell, ( py1 - py0 ) * cell ),
        } )
    return people


def persons_for_imgs( imgs, session, ts, config ):
    """
    People in a batch of BGR uint8 images, one flat list.

    Images in a batch may differ in size; they share the input shape
    the largest needs.
    """
    resolved = resolve_config( config )
    size = int( resolved['picture_size'] )
    blob, placements = prep_batch( imgs, size )
    input_name = session.get_inputs()[0].name
    predictions, prototypes = session.run( None, { input_name: blob } )[:2]
    people = []
    for index, ( img, t ) in enumerate( zip( imgs, ts ) ):
        img_h, img_w = img.shape[:2]
        people.extend( decode_one( predictions[index], prototypes[index], placements[index], img_w, img_h,
                                   blob.shape[3], resolved['min_prob'], resolved['nms_iou'], t ) )
    return people
