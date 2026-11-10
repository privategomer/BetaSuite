"""
segmenters/rf_detr_seg.py - Roboflow RF-DETR-Seg person masks, run as ONNX.

The second person backend beside YOLO11-seg (segmenters/yolo11_seg.py),
to the same contract: one flat list of person dicts per batch. RF-DETR
is a COCO instance segmentation transformer. Roboflow publishes the
rfdetr code and every RF-DETR-Seg size under Apache-2.0. BetaSuite does
not redistribute the weights, and this backend does not import the
rfdetr package.

It runs through onnxruntime like every detector backend, so the runtime
environment gains no PyTorch. tools/setup/fetch_models.py exports the
.onnx files once, in the same throwaway environment as YOLO11, from
Roboflow's .pth releases pinned by sha256.

MODEL OUTPUTS (exported with a dynamic batch; the picture size is fixed)
-----------------------------------------------------------------------
  input:  (B, 3, S, S)      RGB, scaled to 0..1, then ImageNet mean and
                            std. S is baked into the export: the
                            variant's native_size below
  dets:   (B, Q, 4)         per query: box centre x, centre y, width,
                            height, as fractions (0..1) of the input
  labels: (B, Q, 91)        per query: class LOGITS (no sigmoid in the
                            graph), sparse COCO ids, so 'person' is
                            column 1, not 0
  masks:  (B, Q, S/4, S/4)  per query: mask LOGITS over the whole input

Q is the variant's fixed query count (100 or 200). There are no
prototypes: each query carries its own low-resolution mask.

The model only runs at its square size, and Roboflow trains and predicts
by stretching the whole frame to that square, without a letterbox. This
adapter stretches too. Measured on real 16:9 footage, a letterbox found
fewer people and drew worse masks at every size: it leaves 44% of the
square as padding and gives each person fewer pixels.

DETR-style models are trained to give each object one query, so there is
no NMS: a person is any query whose person score clears min_prob. Near
the floor two queries can nest one box inside the other; that is left
alone, because merging could also swallow a second, overlapping person,
and every outline that is not a man's has to stay to be cut out of his.
A person's mask is sigmoid(mask logits), cropped to their box.

Analogy: YOLO shouts a few thousand guesses and NMS shushes the
duplicates; RF-DETR has a fixed panel of Q seats and each person gets
one seat.
"""

import os

import cv2
import numpy as np

import betautils_detector as bu_detector

# native_size: the square picture size the export is fixed at, Roboflow's
# own resolution for that size. Not a setting: it is baked into the .onnx
# file, and persons_for_imgs reads it from there.
VARIANTS = {
    'nano':   { 'model_path': '../resources/model/rf-detr-seg-nano.onnx',   'native_size': 312 },
    'small':  { 'model_path': '../resources/model/rf-detr-seg-small.onnx',  'native_size': 384 },
    'medium': { 'model_path': '../resources/model/rf-detr-seg-medium.onnx', 'native_size': 432 },
}
# Roboflow also publishes large, xlarge and 2xlarge (504, 624, 768 px), all
# Apache-2.0. Left out: on real footage large found no more people than
# small, with about the same outline quality, at 1.9x small's time on a GPU,
# and every size here is another ~260 MB that setup downloads and exports.

DEFAULT_VARIANT = 'small'
DEFAULT_MIN_PROB = 0.35
DEFAULT_BATCH_SIZE = 4

# Column of 'person' in the labels output: the export keeps COCO's sparse
# category ids, where 0 is unused and 1 is person.
PERSON_CLASS = 1
# ImageNet statistics, RGB, as Roboflow's own preprocessing uses them.
MEAN = np.array( [ 0.485, 0.456, 0.406 ], dtype=np.float32 )
STD = np.array( [ 0.229, 0.224, 0.225 ], dtype=np.float32 )

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
        raise ValueError( "person_backend['rf_detr_seg']['model_variant'] = %r is not a known variant "
                          "- valid choices: %s"%( name, ', '.join( sorted( VARIANTS ) ) ) )
    return name, VARIANTS[name]


def resolve_config( config ):
    """This adapter's settings with defaults filled in."""
    variant_name, variant = _resolve_variant( config )
    return {
        'model_variant': variant_name,
        'model_path':    config.get( 'model_path', variant['model_path'] ),
        'min_prob':      config.get( 'min_prob', DEFAULT_MIN_PROB ),
        'batch_size':    config.get( 'batch_size', DEFAULT_BATCH_SIZE ),
    }


def validate_config( config, errors ):
    """Append every problem with person_backend['rf_detr_seg'] to errors."""
    where = "person_backend['rf_detr_seg']"
    try:
        resolved = resolve_config( config )
    except ValueError as err:
        errors.append( str( err ) )
        return
    if 'picture_size' in config:
        errors.append( "%s['picture_size'] is not a setting: RF-DETR's picture size is fixed in the "
                       ".onnx file when it is exported (pick a model_variant instead)"%( where, ) )
    value = resolved['min_prob']
    if not isinstance( value, ( int, float ) ) or isinstance( value, bool ) or not ( 0 < value < 1 ):
        errors.append( "%s['min_prob'] must be a number between 0 and 1, got %r"%( where, value ) )
    batch = resolved['batch_size']
    if not isinstance( batch, int ) or isinstance( batch, bool ) or batch < 1:
        errors.append( "%s['batch_size'] must be a whole number >= 1, got %r"%( where, batch ) )


def identity( config ):
    """
    Every setting that changes which person masks come out, for cache keys.

    No picture size: it is the model file's own, so model_variant and
    model_path already pin it.
    """
    resolved = resolve_config( config )
    return {
        'model_variant': resolved['model_variant'],
        'model_path':    resolved['model_path'],
        'min_prob':      round( float( resolved['min_prob'] ), 6 ),
    }


def get_session( config, logger=None ):
    resolved = resolve_config( config )
    return bu_detector.build_onnx_session( resolved['model_path'], logger=logger,
                                           backend_name='person:rf_detr_seg' )


# ---------------------------------------------------------------------------
# Pre- and postprocessing
# ---------------------------------------------------------------------------

# (rgb / 255 - mean) / std as one multiply and one subtract per pixel.
_GAIN = ( 1.0 / ( 255.0 * STD ) ).astype( np.float32 )
_OFFSET = ( MEAN / STD ).astype( np.float32 )


def prep_batch( imgs, size ):
    """
    Stretch each image to the model's size x size square (each axis
    scaled on its own, Roboflow's convention), normalise, stack.

    The whole frame lands on the whole square, so a position as a
    fraction of the input is the same fraction of the original image:
    there is no placement to undo, whatever the image sizes in a batch.

    Returns:
        (B, 3, size, size) float32, RGB, ImageNet-normalised.
    """
    batch = np.empty( ( len( imgs ), size, size, 3 ), dtype=np.uint8 )
    for index, img in enumerate( imgs ):
        batch[index] = cv2.resize( img, ( size, size ), interpolation=cv2.INTER_LINEAR )
    # BGR to RGB, normalise, channels first.
    rgb = batch[ ..., ::-1 ].astype( np.float32 )
    rgb *= _GAIN
    rgb -= _OFFSET
    return np.ascontiguousarray( rgb.transpose( 0, 3, 1, 2 ) )


def _sigmoid( values ):
    return 1.0 / ( 1.0 + np.exp( -values ) )


def decode_one( boxes, logits, masks, img_w, img_h, min_prob, t ):
    """
    People in one image.

    Args:
        boxes: (Q, 4) centre x, centre y, width, height as fractions of
            the input, for this image.
        logits: (Q, 91) class logits for this image.
        masks: (Q, Hm, Wm) mask logits over the whole input.
        img_w, img_h: The original image size.
        min_prob: Person score floor.
        t: The sample time, copied onto each person.

    Returns:
        A list of person dicts, best score first: 't', 'x', 'y', 'w', 'h'
        (original image pixels, clamped inside it), 'score', 'mask', a
        uint8 array of mask probability (0..255) at the mask resolution,
        cropped to the box, and 'mask_box', the (x, y, w, h) in original
        pixels that mask covers (the box rounded out to whole mask cells).
    """
    scores = _sigmoid( logits[ :, PERSON_CLASS ] )
    keep = np.flatnonzero( scores >= min_prob )
    if not keep.size:
        return []
    _queries, mask_h, mask_w = masks.shape
    # Mask cells per original pixel, per axis: the mask spans the whole
    # input, which is the whole image stretched.
    cells_x = mask_w / float( img_w )
    cells_y = mask_h / float( img_h )
    people = []
    for index in keep[ np.argsort( -scores[keep], kind='stable' ) ]:
        cx, cy, bw, bh = ( float( v ) for v in boxes[index] )
        # Fractions of the input are fractions of the original image too.
        x = max( 0.0, ( cx - bw / 2 ) * img_w )
        y = max( 0.0, ( cy - bh / 2 ) * img_h )
        right = min( float( img_w ), ( cx + bw / 2 ) * img_w )
        bottom = min( float( img_h ), ( cy + bh / 2 ) * img_h )
        if right - x < 1 or bottom - y < 1:
            continue
        # The same box in whole mask cells. The box is at least a pixel
        # wide, so it always reaches into at least one cell. The clamp is
        # for rounding: at some sizes img_w * (mask_w / img_w) lands a hair
        # past mask_w, and its ceil would claim a cell that is not there.
        mx0 = int( np.floor( x * cells_x ) )
        my0 = int( np.floor( y * cells_y ) )
        mx1 = min( mask_w, int( np.ceil( right * cells_x ) ) )
        my1 = min( mask_h, int( np.ceil( bottom * cells_y ) ) )
        crop = _sigmoid( masks[ index, my0:my1, mx0:mx1 ].astype( np.float32 ) )
        # Whole cells, so the mask covers a slightly larger rectangle than
        # the box; that rectangle, in image pixels, is where it belongs.
        people.append( {
            't': t, 'x': x, 'y': y, 'w': right - x, 'h': bottom - y,
            'score': float( scores[index] ),
            'mask': np.clip( crop * 255.0 + 0.5, 0, 255 ).astype( np.uint8 ),
            'mask_box': ( mx0 / cells_x, my0 / cells_y, ( mx1 - mx0 ) / cells_x, ( my1 - my0 ) / cells_y ),
        } )
    return people


def persons_for_imgs( imgs, session, ts, config ):
    """
    People in a batch of BGR uint8 images, one flat list.

    Images in a batch may differ in size; each is stretched to the
    model's square input on its own.
    """
    _variant_name, variant = _resolve_variant( config )
    resolved = resolve_config( config )
    model_input = session.get_inputs()[0]
    # The export fixes the size; the file is the authority.
    size = model_input.shape[3] if isinstance( model_input.shape[3], int ) else variant['native_size']
    blob = prep_batch( imgs, size )
    outputs = dict( zip( ( output.name for output in session.get_outputs() ),
                         session.run( None, { model_input.name: blob } ) ) )
    boxes, logits, masks = outputs['dets'], outputs['labels'], outputs['masks']
    people = []
    for index, ( img, t ) in enumerate( zip( imgs, ts ) ):
        img_h, img_w = img.shape[:2]
        people.extend( decode_one( boxes[index], logits[index], masks[index], img_w, img_h,
                                   resolved['min_prob'], t ) )
    return people
