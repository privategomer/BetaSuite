"""
segmenters/rf_detr_seg.py - RF-DETR-Seg person masks. NOT IMPLEMENTED YET.

Registered so person_backend can name it, betabench can list it beside
YOLO11, and config validation can say plainly that it is not ready,
instead of the backend being unknown. Roboflow publishes the code and
every segmentation size under Apache-2.0, COCO-trained (class 'person').

To finish it:
  1. export the chosen sizes to ONNX (the rfdetr package exports) and add
     them to tools/setup/fetch_models.py
  2. write prep_batch / persons_for_imgs to the same contract as
     segmenters/yolo11_seg.py: one flat list of person dicts with 't',
     'x', 'y', 'w', 'h', 'score' and a uint8 'mask' covering the box
  3. set IMPLEMENTED = True
"""

import os

VARIANTS = {
    'nano':   { 'model_path': '../resources/model/rf-detr-seg-nano.onnx' },
    'small':  { 'model_path': '../resources/model/rf-detr-seg-small.onnx' },
    'medium': { 'model_path': '../resources/model/rf-detr-seg-medium.onnx' },
}

DEFAULT_VARIANT = 'small'
IMPLEMENTED = False

NOT_IMPLEMENTED = ( "person_backend 'rf_detr_seg' is registered but its decoder is not written yet; "
                    "select 'yolo11_seg'" )


def resolve_config( config ):
    import betautils_person as bu_person
    name = ( os.environ.get( bu_person.VARIANT_OVERRIDE_ENV_VAR )
             or config.get( 'model_variant' ) or DEFAULT_VARIANT )
    if name not in VARIANTS:
        raise ValueError( "person_backend['rf_detr_seg']['model_variant'] = %r is not a known variant "
                          "- valid choices: %s"%( name, ', '.join( sorted( VARIANTS ) ) ) )
    return { 'model_variant': name,
             'model_path': config.get( 'model_path', VARIANTS[name]['model_path'] ) }


def validate_config( config, errors ):
    errors.append( NOT_IMPLEMENTED )


def identity( config ):
    return dict( resolve_config( config ), implemented=False )


def get_session( config, logger=None ):
    raise NotImplementedError( NOT_IMPLEMENTED )


def persons_for_imgs( imgs, session, ts, config ):
    raise NotImplementedError( NOT_IMPLEMENTED )
