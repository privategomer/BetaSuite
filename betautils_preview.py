"""
betautils_preview.py - censor looks drawn onto the figure art, or any
image, by the real censor code.

A preview is what a photo with these detections would come out as: the
boxes stand in for detections and go through the same steps as a photo
(betautils_still.still_boxes, then censor_img_for_boxes), and the same
steps as the first frame of a video track. So a preview cannot drift from
the output.

The figure art and its boxes come from betautils_art.

Rendered previews are cached as PNGs under the cache folder, named by a
key over everything that decides their bytes, and the folder is kept
under a size cap, oldest first.
"""

import hashlib
import json
import os
import random

import cv2

import betaconfig
import betautils_art as bu_art
import betautils_cache_paths as bu_cache
import betautils_censor as bu_censor
import betautils_config as bu_config
import betautils_detector as bu_detector
import betautils_still as bu_still

KEY_VERSION = 1
CACHE_MAX_BYTES = 64 * 1024 * 1024     # the folder is trimmed back to 3/4 of this
CACHE_KEEP_FRACTION = 0.75

_CENSOR_SOURCE = None


class BadImage( ValueError ):
    """The image could not be read."""


# --- rendering ------------------------------------------------------------------

def raw_detections( boxes, width, height ):
    """Authored boxes as the raw detections a detector adapter would return."""
    return [ { 'x': int( box['x'] ), 'y': int( box['y'] ), 'w': int( box['w'] ), 'h': int( box['h'] ),
               'class_id': box['label'], 'score': 1.0, 't': 0.0, 'size': max( width, height ) }
             for box in boxes ]


def parts_for( labels, backend_name=None ):
    """The per-label settings table for these labels, resolved as a run resolves them."""
    return { label: bu_config.get_label_settings( label, backend_name ) for label in labels }


def default_labels():
    """The labels a run censors: items_to_censor."""
    return list( betaconfig.items_to_censor )


def read_image( path ):
    image = cv2.imread( path, cv2.IMREAD_COLOR )
    if image is None:
        raise BadImage( path )
    return image


def _censor_source_hash():
    global _CENSOR_SOURCE
    if _CENSOR_SOURCE is None:
        with open( bu_censor.__file__.replace( '.pyc', '.py' ), 'rb' ) as handle:
            _CENSOR_SOURCE = hashlib.sha256( handle.read() ).hexdigest()
    return _CENSOR_SOURCE


def _box_identity( box ):
    keys = ( 'label', 'x', 'y', 'w', 'h', 'censor_style', 'censor_shape', 'censor_sticker_seed' )
    return { key: box.get( key ) for key in keys }


def preview_key( image, boxes, backend_name=None ):
    """
    The cache key of a preview: the image's pixels, the resolved boxes,
    every render setting (the censor key's identity) and the censor code.
    """
    backend_name = backend_name or bu_detector.selected_backend_name()
    identity = {
        'v': KEY_VERSION,
        'image': hashlib.sha256( image.tobytes() ).hexdigest(),
        'shape': list( image.shape ),
        'boxes': [ _box_identity( box ) for box in boxes ],
        'render': bu_cache.censor_identity( backend_name ),
        'code': _censor_source_hash(),
    }
    text = json.dumps( identity, sort_keys=True, default=repr )
    return hashlib.sha256( text.encode( 'utf-8' ) ).hexdigest()


def resolve( image, raw, labels, seed, backend_name=None ):
    """The censorable boxes for these detections, styles resolved with a seeded generator."""
    height, width = image.shape[:2]
    return bu_still.still_boxes( [ raw ], width, height, parts_for( labels, backend_name ),
                                 backend_name, random.Random( seed ) )


def render( image, boxes ):
    """The censored image. The input is left as it is."""
    return bu_censor.censor_img_for_boxes( image.copy(), boxes )


def preview( image, raw, labels, seed=0, backend_name=None, cache_dir=None ):
    """
    Render (or find in the cache) one preview.

    Returns:
        { 'path', 'key', 'cached', 'width', 'height' }
    """
    boxes = resolve( image, raw, labels, seed, backend_name )
    key = preview_key( image, boxes, backend_name )
    path = bu_cache.look_preview_path( key, cache_dir )
    height, width = image.shape[:2]
    if os.path.exists( path ):
        os.utime( path )
        return { 'path': path, 'key': key, 'cached': True, 'width': width, 'height': height }
    out = render( image, boxes )
    ok, encoded = cv2.imencode( '.png', out )
    if not ok:
        raise BadImage( path )
    tmp = bu_cache.temp_sibling( path )
    with open( tmp, 'wb' ) as handle:
        handle.write( encoded.tobytes() )
    os.replace( tmp, path )
    trim_cache( os.path.dirname( path ) )
    return { 'path': path, 'key': key, 'cached': False, 'width': width, 'height': height }


def trim_cache( folder, max_bytes=CACHE_MAX_BYTES ):
    """Delete the oldest previews until the folder is under max_bytes * CACHE_KEEP_FRACTION."""
    entries = []
    total = 0
    for name in os.listdir( folder ):
        if not bu_cache.is_look_preview_name( name ):
            continue
        full = os.path.join( folder, name )
        try:
            stat = os.stat( full )
        except FileNotFoundError:
            continue
        entries.append( ( stat.st_mtime_ns, name, stat.st_size, full ) )
        total += stat.st_size
    if total <= max_bytes:
        return 0
    removed = 0
    target = max_bytes * CACHE_KEEP_FRACTION
    for _mtime, _name, size, full in sorted( entries ):
        if total <= target:
            break
        try:
            os.remove( full )
        except FileNotFoundError:
            pass
        total -= size
        removed += 1
    return removed



# The art, re-exported for callers of this module.
art_styles = bu_art.art_styles
art_item = bu_art.art_item
UnknownArt = bu_art.UnknownArt
