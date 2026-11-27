"""
test_behavior_pins.py - the exact output of the censor drawing, hashing
and RetinaNet adapter helpers is pinned.

Each case builds a seeded synthetic input, runs one function and reduces
its output to a digest (the sha256 of an image's shape, dtype and bytes,
or of a value's JSON). tests/golden/behavior_pins.json holds the digests,
so a rewrite of any of these functions has to reproduce them byte for
byte. The detection and render cache keys depend on dictionary_hash, so
its values are pinned directly as well.

    python -m tests.test_behavior_pins write

records the file. Only do that when an output is meant to change, and
say why in CHANGELOG.
"""

import hashlib
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

import betaconfig
import betautils_censor as bu_censor
import betautils_hash as bu_hash
import betautils_render as bu_render
from detectors import retinanet_v2

GOLDEN = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), 'golden', 'behavior_pins.json' )


def _image( height=120, width=160, seed=7 ):
    return np.random.default_rng( seed ).integers( 0, 255, ( height, width, 3 ), dtype=np.uint8 )


def _digest_image( image ):
    image = np.ascontiguousarray( image )
    head = ( '%s|%s|' % ( image.shape, image.dtype ) ).encode( 'ascii' )
    return hashlib.sha256( head + image.tobytes() ).hexdigest()


def _digest_value( value ):
    return hashlib.sha256( json.dumps( value, sort_keys=True, default=str ).encode( 'utf-8' ) ).hexdigest()


def _box( x, y, w, h, style, shape='box', label='exposed_breast', seed=0.5 ):
    return { 'x': x, 'y': y, 'w': w, 'h': h, 'censor_style': style, 'censor_shape': shape,
             'censor_sticker_seed': seed, 'label': label, 'score': 0.9, 'start': 0.0, 'end': 1.0 }


STYLES = {
    'blur_gaussian': { 'type': 'blur', 'method': 'gaussian', 'strength': 20 },
    'blur_triple_box': { 'type': 'blur', 'method': 'triple_box', 'strength': 15 },
    'pixel_square': { 'type': 'pixel', 'pattern': 'square', 'strength': 12 },
    'pixel_hex': { 'type': 'pixel', 'pattern': 'hex', 'strength': 10 },
    'bar_black': { 'type': 'bar', 'color': ( 0, 0, 0 ), 'thickness': 1 },
    'debug': { 'type': 'debug', 'color': ( 0, 255, 0 ) },
}


def _cases():
    """Every pinned case: name -> zero-argument function returning a digest."""
    cases = {}

    for fw, fh in ( ( 10, 10 ), ( 40, 25 ), ( 160, 120 ), ( 1, 300 ) ):
        cases[ 'censor_scale_for_image_box/%dx%d' % ( fw, fh ) ] = (
            lambda fw=fw, fh=fh: _digest_value( bu_censor.censor_scale_for_image_box( _image(), fw, fh ) ) )

    for args in ( ( 10, 10, 30, 20, 160, 120, 0.0, 0.0 ), ( 10, 10, 30, 20, 160, 120, 0.2, 0.4 ),
                  ( 0, 0, 160, 120, 160, 120, 0.5, 0.5 ), ( 150, 110, 30, 30, 160, 120, 0.1, 0.1 ),
                  ( -5, -5, 20, 20, 160, 120, 0.0, 0.3 ) ):
        cases[ 'compute_safe_geometry/%s' % ( ','.join( str( a ) for a in args ), ) ] = (
            lambda args=args: _digest_value( bu_censor.compute_safe_geometry( *args ) ) )

    for factor in ( 2, 5, 13 ):
        cases[ 'pixelate_block_grid/%d' % factor ] = (
            lambda factor=factor: _digest_image( bu_censor.pixelate_block_grid( _image( 37, 53 ).copy(), 53, 37, factor ) ) )

    for shape in ( 'box', 'circle', 'ellipse' ):
        for feather in ( 0, 4 ):
            def masked( shape=shape, feather=feather ):
                image = _image()
                region = 255 - image[ 20:70, 30:110 ].copy()
                return _digest_image( bu_censor.apply_masked_region( image, 30, 20, 80, 50, region, shape, feather ) )
            cases[ 'apply_masked_region/%s/%d' % ( shape, feather ) ] = masked

    cases[ 'debug_image' ] = lambda: _digest_image(
        bu_censor.debug_image( _image(), _box( 20, 30, 60, 40, STYLES[ 'debug' ] ) ) )
    cases[ 'annotate_image_shape' ] = lambda: _digest_image( bu_censor.annotate_image_shape( _image() ) )
    for enabled in ( False, True ):
        def watermark( enabled=enabled ):
            with mock.patch.object( betaconfig, 'enable_betasuite_watermark', enabled, create=True ):
                return _digest_image( bu_censor.watermark_image( _image( 240, 320 ) ) )
        cases[ 'watermark_image/%s' % enabled ] = watermark

    for name, style in STYLES.items():
        for merge in ( 'none', 'single-pass' ):
            def render( style=style, merge=merge ):
                styled = dict( style, merge=merge )
                boxes = [ _box( 10, 10, 50, 40, styled ), _box( 40, 30, 50, 40, styled, shape='ellipse' ),
                          _box( 120, 80, 30, 30, styled, shape='circle', label='exposed_vulva' ) ]
                return _digest_image( bu_censor.censor_img_for_boxes( _image(), boxes ) )
            cases[ 'censor_img_for_boxes/%s/%s' % ( name, merge ) ] = render

    def windows():
        styled = dict( STYLES[ 'blur_gaussian' ], merge='single-pass' )
        boxes = [ _box( 10, 10, 50, 40, styled ), _box( 40, 30, 50, 40, styled ),
                  _box( 120, 80, 30, 30, dict( STYLES[ 'bar_black' ], merge='none' ), label='exposed_vulva' ) ]
        return _digest_value( bu_censor._censored_windows( boxes, 160, 120 ) )
    cases[ '_censored_windows' ] = windows

    for value, length in ( ( { 'b': 2, 'a': [ 1, 2.5, None ] }, 16 ), ( [ 'x', { 'y': True } ], 32 ),
                           ( 'plain', 8 ), ( { 'nested': { 'z': 1, 'y': [ 0.1 ] } }, 64 ) ):
        cases[ 'dictionary_hash/%s/%d' % ( _digest_value( value )[ :8 ], length ) ] = (
            lambda value=value, length=length: bu_hash.dictionary_hash( value, length ) )

    for codec, quality in ( ( None, None ), ( 'libx265', 22 ), ( 'h264_nvenc', 19 ) ):
        def command( codec=codec, quality=quality ):
            argv = bu_render._chunk_encode_command( 'out.mp4', 641, 359, 29.97, 'fast', 4,
                                                    codec=codec, quality=quality )
            return _digest_value( argv[ 1: ] )   # argv[0] is the ffmpeg path on this machine
        cases[ '_chunk_encode_command/%s/%s' % ( codec, quality ) ] = command

    for height, width, size in ( ( 120, 160, 640 ), ( 1080, 1920, 1280 ), ( 1920, 1080, 960 ), ( 50, 50, 0 ) ):
        cases[ 'retinanet.get_resize_scale/%dx%d/%d' % ( width, height, size ) ] = (
            lambda height=height, width=width, size=size: _digest_value(
                retinanet_v2.get_resize_scale( height, width, size ) ) )
    for size in ( 160, 320 ):
        def prep( size=size ):
            image = _image( 90, 140 )
            scale = retinanet_v2.get_image_resize_scale( image, size )
            return _digest_image( retinanet_v2.prep_img_for_nn( image, size, scale ) )
        cases[ 'retinanet.prep_img_for_nn/%d' % size ] = prep

    def model_output():
        rng = np.random.default_rng( 11 )
        boxes = ( rng.random( ( 2, 300, 4 ) ) * 400 ).astype( np.float32 )
        boxes[ ..., 2: ] += boxes[ ..., :2 ]
        scores = rng.random( ( 2, 300 ) ).astype( np.float32 )
        classes = rng.integers( 0, len( retinanet_v2._MODEL_CLASS_ORDER ), ( 2, 300 ) ).astype( np.int32 )
        return [ boxes, scores, classes ]
    cases[ 'retinanet.raw_boxes_from_model_output' ] = lambda: _digest_value(
        retinanet_v2.raw_boxes_from_model_output( model_output(), [ 0.5, 1.25 ], [ 1.0, 2.0 ] ) )
    cases[ 'retinanet.raw_boxes_from_model_output/sizes' ] = lambda: _digest_value(
        retinanet_v2.raw_boxes_from_model_output( model_output(), [ 0.5, 1.25 ], [ 1.0, 2.0 ], [ 320, 640 ] ) )

    class _Session:
        """Answers like the model: one [boxes, scores, classes] per call, from the batch it got."""

        def __init__( self ):
            self.calls = []

        def get_inputs( self ):
            return [ mock.Mock( name='input', **{ 'name': 'input_1:0' } ) ]

        def get_outputs( self ):
            return [ mock.Mock( **{ 'name': name } ) for name in ( 'boxes', 'scores', 'classes' ) ]

        def run( self, output_names, feed ):
            ( name, batch ), = feed.items()
            batch = np.asarray( batch )
            self.calls.append( ( list( output_names ) if output_names else None, name, list( batch.shape ) ) )
            count = batch.shape[ 0 ]
            rng = np.random.default_rng( int( batch.sum() ) % 1000 )
            boxes = ( rng.random( ( count, 300, 4 ) ) * 200 ).astype( np.float32 )
            boxes[ ..., 2: ] += boxes[ ..., :2 ]
            return [ boxes, rng.random( ( count, 300 ) ).astype( np.float32 ),
                     rng.integers( 0, len( retinanet_v2._MODEL_CLASS_ORDER ), ( count, 300 ) ).astype( np.int32 ) ]

    for batch_size in ( 1, 2, 5 ):
        def detect( batch_size=batch_size ):
            session = _Session()
            images = [ retinanet_v2.prep_img_for_nn( _image( 60, 80, seed ), 160,
                                                     retinanet_v2.get_image_resize_scale( _image( 60, 80, seed ), 160 ) )
                       for seed in range( 3 ) ]
            raw = retinanet_v2.detect_raw_boxes( images, session, [ 1.0, 0.5, 2.0 ], [ 0.0, 0.1, 0.2 ],
                                                 batch_size=batch_size )
            return _digest_value( { 'raw': raw, 'calls': session.calls } )
        cases[ 'retinanet.detect_raw_boxes/batch%d' % batch_size ] = detect

    return cases


def _record():
    return { name: case() for name, case in sorted( _cases().items() ) }


class TestRectanglesIntersect( unittest.TestCase ):
    def test_overlap_touching_and_apart( self ):
        a = { 'x': 10, 'y': 10, 'w': 20, 'h': 20 }
        def b( x, y, w=20, h=20 ):
            return { 'x': x, 'y': y, 'w': w, 'h': h }
        self.assertTrue( bu_censor.rectangles_intersect( a, b( 15, 15 ) ) )
        self.assertTrue( bu_censor.rectangles_intersect( a, b( 30, 10 ) ), "touching on the right edge" )
        self.assertTrue( bu_censor.rectangles_intersect( a, b( 10, 30 ) ), "touching on the bottom edge" )
        self.assertTrue( bu_censor.rectangles_intersect( a, b( -10, 10 ) ), "touching on the left edge" )
        self.assertTrue( bu_censor.rectangles_intersect( a, b( 10, -10 ) ), "touching on the top edge" )
        self.assertTrue( bu_censor.rectangles_intersect( a, b( 12, 12, 2, 2 ) ), "one inside the other" )
        for x, y in ( ( 31, 10 ), ( 10, 31 ), ( -11, 10 ), ( 10, -11 ), ( 31, 31 ) ):
            self.assertFalse( bu_censor.rectangles_intersect( a, b( x, y ) ), ( x, y ) )
            self.assertFalse( bu_censor.rectangles_intersect( b( x, y ), a ), ( x, y ) )


class TestJsonFiles( unittest.TestCase ):
    def test_write_then_read_round_trips_through_gzip( self ):
        value = { 'boxes': [ { 'x': 1.5, 'label': 'exposed_breast' } ], 'n': None, 'ok': True }
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join( folder, 'v.json.gz' )
            bu_hash.write_json( value, path )
            with open( path, 'rb' ) as handle:
                self.assertEqual( handle.read( 2 ), b'\x1f\x8b', "gzip" )
            self.assertEqual( bu_hash.read_json( path ), value )
            self.assertEqual( sorted( os.listdir( folder ) ), [ 'v.json.gz' ], "no temporary file left" )
            bu_hash.write_json( [ 1 ], path )
            self.assertEqual( bu_hash.read_json( path ), [ 1 ], "overwrites" )


class TestPins( unittest.TestCase ):
    maxDiff = None

    def test_every_case_matches_its_recorded_digest( self ):
        with open( GOLDEN, encoding='utf-8' ) as handle:
            golden = json.load( handle )
        got = _record()
        self.assertEqual( sorted( golden ), sorted( got ), "cases added or removed: rerun write" )
        for name in sorted( got ):
            with self.subTest( name ):
                self.assertEqual( got[ name ], golden[ name ] )


if __name__ == '__main__':
    if sys.argv[ 1: ] == [ 'write' ]:
        with open( GOLDEN, 'w', encoding='utf-8' ) as handle:
            json.dump( _record(), handle, indent=1, sort_keys=True )
            handle.write( '\n' )
        print( 'wrote %s' % GOLDEN )
    else:
        unittest.main()
