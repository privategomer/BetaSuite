"""
test_behavior_pins.py - the exact output of the censor rendering, hashing
and RetinaNet adapter helpers and betastare's per-file path is pinned;
debug drawing and the watermark are checked loosely.

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
import betaconst
import betautils_censor as bu_censor
import betautils_config as bu_config
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
}
DEBUG_STYLE = { 'type': 'debug', 'color': ( 0, 255, 0 ) }


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

    for sizes in ( [ 320 ], [ 320, 640 ] ):
        cases[ 'betastare._process_one_file/%s' % '-'.join( str( s ) for s in sizes ) ] = (
            lambda sizes=sizes: _stare_one_file( sizes ) )

    return cases


class _StubDetector:
    """Answers one fixed detection per size, and counts calls."""

    def __init__( self ):
        self.calls = []

    def raw_boxes_for_img( self, image, size, session, t ):
        self.calls.append( ( list( image.shape ), size, t ) )
        return [ { 'x': 20.0 + size % 7, 'y': 15.0, 'w': 60.0, 'h': 50.0, 'class_id': 'exposed_breast',
                   'score': 0.91, 't': t, 'size': size } ]


def _stare_one_file( sizes ):
    """
    betastare's whole per-file path, twice (a cold run, then one served
    from its detection cache), on a synthetic image with a stub detector:
    the output name, the output bytes as read back, the cached detections,
    the detector calls and the log lines with their timings masked.
    """
    import random
    import re
    import betastare
    random.seed( 7 )   # a look with several styles picks one at random
    with tempfile.TemporaryDirectory() as tmp:
        root, out, cache = ( os.path.join( tmp, name ) for name in ( 'in', 'out', 'cache' ) )
        for folder in ( root, out, cache ):
            os.makedirs( folder )
        import cv2
        cv2.imwrite( os.path.join( root, 'photo.png' ), _image( 90, 120, 3 ) )
        detector = _StubDetector()
        logger = mock.Mock()

        def cache_path( image_hash, size, min_prob, backend ):
            return os.path.join( cache, '%s-%s-%s-%s.json.gz' % ( image_hash, size, min_prob, backend ) )

        with mock.patch.object( betastare.bu_detector, 'get_detector', return_value=detector ), \
             mock.patch.object( betastare.bu_detector, 'get_picture_sizes', return_value=sizes ), \
             mock.patch.object( betastare.bu_cache, 'pic_hash_path_for', side_effect=cache_path ), \
             mock.patch.object( betaconfig, 'items_to_censor', [ 'exposed_breast' ] ):
            bu_config.invalidate_config_caches()
            try:
                betastare._process_one_file( root, 'photo.png', out, 1, 2, None, logger )
                first = sorted( os.listdir( out ) )
                written = cv2.imread( os.path.join( out, first[ 0 ] ), cv2.IMREAD_UNCHANGED )
                for name in first:
                    os.remove( os.path.join( out, name ) )
                betastare._process_one_file( root, 'photo.png', out, 2, 2, None, logger )
                betastare._process_one_file( root, 'photo.png', out, 2, 2, None, logger )
                cached = { name: bu_hash.read_json( os.path.join( cache, name ) ) for name in sorted( os.listdir( cache ) ) }
                with open( os.path.join( root, 'notes.txt' ), 'w' ) as handle:
                    handle.write( 'not an image' )
                betastare._process_one_file( root, 'notes.txt', out, 2, 2, None, logger )
            finally:
                bu_config.invalidate_config_caches()
        lines = [ ( call[ 0 ], re.sub( r'\d+\.\d{3}s', '#s', str( call[ 1 ][ 0 ] ) ) )
                  for call in logger.method_calls ]
        return _digest_value( { 'outputs': first, 'image': _digest_image( written ), 'cached': cached,
                                'calls': detector.calls, 'log': lines } )


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


class TestDebugAids( unittest.TestCase ):
    """Debug drawing and the watermark are checked loosely: their pixels may change."""

    def test_debug_box_draws_in_its_colour_and_keeps_the_shape( self ):
        image = np.zeros( ( 120, 160, 3 ), np.uint8 )
        drawn = bu_censor.debug_image( image, _box( 20, 30, 60, 40, DEBUG_STYLE ) )
        self.assertEqual( drawn.shape, ( 120, 160, 3 ) )
        self.assertTrue( ( drawn == ( 0, 255, 0 ) ).all( axis=2 ).any(), "the style colour, as BGR" )

    def test_shape_annotation_draws_and_keeps_the_shape( self ):
        drawn = bu_censor.annotate_image_shape( np.zeros( ( 120, 160, 3 ), np.uint8 ) )
        self.assertEqual( drawn.shape, ( 120, 160, 3 ) )
        self.assertTrue( drawn.any() )

    def test_watermark_only_when_enabled( self ):
        image = _image( 240, 320 )
        with mock.patch.object( betaconfig, 'enable_betasuite_watermark', False, create=True ):
            self.assertIs( bu_censor.watermark_image( image ), image )
        with mock.patch.object( betaconfig, 'enable_betasuite_watermark', True, create=True ):
            marked = bu_censor.watermark_image( image.copy() )
        self.assertEqual( marked.shape, image.shape )
        self.assertFalse( np.array_equal( marked, image ) )

    def test_debug_mode_gives_every_class_a_debug_entry( self ):
        with mock.patch.object( betaconfig, 'debug_mode', 1 ), \
             mock.patch.object( betaconfig, 'items_to_censor', [ 'exposed_breast' ] ):
            bu_config.invalidate_config_caches()
            try:
                parts = bu_config.get_parts_to_blur()
            finally:
                bu_config.invalidate_config_caches()
        normal = bu_config.get_label_settings( 'exposed_breast' )
        for label, colour in betaconst.classes.items():
            self.assertEqual( parts[ label ][ 'censor_style' ], { 'type': 'debug', 'color': colour } )
            self.assertLessEqual( set( parts[ label ] ), set( normal ), "only keys a normal entry has" )
            self.assertLessEqual( { 'min_prob', 'censor_shape', 'time_safety', 'min_track_hits' },
                                  set( parts[ label ] ) )


class TestEdgeCases( unittest.TestCase ):
    """Cases the mutation check found unpinned in the rewritten code."""

    def test_overlapping_boxes_merge_only_with_the_same_label_and_style( self ):
        blur = { 'type': 'blur', 'method': 'gaussian', 'strength': 20, 'merge': 'single-pass' }
        pixel = { 'type': 'pixel', 'pattern': 'square', 'strength': 12, 'merge': 'single-pass' }
        same = [ _box( 10, 10, 50, 40, blur ), _box( 40, 30, 50, 40, dict( blur ) ) ]
        self.assertEqual( 1, len( bu_censor._censored_windows( same, 160, 120 ) ) )
        other_label = [ _box( 10, 10, 50, 40, blur ), _box( 40, 30, 50, 40, blur, label='exposed_vulva' ) ]
        self.assertEqual( 2, len( bu_censor._censored_windows( other_label, 160, 120 ) ) )
        other_style = [ _box( 10, 10, 50, 40, blur ), _box( 40, 30, 50, 40, pixel ) ]
        self.assertEqual( 2, len( bu_censor._censored_windows( other_style, 160, 120 ) ) )

    def test_an_audio_probe_that_fails_means_no_audio( self ):
        import subprocess
        import betautils_video as bu_video
        with mock.patch.object( bu_video.subprocess, 'run', side_effect=subprocess.TimeoutExpired( 'ffprobe', 1 ) ):
            self.assertFalse( bu_video.video_file_has_audio( 'x.mp4' ) )
        failed = mock.Mock( returncode=1, stdout=b'1,audio\n' )
        with mock.patch.object( bu_video.subprocess, 'run', return_value=failed ):
            self.assertFalse( bu_video.video_file_has_audio( 'x.mp4' ), "a failed probe is not trusted" )
        found = mock.Mock( returncode=0, stdout=b'0,video\n1,audio\n' )
        with mock.patch.object( bu_video.subprocess, 'run', return_value=found ):
            self.assertTrue( bu_video.video_file_has_audio( 'x.mp4' ) )

    def test_retinanet_drops_a_score_equal_to_the_floor( self ):
        boxes = np.zeros( ( 1, 300, 4 ), np.float32 )
        boxes[ 0, :2 ] = [ 10, 10, 50, 50 ]
        scores = np.zeros( ( 1, 300 ), np.float32 )
        classes = np.zeros( ( 1, 300 ), np.int32 )
        with mock.patch.object( betaconfig, 'global_min_prob', 0.25 ):
            scores[ 0, 0 ], scores[ 0, 1 ] = 0.25, 0.2501
            raw = retinanet_v2.raw_boxes_from_model_output( [ boxes, scores, classes ], [ 1.0 ], [ 0.0 ] )
        self.assertEqual( [ 0.2501 ], [ round( box[ 'score' ], 4 ) for box in raw[ 0 ] ] )

    def test_retinanet_skips_class_indexes_outside_its_table( self ):
        count = len( retinanet_v2._MODEL_CLASS_ORDER )
        boxes = np.zeros( ( 1, 300, 4 ), np.float32 )
        boxes[ 0, :4 ] = [ 10, 10, 50, 50 ]
        scores = np.zeros( ( 1, 300 ), np.float32 )
        scores[ 0, :4 ] = 0.9
        classes = np.zeros( ( 1, 300 ), np.int32 )
        classes[ 0, :4 ] = [ -1, count, count - 1, 0 ]
        raw = retinanet_v2.raw_boxes_from_model_output( [ boxes, scores, classes ], [ 1.0 ], [ 0.0 ] )
        self.assertEqual( [ retinanet_v2._MODEL_CLASS_ORDER[ -1 ], retinanet_v2._MODEL_CLASS_ORDER[ 0 ] ],
                          [ box[ 'class_id' ] for box in raw[ 0 ] ] )

    def test_retinanet_size_zero_resizes_without_padding( self ):
        image = _image( 90, 140 )
        prepared = retinanet_v2.prep_img_for_nn( image, 0, 0.5 )
        self.assertEqual( prepared.shape, ( 45, 70, 3 ) )
        self.assertEqual( retinanet_v2.prep_img_for_nn( image, 160, 0.5 ).shape, ( 160, 160, 3 ) )

    def test_retinanet_rejects_negative_and_non_integer_sizes( self ):
        errors = []
        with mock.patch.object( retinanet_v2.bu_detector, 'get_picture_sizes', return_value=[ 0, 640, -1, 1.5 ] ):
            retinanet_v2.validate_backend_config( {}, errors )
        self.assertEqual( [ ( 'config.rule', 'backend.size_negative' ) ] * 2,
                          [ ( error.code, error.params[ 'rule' ] ) for error in errors ] )


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
