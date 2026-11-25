"""
test_preview.py - preview.look draws exactly what the censor pipeline does.

The milestone's check: for the same image, boxes and settings, a look
preview equals what betastare.py writes for a photo with those detections
(the detector stubbed to return the boxes), pixel for pixel, for every
preset and for a weighted list with per-style area safety. One case goes
over the engine protocol with per-run overrides. Also: the still path
resolves looks as a video's first frame does (paired_style, per-style
area safety), the cache, the presets and art_style, and the errors.
"""

import os
import random
import shutil
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

import betaconfig
import betastare
import betautils_art as bu_art
import betautils_config_store as bu_store
import betautils_detector as bu_detector
import betautils_log as bu_log
import betautils_looks as bu_looks
import betautils_preview as bu_preview
import betautils_protocol as bu_proto
import betautils_serve_preview as bu_serve_preview
import betautils_settings as bu_settings
import betautils_still as bu_still
import betautils_track as bu_track

from tests.test_serve import _EngineTest

FIGURE = ( 'flat', 'figure.female.front.unclothed' )
LABELS = [ 'exposed_breast', 'exposed_vulva', 'exposed_armpits', 'exposed_feet' ]

# A custom weighted list, like the shipped breast list: its own area
# safety per style, so the per-style padding path is exercised.
CUSTOM = [
    { 'type': 'bar', 'color': [ 0, 0, 0 ], 'thickness': 1.0, 'weight': 1, 'width_area_safety': -0.35 },
    { 'type': 'pixel', 'pattern': 'hex', 'strength': 30, 'weight': 1, 'width_area_safety': 0.1,
      'height_area_safety': 0.1 },
    { 'type': 'blur', 'method': 'box', 'strength': 85, 'feather': 0.2, 'weight': 1 },
]


def _settings( style, labels=LABELS, paired=None ):
    overrides = { ( 'items_to_censor', ): list( labels ) }
    backend = bu_detector.selected_backend_name()
    for label in labels:
        overrides[( 'item_overrides', label, 'censor_style' )] = style
        if paired is not None:
            # a per-backend tunable: the backend's tier wins over the shared block
            overrides[( 'detector_backend', backend, 'item_overrides', label, 'paired_style' )] = paired
    return overrides


def _art( style, item ):
    art = bu_art.art_item( style, item )
    image = cv2.imread( art['path'], cv2.IMREAD_COLOR )
    return art, image, bu_preview.raw_detections( art['boxes'], image.shape[1], image.shape[0] )


class _CacheDir( unittest.TestCase ):

    def setUp( self ):
        self.tmp = tempfile.mkdtemp( prefix='betasuite-preview-' )
        self.addCleanup( shutil.rmtree, self.tmp, True )

    def preview( self, image, raw, labels, seed ):
        result = bu_preview.preview( image, raw, labels, seed, cache_dir=self.tmp )
        return cv2.imread( result['path'], cv2.IMREAD_COLOR ), result


def photo_output( test, image, raw, seed ):
    """What betastare.py writes for this image when the detector returns raw."""
    work = tempfile.mkdtemp( dir=test.tmp )
    root = os.path.join( work, 'in' )
    out = os.path.join( work, 'out' )
    os.makedirs( root )
    os.makedirs( out )
    cv2.imwrite( os.path.join( root, 'figure.png' ), image )
    random.seed( seed )
    with mock.patch.object( betastare, '_raw_boxes_for_all_sizes', return_value=( [ raw ], False ) ):
        betastare._process_one_file( root, 'figure.png', out, 1, 1, None, bu_log.get_logger() )
    written = os.listdir( out )
    test.assertEqual( len( written ), 1, written )
    return cv2.imread( os.path.join( out, written[0] ), cv2.IMREAD_COLOR )


class PreviewEqualsPhotoTest( _CacheDir ):

    def check( self, style, seed=7, art=FIGURE ):
        _art_entry, image, raw = _art( *art )
        with bu_store.overrides_applied( _settings( style ) ):
            preview, _result = self.preview( image, raw, LABELS, seed )
            photo = photo_output( self, image, raw, seed )
        self.assertEqual( preview.shape, photo.shape )
        self.assertTrue( np.array_equal( preview, photo ), 'preview differs from the photo pipeline' )
        self.assertFalse( np.array_equal( preview, image ), 'nothing was censored' )

    def test_every_preset( self ):
        for preset in bu_looks.presets():
            with self.subTest( preset=preset['id'] ):
                self.check( preset['style'] )

    def test_custom_list_with_per_style_area_safety( self ):
        for seed in range( 4 ):
            with self.subTest( seed=seed ):
                self.check( CUSTOM, seed )

    def test_another_style_and_a_tile( self ):
        self.check( bu_looks.presets()[0]['style'], art=( 'cel', 'tile.exposed_breast.female' ) )


class PreviewOverProtocolTest( _EngineTest ):

    def test_engine_preview_equals_the_photo_pipeline( self ):
        engine = self.start()
        self.hello( engine )
        overrides = { 'items_to_censor': LABELS }
        for label in LABELS:
            overrides['item_overrides.%s.censor_style'%( label, )] = CUSTOM
        result = engine.request( 'preview.look', {
            'previews': [ { 'art': { 'style': FIGURE[0], 'item': FIGURE[1] }, 'seed': 3 } ],
            'overrides': overrides } )
        self.assertTrue( result['ok'], result )
        entry = result['data']['previews'][0]
        self.assertFalse( entry['cached'] )
        served = cv2.imread( entry['path'], cv2.IMREAD_COLOR )
        _art_entry, image, raw = _art( *FIGURE )
        with bu_store.overrides_applied( _settings( CUSTOM ) ):
            photo = photo_output( self, image, raw, 3 )
        self.assertTrue( np.array_equal( served, photo ) )

        again = engine.request( 'preview.look', {
            'previews': [ { 'art': { 'style': FIGURE[0], 'item': FIGURE[1] }, 'seed': 3 } ],
            'overrides': overrides } )
        self.assertEqual( again['data']['previews'][0]['key'], entry['key'] )
        self.assertTrue( again['data']['previews'][0]['cached'] )

    def test_errors( self ):
        engine = self.start()
        self.hello( engine )
        self.assertError( engine.request( 'preview.look', { 'previews': [ { 'art': { 'style': 'nope', 'item': 'x' } } ] } ),
                          'preview.unknown_art', { 'style': 'nope', 'item': 'x' } )
        self.assertError( engine.request( 'preview.look', { 'previews': [ { 'art': { 'style': 'flat', 'item': 'tile.nope.female' } } ] } ),
                          'preview.unknown_art' )
        missing = os.path.join( self.tmp, 'missing.png' )
        self.assertError( engine.request( 'preview.look', { 'previews': [ {
            'image': missing, 'boxes': [ { 'label': 'exposed_breast', 'x': 0, 'y': 0, 'w': 4, 'h': 4 } ] } ] } ),
            'preview.bad_image', {} )
        self.assertError( engine.request( 'preview.look', {} ), 'command.bad_params',
                          { 'field': 'previews', 'reason': 'missing' } )
        self.assertError( engine.request( 'preview.look', { 'previews': [] } ), 'command.bad_params',
                          { 'field': 'previews', 'reason': 'value' } )
        self.assertError( engine.request( 'preview.look', { 'previews': [ { 'image': missing } ] } ),
                          'command.bad_params', { 'field': 'previews.0.boxes', 'reason': 'missing' } )
        self.assertError( engine.request( 'preview.look', { 'previews': [ {
            'art': { 'style': 'flat', 'item': FIGURE[1] }, 'labels': [ 'no_such_label' ] } ] } ),
            'command.bad_params', { 'field': 'previews.0.labels', 'reason': 'value' } )
        self.assertError( engine.request( 'preview.look', { 'previews': [ {
            'art': { 'style': 'flat', 'item': FIGURE[1] } } ], 'overrides': { 'encode_crf': 'x' } } ),
            'config.invalid' )
        self.assertError( engine.request( 'preview.look', { 'previews': [ {
            'art': { 'style': 'flat', 'item': FIGURE[1] } } ], 'overrides': { 'no_such_key': 1 } } ),
            'config.unknown_key' )


class LabelsFieldTest( unittest.TestCase ):

    def test_labels_render_a_label_that_is_switched_off( self ):
        server = mock.Mock( jobs=None )
        tile = { 'art': { 'style': 'flat', 'item': 'tile.exposed_vulva.female' } }
        with bu_store.overrides_applied( { ( 'items_to_censor', ): [] } ):
            off = bu_serve_preview.preview_look( server, { 'previews': [ dict( tile ) ] } )['previews'][0]
            on = bu_serve_preview.preview_look( server, { 'previews': [ dict( tile, labels=[ 'exposed_vulva' ] ) ] } )['previews'][0]
        original = cv2.imread( bu_art.art_item( 'flat', 'tile.exposed_vulva.female' )['path'] )
        self.assertTrue( np.array_equal( cv2.imread( off['path'] ), original ) )
        self.assertFalse( np.array_equal( cv2.imread( on['path'] ), original ) )


class BusyTest( unittest.TestCase ):

    def test_refused_while_a_job_holds_the_queue( self ):
        server = mock.Mock()
        server.jobs.busy_job.return_value = 'job-1'
        with self.assertRaises( bu_proto.ProtocolError ) as caught:
            bu_serve_preview.preview_look( server, { 'previews': [ { 'art': { 'style': 'flat', 'item': FIGURE[1] } } ] } )
        self.assertEqual( caught.exception.error, { 'code': 'config.busy', 'params': { 'job': 'job-1' } } )


class StillResolvesLikeVideoTest( unittest.TestCase ):
    """The still path's looks are a video's first frame's."""

    def resolve( self, style, paired, seed ):
        _art_entry, image, raw = _art( *FIGURE )
        breasts = [ box for box in raw if box['class_id'] == 'exposed_breast' ]
        self.assertEqual( len( breasts ), 2 )
        with bu_store.overrides_applied( _settings( style, [ 'exposed_breast' ], paired ) ):
            return bu_still.still_boxes( [ breasts ], image.shape[1], image.shape[0], rng=random.Random( seed ) )

    def test_paired_style_shares_one_look( self ):
        for seed in range( 12 ):
            left, right = self.resolve( CUSTOM, True, seed )
            self.assertEqual( left['censor_style'], right['censor_style'] )
            self.assertEqual( left['censor_sticker_seed'], right['censor_sticker_seed'] )

    def test_unpaired_rolls_each_box( self ):
        differ = 0
        for seed in range( 12 ):
            left, right = self.resolve( CUSTOM, False, seed )
            differ += left['censor_style'] != right['censor_style']
        self.assertGreater( differ, 0 )

    def test_style_area_safety_pads_the_box( self ):
        bar = dict( CUSTOM[0] )
        _art_entry, _image, raw = _art( *FIGURE )
        authored = [ box for box in raw if box['class_id'] == 'exposed_breast' ][0]
        left = self.resolve( [ bar ], True, 0 )[0]
        self.assertLess( left['w'], authored['w'] )
        self.assertEqual( ( left['style_scale_w'], left['style_scale_h'] ), ( left['w'], left['h'] ) )

    def test_same_looks_as_the_video_tracker_on_one_frame( self ):
        """
        Same pairing and the same per-style geometry as smooth_boxes on a
        single frame. Picks come from a counter rather than the random
        generator, because the tracker draws from it in places a still
        does not (a sticker seed it then discards), which would only shift
        which pick each box gets.
        """
        import betautils_censor as bu_censor
        for offset in range( len( CUSTOM ) ):
            for paired in ( True, False ):
                with self.subTest( offset=offset, paired=paired ):
                    got = self._resolved( 'still', offset, paired )
                    want = self._resolved( 'video', offset, paired )
                    self.assertEqual( got, want )
                    self.assertEqual( len( got ), len( LABELS ) + 2 )   # two breasts, two feet

    def _resolved( self, path, offset, paired ):
        import betautils_censor as bu_censor
        calls = [ offset ]

        def pick( style_config, rng=None ):
            if isinstance( style_config, dict ):
                return style_config
            calls[0] += 1
            return style_config[calls[0] % len( style_config )]
        _art_entry, image, raw = _art( *FIGURE )
        height, width = image.shape[:2]
        with bu_store.overrides_applied( _settings( CUSTOM, LABELS, paired ) ), \
                mock.patch.object( bu_censor, 'resolve_censor_style', pick ):
            if path == 'still':
                boxes = bu_still.still_boxes( [ raw ], width, height )
            else:
                parts = bu_preview.parts_for( LABELS )
                fresh = [ bu_censor.process_raw_box( one, width, height, parts ) for one in raw ]
                boxes = bu_track.smooth_boxes( [ box for box in fresh if box ] )
                boxes = boxes[0] if isinstance( boxes, tuple ) else boxes
        return { ( box['label'], box['_raw_x'], box['_raw_y'] ):
                 ( box['censor_style'], box['x'], box['y'], box['w'], box['h'] )
                 for box in boxes if '_raw_x' in box }


class NearestUnambiguousTest( unittest.TestCase ):
    """The pairing test shared by video tracks and stills."""

    settings = mock.Mock( paired_style_max_distance=2.0, paired_style_tiebreak_margin=0.25 )

    def nearest( self, *others ):
        candidates = [ ( index, { 'x': x, 'y': 0, 'w': 10, 'h': 10 } ) for index, x in enumerate( others ) ]
        return bu_track.nearest_unambiguous( candidates, 0, 0, 10, 10, self.settings )

    def test_one_near_candidate_wins( self ):
        self.assertEqual( self.nearest( 12 ), 0 )

    def test_far_candidates_are_ignored( self ):
        self.assertIsNone( self.nearest( 25 ) )

    def test_clearly_nearest_wins( self ):
        self.assertEqual( self.nearest( 18, 5 ), 1 )

    def test_two_at_much_the_same_distance_are_ambiguous( self ):
        self.assertIsNone( self.nearest( 10, 11 ) )

    def test_a_candidate_on_top_wins( self ):
        self.assertEqual( self.nearest( 1, 0 ), 1 )


class CacheTest( _CacheDir ):

    def _key( self ):
        _art_entry, image, raw = _art( *FIGURE )
        with bu_store.overrides_applied( _settings( bu_looks.presets()[0]['style'] ) ):
            boxes = bu_preview.resolve( image, raw, LABELS, 0 )
            return bu_preview.preview_key( image, boxes )

    def test_key_follows_the_censor_code_and_render_settings( self ):
        base = self._key()
        with mock.patch.object( bu_preview, '_censor_source_hash', return_value='other code' ):
            self.assertNotEqual( self._key(), base )
        with bu_store.overrides_applied( { ( 'blur_edge_margin', ): not betaconfig.blur_edge_margin } ):
            self.assertNotEqual( self._key(), base )
        self.assertEqual( self._key(), base )

    def test_a_cache_hit_counts_as_recent( self ):
        _art_entry, image, raw = _art( *FIGURE )
        _out, first = self.preview( image, raw, LABELS, 0 )
        os.utime( first['path'], ns=( 10**9, 10**9 ) )
        _out, again = self.preview( image, raw, LABELS, 0 )
        self.assertTrue( again['cached'] )
        self.assertGreater( os.stat( first['path'] ).st_mtime_ns, 10**9 )

    def test_second_request_is_cached_and_a_change_is_not( self ):
        _art_entry, image, raw = _art( *FIGURE )
        with bu_store.overrides_applied( _settings( bu_looks.presets()[0]['style'] ) ):
            _out, first = self.preview( image, raw, LABELS, 0 )
            _out, second = self.preview( image, raw, LABELS, 0 )
        with bu_store.overrides_applied( _settings( bu_looks.presets()[1]['style'] ) ):
            _out, third = self.preview( image, raw, LABELS, 0 )
        self.assertFalse( first['cached'] )
        self.assertTrue( second['cached'] )
        self.assertEqual( first['key'], second['key'] )
        self.assertNotEqual( first['key'], third['key'] )
        self.assertEqual( ( first['width'], first['height'] ), ( 600, 1200 ) )

    def test_trim_removes_oldest_first( self ):
        names = []
        for index in range( 6 ):
            name = '%064x.png'%( index, )
            path = os.path.join( self.tmp, name )
            with open( path, 'wb' ) as handle:
                handle.write( b'x' * 100 )
            os.utime( path, ns=( index * 10**9, index * 10**9 ) )
            names.append( name )
        with open( os.path.join( self.tmp, 'keep.txt' ), 'wb' ) as handle:
            handle.write( b'x' * 1000 )
        removed = bu_preview.trim_cache( self.tmp, max_bytes=400 )
        self.assertEqual( removed, 3 )
        self.assertEqual( sorted( os.listdir( self.tmp ) ), sorted( names[3:] + [ 'keep.txt' ] ) )
        self.assertEqual( bu_preview.trim_cache( self.tmp, max_bytes=400 ), 0 )


class PresetsAndStyleTest( unittest.TestCase ):

    def test_presets_validate_as_censor_styles( self ):
        store = bu_store.ConfigStore( path=None )
        for preset in bu_looks.presets():
            evaluation = store.evaluate( '', { ( 'item_overrides', 'exposed_breast', 'censor_style' ): preset['style'] } )
            self.assertTrue( evaluation.ok, ( preset['id'], evaluation.errors ) )

    def test_matching_finds_the_preset_and_nothing_else( self ):
        for preset in bu_looks.presets():
            self.assertEqual( bu_looks.matching( preset['style'] ), preset['id'] )
        self.assertEqual( bu_looks.matching( { 'type': 'bar', 'color': ( 0, 0, 0 ), 'thickness': 1.0 } ), 'bars' )
        self.assertIsNone( bu_looks.matching( CUSTOM ) )
        self.assertIsNone( bu_looks.matching( betaconfig.item_overrides.get( 'exposed_breast', {} ).get( 'censor_style' ) ) )

    def test_presets_are_served_and_well_formed( self ):
        schema = bu_settings.schema()
        self.assertEqual( schema['presets'], bu_looks.presets() )
        ids = [ preset['id'] for preset in schema['presets'] ]
        self.assertEqual( ids, [ 'blur', 'pixelate', 'bars', 'mixed' ] )
        for preset in schema['presets']:
            self.assertEqual( preset['name_key'], 'look.preset.' + preset['id'] )
            self.assertGreaterEqual( preset['version'], 1 )

    def test_art_style_choices_are_the_art_folders( self ):
        entry = bu_settings.find( ( 'art_style', ) )
        self.assertEqual( entry.choice_list(), [ 'angular', 'cel', 'flat', 'ink' ] )
        self.assertEqual( entry.moves, frozenset() if isinstance( entry.moves, frozenset ) else type( entry.moves )() )
        self.assertEqual( betaconfig.art_style, 'flat' )


if __name__ == '__main__':
    unittest.main()
