"""
test_reverse_mode.py - reverse mode: obscure the frame except windows,
censoring the censored labels with their own styles.

What each group prevents:
  - Normal mode must not change at all while reverse mode is off: same
    censor key (or the whole library re-renders) and the same resolved
    settings.
  - The compositor: a show window that does not show the source, a
    censored window that leaks the source (the double-alpha ring), a
    censor computed from the blurred frame, censor losing an overlap,
    an untriggered frame obscured under passthrough, a crash on a window
    off the frame edge, and mutation of the shared box list.
  - Resolution: a window inheriting the label's normal-mode look, a
    censored label losing it, a reverse-only key losing to a backend tier
    or a profile, and tracking re-rolling a window into the normal-mode
    style list.
  - The trigger: detections the trigger should see (below a strict
    censored label's min_prob) and should not see (suppressed, a window,
    wrong label), censored boxes holding it on by themselves, the lead
    before the first sample, and the look-ahead fade.
  - The key, validation, the CLI, and one real betatv run on a clip.
"""

import copy
import logging
import os
import shutil
import subprocess
import tempfile
import unittest

import cv2
import numpy as np

import betaconfig
import betautils_cache_paths as bu_cache
import betautils_censor as bu_censor
import betautils_cli as bu_cli
import betautils_config as bu_config
import betautils_detector as bu_detector
import betautils_log as bu_log
import betautils_reverse as bu_reverse
import betautils_track as bu_track

_WATCHED = ( 'reverse', 'items_to_censor', 'item_overrides', 'detector_backend',
             'global_min_prob', 'default_censor_shape', 'enable_betasuite_watermark' )
_MISSING = object()

FRAME_W, FRAME_H = 160, 120


class ConfigHarness( unittest.TestCase ):
    """Deep-saves and restores every betaconfig attribute these tests touch."""

    def setUp( self ):
        self._saved = { name: copy.deepcopy( getattr( betaconfig, name, _MISSING ) )
                        for name in _WATCHED }
        betaconfig.enable_betasuite_watermark = False
        bu_config.invalidate_config_caches()

    def tearDown( self ):
        for name, value in self._saved.items():
            if value is _MISSING:
                if hasattr( betaconfig, name ):
                    delattr( betaconfig, name )
            else:
                setattr( betaconfig, name, value )
        bu_config.invalidate_config_caches()

    def set_reverse( self, **settings ):
        block = { 'enabled': True,
                  'frame_style': { 'type': 'blur', 'method': 'box', 'strength': 20 },
                  'censor': { 'labels': [ 'exposed_breast' ] },
                  'windows': { 'face_femme': {} } }
        block.update( settings )
        betaconfig.reverse = block
        bu_config.invalidate_config_caches()
        return block


def _noise( seed=1 ):
    return np.random.RandomState( seed ).randint( 0, 256, ( FRAME_H, FRAME_W, 3 ), dtype=np.uint8 )


def _box( x, y, w, h, style, shape='box', label='exposed_breast' ):
    return { 'x': x, 'y': y, 'w': w, 'h': h, 'censor_style': style, 'censor_shape': shape,
             'label': label, 'censor_sticker_seed': 0.0, 'start': 0.0, 'end': 1.0 }


SHOW = { 'type': 'show' }
BLACK_FRAME = { 'type': 'bar', 'color': ( 0, 0, 0 ) }
BLUR_FRAME = { 'type': 'blur', 'method': 'box', 'strength': 20 }
MOSAIC = { 'type': 'pixel', 'pattern': 'mosaic', 'strength': 30 }
WHITE_BAR = { 'type': 'bar', 'color': ( 255, 255, 255 ) }


# ---------------------------------------------------------------------------
# Normal mode is untouched while reverse mode is off
# ---------------------------------------------------------------------------

class TestOffMeansUnchanged( ConfigHarness ):

    def _snapshot( self ):
        bu_config.invalidate_config_caches()
        labels = { label: bu_config.get_label_settings( label )
                   for label in bu_config.get_parts_to_blur() }
        return bu_cache.censor_key(), labels

    def test_a_disabled_block_is_indistinguishable_from_no_block( self ):
        if hasattr( betaconfig, 'reverse' ):
            delattr( betaconfig, 'reverse' )
        without = self._snapshot()
        self.set_reverse( enabled=False,
                          censor={ 'labels': [ 'exposed_breast' ], 'overrides': {
                              'exposed_breast': { 'min_prob': 0.9, 'censor_style': MOSAIC } } },
                          windows={ 'exposed_vulva': { 'min_prob': 0.9, 'censor_shape': 'ellipse' } } )
        self.assertEqual( self._snapshot(), without )

    def test_disabled_settings_do_not_move_the_key( self ):
        self.set_reverse( enabled=False )
        before = bu_cache.censor_key()
        self.set_reverse( enabled=False, fade_seconds=3.0, empty_frame='obscure' )
        self.assertEqual( bu_cache.censor_key(), before )

    def test_no_mode_marker_while_off( self ):
        self.set_reverse( enabled=False )
        self.assertEqual( bu_cache.output_mode_marker(), '' )


# ---------------------------------------------------------------------------
# Compositor
# ---------------------------------------------------------------------------

class TestCompositor( ConfigHarness ):

    def test_untriggered_passthrough_renders_only_censored_windows_like_normal_mode( self ):
        source = _noise()
        boxes = [ _box( 10, 10, 40, 30, SHOW ), _box( 80, 50, 40, 40, MOSAIC ) ]
        reverse_out = bu_censor.censor_img_reverse( source.copy(), boxes, 0.0, BLUR_FRAME )
        normal_out = bu_censor.censor_img_for_boxes( source.copy(), [ boxes[1] ] )
        np.testing.assert_array_equal( reverse_out, normal_out )
        # Precondition: the censored window did change something.
        self.assertFalse( np.array_equal( normal_out, source ) )

    def test_an_untriggered_frame_never_pays_for_the_frame_style( self ):
        # Passthrough is what keeps most of a library's frames near free;
        # obscuring and then discarding gives the same pixels at full cost.
        from unittest import mock
        with mock.patch.object( bu_censor, 'obscure_full_frame',
                                side_effect=AssertionError( 'frame style computed' ) ):
            bu_censor.censor_img_reverse( _noise(), [ _box( 10, 10, 40, 30, SHOW ),
                                                      _box( 60, 40, 40, 40, MOSAIC ) ], 0.0, BLUR_FRAME )

    def test_a_triggered_frame_is_obscured_outside_every_window( self ):
        source = _noise()
        out = bu_censor.censor_img_reverse( source.copy(), [ _box( 10, 10, 40, 30, SHOW ) ],
                                            1.0, BLUR_FRAME )
        outside = np.ones( ( FRAME_H, FRAME_W ), dtype=bool )
        outside[ 10:40, 10:50 ] = False
        changed = np.any( out != source, axis=2 )
        self.assertGreater( changed[outside].mean(), 0.99 )

    def test_the_frame_style_covers_the_whole_frame_even_with_a_feather( self ):
        # A feathered frame style would fade to the source at the border.
        out = bu_censor.censor_img_reverse( _noise(), [], 1.0, dict( BLACK_FRAME, feather=0.5 ) )
        self.assertTrue( np.all( out == 0 ) )

    def test_dim_darkens_the_frame_style_and_nothing_else( self ):
        source = _noise()
        window = _box( 10, 10, 40, 30, SHOW )
        censored = _box( 80, 50, 40, 40, MOSAIC )
        plain = bu_censor.censor_img_reverse( source.copy(), [ window, censored ], 1.0, BLUR_FRAME )
        dimmed = bu_censor.censor_img_reverse( source.copy(), [ window, censored ], 1.0,
                                               dict( BLUR_FRAME, dim=0.4 ) )
        outside = np.ones( ( FRAME_H, FRAME_W ), dtype=bool )
        outside[ 10:40, 10:50 ] = outside[ 50:90, 80:120 ] = False
        expected = np.round( plain.astype( np.float64 ) * 0.6 )
        self.assertLessEqual( np.abs( dimmed[outside] - expected[outside] ).max(), 1.0,
                              "the frame style, 40% darker" )
        np.testing.assert_array_equal( dimmed[ 10:40, 10:50 ], source[ 10:40, 10:50 ],
                                       "a show window is the source, never dimmed" )
        np.testing.assert_array_equal( dimmed[ 50:90, 80:120 ], plain[ 50:90, 80:120 ],
                                       "a censored label keeps its normal look" )

    def test_no_dim_is_the_frame_style_exactly( self ):
        source = _noise()
        for frame_style in ( dict( BLUR_FRAME, dim=0 ), dict( BLUR_FRAME, dim=0.0 ) ):
            np.testing.assert_array_equal(
                bu_censor.censor_img_reverse( source.copy(), [], 1.0, frame_style ),
                bu_censor.censor_img_reverse( source.copy(), [], 1.0, BLUR_FRAME ) )
        self.assertTrue( np.all( bu_censor.censor_img_reverse( source.copy(), [], 1.0,
                                                              dict( BLUR_FRAME, dim=1 ) ) == 0 ) )

    def test_dim_darkens_whatever_the_frame_style_drew( self ):
        out = bu_censor.censor_img_reverse( _noise(), [], 1.0, dict( WHITE_BAR, dim=0.4 ) )
        self.assertTrue( np.all( out == 153 ), "a white bar dimmed 40%" )

    def test_a_fade_blends_the_dimmed_frame( self ):
        source = _noise()
        full = bu_censor.censor_img_reverse( source.copy(), [], 1.0, dict( BLUR_FRAME, dim=0.5 ) )
        half = bu_censor.censor_img_reverse( source.copy(), [], 0.5, dict( BLUR_FRAME, dim=0.5 ) )
        expected = 0.5 * full.astype( np.float64 ) + 0.5 * source.astype( np.float64 )
        self.assertLessEqual( np.abs( half - expected ).max(), 1.0 )

    def test_a_hard_edged_show_window_is_exactly_the_source( self ):
        source = _noise()
        out = bu_censor.censor_img_reverse( source.copy(), [ _box( 10, 10, 40, 30, SHOW ) ],
                                            1.0, BLUR_FRAME )
        np.testing.assert_array_equal( out[ 10:40, 10:50 ], source[ 10:40, 10:50 ] )

    def test_a_censored_window_owes_nothing_to_the_source_when_its_style_does_not( self ):
        # A solid bar window on a solid frame style: every pixel in the
        # window, feather ramp and ellipse corners included, must be the
        # same whatever the source was. Applying the window's alpha
        # twice leaves part of the source in the ramp: a reveal.
        white = { 'type': 'bar', 'color': ( 255, 255, 255 ), 'feather': 0.4 }
        boxes = [ _box( 30, 20, 80, 70, white, shape='ellipse' ) ]
        first = bu_censor.censor_img_reverse( _noise( 1 ), boxes, 1.0, BLACK_FRAME )
        second = bu_censor.censor_img_reverse( _noise( 2 ), boxes, 1.0, BLACK_FRAME )
        np.testing.assert_array_equal( first[ 20:90, 30:110 ], second[ 20:90, 30:110 ] )
        # Precondition: the feather really does ramp between the two.
        region = first[ 20:90, 30:110, 0 ]
        self.assertTrue( ( ( region > 10 ) & ( region < 245 ) ).any() )

    def test_a_hard_edged_censored_window_is_the_normal_censor_of_the_original( self ):
        # Rendering the style from the obscured frame instead would give
        # a mosaic of the blur, not of the subject.
        source = _noise()
        box = _box( 40, 30, 60, 50, MOSAIC )
        out = bu_censor.censor_img_reverse( source.copy(), [ box ], 1.0, BLACK_FRAME )
        expected = bu_censor.censor_image( source.copy(), dict( box ) )
        np.testing.assert_array_equal( out[ 30:80, 40:100 ], expected[ 30:80, 40:100 ] )

    def test_censored_windows_look_the_same_triggered_or_not( self ):
        # Same merge as normal mode, so a window does not change look
        # when the frame style switches on around it.
        source = _noise()
        boxes = [ _box( 20, 20, 60, 50, MOSAIC ), _box( 50, 40, 60, 50, MOSAIC ) ]
        untriggered = bu_censor.censor_img_reverse( source.copy(), boxes, 0.0, BLACK_FRAME )
        triggered = bu_censor.censor_img_reverse( source.copy(), boxes, 1.0, BLACK_FRAME )
        np.testing.assert_array_equal( triggered[ 20:90, 20:110 ], untriggered[ 20:90, 20:110 ] )

    def test_censor_wins_where_a_censored_window_overlaps_a_show_window( self ):
        source = _noise()
        show, censored = _box( 10, 10, 80, 60, SHOW ), _box( 50, 30, 60, 60, MOSAIC )
        out = bu_censor.censor_img_reverse( source.copy(), [ censored, show ], 1.0, BLACK_FRAME )
        expected = bu_censor.censor_image( source.copy(), dict( censored ) )
        np.testing.assert_array_equal( out[ 30:70, 50:90 ], expected[ 30:70, 50:90 ] )

    def test_overlapping_show_windows_show_the_source_once( self ):
        source = _noise()
        boxes = [ _box( 10, 10, 60, 60, SHOW ), _box( 40, 30, 60, 60, SHOW ) ]
        out = bu_censor.censor_img_reverse( source.copy(), boxes, 1.0, BLUR_FRAME )
        np.testing.assert_array_equal( out[ 30:70, 40:70 ], source[ 30:70, 40:70 ] )

    def test_a_partial_fade_sits_between_clear_and_obscured( self ):
        source = _noise()
        obscured = bu_censor.censor_img_reverse( source.copy(), [], 1.0, BLACK_FRAME )
        half = bu_censor.censor_img_reverse( source.copy(), [], 0.5, BLACK_FRAME )
        self.assertTrue( np.all( obscured == 0 ) )
        np.testing.assert_allclose( half.astype( float ), source * 0.5, atol=1.0 )

    def test_windows_hanging_off_the_frame_do_not_crash( self ):
        source = _noise()
        boxes = [ _box( -30, -20, 60, 50, SHOW ), _box( 140, 100, 80, 80, MOSAIC, shape='ellipse' ),
                  _box( 150, 5, 40, 30, dict( MOSAIC, feather=0.3 ), shape='circle' ) ]
        out = bu_censor.censor_img_reverse( source.copy(), boxes, 1.0, BLUR_FRAME )
        self.assertEqual( out.shape, source.shape )

    def test_the_shared_box_list_is_not_mutated( self ):
        boxes = [ _box( -30, -20, 60, 50, SHOW ), _box( 10, 10, 40, 40, MOSAIC ),
                  _box( 20, 15, 40, 40, MOSAIC ) ]
        before = copy.deepcopy( boxes )
        bu_censor.censor_img_reverse( _noise(), boxes, 1.0, BLUR_FRAME )
        bu_censor.censor_img_reverse( _noise(), boxes, 0.0, BLUR_FRAME )
        self.assertEqual( boxes, before )

    def _sticker_dir( self, sticker ):
        sticker_dir = tempfile.mkdtemp( prefix='betasuite-sticker-' )
        self.addCleanup( shutil.rmtree, sticker_dir, True )
        cv2.imwrite( os.path.join( sticker_dir, 's.png' ), sticker )
        return sticker_dir

    def test_a_mirrored_window_looks_like_normal_mode_wherever_its_edge_is_opaque( self ):
        # The shipped windows reuse each label's normal-mode style list.
        # Inside the window's shape, at full opacity, every style must
        # draw exactly what normal mode draws there; only the feather
        # ring and the corners outside the shape go to the frame style.
        sticker = np.zeros( ( 40, 40, 4 ), dtype=np.uint8 )
        cv2.circle( sticker, ( 20, 20 ), 14, ( 0, 200, 255, 255 ), -1 )
        styles = {
            'blur': ( { 'type': 'blur', 'method': 'triple_box', 'strength': 50, 'feather': 0.3 }, 'ellipse' ),
            'mosaic': ( dict( MOSAIC, feather=0.3 ), 'ellipse' ),
            'sticker': ( { 'type': 'sticker', 'dir': self._sticker_dir( sticker ), 'feather': 0.3 },
                         'circle' ),
        }
        source = _noise()
        for name, ( style, shape ) in styles.items():
            with self.subTest( style=name ):
                box = _box( 40, 30, 80, 70, style, shape=shape )
                normal = bu_censor.censor_img_for_boxes( source.copy(), [ dict( box ) ] )
                out = bu_censor.censor_img_reverse( source.copy(), [ box ], 1.0, BLACK_FRAME )
                alpha = bu_censor._alpha_for( shape, 80, 70, style['feather'] ).reshape( 70, 80, -1 )[ :, :, 0 ]
                opaque = alpha >= 0.999
                self.assertGreater( opaque.sum(), 1000 )
                np.testing.assert_array_equal( out[ 30:100, 40:120 ][ opaque ],
                                               normal[ 30:100, 40:120 ][ opaque ] )

    def test_a_sticker_window_shows_the_source_through_its_see_through_parts( self ):
        # Pins the documented limitation: the sticker is drawn over the
        # ORIGINAL before going into the frame. A future fix that draws it
        # over the frame style should change this test on purpose.
        sticker = np.zeros( ( 40, 40, 4 ), dtype=np.uint8 )
        sticker[ :, :20 ] = ( 255, 255, 255, 255 )      # left half opaque, right half clear
        sticker_dir = self._sticker_dir( sticker )
        source = _noise()
        box = _box( 40, 40, 40, 40, { 'type': 'sticker', 'dir': sticker_dir } )
        out = bu_censor.censor_img_reverse( source.copy(), [ box ], 1.0, BLACK_FRAME )
        np.testing.assert_array_equal( out[ 40:80, 62:80 ], source[ 40:80, 62:80 ] )
        self.assertTrue( np.all( out[ 40:80, 40:58 ] == 255 ) )


# ---------------------------------------------------------------------------
# Resolution: what a window inherits
# ---------------------------------------------------------------------------

class TestResolution( ConfigHarness ):

    def setUp( self ):
        super().setUp()
        betaconfig.item_overrides['exposed_breast'] = {
            'censor_shape': 'circle',
            'censor_style': [ dict( MOSAIC, weight=1 ), { 'type': 'sticker', 'dir': '.', 'weight': 1 } ] }
        backend = betaconfig.detector_backend.setdefault(
            bu_detector.selected_backend_name(), {} )
        backend.setdefault( 'item_overrides', {} ).setdefault( 'exposed_breast', {} )['min_prob'] = 0.21

    def as_window( self, entry=None, **settings ):
        self.set_reverse( censor={ 'labels': [] }, windows={ 'exposed_breast': entry or {} }, **settings )

    def test_an_empty_window_shows_and_never_takes_the_label_look( self ):
        self.as_window()
        settings = bu_config.get_label_settings( 'exposed_breast' )
        self.assertEqual( settings['censor_style'], SHOW )
        self.assertEqual( settings['censor_shape'], getattr( betaconfig, 'default_censor_shape', 'box' ) )

    def test_a_backend_tuning_key_reaches_the_window( self ):
        self.as_window()
        self.assertEqual( bu_config.get_label_settings( 'exposed_breast' )['min_prob'], 0.21 )

    def test_a_window_key_applies_to_reverse_only( self ):
        self.as_window( { 'min_prob': 0.6, 'censor_shape': 'ellipse' } )
        self.assertEqual( bu_config.get_label_settings( 'exposed_breast' )['min_prob'], 0.6 )
        self.as_window( { 'min_prob': 0.6, 'censor_shape': 'ellipse' }, enabled=False )
        settings = bu_config.get_label_settings( 'exposed_breast' )
        self.assertEqual( settings['min_prob'], 0.21 )
        self.assertEqual( settings['censor_shape'], 'circle' )

    def _rendered_box( self ):
        step = 1.0 / betaconfig.video_censor_fps
        raw = [ _raw( 'exposed_breast', 0.9, k * step, x=400, y=300, w=100, h=80 ) for k in range( 6 ) ]
        boxes, _stats = bu_track.prepare_boxes_for_render( raw, 1920, 1080, style_seed='seed' )
        ( box, ) = { ( b['x'], b['y'], b['w'], b['h'] ) for b in boxes if b['label'] == 'exposed_breast' }
        return box

    def test_a_window_area_safety_grows_it_about_its_centre( self ):
        # 1.0 doubles both sides, centred on the detection: the shipped
        # windows, so the frame style is not a field of detector-sized holes.
        self.as_window( { 'width_area_safety': 1.0, 'height_area_safety': 1.0 } )
        self.assertEqual( self._rendered_box(), ( 350, 260, 200, 160 ) )
        self.as_window( { 'width_area_safety': 0.0, 'height_area_safety': 0.0 } )
        self.assertEqual( self._rendered_box(), ( 400, 300, 100, 80 ) )

    def test_the_shipped_windows_are_twice_the_detection_and_soft_edged( self ):
        import importlib.util
        # A separate module object: the shipped file, untouched by any test.
        spec = importlib.util.spec_from_file_location( 'shipped_betaconfig', betaconfig.__file__ )
        module = importlib.util.module_from_spec( spec )
        spec.loader.exec_module( module )
        shipped = module.reverse['windows']
        self.assertEqual( { label: ( entry.get( 'width_area_safety' ), entry.get( 'height_area_safety' ),
                                     entry.get( 'censor_shape' ), entry.get( 'censor_style' ) )
                            for label, entry in shipped.items() },
                          { label: ( 1.0, 1.0, 'ellipse', { 'type': 'show', 'feather': 0.3 } )
                            for label in shipped } )

    def test_a_soft_window_keeps_the_whole_detection_clear( self ):
        # The feathered ellipse of a doubled window fades only outside
        # the detected box: what was detected is shown in full.
        self.as_window( { 'width_area_safety': 1.0, 'height_area_safety': 1.0, 'censor_shape': 'ellipse',
                          'censor_style': { 'type': 'show', 'feather': 0.3 } } )
        step = 1.0 / betaconfig.video_censor_fps
        raw = [ _raw( 'exposed_breast', 0.9, k * step, x=60, y=40, w=40, h=30 ) for k in range( 6 ) ]
        boxes, _stats = bu_track.prepare_boxes_for_render( raw, FRAME_W, FRAME_H, style_seed='seed' )
        box = next( b for b in boxes if b['label'] == 'exposed_breast' )
        source = _noise()
        out = bu_censor.censor_img_reverse( source.copy(), [ box ], 1.0, BLACK_FRAME )
        np.testing.assert_array_equal( out[ 40:70, 60:100 ], source[ 40:70, 60:100 ] )
        self.assertTrue( np.all( out[ box['y'], box['x'] ] == 0 ), "the window's corner is the frame" )

    def test_a_censored_label_keeps_its_normal_mode_look_and_tuning( self ):
        # The point of a censored label: a breast keeps its mosaic/hex
        # list inside the frame style instead of turning into it.
        self.set_reverse()
        settings = bu_config.get_label_settings( 'exposed_breast' )
        self.assertEqual( settings['censor_shape'], 'circle' )
        self.assertEqual( settings['censor_style'], betaconfig.item_overrides['exposed_breast']['censor_style'] )
        self.assertEqual( settings['min_prob'], 0.21 )

    def test_a_censored_override_applies_last_and_to_reverse_only( self ):
        censor = { 'labels': [ 'exposed_breast' ],
                   'overrides': { 'exposed_breast': { 'censor_style': MOSAIC, 'min_prob': 0.6 } } }
        self.set_reverse( censor=censor )
        settings = bu_config.get_label_settings( 'exposed_breast' )
        self.assertEqual( ( settings['censor_style'], settings['min_prob'], settings['censor_shape'] ),
                          ( MOSAIC, 0.6, 'circle' ) )
        self.set_reverse( censor=censor, enabled=False )
        settings = bu_config.get_label_settings( 'exposed_breast' )
        self.assertEqual( settings['censor_style'], betaconfig.item_overrides['exposed_breast']['censor_style'] )
        self.assertEqual( settings['min_prob'], 0.21 )

    def test_a_label_reverse_mode_does_not_render_keeps_its_normal_resolution( self ):
        # An override for a label that is not censored never applies
        # (validation rejects it too).
        self.set_reverse( censor={ 'labels': [ 'exposed_vulva' ],
                                   'overrides': { 'exposed_vulva': { 'min_prob': 0.9 },
                                                  'exposed_breast': { 'min_prob': 0.9 } } } )
        settings = bu_config.get_label_settings( 'exposed_breast' )
        self.assertEqual( settings['censor_shape'], 'circle' )
        self.assertEqual( settings['censor_style'], betaconfig.item_overrides['exposed_breast']['censor_style'] )
        self.assertEqual( settings['min_prob'], 0.21 )

    def test_rendered_labels_are_the_windows_and_the_censored_labels( self ):
        betaconfig.items_to_censor = [ 'exposed_vulva' ]
        self.set_reverse( censor={ 'labels': None }, windows={ 'face_femme': {} } )
        self.assertEqual( sorted( bu_config.get_parts_to_blur() ), [ 'exposed_vulva', 'face_femme' ] )
        self.set_reverse( censor={ 'labels': [ 'exposed_breast' ] }, windows={ 'face_femme': {} } )
        self.assertEqual( sorted( bu_config.get_parts_to_blur() ), [ 'exposed_breast', 'face_femme' ] )

    def _profile_with_time_safety( self, value ):
        backend = betaconfig.detector_backend[ bu_detector.selected_backend_name() ]
        backend['profiles'] = { 'default': 'p', 'variants': {
            'p': { 'item_overrides': { 'exposed_breast': { 'time_safety': value } } } } }
        bu_config.invalidate_config_caches()

    def test_a_window_key_beats_a_profile_in_tracking( self ):
        self._profile_with_time_safety( 0.06 )
        self.as_window( { 'time_safety': 0.4 } )
        settings = bu_track._LabelSettings( 'exposed_breast', None, 0.1, profile_name='p' )
        self.assertEqual( settings.time_safety, 0.4 )

    def test_a_censored_override_beats_a_profile_in_tracking( self ):
        self._profile_with_time_safety( 0.06 )
        self.set_reverse( censor={ 'labels': [ 'exposed_breast' ],
                                   'overrides': { 'exposed_breast': { 'time_safety': 0.4 } } } )
        settings = bu_track._LabelSettings( 'exposed_breast', None, 0.1, profile_name='p' )
        self.assertEqual( settings.time_safety, 0.4 )

    def test_the_profile_still_applies_to_inherited_keys_and_to_normal_mode( self ):
        self._profile_with_time_safety( 0.06 )
        self.as_window()
        self.assertEqual( bu_track._LabelSettings( 'exposed_breast', None, 0.1, 'p' ).time_safety, 0.06 )
        self.as_window( { 'time_safety': 0.4 }, enabled=False )
        self.assertEqual( bu_track._LabelSettings( 'exposed_breast', None, 0.1, 'p' ).time_safety, 0.06 )

    def test_tracking_rolls_the_window_style_not_the_label_list( self ):
        # A shot cut re-rolls a track's look from this list; reading the
        # shared block put the normal-mode stickers back on a show window.
        self.as_window()
        self.assertEqual( bu_track._LabelSettings( 'exposed_breast', None, 0.1 ).unresolved_censor_style,
                          SHOW )
        self.as_window( enabled=False )
        self.assertEqual( bu_track._LabelSettings( 'exposed_breast', None, 0.1 ).unresolved_censor_style,
                          betaconfig.item_overrides['exposed_breast']['censor_style'] )


# ---------------------------------------------------------------------------
# Trigger
# ---------------------------------------------------------------------------

def _raw( label, score, t, x=10, y=10, w=40, h=40 ):
    return { 'class_id': label, 'score': score, 't': t, 'x': x, 'y': y, 'w': w, 'h': h, 'size': 320 }


class TestTriggerDetections( ConfigHarness ):

    def setUp( self ):
        super().setUp()
        backend = betaconfig.detector_backend.setdefault( bu_detector.selected_backend_name(), {} )
        backend['class_suppression'] = {
            'exposed_breast': [ { 'suppressed_by': 'covered_breast', 'margin': 0.1, 'min_iou': 0.3 } ] }
        backend['class_promotion'] = {}
        self.set_reverse( censor={ 'labels': [ 'exposed_breast' ],
                                   'overrides': { 'exposed_breast': { 'min_prob': 0.8 } } },
                          trigger={ 'labels': None, 'min_prob': 0.3, 'hold_seconds': 0.5 } )

    def times( self, raw_boxes ):
        return bu_track.trigger_detection_times( raw_boxes, FRAME_W, FRAME_H )

    def test_a_detection_below_the_censored_label_but_above_the_trigger_triggers( self ):
        self.assertEqual( self.times( [ _raw( 'exposed_breast', 0.5, 1.0 ) ] ), [ 1.0 ] )

    def test_the_trigger_min_prob_is_a_floor_it_does_not_reach( self ):
        self.assertEqual( self.times( [ _raw( 'exposed_breast', 0.3, 1.0 ),
                                        _raw( 'exposed_breast', 0.31, 2.0 ) ] ), [ 2.0 ] )

    def test_a_suppressed_detection_does_not_trigger( self ):
        raw = [ _raw( 'exposed_breast', 0.5, 1.0 ), _raw( 'covered_breast', 0.9, 1.0 ) ]
        self.assertEqual( self.times( raw ), [] )
        # Precondition: without its suppressor it would have triggered.
        self.assertEqual( self.times( raw[:1] ), [ 1.0 ] )

    def test_only_trigger_labels_trigger_and_a_window_is_not_one_by_default( self ):
        # face_femme is a window: a face on screen must not blur the frame.
        self.assertEqual( self.times( [ _raw( 'face_femme', 0.99, 1.0 ) ] ), [] )
        self.set_reverse( trigger={ 'labels': [ 'face_femme' ], 'min_prob': 0.3 } )
        self.assertEqual( self.times( [ _raw( 'face_femme', 0.99, 1.0 ),
                                        _raw( 'exposed_breast', 0.99, 2.0 ) ] ), [ 1.0 ] )

    def test_the_input_is_not_mutated_and_observers_are_not_told( self ):
        raw = [ _raw( 'exposed_breast', 0.5, 1.0 ), _raw( 'covered_breast', 0.9, 1.0 ) ]
        before = copy.deepcopy( raw )
        seen = []

        class Observer:
            def on_suppressed( self, *args ):
                seen.append( args )

        with bu_track.tracking_observer( Observer() ):
            self.times( raw )
        self.assertEqual( raw, before )
        self.assertEqual( seen, [] )


class TestTimeline( unittest.TestCase ):

    def test_each_detection_holds_from_one_sample_before_to_hold_after( self ):
        self.assertEqual( bu_reverse.trigger_intervals( [ 2.0 ], 0.2, 0.5 ), [ ( 1.8, 2.5 ) ] )

    def test_consecutive_samples_join_and_far_ones_do_not( self ):
        intervals = bu_reverse.trigger_intervals( [ 1.0, 1.2, 1.4, 5.0 ], 0.2, 0.0 )
        self.assertEqual( intervals, [ ( 0.8, 1.4 ), ( 4.8, 5.0 ) ] )

    def test_the_fade_in_finishes_by_the_interval_start( self ):
        timeline = bu_reverse.TriggerTimeline( [ ( 2.0, 3.0 ) ], 0.5 )
        self.assertEqual( timeline.alpha_at( 2.0 ), 1.0 )
        self.assertAlmostEqual( timeline.alpha_at( 1.75 ), 0.5 )
        self.assertEqual( timeline.alpha_at( 1.4 ), 0.0 )

    def test_the_fade_out_starts_after_the_interval_ends( self ):
        timeline = bu_reverse.TriggerTimeline( [ ( 2.0, 3.0 ) ], 0.5 )
        self.assertEqual( timeline.alpha_at( 3.0 ), 1.0 )
        self.assertAlmostEqual( timeline.alpha_at( 3.25 ), 0.5 )
        self.assertEqual( timeline.alpha_at( 3.6 ), 0.0 )

    def test_zero_fade_is_a_hard_switch( self ):
        timeline = bu_reverse.TriggerTimeline( [ ( 2.0, 3.0 ) ], 0.0 )
        self.assertEqual( [ timeline.alpha_at( t ) for t in ( 1.99, 2.0, 3.0, 3.01 ) ],
                          [ 0.0, 1.0, 1.0, 0.0 ] )

    def test_obscure_mode_is_always_on( self ):
        timeline = bu_reverse.TriggerTimeline( [], 0.25, always_on=True )
        self.assertEqual( timeline.alpha_at( 0.0 ), 1.0 )


class TestTimelineFromConfig( ConfigHarness ):

    def test_the_lead_is_one_sample_interval( self ):
        self.set_reverse( fade_seconds=0.0, trigger={ 'hold_seconds': 0.5 } )
        plan = bu_reverse.render_plan_for_detections( [ 2.0 ], 4 )
        self.assertEqual( plan.timeline.intervals, [ ( 1.75, 2.5 ) ] )
        self.assertEqual( plan.frame_alpha( 1.8 ), 1.0 )

    def test_empty_frame_obscure_obscures_untriggered_frames( self ):
        self.set_reverse( empty_frame='obscure' )
        plan = bu_reverse.render_plan_for_detections( [], 9 )
        self.assertEqual( plan.frame_alpha( 0.0 ), 1.0 )
        self.set_reverse( empty_frame='passthrough' )
        self.assertEqual( bu_reverse.render_plan_for_detections( [], 9 ).frame_alpha( 0.0 ), 0.0 )

    def test_a_censored_box_holds_the_trigger_on_over_its_span_by_itself( self ):
        # No trigger detection at all (the box scored under the trigger's
        # min_prob, or tracking held it): still never a clear frame
        # around a censored object.
        self.set_reverse( fade_seconds=0.0 )
        boxes = [ _box( 0, 0, 10, 10, MOSAIC ), _box( 0, 0, 10, 10, SHOW, label='face_femme' ) ]
        boxes[0].update( start=3.0, end=4.0 )
        boxes[1].update( start=6.0, end=7.0 )
        plan = bu_reverse.render_plan_for_detections( [], 9, boxes )
        self.assertEqual( plan.timeline.intervals, [ ( 3.0, 4.0 ) ] )
        self.assertEqual( [ plan.frame_alpha( t ) for t in ( 2.9, 3.0, 4.0, 4.1, 6.5 ) ],
                          [ 0.0, 1.0, 1.0, 0.0, 0.0 ] )

    def test_censored_boxes_and_trigger_detections_merge( self ):
        self.set_reverse( fade_seconds=0.0, trigger={ 'hold_seconds': 0.5 } )
        box = _box( 0, 0, 10, 10, MOSAIC )
        box.update( start=2.2, end=3.0 )
        plan = bu_reverse.render_plan_for_detections( [ 2.0 ], 10, [ box ] )
        self.assertEqual( plan.timeline.intervals, [ ( 1.9, 3.0 ) ] )

    def test_a_still_triggers_on_a_censored_box_alone( self ):
        self.set_reverse( trigger={ 'min_prob': 0.9 } )
        self.assertEqual( bu_reverse.frame_alpha_for_still( [], [ _box( 0, 0, 9, 9, MOSAIC ) ] ), 1.0 )
        self.assertEqual( bu_reverse.frame_alpha_for_still(
            [], [ _box( 0, 0, 9, 9, SHOW, label='face_femme' ) ] ), 0.0 )

    def test_a_still_triggers_on_any_trigger_detection( self ):
        self.set_reverse( trigger={ 'min_prob': 0.3 } )
        self.assertEqual( bu_reverse.frame_alpha_for_still( [ _raw( 'exposed_breast', 0.31, 0 ) ] ), 1.0 )
        self.assertEqual( bu_reverse.frame_alpha_for_still( [ _raw( 'exposed_breast', 0.29, 0 ),
                                                              _raw( 'face_femme', 0.9, 0 ) ] ), 0.0 )
        self.assertEqual( bu_reverse.frame_alpha_for_still( [ _raw( 'exposed_breast', 0.3, 0 ) ] ), 0.0 )


# ---------------------------------------------------------------------------
# Censor key, output name
# ---------------------------------------------------------------------------

class TestCensorKey( ConfigHarness ):

    def key( self ):
        bu_config.invalidate_config_caches()
        return bu_cache.censor_key()

    def assert_moves( self, first, second, what ):
        self.set_reverse( **first )
        before = self.key()
        self.set_reverse( **second )
        self.assertNotEqual( before, self.key(), "%s changes reverse output but not the key"%( what, ) )

    def test_turning_reverse_on_moves_the_key( self ):
        self.set_reverse( enabled=False )
        before = self.key()
        self.set_reverse( enabled=True )
        self.assertNotEqual( before, self.key() )

    def test_every_reverse_setting_moves_the_key( self ):
        self.assert_moves( {}, { 'frame_style': { 'type': 'pixel', 'pattern': 'mosaic', 'strength': 4 } },
                           'frame_style' )
        self.assert_moves( {}, { 'frame_style': dict( BLUR_FRAME, dim=0.4 ) }, 'frame dim' )
        self.assert_moves( {}, { 'empty_frame': 'obscure' }, 'empty_frame' )
        self.assert_moves( {}, { 'fade_seconds': 0.5 }, 'fade_seconds' )
        self.assert_moves( {}, { 'trigger': { 'min_prob': 0.4 } }, 'trigger min_prob' )
        self.assert_moves( {}, { 'trigger': { 'hold_seconds': 1.0 } }, 'trigger hold_seconds' )
        self.assert_moves( {}, { 'trigger': { 'labels': [ 'face_femme' ] } }, 'trigger labels' )
        self.assert_moves( {}, { 'windows': { 'face_femme': { 'censor_style': dict( SHOW, feather=0.3 ) } } },
                           'window feather' )
        self.assert_moves( {}, { 'windows': { 'face_femme': {}, 'face_masc': {} } }, 'a new window' )
        self.assert_moves( {}, { 'censor': { 'labels': [ 'exposed_breast', 'exposed_vulva' ] } },
                           'a new censored label' )
        self.assert_moves( {}, { 'censor': { 'labels': [ 'exposed_breast' ], 'overrides': {
            'exposed_breast': { 'censor_style': MOSAIC } } } }, 'a censored override' )

    def test_a_window_inherited_value_moves_the_key( self ):
        self.set_reverse()
        before = self.key()
        backend = betaconfig.detector_backend.setdefault( bu_detector.selected_backend_name(), {} )
        backend.setdefault( 'item_overrides', {} ).setdefault( 'exposed_breast', {} )['time_safety'] = 0.77
        self.assertNotEqual( before, self.key() )

    def test_the_output_name_says_reverse( self ):
        self.set_reverse()
        self.assertEqual( bu_cache.output_mode_marker(), '-reverse' )
        name = bu_cache.video_output_basename( 'clip', 'abc', [ 320 ], 9, 'd1', 'c1', 'e1', '', '',
                                               bu_cache.output_mode_marker() )
        self.assertTrue( name.endswith( '-ee1-reverse' ), name )


# ---------------------------------------------------------------------------
# Validation and CLI
# ---------------------------------------------------------------------------

class TestValidation( ConfigHarness ):

    def errors( self ):
        bu_config.invalidate_config_caches()
        errors = []
        bu_config._validate_reverse( errors, bu_config._valid_class_names() )
        return errors

    def test_the_shipped_block_validates( self ):
        betaconfig.reverse = copy.deepcopy( self._saved['reverse'] )
        if betaconfig.reverse is _MISSING:
            self.skipTest( 'no reverse block in betaconfig.py' )
        self.assertEqual( self.errors(), [] )

    def test_frame_styles_too_slow_or_meaningless_are_rejected( self ):
        for frame_style in ( { 'type': 'blur', 'method': 'gaussian', 'strength': 20 },
                             { 'type': 'pixel', 'pattern': 'hex', 'strength': 4 },
                             { 'type': 'sticker', 'dir': '.' }, SHOW ):
            self.set_reverse( frame_style=frame_style )
            self.assertTrue( self.errors(), frame_style )
        self.set_reverse( frame_style={ 'type': 'pixel', 'pattern': 'mosaic', 'strength': 4 } )
        self.assertEqual( self.errors(), [] )

    def test_dim_is_a_share_of_brightness( self ):
        for dim in ( 0, 0.4, 1, 1.0 ):
            self.set_reverse( frame_style=dict( BLUR_FRAME, dim=dim ) )
            self.assertEqual( self.errors(), [], dim )
        for dim in ( -0.1, 1.01, '0.4', None, True ):
            self.set_reverse( frame_style=dict( BLUR_FRAME, dim=dim ) )
            self.assertTrue( any( "['dim']" in error for error in self.errors() ), dim )
        self.set_reverse( frame_style=dict( BLUR_FRAME, dim=0.4, dimm=0.4 ) )
        self.assertTrue( self.errors(), "a misspelt key is still caught" )

    def test_a_trigger_below_the_cache_floor_is_rejected( self ):
        betaconfig.global_min_prob = 0.12
        self.set_reverse( trigger={ 'min_prob': 0.10 } )
        self.assertTrue( any( 'global_min_prob' in e for e in self.errors() ) )
        self.set_reverse( trigger={ 'min_prob': 0.12 } )
        self.assertEqual( self.errors(), [] )

    def test_window_mistakes_are_rejected( self ):
        for windows in ( { 'not_a_class': {} }, { 'face_femme': { 'min_prop': 0.5 } },
                         { 'face_femme': { 'censor_style': { 'type': 'show', 'strength': 3 } } },
                         { 'face_femme': { 'censor_shape': 'hexagon' } },
                         { 'face_femme': { 'censor_style': [] } } ):
            self.set_reverse( windows=windows )
            self.assertTrue( self.errors(), windows )

    def test_a_window_is_always_uncensored( self ):
        errors = []
        bu_config._check_censor_style( SHOW, 'item_overrides', errors )
        self.assertTrue( errors, "'show' is a window style only" )
        self.set_reverse( windows={ 'face_femme': {
            'censor_style': [ dict( SHOW, weight=3, feather=0.2, shape='ellipse' ), dict( SHOW, weight=1 ) ] } } )
        self.assertEqual( self.errors(), [] )
        for style in ( MOSAIC, [ dict( SHOW, weight=3 ), dict( MOSAIC, weight=1 ) ] ):
            self.set_reverse( windows={ 'face_femme': { 'censor_style': style } } )
            self.assertTrue( any( "reverse['censor']" in e for e in self.errors() ), style )

    def test_censored_label_mistakes_are_rejected( self ):
        for censor in ( { 'labels': [ 'not_a_class' ] }, { 'labels': 'exposed_breast' },
                        { 'labls': None },
                        { 'labels': [ 'exposed_breast' ], 'overrides': { 'exposed_vulva': {} } },
                        { 'labels': [ 'exposed_breast' ], 'overrides': { 'exposed_breast': { 'min_prop': 1 } } },
                        { 'labels': [ 'exposed_breast' ], 'overrides': {
                            'exposed_breast': { 'censor_style': { 'type': 'show' } } } },
                        { 'labels': [ 'exposed_breast' ], 'overrides': { 'exposed_breast': { 'censor_shape': 'hexagon' } } } ):
            self.set_reverse( censor=censor )
            self.assertTrue( self.errors(), censor )
        self.set_reverse( censor={ 'labels': [ 'exposed_breast' ], 'overrides': {
            'exposed_breast': { 'censor_style': MOSAIC, 'censor_shape': 'ellipse', 'min_prob': 0.4 } } } )
        self.assertEqual( self.errors(), [] )

    def test_a_label_cannot_be_both_shown_and_censored( self ):
        self.set_reverse( windows={ 'exposed_breast': {} } )
        self.assertTrue( any( 'both a window' in e for e in self.errors() ) )
        # None means items_to_censor, which counts the same.
        betaconfig.items_to_censor = [ 'exposed_breast' ]
        self.set_reverse( censor={ 'labels': None }, windows={ 'exposed_breast': {} } )
        self.assertTrue( any( 'both a window' in e for e in self.errors() ) )

    def warnings( self ):
        bu_config.invalidate_config_caches()
        warnings = []
        bu_config._warn_reverse( warnings )
        return warnings

    def test_see_through_censored_styles_warn( self ):
        self.set_reverse( censor={ 'labels': [ 'exposed_breast' ], 'overrides': {
            'exposed_breast': { 'censor_style': { 'type': 'sticker', 'dir': '.' } } } } )
        self.assertTrue( any( 'see-through' in w for w in self.warnings() ) )
        self.set_reverse( censor={ 'labels': [ 'exposed_breast' ], 'overrides': {
            'exposed_breast': { 'censor_style': MOSAIC } } } )
        self.assertFalse( any( 'see-through' in w for w in self.warnings() ) )

    def test_no_windows_and_nothing_to_trigger_warn( self ):
        self.set_reverse( windows={} )
        self.assertTrue( any( 'no windows' in w for w in self.warnings() ) )
        self.set_reverse( censor={ 'labels': [] } )
        self.assertTrue( any( 'nothing can trigger' in w for w in self.warnings() ) )
        self.set_reverse( censor={ 'labels': [] }, trigger={ 'labels': [ 'face_femme' ] } )
        self.assertFalse( any( 'nothing can trigger' in w for w in self.warnings() ) )

    def test_promotion_targets_must_be_rendered_while_reverse_is_on( self ):
        backend = betaconfig.detector_backend.setdefault( bu_detector.selected_backend_name(), {} )
        backend['class_promotion'] = { 'exposed_vulva': [
            { 'from': 'covered_vulva', 'requires': [ { 'label': 'exposed_penis' } ] } ] }
        betaconfig.items_to_censor = [ 'exposed_vulva' ]
        self.set_reverse()
        errors = []
        bu_config._validate_class_promotion( errors, bu_config._valid_class_names() )
        self.assertTrue( any( "reverse mode's windows and censored labels" in e for e in errors ), errors )
        self.set_reverse( censor={ 'labels': None } )
        errors = []
        bu_config._validate_class_promotion( errors, bu_config._valid_class_names() )
        self.assertEqual( [ e for e in errors if 'censored labels' in e ], [] )

    def startup_errors( self, *cli_args ):
        """
        Every error validate_config() reports when run the way betatv's
        startup runs it (CLI flags applied first, then every check over
        every backend and tier), or [] when it passes.
        """
        parser = bu_cli.build_arg_parser( 'test', include_preview=True, include_logging=True )
        bu_cli.apply_cli_overrides( betaconfig, parser.parse_args( list( cli_args ) ) )
        bu_config.invalidate_config_caches()
        records = []
        handler = logging.Handler( logging.ERROR )
        handler.emit = records.append
        logger = bu_log.get_logger()
        logger.addHandler( handler )
        try:
            bu_config.validate_config()
        except SystemExit:
            return [ record.getMessage() for record in records ]
        finally:
            logger.removeHandler( handler )
        return []

    def test_betatv_startup_validates_the_shipped_config_with_reverse_on( self ):
        if self._saved['reverse'] is _MISSING:
            self.skipTest( 'no reverse block in betaconfig.py' )
        errors = [ e for e in self.startup_errors( '--reverse', 'on' ) if 'failed validation' not in e ]
        # Where setup has not exported the person model (a fresh checkout,
        # CI), that is the one error allowed, and it says how to fix it.
        self.assertEqual( [ e for e in errors if 'person model' not in e ], [], errors )
        self.assertTrue( all( 'Run setup' in e for e in errors ), errors )

    def test_show_windows_pass_the_backend_and_variant_tiers( self ):
        # Those tiers re-check each label's fully resolved settings, and a
        # show window's resolved style is 'show', which only the window
        # rules accept. The shipped config hit this on its first real run.
        backend = betaconfig.detector_backend.setdefault( bu_detector.selected_backend_name(), {} )
        backend.setdefault( 'item_overrides', {} )['face_femme'] = { 'min_prob': 0.3 }
        backend['variants'] = { bu_detector.variant_names()[0]: {
            'item_overrides': { 'face_masc': { 'min_prob': 0.25 } } } }
        self.set_reverse( censor={ 'labels': None },
                          windows={ 'face_femme': {}, 'face_masc': { 'censor_style': SHOW } } )
        self.assertEqual( self.startup_errors(), [] )

    def test_a_window_labels_tuning_is_still_checked_once_the_tiers_merge( self ):
        # Only the style is left to the window rules. A tuning combination
        # that is wrong only once the tiers merge is still reported.
        backend = betaconfig.detector_backend.setdefault( bu_detector.selected_backend_name(), {} )
        backend.setdefault( 'item_overrides', {} ).setdefault( 'face_femme', {} )['track_max_gap'] = 0.3
        self.set_reverse( censor={ 'labels': None },
                          windows={ 'face_femme': { 'interpolation_max_gap': 0.5 } } )
        errors = self.startup_errors()
        self.assertTrue( any( "resolved item_overrides['face_femme']['track_max_gap'] (0.3)" in e
                              for e in errors ), errors )

    def test_a_censored_override_is_checked_at_every_tier( self ):
        betaconfig.items_to_censor = [ 'exposed_breast' ]
        self.set_reverse( censor={ 'labels': None, 'overrides': {
            'exposed_breast': { 'censor_style': { 'type': 'pixel', 'pattern': 'triangles' } } } } )
        errors = self.startup_errors()
        self.assertTrue( any( "reverse['censor']['overrides']['exposed_breast']" in e for e in errors ), errors )

    def test_labels_without_a_window_keep_the_normal_checks_at_every_tier( self ):
        betaconfig.item_overrides = dict( betaconfig.item_overrides, exposed_vulva={ 'censor_style': SHOW } )
        self.set_reverse( censor={ 'labels': None } )
        errors = self.startup_errors()
        self.assertTrue( any( "resolved item_overrides['exposed_vulva']['censor_style']" in e
                              and "'show'" in e for e in errors ), errors )


class TestCli( ConfigHarness ):

    def test_the_flag_switches_reverse_without_editing_the_block_in_place( self ):
        block = self.set_reverse( enabled=False )
        parser = bu_cli.build_arg_parser( 'test' )
        bu_cli.apply_cli_overrides( betaconfig, parser.parse_args( [ '--reverse', 'on' ] ) )
        self.assertTrue( bu_reverse.enabled() )
        self.assertFalse( block['enabled'] )
        bu_cli.apply_cli_overrides( betaconfig, parser.parse_args( [ '--reverse', 'off' ] ) )
        self.assertFalse( bu_reverse.enabled() )


# ---------------------------------------------------------------------------
# One real run
# ---------------------------------------------------------------------------

from tests.test_betatv_end_to_end import BetaTvHarness, _StubDetectorModule  # noqa: E402


class _WindowedStub( _StubDetectorModule ):
    """
    Between 1.0 s and 2.0 s only: a censored exposed_breast box (x 40..93,
    y 30..70 at 160x120) and a face_femme window in the top-right corner.
    """

    def raw_boxes_for_imgs( self, imgs, size, session, ts ):
        boxes = [ box for box in super().raw_boxes_for_imgs( imgs, size, session, ts )
                  if 1.0 <= box['t'] <= 2.0 ]
        faces = [ dict( box, x=120.0, y=4.0, w=36.0, h=24.0, class_id='face_femme', score=0.95 )
                  for box in boxes ]
        return boxes + faces


def _decode( path ):
    capture = cv2.VideoCapture( path )
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append( frame.astype( np.int16 ) )
    capture.release()
    return frames


@unittest.skipUnless( shutil.which( 'ffmpeg' ), "ffmpeg is needed to build the test clip" )
class TestReverseRun( BetaTvHarness ):

    def setUp( self ):
        super().setUp()
        self._saved_reverse = copy.deepcopy( getattr( betaconfig, 'reverse', _MISSING ) )
        betaconfig.reverse = { 'enabled': True,
                               'frame_style': { 'type': 'blur', 'method': 'box', 'strength': 30 },
                               'fade_seconds': 0.0,
                               'trigger': { 'min_prob': 0.3, 'hold_seconds': 0.0 },
                               'censor': { 'labels': [ 'exposed_breast' ], 'overrides': {
                                   'exposed_breast': { 'censor_shape': 'box',
                                                       'censor_style': WHITE_BAR } } },
                               'windows': { 'face_femme': {} } }
        betaconfig.video_censor_fps = 10
        bu_config.invalidate_config_caches()
        self.detector = _WindowedStub()
        self._detector_patch.stop()
        from unittest import mock
        import betatv
        self._detector_patch = mock.patch.object( betatv.bu_detector, 'get_detector',
                                                  return_value=self.detector )
        self._detector_patch.start()

    def tearDown( self ):
        if self._saved_reverse is _MISSING:
            if hasattr( betaconfig, 'reverse' ):
                delattr( betaconfig, 'reverse' )
        else:
            betaconfig.reverse = self._saved_reverse
        super().tearDown()

    def render( self ):
        source = os.path.join( self.uncensored, 'clip.mp4' )
        subprocess.run( [ 'ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi', '-i',
                          'testsrc2=size=160x120:rate=10:duration=3', '-c:v', 'libx264',
                          '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', source ], check=True )
        self.assertEqual( self.run_one(), 'processed' )
        outputs = self.outputs()
        self.assertEqual( len( outputs ), 1 )
        self.assertTrue( outputs[0].endswith( '-reverse.mp4' ), outputs[0] )
        original = _decode( source )
        rendered = _decode( os.path.join( self.censored, outputs[0] ) )
        self.assertEqual( len( original ), len( rendered ) )

        def mean_diff( index, rows, cols ):
            return float( np.abs( rendered[index][rows, cols] - original[index][rows, cols] ).mean() )

        return rendered, mean_diff

    # Probes: inside the censored box, inside the face window, and the
    # bottom strip, clear of both.
    CENSORED = ( slice( 36, 64 ), slice( 46, 86 ) )
    FACE = ( slice( 8, 24 ), slice( 124, 152 ) )
    OUTSIDE = ( slice( 85, 120 ), slice( 0, 160 ) )

    def test_triggered_frames_are_obscured_except_windows_and_censored_labels_keep_their_style( self ):
        rendered, mean_diff = self.render()
        before, during, after = 3, 15, 28   # frames: 0.3 s, 1.5 s, 2.8 s
        self.assertLess( mean_diff( before, *self.OUTSIDE ), 4.0, "untriggered frames pass through" )
        # The first detection is at 1.0 s; the frame one sample earlier
        # may already show the target, so it must be obscured too.
        self.assertGreater( mean_diff( 9, *self.OUTSIDE ), 15.0, "the lead covers the sample before" )
        self.assertLess( mean_diff( 8, *self.OUTSIDE ), 4.0, "and no earlier" )
        self.assertGreater( mean_diff( during, *self.OUTSIDE ), 15.0, "a triggered frame is obscured" )
        self.assertLess( mean_diff( during, *self.FACE ), 6.0, "the window shows the source" )
        rows, cols = self.CENSORED
        self.assertGreater( float( rendered[during][rows, cols].mean() ), 240.0,
                            "the censored label keeps its own style, not the frame style" )
        self.assertLess( mean_diff( after, *self.OUTSIDE ), 4.0, "the trigger lets go" )

    def test_a_censored_box_obscures_its_frames_with_no_trigger_detection( self ):
        betaconfig.reverse['trigger']['min_prob'] = 0.99   # above every score the stub reports
        bu_config.invalidate_config_caches()
        rendered, mean_diff = self.render()
        self.assertGreater( mean_diff( 15, *self.OUTSIDE ), 15.0, "a censored box triggers by itself" )
        self.assertLess( mean_diff( 3, *self.OUTSIDE ), 4.0 )
        self.assertLess( mean_diff( 28, *self.OUTSIDE ), 4.0 )


if __name__ == '__main__':
    unittest.main()


class TestBetastareReverse( ConfigHarness ):

    def stare( self, raw ):
        from unittest import mock
        import betastare
        tmp = tempfile.mkdtemp( prefix='betasuite-stare-' )
        self.addCleanup( shutil.rmtree, tmp, True )
        source_dir, out_dir = os.path.join( tmp, 'in' ), os.path.join( tmp, 'out' )
        os.makedirs( source_dir )
        os.makedirs( out_dir )
        cv2.imwrite( os.path.join( source_dir, 'still.png' ), self.source )
        with mock.patch.object( betastare, '_raw_boxes_for_all_sizes', return_value=( [ raw ], False ) ), \
             mock.patch.object( betastare.bu_detector, 'get_picture_sizes', return_value=[ 320 ] ):
            betastare._process_one_file( source_dir, 'still.png', out_dir, 1, 1, None,
                                         betastare.bu_log.get_logger() )
        written = os.listdir( out_dir )
        self.assertEqual( len( written ), 1 )
        self.assertIn( '-reverse', written[0] )
        return cv2.imread( os.path.join( out_dir, written[0] ) )

    def setUp( self ):
        super().setUp()
        self.source = _noise()
        self.set_reverse( frame_style=BLACK_FRAME, trigger={ 'min_prob': 0.3 },
                          censor={ 'labels': [ 'exposed_breast' ], 'overrides': { 'exposed_breast': {
                              'min_prob': 0.2, 'censor_shape': 'box', 'censor_style': WHITE_BAR } } } )

    def test_a_still_with_a_trigger_detection_is_obscured_except_its_window( self ):
        out = self.stare( [ _raw( 'face_femme', 0.9, 0, x=10, y=10, w=50, h=30 ),
                            _raw( 'exposed_breast', 0.9, 0, x=90, y=50, w=40, h=30 ) ] )
        np.testing.assert_array_equal( out[ 10:40, 10:60 ], self.source[ 10:40, 10:60 ] )
        self.assertTrue( np.all( out[ 52:78, 92:128 ] == 255 ), "censored with its own style" )
        self.assertTrue( np.all( out[ 95:, : ] == 0 ) )

    def test_a_still_with_only_a_censored_box_is_obscured( self ):
        # Scores above the censored label's min_prob, below the trigger's.
        out = self.stare( [ _raw( 'exposed_breast', 0.25, 0, x=90, y=50, w=40, h=30 ) ] )
        self.assertTrue( np.all( out[ 95:, : ] == 0 ) )
