"""
test_size_smoothing_and_clamp.py - size is smoothed separately from
position, and the result always stays inside the frame.

WHY SIZE IS SMOOTHED SEPARATELY
-------------------------------
Position and size were smoothed with one alpha. That made the censor
box breathe: the detector's width and height wobble a few pixels frame
to frame even on a motionless subject, and every wobble moved the
composited edge. A blurred edge against a sharp background is a
high-contrast boundary, so it read as flicker - on every style, with
blur merely the most visible.

Holding size steadier is close to free. A censor a few pixels larger
than strictly necessary still covers the subject; one that breathes
reads as broken. So size_alpha defaults well below alpha.

WHY THE CLAMP EXISTS
--------------------
Splitting the alphas introduced a crash. Position and size now move at
different rates, so they can disagree about where the box ends even
when every raw detection was inside the frame. On a box pinned to an
edge, y follows a new detection quickly while h still carries the old
value, and y+h overshoots - up to 17px measured at the shipped alphas.

That surfaced as a render failure rather than a visual glitch:
censor_image builds its feather mask from box w/h but slices the region
out of the frame, and numpy silently truncates a slice at the array
edge. A (79,94,1) mask against a (77,94,3) region raised ValueError and
failed the whole chunk. The hex style failed differently, inside
np.bincount, which is why it looked like two unrelated bugs.

These tests therefore assert the invariant (never outside the frame)
rather than the symptom (a particular exception), so any future style
that composites differently is covered too.
"""

import copy
import os
import sys
import unittest
from unittest import mock

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )

import betaconfig

import betautils_censor as bu_censor
import betautils_config as bu_config
import betautils_track as bu_track


VID_W, VID_H = 1920, 1080


def _box( label, t, x, y, w, h, score=0.9 ):
    return { 'class_id': label, 'score': score, 't': t,
             'x': x, 'y': y, 'w': w, 'h': h }


class SmoothingHarness( unittest.TestCase ):

    def setUp( self ):
        self._saved_backend = copy.deepcopy( betaconfig.detector_backend )
        self._saved = { name: getattr( betaconfig, name, None )
                        for name in ( 'default_position_smoothing',
                                      'default_size_smoothing' ) }
        self.backend = betaconfig.detector_backend['selected']
        bu_config.invalidate_config_caches()

    def tearDown( self ):
        betaconfig.detector_backend = self._saved_backend
        for name, value in self._saved.items():
            if value is None:
                if hasattr( betaconfig, name ):
                    delattr( betaconfig, name )
            else:
                setattr( betaconfig, name, value )
        bu_config.invalidate_config_caches()

    def _override( self, **values ):
        betaconfig.detector_backend = copy.deepcopy( betaconfig.detector_backend )
        overrides = betaconfig.detector_backend[self.backend].setdefault(
            'item_overrides', {} ).setdefault( 'exposed_breast', {} )
        overrides.update( values )
        bu_config.invalidate_config_caches()


class TestSizeIsSmoothedMoreThanPosition( SmoothingHarness ):

    def test_size_smoothing_defaults_below_position_smoothing( self ):
        settings = bu_track._LabelSettings( 'exposed_breast', self.backend, 1.0/9 )
        self.assertLess(
            settings.size_alpha, settings.alpha,
            "size is being smoothed at least as loosely as position, which "
            "is the box-breathing that reads as flicker" )

    def test_size_smoothing_is_configurable_per_label( self ):
        self._override( size_smoothing=0.05 )
        settings = bu_track._LabelSettings( 'exposed_breast', self.backend, 1.0/9 )
        self.assertAlmostEqual( settings.size_alpha, 0.05 )

    def test_a_wobbling_detection_produces_a_steadier_box( self ):
        # The behavioural claim. Feed a subject that does not move but
        # whose detected size oscillates, and the rendered width must
        # vary less than the input did.
        self._override( track_max_gap=5.0, interpolation_max_gap=0.0,
                        min_track_hits=1, size_smoothing=0.1 )
        raw = []
        for index in range( 40 ):
            wobble = 40 if index % 2 else -40
            raw.append( _box( 'exposed_breast', index/9.0,
                              600, 400, 300 + wobble, 280 ) )

        boxes, _stats = bu_track.prepare_boxes_for_render(
            copy.deepcopy( raw ), VID_W, VID_H, backend_name=self.backend )
        rendered = [ b for b in boxes if not b.get( 'interpolated' ) ]
        self.assertGreater( len( rendered ), 10 )

        widths = [ b['w'] for b in rendered[2:] ]
        spread = max( widths ) - min( widths )
        self.assertLess(
            spread, 80,
            "the rendered width swung %d px against an input swing of 80; "
            "size smoothing is not damping the detector's wobble"%( spread, ) )


class TestSmoothedBoxesStayInsideTheFrame( SmoothingHarness ):
    """
    The crash guard. Every rendered box must satisfy
    0 <= x, 0 <= y, x+w <= vid_w, y+h <= vid_h - whatever the alphas do.
    """

    def assert_inside_frame( self, boxes, width=VID_W, height=VID_H ):
        for box in boxes:
            self.assertGreaterEqual( box['x'], 0, box )
            self.assertGreaterEqual( box['y'], 0, box )
            self.assertGreaterEqual( box['w'], 1, box )
            self.assertGreaterEqual( box['h'], 1, box )
            self.assertLessEqual(
                box['x'] + box['w'], width,
                "box runs %d px past the right edge; censor_image would "
                "build a mask wider than the region numpy hands it"
                %( box['x'] + box['w'] - width, ) )
            self.assertLessEqual(
                box['y'] + box['h'], height,
                "box runs %d px past the bottom edge; this is the render "
                "crash the post-smoothing clamp exists to prevent"
                %( box['y'] + box['h'] - height, ) )

    def test_a_box_pinned_to_the_bottom_edge_never_overshoots( self ):
        # The exact failing shape: raw y+h sits on the boundary, and the
        # detection alternates between two heights so position and size
        # disagree about where the bottom is.
        self._override( track_max_gap=5.0, interpolation_max_gap=0.0,
                        min_track_hits=1 )
        raw = []
        for index in range( 40 ):
            height = 280 if index % 2 else 200
            raw.append( _box( 'exposed_breast', index/9.0,
                              600, VID_H - height, 300, height ) )
        boxes, _stats = bu_track.prepare_boxes_for_render(
            copy.deepcopy( raw ), VID_W, VID_H, backend_name=self.backend )
        self.assert_inside_frame( boxes )

    def test_a_box_pinned_to_the_right_edge_never_overshoots( self ):
        self._override( track_max_gap=5.0, interpolation_max_gap=0.0,
                        min_track_hits=1 )
        raw = []
        for index in range( 40 ):
            width = 300 if index % 2 else 220
            raw.append( _box( 'exposed_breast', index/9.0,
                              VID_W - width, 400, width, 280 ) )
        boxes, _stats = bu_track.prepare_boxes_for_render(
            copy.deepcopy( raw ), VID_W, VID_H, backend_name=self.backend )
        self.assert_inside_frame( boxes )

    def test_a_box_in_the_top_left_corner_never_goes_negative( self ):
        self._override( track_max_gap=5.0, interpolation_max_gap=0.0,
                        min_track_hits=1 )
        raw = []
        for index in range( 40 ):
            offset = 0 if index % 2 else 30
            raw.append( _box( 'exposed_breast', index/9.0,
                              offset, offset, 200, 200 ) )
        boxes, _stats = bu_track.prepare_boxes_for_render(
            copy.deepcopy( raw ), VID_W, VID_H, backend_name=self.backend )
        self.assert_inside_frame( boxes )

    def test_extreme_mismatched_alphas_still_stay_inside( self ):
        # Deliberately pathological: position snaps instantly while size
        # barely moves. The clamp must hold as an invariant, not as a
        # property of the shipped values.
        self._override( track_max_gap=5.0, interpolation_max_gap=0.0,
                        min_track_hits=1, position_smoothing=1.0,
                        size_smoothing=0.01 )
        raw = []
        for index in range( 60 ):
            if index % 2:
                raw.append( _box( 'exposed_breast', index/9.0, 10, 10, 600, 600 ) )
            else:
                raw.append( _box( 'exposed_breast', index/9.0,
                                  VID_W - 120, VID_H - 120, 100, 100 ) )
        boxes, _stats = bu_track.prepare_boxes_for_render(
            copy.deepcopy( raw ), VID_W, VID_H, backend_name=self.backend )
        self.assert_inside_frame( boxes )

    def test_a_tiny_frame_still_produces_usable_boxes( self ):
        # Guards the max(1, ...) floors: clamping must never produce a
        # zero or negative dimension, which would slice to an empty
        # array rather than raising anywhere obvious.
        self._override( track_max_gap=5.0, interpolation_max_gap=0.0,
                        min_track_hits=1 )
        raw = [ _box( 'exposed_breast', index/9.0, 30, 30, 80, 80 )
                for index in range( 20 ) ]
        boxes, _stats = bu_track.prepare_boxes_for_render(
            copy.deepcopy( raw ), 64, 48, backend_name=self.backend )
        self.assert_inside_frame( boxes, width=64, height=48 )


class TestClampDoesNotCostCoverage( SmoothingHarness ):
    """
    The standing constraint: a fix for flicker or a crash may never
    reduce what actually gets censored.
    """

    def test_every_detection_still_produces_a_box( self ):
        self._override( track_max_gap=5.0, interpolation_max_gap=0.0,
                        min_track_hits=1 )
        raw = [ _box( 'exposed_breast', index/9.0, 600, 400, 300, 280 )
                for index in range( 30 ) ]
        boxes, _stats = bu_track.prepare_boxes_for_render(
            copy.deepcopy( raw ), VID_W, VID_H, backend_name=self.backend )
        rendered = [ b for b in boxes if not b.get( 'interpolated' ) ]
        self.assertEqual( len( rendered ), len( raw ) )

    def test_an_edge_box_still_covers_most_of_its_detection( self ):
        # Clamping trims a box to the frame; it must not shrink it to a
        # token sliver. The visible part of the detection is what has to
        # survive.
        #
        # The style is PINNED because area safety is part of the resolved
        # style and the draw is random. The live config carries 14 entries
        # with negative area safety, and the narrowest (a bar at
        # width_area_safety -0.55) has an area factor of 0.450 - below
        # this test's 0.5 floor. So roughly one run in thirty failed here
        # for a legitimate style choice rather than a clamping bug, which
        # is the same unseeded-draw flakiness that once made
        # test_censored_area_is_unchanged_by_cuts unreliable.
        style = { 'type': 'blur', 'method': 'gaussian', 'strength': 22,
                  'width_area_safety': 0.0, 'height_area_safety': 0.0 }
        self.addCleanup( mock.patch.stopall )
        mock.patch.object( bu_censor, 'resolve_censor_style',
                           return_value=dict( style ) ).start()
        self._override( track_max_gap=5.0, interpolation_max_gap=0.0,
                        min_track_hits=1 )
        raw = [ _box( 'exposed_breast', index/9.0, VID_W - 300, 400, 300, 280 )
                for index in range( 30 ) ]
        boxes, _stats = bu_track.prepare_boxes_for_render(
            copy.deepcopy( raw ), VID_W, VID_H, backend_name=self.backend )
        rendered = [ b for b in boxes if not b.get( 'interpolated' ) ]
        for box in rendered[2:]:
            self.assertGreater(
                box['w'] * box['h'], 300 * 280 * 0.5,
                "an edge-pinned censor lost more than half its area to "
                "clamping; that is uncensored subject" )


if __name__ == '__main__':
    unittest.main()
