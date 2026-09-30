"""
test_shot_cut_style_reroll.py - shot cuts must be visible to tracking,
and must produce style variety without costing censor coverage.

THE BUG THIS PINS
-----------------
Boxes carry three times: 't' (the frame's real timestamp), and
'start'/'end', which are t -/+ time_safety/2. The safety padding exists
so a censor appears slightly before and lingers slightly after the
detection.

Every cut check used to be written against the PADDED bounds:

    cut_between( track['end'], box['start'] )

For two adjacent sampled frames that interval is
(t + safety/2, t + step - safety/2], which is EMPTY whenever
time_safety >= the frame step. At video_censor_fps = 9 the step is
0.111s and exposed_breast's time_safety is 0.18s, so the interval was
inverted on every single frame pair and no cut was ever seen.

The visible symptom was not a tracking complaint. It was style
monoculture: a 30s preview of a 16-woman compilation with 50 detected
cuts produced 3 tracks and ONE independent style resolve, so every woman
in the clip wore the same censor style. The configured weights and the
style resolver were both correct the whole time; there was simply only
one draw.

These tests therefore assert on TIMESTAMPS, and deliberately use a
time_safety LARGER than the frame step - the exact condition that
silenced the original check.
"""

import os
import sys
import unittest
from collections import Counter
from unittest import mock

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )

import betaconfig
import betautils_censor as bu_censor
import betautils_track as bu_track


FPS = 9.0
STEP = 1.0 / FPS


def breast_boxes( frames=270, pair_offsets=( 0, 260 ), drift=1 ):
    """A continuously-detected breast pair, as a compilation produces."""
    boxes = []
    for index in range( frames ):
        t = index / FPS
        for offset in pair_offsets:
            boxes.append( {
                'class_id': 'exposed_breast', 'score': 0.8, 't': t,
                'x': 600 + offset + ( index % 5 )*drift, 'y': 400,
                'w': 300, 'h': 280,
            } )
    return boxes


def style_key( box ):
    style = box.get( 'censor_style' ) or {}
    return ( style.get( 'type' ),
             style.get( 'pattern' ) or style.get( 'method' ) or '' )


class TestTimeSafetyDoesNotHideCuts( unittest.TestCase ):
    """
    The regression itself, stated as arithmetic rather than behaviour so
    the failure message points straight at the cause.
    """

    def test_the_padded_interval_inverts_at_shipped_settings( self ):
        # Not an assertion about desired behaviour - a demonstration that
        # the OLD comparison could not have worked at these settings. If
        # this ever stops being true (fps raised, time_safety lowered),
        # the padded form would start working by accident, which is
        # exactly the kind of silent coupling worth knowing about.
        overrides = betaconfig.detector_backend['nudenet_v3']['item_overrides']
        time_safety = overrides['exposed_breast']['time_safety']
        step = 1.0 / betaconfig.video_censor_fps
        self.assertGreaterEqual(
            time_safety, step,
            "exposed_breast time_safety (%.3f) is now below the frame step "
            "(%.3f). The padded cut check would coincidentally work again; "
            "re-read this test's docstring before trusting it."
            % ( time_safety, step ) )

    def test_a_cut_is_visible_between_adjacent_frames( self ):
        # The real fix: compare timestamps, which are unpadded.
        raw = breast_boxes( frames=90 )
        cuts = [ 3.0 ]
        with_cuts, stats_with = bu_track.prepare_boxes_for_render(
            list( raw ), 1920, 1080, cuts, backend_name='nudenet_v3' )
        without, stats_without = bu_track.prepare_boxes_for_render(
            list( raw ), 1920, 1080, None, backend_name='nudenet_v3' )
        self.assertGreater(
            stats_with['tracks'], stats_without['tracks'],
            "a cut in the middle of a continuously-detected track did not "
            "change tracking at all - the cut check is not seeing it" )


class TestCutsProduceStyleVariety( unittest.TestCase ):

    def test_many_cuts_yield_many_styles( self ):
        raw = breast_boxes()
        cuts = [ round( 30.0*k/50, 3 ) for k in range( 1, 51 ) ]
        boxes, _stats = bu_track.prepare_boxes_for_render(
            list( raw ), 1920, 1080, cuts, backend_name='nudenet_v3' )
        seen = Counter( style_key( b ) for b in boxes )
        self.assertGreater(
            len( seen ), 1,
            "50 cuts over 30s produced a single censor style; this is the "
            "monoculture symptom the timestamp fix exists to remove" )

    def test_no_cuts_yields_one_style_per_track( self ):
        # The other half of the contract: without a cut, a continuing
        # track must NOT change style. Re-rolling mid-shot is the flicker
        # the whole style-resolution design exists to prevent.
        raw = breast_boxes()
        boxes, _stats = bu_track.prepare_boxes_for_render(
            list( raw ), 1920, 1080, None, backend_name='nudenet_v3' )
        seen = Counter( style_key( b ) for b in boxes )
        self.assertEqual(
            len( seen ), 1,
            "a continuing track changed style with no shot cut present - "
            "that is mid-scene flicker, not variety" )

    def test_style_changes_only_ever_land_on_a_cut( self ):
        raw = breast_boxes()
        cuts = [ round( 30.0*k/50, 3 ) for k in range( 1, 51 ) ]
        boxes, _stats = bu_track.prepare_boxes_for_render(
            list( raw ), 1920, 1080, cuts, backend_name='nudenet_v3' )

        by_time = {}
        for box in boxes:
            by_time.setdefault( round( box.get( 't', box['start'] ), 4 ), [] ).append( box )

        previous = None
        off_cut = []
        for t in sorted( by_time ):
            key = style_key( by_time[t][0] )
            if previous is not None and key != previous:
                if not any( abs( t - cut ) <= STEP for cut in cuts ):
                    off_cut.append( t )
            previous = key
        self.assertEqual(
            off_cut, [],
            "style changed at %s, which is not adjacent to any shot cut"
            % ( off_cut[:5], ) )

    def test_a_breast_pair_shares_one_style_in_every_frame( self ):
        # paired_style's contract. A re-roll must apply to both sides of
        # the pair in the same frame, or one breast is pixelated while
        # the other is blurred.
        raw = breast_boxes()
        cuts = [ round( 30.0*k/50, 3 ) for k in range( 1, 51 ) ]
        boxes, _stats = bu_track.prepare_boxes_for_render(
            list( raw ), 1920, 1080, cuts, backend_name='nudenet_v3' )

        by_time = {}
        for box in boxes:
            by_time.setdefault( round( box.get( 't', box['start'] ), 4 ), [] ).append( box )

        mismatched = [ t for t, group in by_time.items()
                       if len( group ) > 1 and len( { style_key( b ) for b in group } ) > 1 ]
        self.assertEqual(
            mismatched, [],
            "%d frame(s) had the two breasts wearing different styles"
            % ( len( mismatched ), ) )


class TestCutsCostNoCoverage( unittest.TestCase ):
    """
    The constraint that outranks variety: a perf or cosmetic change may
    never reduce what gets censored.
    """

    def test_rendered_box_count_is_unchanged_by_cuts( self ):
        raw = breast_boxes()
        cuts = [ round( 30.0*k/50, 3 ) for k in range( 1, 51 ) ]
        with_cuts, _ = bu_track.prepare_boxes_for_render(
            list( raw ), 1920, 1080, cuts, backend_name='nudenet_v3' )
        without, _ = bu_track.prepare_boxes_for_render(
            list( raw ), 1920, 1080, None, backend_name='nudenet_v3' )
        self.assertEqual(
            len( with_cuts ), len( without ),
            "re-rolling styles at cuts changed how many boxes get rendered; "
            "variety must never cost coverage" )

    def test_censored_area_is_unchanged_by_cuts( self ):
        # Count is not enough: a box could survive but shrink.
        #
        # THIS TEST WAS FLAKY TWICE, AND BOTH BOUNDS WERE THE WRONG FIX.
        #
        # Each style carries its own width/height_area_safety, and which
        # style a track rolls is random. The no-cuts side resolves ONE
        # style for its ONE track, so its total area is a single draw:
        # measured across runs it ranged 20.4M to 54.9M, a 2.7x swing,
        # entirely from which style that one draw landed on (sticker
        # carries 0.13 safety, one pixel entry 0.05, another 0.0 - and
        # 1.13 on each axis is ~1.28x area).
        #
        # Widening the bound (5% -> 25%) did not fix it and could not:
        # no fixed ratio survives a baseline that itself varies 2.7x.
        # Each loosening also made the test less able to catch the thing
        # it exists for.
        #
        # The real fix is to remove the randomness rather than tolerate
        # it. Pinning one style makes both sides comparable, so the
        # assertion can go back to being tight - and a tight assertion
        # is what actually catches a shrinking box.
        style = { 'type': 'blur', 'method': 'gaussian', 'strength': 22,
                  'width_area_safety': 0.0, 'height_area_safety': 0.0 }
        with mock.patch.object( bu_censor, 'resolve_censor_style',
                                return_value=dict( style ) ):
            raw = breast_boxes()
            cuts = [ round( 30.0*k/50, 3 ) for k in range( 1, 51 ) ]
            with_cuts, _ = bu_track.prepare_boxes_for_render(
                list( raw ), 1920, 1080, cuts, backend_name='nudenet_v3' )
            without, _ = bu_track.prepare_boxes_for_render(
                list( raw ), 1920, 1080, None, backend_name='nudenet_v3' )

        def area( boxes ):
            return sum( b['w']*b['h'] for b in boxes )

        with_area, without_area = area( with_cuts ), area( without )
        self.assertGreater(
            with_area, without_area * 0.99,
            "censored area fell %.1f%% when cuts re-rolled styles, with "
            "the style pinned so area safety is identical on both sides - "
            "this is a real loss of coverage, not style variance"
            % ( 100.0*(1 - with_area/max( without_area, 1 )), ) )


if __name__ == '__main__':
    unittest.main()
