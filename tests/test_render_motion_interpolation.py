"""
test_render_motion_interpolation.py - the censor box moves at OUTPUT frame
rate, not at detection sample rate.

THE BUG THIS PINS
-----------------
Detection samples at video_censor_fps; the render writes every source
frame. On a 60fps source sampled at 9fps that is 6.67 output frames per
sample. Two things followed, and the second was the one that mattered.

1. Held geometry. Each box's rectangle was whatever its sample said, held
   until the next sample replaced it, so the box teleported. Measured on a
   real 640m cache (longest breast track, 152 samples): 5.1px median jump,
   10.4px p90, 14.1px max, nine times a second.

2. Overlapping samples. time_safety makes a box live for LONGER than the
   sampling interval - 0.17s against 0.111s, 153% coverage - so for most
   output frames TWO consecutive samples of the same object are live at
   once. Measured: 66 of 120 frames. The blur overlap strategy unions
   them, so the rendered rectangle grew and shrank as the overlap came and
   went: 78x86, 78x87, 77x86, 79x91 on consecutive frames, oscillating at
   the sample rate.

(2) is the pulsing. It is also why no blur setting ever touched it:
strength, method and the edge margin all change what FILLS the box, and
this moves the box's boundary. position_smoothing could not fix it either,
because it smooths between SAMPLES, not between output frames.

WHY THE COLLAPSE IS CONDITIONAL
-------------------------------
Collapsing two live samples to one always removes the oscillation, but
when the object is moving fast the two straddle the motion and their union
covers the whole swept path. That is a real safety margin. Collapsing
unconditionally gave up 8265px on one fast-motion frame of the real cache.
So the collapse applies only where the union is barely bigger than one box
(render_motion_collapse_max_growth, default 1.15), which is exactly the
near-still case where the eye notices pulsing.

Measured across the whole real slice at that default: median
frame-to-frame rendered-area step 0px (159px with no collapse), coverage
98.31% of the un-collapsed behaviour.
"""

import os
import sys
import unittest

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )

import betaconfig

import betautils_config as bu_config
import betautils_render as bu_render


def _box( label, track_id, t, x, y, w=80, h=80, safety=0.17, vid=( 1920, 1080 ) ):
    return { 'label': label, '_track_id': track_id, 't': t,
             'start': t - safety/2, 'end': t + safety/2,
             'x': x, 'y': y, 'w': w, 'h': h,
             '_vid_w': vid[0], '_vid_h': vid[1],
             'censor_style': { 'type': 'blur', 'method': 'gaussian', 'strength': 60 },
             'censor_shape': 'box', 'style_scale_w': w, 'style_scale_h': h,
             'censor_sticker_seed': 0.5 }


def _longest_run_of_zeros( steps ):
    longest = run = 0
    for step in steps:
        run = run + 1 if step == 0 else 0
        longest = max( longest, run )
    return longest


class TestTrackKey( unittest.TestCase ):
    """
    _track_id restarts from 0 for every label, so it cannot identify an
    object on its own.
    """

    def test_the_key_includes_the_label( self ):
        breast = _box( 'exposed_breast', 0, 1.0, 100, 100 )
        vulva = _box( 'covered_vulva', 0, 1.0, 600, 400 )
        self.assertNotEqual( bu_render.track_key( breast ),
                             bu_render.track_key( vulva ) )

    def test_two_labels_sharing_an_id_get_separate_timelines( self ):
        # On the real cache, track id 0 carried a breast at (571,42) and a
        # covered_vulva at (617,446) at 150 separate instants. Keying on
        # the id alone would interpolate between them, sliding a censor
        # across the body.
        boxes = [ _box( 'exposed_breast', 0, 1.0, 100, 100 ),
                  _box( 'exposed_breast', 0, 1.1, 110, 100 ),
                  _box( 'covered_vulva', 0, 1.0, 600, 400 ),
                  _box( 'covered_vulva', 0, 1.1, 601, 400 ) ]
        timelines = bu_render.build_motion_index( boxes )
        self.assertEqual( len( timelines ), 2 )
        self.assertIn( ( 'exposed_breast', 0 ), timelines )
        self.assertIn( ( 'covered_vulva', 0 ), timelines )


class TestMotionIndex( unittest.TestCase ):

    def test_single_sample_objects_are_omitted( self ):
        # Nothing to interpolate between, and leaving them out keeps the
        # per-frame lookup small.
        boxes = [ _box( 'exposed_breast', 0, 1.0, 100, 100 ) ]
        self.assertEqual( bu_render.build_motion_index( boxes ), {} )

    def test_timelines_are_sorted_by_time( self ):
        boxes = [ _box( 'exposed_breast', 0, 1.2, 120, 100 ),
                  _box( 'exposed_breast', 0, 1.0, 100, 100 ),
                  _box( 'exposed_breast', 0, 1.1, 110, 100 ) ]
        times, entries = bu_render.build_motion_index( boxes )[ ( 'exposed_breast', 0 ) ]
        self.assertEqual( times, sorted( times ) )
        self.assertEqual( [ entry['t'] for entry in entries ], times )

    def test_a_box_with_no_timestamp_is_skipped( self ):
        box = _box( 'exposed_breast', 0, 1.0, 100, 100 )
        del box['t']
        self.assertEqual( bu_render.build_motion_index( [ box ] ), {} )


class TestGeometrySlides( unittest.TestCase ):

    def setUp( self ):
        self.boxes = [ _box( 'exposed_breast', 0, 1.0, 100, 200 ),
                       _box( 'exposed_breast', 0, 1.1, 140, 200 ) ]
        self.timelines = bu_render.build_motion_index( self.boxes )

    def _at( self, current_time, span=0.25 ):
        # Only the EARLIER sample, so this exercises the slide in isolation.
        # At 1.05 both samples are genuinely live under time_safety 0.17,
        # and the collapse gate deliberately keeps both here because 40px
        # in 0.1s is fast motion - that path is TestOverlappingSamples-
        # Collapse's job, not this class's.
        return bu_render.interpolate_boxes_for_frame(
            [ self.boxes[0] ], self.timelines, current_time, span )

    def test_halfway_between_samples_is_halfway_in_space( self ):
        result = self._at( 1.05 )
        self.assertEqual( len( result ), 1 )
        self.assertEqual( result[0]['x'], 120 )
        self.assertTrue( result[0]['_motion_interpolated'] )

    def test_the_slide_is_monotonic_across_the_interval( self ):
        xs = [ self._at( 1.0 + step/100.0 )[0]['x'] for step in range( 0, 11 ) ]
        for earlier, later in zip( xs, xs[1:] ):
            self.assertLessEqual( earlier, later )
        self.assertLess( xs[0], xs[-1] )

    def test_landing_exactly_on_the_later_sample_does_not_snap_back( self ):
        # fraction == 1. An earlier draft bailed out here and returned the
        # EARLIER sample's rectangle, so on one frame per sample the box
        # jumped a full interval backwards (100 after reaching 136). That
        # is flicker, produced by the flicker fix.
        result = self._at( 1.1 )[0]
        self.assertEqual( result['x'], 140 )
        self.assertTrue( result['_motion_interpolated'] )

    def test_style_and_seed_come_from_the_earlier_sample( self ):
        # Only the rectangle moves. A style that changed mid-slide would
        # be the flicker this exists to remove.
        result = self._at( 1.05 )[0]
        self.assertEqual( result['censor_sticker_seed'],
                          self.boxes[0]['censor_sticker_seed'] )
        self.assertEqual( result['censor_style'], self.boxes[0]['censor_style'] )

    def test_it_never_extrapolates_before_the_first_sample( self ):
        result = self._at( 0.95 )
        self.assertTrue( all( not box.get( '_motion_interpolated' )
                              for box in result ) )

    def test_it_never_extrapolates_after_the_last_sample( self ):
        live = [ self.boxes[-1] ]
        result = bu_render.interpolate_boxes_for_frame(
            live, self.timelines, 1.15, 0.25 )
        self.assertFalse( result[0].get( '_motion_interpolated' ) )

    def test_a_long_hole_is_held_not_slid( self ):
        # Across a gap the object genuinely was not seen, sliding through
        # would draw a censor along a path nobody detected.
        boxes = [ _box( 'exposed_breast', 0, 1.0, 100, 200 ),
                  _box( 'exposed_breast', 0, 4.0, 900, 200 ) ]
        timelines = bu_render.build_motion_index( boxes )
        result = bu_render.interpolate_boxes_for_frame(
            [ boxes[0] ], timelines, 1.05, max_span=0.25 )
        self.assertFalse( result[0].get( '_motion_interpolated' ) )
        self.assertEqual( result[0]['x'], 100 )

    def test_a_blend_stays_inside_the_frame( self ):
        # Both samples are in frame (1830+80=1910 < 1920); the LATER one is
        # flush against the right edge. Blending toward it must not push the
        # rectangle past 1920 - cv2 would index out of the array.
        boxes = [ _box( 'exposed_breast', 0, 1.0, 1830, 990, w=80, h=80 ),
                  _box( 'exposed_breast', 0, 1.1, 1840, 1000, w=80, h=80 ) ]
        timelines = bu_render.build_motion_index( boxes )
        blended = 0
        for step in range( 0, 11 ):
            result = bu_render.interpolate_boxes_for_frame(
                [ boxes[0] ], timelines, 1.0 + step/100.0, 0.25 )[0]
            if result.get( '_motion_interpolated' ): blended += 1
            self.assertLessEqual( result['x'] + result['w'], 1920 )
            self.assertLessEqual( result['y'] + result['h'], 1080 )
            self.assertGreaterEqual( result['w'], 1 )
            self.assertGreaterEqual( result['h'], 1 )
        # Guard the guard: if nothing blended, the loop proved nothing.
        self.assertGreater( blended, 0 )

    def test_a_blend_past_the_edge_is_clamped_not_dropped( self ):
        # A detector box already hanging off the frame stays whatever it was
        # on pass-through frames, but any BLEND of it is clamped in bounds.
        boxes = [ _box( 'exposed_breast', 0, 1.0, 1900, 1040, w=80, h=80 ),
                  _box( 'exposed_breast', 0, 1.1, 1910, 1050, w=80, h=80 ) ]
        timelines = bu_render.build_motion_index( boxes )
        result = bu_render.interpolate_boxes_for_frame(
            [ boxes[0] ], timelines, 1.05, 0.25 )[0]
        self.assertTrue( result['_motion_interpolated'] )
        self.assertLessEqual( result['x'] + result['w'], 1920 )
        self.assertLessEqual( result['y'] + result['h'], 1080 )
        self.assertGreaterEqual( result['w'], 1 )
        self.assertGreaterEqual( result['h'], 1 )


class TestGeometryHelperDirectly( unittest.TestCase ):
    """
    _interpolated_geometry's guards are unreachable from
    interpolate_boxes_for_frame (it filters sample_time >= current_time
    before calling), but the helper is callable on its own, so its
    contract is pinned here rather than left to a caller that may change.
    """

    def setUp( self ):
        self.earlier = _box( 'exposed_breast', 0, 1.0, 100, 200 )
        self.later = _box( 'exposed_breast', 0, 1.1, 140, 200 )

    def test_it_refuses_to_extrapolate_backwards( self ):
        self.assertIsNone( bu_render._interpolated_geometry(
            self.earlier, self.later, 0.95 ) )

    def test_it_refuses_to_extrapolate_forwards( self ):
        self.assertIsNone( bu_render._interpolated_geometry(
            self.earlier, self.later, 1.2 ) )

    def test_a_zero_or_negative_span_is_refused( self ):
        same = _box( 'exposed_breast', 0, 1.0, 140, 200 )
        self.assertIsNone( bu_render._interpolated_geometry(
            self.earlier, same, 1.0 ) )
        self.assertIsNone( bu_render._interpolated_geometry(
            self.later, self.earlier, 1.05 ) )

    def test_the_blend_rounds_rather_than_truncating( self ):
        # Truncating biases every blended edge toward the earlier sample by
        # up to a pixel. Same direction every frame, so it is a persistent
        # sub-pixel lag rather than noise - small, but free to avoid.
        # 100 + 0.25*(103-100) = 100.75 -> 101 rounded, 100 truncated.
        later = _box( 'exposed_breast', 0, 1.1, 103, 200 )
        result = bu_render._interpolated_geometry( self.earlier, later, 1.025 )
        self.assertEqual( result['x'], 101 )


class TestOverlappingSamplesCollapse( unittest.TestCase ):
    """
    The part that actually removes the pulsing: one box per object per
    frame while the object is near-still.
    """

    def _pair( self, second_x ):
        # time_safety 0.17 against a 0.111 interval, so both are live at
        # 1.06 - the real configuration's overlap.
        return [ _box( 'exposed_breast', 0, 1.0, 100, 200 ),
                 _box( 'exposed_breast', 0, 1.111, second_x, 200 ) ]

    def test_a_near_still_object_renders_one_box( self ):
        boxes = self._pair( 105 )
        timelines = bu_render.build_motion_index( boxes )
        live = [ box for box in boxes if box['start'] <= 1.06 <= box['end'] ]
        self.assertEqual( len( live ), 2, "fixture must have both samples live" )
        result = bu_render.interpolate_boxes_for_frame(
            live, timelines, 1.06, 0.25, max_collapse_growth=1.15 )
        self.assertEqual( len( result ), 1 )

    def test_a_fast_moving_object_keeps_both_boxes( self ):
        # Their union covers the swept path, which is a real safety
        # margin. Collapsing unconditionally cost 8265px on one frame of
        # the real cache.
        boxes = self._pair( 400 )
        timelines = bu_render.build_motion_index( boxes )
        live = [ box for box in boxes if box['start'] <= 1.06 <= box['end'] ]
        result = bu_render.interpolate_boxes_for_frame(
            live, timelines, 1.06, 0.25, max_collapse_growth=1.15 )
        self.assertEqual( len( result ), 2 )

    def test_the_rendered_area_stops_oscillating( self ):
        # The measurable form of the symptom: with two samples alternating
        # between one and two live, the union area pulses. With the
        # collapse it holds.
        boxes = self._pair( 104 )
        timelines = bu_render.build_motion_index( boxes )

        def union_area( group ):
            x0 = min( entry['x'] for entry in group )
            y0 = min( entry['y'] for entry in group )
            x1 = max( entry['x'] + entry['w'] for entry in group )
            y1 = max( entry['y'] + entry['h'] for entry in group )
            return ( x1 - x0 ) * ( y1 - y0 )

        held, collapsed = [], []
        for frame in range( 0, 12 ):
            current = 1.0 + frame / 60.0
            live = [ box for box in boxes if box['start'] <= current <= box['end'] ]
            if not live:
                continue
            held.append( union_area( live ) )
            collapsed.append( union_area( bu_render.interpolate_boxes_for_frame(
                live, timelines, current, 0.25, max_collapse_growth=1.15 ) ) )
        self.assertGreater( max( held ) - min( held ), 0,
                            "fixture does not reproduce the oscillation" )
        self.assertLess( max( collapsed ) - min( collapsed ),
                         max( held ) - min( held ) )

    def test_a_collapsed_box_still_slides_and_never_freezes( self ):
        # The collapse must survive the sample that this frame can actually
        # be interpolated FROM. time_safety makes a sample live before its
        # own timestamp, so the newest live sample is usually still in the
        # future, and the blend loop skips anything at t >= current_time.
        # Keeping that one traded the area oscillation for a positional
        # one: x ran 100, 101, 105, 105, 105, 105 - a 4px jump then five
        # frozen frames, which is the teleporting this module removes.
        boxes = [ _box( 'exposed_breast', 0, time, 100 + index*5, 200 )
                  for index, time in enumerate( [ 1.0, 1.111, 1.222, 1.333 ] ) ]
        timelines = bu_render.build_motion_index( boxes )
        xs = []
        for frame in range( 0, 20 ):
            current = 1.0 + frame/60.0
            live = [ box for box in boxes
                     if box['start'] <= current <= box['end'] ]
            if not live: continue
            result = bu_render.interpolate_boxes_for_frame(
                live, timelines, current, 0.25, max_collapse_growth=1.15 )
            self.assertEqual( len( result ), 1,
                              "near-still samples should collapse to one box" )
            xs.append( result[0]['x'] )
        self.assertGreater( len( xs ), 12, "fixture produced too few frames" )
        steps = [ later - earlier for earlier, later in zip( xs, xs[1:] ) ]
        self.assertTrue( all( step >= 0 for step in steps ), xs )
        # One sample interval of travel is 5px; a per-frame step anywhere
        # near that means the box jumped a whole sample instead of sliding.
        self.assertLessEqual( max( steps ), 3, xs )
        # And it must not sit still for a whole sample interval either.
        self.assertLess( _longest_run_of_zeros( steps ), 5, xs )

    def test_with_three_live_samples_the_newest_eligible_one_wins( self ):
        # Not reachable at time_safety 0.17 against a 0.111 interval, but a
        # structure profile can raise video_censor_fps, and a long enough
        # time_safety then puts three samples of one object on one frame.
        # The survivor must be the NEWEST at or before this frame; an older
        # eligible one would render a rectangle two intervals stale.
        # Near-still, so all four stay inside the growth gate together.
        boxes = [ _box( 'exposed_breast', 0, 1.00, 100, 200, safety=0.40 ),
                  _box( 'exposed_breast', 0, 1.05, 101, 200, safety=0.40 ),
                  _box( 'exposed_breast', 0, 1.10, 110, 200, safety=0.40 ),
                  _box( 'exposed_breast', 0, 1.15, 120, 200, safety=0.40 ) ]
        timelines = bu_render.build_motion_index( boxes )
        live = [ box for box in boxes if box['start'] <= 1.12 <= box['end'] ]
        self.assertGreaterEqual( len( live ), 3,
                                 "fixture must put three samples on one frame" )
        result = bu_render.interpolate_boxes_for_frame(
            live, timelines, 1.12, 0.25, max_collapse_growth=99.0 )
        self.assertEqual( len( result ), 1 )
        # Blending 1.10 (x=110) toward 1.15 (x=120) at 1.12 gives ~114.
        # Surviving 1.00 or 1.05 instead would blend from x=100 or 101.
        self.assertGreaterEqual( result[0]['x'], 110 )

    def test_the_growth_gate_measures_against_the_largest_box( self ):
        # A big box and a small one live at once: their union is barely
        # bigger than the BIG one, so collapsing costs almost nothing and
        # should happen. Measuring the union against the SMALLEST box
        # instead inflates the ratio and refuses the collapse, leaving the
        # pulsing in place. Both boxes here are interpolable at 1.06 so the
        # assertion is about the gate, not about which one survives.
        # The small sample sits wholly inside the big one, so the union IS
        # the big box: free to collapse against the largest, 4x the
        # smallest.
        boxes = [ _box( 'exposed_breast', 0, 1.0, 100, 200, w=200, h=200 ),
                  _box( 'exposed_breast', 0, 1.04, 140, 240, w=100, h=100 ),
                  _box( 'exposed_breast', 0, 1.2, 120, 210, w=200, h=200 ) ]
        timelines = bu_render.build_motion_index( boxes )
        live = boxes[:2]
        union_w = max( b['x']+b['w'] for b in live ) - min( b['x'] for b in live )
        union_h = max( b['y']+b['h'] for b in live ) - min( b['y'] for b in live )
        largest = max( b['w']*b['h'] for b in live )
        smallest = min( b['w']*b['h'] for b in live )
        self.assertLessEqual( union_w*union_h, largest*1.15,
                              "fixture must be collapsible against the largest" )
        self.assertGreater( union_w*union_h, smallest*1.15,
                            "fixture must NOT be collapsible against the smallest" )
        result = bu_render.interpolate_boxes_for_frame(
            live, timelines, 1.06, 0.25, max_collapse_growth=1.15 )
        self.assertEqual( len( result ), 1 )

    def test_objects_are_collapsed_independently( self ):
        boxes = ( self._pair( 105 )
                  + [ _box( 'exposed_breast', 1, 1.0, 900, 200 ),
                      _box( 'exposed_breast', 1, 1.111, 1200, 200 ) ] )
        timelines = bu_render.build_motion_index( boxes )
        live = [ box for box in boxes if box['start'] <= 1.06 <= box['end'] ]
        self.assertEqual( len( live ), 4 )
        result = bu_render.interpolate_boxes_for_frame(
            live, timelines, 1.06, 0.25, max_collapse_growth=1.15 )
        # near-still object collapses to 1, fast one keeps 2
        self.assertEqual( len( result ), 3 )


class TestDisabledAndUntracked( unittest.TestCase ):

    def test_no_timelines_returns_the_input_unchanged( self ):
        live = [ _box( 'exposed_breast', 0, 1.0, 100, 200 ) ]
        self.assertIs( bu_render.interpolate_boxes_for_frame( live, {}, 1.0, 0.25 ),
                       live )

    def test_a_box_with_no_timeline_passes_through( self ):
        tracked = [ _box( 'exposed_breast', 0, 1.0, 100, 200 ),
                    _box( 'exposed_breast', 0, 1.1, 140, 200 ) ]
        timelines = bu_render.build_motion_index( tracked )
        stranger = _box( 'exposed_vulva', 9, 1.05, 700, 300 )
        result = bu_render.interpolate_boxes_for_frame(
            [ stranger ], timelines, 1.05, 0.25 )
        self.assertEqual( len( result ), 1 )
        self.assertIs( result[0], stranger )


class TestSizeStabilisation( unittest.TestCase ):
    """
    The collapse removes pulsing but PAYS for it in coverage: measured on
    a real 120fps/9fps cache at growth 1.15, 561k censored pixels went
    uncovered and 34% of those were inside the nearest real detection,
    up to 4012px on one frame. Holding each track's size at its sampled
    maximum recovers 90% of that AND takes the residual size pulse to
    zero, at the cost of censoring more area.
    """

    def _growing_track( self ):
        # A track whose sampled size varies, which is the real case: the
        # detector's box breathes a few percent frame to frame.
        return [ _box( 'exposed_breast', 0, 1.000, 100, 200, w=80, h=80 ),
                 _box( 'exposed_breast', 0, 1.111, 104, 200, w=92, h=100 ),
                 _box( 'exposed_breast', 0, 1.222, 108, 200, w=78, h=82 ),
                 _box( 'exposed_breast', 0, 1.333, 112, 200, w=84, h=88 ) ]

    def test_off_by_default_in_the_helper( self ):
        # size_window None must leave geometry exactly as the collapse
        # produced it, so the stage is opt-in at the helper level.
        boxes = self._growing_track()
        timelines = bu_render.build_motion_index( boxes )
        plain = bu_render.interpolate_boxes_for_frame(
            [ boxes[0] ], timelines, 1.05, 0.25 )
        self.assertFalse( plain[0].get( '_motion_size_stabilised' ) )

    def test_the_rectangle_size_stops_changing( self ):
        boxes = self._growing_track()
        timelines = bu_render.build_motion_index( boxes )
        sizes = set()
        for frame in range( 0, 40 ):
            current = 1.0 + frame/120.0
            live = [ box for box in boxes
                     if box['start'] <= current <= box['end'] ]
            if not live: continue
            result = bu_render.interpolate_boxes_for_frame(
                live, timelines, current, 0.25, 1.15, size_window=0 )
            for box in result:
                sizes.add( ( box['w'], box['h'] ) )
        self.assertEqual( len( sizes ), 1,
                          "size should be constant across the track, got %s"%( sizes, ) )
        self.assertEqual( sizes.pop(), ( 92, 100 ),
                          "should hold the track's largest sampled size" )

    def test_it_only_ever_grows_a_box( self ):
        # Shrinking would uncover detected area, which is the failure this
        # stage exists to prevent.
        boxes = self._growing_track()
        timelines = bu_render.build_motion_index( boxes )
        for frame in range( 0, 40 ):
            current = 1.0 + frame/120.0
            live = [ box for box in boxes
                     if box['start'] <= current <= box['end'] ]
            if not live: continue
            plain = bu_render.interpolate_boxes_for_frame(
                live, timelines, current, 0.25, 1.15 )
            grown = bu_render.interpolate_boxes_for_frame(
                live, timelines, current, 0.25, 1.15, size_window=0 )
            self.assertEqual( len( plain ), len( grown ) )
            for before, after in zip( plain, grown ):
                self.assertGreaterEqual( after['w'], before['w'] )
                self.assertGreaterEqual( after['h'], before['h'] )

    def test_it_grows_around_the_interpolated_centre( self ):
        # The point of the slide is that the box tracks the subject; growing
        # from a corner would drag it off position.
        boxes = self._growing_track()
        timelines = bu_render.build_motion_index( boxes )
        plain = bu_render.interpolate_boxes_for_frame(
            [ boxes[0] ], timelines, 1.05, 0.25, 1.15 )[0]
        grown = bu_render.interpolate_boxes_for_frame(
            [ boxes[0] ], timelines, 1.05, 0.25, 1.15, size_window=0 )[0]
        self.assertAlmostEqual( plain['x'] + plain['w']/2.0,
                                grown['x'] + grown['w']/2.0, delta=1.0 )
        self.assertAlmostEqual( plain['y'] + plain['h']/2.0,
                                grown['y'] + grown['h']/2.0, delta=1.0 )

    def test_a_window_bounds_which_samples_the_size_comes_from( self ):
        # The big sample at 1.111 must not set the size for a frame far
        # from it once a window is in force.
        boxes = [ _box( 'exposed_breast', 0, 1.000, 100, 200, w=80, h=80 ),
                  _box( 'exposed_breast', 0, 1.111, 104, 200, w=200, h=200 ),
                  _box( 'exposed_breast', 0, 5.000, 108, 200, w=80, h=80 ),
                  _box( 'exposed_breast', 0, 5.111, 112, 200, w=80, h=80 ) ]
        timelines = bu_render.build_motion_index( boxes )
        narrow = bu_render.interpolate_boxes_for_frame(
            [ boxes[2] ], timelines, 5.05, 0.25, 1.15, size_window=0.5 )[0]
        whole = bu_render.interpolate_boxes_for_frame(
            [ boxes[2] ], timelines, 5.05, 0.25, 1.15, size_window=0 )[0]
        self.assertEqual( narrow['w'], 80 )
        self.assertEqual( whole['w'], 200 )

    def test_a_stabilised_box_stays_inside_the_frame( self ):
        boxes = [ _box( 'exposed_breast', 0, 1.000, 1830, 990, w=80, h=80 ),
                  _box( 'exposed_breast', 0, 1.111, 1835, 995, w=200, h=200 ) ]
        timelines = bu_render.build_motion_index( boxes )
        for frame in range( 0, 14 ):
            current = 1.0 + frame/120.0
            result = bu_render.interpolate_boxes_for_frame(
                [ boxes[0] ], timelines, current, 0.25, 1.15, size_window=0 )[0]
            self.assertGreaterEqual( result['x'], 0 )
            self.assertGreaterEqual( result['y'], 0 )
            self.assertLessEqual( result['x'] + result['w'], 1920 )
            self.assertLessEqual( result['y'] + result['h'], 1080 )

    def test_both_boxes_of_an_uncollapsed_track_are_stabilised( self ):
        # When the object is moving fast the collapse deliberately keeps
        # both live samples. Each is a rectangle that will be drawn, so each
        # needs the size guarantee. An earlier version tracked which keys it
        # had already handled and skipped the second, leaving it at its own
        # sample's size on exactly the frames where coverage matters most.
        boxes = [ _box( 'exposed_breast', 0, 1.000, 100, 200, w=80, h=80 ),
                  _box( 'exposed_breast', 0, 1.111, 400, 200, w=150, h=150 ) ]
        timelines = bu_render.build_motion_index( boxes )
        live = [ box for box in boxes if box['start'] <= 1.06 <= box['end'] ]
        self.assertEqual( len( live ), 2, "fixture must keep both samples live" )
        result = bu_render.interpolate_boxes_for_frame(
            live, timelines, 1.06, 0.25, 1.15, size_window=0 )
        self.assertEqual( len( result ), 2, "fast motion should not collapse" )
        for box in result:
            self.assertEqual( ( box['w'], box['h'] ), ( 150, 150 ),
                              "every drawn box needs the track's max size" )

    def test_the_window_is_centred_on_the_sample( self ):
        # It must look BOTH ways in time. Taking only later samples would
        # miss a large sample just behind this frame, which is half the
        # coverage the window is there to keep.
        boxes = [ _box( 'exposed_breast', 0, 1.000, 100, 200, w=200, h=200 ),
                  _box( 'exposed_breast', 0, 1.111, 104, 200, w=80, h=80 ),
                  _box( 'exposed_breast', 0, 1.222, 108, 200, w=80, h=80 ) ]
        timelines = bu_render.build_motion_index( boxes )
        # At the 1.111 sample, a 0.2s window reaches back to 1.000 (w=200).
        result = bu_render.interpolate_boxes_for_frame(
            [ boxes[1] ], timelines, 1.15, 0.25, 1.15, size_window=0.2 )[0]
        self.assertEqual( result['w'], 200,
                          "the window must include samples EARLIER than this one" )

        # And the other half: a large sample just AHEAD must count too, so
        # the box is already big by the time the subject gets there.
        ahead = [ _box( 'exposed_breast', 0, 1.000, 100, 200, w=80, h=80 ),
                  _box( 'exposed_breast', 0, 1.111, 104, 200, w=200, h=200 ),
                  _box( 'exposed_breast', 0, 1.222, 108, 200, w=80, h=80 ) ]
        ahead_lines = bu_render.build_motion_index( ahead )
        forward = bu_render.interpolate_boxes_for_frame(
            [ ahead[0] ], ahead_lines, 1.05, 0.25, 1.15, size_window=0.2 )[0]
        self.assertEqual( forward['w'], 200,
                          "the window must include samples LATER than this one" )

    def test_an_untracked_box_is_left_alone( self ):
        box = _box( 'exposed_breast', 9, 1.0, 100, 200 )
        timelines = bu_render.build_motion_index( self._growing_track() )
        result = bu_render.interpolate_boxes_for_frame(
            [ box ], timelines, 1.0, 0.25, 1.15, size_window=0 )
        self.assertEqual( result[0]['w'], box['w'] )
        self.assertFalse( result[0].get( '_motion_size_stabilised' ) )

    def test_the_source_boxes_are_not_mutated( self ):
        boxes = self._growing_track()
        timelines = bu_render.build_motion_index( boxes )
        snapshot = [ dict( box ) for box in boxes ]
        for frame in range( 0, 40 ):
            bu_render.interpolate_boxes_for_frame(
                list( boxes ), timelines, 1.0 + frame/120.0, 0.25, 1.15,
                size_window=0 )
        self.assertEqual( [ dict( box ) for box in boxes ], snapshot )


class TestSharedIndexIsHonoured( unittest.TestCase ):
    """
    The index is built once for the whole video and handed to every chunk
    and every worker. That is only correct if the callee actually uses
    what it was given.
    """

    def test_a_supplied_index_is_used_rather_than_rebuilt( self ):
        # Hand in an index that knows about the track and one that does
        # not. If the argument were ignored and rebuilt from the boxes,
        # both calls would behave identically.
        boxes = [ _box( 'exposed_breast', 0, 1.0, 100, 200 ),
                  _box( 'exposed_breast', 0, 1.1, 140, 200 ) ]
        real = bu_render.build_motion_index( boxes )
        with_index = bu_render.interpolate_boxes_for_frame(
            [ boxes[0] ], real, 1.05, 0.25 )
        without = bu_render.interpolate_boxes_for_frame(
            [ boxes[0] ], {}, 1.05, 0.25 )
        self.assertTrue( with_index[0].get( '_motion_interpolated' ) )
        self.assertFalse( without[0].get( '_motion_interpolated' ) )

    def test_the_index_is_not_mutated_by_a_frame( self ):
        # Workers share it, so a render that wrote to it would make output
        # depend on chunk ordering - the same hazard the module docstring
        # calls out for the box list.
        boxes = [ _box( 'exposed_breast', 0, 1.0, 100, 200 ),
                  _box( 'exposed_breast', 0, 1.1, 140, 200 ) ]
        timelines = bu_render.build_motion_index( boxes )
        before = { key: ( list( times ), [ dict( e ) for e in entries ] )
                   for key, ( times, entries ) in timelines.items() }
        for step in range( 0, 11 ):
            bu_render.interpolate_boxes_for_frame(
                [ boxes[0] ], timelines, 1.0 + step/100.0, 0.25 )
        after = { key: ( list( times ), [ dict( e ) for e in entries ] )
                  for key, ( times, entries ) in timelines.items() }
        self.assertEqual( before, after )

    def test_the_source_boxes_are_not_mutated( self ):
        boxes = [ _box( 'exposed_breast', 0, 1.0, 100, 200 ),
                  _box( 'exposed_breast', 0, 1.1, 140, 200 ) ]
        timelines = bu_render.build_motion_index( boxes )
        snapshot = [ dict( box ) for box in boxes ]
        for step in range( 0, 11 ):
            bu_render.interpolate_boxes_for_frame(
                list( boxes ), timelines, 1.0 + step/100.0, 0.25 )
        self.assertEqual( [ dict( box ) for box in boxes ], snapshot )


class TestSettingsAreKeyedAndValidated( unittest.TestCase ):

    WATCHED = ( 'render_motion_interpolation', 'render_motion_max_span_seconds',
                'render_motion_collapse_max_growth' )

    def setUp( self ):
        self._saved = { name: getattr( betaconfig, name, None ) for name in self.WATCHED }

    def tearDown( self ):
        for name, value in self._saved.items():
            if value is None:
                if hasattr( betaconfig, name ):
                    delattr( betaconfig, name )
            else:
                setattr( betaconfig, name, value )
        bu_config.invalidate_config_caches()

    def test_the_toggle_is_in_the_censor_key( self ):
        import betautils_cache_paths as bu_cache
        betaconfig.render_motion_interpolation = True
        bu_config.invalidate_config_caches()
        before = bu_cache.censor_key()
        betaconfig.render_motion_interpolation = False
        bu_config.invalidate_config_caches()
        self.assertNotEqual( before, bu_cache.censor_key() )

    def test_the_span_is_in_the_censor_key( self ):
        import betautils_cache_paths as bu_cache
        betaconfig.render_motion_max_span_seconds = None
        bu_config.invalidate_config_caches()
        before = bu_cache.censor_key()
        betaconfig.render_motion_max_span_seconds = 0.5
        bu_config.invalidate_config_caches()
        self.assertNotEqual( before, bu_cache.censor_key() )

    def test_a_non_boolean_toggle_is_rejected( self ):
        betaconfig.render_motion_interpolation = 'yes'
        bu_config.invalidate_config_caches()
        self.assertTrue( any( 'render_motion_interpolation' in error
                              for error in bu_config._collect_validation_errors() ) )

    def test_the_growth_gate_is_in_the_censor_key( self ):
        # It changes which boxes get collapsed, so it changes rendered
        # pixels. A censored video cached under one value must not be
        # served for another.
        import betautils_cache_paths as bu_cache
        betaconfig.render_motion_collapse_max_growth = 1.15
        bu_config.invalidate_config_caches()
        before = bu_cache.censor_key()
        betaconfig.render_motion_collapse_max_growth = 1.40
        bu_config.invalidate_config_caches()
        self.assertNotEqual( before, bu_cache.censor_key() )

    def test_a_growth_gate_below_one_is_rejected( self ):
        # Under 1.0 nothing can ever collapse, so the flicker fix would be
        # off while the toggle still said it was on.
        betaconfig.render_motion_collapse_max_growth = 0.9
        bu_config.invalidate_config_caches()
        self.assertTrue( any( 'render_motion_collapse_max_growth' in error
                              for error in bu_config._collect_validation_errors() ) )

    def test_the_size_window_is_in_the_censor_key( self ):
        import betautils_cache_paths as bu_cache
        betaconfig.render_motion_size_window_seconds = 0
        bu_config.invalidate_config_caches()
        before = bu_cache.censor_key()
        betaconfig.render_motion_size_window_seconds = 1.0
        bu_config.invalidate_config_caches()
        self.assertNotEqual( before, bu_cache.censor_key() )

    def test_a_negative_size_window_is_rejected( self ):
        betaconfig.render_motion_size_window_seconds = -1
        bu_config.invalidate_config_caches()
        self.assertTrue( any( 'render_motion_size_window_seconds' in error
                              for error in bu_config._collect_validation_errors() ) )

    def test_the_size_window_defaults_to_a_quarter_second( self ):
        # Measured: recovers a third of the coverage the collapse gives up
        # for -0.1% area. The whole track (0) recovers far more but costs
        # +20% censored area, which is a choice about the footage.
        if hasattr( betaconfig, 'render_motion_size_window_seconds' ):
            delattr( betaconfig, 'render_motion_size_window_seconds' )
        bu_config.invalidate_config_caches()
        _enabled, _span, _growth, window = bu_render._motion_settings( None )
        self.assertEqual( window, 0.25 )

    def test_the_size_window_can_be_switched_off( self ):
        betaconfig.render_motion_size_window_seconds = None
        bu_config.invalidate_config_caches()
        _enabled, _span, _growth, window = bu_render._motion_settings( None )
        self.assertIsNone( window )

    def test_a_non_numeric_growth_gate_is_rejected( self ):
        betaconfig.render_motion_collapse_max_growth = 'wide'
        bu_config.invalidate_config_caches()
        self.assertTrue( any( 'render_motion_collapse_max_growth' in error
                              for error in bu_config._collect_validation_errors() ) )

    def test_a_non_positive_span_is_rejected( self ):
        betaconfig.render_motion_max_span_seconds = 0
        bu_config.invalidate_config_caches()
        self.assertTrue( any( 'render_motion_max_span_seconds' in error
                              for error in bu_config._collect_validation_errors() ) )

    def test_none_span_means_two_sampling_intervals( self ):
        betaconfig.render_motion_max_span_seconds = None
        betaconfig.video_censor_fps = 9
        bu_config.invalidate_config_caches()
        _enabled, span, _growth, _window = bu_render._motion_settings( None )
        self.assertAlmostEqual( span, 2.0/9 )


if __name__ == '__main__':
    unittest.main()
