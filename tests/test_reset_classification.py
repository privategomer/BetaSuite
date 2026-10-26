"""
test_reset_classification.py - the analysers' idea of a track reset matches the tracker's.

smooth_boxes never continues a track across a shot cut, however close
the boxes. analyze_track_breaks and analyze_jitter classified each new
track by looking for a nearby live track, and until 2026-10-18 neither
looked at cuts: on cut-dense compilations every new shot's first
detection beside where the last shot's track ended counted as an
"unexplained" reset or a "likely reset". These pin the cut-aware
classification, and that the shared bu_track.cut_separates is the one
the tracker uses.
"""

import importlib.util
import os
import unittest

import betautils_track as bu_track

REPO_ROOT = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )


def _load( relative, name ):
    spec = importlib.util.spec_from_file_location( name, os.path.join( REPO_ROOT, relative ) )
    module = importlib.util.module_from_spec( spec )
    spec.loader.exec_module( module )
    return module


BREAKS = _load( os.path.join( 'tools', 'analysis', 'analyze_track_breaks.py' ), '_breaks_under_test' )
JITTER = _load( os.path.join( 'tools', 'analysis', 'analyze_jitter.py' ), '_jitter_under_test' )


def _track( t, x=100, end=None ):
    return { 'x': x, 'y': 100, 'w': 50, 'h': 50, 't': t, 'end': end if end is not None else t + 0.1 }


def _box( t, x=100 ):
    return { 'x': x, 'y': 100, 'w': 50, 'h': 50, 't': t, 'start': t - 0.1, 'end': t + 0.1 }


class TestCutSeparates( unittest.TestCase ):

    def test_a_cut_after_the_earlier_time_and_at_or_before_the_later_counts( self ):
        cuts = [ 1.0, 2.0 ]
        self.assertTrue( bu_track.cut_separates( cuts, 0.9, 1.0 ), "a cut AT the later time counts" )
        self.assertFalse( bu_track.cut_separates( cuts, 1.0, 1.5 ), "a cut AT the earlier time does not" )
        self.assertFalse( bu_track.cut_separates( cuts, 1.1, 1.9 ) )
        self.assertTrue( bu_track.cut_separates( cuts, 1.1, 2.5 ) )
        self.assertFalse( bu_track.cut_separates( [], 0.0, 9.0 ) )
        self.assertFalse( bu_track.cut_separates( None, 0.0, 9.0 ) )

    def test_the_tracker_stops_at_a_cut_the_helper_reports( self ):
        from tests.test_track_pipeline import censorable_box

        class _Count( bu_track.TrackingObserver ):
            new_tracks = 0

            def on_new_track( self, label, tracks, track_index, box, settings ):
                _Count.new_tracks += 1

        later = 1.0 + 1 / 9.0
        cut = 1.05
        self.assertTrue( bu_track.cut_separates( [ cut ], 1.0, later ) )
        for cuts, expected in ( ( [ cut ], 2 ), ( [], 1 ) ):
            _Count.new_tracks = 0
            boxes = [ censorable_box( 'exposed_breast', 1.0, x=100 ),
                      censorable_box( 'exposed_breast', later, x=101 ) ]
            with bu_track.tracking_observer( _Count() ):
                bu_track.smooth_boxes( boxes, shot_cut_times=cuts, cut_bursts=False )
            self.assertEqual( _Count.new_tracks, expected, cuts )


class TestTrackBreaksReasons( unittest.TestCase ):

    def setUp( self ):
        BREAKS._reset_events()
        self.addCleanup( BREAKS._cut_times.clear )

    def reason( self, cuts, prev, box, max_gap=1.0, multiplier=1.0 ):
        BREAKS._cut_times[:] = cuts
        tracks = [ prev, _track( box['t'], x=box['x'] ) ]
        BREAKS._record_new_track_event( 'exposed_breast', tracks, 1, box, max_gap, multiplier )
        return BREAKS._new_track_events[-1]['reason']

    def test_a_new_shot_beside_the_last_shots_track_is_after_cut( self ):
        self.assertEqual( self.reason( [ 1.05 ], _track( 1.0 ), _box( 1.1 ) ), 'after-cut' )

    def test_without_the_cut_the_same_boxes_are_unexplained( self ):
        self.assertEqual( self.reason( [], _track( 1.0 ), _box( 1.1 ) ), 'unexplained' )

    def test_a_cut_outside_the_interval_changes_nothing( self ):
        self.assertEqual( self.reason( [ 0.5, 3.0 ], _track( 1.0 ), _box( 1.1 ) ), 'unexplained' )

    def test_distance_and_gap_still_classify_within_a_shot( self ):
        self.assertEqual( self.reason( [], _track( 1.0, x=100 ), _box( 1.1, x=400 ) ), 'distance-blocked' )
        self.assertEqual( self.reason( [], _track( 1.0, end=1.0 ), _box( 5.0 ), max_gap=1.0 ), 'gap-blocked' )

    def test_a_cut_wins_over_distance( self ):
        self.assertEqual( self.reason( [ 1.05 ], _track( 1.0, x=100 ), _box( 1.1, x=400 ) ), 'after-cut' )

    def test_nothing_else_live_is_no_nearby_track( self ):
        BREAKS._cut_times[:] = [ 1.05 ]
        box = _box( 1.1 )
        BREAKS._record_new_track_event( 'exposed_breast', [ _track( 1.1 ) ], 0, box, 1.0, 1.0 )
        self.assertEqual( BREAKS._new_track_events[-1]['reason'], 'no-nearby-track' )


class TestJitterNearMisses( unittest.TestCase ):

    def setUp( self ):
        JITTER._reset_events()
        self.addCleanup( JITTER._cut_times.clear )

    def event( self, cuts, others, box ):
        JITTER._cut_times[:] = cuts
        tracks = list( others ) + [ _track( box['t'], x=box['x'] ) ]
        JITTER._record_new_track_event( 'exposed_breast', tracks, len( tracks ) - 1, box )
        return JITTER._new_track_events[-1]

    def test_a_track_across_a_cut_is_not_a_near_miss( self ):
        event = self.event( [ 1.05 ], [ _track( 1.0 ) ], _box( 1.1, x=110 ) )
        self.assertIsNone( event['near_miss_dist'] )
        self.assertTrue( event['after_cut'] )

    def test_the_same_boxes_in_one_shot_are_a_near_miss( self ):
        event = self.event( [], [ _track( 1.0 ) ], _box( 1.1, x=110 ) )
        self.assertAlmostEqual( event['near_miss_dist'], 10.0 )
        self.assertFalse( event['after_cut'] )

    def test_a_near_miss_in_the_same_shot_wins_over_one_across_a_cut( self ):
        # One track before the cut, one started after it: the same-shot
        # one makes this a likely reset, not a new shot.
        event = self.event( [ 1.05 ], [ _track( 1.0, x=100 ), _track( 1.08, x=120 ) ], _box( 1.1, x=110 ) )
        self.assertAlmostEqual( event['near_miss_dist'], 10.0 )
        self.assertFalse( event['after_cut'] )

    def test_far_tracks_are_neither( self ):
        event = self.event( [ 1.05 ], [ _track( 1.0, x=900 ) ], _box( 1.1, x=100 ) )
        self.assertIsNone( event['near_miss_dist'] )
        self.assertFalse( event['after_cut'] )

    def test_the_report_counts_post_cut_starts_apart_from_resets( self ):
        lines = []
        result = { 'dims_missing': [], 'frame_deltas_by_label': {}, 'match_dists_by_label': {},
                   'new_track_by_label': { 'exposed_breast': 10 },
                   'near_miss_by_label': { 'exposed_breast': 2 },
                   'after_cut_by_label': { 'exposed_breast': 5 } }
        import builtins
        original = builtins.print
        builtins.print = lambda *args, **kwargs: lines.append( " ".join( str( a ) for a in args ) )
        try:
            JITTER.print_alpha_result( 0.5, result, [ 'exposed_breast' ] )
        finally:
            builtins.print = original
        text = "\n".join( lines )
        self.assertIn( 'likely resets (another track in the same shot', text )
        self.assertIn( '2/10 (20%)', text )
        self.assertIn( 'tracking stops at every cut): 5/10 (50%)', text )


if __name__ == '__main__':
    unittest.main()
