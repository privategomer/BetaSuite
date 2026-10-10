"""
test_cut_bursts.py - merging bursts of shot cuts, and the analysis that
decides whether to.

A fade or flash makes the scan flag a cut at several consecutive
samples. merge_cut_bursts collapses a short burst to one cut, leaving
long chains alone because those are genuine rapid-fire editing.
analyze_cut_bursts replays the caches with and without merging; these
tests pin the merge, the replay hook it uses (tracking cuts change, the
profile does not), the confident-detection coverage it reports, and the
verdicts it draws.
"""

import importlib.util
import os
import unittest
from unittest import mock

import betautils_tuning as bu_tuning
import betautils_video as bu_video


def _load_tool():
    path = os.path.join( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ),
                         'tools', 'analysis', 'analyze_cut_bursts.py' )
    spec = importlib.util.spec_from_file_location( '_cut_bursts_under_test', path )
    module = importlib.util.module_from_spec( spec )
    spec.loader.exec_module( module )
    return module


class TestMergeCutBursts( unittest.TestCase ):

    def test_a_fade_burst_becomes_one_cut( self ):
        cuts = [ 5.0, 10.0, 10.1, 10.2, 10.3, 20.0 ]
        self.assertEqual( bu_video.merge_cut_bursts( cuts, 0.35 ), [ 5.0, 10.3, 20.0 ] )
        self.assertEqual( bu_video.merge_cut_bursts( cuts, 0.35, keep='first' ),
                          [ 5.0, 10.0, 20.0 ] )

    def test_isolated_cuts_are_untouched( self ):
        cuts = [ 1.0, 2.0, 3.0 ]
        self.assertEqual( bu_video.merge_cut_bursts( cuts, 0.35 ), cuts )

    def test_the_window_is_a_strict_gap( self ):
        self.assertEqual( bu_video.merge_cut_bursts( [ 1.0, 1.35 ], 0.35 ), [ 1.0, 1.35 ] )
        self.assertEqual( bu_video.merge_cut_bursts( [ 1.0, 1.34 ], 0.35 ), [ 1.34 ] )

    def test_a_long_chain_is_real_editing_and_stays( self ):
        # Beat cuts every 0.2s for 2s: a span past max_span is left alone.
        chain = [ round( 30.0 + 0.2 * index, 1 ) for index in range( 11 ) ]
        self.assertEqual( bu_video.merge_cut_bursts( chain, 0.35, max_span=0.6 ), chain )
        self.assertEqual( bu_video.merge_cut_bursts( chain, 0.35 ), [ chain[-1] ] )

    def test_a_burst_at_exactly_max_span_merges( self ):
        self.assertEqual( bu_video.merge_cut_bursts( [ 1.0, 1.3, 1.6 ], 0.35, max_span=0.6 ),
                          [ 1.6 ] )

    def test_unsorted_and_empty_input( self ):
        self.assertEqual( bu_video.merge_cut_bursts( [ 10.2, 10.0, 10.1 ], 0.35 ), [ 10.2 ] )
        self.assertEqual( bu_video.merge_cut_bursts( [], 0.35 ), [] )
        self.assertEqual( bu_video.merge_cut_bursts( None, 0.35 ), [] )

    def test_keep_is_validated( self ):
        with self.assertRaises( ValueError ):
            bu_video.merge_cut_bursts( [ 1.0 ], 0.35, keep='middle' )

    def test_bursts_are_found( self ):
        self.assertEqual( bu_video.cut_bursts( [ 1.0, 5.0, 5.1, 5.2, 9.0, 9.2 ], 0.35 ),
                          [ [ 5.0, 5.1, 5.2 ], [ 9.0, 9.2 ] ] )
        self.assertEqual( bu_video.cut_bursts( [ 1.0, 2.0 ], 0.35 ), [] )


class TestConfidentCoverage( unittest.TestCase ):

    def test_confident_detections_are_by_label_and_score( self ):
        raw = [ { 'class_id': 'exposed_breast', 'score': 0.8, 't': 1.0, 'x': 0, 'y': 0, 'w': 10, 'h': 20 },
                { 'class_id': 'exposed_breast', 'score': 0.3, 't': 1.0, 'x': 0, 'y': 0, 'w': 10, 'h': 20 },
                { 'class_id': 'face_femme', 'score': 0.9, 't': 1.0, 'x': 0, 'y': 0, 'w': 10, 'h': 20 } ]
        found = bu_tuning.confident_detections( raw, { 'exposed_breast' }, 0.5 )
        self.assertEqual( found, { 'exposed_breast': [ ( 1.0, 5.0, 10.0 ) ] } )

    def test_the_confident_floor_is_inclusive( self ):
        raw = [ { 'class_id': 'exposed_breast', 'score': 0.5, 't': 2.0, 'x': 0, 'y': 0, 'w': 10, 'h': 10 } ]
        self.assertEqual( bu_tuning.confident_detections( raw, { 'exposed_breast' }, 0.5 ),
                          { 'exposed_breast': [ ( 2.0, 5.0, 5.0 ) ] } )

    def test_a_box_that_already_ended_does_not_cover( self ):
        # The long box elsewhere widens the look-back, so only the end
        # time can rule out the short box that ended at 1.1.
        detections = { 'exposed_breast': [ ( 2.0, 50, 50 ) ] }
        rendered = [ { 'label': 'exposed_breast', 'start': 0.0, 'end': 10.0,
                       'x': 900, 'y': 900, 'w': 10, 'h': 10 },
                     { 'label': 'exposed_breast', 'start': 0.9, 'end': 1.1,
                       'x': 0, 'y': 0, 'w': 100, 'h': 100 } ]
        self.assertEqual( bu_tuning.uncovered_detections( detections, rendered ),
                          { 'exposed_breast': 1 } )

    def test_a_detection_is_covered_by_a_live_box_holding_its_centre( self ):
        detections = { 'exposed_breast': [ ( 1.0, 50, 50 ), ( 3.0, 50, 50 ), ( 1.0, 500, 500 ) ] }
        rendered = [ { 'label': 'exposed_breast', 'start': 0.9, 'end': 1.1,
                       'x': 0, 'y': 0, 'w': 100, 'h': 100 },
                     { 'label': 'exposed_vulva', 'start': 2.9, 'end': 3.1,
                       'x': 0, 'y': 0, 'w': 100, 'h': 100 } ]
        # 1.0 inside: covered. 3.0: only a different label is live. 500,500: outside.
        self.assertEqual( bu_tuning.uncovered_detections( detections, rendered ),
                          { 'exposed_breast': 2 } )

    def test_a_long_box_earlier_in_the_list_still_covers( self ):
        detections = { 'exposed_breast': [ ( 5.0, 50, 50 ) ] }
        rendered = [ { 'label': 'exposed_breast', 'start': 0.0, 'end': 10.0,
                       'x': 0, 'y': 0, 'w': 100, 'h': 100 },
                     { 'label': 'exposed_breast', 'start': 4.9, 'end': 5.0,
                       'x': 900, 'y': 900, 'w': 10, 'h': 10 } ]
        self.assertEqual( bu_tuning.uncovered_detections( detections, rendered ),
                          { 'exposed_breast': 0 } )


class TestCutTransformHook( unittest.TestCase ):
    """The merge changes the cuts tracking stops at, never the profile."""

    def test_tracking_gets_the_transformed_cuts( self ):
        seen = {}

        def fake_prepare( raw, w, h, **kwargs ):
            seen.update( kwargs )
            return [], {}

        context = { 'shot_cut_times': [ 10.0, 10.1, 10.2 ], 'profile_name': 'scene',
                    'sample_fps': 9, 'signals': None }
        with mock.patch.object( bu_tuning, '_read_raw_boxes', return_value=[ { 'class_id': 'x' } ] ), \
             mock.patch.object( bu_tuning, 'frame_size_for_video', return_value=( 640, 360 ) ), \
             mock.patch.object( bu_tuning, 'replay_tracking_context', return_value=context ), \
             mock.patch.object( bu_tuning.bu_track, 'prepare_boxes_for_render', fake_prepare ):
            metrics = bu_tuning.measure_replay(
                { 'h1': [ 'c.gz' ] }, { 'h1': '/v.mp4' }, 'nudenet_v3', [ 'exposed_breast' ],
                cut_transform=lambda cuts: bu_video.merge_cut_bursts( cuts, 0.35 ) )
        self.assertEqual( seen['shot_cut_times'], [ 10.2 ] )
        self.assertIs( seen['logger'], bu_tuning._QUIET_LOGGER,
                       "a replay must not write its per-video lines into the run log" )
        self.assertEqual( seen['profile_name'], 'scene' )
        self.assertEqual( metrics['profiles'], { 'h1': 'scene' } )


class TestAnalysis( unittest.TestCase ):

    def setUp( self ):
        self.tool = _load_tool()

    def metrics( self, missed, risky=100, boxes=1000, unconfirmed=50, resets=200 ):
        return { 'confident_uncovered_by_label': { 'exposed_breast': missed },
                 'confident_by_label': { 'exposed_breast': 10000 },
                 'risky_contention': risky, 'rendered_boxes': boxes,
                 'dropped_unconfirmed': unconfirmed, 'new_tracks': resets,
                 'coverage_by_label': { 'exposed_breast': 1.0 } }

    def test_census_counts_cuts_in_bursts_worst_first( self ):
        entries = [ ( 'a', '/x/a.mp4', [], 'scene', [ 1.0, 5.0, 5.1, 5.2 ] ),
                    ( 'b', '/x/b.mp4', [], 'compilation', [ 1.0, 2.0 ] ),
                    ( 'c', '/x/c.mp4', [], 'scene', None ) ]
        rows = self.tool.burst_census( entries, 0.35 )
        self.assertEqual( [ row['file'] for row in rows ][0], 'a.mp4' )
        self.assertEqual( rows[0]['cuts_in_bursts'], 3 )
        self.assertEqual( rows[0]['bursts'], 1 )
        self.assertFalse( [ row for row in rows if row['file'] == 'c.mp4' ][0]['scanned'] )

    def test_compare_reports_restored_and_risky( self ):
        change = self.tool.compare( self.metrics( 500, risky=100 ), self.metrics( 400, risky=150 ) )
        self.assertEqual( change['confident_restored'], 100 )
        self.assertEqual( change['risky_added'], 50 )

    def test_verdicts( self ):
        helps, _ = self.tool.verdict( { 'confident_restored': 100, 'risky_added': 50 }, 5.0 )
        self.assertTrue( helps )
        too_costly, text = self.tool.verdict( { 'confident_restored': 10, 'risky_added': 80 }, 5.0 )
        self.assertFalse( too_costly )
        self.assertIn( '8.0 risky each', text )
        worse, text = self.tool.verdict( { 'confident_restored': -3, 'risky_added': 0 }, 5.0 )
        self.assertFalse( worse )
        self.assertIn( 'worse', text )
        nothing, _ = self.tool.verdict( { 'confident_restored': 0, 'risky_added': 0 }, 5.0 )
        self.assertFalse( nothing )

    def test_variants_are_replayed_per_profile_and_totalled( self ):
        calls = []

        def fake_measure( videos, hash_to_path, backend, labels, seed=None,
                          confident_score=None, cut_transform=None ):
            calls.append( ( tuple( sorted( videos ) ), cut_transform is not None ) )
            return self.metrics( 400 if cut_transform else 500 )

        entries = [ ( 'a', '/a.mp4', [ 'a.gz' ], 'scene', [] ),
                    ( 'b', '/b.mp4', [ 'b.gz' ], 'compilation', [] ),
                    ( 'c', '/c.mp4', [ 'c.gz' ], 'scene', [] ) ]
        variants = self.tool.variants_from( [ 0.35 ], [ 0.6 ], [ 'last' ] )
        results = self.tool.compare_variants( entries, 'nudenet_v3', [ 'exposed_breast' ],
                                              variants, 1, 0.5, measure=fake_measure )
        self.assertEqual( results['scene']['files'], 2 )
        label = variants[0].label
        self.assertEqual( results['scene'][label]['confident_restored'], 100 )
        self.assertEqual( results['total'][label]['confident_restored'], 200 )
        self.assertIn( ( ( 'a', 'c' ), False ), calls )
        self.assertIn( ( ( 'a', 'c' ), True ), calls )

    def test_best_variant_is_the_one_recovering_most_at_a_fair_price( self ):
        totals = { 'cheap': { 'confident_restored': 100, 'risky_added': 10 },
                   'more': { 'confident_restored': 300, 'risky_added': 400 },
                   'pricey': { 'confident_restored': 500, 'risky_added': 5000 } }
        self.assertEqual( self.tool.best_variant( totals, 5.0 ), 'more' )
        self.assertIsNone( self.tool.best_variant( { 'x': { 'confident_restored': 0,
                                                            'risky_added': 0 } }, 5.0 ) )

    def test_spans_parse( self ):
        self.assertIsNone( self.tool.parse_span( 'none' ) )
        self.assertEqual( self.tool.parse_span( '0.6' ), 0.6 )


if __name__ == '__main__':
    unittest.main()
