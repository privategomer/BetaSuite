"""
test_suppression_bench.py - betabench suppression pairs boxes within a
video, suggests min_iou from the valley, and reports what the rules do.

Pairs used to be formed across videos (grouping by timestamp alone), and
min_iou was the 25th percentile of all overlaps, which sits inside the
cluster of genuinely adjacent body parts. Both are pinned here, along
with the impact table that runs the production suppression function, and
the loader that used to hand a preview configuration the real caches.
"""

import os
import random
import shutil
import tempfile
import types
import unittest
from unittest import mock

import betaconst
import betautils_cache_paths as bu_cache

from tests.test_structure_bench import betabench, _quiet_logger


def _box( label, score, x, y, w=100, h=100, t=1.0, video='a' ):
    return { 'class_id': label, 'score': score, 'x': x, 'y': y, 'w': w, 'h': h,
             't': t, '_file_hash': video }


class TestValley( unittest.TestCase ):

    def bimodal( self, low=900, high=300, seed=1 ):
        rng = random.Random( seed )
        return ( [ rng.uniform( 0.0, 0.15 ) for _ in range( low ) ]
                 + [ rng.uniform( 0.85, 0.99 ) for _ in range( high ) ] )

    def test_the_valley_sits_between_the_clusters( self ):
        valley = betabench.iou_valley( self.bimodal() )
        self.assertIsNotNone( valley )
        self.assertGreater( valley, 0.15 )
        self.assertLess( valley, 0.85 )

    def test_adjacency_alone_has_no_valley( self ):
        rng = random.Random( 2 )
        self.assertIsNone( betabench.iou_valley( [ rng.uniform( 0.0, 0.3 ) for _ in range( 1000 ) ] ) )

    def test_a_token_high_cluster_is_not_a_cluster( self ):
        # 1% at the same spot, below the 3% floor: the face_femme case.
        self.assertIsNone( betabench.iou_valley( self.bimodal( low=990, high=10 ) ) )

    def test_a_shallow_trough_is_not_a_valley( self ):
        rng = random.Random( 3 )
        flat = [ rng.uniform( 0.0, 1.0 ) for _ in range( 2000 ) ]
        self.assertIsNone( betabench.iou_valley( flat ) )

    def test_margin_comes_from_the_same_spot_pairs_only( self ):
        ious = [ 0.05 ] * 200 + [ 0.9 ] * 100
        margins = [ 0.5 ] * 200 + [ 0.1 ] * 100
        min_iou, margin, same_spot, wins = betabench.suggest_suppression_rule( ious, margins )
        self.assertAlmostEqual( margin, 0.1 )
        self.assertEqual( same_spot, 100 )
        self.assertEqual( wins, 1.0 )
        self.assertLess( min_iou, 0.9 )

    def test_no_cluster_no_suggestion( self ):
        self.assertIsNone( betabench.suggest_suppression_rule( [ 0.05 ] * 300, [ 0.2 ] * 300 ) )


class TestSuppressionBench( unittest.TestCase ):

    def setUp( self ):
        self.args = types.SimpleNamespace( min_samples=1 )
        self.config = types.SimpleNamespace( backend_name='nudenet_v3' )
        self.parts = { 'exposed_breast': { 'min_prob': 0.2 } }
        patches = [
            mock.patch.object( betabench.betaconfig, 'items_to_censor', [ 'exposed_breast' ] ),
            mock.patch.object( betabench.bu_config, 'get_parts_to_blur', return_value=self.parts ),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup( patch.stop )

    def run_bench( self, boxes, rules ):
        with mock.patch.object( betabench.bu_detector, 'get_class_suppression', return_value=rules ):
            return betabench._bench_suppression_one( self.args, _quiet_logger(), self.config, boxes )

    def test_boxes_in_different_videos_never_pair( self ):
        boxes = [ _box( 'exposed_breast', 0.5, 0, 0, video='a' ),
                  _box( 'face_femme', 0.9, 0, 0, video='b' ) ]
        result = self.run_bench( boxes, {} )
        self.assertEqual( result['suppression'], [] )

    def test_boxes_in_the_same_video_and_instant_pair( self ):
        boxes = [ _box( 'exposed_breast', 0.5, 0, 0 ), _box( 'face_femme', 0.9, 0, 0 ) ]
        pairs = [ row['pair'] for row in self.run_bench( boxes, {} )['suppression'] ]
        self.assertEqual( pairs, [ 'exposed_breast<-face_femme' ] )

    def test_impact_counts_removals_and_how_many_sat_side_by_side( self ):
        rules = { 'exposed_breast': [ { 'suppressed_by': 'face_femme', 'min_iou': 0.01,
                                        'margin': 0.1 } ] }
        boxes = []
        # Three side-by-side removals (small overlap) and one on top.
        for index in range( 3 ):
            boxes += [ _box( 'exposed_breast', 0.5, 0, 0, t=index ),
                       _box( 'face_femme', 0.9, 90, 0, t=index ) ]
        boxes += [ _box( 'exposed_breast', 0.5, 0, 0, t=9 ), _box( 'face_femme', 0.9, 5, 5, t=9 ) ]
        # Below the label's min_prob: removed but not renderable, not counted
        # either as a removal or as a side-by-side one.
        boxes += [ _box( 'exposed_breast', 0.1, 0, 0, t=10 ), _box( 'face_femme', 0.9, 90, 0, t=10 ) ]
        impact = self.run_bench( boxes, rules )['suppression_impact']
        self.assertEqual( len( impact ), 1 )
        self.assertEqual( impact[0]['removed'], 4 )
        self.assertEqual( impact[0]['side_by_side'], 3 )
        self.assertIsNone( impact[0]['suggested'], "no same-spot cluster here, so no suggestion" )
        self.assertEqual( impact[0]['removed_suggested'], 0 )

    def test_side_by_side_never_exceeds_removals_under_two_rules( self ):
        # A breast touching both a face and a belly is removed once, by the
        # first rule the production function reaches. The 2026-10-10 table
        # credited it to both and showed 15,184 side-by-side removals for a
        # rule that removed 14,295.
        rules = { 'exposed_breast': [
            { 'suppressed_by': 'face_femme', 'min_iou': 0.01, 'margin': 0.1 },
            { 'suppressed_by': 'exposed_belly', 'min_iou': 0.01, 'margin': 0.1 } ] }
        boxes = [ _box( 'exposed_breast', 0.5, 0, 0 ), _box( 'face_femme', 0.9, 90, 0 ),
                  _box( 'exposed_belly', 0.9, 0, 90 ) ]
        impact = { row['rule']: row for row in self.run_bench( boxes, rules )['suppression_impact'] }
        for row in impact.values():
            self.assertLessEqual( row['side_by_side'], row['removed'], row['rule'] )
        self.assertEqual( sum( row['removed'] for row in impact.values() ), 1 )
        self.assertEqual( sum( row['side_by_side'] for row in impact.values() ), 1 )

    def test_impact_never_pairs_across_videos( self ):
        rules = { 'exposed_breast': [ { 'suppressed_by': 'face_femme', 'min_iou': 0.01,
                                        'margin': 0.1 } ] }
        boxes = [ _box( 'exposed_breast', 0.5, 0, 0, video='a' ),
                  _box( 'face_femme', 0.9, 0, 0, video='b' ) ]
        impact = self.run_bench( boxes, rules )['suppression_impact']
        self.assertEqual( impact[0]['removed'], 0 )

    def test_impact_with_a_suggestion_runs_the_suggested_rule( self ):
        rules = { 'exposed_breast': [ { 'suppressed_by': 'face_femme', 'min_iou': 0.01,
                                        'margin': 0.0 } ] }
        boxes = []
        for index in range( 60 ):         # side by side
            boxes += [ _box( 'exposed_breast', 0.5, 0, 0, t=index ),
                       _box( 'face_femme', 0.9, 95, 0, t=index ) ]
        for index in range( 30 ):         # on the same spot
            boxes += [ _box( 'exposed_breast', 0.5, 0, 0, t=100 + index ),
                       _box( 'face_femme', 0.9, 2, 2, t=100 + index ) ]
        impact = self.run_bench( boxes, rules )['suppression_impact'][0]
        self.assertEqual( impact['removed'], 90 )
        self.assertIsNotNone( impact['suggested'] )
        self.assertEqual( impact['removed_suggested'], 30, "only the same-spot overlaps" )


class TestConfigurationCaches( unittest.TestCase ):

    def setUp( self ):
        self.tmp = tempfile.mkdtemp( prefix='betasuite-cfgcache-' )
        self.saved = bu_cache.VID_HASH_DIR
        bu_cache.VID_HASH_DIR = self.tmp

    def tearDown( self ):
        bu_cache.VID_HASH_DIR = self.saved
        shutil.rmtree( self.tmp, ignore_errors=True )

    def touch( self, file_hash, suffix='', key='k1' ):
        path = bu_cache.box_hash_path_for( file_hash, 640, 9, 0.2, 'nudenet_v3',
                                           preview_suffix=suffix, det_key=key )
        open( path, 'w' ).close()
        return path

    def config( self, suffix='', key='k1', hashes=( 'aaa', 'bbb' ) ):
        return bu_cache.CachedConfiguration( 'nudenet_v3', [ 640 ], detection_key=key,
                                             preview_suffix=suffix, file_hashes=hashes )

    def test_a_real_configuration_reads_only_real_caches( self ):
        real = self.touch( 'aaa' )
        self.touch( 'aaa', suffix='-preview@60s' )
        found = bu_cache.cache_paths_for_configuration( self.config(), 9, 0.2 )
        self.assertEqual( found, { 'aaa': [ real ] } )

    def test_a_preview_configuration_reads_only_its_preview_caches( self ):
        self.touch( 'aaa' )
        preview = self.touch( 'aaa', suffix='-preview@60s' )
        self.touch( 'aaa', suffix='-preview@120s' )
        found = bu_cache.cache_paths_for_configuration(
            self.config( suffix='-preview@60s' ), 9, 0.2 )
        self.assertEqual( found, { 'aaa': [ preview ] } )

    def test_other_detection_keys_and_videos_are_left_out( self ):
        self.touch( 'aaa', key='k2' )
        self.touch( 'ccc' )
        self.assertEqual( bu_cache.cache_paths_for_configuration( self.config(), 9, 0.2 ), {} )


if __name__ == '__main__':
    unittest.main()
