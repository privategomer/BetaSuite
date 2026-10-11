"""
test_label_content.py - which videos show which censored label.

The 2026-10 suppression A/B picked its videos by total difference, and a
library that is three-quarters breast picks breast videos. label_content
measures each label's on-screen seconds from the raw caches, and mixed
seconds where two censored labels share a sample; analyze_label_content
ranks videos by each. These tests pin the counting (confident score
inclusive, one sample counted once however many boxes, the cache's own
rate), the mixed and pair counts, and the rankings.
"""

import importlib.util
import os
import unittest
from unittest import mock

import betautils_tuning as bu_tuning

REPO_ROOT = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )


def _raw( t, label, score=0.9 ):
    return { 't': t, 'class_id': label, 'score': score, 'x': 0, 'y': 0, 'w': 10, 'h': 10 }


LABELS = [ 'exposed_breast', 'exposed_vulva' ]


class TestLabelContent( unittest.TestCase ):

    def test_seconds_are_samples_over_the_rate( self ):
        raw = [ _raw( index / 10.0, 'exposed_breast' ) for index in range( 20 ) ]
        found = bu_tuning.label_content( raw, LABELS, 10 )
        self.assertEqual( found['seconds'], { 'exposed_breast': 2.0, 'exposed_vulva': 0.0 } )

    def test_two_boxes_in_one_sample_count_once( self ):
        raw = [ _raw( 1 / 9.0, 'exposed_breast' ), _raw( 1 / 9.0, 'exposed_breast' ),
                _raw( 0.11111111, 'exposed_breast' ) ]
        self.assertEqual( bu_tuning.label_content( raw, LABELS, 9 )['seconds']['exposed_breast'],
                          round( 1 / 9.0, 1 ) )

    def test_the_confident_floor_is_inclusive_and_other_labels_ignored( self ):
        raw = [ _raw( 0.0, 'exposed_breast', 0.5 ), _raw( 0.5, 'exposed_breast', 0.49 ),
                _raw( 1.0, 'face_femme' ) ]
        found = bu_tuning.label_content( raw, LABELS, 2 )
        self.assertEqual( found['seconds']['exposed_breast'], 0.5 )
        self.assertNotIn( 'face_femme', found['seconds'] )

    def test_mixed_and_pairs( self ):
        raw = [ _raw( 0.0, 'exposed_breast' ), _raw( 0.0, 'exposed_vulva' ),
                _raw( 1.0, 'exposed_breast' ), _raw( 2.0, 'exposed_vulva' ),
                _raw( 3.0, 'exposed_breast' ), _raw( 3.0, 'exposed_vulva' ),
                _raw( 3.0, 'exposed_anus' ) ]
        found = bu_tuning.label_content( raw, LABELS + [ 'exposed_anus' ], 1 )
        self.assertEqual( found['mixed_seconds'], 2.0 )
        self.assertEqual( found['pair_seconds'], { 'exposed_anus+exposed_breast': 1.0,
                                                   'exposed_anus+exposed_vulva': 1.0,
                                                   'exposed_breast+exposed_vulva': 2.0 } )

    def test_a_low_score_partner_does_not_make_a_sample_mixed( self ):
        raw = [ _raw( 0.0, 'exposed_breast' ), _raw( 0.0, 'exposed_vulva', 0.2 ) ]
        self.assertEqual( bu_tuning.label_content( raw, LABELS, 1 )['mixed_seconds'], 0.0 )

    def test_each_video_uses_its_own_cache_rate( self ):
        # Three samples from an 18 fps cache are a sixth of a second; read
        # at the global 9 fps they would be a third.
        raw = [ _raw( index, 'exposed_vulva' ) for index in range( 3 ) ]
        with mock.patch.object( bu_tuning, 'raw_boxes_for_reading', return_value=raw ), \
                mock.patch.object( bu_tuning, 'forget_raw_boxes' ) as forget:
            found = bu_tuning.content_by_video(
                { 'h1': [ '/c/h1-nudenet_v3-3-640-18-0.120-dabc.gz' ] }, LABELS )
        self.assertEqual( found['h1']['seconds']['exposed_vulva'], round( 3 / 18.0, 1 ) )
        forget.assert_called_once()


def _load_tool():
    path = os.path.join( REPO_ROOT, 'tools', 'analysis', 'analyze_label_content.py' )
    spec = importlib.util.spec_from_file_location( '_label_content_under_test', path )
    module = importlib.util.module_from_spec( spec )
    spec.loader.exec_module( module )
    return module


class TestRankings( unittest.TestCase ):

    def setUp( self ):
        self.tool = _load_tool()
        content = {
            'a': { 'seconds': { 'exposed_breast': 900.0, 'exposed_vulva': 100.0 }, 'mixed_seconds': 50.0,
                   'pair_seconds': { 'exposed_breast+exposed_vulva': 50.0 } },
            'b': { 'seconds': { 'exposed_breast': 200.0, 'exposed_vulva': 600.0 }, 'mixed_seconds': 300.0,
                   'pair_seconds': { 'exposed_breast+exposed_vulva': 300.0 } },
            'c': { 'seconds': { 'exposed_breast': 0.0, 'exposed_vulva': 70.0 }, 'mixed_seconds': 0.0,
                   'pair_seconds': { 'exposed_breast+exposed_vulva': 0.0 } },
        }
        self.rows = self.tool.content_rows( content, { 'a': 'A.mp4', 'b': 'B.mp4' }, { 'a': 2000.0 } )

    def test_rows_carry_names_and_fall_back_to_the_hash( self ):
        names = sorted( row['name'] for row in self.rows )
        self.assertEqual( names, [ 'A.mp4', 'B.mp4', 'c' ] )

    def test_by_label_most_first_and_zeros_left_out( self ):
        self.assertEqual( [ row['name'] for row in self.tool.top_by( self.rows, 'exposed_vulva', 5 ) ],
                          [ 'B.mp4', 'A.mp4', 'c' ] )
        self.assertEqual( [ row['name'] for row in self.tool.top_by( self.rows, 'exposed_breast', 5 ) ],
                          [ 'A.mp4', 'B.mp4' ] )
        self.assertEqual( [ row['name'] for row in self.tool.top_by( self.rows, 'mixed', 1 ) ], [ 'B.mp4' ] )

    def test_leaning_is_share_of_labelled_seconds_with_a_floor( self ):
        ranked = self.tool.top_leaning( self.rows, 'exposed_vulva', 5 )
        self.assertEqual( [ row['name'] for row in ranked ], [ 'c', 'B.mp4', 'A.mp4' ] )
        self.assertEqual( self.tool.leaning( ranked[0], 'exposed_vulva' ), 1.0 )
        self.assertEqual( [ row['name'] for row in self.tool.top_leaning( self.rows, 'exposed_vulva', 5,
                                                                          min_seconds=80 ) ],
                          [ 'B.mp4', 'A.mp4' ] )
        # The floor itself qualifies: c has exactly 70s.
        self.assertIn( 'c', [ row['name'] for row in self.tool.top_leaning(
            self.rows, 'exposed_vulva', 5, min_seconds=70 ) ] )

    def test_report_prints_every_ranking( self ):
        lines = []
        self.tool.print_report( self.rows, LABELS, 3, out=lines.append )
        text = "\n".join( lines )
        for heading in ( 'Most exposed_breast seconds', 'Most exposed_vulva seconds', 'Most MIXED',
                         'exposed_vulva-leaning' ):
            self.assertIn( heading, text )
        self.assertIn( 'breast+vulva 300s', text )


if __name__ == '__main__':
    unittest.main()
