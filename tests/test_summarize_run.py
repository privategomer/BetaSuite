"""
test_summarize_run.py - coverage for the post-run stats summary.

summarize_run.py is the first thing read after a real run, so its
arithmetic decides where the next evening of tuning goes. The two
things worth guarding are that it never pools rows from different
configurations (a mean across two backends is a number about nothing)
and that a malformed or missing stats file degrades into a message
rather than a traceback - stats are appended by a running pipeline, so
a half-written last line is an ordinary thing to find.
"""

import importlib.util
import json
import os
import sys
import tempfile
import unittest

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )


def _load_tool():
    path = os.path.join( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ),
                         'tools', 'analysis', 'summarize_run.py' )
    spec = importlib.util.spec_from_file_location( 'summarize_run_under_test', path )
    module = importlib.util.module_from_spec( spec )
    spec.loader.exec_module( module )
    return module


summarize_run = _load_tool()


def _row( backend='nudenet_v3', variant='320n', sizes=( 320, ), detection=10.0,
          encode=90.0, total=105.0, frames=2400, fps=24.0, labels=None,
          preview=False, workers=1 ):
    return {
        'detector_backend': backend,
        'detector_variant': variant,
        'picture_sizes': list( sizes ),
        'video_censor_fps': 9.0,
        'nn_batch_size': 2,
        'preview_mode': preview,
        'detection_seconds': detection,
        'encode_seconds': encode,
        'total_seconds': total,
        'video_frames': frames,
        'video_fps': fps,
        'render_chunks': 1,
        'render_chunks_reused': 0,
        'render_frames_written': frames,
        'render_render_workers': workers,
        'track_tracks': 5,
        'track_interpolated': 20,
        'label_counts': labels if labels is not None else { 'exposed_breast': 100 },
        'timestamp': 1_700_000_000,
    }


class TestGrouping( unittest.TestCase ):

    def test_variant_alone_separates_two_groups( self ):
        rows = [ _row( variant='320n', sizes=( 320, ) ),
                 _row( variant='640m', sizes=( 640, ) ) ]
        keys = { summarize_run.group_key( row ) for row in rows }
        self.assertEqual( len( keys ), 2,
            "a 320n run and a 640m run must never be averaged together - the difference "
            "between them is the entire reason both were run" )

    def test_identical_configuration_groups_together( self ):
        rows = [ _row(), _row( detection=12.0 ) ]
        keys = { summarize_run.group_key( row ) for row in rows }
        self.assertEqual( len( keys ), 1 )

    def test_preview_rows_are_a_separate_group( self ):
        keys = { summarize_run.group_key( _row( preview=preview ) )
                 for preview in ( True, False ) }
        self.assertEqual( len( keys ), 2,
            "a preview slice's totals describe a few seconds; pooling them with a real run "
            "makes both numbers wrong" )


class TestAggregation( unittest.TestCase ):

    def test_stage_seconds_and_video_duration_sum( self ):
        rows = [ _row( detection=10.0, encode=90.0, total=105.0, frames=2400, fps=24.0 ),
                 _row( detection=20.0, encode=80.0, total=110.0, frames=2400, fps=24.0 ) ]
        totals = summarize_run.summarise_group( rows )
        self.assertEqual( totals['files'], 2 )
        self.assertAlmostEqual( totals['detection_seconds'], 30.0 )
        self.assertAlmostEqual( totals['encode_seconds'], 170.0 )
        self.assertAlmostEqual( totals['video_seconds'], 200.0 )

    def test_label_counts_accumulate_across_files( self ):
        rows = [ _row( labels={ 'exposed_breast': 100, 'exposed_vulva': 10 } ),
                 _row( labels={ 'exposed_breast': 50 } ) ]
        totals = summarize_run.summarise_group( rows )
        self.assertEqual( totals['label_counts'],
                          { 'exposed_breast': 150, 'exposed_vulva': 10 } )

    def test_worker_histogram_records_what_was_actually_used( self ):
        rows = [ _row( workers=1 ), _row( workers=1 ), _row( workers=2 ) ]
        totals = summarize_run.summarise_group( rows )
        self.assertEqual( totals['worker_counts'], { 1: 2, 2: 1 },
            "the worker histogram is how 'render_workers is set to 8 but nothing ran in "
            "parallel' becomes visible" )

    def test_a_row_with_no_duration_does_not_divide_by_zero( self ):
        totals = summarize_run.summarise_group( [ _row( frames=0, fps=0 ) ] )
        self.assertEqual( totals['video_seconds'], 0.0 )


class TestMalformedInput( unittest.TestCase ):

    def test_a_truncated_last_line_is_skipped_not_fatal( self ):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join( tmp, 'stats.jsonl' )
            with open( path, 'w', encoding='UTF-8' ) as fout:
                fout.write( json.dumps( _row() ) + "\n" )
                fout.write( '{"detector_backend": "nudenet' )  # interrupted mid-write
            rows, skipped = summarize_run.load_rows( path )
            self.assertEqual( len( rows ), 1 )
            self.assertEqual( skipped, 1 )

    def test_a_missing_file_reads_as_empty( self ):
        rows, skipped = summarize_run.load_rows( '/nonexistent/stats.jsonl' )
        self.assertEqual( rows, [] )
        self.assertEqual( skipped, 0 )


if __name__ == '__main__':
    unittest.main()
