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
from unittest import mock

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


def _failure( name='a.mkv', at=1_700_000_100, preview=False ):
    return { 'file': name, 'error': "RenderError('chunk 5 failed')", 'preview_mode': preview,
             'timestamp': at }


def _row_for( name, at, preview=False ):
    row = _row( preview=preview )
    row.update( { 'file': name, 'timestamp': at } )
    return row


class TestFailuresToReport( unittest.TestCase ):

    def test_a_failure_with_no_later_render_is_reported_as_such( self ):
        entries = summarize_run.failures_to_report( [ _failure() ], [], include_preview=False )
        self.assertEqual( entries, [ ( _failure(), None ) ] )

    def test_a_later_render_of_the_same_file_is_named_with_its_time( self ):
        rows = [ _row_for( 'a.mkv', 1_700_000_050 ),    # before the failure: not a fix
                 _row_for( 'a.mkv', 1_700_000_300 ),
                 _row_for( 'a.mkv', 1_700_000_200 ),
                 _row_for( 'b.mkv', 1_700_000_150 ) ]
        entries = summarize_run.failures_to_report( [ _failure() ], rows, include_preview=False )
        self.assertEqual( entries[0][1], 1_700_000_200, "the EARLIEST later render" )

    def test_a_render_at_exactly_the_failure_time_is_not_later( self ):
        rows = [ _row_for( 'a.mkv', 1_700_000_100 ) ]
        entries = summarize_run.failures_to_report( [ _failure() ], rows, include_preview=False )
        self.assertIsNone( entries[0][1] )

    def test_a_preview_render_does_not_clear_a_full_run_failure( self ):
        rows = [ _row_for( 'a.mkv', 1_700_000_200, preview=True ) ]
        entries = summarize_run.failures_to_report( [ _failure() ], rows, include_preview=True )
        self.assertIsNone( entries[0][1] )

    def test_preview_failures_follow_include_preview( self ):
        failures = [ _failure( preview=True ) ]
        self.assertEqual( summarize_run.failures_to_report( failures, [], include_preview=False ), [] )
        self.assertEqual( len( summarize_run.failures_to_report( failures, [], include_preview=True ) ), 1 )

    def test_since_and_the_last_window_drop_older_failures( self ):
        failures = [ _failure( 'old.mkv', at=100 ), _failure( 'on.mkv', at=200 ),
                     _failure( 'new.mkv', at=300 ) ]
        by_since = summarize_run.failures_to_report( failures, [], False, since_timestamp=200 )
        by_window = summarize_run.failures_to_report( failures, [], False, window_start_timestamp=300 )
        self.assertEqual( [ failure['file'] for failure, _later in by_since ], [ 'on.mkv', 'new.mkv' ],
            "--since keeps rows AT the cutoff, so failures at it stay too" )
        self.assertEqual( [ failure['file'] for failure, _later in by_window ], [ 'new.mkv' ],
            "a failure AT the window start is inside it" )

    def test_oldest_first( self ):
        failures = [ _failure( 'second.mkv', at=300 ), _failure( 'first.mkv', at=100 ) ]
        entries = summarize_run.failures_to_report( failures, [], False )
        self.assertEqual( [ failure['file'] for failure, _later in entries ],
                          [ 'first.mkv', 'second.mkv' ] )


class TestFailuresInTheReport( unittest.TestCase ):
    """main() end to end, reading a stats file and the failures file beside it."""

    def _report( self, rows, failures, *args ):
        import contextlib
        import io
        import betautils_log as bu_log
        with tempfile.TemporaryDirectory() as tmp:
            stats_path = os.path.join( tmp, 'betasuite_stats.jsonl' )
            for path, records in ( ( stats_path, rows ),
                                   ( bu_log.failures_path_for( stats_path ), failures ) ):
                if records:
                    with open( path, 'w', encoding='UTF-8' ) as fout:
                        for record in records:
                            fout.write( json.dumps( record ) + "\n" )
            output = io.StringIO()
            argv = [ 'summarize_run.py', '--stats-path', stats_path ] + list( args )
            with mock.patch.object( sys, 'argv', argv ), contextlib.redirect_stdout( output ):
                try:
                    summarize_run.main()
                except SystemExit:
                    pass
            return output.getvalue()

    def test_failed_files_are_listed_after_the_numbers( self ):
        text = self._report( [ _row_for( 'b.mkv', 1_700_000_000 ) ],
                             [ _failure( 'a.mkv', at=1_700_000_100 ) ] )
        self.assertIn( "FAILED FILES: 1 failure(s), 1 not rendered since", text )
        self.assertIn( "a.mkv  [not rendered since]", text )
        self.assertIn( "RenderError('chunk 5 failed')", text )
        self.assertLess( text.index( "FAILED FILES" ), text.index( "next steps" ) )

    def test_no_failures_no_section( self ):
        text = self._report( [ _row_for( 'b.mkv', 1_700_000_000 ) ], [] )
        self.assertNotIn( "FAILED FILES", text )

    def test_a_run_where_every_file_failed_still_shows_its_failures( self ):
        text = self._report( [], [ _failure( 'a.mkv' ) ] )
        self.assertIn( "no stats rows found", text )
        self.assertIn( "a.mkv  [not rendered since]", text )

    def test_last_keeps_only_failures_inside_the_rows_it_kept( self ):
        rows = [ _row_for( 'x.mkv', 1_700_000_000 ), _row_for( 'y.mkv', 1_700_000_500 ) ]
        failures = [ _failure( 'early.mkv', at=1_700_000_100 ),
                     _failure( 'late.mkv', at=1_700_000_600 ) ]
        text = self._report( rows, failures, '--last', '1' )
        self.assertIn( "late.mkv", text )
        self.assertNotIn( "early.mkv", text )


if __name__ == '__main__':
    unittest.main()
