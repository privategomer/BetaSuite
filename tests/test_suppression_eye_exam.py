"""
test_suppression_eye_exam.py - from A/B clips to chosen rules.

The A/B clips change several rules at once, so a difference on screen
says nothing about which rule caused it. The tool attributes every
differing box to the rule that dropped its detection (through the
suppression observer, never a copy of the rule), lists each difference
with its time in the clip, collects one verdict per row, and turns the
verdicts into a rule set, one rule at a time, like an eye exam.
"""

import csv
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

import betautils_censor as bu_censor
import betautils_track as bu_track


def _load_tool():
    path = os.path.join( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ),
                         'tools', 'tuning', 'suppression_ab.py' )
    spec = importlib.util.spec_from_file_location( '_suppression_eye_exam', path )
    module = importlib.util.module_from_spec( spec )
    spec.loader.exec_module( module )
    return module


TOOL = _load_tool()
REPO = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )


def _raw( label, t, x, score, w=100, h=100 ):
    return { 'class_id': label, 't': t, 'x': x, 'y': 0, 'w': w, 'h': h, 'score': score }


def _box( start, end, x=0, label='exposed_breast' ):
    return { 'start': start, 'end': end, 'x': x, 'y': 0, 'w': 100, 'h': 100, 'label': label }


class _Logger:
    def __init__( self ):
        self.lines = []

    def _note( self, message, *args ):
        self.lines.append( message )

    trace = debug = info = warning = error = _note


CURRENT = { 'exposed_breast': [ { 'suppressed_by': 'face_femme', 'margin': 0.1, 'min_iou': 0.01 },
                                { 'suppressed_by': 'covered_breast', 'margin': 0.05, 'min_iou': 0.03 } ],
            'exposed_vulva': [ { 'suppressed_by': 'exposed_anus', 'margin': 0.1, 'min_iou': 0.08 } ] }
PROPOSED = { 'exposed_breast': [ { 'suppressed_by': 'covered_breast', 'margin': 0.081,
                                   'min_iou': 0.425 } ] }


class TestSuppressionObserver( unittest.TestCase ):

    def test_each_dropped_detection_is_reported_with_its_rule_and_suppressor( self ):
        seen = []

        class Watcher( bu_track.TrackingObserver ):
            def on_suppressed( self, raw, rule_key, suppressor ):
                seen.append( ( raw['class_id'], rule_key, suppressor['class_id'] ) )

        raws = [ _raw( 'exposed_breast', 1.0, 0, 0.5 ), _raw( 'face_femme', 1.0, 10, 0.9 ),
                 _raw( 'exposed_breast', 2.0, 0, 0.9 ), _raw( 'face_femme', 2.0, 10, 0.95 ) ]
        with bu_track.tracking_observer( Watcher() ):
            surviving, counts = bu_track.apply_class_suppression( raws, CURRENT )
        self.assertEqual( seen, [ ( 'exposed_breast', 'exposed_breast<-face_femme', 'face_femme' ) ] )
        self.assertEqual( counts['exposed_breast<-face_femme']['total'], 1 )
        self.assertEqual( len( surviving ), 3 )

    def test_without_an_observer_nothing_changes( self ):
        raws = [ _raw( 'exposed_breast', 1.0, 0, 0.5 ), _raw( 'face_femme', 1.0, 10, 0.9 ) ]
        surviving, _counts = bu_track.apply_class_suppression( raws, CURRENT )
        self.assertEqual( [ raw['class_id'] for raw in surviving ], [ 'face_femme' ] )


class TestDebugNote( unittest.TestCase ):

    def test_the_note_is_drawn_under_the_label( self ):
        box = { 'x': 10, 'y': 10, 'w': 150, 'h': 150, 'label': 'exposed_breast', 'score': 0.9,
                'start': 1.0, 'end': 2.0, 'censor_style': { 'type': 'debug', 'color': ( 255, 0, 255 ) } }
        plain = bu_censor.debug_image( np.zeros( ( 200, 200, 3 ), np.uint8 ), dict( box ) )
        noted = bu_censor.debug_image( np.zeros( ( 200, 200, 3 ), np.uint8 ),
                                       dict( box, debug_note='<-face_femme' ) )
        band = ( slice( 95, 115 ), slice( 15, 150 ) )
        self.assertEqual( int( plain[band].sum() ), 0 )
        self.assertGreater( int( noted[band].sum() ), 0 )


class TestAttribution( unittest.TestCase ):

    def test_the_rule_whose_dropped_detection_sits_inside_the_live_box( self ):
        suppressed = [ ( 1.5, 'exposed_breast', 50, 50, 'exposed_breast<-face_femme' ),
                       ( 1.6, 'exposed_breast', 50, 50, 'exposed_breast<-face_femme' ),
                       ( 1.7, 'exposed_breast', 50, 50, 'exposed_breast<-covered_breast' ),
                       ( 9.0, 'exposed_breast', 50, 50, 'exposed_breast<-covered_breast' ),
                       ( 1.5, 'exposed_vulva', 50, 50, 'exposed_vulva<-exposed_anus' ),
                       ( 1.5, 'exposed_breast', 900, 50, 'exposed_breast<-covered_breast' ) ]
        boxes = [ _box( 1.0, 2.0 ), _box( 5.0, 6.0 ), _box( 1.0, 2.0, label='exposed_vulva' ) ]
        self.assertEqual( TOOL.attribute( boxes, suppressed ),
                          [ 'exposed_breast<-face_femme', TOOL.TRACKING_CAUSE,
                            'exposed_vulva<-exposed_anus' ] )

    def test_a_detection_in_the_same_bucket_but_after_the_box_is_not_the_cause( self ):
        suppressed = [ ( 1.45, 'exposed_breast', 50, 50, 'exposed_breast<-face_femme' ) ]
        self.assertEqual( TOOL.attribute( [ _box( 1.0, 1.2 ) ], suppressed ), [ TOOL.TRACKING_CAUSE ] )

    def test_a_long_box_finds_detections_in_any_bucket( self ):
        suppressed = [ ( 12.3, 'exposed_breast', 50, 50, 'exposed_breast<-face_femme' ) ]
        self.assertEqual( TOOL.attribute( [ _box( 0.0, 20.0 ) ], suppressed ),
                          [ 'exposed_breast<-face_femme' ] )

    def test_compare_attributes_each_side_to_the_other_sides_rules( self ):
        def fake_replay( video, caches, backend, rules, tracking, observer=None ):
            if rules is CURRENT:
                observer.on_suppressed( _raw( 'exposed_breast', 1.5, 0, 0.5 ),
                                        'exposed_breast<-face_femme', _raw( 'face_femme', 1.5, 0, 0.9 ) )
                return [ _box( 5.0, 6.0, x=600 ) ]
            observer.on_suppressed( _raw( 'exposed_breast', 5.5, 600, 0.5 ),
                                    'exposed_breast<-covered_breast', _raw( 'covered_breast', 5.5, 600, 0.9 ) )
            return [ _box( 1.0, 2.0 ) ]
        context = { 'shot_cut_times': [], 'profile_name': 'scene', 'sample_fps': 9 }
        with mock.patch.object( TOOL.bu_tuning, 'replay_tracking_context', return_value=context ):
            result = TOOL.compare_with_causes( 'h', '/v.mp4', [ 'c.gz' ], 'nudenet_v3', CURRENT,
                                               PROPOSED, replay=fake_replay )
        self.assertEqual( result['proposed_only_causes'], [ 'exposed_breast<-face_femme' ] )
        self.assertEqual( result['current_only_causes'], [ 'exposed_breast<-covered_breast' ] )


class TestWindowEvents( unittest.TestCase ):

    def test_events_are_clipped_grouped_merged_and_clip_relative( self ):
        boxes = [ _box( 99.0, 100.2 ), _box( 100.4, 100.6 ), _box( 103.0, 103.2 ),
                  _box( 101.0, 101.2, label='exposed_vulva' ), _box( 120.0, 121.0 ) ]
        causes = [ 'r1', 'r1', 'r1', 'r2', 'r1' ]
        events = TOOL.window_events( boxes, causes, 100.0, 115.0, 'RIGHT only' )
        self.assertEqual( events, [
            { 'side': 'RIGHT only', 'label': 'exposed_breast', 'cause': 'r1', 'boxes': 3,
              'spans': [ ( 0.0, 0.6 ), ( 3.0, 3.2 ) ] },
            { 'side': 'RIGHT only', 'label': 'exposed_vulva', 'cause': 'r2', 'boxes': 1,
              'spans': [ ( 1.0, 1.2 ) ] } ] )

    def test_cause_seconds_largest_first( self ):
        totals = TOOL.cause_seconds( [ _box( 0, 1 ), _box( 0, 3 ), _box( 0, 0.5 ) ], [ 'a', 'b', 'a' ] )
        self.assertEqual( list( totals.items() ), [ ( 'b', 3.0 ), ( 'a', 1.5 ) ] )


def _ranking():
    return { 'backend': 'nudenet_v3', 'current_rules': CURRENT, 'proposed_rules': PROPOSED,
             'labels': [ 'exposed_breast' ], 'chosen': [ 'h1' ],
             'videos': [ { 'file_hash': 'h1', 'video': '/v/My Video.mp4', 'name': 'My Video.mp4',
                           'cache_paths': [ 'c.gz' ], 'windows': [ ( 100.0, 115.0 ), ( 300.0, 315.0 ) ] } ] }


def _fake_compare( *args ):
    return { 'current_boxes': [], 'proposed_boxes': [], 'both': [],
             'proposed_only': [ _box( 102.0, 103.0 ), _box( 108.0, 108.5 ) ],
             'proposed_only_causes': [ 'exposed_breast<-face_femme', TOOL.TRACKING_CAUSE ],
             'current_only': [ _box( 110.0, 110.3, label='exposed_vulva' ) ],
             'current_only_causes': [ 'exposed_breast<-covered_breast' ] }


class TestExplain( unittest.TestCase ):

    def setUp( self ):
        self.dir = tempfile.mkdtemp( prefix='betasuite-eye-' )
        self.args = types.SimpleNamespace( out_dir=self.dir, feedback=None )

    def tearDown( self ):
        shutil.rmtree( self.dir, ignore_errors=True )

    def test_every_difference_becomes_a_row_with_time_side_cause_and_change( self ):
        rows = TOOL.phase_explain( self.args, _Logger(), _ranking(), compare=_fake_compare )
        self.assertEqual( [ row['id'] for row in rows ], [ 'v1c1.1', 'v1c1.2', 'v1c1.3' ] )
        first = rows[0]
        self.assertEqual( first['clip'], 'My_Video_clip01_100s_ab.mp4' )
        self.assertEqual( first['when_in_clip'], '0:02.0-0:03.0' )
        self.assertEqual( first['source_seconds'], '102.0-103.0' )
        self.assertEqual( first['censored_on'], 'RIGHT only' )
        self.assertEqual( first['change'], 'dropped' )
        self.assertIn( 'no single change', rows[1]['change'] )
        self.assertEqual( rows[2]['censored_on'], 'LEFT only' )
        self.assertIn( 'retuned', rows[2]['change'] )
        with open( os.path.join( self.dir, 'clips.md' ) ) as handle:
            text = handle.read()
        self.assertIn( '| v1c1.1 | 0:02.0-0:03.0 | exposed_breast | RIGHT only |', text )
        self.assertIn( 'No differences inside this clip.', text )
        with open( os.path.join( self.dir, 'feedback.csv' ) ) as handle:
            self.assertEqual( len( list( csv.DictReader( handle ) ) ), 3 )

    def test_an_existing_feedback_file_is_never_overwritten( self ):
        path = os.path.join( self.dir, 'feedback.csv' )
        with open( path, 'w' ) as handle:
            handle.write( 'mine' )
        TOOL.phase_explain( self.args, _Logger(), _ranking(), compare=_fake_compare )
        with open( path ) as handle:
            self.assertEqual( handle.read(), 'mine' )
        self.assertTrue( os.path.exists( os.path.join( self.dir, 'feedback_template.csv' ) ) )


class TestDecide( unittest.TestCase ):

    def rows( self, *entries ):
        return [ { 'cause': cause, 'verdict': verdict } for cause, verdict in entries ]

    def test_verdict_words_and_tallies( self ):
        tally = TOOL.tally_verdicts( self.rows(
            ( 'a', 'Right' ), ( 'a', 'r' ), ( 'a', 'left' ), ( 'a', '=' ), ( 'a', '' ), ( 'a', '?' ) ) )
        self.assertEqual( tally['a'], { 'proposed': 2, 'current': 1, 'same': 1, 'unjudged': 2 } )

    def test_each_changed_rule_follows_its_own_verdicts( self ):
        tally = TOOL.tally_verdicts( self.rows(
            ( 'exposed_breast<-face_femme', 'right' ), ( 'exposed_breast<-face_femme', 'right' ),
            ( 'exposed_breast<-face_femme', 'left' ),
            ( 'exposed_breast<-covered_breast', 'left' ),
            ( 'exposed_vulva<-exposed_anus', 'right' ), ( 'exposed_vulva<-exposed_anus', 'left' ),
            ( TOOL.TRACKING_CAUSE, 'right' ) ) )
        rules, decisions = TOOL.choose_rules( CURRENT, PROPOSED, tally )
        self.assertEqual( decisions, {
            'exposed_breast<-covered_breast': 'current',
            'exposed_breast<-face_femme': 'proposed',
            'exposed_vulva<-exposed_anus': 'current' } )
        # face_femme dropped (the proposed side won); covered_breast kept as is; the tie keeps anus
        self.assertEqual( rules, {
            'exposed_breast': [ CURRENT['exposed_breast'][1] ],
            'exposed_vulva': [ CURRENT['exposed_vulva'][0] ] } )

    def test_unchanged_rules_are_not_up_for_decision( self ):
        same = { 'exposed_breast': [ { 'suppressed_by': 'covered_breast', 'margin': 0.081, 'min_iou': 0.425 } ] }
        tally = TOOL.tally_verdicts( self.rows( ( 'exposed_breast<-covered_breast', 'left' ) ) )
        rules, decisions = TOOL.choose_rules( same, same, tally )
        self.assertEqual( decisions, {} )
        self.assertEqual( rules, same )

    def test_an_added_rule_is_added_only_when_it_wins( self ):
        proposed = dict( PROPOSED, exposed_vulva=[ { 'suppressed_by': 'exposed_penis', 'margin': 0.1,
                                                     'min_iou': 0.6 } ] )
        current = { 'exposed_vulva': [] }
        won = TOOL.tally_verdicts( self.rows( ( 'exposed_vulva<-exposed_penis', 'right' ) ) )
        rules, _decisions = TOOL.choose_rules( current, proposed, won )
        self.assertEqual( rules['exposed_vulva'], proposed['exposed_vulva'] )
        rules, decisions = TOOL.choose_rules( current, proposed, {} )
        self.assertNotIn( 'exposed_vulva', rules )
        self.assertEqual( decisions['exposed_vulva<-exposed_penis'], 'current (no verdict yet)' )

    def test_a_winning_retune_replaces_the_rule_and_silence_changes_nothing( self ):
        tally = TOOL.tally_verdicts( self.rows( ( 'exposed_breast<-covered_breast', 'right' ) ) )
        rules, decisions = TOOL.choose_rules( CURRENT, PROPOSED, tally )
        self.assertEqual( rules['exposed_breast'], [ CURRENT['exposed_breast'][0],
                                                     PROPOSED['exposed_breast'][0] ] )
        self.assertEqual( decisions['exposed_breast<-face_femme'], 'current (no verdict yet)' )

    def test_decide_reads_the_csv_and_writes_chosen_rules( self ):
        directory = tempfile.mkdtemp( prefix='betasuite-decide-' )
        try:
            with open( os.path.join( directory, 'feedback.csv' ), 'w', newline='' ) as handle:
                writer = csv.DictWriter( handle, fieldnames=TOOL.FEEDBACK_COLUMNS )
                writer.writeheader()
                writer.writerow( { 'id': 'v1c1.1', 'cause': 'exposed_breast<-face_femme', 'verdict': 'R' } )
            logger = _Logger()
            args = types.SimpleNamespace( out_dir=directory, feedback=None )
            TOOL.phase_decide( args, logger, _ranking() )
            with open( os.path.join( directory, 'chosen_rules.json' ) ) as handle:
                chosen = json.load( handle )
            self.assertEqual( [ rule['suppressed_by'] for rule in chosen['exposed_breast'] ],
                              [ 'covered_breast' ] )
            self.assertTrue( any( 'ab_compare.sh confirm' in line for line in logger.lines ),
                             "the next round is one command, not a path to paste" )
        finally:
            shutil.rmtree( directory, ignore_errors=True )


FAKE_PYTHON = r'''#!/bin/bash
if [ "${1:-}" = "-c" ]; then exit 0; fi
case " $* " in *" --run-experiment "*) echo "min_track_hits"; exit 0 ;; esac
echo "$*" >> "__CALLS__"
'''


class TestWrapperModes( unittest.TestCase ):

    def setUp( self ):
        self.root = tempfile.mkdtemp( prefix='betasuite-abwrap-' )
        self.repo = os.path.join( self.root, 'repo' )
        for relative in ( 'tools/tuning/suppression_ab.sh', 'tools/tuning/ab_compare.sh',
                          'tools/tuning/harness_lib.sh', 'tools/lib/python_env.sh' ):
            target = os.path.join( self.repo, relative )
            os.makedirs( os.path.dirname( target ), exist_ok=True )
            shutil.copy( os.path.join( REPO, relative ), target )
        self.calls = os.path.join( self.root, 'calls.txt' )
        self.python = os.path.join( self.root, 'python' )
        with open( self.python, 'w' ) as handle:
            handle.write( FAKE_PYTHON.replace( '__CALLS__', self.calls ) )
        os.chmod( self.python, 0o755 )

    def tearDown( self ):
        shutil.rmtree( self.root, ignore_errors=True )

    def run_mode( self, *args ):
        environment = dict( os.environ, PYTHON_BIN=self.python, PYTHON_BIN_SOURCE='stand-in' )
        return subprocess.run( [ 'bash', 'tools/tuning/suppression_ab.sh' ] + list( args ),
                               cwd=self.repo, env=environment, stdin=subprocess.DEVNULL,
                               capture_output=True, text=True, timeout=60 )

    def make_run( self, name ):
        path = os.path.join( self.root, 'output', 'benchmarks', name )
        os.makedirs( path )
        with open( os.path.join( path, 'ranking.json' ), 'w' ) as handle:
            handle.write( '{}' )
        return path

    def test_explain_uses_the_newest_run( self ):
        self.make_run( 'suppression_ab_20261010_120000' )
        newest = self.make_run( 'suppression_ab_20261010_145028' )
        result = self.run_mode( 'explain' )
        self.assertEqual( result.returncode, 0, result.stdout + result.stderr )
        with open( self.calls ) as handle:
            call = handle.read()
        self.assertIn( '--phase explain', call )
        self.assertIn( '--out-dir %s'%( newest, ), call )

    def test_decide_takes_a_named_run_and_no_run_is_an_error( self ):
        older = self.make_run( 'suppression_ab_20261010_120000' )
        self.make_run( 'suppression_ab_20261010_145028' )
        self.assertEqual( self.run_mode( 'decide', older ).returncode, 0 )
        with open( self.calls ) as handle:
            self.assertIn( '--phase decide --out-dir %s'%( older, ), handle.read() )
        shutil.rmtree( os.path.join( self.root, 'output' ) )
        result = self.run_mode( 'explain' )
        self.assertEqual( result.returncode, 1 )
        self.assertIn( 'run a round first', result.stderr )

    def test_an_unknown_mode_is_refused( self ):
        self.assertEqual( self.run_mode( 'frobnicate' ).returncode, 2 )

    def test_newest_is_by_the_name_across_both_prefixes( self ):
        self.make_run( 'suppression_ab_20261010_145028' )
        newest = self.make_run( 'ab_min_track_hits_20261011_090000' )
        older = self.make_run( 'ab_suppression_20261009_090000' )
        os.utime( older )   # touched last: mtime must not decide
        self.assertEqual( self.run_mode( 'explain' ).returncode, 0 )
        with open( self.calls ) as handle:
            self.assertIn( '--out-dir %s'%( newest, ), handle.read() )

    def test_confirm_runs_the_chosen_side_on_videos_not_yet_shown( self ):
        previous = self.make_run( 'ab_min_track_hits_20261011_090000' )
        with open( os.path.join( previous, 'ranking.json' ), 'w' ) as handle:
            handle.write( '{"experiment": "min_track_hits", "focus": ["exposed_vulva"], "chosen": ["h1"]}' )
        with open( os.path.join( previous, 'chosen.json' ), 'w' ) as handle:
            handle.write( '{"exposed_breast": {"min_track_hits": 1}}' )
        self.run_mode( 'confirm' )
        with open( self.calls ) as handle:
            calls = handle.read()
        rank = [ line for line in calls.splitlines() if '--phase rank' in line ][0]
        self.assertIn( '--experiment min_track_hits', rank )
        self.assertIn( '--confirm-from %s'%( previous, ), rank )
        self.assertIn( 'ab_min_track_hits_', rank )
        self.assertIn( '_confirm', rank )

    def test_confirm_with_nothing_chosen_runs_nothing( self ):
        previous = self.make_run( 'ab_min_track_hits_20261011_090000' )
        with open( os.path.join( previous, 'chosen.json' ), 'w' ) as handle:
            handle.write( '{}' )
        result = self.run_mode( 'confirm' )
        self.assertNotEqual( result.returncode, 0 )
        self.assertFalse( os.path.exists( self.calls ) and '--phase rank' in open( self.calls ).read() )

    def test_the_old_name_runs_the_suppression_experiment( self ):
        self.run_mode()
        with open( self.calls ) as handle:
            rank = [ line for line in handle.read().splitlines() if '--phase rank' in line ][0]
        self.assertIn( '--experiment suppression', rank )


if __name__ == '__main__':
    unittest.main()
