"""
test_ab_compare.py - the A/B engine across experiments.

ab_compare.py replays every cached video as configured and with one
change, ranks by how much the censoring differs, and turns per-row
verdicts into the chosen config. What changes is an experiment
(ab_experiments.py). These tests pin each experiment's parts (what it
proposes, how the pipeline reads a side, what caused a difference, what
the verdicts choose, the lines to paste) and the engine pieces added
with them: the focus on one label, leaving out videos an earlier round
showed, old ranking files, and auto_tune's review values, including the
tightening candidate its price rule can never pick.
"""

import argparse
import copy
import csv
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

REPO_ROOT = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
sys.path.insert( 0, os.path.join( REPO_ROOT, 'tools', 'tuning' ) )

import betaconfig
import betautils_config as bu_config
import betautils_detector as bu_detector
import betautils_tuning as bu_tuning

import ab_compare as ab
import ab_experiments as ex


def _box( start, end, label='exposed_breast', x=0 ):
    return { 'start': start, 'end': end, 't': start, 'x': x, 'y': 0, 'w': 100, 'h': 100,
             'label': label }


class _Logger( object ):

    def __init__( self ):
        self.lines = []

    def info( self, text, *args ):
        self.lines.append( text%args if args else text )

    debug = warning = error = info


# --- engine ----------------------------------------------------------------------

class TestNormalised( unittest.TestCase ):

    def test_old_suppression_runs_read_as_suppression( self ):
        ranking = ab.normalised( { 'current_rules': { 'a': 1 }, 'proposed_rules': { 'b': 2 } } )
        self.assertEqual( ranking['experiment'], 'suppression' )
        self.assertEqual( ranking['current'], { 'a': 1 } )
        self.assertEqual( ranking['proposed'], { 'b': 2 } )

    def test_new_runs_are_left_alone( self ):
        ranking = ab.normalised( { 'experiment': 'min_track_hits', 'current': {},
                                   'proposed': { 'x': { 'min_track_hits': 1 } } } )
        self.assertEqual( ranking['experiment'], 'min_track_hits' )
        self.assertEqual( ranking['proposed'], { 'x': { 'min_track_hits': 1 } } )


class TestFocus( unittest.TestCase ):
    """--focus ranks by one label's differences and picks windows from them."""

    def compare( self, focus ):
        current = [ _box( 0.0, 10.0, 'exposed_breast' ) ]
        proposed = [ _box( 50.0, 51.0, 'exposed_vulva' ) ]

        def replay( video, caches, backend, side, tracking, observer=None ):
            return proposed if side == 'new' else current

        context = { 'shot_cut_times': None, 'profile_name': 'scene', 'sample_fps': 9 }
        with mock.patch.object( bu_tuning, 'replay_tracking_context', return_value=context ), \
                mock.patch.object( bu_tuning, 'raw_boxes_for_reading', return_value=[] ):
            return ab.compare_video( 'h', '/v/a.mp4', [ 'a.gz' ], 'nudenet_v3', 'old', 'new',
                                     { 'exposed_breast', 'exposed_vulva' }, 4.0, 3, replay=replay,
                                     duration_of=lambda path: 100.0, focus=focus )[0]

    def test_without_focus_every_label_counts( self ):
        row = self.compare( None )
        self.assertEqual( row['differing_seconds'], 11.0 )

    def test_with_focus_only_that_label_ranks_and_picks_windows( self ):
        row = self.compare( [ 'exposed_vulva' ] )
        self.assertEqual( row['differing_seconds'], 1.0 )
        self.assertEqual( row['current_only_seconds'], 0.0 )
        self.assertEqual( len( row['windows'] ), 1 )
        start, end = row['windows'][0]
        self.assertTrue( start <= 50.5 <= end, row['windows'] )
        # The per-label split still shows everything, so nothing is hidden.
        self.assertEqual( row['current_only_by_label'], { 'exposed_breast': 10.0 } )

    def test_the_row_carries_the_videos_signature( self ):
        raw = [ { 'class_id': 'exposed_breast', 'score': 0.9, 't': step / 9.0, 'x': 0, 'y': 0,
                  'w': 10, 'h': 10 } for step in range( 90 ) ]
        context = { 'shot_cut_times': None, 'profile_name': 'scene', 'sample_fps': 9 }
        with mock.patch.object( bu_tuning, 'replay_tracking_context', return_value=context ), \
                mock.patch.object( bu_tuning, 'raw_boxes_for_reading', return_value=raw ), \
                mock.patch.object( bu_tuning, 'cache_sample_rate', return_value=9 ):
            row = ab.compare_video( 'h', '/v/a.mp4', [ 'a.gz' ], 'nudenet_v3', 'old', 'new',
                                    { 'exposed_breast', 'exposed_vulva' }, 4.0, 3,
                                    replay=lambda *args, **kwargs: [],
                                    duration_of=lambda path: 100.0 )[0]
        self.assertEqual( row['signature'], { 'duration': 100.0, 'mixed': 0.0, 'seconds': {
            'exposed_breast': 10.0, 'exposed_vulva': 0.0 } } )

    def test_focused_keeps_all_without_a_focus( self ):
        boxes = [ _box( 0, 1, 'a' ), _box( 0, 1, 'b' ) ]
        self.assertEqual( ab.focused( boxes, [] ), boxes )
        self.assertEqual( ab.focused( boxes, [ 'b' ] ), [ boxes[1] ] )


class TestExcluded( unittest.TestCase ):

    def test_chosen_and_earlier_exclusions_both_count_and_a_folder_works( self ):
        directory = tempfile.mkdtemp( prefix='betasuite-ab-' )
        try:
            with open( os.path.join( directory, 'ranking.json' ), 'w' ) as handle:
                json.dump( { 'chosen': [ 'h1', 'h2' ], 'excluded': [ 'h0' ] }, handle )
            self.assertEqual( ab.excluded_hashes( [ directory ] ), { 'h0', 'h1', 'h2' } )
            self.assertEqual( ab.excluded_hashes( [ os.path.join( directory, 'ranking.json' ) ] ),
                              { 'h0', 'h1', 'h2' } )
            self.assertEqual( ab.excluded_hashes( None ), set() )
        finally:
            shutil.rmtree( directory )


# --- per-label overrides --------------------------------------------------------------

class _ConfigSnapshot( unittest.TestCase ):

    def setUp( self ):
        self._backend = copy.deepcopy( betaconfig.detector_backend )
        self._shared = copy.deepcopy( getattr( betaconfig, 'item_overrides', {} ) )
        self.backend_name = bu_detector.selected_backend_name()

    def tearDown( self ):
        betaconfig.detector_backend = self._backend
        betaconfig.item_overrides = self._shared
        bu_config.invalidate_config_caches()

    def set_override( self, label, key, value, profile=None ):
        block = betaconfig.detector_backend[self.backend_name]
        if profile:
            target = block.setdefault( 'profiles', {} ).setdefault( 'variants', {} ).setdefault(
                profile, {} ).setdefault( 'item_overrides', {} )
        else:
            target = block.setdefault( 'item_overrides', {} )
        target.setdefault( label, {} )[key] = value
        bu_config.invalidate_config_caches()


class TestOverrideExperiment( _ConfigSnapshot ):

    def experiment( self ):
        return ex.experiment_named( 'min_track_hits' )

    def test_changes_name_old_new_and_the_profiles_replaced( self ):
        self.set_override( 'exposed_breast', 'min_track_hits', 2 )
        self.set_override( 'exposed_breast', 'min_track_hits', 3, profile='quick_cut' )
        experiment = self.experiment()
        proposed = { 'exposed_breast': { 'min_track_hits': 1 } }
        lines = experiment.change_lines( {}, proposed, self.backend_name )
        self.assertEqual( len( lines ), 1 )
        self.assertIn( 'min_track_hits 2 -> 1', lines[0] )
        self.assertIn( "quick_cut's 3", lines[0] )
        self.assertEqual( experiment.changes_by_cause( {}, proposed, self.backend_name ),
                          { 'exposed_breast': 'min_track_hits 2 -> 1' } )

    def test_a_value_the_config_already_has_is_no_change( self ):
        self.set_override( 'exposed_breast', 'min_track_hits', 2 )
        for variant in ( betaconfig.detector_backend[self.backend_name].get( 'profiles', {} )
                         .get( 'variants', {} ).values() ):
            ( variant.get( 'item_overrides' ) or {} ).get( 'exposed_breast', {} ).pop( 'min_track_hits', None )
        bu_config.invalidate_config_caches()
        experiment = self.experiment()
        self.assertTrue( experiment.is_noop( {}, { 'exposed_breast': { 'min_track_hits': 2 } },
                                             self.backend_name ) )

    def test_a_profile_that_differs_still_makes_it_a_change( self ):
        self.set_override( 'exposed_breast', 'min_track_hits', 1 )
        self.set_override( 'exposed_breast', 'min_track_hits', 2, profile='quick_cut' )
        self.assertFalse( self.experiment().is_noop(
            {}, { 'exposed_breast': { 'min_track_hits': 1 } }, self.backend_name ) )

    def test_the_pipeline_reads_the_side_in_the_profile_too( self ):
        self.set_override( 'exposed_breast', 'min_track_hits', 2 )
        self.set_override( 'exposed_breast', 'min_track_hits', 3, profile='quick_cut' )
        side = { 'exposed_breast': { 'min_track_hits': 1 } }
        with self.experiment().applied( side, self.backend_name, { 'profile_name': 'quick_cut' } ):
            self.assertEqual( bu_detector.get_item_overrides(
                'exposed_breast', self.backend_name )['min_track_hits'], 1 )
            self.assertEqual( bu_detector.get_profile_item_overrides(
                'quick_cut', 'exposed_breast', self.backend_name )['min_track_hits'], 1 )
        self.assertEqual( bu_detector.get_item_overrides(
            'exposed_breast', self.backend_name )['min_track_hits'], 2 )

    def test_the_config_side_changes_nothing( self ):
        before = copy.deepcopy( betaconfig.detector_backend )
        with self.experiment().applied( {}, self.backend_name, { 'profile_name': 'scene' } ):
            self.assertEqual( betaconfig.detector_backend, before )

    def test_a_difference_is_caused_by_its_own_labels_change( self ):
        boxes = [ _box( 0, 1, 'exposed_breast' ), _box( 0, 1, 'exposed_vulva' ) ]
        causes = self.experiment().attribute( boxes, None, { 'exposed_breast': 'x' } )
        self.assertEqual( causes, [ 'exposed_breast', ex.TRACKING_CAUSE ] )

    def test_each_label_takes_the_side_its_verdicts_favour_and_a_tie_keeps_the_config( self ):
        self.set_override( 'exposed_breast', 'min_track_hits', 2 )
        self.set_override( 'exposed_vulva', 'min_track_hits', 2 )
        proposed = { 'exposed_breast': { 'min_track_hits': 1 }, 'exposed_vulva': { 'min_track_hits': 1 } }
        tally = { 'exposed_breast': { 'proposed': 2, 'current': 1 },
                  'exposed_vulva': { 'proposed': 1, 'current': 1 } }
        chosen, decisions = self.experiment().choose( {}, proposed, tally, self.backend_name )
        self.assertEqual( chosen, { 'exposed_breast': { 'min_track_hits': 1 } } )
        self.assertEqual( decisions, { 'exposed_breast': 'proposed', 'exposed_vulva': 'current' } )

    def test_chosen_lines_are_pastable_and_name_the_profiles( self ):
        self.set_override( 'exposed_breast', 'min_track_hits', 3, profile='quick_cut' )
        lines = self.experiment().chosen_lines( { 'exposed_breast': { 'min_track_hits': 1 } }, {},
                                                self.backend_name )
        self.assertEqual( lines[0], "detector_backend[%r]['item_overrides']['exposed_breast']"
                                    "['min_track_hits'] = 1"%( self.backend_name, ) )
        self.assertIn( 'quick_cut', lines[1] )

    def test_note_names_the_setting_under_the_label( self ):
        self.assertEqual( self.experiment().note( 'exposed_breast', { 'exposed_breast': {
            'min_track_hits': 1 } } ), 'min_track_hits' )

    def test_set_builds_the_side_and_rejects_a_malformed_one( self ):
        self.assertEqual( ex.parse_sets( [ 'exposed_breast.min_track_hits=1',
                                           'exposed_breast.match_distance_multiplier=0.75' ] ),
                          { 'exposed_breast': { 'min_track_hits': 1, 'match_distance_multiplier': 0.75 } } )
        for bad in ( 'exposed_breast=1', 'min_track_hits', '.x=1', 'a.=1' ):
            with self.assertRaises( SystemExit ):
                ex.parse_sets( [ bad ] )

    def test_values_parse_to_python( self ):
        self.assertEqual( [ ex.parse_value( text ) for text in ( '1', '0.75', 'none', 'True', 'first' ) ],
                          [ 1, 0.75, None, True, 'first' ] )


class TestProposals( unittest.TestCase ):

    def setUp( self ):
        self.directory = tempfile.mkdtemp( prefix='betasuite-ab-' )
        self._censor = list( betaconfig.items_to_censor )

    def tearDown( self ):
        betaconfig.items_to_censor = self._censor
        shutil.rmtree( self.directory )

    def decisions( self, entries ):
        path = os.path.join( self.directory, 'decisions.json' )
        with open( path, 'w' ) as handle:
            json.dump( entries, handle )
        return path

    def args( self, **values ):
        return argparse.Namespace( set=values.get( 'set' ), auto_tune=values.get( 'auto_tune' ) )

    def test_from_auto_tune_review_values_censored_labels_only( self ):
        betaconfig.items_to_censor = [ 'exposed_breast' ]
        path = self.decisions( [
            { 'stage': 'min_track_hits / nudenet_v3 / exposed_breast', 'review': { 'value': 1 } },
            { 'stage': 'min_track_hits / nudenet_v3 / face_femme', 'review': { 'value': 1 } } ] )
        side, source = ex.experiment_named( 'min_track_hits' ).propose(
            self.args( auto_tune=path ), _Logger(), None, None )
        self.assertEqual( side, { 'exposed_breast': { 'min_track_hits': 1 } } )
        self.assertIn( 'auto_tune', source )

    def test_set_wins_over_auto_tune( self ):
        side, source = ex.experiment_named( 'min_track_hits' ).propose(
            self.args( set=[ 'exposed_vulva.min_track_hits=3' ] ), _Logger(), None, None )
        self.assertEqual( side, { 'exposed_vulva': { 'min_track_hits': 3 } } )
        self.assertEqual( source, '--set' )

    def test_nothing_to_review_says_how_to_name_values( self ):
        path = self.decisions( [] )
        with self.assertRaises( SystemExit ) as raised:
            ex.experiment_named( 'match_distance' ).propose( self.args( auto_tune=path ), _Logger(),
                                                             None, None )
        self.assertIn( 'AB_SET', str( raised.exception ) )

    def test_tracking_needs_set( self ):
        with self.assertRaises( SystemExit ):
            ex.experiment_named( 'tracking' ).propose( self.args(), _Logger(), None, None )


class TestReviewValues( unittest.TestCase ):
    """auto_tune's numbers from 2026-10-11, as decisions.json held them."""

    MATCH_BREAST = { 'stage': 'match_distance / nudenet_v3 / exposed_breast', 'chosen': 2.0,
                     'accepted': False, 'rows': [
        { 'value': 2.0, 'verdict': 'baseline', 'coverage_loss': 0.0 },
        { 'value': 0.75, 'coverage_loss': 0.0, 'resets_avoided': -6631, 'risky_added': -9509 },
        { 'value': 1.25, 'coverage_loss': 0.0, 'resets_avoided': -2720, 'risky_added': -6362 },
        { 'value': 1.5, 'coverage_loss': 0.00027, 'resets_avoided': -1573, 'risky_added': -4373 },
        { 'value': 3.0, 'coverage_loss': 0.0, 'resets_avoided': 1475, 'risky_added': 11325 } ] }
    MATCH_VULVA = { 'stage': 'match_distance / nudenet_v3 / exposed_vulva', 'chosen': 3,
                    'accepted': False, 'rows': [
        { 'value': 3, 'verdict': 'baseline', 'coverage_loss': 0.0 },
        { 'value': 0.75, 'coverage_loss': 0.0, 'resets_avoided': -4027, 'risky_added': -88 },
        { 'value': 2.0, 'coverage_loss': 0.0028, 'resets_avoided': -552, 'risky_added': -50 } ] }
    HITS = { 'stage': 'min_track_hits / nudenet_v3 / exposed_breast', 'chosen': 1, 'accepted': True,
             'rows': [ { 'value': 2, 'verdict': 'baseline' }, { 'value': 1, 'verdict': '+12742' } ] }

    def test_older_files_get_the_tightening_candidate_from_their_rows( self ):
        found = ex.review_values( [ self.MATCH_BREAST, self.MATCH_VULVA ], 'match_distance_multiplier' )
        self.assertEqual( found, { 'exposed_breast': 0.75 } )

    def test_older_files_get_an_accepted_change( self ):
        self.assertEqual( ex.review_values( [ self.HITS ], 'min_track_hits' ), { 'exposed_breast': 1 } )

    def test_a_recorded_review_wins_and_a_recorded_none_means_none( self ):
        reviewed = dict( self.MATCH_BREAST, key='match_distance_multiplier', label='exposed_breast',
                         review={ 'value': 1.25 } )
        self.assertEqual( ex.review_values( [ reviewed ], 'match_distance_multiplier' ),
                          { 'exposed_breast': 1.25 } )
        declined = dict( self.HITS, key='min_track_hits', label='exposed_breast', review=None )
        self.assertEqual( ex.review_values( [ declined ], 'min_track_hits' ), {} )

    def test_other_keys_are_ignored( self ):
        self.assertEqual( ex.review_values( [ self.HITS ], 'match_distance_multiplier' ), {} )

    def test_the_newest_run_is_found( self ):
        root = tempfile.mkdtemp( prefix='betasuite-runs-' )
        try:
            for name in ( '20261009_1', '20261011_1' ):
                os.makedirs( os.path.join( root, name, 'auto_tune' ) )
                open( os.path.join( root, name, 'auto_tune', 'decisions.json' ), 'w' ).close()
            self.assertIn( '20261011_1', ex.latest_auto_tune_decisions( root ) )
            self.assertIsNone( ex.latest_auto_tune_decisions( os.path.join( root, 'none' ) ) )
        finally:
            shutil.rmtree( root )


class TestTighteningReview( unittest.TestCase ):

    def rows( self, *candidates ):
        return [ { 'value': 2.0, 'verdict': 'baseline', 'coverage_loss': 0.0 } ] + [
            { 'value': value, 'coverage_loss': loss, 'resets_avoided': -resets, 'risky_added': -risky }
            for value, loss, resets, risky in candidates ]

    def test_the_most_ambiguity_removed_wins( self ):
        found = bu_tuning.tightening_review(
            self.rows( ( 0.75, 0.0, 6631, 9509 ), ( 1.25, 0.0, 2720, 6362 ) ), 2.0 )
        self.assertEqual( found['value'], 0.75 )
        self.assertEqual( ( found['risky_removed'], found['resets_added'] ), ( 9509, 6631 ) )
        self.assertIn( 'by eye', found['why'] )

    def test_coverage_loss_disqualifies( self ):
        self.assertIsNone( bu_tuning.tightening_review( self.rows( ( 1.5, 0.00027, 1573, 4373 ) ), 2.0 ) )

    def test_fewer_ambiguous_than_resets_added_disqualifies_and_equal_qualifies( self ):
        self.assertIsNone( bu_tuning.tightening_review( self.rows( ( 0.75, 0.0, 4027, 88 ) ), 2.0 ) )
        self.assertIsNotNone( bu_tuning.tightening_review( self.rows( ( 0.75, 0.0, 100, 100 ) ), 2.0 ) )

    def test_only_tighter_values_and_only_real_reductions( self ):
        self.assertIsNone( bu_tuning.tightening_review( self.rows( ( 3.0, 0.0, -100, 500 ) ), 2.0 ) )
        self.assertIsNone( bu_tuning.tightening_review( self.rows( ( 1.0, 0.0, 0, 0 ) ), 2.0 ) )
        self.assertIsNone( bu_tuning.tightening_review( self.rows( ( 2.0, 0.0, 10, 50 ) ), 2.0 ) )


# --- the cut-burst experiment --------------------------------------------------------------

class TestCutBurstExperiment( unittest.TestCase ):

    def setUp( self ):
        self.saved = { key: getattr( betaconfig, key ) for key in ex.CUT_BURST_KEYS
                       if hasattr( betaconfig, key ) }
        for key in ex.CUT_BURST_KEYS:
            if hasattr( betaconfig, key ):
                delattr( betaconfig, key )
        self.experiment = ex.experiment_named( 'cut_bursts' )

    def tearDown( self ):
        for key in ex.CUT_BURST_KEYS:
            if hasattr( betaconfig, key ):
                delattr( betaconfig, key )
        for key, value in self.saved.items():
            setattr( betaconfig, key, value )
        bu_config.invalidate_config_caches()

    def test_changes_are_against_the_config_absent_meaning_off( self ):
        proposed = dict( ex.CUT_BURST_DEFAULT )
        lines = self.experiment.change_lines( {}, proposed, 'nudenet_v3' )
        self.assertEqual( lines, [ "shot_cut_burst_window: 0 -> 0.35" ] )
        self.assertEqual( list( self.experiment.changes_by_cause( {}, proposed, 'nudenet_v3' ) ),
                          [ 'cut_bursts' ] )
        betaconfig.shot_cut_burst_window = 0.35
        self.assertTrue( self.experiment.is_noop( {}, proposed, 'nudenet_v3' ) )

    def test_the_side_is_read_inside_and_absent_keys_come_back_absent( self ):
        with self.experiment.applied( dict( ex.CUT_BURST_DEFAULT ), 'nudenet_v3', {} ):
            self.assertEqual( bu_tuning.bu_track.cut_burst_settings()['window'], 0.35 )
        self.assertFalse( hasattr( betaconfig, 'shot_cut_burst_window' ) )
        self.assertIsNone( bu_tuning.bu_track.cut_burst_settings() )

    def test_every_difference_has_the_one_cause( self ):
        boxes = [ _box( 0, 1 ), _box( 1, 2, 'exposed_vulva' ) ]
        self.assertEqual( self.experiment.attribute( boxes, None, { 'cut_bursts': 'x' } ),
                          [ 'cut_bursts', 'cut_bursts' ] )

    def test_the_verdicts_choose_all_or_nothing( self ):
        proposed = dict( ex.CUT_BURST_DEFAULT )
        chosen, decisions = self.experiment.choose( {}, proposed, { 'cut_bursts': { 'proposed': 3 } },
                                                    'nudenet_v3' )
        self.assertEqual( chosen, proposed )
        self.assertEqual( self.experiment.chosen_lines( chosen, {}, 'nudenet_v3' ),
                          [ "shot_cut_burst_window = 0.35", "shot_cut_burst_max_span = None",
                            "shot_cut_burst_keep = 'first'" ] )
        chosen, decisions = self.experiment.choose( {}, proposed, { 'cut_bursts': { 'proposed': 1,
                                                                                    'current': 1 } },
                                                    'nudenet_v3' )
        self.assertEqual( ( chosen, decisions ), ( {}, { 'cut_bursts': 'current' } ) )

    def test_set_is_checked_against_the_keys( self ):
        args = argparse.Namespace( set=[ 'shot_cut_burst_keep=last' ] )
        side, source = self.experiment.propose( args, _Logger(), None, None )
        self.assertEqual( side['shot_cut_burst_keep'], 'last' )
        self.assertEqual( source, '--set' )
        with self.assertRaises( SystemExit ):
            self.experiment.propose( argparse.Namespace( set=[ 'min_track_hits=1' ] ), _Logger(), None, None )

    def test_the_analysers_best_variant_is_the_proposal( self ):
        root = tempfile.mkdtemp( prefix='betasuite-runs-' )
        try:
            os.makedirs( os.path.join( root, '20261011_1' ) )
            with open( os.path.join( root, '20261011_1', 'analyze_cut_bursts.json' ), 'w' ) as handle:
                json.dump( { 'best_settings': { 'window': 0.25, 'max_span': 0.6, 'keep': 'last' } }, handle )
            found, path = ex.latest_cut_burst_best( root )
            self.assertEqual( found, { 'shot_cut_burst_window': 0.25, 'shot_cut_burst_max_span': 0.6,
                                       'shot_cut_burst_keep': 'last' } )
            self.assertIsNone( ex.latest_cut_burst_best( os.path.join( root, 'none' ) ) )
        finally:
            shutil.rmtree( root )


# --- decide, end to end per experiment --------------------------------------------------------

class TestDecideWritesChosen( _ConfigSnapshot ):

    def decide( self, ranking, verdicts ):
        directory = tempfile.mkdtemp( prefix='betasuite-ab-' )
        self.addCleanup( shutil.rmtree, directory )
        with open( os.path.join( directory, 'feedback.csv' ), 'w', newline='' ) as handle:
            writer = csv.DictWriter( handle, fieldnames=ab.FEEDBACK_COLUMNS )
            writer.writeheader()
            for number, ( cause, verdict ) in enumerate( verdicts ):
                writer.writerow( { 'id': 'v1c1.%d'%( number, ), 'cause': cause, 'verdict': verdict } )
        logger = _Logger()
        args = argparse.Namespace( out_dir=directory, feedback=None )
        chosen, decisions = ab.phase_decide( args, logger, ranking )
        return directory, chosen, decisions, logger

    def test_override_experiment( self ):
        self.set_override( 'exposed_breast', 'min_track_hits', 2 )
        ranking = { 'experiment': 'min_track_hits', 'backend': self.backend_name, 'current': {},
                    'proposed': { 'exposed_breast': { 'min_track_hits': 1 } } }
        directory, chosen, decisions, logger = self.decide(
            ranking, [ ( 'exposed_breast', 'right' ), ( 'exposed_breast', 'r' ) ] )
        with open( os.path.join( directory, 'chosen.json' ) ) as handle:
            self.assertEqual( json.load( handle ), { 'exposed_breast': { 'min_track_hits': 1 } } )
        self.assertFalse( os.path.exists( os.path.join( directory, 'chosen_rules.json' ) ) )
        self.assertTrue( any( "['min_track_hits'] = 1" in line for line in logger.lines ) )
        self.assertTrue( any( 'ab_compare.sh confirm' in line for line in logger.lines ) )

    def test_nothing_won_means_nothing_to_paste( self ):
        self.set_override( 'exposed_breast', 'min_track_hits', 2 )
        ranking = { 'experiment': 'min_track_hits', 'backend': self.backend_name, 'current': {},
                    'proposed': { 'exposed_breast': { 'min_track_hits': 1 } } }
        directory, chosen, decisions, logger = self.decide( ranking, [ ( 'exposed_breast', 'left' ) ] )
        self.assertEqual( chosen, {} )
        self.assertTrue( any( 'nothing to paste' in line for line in logger.lines ) )
        self.assertFalse( any( 'confirm' in line for line in logger.lines ) )

    def test_suppression_writes_both_names( self ):
        current = { 'exposed_breast': [ { 'suppressed_by': 'face_femme', 'margin': 0.1, 'min_iou': 0.01 } ] }
        ranking = { 'backend': self.backend_name, 'current_rules': current, 'proposed_rules': {} }
        directory, chosen, decisions, logger = self.decide(
            ranking, [ ( 'exposed_breast<-face_femme', 'right' ) ] )
        for name in ( 'chosen.json', 'chosen_rules.json' ):
            with open( os.path.join( directory, name ) ) as handle:
                self.assertEqual( json.load( handle ), {} )
        self.assertEqual( decisions, { 'exposed_breast<-face_femme': 'proposed' } )

    def test_a_turned_down_addition_is_offered_for_the_refused_list( self ):
        current = { 'exposed_breast': [ { 'suppressed_by': 'face_femme', 'margin': 0.1, 'min_iou': 0.01 } ] }
        proposed = { 'exposed_breast': [ { 'suppressed_by': 'face_femme', 'margin': 0.1, 'min_iou': 0.01 },
                                         { 'suppressed_by': 'exposed_chest', 'margin': 0.087,
                                           'min_iou': 0.525 } ] }
        ranking = { 'experiment': 'suppression', 'backend': self.backend_name,
                    'current': current, 'proposed': proposed }
        with mock.patch.object( ab.ab_experiments.bu_detector, 'get_class_suppression_refused',
                                return_value=[] ):
            _directory, chosen, decisions, logger = self.decide(
                ranking, [ ( 'exposed_breast<-exposed_chest', 'left' ) ] )
        self.assertEqual( decisions, { 'exposed_breast<-exposed_chest': 'current' } )
        self.assertTrue( any( 'nothing to paste' in line for line in logger.lines ) )
        self.assertTrue( any( "['class_suppression_refused'] = ['exposed_breast<-exposed_chest']" in line
                              for line in logger.lines ), logger.lines )


class TestLaterPhasesNameTheRoundsExperiment( unittest.TestCase ):

    def test_decide_logs_the_experiment_from_the_ranking( self ):
        directory = tempfile.mkdtemp( prefix='betasuite-ab-' )
        self.addCleanup( shutil.rmtree, directory )
        with open( os.path.join( directory, 'ranking.json' ), 'w' ) as handle:
            json.dump( { 'experiment': 'confirm_score', 'backend': 'nudenet_v3', 'current': {},
                         'proposed': { 'exposed_breast': { 'confirm_score': 0.5 } } }, handle )
        with open( os.path.join( directory, 'feedback.csv' ), 'w', newline='' ) as handle:
            csv.DictWriter( handle, fieldnames=ab.FEEDBACK_COLUMNS ).writeheader()
        ab.main( [ '--phase', 'decide', '--out-dir', directory, '--console-level', 'error' ] )
        with open( os.path.join( directory, 'betabench.log' ) ) as handle:
            text = handle.read()
        self.assertIn( 'A/B (confirm_score): output in', text )
        self.assertNotIn( 'A/B (suppression)', text )


class TestChooseVideosSkipsCopies( unittest.TestCase ):
    """A round never spends two of its videos on the same footage."""

    SAME = { 'duration': 1080.0, 'seconds': { 'exposed_breast': 635.7, 'exposed_vulva': 128.7 },
             'mixed': 75.7 }
    OTHER = { 'duration': 2000.0, 'seconds': { 'exposed_breast': 300.0, 'exposed_vulva': 300.0 },
              'mixed': 100.0 }

    def row( self, name, differing, signature ):
        return { 'name': name, 'file_hash': name.lower(), 'differing_seconds': differing,
                 'signature': signature }

    def test_a_copy_of_a_chosen_video_is_skipped( self ):
        logger = _Logger()
        rows = [ self.row( 'A', 90, self.SAME ), self.row( 'A2', 89, self.SAME ),
                 self.row( 'B', 50, self.OTHER ) ]
        chosen = ab.choose_videos( rows, 2, logger=logger )
        self.assertEqual( [ row['name'] for row in chosen ], [ 'A', 'B' ] )
        self.assertIn( 'skipping A2: the same footage as A', logger.lines )

    def test_a_copy_of_an_earlier_rounds_video_is_skipped( self ):
        rows = [ self.row( 'A2', 89, self.SAME ), self.row( 'B', 50, self.OTHER ) ]
        chosen = ab.choose_videos( rows, 2, earlier={ 'A': self.SAME } )
        self.assertEqual( [ row['name'] for row in chosen ], [ 'B' ] )

    def test_videos_that_do_not_differ_are_never_chosen( self ):
        rows = [ self.row( 'A', 90, self.SAME ), self.row( 'B', 0, self.OTHER ) ]
        self.assertEqual( [ row['name'] for row in ab.choose_videos( rows, 2 ) ], [ 'A' ] )

    def test_rows_without_a_signature_are_never_copies( self ):
        rows = [ self.row( 'A', 90, None ), self.row( 'A2', 89, None ) ]
        self.assertEqual( len( ab.choose_videos( rows, 2 ) ), 2 )

    def test_count_is_respected( self ):
        rows = [ self.row( 'A', 90, self.SAME ), self.row( 'B', 50, self.OTHER ) ]
        self.assertEqual( [ row['name'] for row in ab.choose_videos( rows, 1 ) ], [ 'A' ] )

    def test_earlier_signatures_reads_only_the_videos_a_round_showed( self ):
        directory = tempfile.mkdtemp( prefix='betasuite-ab-' )
        self.addCleanup( shutil.rmtree, directory )
        with open( os.path.join( directory, 'ranking.json' ), 'w' ) as handle:
            json.dump( { 'chosen': [ 'a' ], 'videos': [
                self.row( 'A', 90, self.SAME ), self.row( 'B', 50, self.OTHER ),
                { 'name': 'Old', 'file_hash': 'a', 'differing_seconds': 1 } ] }, handle )
        self.assertEqual( ab.earlier_signatures( [ directory ] ), { 'A': self.SAME } )
        self.assertEqual( ab.earlier_signatures( None ), {} )


class TestConfirmScoreExperiment( unittest.TestCase ):
    """auto_tune's review values when it has some, else the confident score for every label."""

    def propose( self, decisions ):
        experiment = ab.ab_experiments.experiment_named( 'confirm_score' )
        directory = tempfile.mkdtemp( prefix='betasuite-ab-' )
        self.addCleanup( shutil.rmtree, directory )
        path = os.path.join( directory, 'decisions.json' )
        with open( path, 'w' ) as handle:
            json.dump( decisions, handle )
        args = argparse.Namespace( set=None, auto_tune=path )
        return experiment.propose( args, _Logger(), None, None )

    def test_without_a_review_every_censored_label_gets_the_confident_score( self ):
        proposed, source = self.propose( [] )
        self.assertEqual( proposed, { label: { 'confirm_score': 0.5 }
                                      for label in sorted( ab.betaconfig.items_to_censor ) } )
        self.assertIn( 'default confirm_score of 0.5', source )

    def test_a_review_value_wins( self ):
        decisions = [ { 'stage': 'confirm_score / nudenet_v3 / exposed_breast', 'key': 'confirm_score',
                        'label': 'exposed_breast', 'review': { 'value': 0.6 } } ]
        proposed, source = self.propose( decisions )
        self.assertEqual( proposed, { 'exposed_breast': { 'confirm_score': 0.6 } } )
        self.assertIn( "auto_tune's review values", source )

    def test_it_is_on_the_menu( self ):
        names = [ experiment.name for experiment in ab.ab_experiments.experiments() ]
        self.assertIn( 'confirm_score', names )

    def test_min_track_hits_still_needs_a_review( self ):
        experiment = ab.ab_experiments.experiment_named( 'min_track_hits' )
        args = argparse.Namespace( set=None, auto_tune='/nonexistent/decisions.json' )
        with self.assertRaises( SystemExit ):
            experiment.propose( args, _Logger(), None, None )


# --- auto_tune flags what only eyes can settle ------------------------------------------------

def _load_auto_tune():
    path = os.path.join( REPO_ROOT, 'tools', 'tuning', 'auto_tune.py' )
    spec = importlib.util.spec_from_file_location( '_auto_tune_for_ab', path )
    module = importlib.util.module_from_spec( spec )
    spec.loader.exec_module( module )
    return module


class TestAutoTuneReview( unittest.TestCase ):

    def setUp( self ):
        self.tool = _load_auto_tune()

    def test_a_rejected_loosening_stage_names_its_tightening_candidate( self ):
        stage = self.tool.TrackMaxGapStage( 'nudenet_v3', 'exposed_breast' )
        rows = TestTighteningReview().rows( ( 0.75, 0.0, 6631, 9509 ) )
        decision = bu_tuning.Decision( 'x', 2.0, False, 'kept', rows )
        self.assertEqual( stage.review( decision, { 'cfg': rows } )['value'], 0.75 )
        accepted = bu_tuning.Decision( 'x', 3.0, True, 'loosened', rows )
        self.assertIsNone( stage.review( accepted, { 'cfg': rows } ) )

    def test_configurations_must_agree( self ):
        stage = self.tool.TrackMaxGapStage( 'nudenet_v3', 'exposed_breast' )
        one = TestTighteningReview().rows( ( 0.75, 0.0, 10, 50 ) )
        other = TestTighteningReview().rows( ( 1.25, 0.0, 10, 50 ) )
        decision = bu_tuning.Decision( 'x', 2.0, False, 'kept', one )
        self.assertIsNone( stage.review( decision, { 'a': one, 'b': other } ) )
        self.assertIsNone( stage.review( decision, { 'a': one, 'b': TestTighteningReview().rows() } ) )

    def test_min_track_hits_puts_every_change_in_front_of_a_person( self ):
        stage = self.tool.MinTrackHitsStage( 'nudenet_v3', 'exposed_breast' )
        changed = bu_tuning.Decision( 'x', 1, True, 'fewer uncensored', [] )
        self.assertEqual( stage.review( changed, {} )['value'], 1 )
        self.assertIn( 'misfire', stage.review( changed, {} )['why'] )
        kept = bu_tuning.Decision( 'x', 2, False, 'kept', [] )
        self.assertIsNone( stage.review( kept, {} ) )

    def test_decisions_json_carries_review_key_and_label( self ):
        decision = bu_tuning.Decision( 's', 1, True, 'r', [], review={ 'value': 1 }, key='min_track_hits',
                                       label='exposed_breast' )
        record = decision.as_dict()
        self.assertEqual( ( record['review'], record['key'], record['label'] ),
                          ( { 'value': 1 }, 'min_track_hits', 'exposed_breast' ) )


class TestConfirming( unittest.TestCase ):
    """A confirm round: the previous round's choice, its focus, none of its videos."""

    def setUp( self ):
        self.previous = tempfile.mkdtemp( prefix='betasuite-ab-' )
        with open( os.path.join( self.previous, 'ranking.json' ), 'w' ) as handle:
            json.dump( { 'experiment': 'min_track_hits', 'focus': [ 'exposed_vulva' ],
                         'chosen': [ 'h1' ] }, handle )

    def tearDown( self ):
        shutil.rmtree( self.previous )

    def chosen( self, value ):
        with open( os.path.join( self.previous, 'chosen.json' ), 'w' ) as handle:
            json.dump( value, handle )

    def args( self, **values ):
        return argparse.Namespace( confirm_from=self.previous, experiment='suppression', proposed=None,
                                   exclude_from=values.get( 'exclude_from' ), focus=values.get( 'focus' ) )

    def test_takes_experiment_choice_focus_and_leaves_its_videos_out( self ):
        self.chosen( { 'exposed_breast': { 'min_track_hits': 1 } } )
        args = ab.confirming( self.args( exclude_from=[ '/earlier' ] ) )
        self.assertEqual( args.experiment, 'min_track_hits' )
        self.assertEqual( args.proposed, os.path.join( self.previous, 'chosen.json' ) )
        self.assertEqual( args.exclude_from, [ '/earlier', self.previous ] )
        self.assertEqual( args.focus, [ 'exposed_vulva' ] )

    def test_a_focus_given_now_wins( self ):
        self.chosen( { 'exposed_breast': { 'min_track_hits': 1 } } )
        self.assertEqual( ab.confirming( self.args( focus=[ 'exposed_breast' ] ) ).focus,
                          [ 'exposed_breast' ] )

    def test_no_choice_or_an_empty_one_stops_with_a_reason( self ):
        with self.assertRaises( SystemExit ) as missing:
            ab.confirming( self.args() )
        self.assertIn( 'decide', str( missing.exception ) )
        self.chosen( {} )
        with self.assertRaises( SystemExit ) as empty:
            ab.confirming( self.args() )
        self.assertIn( 'nothing to confirm', str( empty.exception ) )


class _RoundDir( unittest.TestCase ):

    ROW = { 'file_hash': 'h1', 'name': 'My Video.mp4', 'video': '/v/My Video.mp4',
            'cache_paths': [ 'c.gz' ], 'windows': [ [ 10.0, 25.0 ], [ 100.0, 115.0 ] ] }

    def setUp( self ):
        self.dir = tempfile.mkdtemp( prefix='betasuite-round-' )
        self.addCleanup( shutil.rmtree, self.dir )

    def write( self, name, content='' ):
        with open( os.path.join( self.dir, name ), 'w' ) as handle:
            handle.write( content if isinstance( content, str ) else json.dumps( content ) )

    def ranked( self, chosen=( 'h1', ) ):
        self.write( 'ranking.json', { 'experiment': 'cut_bursts', 'current': {}, 'proposed': {},
                                      'videos': [ self.ROW ], 'chosen': list( chosen ) } )


class TestRoundStatus( _RoundDir ):
    """Where a round stopped, read from its folder, so resume knows what is left."""

    def status( self ):
        return ab.round_status( self.dir )

    def test_nothing_ranked( self ):
        self.assertEqual( self.status()['ranked'], 0 )

    def test_each_step_flips_when_its_files_are_there( self ):
        self.ranked()
        status = self.status()
        self.assertEqual( ( status['ranked'], status['experiment'], status['videos'] ),
                          ( 1, 'cut_bursts', 1 ) )
        self.assertEqual( ( status['explained'], status['clips'], status['full'], status['decided'] ),
                          ( 0, 0, 0, 0 ) )
        self.write( 'clips.md' )
        self.assertEqual( self.status()['explained'], 0, "clips.md alone is half an explain" )
        self.write( 'feedback.csv', 'id,cause,verdict\nv1c1.1,a,\n' )
        self.assertEqual( self.status()['explained'], 1 )
        self.write( 'My_Video_clip01_10s_ab.mp4' )
        self.assertEqual( self.status()['clips'], 0, "one of two windows" )
        self.write( 'My_Video_clip02_100s_current.mp4' )
        self.assertEqual( self.status()['clips'], 0, "one half is not a side by side" )
        self.write( 'My_Video_clip02_100s_proposed.mp4' )
        self.assertEqual( self.status()['clips'], 1, "both halves stand in for a failed stack" )
        self.write( 'My_Video_full_current.mp4' )
        self.assertEqual( self.status()['full'], 0 )
        self.write( 'My_Video_full_proposed.mp4' )
        self.assertEqual( self.status()['full'], 1 )

    def test_verdicts_count_only_real_verdicts( self ):
        self.ranked()
        self.write( 'feedback.csv', 'id,cause,verdict\na,x,right\nb,x, L \nc,x,maybe\nd,x,\n' )
        status = self.status()
        self.assertEqual( ( status['rows'], status['verdicts'] ), ( 4, 2 ) )

    def test_decided_and_adopted( self ):
        self.ranked()
        self.write( 'chosen.json', {} )
        self.assertEqual( ( self.status()['decided'], self.status()['adopted'] ), ( 1, 0 ) )
        self.write( 'chosen.json', { 'shot_cut_burst_window': 0.35 } )
        self.assertEqual( self.status()['adopted'], 1 )

    def test_no_chosen_video_is_never_complete( self ):
        self.ranked( chosen=() )
        status = self.status()
        self.assertEqual( ( status['videos'], status['clips'], status['full'] ), ( 0, 0, 0 ) )


class TestSkipExisting( _RoundDir ):
    """A resumed render keeps every finished file and replays nothing it does not need."""

    def setUp( self ):
        super().setUp()
        self.rendered = []
        self.compared = []

    def render( self, video, file_hash, boxes, path, start, length, logger ):
        self.rendered.append( os.path.basename( path ) )
        open( path, 'w' ).close()

    def compare( self, *args ):
        self.compared.append( args )
        return { 'current_boxes': [], 'proposed_boxes': [], 'both': [], 'current_only': [],
                 'proposed_only': [], 'current_only_causes': [], 'proposed_only_causes': [] }

    def run_render( self, **kwargs ):
        ranking = { 'experiment': 'cut_bursts', 'backend': 'nudenet_v3', 'current': {}, 'proposed': {} }
        return ab.render_video_comparison( self.ROW, ranking, _Logger(), self.dir, render=self.render,
                                           stack=lambda left, right, out, logger: open( out, 'w' ).close() or True,
                                           compare=self.compare, skip_existing=True, **kwargs )

    def test_everything_there_means_no_replay_and_no_render( self ):
        for name in ( 'My_Video_clip01_10s_diff.mp4', 'My_Video_clip01_10s_ab.mp4',
                      'My_Video_clip02_100s_diff.mp4', 'My_Video_clip02_100s_ab.mp4',
                      'My_Video_full_current.mp4', 'My_Video_full_proposed.mp4' ):
            self.write( name )
        self.assertEqual( self.run_render( full=True ), [] )
        self.assertEqual( ( self.rendered, self.compared ), ( [], [] ) )

    def test_only_the_missing_files_are_rendered( self ):
        self.write( 'My_Video_clip01_10s_diff.mp4' )
        self.write( 'My_Video_clip01_10s_ab.mp4' )
        self.write( 'My_Video_full_current.mp4' )
        self.run_render( full=True )
        self.assertEqual( len( self.compared ), 1 )
        self.assertEqual( self.rendered, [ 'My_Video_clip02_100s_diff.mp4', 'My_Video_clip02_100s_current.mp4',
                                           'My_Video_clip02_100s_proposed.mp4', 'My_Video_full_proposed.mp4' ] )

    def test_clips_done_but_whole_videos_missing_still_renders_them( self ):
        for name in ( 'My_Video_clip01_10s_diff.mp4', 'My_Video_clip01_10s_ab.mp4',
                      'My_Video_clip02_100s_diff.mp4', 'My_Video_clip02_100s_ab.mp4' ):
            self.write( name )
        self.run_render( full=True )
        self.assertEqual( self.rendered, [ 'My_Video_full_current.mp4', 'My_Video_full_proposed.mp4' ] )

    def test_without_skip_everything_is_rendered_again( self ):
        self.write( 'My_Video_clip01_10s_ab.mp4' )
        ranking = { 'experiment': 'cut_bursts', 'backend': 'nudenet_v3', 'current': {}, 'proposed': {} }
        ab.render_video_comparison( self.ROW, ranking, _Logger(), self.dir, render=self.render,
                                    stack=lambda left, right, out, logger: True, compare=self.compare,
                                    overlay=False )
        self.assertIn( 'My_Video_clip01_10s_current.mp4', self.rendered )


class TestSideBySideIsWholeOrAbsent( unittest.TestCase ):

    def setUp( self ):
        self.dir = tempfile.mkdtemp( prefix='betasuite-stack-' )
        self.addCleanup( shutil.rmtree, self.dir )
        self.out = os.path.join( self.dir, 'x_ab.mp4' )

    def fake_ffmpeg( self, code ):
        def run( command, capture_output=True, text=True ):
            open( command[-1], 'w' ).close()       # ffmpeg writes wherever it is told
            return mock.Mock( returncode=code, stderr='boom' if code else '' )
        return run

    def test_success_moves_the_finished_file_into_place( self ):
        with mock.patch.object( ab.subprocess, 'run', self.fake_ffmpeg( 0 ) ) as run:
            self.assertTrue( ab.side_by_side( 'l.mp4', 'r.mp4', self.out, _Logger() ) )
        self.assertEqual( os.listdir( self.dir ), [ 'x_ab.mp4' ] )

    def test_the_encoder_arguments_are_the_renders_own( self ):
        seen = {}

        def run( command, capture_output=True, text=True ):
            seen['command'] = command
            open( command[-1], 'w' ).close()
            return mock.Mock( returncode=0, stderr='' )

        with mock.patch.object( betaconfig, 'encode_video_codec', 'h264_nvenc', create=True ), \
                mock.patch.object( betaconfig, 'encode_crf', 21, create=True ), \
                mock.patch.object( betaconfig, 'encode_preset', 'p5', create=True ), \
                mock.patch.object( ab.subprocess, 'run', run ):
            ab.side_by_side( 'l.mp4', 'r.mp4', self.out, _Logger() )
        command = seen['command']
        self.assertNotIn( '-crf', command, "NVENC ignores -crf and falls back to its default bitrate" )
        self.assertEqual( command[ command.index( '-cq' ) + 1 ], '21' )
        self.assertEqual( command[ command.index( '-preset' ) + 1 ], 'p5' )

    def test_failure_leaves_nothing_that_looks_finished( self ):
        with mock.patch.object( ab.subprocess, 'run', self.fake_ffmpeg( 1 ) ):
            self.assertFalse( ab.side_by_side( 'l.mp4', 'r.mp4', self.out, _Logger() ) )
        self.assertEqual( os.listdir( self.dir ), [] )


def _placed( start, end, x, label='exposed_breast', w=100, y=0 ):
    return { 'start': start, 'end': end, 't': start, 'x': x, 'y': y, 'w': w, 'h': 100, 'label': label }


class TestWindowMoments( unittest.TestCase ):
    """One feedback row per moment, and a moment both sides censor differently says so."""

    def result( self, current_boxes, proposed_boxes ):
        both, current_only, proposed_only = ab.diff_boxes( current_boxes, proposed_boxes )
        return { 'current_boxes': current_boxes, 'proposed_boxes': proposed_boxes, 'both': both,
                 'current_only': current_only, 'proposed_only': proposed_only,
                 'current_only_causes': [ 'c' ] * len( current_only ),
                 'proposed_only_causes': [ 'c' ] * len( proposed_only ) }

    def test_one_side_bare( self ):
        moments = ab.window_moments( self.result( [], [ _placed( 12.0, 13.0, 0 ) ] ), 10.0, 25.0 )
        self.assertEqual( moments, [ { 'side': ab.SIDE_RIGHT, 'label': 'exposed_breast', 'cause': 'c',
                                       'span': ( 2.0, 3.0 ) } ] )
        moments = ab.window_moments( self.result( [ _placed( 12.0, 13.0, 0 ) ], [] ), 10.0, 25.0 )
        self.assertEqual( moments[0]['side'], ab.SIDE_LEFT )

    def test_both_censor_in_different_places_is_one_moment( self ):
        # Touching but under MATCH_IOU: each counts as "only" in the diff, both censor on screen.
        moments = ab.window_moments( self.result( [ _placed( 12.0, 13.0, 0 ) ],
                                                  [ _placed( 12.0, 13.0, 80 ) ] ), 10.0, 25.0 )
        self.assertEqual( [ moment['side'] for moment in moments ], [ ab.SIDE_BOTH ] )
        self.assertEqual( moments[0]['span'], ( 2.0, 3.0 ) )

    def test_a_box_elsewhere_or_of_another_label_is_not_the_same_spot( self ):
        apart = ab.window_moments( self.result( [ _placed( 12.0, 13.0, 0 ) ],
                                                [ _placed( 12.0, 13.0, 500 ) ] ), 10.0, 25.0 )
        self.assertEqual( sorted( moment['side'] for moment in apart ), [ ab.SIDE_LEFT, ab.SIDE_RIGHT ] )
        other = ab.window_moments( self.result( [ _placed( 12.0, 13.0, 0, label='exposed_vulva' ) ],
                                                [ _placed( 12.0, 13.0, 0 ) ] ), 10.0, 25.0 )
        self.assertNotIn( ab.SIDE_BOTH, [ moment['side'] for moment in other ] )

    def test_a_box_at_another_time_does_not_count( self ):
        moments = ab.window_moments( self.result( [ _placed( 20.0, 21.0, 0 ) ],
                                                  [ _placed( 12.0, 13.0, 0 ) ] ), 10.0, 25.0 )
        self.assertEqual( sorted( moment['side'] for moment in moments ), [ ab.SIDE_LEFT, ab.SIDE_RIGHT ] )
        # The same half-second bucket, one after the other: still not at the same time.
        moments = ab.window_moments( self.result( [ _placed( 12.30, 12.45, 0 ) ],
                                                  [ _placed( 12.0, 12.2, 0 ) ] ), 10.0, 25.0 )
        self.assertNotIn( ab.SIDE_BOTH, [ moment['side'] for moment in moments ] )

    def test_a_box_straight_above_is_not_the_same_spot( self ):
        moments = ab.window_moments( self.result( [ _placed( 12.0, 13.0, 0, y=500 ) ],
                                                  [ _placed( 12.0, 13.0, 0 ) ] ), 10.0, 25.0 )
        self.assertNotIn( ab.SIDE_BOTH, [ moment['side'] for moment in moments ] )

    def test_moments_split_on_gaps_join_close_ones_sort_by_time_and_clip_to_the_window( self ):
        proposed = [ _placed( 9.0, 10.5, 0 ), _placed( 10.6, 11.0, 0 ), _placed( 14.0, 15.0, 0 ),
                     _placed( 12.0, 12.5, 0, label='exposed_vulva' ), _placed( 26.0, 27.0, 0 ) ]
        moments = ab.window_moments( self.result( [], proposed ), 10.0, 25.0 )
        self.assertEqual( [ ( moment['label'], moment['span'] ) for moment in moments ],
                          [ ( 'exposed_breast', ( 0.0, 1.0 ) ), ( 'exposed_vulva', ( 2.0, 2.5 ) ),
                            ( 'exposed_breast', ( 4.0, 5.0 ) ) ] )

    def test_explain_writes_one_row_per_moment( self ):
        row = { 'file_hash': 'h', 'video': '/v/My Video.mp4', 'name': 'My Video.mp4',
                'cache_paths': [ 'c.gz' ], 'windows': [ ( 10.0, 25.0 ) ] }
        result = self.result( [ _placed( 12.0, 13.0, 0 ) ],
                              [ _placed( 12.0, 13.0, 80 ), _placed( 16.0, 17.0, 900 ),
                                _placed( 20.0, 21.0, 900 ) ] )
        ranking = { 'experiment': 'cut_bursts', 'backend': 'nudenet_v3', 'current': {},
                    'proposed': { 'shot_cut_burst_window': 0.35 } }
        lines, rows = ab.explain_video( row, ranking, 1, compare=lambda *args: result )
        self.assertEqual( [ ( item['when_in_clip'], item['censored_on'] ) for item in rows ],
                          [ ( '0:02.0-0:03.0', ab.SIDE_BOTH ), ( '0:06.0-0:07.0', ab.SIDE_RIGHT ),
                            ( '0:10.0-0:11.0', ab.SIDE_RIGHT ) ] )
        self.assertIn( 'both censor, boxes differ (place or size)', "\n".join( lines ) )


class TestRegistry( unittest.TestCase ):

    def test_every_experiment_is_named_once_and_found( self ):
        names = [ experiment.name for experiment in ex.experiments() ]
        self.assertEqual( len( names ), len( set( names ) ) )
        for name in names:
            self.assertEqual( ex.experiment_named( name ).name, name )
        self.assertEqual( ex.experiment_named( None ).name, 'suppression' )
        with self.assertRaises( SystemExit ):
            ex.experiment_named( 'nope' )

    def test_auto_tune_points_each_reviewed_key_at_an_experiment( self ):
        tool = _load_auto_tune()
        names = set( experiment.name for experiment in ex.experiments() )
        for key, experiment in tool.REVIEW_EXPERIMENTS.items():
            self.assertIn( experiment, names )
            self.assertEqual( ex.experiment_named( experiment ).key, key )


if __name__ == '__main__':
    unittest.main()
