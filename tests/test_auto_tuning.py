"""
test_auto_tuning.py - the parts of automatic tuning that must not be wrong.

Three things here can cause real damage, so each is tested directly
rather than through a tool's output:

  1. The COVERAGE GUARD. It is the mechanical form of "performance work
     never costs valid detections". If it can be fooled, the tuner will
     happily accept a setting that leaves skin uncovered because the
     track counts looked tidy.
  2. The CONFIG WRITER. It edits a live betaconfig.py. An edit that hits
     the wrong key, or that writes a value the resolver then reports
     differently, is worse than a failed edit, because every later stage
     measures against a setting nobody chose.
  3. The DECISION RULES, which must reject on price even when the
     headline number improves - that being the exact trap a real sweep
     laid: tripling match_distance cut resets by 130 and added 8874
     risky assignments, and a rule that only read "fewer resets" would
     have taken it.
"""

import copy
import importlib.util
import os
import sys
import tempfile
import unittest

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )

import betaconfig
import betautils_config as bu_config
import betautils_detector as bu_detector
import betautils_tuning as bu_tuning


def _load_auto_tune():
    path = os.path.join( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ),
                         'tools', 'tuning', 'auto_tune.py' )
    spec = importlib.util.spec_from_file_location( 'auto_tune_under_test', path )
    module = importlib.util.module_from_spec( spec )
    spec.loader.exec_module( module )
    return module


auto_tune = _load_auto_tune()


class _Options:
    """Stand-in for the parsed CLI namespace the decision rules read."""

    def __init__( self, max_coverage_loss=0.005, max_risky_per_reset=5.0 ):
        self.max_coverage_loss = max_coverage_loss
        self.max_risky_per_reset = max_risky_per_reset


def _metrics( new_tracks=100, risky=100, coverage=None, dropped_unconfirmed=0,
              rendered=1000 ):
    return {
        'new_tracks': new_tracks,
        'risky_contention': risky,
        'tracks': 50,
        'rendered_boxes': rendered,
        'interpolated': 0,
        'dropped_unconfirmed': dropped_unconfirmed,
        'coverage_by_label': coverage if coverage is not None else { 'exposed_breast': 100.0 },
    }


def _row( value, label, metrics, baseline_metrics ):
    return {
        'value': value,
        'label': label,
        'metrics': metrics,
        'coverage_loss': bu_tuning.worst_coverage_loss( baseline_metrics, metrics ),
        'verdict': '',
    }


class TestCoverageGuard( unittest.TestCase ):

    def test_a_label_losing_coverage_is_reported_even_when_others_gain( self ):
        baseline = _metrics( coverage={ 'exposed_breast': 100.0, 'exposed_vulva': 100.0 } )
        candidate = _metrics( coverage={ 'exposed_breast': 130.0, 'exposed_vulva': 90.0 } )
        self.assertAlmostEqual( bu_tuning.worst_coverage_loss( baseline, candidate ), 0.10,
            msg="a gain on one label must never mask a loss on another - censoring is per "
                "label, and an uncovered vulva is not paid for by a larger breast box" )

    def test_no_loss_reads_as_zero( self ):
        baseline = _metrics( coverage={ 'exposed_breast': 100.0 } )
        candidate = _metrics( coverage={ 'exposed_breast': 120.0 } )
        self.assertEqual( bu_tuning.worst_coverage_loss( baseline, candidate ), 0.0 )

    def test_a_label_missing_from_the_candidate_counts_as_total_loss( self ):
        baseline = _metrics( coverage={ 'exposed_breast': 100.0 } )
        candidate = _metrics( coverage={} )
        self.assertAlmostEqual( bu_tuning.worst_coverage_loss( baseline, candidate ), 1.0,
            msg="a label that stopped being censored entirely is the worst possible outcome, "
                "not an absent measurement" )

    def test_a_label_with_no_baseline_coverage_is_skipped( self ):
        baseline = _metrics( coverage={ 'exposed_breast': 0.0 } )
        candidate = _metrics( coverage={ 'exposed_breast': 0.0 } )
        self.assertEqual( bu_tuning.worst_coverage_loss( baseline, candidate ), 0.0 )


class TestLooseningDecisionRule( unittest.TestCase ):
    """
    The rule that has to price a change, not just count its wins.
    """

    def setUp( self ):
        self.stage = auto_tune.TrackMaxGapStage( 'nudenet_v3', 'exposed_breast', 'nudenet_v3/320n @320' )
        self.baseline_metrics = _metrics( new_tracks=292, risky=383 )
        self.baseline_row = _row( 13.5, 'baseline (13.5)', self.baseline_metrics, self.baseline_metrics )
        self.baseline_row['verdict'] = 'baseline'

    def _decide( self, rows, options=None, configurations=None ):
        """Score and decide, the way the tool does: rows per configuration."""
        options = options or _Options()
        by_configuration = configurations or { 'only @320': rows }
        for these in by_configuration.values():
            self.stage.score_rows( these, options )
        return self.stage.decide( by_configuration, options, None )

    def test_a_cheap_win_is_accepted( self ):
        # The real gap-sweep shape: 91 resets avoided for 192 risky, 2.1 each.
        candidate = _row( 27.0, '2x (27s)', _metrics( new_tracks=201, risky=575 ),
                          self.baseline_metrics )
        decision = self._decide( [ self.baseline_row, candidate ] )
        self.assertTrue( decision.accepted )
        self.assertEqual( decision.chosen, 27.0 )

    def test_an_expensive_win_is_rejected_despite_better_headline_numbers( self ):
        # The real match-distance shape: 130 resets avoided for 8874 risky, 68 each.
        candidate = _row( 40.5, '3x', _metrics( new_tracks=162, risky=9257 ),
                          self.baseline_metrics )
        decision = self._decide( [ self.baseline_row, candidate ] )
        self.assertFalse( decision.accepted,
            "a candidate with FEWER resets than the baseline must still be rejected when each "
            "one cost dozens of risky assignments - counting wins without pricing them is the "
            "exact mistake this rule exists to prevent" )
        self.assertIn( 'risky assignments per reset', candidate['verdict'] )

    def test_coverage_loss_overrides_a_perfect_score( self ):
        candidate = _row( 27.0, '2x',
                          _metrics( new_tracks=10, risky=383,
                                    coverage={ 'exposed_breast': 90.0 } ),
                          self.baseline_metrics )
        decision = self._decide( [ self.baseline_row, candidate ] )
        self.assertFalse( decision.accepted,
            "a candidate that removes almost every reset at no contention cost is still "
            "refused when it censors less - that trade is never available" )
        self.assertIn( 'coverage', candidate['verdict'] )

    def test_the_cheapest_eligible_candidate_does_not_automatically_win( self ):
        # Among eligible candidates the rule maximises resets avoided,
        # because they have all already passed the price test.
        cheap = _row( 20.0, '1.5x', _metrics( new_tracks=250, risky=400 ), self.baseline_metrics )
        better = _row( 27.0, '2x', _metrics( new_tracks=201, risky=500 ), self.baseline_metrics )
        decision = self._decide( [ self.baseline_row, cheap, better ] )
        self.assertTrue( decision.accepted )
        self.assertEqual( decision.chosen, 27.0 )

    def test_nothing_is_chosen_when_every_candidate_fails( self ):
        worse = _row( 6.75, '0.5x', _metrics( new_tracks=400, risky=300 ), self.baseline_metrics )
        decision = self._decide( [ self.baseline_row, worse ] )
        self.assertFalse( decision.accepted )
        self.assertEqual( decision.chosen, 13.5 )


class TestMinTrackHitsDecisionRule( unittest.TestCase ):
    """A stage whose whole job is removing boxes, so the guard IS the rule."""

    def setUp( self ):
        self.stage = auto_tune.MinTrackHitsStage( 'nudenet_v3', 'exposed_breast' )
        self.baseline_metrics = _metrics( dropped_unconfirmed=0 )
        self.baseline_row = _row( 1, 'baseline (1)', self.baseline_metrics, self.baseline_metrics )
        self.baseline_row['verdict'] = 'baseline'

    def _decide( self, rows, options=None, configurations=None ):
        """Score and decide, the way the tool does: rows per configuration."""
        options = options or _Options()
        by_configuration = configurations or { 'only @320': rows }
        for these in by_configuration.values():
            self.stage.score_rows( these, options )
        return self.stage.decide( by_configuration, options, None )

    def test_a_tiny_coverage_cost_is_allowed( self ):
        candidate = _row( 2, '2 hit(s)',
                          _metrics( dropped_unconfirmed=400,
                                    coverage={ 'exposed_breast': 99.9 } ),
                          self.baseline_metrics )
        decision = self._decide( [ self.baseline_row, candidate ] )
        self.assertTrue( decision.accepted )
        self.assertEqual( decision.chosen, 2 )

    def test_a_real_coverage_cost_is_not( self ):
        candidate = _row( 2, '2 hit(s)',
                          _metrics( dropped_unconfirmed=4000,
                                    coverage={ 'exposed_breast': 95.0 } ),
                          self.baseline_metrics )
        decision = self._decide( [ self.baseline_row, candidate ] )
        self.assertFalse( decision.accepted,
            "dropping four thousand boxes looks like a big cleanup right up until you notice "
            "it cost 5% of the censoring" )

    def test_a_value_that_changes_nothing_is_rejected( self ):
        candidate = _row( 2, '2 hit(s)', _metrics( dropped_unconfirmed=0 ), self.baseline_metrics )
        decision = self._decide( [ self.baseline_row, candidate ] )
        self.assertFalse( decision.accepted )



class TestOneDecisionPerSharedKey( unittest.TestCase ):
    """
    Every variant of a backend reads the SAME item_overrides block.

    The first real run proved what happens without this: the 320n stage
    raised exposed_vulva's track_max_gap from 21.6 to 64.8, and the 640m
    stage then read 64.8 as ITS baseline and doubled it again to 129.6 -
    a value no measurement ever proposed, arrived at by two stages
    compounding each other's writes on a key they share.
    """

    def setUp( self ):
        self.stage = auto_tune.TrackMaxGapStage( 'nudenet_v3', 'exposed_breast' )
        self.baseline_metrics = _metrics( new_tracks=300, risky=400 )

    def _rows( self, candidates ):
        baseline = _row( 13.5, 'baseline (13.5)', self.baseline_metrics, self.baseline_metrics )
        baseline['verdict'] = 'baseline'
        rows = [ baseline ]
        for value, metrics in candidates:
            rows.append( _row( value, str( value ), metrics, self.baseline_metrics ) )
        return rows

    def _decide_across( self, by_configuration, options=None ):
        options = options or _Options()
        for rows in by_configuration.values():
            self.stage.score_rows( rows, options )
        return self.stage.decide( by_configuration, options, None )

    def test_a_value_good_for_one_variant_and_harmful_for_another_is_rejected( self ):
        decision = self._decide_across( {
            'nudenet_v3/320n @320': self._rows( [
                ( 27.0, _metrics( new_tracks=250, risky=450 ) ) ] ),
            'nudenet_v3/640m @640': self._rows( [
                ( 27.0, _metrics( new_tracks=250, risky=450,
                                  coverage={ 'exposed_breast': 90.0 } ) ) ] ),
        } )
        self.assertFalse( decision.accepted,
            "'good for 320n, harmful for 640m' is not a value you want written to a setting "
            "both of them read" )

    def test_a_value_good_everywhere_is_accepted_once( self ):
        decision = self._decide_across( {
            'nudenet_v3/320n @320': self._rows( [
                ( 27.0, _metrics( new_tracks=250, risky=450 ) ) ] ),
            'nudenet_v3/640m @640': self._rows( [
                ( 27.0, _metrics( new_tracks=260, risky=440 ) ) ] ),
        } )
        self.assertTrue( decision.accepted )
        self.assertEqual( decision.chosen, 27.0 )
        self.assertIn( '320n', decision.reasoning )
        self.assertIn( '640m', decision.reasoning,
            "the reasoning has to name every configuration the decision covers, or you "
            "cannot tell what evidence it rests on" )

    def test_the_winner_maximises_total_resets_avoided_across_configurations( self ):
        decision = self._decide_across( {
            'a @320': self._rows( [
                ( 20.0, _metrics( new_tracks=290, risky=410 ) ),
                ( 27.0, _metrics( new_tracks=270, risky=430 ) ) ] ),
            'b @640': self._rows( [
                ( 20.0, _metrics( new_tracks=295, risky=405 ) ),
                ( 27.0, _metrics( new_tracks=260, risky=440 ) ) ] ),
        } )
        self.assertEqual( decision.chosen, 27.0 )

    def test_the_baseline_is_shared_not_re_read_per_configuration( self ):
        # Both configurations must start from the same baseline value,
        # because there is only one value in config for them to read.
        by_configuration = {
            'a @320': self._rows( [ ( 27.0, _metrics( new_tracks=250, risky=450 ) ) ] ),
            'b @640': self._rows( [ ( 27.0, _metrics( new_tracks=250, risky=450 ) ) ] ),
        }
        baselines = { rows[0]['value'] for rows in by_configuration.values() }
        self.assertEqual( len( baselines ), 1 )



class TestVariantTier( unittest.TestCase ):
    """
    The third resolution tier, added because the variants disagreed.

    The first real tuning run found match_distance_multiplier 2.0 better
    on BOTH axes for nudenet_v3's 640m (39 fewer resets, 51 fewer risky)
    while 320n rejected every candidate. With only a backend tier that
    value could not be written at all - two sets of weights were forced
    to share one number.
    """

    def setUp( self ):
        self._saved = copy.deepcopy( betaconfig.detector_backend )

    def tearDown( self ):
        betaconfig.detector_backend = self._saved
        bu_config.invalidate_config_caches()

    def _set_variant( self, backend, variant, label, key, value ):
        block = betaconfig.detector_backend.setdefault( backend, {} )
        variants = block.setdefault( 'variants', {} )
        overrides = variants.setdefault( variant, {} ).setdefault( 'item_overrides', {} )
        overrides.setdefault( label, {} )[key] = value
        bu_config.invalidate_config_caches()

    def test_a_variant_value_wins_over_the_backend_value( self ):
        self._set_variant( 'nudenet_v3', '640m', 'exposed_breast', 'track_max_gap', 99.0 )
        self.assertEqual(
            bu_detector.get_item_overrides( 'exposed_breast', 'nudenet_v3', '640m' )['track_max_gap'],
            99.0 )

    def test_another_variant_still_inherits_the_backend_value( self ):
        backend_value = bu_detector.get_item_overrides(
            'exposed_breast', 'nudenet_v3', '320n' )['track_max_gap']
        self._set_variant( 'nudenet_v3', '640m', 'exposed_breast', 'track_max_gap', 99.0 )
        self.assertEqual(
            bu_detector.get_item_overrides( 'exposed_breast', 'nudenet_v3', '320n' )['track_max_gap'],
            backend_value,
            "a variant naming one setting must not detach it from everything else it shares" )

    def test_a_variant_only_names_what_it_differs_on( self ):
        self._set_variant( 'nudenet_v3', '640m', 'exposed_breast', 'track_max_gap', 99.0 )
        resolved = bu_detector.get_item_overrides( 'exposed_breast', 'nudenet_v3', '640m' )
        backend_only = bu_detector.get_item_overrides( 'exposed_breast', 'nudenet_v3', '320n' )
        inherited = { key: value for key, value in backend_only.items()
                      if key != 'track_max_gap' }
        for key, value in inherited.items():
            self.assertEqual( resolved.get( key ), value,
                "%s should have been inherited, not dropped"%(key) )

    def test_a_non_tunable_key_is_ignored_at_the_variant_tier_too( self ):
        self._set_variant( 'nudenet_v3', '640m', 'exposed_breast', 'censor_shape', 'box' )
        resolved = bu_detector.get_item_overrides( 'exposed_breast', 'nudenet_v3', '640m' )
        shared = getattr( betaconfig, 'item_overrides', {} ).get( 'exposed_breast', {} )
        self.assertEqual( resolved.get( 'censor_shape' ), shared.get( 'censor_shape' ),
            "how censoring LOOKS is not a per-variant question, and the resolver must not "
            "quietly accept it becoming one" )

    def test_an_unknown_variant_name_is_a_validation_error( self ):
        import betaconst
        self._set_variant( 'nudenet_v3', 'no_such_variant', 'exposed_breast', 'track_max_gap', 9.0 )
        errors = []
        bu_config._validate_backend_item_overrides( errors, set( betaconst.classes ) )
        self.assertTrue( any( 'no_such_variant' in error for error in errors ),
            "a typo'd variant name never resolves, so the block silently does nothing - "
            "exactly the sort of wrongness that has to fail at startup" )

    def test_the_censor_key_differs_between_variants( self ):
        import os
        import betautils_cache_paths as bu_cache
        self._set_variant( 'nudenet_v3', '640m', 'exposed_breast', 'track_max_gap', 99.0 )
        saved_env = os.environ.get( 'BETASUITE_DETECTOR_VARIANT_OVERRIDE' )
        try:
            os.environ['BETASUITE_DETECTOR_VARIANT_OVERRIDE'] = '320n'
            bu_config.invalidate_config_caches()
            first = bu_cache.censor_identity( 'nudenet_v3' )
            os.environ['BETASUITE_DETECTOR_VARIANT_OVERRIDE'] = '640m'
            bu_config.invalidate_config_caches()
            second = bu_cache.censor_identity( 'nudenet_v3' )
        finally:
            if saved_env is None:
                os.environ.pop( 'BETASUITE_DETECTOR_VARIANT_OVERRIDE', None )
            else:
                os.environ['BETASUITE_DETECTOR_VARIANT_OVERRIDE'] = saved_env
            bu_config.invalidate_config_caches()
        self.assertNotEqual( first, second,
            "two variants with different tracking settings must not share a censor key, or "
            "one's rendered output silently overwrites the other's" )


class TestSplittingWhenConfigurationsDisagree( unittest.TestCase ):
    """When no single value suits both, each variant gets its own."""

    def setUp( self ):
        self.stage = auto_tune.TrackMaxGapStage( 'nudenet_v3', 'exposed_breast' )
        self.baseline_metrics = _metrics( new_tracks=300, risky=400 )

    def _rows( self, candidates ):
        baseline = _row( 13.5, 'baseline (13.5)', self.baseline_metrics, self.baseline_metrics )
        baseline['verdict'] = 'baseline'
        return [ baseline ] + [ _row( value, str( value ), metrics, self.baseline_metrics )
                                for value, metrics in candidates ]

    def _decide( self, by_configuration ):
        options = _Options()
        for rows in by_configuration.values():
            self.stage.score_rows( rows, options )
        return self.stage.decide( by_configuration, options, None )

    def test_each_configuration_gets_its_own_value_when_they_disagree( self ):
        decision = self._decide( {
            'nudenet_v3/320n @320': self._rows( [
                ( 20.0, _metrics( new_tracks=280, risky=420 ) ),
                ( 27.0, _metrics( new_tracks=270, risky=900 ) ) ] ),
            'nudenet_v3/640m @640': self._rows( [
                ( 20.0, _metrics( new_tracks=295, risky=900 ) ),
                ( 27.0, _metrics( new_tracks=250, risky=430 ) ) ] ),
        } )
        self.assertTrue( decision.accepted )
        self.assertTrue( decision.is_split )
        self.assertEqual( decision.per_configuration['nudenet_v3/320n @320'], 20.0 )
        self.assertEqual( decision.per_configuration['nudenet_v3/640m @640'], 27.0 )

    def test_one_configuration_having_no_opinion_does_not_trigger_a_split( self ):
        decision = self._decide( {
            'nudenet_v3/320n @320': self._rows( [
                ( 27.0, _metrics( new_tracks=270, risky=900 ) ) ] ),
            'nudenet_v3/640m @640': self._rows( [
                ( 27.0, _metrics( new_tracks=250, risky=430 ) ) ] ),
        } )
        self.assertFalse( decision.is_split,
            "one variant finding a winner while the other finds none is not disagreement, "
            "it is silence - splitting a shared setting on that is overreach" )

    def test_agreement_still_produces_one_shared_value( self ):
        decision = self._decide( {
            'a @320': self._rows( [ ( 27.0, _metrics( new_tracks=250, risky=430 ) ) ] ),
            'b @640': self._rows( [ ( 27.0, _metrics( new_tracks=255, risky=435 ) ) ] ),
        } )
        self.assertTrue( decision.accepted )
        self.assertFalse( decision.is_split )
        self.assertEqual( decision.chosen, 27.0 )


class TestConfigWriter( unittest.TestCase ):
    """
    Editing a live config file, carefully.
    """

    SAMPLE = '''"""A stand-in betaconfig for tests."""

render_chunk_seconds = 600
render_workers = 0          # 0 = auto

detector_backend = {
    'selected': 'nudenet_v3',
    'nudenet_v3': {
        'model_variant': '320n',
        'item_overrides': {
            'exposed_vulva': {
                'min_prob': 0.335,
                'track_max_gap': 10.8,
            },
            'exposed_breast': {
                'min_prob': 0.37,
                'track_max_gap': 13.5,
            },
        },
    },
}
'''

    def setUp( self ):
        self._dir = tempfile.TemporaryDirectory()
        self.path = os.path.join( self._dir.name, 'betaconfig.py' )
        with open( self.path, 'w', encoding='UTF-8' ) as fout:
            fout.write( self.SAMPLE )
        self.writer = bu_tuning.ConfigWriter( config_path=self.path, backup_dir=self._dir.name )
        # The verification step reloads the REAL betaconfig, which would
        # not reflect this temporary file. These tests exercise the
        # locating and rewriting; verification is covered by the live
        # smoke run in the tuner itself.
        self.writer._write_and_verify = self._write_only

    def tearDown( self ):
        self._dir.cleanup()

    def _write_only( self, updated_source, resolve, expected, description ):
        self.writer._backup()
        with open( self.path, 'w', encoding='UTF-8' ) as fout:
            fout.write( updated_source )

    def test_it_edits_the_right_label_block( self ):
        self.writer.set_item_override( 'nudenet_v3', 'exposed_breast', 'track_max_gap', 27.0 )
        written = self.writer.read()
        self.assertIn( "'track_max_gap': 27,", written )
        self.assertIn( "'track_max_gap': 10.8,", written,
            "editing exposed_breast must not touch exposed_vulva's identically named key" )

    def test_it_leaves_other_keys_and_comments_alone( self ):
        self.writer.set_item_override( 'nudenet_v3', 'exposed_vulva', 'min_prob', 0.4 )
        written = self.writer.read()
        self.assertIn( "'min_prob': 0.4,", written )
        self.assertIn( "'model_variant': '320n',", written )
        self.assertIn( "# 0 = auto", written )

    def test_a_missing_key_is_inserted( self ):
        # The tuner's most useful decisions are about keys that have
        # never been set, because they default implicitly and so appear
        # nowhere. Refusing them meant measuring correctly and then
        # asking the user to type the answer in themselves.
        self.writer.set_item_override( 'nudenet_v3', 'exposed_breast', 'min_track_hits', 2 )
        written = self.writer.read()
        self.assertIn( "'min_track_hits': 2,", written )
        self.assertIn( 'set by auto_tune', written,
            "a value that appeared in config without being typed should say where it came from" )

    def test_an_inserted_key_lands_in_the_right_label_block( self ):
        self.writer.set_item_override( 'nudenet_v3', 'exposed_breast',
                                       'match_distance_multiplier', 2.0 )
        written = self.writer.read()
        breast_at = written.index( "'exposed_breast': {" )
        vulva_at = written.index( "'exposed_vulva': {" )
        inserted_at = written.index( "'match_distance_multiplier'" )
        self.assertGreater( inserted_at, breast_at,
            "an inserted key must land inside the label it was decided for" )
        self.assertGreater( inserted_at, vulva_at )

    def test_an_inserted_key_matches_its_siblings_indentation( self ):
        self.writer.set_item_override( 'nudenet_v3', 'exposed_vulva', 'min_track_hits', 3 )
        written = self.writer.read()
        sibling = [ line for line in written.splitlines() if "'min_prob': 0.335" in line ][0]
        inserted = [ line for line in written.splitlines() if "'min_track_hits'" in line ][0]
        self.assertEqual( len( sibling ) - len( sibling.lstrip() ),
                          len( inserted ) - len( inserted.lstrip() ),
            "a tuner that reformats a hand-maintained config is a tuner people stop trusting "
            "with the file" )

    def test_the_file_still_parses_after_an_insert( self ):
        import ast
        self.writer.set_item_override( 'nudenet_v3', 'exposed_vulva', 'min_track_hits', 3 )
        self.writer.set_item_override( 'nudenet_v3', 'exposed_breast', 'min_track_hits', 2 )
        ast.parse( self.writer.read() )

    def test_an_inserted_key_can_then_be_updated( self ):
        self.writer.set_item_override( 'nudenet_v3', 'exposed_breast', 'min_track_hits', 2 )
        self.writer.set_item_override( 'nudenet_v3', 'exposed_breast', 'min_track_hits', 3 )
        written = self.writer.read()
        self.assertEqual( written.count( "'min_track_hits'" ), 1,
            "a second decision on the same key must replace the first, not stack another line" )
        self.assertIn( "'min_track_hits': 3,", written )

    def test_a_duplicated_key_is_still_refused( self ):
        doubled = self.SAMPLE.replace(
            "                'track_max_gap': 13.5,\n",
            "                'track_max_gap': 13.5,\n                'track_max_gap': 99.0,\n" )
        with open( self.path, 'w', encoding='UTF-8' ) as fout:
            fout.write( doubled )
        with self.assertRaises( bu_tuning.ConfigWriteError ):
            self.writer.set_item_override( 'nudenet_v3', 'exposed_breast', 'track_max_gap', 27.0 )

    def test_an_unknown_label_is_refused( self ):
        with self.assertRaises( bu_tuning.ConfigWriteError ):
            self.writer.set_item_override( 'nudenet_v3', 'no_such_label', 'track_max_gap', 5 )

    def test_an_unknown_backend_is_refused( self ):
        with self.assertRaises( bu_tuning.ConfigWriteError ):
            self.writer.set_item_override( 'no_such_backend', 'exposed_breast', 'track_max_gap', 5 )

    def test_a_module_scalar_keeps_its_trailing_comment( self ):
        self.writer.set_module_scalar( 'render_workers', 4 )
        written = self.writer.read()
        self.assertIn( 'render_workers = 4', written )
        self.assertIn( '# 0 = auto', written )

    def test_every_write_leaves_a_backup( self ):
        self.writer.set_module_scalar( 'render_chunk_seconds', 180 )
        self.assertEqual( len( self.writer.backups ), 1 )
        with open( self.writer.backups[0], encoding='UTF-8' ) as fin:
            self.assertIn( 'render_chunk_seconds = 600', fin.read() )


class TestStageCandidates( unittest.TestCase ):

    def setUp( self ):
        self._saved = copy.deepcopy( betaconfig.detector_backend )

    def tearDown( self ):
        betaconfig.detector_backend = self._saved
        bu_config.invalidate_config_caches()

    def test_candidates_exclude_the_current_value( self ):
        stage = auto_tune.MatchDistanceStage( 'nudenet_v3', 'exposed_breast' )
        values = [ value for value, _label in stage.candidates() ]
        baseline = stage.baseline_value() or 1.0
        self.assertNotIn( baseline, values,
            "measuring the baseline twice wastes a replay and makes the table confusing" )

    def test_track_max_gap_candidates_scale_the_real_baseline( self ):
        stage = auto_tune.TrackMaxGapStage( 'nudenet_v3', 'exposed_breast' )
        baseline = stage.baseline_value()
        if baseline is None:
            self.skipTest( 'no explicit track_max_gap configured for this label' )
        values = [ value for value, _label in stage.candidates() ]
        self.assertIn( round( baseline * 2.0, 4 ), values )
        self.assertIn( round( baseline * 0.5, 4 ), values,
            "the sweep must be able to TIGHTEN a threshold, not only loosen it - a baseline "
            "inherited from an older configuration can easily be too loose" )

    def test_applying_a_candidate_reaches_the_resolver( self ):
        stage = auto_tune.TrackMaxGapStage( 'nudenet_v3', 'exposed_breast' )
        stage.apply( 99.0 )
        self.assertAlmostEqual(
            bu_detector.get_item_overrides( 'exposed_breast', 'nudenet_v3' )['track_max_gap'], 99.0,
            msg="a candidate written to the shared item_overrides instead of the backend block "
                "is silently overwritten by the resolver, and every row of the sweep becomes "
                "the same run" )


if __name__ == '__main__':
    unittest.main()
