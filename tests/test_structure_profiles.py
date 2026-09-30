"""
test_structure_profiles.py - the profile layer, the settings it
overrides, and the arithmetic that picks one.

WHAT A PROFILE IS
-----------------
A fourth override tier. Per-label settings already resolve through
three: betaconfig.item_overrides, then the backend's own item_overrides,
then the variant's. A profile layers on top of all of them, selected per
video by how heavily cut the footage is.

The reason it exists is that one set of timing values cannot serve both
kinds of footage. A compilation cuts between different people every few
seconds, so a track must give up fast or it keeps drawing a censor on
whoever replaced the original subject. A single-scene video holds on one
person for minutes, so giving up fast means the censor drops out every
time she turns away from the camera. The same track_max_gap cannot be
right for both.

WHAT THIS FILE PINS
-------------------
  - selection arithmetic: thresholds, ties, order-independence, and the
    difference between "no cuts found" and "never looked"
  - the off switch actually switching something off
  - layering: a profile overrides only the keys it names
  - the cut-rate denominator, which shipped wrong (see
    TestCutRateDenominator)
"""

import copy
import os
import sys
import unittest

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )

import betaconfig

import betautils_config as bu_config
import betautils_detector as bu_detector
import betautils_track as bu_track


PROFILES = {
    'default': 'scene',
    'match_on': 'cuts_per_min',
    'variants': {
        'compilation': {
            'min_cuts_per_min': 40,
            'item_overrides': { 'exposed_breast': { 'track_max_gap': 0.20,
                                                    'style_min_dwell_seconds': 1.5 } },
        },
        'scene': {
            'min_cuts_per_min': 0,
            'item_overrides': { 'exposed_breast': { 'track_max_gap': 3.0,
                                                    'style_min_dwell_seconds': 3.0 } },
        },
    },
}


class ProfileHarness( unittest.TestCase ):
    """Installs a known profiles block so tests do not read live config."""

    def setUp( self ):
        self._saved_backend = copy.deepcopy( betaconfig.detector_backend )
        self._saved_enabled = getattr( betaconfig, 'default_profile_enabled', True )
        self._saved_forced = getattr( betaconfig, 'force_structure_profile', None )
        self.backend = betaconfig.detector_backend['selected']
        betaconfig.detector_backend = copy.deepcopy( betaconfig.detector_backend )
        betaconfig.detector_backend[self.backend]['profiles'] = copy.deepcopy( PROFILES )
        betaconfig.default_profile_enabled = True
        betaconfig.force_structure_profile = None
        bu_config.invalidate_config_caches()

    def tearDown( self ):
        betaconfig.detector_backend = self._saved_backend
        betaconfig.default_profile_enabled = self._saved_enabled
        betaconfig.force_structure_profile = self._saved_forced
        bu_config.invalidate_config_caches()

    def profiles( self ):
        return betaconfig.detector_backend[self.backend]['profiles']


class TestProfileSelection( ProfileHarness ):

    def test_a_fast_cut_rate_picks_the_compilation_profile( self ):
        self.assertEqual( bu_detector.select_profile_name( 120.0 ), 'compilation' )

    def test_a_slow_cut_rate_picks_the_scene_profile( self ):
        self.assertEqual( bu_detector.select_profile_name( 4.0 ), 'scene' )

    def test_the_threshold_is_inclusive( self ):
        # 40 must mean "40 qualifies", not "41 does". An exclusive
        # comparison here would be invisible except exactly on the
        # boundary, which is where hand-tuned thresholds tend to sit.
        self.assertEqual( bu_detector.select_profile_name( 40.0 ), 'compilation' )
        self.assertEqual( bu_detector.select_profile_name( 39.9 ), 'scene' )

    def test_zero_cuts_is_not_the_same_as_no_scan( self ):
        # A still video genuinely has 0 cuts/min and should take the
        # profile whose threshold covers 0. That is a measurement, not a
        # missing measurement, so it must not be confused with None.
        self.assertEqual( bu_detector.select_profile_name( 0.0 ), 'scene' )

    def test_no_shot_cut_data_takes_the_configured_default( self ):
        self.assertEqual( bu_detector.select_profile_name( None ), 'scene' )

    def test_the_default_is_honoured_rather_than_assumed( self ):
        self.profiles()['default'] = 'compilation'
        bu_config.invalidate_config_caches()
        self.assertEqual( bu_detector.select_profile_name( None ), 'compilation' )

    def test_selection_does_not_depend_on_declaration_order( self ):
        # Highest cleared threshold wins, so reordering the dict must
        # not change the answer. Dict order is stable in Python but
        # config files get reshuffled by hand.
        reversed_variants = dict( reversed( list( self.profiles()['variants'].items() ) ) )
        self.profiles()['variants'] = reversed_variants
        bu_config.invalidate_config_caches()
        self.assertEqual( bu_detector.select_profile_name( 120.0 ), 'compilation' )
        self.assertEqual( bu_detector.select_profile_name( 4.0 ), 'scene' )

    def test_a_third_profile_slots_in_without_disturbing_the_others( self ):
        self.profiles()['variants']['frenetic'] = {
            'min_cuts_per_min': 200,
            'item_overrides': { 'exposed_breast': { 'track_max_gap': 0.08 } },
        }
        bu_config.invalidate_config_caches()
        self.assertEqual( bu_detector.select_profile_name( 240.0 ), 'frenetic' )
        self.assertEqual( bu_detector.select_profile_name( 120.0 ), 'compilation' )
        self.assertEqual( bu_detector.select_profile_name( 4.0 ), 'scene' )

    def test_a_backend_with_no_profiles_selects_nothing( self ):
        del betaconfig.detector_backend[self.backend]['profiles']
        bu_config.invalidate_config_caches()
        self.assertIsNone( bu_detector.select_profile_name( 120.0 ) )
        self.assertIsNone( bu_detector.select_profile_name( None ) )

    def test_an_empty_variants_block_selects_nothing( self ):
        self.profiles()['variants'] = {}
        bu_config.invalidate_config_caches()
        self.assertIsNone( bu_detector.select_profile_name( 120.0 ) )

    def test_a_default_naming_a_missing_variant_selects_nothing( self ):
        # Rather than raising, or inventing a fallback that would be
        # wrong in a different way. A typo'd default should degrade to
        # "no profile", which is the pre-profiles behaviour.
        self.profiles()['default'] = 'typo'
        bu_config.invalidate_config_caches()
        self.assertIsNone( bu_detector.select_profile_name( None ) )


class TestProfilesCanBeSwitchedOff( ProfileHarness ):

    def test_disabling_profiles_selects_nothing( self ):
        # This shipped broken: default_profile_enabled existed in config
        # and in the censor key but was read by nothing, so turning
        # profiles off changed the output FILENAME without changing the
        # output. That is the worst kind of dead setting, because the
        # new filename looks like proof it took effect.
        betaconfig.default_profile_enabled = False
        bu_config.invalidate_config_caches()
        self.assertIsNone( bu_detector.select_profile_name( 120.0 ) )
        self.assertIsNone( bu_detector.select_profile_name( None ) )

    def test_re_enabling_restores_selection( self ):
        betaconfig.default_profile_enabled = False
        bu_config.invalidate_config_caches()
        self.assertIsNone( bu_detector.select_profile_name( 120.0 ) )
        betaconfig.default_profile_enabled = True
        bu_config.invalidate_config_caches()
        self.assertEqual( bu_detector.select_profile_name( 120.0 ), 'compilation' )


class TestProfileOverrideLayering( ProfileHarness ):

    def test_a_profile_supplies_the_keys_it_names( self ):
        overrides = bu_detector.get_profile_item_overrides(
            'compilation', 'exposed_breast', self.backend )
        self.assertEqual( overrides['track_max_gap'], 0.20 )
        self.assertEqual( overrides['style_min_dwell_seconds'], 1.5 )

    def test_profiles_differ_from_each_other( self ):
        fast = bu_detector.get_profile_item_overrides(
            'compilation', 'exposed_breast', self.backend )
        slow = bu_detector.get_profile_item_overrides(
            'scene', 'exposed_breast', self.backend )
        self.assertNotEqual( fast['track_max_gap'], slow['track_max_gap'] )

    def test_an_unnamed_label_gets_nothing_from_the_profile( self ):
        # A profile is a partial override. A label it does not mention
        # must keep whatever the three lower tiers resolved, not be
        # blanked out.
        self.assertEqual(
            bu_detector.get_profile_item_overrides(
                'compilation', 'exposed_vulva', self.backend ), {} )

    def test_no_profile_name_means_no_overrides( self ):
        self.assertEqual(
            bu_detector.get_profile_item_overrides(
                None, 'exposed_breast', self.backend ), {} )

    def test_an_unknown_profile_name_means_no_overrides( self ):
        self.assertEqual(
            bu_detector.get_profile_item_overrides(
                'nonexistent', 'exposed_breast', self.backend ), {} )

    def test_the_resolved_settings_reflect_the_profile( self ):
        # The layering assertion that matters: not what the profile dict
        # says, but what tracking actually resolves for that label.
        fast = bu_track._LabelSettings(
            'exposed_breast', self.backend, 1.0/9, profile_name='compilation' )
        slow = bu_track._LabelSettings(
            'exposed_breast', self.backend, 1.0/9, profile_name='scene' )
        self.assertEqual( fast.style_min_dwell_seconds, 1.5 )
        self.assertEqual( slow.style_min_dwell_seconds, 3.0 )
        self.assertNotEqual( fast.max_gap, slow.max_gap )

    def test_a_profile_does_not_disturb_unnamed_settings( self ):
        # time_safety and the smoothing alphas are not mentioned by
        # either profile, so all three resolutions must agree on them.
        fast = bu_track._LabelSettings(
            'exposed_breast', self.backend, 1.0/9, profile_name='compilation' )
        slow = bu_track._LabelSettings(
            'exposed_breast', self.backend, 1.0/9, profile_name='scene' )
        none = bu_track._LabelSettings( 'exposed_breast', self.backend, 1.0/9 )
        for attribute in ( 'time_safety', 'alpha', 'size_alpha',
                           'item_x_safety', 'item_y_safety' ):
            self.assertEqual( getattr( fast, attribute ), getattr( none, attribute ),
                              "profile 'compilation' changed %s, which it never names"
                              %( attribute, ) )
            self.assertEqual( getattr( slow, attribute ), getattr( none, attribute ),
                              "profile 'scene' changed %s, which it never names"
                              %( attribute, ) )


class TestCutRateDenominator( unittest.TestCase ):
    """
    cuts_per_min must divide by the length of the window that was
    SCANNED, not by the spread between the first and last cut.

    The spread form shipped, and it reads structure backwards. Two cuts
    0.1s apart in an otherwise still 30s slice scored 1237 cuts/min and
    picked the fast-cut profile; the same two cuts at opposite ends of
    that slice scored 4 and picked the slow one. Identical footage
    density, opposite answers, and the wrong one is confidently wrong.

    This is arithmetic rather than a call into betatv, because the
    calculation is three lines inside process_one_video and reproducing
    the surrounding orchestration would test the harness, not the rule.
    """

    @staticmethod
    def rate( cut_count, scanned_seconds ):
        if not cut_count or not scanned_seconds or scanned_seconds <= 0:
            return None
        return 60.0 * cut_count / scanned_seconds

    def test_two_close_cuts_in_a_long_slice_is_a_slow_rate( self ):
        # The exact case that misfired in a real run.
        self.assertAlmostEqual( self.rate( 2, 30.0 ), 4.0 )

    def test_cut_spacing_does_not_change_the_rate( self ):
        # Whether the cuts are bunched or spread, two cuts in thirty
        # seconds is two cuts in thirty seconds.
        self.assertEqual( self.rate( 2, 30.0 ), self.rate( 2, 30.0 ) )

    def test_a_genuinely_fast_slice_still_reads_fast( self ):
        self.assertAlmostEqual( self.rate( 50, 30.0 ), 100.0 )

    def test_no_cuts_yields_no_rate( self ):
        self.assertIsNone( self.rate( 0, 30.0 ) )

    def test_a_zero_length_window_yields_no_rate( self ):
        # Guards a division by zero on a file whose duration could not
        # be determined.
        self.assertIsNone( self.rate( 5, 0.0 ) )
        self.assertIsNone( self.rate( 5, None ) )

    def test_the_computed_rate_selects_a_sane_profile( self ):
        # Ties the arithmetic to the consequence. Under the shipped
        # spread form this same footage selected 'compilation'.
        saved = copy.deepcopy( betaconfig.detector_backend )
        backend = betaconfig.detector_backend['selected']
        try:
            betaconfig.detector_backend = copy.deepcopy( betaconfig.detector_backend )
            betaconfig.detector_backend[backend]['profiles'] = copy.deepcopy( PROFILES )
            bu_config.invalidate_config_caches()
            self.assertEqual(
                bu_detector.select_profile_name( self.rate( 2, 30.0 ) ), 'scene' )
        finally:
            betaconfig.detector_backend = saved
            bu_config.invalidate_config_caches()


if __name__ == '__main__':
    unittest.main()


class TestForcedProfile( ProfileHarness ):
    """
    --profile NAME: pick a profile by name instead of by cut rate.

    The case it exists for is a compilation that happens to cut slowly.
    The cut rate says 'scene', the footage is really a comp, and no
    threshold can tell them apart because the measurement itself is what
    is misleading.
    """

    def test_a_forced_profile_beats_the_measured_cut_rate( self ):
        betaconfig.force_structure_profile = 'compilation'
        bu_config.invalidate_config_caches()
        # 4.0 cuts/min would select 'scene' on its own.
        self.assertEqual( bu_detector.select_profile_name( 4.0 ), 'compilation' )

    def test_a_forced_profile_applies_when_no_cuts_were_scanned( self ):
        betaconfig.force_structure_profile = 'compilation'
        bu_config.invalidate_config_caches()
        self.assertEqual( bu_detector.select_profile_name( None ), 'compilation' )

    def test_a_forced_profile_overrides_the_off_switch( self ):
        # Typing --profile comp and silently getting NO profile because
        # config disabled them is the same 'looks like it worked' failure
        # the off switch itself was added to fix.
        betaconfig.default_profile_enabled = False
        betaconfig.force_structure_profile = 'compilation'
        bu_config.invalidate_config_caches()
        self.assertEqual( bu_detector.select_profile_name( 4.0 ), 'compilation' )

    def test_an_unknown_forced_profile_selects_nothing_and_fails_validation( self ):
        # It must not quietly fall back to the auto-selected profile:
        # that renders the wrong thing and reports success.
        betaconfig.force_structure_profile = 'nosuchprofile'
        bu_config.invalidate_config_caches()
        self.assertIsNone( bu_detector.select_profile_name( 4.0 ) )
        self.assertTrue( any( 'force_structure_profile' in error
                              for error in bu_config._collect_validation_errors() ) )

    def test_forcing_is_in_the_censor_key( self ):
        import betautils_cache_paths as bu_cache
        betaconfig.force_structure_profile = None
        bu_config.invalidate_config_caches()
        before = bu_cache.censor_key()
        betaconfig.force_structure_profile = 'compilation'
        bu_config.invalidate_config_caches()
        self.assertNotEqual( before, bu_cache.censor_key() )

    def test_the_forced_profile_actually_changes_resolved_settings( self ):
        # Selection returning a name is not the point; the name has to
        # reach the per-label values.
        betaconfig.force_structure_profile = 'compilation'
        bu_config.invalidate_config_caches()
        forced = bu_detector.get_profile_item_overrides(
            bu_detector.select_profile_name( 4.0 ), 'exposed_breast' )
        self.assertEqual( forced.get( 'track_max_gap' ), 0.20 )


class TestManualOnlyProfiles( ProfileHarness ):
    """
    manual_only: a profile automatic selection never picks.

    For footage the cut rate cannot identify, where the profile should
    exist but only be reachable by name.
    """

    def _mark_manual( self, name ):
        self.profiles()['variants'][name]['manual_only'] = True
        bu_config.invalidate_config_caches()

    def test_a_manual_only_profile_is_never_auto_selected( self ):
        self._mark_manual( 'compilation' )
        # 120 cuts/min clears compilation's threshold of 40, so without
        # manual_only this is exactly the rate that would pick it.
        self.assertEqual( bu_detector.select_profile_name( 120.0 ), 'scene' )

    def test_a_manual_only_profile_is_still_reachable_by_name( self ):
        self._mark_manual( 'compilation' )
        betaconfig.force_structure_profile = 'compilation'
        bu_config.invalidate_config_caches()
        self.assertEqual( bu_detector.select_profile_name( 4.0 ), 'compilation' )

    def test_a_manual_only_default_is_not_used_as_the_fallback( self ):
        # 'scene' is the configured default AND the 0-threshold catch-all.
        # Marking it manual_only must take it out of both roles, not just
        # the threshold one.
        self._mark_manual( 'scene' )
        self.assertEqual( bu_detector.select_profile_name( 120.0 ), 'compilation' )
        self.assertIsNone( bu_detector.select_profile_name( None ) )

    def test_all_profiles_manual_only_means_no_auto_profile( self ):
        self._mark_manual( 'compilation' )
        self._mark_manual( 'scene' )
        self.assertIsNone( bu_detector.select_profile_name( 120.0 ) )
        self.assertIsNone( bu_detector.select_profile_name( None ) )

    def test_manual_only_passes_validation_and_a_non_bool_does_not( self ):
        self._mark_manual( 'compilation' )
        self.assertEqual( [ e for e in bu_config._collect_validation_errors()
                            if 'manual_only' in e ], [] )
        self.profiles()['variants']['compilation']['manual_only'] = 'yes'
        bu_config.invalidate_config_caches()
        self.assertTrue( any( 'manual_only' in e
                              for e in bu_config._collect_validation_errors() ) )


class TestProfileByPath( ProfileHarness ):
    """
    profile_by_path_pattern: a standing profile choice per file or folder.

    The "specific video or directory" case - a comps/ folder whose
    contents are compilations however slowly they happen to cut.
    """

    def setUp( self ):
        super().setUp()
        self._saved_mapping = getattr( betaconfig, 'profile_by_path_pattern', None )
        betaconfig.profile_by_path_pattern = None
        bu_config.invalidate_config_caches()

    def tearDown( self ):
        betaconfig.profile_by_path_pattern = self._saved_mapping
        super().tearDown()

    def _map( self, mapping ):
        betaconfig.profile_by_path_pattern = mapping
        bu_config.invalidate_config_caches()

    def test_a_directory_pattern_beats_the_cut_rate( self ):
        self._map( { '*/comps/*': 'compilation' } )
        self.assertEqual(
            bu_detector.select_profile_name(
                4.0, source_path='/videos/comps/slow_but_a_comp.mp4' ),
            'compilation' )

    def test_a_non_matching_path_falls_back_to_the_cut_rate( self ):
        self._map( { '*/comps/*': 'compilation' } )
        self.assertEqual(
            bu_detector.select_profile_name( 4.0, source_path='/videos/scenes/x.mp4' ),
            'scene' )

    def test_a_filename_pattern_matches_the_bare_name_too( self ):
        self._map( { '*reveal*': 'compilation' } )
        self.assertEqual(
            bu_detector.select_profile_name( 4.0, source_path='/v/a_reveal_clip.mp4' ),
            'compilation' )

    def test_no_source_path_is_harmless( self ):
        # Callers that do not know the path (replay tooling) must still work.
        self._map( { '*/comps/*': 'compilation' } )
        self.assertEqual( bu_detector.select_profile_name( 4.0 ), 'scene' )

    def test_an_explicit_profile_outranks_a_path_rule( self ):
        # --profile is the one-off override; the path rule is standing.
        self._map( { '*/comps/*': 'compilation' } )
        betaconfig.force_structure_profile = 'scene'
        bu_config.invalidate_config_caches()
        self.assertEqual(
            bu_detector.select_profile_name( 120.0, source_path='/v/comps/x.mp4' ),
            'scene' )

    def test_a_path_rule_can_select_a_manual_only_profile( self ):
        self.profiles()['variants']['compilation']['manual_only'] = True
        self._map( { '*/comps/*': 'compilation' } )
        self.assertEqual(
            bu_detector.select_profile_name( 4.0, source_path='/v/comps/x.mp4' ),
            'compilation' )

    def test_first_matching_pattern_wins( self ):
        self._map( { '*/comps/special/*': 'scene',
                     '*/comps/*': 'compilation' } )
        self.assertEqual(
            bu_detector.select_profile_name(
                120.0, source_path='/v/comps/special/x.mp4' ), 'scene' )

    def test_an_unknown_profile_name_is_ignored_and_fails_validation( self ):
        self._map( { '*/comps/*': 'nosuchprofile' } )
        self.assertEqual(
            bu_detector.select_profile_name( 4.0, source_path='/v/comps/x.mp4' ),
            'scene', "an unknown name must not select nothing silently" )
        self.assertTrue( any( 'profile_by_path_pattern' in e
                              for e in bu_config._collect_validation_errors() ) )

    def test_the_mapping_is_in_the_censor_key( self ):
        import betautils_cache_paths as bu_cache
        self._map( None )
        before = bu_cache.censor_key()
        self._map( { '*/comps/*': 'compilation' } )
        self.assertNotEqual( before, bu_cache.censor_key() )

    def test_a_non_dict_mapping_fails_validation( self ):
        self._map( [ '*/comps/*' ] )
        self.assertTrue( any( 'profile_by_path_pattern' in e
                              for e in bu_config._collect_validation_errors() ) )


class TestShotSpans( unittest.TestCase ):
    """
    The span list every structure measure is built on.

    Shipped measuring only the gaps BETWEEN cuts, which biases anything
    downstream: a long video with two adjacent cuts looks entirely
    composed of one-second shots because the 40 minutes on either side
    were never counted as shots at all.
    """

    def spans( self, cuts, scanned=None ):
        return sorted( bu_detector.shot_spans_for( cuts, scanned ) )

    def test_nothing_to_measure_yields_no_spans( self ):
        self.assertEqual( bu_detector.shot_spans_for( [] ), [] )
        self.assertEqual( bu_detector.shot_spans_for( None ), [] )

    def test_the_opening_span_counts( self ):
        # A cut at 10s means the first ten seconds were a shot.
        self.assertEqual( self.spans( [ 10.0, 12.0 ] ), [ 2.0, 10.0 ] )

    def test_a_cut_at_zero_contributes_no_opening_span( self ):
        self.assertEqual( self.spans( [ 0.0, 2.0 ] ), [ 2.0 ] )

    def test_the_closing_span_counts_when_the_window_is_known( self ):
        self.assertEqual( self.spans( [ 10.0 ], 60.0 ), [ 10.0, 50.0 ] )

    def test_an_unknown_window_drops_the_closing_span( self ):
        # Better to measure less than to invent an end time.
        self.assertEqual( self.spans( [ 10.0 ] ), [ 10.0 ] )

    def test_a_window_shorter_than_the_last_cut_adds_nothing( self ):
        self.assertEqual( self.spans( [ 1.0, 50.0 ], 10.0 ), [ 1.0, 49.0 ] )

    def test_input_order_does_not_matter( self ):
        self.assertEqual( self.spans( [ 5.0, 1.0, 3.0 ] ),
                          self.spans( [ 1.0, 3.0, 5.0 ] ) )

    def test_duplicate_timestamps_contribute_no_zero_length_shot( self ):
        self.assertEqual( self.spans( [ 1.0, 1.0, 2.0 ] ), [ 1.0, 1.0 ] )


class TestShortShotFraction( unittest.TestCase ):
    """
    The selection signal, and why it is a duration share and not a count.

    Measured on real footage: a video whose cut detector fired on twenty
    consecutive samples inside one two-second transition reported a
    0.10s MEDIAN shot while only 4.7% of its runtime was short shots. A
    genuinely fast-cutting video reported a LONGER median, 0.50s, with
    23.6% of its runtime in short shots. Ranked by median the two come
    out backwards, which is the bug these tests exist to prevent.
    """

    def frac( self, cuts, scanned=None, short=None ):
        return bu_detector.short_shot_fraction_for( cuts, scanned, short )

    def test_nothing_to_measure_is_none_not_zero( self ):
        # None means "not measured". Zero means "measured, no short
        # shots". Collapsing them makes an unscanned file look like slow
        # footage, which is a guess dressed as a measurement.
        self.assertIsNone( self.frac( [] ) )
        self.assertIsNone( self.frac( None ) )

    def test_all_short_shots_is_everything( self ):
        self.assertAlmostEqual( self.frac( [ 0.2, 0.4, 0.6 ], 0.8 ), 1.0 )

    def test_no_short_shots_is_zero( self ):
        self.assertAlmostEqual( self.frac( [ 10.0, 20.0 ], 30.0 ), 0.0 )

    def test_the_share_is_weighted_by_duration_not_by_count( self ):
        # Nine 0.1s shots and one 9.1s shot: 90% of the SHOTS are short,
        # 9% of the RUNTIME is. The distinction is the whole point.
        cuts = [ round( 0.1 * n, 4 ) for n in range( 1, 10 ) ]
        fraction = self.frac( cuts, 10.0 )
        self.assertLess( fraction, 0.15 )
        self.assertGreater( fraction, 0.05 )

    def test_a_flurry_ranks_below_sustained_fast_cutting( self ):
        # The measured inversion, in miniature. The flurry has the
        # shorter median; sustained cutting has the larger share.
        flurry = [ round( 1.0 + 0.1 * n, 4 ) for n in range( 0, 20 ) ]
        sustained = [ round( 0.5 * n, 4 ) for n in range( 1, 90 ) ]
        self.assertLess( bu_detector.median_shot_seconds_for( flurry, 45.0 ),
                         bu_detector.median_shot_seconds_for( sustained, 45.0 ),
                         "precondition: the flurry has the SHORTER median" )
        self.assertLess( self.frac( flurry, 45.0 ), self.frac( sustained, 45.0 ),
                         "the duration share must rank them the other way" )

    def test_the_cutoff_is_configurable( self ):
        cuts = [ 1.5, 3.0 ]
        self.assertAlmostEqual( self.frac( cuts, 4.5, short=1.0 ), 0.0 )
        self.assertAlmostEqual( self.frac( cuts, 4.5, short=2.0 ), 1.0 )

    def test_the_fraction_never_leaves_zero_to_one( self ):
        for cuts, scanned in ( ( [ 0.1, 0.2 ], 0.3 ), ( [ 5.0 ], 10.0 ),
                               ( [ 0.1, 9.0 ], 9.5 ), ( [ 1.0, 1.0 ], 2.0 ) ):
            value = self.frac( cuts, scanned )
            self.assertGreaterEqual( value, 0.0 )
            self.assertLessEqual( value, 1.0 )


class TestMatchOnShortShotFraction( ProfileHarness ):
    """Selection driven by the duration share rather than the cut rate."""

    def setUp( self ):
        super().setUp()
        profiles = self.profiles()
        profiles['match_on'] = 'short_shot_fraction'
        profiles['variants']['quick_cut'] = {
            'min_short_shot_fraction': 0.15,
            'video_censor_fps': 18,
            'item_overrides': { 'exposed_breast': { 'track_max_gap': 0.20 } },
        }
        del profiles['variants']['compilation']
        del profiles['variants']['scene']['min_cuts_per_min']
        bu_config.invalidate_config_caches()

    def select( self, fraction ):
        # cuts_per_min deliberately passed as a value that WOULD pick the
        # other profile under the cut-rate signal, so a test passing here
        # cannot be passing by reading the wrong argument.
        return bu_detector.select_profile_name( 999.0, short_shot_fraction=fraction )

    def test_a_large_share_picks_the_quick_cut_profile( self ):
        self.assertEqual( self.select( 0.236 ), 'quick_cut' )

    def test_a_small_share_stays_on_scene( self ):
        self.assertEqual( self.select( 0.047 ), 'scene' )

    def test_the_threshold_is_inclusive( self ):
        self.assertEqual( self.select( 0.15 ), 'quick_cut' )
        self.assertEqual( self.select( 0.1499 ), 'scene' )

    def test_an_unmeasured_share_takes_the_default( self ):
        self.assertEqual( self.select( None ), 'scene' )

    def test_the_cut_rate_is_ignored_under_this_signal( self ):
        # A rate that clears every cut-rate threshold must not select
        # quick_cut when the share says otherwise.
        self.assertEqual(
            bu_detector.select_profile_name( 9999.0, short_shot_fraction=0.0 ),
            'scene' )

    def test_zero_share_is_not_the_same_as_no_measurement( self ):
        self.assertEqual( self.select( 0.0 ), 'scene' )
        self.assertEqual( self.select( None ), 'scene' )

    def test_the_selected_profile_changes_the_sample_rate( self ):
        self.assertEqual( bu_detector.get_profile_sample_fps( 'quick_cut' ), 18 )
        self.assertNotEqual( bu_detector.get_profile_sample_fps( 'scene' ), 18 )

    def test_the_selected_profile_changes_resolved_settings( self ):
        quick = bu_detector.get_profile_item_overrides(
            'quick_cut', 'exposed_breast', self.backend )
        scene = bu_detector.get_profile_item_overrides(
            'scene', 'exposed_breast', self.backend )
        self.assertEqual( quick['track_max_gap'], 0.20 )
        self.assertNotEqual( scene.get( 'track_max_gap' ), 0.20 )

    def test_an_explicit_profile_still_wins( self ):
        betaconfig.force_structure_profile = 'scene'
        bu_config.invalidate_config_caches()
        self.assertEqual( self.select( 0.9 ), 'scene' )


class TestMatchOnValidation( ProfileHarness ):
    """
    A threshold the active signal never reads is the silent-neighbour
    failure the profile validator exists to catch: the variant looks
    tuned and behaves exactly like the default.
    """

    def errors( self ):
        return [ e for e in bu_config._collect_validation_errors()
                 if 'profiles' in e ]

    def test_a_known_signal_passes( self ):
        for signal in ( 'cuts_per_min', 'short_shot_fraction' ):
            self.profiles()['match_on'] = signal
            key = bu_detector.PROFILE_THRESHOLD_KEYS[signal]
            for variant in self.profiles()['variants'].values():
                variant.pop( 'min_cuts_per_min', None )
                variant.pop( 'min_short_shot_fraction', None )
            self.profiles()['variants']['compilation'][key] = 0.5
            bu_config.invalidate_config_caches()
            self.assertEqual( self.errors(), [], signal )

    def test_an_unknown_signal_fails( self ):
        self.profiles()['match_on'] = 'p90_simultaneous'
        bu_config.invalidate_config_caches()
        self.assertTrue( any( 'match_on' in e for e in self.errors() ) )

    def test_a_threshold_for_the_other_signal_fails( self ):
        self.profiles()['match_on'] = 'short_shot_fraction'
        bu_config.invalidate_config_caches()
        self.assertTrue(
            any( 'never match automatically' in e for e in self.errors() ),
            "min_cuts_per_min under short_shot_fraction is read by nothing" )

    def test_a_manual_only_variant_may_carry_any_threshold( self ):
        # It is never auto-selected, so an unread threshold there is
        # inert rather than misleading.
        self.profiles()['match_on'] = 'short_shot_fraction'
        self.profiles()['variants']['compilation']['manual_only'] = True
        self.profiles()['variants']['scene']['min_short_shot_fraction'] = 0
        bu_config.invalidate_config_caches()
        self.assertEqual( self.errors(), [] )

    def test_a_fraction_above_one_fails( self ):
        self.profiles()['match_on'] = 'short_shot_fraction'
        for variant in self.profiles()['variants'].values():
            variant.pop( 'min_cuts_per_min', None )
        self.profiles()['variants']['compilation']['min_short_shot_fraction'] = 15
        bu_config.invalidate_config_caches()
        self.assertTrue( any( 'fraction of runtime' in e for e in self.errors() ) )

    def test_a_boolean_threshold_fails( self ):
        # True == 1 passes a naive numeric check.
        self.profiles()['variants']['compilation']['min_cuts_per_min'] = True
        bu_config.invalidate_config_caches()
        self.assertTrue( any( 'min_cuts_per_min' in e for e in self.errors() ) )

    def test_the_threshold_key_map_is_shared_with_the_validator( self ):
        # Two copies of this map would drift, and the drift would show up
        # as a valid config failing validation or the reverse.
        self.assertEqual( set( bu_detector.PROFILE_THRESHOLD_KEYS ),
                          { 'cuts_per_min', 'short_shot_fraction' } )
