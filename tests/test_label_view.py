"""
test_label_view.py - the per-label view the app shows (config.get labels)
says what a run really uses, and where it comes from.

The app marks a per-label row "in effect", "overridden by" another
tier, or "not used" with the reason, from betautils_config.label_view.
Wrong here means a user edits a value that changes nothing, or trusts a
value that a higher tier quietly replaces. So for every label, under
each backend, model variant and reverse-mode set-up:

  - each field's value is exactly get_label_settings' value
  - exactly one layer of its chain is in effect, and it is the source
    the resolver itself recorded
  - every other layer that applies has lower precedence than the winner
  - a layer that does not apply says why, and the reason is the real one
"""

import copy
import unittest

import betaconfig
import betautils_config as bu_config
import betautils_detector as bu_detector
import betautils_settings as bu_settings

from tests import config_harness as harness

BACKEND = 'nudenet_v3'
WINDOW_LABEL = 'face_femme'
REVERSE_CENSORED = 'exposed_breast'


def _setups():
    """( name, callable ) that each leave the live config in one shape."""
    def plain():
        pass

    def other_backend():
        others = [ name for name in bu_detector.registered_backend_names() if name != BACKEND ]
        betaconfig.detector_backend['selected'] = others[0]

    def other_variants():
        for variant in bu_detector.variant_names( BACKEND ):
            yield variant

    def reverse_on():
        betaconfig.reverse['enabled'] = True
        betaconfig.reverse['windows'][WINDOW_LABEL] = { 'min_prob': 0.41 }
        betaconfig.reverse['censor']['labels'] = [ REVERSE_CENSORED ]
        betaconfig.reverse['censor']['overrides'][REVERSE_CENSORED] = { 'time_safety': 0.9 }

    found = [ ( 'plain', plain ), ( 'other backend', other_backend ), ( 'reverse on', reverse_on ) ]
    for variant in other_variants():
        found.append( ( 'variant ' + variant,
                        lambda variant=variant: betaconfig.detector_backend[BACKEND].__setitem__(
                            'model_variant', variant ) ) )
    return found


def _check_view( test, where ):
    for view in bu_config.label_views():
        label = view['label']
        expected = bu_config.get_label_settings( label )
        test.assertEqual( set( expected ), set( view['fields'] ), ( where, label ) )
        for key, field in view['fields'].items():
            context = ( where, label, key )
            test.assertEqual( expected[key], field['value'], context )
            chain = field['chain']
            winners = [ index for index, layer in enumerate( chain ) if layer.get( 'wins' ) ]
            test.assertEqual( 1, len( winners ), context )
            winner = chain[winners[0]]
            test.assertEqual( field['source'], winner['source'], context )
            test.assertNotIn( 'inactive', winner, context )
            if 'value' in winner and winner['source'] != bu_detector.SOURCE_BUILTIN:
                test.assertEqual( expected[key], winner['value'], context )
            for index, layer in enumerate( chain ):
                if 'inactive' not in layer and index != winners[0]:
                    test.assertLess( index, winners[0], ( context, 'applies above the winner', layer ) )


class TestEveryFieldIsWhatARunUses( unittest.TestCase ):

    def test_under_each_backend_variant_and_reverse_set_up( self ):
        for name, set_up in _setups():
            with self.subTest( setup=name ), harness.shipped():
                set_up()
                bu_config.invalidate_config_caches()
                _check_view( self, name )

    def test_only_keys_with_no_default_setting_come_from_the_code( self ):
        # A key whose fallback is a default_* setting must name it, so a
        # user sees which global value a label inherits.
        no_default = { 'track_max_gap', 'match_distance_multiplier', 'paired_style' }
        with harness.shipped():
            bu_config.invalidate_config_caches()
            builtin = { key for view in bu_config.label_views() for key, field in view['fields'].items()
                        if field['source'] == bu_detector.SOURCE_BUILTIN }
            area = bu_config.label_view( 'exposed_anus' )['width_area_safety']
        self.assertLessEqual( builtin, no_default )
        if area['source'].startswith( 'default_' ):
            self.assertEqual( 'default_area_safety', area['source'] )
        self.assertEqual( 'default_area_safety', area['chain'][0]['source'] )

    def test_reading_the_view_leaves_the_live_config_as_it_was( self ):
        with harness.shipped():
            before = harness.snapshot()
            bu_config.label_views()
            self.assertEqual( before, harness.snapshot() )


class TestEachReasonIsTheRealOne( unittest.TestCase ):

    def field( self, label, key ):
        bu_config.invalidate_config_caches()
        return bu_config.label_view( label )[key]

    def layer( self, label, key, source ):
        found = [ layer for layer in self.field( label, key )['chain'] if layer['source'] == source ]
        self.assertEqual( 1, len( found ), source )
        return found[0]

    def test_a_value_for_another_backend_does_not_apply( self ):
        label = 'exposed_anus'
        with harness.shipped():
            other = [ name for name in bu_detector.registered_backend_names() if name != BACKEND ][0]
            betaconfig.detector_backend['selected'] = BACKEND
            betaconfig.detector_backend[other].setdefault( 'item_overrides', {} )[label] = { 'min_prob': 0.77 }
            layer = self.layer( label, 'min_prob', 'detector_backend.%s.item_overrides.%s.min_prob'%( other, label ) )
        self.assertEqual( bu_config.INACTIVE_BACKEND, layer['inactive'] )

    def test_a_value_for_another_variant_does_not_apply_and_the_selected_one_wins( self ):
        label = 'exposed_anus'
        variants = bu_detector.variant_names( BACKEND )
        if len( variants ) < 2:
            self.skipTest( 'one variant' )
        with harness.shipped():
            betaconfig.detector_backend['selected'] = BACKEND
            betaconfig.detector_backend[BACKEND]['model_variant'] = variants[0]
            blocks = betaconfig.detector_backend[BACKEND].setdefault( 'variants', {} )
            for index, variant in enumerate( variants[:2] ):
                blocks.setdefault( variant, {} ).setdefault( 'item_overrides', {} )[label] = { 'min_prob': 0.5 + index / 10 }
            field = self.field( label, 'min_prob' )
        by_source = { layer['source']: layer for layer in field['chain'] }
        own = 'detector_backend.%s.variants.%s.item_overrides.%s.min_prob'%( BACKEND, variants[0], label )
        other = 'detector_backend.%s.variants.%s.item_overrides.%s.min_prob'%( BACKEND, variants[1], label )
        self.assertTrue( by_source[own].get( 'wins' ) )
        self.assertEqual( 0.5, field['value'] )
        self.assertEqual( bu_config.INACTIVE_VARIANT, by_source[other]['inactive'] )

    def test_a_backend_value_is_overridden_by_the_variant_value( self ):
        label = 'exposed_anus'
        variant = bu_detector.variant_names( BACKEND )[0]
        with harness.shipped():
            betaconfig.detector_backend['selected'] = BACKEND
            betaconfig.detector_backend[BACKEND]['model_variant'] = variant
            betaconfig.detector_backend[BACKEND].setdefault( 'item_overrides', {} )[label] = { 'time_safety': 0.11 }
            betaconfig.detector_backend[BACKEND].setdefault( 'variants', {} ).setdefault(
                variant, {} ).setdefault( 'item_overrides', {} )[label] = { 'time_safety': 0.22 }
            field = self.field( label, 'time_safety' )
        backend_path = 'detector_backend.%s.item_overrides.%s.time_safety'%( BACKEND, label )
        variant_path = 'detector_backend.%s.variants.%s.item_overrides.%s.time_safety'%( BACKEND, variant, label )
        applying = [ ( layer['source'], layer.get( 'wins', False ) ) for layer in field['chain']
                     if 'inactive' not in layer ]
        self.assertEqual( [ ( 'default_time_safety', False ), ( backend_path, False ), ( variant_path, True ) ],
                          applying )
        self.assertEqual( 0.22, field['value'] )

    def test_a_look_key_at_a_backend_tier_is_not_tunable_there( self ):
        label = 'exposed_anus'
        with harness.shipped():
            betaconfig.detector_backend['selected'] = BACKEND
            betaconfig.detector_backend[BACKEND].setdefault( 'item_overrides', {} )[label] = { 'censor_shape': 'ellipse' }
            layer = self.layer( label, 'censor_shape',
                                'detector_backend.%s.item_overrides.%s.censor_shape'%( BACKEND, label ) )
        self.assertNotIn( 'censor_shape', bu_detector.ITEM_OVERRIDE_BACKEND_TUNABLE_KEYS )
        self.assertEqual( bu_config.INACTIVE_NOT_TUNABLE, layer['inactive'] )

    def test_a_window_drops_its_normal_look_and_shows_the_original( self ):
        with harness.shipped():
            betaconfig.item_overrides[WINDOW_LABEL] = { 'censor_shape': 'ellipse',
                                                        'censor_style': { 'type': 'pixelate' } }
            betaconfig.reverse['enabled'] = True
            betaconfig.reverse['windows'][WINDOW_LABEL] = {}
            field = self.field( WINDOW_LABEL, 'censor_style' )
            shape = self.field( WINDOW_LABEL, 'censor_shape' )
            look = self.layer( WINDOW_LABEL, 'censor_style', 'item_overrides.%s.censor_style'%( WINDOW_LABEL, ) )
        self.assertEqual( bu_config.INACTIVE_WINDOW_LOOK, look['inactive'] )
        self.assertEqual( bu_detector.SOURCE_BUILTIN, field['source'] )
        self.assertEqual( { 'type': 'show' }, field['value'] )
        # The stripped shape falls back to the global default, not to the dropped value.
        self.assertEqual( ( 'default_censor_shape', betaconfig.default_censor_shape ),
                          ( shape['source'], shape['value'] ) )

    def test_reverse_values_do_not_apply_while_reverse_mode_is_off( self ):
        with harness.shipped():
            betaconfig.reverse['enabled'] = False
            betaconfig.reverse['windows'][WINDOW_LABEL] = { 'min_prob': 0.41 }
            layer = self.layer( WINDOW_LABEL, 'min_prob', 'reverse.windows.%s.min_prob'%( WINDOW_LABEL, ) )
        self.assertEqual( bu_config.INACTIVE_REVERSE, layer['inactive'] )

    def test_a_reverse_censor_override_wins_while_on_and_not_for_a_window( self ):
        with harness.shipped():
            betaconfig.reverse['enabled'] = True
            betaconfig.reverse['censor']['labels'] = [ REVERSE_CENSORED ]
            betaconfig.reverse['censor']['overrides'][REVERSE_CENSORED] = { 'time_safety': 0.9 }
            betaconfig.reverse['censor']['overrides'][WINDOW_LABEL] = { 'time_safety': 0.8 }
            betaconfig.reverse['windows'][WINDOW_LABEL] = {}
            censored = self.field( REVERSE_CENSORED, 'time_safety' )
            window = self.layer( WINDOW_LABEL, 'time_safety',
                                 'reverse.censor.overrides.%s.time_safety'%( WINDOW_LABEL, ) )
        self.assertEqual( 'reverse.censor.overrides.%s.time_safety'%( REVERSE_CENSORED, ), censored['source'] )
        self.assertEqual( 0.9, censored['value'] )
        self.assertEqual( bu_config.INACTIVE_IS_WINDOW, window['inactive'] )

    def test_a_profile_value_applies_per_file_on_top( self ):
        label = 'exposed_anus'
        with harness.shipped():
            betaconfig.detector_backend['selected'] = BACKEND
            profiles = betaconfig.detector_backend[BACKEND].setdefault( 'profiles', {} ).setdefault( 'variants', {} )
            profile = sorted( profiles )[0] if profiles else 'any'
            profiles.setdefault( profile, {} ).setdefault( 'item_overrides', {} )[label] = { 'time_safety': 0.05 }
            field = self.field( label, 'time_safety' )
        last = field['chain'][-1]
        self.assertEqual( ( bu_config.PER_PROFILE, profile, 0.05 ),
                          ( last['inactive'], last['profile'], last['value'] ) )
        self.assertNotEqual( last['source'], field['source'] )


class TestOff( unittest.TestCase ):

    def test_paired_style_values_are_off_until_the_label_pairs_its_style( self ):
        label = 'exposed_breast'
        with harness.shipped():
            betaconfig.detector_backend['selected'] = BACKEND
            overrides = betaconfig.detector_backend[BACKEND].setdefault( 'item_overrides', {} )
            overrides.setdefault( label, {} )['paired_style'] = False
            bu_config.invalidate_config_caches()
            off = bu_config.label_view( label )
            overrides[label]['paired_style'] = True
            bu_config.invalidate_config_caches()
            on = bu_config.label_view( label )
        for key in bu_settings.LABEL_REQUIRES:
            self.assertEqual( 'paired_style', off[key].get( 'off' ), key )
            self.assertNotIn( 'off', on[key], key )
        self.assertNotIn( 'off', off['interpolation_max_gap'] )


if __name__ == '__main__':
    unittest.main()
