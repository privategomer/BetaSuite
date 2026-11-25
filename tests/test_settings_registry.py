"""
test_settings_registry.py - the settings registry covers every setting,
and every entry has its words.

The app builds its settings pages, tips, help links and "this change
re-renders" warnings from betautils_settings.REGISTRY. A setting missing
from it cannot be shown or set; an entry with no tip shows a raw key;
a help anchor that names no heading opens nothing. Each of those is
checked here, both ways: config against registry, registry against
config, and both against CONFIG_REFERENCE.md and the catalog.

The cache-key tags are proven in test_settings_cache_tags.py.
"""

import json
import os
import re
import unittest

import betaconfig
import betautils_cli as bu_cli
import betautils_settings as bu_settings

from tests import config_harness as harness

REPO = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
CATALOG_PATH = os.path.join( REPO, 'i18n', 'en-US', 'settings.json' )
REFERENCE_PATH = os.path.join( REPO, 'CONFIG_REFERENCE.md' )

TIP_MAX_CHARS = 120


def _reference():
    with open( REFERENCE_PATH, 'r', encoding='utf-8' ) as handle:
        return handle.read()


def _catalog():
    with open( CATALOG_PATH, 'r', encoding='utf-8' ) as handle:
        return json.load( handle )


def _shipped_leaves():
    # The shipped config, not whatever earlier tests in the suite left behind.
    with harness.shipped():
        return list( bu_settings.concrete_leaves( vars( betaconfig ) ) )


def _shipped_names():
    with harness.shipped():
        return harness.setting_names()


def _dotted( path ):
    return '.'.join( str( part ) for part in path )


def _matching( path ):
    """Every entry a concrete path matches, structured parents included."""
    return [ entry for entry in bu_settings.REGISTRY
             if bu_settings._matches( entry.path, path )
             or ( entry.kind in bu_settings.STRUCTURED_KINDS and len( path ) > len( entry.path )
                  and bu_settings._matches( entry.path, path[:len( entry.path )] ) ) ]


class TestEverySettingIsRegistered( unittest.TestCase ):

    def test_every_shipped_leaf_has_an_entry( self ):
        missing = [ _dotted( path ) for path, _value, entry in _shipped_leaves() if entry is None ]
        self.assertEqual( [], missing )

    def test_every_shipped_leaf_matches_exactly_one_most_specific_entry( self ):
        ambiguous = []
        for path, _value, entry in _shipped_leaves():
            candidates = _matching( path )
            best = max( ( len( c.path ), bu_settings._specificity( c ) ) for c in candidates )
            if sum( 1 for c in candidates
                    if ( len( c.path ), bu_settings._specificity( c ) ) == best ) != 1:
                ambiguous.append( _dotted( path ) )
        self.assertEqual( [], ambiguous )

    def test_every_shipped_value_is_within_its_entry( self ):
        problems = []
        for path, value, entry in _shipped_leaves():
            problems.extend( bu_settings.check_value( _dotted( path ), value, entry ) )
        self.assertEqual( [], problems )

    def test_every_top_level_setting_is_registered( self ):
        roots = { entry.path[0] for entry in bu_settings.REGISTRY }
        self.assertEqual( [], sorted( set( _shipped_names() ) - roots ) )

    def test_every_entry_is_set_by_the_shipped_config_or_has_a_valid_probe( self ):
        shipped = { entry.path for _path, _value, entry in _shipped_leaves() }
        for entry in bu_settings.REGISTRY:
            with self.subTest( entry=entry.dotted ):
                if entry.path in shipped:
                    continue
                self.assertIsNotNone( entry.probe, "not shipped and no probe to prove it is read" )
                self.assertEqual( [], bu_settings.check_value( entry.dotted, entry.probe_for(), entry ) )

    def test_every_probe_is_valid_for_its_entry( self ):
        for entry in bu_settings.REGISTRY:
            with self.subTest( entry=entry.dotted ):
                self.assertIsNotNone( entry.probe )
                self.assertEqual( [], bu_settings.check_value( entry.dotted, entry.probe_for(), entry ) )

    def test_no_two_entries_share_a_path( self ):
        paths = [ entry.path for entry in bu_settings.REGISTRY ]
        self.assertEqual( len( paths ), len( set( paths ) ) )

    def test_every_root_names_a_setting_betaconfig_has( self ):
        roots = { entry.path[0] for entry in bu_settings.REGISTRY }
        self.assertEqual( [], sorted( roots - set( _shipped_names() ) ) )


class TestEntriesAreWellFormed( unittest.TestCase ):

    def test_kinds_levels_tags_and_conditions_are_known( self ):
        for entry in bu_settings.REGISTRY:
            with self.subTest( entry=entry.dotted ):
                self.assertIn( entry.kind, bu_settings.KINDS )
                self.assertIn( entry.level, ( bu_settings.SIMPLE, bu_settings.ADVANCED ) )
                self.assertLessEqual( entry.moves, set( bu_settings.CACHE_KEYS ) )
                self.assertLessEqual( set( entry.when ), set( bu_settings.CONDITIONS ) )
                if entry.when:
                    self.assertTrue( entry.moves, "a condition on moving nothing means nothing" )

    def test_a_choice_has_choices_and_a_range_is_only_on_numbers( self ):
        for entry in bu_settings.REGISTRY:
            with self.subTest( entry=entry.dotted ):
                if entry.kind in ( 'choice', 'labels' ):
                    self.assertTrue( entry.choice_list() )
                if entry.minimum is not None or entry.maximum is not None:
                    self.assertIn( entry.kind, ( 'int', 'number' ) )

    def test_a_nullable_entry_says_what_none_means( self ):
        for entry in bu_settings.REGISTRY:
            with self.subTest( entry=entry.dotted ):
                self.assertEqual( bool( entry.nullable ), bool( entry.none_means ) )

    def test_every_cli_flag_is_a_real_betatv_flag( self ):
        parser = bu_cli.build_arg_parser( 'registry test', include_preview=True, include_logging=True )
        flags = { flag for action in parser._actions for flag in action.option_strings }
        for entry in bu_settings.REGISTRY:
            if entry.cli:
                with self.subTest( entry=entry.dotted ):
                    self.assertIn( entry.cli, flags )

    def test_the_schema_is_plain_json( self ):
        text = json.dumps( bu_settings.schema(), ensure_ascii=True, allow_nan=False )
        self.assertEqual( len( bu_settings.REGISTRY ), len( json.loads( text )['settings'] ) )


class TestScreens( unittest.TestCase ):
    """Every setting is shown somewhere: on a screen or its More options,
    or else on All Settings alone."""

    def test_every_entry_has_a_known_screen( self ):
        for entry in bu_settings.REGISTRY:
            self.assertIn( entry.screen, bu_settings.SCREENS, entry.dotted )

    def test_a_simple_setting_is_on_a_screen_not_only_in_all_settings( self ):
        for entry in bu_settings.REGISTRY:
            if entry.level == bu_settings.SIMPLE:
                self.assertNotEqual( bu_settings.SCREEN_NONE, entry.screen, entry.dotted )

    def test_every_screen_but_none_holds_a_simple_setting( self ):
        simple = { entry.screen for entry in bu_settings.REGISTRY if entry.level == bu_settings.SIMPLE }
        self.assertEqual( set( bu_settings.SCREENS ) - { bu_settings.SCREEN_NONE }, simple )

    def test_every_rule_but_the_catch_all_places_some_entry( self ):
        used = set()
        for entry in bu_settings.REGISTRY:
            for index, ( pattern, _screen ) in enumerate( bu_settings.SCREEN_RULES ):
                if bu_settings.fnmatch.fnmatchcase( entry.dotted, pattern ):
                    used.add( index )
                    break
        self.assertEqual( set( range( len( bu_settings.SCREEN_RULES ) ) ), used )

    def test_rules_place_these_settings_as_listed( self ):
        expect = {
            'encode_crf': 'run', 'video_censor_fps': 'run',
            'items_to_censor': 'what_to_hide', 'item_overrides.<label>.censor_style': 'what_to_hide',
            'default_time_safety': 'what_to_hide', 'detector_backend.selected': 'models',
            'person_backend.<person_backend>.model_variant': 'models', 'gpu_enabled': 'resources',
            'person_backend.<person_backend>.batch_size': 'resources',
            'person_backend.<person_backend>.min_prob': 'reverse', 'reverse.enabled': 'reverse',
            'move_completed_sources': 'library', 'default_min_prob': 'none',
            'detector_backend.<backend>.item_overrides.<label>.time_safety': 'none',
        }
        got = { entry.dotted: entry.screen for entry in bu_settings.REGISTRY if entry.dotted in expect }
        self.assertEqual( expect, got )

    def test_the_schema_sends_each_screen( self ):
        exported = bu_settings.schema()
        self.assertEqual( list( bu_settings.SCREENS ), exported['screens'] )
        self.assertEqual( [ entry.screen for entry in bu_settings.REGISTRY ],
                          [ item['screen'] for item in exported['settings'] ] )


class TestHelpAnchors( unittest.TestCase ):

    @classmethod
    def setUpClass( cls ):
        cls.reference = _reference()
        cls.anchors = set( bu_settings.heading_anchors( cls.reference ) )

    def test_every_help_anchor_names_a_heading( self ):
        missing = sorted( { entry.help for entry in bu_settings.REGISTRY } - self.anchors )
        self.assertEqual( [], missing )

    def test_every_setting_named_in_a_heading_is_registered( self ):
        names = set()
        for entry in bu_settings.REGISTRY:
            names.update( part for part in entry.path if part not in bu_settings.PLACEHOLDERS )
        names.update( bu_settings.STYLE_COMMON_FIELDS )
        for fields in bu_settings.STYLE_TYPE_FIELDS.values():
            names.update( fields )
        # Values a placeholder or a choice takes: 'nudenet_v3', 'triple_box'.
        names.update( bu_settings.backend_names() )
        names.update( bu_settings.label_names() )
        names.update( bu_settings.BLUR_METHODS )
        unknown = set()
        for line in self.reference.splitlines():
            if not line.startswith( '#' ):
                continue
            for token in re.findall( r'`([^`]+)`', line ):
                if re.fullmatch( r'[a-z][a-z0-9]*(_[a-z0-9]+)+', token ) and token not in names:
                    unknown.add( token )
        self.assertEqual( [], sorted( unknown ) )

    def test_every_setting_name_appears_in_the_reference( self ):
        named = set( re.findall( r'[a-z][a-z0-9_]*', self.reference ) )
        missing = sorted( { part for entry in bu_settings.REGISTRY for part in entry.path
                            if part not in bu_settings.PLACEHOLDERS and part not in named
                            and part not in ( 'defaults', 'variants', 'profiles', 'trigger', 'censor',
                                              'person_windows', 'item_overrides' ) } )
        self.assertEqual( [], missing )


class TestCatalog( unittest.TestCase ):

    @classmethod
    def setUpClass( cls ):
        cls.catalog = _catalog()

    def expected_keys( self ):
        keys = set()
        for family in bu_settings.families():
            keys.add( 'setting.%s.label'%( family, ) )
            keys.add( 'setting.%s.tip'%( family, ) )
        for entry in bu_settings.REGISTRY:
            if entry.none_means:
                keys.add( 'setting.%s.none'%( entry.family, ) )
        return keys

    def test_every_entry_has_a_tip_and_a_label( self ):
        for entry in bu_settings.REGISTRY:
            with self.subTest( entry=entry.dotted ):
                self.assertIn( entry.tip, self.catalog )
                self.assertIn( entry.label_key, self.catalog )

    def test_every_style_and_rule_field_has_a_tip( self ):
        for family in bu_settings.style_field_families():
            with self.subTest( family=family ):
                self.assertIn( 'setting.%s.tip'%( family, ), self.catalog )

    def test_the_catalog_has_no_orphans_and_no_gaps( self ):
        self.assertEqual( sorted( self.expected_keys() ), sorted( self.catalog ) )

    def test_every_message_is_one_short_plain_line( self ):
        for key, text in self.catalog.items():
            with self.subTest( key=key ):
                self.assertTrue( text.strip() )
                self.assertNotIn( '\n', text )
                self.assertNotIn( '—', text )   # no em dashes
                self.assertLessEqual( len( text ), TIP_MAX_CHARS )
                self.assertEqual( text.count( '{' ), text.count( '}' ),
                                  "unbalanced ICU braces" )
                self.assertNotIn( '  ', text )

    def test_keys_are_stable_dotted_names( self ):
        for key in self.catalog:
            self.assertRegex( key, r'^setting\.[a-z][a-z0-9_]*\.(label|tip|none)$' )


class TestLookup( unittest.TestCase ):

    def test_a_literal_segment_wins_over_a_placeholder( self ):
        entry = bu_settings.find( 'detector_backend.defaults.nn_batch_size' )
        self.assertEqual( ( 'detector_backend', 'defaults', 'nn_batch_size' ), entry.path )
        entry = bu_settings.find( 'detector_backend.nudenet_v3.nn_batch_size' )
        self.assertEqual( '<backend>', entry.path[1] )

    def test_the_most_specific_entry_wins_whatever_the_registry_order( self ):
        import unittest.mock as mock
        reordered = tuple( reversed( bu_settings.REGISTRY ) )
        bu_settings._find.cache_clear()
        try:
            with mock.patch.object( bu_settings, 'REGISTRY', reordered ):
                entry = bu_settings.find( 'detector_backend.defaults.nn_batch_size' )
                self.assertEqual( ( 'detector_backend', 'defaults', 'nn_batch_size' ), entry.path )
                entry = bu_settings.find( 'detector_backend.defaults.picture_sizes' )
                self.assertEqual( 'defaults', entry.path[1] )
        finally:
            bu_settings._find.cache_clear()

    def test_a_path_inside_a_structured_entry_finds_that_entry( self ):
        entry = bu_settings.find( 'item_overrides.exposed_breast.censor_style.0.strength' )
        self.assertEqual( 'censor_style', entry.kind )

    def test_an_unknown_path_finds_nothing_and_suggests_the_closest( self ):
        self.assertIsNone( bu_settings.find( 'render_worker' ) )
        self.assertEqual( 'render_workers', bu_settings.suggestion( 'render_worker' ) )
        self.assertEqual( 'detector_backend.nudenet_v3.nms_iou',
                          bu_settings.suggestion( 'detector_backend.nudenet_v3.nms_iuo' ) )

    def test_scope_comes_from_the_path( self ):
        self.assertEqual( 'global', bu_settings.find( 'encode_crf' ).scope )
        self.assertEqual( 'backend', bu_settings.find( 'detector_backend.nudenet_v3.nms_iou' ).scope )
        self.assertEqual( 'backend_label', bu_settings.find(
            'detector_backend.nudenet_v3.item_overrides.exposed_breast.min_prob' ).scope )
        self.assertEqual( 'profile_label', bu_settings.find(
            'detector_backend.nudenet_v3.profiles.variants.scene.item_overrides.exposed_breast.track_max_gap'
        ).scope )
        self.assertEqual( 'label', bu_settings.find( 'item_overrides.exposed_breast.censor_shape' ).scope )
        self.assertEqual( 'person_backend', bu_settings.find( 'person_backend.yolo11_seg.min_prob' ).scope )


class TestCheckValue( unittest.TestCase ):

    def codes( self, path, value ):
        entry = bu_settings.find( path )
        return [ problem.code for problem in bu_settings.check_value( path, value, entry ) ]

    def test_type_range_and_choice_have_their_own_codes( self ):
        self.assertEqual( [ 'config.bad_type' ], self.codes( 'encode_crf', 'x' ) )
        self.assertEqual( [ 'config.bad_type' ], self.codes( 'encode_crf', 17.5 ) )
        self.assertEqual( [ 'config.bad_type' ], self.codes( 'encode_crf', True ) )
        self.assertEqual( [ 'config.out_of_range' ], self.codes( 'encode_crf', 52 ) )
        self.assertEqual( [ 'config.out_of_range' ], self.codes( 'encode_crf', -1 ) )
        self.assertEqual( [], self.codes( 'encode_crf', 51 ) )
        self.assertEqual( [], self.codes( 'encode_crf', 0 ) )
        self.assertEqual( [ 'config.bad_choice' ], self.codes( 'render_chunk_container', 'avi' ) )
        self.assertEqual( [ 'config.bad_choice' ], self.codes( 'items_to_censor', [ 'nose' ] ) )

    def test_open_and_closed_bounds( self ):
        self.assertEqual( [ 'config.out_of_range' ], self.codes( 'global_min_prob', 1 ) )
        self.assertEqual( [], self.codes( 'global_min_prob', 0 ) )
        self.assertEqual( [ 'config.out_of_range' ], self.codes( 'video_censor_fps', 0 ) )
        self.assertEqual( [], self.codes( 'video_censor_fps', 0.5 ) )

    def test_none_only_where_it_means_something( self ):
        self.assertEqual( [], self.codes( 'render_decode_threads', None ) )
        self.assertEqual( [ 'config.bad_type' ], self.codes( 'render_workers', None ) )

    def test_odd_numbers( self ):
        self.assertEqual( [ 'config.out_of_range' ], self.codes( 'blur_approximation_min_kernel', 22 ) )
        self.assertEqual( [], self.codes( 'blur_approximation_min_kernel', 23 ) )

    def test_a_color_is_three_whole_numbers_from_0_to_255( self ):
        self.assertEqual( [], self.codes( 'vision_cursor_color', ( 0, 128, 255 ) ) )
        for bad in ( ( 0, 0, 256 ), ( -1, 0, 0 ), ( 0, 0 ), ( 0.5, 0, 0 ), 'red' ):
            self.assertEqual( [ 'config.bad_type' ], self.codes( 'vision_cursor_color', bad ), bad )

    def test_picture_sizes_are_a_non_empty_list_of_whole_numbers_from_0( self ):
        path = 'detector_backend.retinanet_v2.picture_sizes'
        self.assertEqual( [], self.codes( path, [ 0, 1280 ] ) )
        for bad in ( [ -32 ], [], [ 320.0 ], 320 ):
            self.assertEqual( [ 'config.bad_type' ], self.codes( path, bad ), bad )

    def test_a_string_or_path_must_not_be_empty( self ):
        self.assertEqual( [ 'config.bad_type' ], self.codes( 'encode_video_codec', '' ) )
        self.assertEqual( [ 'config.bad_type' ], self.codes( 'log_path', '' ) )
        self.assertEqual( [], self.codes( 'log_path', None ) )
        self.assertEqual( [], self.codes( 'encode_video_codec', 'libx264' ) )

    def test_gpu_enabled_takes_true_and_false_as_one_and_zero( self ):
        self.assertEqual( [], self.codes( 'gpu_enabled', True ) )
        self.assertEqual( [ 'config.bad_choice' ], self.codes( 'gpu_enabled', 2 ) )

    def test_a_problem_is_its_english_text_and_carries_code_and_params( self ):
        entry = bu_settings.find( 'encode_crf' )
        problem, = bu_settings.check_value( 'encode_crf', 99, entry )
        self.assertIsInstance( problem, str )
        self.assertIn( 'encode_crf', problem )
        self.assertEqual( { 'code': 'config.out_of_range',
                            'params': { 'path': 'encode_crf', 'minimum': 0, 'maximum': 51,
                                        'min_open': False, 'max_open': False, 'odd': False } },
                          problem.as_error() )


if __name__ == '__main__':
    unittest.main()
