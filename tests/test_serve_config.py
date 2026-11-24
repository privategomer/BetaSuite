"""
test_serve_config.py - config.schema, config.get, config.set and
config.reset over the engine protocol.

What the app needs to be able to trust:
  - a value set over the contract is what the command-line tools read
    on their next run (the same file, the same merge, the same checks)
  - a change that would fail validation writes nothing and says why in
    codes, and a dry run writes nothing at all
  - "this re-renders" comes from the keys themselves
  - precedence is default < local < per-run

Through a real engine process, so the framing, the schemas and the live
config's reload are all part of what is tested.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import betaconfig
import betautils_cli as bu_cli
import betautils_config_store as bu_store
import betautils_local as bu_local
import betautils_serve_config as bu_serve_config
import betautils_protocol as bu_proto

from tests import config_harness as harness
from tests import protocol_schemas as schemas
from tests.test_serve import REPO, Engine, _EngineTest


def read_setting( local_path, expression ):
    """What a fresh process (the CLI's view) reads for a betaconfig expression."""
    env = dict( os.environ, BETASUITE_LOCAL_CONFIG=local_path )
    out = subprocess.run( [ sys.executable, '-c',
                            'import json, betaconfig; print( json.dumps( %s ) )'%( expression, ) ],
                          cwd=REPO, env=env, capture_output=True, text=True, timeout=60, check=True )
    return json.loads( out.stdout.strip().splitlines()[-1] )


class _ConfigEngineTest( _EngineTest ):

    def setUp( self ):
        super().setUp()
        self.local = os.path.join( self.tmp, 'betaconfig_local.py' )
        saved = os.environ.get( bu_local.ENV_VAR )
        os.environ[bu_local.ENV_VAR] = self.local
        self.addCleanup( lambda: os.environ.__setitem__( bu_local.ENV_VAR, saved ) if saved is not None
                         else os.environ.pop( bu_local.ENV_VAR, None ) )

    def engine( self ):
        engine = self.start()
        self.hello( engine )
        return engine

    def text( self ):
        if not os.path.exists( self.local ):
            return None
        with open( self.local, encoding='utf-8' ) as handle:
            return handle.read()


class TestWhatIsSetIsWhatTheCliReads( _ConfigEngineTest ):

    def test_set_then_a_fresh_process_reads_it_and_reset_puts_it_back( self ):
        engine = self.engine()
        result = engine.request( 'config.set', { 'values': { 'encode_crf': 21 } } )
        self.assertTrue( result['ok'], result )
        self.assertEqual( { 'moves': [ 'encode' ], 'written': True, 'dry_run': False }, result['data'] )
        self.assertEqual( 21, read_setting( self.local, 'betaconfig.encode_crf' ) )

        result = engine.request( 'config.reset', { 'paths': [ 'encode_crf' ] } )
        self.assertTrue( result['ok'], result )
        self.assertEqual( [ 'encode' ], result['data']['moves'] )
        self.assertEqual( 17, read_setting( self.local, 'betaconfig.encode_crf' ) )

    def test_a_key_inside_a_dict_setting_is_set_alone( self ):
        engine = self.engine()
        result = engine.request( 'config.set', { 'values': { 'detector_backend.nudenet_v3.nms_iou': 0.5 } } )
        self.assertTrue( result['ok'], result )
        self.assertEqual( [ 'detection' ], result['data']['moves'] )
        backend = read_setting( self.local, "betaconfig.detector_backend['nudenet_v3']" )
        self.assertEqual( 0.5, backend['nms_iou'] )
        self.assertEqual( 0.15, backend['candidate_floor'] )   # the rest of the block is kept

    def test_the_engine_itself_reads_the_new_value_next( self ):
        engine = self.engine()
        engine.request( 'config.set', { 'values': { 'render_workers': 3 } } )
        result = engine.request( 'config.get', { 'paths': [ 'render_workers' ] } )
        self.assertEqual( [ { 'path': 'render_workers', 'value': 3, 'default': 0,
                              'source': 'local', 'changed': True } ], result['data']['settings'] )

    def test_the_cli_tool_and_the_engine_write_the_same_file_the_same_way( self ):
        tool = os.path.join( REPO, 'tools', 'setup', 'local_config.py' )
        # Its log goes to the test's folder, never to the real output/logs.
        subprocess.run( [ sys.executable, tool, '--log-file', os.path.join( self.tmp, 'local_config.log' ),
                          'set', 'console_palette=deuteranopia' ], cwd=REPO,
                        capture_output=True, timeout=60, check=True,
                        env=dict( os.environ, BETASUITE_LOCAL_CONFIG=self.local ) )
        engine = self.engine()
        engine.request( 'config.set', { 'values': { 'encode_crf': 20 } } )
        text = self.text()
        self.assertIn( "console_palette = 'deuteranopia'", text )
        self.assertIn( "encode_crf = 20", text )
        rows = { row['path']: row for row in engine.request( 'config.get', {} )['data']['settings'] }
        self.assertEqual( 'local', rows['console_palette']['source'] )
        self.assertEqual( 'local', rows['encode_crf']['source'] )
        self.assertEqual( 'default', rows['encode_preset']['source'] )


class TestNothingIsWrittenUnlessItPasses( _ConfigEngineTest ):

    def test_a_dry_run_writes_nothing_and_says_what_it_would_cost( self ):
        engine = self.engine()
        for values, moves in ( ( { 'encode_crf': 21 }, [ 'encode' ] ),
                               ( { 'video_censor_fps': 12 }, [ 'detection' ] ),
                               ( { 'default_censor_shape': 'ellipse' }, [ 'censor' ] ),
                               ( { 'render_workers': 2 }, [] ),
                               ( { 'detector_backend.selected': 'retinanet_v2' }, [ 'detection', 'censor' ] ) ):
            result = engine.request( 'config.set', { 'values': values, 'dry_run': True } )
            self.assertTrue( result['ok'], result )
            self.assertEqual( sorted( moves ), sorted( result['data']['moves'] ), values )
            self.assertFalse( result['data']['written'] )
        self.assertIsNone( self.text() )

    def test_a_bad_value_is_refused_with_codes_and_the_file_is_untouched( self ):
        with open( self.local, 'w', encoding='utf-8' ) as handle:
            handle.write( "encode_crf = 20\n" )
        engine = self.engine()
        result = engine.request( 'config.set', { 'values': { 'encode_crf': 99, 'render_chunk_container': 'avi' } } )
        self.assertError( result, 'config.invalid' )
        codes = sorted( problem['code'] for problem in result['error']['params']['problems'] )
        self.assertEqual( [ 'config.bad_choice', 'config.out_of_range' ], codes )
        self.assertEqual( "encode_crf = 20\n", self.text() )

    def test_a_cross_field_problem_is_a_rule_with_an_id( self ):
        engine = self.engine()
        result = engine.request( 'config.set', { 'values': {
            'detector_backend.nudenet_v3.item_overrides.exposed_breast.min_prob_continue': 0.5 } } )
        self.assertError( result, 'config.invalid' )
        problem, = result['error']['params']['problems']
        self.assertEqual( 'config.rule', problem['code'] )
        self.assertEqual( 'label.hysteresis_inverted', problem['params']['rule'] )
        self.assertIsNone( self.text() )

    def test_an_unknown_path_names_the_closest_real_one( self ):
        engine = self.engine()
        self.assertError( engine.request( 'config.set', { 'values': { 'render_worker': 2 } } ),
                          'config.unknown_key', { 'path': 'render_worker', 'suggestion': 'render_workers' } )
        self.assertError( engine.request( 'config.set', { 'values': { 'qqq': 2 } } ),
                          'config.unknown_key', { 'path': 'qqq', 'suggestion': None } )

    def test_a_value_inside_a_whole_value_is_not_settable( self ):
        engine = self.engine()
        self.assertError( engine.request( 'config.set', { 'values': {
            'item_overrides.exposed_breast.censor_style.0.strength': 10 } } ),
            'config.not_settable', { 'path': 'item_overrides.exposed_breast.censor_style.0.strength',
                                     'setting': 'item_overrides.exposed_breast.censor_style' } )

    def test_malformed_requests_are_bad_params( self ):
        engine = self.engine()
        for name, data, field in ( ( 'config.set', {}, 'values' ), ( 'config.set', { 'values': [] }, 'values' ),
                                   ( 'config.set', { 'values': {} }, 'values' ),
                                   ( 'config.set', { 'values': { 'encode_crf': 1 }, 'dry_run': 'yes' }, 'dry_run' ),
                                   ( 'config.reset', {}, 'paths' ), ( 'config.reset', { 'paths': [] }, 'paths' ),
                                   ( 'config.reset', { 'paths': [ 3 ] }, 'paths' ),
                                   ( 'config.get', { 'paths': 'encode_crf' }, 'paths' ),
                                   ( 'config.get', { 'overrides': [] }, 'overrides' ) ):
            result = engine.request( name, data )
            self.assertError( result, 'command.bad_params' )
            self.assertEqual( field, result['error']['params']['field'], ( name, data ) )

    def test_an_unparsable_local_file_is_not_edited( self ):
        with open( self.local, 'w', encoding='utf-8' ) as handle:
            handle.write( "encode_crf = (\n" )
        engine = self.engine()
        self.assertError( engine.request( 'config.set', { 'values': { 'encode_crf': 20 } } ),
                          'config.local_unparsable', {} )
        self.assertEqual( "encode_crf = (\n", self.text() )

    def test_resetting_what_the_file_does_not_set_changes_nothing( self ):
        engine = self.engine()
        result = engine.request( 'config.reset', { 'paths': [ 'encode_crf' ] } )
        self.assertTrue( result['ok'], result )
        self.assertEqual( { 'moves': [], 'written': False, 'dry_run': False }, result['data'] )

    def test_with_the_local_layer_off_nothing_can_be_written( self ):
        os.environ[bu_local.ENV_VAR] = 'none'
        engine = self.engine()
        self.assertError( engine.request( 'config.set', { 'values': { 'encode_crf': 20 } } ),
                          'config.local_off', {} )


class TestSchemaAndGet( _ConfigEngineTest ):

    def test_the_schema_lists_every_entry_and_fits_one_line( self ):
        import betautils_settings as bu_settings
        engine = self.engine()
        result = engine.request( 'config.schema', {} )
        self.assertTrue( result['ok'] )
        self.assertEqual( len( bu_settings.REGISTRY ), len( result['data']['settings'] ) )
        line = bu_proto.encode( result )
        self.assertLess( len( line ), bu_proto.MAX_LINE_BYTES // 4 )

    def test_get_reports_default_local_and_run_sources( self ):
        with open( self.local, 'w', encoding='utf-8' ) as handle:
            handle.write( "encode_crf = 20\nrender_workers = 3\n" )
        engine = self.engine()
        result = engine.request( 'config.get', { 'paths': [ 'encode_crf', 'render_workers', 'encode_preset' ],
                                                 'overrides': { 'render_workers': 5 } } )
        rows = { row['path']: ( row['value'], row['source'] ) for row in result['data']['settings'] }
        self.assertEqual( { 'encode_crf': ( 20, 'local' ), 'render_workers': ( 5, 'run' ),
                            'encode_preset': ( 'fast', 'default' ) }, rows )
        # Overrides are never written.
        self.assertEqual( "encode_crf = 20\nrender_workers = 3\n", self.text() )

    def test_a_bad_override_is_invalid_and_an_unknown_one_unknown( self ):
        engine = self.engine()
        self.assertError( engine.request( 'config.get', { 'overrides': { 'encode_crf': 'x' } } ), 'config.invalid' )
        self.assertError( engine.request( 'config.get', { 'overrides': { 'encod_crf': 1 } } ), 'config.unknown_key' )

    def test_a_value_the_file_sets_whole_makes_everything_inside_it_local( self ):
        with open( self.local, 'w', encoding='utf-8' ) as handle:
            handle.write( "reverse = { 'windows': { 'face_femme': { 'censor_style': { 'type': 'show' } } } }\n" )
        engine = self.engine()
        rows = engine.request( 'config.get', { 'paths': [ 'reverse.windows' ] } )['data']['settings']
        self.assertEqual( [ ( 'reverse.windows.face_femme.censor_style', 'local' ) ],
                          [ ( row['path'], row['source'] ) for row in rows ] )

    def test_a_path_prefix_selects_a_block( self ):
        engine = self.engine()
        rows = engine.request( 'config.get', { 'paths': [ 'reverse.trigger' ] } )['data']['settings']
        self.assertEqual( [ 'reverse.trigger.hold_seconds', 'reverse.trigger.labels', 'reverse.trigger.min_prob' ],
                          sorted( row['path'] for row in rows ) )


class TestPrecedence( unittest.TestCase ):
    """default < local < per-run, through the store the engine and the CLI tool share."""

    def setUp( self ):
        self.tmp = tempfile.mkdtemp( prefix='betasuite-precedence-' )
        self.addCleanup( shutil.rmtree, self.tmp, ignore_errors=True )
        self.local = os.path.join( self.tmp, 'betaconfig_local.py' )

    def evaluate( self, text, overrides=None, flags=() ):
        def apply_flags( config ):
            if not flags:
                return []
            parser = bu_cli.build_arg_parser( 'test', include_preview=True, include_logging=True )
            return bu_cli.apply_cli_overrides( config, parser.parse_args( list( flags ) ) )
        with harness.shipped():
            store = bu_store.ConfigStore( path=self.local )
            return store.evaluate( text, overrides, extra=apply_flags )

    def test_each_layer_wins_over_the_one_below( self ):
        self.assertEqual( 0, self.evaluate( '' ).values['render_workers'] )
        self.assertEqual( 3, self.evaluate( 'render_workers = 3\n' ).values['render_workers'] )
        self.assertEqual( 5, self.evaluate( 'render_workers = 3\n',
                                            overrides={ ( 'render_workers', ): 5 } ).values['render_workers'] )
        self.assertEqual( 6, self.evaluate( 'render_workers = 3\n',
                                            flags=( '--render-workers', '6' ) ).values['render_workers'] )

    def test_a_lower_layer_shows_through_where_a_higher_one_is_silent( self ):
        evaluation = self.evaluate( 'render_workers = 3\n', overrides={ ( 'encode_crf', ): 25 } )
        self.assertEqual( ( 3, 25, 'fast' ), ( evaluation.values['render_workers'], evaluation.values['encode_crf'],
                                               evaluation.values['encode_preset'] ) )
        self.assertEqual( 3, evaluation.local_values['render_workers'] )
        self.assertEqual( 17, evaluation.local_values['encode_crf'] )

    def test_a_nested_override_keeps_the_rest_of_the_block( self ):
        evaluation = self.evaluate( "detector_backend = { 'nudenet_v3': { 'nms_iou': 0.5 } }\n",
                                    overrides={ ( 'detector_backend', 'nudenet_v3', 'candidate_floor' ): 0.2 } )
        block = evaluation.values['detector_backend']['nudenet_v3']
        self.assertEqual( ( 0.5, 0.2, 'per_class' ), ( block['nms_iou'], block['candidate_floor'], block['nms_mode'] ) )

    def test_evaluating_leaves_the_live_config_as_it_was( self ):
        with harness.shipped():
            before = harness.snapshot()
            store = bu_store.ConfigStore( path=self.local )
            store.evaluate( 'encode_crf = 30\n', { ( 'render_workers', ): 7 } )
            self.assertEqual( before, harness.snapshot() )

    def test_a_problem_betaconfig_py_already_has_is_not_the_local_files_fault( self ):
        with harness.shipped():
            store = bu_store.ConfigStore( path=self.local )
            store.base['encode_crf'] = 99   # as if the shipped value were out of range
            self.assertTrue( store.evaluate( 'render_workers = 2\n' ).ok )
            self.assertFalse( store.evaluate( 'render_workers = -2\n' ).ok )

    def test_the_baseline_is_betaconfig_py_without_the_local_layer( self ):
        with open( self.local, 'w', encoding='utf-8' ) as handle:
            handle.write( "encode_crf = 30\ndetector_backend = { 'nudenet_v3': { 'nms_iou': 0.5 } }\n" )
        with harness.shipped():
            shipped = harness.snapshot()
            bu_local.apply( vars( betaconfig ), path=self.local )
            self.assertEqual( 30, betaconfig.encode_crf )
            base = bu_store.baseline_values()
        self.assertEqual( shipped['encode_crf'], base['encode_crf'] )
        self.assertEqual( shipped['detector_backend'], base['detector_backend'] )


if __name__ == '__main__':
    unittest.main()
