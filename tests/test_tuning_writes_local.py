"""
test_tuning_writes_local.py - nothing the tuners write lands in
betaconfig.py. auto_tune's ConfigWriter writes the local config and
proves each value took; the A/B tools print a local_config.py command;
the bar smoothing test hands betatv.py a per-variant local config.
"""

import copy
import hashlib
import importlib.util
import os
import re
import shutil
import tempfile
import unittest

import betaconfig
import betautils_config as bu_config
import betautils_detector as bu_detector
import betautils_local as bu_local
import betautils_tuning as bu_tuning

REPO = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
SHIPPED = os.path.join( REPO, 'betaconfig.py' )


def _digest( path ):
    with open( path, 'rb' ) as handle:
        return hashlib.sha256( handle.read() ).hexdigest()


class WriterCase( unittest.TestCase ):

    NAMES = ( 'detector_backend', 'render_chunk_seconds', 'items_to_censor' )
    MINE = "console_palette = 'deuteranopia'  # mine\n"

    def setUp( self ):
        self.saved = { name: copy.deepcopy( getattr( betaconfig, name ) ) for name in self.NAMES }
        self.saved_loaded = bu_local.LOADED
        self.addCleanup( self._restore )
        self.dir = tempfile.mkdtemp( prefix='betasuite-writer-' )
        self.addCleanup( shutil.rmtree, self.dir, True )
        self.path = os.path.join( self.dir, 'betaconfig_local.py' )
        with open( self.path, 'w', encoding='utf-8' ) as handle:
            handle.write( bu_local.template_text() + '\n' + self.MINE )
        self.shipped = _digest( SHIPPED )
        self.backup_dir = os.path.join( self.dir, 'run' )
        self.writer = bu_tuning.ConfigWriter( local_path=self.path, backup_dir=self.backup_dir,
                                              comment='set by auto_tune run_x' )
        self.backend = 'nudenet_v3'

    def _restore( self ):
        for name, value in self.saved.items():
            setattr( betaconfig, name, value )
        bu_local.LOADED = self.saved_loaded
        bu_config.invalidate_config_caches()

    def tearDown( self ):
        self.assertEqual( _digest( SHIPPED ), self.shipped, "a tuner wrote betaconfig.py" )

    def settings( self ):
        entries, problems = bu_local.parse( self.writer.read() )
        self.assertEqual( problems, [] )
        return { name: value for name, value, _line in entries }


class TestConfigWriter( WriterCase ):

    def test_a_value_lands_in_the_local_config_and_the_running_config( self ):
        self.writer.set_item_override( self.backend, 'exposed_breast', 'track_max_gap', 27.0 )
        self.assertEqual( self.settings()['detector_backend'],
                          { self.backend: { 'item_overrides': { 'exposed_breast': { 'track_max_gap': 27.0 } } } } )
        self.assertEqual( bu_detector.get_item_overrides( 'exposed_breast', self.backend )['track_max_gap'], 27.0,
                          "later stages must measure the value just written" )
        text = self.writer.read()
        self.assertIn( self.MINE, text, "the person's own line and comment stay" )
        self.assertIn( 'set by auto_tune run_x', text, "a value nobody typed says where it came from" )

    def test_a_second_decision_replaces_the_first_and_keeps_a_backup( self ):
        self.writer.set_item_override( self.backend, 'exposed_breast', 'min_track_hits', 2 )
        self.writer.set_item_override( self.backend, 'exposed_breast', 'min_track_hits', 3 )
        self.assertEqual( self.writer.read().count( "'min_track_hits'" ), 1 )
        self.assertEqual( self.settings()['detector_backend'][self.backend]['item_overrides']['exposed_breast'],
                          { 'min_track_hits': 3 } )
        self.assertEqual( len( self.writer.backups ), 2 )
        with open( self.writer.backups[1], encoding='utf-8' ) as handle:
            self.assertIn( "'min_track_hits': 2", handle.read() )

    def test_each_tier_goes_to_its_own_path( self ):
        variant = bu_detector.selected_variant_name( self.backend )
        self.writer.set_variant_item_override( self.backend, '640m' if variant != '640m' else '320n',
                                               'exposed_vulva', 'track_max_gap', 6.0 )
        self.writer.set_profile_item_override( self.backend, 'quick_cut', 'exposed_breast', 'track_max_gap', 0.3 )
        self.writer.set_module_scalar( 'render_chunk_seconds', 120 )
        written = self.settings()
        block = written['detector_backend'][self.backend]
        self.assertEqual( list( block['variants'].values() ),
                          [ { 'item_overrides': { 'exposed_vulva': { 'track_max_gap': 6.0 } } } ] )
        self.assertEqual( block['profiles'], { 'variants': { 'quick_cut': { 'item_overrides': {
            'exposed_breast': { 'track_max_gap': 0.3 } } } } } )
        self.assertEqual( written['render_chunk_seconds'], 120 )
        self.assertEqual( bu_detector.get_profile_item_overrides( 'quick_cut', 'exposed_breast', self.backend )
                          ['track_max_gap'], 0.3 )
        # The rest of quick_cut's shipped block still applies under the merge.
        self.assertEqual( bu_detector.get_profiles( self.backend )['variants']['quick_cut']['video_censor_fps'],
                          self.saved['detector_backend'][self.backend]['profiles']['variants']['quick_cut']
                          ['video_censor_fps'] )

    def test_what_names_nothing_is_refused_and_nothing_is_written( self ):
        before = self.writer.read()
        for attempt in ( lambda: self.writer.set_item_override( 'no_such_backend', 'exposed_breast', 'track_max_gap', 5 ),
                         lambda: self.writer.set_item_override( self.backend, 'no_such_label', 'track_max_gap', 5 ),
                         lambda: self.writer.set_profile_item_override( self.backend, 'no_such', 'exposed_breast',
                                                                        'track_max_gap', 1 ),
                         lambda: self.writer.set_module_scalar( 'no_such_setting', 1 ) ):
            with self.assertRaises( bu_tuning.ConfigWriteError ):
                attempt()
        self.assertEqual( self.writer.read(), before )
        with self.assertRaises( bu_tuning.ConfigWriteError ) as caught:
            self.writer.set_item_override( self.backend, 'no_such_label', 'track_max_gap', 5 )
        self.assertIn( "'no_such_label' is not a label any detector knows", str( caught.exception ) )

    def test_a_value_validation_rejects_puts_everything_back( self ):
        before = self.writer.read()
        running = copy.deepcopy( betaconfig.detector_backend )
        with self.assertRaises( bu_tuning.ConfigWriteError ) as caught:
            self.writer.set_item_override( self.backend, 'exposed_breast', 'track_max_gap', -1.0 )
        self.assertIn( 'fails validation', str( caught.exception ) )
        self.assertEqual( self.writer.read(), before )
        self.assertEqual( betaconfig.detector_backend, running )

    def test_a_value_the_resolver_does_not_see_puts_everything_back( self ):
        # A model-variant value for the running variant wins over the
        # backend tier, so a backend write would read back the variant's.
        variant = bu_detector.selected_variant_name( self.backend )
        self.writer.set_variant_item_override( self.backend, variant, 'exposed_breast', 'min_track_hits', 4 )
        before = self.writer.read()
        with self.assertRaises( bu_tuning.ConfigWriteError ) as caught:
            self.writer.set_item_override( self.backend, 'exposed_breast', 'min_track_hits', 2 )
        self.assertIn( 'resolver reports 4', str( caught.exception ) )
        self.assertEqual( self.writer.read(), before )
        self.assertNotIn( 'min_track_hits', betaconfig.detector_backend[self.backend]['item_overrides']
                          .get( 'exposed_breast', {} ) if self.saved['detector_backend'][self.backend]
                          ['item_overrides'].get( 'exposed_breast', {} ).get( 'min_track_hits' ) is None else {} )

    def test_a_missing_file_is_created_and_an_unparseable_one_refused( self ):
        os.remove( self.path )
        self.writer.set_module_scalar( 'render_chunk_seconds', 90 )
        self.assertTrue( self.writer.read().startswith( bu_local.template_text() ) )
        with open( self.path, 'w', encoding='utf-8' ) as handle:
            handle.write( "render_chunk_seconds = (\n" )
        with self.assertRaises( bu_tuning.ConfigWriteError ):
            self.writer.set_module_scalar( 'render_chunk_seconds', 60 )

    def test_with_the_layer_off_there_is_nowhere_to_write( self ):
        writer = bu_tuning.ConfigWriter( local_path=None )
        writer.local_path = None
        with self.assertRaises( bu_tuning.ConfigWriteError ) as caught:
            writer.set_module_scalar( 'render_chunk_seconds', 90 )
        self.assertIn( 'turned off', str( caught.exception ) )

    def test_by_default_it_writes_the_configured_local_config( self ):
        # tests/__init__.py turns the layer off, so the default is None here.
        self.assertIsNone( bu_tuning.ConfigWriter().local_path )
        self.assertEqual( bu_tuning.ConfigWriter( local_path=self.path ).local_path, self.path )


class TestNoToolWritesTheShippedConfig( unittest.TestCase ):

    def test_no_tool_builds_a_path_to_betaconfig_py( self ):
        # The regex writer and the bar smoothing test both rewrote it, each
        # from an os.path.join( ..., 'betaconfig.py' ). Code reads
        # betaconfig by importing it; only a writer needs its path.
        pattern = re.compile( r"join\([^)]*'betaconfig\.py'|BETACONFIG_PATH" )
        offenders = []
        for root, dirs, files in os.walk( REPO ):
            dirs[:] = [ name for name in dirs if name not in ( 'tests', '.venv', '__pycache__', '.git' ) ]
            for name in files:
                if not name.endswith( '.py' ):
                    continue
                path = os.path.join( root, name )
                with open( path, encoding='utf-8', errors='replace' ) as handle:
                    text = handle.read()
                if pattern.search( text ):
                    offenders.append( os.path.relpath( path, REPO ) )
        self.assertEqual( offenders, [] )


class TestSetCommand( unittest.TestCase ):

    def test_every_value_survives_the_shell_and_the_parser( self ):
        import shlex
        spec = importlib.util.spec_from_file_location(
            '_local_config_for_set_command', os.path.join( REPO, 'tools', 'setup', 'local_config.py' ) )
        tool = importlib.util.module_from_spec( spec )
        spec.loader.exec_module( tool )
        updates = [ ( ( 'render_workers', ), [ 1, 2 ] ),
                    ( ( 'console_palette', ), 'deuteranopia' ),
                    ( ( 'log_path', ), '../out dir/$HOME "x".log' ),
                    ( ( 'detector_backend', 'nudenet_v3', 'class_suppression', 'exposed_breast' ),
                      [ { 'suppressed_by': 'covered_breast', 'margin': 0.08, 'min_iou': 0.4 } ] ),
                    ( ( 'shot_cut_burst_max_span', ), None ) ]
        words = shlex.split( bu_local.set_command( updates ) )
        self.assertEqual( words[:3], [ 'python', 'tools/setup/local_config.py', 'set' ] )
        self.assertEqual( [ tool.parse_assignment( word ) for word in words[3:] ], updates )


class TestBarSmoothingVariants( unittest.TestCase ):

    def setUp( self ):
        spec = importlib.util.spec_from_file_location(
            '_bar_test_under_test', os.path.join( REPO, 'tools', 'tuning', 'run_bar_smoothing_test.py' ) )
        self.tool = importlib.util.module_from_spec( spec )
        spec.loader.exec_module( self.tool )

    def test_a_variant_is_the_persons_config_plus_its_values( self ):
        base = bu_local.template_text() + "\nconsole_palette = 'deuteranopia'  # mine\n"
        style = [ { 'type': 'bar', 'weight': 1.0 } ]
        text = self.tool.build_variant_text( base, { 'name': 'v1', 'censor_style': style,
                                                     'position_smoothing': 0.4 } )
        entries, problems = bu_local.parse( text )
        self.assertEqual( problems, [] )
        values = { name: value for name, value, _line in entries }
        self.assertEqual( values['console_palette'], 'deuteranopia' )
        self.assertEqual( values['item_overrides'], { 'exposed_breast': { 'censor_style': style } } )
        self.assertEqual( values['default_position_smoothing'], 0.4 )
        self.assertIn( '# mine', text )

    def test_betatv_gets_the_variant_through_the_environment( self ):
        self.assertEqual( self.tool._variant_environment( '/x/v1.py' )[ bu_local.ENV_VAR ], '/x/v1.py' )


if __name__ == '__main__':
    unittest.main()
