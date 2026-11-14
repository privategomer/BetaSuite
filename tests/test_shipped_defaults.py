"""
test_shipped_defaults.py - what betaconfig.py ships: every setting the
code reads is listed there (so the local config can set it), the
shipped values work on a fresh install with either detector backend,
and listing a setting at the value the code already fell back to
changes no output and no cache key.
"""

import ast
import copy
import os
import re
import unittest
import unittest.mock

import betaconfig
import betautils_cache_paths as bu_cache
import betautils_config as bu_config
import betautils_track as bu_track

REPO = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )

# Read only to refuse them: validation names each as removed when it is
# present, so listing one would stop every run.
REMOVED_SETTINGS = { 'class_suppression', 'nn_batch_size', 'picture_sizes' }

READ_PATTERN = re.compile( r"(?:getattr|hasattr)\(\s*betaconfig\s*,\s*'([a-z][a-z_0-9]*)'" )


def _shipped_values():
    """betaconfig.py's own literal assignments, read from the source."""
    with open( os.path.join( REPO, 'betaconfig.py' ), encoding='utf-8' ) as handle:
        tree = ast.parse( handle.read() )
    return { node.targets[0].id: ast.literal_eval( node.value ) for node in tree.body
             if isinstance( node, ast.Assign ) and isinstance( node.targets[0], ast.Name ) }


def _names_read_with_a_fallback():
    """{ setting: [ files ] } for every getattr/hasattr on betaconfig outside the tests."""
    found = {}
    for root, dirs, files in os.walk( REPO ):
        dirs[:] = [ name for name in dirs if name not in ( 'tests', '__pycache__', '.venv', '.git' ) ]
        for name in files:
            if not name.endswith( '.py' ):
                continue
            path = os.path.join( root, name )
            with open( path, encoding='utf-8', errors='replace' ) as handle:
                text = handle.read()
            for match in READ_PATTERN.finditer( text ):
                found.setdefault( match.group( 1 ), [] ).append( os.path.relpath( path, REPO ) )
    return found


class TestEverySettingIsListed( unittest.TestCase ):

    def test_every_setting_the_code_reads_is_in_betaconfig( self ):
        # A setting read with a code fallback and absent from betaconfig.py
        # could not be set in betaconfig_local.py (an unknown name there is
        # an error) and was invisible to anyone reading the shipped file.
        shipped = _shipped_values()
        missing = { name: sorted( set( files ) ) for name, files in _names_read_with_a_fallback().items()
                    if name not in shipped and name not in REMOVED_SETTINGS }
        self.assertEqual( missing, {}, "read in code but not listed in betaconfig.py" )

    def test_the_removed_settings_are_still_refused( self ):
        # If one were no longer refused, it would belong in the check above.
        shipped = _shipped_values()
        for name in sorted( REMOVED_SETTINGS ):
            self.assertNotIn( name, shipped )
            setattr( betaconfig, name, 1 )
            try:
                errors = bu_config._collect_validation_errors()
            finally:
                delattr( betaconfig, name )
            self.assertTrue( any( name in error and 'no longer read' in error for error in errors ), name )


class TestFreshInstall( unittest.TestCase ):

    def setUp( self ):
        self.saved = copy.deepcopy( betaconfig.detector_backend )
        self.addCleanup( self._restore )

    def _restore( self ):
        betaconfig.detector_backend = self.saved
        bu_config.invalidate_config_caches()

    def test_the_shipped_config_validates_with_each_backend( self ):
        # retinanet_v2 had no profiles, so profile_by_source_dir named
        # four variants it lacked and `--backend retinanet_v2` stopped at
        # validation on a fresh install.
        for backend in ( 'nudenet_v3', 'retinanet_v2' ):
            betaconfig.detector_backend = copy.deepcopy( self.saved )
            betaconfig.detector_backend['selected'] = backend
            bu_config.invalidate_config_caches()
            self.assertEqual( bu_config._collect_validation_errors(), [], backend )

    def test_the_shipped_values_need_nothing_a_fresh_install_lacks( self ):
        shipped = _shipped_values()
        # setup downloads only 320n; 640m is a manual download.
        self.assertEqual( shipped['detector_backend']['nudenet_v3']['model_variant'], '320n' )
        # setup and tune_machine set 1 in the local config on a GPU install.
        self.assertEqual( shipped['gpu_enabled'], 0 )
        self.assertEqual( shipped['log_level'], 'debug' )
        # No sticker images ship, so a sticker style renders as a solid bar.
        stickers = [ label for label, override in shipped['item_overrides'].items()
                     for style in override.get( 'censor_style', [] ) if style['type'] == 'sticker' ]
        self.assertEqual( stickers, [] )
        # Taste, so the person's: nothing censored, nothing shown through.
        self.assertEqual( shipped['items_to_censor'], [] )
        self.assertEqual( shipped['reverse']['windows'], {} )
        self.assertIs( shipped['reverse']['person_windows']['enabled'], False )

    def test_no_line_says_a_tool_set_it( self ):
        # The tuners write provenance comments where they write values;
        # in the shipped file they describe one machine's run.
        with open( os.path.join( REPO, 'betaconfig.py' ), encoding='utf-8' ) as handle:
            marked = [ number for number, line in enumerate( handle, 1 )
                       if re.search( r'#.*\bset by\b', line ) ]
        self.assertEqual( marked, [] )

    def test_style_weights_still_sum_to_100( self ):
        for label, override in _shipped_values()['item_overrides'].items():
            total = sum( style['weight'] for style in override['censor_style'] )
            self.assertAlmostEqual( total, 100.0, places=6, msg=label )


class TestListingAtTheFallbackChangesNothing( unittest.TestCase ):
    """The settings betaconfig.py now lists at the value the code used to fall back to."""

    NAMES = ( 'default_confirm_score', 'default_size_smoothing', 'default_geometry_action' )

    def setUp( self ):
        self.saved = { name: copy.deepcopy( getattr( betaconfig, name ) ) for name in self.NAMES }
        self.addCleanup( self._restore )

    def _restore( self ):
        for name, value in self.saved.items():
            setattr( betaconfig, name, value )
        bu_config.invalidate_config_caches()

    def test_the_censor_key_is_the_one_from_before_they_were_listed( self ):
        # Every default_* name is keyed, so listing these three moved the
        # censor key and would have rendered every finished output again.
        listed = bu_cache.censor_key()
        for name in self.NAMES:
            delattr( betaconfig, name )
        bu_config.invalidate_config_caches()
        self.assertEqual( bu_cache.censor_key(), listed )

    def test_each_one_moves_the_censor_key_once_changed( self ):
        before = bu_cache.censor_key()
        for name, value in ( ( 'default_confirm_score', 0.6 ), ( 'default_size_smoothing', 0.1 ),
                             ( 'default_geometry_action', 'flag' ) ):
            setattr( betaconfig, name, value )
            bu_config.invalidate_config_caches()
            self.assertNotEqual( bu_cache.censor_key(), before, name )
            self._restore()
            self.assertEqual( bu_cache.censor_key(), before, "precondition: restoring restores the key" )

    def test_size_smoothing_none_follows_position_smoothing( self ):
        backend = betaconfig.detector_backend['selected']
        betaconfig.default_size_smoothing = None
        settings = bu_track._LabelSettings( 'exposed_breast', backend, 1.0/9 )
        self.assertAlmostEqual( settings.size_alpha, settings.alpha/3.0 )
        betaconfig.default_size_smoothing = 0.12
        settings = bu_track._LabelSettings( 'exposed_breast', backend, 1.0/9 )
        self.assertAlmostEqual( settings.size_alpha, 0.12 )

    def test_the_label_defaults_are_validated( self ):
        self.assertEqual( [ error for error in bu_config._collect_validation_errors()
                            if any( name in error for name in self.NAMES ) ], [] )
        for name, value in ( ( 'default_geometry_action', 'drpo' ),
                             ( 'default_geometry_action', { 'min': 'drop', 'top': 'clamp' } ),
                             ( 'default_geometry_action', { 'max': 'shrink' } ),
                             ( 'default_confirm_score', 0 ), ( 'default_confirm_score', 1.5 ),
                             ( 'default_size_smoothing', 'fast' ), ( 'default_size_smoothing', True ),
                             ( 'default_size_smoothing', 0 ) ):
            setattr( betaconfig, name, value )
            errors = bu_config._collect_validation_errors()
            self._restore()
            self.assertTrue( any( error.startswith( name ) for error in errors ), ( name, value, errors ) )


class TestUnsafeSuppressionPairs( unittest.TestCase ):

    def setUp( self ):
        self.saved = copy.deepcopy( betaconfig.detector_backend )
        self.addCleanup( self._restore )

    def _restore( self ):
        betaconfig.detector_backend = self.saved
        bu_config.invalidate_config_caches()

    def test_they_are_built_in_and_the_shipped_list_is_for_a_persons_own( self ):
        import betautils_detector as bu_detector
        for backend, block in _shipped_values()['detector_backend'].items():
            if isinstance( block, dict ):
                self.assertEqual( block.get( 'class_suppression_refused', [] ), [], backend )
        self.assertEqual( sorted( bu_detector.UNSAFE_SUPPRESSION_PAIRS ), [
            'exposed_vulva<-exposed_buttocks', 'exposed_vulva<-exposed_feet',
            'exposed_vulva<-exposed_penis', 'exposed_vulva<-face_femme' ] )
        self.assertTrue( all( reason for reason in bu_detector.UNSAFE_SUPPRESSION_PAIRS.values() ) )
        betaconfig.detector_backend = copy.deepcopy( self.saved )
        betaconfig.detector_backend['nudenet_v3']['class_suppression_refused'] = [
            'exposed_breast<-exposed_chest', 'exposed_vulva<-face_femme' ]
        bu_config.invalidate_config_caches()
        self.assertEqual( bu_detector.get_class_suppression_refused( 'nudenet_v3' ),
                          list( bu_detector.UNSAFE_SUPPRESSION_PAIRS ) + [ 'exposed_breast<-exposed_chest' ] )
        for backend in bu_detector.registered_backend_names():
            self.assertTrue( set( bu_detector.UNSAFE_SUPPRESSION_PAIRS )
                             <= set( bu_detector.get_class_suppression_refused( backend ) ), backend )

    def test_a_hand_written_unsafe_rule_warns_and_a_safe_one_does_not( self ):
        unsafe = lambda: [ w for w in bu_config._collect_validation_warnings() if 'removes the censor' in w ]
        self.assertEqual( unsafe(), [] )
        betaconfig.detector_backend = copy.deepcopy( self.saved )
        betaconfig.detector_backend['nudenet_v3']['class_suppression']['exposed_vulva'].append(
            { 'suppressed_by': 'exposed_penis', 'margin': 0.1, 'min_iou': 0.5 } )
        bu_config.invalidate_config_caches()
        warnings = unsafe()
        self.assertEqual( len( warnings ), 1, warnings )
        self.assertIn( 'exposed_vulva<-exposed_penis', warnings[0] )


class TestValidatedLabelMaps( unittest.TestCase ):
    """betautils_local skips its new-key warning inside these maps; validation must cover them."""

    NAMES = ( 'detector_backend', 'item_overrides', 'reverse' )

    def setUp( self ):
        self.saved = { name: copy.deepcopy( getattr( betaconfig, name ) ) for name in self.NAMES }
        self.addCleanup( self._restore )

    def _restore( self ):
        for name, value in self.saved.items():
            setattr( betaconfig, name, copy.deepcopy( value ) )
        bu_config.invalidate_config_caches()

    def test_a_misspelled_key_in_each_map_is_a_validation_error( self ):
        import betautils_local as bu_local
        backend = lambda: betaconfig.detector_backend['nudenet_v3']
        cases = {
            'item_overrides': [
                lambda: backend()['item_overrides'].update( exposed_anus={ 'min_porb': 0.3 } ),
                lambda: betaconfig.item_overrides.update( exposed_anus={ 'censor_shap': 'box' } ),
                lambda: backend().update( variants={ '640m': { 'item_overrides': {
                    'exposed_breast': { 'min_porb': 0.3 } } } } ),
                lambda: backend()['profiles']['variants']['scene']['item_overrides'].update(
                    exposed_anus={ 'min_porb': 0.3 } ) ],
            'windows': [ lambda: betaconfig.reverse['windows'].update(
                face_femme={ 'widht_area_safety': 1.0 } ) ],
            'overrides': [ lambda: betaconfig.reverse['censor']['overrides'].update(
                exposed_breast={ 'min_porb': 0.3 } ) ],
        }
        self.assertEqual( sorted( cases ), sorted( bu_local.VALIDATED_LABEL_MAPS ),
                          "a map the warning skips needs a case here proving it is validated" )
        for name, mutations in cases.items():
            for index, mutate in enumerate( mutations ):
                mutate()
                bu_config.invalidate_config_caches()
                errors = bu_config._collect_validation_errors()
                self._restore()
                self.assertTrue( any( 'porb' in error or 'shap' in error or 'widht' in error
                                      for error in errors ), ( name, index, errors ) )


class TestNothingToRender( unittest.TestCase ):

    NAMES = ( 'items_to_censor', 'reverse' )

    def setUp( self ):
        self.saved = { name: copy.deepcopy( getattr( betaconfig, name ) ) for name in self.NAMES }
        self.addCleanup( self._restore )

    def _restore( self ):
        for name, value in self.saved.items():
            setattr( betaconfig, name, copy.deepcopy( value ) )
        bu_config.invalidate_config_caches()

    def test_the_shipped_config_censors_nothing_and_says_so_before_any_work( self ):
        self.assertEqual( _shipped_values()['items_to_censor'], [] )
        betaconfig.items_to_censor = []
        self.assertIn( 'items_to_censor is empty', bu_config.nothing_to_render_reason() )
        logger = unittest.mock.Mock()
        with self.assertRaises( SystemExit ) as stopped:
            bu_config.exit_if_nothing_to_render( logger )
        self.assertEqual( stopped.exception.code, 0, "nothing failed" )
        self.assertIn( 'local_config.py set', logger.warning.call_args[0][0] )

    def test_the_renderers_stop_before_any_work( self ):
        # Run for real on the shipped config: a local file only moves the
        # log and stats out of the real output folder.
        import subprocess, sys, tempfile, shutil
        directory = tempfile.mkdtemp( prefix='betasuite-nothing-' )
        self.addCleanup( shutil.rmtree, directory, True )
        local = os.path.join( directory, 'local.py' )
        log_path = os.path.join( directory, 'betasuite.log' )
        with open( local, 'w', encoding='utf-8' ) as handle:
            handle.write( "log_path = %r\nstats_path = %r\n"%( log_path, os.path.join( directory, 's.jsonl' ) ) )
        env = dict( os.environ, BETASUITE_LOCAL_CONFIG=local )
        for script in ( 'betatv.py', 'betastare.py' ):
            done = subprocess.run( [ sys.executable, script ], cwd=REPO, env=env, capture_output=True,
                                   text=True, timeout=120 )
            output = done.stdout + done.stderr
            self.assertEqual( done.returncode, 0, ( script, output ) )
            self.assertIn( 'items_to_censor is empty', output, script )
            self.assertIn( 'Nothing was processed', output, script )
        # betavision-censor.py imports win32 modules, so read its source.
        with open( os.path.join( REPO, 'betavision-censor.py' ), encoding='utf-8' ) as handle:
            source = handle.read()
        main = source[ source.index( 'def main():' ): ]
        self.assertLess( main.index( 'exit_if_nothing_to_render' ), main.index( 'open_detection_shared_memory()' ) )

    def test_censoring_anything_runs( self ):
        betaconfig.items_to_censor = [ 'exposed_breast' ]
        self.assertIsNone( bu_config.nothing_to_render_reason() )
        bu_config.exit_if_nothing_to_render( unittest.mock.Mock() )

    def test_reverse_mode_runs_when_anything_can_change_a_frame( self ):
        betaconfig.items_to_censor = []
        betaconfig.reverse['enabled'] = True
        bu_config.invalidate_config_caches()
        self.assertIn( 'reverse mode', bu_config.nothing_to_render_reason() )
        for change in ( lambda: betaconfig.reverse.update( empty_frame='obscure' ),
                        lambda: betaconfig.reverse['trigger'].update( labels=[ 'exposed_breast' ] ),
                        lambda: betaconfig.reverse['windows'].update(
                            face_femme={ 'censor_style': { 'type': 'show' } } ) ):
            betaconfig.reverse = copy.deepcopy( self.saved['reverse'] )
            betaconfig.reverse['enabled'] = True
            change()
            bu_config.invalidate_config_caches()
            self.assertIsNone( bu_config.nothing_to_render_reason() )

    def test_promotion_into_an_unrendered_label_is_an_error_only_when_it_loses_a_censor( self ):
        betaconfig.items_to_censor = []
        bu_config.invalidate_config_caches()
        self.assertEqual( [ e for e in bu_config._collect_validation_errors() if 'class_promotion' in e ], [] )
        # covered_vulva rendered, exposed_vulva not: its promoted boxes lose their censor.
        betaconfig.items_to_censor = [ 'covered_vulva' ]
        bu_config.invalidate_config_caches()
        errors = [ e for e in bu_config._collect_validation_errors() if 'class_promotion' in e ]
        self.assertEqual( len( errors ), 1, errors )
        self.assertIn( "'covered_vulva' boxes promoted to it would lose their censor", errors[0] )


if __name__ == '__main__':
    unittest.main()
