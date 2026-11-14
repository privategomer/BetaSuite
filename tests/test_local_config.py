"""
test_local_config.py - the local config (betaconfig_local.py): where it
lives, what it may set and how it merges into betaconfig, its place in
the precedence (default < local < CLI), how validation, the cache keys
and the stats record see it, the editor every writer uses, the
migration from the old machine config, and tools/setup/local_config.py.
"""

import ast
import copy
import io
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import betaconfig
import betautils_cache_paths as bu_cache
import betautils_cli as bu_cli
import betautils_config as bu_config
import betautils_local as bu_local

REPO = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
TOOL = os.path.join( REPO, 'tools', 'setup', 'local_config.py' )
SHIM = os.path.join( REPO, 'tools', 'setup', 'machine_config.py' )


def _namespace():
    """A betaconfig-shaped namespace with the kinds of value the merge treats differently."""
    return {
        'detector_backend': { 'selected': 'nudenet_v3', 'defaults': { 'nn_batch_size': 2 },
                              'nudenet_v3': { 'nn_batch_size': 4, 'candidate_floor': 0.2 } },
        'person_backend': { 'selected': 'yolo11_seg', 'yolo11_seg': { 'batch_size': 4 } },
        'default_censor_style': { 'type': 'blur', 'method': 'triple_box', 'strength': 65 },
        'items_to_censor': [ 'exposed_vulva', 'exposed_breast' ],
        'encode_video_codec': 'libx264', 'encode_preset': 'fast', 'encode_crf': 17,
        'render_workers': 0, 'gpu_enabled': 1, 'default_min_prob': 0.35,
        'console_palette': 'standard',
        'reverse': { 'enabled': False, 'windows': {} },
        # Not settings: what a module namespace also holds.
        '__name__': 'betaconfig', 'os': os, 'helper': lambda: None,
    }


def _shipped_values():
    """betaconfig.py's own literal assignments, read from the source."""
    with open( os.path.join( REPO, 'betaconfig.py' ), encoding='utf-8' ) as handle:
        tree = ast.parse( handle.read() )
    return { node.targets[0].id: ast.literal_eval( node.value ) for node in tree.body
             if isinstance( node, ast.Assign ) and isinstance( node.targets[0], ast.Name ) }


class TempDir( unittest.TestCase ):

    def setUp( self ):
        self.dir = tempfile.mkdtemp( prefix='betasuite-local-' )
        self.addCleanup( shutil.rmtree, self.dir, True )
        self.path = os.path.join( self.dir, 'betaconfig_local.py' )
        # Never this machine's real ../resources file.
        self.legacy = os.path.join( self.dir, 'betaconfig_machine.py' )
        saved = bu_local.LOADED
        self.addCleanup( setattr, bu_local, 'LOADED', saved )

    def write( self, text, path=None ):
        with open( path or self.path, 'w', encoding='utf-8' ) as handle:
            handle.write( text )

    def apply( self, text, namespace=None ):
        self.write( text )
        namespace = namespace if namespace is not None else _namespace()
        stderr = io.StringIO()
        loaded = bu_local.apply( namespace, path=self.path, stderr=stderr, legacy_path=self.legacy )
        return loaded, namespace, stderr.getvalue()


class LiveConfig( TempDir ):
    """Applies a local file to the real betaconfig module, and puts it back."""

    NAMES = ( 'detector_backend', 'render_workers', 'encode_video_codec', 'encode_preset', 'encode_crf',
              'global_min_prob', 'console_palette', 'default_min_prob' )

    def setUp( self ):
        super().setUp()
        self.saved = { name: copy.deepcopy( getattr( betaconfig, name ) ) for name in self.NAMES }
        self.addCleanup( self._restore )

    def _restore( self ):
        for name, value in self.saved.items():
            setattr( betaconfig, name, value )
        bu_config.invalidate_config_caches()

    def apply_live( self, text ):
        loaded, _namespace, _stderr = self.apply( text, vars( betaconfig ) )
        bu_config.invalidate_config_caches()
        return loaded


class TestWhereItLives( unittest.TestCase ):

    def test_the_default_is_beside_betaconfig( self ):
        self.assertEqual( bu_local.default_path(), os.path.join( REPO, 'betaconfig_local.py' ) )
        self.assertEqual( bu_local.legacy_machine_path(),
                          os.path.normpath( os.path.join( REPO, '..', 'resources', 'betaconfig_machine.py' ) ) )

    def test_the_environment_variable_moves_or_turns_it_off( self ):
        self.assertEqual( bu_local.configured_path( {} ), bu_local.default_path() )
        for off in ( 'none', 'NONE', '', '  ', 'off' ):
            self.assertIsNone( bu_local.configured_path( { bu_local.ENV_VAR: off } ), off )
        self.assertEqual( bu_local.configured_path( { bu_local.ENV_VAR: '/x/y.py' } ),
                          os.path.abspath( '/x/y.py' ) )
        self.assertEqual( bu_local.ENV_VAR, 'BETASUITE_LOCAL_CONFIG' )

    def test_the_suite_runs_on_the_shipped_defaults( self ):
        # tests/__init__.py turns it off before betaconfig is imported, so
        # no checkout's own file changes what the suite sees.
        self.assertEqual( os.environ.get( bu_local.ENV_VAR ), 'none' )
        self.assertIsNone( bu_local.LOADED.path if bu_local.LOADED else None )

    def test_git_ignores_it( self ):
        with open( os.path.join( REPO, '.gitignore' ), encoding='utf-8' ) as handle:
            lines = [ line.strip() for line in handle ]
        self.assertIn( '/betaconfig_local.py', lines )

    def test_the_repo_ships_portable_defaults( self ):
        # A fresh checkout must not need an NVIDIA GPU to render, and
        # one person's palette is not everyone's default.
        values = _shipped_values()
        self.assertEqual( ( values['encode_video_codec'], values['encode_preset'], values['encode_crf'] ),
                          ( 'libx264', 'fast', 17 ) )
        self.assertEqual( values['console_palette'], 'standard' )

    def test_betaconfig_applies_the_file_on_import( self ):
        # The hook is betaconfig.py's last line; every reader sees the merge.
        directory = tempfile.mkdtemp( prefix='betasuite-local-' )
        self.addCleanup( shutil.rmtree, directory, True )
        path = os.path.join( directory, 'mine.py' )
        with open( path, 'w', encoding='utf-8' ) as handle:
            handle.write( "encode_video_codec = 'h264_nvenc'\nencode_crf = 21\ndefault_min_prob = 0.3\n"
                          "detector_backend = { 'nudenet_v3': { 'nn_batch_size': 8 } }\n" )
        env = dict( os.environ, **{ bu_local.ENV_VAR: path } )
        out = subprocess.run( [ sys.executable, '-c',
                                "import betaconfig as c; print(c.encode_video_codec, c.encode_crf, c.default_min_prob, "
                                "c.detector_backend['nudenet_v3']['nn_batch_size'], "
                                "c.detector_backend['nudenet_v3']['candidate_floor'])" ],
                              cwd=REPO, env=env, capture_output=True, text=True, timeout=120 )
        self.assertEqual( out.returncode, 0, out.stderr )
        # candidate_floor is not in the local file, so it keeps whatever
        # betaconfig.py says: read from the source, not pinned.
        floor = _shipped_values()['detector_backend']['nudenet_v3']['candidate_floor']
        self.assertEqual( out.stdout.split(), [ 'h264_nvenc', '21', '0.3', '8', str( floor ) ] )

    def test_a_broken_file_never_stops_the_import( self ):
        # betaconfig is imported by every tool, including the one that
        # explains the problem.
        directory = tempfile.mkdtemp( prefix='betasuite-local-' )
        self.addCleanup( shutil.rmtree, directory, True )
        path = os.path.join( directory, 'mine.py' )
        with open( path, 'w', encoding='utf-8' ) as handle:
            handle.write( "render_workers = (\n" )
        env = dict( os.environ, **{ bu_local.ENV_VAR: path } )
        out = subprocess.run( [ sys.executable, '-c', "import betaconfig as c; print(c.render_workers)" ],
                              cwd=REPO, env=env, capture_output=True, text=True, timeout=120 )
        self.assertEqual( out.returncode, 0, out.stderr )
        self.assertEqual( out.stdout.split(), [ str( _shipped_values()['render_workers'] ) ] )
        self.assertIn( 'nothing in the file was applied', out.stderr )


class TestWhatItMaySet( TempDir ):

    def test_any_setting_betaconfig_has_merges_and_the_rest_stays( self ):
        loaded, namespace, stderr = self.apply(
            "render_workers = 6\nencode_video_codec = 'h264_nvenc'\ndefault_min_prob = 0.3\n"
            "console_palette = 'deuteranopia'\n"
            "detector_backend = { 'nudenet_v3': { 'nn_batch_size': 8 }, 'defaults': { 'nn_batch_size': 1 } }\n" )
        self.assertEqual( ( loaded.problems, stderr ), ( [], '' ) )
        self.assertEqual( namespace['render_workers'], 6 )
        self.assertEqual( namespace['default_min_prob'], 0.3, "tuning settings are no longer refused" )
        self.assertEqual( namespace['console_palette'], 'deuteranopia' )
        self.assertEqual( namespace['detector_backend'], {
            'selected': 'nudenet_v3', 'defaults': { 'nn_batch_size': 1 },
            'nudenet_v3': { 'nn_batch_size': 8, 'candidate_floor': 0.2 } } )
        self.assertEqual( loaded.replaced[ ( 'render_workers', ) ], 0 )
        self.assertEqual( loaded.replaced[ ( 'detector_backend', 'nudenet_v3', 'nn_batch_size' ) ], 4 )
        self.assertEqual( loaded.set_names(), [ 'console_palette', 'default_min_prob', 'detector_backend',
                                                'encode_video_codec', 'render_workers' ] )
        self.assertEqual( loaded.new_keys, [] )

    def test_an_unknown_name_is_refused_with_a_suggestion( self ):
        loaded, namespace, stderr = self.apply( "render_worker = 3\nencode_crf = 20\n" )
        self.assertNotIn( 'render_worker', namespace )
        self.assertEqual( namespace['encode_crf'], 20, "the good line still applies" )
        self.assertEqual( len( loaded.problems ), 1 )
        self.assertIn( 'betaconfig_local.py line 1', loaded.problems[0] )
        self.assertIn( 'did you mean render_workers?', loaded.problems[0] )
        self.assertIn( 'skipped', stderr )

    def test_names_that_are_not_settings_are_unknown( self ):
        loaded, namespace, _stderr = self.apply( "os = 1\nhelper = 2\n__name__ = 'x'\nzzzz_qqqq = 1\n" )
        self.assertEqual( len( loaded.problems ), 4 )
        self.assertIs( namespace['os'], os )
        self.assertNotIn( 'did you mean', loaded.problems[3] )

    def test_a_new_key_inside_a_dict_merges_and_is_flagged( self ):
        loaded, namespace, _stderr = self.apply(
            "detector_backend = { 'nudenet_v3': { 'nn_batchsize': 8 } }\n" )
        self.assertEqual( loaded.problems, [] )
        self.assertEqual( namespace['detector_backend']['nudenet_v3']['nn_batchsize'], 8 )
        self.assertEqual( namespace['detector_backend']['nudenet_v3']['nn_batch_size'], 4 )
        self.assertEqual( loaded.new_keys, [ ( ( 'detector_backend', 'nudenet_v3', 'nn_batchsize' ),
                                               'nn_batch_size', 1 ) ] )
        warning = bu_local.new_key_warnings( loaded )[0]
        self.assertIn( "line 1: detector_backend['nudenet_v3']['nn_batchsize']", warning )
        self.assertIn( "closest existing key: 'nn_batch_size'", warning )

    def test_a_new_label_or_per_label_key_is_not_flagged( self ):
        # The tuners write these, and a warning on every run for an
        # ordinary line taught people to ignore the warning.
        loaded, namespace, _stderr = self.apply(
            "detector_backend = { 'nudenet_v3': { 'item_overrides': { 'exposed_anus': { 'min_prob': 0.4 } } } }\n"
            "reverse = { 'windows': { 'face_femme': { 'width_area_safety': 1.0 } } }\n" )
        self.assertEqual( loaded.problems, [] )
        self.assertEqual( loaded.new_keys, [] )
        self.assertEqual( namespace['reverse']['windows']['face_femme'], { 'width_area_safety': 1.0 } )

    def test_a_new_block_has_no_suggestion_when_nothing_is_close( self ):
        loaded, namespace, _stderr = self.apply( "person_backend = { 'rf_detr_seg': { 'batch_size': 1 } }\n" )
        self.assertEqual( namespace['person_backend']['rf_detr_seg'], { 'batch_size': 1 } )
        self.assertEqual( loaded.new_keys, [ ( ( 'person_backend', 'rf_detr_seg' ), None, 1 ) ] )
        self.assertNotIn( 'closest', bu_local.new_key_warnings( loaded )[0] )

    def test_a_complete_censor_style_and_a_list_replace_whole( self ):
        loaded, namespace, _stderr = self.apply(
            "default_censor_style = { 'type': 'pixel', 'pattern': 'mosaic', 'strength': 20 }\n"
            "items_to_censor = [ 'exposed_breast' ]\n" )
        self.assertEqual( namespace['default_censor_style'],
                          { 'type': 'pixel', 'pattern': 'mosaic', 'strength': 20 },
                          "a blur's 'method' must not leak into a pixel style" )
        self.assertEqual( namespace['items_to_censor'], [ 'exposed_breast' ] )
        self.assertEqual( list( loaded.applied ), [ ( 'default_censor_style', ), ( 'items_to_censor', ) ] )

    def test_a_style_without_a_type_tweaks_one_key( self ):
        _loaded, namespace, _stderr = self.apply( "default_censor_style = { 'strength': 80 }\n" )
        self.assertEqual( namespace['default_censor_style'],
                          { 'type': 'blur', 'method': 'triple_box', 'strength': 80 } )

    def test_a_dict_over_a_plain_value_replaces_it( self ):
        _loaded, namespace, _stderr = self.apply( "render_workers = { 'x': 1 }\n" )
        self.assertEqual( namespace['render_workers'], { 'x': 1 } )

    def test_nothing_is_executed( self ):
        marker = os.path.join( self.dir, 'ran' )
        loaded, namespace, _stderr = self.apply(
            "render_workers = __import__('os').mkdir(%r)\nimport os\nrender_workers += 1\n"%( marker, ) )
        self.assertFalse( os.path.exists( marker ) )
        self.assertEqual( namespace['render_workers'], 0 )
        self.assertEqual( len( loaded.problems ), 3 )
        self.assertIn( 'line 1', loaded.problems[0] )
        self.assertIn( 'plain value', loaded.problems[0] )
        self.assertIn( 'line 2', loaded.problems[1] )
        self.assertIn( 'only `name = value`', loaded.problems[1] )

    def test_a_setting_twice_is_a_problem( self ):
        loaded, namespace, _stderr = self.apply( "encode_crf = 20\n\nencode_crf = 22\n" )
        self.assertEqual( namespace['encode_crf'], 22, "Python's own reading: the last line wins" )
        self.assertEqual( loaded.problems, [
            "betaconfig_local.py line 3: encode_crf is already set on line 1; delete one of them" ] )

    def test_a_syntax_error_applies_nothing( self ):
        loaded, namespace, _stderr = self.apply( "render_workers = 3\nencode_crf = (\n" )
        self.assertEqual( namespace['render_workers'], 0 )
        self.assertIn( 'nothing in the file was applied', loaded.problems[0] )

    def test_an_empty_file_or_a_docstring_sets_nothing( self ):
        for text in ( '', '"""my settings"""\n# a comment\n' ):
            loaded, namespace, _stderr = self.apply( text )
            self.assertEqual( ( loaded.applied, loaded.problems ), ( {}, [] ) )
            self.assertTrue( loaded.exists )
            self.assertIn( 'overrides nothing', loaded.describe() )
            self.assertEqual( namespace['render_workers'], 0 )

    def test_a_missing_file_is_not_a_problem( self ):
        namespace = _namespace()
        loaded = bu_local.apply( namespace, path=os.path.join( self.dir, 'absent.py' ), legacy_path=self.legacy )
        self.assertFalse( loaded.exists )
        self.assertEqual( ( loaded.applied, loaded.problems ), ( {}, [] ) )
        self.assertIn( 'none at', loaded.describe() )
        self.assertIs( bu_local.LOADED, loaded )

    def test_turned_off_reads_nothing_and_looks_for_no_old_file( self ):
        self.write( "render_workers = 5\n" )
        self.write( "render_workers = 5\n", self.legacy )
        for off in ( 'none', '' ):
            namespace = _namespace()
            loaded = bu_local.apply( namespace, environ={ bu_local.ENV_VAR: off }, legacy_path=self.legacy )
            self.assertIsNone( loaded.path )
            self.assertEqual( namespace['render_workers'], 0 )
            self.assertIn( 'off', loaded.describe() )
            self.assertIsNone( bu_local.legacy_warning( loaded ) )

    def test_the_environment_variable_names_the_file( self ):
        self.write( "render_workers = 5\n" )
        namespace = _namespace()
        loaded = bu_local.apply( namespace, environ={ bu_local.ENV_VAR: self.path }, legacy_path=self.legacy )
        self.assertEqual( loaded.path, self.path )
        self.assertEqual( namespace['render_workers'], 5 )

    def test_describe_is_the_one_run_start_line( self ):
        loaded, _namespace, _stderr = self.apply(
            "encode_crf = 21\ndetector_backend = { 'nudenet_v3': { 'nn_batch_size': 8 } }\n"
            "items_to_censor = [ 'exposed_vulva', 'exposed_breast', 'exposed_anus', 'exposed_buttocks' ]\n" )
        self.assertEqual( loaded.describe(),
                          "local overrides active: encode_crf=21, detector_backend['nudenet_v3']['nn_batch_size']=8, "
                          "items_to_censor=<list of 4> (from %s)"%( self.path, ) )
        self.assertEqual( loaded.overridden_names(), [ 'encode_crf', "detector_backend['nudenet_v3']['nn_batch_size']",
                                                       'items_to_censor' ] )

    def test_short_values_are_shown_and_long_ones_named( self ):
        self.assertEqual( bu_local.short_value( 'x' * 38 ), repr( 'x' * 38 ) )
        self.assertEqual( bu_local.short_value( 'x' * 39 ), '<str>' )
        self.assertEqual( bu_local.short_value( list( range( 20 ) ) ), '<list of 20>' )


class TestOldMachineConfig( TempDir ):

    def test_while_the_old_file_exists_it_is_named_with_the_command( self ):
        self.write( "encode_crf = 21\n", self.legacy )
        loaded, namespace, _stderr = self.apply( "" )
        self.assertEqual( namespace['encode_crf'], 17, "the old file is not read" )
        warning = bu_local.legacy_warning( loaded )
        self.assertIn( self.legacy, warning )
        self.assertIn( 'local_config.py migrate', warning )
        self.assertNotIn( '\n', warning )

    def test_also_when_no_local_file_exists_yet( self ):
        self.write( "encode_crf = 21\n", self.legacy )
        loaded = bu_local.apply( _namespace(), path=os.path.join( self.dir, 'absent.py' ), legacy_path=self.legacy )
        self.assertIsNotNone( bu_local.legacy_warning( loaded ) )

    def test_no_old_file_no_warning( self ):
        loaded, _namespace, _stderr = self.apply( "" )
        self.assertIsNone( bu_local.legacy_warning( loaded ) )
        self.assertIsNone( bu_local.legacy_warning( None ) )


class TestMigrationPlan( unittest.TestCase ):

    def plan( self, legacy, local ):
        entries, problems = bu_local.parse( legacy )
        self.assertEqual( problems, [] )
        return bu_local.migration_plan( entries, local, _namespace() )

    def test_each_kind_of_line( self ):
        plan = self.plan( "encode_crf = 21\nencode_preset = 'fast'\nrender_workers = 4\ngpu_enabled = 0\n"
                          "detector_backend = { 'nudenet_v3': { 'nn_batch_size': 8 }, 'defaults': { 'nn_batch_size': 2 } }\n",
                          { 'render_workers': 4, 'gpu_enabled': 1 } )
        self.assertEqual( plan['copy'], [ ( ( 'encode_crf', ), 21 ),
                                          ( ( 'detector_backend', 'nudenet_v3', 'nn_batch_size' ), 8 ) ] )
        self.assertEqual( plan['same_as_default'], [ ( 'encode_preset', ),
                                                     ( 'detector_backend', 'defaults', 'nn_batch_size' ) ] )
        self.assertEqual( plan['already_local'], [ ( 'render_workers', ) ] )
        self.assertEqual( plan['conflicts'], [ ( ( 'gpu_enabled', ), 0, 1 ) ] )

    def test_a_nested_local_value_is_compared_key_by_key( self ):
        plan = self.plan( "detector_backend = { 'nudenet_v3': { 'nn_batch_size': 8 } }\n",
                          { 'detector_backend': { 'nudenet_v3': { 'nn_batch_size': 6 } } } )
        self.assertEqual( plan['conflicts'], [ ( ( 'detector_backend', 'nudenet_v3', 'nn_batch_size' ), 8, 6 ) ] )
        self.assertEqual( plan['copy'], [] )


class TestChangedPaths( unittest.TestCase ):

    def test_changes_are_found_at_the_depth_they_happen( self ):
        before = { 'a': 1, 'b': { 'c': 2, 'd': 3 } }
        self.assertEqual( bu_local.changed_paths( before, copy.deepcopy( before ) ), [] )
        self.assertEqual( bu_local.changed_paths( before, { 'a': 1, 'b': { 'c': 9, 'd': 3, 'e': 1 } } ),
                          [ ( 'b', 'c' ), ( 'b', 'e' ) ] )
        self.assertEqual( bu_local.changed_paths( 1, 2, ( 'x', ) ), [ ( 'x', ) ] )
        self.assertEqual( bu_local.changed_paths( { 'a': 1 }, 5, ( 'x', ) ), [ ( 'x', ) ] )


class TestEncoderScaleWarning( TempDir ):

    def test_switching_family_without_quality_warns( self ):
        loaded, _namespace, _stderr = self.apply( "encode_video_codec = 'h264_nvenc'\n" )
        self.assertIn( "NVENC's cq", bu_local.encoder_scale_warning( loaded ) )

    def test_switching_with_quality_or_within_a_family_is_fine( self ):
        loaded, _n, _s = self.apply( "encode_video_codec = 'h264_nvenc'\nencode_crf = 21\n" )
        self.assertIsNone( bu_local.encoder_scale_warning( loaded ) )
        loaded, _n, _s = self.apply( "encode_video_codec = 'libx265'\n" )
        self.assertIsNone( bu_local.encoder_scale_warning( loaded ) )
        namespace = dict( _namespace(), encode_video_codec='h264_nvenc' )
        loaded, _n, _s = self.apply( "encode_video_codec = 'hevc_nvenc'\n", namespace )
        self.assertIsNone( bu_local.encoder_scale_warning( loaded ) )
        self.assertIsNone( bu_local.encoder_scale_warning( None ) )


class TestPrecedence( LiveConfig ):
    """default < local < CLI, through the real betaconfig module and betatv.py's flag code."""

    def flags( self, *argv ):
        parser = bu_cli.build_arg_parser( 'test', include_preview=True, include_logging=True )
        with mock.patch( 'sys.stdout', io.StringIO() ):
            bu_cli.apply_cli_overrides( betaconfig, parser.parse_args( list( argv ) ) )

    def test_each_layer_beats_the_one_below( self ):
        shipped = _shipped_values()
        self.assertEqual( betaconfig.render_workers, shipped['render_workers'] )
        self.apply_live( "render_workers = 6\ndetector_backend = { 'nudenet_v3': { 'nn_batch_size': 8 } }\n" )
        self.assertNotEqual( shipped['render_workers'], 6, "precondition: the layers disagree" )
        self.assertEqual( betaconfig.render_workers, 6, "local beats default" )
        self.flags( '--render-workers', '2', '--nn-batch-size', '3' )
        self.assertEqual( betaconfig.render_workers, 2, "CLI beats local" )
        self.assertEqual( betaconfig.detector_backend['nudenet_v3']['nn_batch_size'], 3 )

    def test_a_flag_not_given_leaves_the_local_value( self ):
        self.apply_live( "render_workers = 6\n" )
        self.flags( '--video-censor-fps', '12' )
        self.assertEqual( betaconfig.render_workers, 6 )


class TestCacheKeysAndStats( LiveConfig ):
    """Overrides reach the keys and the stats record through the effective values."""

    def test_each_key_changes_with_a_local_override_of_its_inputs( self ):
        before = ( bu_cache.detection_key(), bu_cache.censor_key(), bu_cache.encode_key( False ) )
        self.apply_live( "encode_crf = 23\n" )
        self.assertNotEqual( bu_cache.encode_key( False ), before[2] )
        self.assertEqual( ( bu_cache.detection_key(), bu_cache.censor_key() ), before[:2] )
        self._restore()
        self.assertEqual( bu_cache.encode_key( False ), before[2], "precondition: restoring restores the key" )

        self.apply_live( "detector_backend = { 'nudenet_v3': { 'item_overrides': "
                         "{ 'exposed_breast': { 'min_track_hits': 3 } } } }\n" )
        self.assertNotEqual( bu_cache.censor_key(), before[1] )
        self.assertEqual( bu_cache.detection_key(), before[0] )
        self._restore()

        self.apply_live( "global_min_prob = 0.13\n" )
        self.assertNotEqual( bu_cache.detection_key(), before[0] )

    def test_the_stats_record_carries_the_effective_values_and_names_the_overrides( self ):
        import betatv
        self.apply_live( "detector_backend = { 'nudenet_v3': { 'nn_batch_size': 7 } }\nglobal_min_prob = 0.13\n" )
        source = types.SimpleNamespace( transcoded=False, width=64, height=48, vid_fps=10.0, num_frames=20 )
        paths = { 'final': '/x/out.mp4', 'detection_key': 'd', 'censor_key': 'c', 'encode_key': 'e' }
        with mock.patch.object( betatv.bu_detector, 'session_provider_summary', return_value='CPU' ), \
                mock.patch.object( betatv.bu_detector, 'selected_backend_name', return_value='nudenet_v3' ):
            record = betatv._stats_record( 'clip.mp4', paths, False, 0, 1, 1, 2, [ 640 ], None, source, {}, {} )
        self.assertEqual( record['nn_batch_size'], 7 )
        self.assertEqual( record['global_min_prob'], 0.13 )
        self.assertEqual( record['local_overrides'], [ "detector_backend['nudenet_v3']['nn_batch_size']",
                                                       'global_min_prob' ] )


class TestValidation( LiveConfig ):

    def test_a_skipped_line_stops_a_real_run_naming_file_and_line( self ):
        self.apply_live( "\nrender_worker = 3\n" )
        errors = bu_config._collect_validation_errors()
        self.assertIn( 'betaconfig_local.py line 2: render_worker is not a setting betaconfig.py has '
                       '(did you mean render_workers?)', '\n'.join( errors ) )

    def test_validate_config_logs_the_overrides_once_and_says_which_file_set_a_bad_value( self ):
        self.apply_live( "render_workers = -2\nconsole_palette = 'deuteranopia'\n" )
        with self.assertLogs( 'betasuite', level='INFO' ) as logs, self.assertRaises( SystemExit ):
            bu_config.validate_config()
        text = '\n'.join( logs.output )
        self.assertEqual( text.count( 'local overrides active:' ), 1 )
        self.assertIn( "local overrides active: render_workers=-2, console_palette='deuteranopia' (from %s)"%(
            self.path, ), text )
        self.assertIn( 'render_workers must be a non-negative integer', text )
        self.assertIn( "%s sets console_palette, render_workers; an error naming one of those"%( self.path, ), text )

    def test_warnings_reach_the_log( self ):
        self.write( "encode_crf = 21\n", self.legacy )
        self.apply_live( "encode_video_codec = 'h264_nvenc'\nencode_preset = 'p5'\n"
                         "detector_backend = { 'nudenet_v3': { 'nn_batchsize': 8 } }\n" )
        warnings = '\n'.join( bu_config._collect_validation_warnings() )
        self.assertIn( "NVENC's cq", warnings )
        self.assertIn( 'local_config.py migrate', warnings )
        self.assertIn( "closest existing key: 'nn_batch_size'", warnings )


class TestHardwareSettingValidation( unittest.TestCase ):

    def _errors( self, **values ):
        saved = { name: getattr( betaconfig, name, None ) for name in values }
        try:
            for name, value in values.items():
                setattr( betaconfig, name, value )
            errors = []
            bu_config._validate_hardware_settings( errors )
            return errors
        finally:
            for name, value in saved.items():
                setattr( betaconfig, name, value )

    def test_good_values( self ):
        for values in ( dict( render_decode_threads=None ), dict( render_decode_threads=0 ),
                        dict( render_decode_threads=4 ), dict( detection_prefetch_frames=0 ),
                        dict( detection_prefetch_frames=16 ), dict( structure_scan_workers=8 ),
                        dict( gpu_enabled=True ), dict( gpu_enabled=0 ), dict( cuda_device_id=1 ) ):
            self.assertEqual( self._errors( **values ), [], values )

    def test_bad_values( self ):
        for values, needle in ( ( dict( render_decode_threads=-1 ), 'render_decode_threads' ),
                                ( dict( render_decode_threads=2.5 ), 'render_decode_threads' ),
                                ( dict( detection_prefetch_frames=None ), 'detection_prefetch_frames' ),
                                ( dict( detection_prefetch_frames=True ), 'detection_prefetch_frames' ),
                                ( dict( structure_scan_workers=-1 ), 'structure_scan_workers' ),
                                ( dict( gpu_enabled=2 ), 'gpu_enabled' ),
                                ( dict( gpu_enabled='1' ), 'gpu_enabled' ),
                                ( dict( cuda_device_id=-1 ), 'cuda_device_id' ) ):
            errors = self._errors( **values )
            self.assertEqual( len( errors ), 1, ( values, errors ) )
            self.assertIn( needle, errors[0] )


class TestEditor( unittest.TestCase ):

    def setUp( self ):
        self.template = bu_local.template_text()

    def entries( self, text ):
        entries, problems = bu_local.parse( text )
        self.assertEqual( problems, [] )
        return { name: value for name, value, _line in entries }

    def test_the_template_is_a_header_and_sets_nothing( self ):
        self.assertEqual( self.entries( self.template ), {} )
        self.assertTrue( all( line.startswith( '#' ) for line in self.template.splitlines() ) )
        self.assertNotIn( '\u2014', self.template )

    def test_a_new_line_goes_below_the_header_with_a_blank_between( self ):
        text = bu_local.edit_text( self.template, { ( 'encode_crf', ): 21 }, comment='mine' )
        self.assertEqual( text, self.template + "\nencode_crf = 21  # mine\n" )
        text = bu_local.edit_text( text, { ( 'console_palette', ): 'deuteranopia' } )
        self.assertTrue( text.endswith( "\nencode_crf = 21  # mine\nconsole_palette = 'deuteranopia'\n" ) )

    def test_a_section_line_goes_under_its_header_and_others_stay_above( self ):
        section = bu_local.HARDWARE_SECTION
        text = bu_local.edit_text( self.template, { ( 'render_workers', ): 4 }, section=section )
        self.assertIn( "\n\n%s\nrender_workers = 4\n"%( bu_local.section_header( section ), ), text )
        text = bu_local.edit_text( text, { ( 'encode_crf', ): 21 }, section=section )
        text = bu_local.edit_text( text, { ( 'console_palette', ): 'deuteranopia' } )
        body = text[ len( self.template ): ]
        self.assertEqual( body, "\nconsole_palette = 'deuteranopia'\n\n%s\nrender_workers = 4\nencode_crf = 21\n"%(
            bu_local.section_header( section ), ) )
        self.assertEqual( text.count( bu_local.section_header( section ) ), 1 )

    def test_a_section_line_lands_before_the_next_section( self ):
        header = bu_local.section_header( bu_local.HARDWARE_SECTION )
        text = "%s\nrender_workers = 4\n\n# --- mine ---\nencode_crf = 1\n"%( header, )
        edited = bu_local.edit_text( text, { ( 'gpu_enabled', ): 0 }, section=bu_local.HARDWARE_SECTION )
        self.assertEqual( edited, "%s\nrender_workers = 4\ngpu_enabled = 0\n\n# --- mine ---\nencode_crf = 1\n"%(
            header, ) )

    def test_set_replaces_an_existing_line_in_place( self ):
        text = "# mine\nencode_crf = 21\nrender_workers = 2\n"
        edited = bu_local.edit_text( text, { ( 'encode_crf', ): 19 }, section=bu_local.HARDWARE_SECTION )
        self.assertEqual( edited, "# mine\nencode_crf = 19\nrender_workers = 2\n" )

    def test_a_commented_out_line_is_replaced_in_place( self ):
        text = "# encode_crf = 17\nrender_workers = 2\n"
        self.assertEqual( bu_local.edit_text( text, { ( 'encode_crf', ): 21 } ),
                          "encode_crf = 21\nrender_workers = 2\n" )

    def test_unset_deletes_the_line( self ):
        text = bu_local.edit_text( self.template, { ( 'render_workers', ): 6 } )
        self.assertEqual( bu_local.edit_text( text, removals=[ ( 'render_workers', ) ] ).rstrip( '\n' ),
                          self.template.rstrip( '\n' ) )
        text = "a = 1\nrender_workers = (\n    6 )  # mine\nb = 2\n"
        self.assertEqual( bu_local.edit_text( text, removals=[ ( 'render_workers', ) ] ), "a = 1\nb = 2\n" )

    def test_unset_of_what_is_not_there_changes_nothing( self ):
        text = "encode_crf = 21\n"
        self.assertEqual( bu_local.edit_text( text, removals=[ ( 'render_workers', ),
                                                               ( 'detector_backend', 'x', 'y' ) ] ), text )

    def test_nested_settings_merge_and_prune( self ):
        text = bu_local.edit_text( self.template, {
            ( 'detector_backend', 'nudenet_v3', 'nn_batch_size' ): 8 } )
        text = bu_local.edit_text( text, {
            ( 'detector_backend', 'nudenet_v3', 'item_overrides', 'exposed_breast', 'min_prob' ): 0.2,
            ( 'detector_backend', 'defaults', 'nn_batch_size' ): 1 } )
        self.assertEqual( self.entries( text ), { 'detector_backend': {
            'nudenet_v3': { 'nn_batch_size': 8, 'item_overrides': { 'exposed_breast': { 'min_prob': 0.2 } } },
            'defaults': { 'nn_batch_size': 1 } } } )
        text = bu_local.edit_text( text, removals=[
            ( 'detector_backend', 'nudenet_v3', 'item_overrides', 'exposed_breast', 'min_prob' ) ] )
        self.assertEqual( self.entries( text ), { 'detector_backend': {
            'nudenet_v3': { 'nn_batch_size': 8 }, 'defaults': { 'nn_batch_size': 1 } } },
            "emptied dicts are pruned, the sibling key stays" )
        text = bu_local.edit_text( text, removals=[ ( 'detector_backend', 'nudenet_v3', 'nn_batch_size' ),
                                                    ( 'detector_backend', 'defaults', 'nn_batch_size' ) ] )
        self.assertEqual( self.entries( text ), {} )
        self.assertNotIn( 'detector_backend', text )

    def test_a_persons_own_lines_and_comments_survive( self ):
        text = ( "# my laptop, tuned by hand\nrender_workers = 4  # keep it cool\n\n"
                 "encode_crf = 21\n" )
        edited = bu_local.edit_text( text, { ( 'encode_crf', ): 19, ( 'encode_preset', ): 'p5' } )
        self.assertEqual( edited, "# my laptop, tuned by hand\nrender_workers = 4  # keep it cool\n\n"
                                  "encode_crf = 19\nencode_preset = 'p5'\n" )
        edited = bu_local.edit_text( edited, { ( 'render_workers', ): 5 } )
        self.assertIn( "render_workers = 5  # keep it cool\n", edited, "a trailing comment is kept" )

    def test_a_multiline_value_is_replaced_whole( self ):
        text = "detector_backend = {\n    'nudenet_v3': { 'nn_batch_size': 4 },\n}\nrender_workers = 2\n"
        edited = bu_local.edit_text( text, { ( 'detector_backend', 'nudenet_v3', 'nn_batch_size' ): 8 } )
        self.assertEqual( edited, "detector_backend = { 'nudenet_v3': { 'nn_batch_size': 8 } }\n"
                                  "render_workers = 2\n" )

    def test_an_unparseable_file_is_not_touched( self ):
        with self.assertRaises( ValueError ):
            bu_local.edit_text( "render_workers = (\n", { ( 'render_workers', ): 2 } )

    def test_trailing_comments( self ):
        self.assertEqual( bu_local.trailing_comments( "a = 1  # tune_machine x\nb = 2\nc = (\n 3 )  #  hi\n" ),
                          { 'a': 'tune_machine x', 'c': 'hi' } )


class ToolCase( unittest.TestCase ):
    """tools/setup/local_config.py, in-process, against temp files."""

    def setUp( self ):
        import importlib.util
        spec = importlib.util.spec_from_file_location( 'local_config_tool', TOOL )
        self.tool = importlib.util.module_from_spec( spec )
        spec.loader.exec_module( self.tool )
        self.dir = tempfile.mkdtemp( prefix='betasuite-local-tool-' )
        self.addCleanup( shutil.rmtree, self.dir, True )
        self.path = os.path.join( self.dir, 'betaconfig_local.py' )
        self.legacy = os.path.join( self.dir, 'resources', 'betaconfig_machine.py' )
        self.log = os.path.join( self.dir, 'tool.log' )
        saved = bu_local.LOADED
        self.addCleanup( setattr, bu_local, 'LOADED', saved )
        self.addCleanup( os.environ.__setitem__, bu_local.ENV_VAR, os.environ[ bu_local.ENV_VAR ] )
        self.saved = { name: copy.deepcopy( getattr( betaconfig, name ) )
                       for name in bu_local.known_names( vars( betaconfig ) ) }
        self.addCleanup( self._restore )

    def _restore( self ):
        for name, value in self.saved.items():
            setattr( betaconfig, name, value )
        bu_config.invalidate_config_caches()

    def run_tool( self, *argv, stdin_tty=False, answer='' ):
        stdout = io.StringIO()
        with mock.patch( 'sys.stdout', stdout ), mock.patch( 'sys.stdin' ) as stdin, \
                mock.patch( 'builtins.input', return_value=answer ) as asked:
            stdin.isatty.return_value = stdin_tty
            status = self.tool.main( [ '--path', self.path, '--log-file', self.log,
                                       '--log-level', 'trace' ] + list( argv ), legacy_path=self.legacy )
        return status, stdout.getvalue(), asked

    def text( self, path=None ):
        with open( path or self.path, encoding='utf-8' ) as handle:
            return handle.read()

    def write( self, path, text ):
        os.makedirs( os.path.dirname( path ), exist_ok=True )
        with open( path, 'w', encoding='utf-8' ) as handle:
            handle.write( text )

    def settings( self, path=None ):
        entries, problems = bu_local.parse( self.text( path ) )
        self.assertEqual( problems, [] )
        return { name: value for name, value, _line in entries }


class TestTool( ToolCase ):

    def test_init_creates_an_empty_file_and_never_rewrites_one( self ):
        status, _out, _asked = self.run_tool( 'init' )
        self.assertEqual( status, 0 )
        self.assertEqual( self.text(), bu_local.template_text() )
        self.write( self.path, "encode_crf = 21\n" )
        status, out, _asked = self.run_tool( 'init' )
        self.assertEqual( ( status, self.text() ), ( 0, "encode_crf = 21\n" ) )
        self.assertIn( 'already exists', out )

    def test_set_writes_a_valid_change( self ):
        status, out, _asked = self.run_tool( 'set', 'encode_video_codec=h264_nvenc', 'encode_preset=p5',
                                             'encode_crf=21', 'console_palette=deuteranopia',
                                             'detector_backend.nudenet_v3.nn_batch_size=8' )
        self.assertEqual( status, 0, out )
        self.assertEqual( self.settings(), {
            'encode_video_codec': 'h264_nvenc', 'encode_preset': 'p5', 'encode_crf': 21,
            'console_palette': 'deuteranopia', 'detector_backend': { 'nudenet_v3': { 'nn_batch_size': 8 } } } )
        self.assertEqual( betaconfig.encode_video_codec, self.saved['encode_video_codec'],
                          "checking a change leaves the live config as it was" )

    def test_set_refuses_what_betatv_would_stop_on( self ):
        self.run_tool( 'set', 'render_workers=3' )
        before = self.text()
        for bad in ( 'render_workers=-1', 'encode_preset=slow1', 'render_worker=3',
                     'console_palette=plaid', 'detector_backend.nudenet_v3.candidate_floor=7' ):
            status, out, _asked = self.run_tool( 'set', bad )
            self.assertEqual( status, 1, bad )
            self.assertIn( 'not written', out )
            self.assertEqual( self.text(), before, bad )
        status, out, _asked = self.run_tool( 'set', 'render_worker=3' )
        self.assertIn( 'did you mean render_workers?', out )

    def test_tuning_settings_may_be_set( self ):
        status, out, _asked = self.run_tool( 'set', 'default_min_prob=0.3' )
        self.assertEqual( status, 0, out )
        self.assertEqual( self.settings(), { 'default_min_prob': 0.3 } )

    def test_bad_names_are_usage_errors( self ):
        for bad in ( 'render_workers', 'render_workers.x=1', 'encode_codec=x y', '=3',
                     'default_censor_style.strength=1' ):
            status, _out, _asked = self.run_tool( 'set', bad )
            self.assertEqual( status, 2, bad )
        self.assertFalse( os.path.exists( self.path ) )

    def test_hardware_lines_go_in_their_section( self ):
        self.run_tool( 'set', 'console_palette=deuteranopia' )
        status, _out, _asked = self.run_tool( 'set', '--section', 'hardware', '--comment', 'tune_machine x',
                                              'render_workers=6' )
        self.assertEqual( status, 0 )
        self.assertTrue( self.text().endswith( "console_palette = 'deuteranopia'\n\n%s\nrender_workers = 6  "
                                               "# tune_machine x\n"%( bu_local.section_header(
                                                   bu_local.HARDWARE_SECTION ), ) ), self.text() )

    def test_show_labels_every_source( self ):
        self.run_tool( 'set', 'encode_crf=21', 'detector_backend.nudenet_v3.nn_batch_size=8' )
        status, out, _asked = self.run_tool( 'show', '--render-workers', '3', '--video-censor-fps', '7' )
        self.assertEqual( status, 0, out )
        self.assertRegex( out, r"\n  encode_crf\s+21\s+local\s+\(betaconfig.py: 17\)" )
        self.assertRegex( out, r"\n  render_workers\s+3\s+CLI\s+\(betaconfig.py: 0\)" )
        self.assertRegex( out, r"\n  encode_preset\s+'fast'\s+default\n" )
        self.assertRegex( out, r"\n  detector_backend\s+\{.*merged\n\s+\['nudenet_v3'\]\['nn_batch_size'\]\s+8\s+local"
                               r"\s+\(betaconfig.py: 4\)" )
        self.assertIn( "local overrides active: encode_crf=21", out )
        self.assertIn( "command-line overrides: video_censor_fps=7.0, render_workers=3", out )
        self.assertIn( "betaconfig.py  <  betaconfig_local.py  <  command line", out )

    def test_a_flag_beats_the_local_line_for_the_same_setting( self ):
        self.run_tool( 'set', 'render_workers=6' )
        status, out, _asked = self.run_tool( 'show', '--render-workers', '3' )
        self.assertEqual( status, 0, out )
        self.assertRegex( out, r"\n  render_workers\s+3\s+CLI\s+\(betaconfig.py: 0\)" )

    def test_a_flag_inside_a_dict_is_labelled_cli( self ):
        self.run_tool( 'set', 'detector_backend.nudenet_v3.nn_batch_size=8' )
        status, out, _asked = self.run_tool( 'show', '--nn-batch-size', '2' )
        self.assertEqual( status, 0, out )
        self.assertRegex( out, r"\['nudenet_v3'\]\['nn_batch_size'\]\s+2\s+CLI\s+\(betaconfig.py: 4\)" )

    def test_extra_arguments_are_only_for_show( self ):
        with mock.patch( 'sys.stderr', io.StringIO() ), self.assertRaises( SystemExit ):
            self.run_tool( 'check', '--render-workers', '3' )
        with mock.patch( 'sys.stderr', io.StringIO() ):
            status, out, _asked = self.run_tool( 'show', '--no-such-flag' )
        self.assertEqual( status, 2 )

    def test_show_and_check_fail_on_a_problem( self ):
        self.run_tool( 'set', 'encode_crf=21' )
        self.assertEqual( self.run_tool( 'check' )[0], 0 )
        with open( self.path, 'a', encoding='utf-8' ) as handle:
            handle.write( "render_worker = 3\n" )
        self.assertEqual( self.run_tool( 'check' )[0], 1 )
        self.assertEqual( self.run_tool( 'show' )[0], 1 )

    def test_show_and_check_warn_while_the_old_file_waits( self ):
        self.write( self.legacy, "encode_crf = 21\n" )
        for command in ( 'show', 'check' ):
            status, out, _asked = self.run_tool( command )
            self.assertEqual( status, 0 )
            self.assertIn( 'local_config.py migrate', out, command )

    def test_unset( self ):
        self.run_tool( 'set', 'render_workers=3', 'encode_crf=21' )
        status, _out, _asked = self.run_tool( 'unset', 'render_workers' )
        self.assertEqual( status, 0 )
        self.assertEqual( self.settings(), { 'encode_crf': 21 } )

    def test_setup_creates_the_file_and_sets_the_gpu_flag_without_asking( self ):
        # The install that does not match what betaconfig.py ships.
        other = str( 1 - self.saved['gpu_enabled'] )
        status, out, asked = self.run_tool( 'setup', '--gpu-enabled', other, stdin_tty=True )
        self.assertEqual( status, 0, out )
        self.assertFalse( asked.called, "a value from betaconfig.py is setup's to change" )
        self.assertIn( "%s\ngpu_enabled = %s  # set by setup\n"%(
            bu_local.section_header( bu_local.HARDWARE_SECTION ), other ), self.text() )
        status, out, asked = self.run_tool( 'setup', '--gpu-enabled', other, stdin_tty=True )
        self.assertIn( 'matches', out )

    def test_setup_on_a_matching_install_writes_only_the_header( self ):
        status, _out, _asked = self.run_tool( 'setup', '--gpu-enabled', str( self.saved['gpu_enabled'] ) )
        self.assertEqual( status, 0 )
        self.assertEqual( self.text(), bu_local.template_text() )

    def test_setup_asks_before_overriding_the_files_own_value( self ):
        self.run_tool( 'set', 'gpu_enabled=0' )
        status, _out, asked = self.run_tool( 'setup', '--gpu-enabled', '1', stdin_tty=True, answer='n' )
        self.assertTrue( asked.called )
        self.assertIn( 'gpu_enabled = 0', self.text() )
        status, _out, asked = self.run_tool( 'setup', '--gpu-enabled', '1', '--yes', stdin_tty=True )
        self.assertFalse( asked.called )
        self.assertIn( 'gpu_enabled = 1', self.text() )

    def test_the_log_file_gets_everything( self ):
        self.run_tool( '--log-level', 'error', 'set', 'render_workers=3' )
        with open( self.log, encoding='utf-8' ) as handle:
            log = handle.read()
        self.assertIn( 'command:', log )
        self.assertIn( 'set render_workers = 3', log )


class TestMigrate( ToolCase ):

    LEGACY = ( "# old header\n# render_workers = 0\n"
               "render_decode_threads = 0  # tune_machine 20010101_120000\n"
               "encode_video_codec = 'h264_nvenc'\nencode_preset = 'p5'\nencode_crf = 21\n"
               "render_chunk_seconds = 180\n" )

    def test_it_moves_the_values_and_renames_the_old_file( self ):
        self.write( self.legacy, self.LEGACY )
        status, out, _asked = self.run_tool( 'migrate' )
        self.assertEqual( status, 0, out )
        self.assertEqual( self.settings(), { 'render_decode_threads': 0, 'encode_video_codec': 'h264_nvenc',
                                             'encode_preset': 'p5', 'encode_crf': 21 } )
        self.assertIn( "%s\nrender_decode_threads = 0  # tune_machine 20010101_120000; from betaconfig_machine.py\n"%(
            bu_local.section_header( bu_local.HARDWARE_SECTION ), ), self.text() )
        self.assertIn( 'not copied, same as betaconfig.py: render_chunk_seconds', out )
        self.assertFalse( os.path.exists( self.legacy ) )
        self.assertEqual( self.text( self.legacy + '.migrated' ), self.LEGACY, "renamed, not rewritten" )

    def test_a_second_run_is_a_success_that_changes_nothing( self ):
        self.write( self.legacy, self.LEGACY )
        self.run_tool( 'migrate' )
        first = self.text()
        status, out, _asked = self.run_tool( 'migrate' )
        self.assertEqual( ( status, self.text() ), ( 0, first ) )
        self.assertIn( 'already migrated', out )

    def test_an_interrupted_run_finishes_on_the_next( self ):
        # Written, but not yet renamed: the values are already local.
        self.write( self.legacy, "encode_crf = 21\n" )
        self.write( self.path, bu_local.template_text() + "\nencode_crf = 21\n" )
        before = self.text()
        status, out, _asked = self.run_tool( 'migrate' )
        self.assertEqual( ( status, self.text() ), ( 0, before ) )
        self.assertIn( 'already in betaconfig_local.py: encode_crf', out )
        self.assertFalse( os.path.exists( self.legacy ) )

    def test_a_local_value_wins_a_conflict_and_is_reported( self ):
        self.write( self.legacy, "encode_crf = 21\nrender_workers = 4\n" )
        self.run_tool( 'set', 'encode_crf=19' )
        status, out, _asked = self.run_tool( 'migrate' )
        self.assertEqual( status, 0, out )
        self.assertEqual( self.settings(), { 'encode_crf': 19, 'render_workers': 4 } )
        self.assertIn( "kept betaconfig_local.py's encode_crf = 19; betaconfig_machine.py had 21", out )

    def test_an_earlier_migrated_file_is_never_overwritten( self ):
        self.write( self.legacy, "encode_crf = 21\n" )
        self.write( self.legacy + '.migrated', "encode_crf = 1\n" )
        status, out, _asked = self.run_tool( 'migrate' )
        self.assertEqual( status, 1 )
        self.assertEqual( self.text( self.legacy + '.migrated' ), "encode_crf = 1\n" )
        self.assertEqual( self.text( self.legacy ), "encode_crf = 21\n" )
        self.assertIn( 'rename one by hand', out )

    def test_a_bad_old_file_changes_nothing( self ):
        for legacy in ( "encode_crf = (\n", "import os\n", "render_workers = -5\n" ):
            self.write( self.legacy, legacy )
            status, out, _asked = self.run_tool( 'migrate' )
            self.assertEqual( status, 1, legacy )
            self.assertIn( 'nothing changed', out )
            self.assertFalse( os.path.exists( self.path ), legacy )
            self.assertTrue( os.path.exists( self.legacy ), legacy )

    def test_nothing_to_migrate( self ):
        status, out, _asked = self.run_tool( 'migrate' )
        self.assertEqual( status, 0 )
        self.assertIn( 'nothing to migrate', out )
        self.assertFalse( os.path.exists( self.path ) )


class TestCommandLine( unittest.TestCase ):
    """The tool and its old name as real commands."""

    def setUp( self ):
        self.dir = tempfile.mkdtemp( prefix='betasuite-local-cli-' )
        self.addCleanup( shutil.rmtree, self.dir, True )
        self.path = os.path.join( self.dir, 'betaconfig_local.py' )
        self.log = os.path.join( self.dir, 'tool.log' )

    def run_command( self, script, *argv ):
        env = dict( os.environ, **{ bu_local.ENV_VAR: self.path } )
        return subprocess.run( [ sys.executable, script, '--log-file', self.log ] + list( argv ), cwd=self.dir,
                               env=env, capture_output=True, text=True, timeout=120 )

    def test_show_compares_against_the_shipped_values_not_the_file( self ):
        # The file is in effect in the environment, and betaconfig is
        # imported (by the logger) before the file is read.
        with open( self.path, 'w', encoding='utf-8' ) as handle:
            handle.write( "encode_crf = 21\n" )
        out = self.run_command( TOOL, 'show' )
        self.assertEqual( out.returncode, 0, out.stdout + out.stderr )
        self.assertRegex( out.stdout, r"encode_crf\s+21\s+local\s+\(betaconfig.py: 17\)" )

    def test_the_old_name_forwards_with_a_note( self ):
        out = self.run_command( SHIM, 'set', 'encode_crf=21' )
        self.assertEqual( out.returncode, 0, out.stdout + out.stderr )
        self.assertIn( 'machine_config.py is now local_config.py', out.stdout + out.stderr )
        with open( self.path, encoding='utf-8' ) as handle:
            self.assertIn( "\nencode_crf = 21\n", handle.read() )
        with open( self.log, encoding='utf-8' ) as handle:
            self.assertIn( 'machine_config.py is now local_config.py', handle.read() )


if __name__ == '__main__':
    unittest.main()
