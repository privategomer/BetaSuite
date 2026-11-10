"""
test_machine_config.py - the machine config (../resources/betaconfig_machine.py):
what it may set, how it merges into betaconfig, how validation reports
it, and the editor setup and tune_machine write it with.
"""

import ast
import copy
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import betaconfig
import betautils_config as bu_config
import betautils_machine as bu_machine

REPO = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )


def _namespace():
    """A betaconfig-shaped namespace with the blocks the nested settings need."""
    return {
        'detector_backend': { 'selected': 'nudenet_v3', 'defaults': { 'nn_batch_size': 2 },
                              'nudenet_v3': { 'nn_batch_size': 4, 'candidate_floor': 0.2 } },
        'person_backend': { 'selected': 'yolo11_seg', 'yolo11_seg': { 'batch_size': 4 } },
        'encode_video_codec': 'libx264', 'encode_preset': 'fast', 'encode_crf': 17,
        'render_workers': 0, 'gpu_enabled': 1,
    }


class TempDir( unittest.TestCase ):

    def setUp( self ):
        self.dir = tempfile.mkdtemp( prefix='betasuite-machine-' )
        self.addCleanup( shutil.rmtree, self.dir, True )
        self.path = os.path.join( self.dir, 'betaconfig_machine.py' )
        saved = bu_machine.LOADED
        self.addCleanup( setattr, bu_machine, 'LOADED', saved )

    def write( self, text ):
        with open( self.path, 'w', encoding='utf-8' ) as handle:
            handle.write( text )

    def apply( self, text, namespace=None ):
        self.write( text )
        namespace = namespace if namespace is not None else _namespace()
        stderr = io.StringIO()
        loaded = bu_machine.apply( namespace, path=self.path, stderr=stderr )
        return loaded, namespace, stderr.getvalue()


class TestWhereItLives( unittest.TestCase ):

    def test_the_default_is_beside_the_models_not_in_the_repo( self ):
        self.assertEqual( bu_machine.default_path(),
                          os.path.normpath( os.path.join( REPO, '..', 'resources', 'betaconfig_machine.py' ) ) )

    def test_the_environment_variable_moves_or_turns_it_off( self ):
        self.assertEqual( bu_machine.configured_path( {} ), bu_machine.default_path() )
        for off in ( 'none', 'NONE', '', 'off' ):
            self.assertIsNone( bu_machine.configured_path( { bu_machine.ENV_VAR: off } ) )
        self.assertEqual( bu_machine.configured_path( { bu_machine.ENV_VAR: '/x/y.py' } ),
                          os.path.abspath( '/x/y.py' ) )

    def test_the_suite_runs_on_the_repo_defaults( self ):
        # tests/__init__.py turns it off before betaconfig is imported, so
        # no machine's file changes what the suite sees.
        self.assertEqual( os.environ.get( bu_machine.ENV_VAR ), 'none' )

    def test_the_repo_ships_the_portable_encoder( self ):
        # The machine that tuned NVENC keeps it in its own file; a fresh
        # checkout must not need an NVIDIA GPU to render.
        with open( os.path.join( REPO, 'betaconfig.py' ), encoding='utf-8' ) as handle:
            tree = ast.parse( handle.read() )
        values = { node.targets[0].id: ast.literal_eval( node.value ) for node in tree.body
                   if isinstance( node, ast.Assign ) and isinstance( node.targets[0], ast.Name )
                   and isinstance( node.value, ( ast.Constant, ast.Dict, ast.List, ast.Tuple ) ) }
        self.assertEqual( ( values['encode_video_codec'], values['encode_preset'], values['encode_crf'] ),
                          ( 'libx264', 'fast', 17 ) )

    def test_betaconfig_applies_the_file_on_import( self ):
        # The hook is betaconfig.py's last line; every reader sees the merge.
        directory = tempfile.mkdtemp( prefix='betasuite-machine-' )
        self.addCleanup( shutil.rmtree, directory, True )
        path = os.path.join( directory, 'm.py' )
        with open( path, 'w', encoding='utf-8' ) as handle:
            handle.write( "encode_video_codec = 'h264_nvenc'\nencode_crf = 21\n"
                          "detector_backend = { 'nudenet_v3': { 'nn_batch_size': 8 } }\n" )
        env = dict( os.environ, **{ bu_machine.ENV_VAR: path } )
        out = subprocess.run( [ sys.executable, '-c',
                                "import betaconfig as c; print(c.encode_video_codec, c.encode_crf, "
                                "c.detector_backend['nudenet_v3']['nn_batch_size'], "
                                "c.detector_backend['nudenet_v3']['candidate_floor'])" ],
                              cwd=REPO, env=env, capture_output=True, text=True, timeout=120 )
        self.assertEqual( out.returncode, 0, out.stderr )
        # candidate_floor is not in the machine file, so it keeps whatever
        # betaconfig.py says: read from the source rather than pinned, since
        # that file is edited by hand while tuning.
        tree = ast.parse( open( os.path.join( REPO, 'betaconfig.py' ), encoding='utf-8' ).read() )
        backends = [ node.value for node in tree.body if isinstance( node, ast.Assign )
                     and getattr( node.targets[0], 'id', None ) == 'detector_backend' ][0]
        floor = ast.literal_eval( backends )['nudenet_v3']['candidate_floor']
        self.assertEqual( out.stdout.split(), [ 'h264_nvenc', '21', '8', str( floor ) ] )


class TestWhatItMaySet( TempDir ):

    def test_hardware_settings_merge_and_the_rest_stays( self ):
        loaded, namespace, stderr = self.apply(
            "render_workers = 6\nencode_video_codec = 'h264_nvenc'\n"
            "detector_backend = { 'nudenet_v3': { 'nn_batch_size': 8 }, 'defaults': { 'nn_batch_size': 1 } }\n"
            "person_backend = { 'yolo11_seg': { 'batch_size': 2 } }\n" )
        self.assertEqual( loaded.problems, [] )
        self.assertEqual( stderr, '' )
        self.assertEqual( namespace['render_workers'], 6 )
        self.assertEqual( namespace['detector_backend']['nudenet_v3'],
                          { 'nn_batch_size': 8, 'candidate_floor': 0.2 } )
        self.assertEqual( namespace['detector_backend']['defaults'], { 'nn_batch_size': 1 } )
        self.assertEqual( namespace['person_backend']['yolo11_seg'], { 'batch_size': 2 } )
        self.assertEqual( loaded.replaced[ ( 'render_workers', ) ], 0 )
        self.assertEqual( loaded.replaced[ ( 'detector_backend', 'nudenet_v3', 'nn_batch_size' ) ], 4 )
        self.assertEqual( loaded.set_names(), [ 'detector_backend', 'encode_video_codec',
                                                'person_backend', 'render_workers' ] )

    def test_a_tuning_setting_is_refused_and_named( self ):
        loaded, namespace, stderr = self.apply( "default_min_prob = 0.3\nrender_workers = 3\n" )
        self.assertNotIn( 'default_min_prob', namespace )
        self.assertEqual( namespace['render_workers'], 3, "the allowed line still applies" )
        self.assertEqual( len( loaded.problems ), 1 )
        self.assertIn( 'line 1', loaded.problems[0] )
        self.assertIn( 'belongs in betaconfig.py', loaded.problems[0] )
        self.assertIn( 'skipped', stderr )

    def test_a_typo_gets_a_suggestion( self ):
        loaded, _namespace, _stderr = self.apply( "render_worker = 3\n" )
        self.assertIn( 'did you mean render_workers?', loaded.problems[0] )

    def test_only_the_batch_size_inside_a_backend( self ):
        loaded, namespace, _stderr = self.apply(
            "detector_backend = { 'nudenet_v3': { 'nn_batch_size': 8, 'candidate_floor': 0.1 } }\n" )
        self.assertEqual( namespace['detector_backend']['nudenet_v3']['nn_batch_size'], 8 )
        self.assertEqual( namespace['detector_backend']['nudenet_v3']['candidate_floor'], 0.2 )
        self.assertEqual( len( loaded.problems ), 1 )
        self.assertIn( "['candidate_floor'] is not a machine setting", loaded.problems[0] )

    def test_an_unknown_backend_is_refused( self ):
        loaded, namespace, _stderr = self.apply( "detector_backend = { 'nudenet_v9': { 'nn_batch_size': 8 } }\n" )
        self.assertNotIn( 'nudenet_v9', namespace['detector_backend'] )
        self.assertIn( "not a backend betaconfig.py has", loaded.problems[0] )

    def test_a_nested_setting_must_be_a_dict( self ):
        loaded, _namespace, _stderr = self.apply( "detector_backend = 8\n" )
        self.assertIn( 'must be a dict', loaded.problems[0] )
        loaded, _namespace, _stderr = self.apply( "detector_backend = { 'nudenet_v3': 8 }\n" )
        self.assertIn( 'must be a dict', loaded.problems[0] )

    def test_nothing_is_executed( self ):
        marker = os.path.join( self.dir, 'ran' )
        loaded, namespace, _stderr = self.apply(
            "render_workers = __import__('os').mkdir(%r)\nimport os\nrender_workers += 1\n"%( marker, ) )
        self.assertFalse( os.path.exists( marker ) )
        self.assertEqual( namespace['render_workers'], 0 )
        self.assertEqual( len( loaded.problems ), 3 )
        self.assertIn( 'plain value', loaded.problems[0] )
        self.assertIn( 'only `name = value`', loaded.problems[1] )

    def test_a_syntax_error_applies_nothing( self ):
        loaded, namespace, _stderr = self.apply( "render_workers = 3\nencode_crf = (\n" )
        self.assertEqual( namespace['render_workers'], 0 )
        self.assertIn( 'nothing in the file was applied', loaded.problems[0] )

    def test_a_missing_file_is_not_a_problem( self ):
        namespace = _namespace()
        loaded = bu_machine.apply( namespace, path=os.path.join( self.dir, 'absent.py' ) )
        self.assertFalse( loaded.exists )
        self.assertEqual( ( loaded.applied, loaded.problems ), ( {}, [] ) )
        self.assertIn( 'none at', loaded.describe() )
        self.assertIs( bu_machine.LOADED, loaded )

    def test_turned_off_reads_nothing( self ):
        self.write( "render_workers = 5\n" )
        namespace = _namespace()
        loaded = bu_machine.apply( namespace, environ={ bu_machine.ENV_VAR: 'none' } )
        self.assertIsNone( loaded.path )
        self.assertEqual( namespace['render_workers'], 0 )
        self.assertIn( 'off', loaded.describe() )

    def test_describe_names_every_setting( self ):
        loaded, _namespace, _stderr = self.apply(
            "encode_crf = 21\ndetector_backend = { 'nudenet_v3': { 'nn_batch_size': 8 } }\n" )
        self.assertIn( "encode_crf=21", loaded.describe() )
        self.assertIn( "detector_backend['nudenet_v3']['nn_batch_size']=8", loaded.describe() )


class TestEncoderScaleWarning( TempDir ):

    def test_switching_family_without_quality_warns( self ):
        loaded, _namespace, _stderr = self.apply( "encode_video_codec = 'h264_nvenc'\n" )
        warning = bu_machine.encoder_scale_warning( loaded )
        self.assertIn( "NVENC's cq", warning )

    def test_switching_with_quality_or_within_a_family_is_fine( self ):
        loaded, _n, _s = self.apply( "encode_video_codec = 'h264_nvenc'\nencode_crf = 21\n" )
        self.assertIsNone( bu_machine.encoder_scale_warning( loaded ) )
        loaded, _n, _s = self.apply( "encode_video_codec = 'libx265'\n" )
        self.assertIsNone( bu_machine.encoder_scale_warning( loaded ) )
        namespace = dict( _namespace(), encode_video_codec='h264_nvenc' )
        loaded, _n, _s = self.apply( "encode_video_codec = 'hevc_nvenc'\n", namespace )
        self.assertIsNone( bu_machine.encoder_scale_warning( loaded ) )
        self.assertIsNone( bu_machine.encoder_scale_warning( None ) )


class TestValidation( TempDir ):

    def setUp( self ):
        super().setUp()
        self.saved = { name: copy.deepcopy( getattr( betaconfig, name ) )
                       for name in ( 'detector_backend', 'render_workers', 'encode_video_codec',
                                     'encode_preset', 'encode_crf' ) }
        self.addCleanup( self._restore )

    def _restore( self ):
        for name, value in self.saved.items():
            setattr( betaconfig, name, value )

    def test_a_skipped_line_stops_a_real_run( self ):
        self.apply( "default_min_prob = 0.3\n", vars( betaconfig ) )
        errors = bu_config._collect_validation_errors()
        self.assertTrue( any( 'default_min_prob is not a machine setting' in e for e in errors ), errors )

    def test_validate_config_says_which_file_set_a_bad_value( self ):
        self.apply( "render_workers = -2\n", vars( betaconfig ) )
        with self.assertLogs( 'betasuite', level='INFO' ) as logs, self.assertRaises( SystemExit ):
            bu_config.validate_config()
        text = '\n'.join( logs.output )
        self.assertIn( 'machine config: %s sets render_workers=-2'%( self.path, ), text )
        self.assertIn( 'render_workers must be a non-negative integer', text )
        self.assertIn( "%s sets render_workers; an error naming one of those"%( self.path, ), text )

    def test_the_encoder_scale_warning_reaches_the_log( self ):
        self.apply( "encode_video_codec = 'h264_nvenc'\nencode_preset = 'p5'\n", vars( betaconfig ) )
        self.assertTrue( any( "NVENC's cq" in w for w in bu_config._collect_validation_warnings() ) )


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
        self.namespace = _namespace()
        self.template = bu_machine.template_text( self.namespace )

    def entries( self, text ):
        entries, problems = bu_machine.parse( text )
        self.assertEqual( problems, [] )
        return { name: value for name, value, _line in entries }

    def test_the_template_lists_every_setting_and_sets_none( self ):
        self.assertEqual( self.entries( self.template ), {} )
        for name in bu_machine.SETTING_NAMES:
            self.assertRegex( self.template, r'(?m)^# %s = '%( name, ) )
        self.assertIn( "# encode_video_codec = 'libx264'", self.template )
        self.assertIn( "# detector_backend = { 'defaults': { 'nn_batch_size': 2 }, "
                       "'nudenet_v3': { 'nn_batch_size': 4 } }", self.template,
                       "only the machine part of each backend block" )
        self.assertNotIn( '—', self.template )

    def test_set_replaces_the_commented_line_in_place( self ):
        text = bu_machine.edit_text( self.template, { ( 'encode_crf', ): 21 }, namespace=self.namespace,
                                     comment='measured' )
        self.assertEqual( self.entries( text ), { 'encode_crf': 21 } )
        self.assertIn( '\nencode_crf = 21  # measured\n', text )
        self.assertNotIn( '# encode_crf = 17', text )
        self.assertEqual( len( text.splitlines() ), len( self.template.splitlines() ) )

    def test_nested_settings_merge( self ):
        text = bu_machine.edit_text( self.template, {
            ( 'detector_backend', 'nudenet_v3', 'nn_batch_size' ): 8 }, namespace=self.namespace )
        text = bu_machine.edit_text( text, {
            ( 'detector_backend', 'defaults', 'nn_batch_size' ): 1 }, namespace=self.namespace )
        self.assertEqual( self.entries( text ), { 'detector_backend': {
            'nudenet_v3': { 'nn_batch_size': 8 }, 'defaults': { 'nn_batch_size': 1 } } } )
        text = bu_machine.edit_text( text, removals=[ ( 'detector_backend', 'nudenet_v3', 'nn_batch_size' ) ],
                                     namespace=self.namespace )
        self.assertEqual( self.entries( text ), { 'detector_backend': { 'defaults': { 'nn_batch_size': 1 } } } )
        text = bu_machine.edit_text( text, removals=[ ( 'detector_backend', 'defaults', 'nn_batch_size' ) ],
                                     namespace=self.namespace )
        self.assertEqual( self.entries( text ), {} )
        self.assertIn( "# detector_backend = {", text )

    def test_unset_goes_back_to_the_commented_default( self ):
        text = bu_machine.edit_text( self.template, { ( 'render_workers', ): 6 }, namespace=self.namespace )
        text = bu_machine.edit_text( text, removals=[ ( 'render_workers', ) ], namespace=self.namespace )
        self.assertEqual( text, self.template )

    def test_a_persons_own_lines_and_comments_survive( self ):
        text = ( "# my laptop, tuned by hand\nrender_workers = 4  # keep it cool\n\n"
                 "encode_crf = 21\n" )
        edited = bu_machine.edit_text( text, { ( 'encode_crf', ): 19, ( 'encode_preset', ): 'p5' } )
        self.assertEqual( edited, "# my laptop, tuned by hand\nrender_workers = 4  # keep it cool\n\n"
                                  "encode_crf = 19\nencode_preset = 'p5'\n" )
        edited = bu_machine.edit_text( edited, { ( 'render_workers', ): 5 } )
        self.assertIn( "render_workers = 5  # keep it cool\n", edited, "a trailing comment is kept" )

    def test_a_multiline_value_is_replaced_whole( self ):
        text = "detector_backend = {\n    'nudenet_v3': { 'nn_batch_size': 4 },\n}\nrender_workers = 2\n"
        edited = bu_machine.edit_text( text, { ( 'detector_backend', 'nudenet_v3', 'nn_batch_size' ): 8 } )
        self.assertEqual( edited, "detector_backend = { 'nudenet_v3': { 'nn_batch_size': 8 } }\n"
                                  "render_workers = 2\n" )

    def test_an_unparseable_file_is_not_touched( self ):
        with self.assertRaises( ValueError ):
            bu_machine.edit_text( "render_workers = (\n", { ( 'render_workers', ): 2 } )


class TestTool( unittest.TestCase ):
    """tools/setup/machine_config.py, in-process, against a temp file."""

    def setUp( self ):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'machine_config_tool', os.path.join( REPO, 'tools', 'setup', 'machine_config.py' ) )
        self.tool = importlib.util.module_from_spec( spec )
        spec.loader.exec_module( self.tool )
        self.dir = tempfile.mkdtemp( prefix='betasuite-machine-tool-' )
        self.addCleanup( shutil.rmtree, self.dir, True )
        self.path = os.path.join( self.dir, 'betaconfig_machine.py' )
        self.log = os.path.join( self.dir, 'tool.log' )
        saved = bu_machine.LOADED
        self.addCleanup( setattr, bu_machine, 'LOADED', saved )
        self.addCleanup( os.environ.__setitem__, bu_machine.ENV_VAR, os.environ[ bu_machine.ENV_VAR ] )
        self.saved = { name: copy.deepcopy( getattr( betaconfig, name, None ) )
                       for name in bu_machine.SETTING_NAMES }
        self.addCleanup( self._restore )

    def _restore( self ):
        for name, value in self.saved.items():
            setattr( betaconfig, name, value )

    def run_tool( self, *argv, stdin_tty=False, answer='' ):
        stdout = io.StringIO()
        with mock.patch( 'sys.stdout', stdout ), mock.patch( 'sys.stdin' ) as stdin, \
                mock.patch( 'builtins.input', return_value=answer ) as asked:
            stdin.isatty.return_value = stdin_tty
            status = self.tool.main( [ '--path', self.path, '--log-file', self.log,
                                       '--log-level', 'trace' ] + list( argv ) )
        return status, stdout.getvalue(), asked

    def text( self ):
        with open( self.path, encoding='utf-8' ) as handle:
            return handle.read()

    def test_set_writes_a_valid_change( self ):
        status, out, _asked = self.run_tool( 'set', 'encode_video_codec=h264_nvenc', 'encode_preset=p5',
                                             'encode_crf=21', 'detector_backend.nudenet_v3.nn_batch_size=8' )
        self.assertEqual( status, 0, out )
        entries, problems = bu_machine.parse( self.text() )
        self.assertEqual( problems, [] )
        self.assertEqual( { n: v for n, v, _l in entries }, {
            'encode_video_codec': 'h264_nvenc', 'encode_preset': 'p5', 'encode_crf': 21,
            'detector_backend': { 'nudenet_v3': { 'nn_batch_size': 8 } } } )
        self.assertEqual( betaconfig.encode_video_codec, self.saved['encode_video_codec'],
                          "checking a change leaves the live config as it was" )

    def test_set_refuses_what_betatv_would_stop_on( self ):
        self.run_tool( 'set', 'render_workers=3' )
        before = self.text()
        for bad in ( 'render_workers=-1', 'encode_preset=slow1', 'default_min_prob=0.3',
                     'detector_backend.nudenet_v3.candidate_floor=0.1' ):
            status, out, _asked = self.run_tool( 'set', bad )
            self.assertEqual( status, 1, bad )
            self.assertIn( 'not written', out )
            self.assertEqual( self.text(), before, bad )

    def test_bad_names_are_usage_errors( self ):
        for bad in ( 'render_workers', 'detector_backend=4', 'render_workers.x=1', 'encode_codec=x y' ):
            status, _out, _asked = self.run_tool( 'set', bad )
            self.assertEqual( status, 2, bad )

    def test_show_and_check( self ):
        self.run_tool( 'set', 'encode_crf=21' )
        status, out, _asked = self.run_tool( 'show' )
        self.assertEqual( status, 0 )
        self.assertRegex( out, r"encode_crf\s+21\s+machine\s+\(betaconfig.py: 17\)" )
        self.assertRegex( out, r"render_workers\s+0\s+betaconfig.py" )
        with open( self.path, 'a', encoding='utf-8' ) as handle:
            handle.write( "default_min_prob = 0.3\n" )
        self.assertEqual( self.run_tool( 'check' )[0], 1 )
        self.assertEqual( self.run_tool( 'show' )[0], 1 )

    def test_unset( self ):
        self.run_tool( 'set', 'render_workers=3' )
        status, _out, _asked = self.run_tool( 'unset', 'render_workers' )
        self.assertEqual( status, 0 )
        self.assertEqual( bu_machine.parse( self.text() )[0], [] )

    def test_setup_creates_the_file_and_sets_the_gpu_flag_without_asking( self ):
        status, out, asked = self.run_tool( 'setup', '--gpu-enabled', '0', stdin_tty=True )
        self.assertEqual( status, 0, out )
        self.assertFalse( asked.called, "a value from betaconfig.py is setup's to change" )
        self.assertIn( "\ngpu_enabled = 0  # set by setup\n", self.text() )
        self.assertIn( '# render_workers = 0', self.text(), "created from the template" )
        status, out, asked = self.run_tool( 'setup', '--gpu-enabled', '0', stdin_tty=True )
        self.assertIn( 'matches', out )

    def test_setup_asks_before_overriding_the_files_own_value( self ):
        self.run_tool( 'set', 'gpu_enabled=0' )
        status, _out, asked = self.run_tool( 'setup', '--gpu-enabled', '1', stdin_tty=True, answer='n' )
        self.assertTrue( asked.called )
        self.assertIn( 'gpu_enabled = 0', self.text() )
        status, _out, asked = self.run_tool( 'setup', '--gpu-enabled', '1', '--yes', stdin_tty=True )
        self.assertFalse( asked.called )
        self.assertIn( 'gpu_enabled = 1', self.text() )

    def test_the_command_line_compares_against_the_repo_not_the_file( self ):
        # As a real command: the file is in effect in the environment, and
        # betaconfig is imported (by the logger) before the file is read.
        with open( self.path, 'w', encoding='utf-8' ) as handle:
            handle.write( "encode_crf = 21\n" )
        env = dict( os.environ, **{ bu_machine.ENV_VAR: self.path } )
        out = subprocess.run( [ sys.executable, os.path.join( REPO, 'tools', 'setup', 'machine_config.py' ),
                                '--log-file', self.log, 'show' ], cwd=self.dir, env=env,
                              capture_output=True, text=True, timeout=120 )
        self.assertEqual( out.returncode, 0, out.stdout + out.stderr )
        self.assertRegex( out.stdout, r"encode_crf\s+21\s+machine\s+\(betaconfig.py: 17\)" )

    def test_the_log_file_gets_everything( self ):
        self.run_tool( 'set', 'render_workers=3' )
        with open( self.log, encoding='utf-8' ) as handle:
            log = handle.read()
        self.assertIn( 'command:', log )
        self.assertIn( 'set render_workers = 3', log )


if __name__ == '__main__':
    unittest.main()
