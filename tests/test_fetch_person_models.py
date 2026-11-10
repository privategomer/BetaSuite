"""
test_fetch_person_models.py - setup's export of the person models
(YOLO11-seg and RF-DETR-Seg).

What each test prevents:
  - PyTorch reaching the runtime .venv: the export must install into its
    own environment, CPU-only torch from the PyTorch index first
  - a leftover multi-GB export environment after a failure
  - an unpinned or tampered .pt/.pth being exported, or an export that
    fetches anything else while it runs
  - a failed export reported as success, and --no-export still exporting
  - an RF-DETR size the adapter offers with no export, no pinned weights
    or no verify_env check, or exported with YOLO's script
  - a person backend nobody chose being exported anyway, or its package
    installed into the export environment; the default leaving out
    yolo11_seg, or a half-exported backend left unfinished
  - a typo in --person-backends or at the prompt silently exporting
    nothing, and the prompt blocking setup when there is no terminal
"""

import importlib.util
import os
import shutil
import tempfile
import unittest
from unittest import mock

REPO_ROOT = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )


def _load():
    spec = importlib.util.spec_from_file_location(
        'fetch_models_under_test', os.path.join( REPO_ROOT, 'tools', 'setup', 'fetch_models.py' ) )
    module = importlib.util.module_from_spec( spec )
    spec.loader.exec_module( module )
    return module


def _fake_onnx( path ):
    with open( path, 'wb' ) as handle:
        handle.write( b'\x08' + b'\x00' * 2_000_000 )


class TestExport( unittest.TestCase ):

    def setUp( self ):
        self.fm = _load()
        self.tmp = tempfile.mkdtemp( prefix='betasuite-fetch-' )
        self.addCleanup( shutil.rmtree, self.tmp, True )
        self.model_dir = os.path.join( self.tmp, 'model' )
        os.makedirs( self.model_dir )
        self.fm.MODEL_DIR = self.model_dir
        self.log = mock.Mock()
        self.workdirs = []
        real_mkdtemp = tempfile.mkdtemp

        def tracked_mkdtemp( *args, **kwargs ):
            path = real_mkdtemp( *args, dir=self.tmp, **{ k: v for k, v in kwargs.items() if k != 'dir' } )
            self.workdirs.append( path )
            return path
        patcher = mock.patch.object( self.fm.tempfile, 'mkdtemp', side_effect=tracked_mkdtemp )
        patcher.start()
        self.addCleanup( patcher.stop )

    def missing( self, *keys ):
        return [ ( self.fm.MODELS[key], os.path.join( self.model_dir, self.fm.MODELS[key]['file'] ) )
                 for key in keys ]

    def test_the_export_env_gets_cpu_torch_first_then_ultralytics_and_never_the_venv( self ):
        commands = []
        with mock.patch.object( self.fm, 'run_logged', side_effect=lambda d, c, log: commands.append( c ) ):
            python = self.fm.build_export_env( self.tmp, '/usr/bin/uv', self.log, { 'ultralytics', 'rfdetr' } )
        self.assertTrue( python.startswith( os.path.join( self.tmp, 'export-venv' ) ) )
        self.assertEqual( commands[0][:2], [ '/usr/bin/uv', 'venv' ] )
        torch, ultralytics = commands[1], commands[2]
        self.assertIn( '--python', torch )
        self.assertEqual( torch[ torch.index( '--python' ) + 1 ], python )
        self.assertEqual( torch[ torch.index( '--index-url' ) + 1 ], self.fm.EXPORT_TORCH_INDEX )
        self.assertIn( 'torch', torch )
        self.assertTrue( any( package.startswith( 'ultralytics==' ) for package in ultralytics ),
                         "ultralytics is pinned" )
        self.assertTrue( any( package.startswith( 'rfdetr==' ) for package in ultralytics ),
                         "rfdetr is pinned, in the same throwaway environment" )
        self.assertFalse( any( package.startswith( 'torch' ) for package in ultralytics ),
                          "torch comes from the CPU index only" )
        self.assertEqual( ultralytics[ ultralytics.index( '--python' ) + 1 ], python )

    def test_without_uv_it_falls_back_to_venv_and_pip( self ):
        commands = []
        with mock.patch.object( self.fm, 'run_logged', side_effect=lambda d, c, log: commands.append( c ) ):
            python = self.fm.build_export_env( self.tmp, None, self.log, { 'ultralytics' } )
        self.assertEqual( commands[0][1:3], [ '-m', 'venv' ] )
        self.assertEqual( commands[1][:4], [ python, '-m', 'pip', 'install' ] )

    def test_a_successful_export_lands_in_the_model_folder_and_cleans_up( self ):
        def fake_run( command, **kwargs ):
            _fake_onnx( command[4] )
            return mock.Mock( returncode=0, stdout='export success' )
        with mock.patch.object( self.fm, 'build_export_env', return_value='/venv/python' ), \
             mock.patch.object( self.fm, 'download_checked' ) as download, \
             mock.patch.object( self.fm.subprocess, 'run', side_effect=fake_run ):
            failures = self.fm.export_person_models( self.missing( 'yolo11n-seg', 'yolo11s-seg' ), self.log )
        self.assertEqual( failures, 0 )
        self.assertEqual( sorted( os.listdir( self.model_dir ) ), [ 'yolo11n-seg.onnx', 'yolo11s-seg.onnx' ] )
        self.assertEqual( download.call_args_list[0][0][1], self.fm.MODELS['yolo11n-seg']['export']['pt_sha256'] )
        self.assertFalse( any( os.path.exists( path ) for path in self.workdirs ), "export env removed" )

    def test_a_failed_export_is_a_failure_and_still_cleans_up( self ):
        with mock.patch.object( self.fm, 'build_export_env', return_value='/venv/python' ), \
             mock.patch.object( self.fm, 'download_checked' ), \
             mock.patch.object( self.fm.subprocess, 'run',
                                return_value=mock.Mock( returncode=1, stdout='Traceback: boom' ) ):
            failures = self.fm.export_person_models( self.missing( 'yolo11n-seg' ), self.log )
        self.assertEqual( failures, 1 )
        self.assertEqual( os.listdir( self.model_dir ), [] )
        self.assertFalse( any( os.path.exists( path ) for path in self.workdirs ) )

    def test_an_export_that_is_not_a_model_is_refused( self ):
        def fake_run( command, **kwargs ):
            with open( command[4], 'w' ) as handle:
                handle.write( '<html>sign in</html>' )
            return mock.Mock( returncode=0, stdout='' )
        with mock.patch.object( self.fm, 'build_export_env', return_value='/venv/python' ), \
             mock.patch.object( self.fm, 'download_checked' ), \
             mock.patch.object( self.fm.subprocess, 'run', side_effect=fake_run ):
            self.assertEqual( self.fm.export_person_models( self.missing( 'yolo11n-seg' ), self.log ), 1 )
        self.assertEqual( os.listdir( self.model_dir ), [] )

    def test_no_environment_means_every_model_failed( self ):
        with mock.patch.object( self.fm, 'build_export_env', side_effect=RuntimeError( 'no network' ) ):
            self.assertEqual( self.fm.export_person_models( self.missing( 'yolo11n-seg', 'yolo11m-seg' ),
                                                            self.log ), 2 )
        self.assertFalse( any( os.path.exists( path ) for path in self.workdirs ) )

    def test_a_tampered_download_is_refused( self ):
        class Response:
            def __enter__( self ):
                return self
            def __exit__( self, *args ):
                return False
            def read( self ):
                return b'not the weights'
        with mock.patch.object( self.fm.urllib.request, 'urlopen', return_value=Response() ):
            with self.assertRaises( self.fm.NotAModel ):
                self.fm.download_checked( 'https://x/yolo11n-seg.pt', '0' * 64,
                                          os.path.join( self.tmp, 'w.pt' ), self.log )
        self.assertFalse( os.path.exists( os.path.join( self.tmp, 'w.pt' ) ) )

    def test_every_person_model_is_pinned_and_the_variants_match_the_adapter( self ):
        exported = { spec['file'] for spec in self.fm.MODELS.values() if 'export' in spec }
        for spec in self.fm.MODELS.values():
            if 'export' in spec:
                self.assertEqual( len( spec['export']['pt_sha256'] ), 64 )
        from segmenters import rf_detr_seg, yolo11_seg
        offered = { os.path.basename( v['model_path'] )
                    for module in ( yolo11_seg, rf_detr_seg ) for v in module.VARIANTS.values() }
        self.assertEqual( exported, offered )

    def test_each_rf_detr_size_is_exported_by_rfdetr_from_its_own_weights( self ):
        from segmenters import rf_detr_seg
        specs = { spec['file']: spec for spec in self.fm.MODELS.values() if 'export' in spec }
        urls = set()
        for name, variant in rf_detr_seg.VARIANTS.items():
            export = specs[ os.path.basename( variant['model_path'] ) ]['export']
            self.assertEqual( export['tool'], 'rfdetr' )
            self.assertEqual( export['arg'], 'RFDETRSeg' + name.capitalize() )
            self.assertTrue( export['pt_url'].startswith( 'https://repo.roboflow.com/rfdetr/rf-detr-seg-' ) )
            urls.add( export['pt_url'] )
            command = self.fm.export_command( '/venv/python', export, '/w/x.pth', '/w/out.onnx' )
            self.assertEqual( command, [ '/venv/python', '-c', self.fm.EXPORT_SCRIPTS['rfdetr'], '/w/x.pth',
                                         '/w/out.onnx', export['arg'] ] )
        self.assertEqual( len( urls ), len( rf_detr_seg.VARIANTS ), "no two sizes share weights" )
        yolo = self.fm.MODELS['yolo11s-seg']['export']
        self.assertEqual( self.fm.export_command( '/p', yolo, 'a.pt', 'b.onnx' ),
                          [ '/p', '-c', self.fm.EXPORT_SCRIPTS['ultralytics'], 'a.pt', 'b.onnx', '640' ] )

    def test_the_export_scripts_compile_and_rf_detr_keeps_a_dynamic_batch( self ):
        for name, script in self.fm.EXPORT_SCRIPTS.items():
            compile( script, name, 'exec' )
        self.assertIn( 'dynamic_batch=True', self.fm.EXPORT_SCRIPTS['rfdetr'] )
        self.assertIn( 'pretrain_weights=pt', self.fm.EXPORT_SCRIPTS['rfdetr'],
                       "exports the pinned file, not whatever rfdetr would download" )

    def test_the_export_runs_offline( self ):
        envs = []

        def fake_run( command, **kwargs ):
            envs.append( kwargs['env'] )
            _fake_onnx( command[4] )
            return mock.Mock( returncode=0, stdout='' )
        with mock.patch.object( self.fm, 'build_export_env', return_value='/venv/python' ), \
             mock.patch.object( self.fm, 'download_checked' ), \
             mock.patch.object( self.fm.subprocess, 'run', side_effect=fake_run ):
            self.assertEqual( self.fm.export_person_models( self.missing( 'rf-detr-seg-small' ), self.log ), 0 )
        for key in ( 'YOLO_OFFLINE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE' ):
            self.assertEqual( envs[0].get( key ), '1', key )
        self.assertEqual( os.listdir( self.model_dir ), [ 'rf-detr-seg-small.onnx' ] )

    def test_verify_env_checks_every_person_model( self ):
        spec = importlib.util.spec_from_file_location(
            'verify_env_under_test', os.path.join( REPO_ROOT, 'tools', 'setup', 'verify_env.py' ) )
        verify_env = importlib.util.module_from_spec( spec )
        spec.loader.exec_module( verify_env )
        from segmenters import rf_detr_seg, yolo11_seg
        checked = set( verify_env.MODELS.values() )
        for module in ( yolo11_seg, rf_detr_seg ):
            for variant in module.VARIANTS.values():
                self.assertIn( variant['model_path'], checked )

    def test_no_export_skips_the_export( self ):
        for spec in self.fm.MODELS.values():
            if 'export' not in spec and 'manual_url' not in spec:
                _fake_onnx( os.path.join( self.model_dir, spec['file'] ) )
        with mock.patch.object( self.fm, 'export_person_models' ) as export, \
             mock.patch.object( self.fm.sys, 'argv', [ 'fetch_models.py', '--no-export',
                                                       '--log-file', os.path.join( self.tmp, 'f.log' ) ] ), \
             mock.patch.object( self.fm, 'relaunch_in_venv' ):
            self.fm.main()
        export.assert_not_called()
        with mock.patch.object( self.fm, 'export_person_models', return_value=0 ) as export, \
             mock.patch.object( self.fm.sys, 'argv', [ 'fetch_models.py', '--uv', '/x/uv', '--person-backends', 'all',
                                                       '--log-file', os.path.join( self.tmp, 'g.log' ) ] ), \
             mock.patch.object( self.fm, 'relaunch_in_venv' ), \
             mock.patch.object( self.fm.os.path, 'isfile', return_value=True ):
            self.assertEqual( self.fm.main(), 0 )
        self.assertEqual( len( export.call_args[0][0] ),
                          sum( 1 for spec in self.fm.MODELS.values() if 'export' in spec ) )
        self.assertEqual( export.call_args[0][2], '/x/uv' )


class TestPersonBackendChoice( unittest.TestCase ):

    def setUp( self ):
        self.fm = _load()
        self.tmp = tempfile.mkdtemp( prefix='betasuite-fetch-' )
        self.addCleanup( shutil.rmtree, self.tmp, True )
        self.model_dir = os.path.join( self.tmp, 'model' )
        os.makedirs( self.model_dir )
        self.fm.MODEL_DIR = self.model_dir
        self.log = mock.Mock()
        # 320n present, so main() reaches the person models without the network.
        _fake_onnx( os.path.join( self.model_dir, self.fm.MODELS['320n']['file'] ) )

    def files_of( self, backend ):
        return [ spec['file'] for spec in self.fm.MODELS.values() if spec.get( 'person_backend' ) == backend ]

    def run_main( self, *argv, interactive=False, answers=() ):
        """main() with export_person_models mocked; the files it was asked to export, or None."""
        replies = iter( answers )
        with mock.patch.object( self.fm, 'export_person_models', return_value=0 ) as export, \
             mock.patch.object( self.fm.sys, 'argv', [ 'fetch_models.py', '--log-file',
                                                       os.path.join( self.tmp, 'f.log' ) ] + list( argv ) ), \
             mock.patch.object( self.fm, 'relaunch_in_venv' ), \
             mock.patch.object( self.fm.sys.stdin, 'isatty', return_value=interactive ), \
             mock.patch( 'builtins.input', side_effect=lambda prompt: next( replies ) ):
            self.assertEqual( self.fm.main(), 0 )
        if not export.called:
            return None
        return sorted( spec['file'] for spec, _dest in export.call_args[0][0] )

    def test_every_person_model_names_the_backend_whose_adapter_loads_it( self ):
        import betautils_person
        self.assertEqual( set( self.fm.PERSON_BACKENDS ), set( betautils_person.registered_backend_names() ) )
        self.assertEqual( self.fm.DEFAULT_PERSON_BACKENDS, [ betautils_person.DEFAULT_BACKEND ],
                          "setup's default is the backend the app runs by default" )
        for spec in self.fm.MODELS.values():
            if 'export' not in spec:
                continue
            module = betautils_person.get_segmenter( spec['person_backend'] )
            self.assertIn( spec['file'], { os.path.basename( v['model_path'] ) for v in module.VARIANTS.values() },
                           spec['file'] )
            self.assertGreater( spec['approx_mb'], 0 )
        self.assertEqual( self.fm.PERSON_BACKENDS, [ 'yolo11_seg', 'rf_detr_seg' ] )

    def test_parse_names_all_and_none( self ):
        self.assertEqual( self.fm.parse_person_backends( 'rf_detr_seg' ), [ 'rf_detr_seg' ] )
        self.assertEqual( self.fm.parse_person_backends( 'rf_detr_seg, yolo11_seg' ), [ 'yolo11_seg', 'rf_detr_seg' ] )
        self.assertEqual( self.fm.parse_person_backends( 'all' ), [ 'yolo11_seg', 'rf_detr_seg' ] )
        self.assertEqual( self.fm.parse_person_backends( 'none' ), [] )

    def test_parse_refuses_a_typo_naming_it_and_the_choices( self ):
        for text in ( 'yolo11_seg,rfdetr', '', 'all,none' ):
            with self.assertRaises( ValueError ) as caught:
                self.fm.parse_person_backends( text )
            self.assertIn( 'yolo11_seg, rf_detr_seg', str( caught.exception ) )
        with self.assertRaises( ValueError ) as caught:
            self.fm.parse_person_backends( 'yolo11_seg,rfdetr' )
        self.assertIn( 'unknown rfdetr', str( caught.exception ) )

    def test_default_is_yolo_plus_a_backend_already_partly_present( self ):
        self.assertEqual( self.fm.default_person_backends( set() ), [ 'yolo11_seg' ] )
        self.assertEqual( self.fm.default_person_backends( { 'rf-detr-seg-small.onnx' } ),
                          [ 'yolo11_seg', 'rf_detr_seg' ] )
        self.assertEqual( self.fm.default_person_backends( { 'yolo11s-seg.onnx' } ), [ 'yolo11_seg' ] )

    def test_a_typed_answer_takes_numbers_names_or_blank( self ):
        self.assertEqual( self.fm.choose_person_backends( '', set() ), [ 'yolo11_seg' ] )
        self.assertEqual( self.fm.choose_person_backends( '2', set() ), [ 'rf_detr_seg' ] )
        self.assertEqual( self.fm.choose_person_backends( '1 2', set() ), [ 'yolo11_seg', 'rf_detr_seg' ] )
        self.assertEqual( self.fm.choose_person_backends( '2,yolo11_seg', set() ), [ 'yolo11_seg', 'rf_detr_seg' ] )
        self.assertEqual( self.fm.choose_person_backends( 'none', set() ), [] )
        for answer in ( '0', '3', 'yolo' ):
            with self.assertRaises( ValueError ):
                self.fm.choose_person_backends( answer, set() )

    def test_without_a_terminal_it_takes_the_default_and_never_asks( self ):
        ask = mock.Mock( side_effect=AssertionError( 'asked' ) )
        self.assertEqual( self.fm.ask_person_backends( set(), self.log, ask=ask, interactive=False ),
                          [ 'yolo11_seg' ] )
        ask.assert_not_called()

    def test_a_bad_answer_is_asked_again_and_end_of_input_takes_the_default( self ):
        ask = mock.Mock( side_effect=[ 'rfdetr', '2' ] )
        self.assertEqual( self.fm.ask_person_backends( set(), self.log, ask=ask, interactive=True ),
                          [ 'rf_detr_seg' ] )
        self.assertEqual( ask.call_count, 2 )
        self.log.warning.assert_called_once()
        ask = mock.Mock( side_effect=EOFError )
        self.assertEqual( self.fm.ask_person_backends( set(), self.log, ask=ask, interactive=True ),
                          [ 'yolo11_seg' ] )

    def test_the_menu_shows_each_backend_its_size_and_what_is_present( self ):
        lines = self.fm.person_backend_menu( { 'rf-detr-seg-nano.onnx' } )
        self.assertEqual( len( lines ), 2 )
        self.assertIn( 'yolo11_seg', lines[0] )
        self.assertIn( 'about 143 MB', lines[0] )
        self.assertNotIn( 'present', lines[0] )
        self.assertIn( 'about 375 MB, 1 of 3 present', lines[1] )

    def test_the_export_environment_gets_only_the_chosen_tools_packages( self ):
        commands = []
        with mock.patch.object( self.fm, 'run_logged', side_effect=lambda d, c, log: commands.append( c ) ):
            self.fm.build_export_env( self.tmp, '/usr/bin/uv', self.log, { 'ultralytics' } )
        packages = commands[2]
        self.assertTrue( any( package.startswith( 'ultralytics==' ) for package in packages ) )
        self.assertFalse( any( package.startswith( 'rfdetr' ) for package in packages ),
                          "rfdetr is not installed when no RF-DETR model is exported" )
        self.assertIn( 'onnx', packages )
        commands.clear()
        with mock.patch.object( self.fm, 'run_logged', side_effect=lambda d, c, log: commands.append( c ) ):
            self.fm.build_export_env( self.tmp, '/usr/bin/uv', self.log, { 'rfdetr' } )
        self.assertFalse( any( package.startswith( 'ultralytics' ) for package in commands[2] ) )
        self.assertTrue( any( package.startswith( 'rfdetr==' ) for package in commands[2] ) )

    def test_export_person_models_builds_the_environment_for_the_tools_it_exports( self ):
        missing = [ ( self.fm.MODELS['rf-detr-seg-nano'], os.path.join( self.model_dir, 'rf-detr-seg-nano.onnx' ) ) ]
        with mock.patch.object( self.fm, 'build_export_env', side_effect=RuntimeError( 'stop' ) ) as build:
            self.fm.export_person_models( missing, self.log )
        self.assertEqual( build.call_args[0][3], { 'rfdetr' } )

    def test_unattended_exports_only_the_default_backend( self ):
        self.assertEqual( self.run_main( '--yes' ), sorted( self.files_of( 'yolo11_seg' ) ) )
        self.assertEqual( self.run_main(), sorted( self.files_of( 'yolo11_seg' ) ), "no terminal: default" )
        self.assertEqual( self.run_main( '--yes', interactive=True, answers=[ 'rf_detr_seg' ] ),
                          sorted( self.files_of( 'yolo11_seg' ) ), "--yes never asks, even at a terminal" )

    def test_the_flag_picks_the_backends_and_none_exports_nothing( self ):
        self.assertEqual( self.run_main( '--person-backends', 'rf_detr_seg' ), sorted( self.files_of( 'rf_detr_seg' ) ) )
        self.assertEqual( self.run_main( '--person-backends', 'all', '--yes' ),
                          sorted( self.files_of( 'yolo11_seg' ) + self.files_of( 'rf_detr_seg' ) ) )
        self.assertIsNone( self.run_main( '--person-backends', 'none' ) )

    def test_a_bad_flag_value_stops_before_anything_runs( self ):
        with mock.patch.object( self.fm, 'export_person_models' ) as export, \
             mock.patch.object( self.fm.sys, 'argv', [ 'fetch_models.py', '--person-backends', 'rfdetr',
                                                       '--log-file', os.path.join( self.tmp, 'f.log' ) ] ), \
             mock.patch.object( self.fm, 'relaunch_in_venv' ), \
             mock.patch( 'sys.stderr' ):
            with self.assertRaises( SystemExit ) as caught:
                self.fm.main()
        self.assertEqual( caught.exception.code, 2 )
        export.assert_not_called()

    def test_at_a_terminal_the_answer_picks_the_backends( self ):
        self.assertEqual( self.run_main( interactive=True, answers=[ '2' ] ), sorted( self.files_of( 'rf_detr_seg' ) ) )
        self.assertEqual( self.run_main( interactive=True, answers=[ '' ] ), sorted( self.files_of( 'yolo11_seg' ) ) )

    def test_a_present_backend_is_not_exported_again_and_a_partial_one_is_finished( self ):
        for name in self.files_of( 'yolo11_seg' ) + [ 'rf-detr-seg-small.onnx' ]:
            _fake_onnx( os.path.join( self.model_dir, name ) )
        self.assertEqual( self.run_main( '--yes' ), [ 'rf-detr-seg-medium.onnx', 'rf-detr-seg-nano.onnx' ] )
        for name in self.files_of( 'rf_detr_seg' ):
            _fake_onnx( os.path.join( self.model_dir, name ) )
        self.assertIsNone( self.run_main( interactive=True, answers=[ AssertionError ] ),
                           "nothing missing: no prompt, no export" )


if __name__ == '__main__':
    unittest.main()
