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
            python = self.fm.build_export_env( self.tmp, '/usr/bin/uv', self.log )
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
            python = self.fm.build_export_env( self.tmp, None, self.log )
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
             mock.patch.object( self.fm.sys, 'argv', [ 'fetch_models.py', '--uv', '/x/uv',
                                                       '--log-file', os.path.join( self.tmp, 'g.log' ) ] ), \
             mock.patch.object( self.fm, 'relaunch_in_venv' ), \
             mock.patch.object( self.fm.os.path, 'isfile', return_value=True ):
            self.assertEqual( self.fm.main(), 0 )
        self.assertEqual( len( export.call_args[0][0] ),
                          sum( 1 for spec in self.fm.MODELS.values() if 'export' in spec ) )
        self.assertEqual( export.call_args[0][2], '/x/uv' )


if __name__ == '__main__':
    unittest.main()
