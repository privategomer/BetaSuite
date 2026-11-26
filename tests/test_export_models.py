"""
test_export_models.py - the maintainers' export of the person models
(tools/release/export_models.py), from their pinned upstream weights.

What each test prevents:
  - PyTorch reaching the runtime .venv: the export installs into its
    own environment, CPU-only torch from the PyTorch index first
  - a leftover multi-GB export environment after a failure
  - an unpinned or tampered .pt/.pth being exported, or an export that
    fetches anything else while it runs
  - a failed export reported as success, or a web page kept as a model
  - an RF-DETR size exported with YOLO's script or from another size's
    weights; a person model in the catalog the tool cannot export
  - the export environment getting a package no chosen model needs
"""

import hashlib
import importlib.util
import os
import shutil
import tempfile
import unittest
from unittest import mock

import betautils_models as bu_models

REPO_ROOT = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )


def _load():
    spec = importlib.util.spec_from_file_location(
        'export_models_under_test', os.path.join( REPO_ROOT, 'tools', 'release', 'export_models.py' ) )
    module = importlib.util.module_from_spec( spec )
    spec.loader.exec_module( module )
    return module


def _fake_onnx( path ):
    with open( path, 'wb' ) as handle:
        handle.write( b'\x08' + b'\x00' * 2_000_000 )


class TestExport( unittest.TestCase ):

    def setUp( self ):
        self.em = _load()
        self.tmp = tempfile.mkdtemp( prefix='betasuite-export-test-' )
        self.addCleanup( shutil.rmtree, self.tmp, True )
        self.out = os.path.join( self.tmp, 'out' )
        os.makedirs( self.out )
        self.log = mock.Mock()
        self.workdirs = []
        real_mkdtemp = tempfile.mkdtemp

        def tracked_mkdtemp( *args, **kwargs ):
            path = real_mkdtemp( *args, dir=self.tmp, **{ k: v for k, v in kwargs.items() if k != 'dir' } )
            self.workdirs.append( path )
            return path
        patcher = mock.patch.object( self.em.tempfile, 'mkdtemp', side_effect=tracked_mkdtemp )
        patcher.start()
        self.addCleanup( patcher.stop )

    def rows( self, *ids ):
        return [ bu_models.entry( model_id ) for model_id in ids ]

    def test_the_export_env_gets_cpu_torch_first_then_the_tools_and_never_the_venv( self ):
        commands = []
        with mock.patch.object( self.em, 'run_logged', side_effect=lambda d, c, log: commands.append( c ) ):
            python = self.em.build_export_env( self.tmp, '/usr/bin/uv', self.log, { 'ultralytics', 'rfdetr' } )
        self.assertTrue( python.startswith( os.path.join( self.tmp, 'export-venv' ) ) )
        self.assertEqual( commands[0][:2], [ '/usr/bin/uv', 'venv' ] )
        torch, tools = commands[1], commands[2]
        self.assertEqual( torch[ torch.index( '--python' ) + 1 ], python )
        self.assertEqual( torch[ torch.index( '--index-url' ) + 1 ], self.em.EXPORT_TORCH_INDEX )
        self.assertIn( 'torch', torch )
        self.assertTrue( any( package.startswith( 'ultralytics==' ) for package in tools ) )
        self.assertTrue( any( package.startswith( 'rfdetr==' ) for package in tools ) )
        self.assertFalse( any( package.startswith( 'torch' ) for package in tools ), "torch only from the CPU index" )

    def test_without_uv_it_falls_back_to_venv_and_pip( self ):
        commands = []
        with mock.patch.object( self.em, 'run_logged', side_effect=lambda d, c, log: commands.append( c ) ):
            python = self.em.build_export_env( self.tmp, None, self.log, { 'ultralytics' } )
        self.assertEqual( commands[0][1:3], [ '-m', 'venv' ] )
        self.assertEqual( commands[1][:4], [ python, '-m', 'pip', 'install' ] )

    def test_only_the_chosen_tools_packages( self ):
        commands = []
        with mock.patch.object( self.em, 'run_logged', side_effect=lambda d, c, log: commands.append( c ) ):
            self.em.build_export_env( self.tmp, '/usr/bin/uv', self.log, { 'ultralytics' } )
        self.assertFalse( any( package.startswith( 'rfdetr' ) for package in commands[2] ) )
        self.assertIn( 'onnx', commands[2] )

    def test_a_successful_export_lands_in_the_folder_with_its_hash_and_cleans_up( self ):
        def fake_run( command, **kwargs ):
            _fake_onnx( command[4] )
            return mock.Mock( returncode=0, stdout='export success' )
        with mock.patch.object( self.em, 'build_export_env', return_value='/venv/python' ), \
             mock.patch.object( self.em, 'download_checked' ) as download, \
             mock.patch.object( self.em.subprocess, 'run', side_effect=fake_run ):
            done, failures = self.em.export_models( self.rows( 'yolo11n-seg', 'yolo11s-seg' ), self.out, self.log )
        self.assertEqual( failures, 0 )
        self.assertEqual( sorted( os.listdir( self.out ) ), [ 'yolo11n-seg.onnx', 'yolo11s-seg.onnx' ] )
        expected = hashlib.sha256( b'\x08' + b'\x00' * 2_000_000 ).hexdigest()
        self.assertEqual( done['yolo11n-seg'], ( expected, 2_000_001 ) )
        source = bu_models.entry( 'yolo11n-seg' )['sources'][0]
        self.assertEqual( download.call_args_list[0][0][:2], ( source['upstream'], source['sha256'] ) )
        self.assertFalse( any( os.path.exists( path ) for path in self.workdirs ), "export env removed" )

    def test_a_failed_export_is_a_failure_and_still_cleans_up( self ):
        with mock.patch.object( self.em, 'build_export_env', return_value='/venv/python' ), \
             mock.patch.object( self.em, 'download_checked' ), \
             mock.patch.object( self.em.subprocess, 'run',
                                return_value=mock.Mock( returncode=1, stdout='Traceback: boom' ) ):
            done, failures = self.em.export_models( self.rows( 'yolo11n-seg' ), self.out, self.log )
        self.assertEqual( ( done, failures ), ( {}, 1 ) )
        self.assertEqual( os.listdir( self.out ), [] )
        self.assertFalse( any( os.path.exists( path ) for path in self.workdirs ) )

    def test_an_export_that_is_not_a_model_is_refused( self ):
        def fake_run( command, **kwargs ):
            with open( command[4], 'w' ) as handle:
                handle.write( '<html>sign in</html>' )
            return mock.Mock( returncode=0, stdout='' )
        with mock.patch.object( self.em, 'build_export_env', return_value='/venv/python' ), \
             mock.patch.object( self.em, 'download_checked' ), \
             mock.patch.object( self.em.subprocess, 'run', side_effect=fake_run ):
            self.assertEqual( self.em.export_models( self.rows( 'yolo11n-seg' ), self.out, self.log )[1], 1 )
        self.assertEqual( os.listdir( self.out ), [] )

    def test_no_environment_means_every_model_failed( self ):
        with mock.patch.object( self.em, 'build_export_env', side_effect=RuntimeError( 'no network' ) ):
            self.assertEqual( self.em.export_models( self.rows( 'yolo11n-seg', 'yolo11m-seg' ),
                                                     self.out, self.log ), ( {}, 2 ) )
        self.assertFalse( any( os.path.exists( path ) for path in self.workdirs ) )

    def test_a_tampered_download_is_refused( self ):
        class Response:
            def __init__( self ):
                self.data = [ b'not the weights', b'' ]

            def __enter__( self ):
                return self

            def __exit__( self, *args ):
                return False

            def read( self, *_args ):
                return self.data.pop( 0 )
        dest = os.path.join( self.tmp, 'w.pt' )
        with mock.patch.object( self.em.urllib.request, 'urlopen', return_value=Response() ):
            with self.assertRaises( self.em.NotAModel ):
                self.em.download_checked( 'https://x/yolo11n-seg.pt', '0' * 64, dest, self.log )
        self.assertEqual( os.listdir( self.tmp ), [ 'out' ] )

    def test_every_person_model_is_exportable( self ):
        self.assertEqual( { row['id'] for row in self.em.exportable() },
                          { row['id'] for row in bu_models.entries( kind='person' ) } )

    def test_each_rf_detr_size_is_exported_by_rfdetr_from_its_own_weights( self ):
        from segmenters import rf_detr_seg
        urls = set()
        for name in rf_detr_seg.VARIANTS:
            row = bu_models.entry_for( 'rf_detr_seg', name )
            self.assertEqual( row['export'], { 'tool': 'rfdetr', 'arg': 'RFDETRSeg' + name.capitalize() } )
            urls.add( row['sources'][0]['upstream'] )
            self.assertEqual( self.em.export_command( '/venv/python', row, '/w/x.pth', '/w/out.onnx' ),
                              [ '/venv/python', '-c', self.em.EXPORT_SCRIPTS['rfdetr'], '/w/x.pth', '/w/out.onnx',
                                row['export']['arg'] ] )
        self.assertEqual( len( urls ), len( rf_detr_seg.VARIANTS ), "no two sizes share weights" )
        yolo = bu_models.entry( 'yolo11s-seg' )
        self.assertEqual( self.em.export_command( '/p', yolo, 'a.pt', 'b.onnx' ),
                          [ '/p', '-c', self.em.EXPORT_SCRIPTS['ultralytics'], 'a.pt', 'b.onnx', '640' ] )

    def test_the_export_scripts_compile_and_rf_detr_keeps_a_dynamic_batch( self ):
        for name, script in self.em.EXPORT_SCRIPTS.items():
            compile( script, name, 'exec' )
        self.assertIn( 'dynamic_batch=True', self.em.EXPORT_SCRIPTS['rfdetr'] )
        self.assertIn( 'pretrain_weights=pt', self.em.EXPORT_SCRIPTS['rfdetr'],
                       "exports the pinned file, not whatever rfdetr would download" )

    def test_the_export_runs_offline( self ):
        envs = []

        def fake_run( command, **kwargs ):
            envs.append( kwargs['env'] )
            _fake_onnx( command[4] )
            return mock.Mock( returncode=0, stdout='' )
        with mock.patch.object( self.em, 'build_export_env', return_value='/venv/python' ), \
             mock.patch.object( self.em, 'download_checked' ), \
             mock.patch.object( self.em.subprocess, 'run', side_effect=fake_run ):
            self.assertEqual( self.em.export_models( self.rows( 'rf-detr-seg-small' ), self.out, self.log )[1], 0 )
        for key in ( 'YOLO_OFFLINE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE' ):
            self.assertEqual( envs[0].get( key ), '1', key )

    def test_export_builds_the_environment_for_the_tools_it_exports( self ):
        with mock.patch.object( self.em, 'build_export_env', side_effect=RuntimeError( 'stop' ) ) as build:
            self.em.export_models( self.rows( 'rf-detr-seg-nano' ), self.out, self.log )
        self.assertEqual( build.call_args[0][3], { 'rfdetr' } )


if __name__ == '__main__':
    unittest.main()
