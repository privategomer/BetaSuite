"""
test_fetch_models.py - setup's model step (tools/setup/fetch_models.py),
which downloads the project's hosted model files.

What each test prevents:
  - setup finishing without the default detector, or reporting success
    when a model failed to install
  - a person backend nobody chose being installed anyway; the default
    leaving out yolo11_seg, or a half-installed backend left unfinished
  - a typo in --person-backends, --extra-models or at the prompt
    silently installing nothing, and the prompt blocking setup when
    there is no terminal
  - a file at a model's place that is not the pinned one being used,
    or thrown away instead of set aside
  - --no-download downloading, and --no-person (or the old --no-export)
    installing person models
  - verify_env checking a different list of models than the catalog
"""

import importlib.util
import os
import shutil
import tempfile
import unittest
from unittest import mock

import betautils_models as bu_models

REPO_ROOT = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )


def _load( name, *path ):
    spec = importlib.util.spec_from_file_location( name, os.path.join( REPO_ROOT, *path ) )
    module = importlib.util.module_from_spec( spec )
    spec.loader.exec_module( module )
    return module


class _Case( unittest.TestCase ):

    def setUp( self ):
        self.fm = _load( 'fetch_models_under_test', 'tools', 'setup', 'fetch_models.py' )
        self.tmp = tempfile.mkdtemp( prefix='betasuite-fetch-' )
        self.addCleanup( shutil.rmtree, self.tmp, True )
        self.model_dir = os.path.join( self.tmp, 'model' )
        os.makedirs( self.model_dir )
        self.states = {}
        self.installed = []
        self.log = mock.Mock()
        self.fail_ids = set()

        def status( row, should_stop=None ):
            return { 'state': self.states.get( row['id'], 'absent' ), 'bytes': 0 }

        def install( model_id, progress=None, **_kw ):
            if model_id in self.fail_ids:
                raise bu_models.ModelError( 'model.download_failed', model=model_id, reason='network' )
            self.installed.append( model_id )
            self.states[model_id] = 'installed'
            return 'installed'

        for patch in ( mock.patch.object( bu_models, 'model_dir', return_value=self.model_dir ),
                       mock.patch.object( bu_models, 'status', side_effect=status ),
                       mock.patch.object( bu_models, 'install', side_effect=install ),
                       mock.patch.object( self.fm, 'relaunch_in_venv' ) ):
            patch.start()
            self.addCleanup( patch.stop )

    def present( self, *ids ):
        for model_id in ids:
            self.states[model_id] = 'installed'

    def ids_of( self, backend ):
        return [ row['id'] for row in self.fm.rows_of( backend ) ]

    def run_main( self, *argv, interactive=False, answers=() ):
        replies = iter( answers )
        self.installed.clear()
        with mock.patch.object( self.fm.sys, 'argv', [ 'fetch_models.py', '--log-file',
                                                       os.path.join( self.tmp, 'f.log' ) ] + list( argv ) ), \
             mock.patch.object( self.fm.sys.stdin, 'isatty', return_value=interactive ), \
             mock.patch( 'builtins.input', side_effect=lambda prompt: next( replies ) ):
            code = self.fm.main()
        return code, sorted( self.installed )


class TestInstall( _Case ):

    def test_320n_and_the_default_backend_unattended( self ):
        code, installed = self.run_main( '--yes' )
        self.assertEqual( code, 0 )
        self.assertEqual( installed, sorted( [ 'nudenet-320n' ] + self.ids_of( 'yolo11_seg' ) ) )

    def test_present_models_are_not_downloaded_again( self ):
        self.present( 'nudenet-320n', *self.ids_of( 'yolo11_seg' ) )
        self.assertEqual( self.run_main( '--yes' ), ( 0, [] ) )

    def test_a_failure_is_the_exit_status( self ):
        self.fail_ids = { 'nudenet-320n' }
        code, installed = self.run_main( '--yes' )
        self.assertEqual( code, 1 )
        self.assertNotIn( 'nudenet-320n', installed )

    def test_no_download_only_checks( self ):
        self.assertEqual( self.run_main( '--no-download', '--yes' )[0], 1 )
        self.assertEqual( self.installed, [] )
        self.present( 'nudenet-320n', *self.ids_of( 'yolo11_seg' ) )
        self.assertEqual( self.run_main( '--no-download', '--yes' ), ( 0, [] ) )

    def test_no_person_and_the_old_spelling_skip_the_person_models( self ):
        for flag in ( '--no-person', '--no-export' ):
            self.states.clear()
            self.assertEqual( self.run_main( flag, '--yes' ), ( 0, [ 'nudenet-320n' ] ) )

    def test_extras_by_flag( self ):
        self.present( 'nudenet-320n' )
        self.assertEqual( self.run_main( '--no-person', '--extra-models', 'nudenet-640m' ), ( 0, [ 'nudenet-640m' ] ) )
        self.states.clear()
        self.present( 'nudenet-320n' )
        self.assertEqual( self.run_main( '--no-person', '--extra-models', 'all' ),
                          ( 0, [ 'nudenet-640m', 'nudenet-retinanet' ] ) )

    def test_a_bad_flag_value_stops_before_anything_runs( self ):
        for argv in ( ( '--person-backends', 'rfdetr' ), ( '--extra-models', 'nudenet-999' ) ):
            with mock.patch( 'sys.stderr' ):
                with self.assertRaises( SystemExit ) as caught:
                    self.run_main( *argv )
            self.assertEqual( caught.exception.code, 2 )
            self.assertEqual( self.installed, [] )

    def test_a_swapped_file_is_set_aside_and_replaced( self ):
        row = bu_models.entry( 'nudenet-320n' )
        path = os.path.join( self.model_dir, row['file'] )
        with open( path, 'wb' ) as out:
            out.write( b'mine' )
        self.states['nudenet-320n'] = 'modified'
        self.assertEqual( self.run_main( '--no-person' ), ( 0, [ 'nudenet-320n' ] ) )
        with open( path + '.bad', 'rb' ) as handle:
            self.assertEqual( handle.read(), b'mine' )


class TestPersonBackendChoice( _Case ):

    def setUp( self ):
        super().setUp()
        self.present( 'nudenet-320n' )

    def test_every_person_backend_is_one_the_app_has( self ):
        import betautils_person
        self.assertEqual( set( self.fm.PERSON_BACKENDS ), set( betautils_person.registered_backend_names() ) )
        self.assertEqual( self.fm.DEFAULT_PERSON_BACKENDS, [ betautils_person.DEFAULT_BACKEND ],
                          "setup's default is the backend the app runs by default" )
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

    def test_default_is_yolo_plus_a_backend_already_partly_present( self ):
        self.assertEqual( self.fm.default_person_backends( set() ), [ 'yolo11_seg' ] )
        self.assertEqual( self.fm.default_person_backends( { 'rf-detr-seg-small.onnx' } ),
                          [ 'yolo11_seg', 'rf_detr_seg' ] )

    def test_a_typed_answer_takes_numbers_names_or_blank( self ):
        self.assertEqual( self.fm.choose_person_backends( '', set() ), [ 'yolo11_seg' ] )
        self.assertEqual( self.fm.choose_person_backends( '2', set() ), [ 'rf_detr_seg' ] )
        self.assertEqual( self.fm.choose_person_backends( '1 2', set() ), [ 'yolo11_seg', 'rf_detr_seg' ] )
        self.assertEqual( self.fm.choose_person_backends( 'none', set() ), [] )
        for answer in ( '0', '3', 'yolo' ):
            with self.assertRaises( ValueError ):
                self.fm.choose_person_backends( answer, set() )

    def test_without_a_terminal_it_takes_the_default_and_never_asks( self ):
        ask = mock.Mock( side_effect=AssertionError( 'asked' ) )
        self.assertEqual( self.fm.ask_person_backends( set(), self.log, ask=ask, interactive=False ), [ 'yolo11_seg' ] )
        ask.assert_not_called()

    def test_a_bad_answer_is_asked_again_and_end_of_input_takes_the_default( self ):
        ask = mock.Mock( side_effect=[ 'rfdetr', '2' ] )
        self.assertEqual( self.fm.ask_person_backends( set(), self.log, ask=ask, interactive=True ), [ 'rf_detr_seg' ] )
        self.assertEqual( ask.call_count, 2 )
        ask = mock.Mock( side_effect=EOFError )
        self.assertEqual( self.fm.ask_person_backends( set(), self.log, ask=ask, interactive=True ), [ 'yolo11_seg' ] )

    def test_the_menu_shows_each_backend_its_size_and_what_is_present( self ):
        lines = self.fm.person_backend_menu( { 'rf-detr-seg-nano.onnx' } )
        self.assertEqual( len( lines ), 2 )
        self.assertIn( 'yolo11_seg', lines[0] )
        self.assertIn( 'about 144 MB', lines[0] )
        self.assertNotIn( 'present', lines[0] )
        self.assertIn( 'about 377 MB, 1 of 3 present', lines[1] )

    def test_the_flag_picks_the_backends_and_none_installs_nothing( self ):
        self.assertEqual( self.run_main( '--person-backends', 'rf_detr_seg' ), ( 0, sorted( self.ids_of( 'rf_detr_seg' ) ) ) )
        self.states = { 'nudenet-320n': 'installed' }
        self.assertEqual( self.run_main( '--person-backends', 'none' ), ( 0, [] ) )

    def test_at_a_terminal_the_answer_picks_the_backends( self ):
        self.assertEqual( self.run_main( interactive=True, answers=[ '2' ] ), ( 0, sorted( self.ids_of( 'rf_detr_seg' ) ) ) )

    def test_unattended_and_yes_never_ask( self ):
        self.assertEqual( self.run_main(), ( 0, sorted( self.ids_of( 'yolo11_seg' ) ) ) )
        self.states = { 'nudenet-320n': 'installed' }
        self.assertEqual( self.run_main( '--yes', interactive=True, answers=[ 'rf_detr_seg' ] ),
                          ( 0, sorted( self.ids_of( 'yolo11_seg' ) ) ) )

    def test_a_partial_backend_is_finished_and_nothing_missing_asks_nothing( self ):
        self.present( *self.ids_of( 'yolo11_seg' ), 'rf-detr-seg-small' )
        self.assertEqual( self.run_main( '--yes' ), ( 0, [ 'rf-detr-seg-medium', 'rf-detr-seg-nano' ] ) )
        self.assertEqual( self.run_main( interactive=True, answers=[ AssertionError ] ), ( 0, [] ) )


class TestVerifyEnv( unittest.TestCase ):

    def test_verify_env_checks_every_catalog_model( self ):
        verify_env = _load( 'verify_env_under_test', 'tools', 'setup', 'verify_env.py' )
        self.assertEqual( set( verify_env.MODELS.values() ),
                          { bu_models.config_path_for( row ) for row in bu_models.entries() } )


if __name__ == '__main__':
    unittest.main()
