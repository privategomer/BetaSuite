"""
test_serve_models.py - the models.* commands and their events.

What each test prevents:
  - a download that blocks the command loop, or two at once
  - progress events flooding the client, or the last one (bytes equal
    to total) missing
  - a cancel that loses what was downloaded, or a failure the client
    is never told about (and with which code)
  - removing a model while it downloads, while a job runs, or while
    the settings use it
  - "use" leaving a hand-set model_path in force, or the GPU off when
    the recommended model on a GPU machine was chosen
  - answers and events that drift from protocol/v1's schemas
"""

import threading
import unittest
from unittest import mock

import betautils_models as bu_models
import betautils_protocol as bu_proto
import betautils_serve_models as bu_serve_models

from tests import protocol_schemas as schemas


class _Events:

    def __init__( self ):
        self.items = []
        self.cond = threading.Condition()

    def __call__( self, name, data ):
        with self.cond:
            self.items.append( ( name, data ) )
            self.cond.notify_all()

    def wait_for( self, predicate, timeout=5.0 ):
        with self.cond:
            if not self.cond.wait_for( lambda: predicate( self.items ), timeout ):
                raise AssertionError( 'events never arrived: %r'%( self.items, ) )

    def states( self, model_id ):
        return [ d['state'] for n, d in self.items if n == 'models.state' and d['id'] == model_id ]


def _final( model_id ):
    return lambda items: any( n == 'models.state' and d['id'] == model_id
                              and d['state'] in ( 'installed', 'stopped', 'failed' ) for n, d in items )


class TestInstaller( unittest.TestCase ):

    def make( self, install ):
        self.events = _Events()
        installer = bu_serve_models.Installer( self.events, mock.Mock(), install=install )
        self.addCleanup( installer.stop )
        return installer

    def test_install_reports_queued_downloading_progress_installed( self ):
        def install( model_id, progress, should_stop ):
            for done in range( 0, 101, 10 ):
                progress( done, 100 )
            return 'installed'

        installer = self.make( install )
        self.assertTrue( installer.add( 'a' ) )
        self.events.wait_for( _final( 'a' ) )
        self.assertEqual( self.events.states( 'a' ), [ 'queued', 'downloading', 'installed' ] )
        progress = [ d for n, d in self.events.items if n == 'models.progress' ]
        self.assertEqual( progress[-1], { 'id': 'a', 'bytes': 100, 'total': 100 } )
        # Coalesced: everything here arrives within one interval.
        self.assertLess( len( progress ), 11 )
        self.__class__.sample_events = list( self.events.items )

    def test_one_at_a_time_in_order_and_no_duplicates( self ):
        release = threading.Event()
        running = []

        def install( model_id, progress, should_stop ):
            running.append( model_id )
            release.wait( 5 )

        installer = self.make( install )
        installer.add( 'a' )
        installer.add( 'b' )
        self.assertFalse( installer.add( 'a' ) )
        self.assertFalse( installer.add( 'b' ) )
        self.events.wait_for( lambda items: running == [ 'a' ] )
        self.assertEqual( installer.state_of( 'a' ), 'downloading' )
        self.assertEqual( installer.state_of( 'b' ), 'queued' )
        release.set()
        self.events.wait_for( _final( 'b' ) )
        self.assertEqual( running, [ 'a', 'b' ] )

    def test_cancel_stops_the_download_in_progress( self ):
        started = threading.Event()

        def install( model_id, progress, should_stop ):
            started.set()
            while not should_stop():
                pass
            raise bu_models.Stopped()

        installer = self.make( install )
        installer.add( 'a' )
        started.wait( 5 )
        self.assertTrue( installer.cancel( 'a' ) )
        self.events.wait_for( _final( 'a' ) )
        self.assertEqual( self.events.states( 'a' )[-1], 'stopped' )
        self.assertIsNone( installer.state_of( 'a' ) )

    def test_cancel_takes_a_queued_install_off( self ):
        release = threading.Event()
        installer = self.make( lambda model_id, progress, should_stop: release.wait( 5 ) )
        installer.add( 'a' )
        installer.add( 'b' )
        self.assertTrue( installer.cancel( 'b' ) )
        self.assertEqual( self.events.states( 'b' ), [ 'queued', 'stopped' ] )
        self.assertFalse( installer.cancel( 'nothing' ) )
        release.set()

    def test_failure_carries_the_code( self ):
        def install( model_id, progress, should_stop ):
            raise bu_models.ModelError( 'model.download_failed', model=model_id, reason='network' )

        installer = self.make( install )
        installer.add( 'a' )
        self.events.wait_for( _final( 'a' ) )
        last = [ d for n, d in self.events.items if n == 'models.state' ][-1]
        self.assertEqual( last, { 'id': 'a', 'state': 'failed', 'code': 'model.download_failed',
                                  'params': { 'model': 'a', 'reason': 'network' } } )
        bu_proto.error( last['code'], **last['params'] )

    def test_a_crash_is_reported_and_the_worker_lives_on( self ):
        calls = []

        def install( model_id, progress, should_stop ):
            calls.append( model_id )
            if model_id == 'a':
                raise RuntimeError( 'boom' )

        installer = self.make( install )
        installer.add( 'a' )
        installer.add( 'b' )
        self.events.wait_for( _final( 'b' ) )
        self.assertEqual( self.events.states( 'a' )[-1], 'failed' )
        self.assertEqual( self.events.states( 'b' )[-1], 'installed' )

    def test_stop_ends_the_download_in_progress( self ):
        started = threading.Event()

        def install( model_id, progress, should_stop ):
            started.set()
            while not should_stop():
                pass
            raise bu_models.Stopped()

        installer = self.make( install )
        installer.add( 'a' )
        started.wait( 5 )
        installer.stop()
        self.assertFalse( installer._thread.is_alive() )


class _Server:

    def __init__( self ):
        self.events = _Events()
        self._logger = mock.Mock()
        self.jobs = None
        self.models = None

    def send_event( self, name, data ):
        self.events( name, data )


class TestCommands( unittest.TestCase ):

    def setUp( self ):
        self.server = _Server()
        self.addCleanup( lambda: self.server.models and self.server.models.stop() )

    def test_list_answers_the_schema( self ):
        status = { 'state': 'absent', 'bytes': 0 }
        hardware = { 'providers': [], 'gpu': { 'name': 'Card' }, 'cuda_usable': True }
        with mock.patch.object( bu_models, 'status', return_value=status ), \
             mock.patch.object( bu_serve_models, '_hardware', return_value=hardware ), \
             mock.patch.object( bu_models, 'in_use', return_value={ 'detector': 'nudenet-320n', 'person': None,
                                                                    'custom': { 'person': '/x.onnx' } } ):
            result = bu_serve_models.models_list( self.server, {} )
        self.assertEqual( len( result['models'] ), len( bu_models.entries() ) )
        self.assertEqual( result['recommended'], { 'detector': 'nudenet-640m', 'reason': 'gpu' } )
        self.assertEqual( result['gpu']['name'], 'Card' )
        self.assertEqual( result['in_use'], { 'detector': 'nudenet-320n', 'person': None } )

    def test_list_shows_downloads_in_progress( self ):
        installer = mock.Mock()
        installer.state_of.side_effect = lambda model_id: 'downloading' if model_id == 'nudenet-640m' else None
        self.server.models = None
        with mock.patch.object( bu_models, 'status', return_value={ 'state': 'partial', 'bytes': 5 } ), \
             mock.patch.object( bu_serve_models, '_hardware', return_value={ 'cuda_usable': False } ), \
             mock.patch.object( bu_models, 'in_use', return_value={ 'detector': None, 'person': None, 'custom': {} } ):
            self.server.models = installer
            result = bu_serve_models.models_list( self.server, {} )
            self.server.models = None
        states = { row['id']: row['state'] for row in result['models'] }
        self.assertEqual( states['nudenet-640m'], 'downloading' )
        self.assertEqual( states['nudenet-320n'], 'partial' )

    def test_unknown_id_and_bad_params( self ):
        for data, code in ( ( {}, 'command.bad_params' ), ( { 'id': 3 }, 'command.bad_params' ),
                            ( { 'id': 'nope' }, 'model.unknown' ) ):
            with self.assertRaises( bu_proto.ProtocolError ) as caught:
                bu_serve_models.models_install( self.server, data )
            self.assertEqual( caught.exception.code, code )

    def test_install_returns_at_once( self ):
        release = threading.Event()
        with mock.patch.object( bu_models, 'install', side_effect=lambda *a, **k: release.wait( 5 ) ):
            result = bu_serve_models.models_install( self.server, { 'id': 'nudenet-640m' } )
            self.assertEqual( result, { 'id': 'nudenet-640m', 'queued': True } )
            self.assertEqual( bu_serve_models.models_install( self.server, { 'id': 'nudenet-640m' } )['queued'], False )
            self.assertTrue( bu_serve_models.models_cancel( self.server, { 'id': 'nudenet-640m' } )['cancelled'] )
            release.set()

    def test_remove_refused_while_downloading( self ):
        self.server.models = mock.Mock( **{ 'state_of.return_value': 'downloading' } )
        with self.assertRaises( bu_proto.ProtocolError ) as caught:
            bu_serve_models.models_remove( self.server, { 'id': 'nudenet-640m' } )
        self.assertEqual( caught.exception.code, 'model.busy' )
        self.server.models = None

    def test_remove_refused_while_a_job_runs( self ):
        self.server.jobs = mock.Mock( **{ 'busy_job.return_value': 'j1' } )
        with self.assertRaises( bu_proto.ProtocolError ) as caught:
            bu_serve_models.models_remove( self.server, { 'id': 'nudenet-640m' } )
        self.assertEqual( caught.exception.error, { 'code': 'config.busy', 'params': { 'job': 'j1' } } )

    def test_remove_in_use_and_success( self ):
        with mock.patch.object( bu_models, 'remove',
                                side_effect=bu_models.ModelError( 'model.in_use', model='nudenet-320n' ) ):
            with self.assertRaises( bu_proto.ProtocolError ) as caught:
                bu_serve_models.models_remove( self.server, { 'id': 'nudenet-320n' } )
        self.assertEqual( caught.exception.code, 'model.in_use' )
        with mock.patch.object( bu_models, 'remove' ) as remove:
            self.assertEqual( bu_serve_models.models_remove( self.server, { 'id': 'nudenet-640m' } ),
                              { 'id': 'nudenet-640m' } )
        remove.assert_called_once_with( 'nudenet-640m' )
        self.assertEqual( self.server.events.states( 'nudenet-640m' ), [ 'removed' ] )

    def test_import_checks_the_path( self ):
        for data in ( {}, { 'path': 5 }, { 'path': '/no/such/file.onnx' } ):
            with self.assertRaises( bu_proto.ProtocolError ) as caught:
                bu_serve_models.models_import( self.server, data )
            self.assertEqual( caught.exception.code, 'command.bad_params' )
        with mock.patch.object( bu_models, 'import_file', side_effect=bu_models.ModelError( 'model.unknown_file' ) ):
            with self.assertRaises( bu_proto.ProtocolError ) as caught:
                bu_serve_models.models_import( self.server, { 'path': __file__ } )
        self.assertEqual( caught.exception.code, 'model.unknown_file' )


class TestUse( unittest.TestCase ):

    def setUp( self ):
        self.server = _Server()

    def test_detector_values_clear_a_hand_set_path( self ):
        values, resets = bu_serve_models.use_values( bu_models.entry( 'nudenet-640m' ) )
        self.assertEqual( values, { 'detector_backend.selected': 'nudenet_v3',
                                    'detector_backend.nudenet_v3.model_variant': '640m' } )
        self.assertEqual( resets, [ 'detector_backend.nudenet_v3.model_path' ] )

    def test_retinanet_has_only_a_backend( self ):
        values, resets = bu_serve_models.use_values( bu_models.entry( 'nudenet-retinanet' ) )
        self.assertEqual( values, { 'detector_backend.selected': 'retinanet_v2' } )
        self.assertEqual( resets, [] )

    def test_person_values( self ):
        values, resets = bu_serve_models.use_values( bu_models.entry( 'rf-detr-seg-small' ) )
        self.assertEqual( values, { 'person_backend.selected': 'rf_detr_seg',
                                    'person_backend.rf_detr_seg.model_variant': 'small' } )
        self.assertEqual( resets, [ 'person_backend.rf_detr_seg.model_path' ] )

    def test_every_written_path_is_a_setting( self ):
        import betautils_settings as bu_settings
        for row in bu_models.entries():
            values, resets = bu_serve_models.use_values( row, gpu=True )
            for path in list( values ) + resets:
                self.assertIsNotNone( bu_settings.find( tuple( path.split( '.' ) ) ), path )

    def test_gpu_turns_on_with_the_model( self ):
        with mock.patch.object( bu_models, 'status', return_value={ 'state': 'installed', 'bytes': 1 } ), \
             mock.patch( 'betautils_serve_config._edit', return_value={ 'ok': True } ) as edit:
            bu_serve_models.models_use( self.server, { 'id': 'nudenet-640m', 'gpu': True } )
        updates = edit.call_args[0][1]
        self.assertEqual( updates[( 'gpu_enabled', )], 1 )
        self.assertEqual( updates[( 'detector_backend', 'nudenet_v3', 'model_variant' )], '640m' )
        self.assertEqual( edit.call_args[0][2], [ ( 'detector_backend', 'nudenet_v3', 'model_path' ) ] )

    def test_gpu_left_alone_unless_asked( self ):
        with mock.patch.object( bu_models, 'status', return_value={ 'state': 'installed', 'bytes': 1 } ), \
             mock.patch( 'betautils_serve_config._edit', return_value={} ) as edit:
            bu_serve_models.models_use( self.server, { 'id': 'nudenet-320n' } )
        self.assertNotIn( ( 'gpu_enabled', ), edit.call_args[0][1] )

    def test_refused_when_not_installed_busy_or_bad_gpu( self ):
        with mock.patch.object( bu_models, 'status', return_value={ 'state': 'absent', 'bytes': 0 } ):
            with self.assertRaises( bu_proto.ProtocolError ) as caught:
                bu_serve_models.models_use( self.server, { 'id': 'nudenet-640m' } )
        self.assertEqual( caught.exception.code, 'model.not_installed' )
        with self.assertRaises( bu_proto.ProtocolError ) as caught:
            bu_serve_models.models_use( self.server, { 'id': 'nudenet-640m', 'gpu': 'yes' } )
        self.assertEqual( caught.exception.code, 'command.bad_params' )
        self.server.jobs = mock.Mock( **{ 'busy_job.return_value': 'j1' } )
        with self.assertRaises( bu_proto.ProtocolError ) as caught:
            bu_serve_models.models_use( self.server, { 'id': 'nudenet-640m' } )
        self.assertEqual( caught.exception.code, 'config.busy' )


@schemas.needs_jsonschema
class TestSchemas( unittest.TestCase ):

    def test_list_result( self ):
        hardware = { 'providers': [], 'gpu': { 'name': 'Card' }, 'cuda_usable': True }
        with mock.patch.object( bu_models, 'status', return_value={ 'state': 'absent', 'bytes': 0 } ), \
             mock.patch.object( bu_serve_models, '_hardware', return_value=hardware ), \
             mock.patch.object( bu_models, 'in_use', return_value={ 'detector': 'nudenet-320n', 'person': None,
                                                                    'custom': { 'person': '/x.onnx' } } ):
            schemas.validate_command_result( 'models.list', bu_serve_models.models_list( _Server(), {} ) )

    def test_other_results_and_data( self ):
        schemas.validate_command_result( 'models.install', { 'id': 'nudenet-640m', 'queued': True } )
        schemas.validate_command_result( 'models.cancel', { 'id': 'nudenet-640m', 'cancelled': False } )
        schemas.validate_command_result( 'models.import', { 'id': 'nudenet-640m' } )
        schemas.validate_command_result( 'models.remove', { 'id': 'nudenet-640m' } )
        schemas.validate_command_result( 'models.use', { 'moves': [ 'detection' ], 'may_move': [],
                                                         'written': True, 'dry_run': False } )
        schemas.validate_command_data( 'models.use', { 'id': 'nudenet-640m', 'gpu': True } )
        schemas.validate_command_data( 'models.import', { 'path': '/a/b.onnx' } )

    def test_events( self ):
        installer = bu_serve_models.Installer( _Events(), mock.Mock(),
                                               install=lambda model_id, progress, should_stop: progress( 5, 5 ) )
        events = installer._send
        installer.add( 'a' )
        events.wait_for( _final( 'a' ) )
        installer.stop()
        samples = list( events.items ) + [
            ( 'models.state', { 'id': 'a', 'state': 'failed', 'code': 'model.download_failed',
                                'params': { 'model': 'a', 'reason': 'network' } } ),
            ( 'models.state', { 'id': 'a', 'state': 'removed' } ) ]
        for name, data in samples:
            schemas.validate_event_data( name, data )


if __name__ == '__main__':
    unittest.main()
