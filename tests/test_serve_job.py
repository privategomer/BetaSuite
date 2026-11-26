"""
test_serve_job.py - the engine's job commands, in process, with a fake
pipeline whose steps the test releases one at a time.

Covers: submit, its checks and its result coming before the job's
events; the queue running jobs in order; pause, resume and cancel from
every state, and refusals for the wrong state or an unknown job; the run
lock, taken while busy and refusing a command-line run both ways;
config.set and config.reset refused while busy, config.get answered from
the snapshot; shutdown stopping a running job; progress coalescing with
state and error events never dropped; a full disk stopping the queue.

The real pipeline through a real engine process is in
test_serve_job_process.py.
"""

import io
import json
import logging
import os
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

import betaconfig
import betautils_protocol as bu_proto
import betautils_runlock as bu_runlock
import betautils_serve as bu_serve
import betautils_serve_config as bu_serve_config
import betautils_serve_job as bu_serve_job
import betautils_signals as bu_signals

from tests import protocol_schemas as schemas

WAIT = 10


class Wire( io.RawIOBase ):
    """The engine's stdout: thread-safe, parsed into messages, waitable."""

    def __init__( self ):
        super().__init__()
        self._buffer = b''
        self.messages = []
        self._cond = threading.Condition()

    def writable( self ):
        return True

    def write( self, data ):
        with self._cond:
            self._buffer += bytes( data )
            while b'\n' in self._buffer:
                line, self._buffer = self._buffer.split( b'\n', 1 )
                message = json.loads( line )
                if schemas.HAVE_JSONSCHEMA:
                    schemas.validate_message( message )
                    if message['kind'] == 'event':
                        schemas.validate_event_data( message['name'], message['data'] )
                    elif message.get( 'ok' ) and message['name'] in schemas.command_schema_names():
                        schemas.validate_command_result( message['name'], message['data'] )
                self.messages.append( message )
            self._cond.notify_all()
        return len( data )

    def wait_for( self, predicate, what ):
        deadline = time.monotonic() + WAIT
        with self._cond:
            while True:
                for message in self.messages:
                    if predicate( message ):
                        return message
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError( "timed out waiting for %s; got %s"%(
                        what, [ ( m.get( 'name' ), m.get( 'data' ) ) for m in self.messages ] ) )
                self._cond.wait( remaining )

    def events( self, name=None ):
        with self._cond:
            return [ m for m in self.messages if m['kind'] == 'event' and ( name is None or m['name'] == name ) ]


class FakePipeline:
    """
    process_one_video in steps: each file reports 'detect' progress per
    step and checks for a stop between steps. A step runs only when the
    test releases it (or every step, with auto). A file named 'bad*'
    fails, one named 'full*' fails as a full disk.
    """

    def __init__( self, steps=3 ):
        self.steps = steps
        self.auto = True
        self.release = threading.Semaphore( 0 )
        self.started = []
        self.calls = []
        self.bu_detector = types.SimpleNamespace(
            get_detector=lambda *args: types.SimpleNamespace( get_session=lambda: 'session' ) )

    @staticmethod
    def _log_run_banner( logger ):
        pass

    @staticmethod
    def _resolve_active_encode_preset():
        return 'ultrafast'

    @staticmethod
    def _free_space_note( folder ):
        return ''

    def process_one_video( self, root, fname, censored_folder, index, total, session,
                           preview, preview_max, preset, logger, reporter=None ):
        import betautils_log as bu_log
        self.started.append( fname )
        reporter.output_paths( { 'final': os.path.join( censored_folder, fname + '.out.mp4' ) } )
        reporter.decision( 'structure.profile', profile=None, rule='disabled', sample_fps=5 )
        progress = bu_log.ProgressReporter( 'detect 320', total=self.steps, min_interval=0 )
        with progress:
            for step in range( self.steps ):
                bu_signals.check()
                if not self.auto:
                    self.release.acquire( timeout=WAIT )
                bu_signals.check()
                self.calls.append( ( fname, step ) )
                progress.update( step + 1 )
        if fname.startswith( 'bad' ):
            reporter.error( 'file.failed', exception_type='RuntimeError', stage=reporter.stage )
            return 'failed'
        if fname.startswith( 'full' ):
            return 'out_of_space'
        return 'processed'


class _QueueTest( unittest.TestCase ):

    def setUp( self ):
        self.tmp = tempfile.mkdtemp( prefix='betasuite-serve-job-' )
        self.addCleanup( __import__( 'shutil' ).rmtree, self.tmp, True )
        self.files = {}
        for name in ( 'a.mp4', 'b.mp4', 'c.mp4', 'bad.mp4', 'full.mp4' ):
            path = os.path.join( self.tmp, name )
            with open( path, 'wb' ) as handle:
                handle.write( b'x' )
            self.files[name] = path
        self.wire = Wire()
        self.logger = logging.getLogger( 'test_serve_job' )
        self.logger.propagate = False
        if not self.logger.handlers:
            quiet = logging.NullHandler()
            quiet.setLevel( logging.CRITICAL )   # the server traces when a handler takes trace
            self.logger.addHandler( quiet )
        self.server = bu_serve.Server( io.BytesIO(), self.wire, self.logger )
        self.server.protocol = 1
        self.pipeline = FakePipeline()
        self.lock_calls = []
        self.server.jobs = bu_serve_job.JobQueue(
            self.server.send_event, self.logger, pipeline=self.pipeline,
            acquire_lock=self._lock )
        self.addCleanup( self._stop )
        bu_signals.clear()
        self.addCleanup( bu_signals.clear )
        self._ids = iter( range( 1, 10000 ) )

    def _lock( self ):
        held = bu_runlock.acquire_library( 'engine' )
        self.lock_calls.append( held )
        return held

    def _stop( self ):
        self.server.stop_jobs()

    def command( self, name, data=None ):
        request_id = 'c%d'%( next( self._ids ), )
        self.server.handle_line( json.dumps( { 'v': 1, 'kind': 'cmd', 'id': request_id, 'name': name,
                                               'data': data or {} } ).encode() )
        return self.wire.wait_for( lambda m: m.get( 'id' ) == request_id, name )

    def ok( self, name, data=None ):
        result = self.command( name, data )
        self.assertTrue( result['ok'], result )
        return result['data']

    def refused( self, name, data, code ):
        result = self.command( name, data )
        self.assertFalse( result['ok'], result )
        self.assertEqual( result['error']['code'], code, result )
        return result['error']['params']

    def submit( self, *names, overrides=None ):
        data = { 'files': [ self.files[name] for name in names ] }
        if overrides:
            data['overrides'] = overrides
        return self.ok( 'job.submit', data )['job']

    def wait_state( self, job, state ):
        return self.wire.wait_for( lambda m: m['kind'] == 'event' and m['name'] == 'job.state'
                                   and m['data']['job'] == job and m['data']['state'] == state,
                                   'job %s %s'%( job, state ) )['data']

    def story( self, job ):
        """The job's events, progress left out, as short strings."""
        out = []
        for message in self.wire.events():
            data = message['data']
            if data.get( 'job' ) != job or message['name'] == 'job.progress':
                continue
            if message['name'] == 'job.state':
                out.append( 'state:' + data['state'] )
            elif message['name'] == 'job.file':
                out.append( 'file%d:%s'%( data['file'], data['state'] ) )
            elif message['name'] == 'job.error':
                out.append( 'error:' + data['code'] )
            else:
                out.append( 'decision:' + data['code'] )
        return out


class TestSubmit( _QueueTest ):

    def test_a_job_runs_and_its_result_comes_before_its_events( self ):
        job = self.submit( 'a.mp4', 'b.mp4' )
        self.wait_state( job, 'completed' )
        names = [ ( m['kind'], m['name'] ) for m in self.wire.messages ]
        self.assertEqual( names[0], ( 'result', 'job.submit' ) )
        self.assertEqual( self.story( job ), [
            'state:queued', 'state:running',
            'file0:started', 'decision:structure.profile', 'file0:processed',
            'file1:started', 'decision:structure.profile', 'file1:processed',
            'state:completed' ] )
        output = [ m['data'] for m in self.wire.events( 'job.file' ) if 'output' in m['data'] ]
        import betaconst
        self.assertEqual( output[0]['output'], os.path.join( betaconst.video_path_censored, 'a.mp4.out.mp4' ),
                          "a file outside the input folder writes to the output folder itself" )

    def test_jobs_run_in_order_one_at_a_time( self ):
        self.pipeline.auto = False
        first = self.submit( 'a.mp4' )
        second = self.ok( 'job.submit', { 'files': [ self.files['b.mp4'] ] } )
        self.assertEqual( second['position'], 1 )
        self.wait_state( first, 'running' )
        for _ in range( 6 ):
            self.pipeline.release.release()
        self.wait_state( second['job'], 'completed' )
        self.assertEqual( self.pipeline.started, [ 'a.mp4', 'b.mp4' ] )
        events = [ m['data'] for m in self.wire.events( 'job.state' ) ]
        order = [ ( e['job'], e['state'] ) for e in events if e['state'] in ( 'running', 'completed' ) ]
        self.assertEqual( order, [ ( first, 'running' ), ( first, 'completed' ),
                                   ( second['job'], 'running' ), ( second['job'], 'completed' ) ] )

    def test_a_failed_file_does_not_stop_the_queue( self ):
        job = self.submit( 'a.mp4', 'bad.mp4', 'c.mp4' )
        final = self.wait_state( job, 'completed' )
        self.assertEqual( final['counts'], { 'total': 3, 'processed': 2, 'skipped': 0, 'failed': 1 } )
        self.assertIn( 'error:file.failed', self.story( job ) )
        self.assertEqual( self.story( job ).index( 'error:file.failed' ) + 1,
                          self.story( job ).index( 'file1:failed' ) )

    def test_a_full_disk_stops_the_job_and_every_queued_one( self ):
        self.pipeline.auto = False
        first = self.submit( 'full.mp4', 'a.mp4' )
        second = self.submit( 'b.mp4' )
        self.wait_state( first, 'running' )
        for _ in range( 3 ):
            self.pipeline.release.release()
        stopped = self.wait_state( first, 'stopped' )
        self.assertEqual( stopped['code'], 'disk.full' )
        self.assertEqual( self.wait_state( second, 'stopped' )['code'], 'disk.full' )
        self.assertEqual( self.pipeline.started, [ 'full.mp4' ] )

    def test_bad_requests( self ):
        self.assertEqual( self.refused( 'job.submit', {}, 'command.bad_params' ),
                          { 'field': 'files', 'reason': 'missing' } )
        self.assertEqual( self.refused( 'job.submit', { 'files': [] }, 'command.bad_params' ),
                          { 'field': 'files', 'reason': 'value' } )
        self.assertEqual( self.refused( 'job.submit', { 'files': [ 'a.mp4' ] }, 'command.bad_params' ),
                          { 'field': 'files', 'reason': 'value' } )
        self.assertEqual( self.refused( 'job.submit', { 'files': [ self.files['a.mp4'], self.tmp + '/nope.mp4' ] },
                                        'job.no_such_file' ), { 'index': 1 } )
        self.assertEqual( self.refused( 'job.submit', { 'files': [ self.files['a.mp4'] ],
                                                        'overrides': { 'encode_crff': 1 } },
                                        'config.unknown_key' )['path'], 'encode_crff' )
        self.refused( 'job.submit', { 'files': [ self.files['a.mp4'] ], 'overrides': { 'encode_crf': 'x' } },
                      'config.invalid' )
        self.assertEqual( self.refused( 'job.submit', { 'files': [ self.files['a.mp4'] ],
                                                        'overrides': { 'input_delete_probability': 0.5 } },
                                        'job.unsafe_setting' ), { 'path': 'input_delete_probability' } )
        self.refused( 'job.submit', { 'files': [ self.files['a.mp4'] ], 'overrides': { 'items_to_censor': [] } },
                      'job.nothing_to_render' )
        self.assertFalse( self.server.jobs.busy )
        self.assertEqual( self.lock_calls, [], "a refused job never takes the lock" )
        self.assertEqual( self.pipeline.started, [] )

    def test_a_job_queued_behind_another_is_checked_when_it_starts( self ):
        self.pipeline.auto = False
        first = self.submit( 'a.mp4' )
        second = self.ok( 'job.submit', { 'files': [ self.files['b.mp4'] ],
                                          'overrides': { 'encode_crf': 'x' } } )['job']
        for _ in range( 3 ):
            self.pipeline.release.release()
        self.wait_state( first, 'completed' )
        stopped = self.wait_state( second, 'stopped' )
        self.assertEqual( stopped['code'], 'config.invalid' )
        self.assertEqual( self.story( second ), [ 'state:queued', 'state:running', 'error:config.invalid',
                                                  'state:stopped' ] )

    def test_overrides_reach_the_pipeline_and_are_put_back( self ):
        seen = []
        real = self.pipeline.process_one_video

        def spy( *args, **kwargs ):
            seen.append( betaconfig.encode_crf )
            return real( *args, **kwargs )

        self.pipeline.process_one_video = spy
        before = betaconfig.encode_crf
        job = self.submit( 'a.mp4', overrides={ 'encode_crf': 30 } )
        self.wait_state( job, 'completed' )
        self.assertEqual( seen, [ 30 ] )
        self.assertEqual( betaconfig.encode_crf, before )

    def test_a_model_that_will_not_load_stops_the_job_with_its_code( self ):
        self.pipeline.bu_detector = types.SimpleNamespace(
            get_detector=lambda *a: types.SimpleNamespace( get_session=mock.Mock( side_effect=OSError( 'x' ) ) ) )
        job = self.submit( 'a.mp4' )
        stopped = self.wait_state( job, 'stopped' )
        self.assertEqual( ( stopped['code'], stopped['params'] ), ( 'model.load_failed', { 'exception_type': 'OSError' } ) )


class TestPauseResumeCancel( _QueueTest ):

    def setUp( self ):
        super().setUp()
        self.pipeline.auto = False

    def test_pause_then_resume_continues_and_holds_the_queue( self ):
        job = self.submit( 'a.mp4', 'b.mp4' )
        behind = self.submit( 'c.mp4' )
        self.wait_state( job, 'running' )
        self.pipeline.release.release()       # step 0 of a.mp4
        self.wire.wait_for( lambda m: m['name'] == 'job.progress', 'progress' )
        self.assertEqual( self.ok( 'job.pause', { 'job': job } ), { 'job': job, 'state': 'running' } )
        self.pipeline.release.release()       # lets a.mp4 reach its next check
        self.wait_state( job, 'paused' )
        time.sleep( 0.2 )
        self.assertEqual( self.pipeline.started, [ 'a.mp4' ], "the paused job holds the queue" )
        self.assertTrue( self.server.jobs.busy )
        self.refused( 'job.pause', { 'job': job }, 'job.bad_state' )
        self.assertEqual( self.ok( 'job.resume', { 'job': job } )['state'], 'paused' )
        self.pipeline.auto = True
        self.pipeline.release.release()
        self.wait_state( behind, 'completed' )
        self.assertEqual( self.pipeline.started, [ 'a.mp4', 'a.mp4', 'b.mp4', 'c.mp4' ],
                          "the paused file restarts (from its checkpoints); the next ones follow" )
        self.assertEqual( self.story( job ), [
            'state:queued', 'state:running', 'file0:started', 'decision:structure.profile', 'state:paused',
            'state:running', 'file0:started', 'decision:structure.profile', 'file0:processed',
            'file1:started', 'decision:structure.profile', 'file1:processed', 'state:completed' ] )

    def test_cancel_a_running_job( self ):
        job = self.submit( 'a.mp4' )
        self.wait_state( job, 'running' )
        self.assertEqual( self.ok( 'job.cancel', { 'job': job } )['state'], 'running' )
        self.pipeline.release.release()
        self.wait_state( job, 'cancelled' )
        self.assertEqual( self.story( job )[-1], 'state:cancelled' )
        self.refused( 'job.cancel', { 'job': job }, 'job.bad_state' )

    def test_cancel_a_queued_and_a_paused_job( self ):
        job = self.submit( 'a.mp4' )
        queued = self.submit( 'b.mp4' )
        self.assertEqual( self.ok( 'job.cancel', { 'job': queued } )['state'], 'cancelled' )
        self.wire.wait_for( lambda m: m['name'] == 'job.file' and m['data']['job'] == job, 'file started' )
        self.ok( 'job.pause', { 'job': job } )
        self.pipeline.release.release()
        self.wait_state( job, 'paused' )
        self.assertEqual( self.ok( 'job.cancel', { 'job': job } )['state'], 'cancelled' )
        self.assertFalse( self.server.jobs.busy )
        self.assertEqual( self.pipeline.started, [ 'a.mp4' ] )
        self.assertEqual( self.story( queued ), [ 'state:queued', 'state:cancelled' ] )

    def test_wrong_state_and_unknown_job( self ):
        job = self.submit( 'a.mp4' )
        queued = self.submit( 'b.mp4' )
        self.wait_state( job, 'running' )
        self.assertEqual( self.refused( 'job.pause', { 'job': queued }, 'job.bad_state' ),
                          { 'job': queued, 'state': 'queued' } )
        self.refused( 'job.resume', { 'job': job }, 'job.bad_state' )
        self.assertEqual( self.refused( 'job.pause', { 'job': 'nope' }, 'job.unknown' ), { 'job': 'nope' } )
        self.assertEqual( self.refused( 'job.cancel', {}, 'command.bad_params' ),
                          { 'field': 'job', 'reason': 'missing' } )

    def test_a_pause_that_finds_no_safe_point_does_not_reach_the_next_job( self ):
        self.pipeline.steps = 1
        job = self.submit( 'a.mp4' )
        self.wait_state( job, 'running' )
        # The pause lands after the file's last check: the job completes.
        real = self.pipeline.process_one_video

        def pause_late( *args, **kwargs ):
            outcome = real( *args, **kwargs )
            bu_signals.request_stop( bu_signals.PAUSE )
            return outcome
        self.pipeline.process_one_video = pause_late
        self.pipeline.release.release()
        self.pipeline.auto = True
        # a.mp4 was already inside the old function; finish with the new one for b.
        self.wait_state( job, 'completed' )
        self.pipeline.process_one_video = real
        later = self.submit( 'b.mp4' )
        self.wait_state( later, 'completed' )


class TestPreviewOverrides( _QueueTest ):
    """The UI's Preview is a job with the preview settings as overrides, not a job kind of its own."""

    def _spy( self ):
        seen = []
        real = self.pipeline.process_one_video

        def spy( root, fname, censored_folder, index, total, session, preview, preview_max, *rest, **kwargs ):
            seen.append( ( preview, preview_max ) )
            return real( root, fname, censored_folder, index, total, session, preview, preview_max, *rest, **kwargs )

        self.pipeline.process_one_video = spy
        return seen

    def test_preview_overrides_reach_the_pipeline( self ):
        seen = self._spy()
        job = self.submit( 'a.mp4', overrides={ 'preview_mode_enabled': True, 'preview_max_seconds': 20 } )
        self.wait_state( job, 'completed' )
        self.assertEqual( seen, [ ( True, 20 ) ] )

    def test_a_full_run_can_turn_preview_off_over_a_config_that_has_it_on( self ):
        seen = self._spy()
        with mock.patch.object( betaconfig, 'preview_mode_enabled', True, create=True ):
            job = self.submit( 'a.mp4', overrides={ 'preview_mode_enabled': False } )
            self.wait_state( job, 'completed' )
        self.assertEqual( [ preview for preview, _ in seen ], [ False ] )


class TestListAndMove( _QueueTest ):

    def setUp( self ):
        super().setUp()
        self.pipeline.auto = False

    def jobs( self ):
        return self.ok( 'job.list' )['jobs']

    def order( self ):
        return [ ( row['job'], row['position'] ) for row in self.jobs() if row['position'] is not None ]

    def release_all( self ):
        self.pipeline.auto = True
        for _ in range( 20 ):
            self.pipeline.release.release()

    def test_list_is_empty_when_nothing_was_submitted( self ):
        self.assertEqual( self.jobs(), [] )

    def test_list_shows_the_active_job_then_the_waiting_ones_then_the_ended_ones( self ):
        done = self.submit( 'c.mp4' )
        for _ in range( 3 ):
            self.pipeline.release.release()
        self.wait_state( done, 'completed' )
        first = self.submit( 'a.mp4', 'b.mp4', overrides={ 'encode_crf': 30 } )
        second = self.submit( 'b.mp4' )
        self.wait_state( first, 'running' )
        rows = self.jobs()
        self.assertEqual( [ ( row['job'], row['state'], row['position'] ) for row in rows ],
                          [ ( first, 'running', 0 ), ( second, 'queued', 1 ), ( done, 'completed', None ) ] )
        self.assertEqual( rows[0]['files'], [ self.files['a.mp4'], self.files['b.mp4'] ] )
        self.assertEqual( rows[0]['overrides'], { 'encode_crf': 30 } )
        self.assertEqual( rows[0]['counts'], { 'total': 2, 'processed': 0, 'skipped': 0, 'failed': 0 } )
        self.assertEqual( rows[2]['counts']['processed'], 1 )
        self.assertNotIn( 'code', rows[0] )

    def test_ended_jobs_are_newest_first_and_a_stopped_one_carries_its_code( self ):
        older = self.submit( 'c.mp4' )
        self.release_all()
        self.wait_state( older, 'completed' )
        newer = self.submit( 'full.mp4' )
        self.release_all()
        self.wait_state( newer, 'stopped' )
        rows = self.jobs()
        self.assertEqual( [ row['job'] for row in rows ], [ newer, older ] )
        self.assertEqual( rows[0]['code'], 'disk.full' )
        self.assertIn( 'params', rows[0] )

    def test_a_paused_job_holding_the_queue_is_at_position_zero( self ):
        job = self.submit( 'a.mp4' )
        behind = self.submit( 'b.mp4' )
        self.wire.wait_for( lambda m: m['name'] == 'job.file' and m['data']['job'] == job, 'file started' )
        self.ok( 'job.pause', { 'job': job } )
        self.pipeline.release.release()
        self.wait_state( job, 'paused' )
        self.assertEqual( self.order(), [ ( job, 0 ), ( behind, 1 ) ] )

    def test_move_reorders_the_waiting_jobs_and_they_run_in_the_new_order( self ):
        first = self.submit( 'a.mp4' )
        b = self.submit( 'b.mp4' )
        c = self.submit( 'c.mp4' )
        self.wait_state( first, 'running' )
        self.assertEqual( self.ok( 'job.move', { 'job': c, 'position': 1 } ), { 'job': c, 'position': 1 } )
        self.assertEqual( self.order(), [ ( first, 0 ), ( c, 1 ), ( b, 2 ) ] )
        self.release_all()
        self.wait_state( b, 'completed' )
        self.assertEqual( self.pipeline.started, [ 'a.mp4', 'c.mp4', 'b.mp4' ] )

    def test_move_to_the_back( self ):
        first = self.submit( 'a.mp4' )
        b = self.submit( 'b.mp4' )
        c = self.submit( 'c.mp4' )
        self.wait_state( first, 'running' )
        self.assertEqual( self.ok( 'job.move', { 'job': b, 'position': 2 } )['position'], 2 )
        self.assertEqual( self.order(), [ ( first, 0 ), ( c, 1 ), ( b, 2 ) ] )

    def test_a_position_past_either_end_is_taken_as_that_end( self ):
        first = self.submit( 'a.mp4' )
        b = self.submit( 'b.mp4' )
        c = self.submit( 'c.mp4' )
        self.wait_state( first, 'running' )
        self.assertEqual( self.ok( 'job.move', { 'job': c, 'position': 0 } )['position'], 1,
                          "nothing goes ahead of the running job" )
        self.assertEqual( self.order(), [ ( first, 0 ), ( c, 1 ), ( b, 2 ) ] )
        self.assertEqual( self.ok( 'job.move', { 'job': c, 'position': 99 } )['position'], 2 )
        self.assertEqual( self.order(), [ ( first, 0 ), ( b, 1 ), ( c, 2 ) ] )

    def test_moving_ahead_of_a_paused_job_is_not_possible( self ):
        job = self.submit( 'a.mp4' )
        behind = self.submit( 'b.mp4' )
        self.wire.wait_for( lambda m: m['name'] == 'job.file' and m['data']['job'] == job, 'file started' )
        self.ok( 'job.pause', { 'job': job } )
        self.pipeline.release.release()
        self.wait_state( job, 'paused' )
        self.assertEqual( self.ok( 'job.move', { 'job': behind, 'position': 0 } )['position'], 1 )
        self.assertEqual( self.order(), [ ( job, 0 ), ( behind, 1 ) ] )

    def test_only_a_waiting_job_moves( self ):
        running = self.submit( 'a.mp4' )
        queued = self.submit( 'b.mp4' )
        self.wait_state( running, 'running' )
        self.assertEqual( self.refused( 'job.move', { 'job': running, 'position': 1 }, 'job.bad_state' ),
                          { 'job': running, 'state': 'running' } )
        self.ok( 'job.cancel', { 'job': queued } )
        self.assertEqual( self.refused( 'job.move', { 'job': queued, 'position': 0 }, 'job.bad_state' ),
                          { 'job': queued, 'state': 'cancelled' } )

    def test_move_bad_requests( self ):
        job = self.submit( 'a.mp4' )
        self.assertEqual( self.refused( 'job.move', { 'job': job }, 'command.bad_params' ),
                          { 'field': 'position', 'reason': 'missing' } )
        self.assertEqual( self.refused( 'job.move', { 'job': job, 'position': '1' }, 'command.bad_params' ),
                          { 'field': 'position', 'reason': 'type' } )
        self.assertEqual( self.refused( 'job.move', { 'job': job, 'position': True }, 'command.bad_params' ),
                          { 'field': 'position', 'reason': 'type' } )
        self.assertEqual( self.refused( 'job.move', { 'job': job, 'position': -1 }, 'command.bad_params' ),
                          { 'field': 'position', 'reason': 'value' } )
        self.assertEqual( self.refused( 'job.move', { 'job': 'nope', 'position': 0 }, 'job.unknown' ),
                          { 'job': 'nope' } )


class TestEstimate( _QueueTest ):
    """job.estimate through the server: which settings it reads, idle and busy. The arithmetic is in test_estimate."""

    def setUp( self ):
        super().setUp()
        import betautils_serve_catalog as bu_serve_catalog
        import betautils_serve_estimate as bu_serve_estimate
        self.bu_serve_estimate = bu_serve_estimate
        patcher = mock.patch.object( bu_serve_catalog, '_open_quietly', return_value=None )
        patcher.start()
        self.addCleanup( patcher.stop )
        self.seen = []
        real = bu_serve_estimate.answer

        def spy( files, ctx, *rest, **kwargs ):
            self.seen.append( ctx )
            return real( files, ctx, *rest, **kwargs )

        patcher = mock.patch.object( bu_serve_estimate, 'answer', side_effect=spy )
        patcher.start()
        self.addCleanup( patcher.stop )

    def estimate( self, *names, overrides=None ):
        data = { 'files': [ self.files[name] for name in names ] }
        if overrides is not None:
            data['overrides'] = overrides
        return self.ok( 'job.estimate', data )

    def test_an_unreadable_file_and_no_history_give_no_estimate( self ):
        out = self.estimate( 'a.mp4', 'b.mp4' )
        self.assertEqual( out, { 'files': [ { 'seconds': None, 'detect_cache': None } ] * 2, 'seconds': None,
                                 'basis': 'none', 'runs': { 'detect': 0, 'render': 0 }, 'look_seen': False } )

    def test_idle_it_reads_the_settings_with_the_overrides_and_changes_nothing( self ):
        before = ( getattr( betaconfig, 'preview_mode_enabled', False ), betaconfig.encode_preset )
        self.estimate( 'a.mp4', overrides={ 'preview_mode_enabled': True, 'preview_max_seconds': 7,
                                            'encode_preset': 'slow' } )
        [ ctx ] = self.seen
        self.assertEqual( ( ctx['preview'], ctx['preview_max'] ), ( True, 7 ) )
        self.assertEqual( ctx['identity']['preset'], getattr( betaconfig, 'preview_encode_preset', 'ultrafast' ) )
        self.estimate( 'a.mp4', overrides={ 'encode_preset': 'slow' } )
        self.assertEqual( self.seen[1]['identity']['preset'], 'slow' )
        self.assertEqual( ( getattr( betaconfig, 'preview_mode_enabled', False ), betaconfig.encode_preset ), before )

    def test_busy_it_uses_the_settings_taken_when_the_queue_became_busy( self ):
        self.pipeline.auto = False
        job = self.submit( 'a.mp4', overrides={ 'encode_preset': 'veryslow' } )
        self.wait_state( job, 'running' )
        taken = self.server.jobs.estimate_context
        self.assertIsNotNone( taken )
        self.estimate( 'b.mp4', overrides={ 'encode_preset': 'slow', 'preview_mode_enabled': True } )
        [ ctx ] = self.seen
        self.assertTrue( ctx['preview'] )
        self.assertEqual( { key: value for key, value in ctx.items() if key not in ( 'preview', 'identity' ) },
                          { key: value for key, value in taken.items() if key not in ( 'preview', 'identity' ) } )
        self.assertNotEqual( ctx['identity']['preset'], 'slow', "only the preview overrides apply while busy" )
        self.pipeline.auto = True
        for _ in range( 3 ):
            self.pipeline.release.release()
        self.wait_state( job, 'completed' )
        self.assertIsNone( self.server.jobs.estimate_context, "forgotten once the queue is idle" )

    def test_busy_with_no_settings_taken_is_refused_not_evaluated( self ):
        self.pipeline.auto = False
        job = self.submit( 'a.mp4' )
        self.wait_state( job, 'running' )
        self.server.jobs.estimate_context = None
        self.assertEqual( self.refused( 'job.estimate', { 'files': [ self.files['b.mp4'] ] }, 'config.busy' ),
                          { 'job': job } )
        self.assertEqual( self.seen, [] )

    def test_bad_requests( self ):
        self.assertEqual( self.refused( 'job.estimate', {}, 'command.bad_params' ),
                          { 'field': 'files', 'reason': 'missing' } )
        self.assertEqual( self.refused( 'job.estimate', { 'files': [ 'a.mp4' ] }, 'command.bad_params' ),
                          { 'field': 'files', 'reason': 'value' } )
        self.assertEqual( self.refused( 'job.estimate', { 'files': [ self.tmp + '/nope.mp4' ] }, 'job.no_such_file' ),
                          { 'index': 0 } )
        self.refused( 'job.estimate', { 'files': [ self.files['a.mp4'] ], 'overrides': { 'encode_crff': 1 } },
                      'config.unknown_key' )
        self.refused( 'job.estimate', { 'files': [ self.files['a.mp4'] ], 'overrides': { 'encode_crf': 'x' } },
                      'config.invalid' )
        self.assertFalse( self.server.jobs.busy )


class TestRaces( _QueueTest ):
    """Windows the independent review found, each held open by the test."""

    def test_shutdown_while_the_worker_checks_a_job_stops_it_before_any_file( self ):
        checking = threading.Event()
        release = threading.Event()
        real = bu_serve_job.start_checks

        def slow_checks( overrides ):
            if threading.current_thread().name == 'job-worker':
                checking.set()
                release.wait( WAIT )
            return real( overrides )

        with mock.patch.object( bu_serve_job, 'start_checks', slow_checks ):
            job = self.submit( 'a.mp4' )
            self.assertTrue( checking.wait( WAIT ) )
            stopper = threading.Thread( target=self.server.stop_jobs )
            stopper.start()
            time.sleep( 0.1 )
            release.set()
            stopper.join( WAIT )
        self.assertFalse( stopper.is_alive() )
        self.assertEqual( self.wait_state( job, 'stopped' )['code'], 'engine.stopping' )
        self.assertEqual( self.pipeline.started, [], "no file ran while shutdown waited" )

    def test_a_cancel_that_lands_while_the_job_is_pausing_still_cancels_it( self ):
        self.pipeline.auto = False
        real_run = self.server.jobs.runner.run

        def cancel_after_pausing( running_job ):
            state = real_run( running_job )
            if state == 'paused':
                # The cancel arrives after the runner took the pause, before
                # the worker looked again: the job is paused, the request
                # is pending.
                bu_signals.request_stop( bu_signals.CANCEL )
            return state

        with mock.patch.object( self.server.jobs.runner, 'run', cancel_after_pausing ):
            job = self.submit( 'a.mp4' )
            self.wait_state( job, 'running' )
            self.ok( 'job.pause', { 'job': job } )
            self.pipeline.release.release()
            self.wait_state( job, 'cancelled' )
        self.assertFalse( self.server.jobs.busy )
        self.assertFalse( bu_signals.interrupted() )

    def test_the_worker_outlives_a_job_that_raises_anything( self ):
        real = self.pipeline.process_one_video

        def interrupt_first( root, fname, *args, **kwargs ):
            if fname == 'a.mp4':
                raise KeyboardInterrupt()
            return real( root, fname, *args, **kwargs )

        self.pipeline.process_one_video = interrupt_first
        first = self.submit( 'a.mp4' )
        stopped = self.wait_state( first, 'stopped' )
        self.assertEqual( ( stopped['code'], stopped['params'] ),
                          ( 'engine.internal', { 'exception_type': 'KeyboardInterrupt' } ) )
        second = self.submit( 'b.mp4' )
        self.wait_state( second, 'completed' )


class TestShutdown( _QueueTest ):

    def test_shutdown_stops_the_running_job_and_the_queued_ones( self ):
        self.pipeline.auto = False
        job = self.submit( 'a.mp4' )
        queued = self.submit( 'b.mp4' )
        self.wait_state( job, 'running' )
        stopper = threading.Thread( target=self.server.stop_jobs )
        stopper.start()
        time.sleep( 0.1 )
        self.pipeline.release.release()
        stopper.join( WAIT )
        self.assertFalse( stopper.is_alive() )
        self.assertEqual( self.wait_state( job, 'stopped' )['code'], 'engine.stopping' )
        self.assertEqual( self.wait_state( queued, 'stopped' )['code'], 'engine.stopping' )
        self.assertFalse( bu_signals.interrupted() )
        self.refused( 'job.submit', { 'files': [ self.files['c.mp4'] ] }, 'engine.stopping' )

    def test_a_paused_job_is_stopped_by_shutdown( self ):
        self.pipeline.auto = False
        job = self.submit( 'a.mp4' )
        self.wait_state( job, 'running' )
        self.ok( 'job.pause', { 'job': job } )
        self.pipeline.release.release()
        self.wait_state( job, 'paused' )
        self.server.stop_jobs()
        self.assertEqual( self.wait_state( job, 'stopped' )['code'], 'engine.stopping' )


class TestRunLock( _QueueTest ):

    def test_the_lock_is_held_while_busy_and_given_back_when_idle( self ):
        self.pipeline.auto = False
        job = self.submit( 'a.mp4' )
        self.wait_state( job, 'running' )
        self.assertIsNotNone( bu_runlock.probe() )
        self.assertEqual( bu_runlock._read_holder_of_live_lock( bu_runlock.default_path() )['program'], 'engine' )
        for _ in range( 3 ):
            self.pipeline.release.release()
        self.wait_state( job, 'completed' )
        deadline = time.monotonic() + WAIT
        while bu_runlock.probe() is not None and time.monotonic() < deadline:
            time.sleep( 0.01 )
        self.assertIsNone( bu_runlock.probe() )

    def record_states( self ):
        """[ ( job, state, lock free as it was sent ) ] for every job.state sent from here on."""
        seen = []
        events = self.server.jobs.events
        send = events._raw_send

        def recording( name, data ):
            if name == 'job.state':
                seen.append( ( data['job'], data['state'], bu_runlock.probe() is None ) )
            send( name, data )

        events._raw_send = recording
        return seen

    def final( self, seen, job ):
        return [ ( state, free ) for seen_job, state, free in seen
                 if seen_job == job and state in ( 'completed', 'cancelled', 'stopped' ) ]

    def test_the_lock_is_free_when_the_last_job_reports_completed( self ):
        seen = self.record_states()
        job = self.submit( 'a.mp4' )
        self.wait_state( job, 'completed' )
        self.assertEqual( self.final( seen, job ), [ ( 'completed', True ) ] )
        self.assertIsNone( bu_runlock.probe(), "free as soon as the client reads completed" )
        held = bu_runlock.acquire_library( 'betatv' )
        held.release()

    def test_the_lock_is_free_when_a_running_or_paused_job_reports_cancelled( self ):
        seen = self.record_states()
        self.pipeline.auto = False
        running = self.submit( 'a.mp4' )
        self.wait_state( running, 'running' )
        self.ok( 'job.cancel', { 'job': running } )
        self.pipeline.release.release()
        self.wait_state( running, 'cancelled' )
        paused = self.submit( 'b.mp4' )
        self.wait_state( paused, 'running' )
        self.ok( 'job.pause', { 'job': paused } )
        self.pipeline.release.release()
        self.wait_state( paused, 'paused' )
        self.ok( 'job.cancel', { 'job': paused } )
        self.assertEqual( self.final( seen, running ), [ ( 'cancelled', True ) ] )
        self.assertEqual( self.final( seen, paused ), [ ( 'cancelled', True ) ] )

    def test_the_lock_stays_held_while_another_job_is_left( self ):
        seen = self.record_states()
        self.pipeline.auto = False
        running = self.submit( 'a.mp4' )
        self.wait_state( running, 'running' )
        queued = self.submit( 'b.mp4' )
        self.ok( 'job.cancel', { 'job': queued } )
        self.assertEqual( self.final( seen, queued ), [ ( 'cancelled', False ) ] )
        for _ in range( 3 ):
            self.pipeline.release.release()
        self.wait_state( running, 'completed' )
        self.assertEqual( self.final( seen, running ), [ ( 'completed', True ) ] )

    def test_a_full_disk_gives_the_lock_back_with_the_last_stopped_job( self ):
        seen = self.record_states()
        self.pipeline.auto = False
        first = self.submit( 'full.mp4' )
        self.wait_state( first, 'running' )
        second = self.submit( 'a.mp4' )
        for _ in range( 3 ):
            self.pipeline.release.release()
        self.wait_state( second, 'stopped' )
        self.assertEqual( self.final( seen, first ), [ ( 'stopped', False ) ] )
        self.assertEqual( self.final( seen, second ), [ ( 'stopped', True ) ] )

    def test_a_job_submitted_as_the_last_one_ends_runs_under_the_lock( self ):
        # The submit lands after the final state went out, before the
        # worker looks at the queue again: it must take the lock itself.
        seen = self.record_states()
        real_run = self.server.jobs.runner.run
        later = []

        def submit_in_the_gap( job ):
            state = real_run( job )
            if not later:
                later.append( self.submit( 'b.mp4' ) )
            return state

        with mock.patch.object( self.server.jobs.runner, 'run', submit_in_the_gap ):
            first = self.submit( 'a.mp4' )
            self.wait_state( first, 'completed' )
            self.wait_until( lambda: later )
            self.wait_state( later[0], 'completed' )
        self.assertEqual( len( self.lock_calls ), 2 )
        self.assertIn( ( later[0], 'running', False ), seen, "the second job ran under the lock" )
        self.assertEqual( self.final( seen, later[0] ), [ ( 'completed', True ) ] )

    def wait_until( self, predicate ):
        deadline = time.monotonic() + WAIT
        while not predicate() and time.monotonic() < deadline:
            time.sleep( 0.01 )
        self.assertTrue( predicate() )

    def test_a_command_line_run_holding_the_library_refuses_a_job( self ):
        held = bu_runlock.acquire_library( 'betatv' )
        self.addCleanup( held.release )
        params = self.refused( 'job.submit', { 'files': [ self.files['a.mp4'] ] }, 'run.locked' )
        self.assertEqual( params['program'], 'betatv' )
        self.assertFalse( self.server.jobs.busy )

    def test_a_running_job_refuses_a_command_line_run( self ):
        import betatv
        self.pipeline.auto = False
        job = self.submit( 'a.mp4' )
        self.wait_state( job, 'running' )
        with mock.patch.object( betatv, '_startup', return_value=mock.Mock() ) as startup:
            self.assertEqual( betatv.main(), bu_runlock.EXIT_LOCKED )
        self.assertIn( 'engine (pid', startup.return_value.error.call_args[0][0] )
        for _ in range( 3 ):
            self.pipeline.release.release()
        self.wait_state( job, 'completed' )


class TestSettingsWhileBusy( _QueueTest ):

    def setUp( self ):
        super().setUp()
        self.local = os.path.join( self.tmp, 'betaconfig_local.py' )
        bu_serve_config.reset_store_for_tests()
        import betautils_config_store as bu_store
        bu_serve_config._STORE = bu_store.ConfigStore( path=self.local )
        self.addCleanup( bu_serve_config.reset_store_for_tests )
        self.pipeline.auto = False

    def test_set_and_reset_are_refused_and_write_nothing_then_accepted( self ):
        job = self.submit( 'a.mp4' )
        self.wait_state( job, 'running' )
        self.assertEqual( self.refused( 'config.set', { 'values': { 'encode_crf': 20 } }, 'config.busy' ),
                          { 'job': job } )
        self.refused( 'config.reset', { 'paths': [ 'encode_crf' ] }, 'config.busy' )
        self.assertFalse( os.path.exists( self.local ), "nothing was written" )
        before = betaconfig.encode_crf
        for _ in range( 3 ):
            self.pipeline.release.release()
        self.wait_state( job, 'completed' )
        deadline = time.monotonic() + WAIT
        while self.server.jobs.busy and time.monotonic() < deadline:
            time.sleep( 0.01 )
        self.assertEqual( betaconfig.encode_crf, before )
        self.assertTrue( self.ok( 'config.set', { 'values': { 'encode_crf': 20 } } )['written'] )

    def test_set_is_accepted_as_soon_as_the_last_job_reports_completed( self ):
        job = self.submit( 'a.mp4' )
        self.release_all()
        self.wait_state( job, 'completed' )
        self.assertTrue( self.ok( 'config.set', { 'values': { 'encode_crf': 20 } } )['written'] )

    def test_get_answers_from_the_snapshot_and_refuses_overrides( self ):
        idle = self.ok( 'config.get', { 'paths': [ 'encode_crf' ] } )
        job = self.submit( 'a.mp4' )
        self.wait_state( job, 'running' )
        with mock.patch( 'betautils_config_store.ConfigStore.evaluate',
                         side_effect=AssertionError( "the live config must not be touched" ) ):
            busy = self.ok( 'config.get', { 'paths': [ 'encode_crf' ] } )
            self.refused( 'config.get', { 'overrides': { 'encode_crf': 30 } }, 'config.busy' )
            self.ok( 'config.schema' )
        self.assertEqual( busy, idle )
        self.release_all()

    def test_a_job_queued_while_paused_also_keeps_settings_busy( self ):
        job = self.submit( 'a.mp4' )
        self.wait_state( job, 'running' )
        self.ok( 'job.pause', { 'job': job } )
        self.pipeline.release.release()
        self.wait_state( job, 'paused' )
        self.refused( 'config.set', { 'values': { 'encode_crf': 20 } }, 'config.busy' )

    def release_all( self ):
        self.pipeline.auto = True
        for _ in range( 5 ):
            self.pipeline.release.release()


class TestEvents( unittest.TestCase ):

    def test_progress_is_coalesced_and_other_events_flush_it_first( self ):
        sent = []
        now = [ 100.0 ]
        events = bu_serve_job.Events( lambda name, data: sent.append( ( name, dict( data ) ) ),
                                      interval=0.25, clock=lambda: now[0] )
        self.addCleanup( events.close )

        def progress( done, stage='detect' ):
            events( 'job.progress', { 'job': 'j', 'file': 0, 'stage': stage, 'done': done,
                                      'total': 100, 'rate': 1.0 } )

        progress( 1 )                     # first one goes at once
        for done in range( 2, 50 ):
            progress( done )              # within the interval: only the latest is kept
        progress( 3, stage='render' )
        self.assertEqual( [ data['done'] for _name, data in sent ], [ 1 ] )
        events( 'job.file', { 'job': 'j', 'file': 0, 'state': 'processed' } )
        self.assertEqual( [ ( name, data.get( 'stage' ), data.get( 'done' ) ) for name, data in sent ], [
            ( 'job.progress', 'detect', 1 ), ( 'job.progress', 'detect', 49 ),
            ( 'job.progress', 'render', 3 ), ( 'job.file', None, None ) ] )

    def test_waiting_progress_is_sent_without_another_event( self ):
        sent = []
        events = bu_serve_job.Events( lambda name, data: sent.append( data['done'] ), interval=0.05 )
        self.addCleanup( events.close )
        events( 'job.progress', { 'job': 'j', 'file': 0, 'stage': 'render', 'done': 1, 'total': 2, 'rate': 1.0 } )
        events( 'job.progress', { 'job': 'j', 'file': 0, 'stage': 'render', 'done': 2, 'total': 2, 'rate': 1.0 } )
        deadline = time.monotonic() + WAIT
        while sent != [ 1, 2 ] and time.monotonic() < deadline:
            time.sleep( 0.01 )
        self.assertEqual( sent, [ 1, 2 ] )

    def test_state_and_error_events_are_never_dropped( self ):
        sent = []
        events = bu_serve_job.Events( lambda name, data: sent.append( name ), interval=10 )
        self.addCleanup( events.close )
        for _ in range( 50 ):
            events( 'job.state', { 'job': 'j', 'state': 'running', 'counts': {} } )
            events( 'job.error', { 'job': 'j', 'file': 0, 'code': 'file.failed',
                                   'params': { 'exception_type': 'X', 'stage': 'render' } } )
        self.assertEqual( len( sent ), 100 )

    def test_an_event_that_cannot_be_sent_is_logged_not_raised( self ):
        logger = mock.Mock()

        def refuse( name, data ):
            raise ValueError( "Out of range float values are not JSON compliant" )

        events = bu_serve_job.Events( refuse, logger=logger )
        self.addCleanup( events.close )
        events( 'job.file', { 'job': 'j', 'file': 0, 'state': 'started' } )
        logger.error.assert_called_once()

    def test_an_unregistered_error_code_is_a_bug( self ):
        events = bu_serve_job.Events( lambda name, data: None )
        self.addCleanup( events.close )
        with self.assertRaises( ValueError ):
            events( 'job.error', { 'job': 'j', 'code': 'no.such', 'params': {} } )
        with self.assertRaises( ValueError ):
            events( 'job.state', { 'job': 'j', 'state': 'stopped', 'counts': {}, 'code': 'disk.full',
                                   'params': {} } )


class TestRegistration( unittest.TestCase ):

    def test_hello_lists_the_job_commands_and_events( self ):
        server = bu_serve.Server( io.BytesIO(), io.BytesIO(), logging.getLogger( 'test_serve_job' ) )
        result = bu_serve._hello( server, { 'protocol': [ 1 ] } )
        for name in ( 'job.submit', 'job.pause', 'job.resume', 'job.cancel', 'job.list', 'job.move' ):
            self.assertIn( name, result['commands'] )
        self.assertEqual( set( bu_serve_job.EVENTS ) - set( result['events'] ), set() )

    def test_the_config_commands_are_the_wrapped_ones( self ):
        commands = bu_serve.registered_commands()
        self.assertIs( commands['config.get'], bu_serve_job.config_get )
        self.assertIsNot( commands['config.set'], bu_serve_config.config_set )


if __name__ == '__main__':
    unittest.main()
