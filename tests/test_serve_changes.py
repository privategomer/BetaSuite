"""
test_serve_changes.py - settings changed while jobs run: apply after
the jobs, pause and apply, cancel and queue again (betautils_serve_changes).

What each test prevents:
  - a change made while a job runs being lost, or written under the
    running job (which reads the settings as it goes)
  - a change that would not pass being accepted to wait, or one that no
    longer passes being written later without telling the client
  - waiting changes lost by an engine restart, or written twice
  - "pause and apply" writing before the job has paused, or leaving it
    paused; "start again" losing a job's remaining files or overrides
  - answers and events drifting from protocol/v1's schemas
"""

import os
import shutil
import tempfile

import betaconfig
import betautils_config_store as bu_store
import betautils_serve_changes as bu_changes
import betautils_serve_config as bu_serve_config

from tests import config_harness as harness
from tests import protocol_schemas as schemas
from tests.test_serve_job import _QueueTest


class _ChangesTest( _QueueTest ):

    def setUp( self ):
        super().setUp()
        # The shipped config, whatever earlier tests left (as test_serve_config does).
        shipped = harness.shipped()
        shipped.__enter__()
        self.addCleanup( shipped.__exit__, None, None, None )
        self.local = os.path.join( self.tmp, 'betaconfig_local.py' )
        bu_serve_config._STORE = bu_store.ConfigStore( path=self.local )
        self.addCleanup( bu_serve_config.reset_store_for_tests )
        self.server.pending = bu_changes.Pending( path=os.path.join( self.tmp, 'state', 'pending_config.json' ) )

    def local_text( self ):
        try:
            with open( self.local, 'r', encoding='utf-8' ) as handle:
                return handle.read()
        except OSError:
            return ''

    def events_named( self, name ):
        return [ m['data'] for m in self.wire.events() if m['name'] == name ]

    def hold_running( self, *names ):
        """A job that is running and waits for the test to release each step."""
        self.pipeline.auto = False
        job = self.submit( *names )
        self.wait_state( job, 'running' )
        return job

    def let_run( self ):
        self.pipeline.auto = True
        for _ in range( 50 ):
            self.pipeline.release.release()


class TestAfterTheJobs( _ChangesTest ):

    def test_waits_while_a_job_runs_and_is_written_when_idle( self ):
        job = self.hold_running( 'a.mp4' )
        self.refused( 'config.set', { 'values': { 'shot_cut_threshold': 0.5 } }, 'config.busy' )
        answer = self.ok( 'config.set', { 'values': { 'shot_cut_threshold': 0.5 }, 'when': 'after_jobs' } )
        self.assertEqual( answer['when'], 'after_jobs' )
        self.assertFalse( answer['written'] )
        self.assertIn( 'censor', answer['moves'] )
        self.assertNotIn( 'shot_cut_threshold', self.local_text(), 'nothing written under the running job' )
        self.assertEqual( [ p['id'] for p in self.ok( 'config.pending' )['pending'] ], [ answer['id'] ] )
        self.let_run()
        self.wait_state( job, 'completed' )
        applied = self.wire.wait_for( lambda m: m.get( 'name' ) == 'config.applied', 'config.applied' )['data']
        self.assertEqual( applied['id'], answer['id'] )
        self.assertIn( 'shot_cut_threshold = 0.5', self.local_text() )
        self.assertEqual( self.ok( 'config.pending' )['pending'], [] )

    def test_with_nothing_running_it_is_written_at_once( self ):
        answer = self.ok( 'config.set', { 'values': { 'shot_cut_threshold': 0.45 }, 'when': 'after_jobs' } )
        self.assertEqual( answer['when'], 'now' )
        self.assertIn( 'shot_cut_threshold = 0.45', self.local_text() )

    def test_a_change_that_would_not_pass_is_refused_now( self ):
        self.hold_running( 'a.mp4' )
        self.refused( 'config.set', { 'values': { 'shot_cut_threshold': 7 }, 'when': 'after_jobs' }, 'config.invalid' )
        self.assertEqual( self.ok( 'config.pending' )['pending'], [] )

    def test_resets_wait_too( self ):
        self.ok( 'config.set', { 'values': { 'shot_cut_threshold': 0.45 } } )
        job = self.hold_running( 'a.mp4' )
        self.ok( 'config.reset', { 'paths': [ 'shot_cut_threshold' ], 'when': 'after_jobs' } )
        self.assertIn( 'shot_cut_threshold', self.local_text() )
        self.let_run()
        self.wait_state( job, 'completed' )
        self.wire.wait_for( lambda m: m.get( 'name' ) == 'config.applied', 'config.applied' )
        self.assertNotIn( 'shot_cut_threshold', self.local_text() )

    def test_discard( self ):
        self.hold_running( 'a.mp4' )
        first = self.ok( 'config.set', { 'values': { 'shot_cut_threshold': 0.5 }, 'when': 'after_jobs' } )['id']
        self.ok( 'config.set', { 'values': { 'shot_cut_burst_window': 0.4 }, 'when': 'after_jobs' } )
        self.assertEqual( self.ok( 'config.pending_discard', { 'id': first } ), { 'discarded': 1 } )
        self.assertEqual( len( self.ok( 'config.pending' )['pending'] ), 1 )
        self.assertEqual( self.ok( 'config.pending_discard' ), { 'discarded': 1 } )
        self.assertEqual( self.ok( 'config.pending' )['pending'], [] )

    def test_bad_when( self ):
        self.refused( 'config.set', { 'values': { 'shot_cut_threshold': 0.5 }, 'when': 'later' }, 'command.bad_params' )


class TestPendingFile( _ChangesTest ):

    def test_survives_a_restart_and_is_written_once( self ):
        pending = self.server.pending
        pending.add( { 'shot_cut_threshold': 0.55 }, [] )
        again = bu_changes.Pending( path=pending.path )
        self.assertEqual( len( again.items() ), 1 )
        sent = []
        self.assertEqual( again.apply( lambda name, data: sent.append( ( name, data ) ) ), 1 )
        self.assertEqual( [ name for name, _ in sent ], [ 'config.applied' ] )
        self.assertIn( 'shot_cut_threshold = 0.55', self.local_text() )
        self.assertEqual( again.items(), [] )
        self.assertEqual( again.apply( lambda *a: None ), 0 )

    def test_one_that_no_longer_passes_is_dropped_and_told( self ):
        sent = []
        self.server.pending.add( { 'shot_cut_threshold': 9 }, [] )
        self.server.pending.add( { 'shot_cut_threshold': 0.4 }, [] )
        self.server.pending.apply( lambda name, data: sent.append( ( name, data ) ) )
        self.assertEqual( [ name for name, _ in sent ], [ 'config.pending_failed', 'config.applied' ] )
        self.assertEqual( sent[0][1]['error']['code'], 'config.invalid' )
        self.assertIn( 'shot_cut_threshold = 0.4', self.local_text() )

    def test_written_at_start_when_nothing_runs( self ):
        self.server.pending.add( { 'shot_cut_threshold': 0.42 }, [] )
        bu_changes.apply_at_start( self.server )
        self.assertIn( 'shot_cut_threshold = 0.42', self.local_text() )

    def test_a_broken_file_is_an_empty_list( self ):
        os.makedirs( os.path.dirname( self.server.pending.path ), exist_ok=True )
        with open( self.server.pending.path, 'w', encoding='utf-8' ) as handle:
            handle.write( '{not json' )
        self.assertEqual( self.server.pending.items(), [] )


class TestRestartWithChanges( _ChangesTest ):

    def step( self, count=1 ):
        for _ in range( count ):
            self.pipeline.release.release()

    def wait_started( self, name ):
        import time
        deadline = time.monotonic() + 20
        while name not in self.pipeline.started:
            self.assertLess( time.monotonic(), deadline, name + ' never started' )
            time.sleep( 0.01 )

    def index_of( self, predicate ):
        return next( i for i, m in enumerate( self.wire.messages ) if predicate( m ) )

    def test_resume_pauses_writes_and_goes_on( self ):
        job = self.hold_running( 'a.mp4', 'b.mp4' )
        answer = self.ok( 'job.restart_with_changes', { 'mode': 'resume', 'values': { 'shot_cut_threshold': 0.5 } } )
        self.assertEqual( answer, { 'mode': 'resume', 'jobs': [ job ] } )
        self.step()                                   # reach the next safe point
        self.wait_state( job, 'paused' )
        self.wire.wait_for( lambda m: m.get( 'name' ) == 'config.applied', 'config.applied' )
        self.assertIn( 'shot_cut_threshold = 0.5', self.local_text() )
        applied_at = self.index_of( lambda m: m.get( 'name' ) == 'config.applied' )
        self.wire.wait_for( lambda m: any( n['kind'] == 'event' and n['name'] == 'job.state'
                                           and n['data']['state'] == 'running'
                                           for n in self.wire.messages[applied_at:] ), 'running again' )
        # Still running: config.get answers from the snapshot, which now has the change.
        rows = { row['path']: row for row in self.ok( 'config.get' )['settings'] }
        self.assertEqual( rows['shot_cut_threshold']['value'], 0.5 )
        self.let_run()
        self.wait_state( job, 'completed' )
        self.assertEqual( self.pipeline.started.count( 'b.mp4' ), 1 )

    def test_requeue_cancels_and_queues_the_rest_with_overrides( self ):
        first = self.hold_running( 'a.mp4', 'b.mp4' )
        second = self.submit( 'c.mp4', overrides={ 'shot_cut_burst_window': 0.3 } )
        self.step( self.pipeline.steps )              # a.mp4 finishes; b.mp4 starts
        self.wait_started( 'b.mp4' )
        answer = self.ok( 'job.restart_with_changes', { 'mode': 'requeue', 'values': { 'shot_cut_threshold': 0.5 } } )
        self.assertEqual( sorted( answer['jobs'] ), sorted( [ first, second ] ) )
        self.let_run()
        self.wait_state( first, 'cancelled' )
        self.wait_state( second, 'cancelled' )
        self.wire.wait_for( lambda m: m.get( 'name' ) == 'config.applied', 'config.applied' )
        cancelled = [ i for i, m in enumerate( self.wire.messages ) if m['kind'] == 'event'
                      and m['name'] == 'job.state' and m['data']['state'] == 'cancelled' ]
        self.assertLess( max( cancelled ), self.index_of( lambda m: m.get( 'name' ) == 'config.applied' ),
                         'written only once nothing ran' )
        self.assertIn( 'shot_cut_threshold = 0.5', self.local_text() )
        for _ in range( 2 ):     # the two requeued jobs, one at a time
            done = { m['data']['job'] for m in self.wire.events()
                     if m['name'] == 'job.state' and m['data']['state'] == 'completed' }
            self.wire.wait_for( lambda m: m['kind'] == 'event' and m['name'] == 'job.state'
                                and m['data']['state'] == 'completed'
                                and m['data']['job'] not in { first, second } | done, 'requeued job' )
        jobs = self.server.jobs.jobs
        requeued = [ job for job_id, job in jobs.items() if job_id not in ( first, second ) ]
        files = sorted( os.path.basename( entry.path ) for job in requeued for entry in job.files )
        self.assertEqual( files, [ 'b.mp4', 'c.mp4' ], 'a.mp4 was done; it is not queued again' )
        self.assertEqual( self.pipeline.started.count( 'a.mp4' ), 1 )
        with_override = [ job for job in requeued if job.overrides ]
        self.assertEqual( [ dict( job.overrides ) for job in with_override ],
                          [ { ( 'shot_cut_burst_window', ): 0.3 } ] )

    def test_refused_before_anything_when_the_change_would_not_pass( self ):
        job = self.hold_running( 'a.mp4' )
        self.refused( 'job.restart_with_changes', { 'mode': 'resume', 'values': { 'shot_cut_threshold': 7 } },
                      'config.invalid' )
        self.refused( 'job.restart_with_changes', { 'mode': 'later', 'values': { 'shot_cut_threshold': 0.5 } },
                      'command.bad_params' )
        self.refused( 'job.restart_with_changes', { 'mode': 'resume' }, 'command.bad_params' )
        self.let_run()
        self.wait_state( job, 'completed' )
        self.assertNotIn( 'state:paused', self.story( job ) )

    def test_with_no_job_it_is_written_at_once( self ):
        self.assertEqual( self.ok( 'job.restart_with_changes', { 'mode': 'resume', 'values': { 'shot_cut_threshold': 0.5 } } ),
                          { 'mode': 'resume', 'jobs': [] } )
        self.assertIn( 'shot_cut_threshold = 0.5', self.local_text() )


@schemas.needs_jsonschema
class TestSchemas( _ChangesTest ):

    def test_messages( self ):
        job = self.hold_running( 'a.mp4' )
        self.ok( 'config.set', { 'values': { 'shot_cut_threshold': 0.5 }, 'when': 'after_jobs' } )
        self.server.pending.add( { 'shot_cut_threshold': 9 }, [] )
        self.let_run()
        self.wait_state( job, 'completed' )
        self.wire.wait_for( lambda m: m.get( 'name' ) == 'config.pending_failed', 'config.pending_failed' )
        for message in self.wire.messages:
            schemas.validate_message( message )
            if message['kind'] == 'event' and message['name'].startswith( 'config.' ):
                schemas.validate_event_data( message['name'], message['data'] )
            if message['kind'] == 'result' and message.get( 'ok' ) and message['name'].startswith( 'config.' ):
                schemas.validate_command_result( message['name'], message['data'] )
        schemas.validate_command_result( 'config.pending', { 'pending': self.server.pending.items() } )
        schemas.validate_command_result( 'job.restart_with_changes', { 'mode': 'requeue', 'jobs': [ 'j' ] } )
        schemas.validate_command_data( 'job.restart_with_changes', { 'mode': 'resume', 'resets': [ 'a' ] } )
