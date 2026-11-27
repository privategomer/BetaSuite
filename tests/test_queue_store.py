"""
test_queue_store.py - the job queue kept across engine restarts.

Covers the file (round trip, a missing, broken or foreign file, bad rows
left out), what the queue writes as it changes (jobs in order, a file
that ended moving next_index on, ended jobs gone, a shutdown keeping
the jobs it stopped), and what the next queue does with it: restored
jobs held and paused, holding no lock, resumed from their next file
(alone or behind a running job), cancelled, refused a move, dropped when
their settings no longer hold; and a queue that cannot be written going
on all the same.
"""

import json
import os
import time

import betautils_queue_store as bu_queue_store
import betautils_serve_job as bu_serve_job

from tests.test_serve_job import _QueueTest, WAIT


class TestFile( _QueueTest ):

    def test_round_trip_and_bad_rows( self ):
        path = os.path.join( self.tmp, 'q', 'queue.json' )
        rows = [ { 'job': 'a', 'files': [ '/x.mp4' ], 'overrides': {}, 'next_index': 1, 'counts': {} } ]
        bu_queue_store.save( rows + [ { 'job': '', 'files': [ '/y' ] }, { 'job': 'b', 'files': [] },
                                      { 'job': 'c', 'files': [ '/z' ], 'next_index': 5 },
                                      { 'job': 'd', 'files': [ '/z' ], 'next_index': True } ], path )
        self.assertEqual( bu_queue_store.load( path ), rows )
        self.assertEqual( [ name for name in os.listdir( os.path.dirname( path ) ) ], [ 'queue.json' ] )

    def test_missing_broken_or_foreign_files_give_an_empty_queue( self ):
        path = os.path.join( self.tmp, 'queue.json' )
        self.assertEqual( bu_queue_store.load( path ), [] )
        for text in ( '{', json.dumps( { 'version': 99, 'jobs': [] } ), json.dumps( [ 1 ] ),
                      json.dumps( { 'version': 1, 'jobs': 'x' } ) ):
            with open( path, 'w' ) as handle:
                handle.write( text )
            self.assertEqual( bu_queue_store.load( path, self.logger ), [] )


class _Kept( _QueueTest ):

    def setUp( self ):
        super().setUp()
        self.path = os.path.join( self.tmp, 'state', 'queue.json' )
        self.server.stop_jobs()
        self.server.jobs = self.new_queue()

    def _stop( self ):
        self.stop_releasing()

    def stop_releasing( self ):
        """
        stop_jobs, letting a step that waits for its release see the stop.
        Permits go out only while the stop is pending, and any left over are
        taken back, so none lets a later job's step through early.
        """
        import threading
        done = threading.Event()

        def free():
            while not done.wait( 0.02 ):
                self.pipeline.release.release()

        helper = threading.Thread( target=free, daemon=True )
        helper.start()
        try:
            self.server.stop_jobs()
        finally:
            done.set()
            helper.join()
        while self.pipeline.release.acquire( blocking=False ):
            pass

    def new_queue( self ):
        return bu_serve_job.JobQueue( self.server.send_event, self.logger, pipeline=self.pipeline,
                                      acquire_lock=self._lock, store_path=self.path )

    def saved( self ):
        with open( self.path ) as handle:
            return json.load( handle )['jobs']

    def wait_saved( self, predicate, what ):
        deadline = time.monotonic() + WAIT
        while time.monotonic() < deadline:
            try:
                if predicate( self.saved() ):
                    return
            except ( OSError, ValueError ):
                pass
            time.sleep( 0.02 )
        self.fail( "the saved queue never showed %s: %s"%( what, self.saved() ) )

    def restart( self, rows=None ):
        """The engine stops and a new one makes its queue from the file (rows, when given, written in between)."""
        self.stop_releasing()
        if rows is not None:
            bu_queue_store.save( rows, self.path )
        self.pipeline.started.clear()
        self.lock_calls.clear()
        self.server.jobs = self.new_queue()


class TestWhatIsWritten( _Kept ):

    def test_jobs_in_order_and_ended_jobs_gone( self ):
        self.pipeline.auto = False
        first = self.submit( 'a.mp4', 'b.mp4', overrides={ 'encode_crf': 30 } )
        second = self.submit( 'c.mp4' )
        self.wait_saved( lambda rows: len( rows ) == 2, "both jobs" )
        saved = self.saved()
        self.assertEqual( [ row['job'] for row in saved ], [ first, second ] )
        self.assertEqual( saved[0]['files'], [ self.files['a.mp4'], self.files['b.mp4'] ] )
        self.assertEqual( saved[0]['overrides'], { 'encode_crf': 30 } )
        self.wait_state( first, 'running' )
        for _ in range( 3 ):
            self.pipeline.release.release()
        self.wait_saved( lambda rows: rows[0]['next_index'] == 1 and rows[0]['counts']['processed'] == 1,
                         "the first file ended" )
        self.pipeline.auto = True
        for _ in range( 10 ):
            self.pipeline.release.release()
        self.wait_state( second, 'completed' )
        self.wait_saved( lambda rows: rows == [], "an empty queue" )

    def test_a_move_and_a_cancel_are_written( self ):
        self.pipeline.auto = False
        running = self.submit( 'a.mp4' )
        b = self.submit( 'b.mp4' )
        c = self.submit( 'c.mp4' )
        self.wait_state( running, 'running' )
        self.ok( 'job.move', { 'job': c, 'position': 1 } )
        self.assertEqual( [ row['job'] for row in self.saved() ], [ running, c, b ] )
        self.ok( 'job.cancel', { 'job': c } )
        self.assertEqual( [ row['job'] for row in self.saved() ], [ running, b ] )

    def test_a_shutdown_keeps_the_jobs_it_stopped( self ):
        self.pipeline.auto = False
        running = self.submit( 'a.mp4', 'b.mp4' )
        queued = self.submit( 'c.mp4' )
        self.wait_state( running, 'running' )
        for _ in range( 3 ):
            self.pipeline.release.release()
        self.wait_saved( lambda rows: rows[0]['next_index'] == 1, "the first file ended" )
        self.pipeline.release.release()
        self.server.stop_jobs()
        saved = self.saved()
        self.assertEqual( [ row['job'] for row in saved ], [ running, queued ] )
        self.assertEqual( saved[0]['next_index'], 1 )

    def test_a_queue_that_cannot_be_written_goes_on( self ):
        blocker = os.path.join( self.tmp, 'blocker' )
        with open( blocker, 'w' ) as handle:
            handle.write( 'x' )
        self.server.stop_jobs()
        self.path = os.path.join( blocker, 'queue.json' )
        self.server.jobs = self.new_queue()
        job = self.submit( 'a.mp4' )
        self.wait_state( job, 'completed' )
        self.assertTrue( self.server.jobs._store_warned )


class TestRestored( _Kept ):

    def leave_two_jobs( self ):
        """A two-file job stopped after its first file, and a queued one, then a restart."""
        self.pipeline.auto = False
        first = self.submit( 'a.mp4', 'b.mp4' )
        second = self.submit( 'c.mp4' )
        self.wait_state( first, 'running' )
        for _ in range( 3 ):
            self.pipeline.release.release()
        self.wait_saved( lambda rows: rows and rows[0]['next_index'] == 1, "the first file ended" )
        self.restart()
        return first, second

    def test_restored_jobs_are_held_paused_and_hold_nothing( self ):
        first, second = self.leave_two_jobs()
        rows = self.ok( 'job.list' )['jobs']
        self.assertEqual( [ ( row['job'], row['state'], row['position'], row.get( 'restored' ) ) for row in rows ],
                          [ ( first, 'paused', 0, True ), ( second, 'paused', 1, True ) ] )
        self.assertEqual( rows[0]['counts'], { 'total': 2, 'processed': 1, 'skipped': 0, 'failed': 0 } )
        self.assertFalse( self.server.jobs.busy )
        self.assertEqual( self.lock_calls, [], "a held job takes no lock" )
        time = __import__( 'time' )
        time.sleep( 0.2 )
        self.assertEqual( self.pipeline.started, [], "nothing runs until resumed" )
        self.ok( 'config.get', { 'paths': [ 'encode_crf' ], 'overrides': { 'encode_crf': 23 } } )  # config.busy while busy

    def test_a_resumed_job_continues_from_its_next_file( self ):
        first, second = self.leave_two_jobs()
        self.pipeline.auto = True
        self.ok( 'job.resume', { 'job': first } )
        self.assertEqual( len( self.lock_calls ), 1, "resuming takes the lock like a new job" )
        final = self.wait_state( first, 'completed' )
        self.assertEqual( self.pipeline.started, [ 'b.mp4' ] )
        self.assertEqual( final['counts'], { 'total': 2, 'processed': 2, 'skipped': 0, 'failed': 0 } )
        self.assertEqual( [ row['job'] for row in self.saved() ], [ second ], "the held one is still kept" )
        self.assertFalse( self.server.jobs.busy )

    def test_a_job_resumed_behind_a_running_one_waits_its_turn_and_can_be_cancelled( self ):
        first, second = self.leave_two_jobs()
        running = self.submit( 'a.mp4' )
        self.wait_state( running, 'running' )
        self.ok( 'job.resume', { 'job': first } )
        self.ok( 'job.resume', { 'job': second } )
        rows = self.ok( 'job.list' )['jobs']
        self.assertEqual( [ ( row['job'], row['position'] ) for row in rows[:3] ],
                          [ ( running, 0 ), ( first, 1 ), ( second, 2 ) ] )
        self.assertEqual( self.ok( 'job.cancel', { 'job': second } )['state'], 'cancelled' )
        self.pipeline.auto = True
        for _ in range( 10 ):
            self.pipeline.release.release()
        self.wait_state( first, 'completed' )
        self.assertEqual( self.pipeline.started, [ 'a.mp4', 'b.mp4' ] )

    def test_a_held_job_is_cancelled_at_once_and_cannot_be_moved_or_paused( self ):
        first, second = self.leave_two_jobs()
        self.refused( 'job.move', { 'job': first, 'position': 1 }, 'job.bad_state' )
        self.refused( 'job.pause', { 'job': first }, 'job.bad_state' )
        self.assertEqual( self.ok( 'job.cancel', { 'job': first } )['state'], 'cancelled' )
        self.assertEqual( [ row['job'] for row in self.saved() ], [ second ] )
        self.assertEqual( self.story( first )[-1], 'state:cancelled' )

    def test_a_job_whose_settings_no_longer_hold_is_dropped( self ):
        self.restart( [ { 'job': 'old', 'files': [ self.files['a.mp4'] ], 'overrides': { 'encode_crff': 1 },
                          'next_index': 0, 'counts': {} },
                        { 'job': 'good', 'files': [ self.files['b.mp4'] ], 'overrides': {},
                          'next_index': 0, 'counts': { 'processed': -3 } } ] )
        rows = self.ok( 'job.list' )['jobs']
        self.assertEqual( [ row['job'] for row in rows ], [ 'good' ] )
        self.assertEqual( rows[0]['counts']['processed'], 0 )

    def test_a_restored_job_checked_at_resume_against_todays_settings( self ):
        self.restart( [ { 'job': 'old', 'files': [ self.files['a.mp4'] ],
                          'overrides': { 'items_to_censor': [] }, 'next_index': 0, 'counts': {} } ] )
        self.refused( 'job.resume', { 'job': 'old' }, 'job.nothing_to_render' )
        self.assertEqual( self.ok( 'job.list' )['jobs'][0]['state'], 'paused', "still held" )
        self.assertEqual( self.lock_calls, [] )
