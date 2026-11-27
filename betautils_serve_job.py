"""
betautils_serve_job.py - the engine's job commands and job events.

    job.submit    queue chosen files with per-job overrides
    job.pause     stop the running job at its next safe point and hold
                  the queue; its checkpoints stay
    job.resume    run the paused job again from its checkpoints
    job.cancel    drop a queued job, or stop a running or paused one and
                  remove its partial outputs
    job.list      the jobs the queue remembers, in the order they run
    job.move      move a waiting job to another place in the queue

Jobs run one at a time, in queue order, on one worker thread, through the same
betautils_job.Runner the command line uses. The main thread keeps reading
commands, so pause and cancel are answered at once; they take effect at
the pipeline's next safe point, and the job.state event says when.

THE RUN LOCK
    The library's run lock (betautils_runlock) is taken as 'engine' when
    the queue goes from idle to busy, and given back when it is idle
    again, so the command line and the engine refuse each other. The
    last job's final job.state goes out after the lock is given back.

SETTINGS WHILE BUSY
    A job reads the live betaconfig module, and checking a settings
    change means changing that module for a moment (ConfigStore.evaluate).
    So while any job is queued, running or paused: config.set and
    config.reset answer config.busy and write nothing; config.get answers
    from the values taken when the queue became busy, and config.busy
    when it is given overrides. A job submitted while another one runs
    is checked when it starts.

EVENTS
    job.progress is coalesced: the latest count per stage, at most one
    batch per job every PROGRESS_INTERVAL seconds. Every other event is
    sent at once, after any progress still waiting, so order holds and
    nothing but progress is ever dropped.

The engine modules are imported on first use, as in betautils_serve_config.
"""

import collections
import os
import threading
import time
import uuid

import betautils_protocol as bu_proto
import betautils_serve_config as bu_serve_config

# How long a settings change waits for the running job to pause (or the
# jobs to stop) before it gives up and writes nothing.
WAIT_FOR_PAUSE_SECONDS = 600

EVENTS = ( 'job.state', 'job.file', 'job.progress', 'job.decision', 'job.error' )

PROGRESS_INTERVAL = 0.25

# Jobs that have ended are remembered for job.pause, job.resume and
# job.cancel to answer job.bad_state rather than job.unknown; the oldest
# are forgotten past this many.
REMEMBERED_FINISHED = 200


def _bad_params( field, reason ):
    return bu_proto.ProtocolError( 'command.bad_params', { 'field': field, 'reason': reason } )


# --- Events ---------------------------------------------------------------------

class Events:
    """
    The job runner's emit(): coalesces progress, sends everything else at
    once. send( name, data ) writes one event; it is called with this
    object's lock held, so events from different threads never reorder.
    An event that cannot be sent (one that does not encode) is logged and
    dropped, never raised into the pipeline, where it would fail the file.
    """

    def __init__( self, send, interval=PROGRESS_INTERVAL, clock=time.monotonic, logger=None ):
        self._raw_send = send
        self._logger = logger
        self._interval = interval
        self._clock = clock
        self._lock = threading.Lock()
        self._wake = threading.Condition( self._lock )
        self._pending = collections.OrderedDict()   # ( job, file, stage ) -> data
        self._last_sent = {}                         # job -> clock of its last progress batch
        self._closed = False
        self._thread = threading.Thread( target=self._flusher, name='job-events', daemon=True )
        self._thread.start()

    def __call__( self, name, data ):
        with self._lock:
            if name == 'job.progress':
                self._pending[( data['job'], data['file'], data['stage'] )] = data
                now = self._clock()
                if now - self._last_sent.get( data['job'], float( '-inf' ) ) >= self._interval:
                    self._flush_job_locked( data['job'], now )
                else:
                    self._wake.notify()
                return
            self._flush_all_locked()
            self._check( name, data )
            self._send( name, data )

    def _send( self, name, data ):
        try:
            self._raw_send( name, data )
        except ( ValueError, TypeError ):
            if self._logger is not None:
                self._logger.error( "event %s could not be sent"%( name, ), exc_info=True )

    @staticmethod
    def _check( name, data ):
        """An error or a stop code must be a registered one with its params (bu_proto.ERRORS)."""
        if name == 'job.error' or ( name == 'job.state' and 'code' in data ):
            bu_proto.error( data['code'], **data['params'] )

    def _flush_job_locked( self, job_id, now ):
        for key in [ key for key in self._pending if key[0] == job_id ]:
            self._send( 'job.progress', self._pending.pop( key ) )
        self._last_sent[job_id] = now

    def _flush_all_locked( self ):
        now = self._clock()
        for job_id in list( dict.fromkeys( key[0] for key in self._pending ) ):
            self._flush_job_locked( job_id, now )

    def _flusher( self ):
        """Sends progress that would otherwise wait for the next event."""
        with self._lock:
            while not self._closed:
                if not self._pending:
                    self._wake.wait()
                    continue
                now = self._clock()
                due = min( self._last_sent.get( key[0], float( '-inf' ) ) + self._interval
                           for key in self._pending )
                if due > now:
                    self._wake.wait( due - now )
                    continue
                for job_id in list( dict.fromkeys( key[0] for key in self._pending ) ):
                    if now - self._last_sent.get( job_id, float( '-inf' ) ) >= self._interval:
                        self._flush_job_locked( job_id, now )

    def flush( self ):
        with self._lock:
            self._flush_all_locked()

    def close( self ):
        with self._lock:
            self._flush_all_locked()
            self._closed = True
            self._wake.notify()
        self._thread.join()


# --- The queue ------------------------------------------------------------------

class JobQueue:
    """
    The engine's jobs: one runs at a time, a paused one holds the queue.

    Args:
        send: send( name, data ) for one event.
        logger: The shared logger.
        acquire_lock: acquire_lock() -> a held lock with release(), or
            raises bu_runlock.RunLocked. Default the library's run lock.
        pipeline: The per-file pipeline module for the runner (tests).
        load_session: Override for loading the detector session (tests).
        recorder: The betautils_run_record.RunRecorder for each file's
            record; None for the stats and failure logs only.
        types: The betautils_catalog_types.RunTypes for pins and folder
            rules; None for the engine's ladder alone.
        store_path: Where the queue is kept across restarts
            (betautils_queue_store); None keeps it in memory only.

    RESTORED JOBS. Jobs a previous engine left are read back held: paused,
    marked restored, after the waiting ones. A held job runs nothing and
    holds nothing (no run lock, settings stay writable) until job.resume
    puts it back in the queue; job.cancel drops it.
    """

    def __init__( self, send, logger, acquire_lock=None, pipeline=None, load_session=None, recorder=None,
                  types=None, governor=None, store_path=None ):
        import betautils_job as bu_job
        self._bu_job = bu_job
        self.logger = logger
        self.events = Events( send, logger=logger )
        self._acquire_lock = acquire_lock or self._library_lock
        self._load_session_override = load_session
        self.runner = bu_job.Runner( logger, emit=self._emit, pipeline=pipeline,
                                     load_session=self._load_session, recorder=recorder, types=types,
                                     governor=governor )
        self._when_idle = collections.OrderedDict()   # key -> callback, run once the queue is idle
        self._cond = threading.Condition()
        self.jobs = collections.OrderedDict()     # id -> Job, every job not yet forgotten
        self._waiting = collections.deque()       # ids queued, in order
        self.active = None                        # the job running, or paused and holding the queue
        self._lock = None                         # the run lock while busy
        self.snapshot = None                      # config.get's answer while busy
        self.estimate_context = None              # job.estimate's settings while busy
        self._probing = False                     # an estimate probe has the model and the GPU
        self._held = collections.deque()          # ids restored from a previous engine, not yet resumed
        self._store_path = store_path
        self._store_warned = False
        self._stopping = False
        if store_path is not None:
            self._restore()
        self._thread = threading.Thread( target=self._work, name='job-worker', daemon=True )
        self._thread.start()

    # --- kept across restarts ---

    def _restore( self ):
        import betautils_queue_store as bu_queue_store
        bu_job = self._bu_job
        for row in bu_queue_store.load( self._store_path, self.logger ):
            try:
                overrides = bu_serve_config._overrides( { 'overrides': row.get( 'overrides', {} ) } )
            except bu_proto.ProtocolError as refused:
                self.logger.warning( "a saved job is dropped: its settings are no longer valid (%s)"%(
                    refused.code, ) )
                continue
            job = bu_job.Job( row['job'], [ bu_job.JobFile( path, bu_job.output_folder_for( path ) )
                                            for path in row['files'] ], overrides )
            job.next_index = row.get( 'next_index', 0 )
            for outcome in bu_job.OUTCOMES:
                count = row.get( 'counts', {} ).get( outcome, 0 )
                job.counts[outcome] = count if isinstance( count, int ) and count >= 0 else 0
            job.state = bu_job.PAUSED
            job.restored = True
            job.announced = True
            self.jobs[job.id] = job
            self._held.append( job.id )
        if self._held:
            self.logger.info( "%d job(s) restored from the last session, paused"%( len( self._held ), ) )

    def _kept_locked( self ):
        """The jobs to keep, in queue order: not ended, or stopped only by a shutdown."""
        bu_job = self._bu_job
        order = ( ( [ self.active.id ] if self.active is not None else [] ) + list( self._waiting )
                  + list( self._held ) )
        seen = set( order )
        order += [ job_id for job_id, job in self.jobs.items()
                   if job_id not in seen and job.state == bu_job.STOPPED and job.code == 'engine.stopping' ]
        return [ self.jobs[job_id] for job_id in order
                 if self.jobs[job_id].state not in bu_job.FINAL or self.jobs[job_id].code == 'engine.stopping' ]

    def _save_locked( self ):
        if self._store_path is None:
            return
        import betautils_queue_store as bu_queue_store
        try:
            bu_queue_store.save( [ bu_queue_store.row_for( job ) for job in self._kept_locked() ], self._store_path )
        except Exception as err:   # noqa: BLE001  keeping the queue never stops a job
            if not self._store_warned:
                self._store_warned = True
                self.logger.warning( "the job queue could not be saved (%s); it will not survive a restart"%( err, ) )

    # --- helpers ---

    def _library_lock( self ):
        import betautils_runlock as bu_runlock
        return bu_runlock.acquire_library( 'engine', logger=self.logger )

    def _load_session( self ):
        bu_job = self._bu_job
        try:
            if self._load_session_override is not None:
                return self._load_session_override()
            return self.runner.pipeline.bu_detector.get_detector().get_session()
        except bu_job.JobStopped:
            raise
        except Exception as err:
            self.logger.error( "the detection model did not load", exc_info=True )
            raise bu_job.JobStopped( 'model.load_failed', { 'exception_type': type( err ).__name__ } )

    @property
    def busy( self ):
        with self._cond:
            return self._busy_locked()

    def _busy_locked( self ):
        return self.active is not None or bool( self._waiting )

    def busy_job( self ):
        """The id config.busy names: the active job, else the first queued; None when idle."""
        with self._cond:
            return self.busy_job_locked()

    def busy_job_locked( self ):
        if self.active is not None:
            return self.active.id
        return self._waiting[0] if self._waiting else None

    def _job( self, data ):
        job_id = data.get( 'job' )
        if not isinstance( job_id, str ) or not job_id:
            raise _bad_params( 'job', bu_proto.REASON_MISSING if job_id is None else bu_proto.REASON_TYPE )
        job = self.jobs.get( job_id )
        if job is None:
            raise bu_proto.ProtocolError( 'job.unknown', { 'job': job_id } )
        return job

    def _bad_state( self, job ):
        return bu_proto.ProtocolError( 'job.bad_state', { 'job': job.id, 'state': job.state } )

    def forget_ended( self ):
        """Clear history: forget every job that has ended (none run while it is allowed)."""
        with self._cond:
            for job_id in [ job_id for job_id, job in self.jobs.items() if job.state in self._bu_job.FINAL ]:
                del self.jobs[job_id]
            self._save_locked()

    def _forget_finished_locked( self ):
        finished = [ job_id for job_id, job in self.jobs.items() if job.state in self._bu_job.FINAL ]
        for job_id in finished[:max( 0, len( finished ) - REMEMBERED_FINISHED )]:
            del self.jobs[job_id]

    def _emit( self, name, data ):
        """
        The runner's events. A job's final state leaves the queue first, so
        a client that reads it finds the run lock free and settings writable
        when nothing else is queued (the job's overrides are restored by then).
        """
        if name == 'job.state' and data['state'] in self._bu_job.FINAL:
            with self._cond:
                if self.active is not None and self.active.id == data['job']:
                    self.active = None
                self._idle_locked()
        if name == 'job.state' or ( name == 'job.file' and data.get( 'state' ) != 'started' ):
            with self._cond:
                job = self.jobs.get( data.get( 'job' ) )
                if name == 'job.file' and job is not None and isinstance( data.get( 'file' ), int ):
                    # The runner moves next_index on just after this event.
                    job.next_index = max( job.next_index, data['file'] + 1 )
                self._save_locked()
        self.events( name, data )

    def _idle_locked( self ):
        """The queue has nothing left: give the lock back and forget the snapshot."""
        if self._busy_locked():
            return
        if self._lock is not None:
            self._lock.release()
            self._lock = None
            self.logger.info( "job queue idle, run lock released" )
        self.snapshot = None
        self.estimate_context = None
        waiting, self._when_idle = self._when_idle, collections.OrderedDict()
        for key, callback in waiting.items():
            try:
                callback()
            except Exception:
                self.logger.error( "after the jobs: %s failed"%( key, ), exc_info=True )

    def when_idle( self, key, callback ):
        """
        Run callback() once the queue is idle; now when it already is. A
        later callback for the same key replaces an earlier one.
        """
        with self._cond:
            if self._busy_locked():
                self._when_idle.pop( key, None )
                self._when_idle[key] = callback
                return False
        callback()
        return True


    @property
    def probing( self ):
        with self._cond:
            return self._probing

    def start_probe( self, work, done ):
        """
        Run work( session ) on a thread of its own, with the detector's
        session, when the queue is idle and no probe runs. A job
        submitted meanwhile waits for it. done( ok ) is called once it
        has ended and the queue may run again, whether or not work
        succeeded. Returns whether it started.
        """
        with self._cond:
            if self._stopping or self._probing or self._busy_locked():
                return False
            self._probing = True

        def run():
            ok = False
            try:
                work( self._load_session() )
                ok = True
            except BaseException:   # noqa: BLE001  a probe never takes the engine down
                self.logger.warning( "the estimate probe failed", exc_info=True )
            finally:
                with self._cond:
                    self._probing = False
                    self._cond.notify_all()
                done( ok )

        threading.Thread( target=run, name='estimate-probe', daemon=True ).start()
        return True

    def estimate_context_if_busy( self ):
        """
        job.estimate's settings while the queue is busy (taken when it
        became busy), or None when idle and the live settings can be
        evaluated.
        """
        with self._cond:
            if not self._busy_locked():
                return None
            if self.estimate_context is None:
                raise bu_proto.ProtocolError( 'config.busy', { 'job': self.busy_job_locked() } )
            return self.estimate_context

    def _become_busy_locked( self, server, overrides ):
        """
        When the queue is idle, a job about to join it is checked against
        the live config (nothing reads it now), the run lock is taken and
        the settings are captured for config.get and job.estimate.
        """
        if self._busy_locked():
            return
        start_checks( overrides )
        import betautils_runlock as bu_runlock
        try:
            self._lock = self._acquire_lock()
        except bu_runlock.RunLocked as locked:
            raise bu_proto.ProtocolError( locked.error()['code'], locked.error()['params'] )
        try:
            self.snapshot = bu_serve_config.config_get( server, {} )['settings']
            import betautils_serve_estimate as bu_serve_estimate
            self.estimate_context = bu_serve_estimate.context()
        except BaseException:
            self._lock.release()
            self._lock = None
            raise

    # --- commands ---

    def submit( self, server, data, announce_now=False ):
        """
        data: { files: [ absolute path ], overrides?: { dotted path: value } }
        announce_now: send the job's first event at once (a job queued by
        the engine itself, with no command reply to send first).
        """
        bu_job = self._bu_job
        files = data.get( 'files' )
        if not isinstance( files, list ):
            raise _bad_params( 'files', bu_proto.REASON_MISSING if files is None else bu_proto.REASON_TYPE )
        if not files:
            raise _bad_params( 'files', bu_proto.REASON_VALUE )
        for index, path in enumerate( files ):
            if not isinstance( path, str ) or not path:
                raise _bad_params( 'files', bu_proto.REASON_TYPE )
            if not os.path.isabs( path ):
                raise _bad_params( 'files', bu_proto.REASON_VALUE )
            if not os.path.isfile( path ):
                raise bu_proto.ProtocolError( 'job.no_such_file', { 'index': index } )
        overrides = bu_serve_config._overrides( data )

        with self._cond:
            if self._stopping:
                raise bu_proto.ProtocolError( 'engine.stopping', {} )
            self._become_busy_locked( server, overrides )
            job = bu_job.Job( uuid.uuid4().hex, [ bu_job.JobFile( path, bu_job.output_folder_for( path ) )
                                                  for path in files ], overrides )
            self.jobs[job.id] = job
            position = len( self._waiting ) + ( 1 if self.active is not None else 0 )
            self._waiting.append( job.id )
            self._forget_finished_locked()
            # Saved by its queued job.state, announced right after this reply.
        self.logger.info( "job %s submitted: %d file(s), %d override(s), position %d"%(
            job.id, len( job.files ), len( overrides ), position ) )

        # The result goes out before the job's first event; the worker
        # starts it only then.
        def announce():
            self.events( 'job.state', { 'job': job.id, 'state': job.state, 'counts': job.counts_data() } )
            with self._cond:
                job.announced = True
                self._cond.notify_all()
        job.announced = False
        if announce_now:
            announce()
        else:
            server.after_reply( announce )
        return { 'job': job.id, 'position': position }

    def pause( self, _server, data ):
        bu_job = self._bu_job
        with self._cond:
            job = self._job( data )
            if job is not self.active or job.state != bu_job.RUNNING:
                raise self._bad_state( job )
            import betautils_signals as bu_signals
            bu_signals.request_stop( bu_signals.PAUSE )
        self.logger.info( "job %s: pause requested"%( job.id, ) )
        return { 'job': job.id, 'state': job.state }

    def resume( self, server, data ):
        bu_job = self._bu_job
        with self._cond:
            job = self._job( data )
            if job.id in self._held:
                # A restored job joins the queue, checked and locked as a new one is.
                if self._stopping:
                    raise bu_proto.ProtocolError( 'engine.stopping', {} )
                self._become_busy_locked( server, job.overrides )
                self._held.remove( job.id )
                job.restored = False
                self._waiting.append( job.id )
                self._save_locked()
                self._cond.notify_all()
                self.logger.info( "job %s: restored job resumed, from file %d"%( job.id, job.next_index + 1 ) )
                return { 'job': job.id, 'state': job.state }
            if job is not self.active or job.state != bu_job.PAUSED:
                raise self._bad_state( job )
            job.resume_requested = True
            self._cond.notify_all()
        self.logger.info( "job %s: resume requested"%( job.id, ) )
        return { 'job': job.id, 'state': job.state }

    def cancel( self, _server, data ):
        bu_job = self._bu_job
        with self._cond:
            job = self._job( data )
            if job.state in bu_job.FINAL:
                raise self._bad_state( job )
            if job.id in self._held:
                self._held.remove( job.id )
                self.runner.cancel_idle( job )
            elif job.id in self._waiting:
                # Queued, or a restored job resumed and waiting its turn.
                self._waiting.remove( job.id )
                self.runner.cancel_idle( job )
                self._idle_locked()
            elif job is self.active and job.state == bu_job.PAUSED:
                self.runner.cancel_idle( job )
                self.active = None
                self._idle_locked()
                self._cond.notify_all()
            elif job is self.active:
                import betautils_signals as bu_signals
                bu_signals.request_stop( bu_signals.CANCEL )
            else:
                raise self._bad_state( job )
        self.logger.info( "job %s: cancel requested"%( job.id, ) )
        return { 'job': job.id, 'state': job.state }

    # --- settings changed while jobs run (betautils_serve_changes) ---

    def _refresh_snapshot_locked( self, server ):
        """After a change written mid-queue: config.get and job.estimate answer from the new settings."""
        if self._busy_locked():
            self.snapshot = bu_serve_config.config_get( server, {} )['settings']
            import betautils_serve_estimate as bu_serve_estimate
            self.estimate_context = bu_serve_estimate.context()

    def restart_paused( self, server, write, wait=None ):
        """
        Pause the running job at its next safe point, call write() while
        nothing runs, and resume it. With no job running (only queued
        ones), write() at once. Returns the ids of the jobs it touched.
        In the background: answers before the job has paused.
        """
        bu_job = self._bu_job
        import betautils_signals as bu_signals
        with self._cond:
            active = self.active
            touched = [ active.id ] if active is not None else []

        def run():
            deadline = time.monotonic() + ( wait or WAIT_FOR_PAUSE_SECONDS )
            paused_here = set()
            with self._cond:
                while True:
                    job = self.active
                    if job is None or job.state in ( bu_job.PAUSED, bu_job.QUEUED ):
                        # Nothing runs, and the worker waits on this lock.
                        written = write()
                        if written:
                            self._refresh_snapshot_locked( server )
                        if job is not None and job.state == bu_job.PAUSED and job.id in paused_here:
                            job.resume_requested = True
                            self._cond.notify_all()
                        return
                    if job.state == bu_job.RUNNING and job.id not in paused_here:
                        bu_signals.request_stop( bu_signals.PAUSE )
                        paused_here.add( job.id )
                    if time.monotonic() > deadline:
                        self.logger.error( "settings change: the job did not pause in time; nothing written" )
                        return
                    self._cond.wait( 0.5 )

        threading.Thread( target=run, name='betarestart', daemon=True ).start()
        return touched

    def requeue_with( self, server, write, wait=None ):
        """
        Cancel every job, call write() once nothing runs, and queue each
        job's remaining files again with its own overrides. Returns the
        ids of the jobs cancelled. In the background.
        """
        bu_job = self._bu_job
        plans = []
        with self._cond:
            for job in list( self.jobs.values() ):
                # A job restored from the last session and not resumed is
                # left held: it reads the new settings whenever it is resumed.
                if job.state in bu_job.FINAL or job.id in self._held:
                    continue
                start = job.next_index
                files = [ entry.path for entry in job.files[start:] ]
                overrides = { '.'.join( path ): value for path, value in job.overrides.items() }
                plans.append( ( job.id, files, overrides ) )
            for job_id, _files, _overrides in plans:
                try:
                    self.cancel( server, { 'job': job_id } )
                except bu_proto.ProtocolError:
                    pass    # it ended on its own meanwhile

        def run():
            deadline = time.monotonic() + ( wait or WAIT_FOR_PAUSE_SECONDS )
            with self._cond:
                while self._busy_locked():
                    if time.monotonic() > deadline:
                        self.logger.error( "settings change: the jobs did not stop in time; nothing written" )
                        return
                    self._cond.wait( 0.5 )
                written = write()
            if not written:
                return
            for _job_id, files, overrides in plans:
                if not files:
                    continue
                try:
                    self.submit( server, { 'files': files, 'overrides': overrides }, announce_now=True )
                except bu_proto.ProtocolError as refused:
                    self.logger.warning( "settings change: a job could not be queued again: %s"%( refused.error, ) )

        threading.Thread( target=run, name='betarequeue', daemon=True ).start()
        return [ job_id for job_id, _files, _overrides in plans ]

    def _positions_locked( self ):
        """{ job id: position } for the active job (0), every waiting one, then the held ones, in order."""
        order = ( [ self.active.id ] if self.active is not None else [] ) + list( self._waiting ) + list( self._held )
        return { job_id: index for index, job_id in enumerate( order ) }

    def list( self, _server, _data ):
        """
        Every job the queue remembers: the active one first, then the
        waiting ones in the order they will run, then the ended ones,
        newest first. position is None for an ended job.
        """
        with self._cond:
            positions = self._positions_locked()
            ended = [ job for job in self.jobs.values() if job.id not in positions ]
            ordered = [ self.jobs[job_id] for job_id in sorted( positions, key=positions.get ) ]
            rows = [ self._row( job, positions.get( job.id ) ) for job in ordered + ended[::-1] ]
        return { 'jobs': rows }

    @staticmethod
    def _row( job, position ):
        row = { 'job': job.id, 'state': job.state, 'position': position,
                'files': [ entry.path for entry in job.files ], 'counts': job.counts_data(),
                'overrides': { '.'.join( path ): value for path, value in job.overrides.items() } }
        if getattr( job, 'restored', False ):
            row['restored'] = True
        if job.code is not None:
            row['code'] = job.code
            row['params'] = job.params
        return row

    def move( self, _server, data ):
        """
        data: { job, position }. Only a waiting job moves; it cannot go
        ahead of the active one. A position past either end is taken as
        that end. The result's position is where it now is.
        """
        position = data.get( 'position' )
        if isinstance( position, bool ) or not isinstance( position, int ):
            raise _bad_params( 'position', bu_proto.REASON_MISSING if position is None else bu_proto.REASON_TYPE )
        if position < 0:
            raise _bad_params( 'position', bu_proto.REASON_VALUE )
        with self._cond:
            job = self._job( data )
            if job.id not in self._waiting:
                raise self._bad_state( job )
            first = 1 if self.active is not None else 0
            index = min( max( position - first, 0 ), len( self._waiting ) - 1 )
            self._waiting.remove( job.id )
            self._waiting.insert( index, job.id )
            self._save_locked()
            self._cond.notify_all()
            now = index + first
        self.logger.info( "job %s moved to position %d"%( job.id, now ) )
        return { 'job': job.id, 'position': now }

    # --- the worker ---

    def _next_locked( self ):
        """The job to run now, or None to wait."""
        if self._stopping or self._probing:
            return None
        if self.active is not None:
            job = self.active
            if job.state == self._bu_job.PAUSED and getattr( job, 'resume_requested', False ):
                job.resume_requested = False
                return job
            return None
        while self._waiting:
            job = self.jobs[self._waiting[0]]
            if not getattr( job, 'announced', True ):
                return None
            self._waiting.popleft()
            self.active = job
            return job
        return None

    def _work( self ):
        bu_job = self._bu_job
        while True:
            with self._cond:
                job = self._next_locked()
                while job is None and not self._stopping:
                    self._cond.wait()
                    job = self._next_locked()
                if job is None:
                    return
            self._run( job )
            with self._cond:
                import betautils_signals as bu_signals
                # A stop asked for as the job ended found no safe point; it
                # must not reach the next job. A cancel that came while the
                # job was pausing still cancels it.
                leftover = bu_signals.take()
                if leftover == bu_signals.CANCEL and job.state == bu_job.PAUSED:
                    self.runner.cancel_idle( job )
                if job.state != bu_job.PAUSED:
                    self.active = None
                if job.state == bu_job.STOPPED and job.code == 'disk.full':
                    self._stop_waiting_locked( 'disk.full', job.params )
                self._idle_locked()

    def _run( self, job ):
        bu_job = self._bu_job
        try:
            if job.state == bu_job.QUEUED:
                try:
                    start_checks( job.overrides )
                except bu_proto.ProtocolError as refused:
                    self.runner.set_state( job, bu_job.RUNNING )
                    self.runner.stop( job, refused.code, refused.error['params'] )
                    return
            self.runner.run( job )
        except BaseException as err:   # noqa: BLE001  the worker must outlive any job
            self.logger.error( "job %s failed inside the engine"%( job.id, ), exc_info=True )
            if job.state not in bu_job.FINAL:
                if job.state == bu_job.QUEUED:
                    self.runner.set_state( job, bu_job.RUNNING )
                self.runner.stop( job, 'engine.internal', { 'exception_type': type( err ).__name__ } )

    def _stop_waiting_locked( self, code, params ):
        while self._waiting:
            waiting = self.jobs[self._waiting.popleft()]
            self.runner.stop( waiting, code, params )

    def stop( self ):
        """
        End the queue for shutdown: the running job stops at its next safe
        point (checkpoints kept, 'stopped' with engine.stopping), a paused
        or queued one is stopped as it is, and the worker is joined.
        """
        bu_job = self._bu_job
        import betautils_signals as bu_signals
        with self._cond:
            if self._stopping:
                return
            self._stopping = True
            # Queued too: the worker may have taken the job and be checking
            # it, about to run it.
            if self.active is not None and self.active.state in ( bu_job.QUEUED, bu_job.RUNNING ):
                bu_signals.request_stop( bu_signals.SHUTDOWN )
            self._cond.notify_all()
        self._thread.join()
        with self._cond:
            if self.active is not None and self.active.state not in bu_job.FINAL:
                self.runner.set_state( self.active, bu_job.STOPPED, 'engine.stopping' )
            self.active = None
            while self._waiting:
                self.runner.set_state( self.jobs[self._waiting.popleft()], bu_job.STOPPED, 'engine.stopping' )
            if bu_signals.stop_reason() == bu_signals.SHUTDOWN:
                bu_signals.clear()
            self._idle_locked()
            # What the next engine restores: the jobs this shutdown stopped, and the held ones.
            self._save_locked()
        self.events.close()


# --- Checks a job must pass to start --------------------------------------------

def start_checks( overrides ):
    """
    Raise the ProtocolError that keeps a job with these overrides from
    starting, checked as betatv.py checks a run: config.invalid,
    job.unsafe_setting (input_delete_probability above 0: the engine
    cannot ask first, as the command line does), job.nothing_to_render.
    Changes the live config for a moment, so it runs only while no job
    reads it.
    """
    import betaconfig as _betaconfig   # noqa: F401  (the store's module)
    import betautils_config as bu_config

    def extra( module ):
        return { 'delete': getattr( module, 'input_delete_probability', 0 ),
                 'nothing': bu_config.nothing_to_render_reason() is not None }

    store = bu_serve_config.store()
    evaluation = store.evaluate( store.text(), overrides, extra=extra )
    if not evaluation.ok:
        raise bu_serve_config._invalid( evaluation.errors )
    if evaluation.extra['delete']:
        raise bu_proto.ProtocolError( 'job.unsafe_setting', { 'path': 'input_delete_probability' } )
    if evaluation.extra['nothing']:
        raise bu_proto.ProtocolError( 'job.nothing_to_render', {} )


# --- Commands -------------------------------------------------------------------

def queue_for( server ):
    """The server's job queue, made on first use."""
    jobs = getattr( server, 'jobs', None )
    if jobs is None:
        # The catalog hears the job events (queued, processing) and keeps
        # every file's record in its history.
        import betautils_queue_store as bu_queue_store
        import betautils_serve_catalog as bu_serve_catalog
        import betautils_serve_resources as bu_serve_resources
        jobs = JobQueue( bu_serve_catalog.job_events_for( server ), server._logger,
                         recorder=bu_serve_catalog.job_recorder_for( server ),
                         types=bu_serve_catalog.job_types_for( server ),
                         governor=bu_serve_resources.runtime_for( server ),
                         store_path=getattr( server, 'queue_store_path', None ) or bu_queue_store.default_path() )
        server.jobs = jobs
    return jobs


def job_submit( server, data ):
    """
    data: { files: [ absolute path ], overrides?: { dotted path: value } }
    result: { job, position }: position counts the jobs ahead of this one.
    Refused with perf.optimize_running while Optimize measures the machine.
    """
    optimizer = getattr( server, 'optimizer', None )
    if optimizer is not None and optimizer.running:
        raise bu_proto.ProtocolError( 'perf.optimize_running', {} )
    return queue_for( server ).submit( server, data )


def job_pause( server, data ):
    """data: { job }; result: { job, state }. The job.state event says when it has paused."""
    return queue_for( server ).pause( server, data )


def job_resume( server, data ):
    """data: { job }; result: { job, state }."""
    return queue_for( server ).resume( server, data )


def job_cancel( server, data ):
    """data: { job }; result: { job, state }. A queued or paused job is cancelled at once."""
    return queue_for( server ).cancel( server, data )


def job_list( server, data ):
    """data: {}; result: { jobs }: the active job, the waiting ones in order, then the ended ones."""
    return queue_for( server ).list( server, data )


def job_move( server, data ):
    """data: { job, position }; result: { job, position }. Reorders a waiting job."""
    return queue_for( server ).move( server, data )


def _busy( server ):
    jobs = getattr( server, 'jobs', None )
    return jobs.busy_job() if jobs is not None else None


def config_get( server, data ):
    """config.get, answered from the snapshot while a job is queued, running or paused."""
    busy = _busy( server )
    if busy is None:
        return bu_serve_config.config_get( server, data )
    if data.get( 'overrides' ):
        raise bu_proto.ProtocolError( 'config.busy', { 'job': busy } )
    prefixes = data.get( 'paths' )
    if prefixes is not None and ( not isinstance( prefixes, list )
                                  or not all( isinstance( prefix, str ) for prefix in prefixes ) ):
        raise _bad_params( 'paths', bu_proto.REASON_TYPE )
    snapshot = server.jobs.snapshot or []
    rows = [ dict( row ) for row in snapshot
             if not prefixes or any( row['path'] == prefix or row['path'].startswith( prefix + '.' )
                                     for prefix in prefixes ) ]
    return { 'settings': rows }


def _set_after_jobs( server, data, handler ):
    """config.set / config.reset with when: after_jobs (betautils_serve_changes)."""
    import betautils_serve_changes as bu_changes
    if handler is bu_serve_config.config_set:
        updates, removals = bu_changes.parse_change( { 'values': data.get( 'values' ) or {} } )
    else:
        paths = data.get( 'paths' )
        if not isinstance( paths, list ) or not paths:
            raise _bad_params( 'paths', bu_proto.REASON_MISSING if paths is None else bu_proto.REASON_TYPE )
        updates, removals = bu_changes.parse_change( { 'resets': paths } )
    return bu_changes.queue_after_jobs( server, data, updates, removals )


def _refused_while_busy( handler ):
    """
    The handler, refused with config.busy while a job holds the queue,
    unless the request says when: after_jobs: then it is checked now and
    written when the queue is idle (betautils_serve_changes).
    """
    def refuse( server, data ):
        when = data.get( 'when', 'now' )
        if when not in ( 'now', 'after_jobs' ):
            raise _bad_params( 'when', bu_proto.REASON_VALUE )
        if when == 'after_jobs' and not data.get( 'dry_run', False ):
            return _set_after_jobs( server, data, handler )
        busy = _busy( server )
        if busy is not None:
            raise bu_proto.ProtocolError( 'config.busy', { 'job': busy } )
        return handler( server, data )
    refuse.__name__ = handler.__name__
    refuse.__doc__ = handler.__doc__
    return refuse


HANDLERS = {
    'job.submit': job_submit,
    'job.pause': job_pause,
    'job.resume': job_resume,
    'job.cancel': job_cancel,
    'job.list': job_list,
    'job.move': job_move,
    'config.get': config_get,
    'config.set': _refused_while_busy( bu_serve_config.config_set ),
    'config.reset': _refused_while_busy( bu_serve_config.config_reset ),
}

