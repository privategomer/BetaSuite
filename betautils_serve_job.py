"""
betautils_serve_job.py - the engine's job commands and job events.

    job.submit    queue chosen files with per-job overrides
    job.pause     stop the running job at its next safe point and hold
                  the queue; its checkpoints stay
    job.resume    run the paused job again from its checkpoints
    job.cancel    drop a queued job, or stop a running or paused one and
                  remove its partial outputs

Jobs run one at a time, in order, on one worker thread, through the same
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
    """

    def __init__( self, send, logger, acquire_lock=None, pipeline=None, load_session=None ):
        import betautils_job as bu_job
        self._bu_job = bu_job
        self.logger = logger
        self.events = Events( send, logger=logger )
        self._acquire_lock = acquire_lock or self._library_lock
        self._load_session_override = load_session
        self.runner = bu_job.Runner( logger, emit=self._emit, pipeline=pipeline,
                                     load_session=self._load_session )
        self._cond = threading.Condition()
        self.jobs = collections.OrderedDict()     # id -> Job, every job not yet forgotten
        self._waiting = collections.deque()       # ids queued, in order
        self.active = None                        # the job running, or paused and holding the queue
        self._lock = None                         # the run lock while busy
        self.snapshot = None                      # config.get's answer while busy
        self._stopping = False
        self._thread = threading.Thread( target=self._work, name='job-worker', daemon=True )
        self._thread.start()

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

    # --- commands ---

    def submit( self, server, data ):
        """data: { files: [ absolute path ], overrides?: { dotted path: value } }"""
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
            if not self._busy_locked():
                # Idle: nothing reads the live config, so check it now.
                start_checks( overrides )
                import betautils_runlock as bu_runlock
                try:
                    self._lock = self._acquire_lock()
                except bu_runlock.RunLocked as locked:
                    raise bu_proto.ProtocolError( locked.error()['code'], locked.error()['params'] )
                try:
                    self.snapshot = bu_serve_config.config_get( server, {} )['settings']
                except BaseException:
                    self._lock.release()
                    self._lock = None
                    raise
            job = bu_job.Job( uuid.uuid4().hex, [ bu_job.JobFile( path, bu_job.output_folder_for( path ) )
                                                  for path in files ], overrides )
            self.jobs[job.id] = job
            position = len( self._waiting ) + ( 1 if self.active is not None else 0 )
            self._waiting.append( job.id )
            self._forget_finished_locked()
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

    def resume( self, _server, data ):
        bu_job = self._bu_job
        with self._cond:
            job = self._job( data )
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
            if job.state == bu_job.QUEUED and job.id in self._waiting:
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

    # --- the worker ---

    def _next_locked( self ):
        """The job to run now, or None to wait."""
        if self._stopping:
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
        jobs = JobQueue( server.send_event, server._logger )
        server.jobs = jobs
    return jobs


def job_submit( server, data ):
    """
    data: { files: [ absolute path ], overrides?: { dotted path: value } }
    result: { job, position }: position counts the jobs ahead of this one.
    """
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


def _refused_while_busy( handler ):
    def refuse( server, data ):
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
    'config.get': config_get,
    'config.set': _refused_while_busy( bu_serve_config.config_set ),
    'config.reset': _refused_while_busy( bu_serve_config.config_reset ),
}

