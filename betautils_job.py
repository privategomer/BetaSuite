"""
betautils_job.py - a job: a list of files run through the pipeline one
after another, with per-job setting overrides, a state, and events.

betatv.py runs everything in the input folder as one job; the engine's
job commands (betautils_serve_job) run the files a client chose. Both go
through Runner, so the command line and the app cannot drift apart.

    Job            the files, the overrides, the state and the counts
    Runner.run()   run a job from its next file until it ends or pauses
    FileReporter   what the pipeline tells the job about one file

STATES
    queued -> running -> completed | cancelled | stopped
    running <-> paused
    queued, paused -> cancelled | stopped

    'stopped' carries a code: disk.full, config.invalid,
    model.load_failed, engine.stopping, engine.internal. TRANSITIONS is
    the whole table; set_state refuses anything else.

PAUSE, CANCEL AND THE CHECKPOINTS
    Both stop at the pipeline's next safe point (betautils_signals.check),
    the same one Ctrl-C uses. Detection has written its checkpoint and
    finished render chunks are on disk, so a paused file restarts from
    them when the job is resumed. Cancel then removes the file's partial
    outputs (render chunks, intermediate, temps); detection caches and
    checkpoints are cache, not output, and stay for a later run.

EVENTS
    emit( name, data ) gets every event; None (the command line) means
    none are built. Names and shapes are in protocol/v1/events/:
    job.state, job.file, job.progress, job.decision, job.error. A file's
    events come between its job.file 'started' and its outcome; a job's
    last event is its job.state for a final state, or 'paused'.
"""

import collections
import os
import shutil

import betaconfig

import betautils_cache_paths as bu_cache
import betautils_hash as bu_hash
import betautils_log as bu_log
import betautils_signals as bu_signals

QUEUED = 'queued'
RUNNING = 'running'
PAUSED = 'paused'
COMPLETED = 'completed'
CANCELLED = 'cancelled'
STOPPED = 'stopped'

STATES = ( QUEUED, RUNNING, PAUSED, COMPLETED, CANCELLED, STOPPED )
FINAL = frozenset( ( COMPLETED, CANCELLED, STOPPED ) )

TRANSITIONS = {
    QUEUED: frozenset( ( RUNNING, CANCELLED, STOPPED ) ),
    RUNNING: frozenset( ( PAUSED, COMPLETED, CANCELLED, STOPPED ) ),
    PAUSED: frozenset( ( RUNNING, CANCELLED, STOPPED ) ),
    COMPLETED: frozenset(),
    CANCELLED: frozenset(),
    STOPPED: frozenset(),
}

# A file's outcomes, as process_one_video returns them and job.file reports them.
OUTCOMES = ( 'processed', 'skipped', 'failed' )

# The stage a ProgressReporter label names: 'detect 640' is detect at size 640.
_STAGES = { 'detect': 'detect', 'persons': 'persons', 'render': 'render' }


JobFile = collections.namedtuple( 'JobFile', 'path censored_folder' )


class BadTransition( Exception ):
    """A state change TRANSITIONS does not allow."""

    def __init__( self, current, wanted ):
        super().__init__( "%s -> %s"%( current, wanted ) )
        self.current = current
        self.wanted = wanted


class JobStopped( Exception ):
    """Raised inside a run to stop the job with a code and params."""

    def __init__( self, code, params=None ):
        super().__init__( code )
        self.code = code
        self.params = dict( params or {} )


class Job:
    """
    Args:
        job_id: The job's id, a string.
        files: JobFile entries, in the order they run.
        overrides: { path tuple: value } set over the config while the
            job runs, for this job only.
    """

    def __init__( self, job_id, files, overrides=None ):
        self.id = job_id
        self.files = list( files )
        self.overrides = dict( overrides or {} )
        self.state = QUEUED
        self.next_index = 0
        self.counts = { outcome: 0 for outcome in OUTCOMES }
        self.code = None
        self.params = {}
        self.current = None        # the FileReporter of the file in progress or paused

    def counts_data( self ):
        data = { 'total': len( self.files ) }
        data.update( self.counts )
        return data


def can_change( current, wanted ):
    return wanted in TRANSITIONS.get( current, () )


def output_folder_for( path ):
    """
    Where a chosen file's output goes: its folder's mirror under the
    output folder when it sits under the input folder, as the command
    line has always done, else the output folder itself.
    """
    import betaconst
    root = os.path.abspath( betaconst.video_path_uncensored )
    folder = os.path.dirname( os.path.abspath( path ) )
    relative = os.path.relpath( folder, root )
    if relative == os.curdir:
        return betaconst.video_path_censored
    if relative.startswith( os.pardir ) or os.path.isabs( relative ):
        return betaconst.video_path_censored
    return os.path.join( betaconst.video_path_censored, relative )


# --- What the pipeline reports about one file ---------------------------------

class NullReporter:
    """The reporter process_one_video uses when nobody is listening."""

    stage = 'prepare'
    paths = None

    def decision( self, code, **params ):
        pass

    def error( self, code, **params ):
        pass

    def output_paths( self, paths ):
        pass


NULL_REPORTER = NullReporter()


class FileReporter( NullReporter ):
    """One file of a job: turns what the pipeline reports into events."""

    def __init__( self, emit, job_id, index ):
        self._emit = emit
        self.job_id = job_id
        self.index = index
        self.stage = 'prepare'
        self.paths = None

    def decision( self, code, **params ):
        if self._emit is not None:
            self._emit( 'job.decision', { 'job': self.job_id, 'file': self.index,
                                          'code': code, 'params': params } )

    def error( self, code, **params ):
        if self._emit is not None:
            self._emit( 'job.error', { 'job': self.job_id, 'file': self.index,
                                       'code': code, 'params': params } )

    def output_paths( self, paths ):
        self.paths = paths

    def progress( self, label, done, total, rate ):
        """A ProgressReporter's counts (bu_log.set_progress_sink)."""
        parts = str( label ).split()
        stage = _STAGES.get( parts[0] if parts else '' )
        if stage is None:
            return
        self.stage = stage
        if self._emit is None:
            return
        data = { 'job': self.job_id, 'file': self.index, 'stage': stage,
                 'done': int( done ), 'total': int( total ) if total is not None else None,
                 'rate': round( float( rate ), 2 ) }
        if stage == 'detect' and len( parts ) > 1 and parts[1].isdigit():
            data['size'] = int( parts[1] )
        self._emit( 'job.progress', data )


def remove_partial_outputs( reporter, logger ):
    """Delete what an unfinished render of the reporter's file left. Returns how many."""
    if reporter is None or not reporter.paths:
        return 0
    removed = 0
    for path in bu_cache.partial_outputs( reporter.paths['final'] ):
        try:
            os.remove( path )
            removed += 1
        except OSError as err:
            logger.warning( "could not remove partial output %s: %s"%( path, err ) )
    if removed:
        logger.info( "removed %d partial output file(s) of %s"%(
            removed, os.path.basename( reporter.paths['final'] ) ) )
    return removed


def free_bytes( folder ):
    try:
        return shutil.disk_usage( folder ).free
    except OSError:
        return None


# --- The runner -----------------------------------------------------------------

class Runner:
    """
    Runs jobs through the pipeline.

    Args:
        logger: The shared logger.
        emit: emit( name, data ) for events, or None.
        pipeline: The module with the per-file pipeline (betatv). Looked
            up at call time, so a patched function is the one that runs.
        load_session: Callable returning the detector session; default
            the selected backend's get_session(). It may raise JobStopped.
    """

    def __init__( self, logger, emit=None, pipeline=None, load_session=None ):
        if pipeline is None:
            import betatv as pipeline
        self.logger = logger
        self.pipeline = pipeline
        self._emit = emit
        self._load_session = load_session

    # --- state and events ---

    def send( self, name, data ):
        if self._emit is not None:
            self._emit( name, data )

    def set_state( self, job, state, code=None, params=None ):
        """Change the job's state and report it. Raises BadTransition."""
        if not can_change( job.state, state ):
            raise BadTransition( job.state, state )
        job.state = state
        job.code = code
        job.params = dict( params or {} )
        data = { 'job': job.id, 'state': state, 'counts': job.counts_data() }
        if code is not None:
            data['code'] = code
            data['params'] = job.params
        self.send( 'job.state', data )

    def stop( self, job, code, params=None, file_index=None ):
        """End a job as 'stopped' with a code: a job.error, then the state."""
        error = { 'job': job.id, 'code': code, 'params': dict( params or {} ) }
        if file_index is not None:
            error['file'] = file_index
        self.send( 'job.error', error )
        self.set_state( job, STOPPED, code, params )

    def cancel_idle( self, job ):
        """Cancel a job that is queued or paused: nothing is running to stop."""
        if job.state == PAUSED:
            remove_partial_outputs( job.current, self.logger )
        self.set_state( job, CANCELLED )
        job.current = None

    # --- running ---

    def run( self, job ):
        """
        Run a job from its next file until it ends or is paused.

        Returns:
            The job's state: completed, paused, cancelled or stopped.

        Raises:
            betautils_signals.Interrupted: Ctrl-C (reason 'interrupt'),
                after the job's own clean-up; the command line exits 130.
            Anything else the pipeline raises outside a file's own
                handling (a model that will not load, say), as the
                command line always has.
        """
        self.set_state( job, RUNNING )
        try:
            if job.overrides:
                import betautils_config_store as bu_store
                with bu_store.overrides_applied( job.overrides ):
                    self._run_files( job )
            else:
                self._run_files( job )
        except JobStopped as stop:
            self.stop( job, stop.code, stop.params,
                       job.current.index if job.current is not None and stop.code == 'disk.full' else None )
        except bu_signals.Interrupted as stop:
            if stop.reason == bu_signals.INTERRUPT:
                raise
            bu_signals.clear()
            if stop.reason == bu_signals.PAUSE:
                self.logger.info( "job %s paused at file %d of %d"%(
                    job.id, job.next_index + 1, len( job.files ) ) )
                self.set_state( job, PAUSED )
            elif stop.reason == bu_signals.CANCEL:
                remove_partial_outputs( job.current, self.logger )
                job.current = None
                self.logger.info( "job %s cancelled"%( job.id, ) )
                self.set_state( job, CANCELLED )
            else:
                self.logger.info( "job %s stopped: the engine is stopping"%( job.id, ) )
                self.set_state( job, STOPPED, 'engine.stopping' )
        else:
            job.current = None
            self.set_state( job, COMPLETED )
        return job.state

    def _session( self ):
        if self._load_session is not None:
            return self._load_session()
        return self.pipeline.bu_detector.get_detector().get_session()

    def _run_files( self, job ):
        pipeline = self.pipeline
        logger = self.logger
        pipeline._log_run_banner( logger )

        preview_mode_enabled = getattr( betaconfig, 'preview_mode_enabled', False )
        preview_max_seconds = getattr( betaconfig, 'preview_max_seconds', 20 )
        active_encode_preset = pipeline._resolve_active_encode_preset()

        run_marker = bu_log.RunMarker(
            bu_log.run_marker_path_for( bu_log.configured_stats_path() ),
            logger, boot_id=bu_log.current_boot_id() )
        run_marker.report_previous()
        run_marker.start()
        # Room for the failure record and the log if the disk fills.
        bu_log.ensure_reserve()
        try:
            # Inside the try, so a model that fails to load (or a stop
            # while it loads) still removes the run marker on the way out.
            session = self._session()
            total = len( job.files )
            for index in range( job.next_index, total ):
                bu_signals.check()
                entry = job.files[index]
                folder, fname = os.path.split( entry.path )
                os.makedirs( entry.censored_folder, exist_ok=True )
                run_marker.now_processing( fname, preview_mode_enabled )
                reporter = FileReporter( self._emit, job.id, index )
                job.current = reporter
                self.send( 'job.file', { 'job': job.id, 'file': index, 'state': 'started' } )
                previous_sink = bu_log.set_progress_sink(
                    reporter.progress if self._emit is not None else None )
                try:
                    outcome = pipeline.process_one_video(
                        folder, fname, entry.censored_folder, index, total, session,
                        preview_mode_enabled, preview_max_seconds, active_encode_preset,
                        logger, reporter=reporter )
                finally:
                    bu_log.set_progress_sink( previous_sink )
                if outcome == 'out_of_space':
                    job.counts['failed'] += 1
                    self.send( 'job.file', { 'job': job.id, 'file': index, 'state': 'failed' } )
                    self._disk_full( entry.censored_folder )
                job.counts[outcome] += 1
                data = { 'job': job.id, 'file': index, 'state': outcome }
                if outcome == 'processed' and reporter.paths:
                    data['output'] = reporter.paths['final']
                self.send( 'job.file', data )
                job.next_index = index + 1
                job.current = None
        finally:
            bu_hash.flush_file_hash_memo()
            # Every ending this process can still act on removes the marker,
            # so one left behind means the run died without a chance to log.
            run_marker.finish()

    def _disk_full( self, folder ):
        # Stopping is the resumable choice: finished outputs are skipped
        # next time, the file that failed is rebuilt, and detections
        # already cached are not run again.
        note = self.pipeline._free_space_note( folder )
        self.logger.error( "stopping the run: a disk is full%s. Free some space and run again; finished "
                           "files are skipped and the one that failed is rebuilt."%(
                               " (%s)"%( note, ) if note else "" ) )
        raise JobStopped( 'disk.full', { 'free_bytes': free_bytes( folder ) } )
