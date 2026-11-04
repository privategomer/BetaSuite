"""
betautils_log.py - Logging, progress reporting, and structured run stats.

Three responsibilities, all shared by every entry point:

  1. get_logger()      one lazily-configured 'betasuite' logger, writing
                       to a file at `log_level` and to the console at
                       `console_level`, independently settable.
  2. progress()        a throttled progress reporter, so a per-frame
                       loop cannot flood the log with one line per frame
                       (see ProgressReporter).
  3. write_stats()     one JSON-Lines record per processed file, for
                       after-the-fact analysis of real runs.

Level vocabulary is trace / debug / info / warn / error, defaulting to
info. TRACE is a BetaSuite addition below DEBUG, for per-frame and
per-box detail that would drown a debug log.

Everything a run prints goes through this module. Bare print() in the
pipeline is a bug: it bypasses the level filter, bypasses the log file,
and (in the per-frame case) produced 16KB of carriage-return spam per
video in the earlier logs.
"""

import json
import logging
import os
import sys
import threading
import time

import betaconfig


# --- Level vocabulary -------------------------------------------------------

TRACE = 5
logging.addLevelName( TRACE, 'TRACE' )

LEVEL_NAMES = {
    'trace': TRACE,
    'debug': logging.DEBUG,
    'info':  logging.INFO,
    'warn':  logging.WARNING,
    'error': logging.ERROR,
}

DEFAULT_LOG_PATH = '../output/logs/betasuite.log'
DEFAULT_STATS_PATH = '../output/stats/betasuite_stats.jsonl'

_logger = None
_logger_lock = threading.Lock()


def resolve_level( level_name, default=logging.INFO ):
    """
    Translate a betaconfig.py / CLI level string into a logging constant.

    Args:
        level_name: One of LEVEL_NAMES' keys (case-insensitive).
        default: Returned when level_name is not recognised.

    Returns:
        A logging level int.
    """
    if not isinstance( level_name, str ):
        return default
    return LEVEL_NAMES.get( level_name.strip().lower(), default )


class _BetaLogger( logging.Logger ):
    """A Logger that also understands trace()."""

    def trace( self, msg, *args, **kwargs ):
        if self.isEnabledFor( TRACE ):
            self._log( TRACE, msg, args, **kwargs )


FILE_FORMAT = '%(asctime)s [%(levelname)-5s] %(message)s'
CONSOLE_FORMAT = '%(message)s'


def _trace_capable( name ):
    """
    logging.getLogger( name ) as a _BetaLogger, whoever created it first.

    Pipeline code calls logger.trace(). A tool that built its own plain
    logging.Logger and handed it to the render path got AttributeError on
    every chunk (suppression_ab), so every logger in the
    project comes from here.
    """
    logger = logging.getLogger( name )
    if not isinstance( logger, _BetaLogger ):
        # A subclass adding one method: retyping in place keeps every
        # reference, handler and level the logger already had.
        logger.__class__ = _BetaLogger
    return logger


def _as_level( level ):
    return level if isinstance( level, int ) else resolve_level( level )


def tool_logger( name, log_path=None, file_level='info', console_level='info', file_mode='a',
                 file_format=FILE_FORMAT, console_format=CONSOLE_FORMAT, console_stream=None ):
    """
    A tool's own logger: everything at file_level to log_path, and a view
    filtered at console_level to the terminal. Trace-capable, so it can
    be handed to any pipeline function.

    Args:
        name: Logger name, usually the tool's.
        log_path: The log file, or None for none.
        file_level, console_level: Level names ('trace'..'error') or
            logging ints. console_level None means no terminal output.
        file_mode: 'a' to append, 'w' to start the file fresh.
        file_format, console_format: logging format strings.
        console_stream: Where the terminal view goes; stdout by default.

    Returns:
        The logger. Building it again replaces its handlers.
    """
    logger = _trace_capable( name )
    logger.propagate = False
    logger.setLevel( TRACE )
    for handler in list( logger.handlers ):
        logger.removeHandler( handler )
        try:
            handler.close()
        except Exception:
            pass
    if log_path:
        log_dir = os.path.dirname( log_path )
        if log_dir:
            os.makedirs( log_dir, exist_ok=True )
        file_handler = logging.FileHandler( log_path, mode=file_mode, encoding='UTF-8' )
        file_handler.setLevel( _as_level( file_level ) )
        file_handler.setFormatter( logging.Formatter( file_format ) )
        logger.addHandler( file_handler )
    if console_level is not None:
        console_handler = logging.StreamHandler( console_stream or sys.stdout )
        console_handler.setLevel( _as_level( console_level ) )
        console_handler.setFormatter( logging.Formatter( console_format ) )
        logger.addHandler( console_handler )
    return logger


def silent_logger( name ):
    """A trace-capable logger that discards everything, for replays and workers."""
    logger = _trace_capable( name )
    logger.propagate = False
    if not logger.handlers:
        logger.addHandler( logging.NullHandler() )
    return logger


def _build_configured_logger():
    """
    Construct the 'betasuite' logger from betaconfig's current settings.

    Two independent handlers so the file can capture trace detail while
    the console stays readable:
      - FileHandler   at betaconfig.log_level      (default 'info')
      - StreamHandler at betaconfig.console_level  (default 'info')

    When logging_enabled is falsy the file handler is omitted; the
    console handler stays, because a run with no output at all is a
    worse failure mode than a noisy one.

    Returns:
        A configured logging.Logger (uncached - go through get_logger()).
    """
    logger = _trace_capable( 'betasuite' )
    logger.propagate = False

    if logger.handlers:
        return logger

    file_level = resolve_level( getattr( betaconfig, 'log_level', 'info' ) )
    console_level = resolve_level( getattr( betaconfig, 'console_level', 'info' ) )

    # The logger itself must pass through the most permissive of the two,
    # or a handler can never see a record its own level would allow.
    logger.setLevel( min( file_level, console_level ) )

    if getattr( betaconfig, 'logging_enabled', False ):
        log_path = getattr( betaconfig, 'log_path', None ) or DEFAULT_LOG_PATH
        log_dir = os.path.dirname( log_path )
        if log_dir:
            os.makedirs( log_dir, exist_ok=True )
        file_handler = logging.FileHandler( log_path, encoding='UTF-8' )
        file_handler.setLevel( file_level )
        file_handler.setFormatter( logging.Formatter( FILE_FORMAT ) )
        logger.addHandler( file_handler )

    console_handler = logging.StreamHandler( sys.stdout )
    console_handler.setLevel( console_level )
    console_handler.setFormatter( logging.Formatter( CONSOLE_FORMAT ) )
    logger.addHandler( console_handler )

    return logger


def get_logger():
    """
    Return BetaSuite's shared logger, configuring it on first use.

    Thread-safe and idempotent: every caller in the process gets the same
    instance with the same handlers attached exactly once.
    """
    global _logger
    if _logger is None:
        with _logger_lock:
            if _logger is None:
                _logger = _build_configured_logger()
    return _logger


def reset_logger_for_tests():
    """Drop the cached logger and its handlers. Test-support only."""
    global _logger
    with _logger_lock:
        logger = logging.getLogger( 'betasuite' )
        for handler in list( logger.handlers ):
            logger.removeHandler( handler )
            try:
                handler.close()
            except Exception:
                pass
        _logger = None


# --- Progress reporting -----------------------------------------------------

class ProgressReporter:
    """
    Rate-limited progress output for a tight per-frame loop.

    The problem this solves: the earlier detection and render loops each
    called print(..., end='\\r') once per frame. On a tty that is a
    redraw; piped to a log file it is one more copy of the whole line,
    which is how a 90-second video produced a 16KB single-line log.

    Behaviour:
      - On a tty, repaints an in-place status line at most every
        `min_interval` seconds.
      - Off a tty, emits nothing to the console.
      - Either way, emits a real log record (at `level`, default DEBUG)
        at most every `log_interval` seconds, so the log has a readable
        progress trail with a bounded line count.

    Usage:
        with bu_log.ProgressReporter( "detect 1280", total=4200 ) as progress:
            for i, frame in enumerate( frames ):
                ...
                progress.update( i+1 )
    """

    def __init__( self, label, total=None, min_interval=0.25,
                  log_interval=15.0, level=logging.DEBUG, logger=None ):
        self.label = label
        self.total = total
        self.min_interval = min_interval
        self.log_interval = log_interval
        self.level = level
        self.logger = logger or get_logger()
        self._start = time.monotonic()
        # The tty line paints on the first update (immediate feedback);
        # the first LOG record waits a full interval, so a short job
        # produces exactly one summary line rather than a start line
        # plus a summary.
        self._last_paint = 0.0
        self._last_log = self._start
        self._current = 0
        self._painted = False
        self._lock = threading.Lock()
        self._isatty = bool( getattr( sys.stdout, 'isatty', lambda: False )() )

    def _format( self, current ):
        elapsed = max( time.monotonic() - self._start, 1e-6 )
        rate = current / elapsed
        if self.total:
            pct = 100.0 * current / max( self.total, 1 )
            remaining = ( self.total - current ) / rate if rate > 0 else 0
            return "%s: %d/%d (%.1f%%) %.1f/s eta %s"%(
                self.label, current, self.total, pct, rate, format_duration( remaining ) )
        return "%s: %d %.1f/s"%(self.label, current, rate)

    def advance( self, delta=1 ):
        """
        Record `delta` more units of progress.

        The thread-safe counterpart to update(): parallel render workers
        each know how many frames THEY wrote, not the global total.
        """
        with self._lock:
            current = self._current + delta
        self.update( current )

    def update( self, current ):
        """Record progress. Cheap enough to call once per frame."""
        now = time.monotonic()
        with self._lock:
            self._current = current
            if self._isatty and ( now - self._last_paint ) >= self.min_interval:
                self._last_paint = now
                sys.stdout.write( "\r\033[K" + self._format( current ) )
                sys.stdout.flush()
                self._painted = True
            if ( now - self._last_log ) >= self.log_interval:
                self._last_log = now
                self.logger.log( self.level, self._format( current ) )

    def finish( self, note=None ):
        """Clear the in-place line and emit one summary record."""
        with self._lock:
            if self._painted:
                sys.stdout.write( "\r\033[K" )
                sys.stdout.flush()
                self._painted = False
            elapsed = time.monotonic() - self._start
            summary = "%s: done, %d in %s"%(
                self.label, self._current, format_duration( elapsed ) )
            if note:
                summary += " (%s)"%(note)
            self.logger.info( summary )

    def __enter__( self ):
        return self

    def __exit__( self, exc_type, exc, tb ):
        self.finish()
        return False


def format_duration( seconds ):
    """Render a duration as the shortest readable h/m/s string."""
    seconds = max( 0.0, float( seconds ) )
    if seconds < 60:
        return "%.1fs"%(seconds)
    if seconds < 3600:
        return "%dm%02ds"%(int(seconds//60), int(seconds%60))
    return "%dh%02dm"%(int(seconds//3600), int((seconds%3600)//60))


# --- Structured run stats ---------------------------------------------------

_stats_lock = threading.Lock()


def write_stats( record ):
    """
    Append one JSON-Lines record to betaconfig.stats_path.

    Stats are a nice-to-have: a failure here is logged and swallowed,
    never allowed to abort a censoring run.

    Args:
        record: A JSON-serialisable dict. A 'timestamp' key is added if
            absent. The caller's dict is never mutated.
    """
    if not getattr( betaconfig, 'stats_enabled', False ):
        return
    _append_record( getattr( betaconfig, 'stats_path', None ) or DEFAULT_STATS_PATH, record,
                    'stats record' )


def failures_path_for( stats_path ):
    """
    The failures file that sits beside a stats file:
    betasuite_stats.jsonl -> betasuite_stats.failures.jsonl.

    Derived rather than configured, so a tool pointed at any stats file
    (summarize_run --stats-path) finds that run's failures with it.
    """
    stem, extension = os.path.splitext( stats_path )
    return stem + '.failures' + ( extension or '.jsonl' )


def write_failure( record ):
    """
    Append one JSON-Lines record for a file that failed to process.

    A separate file, not a row in the stats file: five tools read every
    stats row as a finished render (timings, label counts, encode
    estimates), and a failure row among them would quietly skew all of
    them. Before this existed a failed file left no trace outside
    betasuite.log, so summarize_run showed it as simply absent: two render
    failures once read as "missing from the run".

    Same switch and same swallow-and-log behaviour as write_stats.
    """
    if not getattr( betaconfig, 'stats_enabled', False ):
        return
    stats_path = getattr( betaconfig, 'stats_path', None ) or DEFAULT_STATS_PATH
    _append_record( failures_path_for( stats_path ), record, 'failure record' )


def _append_record( path, record, what ):
    """Append record as one JSON line to path; log and swallow any failure."""
    payload = dict( record )
    payload.setdefault( 'timestamp', time.time() )
    try:
        directory = os.path.dirname( path )
        if directory:
            os.makedirs( directory, exist_ok=True )
        line = json.dumps( payload, default=str ) + "\n"
        with _stats_lock:
            with open( path, 'a', encoding='UTF-8' ) as fout:
                fout.write( line )
    except Exception as err:
        get_logger().warning( "failed to write %s: %s"%( what, err ) )


# --- Unfinished-run detection -------------------------------------------------

def run_marker_path_for( stats_path ):
    """The run marker that sits beside a stats file: betasuite_stats.jsonl -> betasuite_stats.running.json."""
    stem, _extension = os.path.splitext( stats_path )
    return stem + '.running.json'


def current_boot_id( path='/proc/sys/kernel/random/boot_id' ):
    """This boot's id on Linux, so a marker from before a reboot is never mistaken for a live run."""
    try:
        with open( path, 'r', encoding='ascii' ) as handle:
            return handle.read().strip() or None
    except OSError:
        return None


def betatv_process_running( pid, proc_root='/proc' ):
    """
    True if pid is a running betatv, False if it is not, None where this
    platform cannot tell.

    os.kill( pid, 0 ) only asks whether a pid exists, on POSIX. On
    Windows os.kill with any other signal TERMINATES the process, so it
    is never called there. A pid that exists but whose command line has
    no 'betatv' in it was reused by something else after the run died.
    """
    if os.name != 'posix' or not isinstance( pid, int ) or pid <= 0:
        return None
    try:
        os.kill( pid, 0 )
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    except OSError:
        return None
    try:
        with open( os.path.join( proc_root, str( pid ), 'cmdline' ), 'rb' ) as handle:
            command_line = handle.read()
    except OSError:
        return True
    return b'betatv' in command_line


class RunMarker:
    """
    A small file that exists exactly while a betatv run is running,
    naming the file it is on.

    WHY. A process killed with SIGKILL (the kernel's out-of-memory
    killer, systemd-oomd), a native crash inside a library, or a power
    loss ends the run without a single line in betasuite.log: no
    "failed", no "run finished", nothing the failures file or
    summarize_run could ever show. Two runs once died that way
    on a 4K file and the only trace was the shell printing "Killed". The
    marker is removed by every ending the run can still act on (finish,
    Ctrl-C, an exception), so one found at startup means the last run
    ended in a way it could not report, and this run says so and records
    the file it was on as a failure.

    Args:
        path: run_marker_path_for( stats_path ).
        logger: The shared logger.
        clock: Returns epoch seconds; injected for tests.
        pid: This process's id.
        boot_id: current_boot_id(), or None where unknown.
        is_running: pid -> True/False/None, see betatv_process_running.
    """

    def __init__( self, path, logger, clock=time.time, pid=None, boot_id=None,
                  is_running=betatv_process_running ):
        self.path = path
        self.logger = logger
        self.clock = clock
        self.pid = os.getpid() if pid is None else pid
        self.boot_id = boot_id
        self.is_running = is_running
        self.record = None

    def read( self ):
        """The marker on disk, or None if there is none or it cannot be read."""
        try:
            with open( self.path, 'r', encoding='UTF-8' ) as handle:
                record = json.load( handle )
        except FileNotFoundError:
            return None
        except ( OSError, ValueError ) as err:
            self.logger.debug( "unreadable run marker %s ignored: %s"%( self.path, err ) )
            return None
        return record if isinstance( record, dict ) else None

    def previous_run_state( self, previous ):
        """
        'running', 'ended' or 'unknown' for a marker found at startup.
        A different boot id settles it before asking about the pid,
        because pids are reused across reboots.
        """
        if previous.get( 'boot_id' ) and self.boot_id and previous['boot_id'] != self.boot_id:
            return 'ended'
        running = self.is_running( previous.get( 'pid' ) )
        if running is None:
            return 'unknown'
        return 'running' if running else 'ended'

    def report_previous( self ):
        """
        Say what a leftover marker means and, when the earlier run died
        on a file, record that file as failed so summarize_run lists it.

        Returns:
            'none', 'running', 'ended' or 'unknown'.
        """
        previous = self.read()
        if previous is None:
            return 'none'
        state = self.previous_run_state( previous )
        started = _clock_text( previous.get( 'started_at' ) )
        if state == 'running':
            self.logger.warning( "another betatv run (pid %s, started %s) is still running; two runs "
                                 "over the same folders write the same outputs"%(
                                     previous.get( 'pid' ), started ) )
            return state
        fname = previous.get( 'file' )
        where = ( "while processing %s (since %s)"%( fname, _clock_text( previous.get( 'file_started_at' ) ) )
                  if fname else "between files" )
        caveat = ( " (this platform cannot check whether it is still running)"
                   if state == 'unknown' else "" )
        self.logger.warning(
            "the previous run (pid %s, started %s) ended without finishing %s%s. A run that is killed "
            "(out of memory), crashes inside a library or loses power cannot log why; on Linux "
            "`journalctl -k | grep -i oom` around that time shows an out-of-memory kill"%(
                previous.get( 'pid' ), started, where, caveat ) )
        if fname:
            write_failure( {
                'file': fname,
                'error': "the run ended while processing this file without logging why: killed "
                         "(out of memory), crashed, or lost power",
                'preview_mode': bool( previous.get( 'preview_mode' ) ),
                'timestamp': previous.get( 'file_started_at' ) or previous.get( 'started_at' ),
                'unfinished_run_pid': previous.get( 'pid' ),
            } )
        return state

    def start( self ):
        """Write this run's marker."""
        self.record = { 'pid': self.pid, 'boot_id': self.boot_id, 'started_at': self.clock(),
                        'file': None, 'file_started_at': None, 'preview_mode': None }
        self._write()

    def now_processing( self, fname, preview_mode ):
        """Name the file the run is about to work on."""
        if self.record is None:
            return
        self.record.update( { 'file': fname, 'file_started_at': self.clock(),
                              'preview_mode': bool( preview_mode ) } )
        self._write()

    def finish( self ):
        """Remove the marker, but only this run's: never another run's that replaced it."""
        if self.record is None:
            return
        self.record = None
        on_disk = self.read()
        if on_disk is not None and on_disk.get( 'pid' ) != self.pid:
            return
        try:
            os.remove( self.path )
        except FileNotFoundError:
            pass
        except OSError as err:
            self.logger.warning( "could not remove the run marker %s: %s"%( self.path, err ) )

    def _write( self ):
        # Written whole and renamed into place, so a run killed mid-write
        # leaves the previous marker rather than half a JSON file.
        temporary = self.path + '.tmp'
        try:
            directory = os.path.dirname( self.path )
            if directory:
                os.makedirs( directory, exist_ok=True )
            with open( temporary, 'w', encoding='UTF-8' ) as handle:
                json.dump( self.record, handle )
            os.replace( temporary, self.path )
        except OSError as err:
            self.logger.warning( "could not write the run marker %s: %s"%( self.path, err ) )


def _clock_text( epoch_seconds ):
    """Local time for a marker timestamp, or 'an unknown time'."""
    if not isinstance( epoch_seconds, (int, float) ):
        return "an unknown time"
    return time.strftime( '%Y-%m-%d %H:%M:%S', time.localtime( epoch_seconds ) )


# Historical name, kept so older tooling keeps resolving.
_format_duration = format_duration
