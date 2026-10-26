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
    every chunk (suppression_ab, 2026-10-10), so every logger in the
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
    betasuite.log, so summarize_run showed it as simply absent; on
    2026-10-24 two render failures read as "missing from the run".

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


# Historical name, kept so older tooling keeps resolving.
_format_duration = format_duration
