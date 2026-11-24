"""
betautils_serve.py - The engine's serve mode: betautils_protocol over
stdin and stdout, for the desktop UI.

    run()             isolate stdio, build the logger, serve, exit code
    Server            the read, dispatch, reply loop over two streams
    register()        add a command; every command beyond the built-ins
                      registers here

KEEPING STDOUT PROTOCOL-ONLY
    In serve mode stdout is the wire, and one stray byte on it breaks
    the client's parser. Rebinding sys.stdout is not enough on its own:
    the logger's console handler, ProgressReporter, C libraries and every
    child process that inherits stdout (render's ffmpeg does) write to
    file descriptor 1 directly, and ffmpeg also reads its inherited
    stdin, which here would be the command stream. So isolate_stdio(),
    the first thing run() does, works at the descriptor level:

      1. copies fds 0 and 1 to private descriptors. Python makes those
         non-inheritable, so no child ever receives them, and only the
         protocol reader and writer use them.
      2. points fd 1 at stderr. Anything written to fd 1 from here on,
         by this process or a child, lands on stderr.
      3. points fd 0 at the null device, so a child reads end of input
         instead of commands, and input() returns at once.
      4. replaces sys.stdout with StrayOutput, which turns each printed
         line into a warning in the log, and sys.stdin with the null
         device.

    The logger is then built on the real stderr and the log file. The
    client must keep reading stderr (protocol/README.md says why).

ENDINGS
    shutdown, or end of input (the client closed stdin or died), exit 0.
    SIGINT exits 130 and SIGTERM 143, through the same clean-up. A
    failure before serving starts (a config that does not import)
    prints a traceback to stderr and exits 2.

Imports at the top are the standard library and betautils_protocol
only; the engine modules are imported in run(), after isolate_stdio().
"""

import faulthandler
import io
import os
import platform
import signal
import sys
import threading
import traceback

import betautils_protocol as bu_proto

EXIT_OK = 0
EXIT_STARTUP_FAILED = 2
EXIT_INTERRUPTED = 130
EXIT_TERMINATED = 143

# Allowed before hello.
ALWAYS_ALLOWED = ( 'hello', 'shutdown' )

# Events the engine can send. The job events will join them.
EVENTS = ( bu_proto.REJECTED_EVENT, )

# The longest piece of a message the trace log shows.
_PREVIEW_BYTES = 2048

_COMMANDS = {}


def register( name, handler ):
    """
    Add a command. handler( server, data ) gets the command's data
    object and returns the result's data object (None means {}).

    A handler reports a bad request by raising bu_proto.ProtocolError
    (usually 'command.bad_params'); the server fills in the id and name.
    Any other exception becomes 'engine.internal', with the traceback in
    the log, and the server keeps serving.
    """
    _COMMANDS[name] = handler
    return handler


def registered_commands():
    """A copy of the command table."""
    return dict( _COMMANDS )


# --- Stdio isolation ----------------------------------------------------------

class StrayOutput( io.TextIOBase ):
    """
    sys.stdout in serve mode: each line written becomes a warning in the
    log ("stray stdout: ..."), because in serve mode a print is a bug.

    Until attach() hands it the logger, and whenever the logger itself
    ends up writing here, lines go to stderr instead, so nothing is lost
    and nothing recurses. isatty() is False, which keeps ProgressReporter
    from painting its terminal line. fileno() is 1, which by then points
    at stderr.
    """

    _MAX_PENDING = 64*1024

    def __init__( self ):
        super().__init__()
        self._logger = None
        self._pending = ''
        self._lock = threading.Lock()
        self._local = threading.local()

    def attach( self, logger ):
        self._logger = logger

    @property
    def encoding( self ):
        return 'utf-8'

    def writable( self ):
        return True

    def isatty( self ):
        return False

    def fileno( self ):
        return 1

    def write( self, text ):
        if not isinstance( text, str ):
            raise TypeError( "write() argument must be str, not %s"%( type( text ).__name__, ) )
        with self._lock:
            self._pending += text
            lines = self._pending.split( '\n' )
            self._pending = lines.pop()
            if len( self._pending ) > self._MAX_PENDING:
                lines.append( self._pending )
                self._pending = ''
        for line in lines:
            self._emit( line )
        return len( text )

    def flush( self ):
        with self._lock:
            line, self._pending = self._pending, ''
        if line:
            self._emit( line )

    def _emit( self, line ):
        line = line.rstrip( '\r' )
        if not line.strip():
            return
        if self._logger is None or getattr( self._local, 'busy', False ):
            os.write( 2, ( 'stray stdout: %s\n'%( line, ) ).encode( 'utf-8', 'backslashreplace' ) )
            return
        self._local.busy = True
        try:
            self._logger.warning( "stray stdout: %s", line )
        finally:
            self._local.busy = False


def _binary( fd ):
    if os.name == 'nt':
        import msvcrt
        msvcrt.setmode( fd, os.O_BINARY )
    return fd


def _point_windows_std_handles():
    """
    Make the Windows standard handles match fds 0 and 1 after the dup2s.

    Child processes started without explicit handles inherit the
    process's standard handles, not its C runtime fds. The C runtime
    updates those handles on dup2 only in a console program (python.exe),
    not in pythonw.exe, so this sets them itself either way.
    """
    import ctypes
    import msvcrt
    kernel32 = ctypes.windll.kernel32
    for std_handle, fd in ( ( -10, 0 ), ( -11, 1 ) ):   # STD_INPUT_HANDLE, STD_OUTPUT_HANDLE
        kernel32.SetStdHandle( ctypes.c_uint32( std_handle & 0xFFFFFFFF ),
                               ctypes.c_void_p( msvcrt.get_osfhandle( fd ) ) )


def isolate_stdio():
    """
    Take stdin and stdout for the protocol; see the module docstring.

    Returns:
        ( proto_in, proto_out, stray ): buffered binary streams on the
        private copies of fds 0 and 1, and the StrayOutput now installed
        as sys.stdout.
    """
    for stream in ( sys.stdout, sys.stderr ):
        try:
            stream.flush()
        except Exception:
            pass
    proto_in_fd = _binary( os.dup( 0 ) )
    proto_out_fd = _binary( os.dup( 1 ) )
    os.dup2( 2, 1 )
    null_fd = os.open( os.devnull, os.O_RDONLY )
    try:
        os.dup2( null_fd, 0 )
    finally:
        os.close( null_fd )
    if os.name == 'nt':
        _point_windows_std_handles()
    proto_in = os.fdopen( proto_in_fd, 'rb', buffering=64*1024 )
    proto_out = os.fdopen( proto_out_fd, 'wb', buffering=64*1024 )
    sys.stdin = open( os.devnull, 'r', encoding='utf-8' )
    stray = StrayOutput()
    sys.stdout = stray
    return proto_in, proto_out, stray


# --- The loop -----------------------------------------------------------------

def _preview( line ):
    text = line[:_PREVIEW_BYTES].decode( 'utf-8', 'backslashreplace' )
    if len( line ) > _PREVIEW_BYTES:
        text += ' ... (%d bytes)'%( len( line ), )
    return text


def _traces( logger ):
    import betautils_log as bu_log
    levels = [ handler.level for handler in logger.handlers ]
    return bool( levels ) and min( levels ) <= bu_log.TRACE


class Server:
    """
    Reads commands from proto_in and writes results to proto_out until
    shutdown or end of input. Never raises for anything the client sends.
    """

    def __init__( self, proto_in, proto_out, logger, commands=None, limit=bu_proto.MAX_LINE_BYTES ):
        self._reader = bu_proto.LineReader( proto_in, limit )
        self._writer = bu_proto.MessageWriter( proto_out )
        self._logger = logger
        self._commands = registered_commands() if commands is None else dict( commands )
        self._log_messages = _traces( logger )
        self.limit = limit
        self.protocol = None
        self.stopping = False

    @property
    def command_names( self ):
        return sorted( self._commands )

    def serve( self ):
        """Serve until shutdown, end of input, or a client that went away. Returns 0."""
        while not self.stopping:
            try:
                line = self._reader.read_line()
            except bu_proto.LineTooLong as too_long:
                self._logger.warning( "discarded a %d-byte line (limit %d)"%(
                    too_long.line_bytes, too_long.limit_bytes ) )
                self._reply_error( too_long.as_protocol_error() )
                continue
            if line is None:
                self._logger.info( "command stream closed, stopping" )
                break
            self.handle_line( line )
        return EXIT_OK

    def handle_line( self, line ):
        """Answer one line."""
        if self._log_messages:
            self._logger.trace( "recv %s", _preview( line ) )
        try:
            command = bu_proto.parse_command( line )
        except bu_proto.ProtocolError as err:
            self._logger.debug( "rejected a line: %s %s", err.code, err.error['params'] )
            self._reply_error( err )
            return
        self._dispatch( command )

    def _dispatch( self, command ):
        if self.protocol is None and command.name not in ALWAYS_ALLOWED:
            self._reply_error( bu_proto.ProtocolError(
                'protocol.hello_required', { 'name': command.name }, command.id, command.name ) )
            return
        handler = self._commands.get( command.name )
        if handler is None:
            self._reply_error( bu_proto.ProtocolError(
                'command.unknown', { 'name': command.name }, command.id, command.name ) )
            return
        try:
            data = handler( self, command.data )
        except bu_proto.ProtocolError as err:
            err.request_id, err.request_name = command.id, command.name
            self._reply_error( err )
            return
        except Exception as err:
            self._logger.error( "command %s failed"%( command.name, ), exc_info=True )
            self._reply_internal( command.id, command.name, err )
            return
        self._send_result( bu_proto.ok_result( command.id, command.name, {} if data is None else data ) )

    def _reply_error( self, err ):
        # No readable id: nothing to answer, so say which line was refused.
        if err.request_id is None:
            self._send( bu_proto.rejected( self._reader.line_number, err.error ) )
            return
        self._send_result( bu_proto.error_result( err.request_id, err.request_name, err.error ) )

    def _reply_internal( self, request_id, name, err ):
        self._send( bu_proto.error_result( request_id, name, bu_proto.error(
            'engine.internal', exception_type=type( err ).__name__ ) ) )

    def _send_result( self, message ):
        # A result whose data is not JSON (NaN, a set) becomes an internal
        # error rather than a crash or a line the client cannot parse.
        try:
            self._send( message )
        except ( ValueError, TypeError ) as err:
            self._logger.error( "result for %s could not be encoded"%( message.get( 'name' ), ), exc_info=True )
            self._reply_internal( message.get( 'id' ), message.get( 'name' ), err )

    def _send( self, message ):
        try:
            line = self._writer.send( message )
        except OSError:
            if not self.stopping:
                self._logger.warning( "the client stopped reading; stopping" )
            self.stopping = True
            return
        if self._log_messages:
            self._logger.trace( "send %s", _preview( line[:-1] ) )


# --- Built-in commands --------------------------------------------------------

def _engine_info():
    import betautils_version as bu_version
    return { 'version': bu_version.get_version(), 'python': platform.python_version(),
             'platform': sys.platform }


def _hello( server, data ):
    offered = data.get( 'protocol' )
    if offered is None:
        raise bu_proto.ProtocolError( 'command.bad_params', { 'field': 'protocol', 'reason': 'missing' } )
    if not isinstance( offered, list ):
        raise bu_proto.ProtocolError( 'command.bad_params', { 'field': 'protocol', 'reason': 'type' } )
    client = data.get( 'client', {} )
    if not isinstance( client, dict ):
        raise bu_proto.ProtocolError( 'command.bad_params', { 'field': 'client', 'reason': 'type' } )
    version = bu_proto.choose_version( offered )
    if version is None:
        raise bu_proto.ProtocolError( 'protocol.unsupported_version',
                                      { 'got': offered, 'supported': list( bu_proto.SUPPORTED_VERSIONS ) } )
    server.protocol = version
    server._logger.info( "hello from %s %s, protocol %d"%(
        str( client.get( 'name', '?' ) )[:64], str( client.get( 'version', '?' ) )[:64], version ) )
    return {
        'protocol': version,
        'engine': _engine_info(),
        'pid': os.getpid(),
        'commands': server.command_names,
        'events': list( EVENTS ),
        'limits': { 'max_line_bytes': server.limit },
    }


def _ping( _server, _data ):
    return {}


def _shutdown( server, _data ):
    server._logger.info( "shutdown requested" )
    server.stopping = True
    return {}


register( 'hello', _hello )
register( 'ping', _ping )
register( 'shutdown', _shutdown )

# config.schema, config.get, config.set, config.reset
from betautils_serve_config import HANDLERS as _CONFIG_HANDLERS   # noqa: E402  (after register)
for _name, _handler in _CONFIG_HANDLERS.items():
    register( _name, _handler )


# --- Entry --------------------------------------------------------------------

class Terminated( BaseException ):
    """SIGTERM, raised in the main thread. Not an Exception, so no handler swallows it."""


def _on_sigterm( _signum, _frame ):
    raise Terminated()


def run( log_path=None, log_level=None, console_level=None, commands=None ):
    """
    Serve on this process's stdin and stdout. Returns the exit code.

    Args:
        log_path: The log file. Default: betaconfig.log_path, else
            betautils_log.DEFAULT_LOG_PATH, under BETASUITE_DATA_ROOT
            when that is set (betautils_log.configured_log_path). Serve
            mode always writes one,
            whatever logging_enabled says, since nothing else records it.
        log_level, console_level: Level names for the file and stderr.
            Default: betaconfig's log_level and console_level.
        commands: The command table, default every registered command.
    """
    try:
        proto_in, proto_out, stray = isolate_stdio()
        faulthandler.enable( file=sys.stderr )
        import betaconfig
        import betautils_log as bu_log
        log_path = log_path or bu_log.configured_log_path()
        logger = bu_log.tool_logger(
            'betasuite', log_path=log_path,
            file_level=log_level or getattr( betaconfig, 'log_level', 'info' ),
            console_level=console_level or getattr( betaconfig, 'console_level', 'info' ),
            console_stream=sys.stderr )
    except Exception:
        traceback.print_exc( file=sys.stderr )
        return EXIT_STARTUP_FAILED
    # get_logger() now returns this same logger: it finds the handlers in place.
    stray.attach( logger )
    if threading.current_thread() is threading.main_thread():
        signal.signal( signal.SIGTERM, _on_sigterm )

    server = Server( proto_in, proto_out, logger, commands )
    logger.info( "engine serving: pid %d, protocol %s, log %s"%(
        os.getpid(), '/'.join( str( v ) for v in bu_proto.SUPPORTED_VERSIONS ), log_path ) )
    exit_code = EXIT_OK
    try:
        exit_code = server.serve()
    except KeyboardInterrupt:
        logger.warning( "interrupted, stopping" )
        exit_code = EXIT_INTERRUPTED
    except Terminated:
        logger.warning( "terminated, stopping" )
        exit_code = EXIT_TERMINATED
    finally:
        stray.flush()
        for stream in ( proto_out, proto_in ):
            try:
                stream.close()
            except OSError:
                pass
        logger.info( "engine stopped, exit code %d"%( exit_code, ) )
    return exit_code
