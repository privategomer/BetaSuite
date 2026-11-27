"""
test_serve.py - the engine's serve mode (betautils_serve, betaengine.py).

Most of these start the real engine in a subprocess and talk to it over
pipes, the way the desktop UI will: handshake and shutdown, framing over
a real pipe, endings by end of input and by signal, and the guarantee
that nothing but protocol messages ever reaches stdout, whether it comes
from print, a direct write to fd 1, the logger, or a child process.
Every byte the engine writes to stdout is parsed and, with jsonschema
installed, validated against protocol/v1.

The rest drive Server in-process for dispatch details, and keep the
command table and the command schemas in step.
"""

import io
import json
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid
from unittest import mock

import betautils_log as bu_log
import betautils_protocol as bu_proto
import betautils_serve as bu_serve
from tests import protocol_schemas as schemas

REPO = schemas.REPO
ENGINE = os.path.join( REPO, 'betaengine.py' )
TIMEOUT = 60

# A serve process with test commands registered, for what no production
# command does on purpose. argv[1] is the log file.
DRIVER = r'''
import os, subprocess, sys
import betautils_log as bu_log
import betautils_serve

CHILD = ( "import sys; sys.stdout.write('child stdout line\\n'); sys.stdout.flush(); "
          "sys.stderr.write('child read %r\\n' % sys.stdin.read())" )

def stray( server, data ):
    print( 'stray print from a handler' )
    sys.stdout.write( 'partial line, ' )
    sys.stdout.write( 'finished\n' )
    os.write( 1, b'stray fd write\n' )
    bu_log.get_logger().info( 'handler log line' )
    with bu_log.ProgressReporter( 'detect 640', total=3 ) as progress:
        for i in range( 3 ):
            progress.update( i + 1 )
    inherit_all = subprocess.run( [ sys.executable, '-c', CHILD ], timeout=30 ).returncode
    like_render = subprocess.Popen( [ sys.executable, '-c', CHILD ], stderr=subprocess.PIPE )
    _out, err = like_render.communicate( timeout=30 )
    return { 'inherit_all': inherit_all, 'like_render': like_render.returncode,
             'like_render_stderr': err.decode( 'utf-8', 'replace' ) }

def boom( server, data ):
    raise KeyError( '/private/path/clip.mp4' )

def unterminated( server, data ):
    import faulthandler
    sys.stdout.write( 'unterminated tail' )
    return { 'faulthandler': faulthandler.is_enabled() }

betautils_serve.register( 'test.stray', stray )
betautils_serve.register( 'test.boom', boom )
betautils_serve.register( 'test.unterminated', unterminated )
sys.exit( betautils_serve.run( log_path=sys.argv[1], log_level='trace', console_level='info' ) )
'''


class Engine:
    """A serve process on pipes, with stdout and stderr read on threads."""

    def __init__( self, argv, log_path ):
        self.log_path = log_path
        self.proc = subprocess.Popen( argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, cwd=REPO )
        self.raw = bytearray()
        self.stderr = bytearray()
        self._lines = queue.Queue()
        self._threads = [ threading.Thread( target=self._read_stdout, daemon=True ),
                          threading.Thread( target=self._read_stderr, daemon=True ) ]
        for thread in self._threads:
            thread.start()

    @classmethod
    def serve( cls, test, *extra ):
        log_path = os.path.join( test.tmp, 'engine.log' )
        return cls( [ sys.executable, ENGINE, 'serve', '--log-file', log_path,
                      '--log-level', 'trace' ] + list( extra ), log_path )

    @classmethod
    def driver( cls, test ):
        log_path = os.path.join( test.tmp, 'engine.log' )
        return cls( [ sys.executable, '-c', DRIVER, log_path ], log_path )

    def _read_stdout( self ):
        for line in iter( self.proc.stdout.readline, b'' ):
            self.raw += line
            self._lines.put( line )
        self._lines.put( None )

    def _read_stderr( self ):
        for chunk in iter( lambda: self.proc.stderr.read1( 65536 ), b'' ):
            self.stderr += chunk

    lines_sent = 0

    def write( self, data ):
        self.lines_sent += data.count( b'\n' )
        self.proc.stdin.write( data )
        self.proc.stdin.flush()

    def send( self, name, data=None, request_id=None ):
        request_id = request_id or str( uuid.uuid4() )
        self.write( bu_proto.encode( bu_proto.command( request_id, name, data ) ) )
        return request_id

    def recv( self ):
        line = self._lines.get( timeout=TIMEOUT )
        if line is None:
            raise AssertionError( "engine closed stdout; stderr:\n%s"%( self.stderr.decode( 'utf-8', 'replace' ), ) )
        return check_message( line )

    def request( self, name, data=None ):
        request_id = self.send( name, data )
        message = self.recv()
        if message['id'] != request_id:
            raise AssertionError( "result for %r answered %r"%( request_id, message ) )
        return message

    def stdout_closed( self ):
        return self._lines.get( timeout=TIMEOUT ) is None

    def wait( self ):
        code = self.proc.wait( timeout=TIMEOUT )
        for thread in self._threads:
            thread.join( timeout=TIMEOUT )
        return code

    def log( self ):
        with open( self.log_path, encoding='utf-8' ) as handle:
            return handle.read()

    def close( self ):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait( timeout=TIMEOUT )
        for stream in ( self.proc.stdin, self.proc.stdout, self.proc.stderr ):
            try:
                stream.close()
            except OSError:
                pass


def check_message( line ):
    """One stdout line -> the message, after checking it is a whole, valid envelope."""
    if not line.endswith( b'\n' ) or line.count( b'\n' ) != 1:
        raise AssertionError( "not one line: %r"%( line[:200], ) )
    message = json.loads( line )
    if schemas.HAVE_JSONSCHEMA:
        schemas.validate_message( message )
        if message.get( 'kind' ) == 'result' and message.get( 'ok' ) \
                and message['name'] in schemas.command_schema_names():
            schemas.validate_command_result( message['name'], message['data'] )
        if message.get( 'kind' ) == 'event':
            schemas.validate_event_data( message['name'], message['data'] )
    return message


class _EngineTest( unittest.TestCase ):

    def setUp( self ):
        self.tmp = tempfile.mkdtemp( prefix='betasuite-serve-' )
        self.addCleanup( lambda: __import__( 'shutil' ).rmtree( self.tmp, ignore_errors=True ) )

    def start( self, factory=Engine.serve, *extra ):
        engine = factory( self, *extra )
        self.addCleanup( engine.close )
        return engine

    def hello( self, engine ):
        result = engine.request( 'hello', { 'protocol': [ 1 ], 'client': { 'name': 'tests', 'version': '0' } } )
        self.assertTrue( result['ok'], result )
        return result

    def assertError( self, message, code, params=None ):
        self.assertFalse( message['ok'], message )
        self.assertEqual( message['error']['code'], code )
        if params is not None:
            self.assertEqual( message['error']['params'], params )

    def assertRejected( self, message, line, code, params=None ):
        self.assertEqual( ( message['kind'], message['name'] ), ( 'event', 'protocol.rejected' ), message )
        self.assertNotIn( 'id', message )
        self.assertEqual( message['data']['line'], line )
        self.assertEqual( message['data']['error']['code'], code )
        if params is not None:
            self.assertEqual( message['data']['error']['params'], params )

    def assertStdoutOnlyProtocol( self, engine ):
        lines = bytes( engine.raw ).split( b'\n' )
        self.assertEqual( lines[-1], b'', "stdout ends mid-line" )
        for line in lines[:-1]:
            check_message( line + b'\n' )
        return len( lines ) - 1


class TestHandshake( _EngineTest ):

    def test_hello_ping_shutdown( self ):
        engine = self.start()
        hello = self.hello( engine )['data']
        self.assertEqual( hello['protocol'], 1 )
        self.assertEqual( hello['pid'], engine.proc.pid )
        self.assertEqual( hello['commands'], [ 'catalog.add_files', 'catalog.add_folder', 'catalog.folders',
                                               'catalog.forget', 'catalog.get', 'catalog.list', 'catalog.pin_type',
                                               'catalog.remove_folder', 'catalog.rescan', 'catalog.strip', 'catalog.thumbnail',
                                               'config.get', 'config.pending', 'config.pending_discard',
                                               'config.reset', 'config.schema', 'config.set',
                                               'feedback.add', 'feedback.list', 'feedback.remove',
                                               'governor.enable', 'governor.get', 'governor.override',
                                               'governor.override_clear',
                                               'hello', 'hw.detect', 'job.cancel', 'job.estimate', 'job.list', 'job.move',
                                               'job.pause', 'job.restart_with_changes', 'job.resume', 'job.submit',
                                               'models.cancel', 'models.import', 'models.install', 'models.list',
                                               'models.remove', 'models.use',
                                               'perf.evidence', 'perf.optimize', 'perf.optimize_apply',
                                               'perf.optimize_cancel', 'perf.rebuild', 'ping', 'preview.look', 'resources.watch', 'shutdown', 'storage.clean',
                                               'storage.clear_history', 'storage.history', 'storage.usage' ] )
        self.assertEqual( hello['events'], [ 'protocol.rejected', 'job.state', 'job.file', 'job.progress',
                                             'job.decision', 'job.error', 'job.probed', 'catalog.changed',
                                             'catalog.strip_ready', 'models.state', 'models.progress', 'resource.sample',
                                             'governor.change', 'perf.optimize_progress', 'config.applied',
                                             'config.pending_failed' ] )
        self.assertEqual( hello['limits'], { 'max_line_bytes': bu_proto.MAX_LINE_BYTES } )
        self.assertEqual( hello['engine']['platform'], sys.platform )
        self.assertTrue( engine.request( 'ping' )['ok'] )
        self.assertEqual( engine.request( 'shutdown' )['data'], {} )
        self.assertTrue( engine.stdout_closed() )
        self.assertEqual( engine.wait(), 0 )
        self.assertEqual( self.assertStdoutOnlyProtocol( engine ), 3 )
        log = engine.log()
        self.assertIn( 'hello from tests 0, protocol 1', log )
        self.assertIn( 'engine stopped, exit code 0', log )
        self.assertIn( '[TRACE] recv {"v":1,"kind":"cmd"', log )

    def test_end_of_input_stops_cleanly( self ):
        engine = self.start()
        self.hello( engine )
        engine.proc.stdin.close()
        self.assertEqual( engine.wait(), 0 )
        self.assertIn( 'command stream closed', engine.log() )

    def test_shutdown_works_before_hello_and_other_commands_do_not( self ):
        engine = self.start()
        self.assertError( engine.request( 'ping' ), 'protocol.hello_required', { 'name': 'ping' } )
        self.assertError( engine.request( 'no.such' ), 'protocol.hello_required', { 'name': 'no.such' } )
        self.assertTrue( engine.request( 'shutdown' )['ok'] )
        self.assertEqual( engine.wait(), 0 )

    def test_nothing_is_read_after_shutdown( self ):
        engine = self.start()
        engine.write( bu_proto.encode( bu_proto.command( 's', 'shutdown' ) )
                      + bu_proto.encode( bu_proto.command( 'p', 'ping' ) ) )
        self.assertEqual( engine.recv()['id'], 's' )
        self.assertTrue( engine.stdout_closed() )
        self.assertEqual( engine.wait(), 0 )

    def test_hello_failures( self ):
        engine = self.start()
        self.assertError( engine.request( 'hello', {} ), 'command.bad_params',
                          { 'field': 'protocol', 'reason': 'missing' } )
        self.assertError( engine.request( 'hello', { 'protocol': 1 } ), 'command.bad_params',
                          { 'field': 'protocol', 'reason': 'type' } )
        self.assertError( engine.request( 'hello', { 'protocol': [ 1 ], 'client': 'ui' } ), 'command.bad_params',
                          { 'field': 'client', 'reason': 'type' } )
        self.assertError( engine.request( 'hello', { 'protocol': [ 2, 3 ] } ), 'protocol.unsupported_version',
                          { 'got': [ 2, 3 ], 'supported': [ 1 ] } )
        self.assertError( engine.request( 'ping' ), 'protocol.hello_required' )
        self.assertEqual( engine.request( 'hello', { 'protocol': [ 3, 2, 1 ] } )['data']['protocol'], 1 )
        self.assertTrue( engine.request( 'ping' )['ok'] )

    def test_unknown_command( self ):
        engine = self.start()
        self.hello( engine )
        self.assertError( engine.request( 'job.nope', { 'x': 1 } ), 'command.unknown', { 'name': 'job.nope' } )
        self.assertTrue( engine.request( 'ping' )['ok'] )


class TestFramingOverAPipe( _EngineTest ):

    def test_bad_lines_are_answered_by_id_or_by_line_and_the_engine_carries_on( self ):
        engine = self.start()
        self.hello( engine )
        ping = bu_proto.encode( bu_proto.command( 'p1', 'ping' ) )

        # Split: one command in three writes, with pauses.
        engine.write( ping[:5] )
        time.sleep( 0.2 )
        engine.write( ping[5:20] )
        time.sleep( 0.2 )
        engine.write( ping[20:] )
        self.assertEqual( engine.recv()['id'], 'p1' )

        # Joined: three commands and a blank line in one write, CRLF on one.
        engine.write( bu_proto.encode( bu_proto.command( 'p2', 'ping' ) )
                      + b'\n'
                      + bu_proto.encode( bu_proto.command( 'p3', 'ping' ) )[:-1] + b'\r\n'
                      + bu_proto.encode( bu_proto.command( 'p4', 'ping' ) ) )
        self.assertEqual( [ engine.recv()['id'] for _ in range( 3 ) ], [ 'p2', 'p3', 'p4' ] )

        # No readable id: a protocol.rejected event naming the line. Lines
        # so far: hello 1, the split ping 2, then 3 to 6 (blank included).
        engine.write( b'{"v":1,"kind":"cmd","id":"e1","name":"\xff"}\n' )
        self.assertRejected( engine.recv(), 7, 'protocol.bad_encoding' )
        engine.write( b'this is not json\n' )
        self.assertRejected( engine.recv(), 8, 'protocol.bad_json' )
        engine.write( b'[1,2]\n' )
        self.assertRejected( engine.recv(), 9, 'protocol.bad_envelope', { 'field': 'message', 'reason': 'type' } )
        self.assertEqual( engine.lines_sent, 9 )

        # A readable id: an error result for that id.
        engine.write( b'{"v":7,"kind":"cmd","id":"e2","name":"ping"}\n' )
        reply = engine.recv()
        self.assertError( reply, 'protocol.unsupported_version', { 'got': 7, 'supported': [ 1 ] } )
        self.assertEqual( reply['id'], 'e2' )

        # Unknown fields are ignored.
        engine.write( b'{"v":1,"kind":"cmd","id":"u1","name":"ping","data":{"x":1},"later":[1]}\n' )
        self.assertTrue( engine.recv()['ok'] )

        # Oversized: 1.5 MiB, then a normal command.
        size = bu_proto.MAX_LINE_BYTES + bu_proto.MAX_LINE_BYTES//2
        engine.write( b'{"v":1,"pad":"' + b'x' * ( size - 16 ) + b'"}\n' )
        self.assertRejected( engine.recv(), 12, 'protocol.line_too_long',
                             { 'limit_bytes': bu_proto.MAX_LINE_BYTES, 'line_bytes': size } )
        self.assertEqual( engine.lines_sent, 12 )
        self.assertEqual( engine.request( 'ping' )['data'], {} )

        self.assertTrue( engine.request( 'shutdown' )['ok'] )
        self.assertEqual( engine.wait(), 0 )
        self.assertStdoutOnlyProtocol( engine )


class TestStdoutIsProtocolOnly( _EngineTest ):

    def test_prints_fd_writes_logs_progress_and_children_never_reach_stdout( self ):
        engine = self.start( Engine.driver )
        self.hello( engine )
        stray = engine.request( 'test.stray' )
        self.assertTrue( stray['ok'], stray )
        self.assertEqual( ( stray['data']['inherit_all'], stray['data']['like_render'] ), ( 0, 0 ) )
        # The child read the null device, not the command stream.
        self.assertIn( "child read ''", stray['data']['like_render_stderr'] )
        self.assertTrue( engine.request( 'ping' )['ok'] )
        self.assertTrue( engine.request( 'shutdown' )['ok'] )
        self.assertEqual( engine.wait(), 0 )

        self.assertEqual( self.assertStdoutOnlyProtocol( engine ), 4 )
        for leaked in ( b'stray print', b'stray fd write', b'child stdout line', b'handler log line',
                        b'detect 640', b'partial line' ):
            self.assertNotIn( leaked, bytes( engine.raw ) )
        stderr = engine.stderr.decode( 'utf-8', 'replace' )
        self.assertIn( 'stray fd write', stderr )
        self.assertIn( 'child stdout line', stderr )
        self.assertIn( "child read ''", stderr )
        self.assertIn( 'handler log line', stderr )
        log = engine.log()
        self.assertIn( '[WARNING] stray stdout: stray print from a handler', log )
        self.assertIn( 'stray stdout: partial line, finished', log )
        self.assertIn( 'handler log line', log )

    def test_a_failing_handler_answers_with_a_code_and_the_engine_carries_on( self ):
        engine = self.start( Engine.driver )
        self.hello( engine )
        self.assertError( engine.request( 'test.boom' ), 'engine.internal', { 'exception_type': 'KeyError' } )
        self.assertTrue( engine.request( 'ping' )['ok'] )
        self.assertTrue( engine.request( 'shutdown' )['ok'] )
        self.assertEqual( engine.wait(), 0 )
        self.assertNotIn( b'/private/path', bytes( engine.raw ) )
        log = engine.log()
        self.assertIn( 'command test.boom failed', log )
        self.assertIn( "KeyError: '/private/path/clip.mp4'", log )


    def test_a_print_left_without_newline_reaches_the_log_at_exit( self ):
        engine = self.start( Engine.driver )
        self.hello( engine )
        reply = engine.request( 'test.unterminated' )
        self.assertEqual( reply['data'], { 'faulthandler': True } )
        self.assertTrue( engine.request( 'shutdown' )['ok'] )
        self.assertEqual( engine.wait(), 0 )
        self.assertNotIn( b'unterminated tail', bytes( engine.raw ) )
        log = engine.log()
        # Logged by run() itself, inside the session, not only by the
        # interpreter's own flush at exit.
        self.assertIn( 'stray stdout: unterminated tail', log )
        self.assertLess( log.index( 'stray stdout: unterminated tail' ), log.index( 'engine stopped' ) )


@unittest.skipIf( os.name == 'nt', 'POSIX signals' )
class TestSignals( _EngineTest ):

    def _stopped_by( self, signum ):
        engine = self.start()
        self.hello( engine )
        engine.proc.send_signal( signum )
        code = engine.wait()
        return code, engine.log()

    def test_sigterm_stops_cleanly( self ):
        code, log = self._stopped_by( signal.SIGTERM )
        self.assertEqual( code, 143 )
        self.assertIn( 'terminated, stopping', log )
        self.assertIn( 'engine stopped, exit code 143', log )

    def test_sigint_stops_cleanly( self ):
        code, log = self._stopped_by( signal.SIGINT )
        self.assertEqual( code, 130 )
        self.assertIn( 'engine stopped, exit code 130', log )


class TestStartupFailure( unittest.TestCase ):

    def test_a_failure_before_serving_exits_2_with_nothing_on_stdout( self ):
        with tempfile.TemporaryDirectory() as tmp:
            # A log path under a regular file cannot be created.
            blocker = os.path.join( tmp, 'file' )
            open( blocker, 'w' ).close()
            result = subprocess.run(
                [ sys.executable, ENGINE, 'serve', '--log-file', os.path.join( blocker, 'x.log' ) ],
                capture_output=True, cwd=REPO, timeout=TIMEOUT,
                input=bu_proto.encode( bu_proto.command( 'h', 'hello', { 'protocol': [ 1 ] } ) ) )
        self.assertEqual( result.returncode, 2 )
        self.assertEqual( result.stdout, b'' )
        self.assertIn( b'Traceback', result.stderr )

    @unittest.skipIf( os.name == 'nt', 'needs preexec_fn' )
    def test_a_failure_while_isolating_stdio_also_exits_2( self ):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [ sys.executable, ENGINE, 'serve', '--log-file', os.path.join( tmp, 'x.log' ) ],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=REPO, timeout=TIMEOUT,
                preexec_fn=lambda: os.close( 0 ) )
        self.assertEqual( result.returncode, 2 )
        self.assertEqual( result.stdout, b'' )
        self.assertIn( b'Traceback', result.stderr )


class TestDefaultLogPath( unittest.TestCase ):

    def test_without_log_file_the_log_goes_under_the_data_root( self ):
        with tempfile.TemporaryDirectory() as root:
            env = dict( os.environ, BETASUITE_DATA_ROOT=root, BETASUITE_LOCAL_CONFIG='none' )
            result = subprocess.run( [ sys.executable, ENGINE, 'serve' ], capture_output=True, cwd=REPO,
                                     timeout=TIMEOUT, env=env,
                                     input=bu_proto.encode( bu_proto.command( 's', 'shutdown' ) ) )
            self.assertEqual( result.returncode, 0, result.stderr )
            log_path = os.path.join( root, 'output', 'logs', 'betasuite.log' )
            with open( log_path, encoding='utf-8' ) as handle:
                self.assertIn( 'engine stopped, exit code 0', handle.read() )


class TestCommandLine( unittest.TestCase ):

    def _run( self, *args ):
        return subprocess.run( [ sys.executable, ENGINE ] + list( args ), capture_output=True,
                               cwd=REPO, timeout=TIMEOUT, stdin=subprocess.DEVNULL )

    def test_help_and_usage_errors( self ):
        self.assertEqual( self._run( '--help' ).returncode, 0 )
        self.assertIn( b'serve', self._run( '--help' ).stdout )
        self.assertEqual( self._run().returncode, 2 )
        self.assertEqual( self._run( 'serve', '--log-level', 'loud' ).returncode, 2 )
        self.assertEqual( self._run( 'bogus' ).returncode, 2 )


# --- In-process ---------------------------------------------------------------

class _Gone( io.BytesIO ):
    def write( self, _data ):
        raise BrokenPipeError( 32, 'Broken pipe' )


def _serve( lines, extra=None, out=None, logger=None, hello=True ):
    commands = bu_serve.registered_commands()
    commands.update( extra or {} )
    data = b''
    if hello:
        data += bu_proto.encode( bu_proto.command( 'h', 'hello', { 'protocol': [ 1 ] } ) )
    data += b''.join( line if isinstance( line, bytes ) else bu_proto.encode( line ) for line in lines )
    out = io.BytesIO() if out is None else out
    server = bu_serve.Server( io.BytesIO( data ), out, logger or bu_log.silent_logger( 'tests.serve' ), commands )
    code = server.serve()
    if isinstance( out, _Gone ):
        return code, []
    messages = [ json.loads( line ) for line in out.getvalue().splitlines() ]
    return code, messages[1:] if hello else messages


class TestServerInProcess( unittest.TestCase ):

    def test_results_that_are_not_json_become_internal_errors( self ):
        extra = { 'test.nan': lambda s, d: { 'x': float( 'nan' ) },
                  'test.set': lambda s, d: { 'x': { 1 } },
                  'test.none': lambda s, d: None }
        _code, messages = _serve( [ bu_proto.command( '1', 'test.nan' ), bu_proto.command( '2', 'test.set' ),
                                    bu_proto.command( '3', 'test.none' ) ], extra )
        self.assertEqual( [ ( m['id'], m['ok'] ) for m in messages ], [ ( '1', False ), ( '2', False ), ( '3', True ) ] )
        self.assertEqual( messages[0]['error'], { 'code': 'engine.internal', 'params': { 'exception_type': 'ValueError' } } )
        self.assertEqual( messages[1]['error']['params'], { 'exception_type': 'TypeError' } )
        self.assertEqual( messages[2]['data'], {} )

    def test_a_handler_reports_bad_params_with_the_id_filled_in( self ):
        def picky( _server, _data ):
            raise bu_proto.ProtocolError( 'command.bad_params', { 'field': 'files', 'reason': 'missing' } )

        def careless( _server, _data ):
            raise bu_proto.ProtocolError( 'command.made_up', {} )

        _code, messages = _serve( [ bu_proto.command( 'q', 'test.picky' ), bu_proto.command( 'r', 'test.careless' ) ],
                                  { 'test.picky': picky, 'test.careless': careless } )
        self.assertEqual( ( messages[0]['id'], messages[0]['name'] ), ( 'q', 'test.picky' ) )
        self.assertEqual( messages[0]['error']['code'], 'command.bad_params' )
        self.assertEqual( messages[1]['error'], { 'code': 'engine.internal', 'params': { 'exception_type': 'ValueError' } } )

    def test_an_interrupt_in_a_handler_is_not_swallowed( self ):
        def interrupted( _server, _data ):
            raise KeyboardInterrupt()
        with self.assertRaises( KeyboardInterrupt ):
            _serve( [ bu_proto.command( 'k', 'test.int' ) ], { 'test.int': interrupted } )

    def test_a_client_that_stopped_reading_ends_the_loop( self ):
        calls = []
        extra = { 'test.count': lambda s, d: calls.append( 1 ) }
        code, _ = _serve( [ bu_proto.command( str( i ), 'test.count' ) for i in range( 5 ) ], extra, out=_Gone() )
        self.assertEqual( code, 0 )
        self.assertEqual( calls, [] )

    def test_a_second_hello_is_answered_again( self ):
        _code, messages = _serve( [ bu_proto.command( 'h2', 'hello', { 'protocol': [ 1 ] } ) ] )
        self.assertTrue( messages[0]['ok'] )

    def test_results_come_in_command_order( self ):
        _code, messages = _serve( [ bu_proto.command( str( i ), 'ping' ) for i in range( 50 ) ] )
        self.assertEqual( [ m['id'] for m in messages ], [ str( i ) for i in range( 50 ) ] )

    def test_messages_are_traced_only_when_a_handler_takes_trace( self ):
        with tempfile.TemporaryDirectory() as tmp:
            for level, expected in ( ( 'trace', True ), ( 'debug', False ) ):
                path = os.path.join( tmp, level + '.log' )
                logger = bu_log.tool_logger( 'tests.serve.' + level, log_path=path, file_level=level,
                                             console_level=None )
                _serve( [ bu_proto.command( 'p', 'ping' ) ], logger=logger )
                for handler in list( logger.handlers ):
                    handler.close()
                with open( path, encoding='utf-8' ) as handle:
                    text = handle.read()
                self.assertEqual( 'recv {"v":1' in text and 'send {"v":1' in text, expected, level )

    def test_no_trace_record_is_made_when_no_handler_takes_trace( self ):
        logger = bu_log.tool_logger( 'tests.serve.quiet', console_level='debug', console_stream=io.StringIO() )
        with mock.patch.object( logger, 'trace' ) as trace:
            _serve( [ bu_proto.command( 'p', 'ping' ) ], logger=logger )
        trace.assert_not_called()

    def test_the_preview_is_bounded( self ):
        self.assertEqual( bu_serve._preview( b'abc' ), 'abc' )
        long_line = b'y' * 5000
        preview = bu_serve._preview( long_line )
        self.assertTrue( preview.startswith( 'y' * bu_serve._PREVIEW_BYTES ) )
        self.assertTrue( preview.endswith( '(5000 bytes)' ) )


class TestStrayOutput( unittest.TestCase ):

    def _attached( self ):
        stream = io.StringIO()
        logger = bu_log.tool_logger( 'tests.stray', console_level='trace', console_stream=stream,
                                     console_format='%(levelname)s %(message)s' )
        stray = bu_serve.StrayOutput()
        stray.attach( logger )
        return stray, stream

    def test_lines_become_warnings_and_partial_lines_wait( self ):
        stray, stream = self._attached()
        print( 'one', file=stray )
        stray.write( 'two, ' )
        self.assertEqual( stream.getvalue(), 'WARNING stray stdout: one\n' )
        stray.write( 'still two\n\n   \nthree\r\n' )
        stray.write( 'four' )
        stray.flush()
        self.assertEqual( stream.getvalue().splitlines(), [
            'WARNING stray stdout: one', 'WARNING stray stdout: two, still two',
            'WARNING stray stdout: three', 'WARNING stray stdout: four' ] )

    def test_it_is_not_a_terminal_and_its_fd_is_1( self ):
        stray = bu_serve.StrayOutput()
        self.assertFalse( stray.isatty() )
        self.assertEqual( stray.fileno(), 1 )
        self.assertTrue( stray.writable() )
        with self.assertRaises( TypeError ):
            stray.write( b'bytes' )

    def test_before_attach_and_on_reentry_lines_go_to_stderr( self ):
        stray = bu_serve.StrayOutput()
        with mock.patch.object( bu_serve.os, 'write' ) as raw_write:
            stray.write( 'early\n' )
            raw_write.assert_called_once_with( 2, b'stray stdout: early\n' )
        # A logger that writes back into the stray stream: no recursion.
        logger = bu_log.tool_logger( 'tests.stray.loop', console_level='trace', console_stream=stray )
        stray.attach( logger )
        with mock.patch.object( bu_serve.os, 'write' ) as raw_write:
            stray.write( 'loop\n' )
        self.assertEqual( raw_write.call_count, 1 )
        self.assertIn( b'stray stdout: loop', raw_write.call_args[0][1] )

    def test_a_line_ending_carriage_return_is_dropped( self ):
        stray, stream = self._attached()
        stray.write( 'crlf\r\n' )
        self.assertEqual( stream.getvalue(), 'WARNING stray stdout: crlf\n' )

    def test_an_endless_line_is_cut_into_pieces( self ):
        stray, stream = self._attached()
        stray.write( 'z' * bu_serve.StrayOutput._MAX_PENDING )
        self.assertEqual( stream.getvalue(), '' )
        stray.write( 'z' )
        self.assertEqual( len( stream.getvalue().splitlines() ), 1 )


class TestCommandSchemas( unittest.TestCase ):

    def test_every_registered_command_has_a_schema_and_every_schema_a_command( self ):
        self.assertEqual( sorted( bu_serve.registered_commands() ), schemas.command_schema_names() )

    def test_every_event_has_a_schema_and_every_schema_an_event( self ):
        self.assertEqual( sorted( bu_serve.EVENTS ), schemas.event_schema_names() )

    def test_commands_allowed_before_hello_exist( self ):
        self.assertTrue( set( bu_serve.ALWAYS_ALLOWED ) <= set( bu_serve.registered_commands() ) )

    @schemas.needs_jsonschema
    def test_each_command_schema_has_data_and_result( self ):
        files = schemas.schema_files()
        for name in schemas.command_schema_names():
            with self.subTest( command=name ):
                definitions = files['commands/%s.schema.json'%( name, )]['$defs']
                self.assertIn( 'data', definitions )
                self.assertIn( 'result', definitions )
        schemas.validate_command_data( 'hello', { 'protocol': [ 1 ], 'client': { 'name': 'ui', 'version': '2.0.0' } } )
        with self.assertRaises( schemas.jsonschema.ValidationError ):
            schemas.validate_command_data( 'hello', { 'protocol': [] } )


if __name__ == '__main__':
    unittest.main()
