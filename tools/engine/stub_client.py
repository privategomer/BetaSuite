"""
stub_client.py - start the engine the way the desktop UI will, shake
hands, ping, read the settings, shut it down, and record the session.

    python tools/engine/stub_client.py [--transcript PATH] [--log-file PATH]
        [--engine-log PATH] [--log-level LEVEL] [--console-level LEVEL]
        [--timeout SECONDS]

No input needed. Checks each step (hello negotiates protocol 1, ping
answers, config.schema lists the settings, config.get reads them,
config.set with dry_run reports that a new encode_crf re-encodes and
writes nothing, shutdown answers and the engine exits 0, every stdout
line is a protocol message) and exits 0 when all hold, 1 when one does
not. Nothing it does changes the local config.

Writes, by default next to betaconfig's log_path (under
BETASUITE_DATA_ROOT when that is set):
    stub_client.log            this tool's log, every level at --log-level
                               (default info); --console-level filters
                               the terminal only
    engine_transcript.jsonl    one record per message: milliseconds since
                               the engine started, direction ("send" to
                               the engine, "recv" from it), the message
The engine logs to --engine-log, or to its own default without it. Its
stderr is drained into this tool's log at debug, as a client must.
"""

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid

REPO = os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )
sys.path.insert( 0, REPO )

import betautils_log as bu_log
import betautils_protocol as bu_proto
import betautils_version as bu_version

ENGINE = os.path.join( REPO, 'betaengine.py' )
LEVELS = ( 'trace', 'debug', 'info', 'warn', 'error' )


class StepFailed( Exception ):
    pass


class Session:
    """The engine as a child process, with its stdout and stderr read on threads."""

    def __init__( self, engine_log, timeout, logger, transcript ):
        self.timeout = timeout
        self.logger = logger
        self.transcript = transcript
        argv = [ sys.executable, ENGINE, 'serve' ]
        if engine_log:
            argv += [ '--log-file', engine_log ]
        logger.debug( "starting %s"%( ' '.join( argv ), ) )
        self.started = time.perf_counter()
        self.proc = subprocess.Popen( argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, cwd=REPO )
        self.lines = queue.Queue()
        self.bad_lines = []
        self._record_lock = threading.Lock()
        self._threads = [ threading.Thread( target=self._read_stdout, daemon=True ),
                          threading.Thread( target=self._read_stderr, daemon=True ) ]
        for thread in self._threads:
            thread.start()

    def _ms( self ):
        return round( ( time.perf_counter() - self.started )*1000.0, 1 )

    def _record( self, direction, message ):
        with self._record_lock:
            self.transcript.write( json.dumps( { 'ms': self._ms(), 'dir': direction, 'message': message } ) + '\n' )
            self.transcript.flush()
        self.logger.info( "%s %s"%( direction, json.dumps( message, separators=( ',', ':' ) ) ) )

    def _read_stdout( self ):
        for line in iter( self.proc.stdout.readline, b'' ):
            try:
                message = json.loads( line )
                if not isinstance( message, dict ) or message.get( 'v' ) != 1:
                    raise ValueError( 'not a v1 envelope' )
            except ValueError:
                self.bad_lines.append( line )
                self.logger.error( "engine stdout carried a non-protocol line: %r"%( line[:200], ) )
                continue
            self._record( 'recv', message )
            self.lines.put( message )
        self.lines.put( None )

    def _read_stderr( self ):
        for line in iter( self.proc.stderr.readline, b'' ):
            self.logger.debug( "engine stderr: %s"%( line.decode( 'utf-8', 'replace' ).rstrip(), ) )

    def request( self, name, data=None, allowed_errors=() ):
        message = bu_proto.command( str( uuid.uuid4() ), name, data )
        self._record( 'send', message )
        self.proc.stdin.write( bu_proto.encode( message ) )
        self.proc.stdin.flush()
        try:
            reply = self.lines.get( timeout=self.timeout )
        except queue.Empty:
            raise StepFailed( "no answer to %s within %.0f s"%( name, self.timeout ) ) from None
        if reply is None:
            raise StepFailed( "the engine closed stdout before answering %s"%( name, ) )
        if reply.get( 'kind' ) != 'result' or reply.get( 'id' ) != message['id']:
            raise StepFailed( "expected the result of %s, got %r"%( name, reply ) )
        if not reply.get( 'ok' ):
            if reply.get( 'error', {} ).get( 'code' ) in allowed_errors:
                return { 'error': reply['error'] }
            raise StepFailed( "%s failed: %r"%( name, reply.get( 'error' ) ) )
        return reply['data']

    def finish( self ):
        try:
            code = self.proc.wait( timeout=self.timeout )
        except subprocess.TimeoutExpired:
            self.proc.kill()
            raise StepFailed( "the engine did not exit within %.0f s of shutdown"%( self.timeout, ) ) from None
        for thread in self._threads:
            thread.join( timeout=self.timeout )
        return code

    def kill( self ):
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()


def config_steps( session, logger ):
    """The settings, read and a change checked without writing it."""
    schema = session.request( 'config.schema', {} )
    levels = [ entry['level'] for entry in schema['settings'] ]
    logger.info( "config.schema: %d settings (%d simple, %d advanced)"%(
        len( levels ), levels.count( 'simple' ), levels.count( 'advanced' ) ) )
    settings = session.request( 'config.get', {} )['settings']
    local = [ row['path'] for row in settings if row['source'] == 'local' ]
    logger.info( "config.get: %d values, %d from the local config"%( len( settings ), len( local ) ) )
    crf = next( ( row['value'] for row in settings if row['path'] == 'encode_crf' ), None )
    if not isinstance( crf, int ):
        raise StepFailed( "config.get gave no whole-number encode_crf: %r"%( crf, ) )
    other = crf + 1 if crf < 51 else crf - 1
    checked = session.request( 'config.set', { 'values': { 'encode_crf': other }, 'dry_run': True },
                               allowed_errors=( 'config.local_off', ) )
    if 'error' in checked:
        logger.info( "config.set dry run: the local config is turned off, so it answers config.local_off" )
        return
    if checked != { 'moves': [ 'encode' ], 'written': False, 'dry_run': True }:
        raise StepFailed( "config.set dry run of encode_crf=%d answered %r"%( other, checked ) )
    logger.info( "config.set dry run: encode_crf=%d would move %s, nothing written"%(
        other, ', '.join( checked['moves'] ) ) )


def run_session( engine_log, timeout, logger, transcript ):
    session = Session( engine_log, timeout, logger, transcript )
    try:
        hello = session.request( 'hello', { 'protocol': [ bu_proto.PROTOCOL_VERSION ],
                                            'client': { 'name': 'stub_client',
                                                        'version': bu_version.get_version() } } )
        ready_ms = session._ms()
        if hello.get( 'protocol' ) != bu_proto.PROTOCOL_VERSION:
            raise StepFailed( "hello chose protocol %r"%( hello.get( 'protocol' ), ) )
        logger.info( "engine %s ready in %.0f ms (start to hello result), pid %s, commands %s"%(
            hello['engine']['version'], ready_ms, hello['pid'], ', '.join( hello['commands'] ) ) )
        started = time.perf_counter()
        session.request( 'ping' )
        logger.info( "ping round trip %.2f ms"%( ( time.perf_counter() - started )*1000.0, ) )
        config_steps( session, logger )
        session.request( 'shutdown' )
        code = session.finish()
    finally:
        session.kill()
    if code != 0:
        raise StepFailed( "the engine exited with code %d"%( code, ) )
    if session.bad_lines:
        raise StepFailed( "%d non-protocol line(s) on the engine's stdout"%( len( session.bad_lines ), ) )
    logger.info( "engine exited 0; every stdout line was a protocol message" )


def main( argv=None ):
    log_dir = os.path.dirname( bu_log.configured_log_path() )
    parser = argparse.ArgumentParser( description="Drive the engine through hello, ping, the config "
                                                  "commands and shutdown" )
    parser.add_argument( '--transcript', default=os.path.join( log_dir, 'engine_transcript.jsonl' ),
                         help="where to record the session (default: %(default)s)" )
    parser.add_argument( '--log-file', default=os.path.join( log_dir, 'stub_client.log' ),
                         help="this tool's log (default: %(default)s)" )
    parser.add_argument( '--engine-log', default=None, help="the engine's log (default: the engine's own)" )
    parser.add_argument( '--log-level', choices=LEVELS, default='info' )
    parser.add_argument( '--console-level', choices=LEVELS, default='info' )
    parser.add_argument( '--timeout', type=float, default=30.0, metavar='SECONDS',
                         help="longest wait for any one answer (default: %(default)s)" )
    args = parser.parse_args( argv )

    logger = bu_log.tool_logger( 'stub_client', log_path=args.log_file, file_level=args.log_level,
                                 console_level=args.console_level )
    transcript_dir = os.path.dirname( args.transcript )
    if transcript_dir:
        os.makedirs( transcript_dir, exist_ok=True )
    with open( args.transcript, 'w', encoding='utf-8' ) as transcript:
        try:
            run_session( args.engine_log, args.timeout, logger, transcript )
        except StepFailed as failure:
            logger.error( "%s; see %s and the engine log"%( failure, args.log_file ) )
            return 1
    logger.info( "transcript: %s"%( args.transcript, ) )
    return 0


if __name__ == '__main__':
    raise SystemExit( main() )
